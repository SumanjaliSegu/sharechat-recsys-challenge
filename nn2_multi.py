"""ShareChat RecSys — structured multi-tower network (our own architecture).

Towers follow the data's structure (from the column audit):
  user tower : 257 user-level columns + user categoricals + 7 user embedding vectors
  post tower : post columns + post ages + post categoricals + 7 post embedding vectors
  pair tower : 318 user×post columns + hour
  match block: 49 cosine matches of every user vector with every post vector
               + 7 learned bilinear matches u_k^T W_k p_k
  fusion     : [user, post, pair, user⊙post, match] -> DCN-v2 cross layers + MLP -> logit
Numeric columns get a rank (quantile) transform fitted on training rows only.
Every prediction uses only its own row (competition rule).
SMOKE=1 runs a tiny CPU test.
"""
import os, glob, json, time, math, csv
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

SMOKE = os.environ.get("SMOKE") == "1"
OUT = os.environ.get("OUT", "/kaggle/working" if os.path.isdir("/kaggle/working") else "D:/sharechat/out/nn2")
os.makedirs(OUT, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def find(name):
    for root in ["/kaggle/input", "D:/sharechat/data"]:
        hits = glob.glob(f"{root}/**/{name}", recursive=True)
        if hits:
            return hits[0]
    raise FileNotFoundError(name)


TRAIN, TEST = find("train.parquet"), find("test.parquet")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


GROUPS = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "col_groups.json")))  # from column audit
USER_CATS = ["feature6", "feature7", "feature8", "feature9", "feature10", "feature11", "feature12",
             "feature13", "feature14", "feature16", "feature17", "feature22"]
POST_CATS = ["feature18", "feature19", "feature20", "feature21", "feature23", "feature25"]
U_EMB = [f"feature{i}" for i in range(26, 33)]
P_EMB = [f"feature{i}" for i in range(40, 47)]
NUM = GROUPS["user"] + GROUPS["post"] + GROUPS["pair"]


def list_to_np(col):
    arr = col.combine_chunks()
    out = np.zeros((len(arr), 32), np.float32)
    valid = arr.is_valid().to_numpy(zero_copy_only=False)
    out[valid] = arr.flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(-1, 32)
    return np.nan_to_num(out)


def read(path, rows=None):
    pf = pq.ParquetFile(path)
    cols = [c for c in pf.schema_arrow.names
            if c in set(NUM + USER_CATS + POST_CATS + U_EMB + P_EMB + ["feature1", "feature57", "feature58", "feature59", "watched", "id"])]
    t = pf.read_row_groups([0], columns=cols).slice(0, rows) if rows else pf.read(columns=cols)
    return t


def prepare(t):
    n = t.num_rows
    ts_ms = pc.cast(t["feature1"], pa.int64()).to_numpy().astype(np.float64)
    ts = ts_ms / 1000.0
    b = {}
    for g in ("user", "post", "pair"):
        m = np.empty((n, len(GROUPS[g])), np.float32)
        for i, c in enumerate(GROUPS[g]):
            m[:, i] = t[c].to_numpy(zero_copy_only=False).astype(np.float32)
        b[g] = m
    age = lambda c, s=1.0: (ts - t[c].to_numpy(zero_copy_only=False).astype(np.float64) / s).astype(np.float32)[:, None]
    b["user"] = np.concatenate([b["user"], age("feature57")], 1)
    b["post"] = np.concatenate([b["post"], age("feature58"), age("feature59", 1000.0)], 1)
    b["U"] = np.stack([list_to_np(t[c]) for c in U_EMB], 1)
    b["P"] = np.stack([list_to_np(t[c]) for c in P_EMB], 1)
    b["hour"] = ((ts_ms // 3_600_000) % 24).astype(np.int64)
    b["ucat"] = {c: t[c].to_numpy(zero_copy_only=False) for c in USER_CATS}
    b["pcat"] = {c: t[c].to_numpy(zero_copy_only=False) for c in POST_CATS}
    b["y"] = t["watched"].to_numpy(zero_copy_only=False).astype(np.float32) if "watched" in t.column_names else None
    b["ts_ms"] = ts_ms
    return b


def subset(b, m):
    out = {}
    for k, v in b.items():
        if isinstance(v, np.ndarray):
            out[k] = v[m]
        elif isinstance(v, dict):
            out[k] = {kk: vv[m] for kk, vv in v.items()}
        else:
            out[k] = v
    return out


class Prep:
    NQ = 256

    def fit(self, b):
        rs = np.random.default_rng(0)
        self.q = {}
        for g in ("user", "post", "pair"):
            x = b[g]
            idx = rs.choice(len(x), min(len(x), 300_000), replace=False)
            self.q[g] = np.nanquantile(x[idx], np.linspace(0, 1, self.NQ), axis=0).T  # cols x NQ
        self.vocab = {}
        for c, v in list(b["ucat"].items()) + list(b["pcat"].items()):
            vals, cnt = np.unique(v[v != None].astype(str), return_counts=True)  # noqa: E711
            keep = vals[cnt >= (10 if c == "feature18" else 20)]
            self.vocab[c] = {k: i + 1 for i, k in enumerate(keep)}
        return self

    def _rank(self, x, q):
        out = np.zeros_like(x, dtype=np.float32)
        grid = np.linspace(-1, 1, self.NQ)
        for j in range(x.shape[1]):
            col = x[:, j]; qq = q[j]
            ok = ~np.isnan(col)
            if np.isnan(qq).all():
                continue
            # unique-ify quantiles for interp
            qu, iu = np.unique(qq, return_index=True)
            g = grid[iu] if len(qu) > 1 else np.zeros(1)
            out[ok, j] = np.interp(col[ok], qu, g) if len(qu) > 1 else 0.0
        return out

    def transform(self, b):
        X = {g: self._rank(b[g], self.q[g]) for g in ("user", "post", "pair")}
        enc = lambda d, cols: np.stack([[self.vocab[c].get(v, 0) if v is not None else 0 for v in d[c]] for c in cols], 1).astype(np.int64)
        X["ucat"] = np.concatenate([enc(b["ucat"], USER_CATS), b["hour"][:, None]], 1)
        X["pcat"] = enc(b["pcat"], POST_CATS)
        X["U"] = np.clip(b["U"], -6, 6); X["P"] = np.clip(b["P"], -6, 6)
        return X

    def cards(self):
        return [len(self.vocab[c]) + 1 for c in USER_CATS] + [24], [len(self.vocab[c]) + 1 for c in POST_CATS]


def mlp(d_in, dims, drop):
    layers = []
    for d in dims:
        layers += [nn.Linear(d_in, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(drop)]
        d_in = d
    return nn.Sequential(*layers)


def edim(c):
    return min(24, max(4, int(5 * c ** 0.25)))


class Towers(nn.Module):
    def __init__(self, n_user, n_post, n_pair, ucards, pcards, drop=0.3, H=192, nocat_post_id=False):
        super().__init__()
        self.ue = nn.ModuleList(nn.Embedding(c, edim(c)) for c in ucards)
        self.pe = nn.ModuleList(nn.Embedding(c, edim(c)) for c in pcards)
        du = n_user + sum(edim(c) for c in ucards) + 7 * 32
        dp = n_post + sum(edim(c) for c in pcards) + 7 * 32
        self.user = mlp(du, [512, H], drop)
        self.post = mlp(dp, [384, H], drop)
        self.pair = mlp(n_pair, [512, H], drop)
        self.W = nn.Parameter(torch.randn(7, 32, 32) * 0.02 + torch.eye(32)[None])
        d_f = 4 * H + 49 + 7
        self.cross = nn.ModuleList(nn.Linear(d_f, d_f) for _ in range(2))
        self.head = nn.Sequential(mlp(d_f, [512, 256], drop), nn.Linear(256, 1))
        self.edrop = nn.Dropout(0.1)

    def forward(self, xu, xp, xr, cu, cp, U, P):
        eu = self.edrop(torch.cat([m(cu[:, i]) for i, m in enumerate(self.ue)], 1))
        ep = self.edrop(torch.cat([m(cp[:, i]) for i, m in enumerate(self.pe)], 1))
        hu = self.user(torch.cat([xu, eu, U.flatten(1)], 1))
        hp = self.post(torch.cat([xp, ep, P.flatten(1)], 1))
        hr = self.pair(xr)
        Un = nn.functional.normalize(U, dim=-1); Pn = nn.functional.normalize(P, dim=-1)
        cos = torch.einsum("nud,npd->nup", Un, Pn).flatten(1)                  # 49
        bil = torch.einsum("nkd,kde,nke->nk", U, self.W, P) / 32.0              # 7
        x0 = torch.cat([hu, hp, hr, hu * hp, cos, bil], 1)
        x = x0
        for l in self.cross:
            x = x0 * l(x) + x
        return self.head(x).squeeze(1)


class PLR(nn.Module):
    """Periodic numeric embeddings (Gorishniy et al., NeurIPS 2022): each scalar feature x ->
    ReLU(Linear_j([sin(2*pi*c_j*x), cos(2*pi*c_j*x)])), separate trainable frequencies c_j and
    linear map per feature j. Input: rank-transformed numerics in [-1, 1]."""
    def __init__(self, n, k=16, d=8, sigma=1.0):
        super().__init__()
        self.c = nn.Parameter(torch.randn(n, k) * sigma)
        self.W = nn.Parameter(torch.randn(n, 2 * k, d) * (1.0 / math.sqrt(2 * k)))
        self.b = nn.Parameter(torch.zeros(n, d))
        self.out_dim = n * d

    def forward(self, x):
        z = 2 * math.pi * x.unsqueeze(-1) * self.c                      # (N, n, k)
        z = torch.cat([torch.sin(z), torch.cos(z)], -1)                  # (N, n, 2k)
        return torch.relu(torch.einsum("njk,jkd->njd", z, self.W) + self.b).flatten(1)


class TowersPLR(nn.Module):
    """Towers network whose user / post / pair numeric inputs first pass through PLR embeddings."""
    SIGMA = float(os.environ.get("PLR_SIGMA", "1.0"))

    def __init__(self, n_user, n_post, n_pair, ucards, pcards, drop=0.3, H=192):
        super().__init__()
        K, D = int(os.environ.get("PLR_K", "16")), int(os.environ.get("PLR_D", "8"))
        self.eu, self.ep, self.er = (PLR(n, k=K, d=D, sigma=self.SIGMA) for n in (n_user, n_post, n_pair))
        self.net = Towers(self.eu.out_dim, self.ep.out_dim, self.er.out_dim, ucards, pcards, drop, H)

    def forward(self, xu, xp, xr, cu, cp, U, P):
        return self.net(self.eu(xu), self.ep(xp), self.er(xr), cu, cp, U, P)


class BELinear(nn.Module):
    """BatchEnsemble linear layer (TabM): shared W, per-member input/output scaling r_k, s_k and bias b_k."""
    def __init__(self, d_in, d_out, k, init_r=True):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out, bias=False)
        self.r = nn.Parameter(torch.ones(k, d_in) if not init_r else 1 + 0.1 * torch.randn(k, d_in).sign() * torch.rand(k, d_in))
        self.s = nn.Parameter(torch.ones(k, d_out))
        self.b = nn.Parameter(torch.zeros(k, d_out))

    def forward(self, x):  # x: (n, k, d_in)
        return self.lin(x * self.r) * self.s + self.b


class TabM(nn.Module):
    """TabM (Gorishniy et al. 2024): parameter-efficient ensemble of k MLPs over one flat row vector."""
    K = 16

    def __init__(self, n_user, n_post, n_pair, ucards, pcards, drop=0.3, H=512, nocat_post_id=False):
        super().__init__()
        k = self.K
        self.ue = nn.ModuleList(nn.Embedding(c, edim(c)) for c in ucards)
        self.pe = nn.ModuleList(nn.Embedding(c, edim(c)) for c in pcards)
        d = n_user + n_post + n_pair + sum(edim(c) for c in ucards) + sum(edim(c) for c in pcards) + 2 * 7 * 32 + 49 + 7
        self.W = nn.Parameter(torch.randn(7, 32, 32) * 0.02 + torch.eye(32)[None])
        self.l1, self.l2, self.l3 = BELinear(d, H, k), BELinear(H, H, k, False), BELinear(H, H, k, False)
        self.out = BELinear(H, 1, k, False)
        self.drop = nn.Dropout(drop); self.edrop = nn.Dropout(0.1)

    def forward(self, xu, xp, xr, cu, cp, U, P):
        eu = torch.cat([m(cu[:, i]) for i, m in enumerate(self.ue)], 1)
        ep = torch.cat([m(cp[:, i]) for i, m in enumerate(self.pe)], 1)
        Un = nn.functional.normalize(U, dim=-1); Pn = nn.functional.normalize(P, dim=-1)
        cos = torch.einsum("nud,npd->nup", Un, Pn).flatten(1)
        bil = torch.einsum("nkd,kde,nke->nk", U, self.W, P) / 32.0
        x = torch.cat([xu, xp, xr, self.edrop(eu), self.edrop(ep), U.flatten(1), P.flatten(1), cos, bil], 1)
        x = x[:, None, :].expand(-1, self.K, -1)
        for l in (self.l1, self.l2, self.l3):
            x = self.drop(nn.functional.relu(l(x)))
        return self.out(x).squeeze(-1)  # (n, k) logits; training averages member losses, predict averages probs


class GTF(nn.Module):
    """Feature-group Transformer: numeric columns in chunks of 16, each categorical, each user / post
    embedding and each user*post embedding product become one token; a CLS token reads the result."""
    D, CH = 64, 16

    def __init__(self, n_user, n_post, n_pair, ucards, pcards, drop=0.1, H=128, nocat_post_id=False):
        super().__init__()
        D, CH = self.D, self.CH
        self.sizes = [n_user, n_post, n_pair]
        self.nch = [math.ceil(n / CH) for n in self.sizes]
        self.Wn = nn.ParameterList(nn.Parameter(torch.randn(k, CH, D) * CH ** -0.5) for k in self.nch)
        self.bn = nn.ParameterList(nn.Parameter(torch.zeros(k, D)) for k in self.nch)
        self.ue = nn.ModuleList(nn.Embedding(c, D) for c in ucards)
        self.pe = nn.ModuleList(nn.Embedding(c, D) for c in pcards)
        self.WU = nn.Parameter(torch.randn(7, 32, D) * 32 ** -0.5)
        self.WP = nn.Parameter(torch.randn(7, 32, D) * 32 ** -0.5)
        self.WX = nn.Parameter(torch.randn(7, 32, D) * 32 ** -0.5)
        n_tok = sum(self.nch) + len(ucards) + len(pcards) + 21
        self.pos = nn.Parameter(torch.randn(n_tok, D) * 0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, D))
        layer = nn.TransformerEncoderLayer(D, 4, H, drop, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, 3)
        self.head = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 64), nn.GELU(), nn.Linear(64, 1))

    def _num(self, x, i):
        k, CH = self.nch[i], self.CH
        x = nn.functional.pad(x, (0, k * CH - x.shape[1])).view(-1, k, CH)
        return torch.einsum("nkc,kcd->nkd", x, self.Wn[i]) + self.bn[i]

    def forward(self, xu, xp, xr, cu, cp, U, P):
        Un = nn.functional.normalize(U, dim=-1); Pn = nn.functional.normalize(P, dim=-1)
        toks = [self._num(xu, 0), self._num(xp, 1), self._num(xr, 2),
                torch.stack([m(cu[:, i]) for i, m in enumerate(self.ue)], 1),
                torch.stack([m(cp[:, i]) for i, m in enumerate(self.pe)], 1),
                torch.einsum("nkc,kcd->nkd", U, self.WU), torch.einsum("nkc,kcd->nkd", P, self.WP),
                torch.einsum("nkc,kcd->nkd", Un * Pn * 32 ** 0.5, self.WX)]
        t = torch.cat(toks, 1) + self.pos
        t = torch.cat([self.cls.expand(len(t), -1, -1), t], 1)
        return self.head(self.enc(t)[:, 0]).squeeze(1)


KEYS = ("user", "post", "pair", "ucat", "pcat", "U", "P")


def batch(X, idx):
    return [torch.from_numpy(X[k][idx]).to(DEV, non_blocking=True) for k in KEYS]


def predict(model, X, bs=16384):
    model.eval(); out = []
    n = len(X["user"])
    with torch.no_grad():
        for i in range(0, n, bs):
            idx = np.arange(i, min(i + bs, n))
            z = torch.sigmoid(model(*batch(X, idx)).float())
            out.append((z.mean(1) if z.dim() == 2 else z).cpu().numpy())
    return np.concatenate(out)


def train(X, y, cards, epochs, val=None, lr=1e-3, bs=8192, wd=1e-4, drop=0.3, H=192, seed=0, nopost=False,
          gamma=0.0, ls=0.0, arch="towers"):
    if SMOKE:
        bs = 512
    torch.manual_seed(seed); np.random.seed(seed)
    ucards, pcards = cards
    if nopost:
        pcards = list(pcards); pcards[0] = 1
    M = {"tabm": TabM, "gtf": GTF, "plr": TowersPLR}.get(arch, Towers)
    model = M(X["user"].shape[1], X["post"].shape[1], X["pair"].shape[1], ucards, pcards, drop, H).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * math.ceil(len(y) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    def lossf(z, t):  # BCE; gamma>0 -> focal loss (1-p_t)^gamma * CE; ls>0 -> label smoothing toward 0.5
        if ls:
            t = t * (1 - ls) + 0.5 * ls
        ce = nn.functional.binary_cross_entropy_with_logits(z, t, reduction="none")
        if gamma:
            p = torch.sigmoid(z); pt = p * t + (1 - p) * (1 - t)
            ce = (1 - pt).clamp_min(0).pow(gamma) * ce
        return ce.mean()
    amp = DEV == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    hist = []
    for ep in range(epochs):
        model.train(); perm = np.random.permutation(len(y)); tl = 0.0; k = 0
        for k, i in enumerate(range(0, len(y), bs)):
            idx = perm[i:i + bs]
            xs = batch(X, idx)
            if nopost:
                xs[4] = xs[4].clone(); xs[4][:, 0] = 0
            t = torch.from_numpy(y[idx]).to(DEV)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                zz = model(*xs).float()
                loss = lossf(zz, t[:, None].expand_as(zz) if zz.dim() == 2 else t)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); tl += loss.item()
        msg = f"epoch {ep + 1}/{epochs} loss {tl / (k + 1):.4f}"
        if val is not None:
            Xv, yv = val
            if nopost:
                Xv = dict(Xv); Xv["pcat"] = Xv["pcat"].copy(); Xv["pcat"][:, 0] = 0
            auc = roc_auc_score(yv, predict(model, Xv)); hist.append(auc); msg += f" holdout_auc {auc:.5f}"
        log(msg)
    return model, hist


if __name__ == "__main__":
    ARMS = {"base": dict(gamma=0.0, ls=0.0), "focal1": dict(gamma=1.0, ls=0.0), "focal2": dict(gamma=2.0, ls=0.0),
            "ls01": dict(gamma=0.0, ls=0.01),
            "tabm": dict(gamma=0.0, ls=0.0, arch="tabm", H=512, drop=0.1, lr=2e-3, wd=3e-4),
            "gtf": dict(gamma=0.0, ls=0.0, arch="gtf", H=128, drop=0.1, lr=1e-3, wd=1e-4, bs=2048, epochs=8),
            "plr": dict(gamma=0.0, ls=0.0, arch="plr", lr=float(os.environ.get("PLR_LR", "1e-3")), wd=1e-4,
                        drop=float(os.environ.get("PLR_DROP", "0.3")), H=int(os.environ.get("PLR_H", "192")),
                        epochs=int(os.environ.get("PLR_EPOCHS", "6")))}
    CONFIGS = [{**dict(name=f"t_{arm}_s{sd}", lr=1e-3, bs=8192, wd=1e-4, drop=0.3, H=192, nopost=False, epochs=6, seed=sd,
                     arch="towers"), **ARMS[arm]}
               for arm in os.environ.get("NNARMS", "base").split(",")
               for sd in tuple(int(x) for x in os.environ.get("TSEEDS", "0,1,2").split(","))]
    if SMOKE:
        CONFIGS = CONFIGS[:2]
        for c in CONFIGS:
            c["epochs"] = 2
    log(f"device={DEV} smoke={SMOKE} user={len(GROUPS['user'])} post={len(GROUPS['post'])} pair={len(GROUPS['pair'])}")

    t = read(TRAIN, 20000 if SMOKE else None); b = prepare(t); del t
    rn = np.arange(len(b["y"]))
    cut = np.datetime64("2026-10-27T18:00:00").astype("datetime64[ms]").astype(np.int64)
    is_tr = (rn < 15000) if SMOKE else (b["ts_ms"] < cut)
    btr, bho = subset(b, is_tr), subset(b, ~is_tr); del b
    prep = Prep().fit(btr)
    Xtr, Xho = prep.transform(btr), prep.transform(bho)
    ytr, yho = btr["y"], bho["y"]; del btr, bho
    log(f"valid: train {len(ytr):,} holdout {len(yho):,}")
    np.save(f"{OUT}/nn_holdout_rn.npy", rn[~is_tr])
    results = {}
    for cfg in CONFIGS:
        hp = {k: cfg[k] for k in ("lr", "bs", "wd", "drop", "H", "nopost", "seed", "gamma", "ls", "arch")}
        log(f"== {cfg['name']} {hp}")
        _, hist = train(Xtr, ytr, prep.cards(), cfg["epochs"], val=(Xho, yho), **hp)
        be = int(np.argmax(hist)) + 1
        model, _ = train(Xtr, ytr, prep.cards(), be, **hp)
        Xv = Xho
        if cfg["nopost"]:
            Xv = dict(Xho); Xv["pcat"] = Xho["pcat"].copy(); Xv["pcat"][:, 0] = 0
        p = predict(model, Xv); auc = float(roc_auc_score(yho, p))
        results[cfg["name"]] = dict(hist=hist, best_epoch=be, auc_holdout=auc)
        np.save(f"{OUT}/nn_holdout_pred_{cfg['name']}.npy", p.astype(np.float32))
        log(f"   -> best_epoch {be} retrained AUC {auc:.5f}")
        del model
    json.dump(results, open(f"{OUT}/nn_valid.json", "w"), indent=1)
    del Xtr, Xho

    t = read(TRAIN, 20000 if SMOKE else None); b = prepare(t); del t
    prep = Prep().fit(b); X = prep.transform(b); y = b["y"]; del b
    tt = read(TEST, 5000 if SMOKE else None); ids = tt["id"].to_numpy(); bt = prepare(tt); del tt
    Xt = prep.transform(bt); del bt
    for cfg in CONFIGS:
        hp = {k: cfg[k] for k in ("lr", "bs", "wd", "drop", "H", "nopost", "seed", "gamma", "ls", "arch")}
        be = results[cfg["name"]]["best_epoch"]
        log(f"full: {len(y):,} rows, {be} epochs, {cfg['name']}")
        model, _ = train(X, y, prep.cards(), be, **hp)
        pt = predict(model, Xt); del model
        with open(f"{OUT}/submission_nn2_{cfg['name']}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["id", "watched"]); w.writerows(zip(ids.tolist(), pt.tolist()))
        log(f"wrote submission_nn2_{cfg['name']}.csv mean={pt.mean():.4f}")
