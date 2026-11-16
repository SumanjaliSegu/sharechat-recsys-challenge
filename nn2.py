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


KEYS = ("user", "post", "pair", "ucat", "pcat", "U", "P")


def batch(X, idx):
    return [torch.from_numpy(X[k][idx]).to(DEV, non_blocking=True) for k in KEYS]


def predict(model, X, bs=16384):
    model.eval(); out = []
    n = len(X["user"])
    with torch.no_grad():
        for i in range(0, n, bs):
            idx = np.arange(i, min(i + bs, n))
            out.append(torch.sigmoid(model(*batch(X, idx)).float()).cpu().numpy())
    return np.concatenate(out)


def train(X, y, cards, epochs, val=None, lr=1e-3, bs=8192, wd=1e-4, drop=0.3, H=192, seed=0, nopost=False):
    if SMOKE:
        bs = 512
    torch.manual_seed(seed); np.random.seed(seed)
    ucards, pcards = cards
    if nopost:
        pcards = list(pcards); pcards[0] = 1
    model = Towers(X["user"].shape[1], X["post"].shape[1], X["pair"].shape[1], ucards, pcards, drop, H).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * math.ceil(len(y) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    lossf = nn.BCEWithLogitsLoss(); amp = DEV == "cuda"
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
                loss = lossf(model(*xs).float(), t)
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
    CONFIGS = [
        dict(name="t_base", lr=1e-3, bs=8192, wd=1e-4, drop=0.3, H=192, nopost=False, epochs=4),
        dict(name="t_nopost", lr=1e-3, bs=8192, wd=1e-4, drop=0.3, H=192, nopost=True, epochs=4),
        dict(name="t_wide_drop4", lr=1e-3, bs=8192, wd=3e-4, drop=0.4, H=256, nopost=True, epochs=5),
        dict(name="t_lr2e3_bs16k", lr=2e-3, bs=16384, wd=1e-4, drop=0.3, H=192, nopost=True, epochs=5),
    ]
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
        hp = {k: cfg[k] for k in ("lr", "bs", "wd", "drop", "H", "nopost")}
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
    best = max(results, key=lambda k: results[k]["auc_holdout"])
    cfg = next(c for c in CONFIGS if c["name"] == best); be = results[best]["best_epoch"]
    log(f"BEST {best} {results[best]}")
    del Xtr, Xho

    t = read(TRAIN, 20000 if SMOKE else None); b = prepare(t); del t
    prep = Prep().fit(b); X = prep.transform(b); y = b["y"]; del b
    hp = {k: cfg[k] for k in ("lr", "bs", "wd", "drop", "H", "nopost")}
    log(f"full: {len(y):,} rows, {be} epochs, {best}")
    model, _ = train(X, y, prep.cards(), be, **hp); del X
    tt = read(TEST, 5000 if SMOKE else None); ids = tt["id"].to_numpy(); bt = prepare(tt); del tt
    Xt = prep.transform(bt)
    if cfg["nopost"]:
        Xt["pcat"][:, 0] = 0
    pt = predict(model, Xt)
    with open(f"{OUT}/submission_nn2_{best}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "watched"]); w.writerows(zip(ids.tolist(), pt.tolist()))
    log(f"wrote submission_nn2_{best}.csv rows={len(ids)} mean={pt.mean():.4f}")
