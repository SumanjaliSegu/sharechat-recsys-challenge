"""ShareChat RecSys — DCN-v2 style neural model (row-independent features only).

Stages (run in one go on Kaggle GPU):
  1. valid: train on rows < 2026-10-27 18:00, report holdout AUC per epoch,
     save holdout predictions with train-file row numbers (for blending).
  2. full: retrain on all train for the best epoch count, predict test.

Set SMOKE=1 to run a tiny CPU smoke test.
"""
import os, glob, json, time, math, datetime as dt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

SMOKE = os.environ.get("SMOKE") == "1"
OUT = os.environ.get("OUT", "/kaggle/working" if os.path.isdir("/kaggle/working") else "D:/sharechat/out/nn")
os.makedirs(OUT, exist_ok=True)


def find(name):
    for root in ["/kaggle/input", "D:/sharechat/data"]:
        hits = glob.glob(f"{root}/**/{name}", recursive=True)
        if hits:
            return hits[0]
    raise FileNotFoundError(name)


TRAIN, TEST = find("train.parquet"), find("test.parquet")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CUT = int(dt.datetime(2026, 10, 27, 18).timestamp() * 1000)  # naive -> treat as ms epoch below
torch.manual_seed(0); np.random.seed(0)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


schema = pq.ParquetFile(TRAIN).schema_arrow
DROP = {"request_id", "feature1", "feature2", "feature4", "feature15", "feature5", "feature24", "watched", "id"}
LIST_COLS = [f.name for f in schema if str(f.type).startswith("list")]
STR_COLS = [f.name for f in schema if str(f.type) == "string" and f.name not in DROP]
EPOCH_COLS = ["feature57", "feature58", "feature59"]
NUM_COLS = [f.name for f in schema if f.name not in DROP and f.name not in LIST_COLS
            and f.name not in STR_COLS and not str(f.type).startswith("timestamp")
            and f.name not in EPOCH_COLS]
U_EMB = [f"feature{i}" for i in range(26, 33)]
P_EMB = [f"feature{i}" for i in range(40, 47)]


def read(path, rows=None):
    pf = pq.ParquetFile(path)
    cols = [c for c in pf.schema_arrow.names if c not in ("request_id", "feature2", "feature4", "feature15", "feature24")]
    if rows:
        t = pf.read_row_groups([0], columns=cols).slice(0, rows)
    else:
        t = pf.read(columns=cols)
    return t


def list_to_np(col):
    arr = col.combine_chunks()
    n = len(arr)
    out = np.zeros((n, 32), np.float32)
    valid = arr.is_valid().to_numpy(zero_copy_only=False)
    vals = arr.flatten().to_numpy(zero_copy_only=False).astype(np.float32)
    out[valid] = vals.reshape(-1, 32)
    return np.nan_to_num(out)


def prepare(t):
    """Arrow table -> dict of raw numpy blocks (before normalisation)."""
    ts_ms = pc.cast(t["feature1"], pa.int64()).to_numpy().astype(np.float64)
    ts = ts_ms / 1000.0
    num = np.empty((t.num_rows, len(NUM_COLS) + 3), np.float32)
    for i, c in enumerate(NUM_COLS):
        num[:, i] = t[c].to_numpy(zero_copy_only=False).astype(np.float32)
    for j, c in enumerate(EPOCH_COLS):
        v = t[c].to_numpy(zero_copy_only=False).astype(np.float64)
        if c == "feature59":
            v = v / 1000.0
        num[:, len(NUM_COLS) + j] = (ts - v).astype(np.float32)
    U = np.stack([list_to_np(t[c]) for c in U_EMB], 1)  # n,7,32
    P = np.stack([list_to_np(t[c]) for c in P_EMB], 1)
    dots = np.einsum("nud,npd->nup", U, P).reshape(t.num_rows, -1).astype(np.float32)
    emb = np.concatenate([U.reshape(t.num_rows, -1), P.reshape(t.num_rows, -1)], 1)
    del U, P
    hour = ((ts_ms // 3_600_000) % 24).astype(np.int64)
    strs = {c: t[c].to_numpy(zero_copy_only=False) for c in STR_COLS}
    y = t["watched"].to_numpy(zero_copy_only=False).astype(np.float32) if "watched" in t.column_names else None
    return dict(num=num, dots=dots, emb=emb, hour=hour, strs=strs, y=y, ts_ms=ts_ms)


class Prep:
    """Normalisation + vocab fitted on the training rows only."""
    def fit(self, b):
        x = np.sign(b["num"]) * np.log1p(np.abs(b["num"]))
        self.mu = np.nanmean(x, 0); self.sd = np.nanstd(x, 0) + 1e-6
        d = b["dots"]; self.dmu = d.mean(0); self.dsd = d.std(0) + 1e-6
        self.vocab = {}
        for c, v in b["strs"].items():
            vals, cnt = np.unique(v[v != None].astype(str), return_counts=True)  # noqa: E711
            mc = 5 if c == "feature18" else 20
            keep = vals[cnt >= mc]
            self.vocab[c] = {k: i + 1 for i, k in enumerate(keep)}  # 0 = unknown / null
        return self

    def transform(self, b):
        x = np.sign(b["num"]) * np.log1p(np.abs(b["num"]))
        x = (x - self.mu) / self.sd
        x = np.clip(np.nan_to_num(x, nan=0.0), -6, 6).astype(np.float32)
        d = np.clip((b["dots"] - self.dmu) / self.dsd, -6, 6).astype(np.float32)
        dense = np.concatenate([x, d, np.clip(b["emb"], -6, 6)], 1).astype(np.float32)
        cats = np.zeros((len(dense), len(STR_COLS) + 1), np.int64)
        for i, c in enumerate(STR_COLS):
            m = self.vocab[c]
            cats[:, i] = [m.get(v, 0) if v is not None else 0 for v in b["strs"][c]]
        cats[:, -1] = b["hour"]
        return dense, cats

    def card(self):
        return [len(self.vocab[c]) + 1 for c in STR_COLS] + [24]


class CrossV2(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(d, d) for _ in range(n))

    def forward(self, x0):
        x = x0
        for l in self.layers:
            x = x0 * l(x) + x
        return x


class DCN(nn.Module):
    def __init__(self, n_dense, cards, drop=0.25):
        super().__init__()
        dims = [min(32, max(4, int(6 * c ** 0.25))) for c in cards]
        self.embs = nn.ModuleList(nn.Embedding(c, d) for c, d in zip(cards, dims))
        self.emb_drop = nn.Dropout(0.1)
        d_in = n_dense + sum(dims)
        self.inp = nn.Sequential(nn.Linear(d_in, 512), nn.LayerNorm(512))
        self.cross = CrossV2(512, 3)
        self.deep = nn.Sequential(
            nn.Linear(512, 1024), nn.GELU(), nn.Dropout(drop),
            nn.Linear(1024, 512), nn.GELU(), nn.Dropout(drop),
            nn.Linear(512, 256), nn.GELU())
        self.head = nn.Linear(512 + 256, 1)

    def forward(self, dense, cats):
        e = torch.cat([m(cats[:, i]) for i, m in enumerate(self.embs)], 1)
        h = self.inp(torch.cat([dense, self.emb_drop(e)], 1))
        return self.head(torch.cat([self.cross(h), self.deep(h)], 1)).squeeze(1)


def predict(model, dense, cats, bs=16384):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(dense), bs):
            d = torch.from_numpy(dense[i:i + bs]).to(DEV); c = torch.from_numpy(cats[i:i + bs]).to(DEV)
            out.append(torch.sigmoid(model(d, c)).float().cpu().numpy())
    return np.concatenate(out)


def train(dense, cats, y, cards, epochs, val=None, lr=1e-3, bs=4096, wd=1e-5, drop=0.25, seed=0):
    if SMOKE:
        bs = 512
    torch.manual_seed(seed); np.random.seed(seed)
    model = DCN(dense.shape[1], cards, drop).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * math.ceil(len(y) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    lossf = nn.BCEWithLogitsLoss()
    use_amp = DEV == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    hist = []
    for ep in range(epochs):
        model.train(); perm = np.random.permutation(len(y)); tl = 0.0
        for k, i in enumerate(range(0, len(y), bs)):
            idx = perm[i:i + bs]
            d = torch.from_numpy(dense[idx]).to(DEV); c = torch.from_numpy(cats[idx]).to(DEV)
            t = torch.from_numpy(y[idx]).to(DEV)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = lossf(model(d, c), t)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item()
        msg = f"epoch {ep + 1}/{epochs} train_loss {tl / (k + 1):.4f}"
        if val is not None:
            p = predict(model, val[0], val[1]); auc = roc_auc_score(val[2], p)
            hist.append(auc); msg += f" holdout_auc {auc:.5f}"
        log(msg)
    return model, hist


if __name__ == "__main__":
    EPOCHS = 2 if SMOKE else int(os.environ.get("EPOCHS", 5))
    log(f"device={DEV} smoke={SMOKE} num={len(NUM_COLS)}+3 str={len(STR_COLS)} lists={len(LIST_COLS)}")

    # ---------- stage 1: validation ----------
    t = read(TRAIN, rows=20000 if SMOKE else None)
    b = prepare(t); del t
    rn = np.arange(len(b["y"]))
    cut = np.datetime64("2026-10-27T18:00:00").astype("datetime64[ms]").astype(np.int64)
    is_tr = b["ts_ms"] < cut
    if SMOKE:
        is_tr = rn < 15000
    sub = lambda blk, m: {k: (v[m] if isinstance(v, np.ndarray) else ({kk: vv[m] for kk, vv in v.items()} if isinstance(v, dict) else v)) for k, v in blk.items()}
    btr, bho = sub(b, is_tr), sub(b, ~is_tr)
    del b
    prep = Prep().fit(btr)
    Dtr, Ctr = prep.transform(btr); Dho, Cho = prep.transform(bho)
    log(f"valid: train {len(Dtr):,} holdout {len(Dho):,} dense {Dtr.shape[1]} pos {btr['y'].mean():.4f}")
    I18 = STR_COLS.index("feature18")
    CONFIGS = [
        dict(name="base_wd", lr=1e-3, bs=4096, wd=1e-4, drop=0.3, nopost=False, epochs=4),
        dict(name="nopost", lr=1e-3, bs=4096, wd=1e-4, drop=0.3, nopost=True, epochs=4),
        dict(name="nopost_lr5e4_bs8k", lr=5e-4, bs=8192, wd=1e-4, drop=0.3, nopost=True, epochs=5),
        dict(name="nopost_lr2e3_bs16k", lr=2e-3, bs=16384, wd=1e-4, drop=0.3, nopost=True, epochs=5),
        dict(name="post_wd1e3_drop4", lr=1e-3, bs=8192, wd=1e-3, drop=0.4, nopost=False, epochs=4),
    ]
    if SMOKE:
        CONFIGS = CONFIGS[:2]
        for c in CONFIGS:
            c["epochs"] = 2

    def cats_for(C, cfg):
        if cfg["nopost"]:
            C = C.copy(); C[:, I18] = 0
        return C

    def cards_for(cfg):
        cd = prep.card()
        if cfg["nopost"]:
            cd = list(cd); cd[I18] = 1
        return cd

    results = {}
    np.save(f"{OUT}/nn_holdout_rn.npy", rn[~is_tr])
    for cfg in CONFIGS:
        hp = {k: cfg[k] for k in ("lr", "bs", "wd", "drop")}
        Ct_, Ch_ = cats_for(Ctr, cfg), cats_for(Cho, cfg)
        log(f"== config {cfg['name']} {hp} nopost={cfg['nopost']}")
        _, hist = train(Dtr, Ct_, btr["y"], cards_for(cfg), cfg["epochs"], val=(Dho, Ch_, bho["y"]), **hp)
        be = int(np.argmax(hist)) + 1
        model, _ = train(Dtr, Ct_, btr["y"], cards_for(cfg), be, **hp)  # compressed schedule, as in full stage
        p = predict(model, Dho, Ch_)
        auc = float(roc_auc_score(bho["y"], p))
        results[cfg["name"]] = dict(hist=hist, best_epoch=be, auc_holdout=auc)
        np.save(f"{OUT}/nn_holdout_pred_{cfg['name']}.npy", p.astype(np.float32))
        log(f"   -> best_epoch {be} retrained holdout AUC {auc:.5f}")
        del model
    json.dump(results, open(f"{OUT}/nn_valid.json", "w"), indent=1)
    best_name = max(results, key=lambda k: results[k]["auc_holdout"])
    best_cfg = next(c for c in CONFIGS if c["name"] == best_name)
    best_ep = results[best_name]["best_epoch"]
    log(f"BEST {best_name} {results[best_name]}")
    del Dtr, Ctr, Dho, Cho, btr, bho

    # ---------- stage 2: full train -> test ----------
    t = read(TRAIN, rows=20000 if SMOKE else None)
    b = prepare(t); del t
    prep = Prep().fit(b)
    D, C = prep.transform(b); y = b["y"]; del b
    log(f"full: {len(D):,} rows, {best_ep} epochs, config {best_name}")
    hp = {k: best_cfg[k] for k in ("lr", "bs", "wd", "drop")}
    model, _ = train(D, cats_for(C, best_cfg), y, cards_for(best_cfg), best_ep, **hp)
    del D, C
    tt = read(TEST, rows=5000 if SMOKE else None)
    ids = tt["id"].to_numpy()
    bt = prepare(tt); del tt
    Dt, Ct = prep.transform(bt)
    pt = predict(model, Dt, cats_for(Ct, best_cfg))
    import csv
    with open(f"{OUT}/submission_nn_{best_name}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "watched"]); w.writerows(zip(ids.tolist(), pt.tolist()))
    log(f"wrote {OUT}/submission_nn_{best_name}.csv rows={len(ids)} mean={pt.mean():.4f}")
