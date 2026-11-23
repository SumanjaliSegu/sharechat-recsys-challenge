"""Pretrained in-context tabular model (TabPFN v2, fallback TabICL) as a diversity component.

Row-only: each holdout / test row is predicted from its own raw features, with a random
sample of labelled training rows as context. No request columns, no other test rows.
Outputs: tp_holdout.parquet (rn, p) for rows with feature1 >= cut (fit on rows < cut),
         submission_tabpfn.csv (id, watched) with context drawn from all training rows.
Adapts the number of context bags to a time budget measured on the first chunk.
"""
import os, sys, time, glob, subprocess, datetime as dt
import numpy as np, polars as pl
from sklearn.metrics import roc_auc_score

FEATS = ("feature54,feature493,feature352,feature419,feature418,feature53,feature50,feature515,feature514,feature103,"
         "feature39,feature431,feature430,feature510,feature358,feature511,feature492,feature334,feature150,feature417,"
         "feature356,feature151,feature478,feature490,feature429,feature37,feature357,feature428,feature104,feature491,"
         "feature126,feature439,feature337,feature340,feature310,feature311,feature482,feature484,feature376,feature332,"
         "feature487,feature438,feature503,feature319,feature304,feature509,feature475,feature416,feature346,feature485,"
         "feature480,feature146,feature38,feature82,feature479,feature35,feature321,feature502,feature56,feature96,"
         "feature483,feature488,feature349,feature153,feature443,feature333,feature351,feature477,feature52,feature127,"
         "feature486,feature331,feature507,feature506,feature336,feature314,feature305,feature345,feature85,feature312,"
         "feature512,feature90,feature545,feature109,feature508,feature33,feature335,feature129,feature298,feature437").split(",")
NF = int(os.environ.get("NF", "100")); FEATS = FEATS[:NF]
CTX = int(os.environ.get("CTX", "10000"))
MAXBAGS = int(os.environ.get("BAGS", "4"))
BUDGET = float(os.environ.get("BUDGET_H", "8.0")) * 3600
CHUNK = 25000
T0 = time.time()
k = glob.glob("/kaggle/input/**/train.parquet", recursive=True)
DATA = os.path.dirname(k[0]) if k else "D:/sharechat/data"
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "out/tabpfn"
os.makedirs(OUT, exist_ok=True)
def log(*a): print(f"[{(time.time()-T0)/60:6.1f}m]", *a, flush=True)

def make_model():
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tabpfn==2.0.9"], check=True)
        from tabpfn import TabPFNClassifier
        m = TabPFNClassifier(device="cuda", n_estimators=4, ignore_pretraining_limits=True)
        m.fit(np.random.rand(200, 5).astype(np.float32), np.r_[np.zeros(100), np.ones(100)])
        log("using TabPFN 2.0.9"); return lambda: TabPFNClassifier(device="cuda", n_estimators=4, ignore_pretraining_limits=True), "tabpfn"
    except Exception as e:
        log("TabPFN unavailable:", repr(e)[:300])
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tabicl"], check=True)
    from tabicl import TabICLClassifier
    log("using TabICL"); return lambda: TabICLClassifier(device="cuda"), "tabicl"

Make, NAME = make_model()
if NAME == "tabicl": CTX = int(os.environ.get("CTX_ICL", "50000"))

tr = pl.read_parquet(f"{DATA}/train.parquet", columns=["feature1", "watched"] + FEATS).with_row_index("rn")
te = pl.read_parquet(f"{DATA}/test.parquet", columns=["id"] + FEATS)
cut = dt.datetime(2026, 10, 27, 18)
f1 = tr["feature1"]
if f1.dtype == pl.Utf8: f1 = f1.str.to_datetime()
hold = (f1 >= cut).to_numpy()
X = tr.select(FEATS).cast(pl.Float32).to_numpy(); y = tr["watched"].to_numpy().astype(int); rn = tr["rn"].to_numpy()
XT = te.select(FEATS).cast(pl.Float32).to_numpy(); ids = te["id"]
del tr, te
log(f"{NAME}: feats {len(FEATS)} ctx {CTX} train {len(X)} holdout {hold.sum()} test {len(XT)}")

def run_bag(seed, pool, Xq, tag):
    rng = np.random.default_rng(seed); idx = rng.choice(pool, CTX, replace=False)
    m = Make(); m.fit(X[idx], y[idx]); out = np.empty(len(Xq), np.float32); t = time.time()
    for s in range(0, len(Xq), CHUNK):
        out[s:s+CHUNK] = m.predict_proba(Xq[s:s+CHUNK])[:, 1]
        if s == 0: log(f"{tag} bag{seed}: first chunk {time.time()-t:.1f}s")
    return out, time.time() - t

pre = np.where(~hold)[0]; Xh = X[hold]; yh = y[hold]
ph, sec = run_bag(0, pre, Xh, "holdout")
log(f"holdout bag0 AUC {roc_auc_score(yh, ph):.5f} in {sec/60:.1f}m")
per_row = sec / len(Xh)
left = BUDGET - (time.time() - T0)
bags = int(max(1, min(MAXBAGS, left / (per_row * (len(Xh) + len(XT)) * 1.1) + 1)))
log(f"bags chosen {bags} (per-row {per_row*1e3:.3f} ms)")
from scipy.stats import rankdata
H = [rankdata(ph) / len(ph)]
for b in range(1, bags):
    p, _ = run_bag(b, pre, Xh, "holdout"); H.append(rankdata(p) / len(p))
    log(f"holdout {b+1}-bag AUC {roc_auc_score(yh, np.mean(H, 0)):.5f}")
pl.DataFrame({"rn": rn[hold], "p": np.mean(H, 0)}).write_parquet(f"{OUT}/tp_holdout.parquet")
allidx = np.arange(len(X)); T = []
for b in range(bags):
    if time.time() - T0 > BUDGET * 1.15: log("budget hit, stopping test bags"); break
    p, _ = run_bag(100 + b, allidx, XT, "test"); T.append(rankdata(p) / len(p))
    pl.DataFrame({"id": ids, "watched": np.mean(T, 0)}).write_csv(f"{OUT}/submission_tabpfn.csv")
    log(f"test bag {b+1} written")
log("done")
