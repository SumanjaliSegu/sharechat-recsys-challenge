"""Align NN holdout preds (indexed by train-file row number) with LightGBM holdout
preds (50% sample drawn with rng seed 42 in train_lgb.py) and evaluate rank blends."""
import sys, glob
import numpy as np, polars as pl, pyarrow.parquet as pq, datetime as dt
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score as A

NN_DIR = sys.argv[1] if len(sys.argv) > 1 else "D:/sharechat/out/nn_kaggle"
LGB = sys.argv[2] if len(sys.argv) > 2 else "v2emb_nopost_lr0.015_reg"

pf = pq.ParquetFile("D:/sharechat/data/train.parquet")
cut = dt.datetime(2026, 10, 27, 18)
rng = np.random.default_rng(42)
off = 0
for g in range(pf.metadata.num_row_groups):  # replay train-fold draws
    d = pl.from_arrow(pf.read_row_group(g, columns=["feature1"]))
    rng.random(int((d["feature1"] < cut).sum()))
rns, ys = [], []
off = 0
for g in range(pf.metadata.num_row_groups):
    d = pl.from_arrow(pf.read_row_group(g, columns=["feature1", "watched"])).with_row_index("rn", offset=off)
    off += d.height
    d = d.filter(pl.col("feature1") >= cut)
    if d.height == 0:
        continue
    keep = rng.random(d.height) < 0.5
    rns.append(d["rn"].to_numpy()[keep]); ys.append(d["watched"].to_numpy()[keep])
rn, y = np.concatenate(rns), np.concatenate(ys)

lgb = np.load(f"D:/sharechat/out/hopred_{LGB}.npy")
assert len(lgb) == len(y)
print(f"LGB {LGB}: {A(y, lgb):.5f}")

nn_rn = np.load(f"{NN_DIR}/nn_holdout_rn.npy"); nn_p = np.load(f"{NN_DIR}/nn_holdout_pred.npy")
m = dict(zip(nn_rn.tolist(), nn_p.tolist()))
nn = np.array([m[r] for r in rn.tolist()], np.float32)
print(f"NN  (same rows): {A(y, nn):.5f}  | NN all holdout rows: {len(nn_p):,}")
print(f"corr(rank) LGB vs NN: {np.corrcoef(rankdata(lgb), rankdata(nn))[0, 1]:.4f}")
rl, rnn = rankdata(lgb) / len(y), rankdata(nn) / len(y)
best = max(((A(y, (1 - w) * rl + w * rnn), w) for w in np.linspace(0, 1, 21)))
for w in (0.3, 0.4, 0.5, 0.6):
    print(f"  w_nn={w:.1f}: {A(y, (1 - w) * rl + w * rnn):.5f}")
print(f"best w_nn={best[1]:.2f} AUC {best[0]:.5f}")
