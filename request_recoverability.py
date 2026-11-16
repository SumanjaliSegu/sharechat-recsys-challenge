"""DIAGNOSTIC ONLY (never used for a submission).
Question: do a row's own columns encode its request size / position in the request?
Method: compute those targets from request_id (for the diagnostic only), then train LightGBM to
predict them from the row's own features, evaluating on requests never seen in training.
High R^2 / Spearman -> the info is in the row (usable legitimately); low -> it is not."""
import sys, json
import numpy as np
sys.argv = ["x", "valid", "--emb", "--nopost", "--reg", "--lr=0.03", "--fullho", "--stable", "--nocatid", "--dropE", "--ff=0.5"]
sys.path.insert(0, "/kaggle/working")
import train_lgb as T
import polars as pl
import pyarrow.parquet as pq
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

T.READ_COLS += ["request_id"]
pf = pq.ParquetFile(f"{T.DATA}/train.parquet")
parts = []
for g in range(0, pf.metadata.num_row_groups, 2):  # half the row groups, whole requests kept
    df = pl.from_arrow(pf.read_row_group(g, columns=T.READ_COLS)).with_row_index("rn")
    df = df.with_columns(pl.len().over("request_id").alias("req_size"),
                         pl.col("rn").rank().over("request_id").alias("req_pos"))
    parts.append(df)
df = pl.concat(parts)
vocab = T.build_vocab(df)
X, names = T.to_np(T.featurize(df, vocab))
grp = (df["request_id"].hash(seed=5) % 5).to_numpy()
tr, te = grp != 0, grp == 0
cats = [c for c in T.STR_COLS if c in names]
P = dict(objective="regression", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.5,
         verbose=-1, num_threads=4)
out = {}
for tgt in ("req_size", "req_pos"):
    y = np.log1p(df[tgt].to_numpy().astype(float))
    m = lgb.train(P, lgb.Dataset(X[tr], y[tr], feature_name=names, categorical_feature=cats), 600)
    p = m.predict(X[te])
    r2 = 1 - ((y[te] - p) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()
    out[tgt] = dict(r2=round(float(r2), 4), spearman=round(float(spearmanr(y[te], p).statistic), 4))
    # how much would the PREDICTED value (row-only) help vs the TRUE value?
    w = df["watched"].to_numpy()
    out[tgt]["auc_true_alone"] = round(float(max(roc_auc_score(w[te], y[te]), 1 - roc_auc_score(w[te], y[te]))), 4)
    out[tgt]["auc_rowpred_alone"] = round(float(max(roc_auc_score(w[te], p), 1 - roc_auc_score(w[te], p))), 4)
    T.log(f"{tgt}: {out[tgt]}")
json.dump(out, open(f"{T.OUT}/recoverability.json", "w"), indent=1)
