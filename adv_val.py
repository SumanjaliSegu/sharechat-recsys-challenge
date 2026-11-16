"""Adversarial validation: can a model tell train rows from test rows?
High AUC + its top features = columns that drift over time."""
import sys
import numpy as np, polars as pl, pyarrow.parquet as pq, lightgbm as lgb
from sklearn.metrics import roc_auc_score
sys.argv = ["x", "valid", "--emb", "--nopost"]  # reuse feature code (v2 features)
sys.path.insert(0, "D:/sharechat")
import train_lgb as T

rng = np.random.default_rng(0)


def sample(path, n_per_rg, cols):
    pf = pq.ParquetFile(path); parts = []
    for g in range(pf.metadata.num_row_groups):
        df = pl.from_arrow(pf.read_row_group(g, columns=cols))
        idx = rng.choice(df.height, min(n_per_rg, df.height), replace=False)
        parts.append(df[np.sort(idx)])
    return pl.concat(parts)


cols = [c for c in T.READ_COLS if c != "watched"]
tr = sample(f"{T.DATA}/train.parquet", 12500, cols + ["watched"])
te = sample(f"{T.DATA}/test.parquet", 18200, cols)
vocab = T.build_vocab(tr)
X = pl.concat([T.featurize(tr, vocab), T.featurize(te, vocab)])
X = X.drop([c for c in ("hour",) if c in X.columns])  # time-of-day is not drift
y = np.r_[np.zeros(tr.height), np.ones(te.height)]
print("rows train", tr.height, "test", te.height, "features", X.width)
Xn, names = T.to_np(X)
cats = [c for c in T.STR_COLS if c in names]
perm = rng.permutation(len(y)); cut = int(0.7 * len(y))
a, b = perm[:cut], perm[cut:]
p = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
         feature_fraction=0.5, verbose=-1, num_threads=8, max_cat_to_onehot=8)
m = lgb.train(p, lgb.Dataset(Xn[a], y[a], feature_name=names, categorical_feature=cats), 300)
auc = roc_auc_score(y[b], m.predict(Xn[b]))
imp = pl.DataFrame({"feat": names, "gain": m.feature_importance("gain")}).sort("gain", descending=True)
imp = imp.with_columns((pl.col("gain") / pl.col("gain").sum() * 100).round(2).alias("gain%"))
print(f"ADVERSARIAL AUC = {auc:.4f}")
pl.Config.set_tbl_rows(30)
print(imp.head(25).select("feat", "gain%"))
imp.write_csv("D:/sharechat/out/adv_importance.csv")

# per-feature drift: single-feature AUC train vs test for top 25
print("\nsingle-feature train-vs-test AUC (0.5 = no drift):")
for f in imp["feat"].head(25).to_list():
    x = X[f].cast(pl.Float64).fill_null(-1e30).to_numpy()
    s = roc_auc_score(y, x); print(f"  {f:28s} {max(s, 1 - s):.3f}")
