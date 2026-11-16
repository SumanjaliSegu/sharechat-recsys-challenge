"""LightGBM hyper-parameter search on the far-ahead check (version C features).
Runs on Kaggle CPU (or locally). Loads data once, then tries each config."""
import sys, os, json, time
import numpy as np
sys.argv = ["x", "valid", "--emb", "--nopost", "--reg", "--lr=0.03", "--fullho", "--far", "--stable", "--nocatid"]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
if os.path.isdir("/kaggle/working"):
    sys.path.insert(0, "/kaggle/working")
import train_lgb as T
import polars as pl
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(T.SEED)
tr = T.load_train(rng, pl.col("feature1") < T.TRAIN_END)
ho = T.load_holdout(rng)
vocab = T.build_vocab(tr)
Xtr, names = T.to_np(T.featurize(tr, vocab)); ytr = tr["watched"].to_numpy()
Xho, _ = T.to_np(T.featurize(ho, vocab)); yho = ho["watched"].to_numpy()
del tr, ho
cats = [c for c in T.STR_COLS if c in names]
T.log(f"train {len(ytr):,} holdout {len(yho):,} feats {len(names)}")
dtr = lgb.Dataset(Xtr, ytr, feature_name=names, categorical_feature=cats, free_raw_data=False)
dho = lgb.Dataset(Xho, yho, reference=dtr, feature_name=names, categorical_feature=cats, free_raw_data=False)

base = dict(T.PARAMS)
grid = [
    {},  # current C params (reference)
    dict(num_leaves=31, min_data_in_leaf=800),
    dict(num_leaves=127, min_data_in_leaf=800),
    dict(num_leaves=63, min_data_in_leaf=2000),
    dict(num_leaves=63, min_data_in_leaf=300),
    dict(feature_fraction=0.15),
    dict(feature_fraction=0.5),
    dict(lambda_l2=100.0),
    dict(lambda_l2=3.0),
    dict(min_gain_to_split=1.0),
    dict(max_bin=255),
    dict(num_leaves=31, min_data_in_leaf=2000, feature_fraction=0.2, lambda_l2=100.0),
    dict(extra_trees=True),
    dict(path_smooth=50.0),
]
res = []
for i, g in enumerate(grid):
    p = dict(base); p.update(g)
    t0 = time.time()
    m = lgb.train(p, dtr, 4000, valid_sets=[dho], callbacks=[lgb.early_stopping(150, verbose=False)])
    auc = roc_auc_score(yho, m.predict(Xho, num_iteration=m.best_iteration))
    res.append(dict(i=i, change=g, auc=float(auc), best_iter=m.best_iteration, secs=round(time.time() - t0)))
    T.log(f"[{i}] {g} -> AUC {auc:.5f} iters {m.best_iteration} ({res[-1]['secs']}s)")
    json.dump(res, open(os.path.join(T.OUT, "tune_results.json"), "w"), indent=1)
best = max(res, key=lambda r: r["auc"])
T.log(f"BEST {best}")
