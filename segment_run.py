"""Segment specialists: does a model trained only on the high-watch user group (by feature10)
rank that group better? Compares global G model vs per-segment models vs their average."""
import sys, json
import numpy as np
sys.argv = ["x", "valid", "--emb", "--nopost", "--reg", "--lr=0.03", "--fullho", "--stable", "--nocatid",
            "--dropE", "--ff=0.5"]
sys.path.insert(0, "/kaggle/working")
import train_lgb as T
import polars as pl
import lightgbm as lgb
from sklearn.metrics import roc_auc_score as A

rng = np.random.default_rng(42)
tr = T.load_train(rng, pl.col("feature1") < T.CUT)
ho = T.load_holdout(rng)
# segment rule learned on training rows only: feature10 values with above-median watch rate
rates = tr.group_by("feature10").agg(pl.col("watched").mean().alias("r"), pl.len().alias("n")).filter(pl.col("n") > 200)
thr = float(rates["r"].median())
hi_vals = [v for v in rates.filter(pl.col("r") > thr)["feature10"].to_list() if v is not None]
seg_tr = tr["feature10"].is_in(hi_vals).fill_null(False).to_numpy().astype(bool)
seg_ho = ho["feature10"].is_in(hi_vals).fill_null(False).to_numpy().astype(bool)
T.log(f"high-watch values {len(hi_vals)} | train share {seg_tr.mean():.3f} holdout share {seg_ho.mean():.3f}")

vocab = T.build_vocab(tr)
Xtr, names = T.to_np(T.featurize(tr, vocab)); ytr = tr["watched"].to_numpy()
Xho, _ = T.to_np(T.featurize(ho, vocab)); yho = ho["watched"].to_numpy()
del tr, ho
cats = [c for c in T.STR_COLS if c in names]


def fit(mask_tr, mask_ho):
    dtr = lgb.Dataset(Xtr[mask_tr], ytr[mask_tr], feature_name=names, categorical_feature=cats)
    dho = lgb.Dataset(Xho[mask_ho], yho[mask_ho], reference=dtr, feature_name=names, categorical_feature=cats)
    m = lgb.train(T.PARAMS, dtr, 4000, valid_sets=[dho], callbacks=[lgb.early_stopping(150, verbose=False)])
    return m


allt, allh = np.ones(len(ytr), bool), np.ones(len(yho), bool)
g = fit(allt, allh); pg = g.predict(Xho, num_iteration=g.best_iteration)
mA = fit(seg_tr, seg_ho); mB = fit(~seg_tr, ~seg_ho)
ps = np.where(seg_ho, mA.predict(Xho, num_iteration=mA.best_iteration), mB.predict(Xho, num_iteration=mB.best_iteration))
res = {
    "global": A(yho, pg),
    "global_in_highseg": A(yho[seg_ho], pg[seg_ho]),
    "global_in_lowseg": A(yho[~seg_ho], pg[~seg_ho]),
    "specialists": A(yho, ps),
    "spec_in_highseg": A(yho[seg_ho], ps[seg_ho]),
    "spec_in_lowseg": A(yho[~seg_ho], ps[~seg_ho]),
}
for w in (0.3, 0.5, 0.7):
    res[f"avg_w{w}"] = A(yho, (1 - w) * pg + w * ps)
res = {k: round(float(v), 5) for k, v in res.items()}
T.log(json.dumps(res))
json.dump(res, open(f"{T.OUT}/segment_results.json", "w"), indent=1)
np.save(f"{T.OUT}/seg_holdout_global.npy", pg.astype(np.float32)); np.save(f"{T.OUT}/seg_holdout_spec.npy", ps.astype(np.float32))
