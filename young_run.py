"""Young-post experiments (post age < 72 h, computed from the row's own timestamp and upload time).
Arms per seed: global G · G with young rows upweighted x3 · global + young-only specialist (blended
only on young rows). Scored on the standard check, plain and re-weighted to test's new-post mix."""
import sys, json
import numpy as np
sys.argv = ["x", "valid", "--emb", "--nopost", "--reg", "--lr=0.03", "--fullho", "--stable", "--nocatid",
            "--dropE", "--ff=0.5"]
sys.path.insert(0, "/kaggle/working")
import train_lgb as T
import polars as pl
import lightgbm as lgb
from sklearn.metrics import roc_auc_score as A

for _c in ("feature18", "feature58"):  # read only (to mark new / young posts), not model inputs
    if _c not in T.READ_COLS:
        T.READ_COLS.append(_c)
rng = np.random.default_rng(42)
tr = T.load_train(rng, pl.col("feature1") < T.CUT); ho = T.load_holdout(rng)
age = lambda d: ((d["feature1"].dt.epoch("s").cast(pl.Float64) - d["feature58"].cast(pl.Float64)) / 3600).to_numpy()
ytr_young, yho_young = age(tr) < 72, age(ho) < 72
# test-mix weights: holdout posts unseen in training rows -> weight to test's 16.1% new-post share
seen = ho["feature18"].is_in(tr["feature18"].unique().implode()).to_numpy()
w = np.where(seen, (1 - 0.161) / seen.mean(), 0.161 / (1 - seen.mean()))
vocab = T.build_vocab(tr)
Xtr, names = T.to_np(T.featurize(tr, vocab)); ytr = tr["watched"].to_numpy()
Xho, _ = T.to_np(T.featurize(ho, vocab)); yho = ho["watched"].to_numpy(); del tr, ho
cats = [c for c in T.STR_COLS if c in names]
T.log(f"young share train {ytr_young.mean():.3f} holdout {yho_young.mean():.3f} | holdout new-post {1 - seen.mean():.3f}")


def fit(mask, weight, seed):
    p = dict(T.PARAMS); p["seed"] = seed
    dtr = lgb.Dataset(Xtr[mask], ytr[mask], weight=None if weight is None else weight[mask], feature_name=names, categorical_feature=cats)
    dho = lgb.Dataset(Xho, yho, reference=dtr, feature_name=names, categorical_feature=cats)
    m = lgb.train(p, dtr, 4000, valid_sets=[dho], callbacks=[lgb.early_stopping(150, verbose=False)])
    return m.predict(Xho, num_iteration=m.best_iteration)


def sc(p):
    return dict(plain=A(yho, p), testmix=A(yho, p, sample_weight=w), young=A(yho[yho_young], p[yho_young]),
                new=A(yho[~seen], p[~seen]))


res = {}
allm = np.ones(len(ytr), bool)
for seed in (42, 7):
    g = fit(allm, None, seed)
    up = fit(allm, np.where(ytr_young, 3.0, 1.0).astype(np.float32), seed)
    sp = fit(ytr_young, None, seed)
    for wy in (0.3, 0.5):
        res.setdefault(f"spec_blend_{wy}", []).append(sc(np.where(yho_young, (1 - wy) * g + wy * sp, g)))
    res.setdefault("global", []).append(sc(g)); res.setdefault("young_x3", []).append(sc(up))
    T.log(f"seed {seed} done")
out = {k: {m: round(float(np.mean([r[m] for r in v])), 5) for m in v[0]} for k, v in res.items()}
T.log(json.dumps(out, indent=1)); json.dump(out, open(f"{T.OUT}/young_results.json", "w"), indent=1)
