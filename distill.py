"""VALIDATION-ONLY experiment: does request information used *during training only* help a row-only model?

Never used for a submission unless the host confirms that request_id may be used in training
(forum thread 759936). Inference of every student model is strictly row-independent.

Arms (all students see only row-local features; scored on the time holdout):
  S0  baseline: hard labels                                   (reference)
  S1  distillation: soft target 0.5*y + 0.5*teacher            (cross_entropy objective)
  S2  distillation: soft target = teacher
  L1  ranking loss: lambdarank grouped by request (feature5 = request/session id)
Teacher = LightGBM on row features + request size / position / rows-per-user, out-of-fold over
4 folds grouped by feature5, so a training row's soft label never comes from a model that saw it.
"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
G = ["--emb", "--nopost", "--reg", "--fullho", "--stable", "--nocatid", "--dropE", "--ff=0.5", "--ist"]
sys.argv = ["train_lgb.py", "valid"] + G + ["--xrow", "--neg=1.0", "--lr=0.05"]
sys.path.insert(0, HERE)
import numpy as np
import polars as pl
import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
import train_lgb as T

OUT = T.OUT
rng = np.random.default_rng(42)
T.XTAB = T.build_xrow()
tr = T.load_train(rng, pl.col("feature1") < T.TRAIN_END)
ho = T.load_holdout(rng)
vocab = T.build_vocab(tr)
grp = (tr["feature5"].hash(seed=11) % 1_000_000_007).to_numpy()
Xtr, names = T.to_np(T.featurize(tr, vocab)); ytr = tr["watched"].to_numpy().astype(np.float32)
Xho, _ = T.to_np(T.featurize(ho, vocab)); yho = ho["watched"].to_numpy()
del tr, ho
XR = [i for i, n in enumerate(names) if n.startswith("x_")]
ROW = [i for i, n in enumerate(names) if not n.startswith("x_")]
rnames = [names[i] for i in ROW]
cats_all = [c for c in T.STR_COLS if c in names]
cats_row = [c for c in T.STR_COLS if c in rnames]
T.log(f"distill data: train {len(ytr):,} holdout {len(yho):,} feats {len(names)} (request feats {[names[i] for i in XR]})")
P = dict(T.PARAMS); P.update(learning_rate=0.05, verbose=-1, num_threads=os.cpu_count())
res = {}


def fit(X, y, Xv, yv, params, cats, nm, group=None, rounds=4000):
    t = time.time()
    d = lgb.Dataset(X, y, feature_name=nm, categorical_feature=cats, group=group, free_raw_data=True)
    dv = lgb.Dataset(Xv, yv, reference=d, feature_name=nm, categorical_feature=cats,
                     group=None if group is None else [len(yv)])
    m = lgb.train(params, d, rounds, valid_sets=[dv], callbacks=[lgb.early_stopping(150, verbose=False)])
    T.log(f"  fit done in {time.time() - t:.0f}s, best iter {m.best_iteration}")
    return m


# ---- teacher, out-of-fold (4 folds grouped by request) ----
fold = grp % 4
teach = np.zeros(len(ytr), np.float32)
for f in range(4):
    tri, vai = np.nonzero(fold != f)[0], np.nonzero(fold == f)[0]
    m = fit(Xtr[tri], ytr[tri], Xtr[vai], ytr[vai], P, cats_all, names)
    teach[vai] = m.predict(Xtr[vai], num_iteration=m.best_iteration)
    T.log(f"teacher fold {f}: OOF AUC {roc_auc_score(ytr[vai], teach[vai]):.5f}")
    del m
res["teacher_oof_auc"] = float(roc_auc_score(ytr, teach))
T.log(f"teacher OOF AUC (with request features) {res['teacher_oof_auc']:.5f}")
np.save(f"{OUT}/distill_teacher_oof.npy", teach)

# ---- students: row-only features ----
Xr, Xhr = Xtr[:, ROW], Xho[:, ROW]
del Xtr, Xho
preds = {}
arms = {"S0_hard": ytr, "S1_soft50": 0.5 * ytr + 0.5 * teach, "S2_soft100": teach}
for arm, target in arms.items():
    p = dict(P, objective="binary" if arm == "S0_hard" else "cross_entropy")
    m = fit(Xr, target.astype(np.float32), Xhr, yho, p, cats_row, rnames)
    preds[arm] = m.predict(Xhr, num_iteration=m.best_iteration)
    res[arm] = dict(auc=float(roc_auc_score(yho, preds[arm])), best_iter=m.best_iteration)
    T.log(f"{arm}: holdout AUC {res[arm]['auc']:.5f}")
    np.save(f"{OUT}/hopred_distill_{arm}.npy", preds[arm].astype(np.float32))
    json.dump(res, open(f"{OUT}/distill_result.json", "w"), indent=1)
    del m

# ---- ranking loss grouped by request ----
o = np.argsort(grp, kind="stable")
_, sizes = np.unique(grp[o], return_counts=True)
p = dict(P, objective="lambdarank", metric="auc", lambdarank_truncation_level=12, label_gain=[0, 1])
d = lgb.Dataset(Xr[o], ytr[o].astype(int), group=sizes, feature_name=rnames, categorical_feature=cats_row)
dv = lgb.Dataset(Xhr, yho, reference=d, feature_name=rnames, categorical_feature=cats_row, group=[len(yho)])
m = lgb.train(p, d, 3000, valid_sets=[dv], callbacks=[lgb.early_stopping(150, verbose=False)])
preds["L1_lambdarank"] = m.predict(Xhr, num_iteration=m.best_iteration)
res["L1_lambdarank"] = dict(auc=float(roc_auc_score(yho, preds["L1_lambdarank"])), best_iter=m.best_iteration)
np.save(f"{OUT}/hopred_distill_L1_lambdarank.npy", preds["L1_lambdarank"].astype(np.float32))

R = lambda v: rankdata(v) / len(v)
for arm in ("S1_soft50", "S2_soft100", "L1_lambdarank"):
    res[f"{arm}+S0_rankavg"] = float(roc_auc_score(yho, R(preds[arm]) + R(preds["S0_hard"])))
T.log("DISTILL " + json.dumps(res))
json.dump(res, open(f"{OUT}/distill_result.json", "w"), indent=1)
