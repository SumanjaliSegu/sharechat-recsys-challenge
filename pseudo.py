"""VALIDATION-ONLY experiment: does pseudo-labelling (self-training) help?

The holdout day plays the role of the unlabelled test set: its labels are never used for training,
only its features plus pseudo-labels from a model trained on earlier days. All arms stop early on
the real holdout labels in the same way (same small optimism for every arm; also reported at fixed
rounds). Never used for a submission unless the gain is >= +0.0005 AND the host confirms that
training on unlabelled test rows is allowed.

Arms (LightGBM, feature set G, lr 0.05, all negatives):
  S0   baseline: train days only                                  (reference)
  P1   + holdout rows, soft label = teacher p, weight 1           (cross_entropy)
  P2   + holdout rows, soft label, weight 0.3
  P3   + holdout rows, hard labels for confident rows only (top 1.5% -> 1, bottom 50% -> 0)
  P4   two rounds: P1's predictions become the teacher for a second P1
Teacher = S0's prediction on the holdout rows.
"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
G = ["--emb", "--nopost", "--reg", "--fullho", "--stable", "--nocatid", "--dropE", "--ff=0.5", "--ist"]
sys.argv = ["train_lgb.py", "valid"] + G + ["--neg=1.0", "--lr=0.05"]
sys.path.insert(0, HERE)
import numpy as np
import polars as pl
import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
import train_lgb as T

OUT = T.OUT
rng = np.random.default_rng(42)
tr = T.load_train(rng, pl.col("feature1") < T.TRAIN_END)
ho = T.load_holdout(rng)
vocab = T.build_vocab(tr)
Xtr, names = T.to_np(T.featurize(tr, vocab)); ytr = tr["watched"].to_numpy().astype(np.float32)
Xho, _ = T.to_np(T.featurize(ho, vocab)); yho = ho["watched"].to_numpy()
del tr, ho
cats = [c for c in T.STR_COLS if c in names]
T.log(f"pseudo data: train {len(ytr):,} holdout {len(yho):,} feats {len(names)}")
P = dict(T.PARAMS); P.update(learning_rate=0.05, verbose=-1, num_threads=os.cpu_count())
res = {}
X2 = np.concatenate([Xtr, Xho])          # train rows + unlabelled holdout rows


def fit(X, y, w, objective, nm):
    t = time.time()
    d = lgb.Dataset(X, y, weight=w, feature_name=names, categorical_feature=cats, free_raw_data=True)
    dv = lgb.Dataset(Xho, yho, reference=d, feature_name=names, categorical_feature=cats)
    m = lgb.train(dict(P, objective=objective), d, 4000, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    p = m.predict(Xho, num_iteration=m.best_iteration)
    r = dict(auc=float(roc_auc_score(yho, p)), best_iter=m.best_iteration, secs=round(time.time() - t))
    if "S0_hard" in res:   # same number of rounds as the baseline: no early-stopping advantage
        r["auc_at_S0_iter"] = float(roc_auc_score(yho, m.predict(Xho, num_iteration=res["S0_hard"]["best_iter"])))
    res[nm] = r
    T.log(f"{nm}: {r}")
    np.save(f"{OUT}/hopred_pseudo_{nm}.npy", p.astype(np.float32))
    json.dump(res, open(f"{OUT}/pseudo_result.json", "w"), indent=1)
    return p


teach = fit(Xtr, ytr, None, "binary", "S0_hard")
one = np.ones(len(ytr), np.float32)
p1 = fit(X2, np.concatenate([ytr, teach]).astype(np.float32), None, "cross_entropy", "P1_soft_w1")
fit(X2, np.concatenate([ytr, teach]).astype(np.float32),
    np.concatenate([one, np.full(len(yho), 0.3, np.float32)]), "cross_entropy", "P2_soft_w03")
hi, lo = np.quantile(teach, 0.985), np.quantile(teach, 0.5)
keep = (teach >= hi) | (teach <= lo)
fit(np.concatenate([Xtr, Xho[keep]]), np.concatenate([ytr, (teach[keep] >= hi).astype(np.float32)]),
    None, "binary", "P3_hard_confident")
fit(X2, np.concatenate([ytr, p1]).astype(np.float32), None, "cross_entropy", "P4_soft_round2")
T.log("PSEUDO " + json.dumps(res))
