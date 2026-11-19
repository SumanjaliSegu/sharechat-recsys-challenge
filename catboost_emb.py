"""CatBoost + native embedding features (CPU): the 7 user and 7 post 32-d vectors go in as
CatBoost embedding_features (it builds LDA / nearest-class features from them inside training).
Based on catboost_run.py. CatBoost (GPU) on the G feature set — a third model family for the blend.
valid: train < 2026-10-27 18:00 (all rows), score all holdout rows (same order as NN holdout preds)
full : train on all train, predict test."""
import sys, os, json, time
import numpy as np
sys.argv = ["x", "valid", "--emb", "--nopost", "--reg", "--lr=0.03", "--fullho", "--stable", "--nocatid",
            "--dropE", "--ff=0.5", "--neg=1.0"]
sys.path.insert(0, "/kaggle/working")
import train_lgb as T
import polars as pl
import pyarrow.parquet as pq
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

W = T.OUT
_SEED = int(__import__("os").environ.get("CBSEED", "42"))
SFX = "_embf" + ("" if _SEED == 42 else f"_s{_SEED}")  # default seed keeps the original (frozen-manifest) file names
P = dict(loss_function="Logloss", eval_metric="AUC", task_type="CPU", thread_count=-1, learning_rate=0.08, depth=8,
         l2_leaf_reg=10, border_count=128, random_strength=1.0, bootstrap_type="Bernoulli", subsample=0.8,
         od_type="Iter", od_wait=200, verbose=200, random_seed=_SEED)


EMBC = ["E" + c for c in T.U_EMB + T.P_EMB]


def frame(df, vocab):
    X = T.featurize(df, vocab)
    cats = [c for c in T.STR_COLS if c in X.columns]
    X = X.with_columns([pl.col(c).cast(pl.Int32) for c in cats]).to_pandas()
    for c in T.U_EMB + T.P_EMB:
        a = df[c].fill_null(pl.lit([0.0] * 32, dtype=pl.List(pl.Float32))).list.to_array(32).to_numpy().astype(np.float32)
        X["E" + c] = list(a)
    return X, cats


rng = np.random.default_rng(42)
tr = T.load_train(rng, pl.col("feature1") < T.CUT)
ho = T.load_holdout(rng)
vocab = T.build_vocab(tr)
Xtr, cats = frame(tr, vocab); ytr = tr["watched"].to_numpy()
Xho, _ = frame(ho, vocab); yho = ho["watched"].to_numpy()
del tr, ho
T.log(f"catboost valid: train {len(ytr):,} holdout {len(yho):,} feats {Xtr.shape[1]} cats {cats}")
m = CatBoostClassifier(iterations=3000, **P)
m.fit(Pool(Xtr, ytr, cat_features=cats, embedding_features=EMBC), eval_set=Pool(Xho, yho, cat_features=cats, embedding_features=EMBC), use_best_model=True)
p = m.predict_proba(Pool(Xho, cat_features=cats, embedding_features=EMBC))[:, 1]
best = m.get_best_iteration()
res = dict(auc_holdout=float(roc_auc_score(yho, p)), best_iter=int(best))
T.log(json.dumps(res)); json.dump(res, open(f"{W}/cb_valid{SFX}.json", "w"))
np.save(f"{W}/cb_holdout_pred{SFX}.npy", p.astype(np.float32))
del Xtr, Xho, m

tr = T.load_train(np.random.default_rng(42), pl.lit(True))
vocab = T.build_vocab(tr)
X, cats = frame(tr, vocab); y = tr["watched"].to_numpy(); del tr
n_iter = int(best * 1.15) + 1
T.log(f"catboost full: {len(y):,} rows, {n_iter} iters")
m = CatBoostClassifier(iterations=n_iter, **{k: v for k, v in P.items() if k not in ("od_type", "od_wait")})
m.fit(Pool(X, y, cat_features=cats, embedding_features=EMBC)); del X
pf = pq.ParquetFile(f"{T.DATA}/test.parquet"); ids, preds = [], []
for g in range(pf.metadata.num_row_groups):
    df = pl.from_arrow(pf.read_row_group(g, columns=["id"] + [c for c in T.READ_COLS if c != "watched"]))
    Xt, _ = frame(df, vocab); ids.append(df["id"].to_numpy()); preds.append(m.predict_proba(Pool(Xt, cat_features=cats, embedding_features=EMBC))[:, 1])
sub = pl.DataFrame({"id": np.concatenate(ids), "watched": np.concatenate(preds)})
assert sub.height == 1_037_090
sub.write_csv(f"{W}/submission_catboost_G{SFX}.csv"); T.log("wrote submission")
