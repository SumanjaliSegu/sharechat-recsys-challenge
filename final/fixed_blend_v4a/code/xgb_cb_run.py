"""GPU: XGBoost + CatBoost(depth 10) on the G feature set. Each saves holdout preds (standard check,
all holdout rows, same order as other models) and full-train test predictions."""
import sys, json
import numpy as np
sys.argv = ["x", "valid", "--emb", "--nopost", "--reg", "--lr=0.03", "--fullho", "--stable", "--nocatid",
            "--dropE", "--ff=0.5", "--neg=1.0", "--ist"]
sys.path.insert(0, "/kaggle/working")
import train_lgb as T
import polars as pl
import pyarrow.parquet as pq
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

W = T.OUT


def frame(df, vocab):
    X = T.featurize(df, vocab)
    cats = [c for c in T.STR_COLS if c in X.columns]
    return X.with_columns([pl.col(c).cast(pl.Int32) for c in cats]).to_pandas(), cats


rng = np.random.default_rng(42)
tr = T.load_train(rng, pl.col("feature1") < T.CUT); ho = T.load_holdout(rng)
vocab = T.build_vocab(tr)
Xtr, cats = frame(tr, vocab); ytr = tr["watched"].to_numpy()
Xho, _ = frame(ho, vocab); yho = ho["watched"].to_numpy(); del tr, ho
T.log(f"valid: train {len(ytr):,} holdout {len(yho):,} feats {Xtr.shape[1]}")

XP = dict(objective="binary:logistic", eval_metric="auc", device="cuda", tree_method="hist", max_depth=8,
          learning_rate=0.03, subsample=0.8, colsample_bytree=0.5, min_child_weight=50, reg_lambda=10.0, max_bin=256)
dtr = xgb.DMatrix(Xtr, ytr); dho = xgb.DMatrix(Xho, yho)
bx = xgb.train(XP, dtr, 6000, evals=[(dho, "ho")], early_stopping_rounds=200, verbose_eval=250)
px = bx.predict(dho, iteration_range=(0, bx.best_iteration + 1)); xbest = bx.best_iteration
np.save(f"{W}/xgb_holdout_pred.npy", px.astype(np.float32))
CP = dict(loss_function="Logloss", eval_metric="AUC", task_type="GPU", learning_rate=0.05, depth=10, l2_leaf_reg=20,
          border_count=128, bootstrap_type="Bernoulli", subsample=0.8, random_seed=7, verbose=500)
cb = CatBoostClassifier(iterations=6000, od_type="Iter", od_wait=200, **CP)
cb.fit(Pool(Xtr, ytr, cat_features=cats), eval_set=Pool(Xho, yho, cat_features=cats), use_best_model=True)
pc = cb.predict_proba(Xho)[:, 1]; cbest = cb.get_best_iteration()
np.save(f"{W}/cb2_holdout_pred.npy", pc.astype(np.float32))
res = dict(xgb_auc=float(roc_auc_score(yho, px)), xgb_iter=int(xbest), cb2_auc=float(roc_auc_score(yho, pc)), cb2_iter=int(cbest))
T.log(json.dumps(res)); json.dump(res, open(f"{W}/xgb_cb_valid.json", "w"))
del Xtr, Xho, dtr, dho, bx, cb

tr = T.load_train(np.random.default_rng(42), pl.lit(True)); vocab = T.build_vocab(tr)
X, cats = frame(tr, vocab); y = tr["watched"].to_numpy(); del tr
bx = xgb.train(XP, xgb.DMatrix(X, y), int(xbest * 1.15) + 1)
cb = CatBoostClassifier(iterations=int(cbest * 1.15) + 1, **CP); cb.fit(Pool(X, y, cat_features=cats)); del X
pf = pq.ParquetFile(f"{T.DATA}/test.parquet"); ids, a, b = [], [], []
for g in range(pf.metadata.num_row_groups):
    df = pl.from_arrow(pf.read_row_group(g, columns=["id"] + [c for c in T.READ_COLS if c != "watched"]))
    Xt, _ = frame(df, vocab); ids.append(df["id"].to_numpy())
    a.append(bx.predict(xgb.DMatrix(Xt))); b.append(cb.predict_proba(Xt)[:, 1])
ids = np.concatenate(ids)
pl.DataFrame({"id": ids, "watched": np.concatenate(a)}).write_csv(f"{W}/submission_xgb.csv")
pl.DataFrame({"id": ids, "watched": np.concatenate(b)}).write_csv(f"{W}/submission_cb2.csv")
T.log("wrote submission_xgb.csv, submission_cb2.csv")
