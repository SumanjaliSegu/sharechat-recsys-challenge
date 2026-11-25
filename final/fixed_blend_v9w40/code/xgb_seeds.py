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
from sklearn.metrics import roc_auc_score

W = T.OUT
SEED = int(__import__("os").environ.get("XSEED", "1"))


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
          learning_rate=0.03, subsample=0.8, colsample_bytree=0.5, min_child_weight=50, reg_lambda=10.0, max_bin=256, seed=SEED)
dtr = xgb.DMatrix(Xtr, ytr); dho = xgb.DMatrix(Xho, yho)
bx = xgb.train(XP, dtr, 6000, evals=[(dho, "ho")], early_stopping_rounds=200, verbose_eval=250)
px = bx.predict(dho, iteration_range=(0, bx.best_iteration + 1)); xbest = bx.best_iteration
np.save(f"{W}/xgb_holdout_pred_s{SEED}.npy", px.astype(np.float32))
res = dict(xgb_auc=float(roc_auc_score(yho, px)), xgb_iter=int(xbest))
T.log(json.dumps(res))
del Xtr, Xho, dtr, dho, bx

tr = T.load_train(np.random.default_rng(42), pl.lit(True)); vocab = T.build_vocab(tr)
X, cats = frame(tr, vocab); y = tr["watched"].to_numpy(); del tr
bx = xgb.train(XP, xgb.DMatrix(X, y), int(xbest * 1.15) + 1)
del X
pf = pq.ParquetFile(f"{T.DATA}/test.parquet"); ids, a, b = [], [], []
for g in range(pf.metadata.num_row_groups):
    df = pl.from_arrow(pf.read_row_group(g, columns=["id"] + [c for c in T.READ_COLS if c != "watched"]))
    Xt, _ = frame(df, vocab); ids.append(df["id"].to_numpy())
    a.append(bx.predict(xgb.DMatrix(Xt)))
ids = np.concatenate(ids)
pl.DataFrame({"id": ids, "watched": np.concatenate(a)}).write_csv(f"{W}/submission_xgb_s{SEED}.csv")
T.log("wrote submission_xgb.csv, submission_cb2.csv")
