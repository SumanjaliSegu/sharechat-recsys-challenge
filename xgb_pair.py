"""GPU: pairwise-ranking XGBoost (objective rank:pairwise) on the G feature set - trains on positive-vs-negative
order directly, as ROC-AUC measures it. Query groups are RANDOM blocks of GROUP training rows (not sessions or
requests), so the sampled pairs approximate the global positive/negative pairs of the metric. Prediction is a
plain per-row score. Saves holdout preds (same order as other models) and full-train test predictions."""
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
GROUP = int(__import__("os").environ.get("XGROUP", "1000"))
NPAIR = int(__import__("os").environ.get("XNPAIR", "8"))
ROUNDS = int(__import__("os").environ.get("XROUNDS", "4000"))


def ranked(X, y, seed):
    """Random row order + consecutive blocks of GROUP rows as query ids (sorted, as XGBoost requires)."""
    p = np.random.default_rng(seed).permutation(len(y))
    d = xgb.DMatrix(X.iloc[p], y[p]); d.set_group(np.diff(np.r_[np.arange(0, len(y), GROUP), len(y)]))  # group sizes
    return d


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

XP = dict(objective="rank:pairwise", lambdarank_pair_method="mean", lambdarank_num_pair_per_sample=NPAIR, device="cuda", tree_method="hist", max_depth=8,
          learning_rate=0.03, subsample=0.8, colsample_bytree=0.5, min_child_weight=50, reg_lambda=10.0, max_bin=256, seed=SEED)
dtr = ranked(Xtr, ytr, SEED); dho = xgb.DMatrix(Xho)
bx = xgb.train(XP, dtr, ROUNDS)
curve = {}
for it in range(250, ROUNDS + 1, 250):          # global holdout AUC (the metric), not per-group
    curve[it] = float(roc_auc_score(yho, bx.predict(dho, iteration_range=(0, it)))); T.log(f"iter {it}: {curve[it]:.5f}")
xbest = max(curve, key=curve.get)
px = bx.predict(dho, iteration_range=(0, xbest))
np.save(f"{W}/xgb_holdout_pred_pair_s{SEED}.npy", px.astype(np.float32))
res = dict(xgb_auc=curve[xbest], xgb_iter=int(xbest), curve=curve)
json.dump(res, open(f"{W}/pair_result.json", "w"), indent=1)
T.log(json.dumps(res))
del Xtr, Xho, dtr, dho, bx

tr = T.load_train(np.random.default_rng(42), pl.lit(True)); vocab = T.build_vocab(tr)
X, cats = frame(tr, vocab); y = tr["watched"].to_numpy(); del tr
bx = xgb.train(XP, ranked(X, y, SEED + 100), int(xbest * 1.15) + 1)
del X
pf = pq.ParquetFile(f"{T.DATA}/test.parquet"); ids, a, b = [], [], []
for g in range(pf.metadata.num_row_groups):
    df = pl.from_arrow(pf.read_row_group(g, columns=["id"] + [c for c in T.READ_COLS if c != "watched"]))
    Xt, _ = frame(df, vocab); ids.append(df["id"].to_numpy())
    a.append(bx.predict(xgb.DMatrix(Xt)))
ids = np.concatenate(ids)
pl.DataFrame({"id": ids, "watched": np.concatenate(a)}).write_csv(f"{W}/submission_xgb_pair_s{SEED}.csv")
T.log("wrote pairwise submission")
