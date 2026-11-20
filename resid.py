"""Residual specialist: LightGBM that starts from each video's training-data watch rate and learns only the rest.

init_score = A * logit(smoothed video watch rate from training data) (te.py, out-of-fold for training rows,
full training maps for holdout / test rows). The trees therefore spend their capacity on "which user
watches this video" (our weakest spot, inside-video AUC 0.737) instead of on video popularity.
Allowed: video statistics come from training labels only; every row is predicted from its own columns.
valid: arms A = 1.0 and 0.5 (holdout AUC of trees + offset); full: best arm on all train -> test file.
"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
STAGE = sys.argv[1] if len(sys.argv) > 1 else "valid"
G = ["--emb", "--nopost", "--reg", "--fullho", "--stable", "--nocatid", "--dropE", "--ff=0.5", "--lr=0.03", "--ist", "--te"]
sys.argv = ["train_lgb.py", STAGE] + G
sys.path.insert(0, HERE)
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import train_lgb as T
import te as temod

OUT = T.OUT
KEY = "te_r_feature18"
logit = lambda p: np.log(np.clip(p, 1e-5, 1 - 1e-5) / (1 - np.clip(p, 1e-5, 1 - 1e-5)))


def split(X, names):
    i = names.index(KEY)
    keep = [j for j, n in enumerate(names) if not n.startswith("te_")]
    return X[:, keep], [names[j] for j in keep], logit(X[:, i].astype(np.float64))


def fit(Xa, y, init, nm, cats, rounds, valid=None):
    d = lgb.Dataset(Xa, y, init_score=init, feature_name=nm, categorical_feature=cats, free_raw_data=True)
    kw = {}
    if valid is not None:
        Xv, yv, iv = valid
        kw = dict(valid_sets=[lgb.Dataset(Xv, yv, init_score=iv, reference=d, feature_name=nm, categorical_feature=cats)],
                  callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(100)])
    return lgb.train(T.PARAMS, d, rounds, **kw)


T.TETAB, T.TEMAPS = T.build_te(STAGE)
rng = np.random.default_rng(T.SEED)
res_f = f"{OUT}/resid_result.json"
if STAGE == "valid":
    tr = T.load_train(rng, pl.col("feature1") < T.TRAIN_END); ho = T.load_holdout(rng)
    vocab = T.build_vocab(tr)
    Xtr, names = T.to_np(T.featurize(tr, vocab)); ytr = tr["watched"].to_numpy()
    Xho, _ = T.to_np(T.featurize(ho, vocab)); yho = ho["watched"].to_numpy(); del tr, ho
    Xtr, nm, itr = split(Xtr, names); Xho, _, iho = split(Xho, names)
    cats = [c for c in T.STR_COLS if c in nm]
    res = {"video_rate_only_auc": float(roc_auc_score(yho, iho))}
    for A in (1.0, 0.5):
        t = time.time()
        m = fit(Xtr, ytr, A * itr, nm, cats, 4000, valid=(Xho, yho, A * iho))
        p = m.predict(Xho, num_iteration=m.best_iteration, raw_score=True) + A * iho
        res[f"A{A}"] = dict(auc=float(roc_auc_score(yho, p)), best_iter=m.best_iteration, secs=round(time.time() - t))
        np.save(f"{OUT}/hopred_resid_A{A}.npy", p.astype(np.float32))
        T.log(f"resid A={A}: {res[f'A{A}']}"); json.dump(res, open(res_f, "w"), indent=1)
else:
    res = json.load(open(res_f))
    A = max((1.0, 0.5), key=lambda a: res[f"A{a}"]["auc"]); it = int(1.15 * res[f"A{A}"]["best_iter"]) + 1
    tr = T.load_train(rng, pl.lit(True)); vocab = T.build_vocab(tr)
    X, names = T.to_np(T.featurize(tr, vocab)); y = tr["watched"].to_numpy(); del tr
    X, nm, ini = split(X, names); cats = [c for c in T.STR_COLS if c in nm]
    T.log(f"resid full: A={A}, {it} rounds, {len(y):,} rows")
    m = fit(X, y, A * ini, nm, cats, it); del X
    pf = pq.ParquetFile(f"{T.DATA}/test.parquet"); ids, preds = [], []
    tcols = ["id"] + [c for c in T.READ_COLS if c != "watched"]
    tcols += [k for k in temod.TE_KEYS if k not in tcols]
    for g in range(pf.metadata.num_row_groups):
        df = pl.from_arrow(pf.read_row_group(g, columns=tcols))
        df = pl.concat([df, temod.apply_maps(df, *T.TEMAPS)], how="horizontal")
        Xt, tn = T.to_np(T.featurize(df, vocab)); Xt, _, it_ = split(Xt, tn)
        ids.append(df["id"].to_numpy()); preds.append(m.predict(Xt, raw_score=True) + A * it_)
    p = np.concatenate(preds); p = 1 / (1 + np.exp(-p))
    sub = pl.DataFrame({"id": np.concatenate(ids), "watched": p})
    assert sub.height == 1_037_090
    sub.write_csv(f"{OUT}/submission_resid_A{A}.csv"); T.log("wrote resid submission")
