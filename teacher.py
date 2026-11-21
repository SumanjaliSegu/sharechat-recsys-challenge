"""Teacher for training-time-only distillation (host, forum 759936 / 760657: "The model which predicts on the
test dataset must not use features derived from request_id/request grouping").

The TEACHER (LightGBM, G features + request size / rows-per-user / position) is used only to make soft training
targets. It never scores test rows. The STUDENT (PLR network, nn2_multi.py with SOFT=...) predicts the test set
from each row's own features only.
Out-of-fold over 4 folds grouped by feature5 (session): a row's soft target never comes from a model that saw it.
Outputs: teach_valid.parquet (rows < 2026-10-27 18:00) and teach_full.parquet (all train rows): columns rn, t.
"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
G = ["--emb", "--nopost", "--reg", "--fullho", "--stable", "--nocatid", "--dropE", "--ff=0.5", "--ist"]
sys.argv = ["train_lgb.py", "valid"] + G + ["--xrow", "--neg=1.0", "--lr=0.05"]
sys.path.insert(0, HERE)
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import train_lgb as T

OUT = T.OUT
T.XTAB = T.build_xrow()
P = dict(T.PARAMS); P.update(learning_rate=0.05, verbose=-1, num_threads=os.cpu_count())
ts = pl.read_parquet(f"{T.DATA}/train.parquet", columns=["feature1"])["feature1"]
info = {}
for stage, filt, mask in (("valid", pl.col("feature1") < T.CUT, (ts < T.CUT).to_numpy()),
                          ("full", pl.lit(True), np.ones(len(ts), bool))):
    t0 = time.time()
    tr = T.load_train(np.random.default_rng(42), filt)          # neg=1.0 -> every row, file order
    rn = np.nonzero(mask)[0]
    assert tr.height == len(rn), (tr.height, len(rn))
    grp = (tr["feature5"].hash(seed=11) % 4).to_numpy()
    vocab = T.build_vocab(tr)
    X, names = T.to_np(T.featurize(tr, vocab)); y = tr["watched"].to_numpy(); del tr
    cats = [c for c in T.STR_COLS if c in names]
    teach = np.zeros(len(y), np.float32)
    for f in range(4):
        a, b = np.nonzero(grp != f)[0], np.nonzero(grp == f)[0]
        d = lgb.Dataset(X[a], y[a], feature_name=names, categorical_feature=cats, free_raw_data=True)
        dv = lgb.Dataset(X[b], y[b], reference=d, feature_name=names, categorical_feature=cats)
        m = lgb.train(P, d, 4000, valid_sets=[dv], callbacks=[lgb.early_stopping(150, verbose=False)])
        teach[b] = m.predict(X[b], num_iteration=m.best_iteration)
        T.log(f"{stage} teacher fold {f}: AUC {roc_auc_score(y[b], teach[b]):.5f} (iter {m.best_iteration})")
        del m, d, dv
    info[stage] = dict(oof_auc=float(roc_auc_score(y, teach)), rows=int(len(y)), secs=round(time.time() - t0))
    T.log(f"{stage} teacher OOF AUC {info[stage]['oof_auc']:.5f}")
    pl.DataFrame({"rn": rn.astype(np.uint32), "t": teach}).write_parquet(f"{OUT}/teach_{stage}.parquet")
    json.dump(info, open(f"{OUT}/teacher_info.json", "w"), indent=1)
    del X
