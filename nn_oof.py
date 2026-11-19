"""Hybrid step 1: out-of-fold towers-network scores, used as one extra LightGBM input (train_lgb.py --nnf).

Row-independent: the score of a row is a function of that row's own features only (a trained network).
Out-of-fold: a training row's score always comes from a network that did not see that row.
  valid stage: 5 random folds over rows < 2026-10-27 18:00 -> OOF for those rows; holdout rows get the
               mean of the 5 fold networks (none of them saw holdout rows).
  full stage : 5 random folds over all train rows -> OOF; test rows get the mean of the 5 fold networks.
Outputs (OUT): nnf_valid.parquet (rn, h_nn), nnf_full.parquet (rn, h_nn), nnf_test.parquet (id, h_nn),
nnf_info.json (fold / holdout AUCs). rn = row number in train.parquet.
"""
import json, os
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
import nn2_multi as N

EPOCHS = int(os.environ.get("NNF_EPOCHS", "4"))
K = 5
HP = dict(lr=1e-3, bs=8192, wd=1e-4, drop=0.3, H=192, nopost=False, gamma=0.0, ls=0.0, arch="towers")
info = {}


def oof(X, y, Xo, cards, tag):
    """Returns OOF scores for X rows and fold-mean scores for Xo rows."""
    n = len(y)
    fold = np.random.default_rng(2026).integers(0, K, n)
    o = np.zeros(n, np.float32); po = np.zeros(len(next(iter(Xo.values()))), np.float64)
    for k in range(K):
        tr, va = np.nonzero(fold != k)[0], np.nonzero(fold == k)[0]
        sub = lambda d, i: {kk: v[i] for kk, v in d.items()}
        model, _ = N.train(sub(X, tr), y[tr], cards, EPOCHS, seed=k, **HP)
        o[va] = N.predict(model, sub(X, va))
        po += N.predict(model, Xo) / K
        info[f"{tag}_fold{k}_auc"] = float(roc_auc_score(y[va], o[va]))
        N.log(f"{tag} fold {k}: OOF AUC {info[f'{tag}_fold{k}_auc']:.5f}")
        del model
    info[f"{tag}_oof_auc"] = float(roc_auc_score(y, o))
    return o, po.astype(np.float32)


if __name__ == "__main__":
    OUT = N.OUT
    N.log(f"device={N.DEV} epochs={EPOCHS}")
    t = N.read(N.TRAIN, 20000 if N.SMOKE else None); b = N.prepare(t); del t
    rn = np.arange(len(b["y"]))
    cut = np.datetime64("2026-10-27T18:00:00").astype("datetime64[ms]").astype(np.int64)
    is_tr = (rn < 15000) if N.SMOKE else (b["ts_ms"] < cut)
    btr, bho = N.subset(b, is_tr), N.subset(b, ~is_tr)
    prep = N.Prep().fit(btr)
    Xtr, Xho = prep.transform(btr), prep.transform(bho)
    ytr, yho = btr["y"], bho["y"]; del btr, bho
    o, ph = oof(Xtr, ytr, Xho, prep.cards(), "valid")
    info["valid_holdout_auc"] = float(roc_auc_score(yho, ph))
    N.log(f"valid: OOF {info['valid_oof_auc']:.5f} holdout {info['valid_holdout_auc']:.5f}")
    pl.DataFrame({"rn": np.concatenate([rn[is_tr], rn[~is_tr]]).astype(np.uint32),
                  "h_nn": np.concatenate([o, ph])}).sort("rn").write_parquet(f"{OUT}/nnf_valid.parquet")
    np.save(f"{OUT}/nnf_holdout_pred.npy", ph)
    json.dump(info, open(f"{OUT}/nnf_info.json", "w"), indent=1)
    del Xtr, Xho

    prep = N.Prep().fit(b); X = prep.transform(b); y = b["y"]; del b
    tt = N.read(N.TEST, 5000 if N.SMOKE else None); ids = tt["id"].to_numpy(); bt = N.prepare(tt); del tt
    Xt = prep.transform(bt); del bt
    o, pt = oof(X, y, Xt, prep.cards(), "full")
    pl.DataFrame({"rn": rn.astype(np.uint32), "h_nn": o}).write_parquet(f"{OUT}/nnf_full.parquet")
    pl.DataFrame({"id": ids, "h_nn": pt}).write_parquet(f"{OUT}/nnf_test.parquet")
    json.dump(info, open(f"{OUT}/nnf_info.json", "w"), indent=1)
    N.log("NNF " + json.dumps(info))
