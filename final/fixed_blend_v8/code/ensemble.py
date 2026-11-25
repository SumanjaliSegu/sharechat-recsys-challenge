"""Rebuild a final submission from a frozen blend folder (MANIFEST.json + component predictions).

Each component's test-prediction files are converted to ranks and averaged; components are then
combined with their fixed weights. Validates IDs, row count, NaNs and value range.
Usage: python ensemble.py final/fixed_blend_v1 [--holdout]   (--holdout also reports holdout AUC)
"""
import sys, json, os
import numpy as np
import polars as pl
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(ROOT, sys.argv[1] if len(sys.argv) > 1 else "final/fixed_blend_v1")
DATA = os.path.join(ROOT, "data")
N_TEST = 1_037_090
R = lambda v: rankdata(v) / len(v)

man = json.load(open(os.path.join(FOLDER, "MANIFEST.json")))
ids = pl.read_csv(os.path.join(DATA, "sample_submission.csv"), columns=["id"])
assert ids.height == N_TEST and ids["id"].n_unique() == N_TEST


def load_test(rel):
    d = ids.join(pl.read_csv(os.path.join(FOLDER, rel)), on="id", how="left", maintain_order="left")
    assert d.height == N_TEST and d["watched"].null_count() == 0, rel
    v = d["watched"].to_numpy()
    assert np.isfinite(v).all(), rel
    return R(v)


def group_mean(entries, loader):
    groups = sorted({e.get("group", 0) for e in entries})
    return np.mean([np.mean([loader(e["file"]) for e in entries if e.get("group", 0) == g], 0) for g in groups], 0)


total_w = sum(c["weight"] for c in man["components"].values())
assert abs(total_w - 1.0) < 1e-9, total_w
pred = np.zeros(N_TEST)
for name, c in man["components"].items():
    comp = group_mean(c["test"], load_test)
    pred += c["weight"] * comp
    print(f"{name:10s} weight {c['weight']:.2f}  files {len(c['test'])}")
assert pred.min() >= 0 and pred.max() <= 1
out = ids.with_columns(pl.Series("watched", pred))
dst = os.path.join(FOLDER, "rebuilt_" + man["submission_file"])
out.write_csv(dst)
print("wrote", dst)

if "--holdout" in sys.argv:
    import datetime as dt, pyarrow.parquet as pq
    from sklearn.metrics import roc_auc_score
    pf = pq.ParquetFile(os.path.join(DATA, "train.parquet")); cut = dt.datetime(2026, 10, 27, 18)
    y = np.concatenate([pl.from_arrow(pf.read_row_group(g, columns=["feature1", "watched"]))
                        .filter(pl.col("feature1") >= cut)["watched"].to_numpy() for g in range(pf.metadata.num_row_groups)])
    ho = sum(c["weight"] * group_mean(c["holdout"], lambda f: R(np.load(os.path.join(FOLDER, f))))
             for c in man["components"].values())
    print(f"holdout AUC {roc_auc_score(y, ho):.5f}")
