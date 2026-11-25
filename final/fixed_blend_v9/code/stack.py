"""Stacking / blend-weight search over all models.

Every model contributes (a) holdout predictions on the standard check (all rows from
2026-10-27 18:00, file order) and (b) full-train test predictions. Weights are non-negative
rank-blend weights found by greedy coordinate search maximising holdout AUC. Honesty check:
weights are fitted on one half of holdout users and scored on the other half (and vice versa).
Usage: python stack.py  -> prints CV AUCs and writes out/submission_stack.csv
"""
import glob, json
import numpy as np, polars as pl, pyarrow.parquet as pq, datetime as dt
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score as A

O = "D:/sharechat/out"
R = lambda v: rankdata(v) / len(v)

# ---------- holdout labels / user ids in file order ----------
pf = pq.ParquetFile("D:/sharechat/data/train.parquet"); cut = dt.datetime(2026, 10, 27, 18)
ys, us = [], []
for g in range(pf.metadata.num_row_groups):
    d = pl.from_arrow(pf.read_row_group(g, columns=["feature1", "watched", "feature5"])).filter(pl.col("feature1") >= cut)
    ys.append(d["watched"].to_numpy()); us.append(d["feature5"].hash(seed=11).to_numpy() % 2)
y, half = np.concatenate(ys), np.concatenate(us)

ss = pl.read_csv("D:/sharechat/data/sample_submission.csv", columns=["id"])


def test_rank(files):
    out = []
    for f in files:
        d = ss.join(pl.read_csv(f), on="id", how="left", maintain_order="left")
        assert d.height == 1_037_090 and d["watched"].null_count() == 0, f
        out.append(R(d["watched"].to_numpy()))
    return np.mean(out, 0)


def ho_rank(files):
    return np.mean([R(np.load(f)) for f in files], 0)


# ---------- model registry: name -> (holdout npy files, test csv files) ----------
M = json.load(open("D:/sharechat/stack_models.json"))
H, T = {}, {}
for name, (hf, tf) in M.items():
    hfs = sorted(sum([glob.glob(f"{O}/{p}") for p in hf], [])); tfs = sorted(sum([glob.glob(f"{O}/{p}") for p in tf], []))
    if not hfs or not tfs:
        print(f"skip {name}: holdout {len(hfs)} test {len(tfs)}"); continue
    h = ho_rank(hfs)
    if len(h) != len(y):
        print(f"skip {name}: holdout length {len(h)} != {len(y)}"); continue
    H[name], T[name] = h, test_rank(tfs)
    print(f"{name:22s} holdout AUC {A(y, h):.5f}  ({len(hfs)} holdout / {len(tfs)} test files)")
names = list(H)


def fit(mask, iters=4, grid=np.linspace(0, 1, 21)):
    w = np.zeros(len(names)); w[int(np.argmax([A(y[mask], H[n][mask]) for n in names]))] = 1.0
    for _ in range(iters):
        for i in range(len(names)):
            best = (-1, w[i])
            for g in grid:
                w2 = w.copy(); w2[i] = g
                if w2.sum() == 0: continue
                p = sum(w2[k] * H[names[k]][mask] for k in range(len(names)))
                a = A(y[mask], p)
                if a > best[0]: best = (a, g)
            w[i] = best[1]
    return w / w.sum()


def score(w, mask):
    return A(y[mask], sum(w[k] * H[names[k]][mask] for k in range(len(names))))


w0, w1 = fit(half == 0), fit(half == 1)
cv = (score(w0, half == 1) + score(w1, half == 0)) / 2
wf = fit(np.ones(len(y), bool))
print("\nCV AUC (weights fit on one half of users, scored on the other):", round(cv, 5))
print("full-holdout AUC with final weights:", round(score(wf, np.ones(len(y), bool)), 5))
print("weights:", {n: round(float(x), 3) for n, x in zip(names, wf) if x > 0})
pt = sum(wf[k] * T[names[k]] for k in range(len(names)))
ss.with_columns(pl.Series("watched", pt)).write_csv(f"{O}/submission_stack.csv")
json.dump({"cv_auc": cv, "weights": {n: float(x) for n, x in zip(names, wf)}}, open(f"{O}/stack_result.json", "w"), indent=1)
print("wrote out/submission_stack.csv")
