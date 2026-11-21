"""User-taste clusters for cluster x content training statistics (used by te.py when UCL is set).

Each row's OWN user embeddings (feature26-32, 7 x 32) are reduced with PCA and assigned to KMeans clusters
(k = 32 / 128 / 512); each row's OWN video embeddings (feature40-46) to 256 video clusters. PCA + KMeans are
fitted on training rows before the holdout cut only (unsupervised, no labels), then applied to every row.
Keys = cluster ids and cluster x creator / category / video-cluster / video. Row-local: a test row's keys come
from its own columns; te.py then looks up statistics computed from training labels only.
Writes OUT/ucl_train.parquet (rn + keys, train file order) and OUT/ucl_test.parquet (id + keys).
"""
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = ["train_lgb.py", "valid"]
sys.path.insert(0, HERE)
import numpy as np
import polars as pl
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import train_lgb as T

U = [f"feature{i}" for i in range(26, 33)]
V = [f"feature{i}" for i in range(40, 47)]
OUT = T.OUT


def stack(path, cols, extra):
    pf = pq.ParquetFile(path); Xs, ex = [], []
    for g in range(int(os.environ.get("UCL_SMOKE", pf.metadata.num_row_groups))):
        df = pl.from_arrow(pf.read_row_group(g, columns=cols + extra))
        parts = []
        for c in cols:
            a = df[c].to_list()
            parts.append(np.array([x if x is not None and len(x) == 32 else [0.0] * 32 for x in a], np.float32))
        Xs.append(np.concatenate(parts, 1)); ex.append(df.select(extra))
    return np.concatenate(Xs), pl.concat(ex)


def clusters(Xtr, fit_mask, Xte, ks, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.nonzero(fit_mask)[0], min(int(os.environ.get("UCL_FIT", 400_000)), int(fit_mask.sum())), replace=False)
    pca = PCA(32, random_state=seed).fit(Xtr[idx])
    Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
    out = {}
    for k in ks:
        km = MiniBatchKMeans(k, random_state=seed, batch_size=8192, n_init=3, max_iter=200).fit(Ztr[idx])
        out[k] = (km.predict(Ztr).astype(np.int32), km.predict(Zte).astype(np.int32))
        T.log(f"kmeans k={k}: largest cluster share {np.bincount(out[k][0]).max() / len(Ztr):.3f}")
    return out


t0 = time.time()
Xu, tr = stack(f"{T.DATA}/train.parquet", U, ["feature1", "feature18", "feature20", "feature23"])
Xut, te = stack(f"{T.DATA}/test.parquet", U, ["id", "feature18", "feature20", "feature23"])
fit = (tr["feature1"] < T.CUT).to_numpy()
cu = clusters(Xu, fit, Xut, (32, 128, 512)); del Xu, Xut
Xv, _ = stack(f"{T.DATA}/train.parquet", V, [])
Xvt, _ = stack(f"{T.DATA}/test.parquet", V, [])
cv = clusters(Xv, fit, Xvt, (256,)); del Xv, Xvt
T.log(f"clusters done in {time.time() - t0:.0f}s")


def keys(base, i):
    s = lambda a: pl.Series(a[i]).cast(pl.Utf8)
    u32, u128, u512, v256 = s(cu[32]), s(cu[128]), s(cu[512]), s(cv[256])
    j = lambda a, b: (a + "_" + b.cast(pl.Utf8))  # null content id -> null key
    return pl.DataFrame({
        "u32": u32, "u128": u128, "u512": u512,
        "u32_f20": j(u32, base["feature20"]), "u32_f23": j(u32, base["feature23"]),
        "u128_f23": j(u128, base["feature23"]), "u128_f20": j(u128, base["feature20"]),
        "u128_v256": j(u128, v256), "u512_v256": j(u512, v256), "u32_f18": j(u32, base["feature18"]),
    })


ktr = pl.concat([pl.DataFrame({"rn": np.arange(tr.height, dtype=np.uint32)}), keys(tr, 0)], how="horizontal")
kte = pl.concat([te.select("id"), keys(te, 1)], how="horizontal")
ktr.write_parquet(f"{OUT}/ucl_train.parquet"); kte.write_parquet(f"{OUT}/ucl_test.parquet")
for c in ("u128_f23", "u128_v256", "u32_f18"):
    seen = kte[c].is_in(ktr[c].unique().implode()).mean()
    T.log(f"{c}: {ktr[c].n_unique():,} train keys, test rows with key seen in train {seen:.3f}")
T.log("wrote ucl tables")
