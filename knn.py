"""Embedding-neighbour video statistics ("neighbour history"), train-data only.

Idea: a video with little or no history (16% of test rows are brand-new videos) still has a content
embedding (features 40-46, fixed per video: cosine 0.999 between its rows). Its most similar
*other* videos do have training history, so their watch rates are an estimate of its appeal.

Allowed category (host): item-level statistics computed only from the training data. Uses no
request grouping and no test labels; test rows only contribute their own video's embedding.

Leakage control (same scheme as te.py):
  * reference rows (training period) get out-of-fold stats: 5 folds grouped by hashed user id, so
    a row never sees its own label, its own user's other rows, or other rows of its own request;
  * rows outside the reference set (holdout / test) get stats from the full reference set;
  * a video is never its own neighbour.
OOF counts are rescaled by K/(K-1) to stay on the full-reference scale.
"""
import numpy as np
import polars as pl

K = 5                 # user folds (same hash as te.py)
NNB = 50              # neighbours per video
MIN_N = 20            # min reference impressions for a video to be in the neighbour index
ALPHA = 30.0          # smoothing for neighbour watch rate
BETA = 2.0            # smoothing for length-adjusted ratio (prior ratio 1)
EMB = [f"feature{i}" for i in range(40, 47)]
NAMES = ["knn_rate", "knn_ratio", "knn_ratio10", "knn_sim1", "knn_simmean", "knn_logn"]


def log(*a):
    import time
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def video_embeddings(frames):
    """frames: DataFrames with feature18 + EMB lists. -> (video ids array, L2-normalised matrix)."""
    v = pl.concat([f.select(["feature18"] + EMB) for f in frames]).drop_nulls().unique("feature18", keep="first")
    mats = []
    for c in EMB:
        m = np.asarray(v[c].to_list(), dtype=np.float32)
        mats.append(m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9))
    X = np.concatenate(mats, axis=1)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return v["feature18"].to_numpy(), X


def neighbours(Q, I, self_col, chunk=4096):
    """Top-NNB cosine neighbours in I for every row of Q; self_col[q] = column of q in I (or -1), excluded."""
    nb = np.empty((len(Q), NNB), np.int32); sm = np.empty((len(Q), NNB), np.float32)
    for s in range(0, len(Q), chunk):
        S = Q[s:s + chunk] @ I.T
        sc = self_col[s:s + chunk]; r = np.nonzero(sc >= 0)[0]
        S[r, sc[r]] = -2.0
        top = np.argpartition(-S, NNB, axis=1)[:, :NNB]
        ts = np.take_along_axis(S, top, 1)
        o = np.argsort(-ts, axis=1)
        nb[s:s + chunk] = np.take_along_axis(top, o, 1); sm[s:s + chunk] = np.take_along_axis(ts, o, 1)
    return nb, sm


def _feats(nb, sm, n, w, e, prior):
    """n, w, e: per-index-video impressions, watches, length-expected watches (same scale)."""
    s = np.clip(sm, 0, None)
    N, W, E = n[nb], w[nb], e[nb]
    sn, sw, se = (s * N).sum(1), (s * W).sum(1), (s * E).sum(1)
    t = slice(0, 10)
    rate = (sw + ALPHA * prior) / (sn + ALPHA)
    ratio = (sw + BETA) / (se + BETA)
    ratio10 = ((s[:, t] * W[:, t]).sum(1) + BETA) / ((s[:, t] * E[:, t]).sum(1) + BETA)
    return np.column_stack([rate, ratio, ratio10, sm[:, 0], sm.mean(1), np.log1p(N.sum(1))]).astype(np.float32)


def build(ref, query_frames):
    """ref: reference rows with rn, feature5, feature18, feature54, watched, EMB.
    query_frames: other frames (holdout and/or test) with feature18 + EMB, used only for embeddings.

    Returns (oof_table: rn + NAMES for reference rows, video_map: feature18 + NAMES from full reference).
    """
    prior = float(ref["watched"].mean())
    dur = ref["feature54"].cast(pl.Float64).fill_null(-1)
    bucket = (dur // 10).clip(-1, 40).cast(pl.Int32)
    ref = ref.with_columns(bucket.alias("_b"), (ref["feature5"].hash(seed=7) % K).cast(pl.Int32).alias("_fold"))
    brate = ref.group_by("_b").agg(pl.col("watched").mean().alias("_e"))
    ref = ref.join(brate, on="_b", how="left", maintain_order="left")
    vids, X = video_embeddings([ref] + list(query_frames))
    vpos = {v: i for i, v in enumerate(vids.tolist())}
    st = ref.group_by(["feature18", "_fold"]).agg(pl.len().alias("n"), pl.col("watched").sum().alias("w"),
                                                  pl.col("_e").sum().alias("e"))
    tot = st.group_by("feature18").agg(pl.col("n").sum(), pl.col("w").sum(), pl.col("e").sum())
    idx_v = tot.filter(pl.col("n") >= MIN_N)["feature18"].to_list()
    idx_v = [v for v in idx_v if v in vpos]
    ipos = np.array([vpos[v] for v in idx_v]); I = X[ipos]
    self_col = np.full(len(vids), -1, np.int64); self_col[ipos] = np.arange(len(ipos))
    log(f"knn: {len(vids):,} videos, index {len(idx_v):,} (>= {MIN_N} impressions), dim {X.shape[1]}")
    nb, sm = neighbours(X, I, self_col)
    log("knn: neighbours done")
    # per-index-video counts: full and per fold
    cnt = {}
    full = tot.filter(pl.col("feature18").is_in(idx_v))
    order = {v: i for i, v in enumerate(idx_v)}

    def arr(df):
        n = np.zeros(len(idx_v)); w = np.zeros(len(idx_v)); e = np.zeros(len(idx_v))
        p = np.array([order[v] for v in df["feature18"].to_list()], dtype=np.int64)
        n[p] = df["n"].to_numpy(); w[p] = df["w"].to_numpy(); e[p] = df["e"].to_numpy()
        return n, w, e
    nF, wF, eF = arr(full)
    vmap = pl.DataFrame({"feature18": vids}).with_columns(
        [pl.Series(c, col) for c, col in zip(NAMES, _feats(nb, sm, nF, wF, eF, prior).T)])
    # out-of-fold for reference rows: neighbour counts excluding the row's user fold
    parts = []
    sc = K / (K - 1)
    for f in range(K):
        fo = st.filter((pl.col("_fold") == f) & pl.col("feature18").is_in(idx_v))
        nf, wf, ef = arr(fo)
        ff = _feats(nb, sm, (nF - nf) * sc, (wF - wf) * sc, (eF - ef) * sc, prior)
        tab = pl.DataFrame({"feature18": vids}).with_columns([pl.Series(c, col) for c, col in zip(NAMES, ff.T)])
        rows = ref.filter(pl.col("_fold") == f).select("rn", "feature18")
        parts.append(rows.join(tab, on="feature18", how="left", maintain_order="left").drop("feature18"))
    oof = pl.concat(parts).sort("rn")
    log(f"knn: oof table {oof.shape}")
    return oof, vmap
