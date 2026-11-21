"""Train-period user/item statistics (allowed by host: "user/item-level statistics
computed only from the training data are allowed as features").

For each key column we compute impression count, watch count and a smoothed
watch rate from a reference set of labelled training rows.

Leakage control:
  * rows INSIDE the reference set get out-of-fold stats (5 folds grouped by user,
    so a row never sees its own label or its own user's other rows — this also
    mimics test, where ~93% of users are new);
  * rows OUTSIDE the reference set (holdout / test) get stats from the full
    reference set.
OOF counts are rescaled by K/(K-1) so they are on the same scale as full-ref counts.
"""
import numpy as np
import polars as pl

TE_KEYS = ["feature18", "feature23", "feature20", "feature9", "feature12",
           "feature7", "feature10", "feature14", "feature19", "feature5"]
K = 5
ALPHA = 30.0

# Optional: user-taste cluster keys (ucl_build.py). UCL=<dir with ucl_train.parquet (rn order) / ucl_test.parquet (id)>.
# Clusters come from each row's OWN user / video embeddings (KMeans fitted on training rows); the statistics per
# cluster x content key come from training labels only (same OOF scheme as above). Test rows only look up their keys.
import os as _os
UCL = _os.environ.get("UCL")
RAW_COLS = list(TE_KEYS)                      # columns read from the parquet files
if UCL:
    TE_KEYS = ["u32", "u128", "u512", "u32_f20", "u32_f23", "u128_f23", "u128_f20",
               "u128_v256", "u512_v256", "u32_f18"]
    RAW_COLS = ["feature5"]


def add_keys(df, split):
    """Attach cluster key columns: split='train' joins by row number rn, 'test' by id."""
    if not UCL or all(k in df.columns for k in TE_KEYS):
        return df
    on = "rn" if split == "train" else "id"
    t = pl.read_parquet(f"{UCL}/ucl_{split}.parquet")
    df = df.with_columns(pl.col(on).cast(t[on].dtype))
    return df.join(t, on=on, how="left", maintain_order="left")


def te_names():
    out = []
    for k in TE_KEYS:
        out += [f"te_n_{k}", f"te_r_{k}"]
    return out


def _stats(ref, key, prior):
    return (ref.filter(pl.col(key).is_not_null())
               .group_by(key)
               .agg(pl.len().alias("n"), pl.col("watched").sum().alias("w")))


def _finish(df, key, prior, scale=1.0):
    n = df["n"].fill_null(0).cast(pl.Float64)
    w = df["w"].fill_null(0).cast(pl.Float64)
    return pl.DataFrame({
        f"te_n_{key}": (n * scale).cast(pl.Float32),
        f"te_r_{key}": ((w * scale + ALPHA * prior) / (n * scale + ALPHA)).cast(pl.Float32),
    })


# Optional: TE_TIME=1 -> training rows get PAST-ONLY stats (rows from strictly earlier hours, same key), the way a
# test row sees only earlier data. Holdout / test rows still use the full reference set (all earlier than them).
TE_TIME = _os.environ.get("TE_TIME") == "1"


def past_table(ref):
    prior = float(ref["watched"].mean())
    hr = pl.col("feature1").dt.truncate("1h").alias("_hr")
    ref = ref.with_columns(hr)
    cols = [ref.select("rn")]
    for key in TE_KEYS:
        g = (ref.filter(pl.col(key).is_not_null()).group_by([key, "_hr"])
                .agg(pl.len().alias("nh"), pl.col("watched").sum().alias("wh")).sort([key, "_hr"])
                .with_columns((pl.col("nh").cum_sum().over(key) - pl.col("nh")).alias("n"),
                              (pl.col("wh").cum_sum().over(key) - pl.col("wh")).alias("w")))
        j = ref.select(key, "_hr").join(g.select(key, "_hr", "n", "w"), on=[key, "_hr"], how="left", maintain_order="left")
        cols.append(_finish(j, key, prior))
    return pl.concat(cols, how="horizontal").sort("rn"), prior


def oof_table(ref):
    """ref: DataFrame with rn, watched, feature5 and TE_KEYS. Returns rn + te cols."""
    if TE_TIME:
        return past_table(ref)
    prior = float(ref["watched"].mean())
    # fold by hashed user id -> grouped folds
    fold = (ref["feature5"].hash(seed=7) % K).cast(pl.Int32)
    ref = ref.with_columns(fold.alias("_fold"))
    parts = []
    for f in range(K):
        inn = ref.filter(pl.col("_fold") == f)
        out = ref.filter(pl.col("_fold") != f)
        cols = [inn.select("rn")]
        for key in TE_KEYS:
            st = _stats(out, key, prior)
            j = inn.select(key).join(st, on=key, how="left", maintain_order="left")
            cols.append(_finish(j, key, prior, scale=K / (K - 1)))
        parts.append(pl.concat(cols, how="horizontal"))
    return pl.concat(parts).sort("rn"), prior


def full_maps(ref):
    """Per-key stats from the whole reference set, for rows outside it."""
    prior = float(ref["watched"].mean())
    return {key: _stats(ref, key, prior) for key in TE_KEYS}, prior


def apply_maps(df, maps, prior):
    cols = []
    for key in TE_KEYS:
        j = df.select(key).join(maps[key], on=key, how="left", maintain_order="left")
        cols.append(_finish(j, key, prior))
    return pl.concat(cols, how="horizontal")
