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


def oof_table(ref):
    """ref: DataFrame with rn, watched, feature5 and TE_KEYS. Returns rn + te cols."""
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
