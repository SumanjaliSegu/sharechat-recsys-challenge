"""Memory-light EDA for ShareChat RecSys data.

Reads a sample of row groups from train/test with PyArrow, then profiles each
column with Polars: dtype, nulls, cardinality, range, and (train) correlation
with the label. Also measures overlap of high-cardinality columns (likely IDs)
between train and test.
"""
import sys
import random
import pyarrow.parquet as pq
import polars as pl

DATA = "D:/sharechat/data"
N_GROUPS = int(sys.argv[1]) if len(sys.argv) > 1 else 6


def sample(path, n_groups, seed=0):
    pf = pq.ParquetFile(path)
    k = pf.metadata.num_row_groups
    idx = sorted(random.Random(seed).sample(range(k), min(n_groups, k)))
    return pl.from_arrow(pf.read_row_groups(idx)), pf.metadata.num_rows


tr, ntr = sample(f"{DATA}/train.parquet", N_GROUPS)
te, nte = sample(f"{DATA}/test.parquet", N_GROUPS)
print(f"train rows {ntr:,} (sample {tr.height:,}) | test rows {nte:,} (sample {te.height:,})")
print(f"sample memory: train {tr.estimated_size('mb'):.0f} MB, test {te.estimated_size('mb'):.0f} MB")

if "watched" in tr.columns:
    print(f"watched rate (sample): {tr['watched'].cast(pl.Float64).mean():.4f}")

pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(40)

rows = []
list_cols = [c for c in tr.columns if isinstance(tr[c].dtype, pl.List)]
print("\nList (embedding) columns — length distribution / null%:")
for c in list_cols:
    ln = tr[c].list.len()
    print(f"  {c}: len min={ln.min()} max={ln.max()} null%={100*tr[c].null_count()/tr.height:.1f}")

for c in tr.columns:
    if c in list_cols:
        continue
    s = tr[c]
    num = s.dtype.is_numeric() or s.dtype == pl.Boolean
    row = {
        "col": c,
        "dtype": str(s.dtype),
        "null%": round(100 * s.null_count() / len(s), 2),
        "n_unique": s.n_unique(),
        "min": str(s.min())[:20] if num or s.dtype.is_temporal() else "",
        "max": str(s.max())[:20] if num or s.dtype.is_temporal() else "",
        "example": str(s.drop_nulls().head(1).to_list())[:40],
        "corr_y": None,
        "in_test": c in te.columns,
    }
    if num and c != "watched" and "watched" in tr.columns:
        try:
            row["corr_y"] = round(
                tr.select(pl.corr(pl.col(c).cast(pl.Float64), pl.col("watched").cast(pl.Float64))).item(), 4
            )
        except Exception:
            pass
    rows.append(row)
prof = pl.DataFrame(rows, infer_schema_length=None)
prof.write_csv("D:/sharechat/column_profile.csv")
print("\nNon-numeric / low-cardinality columns:")
print(prof.filter(~pl.col("dtype").str.contains("Float") | (pl.col("n_unique") < 30)).head(120))
print("\nTop 30 numeric features by |corr with watched|:")
print(prof.drop_nulls("corr_y").sort(pl.col("corr_y").abs(), descending=True).head(30))
print("\nnull% distribution across columns:")
print(prof["null%"].describe())
print("constant columns:", prof.filter(pl.col("n_unique") <= 1)["col"].to_list())

print("\nTest-only columns:", [c for c in te.columns if c not in tr.columns])

# Overlap of high-cardinality columns between train and test sample
print("\nID-like column overlap (share of test-sample values seen in train-sample):")
for c in tr.columns:
    if c in te.columns and c not in ("id", "watched") and tr[c].n_unique() > 1000:
        seen = te[c].is_in(tr[c].unique()).mean()
        print(f"  {c}: n_unique train={tr[c].n_unique():,} test={te[c].n_unique():,} test-seen={seen:.3f}")
