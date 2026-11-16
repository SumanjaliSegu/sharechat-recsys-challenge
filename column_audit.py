"""Column audit: for every feature, what level it lives at and how predictive it is.

level: fraction of multi-row groups where the value is constant within
       request / user (feature5) / post (feature18) -> tells us whose attribute it is.
Also: dtype, null%, range, share in [0,1], integer-valued share, single-feature AUC,
LightGBM gain rank. Reads columns in batches to stay within 8 GB RAM.
"""
import numpy as np, polars as pl, pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

DATA = "D:/sharechat/data/train.parquet"
pf = pq.ParquetFile(DATA)
schema = pf.schema_arrow
KEYS = ["request_id", "feature5", "feature18", "watched"]
skip = set(KEYS) | {"feature1"}
cols = [f.name for f in schema if f.name not in skip and not str(f.type).startswith("list")]

# sample 400k rows, whole requests kept together (sample by request hash)
base = pl.read_parquet(DATA, columns=KEYS).with_row_index("rn")
keep = (base["request_id"].hash(seed=3) % 4 == 0)
base = base.filter(keep)
rows = base["rn"].to_numpy()
y = base["watched"].to_numpy()
print("sample rows", len(rows))

imp = pl.read_csv("D:/sharechat/out/importance_v2emb_nopost_lr0.015_reg.csv").sort("gain", descending=True)
imp = imp.with_row_index("gain_rank", offset=1)
gain_rank = dict(zip(imp["feat"].to_list(), imp["gain_rank"].to_list()))
gain_share = dict(zip(imp["feat"].to_list(), (imp["gain"] / imp["gain"].sum()).to_list()))


def const_share(df, key, c):
    g = (df.group_by(key).agg(pl.col(c).n_unique().alias("u"), pl.len().alias("n"))
           .filter(pl.col("n") > 1))
    return float((g["u"] == 1).mean()) if g.height else float("nan")


out = []
B = 40
for i in range(0, len(cols), B):
    chunk = cols[i:i + B]
    t = pl.read_parquet(DATA, columns=chunk)[rows]
    df = pl.concat([base.select("request_id", "feature5", "feature18"), t], how="horizontal")
    for c in chunk:
        s = df[c]
        is_str = s.dtype == pl.String
        r = {"col": c, "num": int(c.replace("feature", "")), "dtype": str(s.dtype),
             "null%": round(100 * s.null_count() / len(s), 1), "n_unique": s.n_unique()}
        r["const_req"] = round(const_share(df, "request_id", c), 3)
        r["const_user"] = round(const_share(df, "feature5", c), 3)
        r["const_post"] = round(const_share(df, "feature18", c), 3)
        if not is_str:
            v = s.cast(pl.Float64)
            nn = v.drop_nulls()
            r["min"] = float(nn.min()) if len(nn) else None
            r["max"] = float(nn.max()) if len(nn) else None
            r["in01"] = round(float(((nn >= 0) & (nn <= 1)).mean()), 3) if len(nn) else None
            r["int_share"] = round(float((nn == nn.round()).mean()), 3) if len(nn) else None
            x = v.fill_null(-1e30).to_numpy()
            a = roc_auc_score(y, x) if np.isfinite(x).all() else 0.5
            r["auc1"] = round(max(a, 1 - a), 4)
        r["gain_rank"] = gain_rank.get(c)
        r["gain%"] = round(100 * gain_share.get(c, 0), 3)
        out.append(r)
    print(f"done {min(i + B, len(cols))}/{len(cols)}", flush=True)

res = pl.DataFrame(out, infer_schema_length=None)


def level(r):
    if r["n_unique"] <= 1:
        return "constant"
    if r["const_req"] >= 0.98 and r["const_user"] >= 0.9:
        return "user"
    if r["const_req"] >= 0.98:
        return "request"
    if r["const_post"] >= 0.9:
        return "post"
    return "user×post"


res = res.with_columns(pl.Series("level", [level(r) for r in res.iter_rows(named=True)]))
res.write_csv("D:/sharechat/out/column_audit.csv")
print("saved D:/sharechat/out/column_audit.csv")
