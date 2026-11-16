p = 'D:/sharechat/train_lgb.py'
s = open(p, encoding='utf8').read()
assert '--age' not in s, "already patched"

old = 'REG = "--reg" in sys.argv'
new = '''AGE = "--age" in sys.argv
XROW = "--xrow" in sys.argv  # DIAGNOSTIC ONLY: cross-row features break the prize rule
if XROW and STAGE != "valid":
    raise SystemExit("--xrow is diagnostic only and is refused outside the valid stage")
if AGE:
    TAG += "_age"
if XROW:
    TAG += "_XROWDIAG"
REG = "--reg" in sys.argv'''
assert s.count(old) == 1; s = s.replace(old, new)

old = '''    if USE_EMB:
        out = pl.concat([out, emb_feats(df)], how="horizontal")
    return out'''
new = '''    if USE_EMB:
        out = pl.concat([out, emb_feats(df)], how="horizontal")
    if AGE:
        ts = df["feature1"].dt.epoch("s").cast(pl.Float64)
        age = pl.DataFrame({
            "age57": (ts - df["feature57"].cast(pl.Float64)).cast(pl.Float32),
            "age58": (ts - df["feature58"].cast(pl.Float64)).cast(pl.Float32),
            "age59": (ts - df["feature59"].cast(pl.Float64) / 1000).cast(pl.Float32),
        })
        out = pl.concat([out.drop(["feature57", "feature58", "feature59"]), age], how="horizontal")
    if XROW:
        out = pl.concat([out, df.select("x_req_size", "x_user_rows", "x_req_pos")], how="horizontal")
    return out'''
assert s.count(old) == 1; s = s.replace(old, new)

old = 'def load_train(rng, filt):'
new = '''XTAB = None


def build_xrow():
    d = pl.read_parquet(f"{DATA}/train.parquet", columns=["request_id", "feature5", "feature1"]).with_row_index("rn")
    d = d.with_columns((pl.col("feature1") >= CUT).alias("per"))
    return d.select(
        "rn",
        pl.len().over(["per", "request_id"]).cast(pl.Float32).alias("x_req_size"),
        pl.len().over(["per", "feature5"]).cast(pl.Float32).alias("x_user_rows"),
        pl.col("rn").rank().over(["per", "request_id"]).cast(pl.Float32).alias("x_req_pos"),
    )


def read_rg(g):
    df = pl.from_arrow(pf_tr.read_row_group(g, columns=READ_COLS))
    if XROW:
        off = sum(pf_tr.metadata.row_group(i).num_rows for i in range(g))
        df = df.with_row_index("rn", offset=off).join(XTAB, on="rn", how="left").drop("rn")
    return df


def load_train(rng, filt):'''
assert s.count(old) == 1; s = s.replace(old, new)

a = '''        df = pl.from_arrow(pf_tr.read_row_group(g, columns=READ_COLS))
        df = df.filter(filt)'''
assert s.count(a) == 1; s = s.replace(a, '        df = read_rg(g).filter(filt)')
a = '''        df = pl.from_arrow(pf_tr.read_row_group(g, columns=READ_COLS))
        df = df.filter(pl.col("feature1") >= CUT)'''
assert s.count(a) == 1; s = s.replace(a, '        df = read_rg(g).filter(pl.col("feature1") >= CUT)')

old = '''    if STAGE == "valid":
        tr = load_train'''
new = '''    if STAGE == "valid":
        if XROW:
            XTAB = build_xrow()
        tr = load_train'''
assert s.count(old) == 1; s = s.replace(old, new)

open(p, 'w', encoding='utf8').write(s)
print("patched")
