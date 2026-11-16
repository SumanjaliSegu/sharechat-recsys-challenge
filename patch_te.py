p = 'D:/sharechat/train_lgb.py'
s = open(p, encoding='utf8').read()
assert '--te' not in s, "already patched"

# flag + tag
old = 'REG = "--reg" in sys.argv'
new = '''TE = "--te" in sys.argv
if TE:
    TAG += "_te"
    sys.path.insert(0, "D:/sharechat")
    import te as temod
REG = "--reg" in sys.argv'''
assert s.count(old) == 1; s = s.replace(old, new)

# featurize: append te columns if present in df
old = '''    if XROW:
        out = pl.concat([out, df.select("x_req_size", "x_user_rows", "x_req_pos")], how="horizontal")
    return out'''
new = '''    if XROW:
        out = pl.concat([out, df.select("x_req_size", "x_user_rows", "x_req_pos")], how="horizontal")
    if TE:
        out = pl.concat([out, df.select(temod.te_names())], how="horizontal")
    return out'''
assert s.count(old) == 1; s = s.replace(old, new)

# read_rg: join per-row te table for training rows
old = '''    if XROW:
        off = sum(pf_tr.metadata.row_group(i).num_rows for i in range(g))
        df = df.with_row_index("rn", offset=off).join(XTAB, on="rn", how="left").drop("rn")
    return df'''
new = '''    off = sum(pf_tr.metadata.row_group(i).num_rows for i in range(g))
    if XROW:
        df = df.with_row_index("rn", offset=off).join(XTAB, on="rn", how="left", maintain_order="left").drop("rn")
    if TE:
        df = df.with_row_index("rn", offset=off).join(TETAB, on="rn", how="left", maintain_order="left").drop("rn")
    return df'''
assert s.count(old) == 1; s = s.replace(old, new)

# table builder
old = '''def read_rg(g):'''
new = '''TETAB = None
TEMAPS = None


def build_te(stage):
    """Training rows in the reference set: OOF stats. Rows outside: full-ref stats."""
    cols = list(dict.fromkeys(["feature1", "watched"] + temod.TE_KEYS))
    d = pl.read_parquet(f"{DATA}/train.parquet", columns=cols).with_row_index("rn")
    if stage == "valid":
        ref = d.filter(pl.col("feature1") < CUT)
        other = d.filter(pl.col("feature1") >= CUT)
        oof, prior = temod.oof_table(ref)
        maps, prior = temod.full_maps(ref)
        oth = pl.concat([other.select("rn"), temod.apply_maps(other, maps, prior)], how="horizontal")
        return pl.concat([oof, oth]).sort("rn"), None
    oof, prior = temod.oof_table(d)
    maps, prior = temod.full_maps(d)
    return oof, (maps, prior)


def read_rg(g):'''
assert s.count(old) == 1; s = s.replace(old, new)

# build in both stages
old = '''    if STAGE == "valid":
        if XROW:
            XTAB = build_xrow()'''
new = '''    if TE:
        TETAB, TEMAPS = build_te(STAGE)
        log(f"te table {TETAB.shape}")
    if STAGE == "valid":
        if XROW:
            XTAB = build_xrow()'''
assert s.count(old) == 1; s = s.replace(old, new)

# test prediction: attach te columns from full maps (keys must be read)
old = '''            df = pl.from_arrow(pf_te.read_row_group(g, columns=["id"] + [c for c in READ_COLS if c != "watched"]))'''
new = '''            tcols = ["id"] + [c for c in READ_COLS if c != "watched"]
            if TE:
                tcols += [k for k in temod.TE_KEYS if k not in tcols]
            df = pl.from_arrow(pf_te.read_row_group(g, columns=tcols))
            if TE:
                df = pl.concat([df, temod.apply_maps(df, *TEMAPS)], how="horizontal")'''
assert s.count(old) == 1; s = s.replace(old, new)

# READ_COLS for training must include TE keys only via table join (te values come from TETAB) — nothing else needed
open(p, 'w', encoding='utf8').write(s)
print("patched te")
