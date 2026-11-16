p = 'D:/sharechat/train_lgb.py'
s = open(p, encoding='utf8').read()
assert '--neg=' not in s, "already patched"

old = 'NEG_RATE = 0.25'
new = 'NEG_RATE = next((float(a.split("=")[1]) for a in sys.argv if a.startswith("--neg=")), 0.25)'
assert s.count(old) == 1; s = s.replace(old, new)

old = 'REG = "--reg" in sys.argv'
new = '''TOPK = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--topk=")), 0)
MSEED = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--seed=")), SEED)
PARAMS["seed"] = MSEED
if NEG_RATE != 0.25:
    TAG += f"_neg{NEG_RATE}"
if TOPK:
    TAG += f"_top{TOPK}"
if MSEED != 42:
    TAG += f"_s{MSEED}"
KEEP = None
if TOPK:
    _imp = pl.read_csv("D:/sharechat/out/importance_v2emb_nopost_lr0.03_reg.csv").sort("gain", descending=True)
    KEEP = _imp["feat"].head(TOPK).to_list()


def to_np(X):
    """Polars frame -> float32 numpy (no pandas copy), optional top-k feature filter."""
    if KEEP is not None:
        X = X.select([c for c in X.columns if c in set(KEEP)])
    names = X.columns
    return X.cast(pl.Float32).to_numpy(), names


REG = "--reg" in sys.argv'''
assert s.count(old) == 1; s = s.replace(old, new)

# valid stage
old = '''        dtr = lgb.Dataset(Xtr.to_pandas(), ytr, categorical_feature=STR_COLS, free_raw_data=True)
        dho = lgb.Dataset(Xho.to_pandas(), yho, reference=dtr, categorical_feature=STR_COLS)'''
new = '''        Xtr, names = to_np(Xtr)
        Xho, _ = to_np(Xho)
        cats = [c for c in STR_COLS if c in names]
        dtr = lgb.Dataset(Xtr, ytr, feature_name=names, categorical_feature=cats, free_raw_data=True)
        dho = lgb.Dataset(Xho, yho, reference=dtr, feature_name=names, categorical_feature=cats)'''
assert s.count(old) == 1; s = s.replace(old, new)
old = '''        p = model.predict(Xho.to_pandas(), num_iteration=model.best_iteration)'''
new = '''        p = model.predict(Xho, num_iteration=model.best_iteration)
        np.save(f"{OUT}/hopred_{TAG}.npy", p.astype(np.float32))'''
assert s.count(old) == 1; s = s.replace(old, new)

# full stage
old = '''        model = lgb.train(PARAMS, lgb.Dataset(Xtr.to_pandas(), ytr, categorical_feature=STR_COLS),'''
new = '''        Xtr, names = to_np(Xtr)
        cats = [c for c in STR_COLS if c in names]
        model = lgb.train(PARAMS, lgb.Dataset(Xtr, ytr, feature_name=names, categorical_feature=cats),'''
assert s.count(old) == 1; s = s.replace(old, new)
old = '''            preds.append(model.predict(featurize(df, vocab).to_pandas()))'''
new = '''            preds.append(model.predict(to_np(featurize(df, vocab))[0]))'''
assert s.count(old) == 1; s = s.replace(old, new)

# seed must reach params
s = s.replace('verbose=-1, seed=SEED,', 'verbose=-1, seed=SEED,')
open(p, 'w', encoding='utf8').write(s)
print("patched mem")
