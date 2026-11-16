"""LightGBM baseline for ShareChat RecSys Challenge.

Row-independent features only (competition rule): every feature comes from the
row itself. Categorical strings are mapped to integer codes using a vocabulary
learned from the training data (a model parameter, like an embedding table).

Usage:
  python train_lgb.py valid   # train < Oct27 18:00, evaluate on last train day
  python train_lgb.py full    # train on all train, predict test -> submission
"""
import sys, json, time, datetime as dt
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

import glob as _glob, os as _os
_k = _glob.glob("/kaggle/input/**/train.parquet", recursive=True)
DATA = _os.path.dirname(_k[0]) if _k else "D:/sharechat/data"
OUT = "/kaggle/working" if _os.path.isdir("/kaggle/working") else "D:/sharechat/out"
STAGE = sys.argv[1] if len(sys.argv) > 1 else "valid"
NEG_RATE = next((float(a.split("=")[1]) for a in sys.argv if a.startswith("--neg=")), 0.25)
HOLDOUT_KEEP = 1.0 if "--fullho" in sys.argv else 0.5
SEED = 42
CUT = dt.datetime(2026, 10, 27, 18)

pf_tr = pq.ParquetFile(f"{DATA}/train.parquet")
schema = pf_tr.schema_arrow
DROP = {"request_id", "feature1", "feature2", "feature4", "feature15",
        "feature5", "feature24", "watched", "id"}
LIST_COLS = [f.name for f in schema if str(f.type).startswith("list")]
STR_COLS = [f.name for f in schema if str(f.type) == "string" and f.name not in DROP]
NUM_COLS = [f.name for f in schema if f.name not in DROP and f.name not in LIST_COLS
            and f.name not in STR_COLS and not str(f.type).startswith("timestamp")]
READ_COLS = ["feature1", "feature5", "watched"] + STR_COLS + NUM_COLS


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


IST = "--ist" in sys.argv  # local Indian time (UTC+5:30) instead of UTC hour
USE_EMB = "--emb" in sys.argv
U_EMB = [f"feature{i}" for i in range(26, 33)]
P_EMB = [f"feature{i}" for i in range(40, 47)]
NOPOST = "--nopost" in sys.argv
LR = next((float(a.split("=")[1]) for a in sys.argv if a.startswith("--lr=")), 0.05)
RAWEMB = "--rawemb" in sys.argv
TAG = ("v2emb" if USE_EMB else "v1") + ("_raw" if RAWEMB else "") + ("_nopost" if NOPOST else "") + (f"_lr{LR}" if LR != 0.05 else "")
if NOPOST:
    STR_COLS.remove("feature18")
    READ_COLS.remove("feature18")
if USE_EMB:
    READ_COLS += U_EMB + P_EMB


def emb_feats(df):
    """Row-local embedding features: all user.post dot products, cosines, norms."""
    def arr(c):
        s = df[c].fill_null(pl.lit([0.0] * 32, dtype=pl.List(pl.Float32)))
        return s.list.to_array(32).to_numpy().astype(np.float32)
    U = {c: arr(c) for c in U_EMB}
    P = {c: arr(c) for c in P_EMB}
    nu = {c: np.linalg.norm(v, axis=1) + 1e-6 for c, v in U.items()}
    npn = {c: np.linalg.norm(v, axis=1) + 1e-6 for c, v in P.items()}
    cols = {}
    for u in U_EMB:
        cols[f"n_{u}"] = nu[u]
        for p in P_EMB:
            d = (U[u] * P[p]).sum(1)
            cols[f"dot_{u}_{p}"] = d
            cols[f"cos_{u}_{p}"] = d / (nu[u] * npn[p])
    for p in P_EMB:
        cols[f"n_{p}"] = npn[p]
    if RAWEMB:
        for u, p in (("feature30", "feature46"), ("feature32", "feature43")):
            prod = U[u] * P[p]
            for i in range(32):
                cols[f"x_{u}_{p}_{i}"] = prod[:, i]
    return pl.DataFrame(cols)


def featurize(df, vocab):
    out = df.select(
        [pl.col(c).cast(pl.Float32) for c in NUM_COLS]
        + ([((pl.col("feature1").dt.epoch("s") + 19800) // 3600 % 24).cast(pl.Float32).alias("hour"),
            ((pl.col("feature1").dt.epoch("s") + 19800) % 86400 / 60.0).cast(pl.Float32).alias("ist_minute_of_day")]
           if IST else [pl.col("feature1").dt.hour().cast(pl.Float32).alias("hour")])
        + [pl.col(c).replace_strict(vocab[c], default=-1, return_dtype=pl.Int32).alias(c)
           for c in STR_COLS]
    )
    if USE_EMB:
        out = pl.concat([out, emb_feats(df)], how="horizontal")
    if STABLE:
        ts = df["feature1"].dt.epoch("s").cast(pl.Float64)
        age_h = ((ts - df["feature58"].cast(pl.Float64)) / 3600.0).clip(0) + 1.0
        st = {"age58_h": age_h, "age59_h": ((ts - df["feature59"].cast(pl.Float64) / 1000) / 3600.0),
              "age57_d": ((ts - df["feature57"].cast(pl.Float64)) / 86400.0)}
        for c in H_COLS:
            st[f"ph_{c}"] = (df[c].cast(pl.Float64).abs() + 1).log() - age_h.log()
        Hs = ["feature237", "feature224", "feature249", "feature246", "feature273", "feature221", "feature250", "feature247", "feature279", "feature264", "feature267"]
        for i in range(len(Hs)):
            for j in range(i + 1, len(Hs)):
                st[f"hr_{Hs[i]}_{Hs[j]}"] = (df[Hs[i]].cast(pl.Float64).abs() + 1).log() - (df[Hs[j]].cast(pl.Float64).abs() + 1).log()
        out = pl.concat([out, pl.DataFrame({k: v.cast(pl.Float32) for k, v in st.items()})], how="horizontal")
    if FE:
        cos = {c: out[c].to_numpy() for c in out.columns if c.startswith("cos_")} or None
        out = pl.concat([out, femod.fe_feats(df, FESPEC, cos)], how="horizontal")
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
    if TE:
        out = pl.concat([out, df.select(temod.te_names())], how="horizontal")
    if KNN:
        out = pl.concat([out, df.select(knnmod.NAMES)], how="horizontal")
    return out


XTAB = None


def build_xrow():
    d = pl.read_parquet(f"{DATA}/train.parquet", columns=["request_id", "feature5", "feature1"]).with_row_index("rn")
    d = d.with_columns((pl.col("feature1") >= CUT).alias("per"))
    return d.select(
        "rn",
        pl.len().over(["per", "request_id"]).cast(pl.Float32).alias("x_req_size"),
        pl.len().over(["per", "feature5"]).cast(pl.Float32).alias("x_user_rows"),
        pl.col("rn").rank().over(["per", "request_id"]).cast(pl.Float32).alias("x_req_pos"),
    )


TETAB = None
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


KNN = "--knn" in sys.argv  # embedding-neighbour video stats from training data (knn.py)
KNTAB = None
KNMAP = None


def build_knn(stage):
    """Reference rows: out-of-fold neighbour stats. Others (holdout/test): full-reference video map."""
    cols = ["feature1", "feature5", "feature18", "feature54", "watched"] + knnmod.EMB
    d = pl.read_parquet(f"{DATA}/train.parquet", columns=cols).with_row_index("rn")
    if stage == "valid":
        ref, other = d.filter(pl.col("feature1") < CUT), d.filter(pl.col("feature1") >= CUT)
        oof, vmap = knnmod.build(ref, [other])
        oth = other.select("rn", "feature18").join(vmap, on="feature18", how="left", maintain_order="left").drop("feature18")
        return pl.concat([oof, oth]).sort("rn"), None
    te = pl.read_parquet(f"{DATA}/test.parquet", columns=["feature18"] + knnmod.EMB)
    return knnmod.build(d, [te])


def read_rg(g):
    df = pl.from_arrow(pf_tr.read_row_group(g, columns=READ_COLS))
    off = sum(pf_tr.metadata.row_group(i).num_rows for i in range(g))
    if KNN:
        df = df.with_row_index("rn", offset=off).join(KNTAB, on="rn", how="left", maintain_order="left").drop("rn")
    if XROW:
        df = df.with_row_index("rn", offset=off).join(XTAB, on="rn", how="left", maintain_order="left").drop("rn")
    if TE:
        df = df.with_row_index("rn", offset=off).join(TETAB, on="rn", how="left", maintain_order="left").drop("rn")
    return df


def load_train(rng, filt):
    """Stream row groups; keep all positives + NEG_RATE of negatives."""
    parts = []
    for g in range(pf_tr.metadata.num_row_groups):
        df = read_rg(g).filter(filt)
        if df.height == 0:
            continue
        keep = (df["watched"] == 1).to_numpy() | (rng.random(df.height) < NEG_RATE)
        parts.append(df.filter(pl.Series(keep)))
        log(f"  rg {g}: kept {parts[-1].height:,}")
    return pl.concat(parts)


def load_holdout(rng):
    parts = []
    for g in range(pf_tr.metadata.num_row_groups):
        df = read_rg(g).filter((pl.col("feature1") >= HO_START) & (pl.col("feature1") < HO_END))
        if df.height == 0:
            continue
        parts.append(df.filter(pl.Series(rng.random(df.height) < HOLDOUT_KEEP)))
    return pl.concat(parts)


def build_vocab(df):
    vocab = {}
    for c in STR_COLS:
        vc = df[c].drop_nulls().value_counts().filter(pl.col("count") >= 20)
        vocab[c] = {v: i for i, v in enumerate(vc[c].to_list())}
    return vocab


PARAMS = dict(
    objective="binary", metric="auc", learning_rate=LR, num_leaves=127,
    min_data_in_leaf=200, feature_fraction=0.4, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=10.0, max_bin=127, cat_smooth=50, min_data_per_group=200,
    max_cat_to_onehot=8, num_threads=8, verbose=-1, seed=SEED,
)

AGE = "--age" in sys.argv
XROW = "--xrow" in sys.argv  # DIAGNOSTIC ONLY: cross-row features break the prize rule
if XROW and STAGE != "valid":
    raise SystemExit("--xrow is diagnostic only and is refused outside the valid stage")
if AGE:
    TAG += "_age"
if XROW:
    TAG += "_XROWDIAG"
FAR = "--far" in sys.argv  # train on first day only, skip a day, score the last day (1-2 days ahead, like test)
FOLD1 = "--fold1" in sys.argv  # rolling fold: train day 1, score day 2 (Oct 26 18:00 - Oct 27 18:00)
TRAIN_END = CUT - dt.timedelta(days=1) if (FAR or FOLD1) else CUT
HO_START = CUT - dt.timedelta(days=1) if FOLD1 else CUT
HO_END = CUT if FOLD1 else dt.datetime(2100, 1, 1)
if FOLD1:
    TAG += "_fold1"
if FAR:
    TAG += "_far"
if IST:
    TAG += "_ist"
DROPH = "--dropH" in sys.argv or "--stable" in sys.argv
STABLE = "--stable" in sys.argv
NOCATID = "--nocatid" in sys.argv
H_COLS = [f"feature{i}" for i in range(154, 287)]
if "--fullho" in sys.argv:
    TAG += "_fullho"
if DROPH:
    TAG += "_dropH"
    for _c in H_COLS + ["feature57", "feature58", "feature59"]:
        if _c in NUM_COLS:
            NUM_COLS.remove(_c)
if STABLE:
    TAG += "_stable"
if NOCATID:
    TAG += "_nocatid"
    for _c in ["feature23", "feature20", "feature12", "feature9", "feature7"]:
        STR_COLS.remove(_c)
        if _c in READ_COLS:
            READ_COLS.remove(_c)
if DROPH:
    for _c in H_COLS + ["feature57", "feature58", "feature59"]:
        if _c in READ_COLS:
            READ_COLS.remove(_c)
    if STABLE:
        READ_COLS.extend(H_COLS + ["feature57", "feature58", "feature59"])
if "--nocatall" in sys.argv:  # D: no categorical columns at all
    TAG += "_nocatall"
    for _c in list(STR_COLS):
        STR_COLS.remove(_c)
        if _c in READ_COLS:
            READ_COLS.remove(_c)
if "--dropE" in sys.argv:  # E: drifting user x post rates
    TAG += "_dropE"
    for _c in ["feature454", "feature442", "feature460", "feature472"]:
        if _c in NUM_COLS:
            NUM_COLS.remove(_c)
if "--dropF" in sys.argv:  # F: user running totals 60-153
    TAG += "_dropF"
    for _c in [f"feature{i}" for i in range(60, 154)]:
        if _c in NUM_COLS:
            NUM_COLS.remove(_c)
_dc = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--dropcols=")), None)
if _dc:  # generic column-group removal: "60-70,454,feature6"
    _names = []
    for part in _dc.split(","):
        part = part.replace("feature", "")
        if "-" in part:
            a, b = part.split("-"); _names += [f"feature{i}" for i in range(int(a), int(b) + 1)]
        else:
            _names.append(f"feature{part}")
    for _c in _names:
        if _c in NUM_COLS:
            NUM_COLS.remove(_c)
        if _c in STR_COLS:
            STR_COLS.remove(_c)
            if _c in READ_COLS:
                READ_COLS.remove(_c)
    TAG += "_dc" + _dc.replace(",", "_")
FE = "--fe" in sys.argv
if FE:
    TAG += "_fe"
    sys.path.insert(0, "D:/sharechat")
    import fe as femod
    FESPEC = femod.load_spec()
    for _c in femod.needed_cols(FESPEC):
        if _c not in READ_COLS:
            READ_COLS.append(_c)
TE = "--te" in sys.argv
if TE:
    TAG += "_te"
    sys.path.insert(0, "D:/sharechat")
    import te as temod
TOPK = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--topk=")), 0)
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


REG = "--reg" in sys.argv
if REG:
    TAG += "_reg"
    PARAMS.update(num_leaves=63, min_data_in_leaf=800, feature_fraction=0.3, lambda_l2=30.0,
                  cat_smooth=200, cat_l2=30.0, min_data_per_group=1000, min_sum_hessian_in_leaf=5.0)

FF = next((float(a.split("=")[1]) for a in sys.argv if a.startswith("--ff=")), None)
if FF is not None:
    PARAMS["feature_fraction"] = FF
    TAG += f"_ff{FF}"
RECENCY = next((float(a.split("=")[1]) for a in sys.argv if a.startswith("--recency=")), None)  # half-life in days
if RECENCY:
    TAG += f"_rec{RECENCY}"


def rec_weights(df):
    if not RECENCY:
        return None
    t = df["feature1"].dt.epoch("s").cast(pl.Float64).to_numpy()
    return (0.5 ** ((t.max() - t) / (RECENCY * 86400.0))).astype(np.float32)


MONO = "--mono" in sys.argv  # longer video (feature54) can never raise watch probability
if MONO:
    TAG += "_mono"


def set_mono(names):
    if MONO:
        PARAMS["monotone_constraints"] = [-1 if n == "feature54" else 0 for n in names]
        PARAMS["monotone_constraints_method"] = "advanced"


DART = "--dart" in sys.argv
LEAVES = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--leaves=")), None)
MINLEAF = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--minleaf=")), None)
VALID_ROUNDS = 4000
if DART:  # dropout boosting; early stopping is unreliable in dart, so a fixed round budget is used
    PARAMS.update(boosting="dart", drop_rate=0.1, skip_drop=0.5, max_drop=50)
    VALID_ROUNDS = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--dartrounds=")), 1500)
    TAG += f"_dart{VALID_ROUNDS}"
if LEAVES:
    PARAMS["num_leaves"] = LEAVES; TAG += f"_lv{LEAVES}"
if MINLEAF:
    PARAMS["min_data_in_leaf"] = MINLEAF; TAG += f"_ml{MINLEAF}"


if KNN:
    import knn as knnmod
    TAG += "_knn"
OPTP = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--params=")), None)  # tuned params (Optuna) JSON
if OPTP:
    PARAMS.update(json.load(open(OPTP))); TAG += "_opt"

ITERS = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--iters=")), None)

if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    log(f"stage={STAGE} | num={len(NUM_COLS)} str={len(STR_COLS)} (lists skipped: {len(LIST_COLS)})")

    if TE:
        TETAB, TEMAPS = build_te(STAGE)
        log(f"te table {TETAB.shape}")
    if KNN:
        KNTAB, KNMAP = build_knn(STAGE)
        log(f"knn table {KNTAB.shape}")
    if STAGE == "valid":
        if XROW:
            XTAB = build_xrow()
        tr = load_train(rng, pl.col("feature1") < TRAIN_END)
        ho = load_holdout(rng)
        log(f"train {tr.height:,} (pos {tr['watched'].mean():.3f}) | holdout {ho.height:,}")
        seen_users = set(tr["feature5"].unique().to_list())
        vocab = build_vocab(tr)
        Xtr, ytr = featurize(tr, vocab), tr["watched"].to_numpy()
        wtr = rec_weights(tr)
        Xho, yho = featurize(ho, vocab), ho["watched"].to_numpy()
        unseen = ~ho["feature5"].is_in(list(seen_users)).to_numpy()
        del tr, ho
        Xtr, names = to_np(Xtr)
        Xho, _ = to_np(Xho)
        cats = [c for c in STR_COLS if c in names]
        set_mono(names)
        dtr = lgb.Dataset(Xtr, ytr, weight=wtr, feature_name=names, categorical_feature=cats, free_raw_data=True)
        dho = lgb.Dataset(Xho, yho, reference=dtr, feature_name=names, categorical_feature=cats)
        cbs = [lgb.log_evaluation(100)] + ([] if DART else [lgb.early_stopping(150)])
        model = lgb.train(PARAMS, dtr, VALID_ROUNDS, valid_sets=[dho], valid_names=["holdout"], callbacks=cbs)
        if DART:
            model.best_iteration = VALID_ROUNDS
        p = model.predict(Xho, num_iteration=model.best_iteration)
        np.save(f"{OUT}/hopred_{TAG}.npy", p.astype(np.float32))
        res = {
            "best_iter": model.best_iteration,
            "auc_holdout": roc_auc_score(yho, p),
            "auc_unseen_users": roc_auc_score(yho[unseen], p[unseen]),
            "unseen_share": float(unseen.mean()),
        }
        log(json.dumps(res))
        json.dump(res, open(f"{OUT}/valid_result_{TAG}.json", "w"), indent=1)
        imp = pl.DataFrame({"feat": model.feature_name(),
                            "gain": model.feature_importance("gain")}).sort("gain", descending=True)
        imp.write_csv(f"{OUT}/importance_{TAG}.csv")
        log("top features:", imp.head(25)["feat"].to_list())

    elif STAGE == "full":
        if ITERS:
            n_iter = ITERS
        else:
            best = json.load(open(f"{OUT}/valid_result_{TAG}.json"))["best_iter"]
            n_iter = int(best * 1.15)  # ~1/3 more data -> a few more rounds
        tr = load_train(rng, pl.lit(True))
        vocab = build_vocab(tr)
        Xtr, ytr = featurize(tr, vocab), tr["watched"].to_numpy()
        wtr = rec_weights(tr)
        del tr
        log(f"full train {len(ytr):,} rows, {n_iter} rounds")
        Xtr, names = to_np(Xtr)
        cats = [c for c in STR_COLS if c in names]
        set_mono(names)
        model = lgb.train(PARAMS, lgb.Dataset(Xtr, ytr, weight=wtr, feature_name=names, categorical_feature=cats),
                          n_iter, callbacks=[lgb.log_evaluation(200)])
        del Xtr
        model.save_model(f"{OUT}/lgb_full_{TAG}.txt")
        pf_te = pq.ParquetFile(f"{DATA}/test.parquet")
        ids, preds = [], []
        for g in range(pf_te.metadata.num_row_groups):
            tcols = ["id"] + [c for c in READ_COLS if c != "watched"]
            if TE:
                tcols += [k for k in temod.TE_KEYS if k not in tcols]
            df = pl.from_arrow(pf_te.read_row_group(g, columns=tcols))
            if TE:
                df = pl.concat([df, temod.apply_maps(df, *TEMAPS)], how="horizontal")
            if KNN:
                if "feature18" not in df.columns:
                    df = df.with_columns(pl.from_arrow(pf_te.read_row_group(g, columns=["feature18"]))["feature18"])
                df = df.join(KNMAP, on="feature18", how="left", maintain_order="left")
            ids.append(df["id"].to_numpy())
            preds.append(model.predict(to_np(featurize(df, vocab))[0]))
            log(f"  test rg {g} predicted")
        sub = pl.DataFrame({"id": np.concatenate(ids), "watched": np.concatenate(preds)})
        ss = pl.read_csv(f"{DATA}/sample_submission.csv", columns=["id"])
        assert sub.height == 1_037_090 == ss.height, sub.height
        assert set(sub["id"].to_list()) == set(ss["id"].to_list())
        sub = ss.join(sub, on="id", how="left")
        assert sub["watched"].null_count() == 0
        sub.write_csv(f"{OUT}/submission_lgb_{TAG}.csv")
        log("wrote submission", sub["watched"].describe())
