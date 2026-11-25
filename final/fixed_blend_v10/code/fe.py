"""Row-local engineered features (allowed: each uses only columns of the same row).

Column choices come from the column audit + LightGBM gain, frozen into fe_spec.json
so train and test use exactly the same recipe.
"""
import json, os
import numpy as np
import polars as pl

SPEC_PATH = "D:/sharechat/fe_spec.json"


def build_spec():
    imp = pl.read_csv("D:/sharechat/out/importance_v2emb_nopost_lr0.015_reg.csv")
    g = dict(zip(imp["feat"], imp["gain"]))

    def top(nums, k):
        cs = [f"feature{i}" for i in nums if f"feature{i}" in g]
        return sorted(cs, key=lambda c: -g[c])[:k]

    spec = {
        "W": top(range(475, 494), 5),                                   # user time-like magnitudes
        "I": top(list(range(391, 443)) + list(range(494, 522)), 8),      # user×post affinities
        "F": top(list(range(329, 360)) + list(range(287, 311)), 6),      # user baseline rates
        "H": top(range(154, 287), 8),                                    # popularity counters
        "G": [f"feature{i}" for i in range(47, 54)],                     # 7 user×post scores
    }
    json.dump(spec, open(SPEC_PATH, "w"), indent=1)
    return spec


def load_spec():
    return json.load(open(SPEC_PATH)) if os.path.exists(SPEC_PATH) else build_spec()


def needed_cols(spec):
    cols = set(spec["W"] + spec["I"] + spec["F"] + spec["H"] + spec["G"])
    return sorted(cols | {"feature54", "feature58", "feature1"})


def fe_feats(df, spec, emb_cos=None):
    """df: polars frame with needed_cols. emb_cos: dict pair->np array of matched cosines."""
    f = lambda c: df[c].cast(pl.Float64).to_numpy()
    out = {}
    eps = 1e-3
    dur = f("feature54")
    # G1 duration vs user magnitudes
    for w in spec["W"]:
        x = f(w)
        out[f"fe_dur_div_{w}"] = dur / (np.abs(x) + 1.0)
        out[f"fe_dur_x_{w}"] = dur * x
    # G2 affinity vs baseline
    for i in spec["I"]:
        a = f(i)
        for b_ in spec["F"]:
            b = f(b_)
            out[f"fe_{i}_m_{b_}"] = a - b
            out[f"fe_{i}_d_{b_}"] = a / (b + eps)
    # G3 popularity velocity + ratios
    ts = df["feature1"].dt.epoch("s").cast(pl.Float64).to_numpy()
    age_h = np.clip((ts - f("feature58")) / 3600.0, 0, None) + 1.0
    H = spec["H"]
    for h in H:
        out[f"fe_{h}_per_h"] = f(h) / age_h
    for a in range(len(H)):
        for b in range(a + 1, len(H)):
            out[f"fe_{H[a]}_r_{H[b]}"] = np.log1p(np.abs(f(H[a]))) - np.log1p(np.abs(f(H[b])))
    out["fe_age_h"] = age_h
    # G4 agreement of the 7 scores
    G = np.stack([f(c) for c in spec["G"]], 1)
    out["fe_g_sum"] = np.nansum(G, 1); out["fe_g_max"] = np.nanmax(G, 1)
    out["fe_g_min"] = np.nanmin(G, 1); out["fe_g_std"] = np.nanstd(G, 1)
    if emb_cos is not None:
        C = np.stack(list(emb_cos.values()), 1)
        out["fe_cos_sum"] = C.sum(1); out["fe_cos_max"] = C.max(1)
        out["fe_cos_min"] = C.min(1); out["fe_cos_std"] = C.std(1)
    res = {k: np.nan_to_num(v, nan=np.nan, posinf=np.nan, neginf=np.nan).astype(np.float32) for k, v in out.items()}
    return pl.DataFrame(res)
