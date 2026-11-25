"""One command to retrain every component of the final blend and build the submission.

    python run_all.py                 # everything (needs a CUDA GPU and ~30 GB RAM, e.g. a Kaggle GPU notebook)
    python run_all.py --only lgb      # one stage: lgb | xgb | catboost | nn | blend

Data: put train.parquet, test.parquet, sample_submission.csv in ./data (or run on Kaggle with the
competition attached — scripts find /kaggle/input automatically). Outputs go to ./out (or /kaggle/working).
Every feature is computed from the row itself; nothing uses other rows or test-set statistics.
"""
import subprocess, sys, os, json, glob, shutil, time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.join(HERE, "out")
FINAL = os.path.join(HERE, "final", "reproduced")
ONLY = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None

# Feature set "G": embedding matches, drift fix, no creator/category ids, no drifting rates, ff 0.5
G = ["--emb", "--nopost", "--reg", "--fullho", "--stable", "--nocatid", "--dropE", "--ff=0.5"]


def sh(args, env=None):
    t = time.time(); print(">>", " ".join(os.path.basename(a) for a in args), flush=True)
    subprocess.run([PY] + args, check=True, cwd=HERE, env=dict(os.environ, **(env or {})))
    print(f"   done in {time.time() - t:.0f}s", flush=True)


def best_iter(tag):
    return json.load(open(os.path.join(OUT, f"valid_result_{tag}.json")))["best_iter"]


def lgb_stage():
    # (a) slow LightGBM on all rows, Indian-local hour
    a = G + ["--lr=0.01", "--ist", "--neg=1.0"]
    sh(["train_lgb.py", "valid"] + a)
    tag = "v2emb_nopost_lr0.01_ist_fullho_dropH_stable_nocatid_dropE_neg1.0_reg_ff0.5"
    sh(["train_lgb.py", "full"] + a + [f"--iters={int(1.15 * best_iter(tag)) + 1}"])
    # (b) LightGBM G, lr 0.03, 5 seeds
    b = G + ["--lr=0.03"]
    its = []
    for sd in (42, 7):
        sh(["train_lgb.py", "valid"] + b + [f"--seed={sd}"])
        its.append(best_iter("v2emb_nopost_lr0.03_fullho_dropH_stable_nocatid_dropE" + ("" if sd == 42 else f"_s{sd}") + "_reg_ff0.5"))
    for sd in (42, 7, 99, 123, 2026):
        sh(["train_lgb.py", "full"] + b + [f"--seed={sd}", f"--iters={int(1.15 * sum(its) / len(its))}"])


STAGES = {
    "lgb": lgb_stage,
    "xgb": lambda: sh(["xgb_cb_run.py"]),          # XGBoost (seed 1) + CatBoost depth 10 (seed 7)
    "catboost": lambda: sh(["catboost_run.py"]),   # CatBoost depth 8
    "nn": lambda: (sh(["nn2_multi.py"]), sh(["nn_dcn_v1.py"])),  # towers x3 seeds + DCN-v2 (v1 settings)
}


def blend_stage():
    """Copy freshly produced predictions into final/reproduced with the frozen manifest's structure."""
    man = json.load(open(os.path.join(HERE, "final", "fixed_blend_v1", "MANIFEST.json")))
    os.makedirs(os.path.join(FINAL, "test_preds"), exist_ok=True); os.makedirs(os.path.join(FINAL, "holdout_preds"), exist_ok=True)
    for comp in man["components"].values():
        for kind in ("test", "holdout"):
            for e in comp[kind]:
                src_name = e["file"].split("__")[-1]
                hits = glob.glob(os.path.join(OUT, "**", src_name), recursive=True)
                assert hits, f"missing reproduced file {src_name}"
                shutil.copy2(hits[0], os.path.join(FINAL, e["file"]))
    json.dump(man, open(os.path.join(FINAL, "MANIFEST.json"), "w"), indent=1)
    sh(["ensemble.py", os.path.relpath(FINAL, HERE), "--holdout"])


if __name__ == "__main__":
    order = ["lgb", "xgb", "catboost", "nn"]
    for s in ([ONLY] if ONLY and ONLY != "blend" else ([] if ONLY == "blend" else order)):
        STAGES[s]()
    if ONLY in (None, "blend"):
        blend_stage()
