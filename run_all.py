"""One command to retrain every component of the final blend (Fixed Blend v9w40) and rebuild its submission.

    python run_all.py                      # everything (CUDA GPU + ~30 GB RAM, e.g. a Kaggle GPU notebook)
    python run_all.py --only lgb           # one stage: lgb | xgb | catboost | nets | teacher | students | blend
    python run_all.py --only blend --manifest final/fixed_blend_v7   # rebuild another frozen blend's structure

Data: put train.parquet, test.parquet, sample_submission.csv in ./data (or run on Kaggle with the competition
attached; scripts find /kaggle/input automatically). Outputs go to ./out (or /kaggle/working).
Every model predicts each test row from that row's own columns only. The teacher (teacher.py) also sees
request-level columns, but ONLY to make soft training targets for the students; it never scores test rows.
Stage order matters only for students (they need the teacher's teach_*.parquet).
Full run ~ 20 GPU/CPU hours in total: run the stages as separate Kaggle sessions (12 h limit each).
"""
import subprocess, sys, os, json, glob, shutil, time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.join(HERE, "out")
ARG = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
ONLY = ARG("--only")
MANIFEST = ARG("--manifest", "final/fixed_blend_v9w40")
FINAL = os.path.join(HERE, "final", "reproduced")

# Feature set "G": embedding matches, drift-stable counters, no creator/category ids, no drifting rates, ff 0.5
G = ["--emb", "--nopost", "--reg", "--fullho", "--stable", "--nocatid", "--dropE", "--ff=0.5"]
# Wide PLR towers network (nn2_multi.py arch "plr"): periodic numeric embeddings, sigma 0.3, 6-epoch OneCycle
PLR = dict(NNARMS="plr", PLR_SIGMA="0.3", PLR_EPOCHS="6", PLR_K="24", PLR_D="16")


def sh(args, env=None):
    t = time.time(); print(">>", " ".join(os.path.basename(a) for a in args), env or "", flush=True)
    subprocess.run([PY] + args, check=True, cwd=HERE, env=dict(os.environ, **(env or {})))
    print(f"   done in {time.time() - t:.0f}s", flush=True)


def lgb_arm(extra):
    """Validate on the last training day, then retrain on all training rows with 1.15x the best iterations."""
    before = set(glob.glob(f"{OUT}/valid_result_*.json"))
    sh(["train_lgb.py", "valid"] + G + extra)
    new = sorted(set(glob.glob(f"{OUT}/valid_result_*.json")) - before, key=os.path.getmtime)
    it = json.load(open(new[-1]))["best_iter"]
    sh(["train_lgb.py", "full"] + G + extra + [f"--iters={int(1.15 * it) + 1}"])


def lgb_stage():
    for sd in (42, 7, 99):                               # slow LightGBM on all rows (lr 0.01), 3 seeds
        lgb_arm(["--lr=0.01", "--ist", "--neg=1.0"] + ([] if sd == 42 else [f"--seed={sd}"]))
    its = []                                             # LightGBM G, lr 0.03: 2 validation seeds, 5 full seeds
    for sd in (42, 7):
        sh(["train_lgb.py", "valid"] + G + ["--lr=0.03", f"--seed={sd}"])
        tag = "v2emb_nopost_lr0.03_fullho_dropH_stable_nocatid_dropE" + ("" if sd == 42 else f"_s{sd}") + "_reg_ff0.5"
        its.append(json.load(open(f"{OUT}/valid_result_{tag}.json"))["best_iter"])
    for sd in (42, 7, 99, 123, 2026):
        sh(["train_lgb.py", "full"] + G + ["--lr=0.03", f"--seed={sd}", f"--iters={int(1.15 * sum(its) / len(its))}"])
    lgb_arm(["--lr=0.05", "--ist", "--dart", "--dartrounds=1500"])                    # DART
    lgb_arm(["--lr=0.01", "--ist", "--neg=1.0", "--leaves=127", "--minleaf=400"])     # big leaves


def xgb_stage():
    sh(["xgb_cb_run.py"])                                # XGBoost seed 1 + CatBoost depth 10 (seed 7)
    for sd in ("2", "3"):
        sh(["xgb_seeds.py"], dict(XSEED=sd))


def catboost_stage():
    sh(["catboost_run.py"]); sh(["catboost_run.py"], dict(CBSEED="7"))  # CatBoost depth 8, seeds default / 7


def nets_stage():                                        # each run in its own folder (file names repeat)
    sh(["nn2_multi.py"], dict(PLR, TSEEDS="0,1", OUT=f"{OUT}/ph_k"))
    sh(["nn2_multi.py"], dict(PLR, TSEEDS="2,3,4,5", OUT=f"{OUT}/ph2_k"))
    sh(["nn2_multi.py"], dict(PLR, TSEEDS="0,1", PLR_K="32", PLR_D="24", OUT=f"{OUT}/pj_k"))


def teacher_stage():
    sh(["teacher.py"])                                   # writes OUT/teach_valid.parquet, teach_full.parquet


def students_stage():
    soft = os.path.dirname(glob.glob(f"{OUT}/**/teach_full.parquet", recursive=True)[0])
    for tag, a, seeds in (("st_a", "0.5", "0,1"), ("st_b", "0.3", "0,1"), ("st_c", "0.1", "0,1,2,3")):
        sh(["nn2_multi.py"], dict(PLR, TSEEDS=seeds, SOFT=soft, SOFT_ALPHA=a, OUT=f"{OUT}/{tag}"))


def resolve(rel):
    """final/<blend>/test_preds/<component>__<run tag>__<file>  ->  OUT/<run tag>/<file> or OUT/<file>."""
    _, tag, name = os.path.basename(rel).split("__", 2)
    for p in (os.path.join(OUT, tag, name), os.path.join(OUT, name)):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"missing reproduced file {name} (run tag {tag})")


def blend_stage():
    """Copy freshly produced predictions into final/reproduced with the frozen manifest's structure, rebuild."""
    man = json.load(open(os.path.join(HERE, MANIFEST, "MANIFEST.json")))
    for d in ("test_preds", "holdout_preds"):
        os.makedirs(os.path.join(FINAL, d), exist_ok=True)
    for comp in man["components"].values():
        for kind in ("test", "holdout"):
            for e in comp[kind]:
                shutil.copy2(resolve(e["file"]), os.path.join(FINAL, e["file"]))
    json.dump(man, open(os.path.join(FINAL, "MANIFEST.json"), "w"), indent=1)
    sh(["ensemble.py", os.path.relpath(FINAL, HERE), "--holdout"])


STAGES = {"lgb": lgb_stage, "xgb": xgb_stage, "catboost": catboost_stage, "nets": nets_stage,
          "teacher": teacher_stage, "students": students_stage, "blend": blend_stage}

if __name__ == "__main__":
    for s in ([ONLY] if ONLY else list(STAGES)):
        STAGES[s]()
