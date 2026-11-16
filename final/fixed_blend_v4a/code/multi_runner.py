"""For each LightGBM variant: validate on the standard check (all holdout rows, saves holdout
preds for stacking), then retrain on all train with 1.15x the best iterations and predict test."""
import subprocess, sys, json, glob, os, time

W = "/kaggle/working"
PY = [sys.executable, f"{W}/train_lgb.py"]
BASE = ["--emb", "--nopost", "--reg", "--fullho", "--stable", "--nocatid", "--dropE", "--ff=0.5"]
ARMS = __ARMS__


def run(args, log):
    t = time.time()
    with open(f"{W}/{log}", "w") as f:
        r = subprocess.run(PY + args, stdout=f, stderr=subprocess.STDOUT)
    print(f"{log}: exit {r.returncode} in {time.time() - t:.0f}s", flush=True)
    return r.returncode


summary = {}
for name, extra in ARMS.items():
    before = set(glob.glob(f"{W}/valid_result_*.json"))
    if run(["valid"] + BASE + extra, f"log_valid_{name}.txt") != 0:
        continue
    new = sorted(set(glob.glob(f"{W}/valid_result_*.json")) - before, key=os.path.getmtime)
    r = json.load(open(new[-1])); summary[name] = r
    print(f"  {name}: AUC {r['auc_holdout']:.5f} iters {r['best_iter']}", flush=True)
    json.dump(summary, open(f"{W}/multi_summary.json", "w"), indent=1)
    run(["full"] + BASE + extra + [f"--iters={int(1.15 * r['best_iter']) + 1}"], f"log_full_{name}.txt")
print("DONE", flush=True)
