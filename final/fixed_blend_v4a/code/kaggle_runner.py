"""Kaggle runner: validate candidate arms (2 seeds, far-ahead check), pick the best by
mean holdout AUC, retrain it on all train and write test predictions."""
import subprocess, sys, json, glob, os, time

W = "/kaggle/working"
PY = [sys.executable, f"{W}/train_lgb.py"]
BASE = ["--emb", "--nopost", "--reg", "--lr=0.03", "--fullho", "--stable", "--nocatid"]
ARMS = __ARMS__  # name -> extra args
SEEDS = __SEEDS__
VALID_FLAGS = __VFLAGS__
FULL_SEEDS = __FSEEDS__
ITER_SCALE = __ISCALE__


def run(args, logname):
    t = time.time()
    with open(f"{W}/{logname}", "w") as f:
        r = subprocess.run(PY + args, stdout=f, stderr=subprocess.STDOUT)
    print(f"{logname}: exit {r.returncode} in {time.time() - t:.0f}s", flush=True)
    return r.returncode


res = {}
for name, extra in ARMS.items():
    aucs, iters = [], []
    for sd in SEEDS:
        args = ["valid"] + BASE + VALID_FLAGS + extra + [f"--seed={sd}"]
        before = set(glob.glob(f"{W}/valid_result_*.json"))
        if run(args, f"log_valid_{name}_s{sd}.txt") != 0:
            continue
        new = sorted(set(glob.glob(f"{W}/valid_result_*.json")) - before, key=os.path.getmtime)
        r = json.load(open(new[-1]))
        aucs.append(r["auc_holdout"]); iters.append(r["best_iter"])
        print(f"  {name} seed {sd}: AUC {r['auc_holdout']:.5f} iters {r['best_iter']}", flush=True)
    if aucs:
        res[name] = dict(aucs=aucs, mean=sum(aucs) / len(aucs), iters=iters)
json.dump(res, open(f"{W}/arm_results.json", "w"), indent=1)
best = max(res, key=lambda k: res[k]["mean"])
print("ARMS:", json.dumps(res), flush=True)
print("BEST ARM:", best, flush=True)

if not FULL_SEEDS:
    print("DONE (valid only)", flush=True); sys.exit(0)
# far-check iters were learned on 1 day of data; full train has ~3 days -> scale up
n_iter = int(ITER_SCALE * sum(res[best]["iters"]) / len(res[best]["iters"]))
for sd in FULL_SEEDS:
    run(["full"] + BASE + ARMS[best] + [f"--seed={sd}", f"--iters={n_iter}"], f"log_full_{best}_s{sd}.txt")
print("DONE", glob.glob(f"{W}/submission_*.csv"), flush=True)
