"""Optuna search over LightGBM parameters (G feature set, Indian-local hour), then a confirmed slow run.

Phase 1: search at lr 0.05 on all positives + 25% negatives (AUC-invariant sampling), scored on the
         full time holdout. Trial 0 = current production parameters (reference).
Phase 2: best parameters -> train_lgb.py valid/full at lr 0.01 on all rows (same recipe as the slow
         LightGBM in the blend), writing hopred_*_opt.npy and submission_*_opt.csv.
"""
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
G = ["--emb", "--nopost", "--reg", "--fullho", "--stable", "--nocatid", "--dropE", "--ff=0.5", "--ist"]
SEARCH_HOURS = float(os.environ.get("SEARCH_HOURS", "5.0"))

sys.argv = ["train_lgb.py", "valid"] + G
sys.path.insert(0, HERE)
import numpy as np
import polars as pl
import lightgbm as lgb
import optuna
from sklearn.metrics import roc_auc_score
import train_lgb as T

OUT = T.OUT
rng = np.random.default_rng(T.SEED)
tr = T.load_train(rng, pl.col("feature1") < T.TRAIN_END)
ho = T.load_holdout(rng)
vocab = T.build_vocab(tr)
Xtr, ytr = T.featurize(tr, vocab), tr["watched"].to_numpy()
Xho, yho = T.featurize(ho, vocab), ho["watched"].to_numpy()
del tr, ho
Xtr, names = T.to_np(Xtr); Xho, _ = T.to_np(Xho)
cats = [c for c in T.STR_COLS if c in names]
T.log(f"optuna data: train {len(ytr):,} holdout {len(yho):,} feats {len(names)}")
dtr = lgb.Dataset(Xtr, ytr, feature_name=names, categorical_feature=cats, free_raw_data=False,
                  params={"feature_pre_filter": False, "max_bin": 127})
dho = lgb.Dataset(Xho, yho, reference=dtr, feature_name=names, categorical_feature=cats)
BASE = dict(T.PARAMS); BASE.update(learning_rate=0.05, verbose=-1, num_threads=os.cpu_count())
TUNED = ["num_leaves", "min_data_in_leaf", "feature_fraction", "bagging_fraction", "lambda_l2", "lambda_l1",
         "min_gain_to_split", "cat_smooth", "cat_l2", "max_cat_threshold", "min_sum_hessian_in_leaf", "path_smooth"]


def objective(trial):
    p = dict(BASE)
    p.update(num_leaves=trial.suggest_int("num_leaves", 31, 255, log=True),
             min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 100, 4000, log=True),
             feature_fraction=trial.suggest_float("feature_fraction", 0.2, 0.8),
             bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
             lambda_l2=trial.suggest_float("lambda_l2", 0.1, 200, log=True),
             lambda_l1=trial.suggest_float("lambda_l1", 1e-3, 20, log=True),
             min_gain_to_split=trial.suggest_float("min_gain_to_split", 0.0, 1.0),
             cat_smooth=trial.suggest_float("cat_smooth", 10, 500, log=True),
             cat_l2=trial.suggest_float("cat_l2", 1, 100, log=True),
             max_cat_threshold=trial.suggest_int("max_cat_threshold", 16, 64),
             min_sum_hessian_in_leaf=trial.suggest_float("min_sum_hessian_in_leaf", 1e-3, 20, log=True),
             path_smooth=trial.suggest_float("path_smooth", 0.0, 50.0))
    t = time.time()
    m = lgb.train(p, dtr, 3000, valid_sets=[dho], callbacks=[lgb.early_stopping(100, verbose=False)])
    auc = roc_auc_score(yho, m.predict(Xho, num_iteration=m.best_iteration))
    trial.set_user_attr("best_iter", m.best_iteration)
    T.log(f"trial {trial.number}: auc {auc:.5f} iters {m.best_iteration} ({time.time() - t:.0f}s) {json.dumps(trial.params)}")
    return auc


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
ref = {k: BASE.get(k, 0.0) for k in TUNED}
ref.update(lambda_l1=1e-3, min_gain_to_split=0.0, path_smooth=0.0, max_cat_threshold=int(BASE.get("max_cat_threshold", 32)),
           cat_l2=float(BASE.get("cat_l2", 10.0)), bagging_fraction=float(BASE.get("bagging_fraction", 0.8)))
study.enqueue_trial(ref)  # trial 0 = production parameters
study.optimize(objective, timeout=SEARCH_HOURS * 3600, gc_after_trial=True)
df = study.trials_dataframe(); df.to_csv(f"{OUT}/optuna_trials.csv", index=False)
ref_auc = study.trials[0].value; best = study.best_trial
summary = dict(n_trials=len(study.trials), ref_auc=ref_auc, best_auc=best.value, gain=best.value - ref_auc,
               best_trial=best.number, best_params=best.params)
T.log("OPTUNA " + json.dumps(summary)); json.dump(summary, open(f"{OUT}/optuna_summary.json", "w"), indent=1)
del dtr, dho, Xtr, Xho

if best.value - ref_auc < 0.0002:
    T.log("no meaningful gain over production parameters -> skipping slow confirmation run"); sys.exit(0)
pfile = f"{OUT}/optuna_best_params.json"; json.dump(best.params, open(pfile, "w"), indent=1)
a = G + ["--lr=0.01", "--neg=1.0", f"--params={pfile}"]
subprocess.run([sys.executable, os.path.join(HERE, "train_lgb.py"), "valid"] + a, check=True)
tag = "v2emb_nopost_lr0.01_ist_fullho_dropH_stable_nocatid_dropE_neg1.0_reg_ff0.5_opt"
bi = json.load(open(f"{OUT}/valid_result_{tag}.json"))["best_iter"]
subprocess.run([sys.executable, os.path.join(HERE, "train_lgb.py"), "full"] + a + [f"--iters={int(1.15 * bi) + 1}"], check=True)
T.log("done")
