# ShareChat RecSys Challenge — experiment log

Metric: global ROC-AUC on `watched`. Rule: features from the row itself only.

## Data facts
- train: 1,537,638 rows, 2026-10-25 18:00 → 10-28 18:00 · test: 1,037,090 rows, 10-28 18:00 → 10-31 24:00
- watched rate ≈ 5%; ~3.5 rows per request
- feature5 = user id (93% of test rows are unseen users) → not used as a feature
- feature18 ≡ feature24 = post id (84% of test rows seen in train)
- feature3 = hour; feature2/4 = date strings (dropped); feature15 constant (dropped)
- feature26–32: 7 × 32-d vectors, constant per request (user/context embeddings)
- feature40–46: 7 × 32-d vectors, per row (post-side embeddings)
- strongest single pairs: feature30·feature46 (AUC 0.711), feature32·feature43 (AUC 0.716)
- feature409–431: per-row affinity rates (vary within user and within post — not a frozen label snapshot)
- feature320–343: per-user profile values; feature54: per-post (likely duration, 45–300)

## Validation
Train on rows < 2026-10-27 18:00, holdout = last 24h of train (50% sample).
Holdout overlap with train fold: users 5%, posts 91% — close to test (7%, 84%).
Train sampling: all positives + 25% negatives (AUC is invariant to uniform negative downsampling).

## Results
| version | change | holdout AUC | unseen-user AUC | best iter | public LB |
|---|---|---|---|---|---|
| v1 | 584 numeric + 18 categorical (incl. post id), hour; lr 0.05 | 0.8291 | 0.8294 | 111 | — |
| v1_nopost | v1 without post-id categorical | 0.8298 | 0.8302 | 87 | — |
| v2emb_nopost | + 49 user·post dot & cosine, 14 norms | 0.8369 | 0.8370 | 119 | — |
| v2emb_nopost_reg_lr0.03 | leaves 63, min_leaf 800, ff 0.3, l2 30, cat_smooth 200, lr 0.03 | **0.8387** | 0.8389 | 362 | **0.83261** |
| + age | replace raw epoch feature57/58/59 with request-time minus epoch | 0.8386 | 0.8387 | 277 | — |
| DIAGNOSTIC (never submitted) | + request size, rows per user, position in request | 0.8875 | 0.8877 | 670 | — |

| + train stats (te) + age | 10 keys, OOF grouped by user | 0.8383 | 0.8384 | 304 | — |

| neg 0.5 + top-560 feats | 2× negatives, 99%-gain features | 0.8382 | 0.8383 | 290 | — |

| lr 0.015 | same as best, lower LR | **0.8391** | 0.8392 | 732 | — |
| seeds 7 / 99 (lr 0.03) | seed variance | 0.8383 / 0.8383 | — | ~300 | — |
| rank-avg 3 seeds | | 0.8389 | — | — | — |
| rank-avg 3 seeds + lr0.015 | | 0.8390 | — | — | — |

| + raw emb products (f30×f46, f32×f43; 64 feats) | | 0.8382 | 0.8384 | 362 | — |

| + 147 engineered row-local feats (fe.py: dur÷user time, affinity−/÷baseline, popularity per hour & ratios, 7-score agreement) | lr 0.03 | 0.8386 | 0.8387 | 245 | — |

Engineered features took over the top importance ranks (fe_dur_div_* is #1) but AUC unchanged → they re-express information the trees already had.

### Drift-fix results (full holdout, 2 seeds)
| arm | seed 42 | seed 7 | mean |
|---|---|---|---|
| base | 0.8390 | 0.8385 | 0.8388 |
| A dropH | 0.8385 | 0.8387 | 0.8386 |
| B stable | 0.8387 | 0.8387 | 0.8387 |
| **C stable + no creator/category id cats** | **0.8396** | **0.8395** | **0.8396** |
C blend 0.7·C + 0.3·mean(NN v1, NN nopost_lr2e3): holdout 0.8401 → **public 0.83378** (+0.0003 vs previous best 0.83344). Rule-following rank 7/15.
Holdout gain +0.0008 only partly transferred (+0.0003 public); still a ~0.006 holdout→LB gap.

### Far-ahead validation (--far): train Oct25 18:00–Oct26 18:00, skip a day, score Oct27 18:00–Oct28 18:00
| model | seed 42 | seed 7 | mean |
|---|---|---|---|
| base (v2 reg) | 0.8377 | 0.8377 | 0.8377 |
| C | 0.8385 | 0.8385 | 0.8385 |
Looking 1–2 days ahead costs only ~0.001 → most of the 0.006 LB gap is test-day mix (incl. Saturday), not time drift. C's +0.0008 holds on both checks.
Step 2 (far check, on top of C): D no categoricals at all · E drop 454/442/460/472 · F drop user running totals 60–153.

Step 2 results (far check, 2 seeds): D no categoricals 0.8384/0.8384 ✗ · E drop 454/442/460/472 0.8386/0.8388 (mean 0.8387, +0.0002) ✓ · F killed by low laptop memory (moved to Kaggle).
LGB tuning on Kaggle CPU (far check, C features, seed 42; ref 0.83847): feature_fraction 0.5 → 0.83889 (best); leaves 31 / min_leaf 2000 → 0.83865; extra_trees worst 0.83751.
Heavy runs moved to Kaggle (kaggle_runner.py): arms G = C+E+ff0.5, H = G+F; 2 seeds far check → best arm retrained on full train (2 seeds).

Kaggle runner v1 (far check, 2 seeds): G = C+E+ff0.5 → 0.83875/0.83892 (mean 0.8388, +0.0003 vs C) ✓ · H = G+F (drop user totals 60–153) → 0.8377 ✗.
Submission: 0.7·G (2-seed rank avg, full train, iters 1.3×far best) + 0.3·mean(NN v1, NN nopost_lr2e3) → **public 0.83415** (best so far).
Launched: NN with drift fix (nn_df.py: per-hour counters, no 454/442/460/472, no creator/category ids, no post id) on GPU; LGB recency weighting (half-life 1 d / 2 d) on standard check + 5-seed full of best arm on CPU.

Kaggle runner v2 (standard check, full holdout, 2 seeds): G 0.83977/0.83976 · G+recency half-life 1 d 0.8396 ✗ · 2 d 0.8398 (=) → recency dropped. G retrained full with 5 seeds.
Drift-fixed DCN (nn_df.py): df_nopost 0.8341 alone (old nopost 0.8335), rank corr w/ G 0.9645.
Blends on full holdout: G2 + oldNN (w .25) 0.84054 · G2 + (oldNN+dfNN)/2 (w .25) 0.84057.
Candidate out/submission_G5_nnold_df.csv = 0.75·G(5 seeds) + 0.25·mean(oldNN pair, dfNN) — held until column sweep finishes.
Not done on purpose: adversarial (test-similarity) weighting — uses test feature distribution in training; grey zone vs "serving-time" rule.

Column-group sweep (far check, seed 42, drop one block from G; ref 0.83875): best 71–83 +0.0002, post cats 19/21/25 +0.00015 (noise level); worst 54–56 −0.0030, 47–53 −0.0009, 33–39 −0.0006. No strongly harmful block left.
Submission G5 blend (0.75·G 5 seeds + 0.25·mean(old NN pair, drift-fixed NN)) → **public 0.83439** (best).
Launched: run3 (creator/category train stats --te, drop 71–83+19/21/25, combo; far check 2 seeds; best → 3-seed full) and CatBoost GPU on G features.

run3 (far, 2 seeds): G 0.83884 · G+creator/category train stats 0.83870 ✗ · G+drop 71–83,19,21,25 0.83885 (=) · both 0.83862 ✗.
CatBoost GPU (G features, all rows): 0.8387 alone (near check), corr w/ G 0.981; blend 0.5 G + 0.25 CB + 0.25 NN = 0.84082 (vs 0.84057) → **public 0.83442** (+0.00003; official 24/37).

### Error analysis (G, near holdout, AUC 0.8401)
- Video length dominates: watch rate ~7% under 60 s, 4.4% at 110 s, then a cliff to ~1% at 120 s and ~0.06% beyond 230 s. Within-length AUC only 0.69–0.80 → much of global AUC is "short vs long".
- feature10 / feature13 split users into two regimes: high watch (5.5–7.5%, AUC ~0.80–0.81) vs low watch (3–4%, AUC ~0.87–0.89) — likely surface/feed type or app.
- Weaker: fresh posts (6–24 h, 0.807), posts unseen in train (0.824), hours 0/6/15; most active users (0.81) vs least active (0.87).

Segment specialists (feature10 high-watch group, 76% of rows): global 0.83977 · specialists 0.83907 · avg w0.3 0.84002 (bagging effect only) → not used.
Dates/timestamp check: feature2/4 = request date (no signal), minute/second AUC 0.500, feature15 constant, feature3 = UTC hour (IST is +5:30 → --ist switch).
Duration × match: watch rate <60 s goes 2.2% → 15.8% from lowest to highest cos(f32,f43) quartile; ≥120 s stays 0.1–0.5%. Already captured by trees.
Hard-subset AUC by model (std): LightGBM best on every subset (<60 s 0.785, 60–120 s 0.795, ≥120 s 0.811, high-watch 0.811, fresh posts 0.807); towers ≈ LGB only on fresh posts (0.8067).
Monotone (longer video never raises p): far 0.83890 vs 0.83884 → not significant; patience network skipped.
New models (std, full holdout): XGBoost GPU (G+ist, all rows) **0.8400**; CatBoost depth 10 0.8388; towers 3 seeds 0.8352/0.8347/0.8347.

9-model learned stack: CV 0.84096 → public 0.83436 (below 0.83442) → learned weights overfit holdout; switch to fixed weights.
Test-mix validation: holdout new-post share 9.0% vs test 16.1%; all models ~0.018 worse on new posts (LGB 0.8237 vs seen 0.8415). New posts are young (median 32 h vs 764 h), not missing data.
LGB variants (std, seed 42; G 0.83977): +IST hour 0.83995 · +lr 0.015 0.84021 · +lr 0.01 all rows 0.84024.
Young-post experiments (2 seeds, test-mix): global 0.83933 · young specialist blend 0.3 0.83937 · young ×3 weight 0.83878 → no gain.
Fixed blend v1 = 0.4 LGB (lr0.01-all + G5) + 0.3 XGB + 0.15 CB×2 + 0.15 NN (towers×3 + DCN v1): holdout 0.84095 / test-mix 0.83984 → **public 0.83447** (best; official 26/40).

### Request-recoverability diagnostic (never submitted)
Train LightGBM to predict request size / position from row-only features (G feature set), test on unseen requests:
req_size R² 0.316 (Spearman 0.56); true size alone AUC 0.759 vs row-predicted size alone 0.638.
req_pos R² 0.154 (Spearman 0.37); true position alone AUC 0.675 vs row-predicted 0.639.
→ The part of request structure that is visible in a row is already available to our model; the extra signal behind the ~0.888 scores is only obtainable from other rows (banned by the host).

Seeds/variants (std, full holdout): XGB s1/s2/s3 0.8400/0.8403/0.8400 · slow LGB s42/s7/s99 0.8402/0.8401/0.8404 · DART (lr .05, 1500 rds) 0.8398 ✗ · LGB 127 leaves/min_leaf 400 0.8402 (=).
Fixed Blend v2 (v1 weights, slow LGB ×3 seeds, XGB ×3 seeds): holdout 0.84102 / test-mix 0.83992 → **public 0.83456** (best; official 26/42). Frozen at final/fixed_blend_v2, rebuilt via ensemble.py.
Blend variants (holdout plain / test-mix): v2 0.84102/0.83992 · +DART+big in LGB group 0.84113/0.84002 · trees-heavy 0.84096 · NN-heavier 0.84106 · equal LGB/XGB 0.84103.
Fixed Blend v2b (LightGBM component = 4 equal groups: slow×3, G×5, DART, 127-leaf) rebuilt holdout 0.84111 → **public 0.83460** (best).
Overnight jobs: sharechat-night-seeds (towers seeds 3–5 + CatBoost seed 7, GPU), sharechat-repro-lgb (run_all.py --only lgb end-to-end reproduction, CPU).
Kaggle env: xgboost 3.4.1, catboost 1.2.10, torch 2.11.0+cu130.

## Structured multi-tower network (nn2.py, Kaggle)
user / post / pair towers + 49 cosine matches + 7 learned bilinear matches + DCN cross; rank-transformed numerics.
| config | full-holdout AUC | rank corr w/ LGB | best blend w/ LGB |
|---|---|---|---|
| t_base (lr 1e-3, bs 8k, drop 0.3, H 192, 4 ep) | **0.8350** | 0.974 | 0.8394 |
| t_nopost / t_wide_drop4 / t_lr2e3_bs16k | 0.8347–0.8350 | 0.974–0.975 | 0.8393–0.8394 |
Best single NN so far and no longer overfits, but *more* similar to LGB → blends less than the old DCN (0.8398). Stop tuning NNs (rule: keep only if ≥0.836 alone or corr <0.95).

## Noise floor
Same config, seeds 42/7/99: 0.8387 / 0.8383 / 0.8383 → single-run differences < ~0.0005 are noise. From here: full holdout + 2 seeds per arm; believe only ≳ +0.001.

## Adversarial validation (adv_val.py) — train vs test separable
AUC 0.981. Top drift drivers: popularity counters 255–286 (+237) ~50% of gain (feature279 alone 16%), upload epochs 58/59 ~9%, post/creator/category ids 23/20/12/7/9 ~15%, some user×post rates 454/442/460/472 ~10%.
Hypothesis: the 0.006 holdout→LB gap is drift (test is 1–3 days past train; holdout only ≤1 day past its fold).
Arms (2 seeds, full holdout): base · A dropH (no raw 154–286, 57–59) · B stable (A + log counter per post-age hour, counter log-ratios, ages) · C stable + no creator/category id categoricals.

## Neural network (nn.py, Kaggle GPU, DCN-v2 cross + deep MLP)
Inputs: 584 numeric (signed log1p + standardise), 49 user·post dots, all 14×32 raw embeddings, 18 categorical embeddings (incl. post id, hour). Full data, no downsampling. ~6 s/epoch on GPU.
| version | holdout AUC (full holdout) | same-rows AUC | blend w/ LGB lr0.015 (rank) |
|---|---|---|---|
| nn v1 (lr 1e-3, wd 1e-5, drop 0.25, best epoch 2 of 5) | 0.8337 | 0.8340 | **0.8398** (w_nn 0.25), rank corr 0.964 |
| nn v2 configs (wd 1e-4/1e-3, drop 0.3/0.4, bs 4k–16k, lr 5e-4–2e-3, ±post id) | 0.8321–0.8338 | 0.8327–0.8338 | 0.8395–0.8398 each |
| LGB + avg of 6 NN (rank, w_nn 0.30) | | | **0.83985** |
| submission_blend_lgb07_nn03.csv: 0.7·LGB(lr0.015) + 0.3·mean(NN v1, NN nopost_lr2e3_bs16k) | | | ≈0.8398 → **public 0.83344** (+0.0008 vs v2) |
Test-set rank correlations (LGB–NN 0.968, NN–NN 0.982) match holdout → blend should transfer.

NN overfits after epoch 2 → v2 run tries 5 configs (wd, dropout, batch, lr, with/without post-id embedding).

Seed averaging adds ~nothing (models too similar). Best single: lr 0.015.

LightGBM has plateaued at ~0.838 on these features; changes within ±0.0005 (holdout noise). Next gains: NN + blending.

Train stats did not help: post rate became a top-3 feature but displaced existing precomputed post-history features (409–431, 480–515).

## Rules clarifications (host, Discussion, 2026-11-14)
- "Features derived from request_id grouping are not allowed." Non-compliant solutions are rejected at review; valid ones move up.
- "user/item-level statistics computed only from the training data, are allowed as features."
- The 0.88 public-LB cluster matches our cross-row diagnostic (0.8875) → almost certainly request-level features.

## Train-statistics features (te.py)
Keys: post (f18), f23, f20, f9, f12, f7, f10, f14, f19, user (f5). Count + smoothed watch rate (α=30).
OOF 5-fold grouped by user for rows inside the reference set; full-reference stats for holdout/test.
Single-feature AUC, OOF vs holdout: post rate 0.711 / 0.702, f20 rate 0.675 / 0.672, user rate 0.50 / 0.51 (consistent → no leak).

Fixed Blend v3 (v2b + towers x6 seeds + CatBoost seed 7): holdout 0.84112 -> **public 0.83462** (best; official 27/43). Frozen at final/fixed_blend_v3.

## 2026-11-16 checklist pass
- Duplicate check: 1,537,553 user×post pairs in 1,537,638 rows; only 85 repeated pairs (170 rows), 5 conflicting (10 rows); 0 exact (user,post,request) duplicates → data clean, nothing to fix.
- End-to-end reproduction (Kaggle sharechat-repro-lgb, `run_all.py --only lgb`): all 6 LightGBM test predictions bit-identical (max diff 0.0) to the frozen ones (multi_k, run_k2); slow LGB holdout 0.84024 reproduced exactly.
- Not applicable for AUC (order-preserving): threshold tuning, calibration (Platt/isotonic). Trees scale-invariant (no scaling/PCA). SMOTE/ADASYN skipped (1.5M rows, ample positives). Pseudo-labelling on test skipped (trains on test rows; same grey zone as adversarial weighting).
- Launched: Kaggle CPU sharechat-optuna-lgb (tune_optuna.py): 5 h TPE search at lr 0.05 / neg 0.25, trial 0 = production params; if best beats it by ≥0.0002 → slow lr 0.01 all-rows valid+full with train_lgb.py --params (tag suffix _opt).
- Launched: Kaggle GPU sharechat-nn-focal — towers net, arms base / focal γ=1 / focal γ=2 / label smoothing 0.01, seeds 0,1 each (nn2_multi.py NNARMS, TSEEDS). Keep an arm only if ≥ +0.0005 over base on the 2-seed mean.

## 2026-11-16 error analysis of Blend v3 (validation day)
Segment AUC (blend / LGB / XGB / CB / NN): seen video 0.8425/0.8421/0.8419/0.8410/0.8386 · NEW video (9% of holdout, 16% of test) 0.8255/0.8246/0.8251/0.8240/0.8193 · <60 s 0.7846 · 60–119 s 0.7904 · 120 s+ 0.8272 · f10 high-watch 0.8151 · other 0.8511.
Mean AUC inside video-length deciles 0.7842 vs global 0.8411. Top-10% disagreement rows: every family ranks LGB ≥ XGB > CB > NN → no segment where another family wins → gating / mixture of experts has nothing to switch to.
Post embeddings 40–46 are fixed per video (cosine 0.999 same video vs 0.226 random) → usable as a video content key.

## Embedding-neighbour video stats (knn.py, --knn)
Per video: 50 nearest other videos (cosine on 7×32 post embeddings) among videos with ≥20 reference impressions; similarity-weighted watch rate, length-adjusted watch ratio (all/top-10), top/mean similarity, log neighbour impressions. Train-data labels only; reference rows OOF by 5 user folds (as te.py); never self.
Sample check (3 row groups → last row group): knn_rate single-feature AUC OOF 0.653 / later rows 0.644 / new videos 0.639 → consistent, no leak sign.
Launched Kaggle CPU sharechat-knn-lgb: G lr0.03 + --knn, seeds 42/7 valid + full. Baseline (same, no knn, repro run): 0.83977 / 0.83976. Keep if ≥ +0.0005 mean, and check new-video AUC.

## 2026-11-16 evening results
- KNN (G lr0.03 + --knn, 2 seeds): 0.84005 / 0.83986 vs base 0.83977 / 0.83976; 2-seed rank-avg 0.84031 vs 0.84008 (+0.0002); new videos 0.8241 vs 0.8237; test-mix 0.83916 vs 0.83891; rank corr with base 0.998. Below +0.0005 bar.
- Focal loss (towers, seeds 0/1): base 0.83524/0.83470 · focal γ1 0.83516/0.83488 · focal γ2 0.83532/0.83471 · label smoothing 0.01 0.83518/0.83461 → no gain.
- Optuna (132 TPE trials, lr 0.05/neg 0.25): best +0.00043 over production params at search scale; slow lr 0.01 all-rows with best params 0.84025 vs 0.84024 → no gain. Params were already near-optimal.
- Blend v3 + KNN group / + Optuna group / + both / + focal nets: plain 0.84109–0.84113, test-mix 0.83998–0.84002 vs v3 0.84112 / 0.84002 → nothing transfers to the blend. Not submitted.
- Cold-start masking skipped: new videos have complete history columns (409-431, 480-515 0% null, same as seen), only smaller counters -> masking would create inputs never seen at test.
- Fixed Blend v4a (v3 + KNN LGB group) holdout 0.84113 -> public 0.83461; v4b (+ Optuna slow LGB group) 0.84112 -> public 0.83461. v3 (0.83462) stays best.
- Overnight: Kaggle CPU sharechat-knn-slow (lr0.01 all rows --ist --knn, seeds 42/7, valid+full); Kaggle GPU sharechat-repro-gpu (run_all.py --only xgb/catboost/nn end-to-end reproduction).

## 2026-11-17 morning — overnight results
- **Slow KNN LGB** (Kaggle sharechat-knn-slow, lr0.01 --ist --neg=1.0 --knn, seeds 42/7): holdout 0.84015 / 0.84014 vs slow LGB without KNN 0.84024 → slightly worse alone.
  In Blend v3 (plain / test-mix): v3 0.84112/0.84002; +slowKNN group 0.84111/0.84000; in slow group 0 0.84112/0.84001; +both KNN groups 0.84112/0.84001. **No gain → no v5, not submitted.** KNN line closed.
- **GPU reproduction** (Kaggle sharechat-repro-gpu, run_all.py --only xgb/catboost/nn): not bit-identical (GPU nondeterminism; spearman 0.997–1.000, DCN 0.976), single-model AUCs within ±0.0002 (DCN −0.0006).
  v3 blend with reproduced XGB/CatBoost/NN files: **0.84113 vs frozen 0.84112 → reproduction PASSED.**
- Official rank 17 Nov 11:28 IST: 29/46, public 0.83462 (Blend v3). 22 teams ≥ 0.88; nearest above 0.83476 (#26), 0.83528–0.83567 (#22–25).

## 2026-11-17 — "are we missing a clue?" check
- Persistent-user-id hunt: every string column scanned (cardinality, rows/value, values/request, train→holdout/test coverage, train-rate AUC). feature5 = 421k values, 3.7 rows each, 1.00 per request, 5% holdout seen, AUC 0.507 → a session id, no persistent user exists. Best item keys already used (f18/f24 0.70, f20 0.67, f23 0.62).
- Forum (read 17 Nov): host ysaraf: "the eval is intended to run on each row of the eval set independently of the other rows". Host Avishek Pal (2 d ago): "Only request_id-derived features are not allowed." Thread 758976: the teams then ranked 16th and 20th (both ≥0.88) wrote publicly that they had used request-level features and dropped them; host: non-compliant solutions rejected, valid ones move up. Public LB keeps old best scores.

## 2026-11-17 — TabM launched
nn2_multi.py: added arch="tabm" (BatchEnsemble MLP, k=16, 3x512, drop 0.1, lr 2e-3, wd 3e-4; flat input = numeric ranks + cat embeddings + raw 2x7x32 embeddings + cos/bilinear). Smoke (20k rows): tabm 0.817 vs towers 0.812. Kaggle sharechat-tabm (GPU): NNARMS=tabm, seeds 0,1,2 -> nn_holdout_pred_t_tabm_s*.npy, submission_nn2_t_tabm_s*.csv.
- TabM result: seeds 0.8329/0.8317/0.8317 (3-seed rank-avg 0.83353) vs networks component 0.83702; spearman 0.984. In Blend v3: as nn group 0.84111/0.84000, extra 5% 0.84111/0.84000, 10% 0.84105 → **no gain, not submitted.**
- Recent-window video rates (holdout, single-feature AUC): all-history 0.7016 · last 24h 0.6909 · 12h 0.6842 · 6h 0.6685 (worse in both early and late holdout hours). All-history rates already gave no model gain → recency stats dropped, not built.

## 2026-11-17 afternoon — controlled seed experiment (option C), rules fixed BEFORE results
Policy: Blend v3 = primary, v2b = backup. Only change the blend on a measured validation gain.
(Option B "user-type split" was already tested 16 Nov — segment specialists 0.83907 vs global 0.83977 → not repeated.)
Launched: Kaggle sharechat-lgb-seeds2 (CPU; slow LGB lr0.01 --ist --neg=1.0, seeds 11/23/2026, valid + full) and sharechat-xgb-seeds2 (GPU; xgb_seeds.py XSEED 4/5/6).
Acceptance rule (decided in advance): add the seeds to their existing groups (slow LGB group 0, XGB group) →
Blend v5 only if holdout gain >= +0.0002 on BOTH plain and test-mix, and every new seed's single AUC is within 0.0005 of the existing seeds.
Otherwise v3 stays and nothing is submitted. Either way: no other blend changes.

## 2026-11-17 evening — fresh "what did we miss" audit (data checks)
- Raw epoch columns: feature57 = user signup time (1 value per request), feature58/59 = video upload time (1 value per video). No per-impression timestamp → no hidden feed position. Gap features AUC 0.45–0.51.
- All ~580 numeric columns (incl. ones dropped in G): max |corr| with position *within* request = 0.01 → no row-local column encodes position. Request size correlates ~0.35 with user activity columns (153, 152, 117, 320) — already used.
- User-group × video train rates (f6, f7, f10, f11, f13, f14, f16, f17, f8 × f18): global AUC 0.64–0.67 (< video rate 0.70); size-weighted AUC *inside* videos 0.497–0.503 → zero information beyond the video rate. Dropped.
- RESULT (18 Nov 00:50): LGB seeds 11/23/2026 = 0.84015/0.84038/0.84021; XGB seeds 4/5/6 = 0.84002/0.84006/0.84012 (all within 0.0005 of existing seeds). Blend: v3 0.84112/0.84002 · +LGB 0.84113/0.84002 · +XGB 0.84113/0.84002 · +both 0.84114/0.84003 (+0.00001). Below the +0.0002 rule → **v3 kept, nothing submitted.** Seed averaging is saturated.
- Public Code tab (18 Nov): only 1 public notebook (avikdas567 "Latent Tensor Geometry & Temporal RecSys Stacking", 11 Nov). Features: raw numeric + per-pair embedding dot/cos/L2/norms + OOF post(f18)/f23 target rates + hour sin/cos + 3 ratios; LGB/XGB/NN rank blend. Its own reported validation: LGB 0.8261, XGB 0.8266, NN 0.8242, blend 0.8279 (ours 0.8411 on a comparable time holdout). Nothing in it we lack (L2 distance is the only extra; derivable from our dot+norms).
- Forum question posted 18 Nov ~01:00 IST (user approved): https://www.kaggle.com/competitions/sharechat/discussion/759936 — asks host for ShareChat row-level baseline AUC and whether request_id may be used in training only (ranking group / distillation) with row-independent inference.
- Launched Kaggle sharechat-distill-valid (CPU, VALIDATION ONLY, distill.py): teacher = LGB + request size/pos/rows-per-user, OOF 4 folds grouped by feature5; students row-only: S0 hard, S1 0.5y+0.5teacher, S2 teacher (cross_entropy), L1 lambdarank grouped by request. Rule: submit only if host OKs training-time request_id AND student gains >= +0.0005 over S0 (and in blend).
- 18 Nov: post edited (user approved) to a neutral version — title "Question on request_id usage and row-level baseline"; removed our score and the named techniques. 0 comments at edit time.

## 2026-11-18 ~01:30 — where exactly are we stuck? (Blend v3, validation day)
- Overall 0.8411. Video-only part (each row = its video's mean score) 0.7646.
- Inside-video AUC (which user watches a given video) **0.7373** (60% of rows in mixed-outcome videos) → main weakness = user side.
- Inside-request AUC (which of a request's videos is watched) 0.7152 (13% of rows).
- Length bands: <60 s 0.7846 · 60–119 s 0.7904 · 120 s+ 0.8272.
- "Oracle" same-day video watch rate looked like 0.856 but that included each row's own label; leave-own-row-out = **0.7063** → better video-popularity estimation is NOT the missing piece. Same-day video impression counts: AUC 0.489, hurt the blend (0.8256) — and would be test statistics anyway.
New model families launched: Kaggle sharechat-gtf (GPU; feature-group Transformer, arch "gtf": ~78 tokens = numeric chunks of 16 + categoricals + 7 user/7 post/7 product embedding tokens, 3 layers d=64, lr 1e-3, bs 2048, 8 epochs, seeds 0,1) and sharechat-lgb-linear (CPU; LightGBM linear_tree, G lr0.03 --ist, valid + full). Smoke (15k rows, 30 steps): gtf 0.598 vs towers 0.812 — slow start expected at that size.

## 2026-11-18 — results of distill-valid / GTF / linear-tree LGB
Holdout (plain / test-mix). Blend v3 = 0.84112 / 0.84002. Acceptance bar: +0.0002 on both.
- **Linear-leaf LightGBM** (G + lr0.03 + ist + linear_tree, lambda 1): single 0.84027 (vs G lr0.03 ist ≈0.83995, +0.0003).
  Spearman 0.996 with the LGB component. As an extra LGB group: 0.84116 / 0.84005 (+0.00004/+0.00003) → below bar, NOT adopted.
- **GTF group Transformer** (2 seeds): single 0.8314 (best epoch 3, then overfits). As a networks group: 0.84111 / 0.84000 → no gain; extra weight hurts. NOT adopted.
- **Distillation (validation only)**: teacher OOF 0.8861 (with request features). Row-only students: S0 hard 0.83934, S1 soft50 0.83972 (+0.0004, below the +0.0005 bar), S2 soft100 0.83931.
  S1 inside Blend v3: 0.84111 / 0.84001 → no gain. Lambdarank stopped at iter 2 (0.732): broken setup, not worth fixing given the S1 result.
  Conclusion: teacher knowledge of request structure does NOT transfer to a row-only model in a useful way. Not submitted (and would need host OK anyway).
- Forum 759936: 0 replies. Official rank 33/54, public 0.83462 (v3). No change.
- Holdout→public drop is a steady ~0.0065 for every submitted blend (v1–v4b), no single-family public scores yet.
- Time-decay check on the holdout (6 h buckets, v3 components): no decay with time (0–6 h 0.854, 6–12 h 0.839, 12–18 h 0.839, 18–24 h 0.841, same shape for every family).
  New-video share rises 2%→13% through the day. Family ranking is the same in every bucket (LGB≈XGB > CB > NN) → no family holds up better later; reweighting by holdout time is pointless.
  Holdout covers only 0–1 day after training; test is 0–3 days → only public single-family submissions can show whether a family decays more slowly on days 2–3.
- **Single-family public diagnostics (18 Nov, user OK, 4 subs)** — Blend v3 components submitted alone:
  | component | holdout plain | public | drop |
  | LightGBM | 0.84069 | 0.83435 | 0.00634 |
  | XGBoost  | 0.84060 | 0.83411 | 0.00649 |
  | CatBoost | 0.83964 | 0.83278 | 0.00686 |
  | networks | 0.83702 | 0.83116 | 0.00586 |
  | Blend v3 | 0.84112 | 0.83462 | 0.00650 |
  Drop is uniform (±0.0003) → no family holds up better on days 2–3; reweighting will not help. Holdout ranks families correctly for test.

## 2026-11-18 — pseudo-labelling, VALIDATION ONLY (user OK)
Rules check (read 18 Nov): Overview "Important Note" = serving-time info only, each row predicted independently using only its own features; Rules 7b bans hand-labelling of test. Pseudo-labelling / test-similarity weighting not named → grey area (learning from future unlabelled rows).
Launched Kaggle CPU sharechat-pseudo-valid (pseudo.py): holdout day = unlabelled "test". S0 train only · P1 + holdout soft labels w1 · P2 w0.3 · P3 confident hard labels (top 1.5% / bottom 50%) · P4 two rounds. Also AUC at S0's iteration count.
Rule (fixed in advance): consider only if best arm >= +0.0005 over S0 AND gain survives inside Blend v3; then ask host before any submission. Otherwise close.
  Networks lose least (0.0059 vs 0.0063–0.0069) → small hint that more network weight may transfer slightly better; NN-heavier blend was −0.00006 on holdout (16 Nov).
- v5nn (v3 files, NN weight 0.25): holdout 0.84109/0.83997 → **public 0.83461** (v3 0.83462). Network weighting closed.
- Overnight 18→19 Nov: launched Kaggle CPU sharechat-lgb-linslow-s42 / -s7 (linear-tree LGB, lr 0.01, all negatives, ist; valid + full).
  Rule: add as a LightGBM group in v3 → freeze v5 and submit only if ≥ +0.0002 on both plain and test-mix. Pseudo-label result: never submitted without host OK.
- **Pseudo-label RESULT** (S0 0.83934): P1 soft w1 0.83953 (+0.0002; at S0's 216 iters 0.83933 = 0 → gain is only from more rounds) · P2 w0.3 0.83939 · P3 confident hard 0.83794 ✗ · P4 two rounds 0.83946.
  In Blend v3: P1 0.84111/0.84000, P4 0.84109/0.83999 vs 0.84112/0.84002 → **no gain. Pseudo-labelling closed; no host question needed.**
- 19 Nov: slow linear-tree LGB seed 42 (lr 0.01, all negatives): **0.83715**, best iter 183 (stops early, worse than lr0.03 version 0.84027); in Blend v3 0.84103/0.83992 ✗. Seed 7 still running — irrelevant. Linear-tree line closed. Nothing submitted.
- Laptop watcher stopped for low system memory (8 GB laptop); not restarted.
- 19 Nov: slow linear-tree seed 7 = **0.84065** (best iter 979; best single LightGBM so far — seed 42's 183-iter stop was an early-stopping accident).
  Blend v3 + s7 group 0.84115/0.84004 · + s7 & lr0.03 linear as two groups 0.84117/0.84007 (+0.00005) → below +0.0002 rule; not submitted.
- Official rank 19 Nov: 34/59 (new team at 0.83482), public 0.83462.
- 19 Nov: **Hybrid launched** (Kaggle GPU sharechat-hybrid): nn_oof.py = towers network, 5 random folds, 4 epochs → out-of-fold score per train row (holdout/test = mean of 5 fold nets); then train_lgb.py --nnf (G lr0.03 ist + h_nn) valid + full.
  Row alignment of the join checked locally (rn offsets exact); smoke run OK. Baseline G lr0.03 ist ≈ 0.83995. Rule: as a LightGBM group in v3 must give ≥ +0.0002 on both plain and test-mix.
- **Hybrid RESULT**: network OOF AUC 0.8320 (train folds), 0.8350 on holdout (5-fold mean). LightGBM + h_nn: **0.83976** (G lr0.03 ist without it ≈ 0.83995) — h_nn became the #1 feature by gain but replaced information instead of adding it.
  In Blend v3: 0.84113/0.84002 (+0.00001) → no gain, not submitted. Hybrid closed.
- 19 Nov evening — two new directions launched (never tried before for trees):
  1. Raw embedding coordinates as LightGBM inputs (train_lgb.py --embval=post: 7×32 post vectors = video content; --embval=all: + 7×32 user vectors). Kaggle CPU sharechat-embval-post / -all, G lr0.03 ist, valid + full. Baseline ≈ 0.83995.
  2. CatBoost native embedding_features (catboost_emb.py; CPU, lr 0.08, depth 8, all negatives; 14 vectors as embedding features → CatBoost's own LDA / nearest-class features). Kaggle CPU sharechat-catboost-emb. Existing CatBoost component 0.8396.
  Rule unchanged: inside Blend v3 ≥ +0.0002 on plain and test-mix.
- **Raw-embedding LightGBM RESULT**: --embval=post 0.83969, --embval=all 0.83974 (base ≈ 0.83995) → no gain alone; in Blend v3 0.84111/0.84001 and 0.84113/0.84002 → not adopted. Trees already get the useful part via dots/cosines.
- Official rank 19 Nov evening: 36/63 (new teams at 0.83558 and 0.83480), public 0.83462. CatBoost-emb still running.
- 19 Nov night — **new angle: tune what the holdout cannot measure, on the public LB.** Holdout = 0–1 day after training; test = 0–3 days (incl. Fri/Sat). Full-fit round count (always 1.15× holdout best) and recency weighting were never judged on test-day data.
  Kaggle CPU sharechat-lbtune: G lr0.03 ist valid once, then 5 full fits: V0 1.15× rounds (reference) · V1 0.80× · V2 1.60× · V3 recency half-life 1 d · V4 half-life 0.5 d. Submit all 5 as diagnostics (single LGB).
  Rule: a variant counts only if ≥ +0.0003 public over V0 (public precision ~0.00001 between near-identical models); then apply the same setting to the blend's LightGBM/XGBoost full fits.
- **lbtune RESULT (public, single LGB G lr0.03 ist, holdout 0.83995, best iter 366)**: V0 1.15× rounds **0.83346** · V1 0.80× 0.83331 (−0.00015) · V2 1.60× 0.83348 (+0.00002) · V3 recency 1 d 0.83319 (−0.00027) · V4 recency 0.5 d 0.83249 (−0.00097).
  → Round count is on a plateau (our 1.15× is right); recency weighting HURTS on test days, more strongly than on holdout → test rewards long history, not recent behaviour. No setting passes +0.0003. Closed.
- Laptop watcher killed again for low memory (not restarted).
- 19 Nov: **everything combined** (v3 + 5 new LightGBM groups: KNN, Optuna, linear-tree ×2, hybrid, raw-embedding ×2; + TabM ×3, GTF ×2 in networks): 0.84112–0.84114 / 0.84001–0.84004 vs v3 0.84112/0.84002 → combining all new models adds nothing (+0.00002 at best).
- Team mergers are allowed (rules 3b; no separate merger deadline on the timeline; combined submission count must be ≤ 5 × days running). Our count: 23. Not pursued.
- 19 Nov night — **diverse feature views** (blend diversity, not single-model strength): Kaggle CPU sharechat-views, LightGBM G lr0.03 ist with deliberately missing information: nohist (no 409–431, 480–515), noemb (no embedding match features), norate (no 409–431). Weaker alone by design; judged only by gain inside Blend v3 (≥ +0.0002 both).
- **Views RESULT**: nohist 0.83952 · noemb 0.83353 · norate 0.83973 (full G ≈ 0.83995). Rank correlation with the LGB component 0.981–0.997 → even with whole blocks removed the models rank rows almost identically.
  In Blend v3 at 5–25%: best 0.84113/0.84003 → no gain. Not adopted.
  Finding: ShareChat's history-rate columns (409–431, 480–515) are almost fully redundant (−0.0004 when removed); embedding match features carry −0.0064.
- 19 Nov research — RecSys Challenge 2023 (ShareChat, 9th place, Kim & Lee arXiv 2310.03778): gains came from frequency encoding (we tried train-period counts/rates in te.py: no gain) and "denoising" scaled-integer columns (arithmetic-sequence values ÷ step).
  Our data checked (row group 3, 484 numeric columns): no injected-noise pattern — the only regular grids are float32 precision steps (2^-8 … 6e-8); "near-integer" flags were small rates (0–0.05). Denoising is also monotone, so trees would not benefit. Closed.
- 19 Nov night — research sweep (TabArena NeurIPS 2025, CTR benchmarks, RecSys 2023). Only strong untried item for us: PLR periodic numeric embeddings (Gorishniy et al. NeurIPS 2022; used inside TabM/RealMLP top configs).
  nn2_multi.py arch "plr" = TowersPLR (per-feature sin/cos frequencies k=16 → linear d=8 → ReLU, before the towers). Smoke (15k rows) 0.728 vs towers 0.812 — slow start on tiny data, as GTF did.
  Kaggle GPU sharechat-plr-a (sigma 1.0) and -plr-b (sigma 0.3), seeds 0,1. Networks component = 0.83702; judged by blend gain (≥ +0.0002 both).
- **PLR RESULT** (2 seeds each): sigma 1.0 0.83652/0.83678 (best ep 3) · **sigma 0.3 0.83869/0.83810 (best ep 6 = last, still rising)**; 2-seed rank-avg 0.83887 = best network ever (towers ×6 component 0.83702). Spearman vs networks 0.981, vs LGB 0.985.
  Blend v3 with PLR-b replacing the networks component: w0.15 0.84119/0.84009 (+0.00007) · w0.20 0.84122/0.84012 · w0.25 0.84123/0.84013 (+0.00011) · w0.30 0.84121/0.84012. PLR-a (sigma 1) hurts.
- CatBoost-emb: stopped at Kaggle's 12 h limit during the full fit; valid 0.83836 (best iter 497); as a CatBoost group 0.84108/0.83996 ✗. Closed.
- Launched (GPU, max 2 at a time): sharechat-plr-c (sigma 0.3, 10 epochs, seeds 2–5), sharechat-plr-d (sigma 0.1, 10 epochs, seeds 0–1).
- Night plan 19→20 Nov: watcher on plr-c / plr-d; build_v5.py swaps PLR nets into the networks component. Builder check: PLR-b w0.15 → 0.84119, w0.25 → 0.84123 (matches eval).
  After 05:30 IST reset: submit the best PLR blend if it beats v3 on both plain and test-mix (candidate final if ≥ +0.0002, otherwise a public check).
- 19 Nov 23:20 — PLR-c (sigma 0.3, 10-epoch schedule, seeds 2–5): 0.83845/0.83840/0.83742/0.83792 (best ep 3–5); PLR-d (sigma 0.1): 0.83779/0.83719.
  Blend: c alone as networks 0.84108/0.83998 ✗ · b+c (6 seeds) w0.25 0.84114/0.84004, old nets+PLR mix 0.84118/0.84008 · b+c+d ≈ v3.
  → PLR-b's +0.00011 was largely seed luck; robust gain ≈ +0.00002–0.00006. Testing whether the 6-epoch schedule matters: plr-f (sigma 0.3, 6 ep, seeds 2–5), plr-g (sigma 0.5, 6 ep, seeds 0–1).
- 20 Nov 00:00 — PLR-f (sigma 0.3, 6 epochs, seeds 2–5): 0.83839/0.83840/0.83848/0.83828; 4-seed avg 0.83908; blend w0.25 0.84120/0.84011 → **6-epoch schedule reproduces the gain** (10-epoch run c did not). PLR-g sigma 0.5 worse.
  **Fixed Blend v5 frozen** (final/fixed_blend_v5): v3 trees (rescaled) + 6 PLR nets (b+f) as networks, weight 0.25 → holdout **0.84122 / 0.84013** (v3 0.84112/0.84002). +0.0001 consistent across seed sets; below +0.0002 rule → public check at 05:40, not auto-final.
  Launched plr-h (k 24, d 16), plr-i (sigma 0.2).
- 20 Nov 00:02: laptop watcher and the 05:40 timer both stopped for low memory; not restarted. v5 is frozen and ready; plr-h/i still running on Kaggle.
- 20 Nov 08:00 — **v5 public 0.83471** (v3 0.83462; holdout +0.00010 → public +0.00009). **New best.**
- PLR-h (k 24, d 16, sigma 0.3, 6 ep): 0.83888/0.83888; 2-seed 0.83923. Blend w0.25 **0.84130/0.84021** (+0.00018/+0.00019). PLR-i sigma 0.2: 0.83826/0.83793, weaker.
  **v6h** (v3 trees + 2 PLR-h nets, w0.25) → **public 0.83476 (new best)**; official 39/72 (tied 0.83476 with #38, passed 0.83475/0.83474).
  Launched plr-h2 (same as h, seeds 2–5: confirm not seed luck) and plr-j (k 32, d 24).
- 20 Nov 22:00 — PLR-h2 (h settings, seeds 2–5): 0.83865/0.83891/0.83867/0.83894; PLR-j (k 32, d 24): 0.83896/0.83885. Robust 8-net set (h ×6 + j ×2) single 0.83947; blend w0.25 0.84126/0.84018.
  **v7** (v3 trees + 8 wide PLR nets, w0.25) → **public 0.83477 (new best)**. Launched plr-j2 (j, seeds 2–5), plr-k (H 256, drop 0.35).
- **LightGBM max_bin was never tuned** (127 everywhere, incl. Optuna). Launched Kaggle CPU sharechat-lgb-bins: G lr0.03 ist with max_bin 255 / 511 / 1023, valid + full (baseline 0.83995).
- 20 Nov 22:40 — **shared-error analysis (user's idea)**: when one family misses a watch (watched row ranked in bottom half), all 4 miss it 55% of the time. All-family misses concentrate in long videos ≥120 s (2% of watches, 27% of misses, miss rate 65%) and high-activity users (20% / 46%, 10.9%).
  Calibration is fine per segment (pred/actual 0.99–1.13); cross-fitted 9-segment recalibration 0.84015 vs 0.84024 → not a bias problem.
  Launched **residual specialist** (resid.py, Kaggle CPU sharechat-resid): LightGBM with init_score = A·logit(training video watch rate, OOF) so trees learn only the user×video part (inside-video AUC 0.737 is our weakest spot). Arms A=1.0/0.5, valid + full.
- 20 Nov 23:10 — **hard-example weighting** (user's "heavy penalty" idea, tree version): train_lgb.py --hardw=DIR:LAM; difficulty from the out-of-fold network score (hybrid job, attached as kernel source): e = 1−rank for watched rows, rank for skipped rows; weight 1+LAM·e, renormalised so each class keeps its total weight. Kaggle CPU sharechat-hardw, LAM 1 and 3, valid + full (baseline 0.83995).
- PLR-k (H 256, drop 0.35, k 24 d 16): 0.83898/0.83900 (2-seed 0.83933); blend w0.25 0.84128/0.84020; v7 nets + k (10) 0.84127/0.84019 (+0.00001 over v7) → PLR family saturating at ~+0.00015.
- 20 Nov 23:40 — loose-end check: tree models never tested with the PLR networks. v7 + linear-tree LGB ×2 + LGB seeds 11/23/2026 + XGB seeds 4/5/6 = **v8: holdout 0.84131 / 0.84023 (+0.00005) → public 0.83470 (worse than v7 0.83477)**.
  → Extra tree models do not transfer to test; the network changes did (v5/v6h/v7 all transferred ~90–100%). v7 stays best.
- 21 Nov 07:40 — overnight results (all vs v7 0.84126/0.84018):
  · Residual specialist (video-rate offset): A=1.0 0.83112, A=0.5 0.83764 alone (base 0.83995); as extra 5–15% → worse. Closed.
  · Hard-example weighting: λ1 0.83952, λ3 0.83824 alone; λ1 as LGB group 0.84125/0.84017 → no gain. Closed.
  · max_bin 255/511/1023: 0.83975/0.83977/0.83996 (127: 0.83995) → bins were not limiting. As LGB group: 0.84126 = v7. Closed.
  · PLR-j2 (k32 d24 seeds 2–5): 0.83873/0.83917/0.83901/0.83917; 12 nets single 0.83954 but blend = v7. PLR family saturated.
- Forum read 20 Nov late: host on our thread 759936 and on 760657 (distillation): "The model which predicts on the test dataset must not use features derived from request_id/request grouping." Host 759267: if top solutions are ineligible, prizes go by private LB rank among the rest.
- 21 Nov — **full unused-column audit** (152 of 624 columns not in the main LightGBM):
  · 137 numeric = raw popularity counters 154–286 + epochs 57–59 + drifting rates 442/454/460/472; single-column AUC only 0.55–0.57 (replaced by per-age-hour versions in --stable).
  · 12 text columns, time-split past-watch-rate AUC: video id f18/f24 0.7015, creator f20 0.6718, f23 0.6162 (already tested as train stats 15 Nov: redundant with 409–431/480–515; video id used by the networks), f9 0.565, f7/f12 ≈0.51, session f5 0.507 (5% seen), dates f2/f4 & constant f15 0.50, request_id 0% seen later (and banned).
  → No unused column hides strong signal. Nothing missed in the data.
- 21 Nov 08:00 — **user approved training-time-only distillation** (host wording 759936/760657: the model predicting test must not use request features). teacher.py: LightGBM G + request size / rows-per-user / position, 4-fold OOF by session, for valid (<cut) and full rows → teach_*.parquet. Never scores test rows.
  Student: wide PLR network (k24 d16, sigma 0.3, 6 ep) with targets alpha·y + (1−alpha)·teacher (nn2_multi.py SOFT=…), inputs = row features only. Jobs ready: student-a (alpha 0.5), student-b (alpha 0.3), seeds 0,1 — push after sharechat-teacher completes. Smoke OK.
  Rule: adopt only with ≥ +0.0002 on both plain and test-mix vs v7; v7 stays the safe final regardless.
- 21 Nov: **PLR teacher–student distillation** (teacher.py: LGB G + request size/pos/user-rows, 4-fold OOF by feature5; student = wide PLR net, row features only, target alpha·y+(1−alpha)·teacher). Teacher OOF 0.8863 valid / 0.8875 full.
  Students (2 seeds each): alpha 0.5 0.84003 · alpha 0.3 0.84015 (v7's 8 PLR nets 0.83947 → best networks so far). Spearman vs v7 nets 0.9967.
  In the v7 blend: best +0.00004 plain / +0.00004 test-mix (nets weight 0.30–0.35, v7 nets + students) → below the +0.0002 bar. Launched student-c (alpha 0.1) and student-d (alpha 0.0), 4 seeds each.
  student-c alpha 0.1 (4 seeds) 0.84011 (spearman v7 nets 0.9933) → not better than alpha 0.3. Best combo v7 nets + all students, weight 0.3–0.4: 0.84130 / 0.84022 (+0.00004) → still below the bar. student-d (alpha 0) running.
  **v9** = v7 + students a/b/c as networks group 1 (nets weight 0.30), final/fixed_blend_v9: holdout 0.84130, public **0.83479** (v7 0.83477, +0.00002) → new best by a hair, rank 45/82.
  Network-weight probe on public (v9 files): 0.30 0.83479 · **0.40 0.83481 (best, final/fixed_blend_v9w40)** · 0.50 0.83478. Holdout 0.84130/0.84130/0.84125 → public peak = holdout plateau edge.
  student-d alpha 0.0 (pure teacher, 4 seeds) 0.83985 single, adds 0 to the blend → alpha 0.3 was the best; distillation line closed at v9w40.
- 21 Nov evening — two NEW ideas from an outside review (the rest of the review was already done or uses other rows):
  · **User-taste clusters** (ucl_build.py + te.py UCL): KMeans on each row's own user embeddings (PCA 32; k 32/128/512) and video embeddings (k 256), fitted on pre-cut training rows; OOF training-label rates for cluster × creator / category / video-cluster / video. Kaggle CPU sharechat-ucl (G lr0.03 ist + --te with cluster keys; baseline 0.83995).
  · **Pairwise-ranking XGBoost** (xgb_pair.py): rank:pairwise, random 1000-row query blocks (not sessions), global holdout AUC per 250 rounds. Kaggle GPU sharechat-pair.
  User embeddings checked as a persistent user key: 435k vectors for 421k sessions, 0.6% of test rows seen → per-session, no user id.
  · **ucl RESULT**: holdout 0.83993 (base 0.83995, best iter 413); cluster rates are used (te_r_u32_f20 #21, te_r_u128_v256 #23 by gain) but replace existing signal (spearman vs LGB component 0.9974); in v9w40 +0 as LGB group, extra 0.05–0.2 ≤ 0 → closed. Cluster stats into the nets not pursued (nets already see the raw embeddings).
  · Launched sharechat-plr-pair: wide PLR net, BCE + 0.5 × in-batch pairwise softplus(−(z⁺−z⁻)) over random batches (PAIR_LAM in nn2_multi.py), seeds 0,1.
  · **plr-pair RESULT**: seeds 0.83926 / 0.83905 (mean 0.83942, below plain PLR 0.8395), spearman vs nets 0.9964; in v9w40 +0.00001 → closed. sharechat-pair (XGB rank:pairwise) stuck QUEUED ~4 h (GPU quota?).
- 21 Nov night — reliability-shrunk rates checked BEFORE building: rate columns 409–431/481 have 85k–267k distinct values per 3 row groups, exact zeros 0.1–15%, small-denominator (≤20) share only 0.2–12% (481: 21%) → rates already come from large counts; no paired support counter is identifiable; and the whole rate family is near-redundant (−0.0004 when removed, 19 Nov views run) → not built.
- 21 Nov late — idea-list re-check (20 Nov research list + outside reviews). Never tried until now:
  · **Blend space**: rank vs logit vs probability averaging (± z-scored families) on v9w40 holdout: all 0.84129–0.84131 / 0.84021–0.84024 → rank kept.
  · **Missing-value counts**: 274/624 columns have nulls, 94% of rows ≥1 null, row null-count AUC 0.4545 alone. train_lgb.py --nullcnt adds nnull_all / nnull_u / nnull_p (row-local). Kaggle CPU sharechat-nullcnt (G lr0.03 ist; base 0.83995).
  · Still untried on purpose: TabR/ModernNCA row retrieval (user-cluster rates, its coarse version, gave 0; costly); video co-occurrence graph from training sessions (built from session/request grouping → against the host's request rule); team merge (user declined).

### 21 Nov: missing-value count features (`--nullcnt`), result
- LGB G lr0.03 + nnull_all / nnull_u / nnull_p: holdout 0.83978 (base 0.83995, -0.00017). Gain: nnull_all 109, nnull_p 80, nnull_u 0 (tiny).
- In v9w40: as an extra LGB group plain 0.84130 / test-mix 0.84022 (base 0.84130 / 0.84023); as a 5-10% extra: no gain. **Rejected.**
- sharechat-pair crashed (`Unknown key: group_ptr` in new XGBoost); fixed with `set_group(sizes)`, re-pushed as version 2.
- 21 Nov 23:50 — **idea audit across all sessions** (user asked what was proposed but never run). Never run: (1) **time-aware training rates** (proposed 20 Nov: our te.py rates for training rows came from user-grouped folds, so they included LATER days, unlike test rows) → te.py `TE_TIME=1` = past-only stats from strictly earlier hours (first-hour rows get n=0, leak check OK); Kaggle CPU sharechat-tetime, G lr0.03 ist --te, valid + full (base 0.83995). (2) LightGBM feature_contri column weighting. (3) test-similarity weighting (grey area, skipped).
- 22 Nov — **time-aware rates RESULT** (sharechat-tetime): LGB G lr0.03 ist + past-only te rates 0.84004 (base 0.83995, +0.00009; te_r_feature18 / te_r_feature20 top-20 by gain). In v9w40: as an LGB group 0.84132 / 0.84024 (base 0.84130 / 0.84023, +0.00002) → below the +0.0002 bar, not submitted. (The old user-fold te rates had been 0; the past-only version fixes that, but the information is already in ShareChat's own history columns.)
- sharechat-pair v2 failed: my re-pack had put xgb_pair.py in place of train_lgb.py. Fixed (both embedded files verified byte-identical), pushed v3.
- 22 Nov — **pairwise XGBoost RESULT** (sharechat-pair v3, rank:pairwise, random 1000-row groups): best 0.83336 at 250 rounds, then falls to 0.818 and slowly recovers to 0.8297 at 4000 (normal XGB ≈ 0.839). In v9w40: as an XGB group 0.84121 / 0.84012, extra 5% 0.84129 / 0.84021 → hurts. Closed. Official rank 51/92.
- 22 Nov — **column weighting** (train_lgb.py --fcontri=IMP:TOPN:HI:LO, LightGBM feature_contri; importance from the G run, imp_g.csv): Kaggle CPU sharechat-fca (top-30 columns x1.5) and sharechat-fcb (columns outside top-100 x0.5), G lr0.03 ist, valid + full (base 0.83995).
- 22 Nov — sharechat-student-e (GPU): wider PLR students k32/d24, alpha 0.1, seeds 4–7 (same teacher). Judge as extra networks group in v9w40.
- 22 Nov — **column weighting RESULT**: fca (top-30 x1.5) 0.83989 (base 0.83995, 0) · **fcb (columns outside top-100 by gain x0.5) 0.84027 (+0.00032, best single LightGBM ever)**. In v9w40: fcb as extra LGB group 0.84134/0.84026; fcb as 50% of the LGB component 0.84137/0.84029; 75–100% 0.84139/0.84031 (+0.00009/+0.00008) — one fcb model beats the whole 10-file LGB component. Below the +0.0002 bar so far; need seeds.
- student-e (wide students k32/d24, alpha 0.1, seeds 4–7): mean 0.84002 single; as networks group 0.84129/0.84021 → no gain.
- Follow-ups launched (Kaggle CPU): sharechat-fcseed (fcb seeds 7, 99), sharechat-fcvar (outside-top-100 x0.25; top-50; top-200), sharechat-fcslow (fcb on the slow lr0.01 neg1.0 arm, the main LGB group).
- 23 Nov — **fcb seeds**: s7 0.83996, s99 0.83982 (s42 0.84027 was a lucky seed); 3-seed rank-avg 0.84042. Variants (single seed 42): outside-top-100 x0.25 0.83988 · top-50 0.84004 · top-200 0.83982 → x0.5 / top-100 kept.
  Blend (v9w40 0.84130/0.84023): 3-seed fcb as 75% of LGB 0.84137/0.84029 (+0.00007/+0.00006). Like-for-like check: our 2 plain G seeds in the same 75% slot 0.84124/0.84016 vs 2 fcb seeds 0.84137/0.84029 → the weighting itself is worth ~+0.00013 over unweighted G models. Still under the +0.0002 bar vs v9w40; waiting for fcslow (weighting on the slow main LGB arm).
- 23 Nov — **fcslow RESULT**: slow lr0.01 neg1.0 + column down-weighting 0.84031 (same arm unweighted 0.84024, +0.00007). Blend: slow-fc as 50–75% of LGB 0.84133/0.84026; (slow-fc + 3 fcb seeds) as 75–100% of LGB 0.84136/0.84029 (+0.00006/+0.00006). Below the +0.0002 bar → v9w40 stays final. Column-weighting line closed (real but small: ~+0.0001 on single models, +0.00006 in the blend).
- 23 Nov — **Fixed Blend v10** (final/fixed_blend_v10 = v9w40 with LightGBM = column-down-weighted slow arm + 3 G seeds; holdout 0.84136): **public 0.83478** vs v9w40 0.83481 → tree gain did not transfer again (as v8). v9w40 stays final, v7 backup.
- 23 Nov — **weight averaging (EMA) for networks** — the one network-side idea never tried. nn2_multi.py EMA=<decay> (torch AveragedModel, per step; validation and predictions use the averaged weights; off by default). Kaggle GPU sharechat-ema1 (0.99) / -ema2 (0.995), student config st_c (alpha 0.1, k24/d16, seeds 0,1); same-seed baseline st_c s0 0.83979 / s1 0.83982.

### Weight averaging (EMA) on distilled PLR students (23 Nov)
- EMA 0.99 / 0.995, seeds 0-1, same setup as st_c otherwise.
- Singles: ema1 0.83981 / 0.83984 vs st_c 0.83979 / 0.83982 (+0.00002); ema2 slightly worse.
- Blend (v9w40 = 0.84130 / 0.84022): swapping in ema1 gives 0.84130 / 0.84022, adding it as an extra group gives 0.84129 / 0.84022. ema2 is worse in both.
- Verdict: no gain, not adopted. Finals stay v9w40 + v7.

### 23 Nov — out-of-the-box round
- Exact-repeat lookup: full rows hashed (all features except id, request_id, feature5, feature1). Test rows found in train 0.0%; holdout rows found in earlier train 0.0% → no repeats, closed.
- Second-pick hedge: Spearman vs v9w40 on test: v7 0.99981 · v9 / v9w50 0.99992 · v10 0.99988 · v8 0.99966 · v3 0.99895. All blends are almost the same ranking, so a "different" second pick cannot hedge in practice; v7 is already the least-correlated high scorer → finals stay v9w40 + v7.
- TabPFN v2 (fallback TabICL) as a new model type: tabpfn_run.py, top 100 raw numeric features, 10k random context rows per bag, up to 4 bags, time budget 8 h. Kaggle GPU sharechat-tabpfn. Rule: add as a networks group in v9w40, accept only at +0.0002 on both checks.
- Watch-time target: train and test differ only by watched / id; no watch-time or watch-fraction column exists → idea 5 not possible, closed.
- Per-day / per-hour offset (23 Nov): train watch rate by day 5.16% → 4.95% → 4.83%; test days 29–31 Oct, blend mean rank by test day 0.49 / 0.51 / 0.50. Holdout in-sample oracle (logit(blend) + time dummies, fitted on holdout labels = upper bound): date 0.84130 (+0.00000) · 6 h 0.84135 · 1 h 0.84150 (24 params, in-sample). Blend already tracks time-level rates → closed.
