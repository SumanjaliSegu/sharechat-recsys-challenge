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
