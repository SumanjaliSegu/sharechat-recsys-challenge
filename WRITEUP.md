# ShareChat RecSys Challenge — Technical write-up

**Team Spirit:** Sumanjali Segu, Pranav Kumar

**Task.** Predict P(watched) for a user–video-post pair. Metric: one global ROC-AUC over all rows.
**Constraint.** Each row must be predicted from that row's features alone (serving-time rule).
**Primary submission (Fixed Blend v9w40).** Public AUC 0.83481; holdout AUC 0.84130 (test-mix weighted 0.84023).
**Backup (Fixed Blend v7).** Public AUC 0.83477; holdout AUC 0.84126 (test-mix 0.84018).
Earlier frozen blends: v1 0.83447 · v2b 0.83460 · v3 0.83462 · v5 0.83471 · v6h 0.83476 · v8 0.83470 · v9 0.83479.

---

## 1. Data understanding
- Train: 1,537,638 rows (2026-10-25 18:00 → 10-28 18:00 UTC). Test: 1,037,090 rows (the following 3¼ days).
- Positive rate ≈ 5%. 622 masked features: 24 strings, 14 list columns (7 user + 7 post 32-d embeddings), ~580 numeric.
- **93% of test users never appear in train** → user-id memorisation is useless; the model must generalise from
  profile/history columns and embeddings. 84% of test posts appear in train, but **16% are new and young**
  (median age 32 h vs 764 h).
- Column roles were inferred from behaviour (constant within user / post / request): user history (~270 cols),
  user×post affinity rates (~180), post popularity counters (~130), post content (video length = feature54).
- **Target structure:** watch rate is ~7% for videos under 60 s, 4.4% at 110 s, then falls off a cliff to ~1%
  at 120 s and ~0.06% beyond 230 s. For short videos, user–post embedding match moves watch rate from 2% to 16%.

## 2. Validation
- **Standard check:** train before 10-27 18:00, score the last 24 h (all rows). User overlap 5% / post overlap 91%,
  close to test (7% / 84%).
- **Far check:** train on day 1 only, score 1–2 days ahead (test is 0–3 days ahead).
- **Test-mix weighting:** holdout re-weighted to test's 16% new-post share.
- **Noise floor:** identical configs with different seeds differ by up to ±0.0004 → single-model decisions used
  2 seeds and a ≳ +0.001 bar; blend changes had to gain ≥ +0.0002 on **both** the plain and the test-mix holdout.

## 3. Features (all row-local)
1. ~450 numeric columns as given, 12 low-cardinality categoricals, Indian-local hour (UTC+5:30).
2. **Embedding matching (largest single gain, +0.0078 AUC):** dot product and cosine of each of the 7 user vectors
   (26–32) with each of the 7 post vectors (40–46) = 49 pairs ×2, plus vector norms.
3. **Drift control** (adversarial validation: a classifier separates train from test with AUC 0.981):
   - raw cumulative popularity counters (154–286) and raw epoch columns (57–59) replaced by
     log(counter) − log(post age in hours), counter log-ratios and ages;
   - creator / category id columns (7, 9, 12, 20, 23) and post id removed — they memorised training-period
     creators that test does not share (+0.0008, consistent across seeds and on the far check);
   - drifting user×post rates 442 / 454 / 460 / 472 removed.

## 4. Models
| Model | Holdout AUC | Notes |
|---|---|---|
| LightGBM slow (lr 0.01, all rows, seeds 42/7/99) | 0.8402 / 0.8401 / 0.8404 | leaves 63, min_leaf 800, feature_fraction 0.5, L2 30, cat_smooth 200, Indian-local hour |
| LightGBM DART (lr 0.05, 1,500 rounds) | 0.8398 | diversity group in the blend |
| LightGBM 127 leaves / min_leaf 400 (lr 0.01) | 0.8402 | diversity group in the blend |
| LightGBM "G" (lr 0.03, 5 seeds) | 0.8398 | negatives downsampled to 25% (AUC-invariant) |
| XGBoost (GPU, seeds 1/2/3) | 0.8400 / 0.8403 / 0.8400 | depth 8, lr 0.03, colsample 0.5, min_child_weight 50 |
| CatBoost ×3 (GPU) | 0.8387 / 0.8388 / seed 7 | depth 8 and 10 |
| Structured multi-tower network (our design, used up to v3) | 0.8347–0.8352 per seed | user / post / pair towers on quantile-ranked inputs, 49 cosine matches + 7 learned bilinear user–post matches, DCN-v2 cross layers; BCE loss, AdamW, OneCycle, AMP |
| **PLR towers network (final networks)** | 0.8390–0.8395 per seed (8 runs: k24/d16 seeds 0–5, k32/d24 seeds 0–1) | the towers network above with **periodic numeric embeddings (PLR)** in front: each numeric input x → [sin 2πcx, cos 2πcx] with k trainable frequencies c (init N(0, σ=0.3)), then a per-feature linear layer (d outputs) + ReLU. 6-epoch OneCycle, lr 1e-3, dropout 0.3, hidden 192. Biggest network gain of the competition (0.8370 → 0.8395) |
| **Distilled PLR students (teacher–student, §7)** | 0.8400 / 0.8402 / 0.8401 (α 0.5 / 0.3 / 0.1) | same PLR network and row-only inputs, trained on α·label + (1−α)·teacher out-of-fold probability; best single networks we built |

## 5. Ensemble
Rank average with **fixed** weights: **0.282 LightGBM + 0.212 XGBoost + 0.106 CatBoost + 0.40 networks**
(= the v3 tree weights 0.40 / 0.30 / 0.15, rescaled to make room for the networks).
Inside a family, files are rank-averaged within a group (e.g. seeds of one configuration) and groups are averaged
equally (LightGBM groups: slow lr 0.01 ×3 seeds, G lr 0.03 ×5 seeds, DART, 127-leaf; network groups: 8 PLR runs,
8 distilled PLR students).
Network weight was the one setting chosen with the public board: holdout is flat between 0.30 and 0.40
(0.84130 both) and drops beyond; public 0.30 → 0.83479, **0.40 → 0.83481**, 0.50 → 0.83478.
A 9-model stack with weights *learned* on the holdout scored higher on holdout (0.8410) but lower on the public
board (0.83436) — the learned weights fit holdout noise — so fixed weights were kept.

## 6. What did not help (each tested with ≥ 2 seeds or on several checks)
147 engineered ratio features · train-period post/creator/user watch-rate statistics · 2× negatives ·
raw embedding products · recency weighting · removing whole column groups (22-group sweep) ·
segment specialists (high-watch user group) · young-post specialists / up-weighting ·
monotone video-length constraint · learned stacking ·
Optuna search of 12 LightGBM parameters (5 h TPE; slow confirmation 0.84025 vs 0.84024) ·
embedding-neighbour video statistics (KNN over post embeddings, out-of-fold; +0.0002 alone, 0 in blend) ·
focal loss γ=1/2 and label smoothing for the networks · TabM (BatchEnsemble MLP, k=16; 0.8335 vs 0.8370) ·
recent-window (6/12/24 h) video watch rates (weaker than all-history rates, which already add nothing) ·
linear-leaf LightGBM (best single LightGBM, 0.8403, but +0.00004 in the blend) ·
feature-group Transformer (0.8314) · extra seeds (+0.00001; seed averaging saturated) ·
network-heavier blend weights (public 0.83461 vs 0.83462) ·
pseudo-labelling, validation only (holdout day used as unlabelled data; +0.0002 alone, which disappears at
equal boosting rounds, 0 in the blend — never used on test) ·
max_bin 255/511/1023 · residual specialist (trees on top of the video's training watch rate) ·
hard-example re-weighting · extra tree models in the PLR blend (v8: holdout +0.00005, public −0.00007) ·
raw embedding coordinates as tree inputs · CatBoost with embedding features ·
**user-taste clusters** (KMeans on each row's own user embeddings, k 32/128/512, × creator / category /
video-cluster watch rates from training labels: used by the trees but 0.83993 vs 0.83995, 0 in the blend) ·
**pairwise AUC loss** for the PLR network (BCE + 0.5 × in-batch watched-vs-skipped softplus; 0.8394, +0.00001
in the blend) · user embeddings as a persistent-user key (they change per session: 0.6% of test rows seen) ·
missing-value counts per row (0.83978 vs 0.83995) · logit / probability-space blending instead of ranks (±0.00001) ·
**time-aware (past-only) training rates** (each training row sees only earlier hours, as a test row does:
0.84004 vs 0.83995, +0.00002 in the blend) · pairwise-ranking XGBoost on random row groups (0.8334) ·
wider distilled students (k32/d24, 0.8400, 0 in the blend) ·
**column down-weighting** (LightGBM `feature_contri` 0.5 outside the top-100 columns by gain: real but small,
+0.00007 on the slow arm and +0.00013 over like-for-like unweighted models; blend v10 holdout 0.84136 vs 0.84130,
but **public 0.83478 vs 0.83481** — like v8, tree-side holdout gains below +0.0002 did not transfer) ·
**TabPFN v2** (pretrained in-context tabular model, 90 raw features, 10k-row context bags, `tabpfn_run.py`:
0.8168 alone, lower blend AUC at every weight) · EMA of network weights (0).
**Drop from validation to public is uniform:** each family submitted alone loses 0.0059–0.0069
(LightGBM 0.84069→0.83435, XGBoost 0.84060→0.83411, CatBoost 0.83964→0.83278, networks 0.83702→0.83116),
and validation AUC shows no decay across the 24 h holdout → no family is under-weighted for the later test days.
**Why:** the dataset already ships ShareChat's own history rates (409–431, user×post affinity) — extra
train-period statistics are largely redundant with them. Error analysis: no model family beats LightGBM in any
segment (new videos, length bands, user groups), so gating / mixture-of-experts has nothing to switch to.

## 7. Rule compliance and the request-structure investigation
Every feature is computed from the row itself; category vocabularies, quantiles and normalisation constants
are fitted on training data only (model parameters). No test-set statistics, no other-row information.

To understand the ~0.888 public scores we ran **diagnostics that were never submitted**:
- adding request size, position-in-request and rows-per-user (all need other rows) lifts our model from
  0.840 to **0.8875**;
- a model predicting request size / position **from row features alone** reaches R² 0.32 / 0.15; the
  row-predicted size carries AUC 0.638 versus 0.759 for the true size.
**Conclusion:** the part of request structure visible in a row is already used by our model; the remaining
signal exists only across rows, which the competition prohibits. Every model that scores test rows is strictly
row-independent, matching ShareChat's serving-time requirement.

**Training-time-only distillation (used in the final blend).** The host clarified on the forum (threads 759936,
760657) that *"the model which predicts on the test dataset must not use features derived from request_id/request
grouping"*. We use request features only as a training signal:
- **Teacher** (`teacher.py`): LightGBM on the G features + request size, position in request and rows per user,
  trained on training rows only, 4-fold out-of-fold grouped by session (a row's target never comes from a model
  that saw that row). OOF AUC 0.886 (validation stage) / 0.888 (all training rows). **It never scores a test row.**
- **Students** (`nn2_multi.py` with `SOFT`): the PLR network on row-only inputs, trained on
  α·label + (1−α)·teacher probability (α 0.5 / 0.3 / 0.1). At test time they receive exactly the same inputs as
  every other model. Gain: +0.0007 single-network AUC; +0.00002 public in the blend (v9 vs v7).

## 8. Serving cost
Measured with `serve_time.py` (one CPU thread): the fast LightGBM (421 trees, 3.1 MB) takes **0.14 ms per row**;
the slow LightGBM (1,625 trees, 12 MB) 0.63 ms per row. The full blend gains about +0.0014 AUC over one model, so
a production version would be one LightGBM plus one distilled student network, both row-only.

## 9. Limitations and next steps
- New / young posts remain hardest (AUC 0.825 vs 0.842 for posts seen in training).
- The validation→public gap (~0.006) is partly explained by test's larger share of new posts (≈0.001).
- With serving-time request context (e.g. a feed-slot feature logged at impression time), position-aware
  modelling would be the largest available gain.

## 10. Reproduce
See `README.md`. `final/fixed_blend_v9w40/MANIFEST.json` records every component (files, groups, weights,
checksums); `python ensemble.py final/fixed_blend_v9w40 --holdout` rebuilds the exact submitted file and reports
holdout AUC 0.84130. `python run_all.py` retrains every component (stages lgb, xgb, catboost, nets, teacher,
students) and rebuilds the blend; `python run_all.py --only blend` on the original outputs reproduces the
submitted v9w40 file exactly (max difference 0.0 over 1,037,090 rows, 21 Nov).

**End-to-end reproduction from code (`run_all.py`, fresh Kaggle notebooks, 16–17 Nov):**
- LightGBM (`--only lgb`, CPU): all 6 test predictions bit-identical to the frozen files (max diff 0.0).
- XGBoost / CatBoost / networks (`--only xgb|catboost|nn`, GPU): not bit-identical (GPU nondeterminism;
  rank correlation 0.997–1.000, DCN 0.976), single-model AUC within ±0.0002 (DCN −0.0006);
  Blend v3 rebuilt with the reproduced files scores **0.84113 vs 0.84112 frozen**.

**Compute.** Training on Kaggle Notebooks: CPU notebooks for LightGBM; GPU notebooks (CUDA; card as assigned by Kaggle) for XGBoost,
CatBoost and networks. Blending runs on a laptop (8 GB RAM). Python 3.12; versions in `requirements.txt`.
Seeds: LightGBM 42/7/99 (slow), 42/7/99/123/2026 (G); XGBoost 1/2/3; CatBoost 42/7; PLR networks 0–5 (k24) and
0–1 (k32); students 0–1 (α 0.5, 0.3), 0–3 (α 0.1); teacher LightGBM seed 42.
