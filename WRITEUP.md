# ShareChat RecSys Challenge — Technical write-up

**Task.** Predict P(watched) for a user–video-post pair. Metric: one global ROC-AUC over all rows.
**Constraint.** Each row must be predicted from that row's features alone (serving-time rule).
**Primary submission (Fixed Blend v3).** Public AUC 0.83462; holdout AUC 0.84112 (test-mix weighted 0.84002).
**Backup (Fixed Blend v2b).** Public AUC 0.83460; holdout AUC 0.84111.
Every submitted blend ranked in the same order on validation and on the public board.

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
- **Noise floor:** identical configs with different seeds differ by up to ±0.0004 → every decision used 2 seeds
  and a ≳ +0.001 bar; single-seed differences below that were treated as noise.

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
| **Structured multi-tower network (our design)** | 0.8347–0.8352 per seed (×6 seeds in the blend) | user / post / pair towers on quantile-ranked inputs, 49 cosine matches + 7 learned bilinear user–post matches, DCN-v2 cross layers; BCE loss, AdamW, OneCycle, AMP |
| DCN-v2 network | 0.8337 | 1,081 dense inputs + categorical embeddings |

## 5. Ensemble
Rank average with **fixed** weights: 0.40 LightGBM + 0.30 XGBoost + 0.15 CatBoost + 0.15 networks.
Inside a family, files are rank-averaged within a group (e.g. seeds of one configuration) and groups are averaged
equally (LightGBM groups: slow lr 0.01 ×3 seeds, G lr 0.03 ×5 seeds, DART, 127-leaf).
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
equal boosting rounds, 0 in the blend — never used on test).
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
- teacher–student check (validation only): a teacher that sees request features scores 0.886 out-of-fold, but
  a row-only student trained on its soft labels gains only +0.0004 alone and 0 in the blend.
**Conclusion:** the part of request structure visible in a row is already used by our model; the remaining
signal exists only across rows, which the competition prohibits. We therefore kept a strictly row-independent
solution, matching ShareChat's serving-time requirement.

## 8. Limitations and next steps
- New / young posts remain hardest (AUC 0.825 vs 0.842 for posts seen in training).
- The validation→public gap (~0.006) is partly explained by test's larger share of new posts (≈0.001).
- With serving-time request context (e.g. a feed-slot feature logged at impression time), position-aware
  modelling would be the largest available gain.

## 9. Reproduce
See `README.md`. `final/fixed_blend_v3/MANIFEST.json` records every component (script, flags, Kaggle notebook,
checksums); `python ensemble.py final/fixed_blend_v3 --holdout` rebuilds the exact submitted file and reports
holdout AUC 0.84112.

**End-to-end reproduction from code (`run_all.py`, fresh Kaggle notebooks, 16–17 Nov):**
- LightGBM (`--only lgb`, CPU): all 6 test predictions bit-identical to the frozen files (max diff 0.0).
- XGBoost / CatBoost / networks (`--only xgb|catboost|nn`, GPU): not bit-identical (GPU nondeterminism;
  rank correlation 0.997–1.000, DCN 0.976), single-model AUC within ±0.0002 (DCN −0.0006);
  Blend v3 rebuilt with the reproduced files scores **0.84113 vs 0.84112 frozen**.

**Compute.** Training on Kaggle Notebooks: CPU notebooks for LightGBM; GPU notebooks (CUDA; card as assigned by Kaggle) for XGBoost,
CatBoost and networks. Blending runs on a laptop (8 GB RAM). Python 3.12; versions in `requirements.txt`.
Seeds: LightGBM 42/7/99 (slow), 42/7/99/123/2026 (G); XGBoost 1/2/3; CatBoost 42/7; networks 0–5.
