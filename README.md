# ShareChat RecSys Challenge — solution

**Team Spirit:** Sumanjali Segu, Pranav Kumar

Predict the probability that a user watches a video post. Metric: global ROC-AUC.
**Every model that scores test rows uses only the features of that row** (competition serving-time rule).
Request-level columns appear in exactly one place: the *teacher* in `teacher.py`, which only makes soft training
targets for row-only student networks and never scores a test row (see WRITEUP.md §7).

Final submission: **Fixed Blend v9w40**: public AUC 0.83481, holdout 0.84130. Backup: Fixed Blend v7, 0.83477.

## Rule compliance at a glance (for reviewers)
- [x] Every model that scores a test row sees **only that row's features**: no request_id grouping, no counts or
      positions over other rows, no test-set statistics.
- [x] Request-level columns are used **only** by `teacher.py`, out-of-fold on training rows, to make soft training
      targets. The teacher never scores a test row (WRITEUP.md §7).
- [x] Vocabularies, quantiles and normalisation constants are fitted on training data only.
- [x] Both selected final submissions rebuild exactly from frozen predictions:
      `python ensemble.py final/fixed_blend_v9w40` and `python ensemble.py final/fixed_blend_v7`.
- [x] No external data; open-source libraries only (`requirements.txt`).

## Serving cost
One LightGBM from the blend (421 trees, 3.1 MB) scores **0.14 ms per row on one CPU core** (`serve_time.py`).
The full 32-model blend adds only about +0.0014 AUC over a single model, so the single model plus one distilled
student network is what we would ship.

## Approach (summary)
1. **Validation that mirrors test.** Train on rows before 2026-10-27 18:00 and score the last 24 h of train.
   We also report a test-mix AUC, re-weighted to the test set's 16% share of new videos.
2. **Features (row-local).** ~450 numeric columns, 12 low-cardinality categoricals, Indian-time hour,
   and 49 user×post **embedding match scores** (dot and cosine of the 7 user vectors 26–32 with the
   7 post vectors 40–46) plus their norms.
3. **Drift control** (found with adversarial validation, train-vs-test AUC 0.98):
   - raw cumulative counters (154–286) and epochs (57–59) are replaced by per-post-age-hour versions, ratios and ages;
   - creator/category ids and the post id are not used;
   - drifting user×post rates 442/454/460/472 are dropped.
4. **Models.**
   - LightGBM: slow, G, DART and 127-leaf variants, several seeds.
   - XGBoost and CatBoost, on GPU.
   - **PLR towers networks**: our user/post/pair towers network with periodic numeric embeddings.
   - **Distilled PLR students**, trained on label + teacher out-of-fold probability.
5. **Blend.** Rank average with **fixed** weights: 0.282 LightGBM, 0.212 XGBoost, 0.106 CatBoost, 0.40 networks.

## Reproduce
```
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
# put train.parquet, test.parquet, sample_submission.csv in data/

python ensemble.py final/fixed_blend_v9w40 --holdout   # rebuild the submitted file from the frozen predictions
python run_all.py                                       # retrain everything (GPU; ~20 h, split by --only <stage>)
python run_all.py --only blend                          # assemble final/reproduced from the retrained outputs
```
Stages: `lgb` (CPU), `xgb`, `catboost`, `nets` (GPU), `teacher` (CPU), `students` (GPU, after `teacher`), `blend`.
Training ran on Kaggle Notebooks; blending runs on a laptop (8 GB RAM). Python 3.12, versions in `requirements.txt`.

See `WRITEUP.md` for the technical write-up and `EXPERIMENTS.md` for every experiment and its result.

License: MIT (required for winners by the competition rules).
