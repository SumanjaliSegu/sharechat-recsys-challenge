# ShareChat RecSys Challenge — solution

Predict the probability that a user watches a video post. Metric: global ROC-AUC.
Every prediction uses **only the features of its own row** (competition rule); no request-level
or other-row information is used anywhere in training or inference.

## Approach (summary)
1. **Validation that mirrors test.** Train on rows before 2026-10-27 18:00, score the last 24 h of
   train ("standard check"); a stricter "far check" trains on the first day only and scores 1–2 days
   ahead, like test. All decisions use 2 seeds and the full holdout (single-run noise ≈ ±0.0004).
2. **Features (row-local).** ~450 numeric columns, 12 low-cardinality categoricals, Indian-time hour,
   and 49 user×post **embedding match scores** (dot and cosine of the 7 user vectors 26–32 with the
   7 post vectors 40–46) plus their norms.
3. **Drift control** (found with adversarial validation, train-vs-test AUC 0.98):
   raw cumulative popularity counters (154–286) and raw epoch columns (57–59) are replaced by
   per-hour-of-post-age versions, counter ratios and ages; creator/category id columns
   (7, 9, 12, 20, 23) and post id are not used; drifting user×post rates 442/454/460/472 dropped.
4. **Models.** LightGBM (regularised, feature_fraction 0.5, 5 seeds), CatBoost (GPU),
   XGBoost (GPU), a DCN-v2 network, and our own structured multi-tower network
   (user / post / pair towers + learned bilinear user–post matching).
5. **Blend.** Rank-average with non-negative weights learned on the holdout (cross-validated by user halves).

## Reproduce
```
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
# put train.parquet, test.parquet, sample_submission.csv in data/
python train_lgb.py valid  --emb --nopost --reg --lr=0.03 --fullho --stable --nocatid --dropE --ff=0.5 --ist
python train_lgb.py full   --emb --nopost --reg --lr=0.03 --fullho --stable --nocatid --dropE --ff=0.5 --ist --seed=42
# networks / CatBoost / XGBoost: nn_df.py, nn2.py, catboost_run.py, xgb_cb_run.py (GPU)
python stack.py
```
Compute: LightGBM runs on 8 GB RAM (negatives downsampled 25%, AUC-invariant); networks,
CatBoost and XGBoost were run on a Kaggle GPU (P100/T4, 30 GB RAM).

See `EXPERIMENTS.md` for every experiment, its validation score and public-leaderboard result.

_Final single-command pipeline and requirements.txt: in progress._

License: MIT (required for winners by the competition rules).
