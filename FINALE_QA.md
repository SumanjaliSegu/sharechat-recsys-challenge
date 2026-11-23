# Finale Q&A prep — ShareChat RecSys Challenge

Short answers, our real numbers. Say the first sentence, add detail only if asked.

## About the approach

**1. Walk me through your solution in one minute.**
Four model families (LightGBM, XGBoost, CatBoost, neural networks) on row-only features, rank-averaged with fixed weights. Two ideas moved the score: PLR numeric embeddings for the networks, and training-only distillation from a teacher that sees request context. Public 0.83481, every submission rebuilds bit-for-bit.

**2. Why are you ~0.05 behind the top of the leaderboard?**
That gap is request-level context (request size, position in the feed). We measured it: our teacher with request columns reaches 0.886 out-of-fold. The host said the model predicting test must not use request-derived features, so we used that signal only during training, through distillation.

**3. Wasn't that a big sacrifice?**
It was a choice to build what would work at serving time. In a live feed you score each item before the request is complete, so a model that needs request grouping can't run as-is. I'd rather show a model that could ship.

**4. How did you validate?**
Holdout = last training day (from 27 Oct 18:00), because test is the days right after training. Test has more new videos (16% vs 9% in holdout), so I also re-weighted holdout rows to match that mix. A change had to gain +0.0002 on both checks to be accepted.

**5. Why such a strict bar?**
Twice, tree-side gains smaller than that showed up on holdout and disappeared on public. Seed noise is about that size. The bar kept us from chasing noise.

**6. Why fixed blend weights instead of learned stacking?**
A learned stack scored higher on holdout but lower on public. With one holdout day it was fitting noise. Fixed weights are more robust for the private leaderboard.

**7. What is PLR and why did it help?**
Periodic-linear-ReLU embeddings (Gorishniy et al., NeurIPS 2022): each number is turned into a small vector of sine/cosine features before the network. It lets networks model sharp, non-linear effects that trees get for free — like the watch-rate cliff at 120 s video length.

**8. Explain the distillation.**
A LightGBM teacher sees request size, position and rows-per-session; trained out-of-fold on training rows only, it never scores a test row. Student networks see only row features and learn from a mix of label and teacher (best mix: 0.3·label + 0.7·teacher). Students became our best single networks (0.8402 holdout).

## About the data

**9. What was the most important feature?**
Video length (feature54). Watch rate is ~7% under 60 s, ~1% at 120 s, almost 0 beyond 230 s. Within one length band, AUC drops to 0.69–0.80 — much of the global AUC is "short vs long".

**10. What surprised you?**
There is no persistent user id: feature5 is a session (3.7 rows each, 93% of test sessions unseen). So personalisation has to come from ShareChat's own history columns, not from user-id tricks.

**11. How did you handle new videos?**
16% of test rows are videos never seen in training. We checked: their history columns are complete, just smaller counts, so no special masking was needed. They score lower (0.8255 vs 0.8425 for seen videos) — that's the main place to improve.

## About production

**12. How would you deploy this?**
Not the 32-model blend — it buys only +0.0014 AUC. I'd ship one LightGBM (0.14 ms per row on one CPU core, 3 MB) and add one distilled student network if the latency budget allows.

**13. What would you do with 3 more months at ShareChat?**
(a) Log feed position at impression time — it is legal at serving time and worth up to ~0.05 AUC. (b) A cold-start model for new videos using content embeddings. (c) Online A/B test of the single-model version against current production.

**14. How would you monitor it after launch?**
Daily AUC by segment (new vs seen videos, length bands), feature-drift checks on the history columns, and calibration (predicted vs actual watch rate) per segment.

## About you

**15. What didn't work, and what did you learn?**
40+ experiments: Optuna (132 trials), focal loss, TabM, a Transformer, KNN neighbours, pseudo-labelling, recency weighting (hurt public by −0.00027), segment specialists, learned stacking. Lesson: once strong history features exist, new kinds of models help more than more tuning — and a strict validation bar saves time.

**16. Why should we hire you?**
I found the strongest signal in the data, measured it, and still respected the serving-time rule. Everything is logged and reproducible with one command. And I think about what would actually ship, not just the leaderboard.

## Tips
- If you don't know: "I didn't test that — here's how I would." Never guess a number.
- Keep each answer under 30 seconds unless they ask for more.
- Bring it back to product: "In a live feed, that would mean…"
