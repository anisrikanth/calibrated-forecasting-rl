# Test-set plan, fixed before the test split is scored

Written 28 October 2026 and amended once on 29 October (see "Changes made before scoring"), both before any model has seen or been scored on the test split. It is committed to the repo before the test notebook (`notebooks/phase4_test_eval.ipynb`) runs; the notebook refuses to run without it and prints the repo commit, so the commit history shows the plan came first. Nothing below changes after the test results are in.

So far the only things looked at in the test split are its size, dates, series and yes-rate (below). No model, rule or baseline has been scored on it.

## What is scored

**Questions.** The test split: 495 econ questions, 8 series, 36 releases, settling 5 June to 14 October 2026. As in validation, every method is scored on the 479 that have a market price. Yes-rate 0.396.

**Models.** Each run's saved LoRA adapter, unchanged, plus the untrained model:

| Group | Models | Prompt | GRPO std scaling |
| --- | --- | --- | --- |
| Main setup | Runs 4, 5, 6 (training seeds 3407, 1, 2) | Key numbers | Off |
| Std scaling on | Runs 3, 9, 10 (training seeds 3407, 1, 2) | Key numbers | On |
| Untrained | Untrained, key numbers; untrained, plain | Both | — |
| Controls | Run 1 (std on), run 7 (std off) | Plain | On / off |
| | Run 8 (shuffled outcomes) | Key numbers | Off |
| Optional | Run 2 (2 passes, learning rate 2e-5) | Plain | On |

**Baselines, on the same questions.** Market price, base rate (each series' yes-rate in train), counting rule, latest-result rule, logistic regression on the key numbers (fitted on train only, as in validation, so it learns from the same questions as the models), always 0.5.

**Fixed procedure.** Each model gets the prompt it was trained with. 4 answers per question at temperature 0.7, top-p 0.95, at most 320 tokens, sampling seed 3407. An unreadable answer counts as 0.5. The forecast is the average of the 4 answers. Metrics: Brier score, log loss, calibration error (10 equal-width bins) and AUC. Intervals are 95%, from 4,000 bootstrap draws that resample whole releases. A group comparison averages the group's models within every draw, so its interval covers question sampling, not seed variation; the spread across seeds is reported separately (lowest and highest).

**Loading check, before any test question.** Every model re-answers 80 validation questions with the same prompt and seed, and its answers are compared with the ones that run gave during training (`eval/answers/`). A correctly loaded model reproduces most answers word for word; a different seed of the same setup reproduces almost none. A model that fails is flagged in the results. If no adapter passes, the notebook stops before the test set.

## Primary comparisons

**P1. The main setup improves on the untrained model with the same prompt.** Runs 4 to 6 (averaged) vs the untrained model, key numbers.
- Validation: Brier −0.031 (−0.049 to −0.013), AUC +0.070 (+0.029 to +0.115).
- Prediction: lower Brier and higher AUC.

**P2. The main setup beats the std-scaling-on runs.** Runs 4 to 6 vs runs 3, 9 and 10 (each group averaged).
- Validation, all three seeds of each: log loss −0.468 (−0.773 to −0.198), calibration error −0.148 (−0.199 to −0.019), Brier −0.048 (−0.099 to −0.003), AUC +0.045 (+0.018 to +0.077).
- Prediction: lower log loss and lower calibration error.

**P3. The std-scaling-on runs collapse onto the latest-result rule again.** Measured on single answers to the questions with a numeric threshold (Fed decisions have no rule).
- Validation: 99.4%, 98.8% and 99.1% of answers (runs 3, 9, 10) are exactly 0 or 1, and 99.9%, 98.7% and 94.6% are exactly the rule's 0 or 1. For comparison, the std-off runs 4 to 6 have under 1% of answers at exactly 0 or 1.
- Prediction: for each of runs 3, 9 and 10, at least 90% of answers exactly 0 or 1, and at least 90% exactly the rule's 0 or 1.

**How each is read.**
- **P1 holds** if the Brier interval lies entirely below 0. AUC is reported with it and counts as confirmed if its interval lies entirely above 0.
- **P2 holds** if the intervals for log loss and calibration error both lie entirely below 0.
- **P3 holds** if both thresholds are met for each of the three runs.
- The right direction with an interval that includes 0 is reported as **consistent but not confirmed**. The wrong direction is reported as **not replicated**.
- Every result is reported in full, whatever it shows.

## Secondary comparisons (reported, not used to judge the main claim)

- **S1.** Main setup vs logistic regression. Validation: a tie, Brier +0.008 (−0.035 to +0.041).
- **S2.** Untrained model, key numbers vs plain prompt (the elicitation effect). Validation: Brier −0.071, AUC +0.237.
- **S3.** Std-scaling-on runs vs the untrained model, key numbers. Validation: Brier +0.017 (−0.038 to +0.080); log loss +0.396 (+0.101 to +0.730) and calibration error +0.156 (+0.038 to +0.204), both clearly worse.
- **S4 and S5.** Run 7 and run 1 (plain prompt) vs the untrained model, plain prompt. Validation: no improvement in Brier or AUC.
- **S6 and S7.** Run 8 (shuffled outcomes) vs the untrained model, key numbers, and the main setup vs run 8. Validation: run 8 worse than untrained (Brier +0.035), main setup better than run 8.
- Each run on its own vs the untrained model, Brier by series, and threshold-ladder consistency.

## Known differences from validation

- **The test yes-rate is lower:** 0.396, against 0.491 in training and 0.461 in validation. Everything that learned the training rate (the base rate, the logistic regression, every model) will tend to forecast too high. Calibration error and Brier will likely be worse for all methods than in validation, so the comparisons between methods matter more than the levels.
- **Weather is not re-scored.** The weather questions were held out throughout, and every run already reports them.

## Changes made before scoring

- **29 October 2026, after run 10 finished on validation and before any test scoring.** P2 and S3 now use all three std-scaling-on seeds; the first version had only runs 3 and 9, because run 10 was still training. P3's second test changed. The first version asked that each run's 4-answer average fall on the rule's side of 0.5 on at least 95% of all questions, a threshold set from runs 3 and 9 (99.5% and 99.3%). Run 10 meets it on only 92.4% of validation questions: its single answers are still 0 or 1, but on a few questions its 4 answers disagree, so the average lands between 0 and 1. A threshold that validation already contradicts would not test anything. So P3 now counts single answers that are exactly the rule's 0 or 1, on the questions that have a rule. On validation, this measure separates the collapsed runs (94.6% to 99.9%) from the std-off runs (under 1%).

## What will not change after scoring

- No retraining, and no prompt, sampling or metric changes.
- No models or questions dropped.
- Anything computed beyond this plan is labeled exploratory in the write-up.
