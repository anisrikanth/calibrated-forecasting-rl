# Test-set plan, fixed before the test split is scored

Written 28 October 2026 and amended three times on 29 October (see "Changes made before scoring"), all before any model has seen or been scored on the test split. It is committed to the repo before the test notebook (`notebooks/phase4_test_eval.ipynb`) runs; the notebook refuses to run without it and prints the repo commit, so the commit history shows the plan came first. Nothing below changes after the test results are in.

So far the only things looked at in the test split are its size, dates, series, announcement dates, question timing and yes-rate (below). No model, rule or baseline has been scored on it.

## What is scored

**Questions.** The test split has 495 eligible econ questions: 8 series, 36 releases (Kalshi events, one per series per release, each a ladder of thresholds) and 13 announcements (CPI's four series come out together, as do payrolls and unemployment, and the Fed rate and decision), settling 5 June to 14 October 2026. As in validation, every method is scored on the questions that have a market price, so all methods share one population: 479 questions, still 36 releases and 13 announcements. The 16 without a price are 12 CPI year-over-year, 3 core CPI and 1 core CPI year-over-year. Yes-rate: 0.396 of the eligible questions, 0.409 of the scored ones.

**Timing.** Every econ question in every split is asked exactly 24 hours before its market closes; none needed shifting to fit a market's opening time. Markets close at the scheduled release: 8:25 to 8:30 Eastern for CPI and the jobs report, a few minutes before the 2 pm Fed statement, and in a handful of cases shortly after (9:00 or 10:00 Eastern), still the same morning. So each question comes about a day before its announcement. The only markets that settled long after closing are training questions whose releases the 2025 government shutdown delayed; those markets closed before the delayed release. The context only includes results whose markets settled before the question time (audited, zero violations).

**Models.** Each run's saved LoRA adapter, unchanged, plus the untrained model:

| Group | Models | Prompt | GRPO std scaling |
| --- | --- | --- | --- |
| Main setup | Runs 4, 5, 6 (training seeds 3407, 1, 2) | Key numbers | Off |
| Std scaling on | Runs 3, 9, 10 (training seeds 3407, 1, 2) | Key numbers | On |
| Untrained | Untrained, key numbers; untrained, plain | Both | — |
| Controls | Run 1 (std on), run 7 (std off) | Plain | On / off |
| | Run 8 (shuffled outcomes) | Key numbers | Off |
| Optional | Run 2 (2 passes, learning rate 2e-5) | Plain | On |

**Baselines, on the same questions.** Market price, base rate (each series' yes-rate in train), counting rule, latest-result rule, logistic regression on the key numbers (fitted on train only, as in validation, so it learns from the same questions as the models), always 0.5. Also two gradient-boosted-tree models on the key numbers, one with the series as an extra feature, fitted on train with settings chosen by cross-validation on train (`eval/analysis/gbm_baseline.py`). They involve no language model, so they are scored by running that script on the test split rather than in the notebook.

**Fixed procedure.**
- Each model gets the prompt it was trained with: 4 answers per question at temperature 0.7, top-p 0.95, at most 320 tokens, sampling seed 3407.
- The forecast is the average of the question's readable answers, and 0.5 if none is readable. On validation, every trained run's answers were all readable; the untrained models' were 97.7% to 98.4% readable. We also report the average with each unreadable answer counted as 0.5; on validation the two differ by at most 0.002 in Brier.
- Metrics: Brier score, log loss, calibration error (10 equal-width bins) and AUC, pooled over the scored questions, so an announcement with more thresholds carries more weight. We also report the mean Brier score of the single answers, and Brier with each announcement weighted equally, as descriptive checks.
- Intervals are 95%, from 4,000 paired bootstrap draws that resample whole announcements. Resampling single releases, as in the first version of this plan, is reported as a sensitivity check.
- A group comparison averages the group's models within every draw, so its interval covers question sampling for these trained models, not variation across training seeds. The spread across seeds is reported separately (lowest and highest).
- With 13 announcements, the intervals are rough; that is the effective sample size.

**Loading check, before any test question.** Every model re-answers 80 validation questions with the same prompt and seed, and its answers are compared with the ones that run gave during training (`eval/answers/`). A correctly loaded model reproduces most answers word for word; a different seed of the same setup reproduces almost none. If no adapter passes, the notebook stops before the test set. An adapter that fails is still scored and reported, but is left out of the group comparisons (P1, P2, S1, S3, S7, S8). If every model in a group fails, that comparison is reported as not evaluable. A failed check on an untrained model is reported, but does not exclude it.

## Primary comparisons

**P1. The main setup improves on the untrained model with the same prompt.** Runs 4 to 6 (averaged) vs the untrained model, key numbers.
- Validation (resampling announcements): Brier −0.031 (−0.045 to −0.016), AUC +0.070 (+0.036 to +0.119).
- Prediction: lower Brier and higher AUC, read as two separate conclusions.

**P2. The main setup beats the std-scaling-on runs.** Runs 4 to 6 vs runs 3, 9 and 10 (each group averaged).
- Validation, all three seeds of each, resampling announcements: log loss −0.468 (−0.741 to −0.236), calibration error −0.148 (−0.194 to −0.027), Brier −0.048 (−0.094 to −0.007), AUC +0.045 (+0.021 to +0.076).
- Prediction: lower log loss and lower calibration error.

**P3. The std-scaling-on runs collapse onto the latest-result rule again.** Measured on single answers to the questions with a numeric threshold (Fed decisions have no rule).
- Validation: 99.4%, 98.8% and 99.1% of answers (runs 3, 9, 10) are exactly 0 or 1, and 99.9%, 98.7% and 94.6% are exactly the rule's 0 or 1. For comparison, the std-off runs 4 to 6 have under 1% of answers at exactly 0 or 1.
- Prediction: for each of runs 3, 9 and 10, at least 90% of answers exactly 0 or 1, and at least 90% exactly the rule's 0 or 1.

**How each is read.**
- **P1** has two separate conclusions. The Brier improvement holds if its interval lies entirely below 0; the AUC improvement holds if its interval lies entirely above 0. A Brier-only success does not confirm better discrimination.
- **P2 holds** if the intervals for log loss and calibration error both lie entirely below 0 (both are required).
- **P3 holds** if both thresholds are met for each of the three runs.
- The right direction with an interval that includes 0 is reported as **consistent but not confirmed**. The wrong direction is reported as **not replicated**.
- Every result is reported in full, whatever it shows. The intervals for secondary comparisons are descriptive: no correction for multiple comparisons is applied, and no secondary result is promoted if a primary one fails.

## Secondary comparisons (reported, not used to judge the main claim)

- **S1.** Main setup vs logistic regression. Validation: the difference is not resolved, Brier +0.008 (−0.031 to +0.044), AUC −0.028 (−0.057 to +0.011).
- **S2.** Untrained model, key numbers vs plain prompt (the elicitation effect). Validation: Brier −0.071, AUC +0.237.
- **S3.** Std-scaling-on runs vs the untrained model, key numbers. Validation: Brier +0.017 (−0.032 to +0.074); log loss +0.396 (+0.142 to +0.690) and calibration error +0.156 (+0.045 to +0.199), both clearly worse.
- **S4 and S5.** Run 7 and run 1 (plain prompt) vs the untrained model, plain prompt. Validation: no improvement in Brier or AUC.
- **S6 and S7.** Run 8 (shuffled outcomes) vs the untrained model, key numbers, and the main setup vs run 8. Validation: run 8 worse than untrained (Brier +0.035, +0.011 to +0.054), main setup better than run 8.
- **S8.** Main setup vs gradient-boosted trees on the key numbers, both versions. The trees' settings are chosen by cross-validation on the training questions with each announcement kept in one fold: depth 3, learning rate 0.03 and 100 trees for both versions, recorded here before scoring. Validation: AUC −0.042 (−0.069 to −0.018) against trees with the series, −0.033 (−0.053 to −0.013) against trees without; Brier not resolved (`extra_analyses.md`).
- Each run on its own vs the untrained model, Brier by series, and threshold-ladder consistency.

## Known differences from validation

- **The test yes-rate is lower:** 0.409 of the scored questions, against 0.491 in training and 0.461 in validation. That may hurt calibration for methods that learned the training rate, most directly the base rate. Models that condition on the evidence can lower their average through their inputs, so this is not a prediction about any one method. Comparisons between methods matter more than levels.
- **Weather is not re-scored.** The weather questions were held out throughout, and every run already reports them.

## Changes made before scoring

- **29 October 2026, after run 10 finished on validation and before any test scoring.** P2 and S3 now use all three std-scaling-on seeds; the first version had only runs 3 and 9, because run 10 was still training. P3's second test changed. The first version asked that each run's 4-answer average fall on the rule's side of 0.5 on at least 95% of all questions, a threshold set from runs 3 and 9 (99.5% and 99.3%). Run 10 meets it on only 92.4% of validation questions: its single answers are still 0 or 1, but on a few questions its 4 answers disagree, so the average lands between 0 and 1. A threshold that validation already contradicts would not test anything. So P3 now counts single answers that are exactly the rule's 0 or 1, on the questions that have a rule. On validation, this measure separates the collapsed runs (94.6% to 99.9%) from the std-off runs (under 1%).

- **29 October 2026, later the same day, still before any test scoring.** Added the gradient-boosted-tree baselines and S8, after reviewer feedback asked for a stronger simple competitor. They were fitted and scored on validation only.

- **29 October 2026, later the same day, after an outside review and still before any test scoring.** The review found that 36 releases are not 36 independent announcements. On validation (13 announcements), resampling announcements instead of releases leaves every validation conclusion unchanged and the intervals similar; the validation numbers above are now announcement-level. The test intervals resample announcements, with releases as a sensitivity check. The trees' cross-validation now keeps announcements whole (same settings chosen). Also:
  - the aggregation rule is stated as implemented (the average of readable answers), with the alternative rule reported;
  - the scored population, its yes-rate and the timing audit are stated;
  - P1's two conclusions are made explicit, as is the handling of adapters that fail the loading check.

  No prediction or threshold changed.

## What will not change after scoring

- No retraining, and no prompt, sampling or metric changes.
- No models or questions dropped.
- Anything computed beyond this plan is labeled exploratory in the write-up.
