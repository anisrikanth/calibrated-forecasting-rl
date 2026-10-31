# Test split: the one scored evaluation

The test split was scored once on 30 October 2026 (Kaggle, one T4; the run started 00:28 UTC on 31 October), with `notebooks/phase4_test_eval.ipynb` at repo commit `7e515ef`, against the model comparisons fixed in advance in `test_plan.md`. The test questions were not entirely unseen: dry runs of the evaluation code had computed the baselines on them (see the timeline below). No model was retrained or changed: each run's saved LoRA adapter was loaded as it was, plus the untrained model.

- **Questions:** 479 test questions with a market price (36 releases, 13 announcements, June to October 2026; yes-rate 0.409), out of 495 eligible.
- **Models:** 11 (untrained with each prompt, runs 1 and 3 to 10; run 2 was optional and not attached).
- **Loading check:** before any test question, every model re-answered 80 validation questions; all 11 reproduced their stored answers word for word on 83% to 100% of answers ("match"). No adapter was left out of any comparison.
- **The verdicts** come from `analysis/analyze_test.py` and the trees comparison (S8) from `analysis/gbm_baseline.py`, both committed before the run. Everything under "Exploratory" was computed afterwards and is labeled so.

## Primary comparisons: all three predictions hold

Change = runs 4-6 (averaged in each draw) minus the reference. 95% intervals from 4,000 draws resampling whole announcements.

| Prediction | Validation | Test | Reading | Resampling releases instead |
| --- | --- | --- | --- | --- |
| **P1 Brier:** runs 4-6 lower than untrained, key numbers | −0.031 (−0.045 to −0.016) | **−0.025 (−0.034 to −0.016)** | holds | holds (−0.038 to −0.012) |
| **P1 AUC:** runs 4-6 higher | +0.070 (+0.036 to +0.119) | **+0.033 (+0.018 to +0.054)** | holds | holds (+0.009 to +0.058) |
| **P2 log loss:** runs 4-6 lower than runs 3, 9, 10 | −0.468 (−0.741 to −0.236) | **−0.409 (−0.676 to −0.195)** | holds (both required) | holds |
| **P2 calibration error** | −0.148 (−0.194 to −0.027) | **−0.109 (−0.134 to −0.004)** | | (−0.146 to −0.012) |
| **P3:** runs 3, 9, 10 each ≥ 90% of answers exactly 0 or 1 and ≥ 90% exactly the rule's 0 or 1 | 99.4 / 98.8 / 99.1%; 99.9 / 98.7 / 94.6% | **99.8 / 99.5 / 99.3%; 99.8 / 99.0 / 95.1%** | holds | (no interval) |

- **P1:** each seed improves on its own, with every interval excluding 0: Brier −0.030, −0.027, −0.018 and AUC +0.031, +0.037, +0.033 (runs 4, 5, 6). The Brier gain is a little smaller than on validation, and the AUC gain about half as large. The untrained model ranks the test questions much better than the validation ones (AUC 0.855 against 0.757), so there was less room.
- **P2:** the calibration-error interval only just excludes 0 (upper end −0.004). Not part of the prediction, but also in the same direction: Brier −0.047 (−0.090 to −0.013), AUC +0.071 (+0.046 to +0.108).
- **P3:** the collapse is as complete as on validation. Run 10 again departs from the rule most often (95.1% of answers equal to it).

`test_results/test_verdicts.csv`, `test_p3.csv`.

## Secondary comparisons (descriptive, no verdicts)

The plan applies no correction for multiple comparisons to these, and none is promoted.

| | Comparison | Validation: Brier / AUC change | Test: Brier change | Test: AUC change |
| --- | --- | --- | --- | --- |
| S1 | Runs 4-6 vs logistic regression | +0.008 / −0.028 | +0.010 (−0.014 to +0.029) | −0.011 (−0.035 to +0.016) |
| S8 | Runs 4-6 vs trees, key numbers | +0.012 / **−0.033** | +0.012 (−0.010 to +0.030) | −0.013 (−0.039 to +0.017) |
| S8 | Runs 4-6 vs trees, key numbers + series | +0.019 / **−0.042** | +0.015 (−0.003 to +0.028) | −0.017 (−0.039 to +0.005) |
| S2 | Untrained: key numbers vs plain | **−0.071 / +0.237** | **−0.100** (−0.129 to −0.074) | **+0.248** (+0.148 to +0.367) |
| S3 | Runs 3, 9, 10 vs untrained, key numbers | +0.017 / +0.025 | +0.022 (−0.018 to +0.073) | **−0.037** (−0.075 to −0.006) |
| S4 | Run 7 (plain, std off) vs untrained, plain | −0.009 / −0.035 | +0.000 (−0.026 to +0.019) | −0.013 (−0.089 to +0.090) |
| S5 | Run 1 (plain, std on) vs untrained, plain | −0.005 / +0.024 | **−0.022** (−0.045 to −0.003) | +0.021 (−0.059 to +0.116) |
| S6 | Run 8 (shuffled outcomes) vs untrained, key numbers | **+0.035 / −0.048** | **+0.061** (+0.046 to +0.075) | −0.027 (−0.060 to +0.004) |
| S7 | Runs 4-6 vs run 8 | **−0.065 / +0.118** | **−0.087** (−0.108 to −0.063) | **+0.061** (+0.035 to +0.098) |

Bold: interval excludes 0. All intervals resample announcements; validation intervals are in `review_checks.md`.

- **Simple models on the same features (S1, S8).** As on validation, the trained models do not outperform the logistic regression or the trees, which are ahead on every metric. But on test no Brier or AUC difference is resolved. The AUC gap to the trees, which excluded 0 on validation, is smaller here (0.013 and 0.017) and its intervals include 0. The one interval that excludes 0 is log loss against the trees with the series: +0.054 (+0.006 to +0.091).
- **The elicitation effect (S2) is as large as on validation.**
- **Std scaling on vs the untrained model (S3).** Log loss is worse again: +0.343 (+0.113 to +0.629). Calibration error is worse, +0.075, but its interval now includes 0 (−0.011 to +0.135). AUC is now worse too, which fits the collapse: on test the untrained model ranks better than the latest-result rule that the collapsed runs copy (0.855 against 0.804), while on validation it ranked worse (0.757 against 0.775).
- **Plain prompt (S4, S5).** Run 7 still changes nothing. Run 1 improves Brier and log loss on test (log loss −0.053, −0.104 to −0.011), which it did not on validation; its AUC does not change. There is no correction for multiple comparisons, and this is one of 36 secondary intervals (nine comparisons, four metrics). Exploratory: run 1 again lowers its average forecast by about 0.06 (332 of 479 forecasts fall, 145 rise), and the untrained plain model overpredicts YES more on test (average 0.573 against a yes-rate of 0.409), so a shift helps more here. The untrained model's forecasts shifted down uniformly by the same amount score 0.249, which accounts for 0.016 of run 1's 0.022.
- **Shuffled outcomes (S6, S7).** Run 8 is again worse than the untrained model, and the main setup beats it on every metric.

`test_results/test_secondary.csv`, `gbm_test_paired_bootstrap.csv`.

## Every method on the test split

| Test econ (479 questions) | Brier | Log loss | Calibration error | AUC | Average forecast |
| --- | --- | --- | --- | --- | --- |
| Market price | 0.073 | 0.236 | 0.038 | 0.967 | 0.424 |
| Gradient-boosted trees, key numbers + series | **0.125** | **0.384** | 0.064 | **0.906** | 0.409 |
| Gradient-boosted trees, key numbers | 0.127 | 0.393 | 0.046 | 0.901 | 0.407 |
| Logistic regression, key numbers | 0.130 | 0.400 | 0.057 | 0.899 | 0.402 |
| Counting rule | 0.138 | 0.461 | 0.059 | 0.880 | 0.419 |
| Latest-result rule (0/1) | 0.193 | 0.889 | 0.188 | 0.804 | 0.430 |
| Base rate | 0.247 | 0.687 | 0.085 | 0.554 | 0.495 |
| Always 0.5 | 0.250 | 0.693 | 0.091 | 0.500 | 0.500 |
| Untrained, plain prompt | 0.265 | 0.732 | 0.168 | 0.607 | 0.573 |
| Run 1: plain, std on | 0.243 | 0.679 | 0.132 | 0.627 | 0.514 |
| Run 7: plain, std off | 0.265 | 0.724 | 0.174 | 0.594 | 0.582 |
| Untrained, key numbers | 0.165 | 0.503 | 0.110 | 0.855 | 0.495 |
| Runs 3, 9, 10: key numbers, std on | 0.177 to 0.194 | 0.78 to 0.90 | 0.17 to 0.19 | 0.80 to 0.84 | 0.42 to 0.48 |
| Runs 4, 5, 6: key numbers, std off | 0.135 to 0.146 | 0.42 to 0.46 | 0.05 to 0.09 | 0.886 to 0.892 | 0.44 to 0.50 |
| Run 8: key numbers, std off, shuffled outcomes | 0.226 | 0.645 | 0.176 | 0.828 | 0.536 |

Bold: best short of the market. Ranges are across the three seeds. `test_results/test_metrics.csv`, `gbm_test_scores.csv`.

## Checks the plan asked for

- **Aggregation rule.** Every trained run's answers were at least 99.95% readable; the untrained models', 96.8% (key numbers) and 98.8% (plain). Counting each unreadable answer as 0.5 instead of averaging the readable ones changes no Brier score by more than 0.002 (`test_descriptive.csv`).
- **Each announcement weighted equally.** No ordering changes: runs 4-6 0.134 to 0.151, untrained key numbers 0.167, logistic regression 0.132, trees with series 0.127, market 0.080.
- **Single answers.** Runs 4-6 score 0.143 to 0.157 as single answers, the untrained model 0.196, so as on validation the four-answer average flatters the untrained model more.
- **Brier by series** is in `test_results/test_series.csv` (descriptive).

## Exploratory (computed after scoring; not part of the plan)

`analysis/test_diagnostics.py`.

- **Methods that read the recent results scored better on test.** Every such method has a lower Brier score and a higher AUC than on validation (calibration error does not always improve: the untrained key-numbers model's rises from 0.060 to 0.110): the untrained key-numbers model's AUC goes from 0.757 to 0.855, the logistic regression's from 0.855 to 0.899, runs 4-6 from 0.82–0.83 to 0.886–0.892. The market (Brier 0.062 to 0.073) and the base rate (0.243 to 0.247) score slightly worse, and run 7 is unchanged. These comparisons describe the period; they do not show why it differed. That is why the plan compares methods, not levels.
- **Calibration on test.**
  - Runs 4-6 forecast YES too often: average forecast 0.44 to 0.50 against a yes-rate of 0.409, and the reliability curve sits below the diagonal between about 0.1 and 0.4 (`figures/reliability_test.png`).
  - Their calibration slopes are 1.19 to 1.51, above 1, so their forecasts are not extreme enough for this period (validation: 0.80 to 1.07). The untrained model's is 1.56; the logistic regression's 0.89 and the trees' 1.05 to 1.09.
  - The std-on runs remain far too extreme: slopes 0.31 to 0.36.
- **Other comparisons from the validation write-up, not in the plan.**
  - Runs 4-6 vs the counting rule: Brier +0.002 (−0.029 to +0.028), AUC +0.008 (−0.032 to +0.047).
  - Runs 3, 9, 10 vs the latest-result rule: Brier −0.006 (−0.014 to +0.003), AUC +0.014 (+0.001 to +0.029). Still close to the rule; the small AUC gain comes mostly from run 10's departures.
- **Payrolls stayed the hard series.** Runs 4-6 score 0.261 to 0.284 there, the untrained model 0.256 and the logistic regression 0.317. The base rate (0.242) beats every method, the market included (0.265). The std-on runs, copying the latest payrolls result, score 0.43 to 0.48.
- **Skill relative to the base rate:** market 0.70, trees 0.49, logistic regression 0.48, counting rule 0.44, runs 4-6 0.41 to 0.46, untrained key numbers 0.33, runs 3, 9, 10 0.22 to 0.28, run 8 0.09 (`test_skill_scores.csv`).
- **Ladder violations** (431 adjacent threshold pairs): market 4%, logistic regression 4%, trees 1%, untrained key numbers 40%, runs 4-6 33% to 36%, runs 3, 9, 10 0% to 2%.

## Timeline

All times U.S. Eastern. Commits are in this repository; `test_plan.md` records each amendment and its reason.

| When | Commit | Event |
| --- | --- | --- |
| 28 Oct, 22:14 | | First dry runs of the test notebook. They pair made-up model answers with the real baselines, so they compute and print the baselines' test scores (market price, base rate, rules, logistic regression). Dry runs continued through the revisions below. |
| 29 Oct, 08:06 | `17b601b` | Plan and test notebook first committed. The plan stated, wrongly, that no baseline had been scored. |
| 29 Oct, 08:25 | `f113ac1` | Amendment 1: P2 and S3 include run 10; P3's second criterion changed because validation run 10 did not meet the original one. |
| 29 Oct, 08:57 | `10b109c` | Amendment 2: gradient-boosted trees (S8) added after a reviewer asked for a stronger baseline; settings chosen by cross-validation on the training questions. |
| 29 Oct, 12:10 | `ba3e6ef` | Amendment 3, after an outside review: announcement-level intervals and wording. No prediction or threshold changed. |
| 29 Oct, 12:45 | `ba3e6ef` | The loading check run alone on Kaggle: every model answered 80 validation questions; no test question was scored. |
| 29 Oct, 13:26–13:38 | | Last dry runs: the notebook, the trees script in test mode (computing the trees' test scores) and the verdict script, all with made-up model answers. |
| 29 Oct, 15:12 | `7e515ef` | Amendment 4: the baseline exposure disclosed; verdict script and the trees' test mode added; P3's wording matched to its computation. No prediction or threshold changed. |
| 30 Oct, 20:28 | `7e515ef` | The scored run: the models' first answers to test questions. |
| 30 Oct, 23:47 | `f35629a` | Results committed. |

No language model answered a test question before the scored run. The baselines' test scores were available from the first dry run on, before the plan was first committed and before amendments 1 to 3, so they could have been seen when those amendments were made. The amendments' stated reasons are validation results and outside reviews, and every baseline's settings were fixed on training questions only.

## Files

`test_results/`:
- **From the notebook:** `test_answers.csv` (every answer, with its text), `test_check.csv`, `test_metrics.csv`, `test_comparisons.csv`, `test_comparisons_by_release.csv`, `test_collapse.csv`, `test_by_series.csv`, and `notebook_summary.txt`, the run's printed summary.
- **From `gbm_baseline.py` (test mode):** `gbm_test_scores.csv`, `gbm_test_paired_bootstrap.csv`, `gbm_test_forecasts.csv`.
- **From `analyze_test.py`:** `test_verdicts.csv`, `test_p3.csv`, `test_secondary.csv`, `test_descriptive.csv`.
- **From `test_diagnostics.py` (exploratory):** `test_diagnostics.csv`, `test_series.csv`, `test_skill_scores.csv`, `test_exploratory_bootstrap.csv`, `reliability_test.csv`.

To reproduce the analysis from the saved answers (with `markets.csv` and the notebook's CSVs in `<data>`):

```bash
python -I eval/analysis/gbm_baseline.py . <data> eval/test_results test
cp eval/test_results/gbm_test_paired_bootstrap.csv <data>/
python -I eval/analysis/analyze_test.py . <data> eval/test_results
python -I eval/analysis/test_diagnostics.py . <data> eval/test_results
```
