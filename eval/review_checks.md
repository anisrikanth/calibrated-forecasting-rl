# Checks added after outside review (29 October 2026)

An outside review of the paper draft asked for several checks. All of them use the saved validation answers; none needed new generation. Scripts: `analysis/review_checks.py`, `analysis/announcement_intervals_extra.py`, `analysis/gbm_baseline.py`.

## 1. Independent announcements, not releases

CPI's four series come out in one release, payrolls and unemployment share the jobs report, and the Fed rate and Fed decision share a meeting. So the 36 validation releases are only **13 independent announcements**; the test split also has 13. Resampling whole announcements leaves every conclusion unchanged, and the intervals are similar (`analysis/review_paired_bootstrap.csv`, both resampling units):

| Comparison (validation econ) | Brier, announcements | Brier, releases | AUC, announcements | AUC, releases |
| --- | --- | --- | --- | --- |
| Runs 4-6 vs untrained, key numbers | **−0.031** (−0.045 to −0.016) | **−0.031** (−0.049 to −0.013) | **+0.070** (+0.036 to +0.119) | **+0.070** (+0.030 to +0.113) |
| Runs 4-6 vs runs 3, 9, 10 | **−0.048** (−0.094 to −0.007) | **−0.048** (−0.099 to −0.003) | **+0.045** (+0.021 to +0.076) | **+0.045** (+0.017 to +0.077) |
| Runs 4-6 vs logistic regression | +0.008 (−0.031 to +0.044) | +0.008 (−0.035 to +0.041) | −0.028 (−0.057 to +0.011) | −0.028 (−0.060 to +0.006) |
| Runs 4-6 vs counting rule | +0.003 (−0.026 to +0.040) | +0.003 (−0.028 to +0.033) | −0.019 (−0.091 to +0.031) | −0.019 (−0.078 to +0.033) |
| Runs 4-6 vs trees, key numbers + series | +0.019 (−0.006 to +0.049) | +0.019 (−0.007 to +0.045) | **−0.042** (−0.069 to −0.018) | **−0.042** (−0.071 to −0.015) |
| Runs 4-6 vs untrained, best three-line prompt | −0.019 (−0.034 to +0.002) | **−0.019** (−0.036 to −0.001) | **+0.038** (+0.002 to +0.074) | **+0.038** (+0.002 to +0.073) |
| Run 8 (shuffled) vs untrained, key numbers | **+0.035** (+0.011 to +0.054) | **+0.035** (+0.007 to +0.059) | **−0.048** (−0.105 to −0.008) | −0.048 (−0.100 to +0.003) |

Bold: interval excludes 0. Each of runs 4, 5 and 6 alone still improves on the untrained model in both Brier and AUC with announcement-level intervals. With 13 announcements, these intervals are rough; that is the effective sample size. The test plan now resamples announcements (`test_plan.md`, third amendment, before scoring).

## 2. Unreadable answers

The code averages the readable answers (0.5 if none is readable); the draft paper said each unreadable answer counted as 0.5 before averaging. Every trained run's validation answers were all readable, so their numbers are unaffected. The untrained models were 97.7% (key numbers) and 98.4% (plain) readable. Counting unreadable answers as 0.5 moves their Brier from 0.2014 to 0.1998 and from 0.2720 to 0.2707. The paper and plan now describe the implemented rule.

## 3. Question timing

Every econ question in every split is asked exactly 24 hours before its market closes; none was shifted to fit the market's opening. Markets close at the scheduled announcement (8:25 to 8:30 Eastern, or just before the 2 pm Fed statement). In four validation releases they close shortly after it, the same morning. So each question comes about a day before its announcement. Markets that settled long after closing are all training questions delayed by the 2025 shutdown, and they closed before the delayed release.

## 4. Single answers vs four-answer averages, ladders, calibration slope, equal weighting

From `analysis/review_diagnostics.csv` (validation econ):

| | Brier (4-answer) | Brier (single answers) | Ladder violations | Calibration slope | Brier (equal weight per announcement) |
| --- | --- | --- | --- | --- | --- |
| Untrained, key numbers | 0.201 | 0.233 | 38% | 0.82 | 0.211 |
| Runs 4-6 (std off) | 0.169 to 0.171 | 0.177 to 0.182 | 28% to 35% | 0.80 to 1.07 | 0.174 to 0.181 |
| Runs 3, 9, 10 (std on) | 0.211 to 0.223 | 0.224 to 0.230 | 0% to 2% | 0.27 to 0.30 | 0.221 to 0.226 |
| Logistic regression | 0.163 | | 6% | 0.61 | 0.162 |
| Counting rule | 0.168 | | 0% | 0.48 | 0.160 |

- **Averaging four answers helps the untrained model more.** Its answers vary more within a question (variance 0.033, against 0.005 to 0.011 for runs 4-6), so on single answers the trained models' gain is larger.
- **The trained models still order 28% to 35% of adjacent thresholds the wrong way.** Pooled AUC partly reflects ordering thresholds within a release, so it shouldn't be read as general skill at predicting the next economic value.
- **Calibration slope:** below 1 means forecasts too extreme. The collapsed runs are far too extreme; runs 4-6 are close to 1.
- Reliability diagrams with bin counts: `figures/reliability_validation.png`.

## 5. Shuffled-outcome control

Run 8's training outcomes are permuted once, at random, across all 1,391 training questions (fixed by the seed) and kept for the whole run. That keeps the yes-rate. It breaks the link between questions and outcomes, and leaves 26% of adjacent threshold pairs in a training ladder inconsistent (none in the real outcomes).
