# Runs 9 and 10: two more seeds of the collapse

Runs 9 and 10 repeat run 3 exactly, with training seeds 1 and 2; run 3 used seed 3407. The settings are key numbers in the prompt, GRPO std scaling on, Brier plus format reward, and one pass of 347 steps at learning rate 1e-5. Run names `3b_brier_keynum_seed1` and `3b_brier_keynum_seed2`, repo commit 5e20686, 103 and 94 minutes of training. Evaluation is as for every run: 410 validation econ questions (36 releases) and 360 held-out weather questions (60 days), 4 answers at temperature 0.7, averaged, with the same eval days and sampling seed. With runs 4 to 6 (std scaling off, the same three seeds), there are now three seeds on each side of the std-scaling comparison.

## Results in brief

1. **The collapse replicates in all three seeds.**
   - Runs 3, 9 and 10 give 99.4%, 98.8% and 99.1% of their validation answers as exactly 0.00 or 1.00.
   - The std-off runs 4 to 6 give under 1% of their answers as exactly 0 or 1.
   - The timing barely moves. The share of near-certain training answers passes 90% at step 124 (runs 3 and 9) and step 136 (run 10), about a third of the way through the pass.
2. **It is the same shortcut, the latest-result rule.**
   - On the 395 validation questions with a numeric threshold, 99.9%, 98.7% and 94.6% of single answers are exactly the rule's 0 or 1.
   - Averaged over the three seeds, the std-on runs cannot be told apart from the rule: Brier +0.001 (−0.017 to +0.017), AUC +0.007 (−0.006 to +0.023).
3. **Run 10 departs from the rule a little, and not because it judges better.**
   - On 25 threshold questions, 10 of them payrolls, its 4-answer average lands on the other side of 0.5 from the rule. On 72% of those, its 4 answers disagree with each other.
   - Single answers stay at 0 or 1. Some of them lean on the count of recent results instead of the latest one, for example: "the latest result (−92,000) is below the threshold and all recent results have been above the threshold, the question will resolve to YES. Probability: 1.00".
   - On those 25 questions, run 10's side is right 56% of the time and the rule's 44%, too few to mean much.
   - Averaging 4 disagreeing 0/1 answers gives forecasts between 0 and 1, which helps a little: Brier 0.211 and AUC 0.800, against the rule's 0.218 and 0.775 (AUC +0.026, −0.002 to +0.064).
4. **Std scaling off beats std scaling on, now seed for seed.** Runs 4 to 6 vs runs 3, 9 and 10, each group averaged:
   - Brier −0.048 (−0.099 to −0.003). With run 3 alone the Brier interval included 0 (−0.053, −0.116 to +0.002).
   - Log loss −0.468 (−0.773 to −0.198).
   - Calibration error −0.148 (−0.199 to −0.019).
   - AUC +0.045 (+0.018 to +0.077).
5. **Against the untrained model, std scaling on mostly costs calibration.**
   - Brier +0.017 (−0.038 to +0.080) and AUC +0.025 (−0.028 to +0.078), both within noise.
   - Log loss +0.396 (+0.101 to +0.730) and calibration error +0.156 (+0.038 to +0.204), both clearly worse.
6. **Stopping reasoning is not part of the collapse.** Run 3's answers shrank to a median of 7 tokens. Runs 9 and 10 kept short reasoning (73 and 37 tokens) and still answer 0 or 1. What replicates is the 0/1 answer, not the length.
7. **Weather gets worse, as with run 3.** Brier 0.298, 0.261 and 0.289, against 0.223 untrained. The three-seed average is +0.060 (+0.038 to +0.080). The std-off runs score 0.208 to 0.223.

## Validation econ: every method on the same 410 questions

| Method | Brier | Log loss | Calibration error | AUC | Average forecast | Spread (std. dev.) |
| --- | --- | --- | --- | --- | --- | --- |
| Market price | 0.062 | 0.200 | 0.023 | 0.977 | 0.460 | 0.417 |
| Logistic regression on key numbers | 0.163 | 0.522 | 0.087 | 0.855 | 0.418 | 0.375 |
| Runs 4 to 6 (std off, 3 seeds) | 0.169 to 0.171 | 0.519 to 0.527 | 0.054 to 0.088 | 0.820 to 0.832 | 0.47 to 0.53 | 0.27 to 0.32 |
| Untrained, key numbers | 0.201 | 0.594 | 0.060 | 0.757 | 0.482 | 0.244 |
| Run 10 (std on, seed 2) | 0.211 | 0.934 | 0.206 | 0.800 | 0.482 | 0.486 |
| Latest-result rule (0/1) | 0.218 | 1.003 | 0.212 | 0.775 | 0.424 | 0.489 |
| Run 9 (std on, seed 1) | 0.222 | 1.005 | 0.217 | 0.774 | 0.425 | 0.484 |
| Run 3 (std on, seed 3407) | 0.223 | 1.030 | 0.224 | 0.771 | 0.422 | 0.493 |
| Base rate | 0.243 | 0.679 | 0.042 | 0.577 | 0.495 | 0.083 |

Yes-rate 0.461. `analysis/runs9_10_scores.csv`.

## How completely each run collapsed (validation econ)

| | Answers exactly 0 or 1 | Answers exactly the rule's 0 or 1 (threshold questions) | All 4 answers identical | Median answer length (tokens) | Training step when 90% of answers are near-certain |
| --- | --- | --- | --- | --- | --- |
| Run 3 (std on, seed 3407) | 99.4% | 99.9% | 98.3% | 7 | 124 |
| Run 9 (std on, seed 1) | 98.8% | 98.7% | 94.2% | 73 | 124 |
| Run 10 (std on, seed 2) | 99.1% | 94.6% | 91.7% | 37 | 136 |
| Run 4 (std off, seed 3407) | 0.6% | 0.6% | 6.6% | 74 | never |
| Run 5 (std off, seed 1) | 0.0% | 0.0% | 2.9% | 66 | never |
| Run 6 (std off, seed 2) | 0.2% | 0.2% | 1.0% | 68 | never |
| Untrained, key numbers | 4.6% | 4.2% | 0.2% | 110 | |

"Near-certain" means at or below 0.05, or at or above 0.95, over a rolling window of 512 training answers. `analysis/runs9_10_collapse.csv`, `analysis/run10_off_rule.json`.

![Spread of stated probabilities during training, 2 × 2 with three seeds per key-numbers cell](figures/spread_2x2.png)

## Paired comparisons (validation econ, 95% intervals)

Event-level bootstrap, 4,000 draws. For groups of runs, each draw averages the group's runs on the same resampled releases, so the interval covers question sampling, not seed variation.

| Comparison | Brier change | Log loss change | Calibration error change | AUC change |
| --- | --- | --- | --- | --- |
| Runs 4 to 6 (std off) vs runs 3, 9, 10 (std on) | **−0.048** (−0.099 to −0.003) | **−0.468** (−0.773 to −0.198) | **−0.148** (−0.199 to −0.019) | **+0.045** (+0.018 to +0.077) |
| Runs 3, 9, 10 vs untrained (key numbers) | +0.017 (−0.038 to +0.080) | **+0.396** (+0.101 to +0.730) | **+0.156** (+0.038 to +0.204) | +0.025 (−0.028 to +0.078) |
| Runs 3, 9, 10 vs latest-result rule | +0.001 (−0.017 to +0.017) | −0.013 (−0.095 to +0.058) | +0.003 (−0.018 to +0.032) | +0.007 (−0.006 to +0.023) |
| Run 9 vs latest-result rule | +0.004 (−0.004 to +0.014) | +0.002 (−0.043 to +0.054) | +0.005 (−0.006 to +0.032) | −0.000 (−0.012 to +0.011) |
| Run 10 vs latest-result rule | −0.007 (−0.053 to +0.028) | −0.070 (−0.280 to +0.084) | −0.007 (−0.057 to +0.042) | +0.026 (−0.002 to +0.064) |
| Run 10 vs run 3 | −0.012 (−0.057 to +0.021) | −0.097 (−0.304 to +0.047) | −0.019 (−0.067 to +0.024) | **+0.029** (+0.002 to +0.065) |

Bold: interval excludes 0. Full table, including weather: `analysis/runs9_10_paired_bootstrap.csv`.

## Per series (validation econ, Brier, point estimates)

| Series | Latest-result rule | Run 3 | Run 9 | Run 10 |
| --- | --- | --- | --- | --- |
| Core CPI (month) | 0.095 | 0.095 | 0.104 | 0.086 |
| Core CPI (year) | 0.145 | 0.145 | 0.141 | 0.149 |
| CPI (month) | 0.267 | 0.267 | 0.257 | 0.262 |
| CPI (year) | 0.227 | 0.227 | 0.227 | 0.239 |
| Fed funds rate | 0.000 | 0.000 | 0.003 | 0.048 |
| Unemployment | 0.060 | 0.060 | 0.062 | 0.053 |
| Payrolls | 0.541 | 0.541 | 0.530 | 0.417 |
| Fed decision (no rule; base rate shown) | 0.160 | 0.299 | 0.330 | 0.456 |

Run 3 matches the rule in every threshold series. Run 9 is within about 0.01 of it in each. Run 10 differs most on payrolls, where the latest result is a poor guide and its mixed answers help (0.417 vs 0.541).

## What this means

- **The std-scaling finding is not one unlucky seed.**
  - With GRPO's default std scaling, all three seeds turned the model into a 0/1 lookup rule, at about the same point in training.
  - With it off, all three seeds kept graded forecasts and improved on the untrained model.
  - The difference between the two groups holds on every metric, with intervals that exclude 0.
- **Why std scaling does this:** see `writeup/hacks.md`, entries 3 and 5. Dividing each answer's advantage by its group's reward spread favours confident answers and feeds on itself ([Bereket & Leskovec 2025](https://arxiv.org/pdf/2508.11800)). The 0/1 answers are not even what the reward prefers: on the training questions, the latest-result rule stated at its hit rates earns a higher average Brier reward than its 0/1 version (0.870 vs 0.847). The three seeds show the collapse comes from the scaling, not from chance.
- **Phase 3 training is finished.** Ten runs in total. Next is the one evaluation on the untouched test split, with the comparisons fixed in advance in `eval/test_plan.md`.
