# Calibrated Forecasting with RL

**When does RL with a proper scoring rule help a small language model forecast?** We train Llama 3.2 3B Instruct with GRPO and a Brier-score reward on resolved [Kalshi](https://kalshi.com) markets about U.S. economic releases. The evidence in every prompt is held fixed: the series' recent results, all public before the question. That lets us measure what RL changes against simple rules built from the same numbers. Paper draft: [`paper/`](paper/).

## Research questions

1. **Evidence or level?** With the evidence held fixed, does proper-score RL teach a small model to use that evidence, or does it only move the level and spread of its probabilities?
2. **The optimizer.** What does GRPO's division of advantages by the group's reward standard deviation ("std scaling", TRL's default) do with a proper-scoring-rule reward?
3. **Simple models.** How does the trained model compare with simple models on the same evidence (a counting rule, a logistic regression, gradient-boosted trees), and with the market price?
4. **Transfer.** Does anything carry over to a domain it never trained on (daily temperature markets)?
5. **Reward hacking.** What shortcuts appear during training, and what measurements catch them?

> **Findings** (validation split: 410 questions, 13 announcements; intervals resample whole announcements)
>
> - **Held on the test split.** The newest 479 questions (13 announcements, June to October 2026) were scored once, against precommitted model comparisons ([`eval/test_plan.md`](eval/test_plan.md); the baselines' test scores were computed earlier, during code testing, as the plan records). All three held: against the untrained model, the main setup's Brier changed by −0.025 (−0.034 to −0.016) and its AUC by +0.033 (+0.018 to +0.054); it beat the std-on runs in log loss and calibration error; and the std-on models kept their collapsed behavior on the test questions ([`eval/test_results.md`](eval/test_results.md)).
> - **The plain prompt elicits little discrimination.** The untrained model scores AUC 0.52, while counting how many recent results would have resolved YES reaches 0.85. A deterministic "key numbers" block that spells out the comparison lifts it to 0.76 with no training. A prompt ablation traces most of the gain to restating the YES condition and the count ([`eval/prompt_ablation.md`](eval/prompt_ablation.md)).
> - **Among the configurations tested, only the key-numbers prompt with std scaling off improved both Brier and AUC** (one pass of training). Over three seeds: Brier −0.031 (95% interval −0.045 to −0.016), AUC +0.070 (+0.036 to +0.119), with no detectable change in calibration error. With outcomes shuffled, the same setup gets worse ([`eval/rl_runs5to7.md`](eval/rl_runs5to7.md), [`eval/rl_run8.md`](eval/rl_run8.md)).
> - **With std scaling on, RL collapses onto a lookup rule.** In all three seeds, near-certain answers pass 90% by about step 130 of 347. More than 98% of evaluation answers end up exactly 0 or 1, matching the rule "has the latest result already crossed the threshold?" Under the training reward itself, that rule scores worse than its calibrated version ([`eval/rl_runs9_10.md`](eval/rl_runs9_10.md)).
> - **Without usable evidence or a real signal, forecasts contract toward a constant**, consistent with hedging when the evidence isn't used.
> - **Against simple models:** the trained models don't outperform a logistic regression or the counting rule on the same features (intervals include 0), and trail gradient-boosted trees on AUC (−0.03 to −0.04; on test −0.01 to −0.02, not resolved). The market price (never shown to the model) is far ahead: Brier 0.062 against 0.170. No useful discrimination on the weather brackets ([`eval/extra_analyses.md`](eval/extra_analyses.md), [`eval/review_checks.md`](eval/review_checks.md)).

![Spread of stated probabilities during training: std scaling on vs off, plain prompt vs key numbers](eval/figures/spread_2x2.png)

## Setup at a glance

| Piece | Choice |
| --- | --- |
| Model | Llama 3.2 3B Instruct (1B only in the Phase 2 baselines) |
| Questions | 2,296 Kalshi economic questions in 249 releases: CPI (4 series), payrolls, unemployment, Fed funds rate, Fed decisions, January 2024 to October 2026 |
| Splits | By release date: train 1,391 (2024-25), validation 410 (Jan-May 2026), test 495 (Jun-Oct 2026, scored once) |
| Evidence | The six most recent results that settled before the question; optionally a "key numbers" block computed from them |
| Leakage and contamination | Context only from markets settled before the question time (audited). Evaluation questions are all from 2026, after the model's release (Sept 2024) and stated knowledge cutoff (Dec 2023) |
| Training | GRPO with TRL + Unsloth, 16-bit LoRA rank 32, one pass, 4 answers per question |
| Reward | 1 − (stated probability − outcome)², plus a small format reward |
| Out-of-domain set | Kalshi daily high-temperature brackets, New York and Chicago, never trained on |
| Evaluation | Brier score, log loss, calibration error, AUC; 95% intervals from resampling whole announcements (series released together); simple-rule baselines; a shift-matched untrained model |
| Benchmark | The market price at question time (never shown to the model) |
| Compute | About 30 GPU-hours on free Kaggle T4s; $0 |

## Results and write-ups

| What | Where |
| --- | --- |
| Untrained baselines (1B and 3B) | [`eval/baselines.md`](eval/baselines.md) |
| Run 1 (plain prompt): a level shift, not better ordering | [`eval/rl_run1.md`](eval/rl_run1.md) |
| Runs 2-3: overconfidence, and the first collapse | [`eval/rl_run2_run3.md`](eval/rl_run2_run3.md) |
| Run 4: std scaling off | [`eval/rl_run4.md`](eval/rl_run4.md) |
| Runs 5-7: three seeds and the full 2 × 2 | [`eval/rl_runs5to7.md`](eval/rl_runs5to7.md) |
| Run 8: shuffled-outcome control | [`eval/rl_run8.md`](eval/rl_run8.md) |
| Runs 9-10: two more seeds of the collapse | [`eval/rl_runs9_10.md`](eval/rl_runs9_10.md) |
| Prompt ablation | [`eval/prompt_ablation.md`](eval/prompt_ablation.md) |
| Training diagnostics, tree baselines, skill scores, compute | [`eval/extra_analyses.md`](eval/extra_analyses.md) |
| Test-set plan (fixed before scoring) | [`eval/test_plan.md`](eval/test_plan.md) |
| Test-set results (scored once) | [`eval/test_results.md`](eval/test_results.md) |
| Reward hacks and parser issues | [`writeup/hacks.md`](writeup/hacks.md) |
| Limitations and data quirks | [`writeup/limitations.md`](writeup/limitations.md) |

## Status

- [x] Phase 0: setup and GRPO smoke test on Kaggle (20 steps, plus checkpoint resume)
- [x] Phase 1: Kalshi data pipeline (14,662 markets, 10 series), context snapshots, leakage audit, date-based splits
- [x] Phase 2: forecasting environment, baselines and leak probe (`eval/baselines.md`)
- [x] Phase 3: GRPO training and reward-hacking log. Runs 1-10 done (`eval/rl_run1.md`, `eval/rl_run2_run3.md`, `eval/rl_run4.md`, `eval/rl_runs5to7.md`): the full prompt × std-scaling 2 × 2 and three seeds of the main setup; 5 entries in `writeup/hacks.md`; shuffled-outcome control (run 8, `eval/rl_run8.md`); prompt ablation (`eval/prompt_ablation.md`); two more seeds of the collapse (runs 9 and 10, `eval/rl_runs9_10.md`) reproduce it
- [x] Phase 4: one evaluation on the untouched test split, with the comparisons fixed in advance in `eval/test_plan.md` (`notebooks/phase4_test_eval.ipynb`): all three predictions held (`eval/test_results.md`); extra analyses (`eval/extra_analyses.md`: training diagnostics, tree baselines, skill scores, compute)
- [ ] Phase 5: write-up. Full draft with test results in `paper/`; outside review, then arXiv

## Repository layout

```
data/        Kalshi pipeline (see data/README.md); raw/ and processed/ are git-ignored
env/         forecasting environment: prompt, parser, reward (Phase 2)
train/       GRPO training configs and scripts (Phase 3)
eval/        metrics, reliability diagrams, results tables
notebooks/   Kaggle notebooks
paper/       the preprint (LaTeX)
tests/       unit tests (run with pytest)
writeup/     notes, hacks.md (reward hacks and parser issues found), drafts
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Pull resolved Kalshi markets and build the dataset (no API key needed)
python -m data.pull_kalshi discover   # check which series tickers exist
python -m data.pull_kalshi all        # markets -> prices -> build

pytest                                # offline unit tests
```

## License

Code is MIT-licensed (see `LICENSE`). Notebooks in `notebooks/` are adapted from [Unsloth's notebooks](https://github.com/unslothai/notebooks) and remain under LGPL-3.0. Kalshi market data is subject to Kalshi's terms and is not redistributed here; the pipeline rebuilds it.
