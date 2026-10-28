# Run 8: shuffled-outcome control

Run 8 repeats run 4 exactly (key numbers in the prompt, GRPO std scaling off, training seed 3407, one pass at learning rate 1e-5), except that each training question's outcome is randomly reassigned across the training set before rewards are computed (`SHUFFLE_OUTCOMES = True`). The yes-rate stays the same (0.491), but the link between a question and its outcome is gone: 48.4% of rewards happen to use the real outcome, about what chance gives. Run name `3b_brier_keynum_nostd_shuffled`, repo commit 0f576a3, 124 minutes of training. Evaluation uses the real outcomes, as always.

The question it answers: **how much of what RL did in runs 4 to 6 comes from the outcomes, and how much from the training process alone?**

## Result

The training process alone makes the model worse. Without a real signal, RL squeezes the forecasts toward a single number: the same hedging as run 7, where the outcomes were real but the model could not read its evidence.

| Validation econ (410 questions) | Brier | AUC | Calibration error | Average forecast | Spread (std. dev.) |
| --- | --- | --- | --- | --- | --- |
| Untrained, key numbers | 0.201 | 0.757 | 0.060 | 0.482 | 0.244 |
| Run 4 (real outcomes, same seed) | 0.169 | 0.828 | 0.063 | 0.472 | 0.319 |
| **Run 8 (shuffled outcomes)** | **0.236** | **0.708** | 0.113 | 0.538 | **0.074** |

Paired comparisons, event-level bootstrap with 4,000 draws (`analysis/run8_paired_bootstrap.csv`):

| Comparison | Eval set | Brier change | AUC change |
| --- | --- | --- | --- |
| Run 8 vs untrained (key numbers) | Econ | **+0.035** (+0.007 to +0.059) | −0.048 (−0.100 to +0.003) |
| Run 8 vs run 4 | Econ | **+0.067** (+0.028 to +0.104) | **−0.120** (−0.172 to −0.075) |
| Run 8 vs untrained (key numbers) | Weather | 0.000 (−0.016 to +0.017) | +0.037 (−0.027 to +0.097) |

Bold: interval excludes 0.

- On validation econ, every run 8 forecast lies between 0.32 and 0.81, and 39% lie between 0.45 and 0.55 (untrained: 14%; run 4: 5%).
- During training, the spread of stated probabilities fell from 0.29 to 0.11 (dashed line in the figure below). Runs 4 to 6 held theirs near 0.3.
- The ordering partly survives the squeeze (AUC 0.71), because the prompt still carries the key numbers. RL just stops acting on them.

![Spread of stated probabilities during training, with the shuffled-outcome control](figures/spread_2x2.png)

## What it means

- **Runs 4 to 6 improved because of the outcomes**, not because training itself sharpens or reformats answers. The same setup with the signal removed lost 0.035 Brier instead of gaining 0.031.
- **Run 7's hedging looks like the no-signal case.** With real outcomes but no readable evidence (run 7), and with readable evidence but no real outcomes (run 8), RL does the same thing: it moves toward one number, the proper-score optimum for a forecaster whose answers carry no usable information.
- **It also rules out a format or length effect.** Run 8's answers keep their reasoning (median 90 tokens) and are all readable, like runs 4 to 6, so the difference between them is the forecasts themselves.
