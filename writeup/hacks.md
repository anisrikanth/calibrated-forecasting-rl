# Reward hacks and parser issues

One entry per issue: what happened, how it was spotted, the fix, and the before/after effect. Parser and format issues count too, because they change what the reward measures.

## 1. Fractions before the decimal (parser bug)

- **Found:** Phase 0 smoke test, Oct 25, 2026, while reading printed samples.
- **What happened:** the model wrote `Probability: 8/17 = 0.47`. The parser took the first number after `Probability:` ("8"), rejected it as out of range, and the answer got reward 0 plus the format penalty, even though 0.47 was a reasonable forecast.
- **Fix:** parse the last `Probability:` line and take the last number on it, so `8/17 = 0.47` gives 0.47. Percentages are still rejected because the prompt asks for `0.XX`.
- **Effect:** not measured (found after 20 steps). Watch for a drop in format failures in the next run.

## 2. A uniform downward shift that looks like a held-out gain

- **Found:** RL run 1 (3B, Brier reward, one pass), Oct 27, 2026, in the paired bootstrap analysis (`eval/rl_run1.md`).
- **What happened:** after training, every forecast was about 0.06–0.07 lower on both eval sets. On held-out weather, where only 1 bracket in 6 resolves yes, that alone improved the Brier score significantly (0.286 → 0.254), which reads like generalization. But weather AUC didn't improve, and shifting the untrained model's forecasts down by the same amount scores even better (0.242).
- **Why it happens:** each GRPO group answers the same question, so the reward favors whichever answers sit closer to that question's outcome. When the model finds nothing in the prompt that predicts the outcome, the easiest consistent improvement is moving its overall level toward the training base rate and pulling in overconfident extremes.
- **Fix (in evaluation, not training):** report AUC (discrimination) and a shift-matched baseline next to every Brier score, and judge runs on discrimination.
- **Effect:** stops a calibration-in-the-large adjustment from being reported as forecasting skill.
