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

## 3. Overconfidence collapse with more training

- **Found:** RL run 2 (3B, Brier reward, 2 passes, learning rate 2e-5), Oct 27, 2026.
- **What happened:** by the end of training the model answered almost only near 0 or near 1: across the last 512 training answers the spread of stated probabilities was 0.48 and none fell between 0.45 and 0.55 (11% did at the start). On new econ questions the Brier score got worse than the untrained model (0.320 vs 0.272) and calibration error rose from 0.12 to 0.30. Payrolls, the noisiest series, scored 0.55. Late in training many groups had a reward spread near 0: all 4 answers gave the same extreme number, so there was nothing left to compare.
- **Why it likely happens:** GRPO divides each answer's advantage by the spread of rewards in its group. With a proper-scoring reward this favors confident answers and feeds on itself ([Bereket & Leskovec 2025](https://arxiv.org/pdf/2508.11800)); [Turtel et al. 2025](https://arxiv.org/html/2505.17989v4) saw 39% extreme forecasts with plain GRPO. TRL 0.22.2 does this by default (`scale_rewards="group"`).
- **Fix to test:** run 4 repeats run 2 with `scale_rewards="none"` (subtract the group mean only).
- **Effect:** pending run 4.

## 4. The in-training gap stops meaning anything after one pass

- **Found:** RL run 2, comparing training and held-out numbers.
- **What happened:** the yes-minus-no gap on training answers rose from +0.03 (first 512) to +0.38 (last 512), which reads like the model learned to discriminate. But the training Brier score didn't improve (0.300 vs 0.298), and on held-out econ questions the Brier score got worse.
- **Why:** in the second pass every question had already been rewarded with its outcome in the first pass, so the model can partly recall answers instead of forecasting them. Turtel et al. (2025) train in a single pass for this reason.
- **Fix:** read training curves only within the first pass; judge every run on held-out questions.

