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
- **What happened:** the collapse started inside the first pass. The share of answers at or below 0.05 or at or above 0.95 passed 50% around step 300. The average forecast then swung to 0.04 at step 384 (almost every answer "no") before settling into a two-sided 0/1 pattern: 94% near-certain at the end, none between 0.45 and 0.55. On new econ questions, 72% of forecasts were near-certain, calibration error rose from 0.12 to 0.30, and the Brier score got worse than the untrained model's (0.320 vs 0.272; payrolls 0.55). Underneath, the ordering did improve (AUC 0.52 to 0.69, interval +0.10 to +0.25), and after recalibration it beats the untrained model (0.238 vs 0.259). Late in training many groups had a reward spread near 0: all 4 answers gave the same extreme number, so there was nothing left to compare.
- **Why it likely happens:** GRPO divides each answer's advantage by the spread of rewards in its group. With a proper-scoring reward this favors confident answers and feeds on itself ([Bereket & Leskovec 2025](https://arxiv.org/pdf/2508.11800)); [Turtel et al. 2025](https://arxiv.org/html/2505.17989v4) saw 39% extreme forecasts with plain GRPO. TRL 0.22.2 does this by default (`scale_rewards="group"`).
- **Fix:** turn the scaling off (`scale_rewards="none"`, subtract the group mean only).
- **Effect (run 4, one seed):** on run 3's settings, which collapsed, turning it off kept 84% of econ forecasts away from 0 and 1 (run 3: 1%), kept calibration error at 0.06 (run 3: 0.22), and improved the Brier score to 0.169 (run 3: 0.223) and AUC to 0.83 (run 3: 0.77). See `eval/rl_run4.md`.

## 4. In-training numbers overstate what was learned

- **Found:** RL run 2, comparing training and held-out numbers.
- **What happened:** the yes-minus-no gap on training answers rose from +0.03 (first 512) to +0.25 (last 512 of the first pass) and +0.38 (end of the second pass), while the training Brier score stayed flat (0.300 vs 0.298). On held-out econ questions the ordering did improve (AUC 0.52 to 0.69), but the Brier score got worse.
- **Why:** the gap grows when forecasts are pushed toward 0 and 1, whether or not the ordering improves (run 3 shows the gap jumping from 0.22 to 0.54 with no AUC change). In a second pass, every question has also been rewarded with its outcome once, so part of any further gain can be recall; Turtel et al. (2025) train in a single pass for this reason.
- **Fix:** train in a single pass, track AUC rather than the gap, and judge every run on held-out questions.

## 5. Collapse onto a copy-the-latest-result rule

- **Found:** RL run 3 (3B, Brier reward, key numbers in the prompt, one pass, learning rate 1e-5), Oct 27, 2026.
- **What happened:** most forecasts were at 0 or 1 within about 100 of 347 steps; by about step 240 answers had shrunk from ~110 tokens of reasoning to ~7, and all 4 answers in each group were identical (reward spread 0). On the validation econ questions, the trained model's Brier score matches the hard rule "1 if the latest result already satisfies the question, else 0" to within 0.0001 for every threshold series (Fed 0.000, unemployment 0.060, core CPI 0.095, payrolls 0.541). It answers exactly 0.00 or 1.00 and sits on the rule's side for 99.9% of threshold questions (`eval/rl_run2_run3.md`). Like that rule, it has zero ladder violations and weather brackets that sum to 1. AUC barely moved (0.757 to 0.771, interval includes 0), while calibration error rose from 0.06 to 0.22 and log loss nearly doubled.
- **Why:** the key-numbers block made one cue easy to read ("latest result: 0.4 above the threshold"). Answers that follow it score better on average, and GRPO's std scaling (entry 3) pushes them to certainty. The certainty itself is not what the reward wants: on the training questions the rule stated at its hit rates (84% / 15%) earns a higher average Brier reward (0.870) than the 0/1 version (0.847). The rule is right for slow series and badly wrong for noisy ones (payrolls).
- **How to catch it:** compare the trained model with simple rules built from the prompt (here, persistence of the latest result). The gap metric rose sharply (0.22 to 0.54) without any AUC gain, so gap rewards sharpening and is not a discrimination measure; use AUC.
- **Fix:** std scaling off (entry 3). Run 4 did not collapse: its forecasts agree with the latest-result rule on 93% of threshold questions rather than 100%, and its AUC beats the rule's by 0.054 (interval +0.010 to +0.101).

