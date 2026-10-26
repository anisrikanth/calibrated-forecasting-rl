# Reward hacks and parser issues

One entry per issue: what happened, how it was spotted, the fix, and the before/after effect. Parser and format issues count too, because they change what the reward measures.

## 1. Fractions before the decimal (parser bug)

- **Found:** Phase 0 smoke test, Oct 25, 2026, while reading printed samples.
- **What happened:** the model wrote `Probability: 8/17 = 0.47`. The parser took the first number after `Probability:` ("8"), rejected it as out of range, and the answer got reward 0 plus the format penalty, even though 0.47 was a reasonable forecast.
- **Fix:** parse the last `Probability:` line and take the last number on it, so `8/17 = 0.47` gives 0.47. Percentages are still rejected because the prompt asks for `0.XX`.
- **Effect:** not measured (found after 20 steps). Watch for a drop in format failures in the next run.
