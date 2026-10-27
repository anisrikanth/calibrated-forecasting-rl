# Environment

| File | What it does |
| --- | --- |
| `prompt.py` | Builds the chat prompt from a row of `data/processed/markets.csv`: question time (US Eastern), the market's resolution rule as the question, and optionally the series' recent results. Never includes the market price or outcome. `chat_template_kwargs` passes the question's date to Llama's chat template, which otherwise writes a fixed 2024 "Today Date" into the system header |
| `features.py` | The optional **key numbers** block (run 3): spells out the comparison between the question's threshold and the recent results already in the context (latest result vs threshold, change from the result before, how many recent results would have resolved YES, average and range). Uses nothing beyond the question and the leak-audited context. Also gives `recent_hit_rate`, used as a rule-based baseline |
| `parser.py` | Reads the final `Probability: 0.XX` line. Takes the last number on the last such line (so `8/17 = 0.47` gives 0.47); rejects percentages and values outside [0, 1] |
| `rewards.py` | Brier reward (1 − squared error), log-score reward (clipped to [0.01, 0.99]), format reward (+0.5 / −1.0), and wrappers in the shape TRL's `GRPOTrainer` expects |
| `evaluation.py` | Shared by the baseline and training notebooks: picks the evaluation questions (validation econ, whole held-out weather days, base rates), renders prompts, collects sampled answers, and builds the score, per-series, consistency and leak-probe tables. Adds the `recent_hits` rule baseline, a shift-matched baseline (the untrained model moved to the trained model's average), and a paired bootstrap that resamples whole events for 95% intervals |
| `training.py` | GRPO training records (prompts rendered with each question's date) and `MonitoredScoreReward`, a Brier or log-score reward that records every stated probability and prints the format rate, probability spread, share near 0.5, and the yes-minus-no **gap** that exposes hedging |
| `metrics.py` | Brier score, log loss, expected calibration error, AUC, the yes-minus-no gap, the Murphy decomposition (reliability, resolution, uncertainty), reliability tables, ladder violations, and bracket-sum error. Definitions are written out in the module docstring |

All of it is covered by `tests/test_env.py`, `tests/test_features.py` and `tests/test_evaluation_training.py`, including a check that both rewards are proper (stating your true belief maximizes expected reward).
