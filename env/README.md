# Environment

| File | What it does |
| --- | --- |
| `prompt.py` | Builds the chat prompt from a row of `data/processed/markets.csv`: question time (US Eastern), the market's resolution rule as the question, and optionally the series' recent results. Never includes the market price or outcome. `chat_template_kwargs` passes the question's date to Llama's chat template, which otherwise writes a fixed 2024 "Today Date" into the system header |
| `parser.py` | Reads the final `Probability: 0.XX` line. Takes the last number on the last such line (so `8/17 = 0.47` gives 0.47); rejects percentages and values outside [0, 1] |
| `rewards.py` | Brier reward (1 − squared error), log-score reward (clipped to [0.01, 0.99]), format reward (+0.5 / −1.0), and wrappers in the shape TRL's `GRPOTrainer` expects |
| `metrics.py` | Brier score, log loss, expected calibration error, the Murphy decomposition (reliability, resolution, uncertainty), reliability tables, ladder violations, and bracket-sum error. Definitions are written out in the module docstring |

All of it is covered by `tests/test_env.py`, including a check that both rewards are proper (stating your true belief maximizes expected reward).
