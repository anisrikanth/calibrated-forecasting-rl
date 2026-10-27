# Evaluation results

Metric code lives in `env/metrics.py`. This folder holds results: copy each notebook's metrics CSV and reliability diagram here (e.g. `baseline_metrics_Llama-3.2-3B-Instruct.csv`, `reliability_Llama-3.2-3B-Instruct.png`) so the write-up can cite them.

Results so far:

- `baselines.md`: untrained 1B and 3B models against the market, base rates and always 0.5
- `rl_run1.md`: run 1, forecasts shift toward the base rate without better ordering
- `rl_run2_run3.md`: runs 2 and 3, the key-numbers prompt, overconfidence and the collapse onto a lookup rule
- `analysis/`: the scripts and tables behind them; `figures/`: the plots

Baselines compared in every run: the market price at question time, each series' base rate, always 0.5, and the model with and without context (first answer, and the average of 4).
