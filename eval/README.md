# Evaluation results

Metric code lives in `env/metrics.py`. This folder holds results: copy each notebook's metrics CSV and reliability diagram here (e.g. `baseline_metrics_Llama-3.2-3B-Instruct.csv`, `reliability_Llama-3.2-3B-Instruct.png`) so the write-up can cite them.

Results so far:

- `baselines.md`: untrained 1B and 3B models against the market, base rates and always 0.5
- `rl_run1.md`: run 1, forecasts shift toward the base rate without better ordering
- `rl_run2_run3.md`: runs 2 and 3, the key-numbers prompt, overconfidence and the collapse onto a lookup rule
- `rl_run4.md`: run 4, GRPO without std scaling: no collapse, better ordering at unchanged calibration
- `rl_runs5to7.md`: three seeds of the main setup and the full 2 × 2 (prompt × std scaling)
- `rl_run8.md`: shuffled-outcome control: without a real signal, RL makes the model worse and hedges
- `prompt_ablation.md`: which lines of the key numbers raise the untrained model's discrimination
- `rl_runs9_10.md`: two more seeds of run 3: with std scaling on, all three seeds collapse onto the latest-result rule
- `test_plan.md`: the comparisons fixed in advance for the one evaluation on the test split
- `answers/`: each model's validation answers (probability and a fingerprint of the text), used to check saved adapters load correctly
- `analysis/`: the scripts and tables behind them; `figures/`: the plots

Baselines compared in every run: the market price at question time, each series' base rate, always 0.5, and the model with and without context (first answer, and the average of 4).
