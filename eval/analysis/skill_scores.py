"""Brier skill score of every method against the base rate, on both eval sets:
skill = 1 - Brier(method) / Brier(base rate). 0 = no better than each series' training yes-rate, 1 = perfect,
negative = worse than the base rate. (The base rate uses each econ series' training yes-rate, and each
weather series' yes-rate on held-out days before 2026.)

Usage: python -I eval/analysis/skill_scores.py <repo_dir> <data_dir> <out_dir>
data_dir holds markets.csv and the eval answer CSVs (renamed as in RUNS); out_dir holds gbm_val_forecasts.csv
from gbm_baseline.py, and gets skill_scores.csv.
"""
import os
import sys

import pandas as pd

REPO, DATA, OUT = sys.argv[1:4]
sys.path.insert(0, REPO)
from env import metrics as M                                                # noqa: E402
from env.evaluation import add_forecast_columns, load_markets, pick_eval_sets  # noqa: E402

RUNS = {  # column -> (file, condition, label)
    "untrained_plain": ("eval_answers_v1.csv", "base", "Untrained, plain prompt"),
    "untrained_keynum": ("eval_answers_run3.csv", "base", "Untrained, key numbers"),
    "run1": ("eval_answers_lr1e-5.csv", "trained", "Run 1: plain, std on"),
    "run2": ("eval_answers_run2.csv", "trained", "Run 2: plain, std on, 2 passes at lr 2e-5"),
    "run7": ("eval_answers_run7.csv", "trained", "Run 7: plain, std off"),
    "run3": ("eval_answers_run3.csv", "trained", "Run 3: key numbers, std on, seed 3407"),
    "run9": ("eval_answers_run9.csv", "trained", "Run 9: key numbers, std on, seed 1"),
    "run10": ("eval_answers_run10.csv", "trained", "Run 10: key numbers, std on, seed 2"),
    "run4": ("eval_answers_run4.csv", "trained", "Run 4: key numbers, std off, seed 3407"),
    "run5": ("eval_answers_run5.csv", "trained", "Run 5: key numbers, std off, seed 1"),
    "run6": ("eval_answers_run6.csv", "trained", "Run 6: key numbers, std off, seed 2"),
    "run8": ("eval_answers_run8.csv", "trained", "Run 8: key numbers, std off, shuffled outcomes"),
}
df = load_markets(os.path.join(DATA, "markets.csv"))
ev = pick_eval_sets(df, weather_events=60, seed=3407)
for col, (path, cond, _) in RUNS.items():
    ev = add_forecast_columns(ev, pd.read_csv(os.path.join(DATA, path)), cond, col)
gbm = pd.read_csv(os.path.join(OUT, "gbm_val_forecasts.csv"))[["market_ticker", "gbm", "gbm_series"]]
ev = ev.merge(gbm, on="market_ticker", how="left")
ev = ev[ev["market_prob"].notna()]
methods = {"market": "Market price", "gbm_series": "Gradient-boosted trees (key numbers + series)",
           "gbm": "Gradient-boosted trees (key numbers)", "logistic": "Logistic regression (key numbers)",
           "recent_hits": "Counting rule", "latest_rule": "Latest-result rule (0/1)", "always_0.5": "Always 0.5",
           **{f"{c}_mean4": label for c, (_, _, label) in RUNS.items()}}
rows = []
for es, g in ev.groupby("eval_set"):
    base = M.brier(g["base_rate"], g["outcome"])
    for col, label in methods.items():
        b = M.brier(g[col], g["outcome"])
        rows.append({"eval_set": es, "method": label, "brier": b, "base_rate_brier": base, "skill": 1 - b / base})
t = pd.DataFrame(rows)
t.round(4).to_csv(os.path.join(OUT, "skill_scores.csv"), index=False)
wide = t.pivot(index="method", columns="eval_set", values="skill").loc[list(methods.values())]
print(wide.round(3).to_string())
