"""Announcement-level intervals for the prompt ablation and the weather comparisons cited in the paper
(the earlier tables resampled single releases or single city-days).

Usage: python -I eval/analysis/announcement_intervals_extra.py <repo_dir> <data_dir> <out_dir>
"""
import os
import sys

import pandas as pd

REPO, DATA, OUT = sys.argv[1:4]
sys.path.insert(0, REPO)
from env.evaluation import add_forecast_columns, load_markets, paired_bootstrap_groups, pick_eval_sets  # noqa: E402

df = load_markets(os.path.join(DATA, "markets.csv"))
ev = pick_eval_sets(df, weather_events=60, seed=3407)
abl = pd.read_csv(os.path.join(DATA, "prompt_ablation_answers.csv"))
for v in abl["condition"].unique():
    ev = add_forecast_columns(ev, abl, v, f"abl_{v}")
for c, (f, cond) in {"untrained_plain": ("eval_answers_v1.csv", "base"), "untrained_keynum": ("eval_answers_run3.csv", "base"),
                     "run1": ("eval_answers_lr1e-5.csv", "trained"),
                     **{f"run{i}": (f"eval_answers_run{i}.csv", "trained") for i in (3, 4, 5, 6, 9, 10)}}.items():
    ev = add_forecast_columns(ev, pd.read_csv(os.path.join(DATA, f)), cond, c)
ev = ev[ev["market_prob"].notna()]
econ = ev[ev["eval_set"] == "val_econ"].reset_index(drop=True)
wx = ev[ev["eval_set"] == "heldout_weather"].reset_index(drop=True)
m4 = lambda xs: [f"{x}_mean4" for x in xs]  # noqa: E731
rows = []
for ref in ("plain", "full"):
    for v in ("plain", "rule", "rule_latest", "rule_latest_hits", "full", "rule_hits", "estimate_first"):
        if v != ref:
            t = paired_bootstrap_groups(econ, [f"abl_{ref}_mean4"], [f"abl_{v}_mean4"], n_boot=4000, seed=0, cluster="announcement")
            rows.append(t.assign(eval_set="val_econ", comparison=f"{v} vs {ref}"))
# each prompt line's own step, as in the ablation write-up
for a, b in (("rule", "rule_latest"), ("rule_latest", "rule_latest_hits")):
    t = paired_bootstrap_groups(econ, [f"abl_{a}_mean4"], [f"abl_{b}_mean4"], n_boot=4000, seed=0, cluster="announcement")
    rows.append(t.assign(eval_set="val_econ", comparison=f"{b} vs {a}"))
for name, a, b in (("Run 1 vs untrained, plain", m4(["untrained_plain"]), m4(["run1"])),
                   ("Runs 4-6 vs untrained, key numbers", m4(["untrained_keynum"]), m4(["run4", "run5", "run6"])),
                   ("Runs 3, 9, 10 vs untrained, key numbers", m4(["untrained_keynum"]), m4(["run3", "run9", "run10"]))):
    t = paired_bootstrap_groups(wx, a, b, n_boot=4000, seed=0, cluster="announcement")
    rows.append(t.assign(eval_set="heldout_weather", comparison=name))
out = pd.concat(rows, ignore_index=True)[["eval_set", "comparison", "metric", "a", "b", "change", "ci_low", "ci_high"]]
out.round(4).to_csv(os.path.join(OUT, "announcement_intervals_extra.csv"), index=False)
pd.set_option("display.width", 200)
print(out[out.metric.isin(["brier", "auc"])].round(3).to_string(index=False))
print("weather dates:", wx["announcement"].nunique(), "of", wx["event_ticker"].nunique(), "city-days")
