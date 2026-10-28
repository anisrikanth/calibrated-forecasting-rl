"""Runs 1-7 together: the 2 x 2 (plain vs key-numbers prompt, GRPO std scaling on vs off), the three
seeds of the main setup (runs 4-6), and run 7's hedging.

Usage: python -I eval/analysis/analyze_runs1to7.py <repo_dir> <data_dir> <out_dir>
data_dir holds markets.csv and the eval answer CSVs downloaded from Kaggle, renamed as in RUNS below.
"""
import json
import sys

import numpy as np
import pandas as pd

REPO, DATA, OUT = sys.argv[1:4]
sys.path.insert(0, REPO)
from env import metrics as M                                                       # noqa: E402
from env.evaluation import (BOOTSTRAP_STATS, add_forecast_columns, load_markets, paired_bootstrap,  # noqa: E402
                            pick_eval_sets)

B = 4000
# column -> (answer file, condition in that file, description)
RUNS = {
    "plain_untrained": ("eval_answers_v1.csv", "base", "Untrained, plain prompt"),
    "keynum_untrained": ("eval_answers_run3.csv", "base", "Untrained, key numbers"),
    "run1": ("eval_answers_lr1e-5.csv", None, "Run 1: plain, std on"),
    "run3": ("eval_answers_run3.csv", "trained", "Run 3: key numbers, std on"),
    "run4": ("eval_answers_run4.csv", "trained", "Run 4: key numbers, std off (seed 3407)"),
    "run5": ("eval_answers_run5.csv", "trained", "Run 5: key numbers, std off (seed 1)"),
    "run6": ("eval_answers_run6.csv", "trained", "Run 6: key numbers, std off (seed 2)"),
    "run7": ("eval_answers_run7.csv", "trained", "Run 7: plain, std off"),
}
df = load_markets(f"{DATA}/markets.csv")
ev = pick_eval_sets(df, weather_events=60, seed=3407)
for col, (path, cond, _) in RUNS.items():
    a = pd.read_csv(f"{DATA}/{path}")
    if cond is None:
        a, cond = a.assign(condition="trained"), "trained"
    assert set(a["market_ticker"]) == set(ev["market_ticker"]), f"eval set differs in {path}"
    ev = add_forecast_columns(ev, a, cond, col)
ev = ev[ev["market_prob"].notna()].reset_index(drop=True)

# 1. Every method on both eval sets
methods = {"market": "Market", "logistic": "Logistic regression", "recent_hits": "Counting rule",
           "latest_rule": "Latest-result rule", "base_rate": "Base rate",
           **{f"{c}_mean4": d for c, (_, _, d) in RUNS.items()}}
rows = []
for es, g in ev.groupby("eval_set"):
    for m, name in methods.items():
        s = M.summarize(g[m], g["outcome"])
        rows.append({"eval_set": es, "method": name, **{k: s[k] for k in ("brier", "log_loss", "ece", "auc", "mean_p")},
                     "sd_p": float(g[m].std()), "share_extreme": float(((g[m] <= 0.05) | (g[m] >= 0.95)).mean()),
                     "share_near_half": float(((g[m] >= 0.45) & (g[m] <= 0.55)).mean())})
scores = pd.DataFrame(rows)

# 2. Paired comparisons: each trained run vs the untrained model with the same prompt
PAIRS = [("plain_untrained", "run1"), ("plain_untrained", "run7"), ("keynum_untrained", "run3"),
         ("keynum_untrained", "run4"), ("keynum_untrained", "run5"), ("keynum_untrained", "run6"),
         ("run3", "run4"), ("run1", "run7"), ("logistic", "run4"), ("logistic", "run5"), ("logistic", "run6")]
boots = []
for es, g in ev.groupby("eval_set"):
    for a, b in PAIRS:
        ca = a if a in ("logistic",) else f"{a}_mean4"
        t = paired_bootstrap(g, ca, f"{b}_mean4", n_boot=B, seed=0).rename(columns={ca: "a", f"{b}_mean4": "b"})
        t.insert(0, "comparison", f"{b} vs {a}")
        t.insert(0, "eval_set", es)
        boots.append(t)
boot = pd.concat(boots, ignore_index=True)


# 3. Three seeds of the main setup: mean over seeds of (trained - reference), with an event-level interval
def seed_average(g, ref_col, seed_cols, n_boot=B, seed=0):
    rng = np.random.default_rng(seed)
    ev_ids = g["event_ticker"].to_numpy()
    events = pd.unique(ev_ids)
    idx = {e: np.flatnonzero(ev_ids == e) for e in events}
    y = g["outcome"].to_numpy()
    ref = g[ref_col].to_numpy(float)
    seeds = [g[c].to_numpy(float) for c in seed_cols]
    out = []
    for k, f in BOOTSTRAP_STATS.items():
        point = np.mean([f(s, y) - f(ref, y) for s in seeds])
        per_seed = [f(s, y) for s in seeds]
        draws = np.empty(n_boot)
        for i in range(n_boot):
            pick = np.concatenate([idx[e] for e in rng.choice(events, size=len(events), replace=True)])
            draws[i] = np.mean([f(s[pick], y[pick]) - f(ref[pick], y[pick]) for s in seeds])
        lo, hi = np.nanpercentile(draws, [2.5, 97.5])
        out.append({"metric": k, "reference": f(ref, y), "seed_mean": float(np.mean(per_seed)),
                    "seed_sd": float(np.std(per_seed, ddof=1)), "seed_min": float(np.min(per_seed)),
                    "seed_max": float(np.max(per_seed)), "change": point, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(out)


seeds = ["run4_mean4", "run5_mean4", "run6_mean4"]
pooled = []
for es, g in ev.groupby("eval_set"):
    for ref, label in (("keynum_untrained_mean4", "vs untrained (key numbers)"), ("logistic", "vs logistic regression"),
                       ("run3_mean4", "vs run 3 (std on)")):
        t = seed_average(g.reset_index(drop=True), ref, seeds)
        t.insert(0, "comparison", f"Runs 4-6 (3 seeds) {label}")
        t.insert(0, "eval_set", es)
        pooled.append(t)
pooled = pd.concat(pooled, ignore_index=True)

# 4. Run 7 hedging, and per-series for the seeds
e = ev[ev["eval_set"] == "val_econ"]
checks = {}
for c in ("plain_untrained", "run1", "run7", "keynum_untrained", "run4", "run5", "run6", "run3"):
    v = e[f"{c}_mean4"]
    checks[c] = {"sd": float(v.std()), "near_half": float(((v >= 0.45) & (v <= 0.55)).mean()),
                 "mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}
series = e.groupby("series_ticker").apply(lambda g: pd.Series({
    "releases": g["event_ticker"].nunique(),
    **{RUNS[c][2]: M.brier(g[f"{c}_mean4"], g["outcome"]) for c in ("keynum_untrained", "run4", "run5", "run6", "run7")},
    "Logistic regression": M.brier(g["logistic"], g["outcome"])}), include_groups=False)

pd.set_option("display.width", 240)
print(scores.round(4).to_string(index=False)); print()
print(boot[boot.metric.isin(["brier", "auc", "ece", "mean_p"])].round(4).to_string(index=False)); print()
print(pooled[pooled.metric.isin(["brier", "auc", "ece", "log_loss"])].round(4).to_string(index=False)); print()
print(json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in checks.items()}, indent=1)); print()
print(series.round(3).to_string())

scores.round(4).to_csv(f"{OUT}/runs1to7_scores.csv", index=False)
boot.round(4).to_csv(f"{OUT}/runs1to7_paired_bootstrap.csv", index=False)
pooled.round(4).to_csv(f"{OUT}/runs4to6_seed_average.csv", index=False)
series.round(4).to_csv(f"{OUT}/runs4to7_per_series.csv")
json.dump(checks, open(f"{OUT}/runs1to7_spread.json", "w"), indent=1)
ev.to_csv(f"{OUT}/runs1to7_eval_forecasts.csv", index=False)
