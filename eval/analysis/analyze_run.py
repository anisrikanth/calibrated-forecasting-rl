"""Paired, cluster-bootstrapped comparison of untrained vs trained 3B, plus training curves.

Usage: python -I analyze.py <repo_dir> <data_dir> <out_dir>
"""
import json
import sys

import numpy as np
import pandas as pd

REPO, DATA, OUT = sys.argv[1:4]
sys.path.insert(0, REPO)
from env import metrics as M                                           # noqa: E402
from env.evaluation import add_forecast_columns, load_markets, pick_eval_sets  # noqa: E402
from env.training import window_stats                                   # noqa: E402

rng = np.random.default_rng(0)
B = 4000

df = load_markets(f"{DATA}/markets.csv")
ev = pick_eval_sets(df, weather_events=60, seed=3407)
v1 = pd.read_csv(f"{DATA}/eval_answers_v1.csv")
full = pd.read_csv(f"{DATA}/eval_answers_lr1e-5.csv")
assert set(v1["market_ticker"]) == set(ev["market_ticker"]) == set(full["market_ticker"]), "eval sets differ"

ev = add_forecast_columns(ev, v1, "base", "base")
ev = add_forecast_columns(ev, full.assign(condition="trained"), "trained", "trained")
ev = ev[ev["market_prob"].notna()].copy()


def auc(p, y):
    """Probability that a random yes-question gets a higher forecast than a random no-question (ties count half)."""
    p, y = np.asarray(p, float), np.asarray(y, int)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    allp = np.concatenate([pos, neg])
    # average ranks for ties
    s = pd.Series(allp).rank(method="average").to_numpy()
    return float((s[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def gap(p, y):
    p, y = np.asarray(p, float), np.asarray(y, int)
    return float(p[y == 1].mean() - p[y == 0].mean())


STATS = {"brier": M.brier, "log_loss": M.log_loss, "ece": M.ece, "auc": auc, "gap": gap,
         "mean_p": lambda p, y: float(np.mean(p))}


def paired_bootstrap(g, a, b):
    """Resample whole events (a weather day, or one release's ladder) with replacement; report the
    difference b − a with a 95% interval. Questions in one event share an outcome driver, so they
    move together; resampling questions one by one would understate the uncertainty."""
    events = g["event_ticker"].unique()
    idx = {e: np.flatnonzero(g["event_ticker"].to_numpy() == e) for e in events}
    y = g["outcome"].to_numpy()
    pa, pb = g[a].to_numpy(), g[b].to_numpy()
    point = {k: (f(pa, y), f(pb, y)) for k, f in STATS.items()}
    draws = {k: [] for k in STATS}
    for _ in range(B):
        pick = np.concatenate([idx[e] for e in rng.choice(events, size=len(events), replace=True)])
        for k, f in STATS.items():
            draws[k].append(f(pb[pick], y[pick]) - f(pa[pick], y[pick]))
    rows = []
    for k in STATS:
        d = np.array(draws[k]); d = d[~np.isnan(d)]
        lo, hi = np.percentile(d, [2.5, 97.5])
        rows.append({"metric": k, "untrained": point[k][0], "trained": point[k][1],
                     "change": point[k][1] - point[k][0], "ci_low": lo, "ci_high": hi,
                     "share_of_draws_improved": float(np.mean(d < 0) if k not in ("auc", "gap") else np.mean(d > 0))})
    return pd.DataFrame(rows)


results = {}
for eval_set, g in ev.groupby("eval_set"):
    t = paired_bootstrap(g.reset_index(drop=True), "base_mean4", "trained_mean4")
    t.insert(0, "eval_set", eval_set)
    t["events"] = g["event_ticker"].nunique()
    t["questions"] = len(g)
    results[eval_set] = t
table = pd.concat(results.values(), ignore_index=True)

# Recomputed point estimates must match the notebook's printed numbers
check = {(r.eval_set, r.metric): (round(r.untrained, 4), round(r.trained, 4)) for r in table.itertuples()}
print("Brier check (notebook: val 0.2720 -> 0.2669, weather 0.2861 -> 0.2541):",
      check[("val_econ", "brier")], check[("heldout_weather", "brier")])

# How much of the weather gain is "less extreme / lower" vs ordering? Shift-only comparison:
w = ev[ev["eval_set"] == "heldout_weather"]
shift = w["trained_mean4"].mean() - w["base_mean4"].mean()
shifted_base = np.clip(w["base_mean4"] + shift, 0, 1)
explain = {
    "weather_mean_forecast_untrained": float(w["base_mean4"].mean()),
    "weather_mean_forecast_trained": float(w["trained_mean4"].mean()),
    "weather_yes_rate": float(w["outcome"].mean()),
    "weather_brier_untrained": M.brier(w["base_mean4"], w["outcome"]),
    "weather_brier_untrained_shifted_to_trained_mean": M.brier(shifted_base, w["outcome"]),
    "weather_brier_trained": M.brier(w["trained_mean4"], w["outcome"]),
}

# Per-series change on val econ (point estimates; too few events per series for intervals)
e = ev[ev["eval_set"] == "val_econ"]
series = e.groupby("series_ticker").apply(lambda g: pd.Series({
    "events": g["event_ticker"].nunique(), "questions": len(g),
    "untrained": M.brier(g["base_mean4"], g["outcome"]), "trained": M.brier(g["trained_mean4"], g["outcome"]),
    "market": M.brier(g["market"], g["outcome"]), "base_rate": M.brier(g["base_rate"], g["outcome"]),
}), include_groups=False)
series["change"] = series["trained"] - series["untrained"]

# Training curve: rolling stats over training answers, in order
tr = pd.read_csv(f"{DATA}/train_answers_lr1e-5.csv")
tr["prob"] = pd.to_numeric(tr["prob"], errors="coerce")
win = 512
curve = []
for end in range(win, len(tr) + 1, 64):
    s = tr.iloc[end - win:end]
    st = window_stats([None if np.isnan(x) else x for x in s["prob"]], s["outcome"].tolist())
    curve.append({"answers": end, "step": end / 16, **st})
curve = pd.DataFrame(curve)

pd.set_option("display.width", 200)
fmt = table.copy()
for c in ("untrained", "trained", "change", "ci_low", "ci_high"):
    fmt[c] = fmt[c].round(4)
fmt["share_of_draws_improved"] = fmt["share_of_draws_improved"].round(3)
print(fmt.to_string(index=False))
print(json.dumps({k: round(v, 4) for k, v in explain.items()}, indent=1))
print(series.round(4).to_string())
print(curve.iloc[[0, len(curve) // 4, len(curve) // 2, 3 * len(curve) // 4, -1]].round(3).to_string(index=False))

table.to_csv(f"{OUT}/paired_bootstrap.csv", index=False)
series.to_csv(f"{OUT}/per_series.csv")
curve.to_csv(f"{OUT}/training_curve.csv", index=False)
ev.to_csv(f"{OUT}/eval_forecasts.csv", index=False)
json.dump(explain, open(f"{OUT}/weather_shift.json", "w"), indent=1)
