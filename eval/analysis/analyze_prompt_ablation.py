"""Prompt ablation (untrained 3B, validation econ): each prompt's Brier and AUC with 95% intervals,
a reproducibility check against the untrained answers from runs 1 and 3, and per-series AUC.

Usage: python -I eval/analysis/analyze_prompt_ablation.py <repo_dir> <data_dir> <out_dir>
data_dir holds markets.csv, prompt_ablation_answers.csv, eval_answers_v1.csv (run 1's untrained
answers) and eval_answers_run3.csv (run 3's untrained answers with the full key numbers).
"""
import json
import sys

import numpy as np
import pandas as pd

REPO, DATA, OUT = sys.argv[1:4]
sys.path.insert(0, REPO)
from env import metrics as M                                                        # noqa: E402
from env.evaluation import add_forecast_columns, load_markets, pick_eval_sets          # noqa: E402

VARIANTS = ["plain", "rule", "rule_latest", "rule_latest_hits", "full", "rule_hits", "estimate_first"]
df = load_markets(f"{DATA}/markets.csv")
ev = pick_eval_sets(df, weather_events=60, seed=3407)
ev = ev[(ev["eval_set"] == "val_econ") & ev["market_prob"].notna()].reset_index(drop=True)
ab = pd.read_csv(f"{DATA}/prompt_ablation_answers.csv")
for v in VARIANTS:
    ev = add_forecast_columns(ev, ab, v, v)
ev = add_forecast_columns(ev, pd.read_csv(f"{DATA}/eval_answers_v1.csv"), "base", "run1_plain")
ev = add_forecast_columns(ev, pd.read_csv(f"{DATA}/eval_answers_run3.csv"), "base", "run3_full")

def identical_share(variant, old_file):
    """Share of the ablation's answers whose text matches the earlier run's answer word for word."""
    now = ab[ab["condition"] == variant][["market_ticker", "sample", "text"]]
    before = pd.read_csv(f"{DATA}/{old_file}").query("condition == 'base'")[["market_ticker", "sample", "text"]]
    m = now.merge(before, on=["market_ticker", "sample"], suffixes=("_now", "_before"))
    return float((m["text_now"] == m["text_before"]).mean())


repro = {}
for v, old in (("plain", "run1_plain"), ("full", "run3_full")):
    a, b = ev[f"{v}_mean4"], ev[f"{old}_mean4"]
    repro[v] = {"brier_now": M.brier(a, ev["outcome"]), "brier_before": M.brier(b, ev["outcome"]),
                "auc_now": M.auc(a, ev["outcome"]), "auc_before": M.auc(b, ev["outcome"]),
                "corr_of_forecasts": float(np.corrcoef(a, b)[0, 1]),
                "share_identical_answers": identical_share(v, "eval_answers_v1.csv" if v == "plain" else "eval_answers_run3.csv")}

rng = np.random.default_rng(0)
events = ev["event_ticker"].unique()
idx = {e: np.flatnonzero(ev["event_ticker"].to_numpy() == e) for e in events}
y = ev["outcome"].to_numpy()
rows = []
cols = VARIANTS + ["logistic", "recent_hits", "base_rate"]
draws = {c: {"brier": [], "auc": []} for c in cols}
for _ in range(4000):
    pick = np.concatenate([idx[e] for e in rng.choice(events, size=len(events), replace=True)])
    for c in cols:
        p = ev[c if c in ("logistic", "recent_hits", "base_rate") else f"{c}_mean4"].to_numpy()[pick]
        draws[c]["brier"].append(M.brier(p, y[pick]))
        draws[c]["auc"].append(M.auc(p, y[pick]))
for c in cols:
    p = ev[c if c in ("logistic", "recent_hits", "base_rate") else f"{c}_mean4"]
    s = M.summarize(p, ev["outcome"])
    lo_b, hi_b = np.nanpercentile(draws[c]["brier"], [2.5, 97.5])
    lo_a, hi_a = np.nanpercentile(draws[c]["auc"], [2.5, 97.5])
    rows.append({"prompt": c, "brier": s["brier"], "brier_low": lo_b, "brier_high": hi_b, "auc": s["auc"],
                 "auc_low": lo_a, "auc_high": hi_a, "ece": s["ece"], "log_loss": s["log_loss"], "mean_p": s["mean_p"],
                 "readable": float(ab.loc[ab["condition"] == c, "prob"].notna().mean()) if c in VARIANTS else np.nan,
                 "median_tokens": float(ab.loc[ab["condition"] == c, "n_tokens"].median()) if c in VARIANTS else np.nan})
table = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in repro.items()}, indent=1))
print(table.round(4).to_string(index=False))
table.round(4).to_csv(f"{OUT}/prompt_ablation_scores.csv", index=False)
json.dump(repro, open(f"{OUT}/prompt_ablation_reproducibility.json", "w"), indent=1)
