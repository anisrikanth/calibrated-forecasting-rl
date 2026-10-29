"""Runs 9 and 10: two more training seeds of run 3 (key numbers, GRPO std scaling on). Does the
collapse onto the latest-result rule replicate, and how do the three std-on seeds compare with the
three std-off seeds (runs 4-6)?

Usage: python -I eval/analysis/analyze_runs9_10.py <repo_dir> <data_dir> <out_dir>
data_dir holds markets.csv and the eval/train answer CSVs downloaded from Kaggle, renamed as in RUNS.
"""
import json
import sys

import numpy as np
import pandas as pd

REPO, DATA, OUT = sys.argv[1:4]
sys.path.insert(0, REPO)
from env import metrics as M                                                        # noqa: E402
from env.baselines import latest_rule_column                                        # noqa: E402
from env.evaluation import (add_forecast_columns, load_markets, paired_bootstrap_groups,  # noqa: E402
                            pick_eval_sets)

B = 4000
RUNS = {  # column -> (answer file, condition)
    "untrained": ("eval_answers_run3.csv", "base"),
    "run3": ("eval_answers_run3.csv", "trained"), "run9": ("eval_answers_run9.csv", "trained"),
    "run10": ("eval_answers_run10.csv", "trained"),
    "run4": ("eval_answers_run4.csv", "trained"), "run5": ("eval_answers_run5.csv", "trained"),
    "run6": ("eval_answers_run6.csv", "trained"),
}
STD_ON, STD_OFF = ["run3", "run9", "run10"], ["run4", "run5", "run6"]
df = load_markets(f"{DATA}/markets.csv")
ev = pick_eval_sets(df, weather_events=60, seed=3407)
answers = {}
for col, (path, cond) in RUNS.items():
    a = pd.read_csv(f"{DATA}/{path}")
    answers[col] = a[a["condition"] == cond]
    ev = add_forecast_columns(ev, a, cond, col)
ev = ev[ev["market_prob"].notna()].reset_index(drop=True)
ev["has_rule"] = latest_rule_column(ev).notna()
g = ev[ev["eval_set"] == "val_econ"].reset_index(drop=True)
m4 = {c: f"{c}_mean4" for c in RUNS}

# 1. Scores
rows = []
for es, h in ev.groupby("eval_set"):
    for name, col in [("Market", "market"), ("Logistic regression", "logistic"), ("Latest-result rule", "latest_rule"),
                      ("Base rate", "base_rate")] + [(c, m4[c]) for c in RUNS]:
        s = M.summarize(h[col], h["outcome"])
        rows.append({"eval_set": es, "method": name, **{k: s[k] for k in ("brier", "log_loss", "ece", "auc", "mean_p")},
                     "sd_p": float(h[col].std())})
scores = pd.DataFrame(rows)


# 2. Collapse checks (validation econ). answers_equal_to_rule: share of single answers, on the questions
# with a numeric threshold, that are exactly the latest-result rule's 0 or 1. same_side_as_rule: share of
# those questions where the 4-answer average is on the rule's side of 0.5 (an average of exactly 0.5
# counts as YES).
def first_collapse_step(path):
    p = pd.to_numeric(pd.read_csv(path)["prob"], errors="coerce").to_numpy()
    for end in range(512, len(p) + 1, 64):
        w = p[end - 512:end]
        w = w[~np.isnan(w)]
        if ((w <= 0.05) | (w >= 0.95)).mean() >= 0.9:
            return end / 16
    return None


rule_q = g[g["has_rule"]]
collapse = []
for c in STD_ON + STD_OFF + ["untrained"]:
    a = answers[c][answers[c]["market_ticker"].isin(g["market_ticker"])]
    same4 = a.groupby("market_ticker")["prob"].nunique().eq(1).mean()
    v = rule_q[m4[c]]
    ar = a.merge(rule_q[["market_ticker", "latest_rule"]], on="market_ticker")   # answers on threshold questions
    train_file = {"untrained": None}.get(c, f"{DATA}/train_answers_{c}.csv")
    collapse.append({"model": c, "answers_exactly_0_or_1": float(a["prob"].isin([0.0, 1.0]).mean()),
                     "all_4_answers_identical": float(same4),
                     "answers_equal_to_rule": float((ar["prob"] == ar["latest_rule"]).mean()),
                     "same_side_as_rule": float(((v >= 0.5) == (rule_q["latest_rule"] >= 0.5)).mean()),
                     "mean_abs_diff_from_rule": float((v - rule_q["latest_rule"]).abs().mean()),
                     "median_tokens": float(a["n_tokens"].median()),
                     "first_step_90pct_near_certain": first_collapse_step(train_file) if train_file else None})
collapse = pd.DataFrame(collapse)

# 3. Where run 10 leaves the rule
d = rule_q[(rule_q[m4["run10"]] >= 0.5) != (rule_q["latest_rule"] >= 0.5)]
run10_off_rule = {"questions": int(len(d)), "by_series": d["series_ticker"].value_counts().to_dict(),
                  "run10_side_correct": float(((d[m4["run10"]] >= 0.5) == d["outcome"]).mean()),
                  "rule_side_correct": float(((d["latest_rule"] >= 0.5) == d["outcome"]).mean()),
                  "share_with_mixed_answers": float(((d[m4["run10"]] > 0) & (d[m4["run10"]] < 1)).mean())}
by_series = g.groupby("series_ticker").apply(lambda h: pd.Series({
    "n": len(h), "Latest-result rule": M.brier(h["latest_rule"], h["outcome"]),
    **{c: M.brier(h[m4[c]], h["outcome"]) for c in STD_ON}}), include_groups=False)

# 4. Paired comparisons, groups averaged within each draw
COMPARE = [("Std on (runs 3, 9, 10) vs untrained, key numbers", ["untrained"], STD_ON),
           ("Std on (runs 3, 9, 10) vs latest-result rule", ["latest_rule"], STD_ON),
           ("Std off (runs 4-6) vs std on (runs 3, 9, 10)", STD_ON, STD_OFF),
           ("Run 9 vs untrained, key numbers", ["untrained"], ["run9"]),
           ("Run 10 vs untrained, key numbers", ["untrained"], ["run10"]),
           ("Run 9 vs latest-result rule", ["latest_rule"], ["run9"]),
           ("Run 10 vs latest-result rule", ["latest_rule"], ["run10"]),
           ("Run 10 vs run 3", ["run3"], ["run10"])]
boots = []
for es, h in ev.groupby("eval_set"):
    for name, a_cols, b_cols in COMPARE:
        cols = lambda xs: [m4.get(x, x) for x in xs]  # noqa: E731
        t = paired_bootstrap_groups(h.reset_index(drop=True), cols(a_cols), cols(b_cols), n_boot=B, seed=0)
        boots.append(t.assign(eval_set=es, comparison=name))
boot = pd.concat(boots, ignore_index=True)[["eval_set", "comparison", "metric", "a", "b", "change", "ci_low", "ci_high",
                                            "share_better", "b_min", "b_max"]]

pd.set_option("display.width", 240)
print(scores.round(4).to_string(index=False)); print()
print(collapse.round(4).to_string(index=False)); print()
print(json.dumps(run10_off_rule, indent=1)); print()
print(by_series.round(3).to_string()); print()
print(boot[boot.metric.isin(["brier", "log_loss", "ece", "auc"])].round(4).to_string(index=False))
scores.round(4).to_csv(f"{OUT}/runs9_10_scores.csv", index=False)
collapse.round(4).to_csv(f"{OUT}/runs9_10_collapse.csv", index=False)
boot.round(4).to_csv(f"{OUT}/runs9_10_paired_bootstrap.csv", index=False)
by_series.round(4).to_csv(f"{OUT}/runs9_10_per_series.csv")
json.dump(run10_off_rule, open(f"{OUT}/run10_off_rule.json", "w"), indent=1)
