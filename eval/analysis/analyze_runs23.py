"""Runs 2 and 3: paired event-level bootstrap, lookup-rule check, recalibration test, training dynamics.

Usage: python -I eval/analysis/analyze_runs23.py <repo_dir> <data_dir> <out_dir>
(data_dir holds markets.csv and the eval/train answer CSVs downloaded from Kaggle; see the read_csv calls)
"""
import json
import sys

import numpy as np
import pandas as pd

REPO, DATA, OUT = sys.argv[1:4]
sys.path.insert(0, REPO)
from env import metrics as M                                                       # noqa: E402
from env.evaluation import add_forecast_columns, load_markets, paired_bootstrap, pick_eval_sets  # noqa: E402
from env.features import key_numbers, threshold_rule, unit_of                      # noqa: E402
from env.training import window_stats                                              # noqa: E402

B = 4000
df = load_markets(f"{DATA}/markets.csv")
ev = pick_eval_sets(df, weather_events=60, seed=3407)

v1 = pd.read_csv(f"{DATA}/eval_answers_v1.csv")
r1 = pd.read_csv(f"{DATA}/eval_answers_lr1e-5.csv")
r2 = pd.read_csv(f"{DATA}/eval_answers_run2.csv")
r3 = pd.read_csv(f"{DATA}/eval_answers_run3.csv")
for a in (v1, r1, r2, r3):
    assert set(a["market_ticker"]) == set(ev["market_ticker"]), "eval sets differ"

ev = add_forecast_columns(ev, v1, "base", "plain_untrained")
ev = add_forecast_columns(ev, r1.assign(condition="t"), "t", "run1")
ev = add_forecast_columns(ev, r2.assign(condition="t"), "t", "run2")
ev = add_forecast_columns(ev, r3, "base", "keynum_untrained")
ev = add_forecast_columns(ev, r3, "trained", "run3")


def latest_rule(r):
    rule = threshold_rule(r, unit_of(r))
    k = key_numbers(r)
    if rule is None or not k or not k["values"]:
        return np.nan
    return float(rule[1](k["values"][-1]))


ev["latest_rule"] = [latest_rule(r) for r in ev.to_dict("records")]
ev["has_rule"] = ev["latest_rule"].notna()
ev["latest_rule"] = ev["latest_rule"].fillna(ev["base_rate"])
ev = ev[ev["market_prob"].notna()].reset_index(drop=True)

METHODS = {"market": "Market", "base_rate": "Base rate", "recent_hits": "Counting rule",
           "latest_rule": "Latest-result rule", "plain_untrained_mean4": "Untrained, plain prompt",
           "run1_mean4": "Run 1", "run2_mean4": "Run 2", "keynum_untrained_mean4": "Untrained, key numbers",
           "run3_mean4": "Run 3"}

# 1. Score table
rows = []
for es, g in ev.groupby("eval_set"):
    for m, name in METHODS.items():
        s = M.summarize(g[m], g["outcome"])
        rows.append({"eval_set": es, "method": name, "brier": s["brier"], "log_loss": s["log_loss"],
                     "ece": s["ece"], "auc": s["auc"], "mean_p": s["mean_p"], "reliability": s["reliability"],
                     "resolution": s["resolution"],
                     "share_extreme": float(((g[m] <= 0.05) | (g[m] >= 0.95)).mean())})
scores = pd.DataFrame(rows)

# 2. Paired bootstrap comparisons
COMPARE = [("plain_untrained_mean4", "keynum_untrained_mean4", "Key numbers, no training"),
           ("plain_untrained_mean4", "run2_mean4", "Run 2 vs untrained"),
           ("keynum_untrained_mean4", "run3_mean4", "Run 3 vs untrained (key numbers)"),
           ("latest_rule", "run3_mean4", "Run 3 vs latest-result rule"),
           ("keynum_untrained_mean4", "recent_hits", "Counting rule vs untrained (key numbers)")]
boots = []
for es, g in ev.groupby("eval_set"):
    for a, b, label in COMPARE:
        t = paired_bootstrap(g, a, b, n_boot=B, seed=0)
        t = t.rename(columns={a: "a", b: "b"})
        t.insert(0, "comparison", label)
        t.insert(0, "eval_set", es)
        boots.append(t)
boot = pd.concat(boots, ignore_index=True)

# 3. Run 3 vs the latest-result rule, question by question
g = ev[ev["has_rule"]]
p1 = r3[r3["condition"] == "trained"]
same4 = p1.groupby("market_ticker")["prob"].nunique().eq(1).mean()
rule_check = {
    "threshold_questions": int(len(g)),
    "run3_extreme_share": float(((g["run3_mean4"] <= 0.02) | (g["run3_mean4"] >= 0.98)).mean()),
    "run3_side_agrees_with_rule": float(((g["run3_mean4"] >= 0.5) == (g["latest_rule"] >= 0.5)).mean()),
    "run3_abs_diff_from_rule_mean": float((g["run3_mean4"] - g["latest_rule"]).abs().mean()),
    "run3_4_samples_identical": float(same4),
    "run3_median_tokens_trained": float(p1["n_tokens"].median()),
    "run3_median_tokens_base": float(r3.loc[r3["condition"] == "base", "n_tokens"].median()),
    "run3_distinct_probs": sorted(p1["prob"].dropna().round(3).value_counts().head(8).to_dict().items()),
}

# 4. Recalibration test: Platt scaling fitted out-of-fold, folds = whole releases (events)
from sklearn.linear_model import LogisticRegression  # noqa: E402


def recalibrate(gg, col, folds=6, seed=0):
    events = gg["event_ticker"].unique()
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(events, rng.permutation(len(events)) % folds))
    f = gg["event_ticker"].map(fold_of).to_numpy()
    x = np.log(np.clip(gg[col].to_numpy(float), 0.01, 0.99) / (1 - np.clip(gg[col].to_numpy(float), 0.01, 0.99)))
    y = gg["outcome"].to_numpy()
    out = np.empty(len(gg))
    for k in range(folds):
        tr, te = f != k, f == k
        lr = LogisticRegression(C=1e6).fit(x[tr, None], y[tr])
        out[te] = lr.predict_proba(x[te, None])[:, 1]
    return out


recal = []
for es, gg in ev.groupby("eval_set"):
    gg = gg.reset_index(drop=True)
    for col in ["plain_untrained_mean4", "run1_mean4", "run2_mean4", "keynum_untrained_mean4", "run3_mean4",
                "recent_hits", "latest_rule"]:
        r = recalibrate(gg, col)
        recal.append({"eval_set": es, "method": METHODS[col], "brier_raw": M.brier(gg[col], gg["outcome"]),
                      "brier_recalibrated": M.brier(r, gg["outcome"]), "auc": M.auc(gg[col], gg["outcome"])})
recal = pd.DataFrame(recal)

# 5. Training dynamics, per 64 answers (= 4 steps), rolling window of 512 answers
curves = []
for name, path in (("Run 1", "train_answers_lr1e-5.csv"), ("Run 2", "train_answers_run2.csv"),
                   ("Run 3", "train_answers_run3.csv")):
    tr = pd.read_csv(f"{DATA}/{path}")
    tr["prob"] = pd.to_numeric(tr["prob"], errors="coerce")
    for end in range(512, len(tr) + 1, 64):
        s = tr.iloc[end - 512:end]
        st = window_stats([None if np.isnan(x) else x for x in s["prob"]], s["outcome"].tolist())
        curves.append({"run": name, "answers": end, "step": end / 16, **st})
curves = pd.DataFrame(curves)

tr2 = pd.read_csv(f"{DATA}/train_answers_run2.csv")
tr2["prob"] = pd.to_numeric(tr2["prob"], errors="coerce")
half = len(tr2) // 2
epochs = {}
for label, s in (("pass1_last512", tr2.iloc[half - 512:half]), ("pass2_first512", tr2.iloc[half:half + 512]),
                 ("pass2_last512", tr2.iloc[-512:])):
    epochs[label] = window_stats([None if np.isnan(x) else x for x in s["prob"]], s["outcome"].tolist())

pd.set_option("display.width", 220)
print(scores.round(4).to_string(index=False))
print()
print(boot[boot.metric.isin(["brier", "auc", "ece", "log_loss", "mean_p"])].round(4).to_string(index=False))
print()
print(json.dumps(rule_check, indent=1, default=str))
print()
print(recal.round(4).to_string(index=False))
print()
print(json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in epochs.items()}, indent=1))
for name, c in curves.groupby("run"):
    idx = [0, len(c) // 4, len(c) // 2, 3 * len(c) // 4, len(c) - 1]
    print(name)
    print(c.iloc[idx][["step", "format_ok", "mean_p", "sd_p", "near_half", "brier", "gap"]].round(3).to_string(index=False))

scores.to_csv(f"{OUT}/runs23_scores.csv", index=False)
boot.to_csv(f"{OUT}/runs23_paired_bootstrap.csv", index=False)
recal.to_csv(f"{OUT}/runs23_recalibration.csv", index=False)
curves.to_csv(f"{OUT}/training_curves_runs123.csv", index=False)
json.dump({"rule_check": rule_check, "run2_passes": epochs}, open(f"{OUT}/runs23_checks.json", "w"), indent=1, default=str)
ev.to_csv(f"{OUT}/runs23_eval_forecasts.csv", index=False)
