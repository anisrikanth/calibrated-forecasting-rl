"""Run 4 (key numbers, GRPO without std scaling): paired event-level bootstrap against the untrained
model, run 3 (same settings with std scaling), and the simple rules; collapse checks; recalibration.

Usage: python -I eval/analysis/analyze_run4.py <repo_dir> <data_dir> <out_dir>
(data_dir holds markets.csv and the eval answer CSVs downloaded from Kaggle; see the read_csv calls)
"""
import json
import sys

import numpy as np
import pandas as pd

REPO, DATA, OUT = sys.argv[1:4]
sys.path.insert(0, REPO)
from env import metrics as M                                                       # noqa: E402
from env.evaluation import add_forecast_columns, load_markets, paired_bootstrap, pick_eval_sets  # noqa: E402

B = 4000
df = load_markets(f"{DATA}/markets.csv")
ev = pick_eval_sets(df, weather_events=60, seed=3407)
v1 = pd.read_csv(f"{DATA}/eval_answers_v1.csv")
r3 = pd.read_csv(f"{DATA}/eval_answers_run3.csv")
r4 = pd.read_csv(f"{DATA}/eval_answers_run4.csv")
for a in (v1, r3, r4):
    assert set(a["market_ticker"]) == set(ev["market_ticker"]), "eval sets differ"
ev = add_forecast_columns(ev, v1, "base", "plain_untrained")
ev = add_forecast_columns(ev, r3, "base", "keynum_untrained")
ev = add_forecast_columns(ev, r3, "trained", "run3")
ev = add_forecast_columns(ev, r4, "trained", "run4")
ev4b = add_forecast_columns(ev, r4, "base", "check")
same_base = float((ev4b["check_mean4"] - ev4b["keynum_untrained_mean4"]).abs().max())
ev = ev[ev["market_prob"].notna()].reset_index(drop=True)

METHODS = {"market": "Market", "logistic": "Logistic regression", "recent_hits": "Counting rule",
           "latest_rule": "Latest-result rule", "base_rate": "Base rate",
           "plain_untrained_mean4": "Untrained, plain prompt", "keynum_untrained_mean4": "Untrained, key numbers",
           "run3_mean4": "Run 3 (std scaling on)", "run4_mean4": "Run 4 (std scaling off)"}

rows = []
for es, g in ev.groupby("eval_set"):
    for m, name in METHODS.items():
        s = M.summarize(g[m], g["outcome"])
        rows.append({"eval_set": es, "method": name, **{k: s[k] for k in ("brier", "log_loss", "ece", "auc", "mean_p",
                                                                          "reliability", "resolution")},
                     "share_extreme": float(((g[m] <= 0.05) | (g[m] >= 0.95)).mean())})
scores = pd.DataFrame(rows)

COMPARE = [("keynum_untrained_mean4", "run4_mean4", "Run 4 vs untrained (key numbers)"),
           ("run3_mean4", "run4_mean4", "Run 4 vs run 3 (std scaling off vs on)"),
           ("recent_hits", "run4_mean4", "Run 4 vs counting rule"),
           ("logistic", "run4_mean4", "Run 4 vs logistic regression"),
           ("latest_rule", "run4_mean4", "Run 4 vs latest-result rule"),
           ("plain_untrained_mean4", "run4_mean4", "Run 4 vs untrained (plain prompt)")]
boots = []
for es, g in ev.groupby("eval_set"):
    for a, b, label in COMPARE:
        t = paired_bootstrap(g, a, b, n_boot=B, seed=0).rename(columns={a: "a", b: "b"})
        t.insert(0, "comparison", label)
        t.insert(0, "eval_set", es)
        boots.append(t)
boot = pd.concat(boots, ignore_index=True)

t4 = r4[r4["condition"] == "trained"]
e = ev[ev["eval_set"] == "val_econ"]
thr = e[e["series_ticker"] != "KXFEDDECISION"]
checks = {
    "untrained_answers_identical_to_run3": same_base == 0.0,
    "run4_median_tokens": float(t4["n_tokens"].median()),
    "run4_readable": float(t4["prob"].notna().mean()),
    "run4_4_samples_identical": float(t4.groupby("market_ticker")["prob"].nunique().eq(1).mean()),
    "run4_share_exact_0_or_1": float(t4["prob"].isin([0.0, 1.0]).mean()),
    "run4_econ_same_side_as_latest_rule": float(((thr["run4_mean4"] >= 0.5) == (thr["latest_rule"] >= 0.5)).mean()),
    "run3_econ_same_side_as_latest_rule": float(((thr["run3_mean4"] >= 0.5) == (thr["latest_rule"] >= 0.5)).mean()),
    "keynum_untrained_econ_same_side_as_latest_rule": float(((thr["keynum_untrained_mean4"] >= 0.5) == (thr["latest_rule"] >= 0.5)).mean()),
    "run4_auc_within_latest_rule_yes": M.auc(thr.loc[thr.latest_rule == 1, "run4_mean4"], thr.loc[thr.latest_rule == 1, "outcome"]),
    "run4_auc_within_latest_rule_no": M.auc(thr.loc[thr.latest_rule == 0, "run4_mean4"], thr.loc[thr.latest_rule == 0, "outcome"]),
    "keynum_untrained_auc_within_latest_rule_yes": M.auc(thr.loc[thr.latest_rule == 1, "keynum_untrained_mean4"], thr.loc[thr.latest_rule == 1, "outcome"]),
    "keynum_untrained_auc_within_latest_rule_no": M.auc(thr.loc[thr.latest_rule == 0, "keynum_untrained_mean4"], thr.loc[thr.latest_rule == 0, "outcome"]),
    "corr_run4_logistic_econ": float(np.corrcoef(e["run4_mean4"], e["logistic"])[0, 1]),
    "corr_run4_counting_econ": float(np.corrcoef(e["run4_mean4"], e["recent_hits"])[0, 1]),
}

series = e.groupby("series_ticker").apply(lambda g: pd.Series({
    "questions": len(g), "releases": g["event_ticker"].nunique(),
    **{name: M.brier(g[m], g["outcome"]) for m, name in METHODS.items() if m != "plain_untrained_mean4"}}),
    include_groups=False)

from sklearn.linear_model import LogisticRegression  # noqa: E402


def recalibrate(gg, col, folds=6, seed=0):
    events = gg["event_ticker"].unique()
    fold_of = dict(zip(events, np.random.default_rng(seed).permutation(len(events)) % folds))
    f = gg["event_ticker"].map(fold_of).to_numpy()
    p = np.clip(gg[col].to_numpy(float), 0.01, 0.99)
    x, y = np.log(p / (1 - p)), gg["outcome"].to_numpy()
    out = np.empty(len(gg))
    for k in range(folds):
        lr = LogisticRegression(C=1e6).fit(x[f != k, None], y[f != k])
        out[f == k] = lr.predict_proba(x[f == k, None])[:, 1]
    return out


recal = []
for es, gg in ev.groupby("eval_set"):
    gg = gg.reset_index(drop=True)
    for col in ["keynum_untrained_mean4", "run3_mean4", "run4_mean4", "recent_hits", "logistic"]:
        recal.append({"eval_set": es, "method": METHODS[col], "brier_raw": M.brier(gg[col], gg["outcome"]),
                      "brier_recalibrated": M.brier(recalibrate(gg, col), gg["outcome"])})
recal = pd.DataFrame(recal)

pd.set_option("display.width", 230)
print(scores.round(4).to_string(index=False)); print()
print(boot[boot.metric.isin(["brier", "auc", "ece", "log_loss", "mean_p"])].round(4).to_string(index=False)); print()
print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in checks.items()}, indent=1)); print()
print(series.round(4).to_string()); print()
print(recal.round(4).to_string(index=False))

scores.round(4).to_csv(f"{OUT}/run4_scores.csv", index=False)
boot.round(4).to_csv(f"{OUT}/run4_paired_bootstrap.csv", index=False)
series.round(4).to_csv(f"{OUT}/run4_per_series.csv")
recal.round(4).to_csv(f"{OUT}/run4_recalibration.csv", index=False)
json.dump(checks, open(f"{OUT}/run4_checks.json", "w"), indent=1, default=str)
