"""Applies the reading rules fixed in eval/test_plan.md to the test notebook's outputs.

Written and committed before the test split was scored, so the verdicts come from code that predates the
results. It reads what notebooks/phase4_test_eval.ipynb and eval/analysis/gbm_baseline.py (test mode) wrote;
it fits nothing and changes no forecast.

Rules (eval/test_plan.md, "How each is read"):
- P1 has two separate conclusions: the Brier improvement holds if its interval lies entirely below 0, the
  AUC improvement if its interval lies entirely above 0.
- P2 holds if the intervals for log loss and calibration error both lie entirely below 0.
- P3 holds if, for each of runs 3, 9 and 10, at least 90% of single answers are exactly 0 or 1 (all scored
  questions, as computed on validation and in the notebook) and at least 90% of single answers on the
  questions with a numeric threshold are exactly the latest-result rule's 0 or 1.
- The right direction with an interval that includes 0 is "consistent but not confirmed"; the wrong
  direction (or no change) is "not replicated"; a comparison whose models all failed the loading check is
  "not evaluable".
Intervals resample announcements. The same rules applied to the release-level intervals are reported as a
sensitivity check. Secondary comparisons get no verdict: only their direction and whether the interval
excludes 0.

Also computes two descriptive checks the plan promises that the notebook does not print:
- Brier with each unreadable answer counted as 0.5, instead of averaging the readable answers;
- Brier with each announcement weighted equally.

Usage: python -I eval/analysis/analyze_test.py <repo_dir> <data_dir> <out_dir>
data_dir: markets.csv and the notebook's test_answers.csv, test_check.csv, test_comparisons.csv,
test_comparisons_by_release.csv and test_collapse.csv; plus gbm_test_paired_bootstrap.csv (S8) if present.
Writes test_verdicts.csv, test_p3.csv, test_secondary.csv and test_descriptive.csv to out_dir.
"""
import os
import sys

import numpy as np
import pandas as pd

REPO, DATA, OUT = sys.argv[1:4]
sys.path.insert(0, REPO)
from env import metrics as M                                                         # noqa: E402
from env.baselines import latest_rule_column                                         # noqa: E402
from env.evaluation import add_forecast_columns, load_markets, pick_eval_sets, scored_subset  # noqa: E402

LOWER_IS_BETTER = {"brier": True, "log_loss": True, "ece": True, "auc": False}
METRICS = list(LOWER_IS_BETTER)
P3_RUNS, P3_MIN = ["run3", "run9", "run10"], 0.90
HOLDS, CONSISTENT, NOT_REPLICATED, NOT_EVALUABLE = ("holds", "consistent but not confirmed", "not replicated",
                                                    "not evaluable")
BASELINES = ["market", "base_rate", "recent_hits", "latest_rule", "logistic", "always_0.5"]


def read(name, required=True):
    path = os.path.join(DATA, name)
    if os.path.exists(path):
        return pd.read_csv(path)
    if required:
        sys.exit(f"missing {path}")
    return None


def get(table, cid, metric):
    r = table[(table["id"] == cid) & (table["metric"] == metric)]
    return None if r.empty else r.iloc[0]


def reading(row, metric):
    """The plan's reading of one interval (change = compared group minus reference)."""
    if row is None or pd.isna(row["change"]):
        return NOT_EVALUABLE
    lower = LOWER_IS_BETTER[metric]
    if (row["ci_high"] < 0) if lower else (row["ci_low"] > 0):
        return HOLDS
    return CONSISTENT if ((row["change"] < 0) if lower else (row["change"] > 0)) else NOT_REPLICATED


def both_required(statuses):
    if NOT_EVALUABLE in statuses:
        return NOT_EVALUABLE
    if all(s == HOLDS for s in statuses):
        return HOLDS
    return NOT_REPLICATED if NOT_REPLICATED in statuses else CONSISTENT


def describe(row, metric):
    """Secondary comparisons: direction and whether the interval excludes 0, no verdict."""
    if row is None or pd.isna(row["change"]):
        return "not evaluable"
    better = (row["change"] < 0) if LOWER_IS_BETTER[metric] else (row["change"] > 0)
    excludes = row["ci_high"] < 0 or row["ci_low"] > 0
    return ("better" if better else "worse" if row["change"] != 0 else "no change") + \
        (", interval excludes 0" if excludes else ", interval includes 0")


comp, by_rel = read("test_comparisons.csv"), read("test_comparisons_by_release.csv")
check, answers = read("test_check.csv"), read("test_answers.csv")
collapse_nb = read("test_collapse.csv")
gbm = read("gbm_test_paired_bootstrap.csv", required=False)
bad = sorted(m for m in check.loc[check["status"] == "MISMATCH", "model"] if m.startswith("run"))
os.makedirs(OUT, exist_ok=True)

# ---- primary comparisons
COLS = ["comparison", "models", "a", "b", "change", "ci_low", "ci_high", "share_better", "b_min", "b_max"]
rows = []
for cid, metric, part in [("P1", "brier", "P1 Brier"), ("P1", "auc", "P1 AUC"), ("P2", "log_loss", "P2"),
                          ("P2", "ece", "P2"), ("P2", "brier", "reported, not predicted"),
                          ("P2", "auc", "reported, not predicted")]:
    r, rr = get(comp, cid, metric), get(by_rel, cid, metric)
    rows.append({"part": part, "id": cid, "metric": metric, **{k: (r[k] if r is not None else np.nan) for k in COLS},
                 "reading": reading(r, metric),
                 "release_ci_low": rr["ci_low"] if rr is not None else np.nan,
                 "release_ci_high": rr["ci_high"] if rr is not None else np.nan,
                 "reading_by_release": reading(rr, metric)})
primary = pd.DataFrame(rows)


def part_reading(part, col):
    s = primary.loc[primary["part"] == part, col].tolist()
    return both_required(s) if part == "P2" else s[0]


# ---- P3, recomputed from the single answers (unrounded) and checked against the notebook's table
test_df = pick_eval_sets(load_markets(os.path.join(DATA, "markets.csv")), weather_events=0, econ_split="test")
g = scored_subset(test_df).reset_index(drop=True)
rule_q = g[latest_rule_column(g).notna()]
p3 = []
for run in P3_RUNS:
    a = answers[(answers["condition"] == run) & answers["market_ticker"].isin(g["market_ticker"])]
    if a.empty:
        p3.append({"model": run, "reading": NOT_EVALUABLE})
        continue
    ar = a.merge(rule_q[["market_ticker", "latest_rule"]], on="market_ticker")
    at01 = float(a["prob"].isin([0.0, 1.0]).mean())
    eq = float((ar["prob"] == ar["latest_rule"]).mean())
    at01_rule_q = float(ar["prob"].isin([0.0, 1.0]).mean())
    nb = collapse_nb.set_index("model").loc[run] if run in set(collapse_nb["model"]) else None
    if nb is not None:
        assert abs(nb["answers_at_0_or_1"] - at01) < 6e-4 and abs(nb["answers_equal_to_rule"] - eq) < 6e-4, \
            f"{run}: P3 recomputation differs from the notebook's test_collapse.csv"
    p3.append({"model": run, "answers": len(a), "answers_at_0_or_1": at01, "threshold_answers": len(ar),
               "answers_equal_to_rule": eq, "answers_at_0_or_1_threshold_questions_only": at01_rule_q,
               "meets_both": at01 >= P3_MIN and eq >= P3_MIN, "failed_loading_check": run in bad,
               "reading": HOLDS if (at01 >= P3_MIN and eq >= P3_MIN) else NOT_REPLICATED})
p3 = pd.DataFrame(p3)
p3_reading = (NOT_EVALUABLE if (p3["reading"] == NOT_EVALUABLE).all() else
              HOLDS if (p3["reading"] == HOLDS).all() else NOT_REPLICATED)

verdicts = pd.DataFrame([
    {"prediction": "P1 Brier: runs 4-6 lower than untrained (key numbers)",
     "reading": part_reading("P1 Brier", "reading"), "reading_by_release": part_reading("P1 Brier", "reading_by_release")},
    {"prediction": "P1 AUC: runs 4-6 higher than untrained (key numbers)",
     "reading": part_reading("P1 AUC", "reading"), "reading_by_release": part_reading("P1 AUC", "reading_by_release")},
    {"prediction": "P2: runs 4-6 lower log loss and calibration error than runs 3, 9, 10",
     "reading": part_reading("P2", "reading"), "reading_by_release": part_reading("P2", "reading_by_release")},
    {"prediction": "P3: runs 3, 9, 10 each >= 90% answers at 0/1 and >= 90% equal to the rule",
     "reading": p3_reading, "reading_by_release": "(no interval)"},
])

# ---- secondary comparisons: S1-S7 and each run on its own from the notebook, S8 from gbm_baseline.py
sec = []
for _, r in comp[comp["id"].ne("P1") & comp["id"].ne("P2") & comp["metric"].isin(METRICS)].iterrows():
    sec.append({"id": r["id"], "comparison": r["comparison"], "cluster": "announcement", "metric": r["metric"],
                **{k: r.get(k, np.nan) for k in COLS[1:]}, "description": describe(r, r["metric"])})
for _, r in by_rel[by_rel["id"].eq("S1") & by_rel["metric"].isin(METRICS)].iterrows():
    sec.append({"id": "S1", "comparison": r["comparison"], "cluster": "release", "metric": r["metric"],
                **{k: r.get(k, np.nan) for k in COLS[1:]}, "description": describe(r, r["metric"])})
if gbm is not None:
    for _, r in gbm[gbm["comparison"].str.startswith("runs 4-6") & gbm["metric"].isin(METRICS)].iterrows():
        sec.append({"id": "S8", "comparison": r["comparison"], "cluster": "announcement", "metric": r["metric"],
                    **{k: r.get(k, np.nan) for k in COLS[1:]}, "description": describe(r, r["metric"])})
secondary = pd.DataFrame(sec)

# ---- descriptive checks promised by the plan
desc = []
conditions = list(dict.fromkeys(answers["condition"]))
for m in BASELINES + conditions:
    if m in conditions:
        a = answers[answers["condition"] == m]
        p = add_forecast_columns(g, answers, m, m)[f"{m}_mean4"]
        alt = g["market_ticker"].map(a.assign(prob=a["prob"].fillna(0.5)).groupby("market_ticker")["prob"].mean())
        readable = float(a.loc[a["market_ticker"].isin(g["market_ticker"]), "prob"].notna().mean())
    else:
        p, alt, readable = g[m], None, np.nan
    sq = (p.to_numpy(float) - g["outcome"].to_numpy(float)) ** 2
    desc.append({"method": m, "brier": M.brier(p, g["outcome"]),
                 "brier_announcements_equal": float(pd.Series(sq).groupby(g["announcement"]).mean().mean()),
                 "readable": readable,
                 "brier_unreadable_as_0.5": M.brier(alt, g["outcome"]) if alt is not None else np.nan})
descriptive = pd.DataFrame(desc)
descriptive["unreadable_rule_change"] = descriptive["brier_unreadable_as_0.5"] - descriptive["brier"]

primary.to_csv(os.path.join(OUT, "test_verdicts.csv"), index=False)
p3.to_csv(os.path.join(OUT, "test_p3.csv"), index=False)
secondary.to_csv(os.path.join(OUT, "test_secondary.csv"), index=False)
descriptive.to_csv(os.path.join(OUT, "test_descriptive.csv"), index=False)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print(f"test: {len(g)} scored questions, {g['announcement'].nunique()} announcements | "
      f"adapters that failed the loading check: {bad or 'none'}\n")
print(verdicts.to_string(index=False), "\n")
print(primary.round(4).to_string(index=False), "\n")
print(p3.round(4).to_string(index=False), "\n")
print(secondary.round(4).to_string(index=False), "\n")
print(descriptive.round(4).to_string(index=False))
