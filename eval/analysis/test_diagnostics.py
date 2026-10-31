"""Descriptive diagnostics on the test split, computed after it was scored, from the saved test answers.

None of this is part of the comparisons fixed in eval/test_plan.md (analyze_test.py applies those); everything here
is descriptive or exploratory and is labeled so in the write-up. It mirrors the validation diagnostics
(review_checks.py, skill_scores.py) so the two splits can be read side by side:
1. Per model: Brier of the four-answer average and of single answers, ladder violations, Brier with each
   announcement weighted equally, calibration intercept and slope.
2. Brier by series, with the latest-result rule and the trees.
3. Brier skill relative to the base rate.
4. Exploratory comparisons with intervals (resampling announcements) that the validation write-up reported but
   the plan did not include: runs 4-6 vs the counting rule, runs 3, 9, 10 vs the latest-result rule.
5. A reliability diagram, as for validation.

Usage: python -I eval/analysis/test_diagnostics.py <repo_dir> <data_dir> <out_dir>
data_dir: markets.csv and test_answers.csv; out_dir: holds gbm_test_forecasts.csv (gbm_baseline.py, test mode)
and gets test_diagnostics.csv, test_series.csv, test_skill_scores.csv, test_exploratory_bootstrap.csv and
reliability_test.csv; the figure goes to eval/figures/reliability_test.png.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO, DATA, OUT = sys.argv[1:4]
sys.path.insert(0, REPO)
from env import metrics as M                                                         # noqa: E402
from env.baselines import fit_logistic                                               # noqa: E402
from env.evaluation import (add_forecast_columns, load_markets, paired_bootstrap_groups,  # noqa: E402
                            pick_eval_sets, scored_subset)

MAIN, ON = ["run4", "run5", "run6"], ["run3", "run9", "run10"]
LABELS = {"untrained_plain": "Untrained, plain prompt", "run1": "Run 1: plain, std on", "run7": "Run 7: plain, std off",
          "untrained_keynum": "Untrained, key numbers", "run3": "Run 3: key numbers, std on, seed 3407",
          "run9": "Run 9: key numbers, std on, seed 1", "run10": "Run 10: key numbers, std on, seed 2",
          "run4": "Run 4: key numbers, std off, seed 3407", "run5": "Run 5: key numbers, std off, seed 1",
          "run6": "Run 6: key numbers, std off, seed 2", "run8": "Run 8: key numbers, std off, shuffled outcomes"}
answers = pd.read_csv(os.path.join(DATA, "test_answers.csv"))
ev = pick_eval_sets(load_markets(os.path.join(DATA, "markets.csv")), weather_events=0, econ_split="test")
for c in LABELS:
    ev = add_forecast_columns(ev, answers, c, c)
ev = ev.merge(pd.read_csv(os.path.join(OUT, "gbm_test_forecasts.csv"))[["market_ticker", "gbm", "gbm_series"]],
              on="market_ticker", how="left")
g = scored_subset(ev).reset_index(drop=True)
y = g["outcome"].to_numpy()
m4 = lambda labels: [f"{l}_mean4" for l in labels]  # noqa: E731

# 1. per-model diagnostics
models = {**{c: f"{c}_mean4" for c in LABELS}, "logistic": "logistic", "gbm_series": "gbm_series", "gbm": "gbm",
          "recent_hits": "recent_hits", "latest_rule": "latest_rule", "base_rate": "base_rate", "market": "market"}
ann = g["announcement"].to_numpy()
diag = []
for name, col in models.items():
    p = g[col].to_numpy(float)
    row = {"model": name, "brier_mean4": M.brier(p, y), "auc": M.auc(p, y), "mean_p": float(p.mean())}
    if name in LABELS:
        a = answers[(answers["condition"] == name) & answers["market_ticker"].isin(g["market_ticker"])]
        yy = a["market_ticker"].map(g.set_index("market_ticker")["outcome"]).to_numpy()
        row["brier_single_answers"] = float(np.mean((a["prob"].fillna(0.5).to_numpy(float) - yy) ** 2))
        row["readable"] = float(a["prob"].notna().mean())
    lad = M.ladder_violations(g, col)
    row["ladder_pairs"], row["ladder_violation_rate"] = lad["pairs"], lad["violation_rate"]
    row["brier_equal_announcement_weight"] = float(pd.Series((p - y) ** 2).groupby(ann).mean().mean())
    if np.std(p) > 0:
        z = np.log(np.clip(p, 0.01, 0.99) / (1 - np.clip(p, 0.01, 0.99)))
        w = fit_logistic(z[:, None], y.astype(float), l2=0.0)
        row["calibration_intercept"], row["calibration_slope"] = float(w[0]), float(w[1])
    diag.append(row)
diag = pd.DataFrame(diag)

# 2. by series
series = g.groupby("series_ticker").apply(
    lambda s: pd.Series({"n": len(s), **{m: M.brier(s[c], s["outcome"]) for m, c in models.items()}}),
    include_groups=False)

# 3. skill relative to the base rate
base = M.brier(g["base_rate"], y)
skill = pd.DataFrame([{"method": m, "brier": M.brier(g[c], y), "base_rate_brier": base,
                       "skill": 1 - M.brier(g[c], y) / base} for m, c in models.items()])

# 4. exploratory comparisons (not in the plan)
EXTRA = [("Runs 4-6 vs counting rule", ["recent_hits"], m4(MAIN)),
         ("Runs 3, 9, 10 vs latest-result rule", ["latest_rule"], m4(ON))]
boot = pd.concat([paired_bootstrap_groups(g, a, b, n_boot=4000, seed=0, cluster=cl).assign(comparison=n, cluster=cl)
                  for cl in ("announcement", "event_ticker") for n, a, b in EXTRA], ignore_index=True)

# 5. reliability diagram, as for validation
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "font.family": "DejaVu Sans"})
panels = [("Untrained, key numbers", ["untrained_keynum_mean4"], "#52514e"),
          ("Runs 4-6 (std off), pooled", m4(MAIN), "#2a78d6"),
          ("Runs 3, 9, 10 (std on), pooled", m4(ON), "#eb6834"),
          ("Logistic regression", ["logistic"], "#1baf7a")]
fig, axes = plt.subplots(1, 4, figsize=(13, 3.9), facecolor="#fcfcfb", sharey=True)
rel_rows = []
for ax, (title, cols, color) in zip(axes, panels):
    p = np.concatenate([g[c].to_numpy(float) for c in cols])
    t = M.reliability_table(p, np.tile(y, len(cols)))
    ax.set_facecolor("#fcfcfb")
    ax.plot([0, 1], [0, 1], color="#52514e", lw=1, ls=(0, (3, 3)))
    ax.scatter(t["mean_forecast"], t["frac_yes"], s=np.clip(10 + 1.2 * t["count"] / len(cols), 10, 120), color=color,
               alpha=0.85, edgecolor="#fcfcfb", linewidth=1.5, zorder=3, clip_on=False)
    ax.plot(t["mean_forecast"], t["frac_yes"], color=color, lw=1.4, zorder=2)
    for _, r in t.iterrows():
        ax.annotate(f"{int(r['count'])}", (r["mean_forecast"], r["frac_yes"]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7.5, color="#52514e", annotation_clip=False)
    ax.set_title(title, loc="left", fontsize=10, color="#0b0b0b")
    ax.set_xlim(-0.04, 1.04); ax.set_ylim(-0.05, 1.12)
    ax.set_xlabel("Forecast (bin average)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    rel_rows += [dict(panel=title, **r) for r in t.to_dict("records")]
axes[0].set_ylabel("Share that resolved YES")
fig.suptitle("Reliability on test econ (10 equal-width bins; labels: forecasts per bin, seeds pooled)", x=0.01, y=0.98,
             ha="left", fontsize=12, color="#0b0b0b")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(os.path.join(REPO, "eval", "figures", "reliability_test.png"), dpi=160, facecolor="#fcfcfb")

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print(diag.round(4).to_string(index=False)); print()
print(series.round(3).to_string()); print()
print(skill.round(3).to_string(index=False)); print()
print(boot[boot.metric.isin(["brier", "log_loss", "ece", "auc"])].round(4).to_string(index=False))
diag.round(4).to_csv(os.path.join(OUT, "test_diagnostics.csv"), index=False)
series.round(4).to_csv(os.path.join(OUT, "test_series.csv"))
skill.round(4).to_csv(os.path.join(OUT, "test_skill_scores.csv"), index=False)
boot.round(4).to_csv(os.path.join(OUT, "test_exploratory_bootstrap.csv"), index=False)
pd.DataFrame(rel_rows).round(4).to_csv(os.path.join(OUT, "reliability_test.csv"), index=False)
print("announcements:", g["announcement"].nunique(), "releases:", g["event_ticker"].nunique(), "questions:", len(g))
