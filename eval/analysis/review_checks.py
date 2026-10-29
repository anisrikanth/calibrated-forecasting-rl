"""Checks suggested in outside review, all computed from the saved validation answers (no new generation).

1. Intervals that resample whole announcements (CPI's four series are released together, as are payrolls and
   unemployment, and the Fed rate and decision) next to the earlier ones that resample single releases.
2. Brier of the four-answer average vs the mean Brier of the single answers (their gap is the average
   within-question variance of the four answers), and AUC of the first answer alone.
3. Threshold-ladder violations for every condition.
4. Brier with each announcement weighted equally.
5. Calibration intercept and slope (logistic recalibration of the forecasts on the outcomes, descriptive)
   and a reliability diagram with bin counts.
6. The trained models against the best untrained prompt from the ablation (three key-number lines).
7. The two aggregation rules for unreadable answers.

Usage: python -I eval/analysis/review_checks.py <repo_dir> <data_dir> <out_dir>
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
from env.evaluation import add_forecast_columns, load_markets, paired_bootstrap_groups, pick_eval_sets  # noqa: E402

B = 4000
RUNS = {  # label -> (file, condition, description)
    "untrained_plain": ("eval_answers_v1.csv", "base", "Untrained, plain prompt"),
    "untrained_keynum": ("eval_answers_run3.csv", "base", "Untrained, key numbers"),
    "untrained_three_lines": ("prompt_ablation_answers.csv", "rule_latest_hits", "Untrained, three key-number lines"),
    "run1": ("eval_answers_lr1e-5.csv", "trained", "Run 1: plain, std on"),
    "run2": ("eval_answers_run2.csv", "trained", "Run 2: plain, std on, 2 passes"),
    "run7": ("eval_answers_run7.csv", "trained", "Run 7: plain, std off"),
    **{f"run{i}": (f"eval_answers_run{i}.csv", "trained", d) for i, d in [
        (3, "Run 3: key numbers, std on"), (9, "Run 9: key numbers, std on"), (10, "Run 10: key numbers, std on"),
        (4, "Run 4: key numbers, std off"), (5, "Run 5: key numbers, std off"), (6, "Run 6: key numbers, std off"),
        (8, "Run 8: key numbers, std off, shuffled")]},
}
MAIN, ON = ["run4", "run5", "run6"], ["run3", "run9", "run10"]
df = load_markets(os.path.join(DATA, "markets.csv"))
ev = pick_eval_sets(df, weather_events=60, seed=3407)
answers = {}
for c, (f, cond, _) in RUNS.items():
    a = pd.read_csv(os.path.join(DATA, f))
    answers[c] = a[a["condition"] == cond]
    ev = add_forecast_columns(ev, a, cond, c)
ev = ev.merge(pd.read_csv(os.path.join(OUT, "gbm_val_forecasts.csv"))[["market_ticker", "gbm", "gbm_series"]],
              on="market_ticker", how="left")
g = ev[(ev["eval_set"] == "val_econ") & ev["market_prob"].notna()].reset_index(drop=True)
y = g["outcome"].to_numpy()
m4 = lambda labels: [f"{l}_mean4" for l in labels]  # noqa: E731

# 1. announcement vs release resampling
COMPARE = [("P1: runs 4-6 vs untrained, key numbers", m4(["untrained_keynum"]), m4(MAIN)),
           ("P2: runs 4-6 vs runs 3, 9, 10", m4(ON), m4(MAIN)),
           ("Runs 4-6 vs logistic regression", ["logistic"], m4(MAIN)),
           ("Runs 4-6 vs counting rule", ["recent_hits"], m4(MAIN)),
           ("Runs 4-6 vs trees (key numbers)", ["gbm"], m4(MAIN)),
           ("Runs 4-6 vs trees (key numbers + series)", ["gbm_series"], m4(MAIN)),
           ("Runs 4-6 vs untrained, three key-number lines", m4(["untrained_three_lines"]), m4(MAIN)),
           ("Runs 3, 9, 10 vs untrained, key numbers", m4(["untrained_keynum"]), m4(ON)),
           ("Runs 3, 9, 10 vs latest-result rule", ["latest_rule"], m4(ON)),
           ("Run 8 vs untrained, key numbers", m4(["untrained_keynum"]), m4(["run8"])),
           ("Runs 4-6 vs run 8", m4(["run8"]), m4(MAIN)),
           ("Run 7 vs untrained, plain", m4(["untrained_plain"]), m4(["run7"])),
           ("Run 1 vs untrained, plain", m4(["untrained_plain"]), m4(["run1"])),
           ("Untrained: key numbers vs plain", m4(["untrained_plain"]), m4(["untrained_keynum"]))]
COMPARE += [(f"{r} vs untrained, key numbers", m4(["untrained_keynum"]), m4([r])) for r in MAIN + ON]
rows = []
for cluster in ("announcement", "event_ticker"):
    for name, a, b in COMPARE:
        t = paired_bootstrap_groups(g, a, b, n_boot=B, seed=0, cluster=cluster)
        rows.append(t.assign(comparison=name, cluster=cluster))
boot = pd.concat(rows, ignore_index=True)[["comparison", "cluster", "metric", "a", "b", "change", "ci_low", "ci_high",
                                           "b_min", "b_max"]]

# 2.-5. per-model diagnostics
models = {**{c: f"{c}_mean4" for c in RUNS}, "logistic": "logistic", "recent_hits": "recent_hits",
          "latest_rule": "latest_rule", "gbm_series": "gbm_series", "base_rate": "base_rate"}
first_auc = {}
diag = []
ann = g["announcement"].to_numpy()
for name, col in models.items():
    p = g[col].to_numpy(float)
    row = {"model": name, "brier_mean4": M.brier(p, y), "auc_mean4": M.auc(p, y)}
    if name in RUNS:
        a = answers[name][answers[name]["market_ticker"].isin(g["market_ticker"])]
        yy = a["market_ticker"].map(g.set_index("market_ticker")["outcome"]).to_numpy()
        ps = a["prob"].fillna(0.5).to_numpy(float)
        row["brier_single_answers"] = float(np.mean((ps - yy) ** 2))
        row["within_question_variance"] = row["brier_single_answers"] - M.brier(
            a.assign(p=ps).groupby("market_ticker")["p"].mean().reindex(g["market_ticker"]).to_numpy(), y)
        row["auc_first_answer"] = M.auc(g[f"{name}_single"].to_numpy(float), y)
        row["unreadable"] = float(a["prob"].isna().mean())
        alt = a.assign(p=ps).groupby("market_ticker")["p"].mean().reindex(g["market_ticker"]).to_numpy()
        row["brier_unreadable_as_half"] = M.brier(alt, y)
    lad = M.ladder_violations(g, col)
    row["ladder_pairs"], row["ladder_violation_rate"] = lad["pairs"], lad["violation_rate"]
    per_ann = pd.Series((p - y) ** 2).groupby(ann).mean()
    row["brier_equal_announcement_weight"] = float(per_ann.mean())
    if np.std(p) > 0:
        z = np.log(np.clip(p, 0.01, 0.99) / (1 - np.clip(p, 0.01, 0.99)))
        w = fit_logistic(z[:, None], y.astype(float), l2=0.0)
        row["calibration_intercept"], row["calibration_slope"] = float(w[0]), float(w[1])
    diag.append(row)
diag = pd.DataFrame(diag)

# reliability diagram with counts
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
    yy = np.tile(y, len(cols))
    t = M.reliability_table(p, yy)
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
fig.suptitle("Reliability on validation econ (10 equal-width bins; labels: forecasts per bin, seeds pooled)", x=0.01, y=0.98,
             ha="left", fontsize=12, color="#0b0b0b")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(os.path.join(REPO, "eval", "figures", "reliability_validation.png"), dpi=160, facecolor="#fcfcfb")

pd.set_option("display.width", 250)
print(boot[boot.metric.isin(["brier", "auc", "ece"])].round(4).to_string(index=False)); print()
print(diag.round(4).to_string(index=False))
boot.round(4).to_csv(os.path.join(OUT, "review_paired_bootstrap.csv"), index=False)
diag.round(4).to_csv(os.path.join(OUT, "review_diagnostics.csv"), index=False)
pd.DataFrame(rel_rows).round(4).to_csv(os.path.join(OUT, "reliability_validation.csv"), index=False)
print("announcements:", g["announcement"].nunique(), "releases:", g["event_ticker"].nunique())
