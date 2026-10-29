"""Training diagnostics for the one-pass runs: rolling Brier score, AUC, calibration error and share of
near-certain answers over the training answers, and histograms of the stated probabilities at the start
and end of training.

Each training question is answered before the model is updated on it, so a rolling window of training
answers scores the current model on questions it has not been rewarded on yet. It is still not a clean
forward test: other questions from the same release (other thresholds, same outcome value) may already
have been rewarded, so these curves can be optimistic. Run 8's rewards used shuffled outcomes; here every
run is scored against the real outcomes.

Usage: python -I eval/analysis/training_diagnostics.py <repo_dir> <data_dir> <out_dir>
data_dir holds the train answer CSVs downloaded from Kaggle, renamed as in RUNS.
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
from env import metrics as M  # noqa: E402

WINDOW, EVERY, PER_STEP = 512, 64, 16
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
ON, OFF, CONTROL = "#eb6834", "#2a78d6", "#1baf7a"
RUNS = {  # run -> (file, prompt, std scaling, note)
    "run1": ("train_answers_lr1e-5.csv", "plain", "on", ""),
    "run7": ("train_answers_run7.csv", "plain", "off", ""),
    "run3": ("train_answers_run3.csv", "key numbers", "on", "seed 3407"),
    "run9": ("train_answers_run9.csv", "key numbers", "on", "seed 1"),
    "run10": ("train_answers_run10.csv", "key numbers", "on", "seed 2"),
    "run4": ("train_answers_run4.csv", "key numbers", "off", "seed 3407"),
    "run5": ("train_answers_run5.csv", "key numbers", "off", "seed 1"),
    "run6": ("train_answers_run6.csv", "key numbers", "off", "seed 2"),
    "run8": ("train_answers_run8.csv", "key numbers", "off", "shuffled outcomes"),
}


def load(run):
    t = pd.read_csv(os.path.join(DATA, RUNS[run][0]))
    y = t["true_outcome"] if "true_outcome" in t else t["outcome"]
    return pd.to_numeric(t["prob"], errors="coerce").to_numpy(), y.to_numpy(int)


def window_stats(p, y):
    ok = ~np.isnan(p)
    p, y = p[ok], y[ok]
    return {"readable": float(ok.mean()), "brier": M.brier(p, y), "auc": M.auc(p, y), "ece": M.ece(p, y),
            "near_certain": float(((p <= 0.05) | (p >= 0.95)).mean()), "mean_p": float(p.mean()), "sd_p": float(p.std())}


rows, ends = [], []
for run in RUNS:
    p, y = load(run)
    for end in range(WINDOW, len(p) + 1, EVERY):
        rows.append({"run": run, "step": end / PER_STEP, **window_stats(p[end - WINDOW:end], y[end - WINDOW:end])})
    for part, sl in (("first 512", slice(0, WINDOW)), ("last 512", slice(len(p) - WINDOW, len(p)))):
        ends.append({"run": run, "prompt": RUNS[run][1], "std_scaling": RUNS[run][2], "note": RUNS[run][3],
                     "answers": part, **window_stats(p[sl], y[sl])})
roll = pd.DataFrame(rows)
ends = pd.DataFrame(ends)
roll.round(4).to_csv(os.path.join(OUT, "training_diagnostics_rolling.csv"), index=False)
ends.round(4).to_csv(os.path.join(OUT, "training_diagnostics_first_last.csv"), index=False)
pd.set_option("display.width", 200)
print(ends.round(3).to_string(index=False))

# Figure 1: rolling metrics for the key-numbers runs
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "DejaVu Sans"})
fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), facecolor=SURFACE, sharex=True)
panels = [("brier", "Brier score (lower is better)", (0.10, 0.32)),
          ("auc", "AUC (higher is better)", (0.6, 1.0)),
          ("ece", "Calibration error (lower is better)", (0.0, 0.32)),
          ("near_certain", "Share of answers at or below 0.05, or at or above 0.95", (0.0, 1.05))]
lines = [("run3", ON, "Runs 3, 9, 10: std scaling on, 3 seeds"), ("run9", ON, None), ("run10", ON, None),
         ("run4", OFF, "Runs 4-6: std scaling off, 3 seeds"), ("run5", OFF, None), ("run6", OFF, None),
         ("run8", CONTROL, "Run 8: std scaling off, shuffled outcomes")]
for ax, (col, title, ylim) in zip(axes.flat, panels):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for run, color, label in lines:
        d = roll[roll["run"] == run]
        ax.plot(d["step"], d[col], color=color, lw=2 if label else 1.4, alpha=1 if label else 0.75, label=label,
                ls=(0, (4, 2)) if run == "run8" else "-", solid_capstyle="round")
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    ax.set_ylim(*ylim)
    ax.set_xlim(0, 360)
for ax in axes[1]:
    ax.set_xlabel("Training step (one pass, 4 questions per step)")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=INK, bbox_to_anchor=(0.5, 0.035))
fig.suptitle("During training both settings improve AUC; only std scaling on drives answers to 0 or 1 and raises calibration error",
             x=0.01, y=0.985, ha="left", fontsize=12, color=INK)
fig.text(0.01, 0.005, "Rolling window of 512 training answers (32 steps), each scored against the real outcome before the "
         "model is updated on that question. Single answers, not averages of 4.", fontsize=8.5, color=MUTED)
fig.tight_layout(rect=(0, 0.08, 1, 0.965))
fig_dir = os.path.join(REPO, "eval", "figures")
fig.savefig(os.path.join(fig_dir, "training_diagnostics.png"), dpi=160, facecolor=SURFACE)

# Figure 2: stated probabilities, first vs last 512 training answers (seeds pooled)
groups = [("Key numbers, std scaling on (runs 3, 9, 10)", ["run3", "run9", "run10"], ON),
          ("Key numbers, std scaling off (runs 4-6)", ["run4", "run5", "run6"], OFF),
          ("Key numbers, shuffled outcomes (run 8)", ["run8"], CONTROL),
          ("Plain prompt, std scaling off (run 7)", ["run7"], OFF)]
bins = np.linspace(0, 1, 21)
fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), facecolor=SURFACE, sharey=True)
hist_rows = []
for ax, (title, runs, color) in zip(axes, groups):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for part, c, lw, ls in (("first", MUTED, 1.6, (0, (3, 2))), ("last", color, 2.2, "-")):
        vals = []
        for r in runs:
            p, _ = load(r)
            p = p[:WINDOW] if part == "first" else p[-WINDOW:]
            vals.append(p[~np.isnan(p)])
        v = np.concatenate(vals)
        h, _ = np.histogram(np.clip(v, 0, 1), bins=bins)
        share = h / len(v)
        ax.stairs(share, bins, color=c, lw=lw, ls=ls, label=f"{part} 512 answers" + (" per run" if len(runs) > 1 else ""))
        hist_rows += [{"group": title, "answers": part, "bin_low": lo, "share": s} for lo, s in zip(bins[:-1], share)]
    ax.set_title(title, loc="left", fontsize=9.5, color=INK)
    ax.set_xlabel("Stated probability")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper center", frameon=False, fontsize=8.5, labelcolor=INK)
axes[0].set_ylabel("Share of answers (bins of 0.05)")
fig.suptitle("Stated probabilities at the start and end of training", x=0.01, y=0.98, ha="left", fontsize=12, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(os.path.join(fig_dir, "probability_histograms.png"), dpi=160, facecolor=SURFACE)
pd.DataFrame(hist_rows).round(4).to_csv(os.path.join(OUT, "probability_histograms.csv"), index=False)
print("wrote figures")
