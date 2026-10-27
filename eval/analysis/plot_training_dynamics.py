"""Training dynamics of runs 1-3 (or 1-4): share of near-certain answers and the average stated
probability, over a rolling window of 512 training answers (32 steps).

Usage: python eval/analysis/plot_training_dynamics.py <run1_train_answers.csv> ... <runN_...> <out_dir>
With 3 runs writes training_dynamics_runs123.{csv,png}; with 4, training_dynamics_runs1234.{csv,png}
(the png goes to eval/figures/).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RUNS = [("Run 1: one pass, lr 1e-5", "#2a78d6"),
        ("Run 2: two passes, lr 2e-5", "#eb6834"),
        ("Run 3: key numbers, one pass, lr 1e-5", "#1baf7a"),
        ("Run 4: as run 3, std scaling off", "#eda100")]
LEGEND = ["Run 1 (1 pass, lr 1e-5)", "Run 2 (2 passes, lr 2e-5)", "Run 3 (key numbers, 1 pass, lr 1e-5)",
          "Run 4 (as run 3, std scaling off)"]
LABEL_AT = {"Run 1": (355, 0.09), "Run 2": (615, 0.80), "Run 3": (190, 1.02), "Run 4": (205, 0.40)}
WINDOW, EVERY, PER_STEP = 512, 64, 16
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def rolling(path):
    tr = pd.read_csv(path)
    p = pd.to_numeric(tr["prob"], errors="coerce").to_numpy()
    rows = []
    for end in range(WINDOW, len(p) + 1, EVERY):
        w = p[end - WINDOW:end]
        w = w[~np.isnan(w)]
        rows.append({"step": end / PER_STEP, "near_certain": float(((w <= 0.05) | (w >= 0.95)).mean()),
                     "mean_p": float(w.mean())})
    return pd.DataFrame(rows)


def main(paths, out_dir):
    runs = RUNS[:len(paths)]
    tag = "runs" + "".join(str(i + 1) for i in range(len(runs)))
    data = pd.concat([rolling(p).assign(run=name) for p, (name, _) in zip(paths, runs)], ignore_index=True)
    data.to_csv(os.path.join(out_dir, f"training_dynamics_{tag}.csv"), index=False)

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), facecolor=SURFACE)
    panels = [("near_certain", "Share of answers at or below 0.05, or at or above 0.95"),
              ("mean_p", "Average stated probability (dotted: training yes-rate, 0.49)")]
    legend = {name: LEGEND[i] for i, (name, _) in enumerate(runs)}
    for ax, (col, title) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.axvline(1391 / 4, color=MUTED, lw=1, ls=(0, (3, 3)))
        for name, color in runs:
            d = data[data["run"] == name]
            ax.plot(d["step"], d[col], color=color, lw=2, label=legend[name], solid_capstyle="round")
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
        ax.set_xlabel("Training step (4 questions per step)")
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlim(0, 720)
    # Direct labels (left panel), placed clear of the lines
    a = axes[0]
    for name, _ in runs:
        short = name.split(":")[0]
        x, y = LABEL_AT[short]
        a.text(x, y, short, color=INK, fontsize=9, va="bottom" if short == "Run 3" else "center")
    b = axes[1]
    b.text(1391 / 4 + 8, 1.0, "End of the first pass\n(only run 2 continues)", color=MUTED, fontsize=8.5, va="top")
    b.axhline(0.491, color=MUTED, lw=1, ls=(0, (1, 2)))
    title = ("Runs 2 and 3 collapse to near-certain forecasts; run 1 does not" if len(runs) == 3 else
             "With GRPO's std scaling, runs 2 and 3 collapse to near-certain forecasts; with it off, run 4 does not")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(runs) if len(runs) <= 3 else 2, frameon=False,
               fontsize=9, labelcolor=INK,
               bbox_to_anchor=(0.5, 0.045))
    scaling = ("All three runs use GRPO's default std scaling." if len(runs) == 3 else
               "Runs 1-3 use GRPO's default std scaling; run 4 turns it off.")
    fig.text(0.01, 0.005, "Rolling window of 512 training answers (32 steps). Dashed line: end of the first pass over "
             "the 1,391 training questions. " + scaling, fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.1 if len(runs) <= 3 else 0.13, 1, 0.95))
    fig_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    out = os.path.join(fig_dir, f"training_dynamics_{tag}.png")
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print("wrote", os.path.abspath(out))


if __name__ == "__main__":
    main(sys.argv[1:-1], sys.argv[-1])
