"""The 2 x 2 in one figure: how spread out the model's stated probabilities are during training.

Left: plain prompt (run 1 std scaling on, run 7 off). Right: key numbers (run 3 on, runs 4-6 off).
Spread = standard deviation of the stated probabilities over a rolling window of 512 training
answers. A model that answers only 0 or 1 on balanced questions has spread 0.5; one that hedges
toward a single number has spread near 0.

Usage: python eval/analysis/plot_spread_2x2.py <run1> <run7> <run3> <run4> <run5> <run6> [<run8>] <out_dir>
(each argument is that run's train_answers CSV; run 8, the shuffled-outcome control, is optional). Writes eval/figures/spread_2x2.png and
<out_dir>/spread_2x2.csv.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

WINDOW, EVERY, PER_STEP = 512, 64, 16
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
ON, OFF, CONTROL = "#eb6834", "#2a78d6", "#1baf7a"   # std scaling on / off / shuffled-outcome control


def rolling_sd(path):
    p = pd.to_numeric(pd.read_csv(path)["prob"], errors="coerce").to_numpy()
    rows = []
    for end in range(WINDOW, len(p) + 1, EVERY):
        w = p[end - WINDOW:end]
        w = w[~np.isnan(w)]
        rows.append({"step": end / PER_STEP, "sd": float(w.std())})
    return pd.DataFrame(rows)


def main(paths, out_dir):
    names = ["Run 1", "Run 7", "Run 3", "Run 4", "Run 5", "Run 6", "Run 8"][:len(paths)]
    data = pd.concat([rolling_sd(p).assign(run=n) for p, n in zip(paths, names)], ignore_index=True)
    data.to_csv(os.path.join(out_dir, "spread_2x2.csv"), index=False)

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE, sharey=True)
    panels = [("Plain prompt (recent results only)", [("Run 1", ON, "Run 1: std scaling on"),
                                                      ("Run 7", OFF, "Run 7: std scaling off")]),
              ("Key numbers in the prompt", [("Run 3", ON, "Run 3: std scaling on"),
                                             ("Run 4", OFF, "Runs 4-6: std scaling off, 3 seeds"),
                                             ("Run 5", OFF, None), ("Run 6", OFF, None)]
               + ([("Run 8", CONTROL, "Run 8: std scaling off, shuffled outcomes")] if "Run 8" in names else []))]
    for ax, (title, lines) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (1, 2)))
        for run, color, label in lines:
            d = data[data["run"] == run]
            ax.plot(d["step"], d["sd"], color=color, lw=2 if label else 1.4, alpha=1 if label else 0.75,
                    label=label, solid_capstyle="round", ls=(0, (4, 2)) if run == "Run 8" else "-")
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
        ax.set_xlabel("Training step (one pass, 4 questions per step)")
        ax.set_xlim(0, 360)
        ax.set_ylim(0, 0.56)
        ax.legend(loc="lower left", frameon=False, fontsize=9, labelcolor=INK)
    axes[0].set_ylabel("Spread of stated probabilities (std. dev.)")
    axes[1].text(358, 0.51, "all answers at 0 or 1", color=MUTED, fontsize=8.5, ha="right", va="bottom")
    axes[0].text(355, 0.035, "hedging toward one number", color=MUTED, fontsize=8.5, ha="right", va="bottom")
    fig.suptitle("Std scaling on: collapse to 0 or 1 once a cue is readable. Off: spread kept with key numbers, hedging without",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.text(0.01, 0.01, "Rolling window of 512 training answers (32 steps). Learning rate 1e-5, Brier reward, "
             "Llama 3.2 3B with LoRA rank 32.", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    out = os.path.join(fig_dir, "spread_2x2.png")
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print("wrote", os.path.abspath(out))


if __name__ == "__main__":
    main(sys.argv[1:-1], sys.argv[-1])
