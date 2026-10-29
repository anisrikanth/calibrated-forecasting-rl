"""Prompt ablation figure: AUC and Brier of the untrained 3B under each prompt (validation econ),
with 95% event-level bootstrap intervals, against the simple rules on the same evidence.

Usage: python eval/analysis/plot_prompt_ablation.py eval/analysis/prompt_ablation_scores.csv
Writes eval/figures/prompt_ablation.png.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

INK, MUTED, GRID, SURFACE, MARK = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6"
ORDER = [("plain", "Recent results only"),
         ("rule", "+ the YES rule"),
         ("rule_latest", "+ latest result vs threshold"),
         ("rule_latest_hits", "+ count of recent results that would resolve YES"),
         ("full", "+ change and range (full key numbers)"),
         ("rule_hits", "YES rule + count only"),
         ("estimate_first", "Estimate the number first, no key numbers")]


def main(path):
    t = pd.read_csv(path).set_index("prompt")
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE, sharey=True)
    ys = list(range(len(ORDER)))[::-1]
    # a gap between the cumulative ladder (first 5) and the two alternatives
    ys = [y + (0.6 if i < 5 else 0) for i, y in enumerate(ys)]
    for ax, (metric, title, refs, xlim) in zip(axes, [
        ("auc", "AUC (higher is better)", [("logistic", "logistic"), (0.5, "no ordering")], (0.4, 0.95)),
        ("brier", "Brier score (lower is better)", [("logistic", "logistic"), ("base_rate", "base rate")], (0.14, 0.31)),
    ]):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        for key, label in refs:
            v = key if isinstance(key, float) else t.loc[key, metric]
            ax.axvline(v, color=MUTED, lw=1, ls=(0, (1, 2)) if isinstance(key, float) else (0, (3, 3)))
            text = f"{label} {v:.3f}" if not isinstance(key, float) else f"{label} {v:.1f}"
            if metric == "auc" and key == "logistic":
                text = f"logistic {v:.3f} (counting rule {t.loc['recent_hits', 'auc']:.3f})"
            ax.text(v, max(ys) + 0.75, text, color=MUTED, fontsize=8.5, ha="center", va="bottom")
        for (key, _), y in zip(ORDER, ys):
            r = t.loc[key]
            ax.plot([r[f"{metric}_low"], r[f"{metric}_high"]], [y, y], color=MARK, lw=2, solid_capstyle="round")
            ax.plot(r[metric], y, "o", color=MARK, ms=7, mec=SURFACE, mew=1.5)
            ax.text(r[f"{metric}_high"] + (xlim[1] - xlim[0]) * 0.015, y, f"{r[metric]:.3f}", color=INK,
                    fontsize=8.5, va="center")
        ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=22)
        ax.set_xlim(*xlim)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([label for _, label in ORDER], color=INK)
    fig.suptitle("Untrained 3B on 410 validation econ questions: the YES rule, the latest result and the count do the work",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.text(0.01, 0.01, "Each prompt shows the recent results; the first five add key-number lines one at a time. "
             "Dots: 4 answers averaged per question. Lines: 95% intervals, resampling whole releases.",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    out = os.path.join(os.path.dirname(__file__), "..", "figures", "prompt_ablation.png")
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print("wrote", os.path.abspath(out))


if __name__ == "__main__":
    main(sys.argv[1])
