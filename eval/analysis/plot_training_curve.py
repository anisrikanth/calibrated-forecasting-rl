import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

c = pd.read_csv(sys.argv[1]); out = sys.argv[2]
INK, INK2, GRID, BLUE, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK, "figure.facecolor": SURF,
                     "axes.facecolor": SURF})
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
panels = [("brier", "Brier score on fresh training questions (lower is better)", 0.25, "always 0.5"),
          ("gap", "Gap: avg. forecast on yes minus on no (higher is better)", 0.0, "no discrimination")]
for ax, (col, title, ref, ref_label) in zip(axes, panels):
    ax.axhline(ref, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(c["step"].iloc[-1], ref, f" {ref_label}", va="bottom", ha="right", fontsize=8.5, color=INK2)
    ax.plot(c["step"], c[col], color=BLUE, lw=2, solid_capstyle="round")
    ax.set_title(title, fontsize=10, loc="left", pad=8)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_xlabel("Training step (4 questions per step)")
    ax.annotate(f"{c[col].iloc[0]:.3f}", (c['step'].iloc[0], c[col].iloc[0]), xytext=(4, 6),
                textcoords="offset points", fontsize=8.5, color=INK)
    ax.annotate(f"{c[col].iloc[-1]:.3f}", (c['step'].iloc[-1], c[col].iloc[-1]), xytext=(-4, 6),
                textcoords="offset points", fontsize=8.5, color=INK, ha="right")
fig.suptitle("Llama 3.2 3B, one GRPO pass with the Brier reward: rolling 512 answers", x=0.01, ha="left",
             fontsize=11.5, color=INK, fontweight="bold")
fig.tight_layout()
fig.savefig(out, dpi=170)
print("saved", out)
