#!/usr/bin/env python3
"""
Plot FA4 class counts by condition (control vs ycomp).
Output: results/fa4_condition_counts.png
"""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABDIR = Path("/net/projects/CLS/lding/data/fa_data_analysis/labelling/stage2_rebuild")
OUT    = Path(__file__).resolve().parents[1] / "results"

FA4 = ["Nascent Adhesion", "focal complex", "focal adhesion", "fibrillar adhesion"]
COLORS = {"control": "#4C7DC4", "ycomp": "#E05C2A"}
SHORT  = {
    "Nascent Adhesion":   "Nascent\nAdhesion",
    "focal complex":      "Focal\nComplex",
    "focal adhesion":     "Focal\nAdhesion",
    "fibrillar adhesion": "Fibrillar\nAdhesion",
}

labels = pd.read_csv(LABDIR / "combined_labels.csv")
labels = labels[labels["label"].isin(FA4)]

counts = labels.groupby(["condition", "label"]).size().unstack(fill_value=0)
counts = counts[FA4]  # fix column order

x      = np.arange(len(FA4))
width  = 0.35
fig, ax = plt.subplots(figsize=(7, 4.5))

for i, cond in enumerate(["control", "ycomp"]):
    vals = counts.loc[cond, FA4].values if cond in counts.index else np.zeros(len(FA4))
    bars = ax.bar(x + (i - 0.5) * width, vals, width, label=cond.capitalize(),
                  color=COLORS[cond], edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(int(v)), ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([SHORT[c] for c in FA4], fontsize=12)
ax.set_ylabel("Number of labelled patches", fontsize=12)
ax.set_title("FA4 labelled patches by class and condition (B1+B2, DS1)", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.set_ylim(0, max(counts.values.max() * 1.18, 50))
ax.grid(axis="y", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
out_path = OUT / "fa4_condition_counts.png"
fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out_path}")
