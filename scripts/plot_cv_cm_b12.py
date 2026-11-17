#!/usr/bin/env python3
"""
plot_cv_cm_b12.py

Confusion matrices for B1+B2 training source only (2-row: test B1 / test B2),
for CP, Ilastik, and SupCon-AE. Font size 14 throughout.

Outputs:
  results/cv_b1b2/cv_cm_b12_CP.png
  results/cv_b1b2/cv_cm_b12_IL.png
  results/cv_b1b2/cv_cm_b12_AE.png
"""
import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv_common import FA4

OUT = Path(__file__).resolve().parents[1] / "results" / "cv_b1b2"
oof = pd.read_csv(OUT / "oof.csv"); oof = oof[oof.seed == 0]

SHORT = {
    "Nascent Adhesion":   "nascent",
    "focal complex":      "f.complex",
    "focal adhesion":     "f.adhesion",
    "fibrillar adhesion": "fibrillar",
}
FEAT_MAP = {
    "CP":  "CP",
    "IL":  "IL",
    "AE":  "AE_b12",
}
SRC = "B1+B2"
FS = 14

for key, rep in FEAT_MAP.items():
    # Pool B1 and B2 test patches together
    d = oof[(oof.rep == rep) & (oof.src == SRC)]
    if d.empty:
        print(f"No data for {key}, skipping"); continue

    cm  = confusion_matrix(d.y, d.pred, labels=FA4)
    cmn = cm / cm.sum(1, keepdims=True)
    bal = balanced_accuracy_score(d.y, d.pred)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for a in range(4):
        for b in range(4):
            ax.text(b, a, f"{cmn[a, b]:.2f}\n({cm[a, b]})",
                    ha="center", va="center", fontsize=FS,
                    color="white" if cmn[a, b] > 0.55 else "black")
    ax.set_xticks(range(4))
    ax.set_xticklabels([SHORT[c] for c in FA4], rotation=30, ha="right", fontsize=FS)
    ax.set_yticks(range(4))
    ax.set_yticklabels([SHORT[c] for c in FA4], fontsize=FS)
    ax.set_title(f"{key} — train B1+B2, test B1+B2  bal.acc {bal:.3f}\n"
                 f"out-of-fold CV (frame-grouped, seed 0)", fontsize=FS)
    ax.set_ylabel("true", fontsize=FS)
    ax.set_xlabel("predicted", fontsize=FS)

    plt.tight_layout()
    out_path = OUT / f"cv_cm_b12_{key}.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")
