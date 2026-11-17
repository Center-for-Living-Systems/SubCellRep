#!/usr/bin/env python3
"""
add_threshold_slide.py

Append a slide to pred_confidence_image_eff_b1b2v2_cp_lgbm.pptx showing:
  Top row  — 5 confusion matrices at thresholds 0.1 / 0.3 / 0.5 / 0.7 / 0.9
  Bottom   — metrics vs threshold curve (bal_acc, recall_ad, recall_noad, prec_ad)

Usage:
  python scripts/add_threshold_slide.py
"""
from __future__ import annotations
import io, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (confusion_matrix, balanced_accuracy_score,
                              recall_score, precision_score)

from pptx import Presentation
from pptx.util import Inches, Pt

REPO     = Path(__file__).resolve().parent.parent
PRED_DIR = REPO / "eval_results" / "patch_preds"
RES      = REPO / "results"
PPT_PATH = RES / "pred_confidence_image_eff_b1b2v2_cp_lgbm.pptx"

THRESHOLDS  = [0.1, 0.3, 0.5, 0.7, 0.9]
CURVE_THRS  = np.round(np.arange(0.05, 0.96, 0.05), 2)


def _load_dedup() -> pd.DataFrame:
    df = pd.read_csv(PRED_DIR / "image_eff_b1b2v2_lgbm.csv")
    df = df.rename(columns={"feature": "feat"})
    df = df[df["feat"] == "cp"].copy()
    return (df.groupby("fn")
              .agg(y_true=("y_true","first"),
                   prob_adhesion=("prob_adhesion","median"))
              .reset_index())


def _metrics_at(y_true, prob, thr):
    yp = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, yp).ravel()
    return dict(
        thr=thr, tp=tp, fp=fp, tn=tn, fn=fn,
        bal_acc   = balanced_accuracy_score(y_true, yp),
        recall_ad = recall_score(y_true, yp, pos_label=1, zero_division=0),
        recall_no = recall_score(y_true, yp, pos_label=0, zero_division=0),
        prec_ad   = precision_score(y_true, yp, pos_label=1, zero_division=0),
    )


def _build_figure(agg: pd.DataFrame) -> io.BytesIO:
    y_true = agg["y_true"].values
    prob   = agg["prob_adhesion"].values

    key_stats = [_metrics_at(y_true, prob, t) for t in THRESHOLDS]
    curve     = [_metrics_at(y_true, prob, t) for t in CURVE_THRS]

    n_thr = len(THRESHOLDS)
    fig   = plt.figure(figsize=(14, 7.2))
    gs    = gridspec.GridSpec(2, n_thr + 1,
                               height_ratios=[1.15, 1],
                               hspace=0.48, wspace=0.35)

    # ── Confusion matrices (top row) ─────────────────────────────────────────
    vmax = max(s["tp"] + s["fp"] for s in key_stats)   # for consistent colour scale
    best_thr = max(key_stats, key=lambda s: s["bal_acc"])["thr"]

    for i, s in enumerate(key_stats):
        ax = fig.add_subplot(gs[0, i])
        cm = np.array([[s["tn"], s["fp"]],
                       [s["fn"], s["tp"]]])

        # cell colours: TP/TN = Blues, FP/FN = Reds
        cell_colors = np.array([["#c6dcf0","#ffd0c0"],
                                 ["#ffd0c0","#3a85c7"]], dtype=object)
        for r in range(2):
            for c in range(2):
                val = cm[r, c]
                ax.add_patch(plt.Rectangle((c, 1-r), 1, 1,
                                            facecolor=cell_colors[r,c],
                                            edgecolor="white", lw=1.5))
                ax.text(c+0.5, 1-r+0.5, str(val),
                        ha="center", va="center",
                        fontsize=15, fontweight="bold",
                        color="#1a1a1a")

        # Axis labels
        ax.set_xlim(0, 2); ax.set_ylim(0, 2)
        ax.set_xticks([0.5, 1.5]); ax.set_yticks([0.5, 1.5])
        ax.set_xticklabels(["Pred\nNo-adh", "Pred\nAdhesion"], fontsize=7.5)
        ax.set_yticklabels(["True\nAdhesion", "True\nNo-adh"], fontsize=7.5)
        ax.tick_params(length=0)

        # Threshold label + bal_acc
        star = " ★" if s["thr"] == best_thr else ""
        ax.set_title(f"Threshold = {s['thr']}{star}\nbal acc = {s['bal_acc']:.3f}",
                     fontsize=9.5, fontweight="bold" if s["thr"] == best_thr else "normal",
                     color="#2060a0" if s["thr"] == best_thr else "black")

        # Row/col header on leftmost
        if i == 0:
            ax.set_ylabel("True label →", fontsize=7.5, labelpad=4)
        ax.set_xlabel("Predicted →", fontsize=7.5, labelpad=3)

    # ── Legend for cell colours ───────────────────────────────────────────────
    ax_leg = fig.add_subplot(gs[0, n_thr])
    ax_leg.set_axis_off()
    legend_items = [
        ("#3a85c7", "TP — true adhesion\ncorrectly predicted"),
        ("#c6dcf0", "TN — true no-adhesion\ncorrectly predicted"),
        ("#ffd0c0", "FP / FN — errors"),
    ]
    y0 = 0.85
    for color, label in legend_items:
        ax_leg.add_patch(plt.Rectangle((0.05, y0 - 0.08), 0.18, 0.12,
                                        facecolor=color, edgecolor="#888", lw=0.8,
                                        transform=ax_leg.transAxes))
        ax_leg.text(0.28, y0 - 0.02, label, fontsize=8.5,
                    va="center", transform=ax_leg.transAxes, color="#333333")
        y0 -= 0.22
    ax_leg.text(0.5, 0.15, "★ = best\nbal. acc.",
                ha="center", va="center", fontsize=9, color="#2060a0",
                fontweight="bold", transform=ax_leg.transAxes)

    # ── Metrics curve (bottom row, spanning all columns) ──────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    thrs_c     = [s["thr"]     for s in curve]
    bal_acc_c  = [s["bal_acc"] for s in curve]
    recall_ad  = [s["recall_ad"] for s in curve]
    recall_no  = [s["recall_no"] for s in curve]
    prec_ad    = [s["prec_ad"]   for s in curve]

    ax2.plot(thrs_c, bal_acc_c,  color="#333333",  lw=2.2, label="Balanced accuracy",  zorder=4)
    ax2.plot(thrs_c, recall_ad,  color="#2166ac",  lw=1.8, ls="--", label="Recall — adhesion",    zorder=3)
    ax2.plot(thrs_c, recall_no,  color="#92c5de",  lw=1.8, ls="--", label="Recall — no-adhesion", zorder=3)
    ax2.plot(thrs_c, prec_ad,    color="#d6604d",  lw=1.8, ls=":",  label="Precision — adhesion", zorder=3)

    # Mark the 5 thresholds
    for s in key_stats:
        ax2.axvline(s["thr"], color="#cccccc", lw=0.9, ls="--", zorder=1)
        ax2.scatter(s["thr"], s["bal_acc"], color="#333333", s=40, zorder=5)

    # Highlight best threshold
    ax2.axvline(best_thr, color="#2060a0", lw=1.4, ls="-", alpha=0.6, zorder=2)
    ax2.text(best_thr + 0.01, 0.62, f"best\n(thr={best_thr})",
             fontsize=8, color="#2060a0", va="bottom")

    ax2.set_xlabel("Classification threshold", fontsize=10)
    ax2.set_ylabel("Score", fontsize=10)
    ax2.set_xlim(0.03, 0.97)
    ax2.set_ylim(0.60, 1.01)
    ax2.legend(fontsize=9, loc="lower left", framealpha=0.85, ncol=2)
    ax2.spines[["top","right"]].set_visible(False)
    ax2.set_title("Metrics vs classification threshold", fontsize=11, fontweight="bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


def main():
    print(f"Loading {PPT_PATH.name} ...")
    prs = Presentation(str(PPT_PATH))
    print(f"  {len(prs.slides)} slides")

    print("Computing confusion matrices ...")
    agg = _load_dedup()
    print(f"  {len(agg)} patches | adhesion={agg['y_true'].sum()} "
          f"no-adhesion={(agg['y_true']==0).sum()}")

    buf = _build_figure(agg)

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    for ph in list(slide.placeholders):
        ph.element.getparent().remove(ph.element)
    W, H = prs.slide_width, prs.slide_height

    txb = slide.shapes.add_textbox(Inches(0.2), Inches(0.06),
                                    W - Inches(0.4), Inches(0.42))
    tf  = txb.text_frame
    tf.text = "Confusion matrices at different classification thresholds  (image eff B1/B2v2, CP, LightGBM)"
    tf.paragraphs[0].runs[0].font.size = Pt(18)
    tf.paragraphs[0].runs[0].font.bold = True

    slide.shapes.add_picture(buf, Inches(0.15), Inches(0.52),
                              W - Inches(0.3), H - Inches(0.62))

    prs.save(str(PPT_PATH))
    sz = PPT_PATH.stat().st_size // 1024
    print(f"\nSaved → {PPT_PATH}  ({sz} KB, {len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
