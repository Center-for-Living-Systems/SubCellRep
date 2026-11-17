#!/usr/bin/env python3
"""
add_confidence_zone_slide.py

Append a slide showing performance split by confidence zone:
  High-confidence zone: prob < 0.3  or  prob > 0.7
  Low-confidence zone:  0.3 ≤ prob ≤ 0.7

Layout:
  Left   — probability distribution with zone shading + patch count
  Middle — confusion matrix + metrics for high-confidence zone
  Right  — confusion matrix + metrics for low-confidence zone
  Bottom — key takeaway box
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
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
from sklearn.metrics import (confusion_matrix, balanced_accuracy_score,
                              recall_score, precision_score)

from pptx import Presentation
from pptx.util import Inches, Pt

REPO     = Path(__file__).resolve().parent.parent
PRED_DIR = REPO / "eval_results" / "patch_preds"
RES      = REPO / "results"
PPT_PATH = RES / "pred_confidence_image_eff_b1b2v2_cp_lgbm.pptx"

LOW_LO, LOW_HI = 0.3, 0.7   # uncertain zone boundaries


def _load() -> pd.DataFrame:
    df = pd.read_csv(PRED_DIR / "image_eff_b1b2v2_lgbm.csv")
    df = df.rename(columns={"feature": "feat"})
    df = df[df["feat"] == "cp"].copy()
    agg = (df.groupby("fn")
             .agg(y_true=("y_true","first"),
                  prob_adhesion=("prob_adhesion","median"))
             .reset_index())
    agg["y_pred"] = (agg["prob_adhesion"] >= 0.5).astype(int)
    agg["zone"]   = np.where(
        (agg["prob_adhesion"] >= LOW_LO) & (agg["prob_adhesion"] <= LOW_HI),
        "low", "high")
    return agg


def _zone_metrics(sub: pd.DataFrame) -> dict:
    y_true = sub["y_true"].values
    y_pred = sub["y_pred"].values
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    errors = fp + fn
    return dict(
        tp=tp, fp=fp, tn=tn, fn=fn,
        n=len(sub), n_ad=int(y_true.sum()), n_no=int((y_true==0).sum()),
        errors=errors, error_pct=errors/len(sub)*100,
        bal_acc  = balanced_accuracy_score(y_true, y_pred),
        recall_ad= recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        recall_no= recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        prec_ad  = precision_score(y_true, y_pred, pos_label=1, zero_division=0),
    )


def _draw_cm(ax, m: dict, title: str, title_color: str):
    """Draw a 2×2 confusion matrix on ax."""
    cm = np.array([[m["tn"], m["fp"]],
                   [m["fn"], m["tp"]]])
    cell_colors = [["#c6dcf0","#ffd0c0"],
                   ["#ffd0c0","#3a85c7"]]
    labels = [["TN","FP"],["FN","TP"]]

    for r in range(2):
        for c in range(2):
            ax.add_patch(plt.Rectangle(
                (c, 1-r), 1, 1,
                facecolor=cell_colors[r][c],
                edgecolor="white", lw=2))
            ax.text(c+0.5, 1-r+0.62, labels[r][c],
                    ha="center", va="center", fontsize=9,
                    color="#666666", fontweight="bold")
            ax.text(c+0.5, 1-r+0.35, str(cm[r,c]),
                    ha="center", va="center", fontsize=17,
                    fontweight="bold", color="#1a1a1a")

    ax.set_xlim(0,2); ax.set_ylim(0,2)
    ax.set_xticks([0.5,1.5]); ax.set_yticks([0.5,1.5])
    ax.set_xticklabels(["Pred\nNo-adh","Pred\nAdhesion"], fontsize=8)
    ax.set_yticklabels(["True\nAdhesion","True\nNo-adh"], fontsize=8)
    ax.tick_params(length=0)
    ax.set_title(title, fontsize=11, fontweight="bold", color=title_color, pad=6)


def _draw_metrics(ax, m: dict, zone_color: str):
    """Draw a metrics text panel on ax."""
    ax.set_axis_off()
    lines = [
        (f"n = {m['n']} patches", 10, True,  "#222222"),
        (f"Adhesion: {m['n_ad']}  |  No-adh: {m['n_no']}", 8.5, False, "#555555"),
        ("", 5, False, "#ffffff"),
        (f"Balanced accuracy:  {m['bal_acc']:.3f}", 10, True,  zone_color),
        (f"Recall (adhesion):  {m['recall_ad']:.3f}", 9.5, False, "#2166ac"),
        (f"Recall (no-adh):    {m['recall_no']:.3f}", 9.5, False, "#2166ac"),
        (f"Precision (adh):    {m['prec_ad']:.3f}", 9.5, False, "#d6604d"),
        ("", 5, False, "#ffffff"),
        (f"Errors: {m['errors']} / {m['n']}  ({m['error_pct']:.1f}%)", 10, True,
         "#cc3300" if m["error_pct"] > 20 else "#226600"),
    ]
    y = 0.97
    for text, size, bold, color in lines:
        ax.text(0.05, y, text, transform=ax.transAxes,
                fontsize=size, fontweight="bold" if bold else "normal",
                color=color, va="top")
        y -= 0.115 if size >= 10 else 0.09
    # border
    for spine in ["top","bottom","left","right"]:
        ax.spines[spine].set_visible(False)


def _build_figure(agg: pd.DataFrame) -> io.BytesIO:
    high = agg[agg["zone"] == "high"]
    low  = agg[agg["zone"] == "low"]
    mh   = _zone_metrics(high)
    ml   = _zone_metrics(low)

    fig  = plt.figure(figsize=(14, 6.8))
    gs   = gridspec.GridSpec(3, 3,
                              height_ratios=[0.12, 1.1, 0.72],
                              width_ratios=[1.1, 1, 1],
                              hspace=0.10, wspace=0.35)

    # ── Column headers ────────────────────────────────────────────────────────
    for col_i, (label, color) in enumerate([
        ("Probability distribution", "#333333"),
        ("High-confidence zone\n(prob < 0.3  or  prob > 0.7)", "#226600"),
        ("Low-confidence zone\n(prob 0.3 – 0.7)", "#cc5500"),
    ]):
        ax_hdr = fig.add_subplot(gs[0, col_i])
        ax_hdr.set_axis_off()
        ax_hdr.text(0.5, 0.5, label, ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color,
                    transform=ax_hdr.transAxes)

    # ── Left: probability distribution with zone shading ─────────────────────
    ax_dist = fig.add_subplot(gs[1:, 0])
    prob_all = agg["prob_adhesion"].values
    xs = np.linspace(0, 1, 300)

    # KDE per true label
    for label, vals, color, lbl in [
        (1, agg[agg["y_true"]==1]["prob_adhesion"].values, "#2166ac", "Adhesion"),
        (0, agg[agg["y_true"]==0]["prob_adhesion"].values, "#d6604d", "No-adhesion"),
    ]:
        kde = gaussian_kde(vals, bw_method=0.12)
        ys  = kde(xs)
        ax_dist.plot(xs, ys, color=color, lw=2, label=lbl)
        ax_dist.fill_between(xs, ys, alpha=0.15, color=color)

    ymax = ax_dist.get_ylim()[1]

    # Zone shading
    ax_dist.axvspan(LOW_LO, LOW_HI, alpha=0.12, color="#ff8800",
                    label=f"Low-conf zone (n={len(low)})")
    ax_dist.axvspan(0, LOW_LO,   alpha=0.07, color="#00aa44")
    ax_dist.axvspan(LOW_HI, 1.0, alpha=0.07, color="#00aa44")

    # Zone boundaries + labels
    for v in [LOW_LO, LOW_HI]:
        ax_dist.axvline(v, color="#ff8800", lw=1.4, ls="--")
    ax_dist.text(LOW_LO/2,      ymax*0.92, f"High\nn={len(high[high['prob_adhesion']<LOW_LO])}",
                 ha="center", fontsize=8, color="#226600", fontweight="bold")
    ax_dist.text((LOW_LO+LOW_HI)/2, ymax*0.92, f"Low\nn={len(low)}",
                 ha="center", fontsize=8, color="#cc5500", fontweight="bold")
    ax_dist.text((LOW_HI+1)/2,  ymax*0.92, f"High\nn={len(high[high['prob_adhesion']>LOW_HI])}",
                 ha="center", fontsize=8, color="#226600", fontweight="bold")

    # Decision threshold
    ax_dist.axvline(0.5, color="#333333", lw=1.0, ls=":", alpha=0.7)
    ax_dist.text(0.51, ymax*0.50, "decision\nthr=0.5", fontsize=7.5,
                 color="#333333", va="center")

    ax_dist.set_xlabel("P(adhesion)", fontsize=10)
    ax_dist.set_ylabel("Density", fontsize=10)
    ax_dist.set_xlim(0, 1)
    ax_dist.legend(fontsize=8.5, loc="upper center", framealpha=0.8)
    ax_dist.spines[["top","right"]].set_visible(False)

    # Patch count bar below distribution
    total = len(agg)
    ax_bar = ax_dist.inset_axes([0, -0.22, 1, 0.14])
    ax_bar.barh(0, len(high)/total, color="#00aa44", alpha=0.65,
                label=f"High conf ({len(high)/total*100:.0f}%)")
    ax_bar.barh(0, len(low)/total, left=len(high)/total,
                color="#ff8800", alpha=0.65,
                label=f"Low conf ({len(low)/total*100:.0f}%)")
    ax_bar.set_xlim(0,1); ax_bar.set_ylim(-0.5,0.5)
    ax_bar.set_axis_off()
    ax_bar.text(len(high)/total/2, 0,
                f"{len(high)/total*100:.0f}% high conf\n({len(high)} patches)",
                ha="center", va="center", fontsize=8.5, fontweight="bold", color="#226600")
    ax_bar.text(len(high)/total + len(low)/total/2, 0,
                f"{len(low)/total*100:.0f}% low conf\n({len(low)} patches)",
                ha="center", va="center", fontsize=8.5, fontweight="bold", color="#cc5500")

    # ── Middle: high-conf confusion matrix + metrics ──────────────────────────
    ax_cm_h = fig.add_subplot(gs[1, 1])
    _draw_cm(ax_cm_h, mh, "High-confidence zone", "#226600")

    ax_met_h = fig.add_subplot(gs[2, 1])
    _draw_metrics(ax_met_h, mh, "#226600")

    # ── Right: low-conf confusion matrix + metrics ────────────────────────────
    ax_cm_l = fig.add_subplot(gs[1, 2])
    _draw_cm(ax_cm_l, ml, "Low-confidence zone", "#cc5500")

    ax_met_l = fig.add_subplot(gs[2, 2])
    _draw_metrics(ax_met_l, ml, "#cc5500")

    fig.suptitle(
        "Performance split by confidence zone  (decision threshold = 0.5  |  "
        "image eff B1/B2v2, CP, LightGBM)",
        fontsize=12, fontweight="bold", y=1.01)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


def main():
    print(f"Loading {PPT_PATH.name} ...")
    prs = Presentation(str(PPT_PATH))
    print(f"  {len(prs.slides)} slides")

    agg = _load()
    print(f"  {len(agg)} patches  |  "
          f"high-conf: {(agg['zone']=='high').sum()}  "
          f"low-conf: {(agg['zone']=='low').sum()}")

    buf = _build_figure(agg)

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    for ph in list(slide.placeholders):
        ph.element.getparent().remove(ph.element)
    W, H = prs.slide_width, prs.slide_height

    slide.shapes.add_picture(buf, Inches(0.10), Inches(0.10),
                              W - Inches(0.20), H - Inches(0.20))

    prs.save(str(PPT_PATH))
    sz = PPT_PATH.stat().st_size // 1024
    print(f"\nSaved → {PPT_PATH}  ({sz} KB, {len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
