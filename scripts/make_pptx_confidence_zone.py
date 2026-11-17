#!/usr/bin/env python3
"""
make_pptx_confidence_zone.py

Editable version of the confidence zone slide (slide 13 of
pred_confidence_image_eff_b1b2v2_cp_lgbm.pptx).

Layout (3 columns):
  Left   — KDE probability distribution [matplotlib image — intrinsically graphical]
  Middle — High-confidence zone: native 2×2 CM shapes + editable metric text
  Right  — Low-confidence zone:  native 2×2 CM shapes + editable metric text

Output: results/confidence_zone_editable.pptx
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
from scipy.stats import gaussian_kde
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                              recall_score, precision_score)

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

REPO     = Path(__file__).resolve().parents[1]
PRED_DIR = REPO / "eval_results" / "patch_preds"
OUT      = REPO / "results" / "confidence_zone_editable.pptx"

ZONE_LO, ZONE_HI = 0.3, 0.7
SLIDE_W, SLIDE_H = 13.33, 7.5

C_HIGH   = "#226600"
C_LOW    = "#cc5500"
C_BLUE   = "#2166ac"
C_RED    = "#d6604d"
C_DARK   = "#1a2a3a"
C_GREY   = "#555555"

# CM cell colors
CM_COLORS = {  # (row, col): fill hex
    (0, 0): "#c6dcf0",  # TN
    (0, 1): "#ffd0c0",  # FP
    (1, 0): "#ffd0c0",  # FN
    (1, 1): "#3a85c7",  # TP
}
CM_LABELS = {(0,0):"TN", (0,1):"FP", (1,0):"FN", (1,1):"TP"}


def _hex(h):
    h = h.lstrip("#")
    return RGBColor(int(h[0:2],16), int(h[2:4],16), int(h[4:6],16))


def _txt(slide, text, left, top, w, h, size=11, bold=False,
         color="#222222", align=PP_ALIGN.LEFT, italic=False):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    p  = tf.paragraphs[0]
    p.alignment = align
    r  = p.add_run()
    r.text = text
    r.font.size  = Pt(size)
    r.font.bold  = bold
    r.font.italic = italic
    r.font.color.rgb = _hex(color)
    return tb


def _rect(slide, left, top, w, h, fill, line=None):
    from pptx.enum.shapes import MSO_SHAPE_TYPE
    shp = slide.shapes.add_shape(
        1, Inches(left), Inches(top), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = _hex(fill)
    if line:
        shp.line.color.rgb = _hex(line)
        shp.line.width = Pt(1.5)
    else:
        shp.line.fill.background()
    return shp


def _paste(slide, img_bytes, left, top, w, h):
    img_bytes.seek(0)
    slide.shapes.add_picture(img_bytes, Inches(left), Inches(top),
                             Inches(w), Inches(h))


def _load():
    df = pd.read_csv(PRED_DIR / "image_eff_b1b2v2_lgbm.csv")
    df = df.rename(columns={"feature": "feat"})
    df = df[df["feat"] == "cp"].copy()
    agg = (df.groupby("fn")
             .agg(y_true=("y_true","first"),
                  prob_adhesion=("prob_adhesion","median"))
             .reset_index())
    agg["y_pred"] = (agg["prob_adhesion"] >= 0.5).astype(int)
    agg["zone"]   = np.where(
        (agg["prob_adhesion"] >= ZONE_LO) & (agg["prob_adhesion"] <= ZONE_HI),
        "low", "high")
    return agg


def _zone_metrics(sub):
    y_true, y_pred = sub["y_true"].values, sub["y_pred"].values
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    errors = fp + fn
    return dict(
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
        n=len(sub), n_ad=int(y_true.sum()), n_no=int((y_true==0).sum()),
        errors=errors, error_pct=errors/len(sub)*100,
        bal_acc   = balanced_accuracy_score(y_true, y_pred),
        recall_ad = recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        recall_no = recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        prec_ad   = precision_score(y_true, y_pred, pos_label=1, zero_division=0),
    )


def _kde_image(agg, high, low):
    """Render the KDE distribution panel as a PNG buffer."""
    fig, ax = plt.subplots(figsize=(4.2, 5.5))
    fig.patch.set_facecolor("white")
    xs = np.linspace(0, 1, 300)
    for y_val, color, lbl in [(1, C_BLUE, "Adhesion"), (0, C_RED, "No-adhesion")]:
        vals = agg[agg["y_true"] == y_val]["prob_adhesion"].values
        kde = gaussian_kde(vals, bw_method=0.12)
        ys  = kde(xs)
        ax.plot(xs, ys, color=color, lw=2, label=lbl)
        ax.fill_between(xs, ys, alpha=0.15, color=color)
    ymax = ax.get_ylim()[1]
    ax.axvspan(ZONE_LO, ZONE_HI, alpha=0.12, color="#ff8800")
    ax.axvspan(0, ZONE_LO,       alpha=0.07, color="#00aa44")
    ax.axvspan(ZONE_HI, 1.0,     alpha=0.07, color="#00aa44")
    for v in [ZONE_LO, ZONE_HI]:
        ax.axvline(v, color="#ff8800", lw=1.4, ls="--")
    n_hi_lo = int((high["prob_adhesion"] < ZONE_LO).sum())
    n_hi_hi = int((high["prob_adhesion"] > ZONE_HI).sum())
    ax.text(ZONE_LO/2,          ymax*0.90, f"High\nn={n_hi_lo}", ha="center",
            fontsize=8, color=C_HIGH, fontweight="bold")
    ax.text((ZONE_LO+ZONE_HI)/2, ymax*0.90, f"Low\nn={len(low)}", ha="center",
            fontsize=8, color=C_LOW, fontweight="bold")
    ax.text((ZONE_HI+1)/2,       ymax*0.90, f"High\nn={n_hi_hi}", ha="center",
            fontsize=8, color=C_HIGH, fontweight="bold")
    ax.axvline(0.5, color="#333333", lw=1.0, ls=":", alpha=0.7)
    ax.text(0.51, ymax*0.50, "decision\nthr=0.5", fontsize=7.5,
            color="#333333", va="center")
    ax.set_xlabel("P(adhesion)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=9, loc="upper center", framealpha=0.8)
    ax.spines[["top","right"]].set_visible(False)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    plt.close(fig)
    return buf


def _draw_cm(slide, m, x0, y0, cell_w=0.95, cell_h=0.80):
    """Native PPTX 2×2 confusion matrix starting at (x0, y0)."""
    cm_vals = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
    for r in range(2):
        for c in range(2):
            fill = CM_COLORS[(r, c)]
            lbl  = CM_LABELS[(r, c)]
            val  = cm_vals[r][c]
            cx   = x0 + c * cell_w
            cy   = y0 + r * cell_h
            _rect(slide, cx, cy, cell_w, cell_h, fill, line="#ffffff")
            # label (TN/FP/FN/TP) small at top
            _txt(slide, lbl, cx, cy + 0.04, cell_w, 0.25,
                 size=8, bold=True, color="#666666", align=PP_ALIGN.CENTER)
            # count large in middle
            _txt(slide, str(val), cx, cy + 0.28, cell_w, 0.45,
                 size=20, bold=True, color="#1a1a1a", align=PP_ALIGN.CENTER)

    # Axis labels
    _txt(slide, "Pred No-adh", x0,            y0 + 2*cell_h + 0.02, cell_w, 0.22,
         size=8, color=C_GREY, align=PP_ALIGN.CENTER)
    _txt(slide, "Pred Adh",    x0 + cell_w,   y0 + 2*cell_h + 0.02, cell_w, 0.22,
         size=8, color=C_GREY, align=PP_ALIGN.CENTER)
    _txt(slide, "True\nNo-adh", x0 - 0.62, y0,          0.60, cell_h,
         size=8, color=C_GREY, align=PP_ALIGN.RIGHT)
    _txt(slide, "True\nAdh",    x0 - 0.62, y0 + cell_h, 0.60, cell_h,
         size=8, color=C_GREY, align=PP_ALIGN.RIGHT)


def _draw_metrics(slide, m, x0, y0, w, zone_color):
    """Native PPTX metrics text box."""
    lines = [
        (f"n = {m['n']} patches  (adhesion: {m['n_ad']}  |  no-adh: {m['n_no']})",
         10, True,  C_DARK),
        ("", 5, False, "#ffffff"),
        (f"Balanced accuracy:   {m['bal_acc']:.3f}",  11, True,  zone_color),
        (f"Recall (adhesion):   {m['recall_ad']:.3f}", 10, False, C_BLUE),
        (f"Recall (no-adh):     {m['recall_no']:.3f}", 10, False, C_BLUE),
        (f"Precision (adh):     {m['prec_ad']:.3f}",  10, False, C_RED),
        ("", 5, False, "#ffffff"),
        (f"Errors: {m['errors']} / {m['n']}  ({m['error_pct']:.1f}%)",
         11, True, "#cc3300" if m["error_pct"] > 20 else "#226600"),
    ]
    tb = slide.shapes.add_textbox(Inches(x0), Inches(y0), Inches(w), Inches(2.4))
    tf = tb.text_frame
    tf.word_wrap = True
    first = True
    for text, size, bold, color in lines:
        if first:
            p = tf.paragraphs[0]; first = False
        else:
            p = tf.add_paragraph()
        run = p.add_run()
        run.text = text
        run.font.size  = Pt(size)
        run.font.bold  = bold
        run.font.color.rgb = _hex(color)


def build_slide(prs, agg, high, low, mh, ml):
    s = prs.slides.add_slide(prs.slide_layouts[6])

    # Background
    _rect(s, 0, 0, SLIDE_W, SLIDE_H, "#ffffff")

    # Title
    _txt(s,
         "Performance split by confidence zone  "
         "(decision threshold = 0.5  |  image eff B1/B2v2, CP, LightGBM)",
         0.25, 0.06, SLIDE_W - 0.5, 0.38,
         size=13, bold=True, color=C_DARK, align=PP_ALIGN.LEFT)
    # Divider
    _rect(s, 0.25, 0.46, SLIDE_W - 0.5, 0.014, "#cccccc")

    BODY_TOP = 0.52

    # ── Column headers ────────────────────────────────────────────────────────
    _txt(s, "Probability distribution", 0.25, BODY_TOP, 4.1, 0.32,
         size=11, bold=True, color="#333333", align=PP_ALIGN.CENTER)
    _txt(s, "High-confidence zone\n(prob < 0.3  or  prob > 0.7)",
         4.55, BODY_TOP, 4.1, 0.40,
         size=11, bold=True, color=C_HIGH, align=PP_ALIGN.CENTER)
    _txt(s, "Low-confidence zone\n(prob 0.3 – 0.7)",
         8.85, BODY_TOP, 4.1, 0.40,
         size=11, bold=True, color=C_LOW, align=PP_ALIGN.CENTER)

    CONTENT_TOP = BODY_TOP + 0.44

    # ── Left: KDE image ───────────────────────────────────────────────────────
    kde_buf = _kde_image(agg, high, low)
    _paste(s, kde_buf, 0.25, CONTENT_TOP, 4.1, 6.5 - CONTENT_TOP)

    # ── Middle: high-confidence zone ──────────────────────────────────────────
    CM_TOP = CONTENT_TOP + 0.10
    _draw_cm(s, mh, x0=5.20, y0=CM_TOP)
    _draw_metrics(s, mh, x0=4.55, y0=CM_TOP + 1.75, w=4.1, zone_color=C_HIGH)

    # ── Right: low-confidence zone ────────────────────────────────────────────
    _draw_cm(s, ml, x0=9.50, y0=CM_TOP)
    _draw_metrics(s, ml, x0=8.85, y0=CM_TOP + 1.75, w=4.1, zone_color=C_LOW)

    # ── Summary bar at bottom ─────────────────────────────────────────────────
    total = len(agg)
    pct_h = len(high) / total * 100
    pct_l = len(low)  / total * 100
    _txt(s,
         f"High-confidence: {len(high)} patches ({pct_h:.0f}%)   ·   "
         f"Low-confidence: {len(low)} patches ({pct_l:.0f}%)",
         0.25, 6.95, SLIDE_W - 0.5, 0.35,
         size=11, color=C_GREY, align=PP_ALIGN.CENTER)


def main():
    print("Loading predictions …")
    agg  = _load()
    high = agg[agg["zone"] == "high"]
    low  = agg[agg["zone"] == "low"]
    mh   = _zone_metrics(high)
    ml   = _zone_metrics(low)
    print(f"  Total: {len(agg)}  |  high: {len(high)}  |  low: {len(low)}")

    prs = Presentation()
    prs.slide_width  = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)

    build_slide(prs, agg, high, low, mh, ml)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    print(f"Saved → {OUT}")


if __name__ == "__main__":
    main()
