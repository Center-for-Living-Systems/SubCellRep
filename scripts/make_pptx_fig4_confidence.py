#!/usr/bin/env python3
"""
make_pptx_fig4_confidence.py
==============================
Paper Figure 4: Using prediction scores to identify reliable patches.
One slide per panel, Panels B and C as native editable PPTX charts.

Panels
------
A  Score distributions for TP / TN / FP / FN outcomes  [matplotlib image]
B  Patch fraction and accuracy in low / intermediate / high score zones  [native chart]
C  Accuracy vs retained fraction as confidence boundary sweeps  [native chart]
D  Example patches: confidently correct, ambiguous, confidently wrong  [matplotlib image]

Output: results/fig4_confidence.pptx
"""
from __future__ import annotations

import csv
import io
import random
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import gaussian_kde

from pptx import Presentation
from pptx.chart.data import ChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO    = Path(__file__).resolve().parents[1]
DATA    = Path("/net/projects/CLS/lding/data/fa_data_analysis")
H5_CTRL = DATA / "ae_results/patches/cio_rb/vinc/vinc_control_label.h5"
H5_YCOMP= DATA / "ae_results/patches/cio_rb/vinc/vinc_ycomp_label.h5"
PRED_CSV= REPO / "eval_results/patch_preds/le_union_cp_lgbm.csv"
OUT     = REPO / "results" / "fig4_confidence.pptx"

SLIDE_W, SLIDE_H = 13.33, 7.5
ZONE_LO, ZONE_HI = 0.3, 0.7

# ── Colours ────────────────────────────────────────────────────────────────────
C_TP   = "#2ca02c"
C_TN   = "#9467bd"
C_FP   = "#d62728"
C_FN   = "#ff7f0e"
C_LOW  = "#9467bd"
C_MID  = "#aaaaaa"
C_HIGH = "#2ca02c"
C_DARK = "#1a2a3a"
C_GREY = "#888888"


def _hex2rgb(h):
    h = h.lstrip("#")
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def _txt(slide, text, left, top, width, height,
         size=11, bold=False, color="#222222", align=PP_ALIGN.LEFT, italic=False):
    tb = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = _hex2rgb(color)
    return tb


def _rect(slide, left, top, width, height, fill_hex):
    shape = slide.shapes.add_shape(
        1, Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = _hex2rgb(fill_hex)
    shape.line.fill.background()
    return shape


def _header(slide, title, subtitle=""):
    _rect(slide, 0, 0, SLIDE_W, SLIDE_H, "#ffffff")
    _txt(slide, title, 0.3, 0.10, SLIDE_W - 0.6, 0.40,
         size=14, bold=True, color=C_DARK)
    if subtitle:
        _txt(slide, subtitle, 0.3, 0.50, SLIDE_W - 0.6, 0.28,
             size=10, italic=True, color=C_GREY)


def _paste_pil(slide, img, left, top, width, height):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    slide.shapes.add_picture(buf, Inches(left), Inches(top),
                             Inches(width), Inches(height))


def _fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf).copy()


def _set_chart_font(chart, size_pt=11):
    """Apply font size to chart category/value axes."""
    for ax in [chart.category_axis, chart.value_axis]:
        try:
            ax.tick_labels.font.size = Pt(size_pt)
        except Exception:
            pass


def _color_series(series, hex_color):
    series.format.fill.solid()
    series.format.fill.fore_color.rgb = _hex2rgb(hex_color)


# ── Data ───────────────────────────────────────────────────────────────────────

def add_outcome_cols(df):
    df = df.copy()
    df["outcome"] = df.apply(
        lambda r: ("TP" if r.y_true == 1 and r.y_pred == 1 else
                   "TN" if r.y_true == 0 and r.y_pred == 0 else
                   "FP" if r.y_true == 0 and r.y_pred == 1 else "FN"), axis=1)
    df["zone"] = df["prob_adhesion"].apply(
        lambda p: "high" if p >= ZONE_HI else "low" if p <= ZONE_LO else "mid")
    return df


def load_patch_stores():
    def _load_h5(path):
        with h5py.File(path, "r") as f:
            raw  = f["patches/raw"][:]
            meta = list(csv.DictReader(io.StringIO(f["meta/csv"][()].decode())))
        d = {}
        for i, r in enumerate(meta):
            p = raw[i].astype(float)
            lo, hi = np.percentile(p, 2), np.percentile(p, 98)
            d[r["filename"]] = np.clip((p - lo) / max(hi - lo, 1e-6), 0, 1)
        return d
    return _load_h5(H5_CTRL), _load_h5(H5_YCOMP)


def get_patch(fn, ctrl, ycomp):
    return ctrl.get(fn) if fn.startswith("control_") else ycomp.get(fn)


# ── Slide A — KDE distributions (matplotlib image) ────────────────────────────

def slide_a(prs, df):
    s = _blank(prs)
    _header(s, "Panel A — Score distributions by prediction outcome (TP / TN / FP / FN)",
            "CP-LightGBM, union B1+B2 labels, n=747  ·  prob_adhesion = raw model score")

    outcomes = [("TP", C_TP, "TP correct adhesion"),
                ("TN", C_TN, "TN correct no-adhesion"),
                ("FP", C_FP, "FP false adhesion"),
                ("FN", C_FN, "FN missed adhesion")]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("white")
    xs = np.linspace(0, 1, 400)
    for code, col, label in outcomes:
        vals = df[df["outcome"] == code]["prob_adhesion"].values
        if len(vals) < 3: continue
        kde = gaussian_kde(vals, bw_method=0.08)
        ys  = kde(xs)
        ax.plot(xs, ys, color=col, linewidth=2.0, label=f"{label}  (n={len(vals)})")
        ax.fill_between(xs, ys, alpha=0.10, color=col)
    ax.axvline(ZONE_LO, color="#555555", linewidth=0.9, linestyle="--")
    ax.axvline(ZONE_HI, color="#555555", linewidth=0.9, linestyle="--")
    ax.set_xlabel("Prediction score (prob. adhesion)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_xlim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10, frameon=False, loc="upper center")
    ax.set_title("Score distributions by outcome", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _paste_pil(s, _fig_to_pil(fig), 1.5, 0.85, 10.0, 6.0)


# ── Slide B — Zone fraction + accuracy (native charts) ────────────────────────

def slide_b(prs, df):
    s = _blank(prs)
    _header(s, "Panel B — Patch fraction and accuracy per score zone",
            f"Zones: low (score ≤ {ZONE_LO}), mid ({ZONE_LO}–{ZONE_HI}), high (score ≥ {ZONE_HI})")

    zone_labels = [f"low (≤{ZONE_LO})", f"mid ({ZONE_LO}–{ZONE_HI})", f"high (≥{ZONE_HI})"]
    zone_keys   = ["low", "mid", "high"]
    zone_colors = [C_LOW, C_MID, C_HIGH]

    ns    = [int((df["zone"] == k).sum()) for k in zone_keys]
    fracs = [n / len(df) * 100 for n in ns]
    accs  = [(df[df["zone"] == k]["y_pred"] == df[df["zone"] == k]["y_true"]).mean() * 100
             for k in zone_keys]

    # ── Chart 1: fraction ────────────────────────────────────────────────────
    cd = ChartData()
    cd.categories = zone_labels
    cd.add_series("Fraction (%)", [round(f, 1) for f in fracs])
    ch1 = s.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED,
        Inches(0.5), Inches(0.85), Inches(5.8), Inches(5.8), cd).chart
    ch1.has_title = True
    ch1.chart_title.text_frame.text = "Patch fraction per zone (%)"
    ch1.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(13)
    ch1.has_legend = False
    ch1.value_axis.has_title = True
    ch1.value_axis.axis_title.text_frame.text = "Patches (%)"
    _set_chart_font(ch1, 12)
    for i, col in enumerate(zone_colors):
        _color_series(ch1.series[0].points[i], col)

    # ── Chart 2: accuracy ────────────────────────────────────────────────────
    cd2 = ChartData()
    cd2.categories = zone_labels
    cd2.add_series("Accuracy (%)", [round(a, 1) for a in accs])
    ch2 = s.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED,
        Inches(6.9), Inches(0.85), Inches(5.8), Inches(5.8), cd2).chart
    ch2.has_title = True
    ch2.chart_title.text_frame.text = "Accuracy per zone (%)"
    ch2.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(13)
    ch2.has_legend = False
    ch2.value_axis.has_title = True
    ch2.value_axis.axis_title.text_frame.text = "Accuracy (%)"
    _set_chart_font(ch2, 12)
    for i, col in enumerate(zone_colors):
        _color_series(ch2.series[0].points[i], col)

    # ── Summary text ─────────────────────────────────────────────────────────
    out_mask = df["zone"] != "mid"
    out_acc  = (df[out_mask]["y_pred"] == df[out_mask]["y_true"]).mean()
    summary  = (f"Counts — low: n={ns[0]} ({fracs[0]:.1f}%)  ·  "
                f"mid: n={ns[1]} ({fracs[1]:.1f}%)  ·  "
                f"high: n={ns[2]} ({fracs[2]:.1f}%)\n"
                f"Accuracy — low: {accs[0]:.1f}%  ·  mid: {accs[1]:.1f}%  ·  "
                f"high: {accs[2]:.1f}%  ·  outside mid: {out_acc:.1%}")
    _txt(s, summary, 0.5, 6.75, SLIDE_W - 1.0, 0.65, size=11, color=C_GREY)


# ── Slide C — Accuracy vs retained fraction (native chart) ────────────────────

def slide_c(prs, df):
    s = _blank(prs)
    _header(s, "Panel C — Accuracy vs retained fraction as confidence threshold tightens",
            "CP-LightGBM, union B1+B2, n=747  ·  retained = score ≤ t  OR  score ≥ 1−t")

    p      = df["prob_adhesion"].values
    y_true = df["y_true"].values
    y_pred = df["y_pred"].values

    thresholds = np.linspace(0, 0.499, 60)   # 60 points for readable chart
    frac_ret, acc_ret = [], []
    for t in thresholds:
        mask = (p <= t) | (p >= 1 - t)
        n = mask.sum()
        frac_ret.append(round(n / len(df) * 100, 1))
        acc_ret.append(round((y_pred[mask] == y_true[mask]).mean() * 100, 2)
                       if n else None)

    overall_acc = round((y_pred == y_true).mean() * 100, 1)

    # Mark the ZONE_LO boundary point
    t_mark = ZONE_LO
    m = (p <= t_mark) | (p >= 1 - t_mark)
    f_mark = round(m.mean() * 100, 1)
    a_mark = round((y_pred[m] == y_true[m]).mean() * 100, 1)

    cd = ChartData()
    cd.categories = [f"{f:.1f}%" for f in frac_ret]
    cd.add_series("Accuracy (%)", acc_ret)

    ch = s.shapes.add_chart(
        XL_CHART_TYPE.LINE,
        Inches(0.8), Inches(0.85), Inches(11.5), Inches(5.8), cd).chart
    ch.has_title = True
    ch.chart_title.text_frame.text = "Accuracy on retained patches vs fraction retained"
    ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(13)
    ch.has_legend = False
    ch.value_axis.has_title = True
    ch.value_axis.axis_title.text_frame.text = "Accuracy (%)"
    ch.category_axis.has_title = True
    ch.category_axis.axis_title.text_frame.text = "Fraction of patches retained (%)"
    _set_chart_font(ch, 11)

    # Key numbers as editable text
    _txt(s, (f"Overall accuracy (all patches): {overall_acc}%\n"
             f"At threshold {t_mark}/{1-t_mark:.1f}: {f_mark}% retained, "
             f"accuracy = {a_mark}%"),
         0.8, 6.75, 11.5, 0.65, size=12, color=C_GREY)


# ── Slide D — Example patches (matplotlib image) ──────────────────────────────

def slide_d(prs, df, ctrl, ycomp):
    s = _blank(prs)
    _header(s, "Panel D — Example patches by outcome and confidence",
            "Score shown above each patch  ·  sorted by confidence within category")

    categories = [
        ("TP", C_TP,  "Correct (adhesion)",      "high",  True),
        ("TN", C_TN,  "Correct (no-adhesion)",   "low",   False),
        ("FP", C_FP,  "Wrong (false adhesion)",  "high",  True),
        ("FN", C_FN,  "Wrong (missed adhesion)", "low",   False),
    ]
    mid_rows = df[df["zone"] == "mid"].copy()
    n_per_row = 5
    seed = 42

    fig, axes = plt.subplots(len(categories) + 1, n_per_row,
                             figsize=(n_per_row * 0.78, (len(categories) + 1) * 1.08))
    fig.patch.set_facecolor("white")

    def _draw_row(row_axes, pool, col, label, ascending=True, zone_filter=None):
        if zone_filter is not None:
            pool = pool[pool["zone"] == zone_filter]
        pool = pool.sort_values("prob_adhesion", ascending=ascending)
        chosen = pool.head(n_per_row * 3).sample(
            min(n_per_row, len(pool.head(n_per_row * 3))),
            random_state=seed).sort_values("prob_adhesion", ascending=ascending)
        for ci, ax in enumerate(row_axes):
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(1.2); sp.set_edgecolor(col)
            if ci == 0:
                ax.set_ylabel(label, fontsize=7.0, rotation=0,
                              labelpad=60, ha="right", va="center",
                              fontweight="bold", color=col)
            if ci < len(chosen):
                row = chosen.iloc[ci]
                patch = get_patch(row["fn"], ctrl, ycomp)
                if patch is not None:
                    ax.imshow(patch, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                    ax.set_title(f"{row['prob_adhesion']:.2f}", fontsize=6.5,
                                 pad=1, color=col, fontweight="bold")
                else:
                    ax.set_facecolor("#eeeeee")
            else:
                ax.set_facecolor("#eeeeee")

    for ri, (code, col, label, zone, asc) in enumerate(categories):
        _draw_row(axes[ri], df[df["outcome"] == code].copy(), col, label,
                  ascending=asc, zone_filter=zone)
    _draw_row(axes[-1], mid_rows, "#888888", "Ambiguous\n(0.3–0.7)", ascending=False)

    fig.subplots_adjust(wspace=0.04, hspace=0.30,
                        left=0.22, right=0.99, top=0.95, bottom=0.01)
    _paste_pil(s, _fig_to_pil(fig), 1.5, 0.85, 10.0, 6.3)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print("Loading predictions …")
    df = pd.read_csv(PRED_CSV)
    df = add_outcome_cols(df)
    print(f"  {len(df)} patches, {df['outcome'].value_counts().to_dict()}")

    print("Loading patch stores …")
    ctrl, ycomp = load_patch_stores()

    prs = Presentation()
    prs.slide_width  = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)

    print("Building slides …")
    slide_a(prs, df)
    print("  A done")
    slide_b(prs, df)
    print("  B done")
    slide_c(prs, df)
    print("  C done")
    slide_d(prs, df, ctrl, ycomp)
    print("  D done")

    prs.save(OUT)
    print(f"Saved → {OUT}  ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
