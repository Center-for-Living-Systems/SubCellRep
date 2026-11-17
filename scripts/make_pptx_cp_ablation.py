#!/usr/bin/env python3
"""
make_pptx_cp_ablation.py

Focused ablation comparison deck: cp_s32 as baseline vs feature additions.
All white backgrounds. 7 slides, 4 chart panels per content slide.
  Top row:    LE B1 (left) | IE B1 (right)   — axis 65–88%
  Bottom row: LE B2v2 (left) | IE B2v2 (right) — axis 88–101%
Grey bar = cp_s32 baseline; dashed line = baseline reference level.

LE note: budget=0 = all labels, 5-fold CV → ~623 training labels (80% of
combined B1+B2v2 pool of ~786). Classifier: LightGBM, mixed strategy.
IE note: LOIO with label_budget=all, mean over 22 held-out images.
"""
from __future__ import annotations
import argparse, io, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_pptx_le_benchmark import (
    _blank, _txt,
    C_HEAD, C_GREY, C_WHITE,
    SW, SH, PP_ALIGN, RGBColor, Inches, Pt,
)
from pptx import Presentation

# ── Paths ──────────────────────────────────────────────────────────────────
REPO     = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO / "eval_results"
LE_CSV   = EVAL_DIR / "b1b2v2/le_b1b2v2_budget_cp_combo_lgbm.csv"
IE_CSV   = EVAL_DIR / "image_efficiency_cp_combo_lgbm.csv"
OUT_DIR  = REPO / "results"

# ── Colours (all white slides) ─────────────────────────────────────────────
NAVY    = RGBColor(0x1B, 0x2A, 0x3B)   # text
MED     = RGBColor(0x37, 0x41, 0x51)   # body text
GREY    = RGBColor(0x6B, 0x72, 0x80)   # subtitle / footnote
LTGRY   = RGBColor(0xF3, 0xF4, 0xF6)   # card background
BRDGRY  = RGBColor(0xD1, 0xD5, 0xDB)   # border / divider
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
B1C     = "#2563AB"   # B1 bars (blue)
B2C     = "#B45309"   # B2v2 bars (amber/dark gold)
BASEC   = "#9CA3AF"   # cp_s32 baseline bars (medium grey)
BESTC   = "#C2410C"   # best-in-group highlight (terracotta)
B1C_HEX = "#2563AB"
B2C_HEX = "#B45309"

# ── Feature labels ─────────────────────────────────────────────────────────
FEAT_LABEL = {
    "cp_s32":                "cp32\n(baseline)",
    "cp_s16+cp_s32":         "cp16+32",
    "cp_s32+s128":           "cp32+128",
    "cp_s32+cp_s64":         "cp32+64",
    "cp_s32+dist":           "cp32\n+dist",
    "cp_s32+s128+dist":      "cp32+128\n+dist",
    "cp_s32+cp_s16_ch2":     "cp32\n+ch2-16",
    "cp_s32+cp_s32_ch2":     "cp32\n+ch2-32",
    "cp_s32+cp_s64_ch2":     "cp32\n+ch2-64",
    "cp_s32+cp_s128_ch2":    "cp32\n+ch2-128",
    "cp_s32+cp_s16_ch3":     "cp32\n+ch3-16",
    "cp_s32+cp32ch3":        "cp32\n+ch3-32",
    "cp_s32+cp_s64_ch3":     "cp32\n+ch3-64",
    "cp_s32+cp_s128_ch3":    "cp32\n+ch3-128",
    "supcon_z":              "SupCon-z",
    "il_s32":                "ilastik",
    "cp_s32+supcon_z":       "cp32\n+SupCon",
    "cp_s32+il_s32":         "cp32\n+ilastik",
    "supcon_z+il_s32":       "SupCon\n+ilastik",
    "cp_s32+supcon_z+il_s32":"cp32+SupCon\n+ilastik",
}

# ── Data loaders ───────────────────────────────────────────────────────────
def load_data():
    le = pd.read_csv(LE_CSV)
    ie = pd.read_csv(IE_CSV)
    le_sub = le[(le["budget"]==0) & (le["strategy"]=="mixed")]
    le_piv = le_sub.groupby(["feat","test_batch"])["bal_acc"].mean().unstack("test_batch")
    ie_sub = ie[(ie["experiment"]=="loio") & (ie["label_budget"]=="all")]
    ie_piv = ie_sub.groupby(["feature","annotator"])["bal_acc"].mean().unstack("annotator")
    return le_piv, ie_piv

def _get(piv, feat, col):
    if piv is None or feat not in piv.index or col not in piv.columns:
        return np.nan
    return float(piv.loc[feat, col])

# ── PPTX helpers ───────────────────────────────────────────────────────────
def _prs():
    prs = Presentation()
    prs.slide_width  = SW
    prs.slide_height = SH
    return prs

def _fig_to_buf(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf

def _add_fig(slide, fig, left, top, width, height):
    slide.shapes.add_picture(_fig_to_buf(fig), left, top, width=width, height=height)

def _hline(slide, y, x=Inches(0.22), w=Inches(12.88), color=BRDGRY):
    ln = slide.shapes.add_shape(1, x, y, w, Inches(0.01))
    ln.fill.solid(); ln.fill.fore_color.rgb = color
    ln.line.fill.background()

# ── Chart panel constants ──────────────────────────────────────────────────
# Slide 13.33" × 7.5" (LAYOUT_13_33x7_5)
# Title area: y=0.10, h=0.48   subtitle: y=0.60, h=0.24
# Row-1 label: y=0.90   Row-1 charts: y=1.08, h=2.82
# Row-2 label: y=3.98   Row-2 charts: y=4.16, h=3.12
# Left col: x=0.22, w=6.30   Right col: x=6.82, w=6.30

LX, RX = Inches(0.22), Inches(6.82)
CW = Inches(6.30)
R1Y, R1H = Inches(1.08), Inches(2.82)
R2Y, R2H = Inches(4.16), Inches(3.12)
LBL1_Y = Inches(0.90)
LBL2_Y = Inches(3.98)

# ── Single bar-chart panel ─────────────────────────────────────────────────
def _bar_chart(combos, values, bar_colors, baseline_val,
               chart_title, ymin, ymax, width_in, height_in):
    """Return a matplotlib Figure with a clean grouped bar chart."""
    fig, ax = plt.subplots(figsize=(width_in, height_in), facecolor="white")
    fig.subplots_adjust(left=0.11, right=0.98, top=0.88, bottom=0.28)

    xs = np.arange(len(combos))
    W = 0.58
    bars = ax.bar(xs, values, W, color=bar_colors, zorder=3,
                  edgecolor="white", linewidth=0.6)

    # baseline reference dashed line
    ax.axhline(baseline_val, color="#6B7280", lw=1.3, ls="--",
               alpha=0.75, zorder=2, label=f"baseline {baseline_val:.1f}%")

    # data labels
    for bar, v in zip(bars, values):
        if not np.isnan(v):
            ypos = v + (ymax - ymin) * 0.015
            ax.text(bar.get_x() + bar.get_width()/2, ypos, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=7.5,
                    color="#1E293B", fontweight="normal")

    labels = [FEAT_LABEL.get(c, c) for c in combos]
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7.2, ha="center")
    ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.14)
    ax.set_ylabel("Balanced acc. (%)", fontsize=8, color="#374151")
    ax.set_title(chart_title, fontsize=9.5, fontweight="bold",
                 color="#1B2A3B", pad=5)
    ax.tick_params(axis="y", labelsize=7.5, labelcolor="#374151")
    ax.set_facecolor("white")
    ax.grid(axis="y", color="#E5E7EB", lw=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#D1D5DB")
    return fig

def _four_panel(slide, combos, le_piv, ie_piv):
    """Draw 4 bar-chart panels; baseline (cp_s32) always grey."""
    fw1 = CW / Inches(1); fh1 = R1H / Inches(1)
    fw2 = CW / Inches(1); fh2 = R2H / Inches(1)

    panels = [
        # (piv, col, ymin, ymax, title, bar_color_non_base, left, top, fw, fh)
        (le_piv, "b1",   65, 88,  "Label Efficiency — B1 (Margaret)",   B1C, LX, R1Y, fw1, fh1),
        (ie_piv, "b1",   65, 88,  "Image Efficiency LOIO — B1 (Margaret)", B1C, RX, R1Y, fw1, fh1),
        (le_piv, "b2v2", 88, 101, "Label Efficiency — B2v2 (Annabel)",  B2C, LX, R2Y, fw2, fh2),
        (ie_piv, "b2v2", 88, 101, "Image Efficiency LOIO — B2v2 (Annabel)", B2C, RX, R2Y, fw2, fh2),
    ]

    for piv, col, ymin, ymax, title, bar_color, left, top, fw, fh in panels:
        vals = [_get(piv, c, col) * 100 for c in combos]
        baseline_val = _get(piv, "cp_s32", col) * 100
        colors = [BASEC if c == "cp_s32" else bar_color for c in combos]
        fig = _bar_chart(combos, vals, colors, baseline_val, title,
                         ymin, ymax, fw, fh)
        _add_fig(slide, fig, left, top, CW, Inches(fh))

    # Row labels (plain text, no background box)
    _txt(slide, "B1 — Margaret (primary annotator)   axis 65 – 88%",
         Inches(0.22), LBL1_Y, Inches(12.88), Inches(0.20),
         bold=True, size_pt=8.5, color=RGBColor(0x2B, 0x50, 0x87))
    _txt(slide, "B2v2 — Annabel (secondary annotator)   axis 88 – 101%",
         Inches(0.22), LBL2_Y, Inches(12.88), Inches(0.20),
         bold=True, size_pt=8.5, color=RGBColor(0x92, 0x40, 0x0E))

    # Thin vertical divider between left and right columns
    _hline(slide, Inches(0.92), x=Inches(6.60), w=Inches(0.01))
    div = slide.shapes.add_shape(1, Inches(6.60), Inches(0.92),
                                 Inches(0.01), Inches(6.42))
    div.fill.solid(); div.fill.fore_color.rgb = BRDGRY
    div.line.fill.background()

    # Footnote
    _txt(slide,
         "Grey bar / dashed line = cp_s32 baseline   ·   "
         "LE: budget=all mixed strategy (~623 train labels, 5-fold CV)   ·   "
         "IE: LOIO label_budget=all (mean over 22 images)   ·   LightGBM",
         Inches(0.22), Inches(7.28), Inches(12.88), Inches(0.18),
         size_pt=6.5, color=GREY)

# ── White header (title + subtitle, no background band) ────────────────────
def _white_header(slide, title, subtitle):
    _txt(slide, title,
         Inches(0.22), Inches(0.10), Inches(12.88), Inches(0.48),
         bold=True, size_pt=18, color=NAVY)
    _txt(slide, subtitle,
         Inches(0.22), Inches(0.60), Inches(12.88), Inches(0.28),
         size_pt=9, color=GREY)

# ── Slide 1: Title ─────────────────────────────────────────────────────────
def _slide_title(prs):
    slide = _blank(prs)

    _txt(slide, "CP32 Feature Ablation Study",
         Inches(0.6), Inches(0.9), Inches(12.1), Inches(1.0),
         bold=True, size_pt=36, color=NAVY, align=PP_ALIGN.CENTER)
    _txt(slide, "Direct Pairwise Comparisons — DS1 Vinculin CIO Imaging",
         Inches(0.6), Inches(2.0), Inches(12.1), Inches(0.45),
         size_pt=16, color=MED, align=PP_ALIGN.CENTER)
    _txt(slide,
         "LightGBM classifier  ·  "
         "LE = label-efficiency (budget=all, ~623 train labels, 5-fold CV)  ·  "
         "IE = image-efficiency LOIO (label_budget=all)",
         Inches(0.6), Inches(2.52), Inches(12.1), Inches(0.35),
         size_pt=11, color=GREY, align=PP_ALIGN.CENTER)

    # Annotation note
    _txt(slide,
         "Annotators:  B1 = Margaret (primary, 393 patches, 43% adhesion)  ·  "
         "B2v2 = Annabel (secondary Sep-2026, 393 patches, 37% adhesion)",
         Inches(0.6), Inches(2.95), Inches(12.1), Inches(0.30),
         size_pt=10, color=GREY, align=PP_ALIGN.CENTER)

    groups = [
        ("① Scale ch1",    "cp_s32 vs cp_s16+32, cp_s32+128, cp_s32+64"),
        ("② + Actin ch3",  "cp_s32 vs +actin at 16/32/64/128 px"),
        ("③ + Zyxin ch2",  "cp_s32 vs +zyxin at 16/32/64/128 px"),
        ("④ + Distance",   "cp_s32 vs +dist and +128+dist"),
        ("⑤ + Learned",    "cp_s32 vs SupCon AE + ilastik combos"),
        ("⑥ vs Best",      "cp_s32 vs top performers (B1 LE/IE & B2v2 LE/IE)"),
    ]
    for i, (g_title, g_sub) in enumerate(groups):
        col, row = i % 3, i // 3
        bx = Inches(1.1 + col * 3.72)
        by = Inches(3.55 + row * 1.55)
        box = slide.shapes.add_shape(1, bx, by, Inches(3.5), Inches(1.35))
        box.fill.solid(); box.fill.fore_color.rgb = LTGRY
        box.line.color.rgb = BRDGRY; box.line.width = Pt(0.75)
        _txt(slide, g_title, bx+Inches(0.14), by+Inches(0.12),
             Inches(3.22), Inches(0.36),
             bold=True, size_pt=12, color=NAVY)
        _txt(slide, g_sub, bx+Inches(0.14), by+Inches(0.52),
             Inches(3.22), Inches(0.65),
             size_pt=9, color=MED)

# ── Generic comparison slide ───────────────────────────────────────────────
def _comp_slide(prs, title, subtitle, combos, le_piv, ie_piv):
    slide = _blank(prs)
    _white_header(slide, title, subtitle)
    _four_panel(slide, combos, le_piv, ie_piv)

# ── Slide 7: vs Best (white, highlight best bar) ───────────────────────────
def _slide_best(prs, le_piv, ie_piv):
    slide = _blank(prs)
    _white_header(slide,
        "⑥ cp_s32 vs Top Performers",
        "Best LE B1: cp_s32+128 = 82.0%  ·  Best IE B1: cp_s32+SupCon = 80.6%  ·  "
        "Best LE B2v2: cp_s32+ch2-128 = 97.1%  ·  Best IE B2v2: cp_s32+ch2-16 = 99.6%")

    combos = ["cp_s32","cp_s32+s128","cp_s32+supcon_z","cp_s32+cp_s128_ch2","cp_s32+cp_s16_ch2"]

    fw1 = CW / Inches(1); fh1 = R1H / Inches(1)
    fw2 = CW / Inches(1); fh2 = R2H / Inches(1)

    panels = [
        (le_piv, "b1",   65, 88,  "Label Efficiency — B1 (Margaret)",    B1C, LX, R1Y, fw1, fh1),
        (ie_piv, "b1",   65, 88,  "Image Efficiency LOIO — B1 (Margaret)", B1C, RX, R1Y, fw1, fh1),
        (le_piv, "b2v2", 88, 101, "Label Efficiency — B2v2 (Annabel)",   B2C, LX, R2Y, fw2, fh2),
        (ie_piv, "b2v2", 88, 101, "Image Efficiency LOIO — B2v2 (Annabel)", B2C, RX, R2Y, fw2, fh2),
    ]

    for piv, col, ymin, ymax, title, bar_color, left, top, fw, fh in panels:
        vals = [_get(piv, c, col) * 100 for c in combos]
        baseline_val = _get(piv, "cp_s32", col) * 100
        best_idx = int(np.nanargmax(vals))
        colors = []
        for ci, c in enumerate(combos):
            if c == "cp_s32":
                colors.append(BASEC)
            elif ci == best_idx:
                colors.append(BESTC)   # terracotta highlight for best
            else:
                colors.append(bar_color)
        fig = _bar_chart(combos, vals, colors, baseline_val, title,
                         ymin, ymax, fw, fh)
        _add_fig(slide, fig, left, top, CW, Inches(fh))

    _txt(slide, "B1 — Margaret (primary annotator)   axis 65 – 88%",
         Inches(0.22), LBL1_Y, Inches(12.88), Inches(0.20),
         bold=True, size_pt=8.5, color=RGBColor(0x2B, 0x50, 0x87))
    _txt(slide, "B2v2 — Annabel (secondary annotator)   axis 88 – 101%",
         Inches(0.22), LBL2_Y, Inches(12.88), Inches(0.20),
         bold=True, size_pt=8.5, color=RGBColor(0x92, 0x40, 0x0E))

    div = slide.shapes.add_shape(1, Inches(6.60), Inches(0.92),
                                 Inches(0.01), Inches(6.42))
    div.fill.solid(); div.fill.fore_color.rgb = BRDGRY
    div.line.fill.background()

    _txt(slide,
         "Terracotta bar = best in group  ·  Grey bar = cp_s32 baseline  ·  "
         "--- = baseline reference  ·  "
         "LE: ~623 train labels (5-fold CV, budget=all, mixed B1+B2v2)  ·  "
         "IE: LOIO label_budget=all",
         Inches(0.22), Inches(7.28), Inches(12.88), Inches(0.18),
         size_pt=6.5, color=GREY)

# ── Build ──────────────────────────────────────────────────────────────────
def build(out_path: Path):
    print("Loading data …")
    le_piv, ie_piv = load_data()
    print(f"  LE combos: {len(le_piv)}, IE combos: {len(ie_piv)}")

    prs = _prs()

    print("  Slide 1 — Title")
    _slide_title(prs)

    print("  Slide 2 — Scale ch1")
    _comp_slide(prs,
        "① Scale Ablation — Same Channel (paxillin ch1)",
        "Does adding a second ch1 patch size help? "
        "cp_s16+32 = add finer texture  ·  cp_s32+128 = add coarser context  ·  cp_s32+64 = mid-large",
        ["cp_s32","cp_s16+cp_s32","cp_s32+s128","cp_s32+cp_s64"],
        le_piv, ie_piv)

    print("  Slide 3 — +Actin ch3")
    _comp_slide(prs,
        "② Cross-Channel: Actin (ch3)",
        "Does adding actin-channel CellProfiler features on top of paxillin cp_s32 help? 4 patch sizes: 16/32/64/128 px",
        ["cp_s32","cp_s32+cp_s16_ch3","cp_s32+cp32ch3","cp_s32+cp_s64_ch3","cp_s32+cp_s128_ch3"],
        le_piv, ie_piv)

    print("  Slide 4 — +Zyxin ch2")
    _comp_slide(prs,
        "③ Cross-Channel: Zyxin (ch2)",
        "Does adding zyxin-channel CellProfiler features on top of paxillin cp_s32 help? 4 patch sizes: 16/32/64/128 px",
        ["cp_s32","cp_s32+cp_s16_ch2","cp_s32+cp_s32_ch2","cp_s32+cp_s64_ch2","cp_s32+cp_s128_ch2"],
        le_piv, ie_piv)

    print("  Slide 5 — +Distance")
    _comp_slide(prs,
        "④ Distance Features",
        "Geometric distance-to-boundary (8-ray, 9 features)  ·  "
        "cp_s32+dist adds only dist  ·  cp_s32+128+dist adds both larger scale and distance",
        ["cp_s32","cp_s32+dist","cp_s32+s128+dist"],
        le_piv, ie_piv)

    print("  Slide 6 — +Learned embeddings")
    _comp_slide(prs,
        "⑤ Learned Embeddings — SupCon AE & ilastik",
        "SupCon AE: 32-dim encoder latent avg over 3 seeds  ·  ilastik: 32 px filter bank (80 feat)  ·  "
        "both alone and combined with cp_s32",
        ["cp_s32","supcon_z","il_s32","cp_s32+supcon_z","cp_s32+il_s32","supcon_z+il_s32","cp_s32+supcon_z+il_s32"],
        le_piv, ie_piv)

    print("  Slide 7 — vs Best performers")
    _slide_best(prs, le_piv, ie_piv)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out_path))
    print(f"\nSaved → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DIR / "cp_ablation_comparison.pptx"))
    args = ap.parse_args()
    build(Path(args.out))


if __name__ == "__main__":
    main()
