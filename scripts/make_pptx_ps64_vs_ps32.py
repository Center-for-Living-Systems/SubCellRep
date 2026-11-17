#!/usr/bin/env python3
"""
make_pptx_ps64_vs_ps32.py

Comparison PPT for patch size 64 vs 32 experiments on DS1 B2 (Annabel).
SupCon contrastive AE latent features + LightGBM classifier.

Slides:
  1. Title
  2. Config table (native PPTX table)
  3. Results figure — bar chart (mean±std) + per-fold line chart
  4. Summary / discussion

Usage:
  python scripts/make_pptx_ps64_vs_ps32.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Import shared helpers from make_pptx_le_benchmark.py
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_pptx_le_benchmark import (
    _blank, _txt, _slide_header, _add_fig, _panel_bg, _rule,
    C_HEAD, C_BODY, C_GREY, C_WHITE, SW, SH, PP_ALIGN, RGBColor, Inches, Pt,
)
from pptx import Presentation

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT  = Path(__file__).resolve().parents[1]
EVAL_JSON  = REPO_ROOT / "results" / "ms_b2_ds1_ps64_eval.json"
OUT_FILE   = REPO_ROOT / "results" / "ps64_vs_ps32.pptx"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
with EVAL_JSON.open() as f:
    EVAL = json.load(f)

# Config definitions (key order = display order)
CONFIGS = [
    {
        "key":        "ms_b2_ds1_ps64_lat12p8",
        "name":       "PS64 / lat12p8",
        "patch_size": "64",
        "latent_dim": "12",
        "proj_dim":   "8",
        "notes":      "Baseline (larger patches)",
        "color":      "#e07b39",
    },
    {
        "key":        "ms_b2_ds1_ps32_lat12p8",
        "name":       "PS32 / lat12p8",
        "patch_size": "32",
        "latent_dim": "12",
        "proj_dim":   "8",
        "notes":      "Same latent/proj, smaller patch",
        "color":      "#2e86ab",
    },
    {
        "key":        "ms_b2_ds1_ps32_lat64p32",
        "name":       "PS32 / lat64p32",
        "patch_size": "32",
        "latent_dim": "64",
        "proj_dim":   "32",
        "notes":      "Larger latent/proj, smaller patch",
        "color":      "#2ca02c",
    },
]

# ---------------------------------------------------------------------------
# PPT factory
# ---------------------------------------------------------------------------

def _prs() -> Presentation:
    prs = Presentation()
    prs.slide_width  = SW
    prs.slide_height = SH
    return prs


def _tbl_cell(cell, text, bold=False, size_pt=10.5,
              font_color=None, bg_color=None,
              align=PP_ALIGN.CENTER):
    font_color = font_color or RGBColor(0x1A, 0x1A, 0x1A)
    if bg_color is not None:
        cell.fill.solid()
        cell.fill.fore_color.rgb = bg_color
    tf = cell.text_frame
    tf.word_wrap = True
    para = tf.paragraphs[0]
    para.alignment = align
    run = para.add_run()
    run.text = text
    run.font.size      = Pt(size_pt)
    run.font.bold      = bold
    run.font.color.rgb = font_color


# ---------------------------------------------------------------------------
# Slide 1 — Title
# ---------------------------------------------------------------------------

def _slide_title(prs):
    slide = _blank(prs)
    _txt(slide,
         "Patch Size Comparison: 32×32 vs 64×64 — DS1 Multiscale AE",
         Inches(0.8), Inches(2.3), Inches(11.7), Inches(1.3),
         bold=True, size_pt=32, color=C_HEAD, align=PP_ALIGN.CENTER)
    _txt(slide,
         "SupCon Contrastive AE  ·  LightGBM Classifier  ·  5-Fold CV",
         Inches(0.8), Inches(3.65), Inches(11.7), Inches(0.5),
         size_pt=18, color=C_GREY, align=PP_ALIGN.CENTER)
    _txt(slide,
         "DS1 B2 (Annabel)  ·  vinculin channel  ·  CIO imaging",
         Inches(0.8), Inches(4.25), Inches(11.7), Inches(0.4),
         size_pt=13, color=C_GREY, align=PP_ALIGN.CENTER)


# ---------------------------------------------------------------------------
# Slide 2 — Config table (native PPTX)
# ---------------------------------------------------------------------------

def _slide_config_table(prs):
    slide = _blank(prs)
    _slide_header(slide,
                  "Experiment Configurations",
                  "Three SupCon AE variants: patch size × latent dim × proj dim")

    # Table geometry
    tbl_left = Inches(0.6)
    tbl_top  = Inches(1.25)
    tbl_w    = Inches(12.1)
    tbl_h    = Inches(5.5)

    n_rows = 1 + len(CONFIGS)   # header + 3 data rows
    n_cols = 5                   # Config Name, Patch Size, Latent Dim, Proj Dim, Notes

    col_widths = [
        Inches(2.8),   # Config Name
        Inches(1.5),   # Patch Size
        Inches(1.5),   # Latent Dim
        Inches(1.5),   # Proj Dim
        Inches(4.8),   # Notes
    ]
    # Fix rounding
    col_widths[-1] = tbl_w - sum(col_widths[:-1])

    hdr_h  = Inches(0.6)
    data_h = (tbl_h - hdr_h) // len(CONFIGS)
    row_heights = [hdr_h] + [data_h] * len(CONFIGS)

    tbl_shape = slide.shapes.add_table(n_rows, n_cols, tbl_left, tbl_top, tbl_w, tbl_h)
    tbl = tbl_shape.table

    for ci, w in enumerate(col_widths):
        tbl.columns[ci].width = w
    for ri, h in enumerate(row_heights):
        tbl.rows[ri].height = h

    C_HDR_BG   = RGBColor(0x16, 0x21, 0x3E)
    C_HDR_TEXT = RGBColor(0xFF, 0xFF, 0xFF)
    C_ROW_A    = RGBColor(0xF5, 0xF7, 0xFF)
    C_ROW_B    = RGBColor(0xE8, 0xED, 0xFF)

    headers = ["Config Name", "Patch Size (px)", "Latent Dim", "Proj Dim", "Notes"]
    for ci, hdr in enumerate(headers):
        _tbl_cell(tbl.cell(0, ci), hdr,
                  bold=True, size_pt=11,
                  font_color=C_HDR_TEXT, bg_color=C_HDR_BG)

    for ri, cfg in enumerate(CONFIGS):
        bg = C_ROW_A if ri % 2 == 0 else C_ROW_B
        row_vals = [cfg["name"], cfg["patch_size"], cfg["latent_dim"],
                    cfg["proj_dim"], cfg["notes"]]
        for ci, val in enumerate(row_vals):
            align = PP_ALIGN.LEFT if ci in (0, 4) else PP_ALIGN.CENTER
            _tbl_cell(tbl.cell(ri + 1, ci), val,
                      size_pt=11, bg_color=bg,
                      font_color=RGBColor(0x1A, 0x1A, 0x1A),
                      align=align)


# ---------------------------------------------------------------------------
# Slide 3 — Results figure
# ---------------------------------------------------------------------------

def _make_results_fig() -> plt.Figure:
    names   = [c["name"]  for c in CONFIGS]
    colors  = [c["color"] for c in CONFIGS]
    means   = [EVAL[c["key"]]["mean"] for c in CONFIGS]
    stds    = [EVAL[c["key"]]["std"]  for c in CONFIGS]
    all_acc = [EVAL[c["key"]]["accs"] for c in CONFIGS]
    n_folds = 5

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor("white")

    # ---- Left: bar chart with error bars ----
    ax = axes[0]
    x = np.arange(len(CONFIGS))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors, alpha=0.85,
                  error_kw=dict(elinewidth=1.5, ecolor="#333333", capthick=1.5),
                  width=0.55)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Balanced Accuracy", fontsize=11)
    ax.set_title("Mean ± Std Balanced Accuracy (5-Fold CV)", fontsize=11, fontweight="bold")
    ax.set_ylim(0.90, 0.99)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3f}"))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    # Annotate bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.001,
                f"{mean:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # ---- Right: per-fold line chart ----
    ax2 = axes[1]
    fold_x = np.arange(1, n_folds + 1)
    for cfg, accs, color, name in zip(CONFIGS, all_acc, colors, names):
        ax2.plot(fold_x, accs, marker="o", color=color, linewidth=1.8,
                 markersize=6, label=name)

    ax2.set_xticks(fold_x)
    ax2.set_xlabel("Fold", fontsize=11)
    ax2.set_ylabel("Balanced Accuracy", fontsize=11)
    ax2.set_title("Per-Fold Balanced Accuracy", fontsize=11, fontweight="bold")
    ax2.set_ylim(0.90, 0.99)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3f}"))
    ax2.legend(fontsize=9, loc="lower right")
    ax2.grid(linestyle="--", alpha=0.5)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(pad=2.0)
    return fig


def _slide_results(prs):
    slide = _blank(prs)
    _slide_header(slide,
                  "Classification Results — 5-Fold Balanced Accuracy",
                  "SupCon AE latent features + LightGBM  ·  DS1 B2 (Annabel)  ·  vinculin channel")

    fig = _make_results_fig()
    _add_fig(slide, fig,
             Inches(0.3), Inches(1.05),
             Inches(12.7), Inches(6.2))


# ---------------------------------------------------------------------------
# Slide 4 — Summary / Discussion
# ---------------------------------------------------------------------------

def _slide_summary(prs):
    slide = _blank(prs)
    _slide_header(slide,
                  "Summary & Conclusions",
                  "Patch size 32 vs 64 — all configurations on DS1 B2 vinculin")

    # Background panel (full content area)
    _panel_bg(slide, Inches(0.35), Inches(1.05), Inches(12.6), Inches(6.1))

    y = Inches(1.25)
    gap = Inches(0.08)

    # Section: Key Observations
    _txt(slide, "Key Observations",
         Inches(0.65), y, Inches(11.8), Inches(0.38),
         bold=True, size_pt=14, color=C_HEAD)
    y += Inches(0.42)

    observations = [
        "PS32 / lat12p8  (mean = 0.9550)  ≈  PS64 / lat12p8  (mean = 0.9524) — "
        "within ~0.3 percentage points, well within noise.",
        "PS32 / lat64p32  (mean = 0.9480)  is slightly lower: larger latent / proj dims "
        "do not improve and may marginally hurt.",
        "All three configs show similar per-fold variance (~1.2–1.4% std), indicating "
        "results are dominated by label noise / fold composition, not patch size.",
    ]
    for obs in observations:
        _txt(slide, f"•  {obs}",
             Inches(0.75), y, Inches(11.5), Inches(0.65),
             size_pt=12, color=C_BODY)
        y += Inches(0.72)

    y += gap
    _rule(slide, y, width=Inches(11.7), left=Inches(0.65), thickness_pt=0.75)
    y += Inches(0.18)

    # Section: Conclusion
    _txt(slide, "Conclusion",
         Inches(0.65), y, Inches(11.8), Inches(0.38),
         bold=True, size_pt=14, color=C_HEAD)
    y += Inches(0.42)

    conclusions = [
        "Patch size 32×32 is sufficient for this dataset — no benefit from 64×64.",
        "Smaller latent/proj (lat=12, proj=8) performs at least as well as larger dims "
        "(lat=64, proj=32) with fewer parameters.",
        "Recommended config for follow-up: PS32 / lat12p8 (efficient and effective).",
    ]
    for con in conclusions:
        _txt(slide, f"•  {con}",
             Inches(0.75), y, Inches(11.5), Inches(0.55),
             size_pt=12, color=C_BODY)
        y += Inches(0.60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    prs = _prs()

    print("Building slide 1 — Title")
    _slide_title(prs)

    print("Building slide 2 — Config table")
    _slide_config_table(prs)

    print("Building slide 3 — Results figure")
    _slide_results(prs)

    print("Building slide 4 — Summary")
    _slide_summary(prs)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT_FILE))
    size_kb = OUT_FILE.stat().st_size / 1024
    print(f"\nSaved: {OUT_FILE}")
    print(f"Slides: {len(prs.slides)}")
    print(f"File size: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
