#!/usr/bin/env python3
"""
add_feature_explanation_slides.py

Replace slides 9-10 of pred_confidence_image_eff_b1b2v2_cp_lgbm.pptx with
fully editable native PPTX shapes (text boxes, rectangles — no burned-in text).

  Slide 9 — Feature selection method
    • Scatter plot image (Spearman ρ vs ANOVA F) on the left 60%
    • All text on the right + quadrant labels = native editable text boxes

  Slide 10 — Feature interpretation
    • Entirely native: header box, 8 feature cards (rectangles + text),
      bottom insight box — zero matplotlib

Usage:
  python scripts/add_feature_explanation_slides.py
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
from scipy import stats

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
import lxml.etree as etree

REPO     = Path(__file__).resolve().parent.parent
PRED_DIR = REPO / "eval_results" / "patch_preds"
RES      = REPO / "results"
CP_CSV   = Path("/net/projects/CLS/lding/data/fa_data_analysis/ae_results/features/cellprofiler/ds1.csv")

PPT_PATH = RES / "pred_confidence_image_eff_b1b2v2_cp_lgbm.pptx"

# ── Colour palette ────────────────────────────────────────────────────────────
C_POS_FILL  = RGBColor(0xDD, 0xEE, 0xFF)   # blue card fill
C_POS_EDGE  = RGBColor(0x21, 0x66, 0xAC)   # blue card border / text accent
C_NEG_FILL  = RGBColor(0xFF, 0xEE, 0xEE)   # red card fill
C_NEG_EDGE  = RGBColor(0xD6, 0x60, 0x4D)   # red card border / text accent
C_HDR_FILL  = RGBColor(0xEE, 0xF4, 0xFB)   # header box fill
C_INS_FILL  = RGBColor(0xFF, 0xF9, 0xE6)   # insight box fill
C_INS_EDGE  = RGBColor(0xE0, 0xC0, 0x60)   # insight box border
C_GREY_EDGE = RGBColor(0xAA, 0xCC, 0xEE)
C_BLACK     = RGBColor(0x22, 0x22, 0x22)
C_DARK      = RGBColor(0x33, 0x33, 0x33)
C_TEXT_BLUE = RGBColor(0x21, 0x66, 0xAC)
C_TEXT_RED  = RGBColor(0xD6, 0x60, 0x4D)

# ── Feature descriptions (ordered by |ρ|) ─────────────────────────────────────
FEATURES = [
    # (csv_key,  display_name,   rho_str,  is_positive, description)
    ("inverse_difference_moment_d1",
     "IDM (d=1) · Inverse Difference Moment / Homogeneity",
     "ρ = +0.75 ↑", True,
     "Measures how close the GLCM distribution is to the diagonal. "
     "High IDM = neighbouring pixels have similar intensities = smooth, uniform texture. "
     "Adhesion patches are spatially compact and bright with little local variation."),

    ("info_meas1_d1",
     "IM-1 (d=1) · Information Measure of Correlation 1",
     "ρ = +0.74 ↑", True,
     "Entropy-based measure of correlation between neighbouring pixel pairs. "
     "High IM-1 = structured, non-random co-occurrence pattern. "
     "Adhesion patches show more organised spatial relationships than background."),

    ("inverse_difference_moment_d2",
     "IDM (d=2) · Inverse Difference Moment at 2-px offset",
     "ρ = +0.74 ↑", True,
     "Same homogeneity measure computed at a 2-pixel offset. "
     "The high ρ at both d=1 and d=2 confirms that adhesion smoothness "
     "persists across spatial scales, not just at immediate neighbours."),

    ("difference_entropy_d1",
     "Diff. Entropy (d=1) · Entropy of pairwise differences",
     "ρ = −0.72 ↓", False,
     "Entropy of the distribution of absolute grey-level differences between "
     "neighbouring pixels. High = heterogeneous local transitions. "
     "Non-adhesion regions (background/cytoplasm) have more random local variation."),

    ("angular_second_moment_d1",
     "ASM / Energy (d=1) · Angular Second Moment",
     "ρ = +0.71 ↑", True,
     "Sum of squared GLCM entries. High ASM = few dominant grey-level pair transitions "
     "= ordered, repetitive texture. "
     "Adhesion patches consist of a small set of characteristic intensity patterns."),

    ("correlation_d1",
     "Correlation (d=1) · Grey-level correlation",
     "ρ = +0.69 ↑", True,
     "Linear dependency of grey levels between neighbouring pixels. "
     "High correlation = regular directional patterns. "
     "Focal adhesion patches exhibit structured spatial gradients reflecting their elongated morphology."),

    ("contrast_d1",
     "Contrast (d=1) · Local intensity variation",
     "ρ = −0.69 ↓", False,
     "Weighted sum of squared differences between neighbouring pixel intensities. "
     "High contrast = large local fluctuations. "
     "Non-adhesion areas (background/cytoplasm boundaries) show strong local intensity jumps."),

    ("entropy_d1",
     "Entropy (d=1) · Overall texture disorder",
     "ρ = −0.67 ↓", False,
     "Entropy of the GLCM — measures how many distinct grey-level pairs occur. "
     "High entropy = disordered, unpredictable texture. "
     "Non-adhesion patches are texturally more random; adhesion patches more predictable."),
]


# ── python-pptx helpers ───────────────────────────────────────────────────────

def _remove_last_n_slides(prs: Presentation, n: int):
    """Remove the last n slides from prs in place."""
    sldIdLst = prs.slides._sldIdLst
    for _ in range(n):
        idx   = len(list(prs.slides)) - 1
        sldId = sldIdLst[idx]
        rId   = sldId.get(qn("r:id"))
        prs.part.drop_rel(rId)
        sldIdLst.remove(sldId)


def _blank_slide(prs: Presentation):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    for ph in list(slide.placeholders):
        ph.element.getparent().remove(ph.element)
    return slide


def _txbox(slide, left, top, width, height,
           text="", size=11, bold=False, italic=False,
           color=None, align=PP_ALIGN.LEFT,
           fill_rgb=None, edge_rgb=None, edge_pt=None,
           word_wrap=True) -> object:
    """Add a text box (or shape background) and return the shape."""
    shape = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height))
    tf = shape.text_frame
    tf.word_wrap = word_wrap
    if text:
        p = tf.paragraphs[0]
        p.alignment = align
        run = p.add_run()
        run.text = text
        run.font.size   = Pt(size)
        run.font.bold   = bold
        run.font.italic = italic
        if color:
            run.font.color.rgb = color
    # Optional background / border on the text box shape
    if fill_rgb:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill_rgb
    if edge_rgb:
        shape.line.color.rgb = edge_rgb
        shape.line.width     = Pt(edge_pt or 1.0)
    return shape


def _rect(slide, left, top, width, height,
          fill_rgb=None, edge_rgb=None, edge_pt=1.0, radius=0.1):
    """Add a rounded rectangle shape with no text."""
    from pptx.enum.shapes import MSO_SHAPE_TYPE
    from pptx.util import Emu
    # Use add_shape with ROUNDED_RECTANGLE
    from pptx.enum.shapes import PP_PLACEHOLDER
    from pptx.oxml.ns import nsmap
    # python-pptx doesn't expose rounded-rect directly; use add_shape
    from pptx.util import Emu as _E
    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE = 1 (not rounded, but cleaner)
        Inches(left), Inches(top), Inches(width), Inches(height))
    shape.text = ""
    if fill_rgb:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill_rgb
    else:
        shape.fill.background()
    if edge_rgb:
        shape.line.color.rgb = edge_rgb
        shape.line.width     = Pt(edge_pt)
    else:
        shape.line.fill.background()
    return shape


def _add_para(tf, text, size=10, bold=False, italic=False,
              color=None, align=PP_ALIGN.LEFT, space_before=0):
    """Append a paragraph to an existing text frame."""
    from pptx.oxml.ns import qn as _qn
    from pptx.util import Pt as _Pt
    p = tf.add_paragraph()
    p.alignment = align
    if space_before:
        p.space_before = Pt(space_before)
    run = p.add_run()
    run.text        = text
    run.font.size   = Pt(size)
    run.font.bold   = bold
    run.font.italic = italic
    if color:
        run.font.color.rgb = color
    return p


# ── Compute feature stats ────────────────────────────────────────────────────

def _compute_stats() -> pd.DataFrame:
    df = pd.read_csv(PRED_DIR / "image_eff_b1b2v2_lgbm.csv")
    df = df.rename(columns={"feature": "feat"})
    df = df[df["feat"] == "cp"].copy()
    agg = (df.groupby("fn")
             .agg(y_true=("y_true","first"),
                  prob_adhesion=("prob_adhesion","median"),
                  y_pred=("y_pred", lambda x: x.mode()[0]))
             .reset_index())
    agg["group"] = "TN"
    agg.loc[(agg["y_true"]==1)&(agg["y_pred"]==1),"group"] = "TP"
    agg.loc[(agg["y_true"]==0)&(agg["y_pred"]==1),"group"] = "FP"
    agg.loc[(agg["y_true"]==1)&(agg["y_pred"]==0),"group"] = "FN"
    feat_df = pd.read_csv(CP_CSV).set_index("filename")
    merged  = agg.set_index("fn").join(feat_df, how="inner")
    feat_cols = feat_df.columns.tolist()
    rows = []
    groups = {g: merged[merged["group"]==g] for g in ["TP","FP","TN","FN"]}
    for col in feat_cols:
        vals = merged[col].dropna().values
        prob = merged.loc[merged[col].notna(),"prob_adhesion"].values
        rho, _ = stats.spearmanr(vals, prob)
        gv = [groups[g][col].dropna().values for g in ["TP","FP","TN","FN"]]
        gv = [v for v in gv if len(v)>=3]
        f  = stats.f_oneway(*gv)[0] if len(gv)>=2 else 0.0
        is_intensity = not any(x in col for x in [
            "moment","contrast","correlat","varianc","entropy","average","sum_","info_"])
        rows.append({"feature":col,"rho":rho,
                     "f_stat":float(f) if np.isfinite(f) else 0.0,
                     "type":"intensity" if is_intensity else "texture"})
    return pd.DataFrame(rows)


# ── Scatter plot (image only, no text annotations) ───────────────────────────

def _scatter_image(stats_df: pd.DataFrame) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    color_map = {"intensity":"#e08020","texture":"#1a6eb5"}
    for ftype, grp in stats_df.groupby("type"):
        ax.scatter(grp["rho"], grp["f_stat"],
                   c=color_map[ftype], s=55, alpha=0.75, zorder=3,
                   label=f"{ftype} (n={len(grp)})")
    # Reference lines only — no text labels on the image
    for v in [-0.5, 0.5]:
        ax.axvline(v, color="grey", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(100, color="grey", lw=0.8, ls="--", alpha=0.5)
    ax.axvline(0,   color="black", lw=0.6, alpha=0.35)
    ax.set_xlabel("Spearman ρ  (correlation with P(adhesion))", fontsize=11)
    ax.set_ylabel("ANOVA F-statistic  (across TP / FP / TN / FN groups)", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


# ── Slide 9: feature selection method ────────────────────────────────────────

def _build_slide9(prs: Presentation, stats_df: pd.DataFrame):
    slide = _blank_slide(prs)
    W, H  = 13.33, 7.5   # inches

    # Title
    _txbox(slide, 0.2, 0.08, W - 0.4, 0.45,
           text="How discriminating features are selected",
           size=20, bold=True, color=C_BLACK)

    # Scatter image — left 60% of slide
    img_w, img_h = W * 0.60, H - 0.65
    buf = _scatter_image(stats_df)
    slide.shapes.add_picture(buf, Inches(0.2), Inches(0.58),
                              Inches(img_w), Inches(img_h))

    # ── Right panel: method explanation ──────────────────────────────────────
    rx = img_w + 0.3    # right panel left edge
    rw = W - rx - 0.15  # right panel width

    # Background rectangle for right panel
    _rect(slide, rx, 0.58, rw, H - 0.73,
          fill_rgb=RGBColor(0xF7,0xF7,0xF7),
          edge_rgb=RGBColor(0xCC,0xCC,0xCC), edge_pt=1.0)

    # Panel heading
    _txbox(slide, rx+0.12, 0.68, rw-0.2, 0.38,
           text="Selection criteria", size=13, bold=True, color=C_TEXT_BLUE)

    # Criterion 1
    _txbox(slide, rx+0.12, 1.10, rw-0.2, 0.28,
           text="① Spearman ρ", size=11, bold=True, color=C_BLACK)
    _txbox(slide, rx+0.12, 1.38, rw-0.2, 0.85,
           text=("Monotonic correlation of each\nfeature with the classifier's\n"
                 "P(adhesion) score.\n\n|ρ| > 0.5 reference line shown."),
           size=10, color=C_DARK)

    # Criterion 2
    _txbox(slide, rx+0.12, 2.30, rw-0.2, 0.28,
           text="② ANOVA F-statistic", size=11, bold=True, color=C_BLACK)
    _txbox(slide, rx+0.12, 2.58, rw-0.2, 0.90,
           text=("One-way ANOVA across the\n4 outcome groups:\n"
                 "TP / FP / TN / FN.\n\nF > 100 reference line shown."),
           size=10, color=C_DARK)

    # Divider line
    _rect(slide, rx+0.12, 3.56, rw-0.24, 0.02,
          fill_rgb=RGBColor(0xCC,0xCC,0xCC), edge_rgb=None)

    # Key finding
    _txbox(slide, rx+0.12, 3.65, rw-0.2, 0.28,
           text="Key finding", size=11, bold=True, color=C_BLACK)
    _txbox(slide, rx+0.12, 3.93, rw-0.2, 1.60,
           text=("Features ranking high on BOTH\n"
                 "criteria are most informative.\n\n"
                 "All top features are GLCM\n"
                 "texture features — intensity\n"
                 "features (orange dots) rank\n"
                 "consistently lower."),
           size=10, color=C_DARK)

    # ── Quadrant annotation text boxes over the scatter image ─────────────────
    # Upper-right quadrant label
    _txbox(slide, img_w * 0.60, 0.65, img_w * 0.38, 0.60,
           text="High ρ + High F\n→ Correlated with adhesion\n   AND group-discriminating",
           size=8.5, italic=True, color=C_TEXT_BLUE,
           fill_rgb=RGBColor(0xEE,0xF4,0xFF),
           edge_rgb=C_POS_EDGE, edge_pt=0.8)

    # Upper-left quadrant label
    _txbox(slide, 0.22, 0.65, img_w * 0.38, 0.60,
           text="Low ρ + High F\n→ Correlated with non-adhesion\n   AND group-discriminating",
           size=8.5, italic=True, color=C_TEXT_RED,
           fill_rgb=RGBColor(0xFF,0xEE,0xEE),
           edge_rgb=C_NEG_EDGE, edge_pt=0.8)


# ── Slide 10: feature interpretation ─────────────────────────────────────────

def _build_slide10(prs: Presentation):
    slide = _blank_slide(prs)
    W, H  = 13.33, 7.5

    # Title
    _txbox(slide, 0.2, 0.05, W - 0.4, 0.40,
           text="Top GLCM features — meaning and adhesion relevance",
           size=19, bold=True, color=C_BLACK)

    # Header: what is GLCM
    hdr = slide.shapes.add_textbox(Inches(0.2), Inches(0.50),
                                    Inches(W-0.4), Inches(0.58))
    hdr.fill.solid()
    hdr.fill.fore_color.rgb = C_HDR_FILL
    hdr.line.color.rgb      = C_GREY_EDGE
    hdr.line.width          = Pt(0.8)
    tf = hdr.text_frame
    tf.word_wrap = True
    tf.margin_left   = Inches(0.12)
    tf.margin_top    = Inches(0.06)
    tf.margin_right  = Inches(0.12)
    tf.margin_bottom = Inches(0.06)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run = p.add_run()
    run.text = "What is GLCM? "
    run.font.size = Pt(10)
    run.font.bold = True
    run.font.color.rgb = C_TEXT_BLUE
    run2 = p.add_run()
    run2.text = ("Gray-Level Co-occurrence Matrix counts how often pixel pairs with given "
                 "intensity values appear at a fixed offset distance d. "
                 "Features are computed at d = 1, 2, 4 pixels in all directions and averaged. "
                 "All 50 CP features = 12 intensity statistics + 38 GLCM features "
                 "(13 measures × 3 distances).")
    run2.font.size = Pt(10)
    run2.font.color.rgb = C_DARK

    # ── Feature cards (2 columns × 4 rows) ───────────────────────────────────
    ncols, nrows = 2, 4
    margin_l = 0.20
    margin_r = 0.20
    gap_col  = 0.14
    gap_row  = 0.07
    card_top_start = 1.14
    card_area_h = H - card_top_start - 0.58  # reserve bottom for insight box
    card_h = (card_area_h - gap_row * (nrows - 1)) / nrows
    card_w = (W - margin_l - margin_r - gap_col) / ncols

    for idx, (col_key, display_name, rho_str, is_pos, desc) in enumerate(FEATURES):
        col = idx % ncols
        row = idx // ncols
        cx  = margin_l + col * (card_w + gap_col)
        cy  = card_top_start + row * (card_h + gap_row)

        fill = C_POS_FILL if is_pos else C_NEG_FILL
        edge = C_POS_EDGE if is_pos else C_NEG_EDGE
        acc  = C_TEXT_BLUE if is_pos else C_TEXT_RED

        # Card background rectangle
        bg = slide.shapes.add_shape(1,
             Inches(cx), Inches(cy), Inches(card_w), Inches(card_h))
        bg.fill.solid()
        bg.fill.fore_color.rgb = fill
        bg.line.color.rgb      = edge
        bg.line.width          = Pt(1.2)
        bg.text = ""

        # Feature name (left, bold)
        name_w = card_w * 0.65
        name_shape = slide.shapes.add_textbox(
            Inches(cx + 0.10), Inches(cy + 0.06),
            Inches(name_w),    Inches(card_h * 0.30))
        tf = name_shape.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        run = p.add_run()
        run.text = display_name
        run.font.size  = Pt(9)
        run.font.bold  = True
        run.font.color.rgb = C_BLACK

        # ρ value (right, bold, accent colour)
        rho_shape = slide.shapes.add_textbox(
            Inches(cx + name_w + 0.05), Inches(cy + 0.06),
            Inches(card_w - name_w - 0.15), Inches(card_h * 0.28))
        tf2 = rho_shape.text_frame
        p2  = tf2.paragraphs[0]
        p2.alignment = PP_ALIGN.RIGHT
        run2 = p2.add_run()
        run2.text = rho_str
        run2.font.size  = Pt(10)
        run2.font.bold  = True
        run2.font.color.rgb = acc

        # Description text
        desc_shape = slide.shapes.add_textbox(
            Inches(cx + 0.10), Inches(cy + card_h * 0.36),
            Inches(card_w - 0.20), Inches(card_h * 0.60))
        tf3 = desc_shape.text_frame
        tf3.word_wrap = True
        p3 = tf3.paragraphs[0]
        run3 = p3.add_run()
        run3.text = desc
        run3.font.size  = Pt(8.5)
        run3.font.color.rgb = C_DARK

    # ── Bottom insight box ────────────────────────────────────────────────────
    insight_top = H - 0.52
    ins = slide.shapes.add_textbox(
        Inches(0.20), Inches(insight_top),
        Inches(W - 0.40), Inches(0.46))
    ins.fill.solid()
    ins.fill.fore_color.rgb = C_INS_FILL
    ins.line.color.rgb      = C_INS_EDGE
    ins.line.width          = Pt(1.0)
    tf = ins.text_frame
    tf.word_wrap    = True
    tf.margin_left  = Inches(0.12)
    tf.margin_top   = Inches(0.05)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = ("Key insight:  Focal adhesion patches are compact, bright, spatially uniform structures "
                "(high homogeneity / energy, low entropy / contrast).  "
                "The classifier uses texture order — not raw intensity — to distinguish them.")
    run.font.size   = Pt(10)
    run.font.italic = True
    run.font.color.rgb = RGBColor(0x55, 0x44, 0x00)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading {PPT_PATH.name} ...")
    prs = Presentation(str(PPT_PATH))
    n   = len(prs.slides)
    print(f"  {n} slides")

    # Remove slides 9 and 10 if they exist (added by a previous run)
    while len(prs.slides) > 8:
        print(f"  Removing slide {len(prs.slides)} ...")
        _remove_last_n_slides(prs, 1)

    print("Computing feature statistics ...")
    stats_df = _compute_stats()

    print("Building slide 9 (feature selection — native shapes + scatter image) ...")
    _build_slide9(prs, stats_df)

    print("Building slide 10 (feature interpretation — fully native shapes) ...")
    _build_slide10(prs)

    prs.save(str(PPT_PATH))
    sz = PPT_PATH.stat().st_size // 1024
    print(f"\nSaved → {PPT_PATH}  ({sz} KB, {len(prs.slides)} slides)")
    print("Slides 9 and 10 are now fully editable in PowerPoint.")


if __name__ == "__main__":
    main()
