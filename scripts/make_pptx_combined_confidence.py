#!/usr/bin/env python3
"""
make_pptx_combined_confidence.py

Merge performance PPTs with pred-confidence PPTs into 3 combined decks:
  1. combined_le_confidence.pptx
       clf_comparison slides 1-4 (LE cross/union/budget perf)
       + pred_confidence_le_cross_b1_to_b2v2_{feat}_{clf}.pptx (prob dist, patches, features)

  2. combined_image_eff_b1_full_confidence.pptx
       image_efficiency_b1_full.pptx (all perf slides)
       + pred_confidence_image_eff_b1_full_{feat}_{clf}.pptx

  3. combined_image_eff_b1b2v2_confidence.pptx
       image_efficiency_b1b2v2.pptx (all perf slides)
       + pred_confidence_image_eff_b1b2v2_{feat}_{clf}.pptx

Strategy: extract title text + picture bytes from each source slide and
recreate them in the combined PPT — avoids the XML/relationship merge mess.

Usage:
  python scripts/make_pptx_combined_confidence.py [--feat cp] [--clf lgbm]
"""
from __future__ import annotations
import argparse, io
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.shapes import MSO_SHAPE_TYPE

REPO    = Path(__file__).resolve().parent.parent
RES     = REPO / "results"
OUT_DIR = RES


def _extract_slides(pptx_path: Path) -> list[dict]:
    """Return list of {title, img_buf} for each slide in the given PPT."""
    prs   = Presentation(str(pptx_path))
    slides = []
    for slide in prs.slides:
        title = ""
        img_buf = None
        for shape in slide.shapes:
            if shape.has_text_frame and not title:
                t = shape.text_frame.text.strip()
                if t:
                    title = t
            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE and img_buf is None:
                img_buf = io.BytesIO(shape.image.blob)
        slides.append({"title": title, "img_buf": img_buf})
    return slides


def _new_prs() -> Presentation:
    prs = Presentation()
    prs.slide_width  = Inches(13.33)
    prs.slide_height = Inches(7.5)
    return prs


def _add_slide(prs: Presentation, title: str, img_buf: io.BytesIO | None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    W, H  = prs.slide_width, prs.slide_height

    # Remove any placeholder shapes from the layout
    for ph in list(slide.placeholders):
        ph.element.getparent().remove(ph.element)

    if title:
        txb = slide.shapes.add_textbox(Inches(0.3), Inches(0.1),
                                        W - Inches(0.6), Inches(0.5))
        tf  = txb.text_frame
        tf.text = title
        tf.paragraphs[0].runs[0].font.size = Pt(20)
        tf.paragraphs[0].runs[0].font.bold = True

    if img_buf is not None:
        img_buf.seek(0)
        margin = Inches(0.3)
        top    = Inches(0.65) if title else Inches(0.1)
        slide.shapes.add_picture(img_buf, margin, top,
                                  W - 2 * margin, H - top - Inches(0.1))


def _add_divider(prs: Presentation, text: str):
    """Add a plain text divider/section slide."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    W, H  = prs.slide_width, prs.slide_height
    for ph in list(slide.placeholders):
        ph.element.getparent().remove(ph.element)
    txb = slide.shapes.add_textbox(Inches(1), Inches(2.8), W - Inches(2), Inches(2))
    tf  = txb.text_frame
    tf.word_wrap = True
    p   = tf.paragraphs[0]
    p.text = text
    p.runs[0].font.size = Pt(28)
    p.runs[0].font.bold = True
    p.alignment = 2  # CENTER


def _copy_slides(prs: Presentation, src_slides: list[dict],
                 indices: list[int] | None = None):
    """Copy specified (or all) slides from extracted slide list into prs."""
    to_copy = src_slides if indices is None else [src_slides[i] for i in indices]
    for s in to_copy:
        _add_slide(prs, s["title"], s["img_buf"])


def _confidence_pptx(source: str, feat: str, clf: str) -> Path:
    """Resolve path to a pred_confidence PPT."""
    if source == "le_cross":
        return RES / f"pred_confidence_le_cross_b1_to_b2v2_{feat}_{clf}.pptx"
    return RES / f"pred_confidence_{source}_{feat}_{clf}.pptx"


def build(feat: str, clf: str):
    feat_lbl = "CellProfiler" if feat == "cp" else "ilastik"
    clf_lbl  = "LightGBM"     if clf  == "lgbm" else "LogReg"

    # ── 1. combined_le_confidence.pptx ───────────────────────────────────────
    print("Building combined_le_confidence.pptx ...")
    prs = _new_prs()

    # Title slide
    _add_divider(prs, "Label Efficiency — Performance + Confidence Analysis")

    # Performance slides from clf_comparison.pptx (slides 0-3: title, cross, union, budget)
    clf_cmp = _extract_slides(RES / "clf_comparison.pptx")
    # slide 0 = title (skip, we added our own)
    # slides 1-3 = cross, union, budget perf
    _copy_slides(prs, clf_cmp, indices=[1, 2, 3])

    # Divider: confidence analysis
    _add_divider(prs, f"Confidence Analysis — LE Cross (B1→B2v2)\n{feat_lbl} / {clf_lbl}")

    # All confidence slides except title (index 0)
    conf_le = _extract_slides(_confidence_pptx("le_cross", feat, clf))
    _copy_slides(prs, conf_le, indices=list(range(1, len(conf_le))))

    out1 = OUT_DIR / f"combined_le_confidence_{feat}_{clf}.pptx"
    prs.save(str(out1))
    print(f"  Saved → {out1}  ({out1.stat().st_size // 1024} KB, {len(prs.slides)} slides)")

    # ── 2. combined_image_eff_b1_full_confidence.pptx ────────────────────────
    print("Building combined_image_eff_b1_full_confidence.pptx ...")
    prs = _new_prs()

    _add_divider(prs, "Image Efficiency (B1 Full) — Performance + Confidence Analysis")

    b1_full_perf = _extract_slides(RES / "image_efficiency_b1_full.pptx")
    # Skip the first title slide of the source PPT
    start = 1 if b1_full_perf[0]["img_buf"] is None else 0
    _copy_slides(prs, b1_full_perf, indices=list(range(start, len(b1_full_perf))))

    _add_divider(prs, f"Confidence Analysis — Image Efficiency B1 Full\n{feat_lbl} / {clf_lbl}")

    conf_b1f = _extract_slides(_confidence_pptx("image_eff_b1_full", feat, clf))
    _copy_slides(prs, conf_b1f, indices=list(range(1, len(conf_b1f))))

    out2 = OUT_DIR / f"combined_image_eff_b1_full_confidence_{feat}_{clf}.pptx"
    prs.save(str(out2))
    print(f"  Saved → {out2}  ({out2.stat().st_size // 1024} KB, {len(prs.slides)} slides)")

    # ── 3. combined_image_eff_b1b2v2_confidence.pptx ─────────────────────────
    print("Building combined_image_eff_b1b2v2_confidence.pptx ...")
    prs = _new_prs()

    _add_divider(prs, "Image Efficiency (B1/B2v2) — Performance + Confidence Analysis")

    b1b2v2_perf = _extract_slides(RES / "image_efficiency_b1b2v2.pptx")
    start = 1 if b1b2v2_perf[0]["img_buf"] is None else 0
    _copy_slides(prs, b1b2v2_perf, indices=list(range(start, len(b1b2v2_perf))))

    _add_divider(prs, f"Confidence Analysis — Image Efficiency B1/B2v2\n{feat_lbl} / {clf_lbl}")

    conf_b2 = _extract_slides(_confidence_pptx("image_eff_b1b2v2", feat, clf))
    _copy_slides(prs, conf_b2, indices=list(range(1, len(conf_b2))))

    out3 = OUT_DIR / f"combined_image_eff_b1b2v2_confidence_{feat}_{clf}.pptx"
    prs.save(str(out3))
    print(f"  Saved → {out3}  ({out3.stat().st_size // 1024} KB, {len(prs.slides)} slides)")

    print("\nDone.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="cp",   choices=["cp", "ilastik"])
    ap.add_argument("--clf",  default="lgbm", choices=["lgbm", "logreg"])
    args = ap.parse_args()
    build(args.feat, args.clf)


if __name__ == "__main__":
    main()
