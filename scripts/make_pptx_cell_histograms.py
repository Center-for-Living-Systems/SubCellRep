#!/usr/bin/env python3
"""
make_pptx_cell_histograms.py

Deck: DS1 91 cells, grid patches, 5-class prediction, per-cell histogram, examples with overlays.
Inputs : results/cell_histograms/* (cell_class_histograms.py)
Output : results/cell_histograms.pptx
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_pptx_le_benchmark import (_prs, _blank, _txt, _slide_header, _img, _bullet_box, C_HEAD, C_WHITE, SW, SH, RGBColor, Inches)
from make_pptx_ordinal_cp import table, fig_slide

REPO = Path(__file__).resolve().parents[1]
D = REPO / "results" / "cell_histograms"; OUT = REPO / "results" / "cell_histograms.pptx"
S = json.load(open(D / "summary.json")); E = pd.read_csv(D / "examples.csv"); CL = pd.read_csv(D / "cell_classifier.csv")
H = pd.read_csv(D / "cell_histograms.csv", index_col=0)
SHORT = ["No adhesion", "nascent", "focal complex", "focal adhesion", "fibrillar"]
mc, my = S["mean_fractions"]["control"], S["mean_fractions"]["ycomp"]
pc = lambda d, k: f"{d['f_' + k]:.0%}"


def main():
    prs = _prs(); s = _blank(prs)
    bg = s.shapes.add_shape(1, 0, 0, SW, SH); bg.fill.solid(); bg.fill.fore_color.rgb = C_HEAD; bg.line.fill.background()
    _txt(s, "What is each cell made of?", Inches(0.8), Inches(2.3), Inches(11.7), Inches(1.0), bold=True, size_pt=38, color=C_WHITE)
    _txt(s, "Per-cell histogram of the 5 predicted patch classes (no adhesion, nascent, focal complex, focal adhesion, fibrillar) on grid patches, as a cell-level representation",
         Inches(0.8), Inches(3.5), Inches(11.5), Inches(1.0), size_pt=17, color=RGBColor(0xCA, 0xDC, 0xFC))
    _txt(s, "DS1 vinculin · 91 cells (50 control, 41 ycomp) · 27,637 grid patches · temporary results", Inches(0.8), Inches(4.7), Inches(11.5), Inches(0.5), size_pt=14, color=RGBColor(0xCA, 0xDC, 0xFC))

    s = _blank(prs); _slide_header(s, "Idea and pipeline", "From patch predictions to a cell-level description")
    _bullet_box(s, Inches(0.5), Inches(1.3), Inches(6.1), Inches(5.6), "Pipeline", [
        "Each cell (one DS1 frame) is tiled by non-overlapping 32x32 grid patches that fall on the cell; median 293 patches per cell (158–464)",
        "Stage 1: CP features + class-balanced logistic regression, adhesion vs no adhesion (trained on all B1+B2 labels)",
        "Stage 2: CP features + class-balanced logistic regression on the 4 FA classes, applied to patches that stage 1 calls adhesion",
        "Result: every patch gets one of 5 classes; a cell becomes a 5-bin histogram (fractions sum to 1)",
        "Grid patches only; the local-maximum sampling is not used"], bullet_size=12)
    _bullet_box(s, Inches(6.8), Inches(1.3), Inches(6.0), Inches(5.6), "Read with care", [
        f"The models were fit on all B1+B2 labels, with no hold-out. About {S['labelled_patch_fraction']:.0%} of these patches are labelled, so for them the predictions are training predictions",
        "Held-out accuracy from the earlier CV: gate 0.92 balanced accuracy; FA4 about 0.56 (B1 test) to 0.63 (B2 test), so single-patch calls are noisy",
        "Only 38 fibrillar labels went into training, so the fibrillar fraction is the least reliable bin",
        "The cell histogram averages over hundreds of patches, which smooths some of the patch-level errors"], bullet_size=12)

    fig_slide(prs, "Histogram of every cell", "Each column is one cell; sorted by no-adhesion fraction within each condition", D / "overview_stacked.png", 12.3, [
        f"Control cells hold more mature classes: focal adhesion {pc(mc, 'focal adhesion')} and fibrillar {pc(mc, 'fibrillar')} of patches on average, no adhesion {pc(mc, 'No adhesion')}",
        f"ycomp cells are dominated by no adhesion ({pc(my, 'No adhesion')}) and nascent ({pc(my, 'nascent')}); focal adhesion {pc(my, 'focal adhesion')}, fibrillar {pc(my, 'fibrillar')}",
        "Within each condition the cells vary a lot: no-adhesion fraction runs from about 0.26 to 0.77, so the histogram carries cell-to-cell information"], size=12)
    fig_slide(prs, "Mean histogram per condition", "Bars: mean ± SD over cells; dots: single cells", D / "mean_histograms.png", 11.2, [
        f"Biggest shifts, control to ycomp: focal adhesion {pc(mc, 'focal adhesion')} to {pc(my, 'focal adhesion')}, fibrillar {pc(mc, 'fibrillar')} to {pc(my, 'fibrillar')}, nascent {pc(mc, 'nascent')} to {pc(my, 'nascent')}",
        "Focal complex is similar in both conditions"], size=12)

    for _, r in E.iterrows():
        f = {k: r[f"f_{k}"] for k in SHORT}; top = sorted(f, key=f.get, reverse=True)[:2]
        fig_slide(prs, f"Example {r.id}: {r.cell} ({r.cond})",
                  f"{r.tag} predicted-adhesion cell of its condition · {int(r.n)} grid patches · {r.f_adhesion_any:.0%} of patches predicted adhesion",
                  D / f"example_{r.id}.png", 12.3, [
            f"Dominant classes: {top[0]} {f[top[0]]:.0%} and {top[1]} {f[top[1]]:.0%}; nascent {f['nascent']:.0%}, focal complex {f['focal complex']:.0%}, focal adhesion {f['focal adhesion']:.0%}, fibrillar {f['fibrillar']:.0%}",
            "Left: whole cell rebuilt from its grid patches. Middle: the same cell with each patch coloured by predicted class. Right: the cell's histogram"], size=12)

    s = _blank(prs); _slide_header(s, "The histogram as a cell representation", "5-d composition per cell; examples E1–E6 circled")
    p = D / "cell_pca.png"; w, h = Image.open(p).size; fw = 5.9
    _img(s, p, Inches(0.4), Inches(1.2), width=Inches(fw))
    rows = [["Features (control vs ycomp)", "AUROC"]] + [[r.features, f"{r.auroc_mean:.3f} ± {r.auroc_sd:.3f}"] for r in CL.itertuples()]
    table(s, rows, Inches(6.6), Inches(1.3), Inches(6.3), [4.2, 2.1], row_h=0.42, size=12)
    _bullet_box(s, Inches(6.6), Inches(3.6), Inches(6.3), Inches(3.8), "Interpretation", [
        f"Two PCs hold {S['pca_var'][0] + S['pca_var'][1]:.0%} of the histogram variance; PC1 ({S['pca_var'][0]:.0%}) runs from no adhesion to focal adhesion/fibrillar",
        "A logistic regression on the 5 fractions separates control from ycomp cells almost perfectly (repeated 5-fold CV over cells); cell size alone does not (0.53)",
        "This is a toy check, not a validated cell-status classifier: the two conditions may also differ in acquisition or normalisation, and the models saw the labelled patches of these cells"], bullet_size=11)

    s = _blank(prs); _slide_header(s, "Caveats and next steps", "")
    _bullet_box(s, Inches(0.5), Inches(1.3), Inches(6.1), Inches(5.6), "Caveats", [
        "Temporary result: models trained on all labels, predictions not held-out for labelled patches",
        "Fibrillar is trained on 38 labels and probably over-called in control cells; read it as a trend, not a count",
        "Fractions use all grid patches on the cell, so cell size and shape change the denominator",
        "Control vs ycomp differences could partly be batch effects; only 2 conditions are available here"], bullet_size=12)
    _bullet_box(s, Inches(6.8), Inches(1.3), Inches(6.0), Inches(5.6), "Possible next steps", [
        "Re-run with out-of-fold predictions (frame-grouped CV) so no cell is scored by a model that saw its labels",
        "Weight each patch by prediction confidence, or use the soft probabilities instead of the hard class",
        "Add spatial descriptors (distance to the cell edge, patch neighbourhood) to the histogram",
        "Check the histogram against a cell-level read-out from the biology team, e.g. treatment or migration state"], bullet_size=12)
    prs.save(OUT); print("saved", OUT, len(prs.slides._sldIdLst), "slides")


if __name__ == "__main__":
    main()
