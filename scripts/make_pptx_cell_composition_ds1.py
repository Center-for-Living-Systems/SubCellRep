#!/usr/bin/env python3
"""
make_pptx_cell_composition_ds1.py — deck: two-stage FA pipeline (adhesion gate -> FA4) and DS1 per-cell
5-class composition (91 cells). Inputs: eval_results/lm_fa_grid/*, eval_results/cell_composition/*.
Usage: python scripts/make_pptx_cell_composition_ds1.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.util import Inches, Pt

R = Path(__file__).resolve().parents[1]
E = R / "eval_results"
BEST = "il_logreg_ps32"
BD = E / "lm_fa_grid" / BEST
CD = E / "cell_composition"
FIG = CD / f"ds1_cell_composition_lm_fa_grid_{BEST}.png"
OUT = R / "results" / "ds1_cell_composition.pptx"
OUT.parent.mkdir(exist_ok=True)
CL = ["No adhesion", "Nascent Adhesion", "focal complex", "focal adhesion", "fibrillar adhesion"]
COL = ["9467BD", "1F77B4", "FF7F0E", "2CA02C", "D62728"]
SHORT = ["NA", "FC", "FA", "FiBA"]

prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
DARK = RGBColor(0x22, 0x2B, 0x38)


def tb(s, x, y, w, h, text, size=16, bold=False, color=DARK):
    t = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = t.text_frame; tf.word_wrap = True
    lines = text if isinstance(text, list) else [text]
    for i, l in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = l; p.font.size = Pt(size); p.font.bold = bold; p.font.color.rgb = color; p.space_after = Pt(6)
    return t


def slide(title, sub=None):
    s = prs.slides.add_slide(BLANK)
    tb(s, .5, .3, 12.3, .8, title, 28, True)
    if sub: tb(s, .5, 1.0, 12.3, .5, sub, 15, color=RGBColor(0x66, 0x66, 0x66))
    return s


def pic(s, path, x, y, w=None, h=None):
    s.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)


def table(s, rows, x, y, w, h, size=12, hl=None):
    t = s.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(h)).table
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            c = t.cell(i, j); c.text = str(v)
            for p in c.text_frame.paragraphs:
                p.font.size = Pt(size); p.font.bold = (i == 0) or (hl is not None and i == hl)
    return t


def bullets(s, items, x=.6, y=1.6, w=12, size=18):
    tb(s, x, y, w, 5, ["• " + i for i in items], size)


m = json.load(open(BD / "metrics.json"))
comp = pd.read_csv(CD / f"ds1_cell_composition_lm_fa_grid_{BEST}.csv")
cm = np.array(m["stageB_confusion"])

# crops of the summary figure
im = Image.open(FIG); W, H = im.size
im.crop((0, int(.07 * H), W, int(.54 * H))).save(CD / "_crop_stack.png")
im.crop((0, int(.53 * H), W, int(.77 * H))).save(CD / "_crop_hist.png")
im.crop((0, int(.77 * H), W, H)).save(CD / "_crop_pca.png")

# 1 title
s = prs.slides.add_slide(BLANK)
tb(s, .8, 2.3, 11.5, 1.2, "Per-cell adhesion composition on DS1", 40, True)
tb(s, .8, 3.5, 11.5, 1.5, ["Two-stage FA classifier (adhesion gate → FA4) aggregated into 5-class histograms for 91 cells",
                           "Temporary result illustration"], 20)

# 2 summary
s = slide("Summary")
bullets(s, [
    "Best two-stage model among 8 non-AE configs: intensity-lowlevel (IL) features + logistic regression, ps32, grid patches.",
    f"Stage A (adhesion vs no adhesion): balanced acc {m['stageA_oof_bal_acc']:.3f}.  Stage B (FA 4-class): balanced acc {m['stageB_oof_bal_acc']:.3f}.  End-to-end acc {m['end_to_end_acc_all_true_adhesion']:.3f}.",
    "Every patch of each of the 91 DS1 cells gets one of 5 labels; each cell becomes a 5-number composition vector.",
    "Control cells: ~50% no adhesion, ~27% focal adhesion (FA), ~7% fibrillar (FiBA). Ycomp cells: ~68% no adhesion, ~18% nascent (NA), ~4% FA, <1% FiBA.",
    "Composition alone separates control vs ycomp (leave-one-cell-out AUC 0.99) → candidate cell-status representation / classifier.",
    "Caveats: stage B is moderate (0.64 bal. acc), FiBA is poorly resolved (17 labelled cells), 91 cells only.",
])

# 3 pipeline
s = slide("Pipeline", "Out-of-fold predictions for labelled cells; models trained on all labels for unlabelled patches")
bullets(s, [
    "Patches: 32×32 grid patches on DS1 pax channel (CIO-normalised), 15,513 patches over 91 images.",
    "Labels: B1 ∪ B2 (B2 wins on disagreement); 1,561 labelled patches (adhesion: 149 NA / 216 FC / 244 FA / 17 FiBA).",
    "Features: IL = multi-scale Gaussian / LoG / gradient / DoG / structure-tensor / Hessian mean+std statistics (80 features).",
    "Stage A: adhesion vs no-adhesion logistic regression (standardised features).",
    "Stage B: 4-class FA logistic regression trained on labelled adhesion patches; applied to patches that pass stage A.",
    "5-class call per patch = No adhesion, or FA class from stage B. Image-grouped 5-fold CV for all labelled metrics.",
    "Cell composition = fraction of the cell's patches in each of the 5 classes.",
], size=17)

# 4 model selection table
runs = []
for r in ["il", "cp"]:
    for c in ["logreg", "lgbm"]:
        for ps in [32, 64]:
            mm = json.load(open(E / "lm_fa_grid" / f"{r}_{c}_ps{ps}" / "metrics.json"))
            runs.append([r.upper(), c, ps, f"{mm['stageA_oof_bal_acc']:.3f}", f"{mm['stageB_oof_bal_acc']:.3f}",
                         f"{mm['stageB_oof_acc']:.3f}", f"{mm['end_to_end_acc_all_true_adhesion']:.3f}"])
s = slide("Model selection: which two-stage combination is best?", "Grid patches, out-of-fold, image-grouped 5-fold")
table(s, [["Features", "Classifier", "Patch", "Stage A bal acc", "Stage B bal acc", "Stage B acc", "End-to-end acc"]] + runs,
      .6, 1.7, 12, 4.2, 14, hl=1)
tb(s, .6, 6.1, 12, 1, ["IL + logreg ps32 (bold) has the best stage B balanced accuracy and near-best stage A / end-to-end.",
                       "Logistic regression beats LightGBM by ~0.1 on stage B balanced accuracy (LightGBM likely overfits small classes). CP = CellProfiler-style features."], 14)

# 5 stage A / B detail
s = slide(f"Best model: {BEST}", "Stage B confusion (labelled adhesion patches, out-of-fold)")
pic(s, BD / "fa_confusion.png", .4, 1.6, w=8.2)
g = m["stageA_oof_gate_pass_rate_by_fa_class"]
table(s, [["Metric", "Value"],
          ["Stage A bal acc", f"{m['stageA_oof_bal_acc']:.3f}"],
          ["Stage A recall adhesion / no-adh", f"{m['stageA_oof_recall_adhesion']:.3f} / {m['stageA_oof_recall_noadhesion']:.3f}"],
          ["Stage B bal acc / acc", f"{m['stageB_oof_bal_acc']:.3f} / {m['stageB_oof_acc']:.3f}"],
          ["End-to-end acc (given gate pass)", f"{m['end_to_end_acc_given_gate_pass']:.3f}"],
          ["Gate pass NA / FC / FA / FiBA", " / ".join(f"{v:.2f}" for v in g.values())]], 8.8, 1.8, 4.2, 2.8, 11)
tb(s, 8.8, 4.9, 4.2, 2, ["Errors are mostly between neighbouring classes (NA↔FC↔FA).",
                         "FiBA: only 17 labelled, and the gate passes 41% of them."], 13)

# 6 embeddings
s = slide("Predicted adhesion patches: embeddings", "Rows: predicted FA class, true FA class; columns: PCA / UMAP / PHATE")
pic(s, BD / "fa_embeddings.png", 1.5, 1.5, h=5.8)

# 7 per-cell stack
s = slide("Per-cell 5-class composition (91 cells)", "Each bar = one cell; strip: red = control (50), green = ycomp (41); sorted by adhesion fraction")
pic(s, CD / "_crop_stack.png", .4, 1.6, w=12.5)

# 7b example cells with overlays
ex = pd.read_csv(CD / "overlays" / "examples.csv")
for c, note in [("control", ["Top: most FA-rich control cell, strong FA + FiBA.",
                             "Middle: median control cell; FA at the edge, no-adhesion in the interior.",
                             "Bottom: least FA-rich control cell, mostly no-adhesion with a few FA at one edge."]),
                ("ycomp", ["Ycomp cells show mainly NA / FC at the periphery and almost no FA / FiBA.",
                           "Top/middle/bottom: most FA-rich, median, least FA-rich ycomp cell (by composition PC1)."])]:
    s = slide(f"Example {c} cells: prediction overlay and 5-class histogram",
              "Raw pax image | 5-class call on each 32 px grid patch (cell patches only) | fraction of the cell's patches per class")
    pic(s, CD / "overlays" / f"{c}_examples.png", .3, 1.45, h=5.95)
    e = ex[ex.condition == c]
    tb(s, 6.9, 1.7, 6, 5, note + [f"Frames shown: {', '.join(str(x) for x in e.frame_idx)} (rows top→bottom).",
                                 "Overlay colours = histogram colours; selection by position on composition PC1, not cherry-picked by eye."], 15)

# 8 mean composition chart + table
s = slide("Mean composition by condition")
cd = CategoryChartData(); cd.categories = ["control", "ycomp"]
mean = comp.groupby("condition")[CL].mean()
for c in CL: cd.add_series(c, [float(mean.loc["control", c]), float(mean.loc["ycomp", c])])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_STACKED_100, Inches(.5), Inches(1.3), Inches(6), Inches(5.8), cd)
ch = gf.chart; ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False
for ser, col in zip(ch.plots[0].series, COL):
    ser.format.fill.solid(); ser.format.fill.fore_color.rgb = RGBColor.from_string(col)
sd = comp.groupby("condition")[CL].std()
rows = [["Class", "Control mean±sd", "Ycomp mean±sd"]] + [[c, f"{mean.loc['control', c]:.3f} ± {sd.loc['control', c]:.3f}",
                                                         f"{mean.loc['ycomp', c]:.3f} ± {sd.loc['ycomp', c]:.3f}"] for c in CL]
table(s, rows, 6.9, 1.7, 6, 3, 12)
tb(s, 6.9, 5, 6, 2, ["Ycomp shifts mass from FA / FiBA to no-adhesion and NA.",
                     f"Patches per cell: control {comp[comp.condition=='control'].n_patches.mean():.0f}, ycomp {comp[comp.condition=='ycomp'].n_patches.mean():.0f} (fractions are normalised per cell)."], 14)

# 9 histograms
s = slide("Histograms of per-class cell fraction", "Control vs ycomp; each count is a cell")
pic(s, CD / "_crop_hist.png", .4, 1.8, w=12.5)
tb(s, .6, 5.6, 12, 1.5, ["FA and FiBA fractions separate the conditions almost completely; NA and no-adhesion shift in the opposite direction.",
                         "Focal complex overlaps heavily — it is the least condition-specific class."], 16)

# 10 representation / classifier
s = slide("From composition to a cell-status representation", "PCA of the 5-d composition and leave-one-cell-out logistic regression (log fractions)")
pic(s, CD / "_crop_pca.png", .4, 1.7, w=12.5)
tb(s, .6, 5.6, 12, 1.6, ["PC1 (81% of variance) is essentially the 'mature adhesion (FA/FiBA) vs no-adhesion/NA' axis and separates the conditions.",
                         "Control cells spread along PC1 → a continuous cell-status score, not just a binary label."], 16)

# 11 caveats
s = slide("Caveats")
bullets(s, [
    "AUC 0.99 is for control vs ycomp on 91 cells with a 5-number input; image-level batch/intensity differences can contribute. Illustration, not a validated result.",
    "Stage B is moderate (0.64 balanced acc). Per-cell class fractions inherit its confusions, especially NA↔FC↔FA.",
    "FiBA: 17 labelled cells; the model predicts ~592 FiBA patches across all patches vs 13 correct among labelled ones. FiBA fractions are likely over-estimated and noisy.",
    "Gate errors (~9%) add false adhesion calls in no-adhesion regions and drop some true adhesions.",
    "Labelled patches use out-of-fold predictions; unlabelled ones use models trained on all labelled cells.",
    "Grid patches tile the whole image, so 'no adhesion' fraction partly reflects how much of the field is cell vs background.",
])

# 12 next steps
s = slide("Next steps")
bullets(s, [
    "Define a continuous cell-status score (PC1 or mature-adhesion fraction) and test it against cell-level phenotypes.",
    "Validate on held-out datasets / replicates (DS2–DS4) with composition vectors computed by the same frozen model.",
    "Restrict per-cell fractions to cell-masked patches to remove background effects.",
    "Collect more FiBA labels and/or merge FA+FiBA for a more robust 'mature' class.",
    "Add soft composition (summed probabilities) alongside hard calls; compare with AE-based features for stage B.",
    "Cluster cells in composition space to look for sub-states within control.",
])
prs.save(OUT); print(OUT)
