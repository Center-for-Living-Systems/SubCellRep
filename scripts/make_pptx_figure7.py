#!/usr/bin/env python3
"""
make_pptx_figure7.py

Deck of the panels for Figure 7: four FA subclasses; binary vs subtype features (CP features, DS1 vinc).
Inputs : results/figure7/*, results/cv_b1b2/{summary.csv,cv_cm_*.png}, results/ordinal_cp/fsel_*
Output : results/figure7.pptx
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_pptx_le_benchmark import (_prs, _blank, _txt, _slide_header, _bullet_box, C_HEAD, C_WHITE, SW, SH, RGBColor, Inches)
from make_pptx_ordinal_cp import table, fig_slide

REPO = Path(__file__).resolve().parents[1]
F7, CV, OD = REPO / "results" / "figure7", REPO / "results" / "cv_b1b2", REPO / "results" / "ordinal_cp"
OUT = REPO / "results" / "figure7.pptx"
CNT = pd.read_csv(F7 / "class_counts.csv", index_col=0); G = pd.read_csv(F7 / "gate_per_subclass.csv").set_index("cls")
S = pd.read_csv(CV / "summary.csv"); PC = pd.read_csv(F7 / "fa4_per_class.csv")
FIN = pd.read_csv(OD / "fsel_final_sets.csv").set_index("task"); FQ = pd.read_csv(OD / "fsel_pick_freq.csv")
SJ = json.load(open(F7 / "summary.json"))


def sm(rep, test, col):
    return S[(S.rep == rep) & (S.src == "B1+B2") & (S.test == test)].iloc[0][col]


def pcs(m, t, c, col):
    return PC[(PC.method == m) & (PC.test == t) & (PC.cls == c)].iloc[0][col]


def main():
    prs = _prs(); s = _blank(prs)
    bg = s.shapes.add_shape(1, 0, 0, SW, SH); bg.fill.solid(); bg.fill.fore_color.rgb = C_HEAD; bg.line.fill.background()
    _txt(s, "Figure 7 panels: four FA subclasses", Inches(0.8), Inches(2.3), Inches(11.7), Inches(1.0), bold=True, size_pt=38, color=C_WHITE)
    _txt(s, "Subclass labels, binary and four-class classifiers, and which features separate adhesion from no adhesion versus the subtypes",
         Inches(0.8), Inches(3.5), Inches(11.5), Inches(1.0), size_pt=17, color=RGBColor(0xCA, 0xDC, 0xFC))
    _txt(s, "DS1 vinculin · B1 (random) + B2 (prototype) labels · frame-grouped cross-validation · draft panels", Inches(0.8), Inches(4.7), Inches(11.5), Inches(0.5), size_pt=14, color=RGBColor(0xCA, 0xDC, 0xFC))

    # 1 counts
    c = CNT["B1+B2 combined"]
    fig_slide(prs, "1. Four FA subclasses: definitions and label counts", "Purple = no adhesion (the binary negative); blue / orange / green / red = nascent / focal complex / focal adhesion / fibrillar",
              F7 / "class_counts.png", 8.6, [
        f"Four ordered subclasses: nascent < focal complex < focal adhesion < fibrillar; no adhesion is the negative class of the binary task. Combined set (B2 priority): {int(c.iloc[1])} nascent, {int(c.iloc[2])} focal complex, {int(c.iloc[3])} focal adhesion, {int(c.iloc[4])} fibrillar, {int(c.iloc[0])} no adhesion",
        f"B1 and B2 differ in class mix: B1 has more focal complex ({int(CNT.loc['focal complex', 'B1 (random)'])} vs {int(CNT.loc['focal complex', 'B2 (prototype)'])}), B2 more focal adhesion ({int(CNT.loc['focal adhesion', 'B2 (prototype)'])} vs {int(CNT.loc['focal adhesion', 'B1 (random)'])})",
        f"Fibrillar is very scarce ({int(CNT.loc['fibrillar adhesion', 'B1 (random)'])} in B1, {int(CNT.loc['fibrillar adhesion', 'B2 (prototype)'])} in B2), so every fibrillar number is exploratory"], size=11)
    fig_slide(prs, "Examples of each class", "Random labelled patches, 32x32, same brightness scale across classes", F7 / "patch_examples.png", 12.0, [
        "Nascent and focal complex patches are dark with small spots; focal adhesion shows bright elongated structures; fibrillar patches are broader and fainter fibres; no adhesion is diffuse texture",
        "Adjacent classes look alike, which matches where the classifiers confuse them"], size=11)

    # 2 gate
    fig_slide(prs, "2. Binary adhesion / no-adhesion classifier on the subclass labels", f"CP features + class-balanced logistic regression; frame-grouped CV; balanced accuracy {SJ['gate_bal_acc']:.3f}", F7 / "gate_per_subclass.png", 11.6, [
        f"Correct calls: no adhesion {G.loc['No adhesion', 'correct_rate']:.0%}, nascent {G.loc['nascent', 'correct_rate']:.0%}, focal complex {G.loc['focal complex', 'correct_rate']:.0%}, focal adhesion {G.loc['focal adhesion', 'correct_rate']:.0%}",
        f"Fibrillar is the weak subclass: only {G.loc['fibrillar', 'correct_rate']:.0%} called adhesion ({int(G.loc['fibrillar', 'n_wrong'])} of {int(G.loc['fibrillar', 'n'])}), with a wide interval from just 38 patches",
        f"Largest error count is no adhesion called adhesion ({int(G.loc['No adhesion', 'n_wrong'])} patches); focal complex is the least reliable of the three common subclasses"], size=11)

    # 3 four-class metrics
    fig_slide(prs, "3. Four-class classifier: CP, ilastik, CP+ilastik, SupCon-AE", "Trained on B1+B2 from other cells, tested on B1 (representative) and on B2", F7 / "fa4_metrics.png", 11.0, [
        f"Test B1: balanced accuracy CP {sm('CP', 'B1', 'bal_acc'):.2f}, ilastik {sm('IL', 'B1', 'bal_acc'):.2f}, CP+IL {sm('CP+IL', 'B1', 'bal_acc'):.2f}, SupCon-AE {sm('AE_b12', 'B1', 'bal_acc'):.2f}; macro-F1 {sm('CP', 'B1', 'macro_f1'):.2f} / {sm('IL', 'B1', 'macro_f1'):.2f} / {sm('CP+IL', 'B1', 'macro_f1'):.2f} / {sm('AE_b12', 'B1', 'macro_f1'):.2f}",
        f"Test B2: balanced accuracy {sm('CP', 'B2', 'bal_acc'):.2f} / {sm('IL', 'B2', 'bal_acc'):.2f} / {sm('CP+IL', 'B2', 'bal_acc'):.2f} / {sm('AE_b12', 'B2', 'bal_acc'):.2f}; macro-F1 {sm('CP', 'B2', 'macro_f1'):.2f} / {sm('IL', 'B2', 'macro_f1'):.2f} / {sm('CP+IL', 'B2', 'macro_f1'):.2f} / {sm('AE_b12', 'B2', 'macro_f1'):.2f}",
        "The hand-crafted sets are statistically indistinguishable from each other; the AE is lower on both tests and its interval overlaps them only partly",
        "Hand-crafted rows pool 5 fold assignments; the AE row is one fold assignment (about 600 patches on B1), so its interval is wider"], size=11)

    # 4 per-class precision / recall
    fig_slide(prs, "Per-class recall and precision", "Same out-of-fold predictions as the previous slide", F7 / "fa4_per_class.png", 9.2, [
        f"Fibrillar: recall is high but precision is low (CP on B1: recall {pcs('CP', 'B1', 'fibrillar', 'recall'):.2f}, precision {pcs('CP', 'B1', 'fibrillar', 'precision'):.2f}); the classifier over-calls fibrillar, mostly on focal adhesion patches",
        f"The hardest classes differ by test set: nascent on B1 (CP F1 {pcs('CP', 'B1', 'nascent', 'f1'):.2f}) and focal complex on B2 ({pcs('CP', 'B2', 'focal complex', 'f1'):.2f}); focal complex is confused with nascent and focal adhesion",
        f"Focal adhesion is the most reliable (CP precision {pcs('CP', 'B2', 'focal adhesion', 'precision'):.2f} on B2); the AE has a lower F1 than CP in 7 of the 8 class / test combinations (B2 nascent is the exception)"], size=11)

    # 5 confusion matrices
    for rep, nm in (("CP", "CP"), ("IL", "ilastik"), ("AE", "SupCon-AE")):
        fig_slide(prs, f"Confusion matrices: {nm}", "Out-of-fold, frame-grouped 5-fold CV (fold assignment seed 0); rows = true class, row-normalised with counts", CV / f"cv_cm_{rep}.png", 10.0, [
            "Top row tests on B1, bottom row on B2; columns are the training label set (B1, B2, B1+B2). Errors sit next to the diagonal: nascent / focal complex and focal adhesion / fibrillar",
            "Balanced accuracy is highest when training on B1+B2 (CP: 0.58 on the B1 test, 0.66 on the B2 test); the B1 test is the more representative one"] if rep == "CP" else [
            "Same layout as the CP slide: top row tests on B1, bottom row on B2; columns are the training label set"], size=10)

    # 6 features
    fa = FIN.loc["fa4", "full_order"].split(", ")[:6]; ad = FIN.loc["adnoad", "full_order"].split(", ")[:6]
    fig_slide(prs, "4. Binary vs subtype: how many features are needed", "Forward selection nested inside each CV fold; black line = all 50 CP features, red dot = one per correlated cluster",
              OD / "fsel_curve.png", 12.0, [
        "Adhesion vs no adhesion keeps improving to about 6 features (0.916 vs 0.923 for all 50); FA4 is flat from about 3 to 6 features (about 0.59 on B1, 0.62-0.64 on B2)",
        "The two tasks pick different features, so one shared compact set is a compromise"], size=11)

    s = _blank(prs); _slide_header(s, "Useful features for each task", "First six forward-selected CP features on all labelled data; (x/5) = folds that picked the feature among their first 10")
    def fq(task, f):
        d = FQ[(FQ.task == task) & (FQ.feature == f)]; return f"{f} ({int(d.folds_in_top10.iloc[0]) if len(d) else 0}/5)"
    rows = [["Pick", "Adhesion vs no adhesion", "Four FA subclasses"]] + [[str(i + 1), fq("adnoad", ad[i]), fq("fa4", fa[i])] for i in range(6)]
    table(s, rows, Inches(0.5), Inches(1.3), Inches(12.3), [0.8, 5.5, 5.5], row_h=0.42, size=13)
    _bullet_box(s, Inches(0.5), Inches(4.6), Inches(12.3), Inches(2.8), "Interpretation", [
        "The binary task leans on pixel-pair correlation (d1, d4) and brightness relative to the patch range (sum_average_d1); the subtype task leans on spread and low-intensity level (sum_variance_d2, intensity_p25) plus correlation_d2",
        "Only info_meas1_d4 appears in both first-four lists, but both tasks use correlation texture, at different distances",
        "Several features are near-duplicates (correlation_d1/d4, d2), so the exact distance picked is partly arbitrary"], bullet_size=12)

    fig_slide(prs, "Are the selected features stable across splits?", "Rank at which each feature was picked in each of the 5 CV folds", F7 / "feature_stability.png", 12.3, [
        "FA4: sum_variance_d2 and intensity_p25 are picks 1 and 2 in all five folds; the rest of the list changes from fold to fold, and fold 0 differs most",
        "Binary: correlation_d1, correlation_d4, sum_average_d1 and info_meas1_d4 are consistent in folds 1-4; fold 0 picks a different order (sum_average_d1 at rank 21)",
        "Stability is good for the top two FA4 features and for the correlation family; beyond that, treat the lists as indicative"], size=11)

    fig_slide(prs, "Class-specific distributions of the top features", "z-scored values on all B1+B2 labelled patches; boxes = quartiles, whiskers = 1.5 IQR", F7 / "feature_distributions.png", 12.0, [
        "Binary row: correlation features are low for no adhesion and rise for the FA classes; sum_average_d1 is high for no adhesion, lowest for nascent and focal complex, so it works together with the others, not alone",
        "FA4 row: sum_variance_d2 separates focal adhesion (high); intensity_p25 separates fibrillar and no adhesion (high) from nascent, complex and focal adhesion; correlation_d2 orders the classes with fibrillar below focal adhesion"], size=11)

    # notes
    s = _blank(prs); _slide_header(s, "Scope and caveats", "")
    _bullet_box(s, Inches(0.5), Inches(1.3), Inches(6.1), Inches(5.6), "Notes on the numbers", [
        "All results are frame-grouped (one DS1 frame = one cell), so no cell is in both training and test",
        "Hand-crafted rows pool 5 fold assignments; the SupCon-AE row has one",
        "The binary and FA4 feature selections use different training samples (all labels vs FA4 labels only), so they are matched in features, not in samples",
        "Fibrillar (38 labels) drives most of the uncertainty"], bullet_size=12)
    _bullet_box(s, Inches(6.8), Inches(1.3), Inches(6.0), Inches(5.6), "Not included here", [
        "The Figure 2 binary texture reference: not in this project's files, so it is not compared",
        "Subtype label / image-efficiency curves: the existing image-efficiency runs use matched B1 / B2v2 label subsets, and I have not confirmed they are the FA4 task, so they are not plotted",
        "Linear classifiers only; a tree model might select different features",
        "Per-class feature importance on ilastik or the AE latents"], bullet_size=12)
    prs.save(OUT); print("saved", OUT, len(prs.slides._sldIdLst), "slides")


if __name__ == "__main__":
    main()
