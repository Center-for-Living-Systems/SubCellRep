#!/usr/bin/env python3
"""
make_pptx_cp_feature_set.py

Deck: smallest CP feature subset that matches the full 50 features, for the FA4 classifier and the ad vs no-ad gate.
Inputs : results/ordinal_cp/fsel_*.csv, fsel_curve.png   (select_cp_feature_set.py + select_cp_feature_set_post.py)
Output : results/cp_feature_set.pptx
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_pptx_le_benchmark import (_prs, _blank, _txt, _slide_header, _bullet_box, C_HEAD, C_WHITE, SW, SH, RGBColor, Inches)
from make_pptx_ordinal_cp import table, fig_slide

REPO = Path(__file__).resolve().parents[1]
OD = REPO / "results" / "ordinal_cp"
OUT = REPO / "results" / "cp_feature_set.pptx"
C = pd.read_csv(OD / "fsel_curve.csv")
F = pd.read_csv(OD / "fsel_final_sets.csv").set_index("task")
FQ = pd.read_csv(OD / "fsel_pick_freq.csv")


def v(task, test, method, k=None, col="bal_acc"):
    d = C[(C.task == task) & (C.test == test) & (C.method == method)]
    if k is not None:
        d = d[d.k == k]
    return d.iloc[0][col]


def cell(task, test, method, k=None):
    return f"{v(task, test, method, k):.3f} ± {v(task, test, method, k, 'se'):.3f}"


def main():
    prs = _prs()
    s = _blank(prs)
    bg = s.shapes.add_shape(1, 0, 0, SW, SH); bg.fill.solid(); bg.fill.fore_color.rgb = C_HEAD; bg.line.fill.background()
    _txt(s, "How few CP features are enough?", Inches(0.8), Inches(2.3), Inches(11.7), Inches(1.0), bold=True, size_pt=38, color=C_WHITE)
    _txt(s, "A compact subset of the 50 intensity + Haralick features that performs like the full set, for the FA4 classifier and the adhesion vs no-adhesion gate",
         Inches(0.8), Inches(3.5), Inches(11.5), Inches(1.0), size_pt=17, color=RGBColor(0xCA, 0xDC, 0xFC))
    _txt(s, "DS1 vinc · B1 + B2 labels · frame-grouped 5-fold CV, selection nested inside each fold", Inches(0.8), Inches(4.7), Inches(11.5), Inches(0.5), size_pt=14, color=RGBColor(0xCA, 0xDC, 0xFC))

    s = _blank(prs); _slide_header(s, "Question and protocol", "Select on training images only, score on images never seen")
    _bullet_box(s, Inches(0.5), Inches(1.3), Inches(6.1), Inches(5.6), "Question", [
        "Which smallest set of CP features gives the same accuracy as all 50?",
        "Two tasks: FA4 (nascent / focal complex / focal adhesion / fibrillar) and adhesion vs no adhesion",
        "The 50 columns are heavily redundant (e.g. sum_average at d1/d2/d4 correlate 0.9995), so a small set should be enough"], bullet_size=13)
    _bullet_box(s, Inches(6.8), Inches(1.3), Inches(6.0), Inches(5.6), "Protocol", [
        "Frame-grouped 5-fold CV; every number is on held-out images",
        "Greedy forward selection inside each fold, using only that fold's training frames (inner 3-fold grouped CV, log-loss criterion)",
        "Classifier: standardised, class-balanced logistic regression, the same as the ad/no-ad gate",
        "FA4: train on B1+B2, test on B1 (representative) and on B2 separately. Ad/no-ad: combined labels",
        "Reference points: all 50 features, one-per-correlated-cluster (earlier list), random k features",
        "Final set: forward selection on all labelled data; ± values are the standard error over the 5 folds"], bullet_size=12)

    fig_slide(prs, "Accuracy against number of features", "Green = forward selection (nested), grey dashed = random k features, black = all 50, red = one-per-cluster",
              OD / "fsel_curve.png", 12.3, [
        "Forward selection climbs fast and sits above random picks at small k, then merges with the full-set line",
        "FA4 (B1 and B2): flat from about k = 3 to 6; extra features add nothing measurable",
        "Ad/no-ad: keeps improving slowly up to about k = 6 and is close to the full set by k ≈ 6–10"], size=11)

    s = _blank(prs); _slide_header(s, "Numbers: balanced accuracy on held-out images", "mean ± standard error over 5 folds (fold-weighted by test size)")
    rows = [["Set", "FA4, test B1", "FA4, test B2", "Ad vs no-ad"]]
    for lab, m, k in [("All 50 features", "full50", None), ("One per cluster (8/8/10)", "one_per_cluster", None)]:
        rows.append([lab, cell("fa4", "B1", m), cell("fa4", "B2", m), cell("adnoad", "all", m)])
    for k in (2, 3, 4, 6, 10):
        rows.append([f"Forward selection, k = {k}", cell("fa4", "B1", "forward", k), cell("fa4", "B2", "forward", k), cell("adnoad", "all", "forward", k)])
    rows.append(["Random, k = 6 (mean)", f"{v('fa4','B1','random',6):.3f}", f"{v('fa4','B2','random',6):.3f}", f"{v('adnoad','all','random',6):.3f}"])
    table(s, rows, Inches(0.5), Inches(1.3), Inches(12.3), [3.6, 2.9, 2.9, 2.9], row_h=0.42, size=13)
    _bullet_box(s, Inches(0.5), Inches(5.3), Inches(12.3), Inches(1.9), "Interpretation", [
        "FA4: the full 50-feature model is no better than 3–10 well-chosen features (0.56 vs about 0.59 on B1); 50 collinear columns on ~1.4k labels slightly overfit",
        "Ad/no-ad: 6 features reach 0.916 against 0.923 for all 50; the one-per-cluster set (10 features) is worse, 0.889",
        "Fold-to-fold spread on FA4 is large (± 0.04–0.05 for the full set), so differences of a few hundredths between sets are not resolved"], bullet_size=12)

    s = _blank(prs); _slide_header(s, "Recommended sets", "Forward selection on all labelled data; stability = how many of the 5 folds picked the feature among their first 10")
    def stab(task, feats):
        d = FQ[FQ.task == task].set_index("feature").folds_in_top10
        return [f"{f} ({int(d.get(f, 0))}/5)" for f in feats]
    fa = F.loc["fa4", "full_order"].split(", ")[:6]; ad = F.loc["adnoad", "full_order"].split(", ")[:6]
    _bullet_box(s, Inches(0.5), Inches(1.3), Inches(6.1), Inches(4.0), "FA4 (first 6, in selection order)",
                stab("fa4", fa) + ["Rule 'within 1 SE of the full set' stops at k = 2, but the SE is wide; k ≈ 4–6 is the safer choice"], bullet_size=12)
    _bullet_box(s, Inches(6.8), Inches(1.3), Inches(6.0), Inches(4.0), f"Ad vs no-ad (k = {int(F.loc['adnoad','k'])})",
                stab("adnoad", ad) + ["Texture-correlation features lead; all six were picked in 4–5 of the 5 folds"], bullet_size=12)
    _bullet_box(s, Inches(0.5), Inches(5.35), Inches(12.3), Inches(1.9), "Reading the lists", [
        "FA4 uses relative brightness and spread (sum_variance_d2, intensity_p25, intensity_p90) plus correlation / information-measure texture (correlation_d2, info_meas1_d4)",
        "The ad/no-ad gate leans on pixel-pair correlation (d1, d4) and brightness (sum_average_d1, variance_d1); these overlap little with the FA4 set",
        "Several picks are near-duplicates (correlation_d1/d4, info_meas1_d2/d4), so a swap within a correlated family changes accuracy very little"], bullet_size=12)

    s = _blank(prs); _slide_header(s, "Conclusions and caveats", "")
    _bullet_box(s, Inches(0.5), Inches(1.3), Inches(6.1), Inches(5.6), "Conclusions", [
        "FA4: about 4–6 features perform as well as all 50 (about 0.59 B1, 0.62–0.64 B2)",
        "Ad/no-ad: about 6 features are within 0.01 of all 50 (0.916 vs 0.923)",
        "Selection by performance beats both random picks and the one-per-cluster list at the same size",
        "The two tasks need different features, so one shared compact set is a compromise"], bullet_size=13)
    _bullet_box(s, Inches(6.8), Inches(1.3), Inches(6.0), Inches(5.6), "Caveats", [
        "One fold assignment (seed 0); FA4 fold noise is large, with only 38 fibrillar labels in B1+B2",
        "The final set was chosen on all labels, so its own accuracy is estimated by the nested curves, not by a separate test",
        "Linear classifier only; forward selection may differ with a tree model",
        "Balanced accuracy on four classes is dominated by the smallest ones",
        "Not tested: ilastik features, or whether the compact set transfers to other datasets (ds2–ds4)"], bullet_size=12)
    prs.save(OUT); print("saved", OUT, len(prs.slides._sldIdLst), "slides")


if __name__ == "__main__":
    main()
