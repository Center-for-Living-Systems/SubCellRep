#!/usr/bin/env python3
"""
make_pptx_ordinal_cp.py

Focused deck: ordinal / regression view of the 4 FA classes with CellProfiler (CP) features.
Inputs : results/ordinal_cp/*, results/stage2_rebuild/embed_cp_stage2.png, results/cv_b1b2/cv_cm_CP.png,
         results/stage2_rebuild/axis_order_spearman.csv
Output : results/ordinal_cp.pptx
"""
from __future__ import annotations
import io, json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_pptx_le_benchmark import (_prs, _blank, _txt, _slide_header, _img, _bullet_box,
                                    C_HEAD, C_BODY, C_GREY, C_WHITE, C_ACCENT, C_ROW_A, C_ROW_B, SW, SH, PP_ALIGN, RGBColor, Inches, Pt)

REPO = Path(__file__).resolve().parents[1]
OD, RS, CV = REPO / "results" / "ordinal_cp", REPO / "results" / "stage2_rebuild", REPO / "results" / "cv_b1b2"
OUT = REPO / "results" / "ordinal_cp.pptx"
S = json.load(open(OD / "summary.json"))
M = pd.read_csv(OD / "metrics.csv")
SP = pd.read_csv(RS / "axis_order_spearman.csv")


def m(src, test):
    return M[(M.src == src) & (M.test == test)].iloc[0]


def table(slide, rows, left, top, width, col_w, row_h=0.36, size=12):
    nr, nc = len(rows), len(rows[0])
    tb = slide.shapes.add_table(nr, nc, left, top, width, Inches(row_h * nr)).table
    for j, w in enumerate(col_w):
        tb.columns[j].width = int(width * w / sum(col_w))
    for i, r in enumerate(rows):
        tb.rows[i].height = Inches(row_h)
        for j, v in enumerate(r):
            c = tb.cell(i, j); c.text = str(v)
            c.margin_left = c.margin_right = Inches(0.06); c.margin_top = c.margin_bottom = Inches(0.02)
            p = c.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            f = p.runs[0].font; f.size = Pt(size); c.fill.solid()
            if i == 0:
                c.fill.fore_color.rgb = C_ACCENT; f.color.rgb = C_WHITE; f.bold = True
            else:
                c.fill.fore_color.rgb = C_ROW_A if i % 2 else C_ROW_B; f.color.rgb = C_BODY; f.bold = (j == 0)


def fig_slide(prs, title, sub, img, width, bullets, top=1.15, box_h=None, size=12):
    s = _blank(prs); _slide_header(s, title, sub)
    p = Path(img); from PIL import Image
    w, h = Image.open(p).size; fh = width * h / w
    _img(s, p, Inches((13.33 - width) / 2), Inches(top), width=Inches(width))
    y = top + fh + 0.12
    if bullets:
        _bullet_box(s, Inches(0.5), Inches(y), Inches(12.3), Inches(min(box_h or (7.4 - y), 7.4 - y)), "Interpretation", bullets, bullet_size=size)
    return s


def title_slide(prs):
    s = _blank(prs)
    bg = s.shapes.add_shape(1, 0, 0, SW, SH); bg.fill.solid(); bg.fill.fore_color.rgb = C_HEAD; bg.line.fill.background()
    _txt(s, "Ordinal / regression view of FA subtypes", Inches(0.8), Inches(2.3), Inches(11.7), Inches(1.0), bold=True, size_pt=38, color=C_WHITE)
    _txt(s, "CellProfiler features as the worked example: ordinal axis vs PC1, regression quality, and whether confused cases are transitions between neighbouring classes",
         Inches(0.8), Inches(3.5), Inches(11.5), Inches(1.0), size_pt=17, color=RGBColor(0xCA, 0xDC, 0xFC))
    _txt(s, "DS1 vinc · B1 (random) and B2 (prototype) labels · frame-grouped 5-fold CV", Inches(0.8), Inches(4.7), Inches(11.5), Inches(0.5), size_pt=14, color=RGBColor(0xCA, 0xDC, 0xFC))


def main():
    prs = _prs(); title_slide(prs)
    a = m("B1+B2", "B1")

    # 2 question + setup
    s = _blank(prs); _slide_header(s, "Question and setup", "Can the four FA types be treated as an ordered scale, and what does that tell us about the confused cases?")
    _bullet_box(s, Inches(0.5), Inches(1.3), Inches(6.1), Inches(5.6), "Question", [
        "Instead of a 4-way classifier, predict a continuous maturity score (regression)",
        "Does the ordered view work, i.e. how well does a score reproduce nascent < focal complex < focal adhesion < fibrillar?",
        "Are the confused cases transitions between neighbouring classes, and are they flagged by lower probability?"], bullet_size=13)
    _bullet_box(s, Inches(6.8), Inches(1.3), Inches(6.0), Inches(5.6), "Setup", [
        "Features: 50 CellProfiler features (intensity + Haralick texture), standardised",
        "Rank encoding: nascent 0, focal complex 1, focal adhesion 2, fibrillar 3 (an assumption)",
        "Frame-grouped 5-fold CV: test images never seen in training; pooled out-of-fold predictions",
        f"Main test: B1 patches (representative; n={int(a.n)}: {int(a.n_nascent)} nascent, {int(a['n_f.complex'])} complex, {int(a['n_f.adhesion'])} adhesion, {int(a.n_fibrillar)} fibrillar), probe trained on B1+B2 of the other folds",
        "Models: ridge regression on the rank (alpha 10) and balanced logistic regression (soft rank = Σ p_k·k)"], bullet_size=12)

    # 3 how the panel is built
    s = _blank(prs); _slide_header(s, "How the ordinal axis vs PC1 panel is built", "One supervised axis against the data's own dominant direction")
    _bullet_box(s, Inches(0.5), Inches(1.3), Inches(6.1), Inches(5.6), "y: ordinal score", [
        "Ridge regression of the class rank on standardised features, fit on the training split only",
        "Applied to every patch → one linear score per patch",
        "A perfect ordered ladder would give scores 0, 1, 2, 3 for the four classes",
        "Out-of-fold version (later slides) is the honest regression performance"], bullet_size=13)
    _bullet_box(s, Inches(6.8), Inches(1.3), Inches(6.0), Inches(5.6), "x: PC1", [
        "First principal component of the same features, no labels used",
        f"Captures {S['pc1_var']:.0%} of CP variance (57% within the stage-2 pool)",
        "Its sign is arbitrary",
        "A diagonal band means the main variation in CP features runs along the maturity direction; no trend means PC1 is something else (e.g. brightness)"], bullet_size=13)

    fig_slide(prs, "CP: ordinal axis vs PC1", f"Spearman(ordinal score, PC1) = {S['spearman_ordinal_vs_pc1_all_labelled']:.2f} on labelled patches (train-fit axis; black edge = validation split)",
              OD / "ordinal_vs_pc1.png", 6.4, [], top=1.15)
    # place bullets to the right of that figure
    s = prs.slides[-1]
    _bullet_box(s, Inches(7.4), Inches(1.25), Inches(5.5), Inches(5.6), "Interpretation", [
        "Clear diagonal: higher PC1 goes with higher ordinal score; nascent (blue) low, focal adhesion (green) high",
        "Nascent and focal complex overlap throughout the lower-left; fibrillar (red) sits with focal adhesion at the top",
        "Validation points (black edge) follow the same band, so the axis is not just fitting the training points",
        f"Yet the two directions are almost orthogonal in feature space (cosine {S['cos_pc1_ridge']:.2f}): the correlation arises because the features are collinear (next slide)"], bullet_size=12)

    fig_slide(prs, "What drives PC1 and the ordinal axis in CP", "Top loadings / coefficients (display fit on all B1+B2 labelled patches)",
              OD / "features.png", 10.0, [
        "PC1 is intensity (min, 10th/25th percentile, median) plus local contrast / homogeneity: a brightness-and-contrast axis",
        "The ordinal axis is dominated by Haralick texture: correlation_d2 (+0.49), difference_variance_d4 (+0.30), contrast_d1 (+0.25)",
        "Correlated features take opposite signs across offsets (d1 vs d2), so individual coefficients are not interpretable on their own",
        "So 'maturity' in CP is mostly a texture signal that happens to co-vary with the main brightness/contrast axis"], size=11)

    # 6 CP embedding context
    sp = SP[(SP.rep == "CP features") & (SP.run == "stage2")].iloc[0]
    fig_slide(prs, "CP: alternative embeddings (stage-2 pool, B1+B2 labels)",
              "PCA · LDA · ordinal vs PC1 · UMAP · PHATE · diffusion map · Isomap; black edge = validation patch", RS / "embed_cp_stage2.png", 9.6, [
        f"|Spearman| with class order on validation patches: PC1 {sp.PC1:.2f}, LD1 {sp.LD1:.2f}, ordinal {sp.ordinal:.2f}, UMAP1 {sp.UMAP1:.2f}, PHATE1 {sp.PHATE1:.2f}, DC1 {sp.DC1:.2f}, Isomap1 {sp.Iso1:.2f}",
        "PHATE and the diffusion map show a continuous arc (nascent/complex mixed at one end, focal adhesion at the other); the supervised axes order the classes best"],
        size=11, box_h=1.7)

    # 7 regression performance
    s = _blank(prs); _slide_header(s, "Regression performance (out-of-fold, B1 test)", "Probe trained on B1+B2 of the other images; baseline = always predict the median class")
    rows = [["Predictor", "Spearman", "MAE (rank units)", "Within-one-class acc.", "Exact acc."],
            ["ridge regression on rank", f"{a.ridge_spearman:.3f}", f"{a.ridge_mae:.3f}", "–", "–"],
            ["soft rank (logistic regression)", f"{a.soft_spearman:.3f}", f"{a.soft_mae:.3f}", "–", "–"],
            ["logistic regression, argmax class", "–", "–", f"{a.within1:.3f}", f"{a.acc:.3f}"],
            ["baseline: median class", "0", f"{a.median_mae:.3f}", "–", "–"]]
    table(s, rows, Inches(0.5), Inches(1.3), Inches(12.3), [3.4, 1.6, 2.0, 2.4, 1.6], row_h=0.45, size=13)
    _bullet_box(s, Inches(0.5), Inches(3.8), Inches(12.3), Inches(3.2), "Interpretation", [
        f"Rank correlation about {a.ridge_spearman:.2f}–{a.soft_spearman:.2f} and MAE {a.ridge_mae:.2f}–{a.soft_mae:.2f} vs 0.60 for the median baseline: a real but moderate regression signal",
        f"Predictions are within one class of the truth {a.within1:.0%} of the time, although only {a.acc:.0%} are exactly right",
        "Two regressors (ridge, soft rank) agree, so the signal is not an artefact of one model",
        "Results are for one fold assignment (seed 0); the B1 test set has 609 patches, so differences of a few hundredths are within noise"], bullet_size=13)

    fig_slide(prs, "Where each class falls on the score (B1 test, out-of-fold)", "Dashed line = perfect score; left: ridge on rank, right: soft rank from logistic probabilities",
              OD / "score_by_class.png", 10.0, [
        f"Ridge class means: nascent {S['class_mean_score']['nascent']:.2f}, complex {S['class_mean_score']['f.complex']:.2f}, adhesion {S['class_mean_score']['f.adhesion']:.2f}, fibrillar {S['class_mean_score']['fibrillar']:.2f}; spread ~0.35–0.46 within each class",
        "Two groups, not four steps: nascent ≈ complex (gap 0.18, far below the spread) and adhesion, with a clear jump between them (about 0.75)",
        "The linear ridge score puts fibrillar on top of focal adhesion; the soft rank (mean 2.50 vs 1.97) pulls fibrillar beyond it, but n = 16 and one patch sits near 0.5"], size=11, box_h=1.7)

    fig_slide(prs, "Confusion matrices (CP, frame-grouped CV)", "Rows of panels: test on B1 / B2; columns: probe trained on B1, B2, B1+B2", CV / "cv_cm_CP.png", 9.4, [
        "On B1 (top row), nascent ↔ focal complex dominates (nascent→complex 0.50, complex→nascent 0.23); complex → adhesion is next; fibrillar mixes with focal adhesion",
        "Cross-annotator (train B1 → test B2): 59% of B2's focal complex is called nascent: the two annotators place the nascent/complex boundary differently"], size=11, box_h=1.7)

    fig_slide(prs, "Structure of the errors", f"B1 test, probe trained on B1+B2: {S['n_errors']} misclassified patches of {int(a.n)}", OD / "error_structure.png", 10.6, [
        f"Exact accuracy {a.acc:.2f}, but {a.within1:.2f} within one class and the true class is in the top-2 probabilities for {a.top2:.2f}",
        f"{a.err_adjacent:.0%} of errors go to an adjacent class, against {a.err_adjacent_chance:.0%} expected if errors were random: errors are neighbour-to-neighbour",
        f"The misclassified patches put {a.mass_true_pred_wrong:.0%} of their probability mass on just the true and the predicted class: two-class ambiguity, not diffuse uncertainty"], size=12, box_h=1.9)

    fig_slide(prs, "Are the errors transitions along the score?", f"Misclassification rate and class mix by octile of the out-of-fold ridge score", OD / "boundary.png", 10.0, [
        f"Error rate is roughly flat along the score (0.32–0.51), not concentrated at one boundary: the low-score end errs on nascent vs complex, the high-score end on adhesion vs neighbours",
        f"Of {S['n_adjacent_errors']} adjacent-class errors, {S['adj_errors_closer_to_pred_mean']:.0%} have a score closer to the predicted class's mean than the true class's mean, but {S['adj_errors_between_means']:.0%} lie between the two class means; this check is partly circular (same features)",
        "Composition shows the transition zone: scores 0.9–1.4 are a mix of nascent, complex and some adhesion; above 1.6 mostly adhesion"], size=11, box_h=2.0)

    fig_slide(prs, "Do confused cases have lower probability?", f"Max class probability, B1 test: correct {a.conf_correct:.2f} vs misclassified {a.conf_wrong:.2f}", OD / "confidence.png", 10.0, [
        f"Only weakly: AUROC for detecting an error from max probability is {a.auc_maxprob:.2f} (margin {a.auc_margin:.2f}); chance is 0.50",
        f"Over-confident: even the most confident bin (p ≥ 0.75, n = {S['error_rate_by_conf_bin']['0.75-1.00'][1]}) is wrong {S['error_rate_by_conf_bin']['0.75-1.00'][0]:.0%} of the time; the least confident bin {S['error_rate_by_conf_bin']['0.30-0.45'][0]:.0%}",
        f"Within-annotator on B2 the probability is far more informative (AUROC {m('B1+B2', 'B2').auc_maxprob:.2f}), so on B1 the ambiguity is largely in the labels/features, not in the confidence score"], size=11, box_h=2.0)

    # 13 label source
    s = fig_slide(prs, "Regression by training label source", "CP ridge regression on rank; tested on B1 (representative) or B2 held-out images", OD / "sources.png", 10.2, [], top=1.15)
    rows = [["train → test", "Spearman", "MAE", "Within-one acc.", "median-baseline MAE"]]
    for src in ("B1", "B2", "B1+B2"):
        for test in ("B1", "B2"):
            r = m(src, test); rows.append([f"{src} → {test}", f"{r.ridge_spearman:.3f}", f"{r.ridge_mae:.3f}", f"{r.within1:.3f}", f"{r.median_mae:.3f}"])
    table(s, rows, Inches(0.5), Inches(4.25), Inches(7.3), [2, 1.4, 1.2, 1.8, 2.2], row_h=0.38, size=11)
    _bullet_box(s, Inches(8.0), Inches(4.25), Inches(4.9), Inches(3.05), "Interpretation", [
        "On B1 test, B1-trained is as good as or better than B1+B2 (0.70 vs 0.66); B2-trained is worst (0.61, MAE 0.54)",
        "B1 → B2 gives MAE 0.64, barely below the 0.66 median baseline: B1's ordering does not transfer to B2",
        "B2 → B2 is best (0.76) = annotator-style consistency"], bullet_size=10)

    # 14 conclusions
    s = _blank(prs); _slide_header(s, "Conclusions and what to do next", "CP features, representative B1 test set")
    _bullet_box(s, Inches(0.5), Inches(1.3), Inches(6.1), Inches(5.6), "What we can say", [
        f"The ordered view works moderately: Spearman ~{a.ridge_spearman:.2f}, MAE {a.ridge_mae:.2f} (baseline 0.60), within-one-class accuracy {a.within1:.2f}",
        "The ordinal axis is mostly a coarse split: {nascent, complex} vs {adhesion, fibrillar}; nascent and complex are not resolved",
        "Confused cases are mostly neighbour-to-neighbour (89% adjacent vs 60% chance) with probability mass on two classes",
        "But low probability does not flag them (AUROC 0.57) and the classifier is over-confident on B1",
        "Fibrillar (16 test patches) is only separated by the nonlinear soft rank, not by the linear ridge"], bullet_size=12)
    _bullet_box(s, Inches(6.8), Inches(1.3), Inches(6.0), Inches(5.6), "Recommendations", [
        "Use a regression/ordinal formulation for the coarse maturity gradient; report MAE, Spearman and within-one accuracy, not only exact accuracy",
        "Do not expect it to separate nascent from focal complex with CP; consider merging them or adding features",
        "Handle fibrillar separately (too few labels for an ordinal position)",
        "Calibrate probabilities (Platt/isotonic on held-out folds) before using confidence to flag transitions",
        "Estimate the label-noise ceiling from the 66 patches both annotators labelled (68% agreement) before judging adjacent errors; train on B1 or B1+B2, not B2 alone",
        "Caveat: single fold assignment, ranks assumed, collinear features limit interpretation"], bullet_size=11)

    prs.save(OUT); print("saved", OUT, len(prs.slides), "slides")


if __name__ == "__main__":
    main()
