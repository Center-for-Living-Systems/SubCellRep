#!/usr/bin/env python3
"""
make_pptx_stage2_rebuild.py

Deck for the stage-2 rebuild (ad/no-ad gate -> stage-2 pool -> SupCon-AE with FA4 labels).
Inputs : results/stage2_rebuild/{umap_*.png, cm_*.png, val_metrics.csv},
         labelling/stage2_rebuild/{adnoad_cv.json, confidence_threshold_table.csv}
Output : results/stage2_rebuild.pptx

Usage:  python scripts/make_pptx_stage2_rebuild.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_pptx_le_benchmark import (_prs, _blank, _txt, _slide_header, _add_fig, _img, _bullet_box,
                                    C_HEAD, C_BODY, C_GREY, C_WHITE, C_PANEL, C_ACCENT, C_ROW_A, C_ROW_B,
                                    SW, SH, PP_ALIGN, RGBColor, Inches, Pt)

REPO = Path(__file__).resolve().parents[1]
RES  = REPO / "results" / "stage2_rebuild"
PREP = Path("/net/projects/CLS/lding/data/fa_data_analysis/labelling/stage2_rebuild")
OUT  = REPO / "results" / "stage2_rebuild.pptx"

CV   = json.load(open(PREP / "adnoad_cv.json"))
THR  = pd.read_csv(PREP / "confidence_threshold_table.csv")
VAL  = pd.read_csv(RES / "val_metrics.csv").set_index("run")
VALS = {"SupCon-AE latents": VAL,
        "CP features": pd.read_csv(RES / "val_metrics_cp.csv").set_index("run"),
        "ILASTIK features": pd.read_csv(RES / "val_metrics_ilastik.csv").set_index("run")}
SPEAR = pd.read_csv(RES / "axis_order_spearman.csv")
CVD  = REPO / "results" / "cv_b1b2"
CVS  = pd.read_csv(CVD / "summary.csv")
ARM  = {"B1": "b1", "B2": "b2", "B1+B2": "b12"}
REP_TAG = {"SupCon-AE latents": "", "CP features": "cp_", "ILASTIK features": "ilastik_"}
REP_COL = {"SupCon-AE latents": "#0f3d79", "CP features": "#d65f0e", "ILASTIK features": "#2e8b57"}

RUNS = [  # key, title, pool, labels
    ("stage2",           "Stage-2 pool (p≥0.5) · B1+B2 FA4 labels",      "12,571", "B1+B2, 4 FA classes"),
    ("stage2_conf90",    "Stage-2 pool (p≥0.9) · B1+B2 FA4 labels",      "8,260",  "B1+B2, 4 FA classes"),
    ("stage2_b1",        "Stage-2 pool (p≥0.5) · B1 FA4 labels only",    "12,571", "B1 only, 4 FA classes"),
    ("stage2_b1_conf90", "Stage-2 pool (p≥0.9) · B1 FA4 labels only",    "8,260",  "B1 only, 4 FA classes"),
    ("stage2_b2",        "Stage-2 pool (p≥0.5) · B2 FA4 labels only",    "12,571", "B2 only, 4 FA classes"),
    ("stage2_b2_conf90", "Stage-2 pool (p≥0.9) · B2 FA4 labels only",    "8,260",  "B2 only, 4 FA classes"),
    ("whole",            "Whole set (27,637) · No-adhesion + FA4 labels", "27,637", "B1+B2, 5 classes"),
]


def add_fig_w(slide, fig, left, top, width):
    """Add a matplotlib figure at the given width, keeping its aspect ratio."""
    import io
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=150, facecolor="white"); plt.close(fig); buf.seek(0)
    slide.shapes.add_picture(buf, left, top, width=width)


def table(slide, rows, left, top, width, col_w, row_h=0.32, size=11, bold_first_col=False):
    nr, nc = len(rows), len(rows[0])
    tb = slide.shapes.add_table(nr, nc, left, top, width, Inches(row_h * nr)).table
    tot = sum(col_w)
    for j, w in enumerate(col_w):
        tb.columns[j].width = int(width * w / tot)
    for i, r in enumerate(rows):
        tb.rows[i].height = Inches(row_h)
        for j, v in enumerate(r):
            c = tb.cell(i, j); c.text = str(v)
            c.margin_left = c.margin_right = Inches(0.06); c.margin_top = c.margin_bottom = Inches(0.02)
            p = c.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            f = p.runs[0].font; f.size = Pt(size)
            c.fill.solid()
            if i == 0:
                c.fill.fore_color.rgb = C_ACCENT; f.color.rgb = C_WHITE; f.bold = True
            else:
                c.fill.fore_color.rgb = C_ROW_A if i % 2 else C_ROW_B; f.color.rgb = C_BODY
                f.bold = bold_first_col and j == 0
    return tb


def title_slide(prs):
    s = _blank(prs)
    bg = s.shapes.add_shape(1, 0, 0, SW, SH); bg.fill.solid(); bg.fill.fore_color.rgb = C_HEAD; bg.line.fill.background()
    _txt(s, "Stage-2 rebuild: FA4 SupCon-AE", Inches(0.8), Inches(2.4), Inches(11.5), Inches(1.0), bold=True, size_pt=40, color=C_WHITE)
    _txt(s, "Ad/no-ad gate (CP + logreg) → stage-2 patch pool → SupCon-AE vs CP vs ilastik features, FA subtype labels",
         Inches(0.8), Inches(3.5), Inches(11.5), Inches(0.8), size_pt=18, color=RGBColor(0xCA, 0xDC, 0xFC))
    _txt(s, "DS1 vinc ctrl+ycomp · B1 (random) vs B2 (prototype) labels · AE vs CP vs ilastik · cross-annotator CV", Inches(0.8), Inches(4.5), Inches(11.5), Inches(0.5),
         size_pt=14, color=RGBColor(0xCA, 0xDC, 0xFC))


def design_slide(prs):
    s = _blank(prs); _slide_header(s, "Pipeline", "Rebuild of the gated two-stage setup (original gated runs were deleted from the cluster)")
    steps = [
        ("1 · Ad vs no-ad gate", [
            "CellProfiler features (50) on all 27,637 DS1 patches",
            "Logistic regression (standardised, balanced) on combined B1+B2 labels: 2,983 labels (B2 priority on overlap, Uncertain dropped)",
            f"Grouped 5-fold CV bal. acc {CV['cv_bal_acc_mean']:.3f} ± {CV['cv_bal_acc_std']:.3f}",
            f"Predicts adhesion for {CV['n_pred_adhesion']:,} patches"]),
        ("2 · Stage-2 pool", [
            f"Pool = predicted adhesion ∪ all FA4-labelled patches = {CV['stage2_pool']:,} (control 7,942 / ycomp 4,629)",
            f"{CV['n_fa4_labelled']:,} FA4-labelled patches; {CV['n_fa4_labelled_not_predicted_ad']} were gated out and added back",
            "High-confidence variant: p(adhesion) ≥ 0.9 → 8,260 patches"]),
        ("3 · SupCon-AE", [
            "Latent 32, projection 16, 1 ch, 32×32 patches, nl1 recon + SupCon (λ=0.5, τ=0.5)",
            "Batch 256 (2× usual), 300 epochs, lr 1e-3, 80/20 split grouped by image",
            "Contrastive loss only on labelled patches; all others reconstruct only",
            "Latents → UMAP; held-out check = logreg on latents (train-split fit, val-split score)"]),
    ]
    x = 0.5
    for h, b in steps:
        _bullet_box(s, Inches(x), Inches(1.3), Inches(4.0), Inches(5.4), h, b, bullet_size=12)
        x += 4.15


def gate_slide(prs):
    s = _blank(prs); _slide_header(s, "Ad/no-ad gate: choosing a confidence threshold",
                                   "Precision / recall from grouped-CV out-of-fold probabilities on B1+B2 labels")
    rows = [["p(adhesion) ≥", "OOF precision", "OOF recall", "Predicted-ad patches", "Stage-2 pool"]]
    for _, r in THR.iterrows():
        rows.append([f"{r.threshold:.2f}", f"{r.oof_precision_ad:.3f}", f"{r.oof_recall_ad:.3f}", f"{int(r.n_pred_ad):,}", f"{int(r.n_pool):,}"])
    table(s, rows, Inches(0.5), Inches(1.3), Inches(7.6), [1.3, 1.3, 1.2, 1.8, 1.4], row_h=0.42, size=12)
    fig, ax = plt.subplots(figsize=(4.6, 3.6), constrained_layout=True)
    ax.plot(THR.threshold, THR.oof_precision_ad, "o-", c="#1f77b4", label="precision")
    ax.plot(THR.threshold, THR.oof_recall_ad, "s-", c="#d62728", label="recall")
    ax.axvline(0.9, ls="--", c="grey", lw=1); ax.set_xlabel("threshold"); ax.set_ylim(0.4, 1.0); ax.legend(frameon=False); ax.grid(alpha=0.3)
    add_fig_w(s, fig, Inches(8.4), Inches(1.3), Inches(4.4))
    _txt(s, "Precision gains flatten while recall drops fast. Bimodal scores: median p = 0.34, 75th pct 0.94.\n"
            "0.9 was chosen as the high-confidence cut (precision 0.977, recall 0.728). Precision is likely optimistic: labelled patches are more typical than the unlabelled pool.",
         Inches(0.5), Inches(4.7), Inches(7.6), Inches(1.5), size_pt=13, color=C_BODY)


def key_slide(prs):
    s = _blank(prs); _slide_header(s, "How to read the figures", "Colour scheme (all figures) and the views used")
    items = [("No adhesion", "#8e44ad"), ("Nascent adhesion", "#1f77b4"), ("Focal complex", "#ff7f0e"),
             ("Focal adhesion", "#2ca02c"), ("Fibrillar adhesion", "#d62728"), ("Unlabelled", "#d9d9d9")]
    for i, (n, c) in enumerate(items):
        y = 1.35 + i * 0.5
        dot = s.shapes.add_shape(9, Inches(0.7), Inches(y), Inches(0.3), Inches(0.3)); dot.fill.solid()
        dot.fill.fore_color.rgb = RGBColor.from_string(c[1:].upper()); dot.line.fill.background()
        _txt(s, n, Inches(1.2), Inches(y - 0.03), Inches(3.5), Inches(0.4), size_pt=16)
    _bullet_box(s, Inches(5.2), Inches(1.3), Inches(7.6), Inches(5.5), "Views and representations", [
        "Representations: SupCon-AE latents (32-d) · CellProfiler features (50) · ilastik features (80), same patches and same train/val split in every run",
        "UMAP (default init) is shown for the AE latents only; every representation also gets PCA, LDA (train-fit), ordinal axis vs PC1, UMAP (init=pca, nn=150), PHATE, diffusion map, Isomap",
        "PHATE / diffusion map / Isomap use a subsample of 8,000 patches (all labelled + random unlabelled)",
        "Black edge = validation-split labelled patch (not used to fit LDA / ordinal / classifier)",
        "Confusion matrices: logistic regression on the representation, fit on train split, scored on val split, rows normalised"], bullet_size=13)


def embed_slide(prs, key, title, rep):
    s = _blank(prs); _slide_header(s, f"{rep}: {title}", "Alternative 2-D views looking for a continuous progression (stripe)")
    _img(s, RES / f"embed_{REP_TAG[rep]}{key}.png", Inches(0.5), Inches(1.15), width=Inches(12.3))


def cm_pair_slide(prs, key, title):
    s = _blank(prs); _slide_header(s, f"Confusion matrices, CP vs ilastik: {title}", "Logistic regression on hand-crafted features, same val split as the AE run")
    _img(s, RES / f"cm_cp_{key}.png", Inches(0.6), Inches(1.2), height=Inches(5.4))
    _img(s, RES / f"cm_ilastik_{key}.png", Inches(7.0), Inches(1.2), height=Inches(5.4))
    for rep, x in (("CP features", 0.6), ("ILASTIK features", 7.0)):
        v = VALS[rep].loc[key]
        _txt(s, f"val bal. acc {v.bal_acc_val:.3f} · train {v.bal_acc_train:.3f} · n_val {int(v.n_val)}", Inches(x), Inches(6.7), Inches(6), Inches(0.35), size_pt=12, color=C_GREY)


def compare_slide(prs):
    s = _blank(prs); _slide_header(s, "Representation comparison: held-out balanced accuracy",
                                   "Logreg on each representation, fit on train split, scored on val split (same patches, same split)")
    keys = [k for k, *_ in RUNS]
    rows = [["Run", "AE val", "CP val", "ilastik val", "AE train", "CP train", "ilastik train"]]
    for k in keys:
        rows.append([k] + [f"{VALS[r].loc[k].bal_acc_val:.3f}" for r in VALS] + [f"{VALS[r].loc[k].bal_acc_train:.3f}" for r in VALS])
    table(s, rows, Inches(0.4), Inches(1.25), Inches(6.6), [2.2, 1, 1, 1.1, 1.1, 1.1, 1.3], row_h=0.42, size=11, bold_first_col=True)
    fig, ax = plt.subplots(figsize=(6.0, 3.8), constrained_layout=True)
    x = np.arange(len(keys))
    for i, r in enumerate(VALS):
        ax.bar(x + (i - 1) * 0.27, [VALS[r].loc[k].bal_acc_val for k in keys], 0.26, color=REP_COL[r], label=r.split()[0])
    ax.set_xticks(x); ax.set_xticklabels(keys, rotation=25, ha="right", fontsize=7); ax.set_ylim(0, 1); ax.set_ylabel("val balanced acc.")
    ax.legend(frameon=False, ncol=3, fontsize=8); ax.grid(axis="y", alpha=0.3)
    add_fig_w(s, fig, Inches(7.2), Inches(1.25), Inches(5.7))
    _txt(s, "CP and ilastik generalise better than the AE latents (smaller train-val gap). p≥0.9 pools give identical CP/ilastik numbers by construction: the pool only changes the unlabelled patches, which a supervised classifier never sees.",
         Inches(0.5), Inches(5.0), Inches(6.5), Inches(1.6), size_pt=13, color=C_BODY)


def recall_slide(prs):
    s = _blank(prs); _slide_header(s, "Per-class recall on the validation split", "Recall (n val patches in class); B1+B2 and B2 runs use the p≥0.5 pool")
    cols = [("nasc", "Nascent"), ("fcomp", "Focal cplx"), ("fadh", "Focal adh."), ("fibr", "Fibrillar"), ("noad", "No-ad")]
    rows = [["Run", "Representation"] + [c[1] for c in cols]]
    for k in ["stage2", "stage2_b1", "stage2_b2", "whole"]:
        for r, v in VALS.items():
            row = v.loc[k]
            rows.append([k, r.split()[0]] + [("–" if pd.isna(row.get("rec_" + c)) else f"{row['rec_' + c]:.2f} ({int(row['nval_' + c])})") for c, _ in cols])
    table(s, rows, Inches(0.5), Inches(1.25), Inches(12.3), [1.6, 1.6, 1.4, 1.4, 1.4, 1.4, 1.3], row_h=0.36, size=11, bold_first_col=True)
    _txt(s, "Very small validation counts for some classes (e.g. 4 fibrillar in B1, 6 focal complex in B2): those recalls are noise. Focal complex vs nascent is the weak spot for every representation except B2-labelled ilastik/CP.",
         Inches(0.5), Inches(6.2), Inches(12.3), Inches(0.9), size_pt=12, color=C_GREY)


def stripe_slide(prs):
    s = _blank(prs); _slide_header(s, "Does a stripe / 1-D axis order the classes?",
                                   "|Spearman| between each axis and the class order nascent < focal complex < focal adhesion < fibrillar (val-split labelled, stage2 pool, B1+B2)")
    ax_cols = ["PC1", "LD1", "ordinal", "UMAP1", "PHATE1", "DC1", "Iso1"]
    d = SPEAR[SPEAR.run == "stage2"]
    rows = [["Representation"] + ax_cols]
    for _, r in d.iterrows():
        rows.append([r.rep] + [f"{r[c]:.2f}" for c in ax_cols])
    table(s, rows, Inches(0.5), Inches(1.3), Inches(12.3), [2.4, 1, 1, 1, 1, 1, 1, 1], row_h=0.45, size=12, bold_first_col=True)
    _bullet_box(s, Inches(0.5), Inches(3.4), Inches(12.3), Inches(3.4), "Reading", [
        "PHATE and the diffusion map turn all three representations into a continuous arc/stripe; ilastik gives the cleanest, thinnest arc. UMAP (init=pca) and Isomap stay blob-like; PCA and LDA are not stripe-like",
        "A clean arc does not mean the arc orders the classes: the supervised axes (LD1, ordinal) order them best (0.68–0.74), while ilastik's arc axes score lowest (DC1 0.51, PHATE1 0.57). The dominant arc likely encodes another variable (e.g. intensity or patch content — not verified)",
        "Even the best axis only reaches |ρ| ≈ 0.7 on 250 val patches, and nascent / focal complex overlap everywhere: the data support a coarse nascent → focal adhesion gradient, not four ordered stages"], bullet_size=13)



def cv_get(rep, src, test, col="bal_acc"):
    r = CVS[(CVS.rep == rep) & (CVS.src == src) & (CVS.test == test)]
    return None if r.empty else r.iloc[0]


def cv_rep_name(rep, src):
    return {"AE": f"AE_{ARM[src]}", "CP+IL+AE": f"CP+IL+AE_{ARM[src]}"}.get(rep, rep) if src in ARM else rep


def cv_design_slide(prs):
    s = _blank(prs); _slide_header(s, "B1 vs B2: cross-annotator cross-validation design",
                                   "B1 = randomised sample (representative of unseen images); B2 = prototype-picked. Goal: is B2 useful, and does the AE add anything?")
    _img(s, CVD / "label_sets.png", Inches(0.4), Inches(1.2), width=Inches(6.0))
    _txt(s, "B1 spans 89 images, B2 only 27 (26 of them also in B1); 66 patches labelled by both, agreeing 68%. B2 FA4 is 59% focal adhesion and 14% focal complex; B1 is 36% / 43%.",
         Inches(0.5), Inches(4.4), Inches(5.9), Inches(1.4), size_pt=12, color=C_BODY)
    _bullet_box(s, Inches(6.7), Inches(1.25), Inches(6.2), Inches(5.7), "Design", [
        "Frame-grouped 5-fold CV over all labelled images: no patch from a test image is used for training the encoder or the probe",
        "Test sets: B1 patches (representative) and B2 patches in the held-out images; pooled out-of-fold predictions, 95% CI by bootstrapping images",
        "Probe training sources: B1, B2, B2 size-matched to B1, B1+B2",
        "Representations: CP, ilastik, CP+ilastik, ConAE (unsupervised), SupCon-AE trained with B1 / B2 / B1+B2 labels (labels of the test images hidden; 150 epochs; seed-0 folds), and CP+ilastik+AE",
        "Probe-only arms repeated over 5 fold seeds; AE arms use seed 0"], bullet_size=12)


def cv_tables_slide(prs):
    s = _blank(prs); _slide_header(s, "Cross-validated balanced accuracy by training label source",
                                   "Top: tested on B1 (representative) with 95% CI · Bottom: tested on B2 · AE rows use the encoder trained with the matching label arm")
    reps = [("CP", "CP"), ("IL", "ilastik"), ("CP+IL", "CP+ilastik"), ("ConAE", "ConAE (unsup.)"), ("AE", "SupCon-AE"), ("CP+IL+AE", "CP+ilastik+AE")]
    srcs = ["B1", "B2", "B2m", "B1+B2"]
    for k, (test, top) in enumerate((("B1", 1.2), ("B2", 4.15))):
        rows = [[f"test on {test}"] + [f"train {x}" for x in srcs]]
        for rep, nm in reps:
            row = [nm]
            for src in srcs:
                r = cv_get(cv_rep_name(rep, src), src, test) if (src in ARM or rep not in ("AE", "CP+IL+AE")) else None
                row.append("–" if r is None else (f"{r.bal_acc:.3f} [{r.ci_lo:.2f},{r.ci_hi:.2f}]" if test == "B1" else f"{r.bal_acc:.3f}"))
            rows.append(row)
        table(s, rows, Inches(0.5), Inches(top), Inches(12.3), [2.2, 2.4, 2.4, 2.4, 2.4], row_h=0.34, size=11, bold_first_col=True)


def cv_cross_slide(prs):
    s = _blank(prs); _slide_header(s, "Within- vs cross-annotator: B1→B2 and B2→B1",
                                   "Probe trained on one annotator's labels (AE: encoder + probe), tested on the held-out images of either annotator")
    reps = [("CP", "CP"), ("IL", "ilastik"), ("ConAE", "ConAE"), ("AE", "SupCon-AE")]
    combos = [("B1", "B1", "#0f3d79", "train B1 → test B1"), ("B2", "B1", "#6fa8dc", "train B2 → test B1"),
              ("B1", "B2", "#d65f0e", "train B1 → test B2"), ("B2", "B2", "#f6b26b", "train B2 → test B2")]
    fig, ax = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)
    x = np.arange(len(reps))
    for i, (src, test, col, lab) in enumerate(combos):
        v = [cv_get(cv_rep_name(r, src), src, test).bal_acc for r, _ in reps]
        ax.bar(x + (i - 1.5) * 0.2, v, 0.19, color=col, label=lab)
        for xx, vv in zip(x + (i - 1.5) * 0.2, v):
            ax.text(xx, vv + 0.01, f"{vv:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in reps]); ax.set_ylim(0, 0.85); ax.set_ylabel("balanced accuracy (4 FA classes)")
    ax.legend(frameon=False, fontsize=8, ncol=2); ax.grid(axis="y", alpha=0.3)
    add_fig_w(s, fig, Inches(0.4), Inches(1.2), Inches(7.9))
    _bullet_box(s, Inches(8.5), Inches(1.25), Inches(4.4), Inches(5.6), "Reading", [
        "Tested on B1, a B2-trained model is no better than a B1-trained one (CP 0.55 vs 0.55, ilastik 0.58 vs 0.56; AE 0.46 vs 0.49)",
        "B2 → B2 is high (0.61–0.69 for CP/ilastik) but B1 → B2 is not (0.41–0.54 across representations): the B2 test set is not intrinsically easier; train and test share the annotator's labelling style",
        "So B2-vs-B2 accuracy overstates performance on unseen images; B1-test is the honest benchmark (~0.55–0.60)"], bullet_size=11)


def cv_encoder_slide(prs):
    s = _blank(prs); _slide_header(s, "SupCon-AE: encoder label source × probe label source",
                                   "Balanced accuracy, frame-grouped CV, seed 0 (rows = labels the encoder was trained with; columns = labels the probe was trained with)")
    for test, top in (("B1", 1.25), ("B2", 3.9)):
        rows = [[f"test on {test}", "probe B1", "probe B2", "probe B2 matched", "probe B1+B2"]]
        for arm, nm in (("b1", "encoder: B1 labels"), ("b2", "encoder: B2 labels"), ("b12", "encoder: B1+B2 labels")):
            row = [nm]
            for src in ("B1", "B2", "B2m", "B1+B2"):
                r = cv_get(f"AE_{arm}", src, test)
                row.append(f"{r.bal_acc:.3f}" + (f" [{r.ci_lo:.2f},{r.ci_hi:.2f}]" if test == "B1" else ""))
            rows.append(row)
        rows.append(["ConAE (no labels)"] + [f"{cv_get('ConAE', src, test).bal_acc:.3f}" + (f" [{cv_get('ConAE', src, test).ci_lo:.2f},{cv_get('ConAE', src, test).ci_hi:.2f}]" if test == "B1" else "") for src in ("B1", "B2", "B2m", "B1+B2")])
        table(s, rows, Inches(0.5), Inches(top), Inches(12.3), [2.4, 2.4, 2.4, 2.4, 2.4], row_h=0.42, size=12, bold_first_col=True)
    _txt(s, "On B1 the B2-supervised encoder (0.46–0.53 across probes) is no better than the B1-supervised one (0.49–0.53); the unsupervised ConAE is lowest (0.41–0.48). All intervals overlap widely.",
         Inches(0.5), Inches(6.35), Inches(12.3), Inches(0.9), size_pt=12, color=C_GREY)


def cross_umap_slide(prs, run, desc):
    s = _blank(prs); _slide_header(s, f"Cross-annotator UMAP: {desc}", "Same embedding, coloured by B1 labels (left) and B2 labels (right); grey = unlabelled")
    _img(s, CVD / f"cross_umap_{run}.png", Inches(0.5), Inches(1.15), width=Inches(12.3))


def cv_cm_slide(prs, key, title):
    s = _blank(prs); _slide_header(s, f"Confusion matrices (CV): {title}", "Rows of panels = test set (B1, B2); columns = training label source; row-normalised, counts in brackets")
    _img(s, CVD / f"cv_cm_{key}.png", Inches(0.5), Inches(1.1), width=Inches(12.3))


def cv_answers_slide(prs):
    s = _blank(prs); _slide_header(s, "Answers to the two questions")
    _bullet_box(s, Inches(0.5), Inches(1.3), Inches(6.1), Inches(5.6), "1 · Does the AE add anything over CP / ilastik?", [
        "Not on the representative B1 test set: matched-arm SupCon-AE 0.49 / 0.46 / 0.51 (B1 / B2 / B1+B2 labels) vs CP 0.55 / 0.55 / 0.58 and ilastik 0.56 / 0.58 / 0.60",
        "Supervision helps the AE a little (unsupervised ConAE 0.41–0.48), but not beyond hand-crafted features",
        "Appending AE latents to CP+ilastik lowered accuracy in all three arms (0.46–0.52 vs 0.54–0.57); with ~1000 labels and 160+ unregularised features this may be dimensionality, not proof of no complementarity",
        "Intervals overlap (±0.06–0.07): the data show no AE advantage, not a proven AE deficit"], bullet_size=12)
    _bullet_box(s, Inches(6.8), Inches(1.3), Inches(6.0), Inches(5.6), "2 · Is B2 better than B1 for defining the embedding?", [
        "No. Evaluated on B1 test patches, B2-trained probes (CP 0.55, ilastik 0.58) equal B1-trained ones (0.55, 0.56), also size-matched; the B2-supervised AE is not better than the B1-supervised one",
        "B1+B2 is the best or tied-best arm (about +0.03 for probes; AE 0.51 vs 0.49 / 0.46), consistent with more labels, within noise",
        "B2's high accuracy is within-annotator consistency: B2→B2 0.61–0.69 (CP/ilastik), but B1→B2 0.41–0.54, similar to B1→B1 (0.49–0.56)",
        "In the B1-supervised AE, B2's nascent patches appear in a different region from B1's nascent patches (visual impression): the annotators' 'nascent' may differ"], bullet_size=12)


def run_slide(prs, key, title, pool, labels):
    v = VAL.loc[key]
    s = _blank(prs); _slide_header(s, title, f"pool {pool} patches · labels: {labels} · val bal. acc {v.bal_acc_val:.3f} (train {v.bal_acc_train:.3f}) · n_val = {int(v.n_val)}")
    _img(s, RES / f"umap_{key}.png", Inches(0.4), Inches(1.15), height=Inches(5.9))
    _img(s, RES / f"cm_{key}.png", Inches(7.3), Inches(1.15), height=Inches(5.0))
    _txt(s, "Left: UMAP of latents (all patches; labelled coloured, others grey).  Right: val-split confusion matrix, logreg on latents.",
         Inches(0.5), Inches(7.1), Inches(12.3), Inches(0.3), size_pt=10, color=C_GREY)


def results_slide(prs):
    s = _blank(prs); _slide_header(s, "Held-out results", "Logreg on latents, fit on train-split labelled patches, scored on val-split labelled patches (grouped by image)")
    rows = [["Run", "n val", "Val bal. acc", "Train bal. acc", "Nascent", "Focal cplx", "Focal adh.", "Fibrillar", "No-ad"]]
    for key, *_ in RUNS:
        v = VAL.loc[key]
        f = lambda k: ("–" if pd.isna(v.get("rec_" + k)) else f"{v['rec_' + k]:.2f} ({int(v['nval_' + k])})")
        rows.append([key, int(v.n_val), f"{v.bal_acc_val:.3f}", f"{v.bal_acc_train:.3f}", f("nasc"), f("fcomp"), f("fadh"), f("fibr"), f("noad")])
    table(s, rows, Inches(0.4), Inches(1.25), Inches(12.5), [2.2, 0.8, 1.2, 1.3, 1.3, 1.3, 1.3, 1.2, 1.2], row_h=0.4, size=12, bold_first_col=True)
    _txt(s, "Per-class cells: recall (n val patches in that class). Val sets are small: e.g. 4 fibrillar (B1), 6 focal complex (B2), 12 fibrillar (B1+B2). Differences of a few points are noise.",
         Inches(0.5), Inches(4.55), Inches(12.3), Inches(0.6), size_pt=12, color=C_GREY)
    fig, ax = plt.subplots(figsize=(9.0, 2.2), constrained_layout=True)
    keys = [k for k, *_ in RUNS]; x = np.arange(len(keys))
    ax.bar(x - 0.2, [VAL.loc[k].bal_acc_train for k in keys], 0.38, color="#bbc8ee", label="train")
    ax.bar(x + 0.2, [VAL.loc[k].bal_acc_val for k in keys], 0.38, color="#0f3d79", label="val")
    ax.set_xticks(x); ax.set_xticklabels(keys, rotation=15, ha="right", fontsize=8); ax.set_ylim(0, 1); ax.legend(frameon=False, ncol=2); ax.grid(axis="y", alpha=0.3)
    add_fig_w(s, fig, Inches(2.2), Inches(5.2), Inches(8.9))


def summary_slide(prs):
    s = _blank(prs); _slide_header(s, "Takeaways and caveats")
    _bullet_box(s, Inches(0.5), Inches(1.3), Inches(6.1), Inches(5.6), "What the results show", [
        "Hand-crafted features beat AE latents on held-out patches: B1+B2 0.65 (CP) / 0.62 (ilastik) vs 0.58 (AE); B2-only 0.74 / 0.80 vs 0.58. AE train accuracy is 0.83–0.98, so it overfits",
        "B2 labels beat B1 labels for every representation (B2 0.74–0.80 vs B1 0.48–0.51 for CP/ilastik); val sets differ, so treat as indicative",
        "p≥0.9 pool gave no gain for the AE (and cannot matter for CP/ilastik)",
        "Whole set with a no-adhesion class: 0.62–0.64 over 5 classes, no-ad recall 0.80–0.85",
        "PHATE / diffusion map give stripe-like arcs for every representation, but the arc is not a class-order axis (best |ρ| ≈ 0.7)"], bullet_size=12)
    _bullet_box(s, Inches(6.8), Inches(1.3), Inches(6.0), Inches(5.6), "Caveats", [
        "Single 80/20 image-grouped split per run, 112–529 val patches; per-class recalls rest on 4–141 patches",
        "AE best checkpoint at epoch 36–72 of 300, then val loss rises; no early stopping or tuning",
        "Labelled UMAP points are mostly training samples, so visual separation is optimistic",
        "Gate probabilities were fit on labelled patches; gate precision is optimistic",
        "Hand-crafted features: standardised only, no tuning, no log transform",
        "Next: CV over image groups, early-stopped / smaller AE, concatenate CP+ilastik+AE features"], bullet_size=12)


def main():
    prs = _prs()
    title_slide(prs); design_slide(prs); gate_slide(prs); key_slide(prs)
    for r in RUNS:                                   # AE: UMAP + confusion matrix
        run_slide(prs, *r)
    for rep in VALS:                                 # alternative embeddings per representation
        for key, title, *_ in RUNS:
            embed_slide(prs, key, title, rep)
    for key, title, *_ in RUNS:
        cm_pair_slide(prs, key, title)
    cv_design_slide(prs); cv_tables_slide(prs); cv_cross_slide(prs); cv_encoder_slide(prs)
    for run, desc in (("stage2_b1", "AE trained with B1 labels"), ("stage2_b2", "AE trained with B2 labels"), ("stage2", "AE trained with B1+B2 labels")):
        cross_umap_slide(prs, run, desc)
    for key, title in (("CP", "CP features"), ("IL", "ilastik features"), ("CP_IL", "CP+ilastik"), ("ConAE", "ConAE (unsupervised)"),
                       ("AE", "SupCon-AE (matching label arm)"), ("CP_IL_AE", "CP+ilastik+AE")):
        cv_cm_slide(prs, key, title)
    cv_answers_slide(prs)
    compare_slide(prs); recall_slide(prs); stripe_slide(prs); summary_slide(prs)
    prs.save(OUT); print("saved", OUT, len(prs.slides), "slides")


if __name__ == "__main__":
    main()
