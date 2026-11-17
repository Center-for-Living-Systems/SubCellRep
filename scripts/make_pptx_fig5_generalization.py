#!/usr/bin/env python3
"""
make_pptx_fig5_generalization.py
================================
Paper Figure 5: Generalization to new images / datasets and image efficiency.

Collects every existing result relevant to Fig 5 into one deck (panel candidates + gap list).
New plots are drawn from CSVs and saved individually to results/fig5/ for panel assembly.
Legacy figures (extracted from older decks) live in results/fig5/legacy/.

  A  New images, same day     : LOIO per-image, frame-grouped CV (FA4), legacy image-held-out LE
  B  New dataset / day        : zero-shot DS1->DS2/DS3, blind test, transfer vs adaptation, intensity / normalization
  C  Image efficiency         : balanced accuracy vs number of training images (CP / ilastik / SupCon-AE)

Output: results/fig5_generalization.pptx
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

REPO = Path(__file__).resolve().parents[1]
EV = REPO / "eval_results"
FEAT_EV = Path("/net/projects/CLS/lding/data/fa_data_analysis/ae_results/features/eval_results")
FIG = REPO / "results" / "fig5"
LEG = FIG / "legacy"
OUT = REPO / "results" / "fig5_generalization.pptx"
FIG.mkdir(parents=True, exist_ok=True)

C = {"cp": "#1f77b4", "ilastik": "#ff7f0e", "supcon": "#2ca02c"}
NAMES = {"cp": "CellProfiler", "ilastik": "ilastik", "supcon": "SupCon-AE (leaky*)"}
ANN = {"b1": "B1 (Margaret)", "b2v2": "B2v2 (Annabel)"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    p = FIG / name
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    return p


# ── data ──────────────────────────────────────────────────────────────────────
ie = pd.read_csv(EV / "image_efficiency_b1b2v2_lgbm.csv")
ie["label_budget"] = ie["label_budget"].astype(str)
cb = pd.read_csv(EV / "image_efficiency_cp_combo_lgbm.csv")
cb["label_budget"] = cb["label_budget"].astype(str)
sup = cb[cb.feature == "supcon_z"].assign(feature="supcon")
ie_all = pd.concat([ie, sup[ie.columns]], ignore_index=True)
full = pd.read_csv(EV / "image_efficiency_b1_full_lgbm.csv")
full["label_budget"] = full["label_budget"].astype(str)
FEATS = ["cp", "ilastik", "supcon"]


# ── A1: LOIO per-image ───────────────────────────────────────────────────────
def plot_loio():
    d = ie_all[(ie_all.experiment == "loio") & (ie_all.label_budget == "all")]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, a in zip(axs, ["b1", "b2v2"]):
        for i, f in enumerate(FEATS):
            g = d[(d.annotator == a) & (d.feature == f)].groupby(["held_out_image", "n_test_patches"]).bal_acc.mean().reset_index()
            x = i + np.random.RandomState(0).uniform(-.18, .18, len(g))
            cond = g.held_out_image.str.startswith("ycomp")
            ax.scatter(x, g.bal_acc, s=10 + 3 * g.n_test_patches, c=np.where(cond, "#2ca02c", "#d62728"), alpha=.7,
                       edgecolor="k", lw=.3, hatch=None)
            ax.hlines(g.bal_acc.mean(), i - .3, i + .3, color="k", lw=2)
            ax.text(i, 0.30, f"mean {g.bal_acc.mean():.2f}\nmedian {g.bal_acc.median():.2f}", ha="center", fontsize=9)
        ax.set_xticks(range(3)); ax.set_xticklabels([NAMES[f] for f in FEATS], fontsize=9)
        ax.set_title(f"{ANN[a]} — 22 held-out images"); ax.set_ylim(0.25, 1.03)
    axs[0].set_ylabel("Balanced accuracy on held-out image")
    axs[1].scatter([], [], c="#d62728", label="control"); axs[1].scatter([], [], c="#2ca02c", label="ycomp")
    axs[1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), title="dot size ∝\npatches in image", fontsize=8, title_fontsize=8, frameon=False)
    fig.suptitle("Leave-one-image-out (all labels of the 21 other images), LightGBM", y=1.0)
    fig.text(0.01, -0.03, "*SupCon-AE features come from a fixed model trained with B1+B2 labels on all DS1 images -> test-image leakage; shown for reference only.", fontsize=8, color="#b02020")
    return save(fig, "A1_loio_per_image.png")


# ── A2: frame-grouped CV FA4 ──────────────────────────────────────────────────
def plot_cv():
    s = pd.read_csv(REPO / "results/cv_b1b2/summary.csv")
    reps = ["CP", "IL", "CP+IL", "ConAE", "AE_b12"]
    lab = {"CP": "CP", "IL": "ilastik", "CP+IL": "CP+ilastik", "ConAE": "ConAE\n(unsup.)", "AE_b12": "SupCon-AE\n(B1+B2)"}
    fig, ax = plt.subplots(figsize=(8.5, 4.2)); w = .38
    for j, (t, col) in enumerate([("B1", "#1f77b4"), ("B2", "#d62728")]):
        r = s[(s.src == t) & (s.test == t)].set_index("rep").loc[reps]
        ax.bar(np.arange(len(reps)) + (j - .5) * w, r.bal_acc, w, color=col, label=f"train {t} → test {t} (n={int(r.n.iloc[0])}±)",
               yerr=[r.bal_acc - r.ci_lo, r.ci_hi - r.bal_acc], capsize=3)
    ax.axhline(.25, color="grey", ls=":", lw=1); ax.text(4.6, .26, "chance (4-class)", fontsize=8, ha="right", color="grey")
    ax.set_xticks(range(len(reps))); ax.set_xticklabels([lab[r] for r in reps], fontsize=9)
    ax.set_ylabel("Balanced accuracy (FA4, frame-grouped 5-fold)"); ax.set_ylim(0, .85); ax.legend(fontsize=8)
    ax.set_title("4-class FA subtype on held-out frames (95% CI: bootstrap over frames)")
    return save(fig, "A2_frame_grouped_cv_fa4.png")


# ── B1: zero-shot DS1 -> DS2/DS3 ──────────────────────────────────────────────
def plot_xds():
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    order = ["10", "20", "25", "50", "75", "100", "150", "200", "300", "500", "750", "all"]
    for ax, ds, nm in zip(axs, ["ds2", "ds3"], ["DS2 (pFAK, ctrl)", "DS3 (pPax, ctrl)"]):
        for f, fn in [("cp", "cp_cross_ds.csv"), ("ilastik", "ilastik_cross_ds.csv"), ("supcon", "supcon_cross_ds.csv")]:
            d = pd.read_csv(FEAT_EV / fn); d = d[d.ds_test == ds]; d["budget"] = d.budget.astype(str)
            g = d.groupby("budget").bal_acc.agg(["mean", "std"]).reindex([o for o in order if o in set(d.budget)])
            ax.errorbar(range(len(g)), g["mean"], g["std"], marker="o", ms=4, capsize=2, color=C[f],
                        label={"supcon": "SupCon-AE"}.get(f, NAMES[f]))
            ax.set_xticks(range(len(order))); ax.set_xticklabels(order, fontsize=8)
        ax.axhline(.5, color="grey", ls=":"); ax.set_title(f"train DS1 (B2) → test {nm}"); ax.set_xlabel("# DS1 training labels")
    axs[0].set_ylabel("Balanced accuracy (zero-shot, no adaptation)"); axs[0].legend(fontsize=8)
    fig.text(0.01, -0.04, "DS3 B2 labels are all 'adhesion' -> balanced accuracy is ill-defined (single class); DS2 has 211 patches / 4 images. Old SupCon run (Aug 28), 25 reps.", fontsize=8, color="#b02020")
    return save(fig, "B1_zero_shot_ds1_to_ds2_ds3.png")


# ── C1: image budget ──────────────────────────────────────────────────────────
def plot_ib():
    d = ie_all[(ie_all.experiment == "image_budget") & (ie_all.label_budget == "all")]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, a in zip(axs, ["b1", "b2v2"]):
        for f in FEATS:
            g = d[(d.annotator == a) & (d.feature == f)].groupby("k_images").bal_acc.agg(["mean", "std"])
            ax.plot(g.index, g["mean"], marker="o", ms=4, color=C[f], ls="--" if f == "supcon" else "-", label=NAMES[f])
            ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=C[f], alpha=.15)
        ax.axhline(.9, color="grey", ls=":"); ax.set_xlabel("# training images (K)"); ax.set_title(ANN[a])
    axs[0].set_ylabel("Balanced accuracy on remaining images"); axs[0].legend(fontsize=8, loc="lower right")
    fig.text(0.01, -0.03, "All labels of the K sampled images used; 20 random image subsets per K; test = all other images (test set shrinks as K grows). *SupCon leaky, reference only.", fontsize=8, color="#b02020")
    return save(fig, "C1_image_budget_curves.png")


def plot_ib_labels():
    d = ie[(ie.experiment == "image_budget")]
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    bud = ["5", "20", "50", "100", "200", "all"]; cm = plt.cm.viridis(np.linspace(0, .9, len(bud)))
    for r, a in enumerate(["b1", "b2v2"]):
        for c, f in enumerate(["cp", "ilastik"]):
            ax = axs[r, c]
            for b, col in zip(bud, cm):
                g = d[(d.annotator == a) & (d.feature == f) & (d.label_budget == b)].groupby("k_images").bal_acc.mean()
                ax.plot(g.index, g.values, "-o", ms=3, color=col, label=f"≤{b} labels" if b != "all" else "all labels")
            ax.set_title(f"{ANN[a]} — {NAMES[f]}"); ax.axhline(.9, color="grey", ls=":")
            if r == 1: ax.set_xlabel("# training images (K)")
            if c == 0: ax.set_ylabel("Balanced accuracy")
    axs[0, 0].legend(fontsize=7, ncol=2)
    return save(fig, "C2_image_budget_x_label_budget.png")


def plot_full():
    d = full[(full.experiment == "image_budget")]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    bud = ["20", "100", "300", "600", "1200", "all"]; cm = plt.cm.viridis(np.linspace(0, .9, len(bud)))
    for ax, f in zip(axs, ["cp", "ilastik"]):
        for b, col in zip(bud, cm):
            g = d[(d.feature == f) & (d.label_budget == b)].groupby("k_images").bal_acc.agg(["mean", "std"])
            ax.errorbar(g.index, g["mean"], g["std"], marker="o", ms=3, capsize=2, color=col, label=b)
        ax.set_xscale("log"); ax.set_xlabel("# training images (K, log)"); ax.set_title(f"B1 full set (89 images) — {NAMES[f]}")
        ax.axhline(.9, color="grey", ls=":")
    axs[0].set_ylabel("Balanced accuracy"); axs[0].legend(title="label budget", fontsize=7, ncol=2)
    return save(fig, "C3_image_budget_b1_full.png")


def plot_var():
    d = ie[(ie.experiment == "image_budget") & (ie.annotator == "b2v2") & (ie.label_budget == "all")]
    ks = sorted(d.k_images.unique())
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for f, col in [("cp", C["cp"]), ("ilastik", C["ilastik"])]:
        x = d[d.feature == f]
        axs[0].boxplot([x[x.k_images == k].bal_acc for k in ks], positions=np.arange(len(ks)) + (.2 if f == "ilastik" else -.2),
                       widths=.35, patch_artist=True, boxprops=dict(facecolor=col, alpha=.5), medianprops=dict(color="k"), showfliers=False)
        axs[1].plot(range(len(ks)), [(x[x.k_images == k].bal_acc <= .55).mean() for k in ks], "-o", color=col, label=NAMES[f])
    for ax in axs[:2]:
        ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks); ax.set_xlabel("# training images (K)")
    axs[0].set_ylabel("Bal. acc. across 20 repeats"); axs[0].set_title("Spread across random image subsets (B2v2)")
    axs[1].set_ylabel("Fraction of repeats with bal.acc ≤ 0.55"); axs[1].set_title("Proxy for missing class coverage"); axs[1].legend(fontsize=8)
    p = d[d.feature == "cp"].groupby("k_images").n_train_pool.agg(["mean", "min", "max"])
    axs[2].errorbar(range(len(p)), p["mean"], [p["mean"] - p["min"], p["max"] - p["mean"]], marker="o", color="k", capsize=3)
    axs[2].set_xticks(range(len(p))); axs[2].set_xticklabels(p.index); axs[2].set_xlabel("# training images (K)")
    axs[2].set_ylabel("# labeled training patches (pool)"); axs[2].set_title("Patches available vs images (mean, min–max)")
    fig.tight_layout()
    return save(fig, "C4_variability_class_coverage.png")


def key_numbers():
    d = ie_all[(ie_all.experiment == "image_budget") & (ie_all.label_budget == "all")]
    rows = []
    for a in ["b1", "b2v2"]:
        for f in FEATS:
            g = d[(d.annotator == a) & (d.feature == f)].groupby("k_images").bal_acc.mean()
            r = [ANN[a], NAMES[f]]
            for t in (.8, .9):
                ok = g[g >= t]; r.append(str(ok.index.min()) if len(ok) else "not reached")
            r.append(f"{g.loc[19]:.3f}")
            rows.append(r)
    return [["Annotator", "Features", "K for ≥0.80", "K for ≥0.90", "bal.acc @K=19"]] + rows


# ── pptx helpers ──────────────────────────────────────────────────────────────
prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.33), Inches(7.5)
DARK = RGBColor(0x1A, 0x2A, 0x3A); GREY = RGBColor(0x66, 0x66, 0x66); RED = RGBColor(0xB0, 0x20, 0x20)


def slide(title, sub=""):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    for txt, top, sz, bold, col in [(title, .2, 24, True, DARK), (sub, .82, 12, False, GREY)]:
        tb = s.shapes.add_textbox(Inches(.4), Inches(top), Inches(12.5), Inches(.5)); tf = tb.text_frame; tf.word_wrap = True
        r = tf.paragraphs[0].add_run(); r.text = txt; r.font.size = Pt(sz); r.font.bold = bold; r.font.color.rgb = col
    return s


def text(s, lines, l, t, w, h, sz=12, col=DARK):
    tb = s.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run(); r.text = ln; r.font.size = Pt(sz); r.font.color.rgb = col; p.space_after = Pt(4)


def pic(s, path, l, t, w=None, h=None):
    path = Path(path)
    iw, ih = Image.open(path).size
    if w is not None and h is not None:
        sc = min(w / iw, h / ih); w, h = iw * sc, ih * sc
    elif w is not None: h = w * ih / iw
    else: w = h * iw / ih
    s.shapes.add_picture(str(path), Inches(l), Inches(t), Inches(w), Inches(h))


def table(s, rows, l, t, w, h, sz=11):
    sh = s.shapes.add_table(len(rows), len(rows[0]), Inches(l), Inches(t), Inches(w), Inches(h)).table
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            c = sh.cell(i, j); c.text = str(v)
            for p in c.text_frame.paragraphs:
                for run in p.runs: run.font.size = Pt(sz); run.font.bold = (i == 0)


def fig_slide(title, sub, path, notes, w=8.6, legacy=False):
    s = slide(title, sub)
    pic(s, path, .3, 1.35, w=w, h=5.8)
    text(s, notes, 9.1, 1.4, 4.0, 5.6, sz=11)
    if legacy: text(s, ["LEGACY figure (older labels / model); re-run before use."], .4, 7.1, 8, .3, sz=10, col=RED)
    return s


# ── build ─────────────────────────────────────────────────────────────────────
pA1, pA2, pB1, pC1, pC2, pC3, pC4 = plot_loio(), plot_cv(), plot_xds(), plot_ib(), plot_ib_labels(), plot_full(), plot_var()

s = slide("Figure 5 — Generalization to new datasets/images and image efficiency", "Candidate panels, sources and gaps · all numbers from existing CSVs · DS1 = vinc, DS2 = pFAK, DS3 = pPax")
text(s, ["A  New images, same day: LOIO per-image (A1), frame-grouped FA4 CV (A2), legacy image-held-out LE (A3)",
         "B  New dataset/day: zero-shot DS1→DS2/DS3 (B1), blind cross-dataset test (B2), transfer vs adaptation (B3), intensity + normalization (B4–B5)",
         "C  Image efficiency: bal. acc vs # training images — CP / ilastik / SupCon-AE (C1–C4, summary C5)",
         "Final slide lists what is missing, unclear, or violates the checks (leakage, test-set definition, patches ≠ replicates)."], .5, 1.5, 12, 3, sz=15)

s = slide("Evidence map: Fig 5 panel → source → status")
table(s, [["Panel", "Content", "Source", "Status"],
          ["A", "LOIO per-image (DS1, 22 imgs)", "eval_results/image_efficiency_b1b2v2_lgbm.csv", "OK (CP, ilastik); SupCon leaky"],
          ["A", "Frame-grouped CV, FA4", "results/cv_b1b2/summary.csv", "OK; 4-class, seed-0 AE only"],
          ["A", "Image-held-out vs random split", "noad_vs_ad_story.pptx s56–58", "Legacy (ctrl only, old AE)"],
          ["B", "Zero-shot DS1→DS2/DS3 (CP/IL/SupCon)", "features/eval_results/*_cross_ds.csv", "Partial: DS3 single-class, old SupCon, B2 only"],
          ["B", "Blind cross-dataset test, accuracy", "noad_vs_ad_story s34, blind_test_crossds.pptx", "Legacy; accuracy not bal.acc"],
          ["B", "Transfer vs adaptation (fine-tuning)", "noad_vs_ad_story s63–70", "Legacy; single split seed=42"],
          ["B", "Intensity variation / normalization", "generalizability_overview.pptx, intensity_*.pptx", "Recon error only; no classification"],
          ["C", "Image budget K=1..19 (22 imgs)", "image_efficiency_b1b2v2*.csv, cp_combo", "OK for CP/ilastik"],
          ["C", "Image budget K=1..88 (B1 full)", "image_efficiency_b1_full*.csv", "OK for CP/ilastik"],
          ["C", "SupCon-AE image efficiency", "cp_combo 'supcon_z'", "Invalid (leaky); needs retrain"]],
      .4, 1.4, 12.5, 5, sz=11)

slide("Panel A — New images, same day").shapes  # section divider
s = prs.slides[-1]; text(s, ["A1 LOIO per-image · A2 frame-grouped FA4 CV · A3 legacy image-held-out vs random-split"], .5, 1.5, 12, 1, sz=16)

fig_slide("A1 — Leave-one-image-out (DS1)", "22 valid images (4 sparse images excluded) · all labels of other 21 images", pA1,
          ["B2v2 labels: CP ≈0.99, ilastik ≈0.92 mean per-image bal.acc.", "B1 labels: ≈0.75 for all features, large spread (SD ≈0.16–0.19).",
           "Per-image patch counts are small (4–16 for many images) → per-image bal.acc is noisy; report pooled and image-level CI.",
           "Report: 22 images / ~393 patches; patches from the same image are not independent."], w=8.6)
fig_slide("A2 — Frame-grouped 5-fold CV, 4 FA classes", "Test = patches from held-out frames (B1: ~609, B2: 790)", pA2,
          ["Whole frames held out; CIs bootstrap over frames.", "AE arms: SupCon-AE trained with fold labels hidden (seed 0 only).",
           "Absolute performance modest (0.4–0.7) — FA4 is a harder task than binary.", "Source: results/cv_b1b2/summary.csv; per-seed spread in summary_seeds.csv."], w=8)
s = slide("A3 — Image-held-out vs random patch split (legacy)", "dataset1/vinc/control, Annabel labels, AE latents + LightGBM")
pic(s, LEG / "story_s57_1.png", .3, 1.4, w=8.8)
text(s, ["Random split is only slightly optimistic vs image-held-out (≈1–2 points).", "Legacy: ctrl only, ≤3 training images, old (contaminated→clean) AE protocol.",
         "Replace with image-grouped result using current CP/ilastik/SupCon (A1)."], 9.3, 1.5, 3.8, 4, sz=11)
text(s, ["LEGACY figure; re-run before use."], .4, 7.0, 8, .3, sz=10, col=RED)

slide("Panel B — New dataset / day")
s = prs.slides[-1]; text(s, ["B1 zero-shot feature classifiers · B2 blind test · B3 transfer vs adaptation · B4 intensity · B5 recon generalization"], .5, 1.5, 12, 1, sz=16)
fig_slide("B1 — Zero-shot transfer DS1 → DS2 / DS3", "LightGBM trained on DS1 (B2) labels at each budget, tested on all labeled DS2/DS3 patches", pB1,
          ["ilastik transfers best (≈0.75–0.8 at high budget); CP flat ≈0.65–0.70; SupCon-AE reaches ≈0.77 at 750 labels.",
           "Flat CP curve → little gain from more source labels, likely domain shift in intensity/texture.",
           "DS3 test has one class only → not a valid balanced accuracy.", "No normalization was applied / documented for these runs."], w=8.6)
s = slide("B2 — Blind cross-dataset test (legacy)", "SupCon-2cls z_recon, Annabel vinc/ctrl-trained, tested on Margaret/Ernest labels")
pic(s, LEG / "story_s34_1.png", .3, 1.5, w=8.6); pic(s, LEG / "blind_s22_1.png", .3, 4.4, w=6.0)
text(s, ["Accuracy: in-domain 0.78–0.80; ycomp 0.61–0.72; ppax 0.71–0.75; pfak 0.93–1.0.", "pfak/ppax sets are tiny (54/60 labels) and class-imbalanced → accuracy inflated.",
         "Macro-F1 per model in blind_test_crossds.pptx (22 slides)."], 9.3, 1.5, 3.8, 4, sz=11)
s = slide("B3 — Transfer without retraining vs adaptation with new labels (legacy)", "Must be reported separately (checklist)")
pic(s, LEG / "story_s70_1.png", .3, 1.4, w=8.6)
text(s, ["Left: zero-shot (no fine-tuning). Right: full-AE fine-tuning using a % of target labels (target-only vs ctrl+target).",
         "ycomp: 0.67 zero-shot → ≈0.84 with 10% labels. pfak 100% at 0% looks like a tiny/one-class test set.",
         "Single 80/20 split, seed 42 — no variability estimate.", "More: story_s66–69 in results/fig5/legacy/."], 9.3, 1.5, 3.8, 4, sz=11)
text(s, ["LEGACY figure; re-run before use."], .4, 7.0, 8, .3, sz=10, col=RED)
s = slide("B4 — Intensity variation across datasets / days", "Dataset-level and per-image intensity histograms (raw vs CIO-RB, histogram-matched)")
pic(s, LEG / "dataset_intensity_histograms.png", .3, 1.4, w=6.4); pic(s, LEG / "gen_s3_1.png", 6.8, 1.4, w=6.2)
pic(s, LEG / "per_image_intensity_histograms.png", .3, 4.5, w=6.4); pic(s, LEG / "gen_s4_1.png", 6.8, 4.5, w=6.2)
s = slide("B5 — Cross-dataset reconstruction error vs normalization / training diversity", "AE trained on vinc (± ppax, ± histogram match); external datasets never seen in training (L1)")
pic(s, LEG / "gen_s7_1.png", .3, 1.4, w=12.7)
text(s, ["Adding ppax and histogram matching lowers L1 on external datasets; hist-match raises vinc val L1.", "Reconstruction only — no classification metric. Hist-match reference statistics: fit on training data? (verify)."], .4, 5.2, 12.5, 1.5, sz=12)

slide("Panel C — Image efficiency")
s = prs.slides[-1]; text(s, ["C1 curves · C2 × label budget · C3 B1 full set · C4 variability · C5 numbers"], .5, 1.5, 12, 1, sz=16)
fig_slide("C1 — Balanced accuracy vs number of training images", "DS1, all labels of the K images; 20 random subsets per K", pC1,
          ["B2v2: ≥0.90 by ~5–7 images for CP; B1 labels plateau ≈0.78–0.8.", "Label-quality difference dominates image count.", "Test set = remaining images, varies with K (not fixed).",
           "SupCon dashed: leaky features, not a valid claim."], w=8.6)
fig_slide("C2 — Image budget × label budget (CP, ilastik)", "Rows: annotator · lines: cap on labels drawn from the K images", pC2,
          ["Separates image coverage from label count.", "At small label caps, adding images adds little; at 'all', curves saturate at ~10 images."], w=8)
fig_slide("C3 — B1 full set (89 images)", "K up to 88; log x axis", pC3, ["Longer image range than the 22-image matched set.", "CP / ilastik only; no SupCon-AE."], w=8.6)
fig_slide("C4 — Variability, class coverage, patches vs images", "Spread across 20 repeats (B2v2, all labels)", pC4,
          ["K=1–2: wide spread, many repeats ≈ chance (single class or too few patches).", "Patches per image are very uneven (pool size ranges widely at fixed K).",
           "Class coverage is only inferred from bal.acc ≤0.55; not logged directly."], w=12.6)
s = slide("C5 — Key numbers (K images needed)", "Mean over 20 repeats, all labels; SupCon rows are leaky")
table(s, key_numbers(), .6, 1.5, 12, 3, sz=13)

s = slide("Gaps, unclear items, and checklist status", "What is not available or needs a decision before the Fig 5 conclusion")
text(s, [
    "MISSING",
    "• SupCon-AE image efficiency without leakage (current supcon_z = B1+B2 labels, all DS1 images). Needs per-split retraining.",
    "• Fixed test set across K in image-budget curves (current test = remaining images; shrinks 21→3 images).",
    "• Image-grouped label-efficiency for SupCon-AE (A) with current protocol; A3 is legacy.",
    "• Direct class-coverage logging per repeat (only proxied).",
    "• Normalization comparison on classification (only recon error and histograms exist).",
    "• Transfer to DS2/DS3 with B1/B12 labels; DS2→DS1 and reverse; DS4/nih3t3 labels absent.",
    "UNCLEAR / NEEDS DECISION",
    "• DS3 B2 is all adhesion → bal.acc invalid; drop or use a different metric (adhesion recall).",
    "• Treat ctrl and ycomp as separate 'days' or one dataset? (open concern in efficiency_test_plan.md)",
    "• Legacy blind-test/fine-tuning figures use accuracy and one split; re-run with repeats + bal.acc?",
    "• Fig 3 panel D already shows CP/ilastik image efficiency — avoid duplicating in Fig 5C.",
    "• Was any normalization/hist-match fit using test images? Not recorded for cross_ds or generalizability runs.",
    "CHECKS",
    "• Report cells/images: LOIO 22 images, matched set 393 patches; B1 full 89 images. Cell counts per image not tabulated.",
    "• Many patches from few images ≠ independent replicates: use image-level CIs (done in A2; not in C1).",
], .4, 1.3, 12.5, 6, sz=12)

prs.save(OUT)
print("saved", OUT)
