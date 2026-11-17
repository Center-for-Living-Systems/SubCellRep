#!/usr/bin/env python3
"""
make_pptx_fig4_annotation_reliability.py
=========================================
Self-readable explainer deck for the paper figure "Annotation strategy and
prediction reliability" (user numbering: Figure 4).  One section per panel:

  A  B1 vs B2: sampling rules, counts, class mix, example patches
  B  Matched cross-evaluation (B1/B2 x B1/B2) + common-test-population budget curves
  C  Score distributions TP/TN/FP/FN, accuracy AND coverage at 0.3/0.7,
     accuracy vs retained fraction, representative errors
  D  Human agreement vs score (DS1 patches annotated independently by 2 annotators)
  +  Verification of the reported numbers, open tests

Every panel plot is also saved as PNG in results/fig4_panels/ for figure assembly.

Output: results/fig4_annotation_reliability.pptx
"""
from __future__ import annotations

import io
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, spearmanr, mannwhitneyu
from PIL import Image

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_pptx_fig4_confidence as f4          # reuse patch loader / outcome helpers

REPO   = Path(__file__).resolve().parents[1]
DATA   = Path("/net/projects/CLS/lding/data/fa_data_analysis")
LAB    = DATA / "labelling"
EV     = REPO / "eval_results"
PRED   = EV / "patch_preds"
PANELS = REPO / "results" / "fig4_panels"
OUT    = REPO / "results" / "fig4_annotation_reliability.pptx"
PANELS.mkdir(parents=True, exist_ok=True)

SW, SH = 13.33, 7.5
C_B1, C_B2, C_MIX = "#1f77b4", "#d62728", "#9467bd"
C_DARK, C_GREY = "#1a2a3a", "#666666"
C_TP, C_TN, C_FP, C_FN = "#2ca02c", "#9467bd", "#d62728", "#ff7f0e"
CLASS_ORDER = ["No adhesion", "Nascent Adhesion", "focal complex",
               "focal adhesion", "fibrillar adhesion"]
CLASS_COL = ["#9467bd", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
CLASS_SHORT = ["No adh.", "Nascent", "Focal cplx", "Focal adh.", "Fibrillar"]
LO, HI = 0.3, 0.7

plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})


# ═════════════════════════════ PPTX helpers ═════════════════════════════════
def rgb(h):
    h = h.lstrip("#")
    return RGBColor(int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16))


def new_slide(prs, title, subtitle=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bar = s.shapes.add_shape(1, 0, 0, Inches(SW), Inches(0.95))
    bar.fill.solid(); bar.fill.fore_color.rgb = rgb(C_DARK); bar.line.fill.background()
    text(s, title, 0.35, 0.08, SW - 0.7, 0.5, size=22, bold=True, color="#ffffff")
    if subtitle:
        text(s, subtitle, 0.35, 0.55, SW - 0.7, 0.35, size=13, color="#c9d6e3")
    return s


def text(slide, s, left, top, width, height, size=13, bold=False, color="#222222",
         italic=False, align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = align
    r = p.add_run(); r.text = s
    r.font.size = Pt(size); r.font.bold = bold; r.font.italic = italic
    r.font.color.rgb = rgb(color)
    return tb


def bullets(slide, items, left, top, width, height, size=13, head=None, head_col=C_DARK,
            fill=None):
    """items: list of str or (str, dict(bold/color/indent))."""
    if fill:
        bx = slide.shapes.add_shape(1, Inches(left), Inches(top), Inches(width), Inches(height))
        bx.fill.solid(); bx.fill.fore_color.rgb = rgb(fill); bx.line.fill.background()
    tb = slide.shapes.add_textbox(Inches(left + 0.05), Inches(top + 0.03),
                                  Inches(width - 0.1), Inches(height - 0.06))
    tf = tb.text_frame; tf.word_wrap = True
    first = True
    if head:
        p = tf.paragraphs[0]; first = False
        r = p.add_run(); r.text = head
        r.font.size = Pt(size + 1); r.font.bold = True; r.font.color.rgb = rgb(head_col)
        p.space_after = Pt(3)
    for it in items:
        s, o = (it, {}) if isinstance(it, str) else it
        p = tf.paragraphs[0] if first else tf.add_paragraph(); first = False
        r = p.add_run(); r.text = ("• " if not o.get("nobullet") else "") + s
        r.font.size = Pt(o.get("size", size)); r.font.bold = o.get("bold", False)
        r.font.color.rgb = rgb(o.get("color", "#222222"))
        p.space_after = Pt(4)
        if o.get("indent"):
            p.level = 1
    return tb


def picture(slide, img, left, top, width=None, height=None):
    if isinstance(img, Path):
        pil = Image.open(img)
    else:
        pil = img
    w, h = pil.size
    if width and not height:
        height = width * h / w
    elif height and not width:
        width = height * w / h
    elif width and height:      # fit inside box, keep aspect, centre
        sc = min(width / w, height / h)
        nw, nh = w * sc, h * sc
        left += (width - nw) / 2; top += (height - nh) / 2
        width, height = nw, nh
    buf = io.BytesIO(); pil.save(buf, format="PNG"); buf.seek(0)
    slide.shapes.add_picture(buf, Inches(left), Inches(top), Inches(width), Inches(height))
    return width, height


def table(slide, rows, left, top, width, col_w=None, size=11, row_h=0.32, hdr_col=C_DARK,
          bold_first_col=True):
    nr, nc = len(rows), len(rows[0])
    gs = slide.shapes.add_table(nr, nc, Inches(left), Inches(top), Inches(width),
                                Inches(row_h * nr))
    t = gs.table
    if col_w:
        for i, w in enumerate(col_w):
            t.columns[i].width = Inches(w)
    for i, row in enumerate(rows):
        t.rows[i].height = Inches(row_h)
        for j, v in enumerate(row):
            c = t.cell(i, j); c.text = str(v)
            c.margin_left = c.margin_right = Inches(0.05)
            c.margin_top = c.margin_bottom = Inches(0.02)
            for p in c.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(size)
                    r.font.bold = (i == 0) or (j == 0 and bold_first_col)
                    r.font.color.rgb = rgb("#ffffff" if i == 0 else "#222222")
            c.fill.solid()
            c.fill.fore_color.rgb = rgb(hdr_col if i == 0 else ("#eef2f6" if i % 2 else "#ffffff"))
    return t


def savefig(fig, name, dpi=220):
    p = PANELS / f"{name}.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


# ═════════════════════════════ statistics helpers ═══════════════════════════
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def cohen_kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean()
    cats = set(a) | set(b)
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def zone_stats(df, lo=LO, hi=HI):
    p = df.prob_adhesion.values
    ok = (df.y_true == df.y_pred).values
    mid = (p > lo) & (p < hi)           # strictly inside the borderline band
    return dict(n=len(df), acc=ok.mean(),
                cov=(~mid).mean(), acc_out=ok[~mid].mean() if (~mid).any() else np.nan,
                n_mid=int(mid.sum()), acc_mid=ok[mid].mean() if mid.any() else np.nan,
                k_mid=int(ok[mid].sum()), k_out=int(ok[~mid].sum()), n_out=int((~mid).sum()))


# ═════════════════════════════ data loading ═════════════════════════════════
def bin_label(s):
    return s.map(lambda x: 0 if str(x).lower().startswith("no ad") else 1)


def load_labels():
    b1 = pd.read_csv(LAB / "labels_vinc_20260521.csv")
    b1 = b1[b1.classification != "Uncertain"].copy()
    b1["uid"] = b1.unique_ID.str.replace("-", "_", n=1)
    b1["cls"] = b1.classification
    b2 = pd.read_csv(LAB / "vinc_combined_label_Annabel_20260816.csv")
    b2["uid"] = b2.filename
    b2["cls"] = b2.label
    for d in (b1, b2):
        d["image"] = d.uid.str.extract(r"^((?:control|ycomp)_f\d+)x")[0]
        d["cond"] = d.uid.str.extract(r"^(control|ycomp)_")[0]
        d["y"] = bin_label(d.cls)
    return b1, b2


def load_patch_examples_df():
    return f4.load_patch_stores()


# ═════════════════════════════ PANEL A ══════════════════════════════════════
def panel_a_composition(b1, b2):
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6),
                            gridspec_kw={"width_ratios": [1.15, 1, 1]})
    # (1) 5-class composition
    ax = axs[0]
    for i, (nm, d, col) in enumerate([("B1", b1, C_B1), ("B2", b2, C_B2)]):
        cnt = d.cls.value_counts().reindex(CLASS_ORDER).fillna(0)
        fr = cnt / cnt.sum(); left = 0
        for k, c in enumerate(CLASS_ORDER):
            ax.barh(1 - i, fr[c], left=left, color=CLASS_COL[k], edgecolor="white",
                    height=0.6, label=CLASS_SHORT[k] if i == 0 else None)
            if fr[c] > 0.06:
                ax.text(left + fr[c] / 2, 1 - i, f"{int(cnt[c])}", ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold")
            left += fr[c]
        ax.text(-0.02, 1 - i, f"{nm}\nn={int(cnt.sum())}", ha="right", va="center",
                color=col, fontweight="bold", fontsize=11)
    ax.set_xlim(0, 1); ax.set_yticks([]); ax.set_xlabel("Fraction of patches")
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax.set_title("Class composition (counts in bars)", fontweight="bold", fontsize=11)
    # (2) patches per image
    ax = axs[1]
    for nm, d, col in [("B1", b1, C_B1), ("B2", b2, C_B2)]:
        ppi = d.groupby("image").size().sort_values(ascending=False).values
        ax.plot(np.arange(1, len(ppi) + 1), ppi, "o-", color=col, ms=3.5, lw=1.4,
                label=f"{nm}: {len(ppi)} images")
    ax.set_xlabel("Image rank"); ax.set_ylabel("Labelled patches per image")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Image coverage", fontweight="bold", fontsize=11)
    # (3) B1 Position annotation (cell-front bias evidence)
    ax = axs[2]
    pos = b1.Position.fillna("n/a").value_counts()
    ax.barh(range(len(pos)), pos.values[::-1], color=C_B1)
    ax.set_yticks(range(len(pos))); ax.set_yticklabels(pos.index[::-1], fontsize=9)
    for i, v in enumerate(pos.values[::-1]):
        ax.text(v + 5, i, f"{v} ({v / pos.sum():.0%})", va="center", fontsize=8.5)
    ax.set_xlim(0, pos.max() * 1.35)
    ax.set_xlabel("B1 patches"); ax.set_title("B1: annotator-recorded patch location",
                                              fontweight="bold", fontsize=11)
    fig.tight_layout()
    return savefig(fig, "A1_composition")


def panel_a_examples(b1, b2, ctrl, ycomp, n=5, seed=3):
    rng = np.random.RandomState(seed)
    fig, axs = plt.subplots(5, 2 * n + 1, figsize=(12.5, 6.4),
                            gridspec_kw={"width_ratios": [1] * n + [0.25] + [1] * n})
    for r, c in enumerate(CLASS_ORDER):
        for side, (d, col) in enumerate([(b1, C_B1), (b2, C_B2)]):
            pool = d[d.cls == c]
            sel = pool.sample(min(n, len(pool)), random_state=rng) if len(pool) else pool
            for k in range(n):
                ax = axs[r, k + side * (n + 1)]
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_edgecolor(col); sp.set_linewidth(1.3)
                if k < len(sel):
                    pt = f4.get_patch(sel.iloc[k].uid, ctrl, ycomp)
                    if pt is not None:
                        ax.imshow(pt, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                    else:
                        ax.set_facecolor("#eeeeee")
                else:
                    ax.set_facecolor("#eeeeee")
                if k == 0 and side == 0:
                    ax.set_ylabel(CLASS_SHORT[r], rotation=0, ha="right", va="center",
                                  fontsize=11, fontweight="bold", color=CLASS_COL[r], labelpad=8)
        axs[r, n].axis("off")
    axs[0, 2].set_title("B1 — broadly sampled", color=C_B1, fontweight="bold", fontsize=12)
    axs[0, n + 1 + 2].set_title("B2 — prototype-selected", color=C_B2, fontweight="bold", fontsize=12)
    fig.subplots_adjust(wspace=0.05, hspace=0.06)
    return savefig(fig, "A2_examples")


# ═════════════════════════════ PANEL B ══════════════════════════════════════
def panel_b_cross():
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.0))
    mats = {}
    for ax, feat, ttl in zip(axs, ["cp", "ilastik"], ["CellProfiler features", "ilastik features"]):
        d = pd.read_csv(EV / "b1b2v2" / f"le_b1b2v2_cross_{feat}_lgbm.csv")
        M = np.zeros((2, 2)); S = np.zeros((2, 2))
        for i, tr in enumerate(["b1", "b2v2"]):
            for j, te in enumerate(["b1", "b2v2"]):
                v = d[(d.train_batch == tr) & (d.test_batch == te)].bal_acc
                M[i, j], S[i, j] = v.mean(), v.std()
        mats[feat] = (M, S)
        im = ax.imshow(M, cmap="RdYlGn", vmin=0.55, vmax=1.0)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{M[i, j]:.3f}\n±{S[i, j]:.3f}", ha="center", va="center",
                        fontsize=12, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["B1 test", "B2 test"], fontsize=11)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["train B1", "train B2"], fontsize=11)
        ax.set_title(ttl, fontweight="bold", fontsize=11)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle("Balanced accuracy, 5-fold CV, LightGBM (mean ± SD over folds)", fontsize=10, y=0.02)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return savefig(fig, "B1_cross_heatmap"), mats


def panel_b_budget():
    out = {}
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 6.6), sharey=True)
    for r, feat in enumerate(["cp", "ilastik"]):
        d = pd.read_csv(EV / "b1b2v2" / f"le_b1b2v2_budget_{feat}_lgbm.csv")
        d = d[d.budget > 0]
        for c, te in enumerate(["b1", "b2v2"]):
            ax = axs[r, c]
            for st, col, lab in [("b1_only", C_B1, "train B1"), ("b2v2_only", C_B2, "train B2"),
                                 ("mixed", C_MIX, "train 50:50 mix")]:
                g = d[(d.strategy == st) & (d.test_batch == te)].groupby("budget").bal_acc
                m, s = g.mean(), g.std()
                m = m[m.index <= 300] if st != "mixed" else m[m.index <= 300]
                ax.plot(m.index, m.values, color=col, lw=2, label=lab)
                ax.fill_between(m.index, (m - s.loc[m.index]).values, (m + s.loc[m.index]).values,
                                color=col, alpha=0.12)
                out[(feat, te, st)] = m
            ax.set_ylim(0.5, 1.0); ax.grid(alpha=0.25)
            ax.set_title(f"{'CellProfiler' if feat == 'cp' else 'ilastik'} — tested on "
                         f"{'B1' if te == 'b1' else 'B2'} test patches", fontsize=10.5, fontweight="bold")
            if r == 1:
                ax.set_xlabel("Training labels (budget)")
            if c == 0:
                ax.set_ylabel("Balanced accuracy")
            if r == 0 and c == 0:
                ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    return savefig(fig, "B2_budget_common_test"), out


# ═════════════════════════════ PANEL C ══════════════════════════════════════
def outcome(df):
    return f4.add_outcome_cols(df)


def kde_ax(ax, df, title):
    xs = np.linspace(0, 1, 400)
    for code, col, lab in [("TP", C_TP, "TP"), ("TN", C_TN, "TN"), ("FP", C_FP, "FP"), ("FN", C_FN, "FN")]:
        v = df[df.outcome == code].prob_adhesion.values
        if len(v) < 3:
            continue
        ys = gaussian_kde(v, bw_method=0.12)(xs) * len(v) / len(df)   # density scaled by share
        ax.plot(xs, ys, color=col, lw=2, label=f"{lab} (n={len(v)})")
        ax.fill_between(xs, ys, color=col, alpha=0.10)
    ax.axvline(LO, color="#555", ls="--", lw=1); ax.axvline(HI, color="#555", ls="--", lw=1)
    ax.axvspan(LO, HI, color="#dddddd", alpha=0.5, zorder=0)
    ax.set_xlim(0, 1); ax.set_xlabel("Prediction score (prob. adhesion)")
    ax.set_ylabel("Density × share of patches")
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper center")


def panel_c_dist(pops):
    keys = ["union B1 test", "union B2 test"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.7), sharey=True)
    for ax, k in zip(axs, keys):
        kde_ax(ax, pops[k], f"{k.replace('union ', 'Union model → ')} (n={len(pops[k])})")
    fig.tight_layout()
    return savefig(fig, "C1_score_distributions")


def panel_c_zone_bars(pops, keys):
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.7))
    w = 0.26
    cols = [C_TN, "#aaaaaa", C_TP]
    zn = ["≤ 0.3", "0.3–0.7", "≥ 0.7"]
    for pi, k in enumerate(keys):
        df = pops[k]; p = df.prob_adhesion.values; ok = (df.y_true == df.y_pred).values
        masks = [p <= LO, (p > LO) & (p < HI), p >= HI]
        for zi, m in enumerate(masks):
            x = zi + (pi - (len(keys) - 1) / 2) * w
            axs[0].bar(x, m.mean() * 100, w * 0.92, color=cols[zi], alpha=0.45 + 0.5 * pi / max(1, len(keys) - 1),
                       edgecolor="white")
            if m.any():
                axs[1].bar(x, ok[m].mean() * 100, w * 0.92, color=cols[zi],
                           alpha=0.45 + 0.5 * pi / max(1, len(keys) - 1), edgecolor="white")
                lo, hi = wilson(ok[m].sum(), m.sum())
                axs[1].plot([x, x], [lo * 100, hi * 100], color="#333", lw=1)
                axs[1].text(x, 3, f"n={m.sum()}", ha="center", fontsize=7.5, rotation=90, color="white")
    for ax, yl, t in zip(axs, ["Patches in zone (%)", "Accuracy in zone (%)"],
                         ["Coverage per score zone", "Accuracy per score zone (95% Wilson CI)"]):
        ax.set_xticks(range(3)); ax.set_xticklabels(zn); ax.set_ylabel(yl)
        ax.set_title(t, fontweight="bold", fontsize=11)
    axs[1].set_ylim(0, 105)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#888", alpha=0.45 + 0.5 * i / max(1, len(keys) - 1))
               for i in range(len(keys))]
    axs[0].legend(handles, keys, frameon=False, fontsize=9)
    fig.tight_layout()
    return savefig(fig, "C2_zone_coverage_accuracy")


def panel_c_curves(pops, keys, colors):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.0))
    ts = np.linspace(0, 0.5, 200)
    for k, col in zip(keys, colors):
        df = pops[k]; p = df.prob_adhesion.values; ok = (df.y_true == df.y_pred).values
        cov, acc = [], []
        for t in ts:
            m = (p <= t) | (p >= 1 - t)
            cov.append(m.mean()); acc.append(ok[m].mean() if m.any() else np.nan)
        cov, acc = np.array(cov), np.array(acc)
        valid = cov > 0.4
        axs[0].plot(cov[valid] * 100, acc[valid] * 100, color=col, lw=2, label=f"{k} (n={len(df)})")
        z = zone_stats(df)
        axs[0].scatter([z["cov"] * 100], [z["acc_out"] * 100], color=col, s=55, zorder=5,
                       edgecolor="k", linewidth=0.8)
        axs[1].plot(ts, cov * 100, color=col, lw=2, ls="--")
        axs[1].plot(ts, acc * 100, color=col, lw=2, label=k)
    axs[0].set_xlabel("Coverage: patches retained (%)"); axs[0].set_ylabel("Accuracy on retained (%)")
    axs[0].set_title("Accuracy vs retained fraction (● = 0.3/0.7 boundary)", fontweight="bold", fontsize=11)
    axs[0].legend(frameon=False, fontsize=8.5, loc="lower left"); axs[0].grid(alpha=0.25)
    axs[1].axvline(0.3, color="#555", ls=":", lw=1)
    axs[1].text(0.305, 41, "0.3/0.7", fontsize=9, color="#555")
    axs[1].set_xlabel("Score margin t  (retain if score ≤ t or ≥ 1−t)")
    axs[1].set_ylabel("%  (solid = accuracy, dashed = coverage)")
    axs[1].set_ylim(40, 101); axs[1].grid(alpha=0.25)
    axs[1].set_title("Accuracy and coverage as the boundary tightens", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return savefig(fig, "C3_acc_vs_retained")


def panel_c_examples(df, ctrl, ycomp):
    img = f4._fig_panel_d(df, ctrl, ycomp, n_per_row=8)
    p = PANELS / "C4_examples.png"; img.save(p)
    return p


# ═════════════════════════════ PANEL D ══════════════════════════════════════
def agreement_tables(b1, b2):
    rows = {}
    m = b1.merge(b2, on="uid", suffixes=("_1", "_2"))
    rows["DS1 B1 vs B2 (full sets)"] = dict(
        n=len(m), agree2=(m.y_1 == m.y_2).mean(), k2=int((m.y_1 == m.y_2).sum()),
        kappa2=cohen_kappa(m.y_1, m.y_2),
        agree5=(m.cls_1.str.lower() == m.cls_2.str.lower()).mean(),
        who="Margaret / Annabel")
    # matched v2 sets (24 overlapping patches)
    a = pd.read_csv(LAB / "vinc_b1_matched_ds1_v2.csv"); b = pd.read_csv(LAB / "vinc_b2v2_matched_ds1.csv")
    mm = a.merge(b, on="unique_ID", suffixes=("_1", "_2"))
    rows["DS1 B1 vs B2v2 matched sets"] = dict(
        n=len(mm), agree2=(mm.label_1 == mm.label_2).mean(), k2=int((mm.label_1 == mm.label_2).sum()),
        kappa2=cohen_kappa(mm.label_1, mm.label_2), agree5=np.nan, who="Margaret / Annabel")
    # Margaret relabelling the same patches on two days
    x = pd.read_excel(LAB / "Margaret_12_18_and_12_19_labelling_twice.xlsx")
    c1, c2 = x["classification_12-18"], x["classification_12-19"]
    keep = (c1 != "Uncertain") & (c2 != "Uncertain")
    y1, y2 = bin_label(c1[keep]), bin_label(c2[keep])
    rows["DS1 Margaret, same patches, 2 days"] = dict(
        n=int(keep.sum()), agree2=(y1.values == y2.values).mean(), k2=int((y1.values == y2.values).sum()),
        kappa2=cohen_kappa(y1.values, y2.values),
        agree5=(c1[keep].str.lower() == c2[keep].str.lower()).mean(), who="Margaret (intra-rater)")
    return rows, m


def panel_d_plot(m, b1_scores):
    """m: overlap B1∩B2 (binary labels y_1,y_2); b1_scores: Series uid -> score (B1-trained LOIO)."""
    d = m.merge(b1_scores.rename("score"), left_on="uid", right_index=True)
    d["disagree"] = (d.y_1 != d.y_2).astype(int)
    d["margin"] = (d.score - 0.5).abs()
    d["zone"] = pd.cut(d.score, [-0.01, LO, HI, 1.01], labels=["≤0.3", "0.3–0.7", "≥0.7"])
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axs[0]
    rng = np.random.RandomState(0)
    for dis, col, lab in [(0, "#2ca02c", "annotators agree"), (1, "#d62728", "annotators disagree")]:
        g = d[d.disagree == dis]
        ax.scatter(g.score, dis + rng.uniform(-0.18, 0.18, len(g)), color=col, s=40, alpha=0.8,
                   edgecolor="k", linewidth=0.4, label=f"{lab} (n={len(g)})")
    ax.axvspan(LO, HI, color="#dddddd", alpha=0.5, zorder=0)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["agree", "disagree"]); ax.set_ylim(-0.6, 1.6)
    ax.set_xlim(0, 1); ax.set_xlabel("Classifier score (B1-trained, leave-one-image-out)")
    ax.set_title(f"Each dot = one patch labelled by both annotators (n={len(d)})",
                 fontweight="bold", fontsize=10.5)
    ax.legend(frameon=False, fontsize=9, loc="center left")
    ax = axs[1]
    stats = []
    for zi, z in enumerate(["≤0.3", "0.3–0.7", "≥0.7"]):
        g = d[d.zone == z]; k = g.disagree.sum(); n = len(g)
        lo, hi = wilson(k, n); r = k / n if n else np.nan
        stats.append((z, n, int(k), r, lo, hi))
        if n:
            ax.bar(zi, r * 100, color=["#9467bd", "#aaaaaa", "#2ca02c"][zi], width=0.6)
            ax.plot([zi, zi], [lo * 100, hi * 100], color="#333")
        ax.text(zi, 2, f"{k}/{n}", ha="center", color="#222" if not n or r < 0.2 else "white", fontsize=10,
                fontweight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels(["≤0.3", "0.3–0.7", "≥0.7"])
    ax.set_ylabel("Annotator disagreement (%)"); ax.set_ylim(0, 105)
    ax.set_title("Disagreement rate by score zone (95% CI)", fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    rho, pr = spearmanr(d.margin, d.disagree)
    mid = d[d.zone == "0.3–0.7"].disagree; out = d[d.zone != "0.3–0.7"].disagree
    return savefig(fig, "D1_agreement_vs_score"), dict(stats=stats, rho=rho, p=pr, n=len(d),
                                                       n_dis=int(d.disagree.sum()), d=d)


# ═════════════════════════════ build deck ═══════════════════════════════════
def main():
    b1, b2 = load_labels()
    ctrl, ycomp = f4.load_patch_stores()

    # ---- populations for panel C ----
    un = outcome(pd.read_csv(PRED / "le_union_cp_lgbm.csv"))
    pops = {"union B1 test": un[un.test_batch == "b1"].copy(),
            "union B2 test": un[un.test_batch == "b2v2"].copy(),
            "union B1+B2 pooled": un.copy()}
    ie = pd.read_csv(PRED / "image_eff_b1b2v2_lgbm.csv")
    ie_cp = outcome(ie[ie.feature == "cp"].copy())
    ie_both = outcome(ie.copy())
    pops["LOIO image-eff. (CP)"] = ie_cp
    zs = {k: zone_stats(v) for k, v in pops.items()}
    zs["LOIO image-eff. (CP+ilastik pooled)"] = zone_stats(ie_both)

    # ---- plots ----
    pA1 = panel_a_composition(b1, b2)
    pA2 = panel_a_examples(b1, b2, ctrl, ycomp)
    pB1, mats = panel_b_cross()
    pB2, bud = panel_b_budget()
    pC1 = panel_c_dist(pops)
    keysC = ["union B1 test", "union B2 test", "union B1+B2 pooled"]
    pC2 = panel_c_zone_bars(pops, ["union B1 test", "union B2 test"])
    pC3 = panel_c_curves(pops, keysC + ["LOIO image-eff. (CP)"],
                         [C_B1, C_B2, "#555555", C_MIX])
    pC4 = panel_c_examples(pops["union B1+B2 pooled"], ctrl, ycomp)
    agr, ov = agreement_tables(b1, b2)
    loio_b1 = pd.read_csv(PRED / "image_eff_b1_full_lgbm.csv")
    loio_b1 = loio_b1[loio_b1.feature == "cp"].set_index("fn").prob_adhesion
    pD1, dstats = panel_d_plot(ov, loio_b1)

    # B1 within-population accuracy numbers for verification
    cross = pd.read_csv(PRED / "le_cross_cp_lgbm.csv")
    cross_acc = cross.groupby(["train_batch", "test_batch"]).apply(lambda g: (g.y_true == g.y_pred).mean())

    # ---- native stats for slide A ----
    def nimg(d, c=None):
        return d[d.cond == c].image.nunique() if c else d.image.nunique()
    shared_imgs = len(set(b1.image) & set(b2.image))
    shared_patch = len(set(b1.uid) & set(b2.uid))

    # ═════ slides ═════
    prs = Presentation(); prs.slide_width = Inches(SW); prs.slide_height = Inches(SH)

    # 1 — overview
    s = new_slide(prs, "Figure 4 — Annotation strategy and prediction reliability",
                  "Overview: what the figure claims, which panel carries which evidence, and what is still open")
    bullets(s, [
        "Broadly sampled annotations (B1) test whether a classifier works on ALL patches; "
        "prototype annotations (B2) make an easier benchmark.",
        "The classifier's prediction score separates more reliable from less reliable patches — "
        "but the score is not a calibrated probability of being correct.",
        "Dataset: DS1 (vinculin channel, U2OS, control + Y-compound), 32×32 px patches, "
        "binary task adhesion vs. no adhesion. Classifier: LightGBM on CellProfiler (CP) or ilastik features.",
    ], 0.35, 1.1, 12.6, 1.75, size=14, head="Proposed conclusion", fill="#eef2f6")
    rows = [["Panel", "Question it answers", "Evidence (slides)", "Status"],
            ["A", "How do B1 and B2 differ?", "Sampling rules, patch/image/class counts, examples (3–4)", "Descriptive; sampling text needs your confirmation"],
            ["B", "Which labels train a classifier that serves all patches?", "2×2 cross-evaluation + budget curves on a common test set (5–6)", "Supported, with annotator confound"],
            ["C", "Can scores flag reliable predictions?", "TP/TN/FP/FN scores, coverage + accuracy at 0.3/0.7, errors (7–10)", "Supported; population-specific"],
            ["D", "Do uncertain scores coincide with annotator disagreement?", "Independent double-annotated patches (11)", "Preliminary signal only (n = %d); not established" % dstats["n"]]]
    table(s, rows, 0.35, 3.05, 12.6, col_w=[0.8, 3.6, 5.0, 3.2], size=12, row_h=0.5)
    bullets(s, [
        "B1 accuracy and annotator agreement are each ≈ 80%; B2 accuracy can exceed 95% — but these come from DIFFERENT "
        "evaluation populations and must not be merged into a single performance claim (slide 12 re-derives each number).",
        "Figure numbering: your note says Figure 4; the pasted section header says Figure 3. "
        "Existing decks fig3_annotation_strategy.pptx / fig4_confidence.pptx are the earlier one-slide drafts.",
    ], 0.35, 5.75, 12.6, 1.5, size=13, head="Read this first", head_col="#aa3333", fill="#fdeeee")

    # 2 — experimental logic
    s = new_slide(prs, "How to read the figure: four evaluation populations",
                  "Every accuracy in this figure belongs to exactly one of these settings")
    rows = [["Setting", "Train", "Test", "Used for", "Prediction scores from"],
            ["Within-set CV", "B1 (or B2) folds", "held-out B1 (or B2) fold", "Panel B diagonal", "5-fold CV, LightGBM"],
            ["Cross-set", "B1 → B2 or B2 → B1", "the other set's held-out patches", "Panel B off-diagonal", "5-fold CV"],
            ["Union", "B1 ∪ B2 (conflicts removed)", "held-out B1 and B2 patches reported separately", "Panel C (main)", "5-fold CV, n = 747 test patches"],
            ["Leave-one-image-out", "all images but one", "the held-out image", "Panel C robustness; claim check", "LOIO, CP and ilastik"]]
    table(s, rows, 0.35, 1.2, 12.6, col_w=[2.2, 2.6, 3.3, 2.3, 2.2], size=12, row_h=0.55)
    bullets(s, [
        "Test populations differ in DIFFICULTY, not just label noise: B2 patches are prototypes, so the same classifier "
        "scores far higher on them (see B→B2 column). Always compare columns of one test set, never across test sets.",
        "Accuracy = fraction of patches where thresholded prediction equals the annotator label; "
        "balanced accuracy (panel B) = mean of adhesion recall and no-adhesion recall.",
        "Score = LightGBM probability of 'adhesion'. 0.3/0.7 are the borderline boundaries used throughout.",
        "Annotators: B1 = Margaret (plus Annabel earlier); B2 = Annabel. Strategy, annotator, interface and date all differ "
        "between B1 and B2 — they cannot be separated with the current data.",
    ], 0.35, 4.3, 12.6, 3.0, size=14, head="Reading rules")

    # 3 — panel A table + composition
    s = new_slide(prs, "Panel A — B1 versus B2: what was annotated",
                  "B1 = broadly sampled (not purely random); B2 = selected prototypes")
    picture(s, pA1, 0.3, 1.05, width=8.3)
    rows = [["", "B1", "B2"],
            ["Patches (binary-usable)", f"{len(b1):,}", f"{len(b2):,}"],
            ["Adhesion / no adhesion", f"{int(b1.y.sum())} / {int((1 - b1.y).sum())}", f"{int(b2.y.sum())} / {int((1 - b2.y).sum())}"],
            ["Images (ctrl / ycomp)", f"{nimg(b1)} ({nimg(b1,'control')} / {nimg(b1,'ycomp')})",
             f"{nimg(b2)} ({nimg(b2,'control')} / {nimg(b2,'ycomp')})"],
            ["Patches / image (mean)", f"{b1.groupby('image').size().mean():.1f}", f"{b2.groupby('image').size().mean():.1f}"],
            ["Shared images / patches", f"{shared_imgs} / {shared_patch}", "(same)"]]
    table(s, rows, 8.75, 1.15, 4.3, col_w=[1.9, 1.2, 1.2], size=11, row_h=0.36)
    bullets(s, [
        "B1: broad sampling of cell regions, biased toward the cell front, plus error-focused additions "
        "(patches added where an early model was wrong). Many images, few patches per image. "
        "Annotator-recorded location (right plot): 40% lamella or protruding edge, 27% cell body, 33% periphery/other — "
        "a front bias is present but moderate.",
        "B2: annotator clicks clear prototypes of each class. Fewer images, many patches per image, "
        "class mix shifted toward clean no-adhesion / focal-adhesion examples.",
    ], 0.35, 4.75, 8.3, 2.5, size=13, head="Sampling rules (as stated in your notes)")
    bullets(s, [
        "To confirm: cell-front bias and error-focused additions are taken from your brief and the earlier fig3 "
        "draft; no protocol document in the repo states them (docs/ only has random- vs prototype-labelling protocols).",
        "Counts are DS1 only; 5-class labels collapsed to adhesion (NA, FC, FA, Fib) vs no adhesion; 'Uncertain' dropped from B1.",
    ], 8.75, 3.5, 4.3, 3.7, size=12, head="Caveats", head_col="#aa3333", fill="#fdeeee")

    # 4 — examples
    s = new_slide(prs, "Panel A — example patches per class",
                  "Random draws (seed 3) from each label set; border colour = batch")
    picture(s, pA2, 0.3, 1.05, width=9.3)
    bullets(s, [
        "Rows are the 5 classes (collapsed to adhesion vs no adhesion for the classifier).",
        "Use these as panel A inset images; re-draw by changing the seed in panel_a_examples().",
        "At this size the two sets are not visually distinguishable by eye; any difficulty gap is shown only by the classifier results (panel B).",
    ], 9.75, 1.2, 3.3, 5.5, size=12, head="How to read")

    # 5 — panel B cross heatmap
    cp_M, cp_S = mats["cp"]
    s = new_slide(prs, "Panel B — matched cross-evaluation: B1→B1, B1→B2, B2→B1, B2→B2",
                  "Matched label budget (~314 training patches per fold each), 5-fold CV, LightGBM")
    picture(s, pB1, 0.3, 1.1, width=7.6)
    bullets(s, [
        f"Rows = training labels, columns = test labels. The B2 column is high for both models "
        f"(B1→B2 {cp_M[0,1]:.2f}, B2→B2 {cp_M[1,1]:.2f}): B2 patches are easy.",
        f"On the B1 column — the broadly sampled, all-patch-like population — the B1-trained model wins "
        f"(B1→B1 {cp_M[0,0]:.2f} vs B2→B1 {cp_M[1,0]:.2f}). B2 training does not transfer to hard patches.",
        "So B2→B2 > B1→B1 reflects an easier test set, not better training data.",
        "Matching: each set has ~393 patches (≈314 train / 79 test per fold); same DS1 images and features.",
    ], 8.05, 1.1, 5.0, 3.9, size=13, head="What it shows")
    bullets(s, [
        "B1 and B2 sets share only 24 patches (not the same patches relabelled), so the two test columns are DIFFERENT patch populations. "
        "docs/le_benchmark_overview.md says 'exact same patches' — the data disagree.",
        "Annotator (Margaret vs Annabel) is confounded with sampling strategy.",
    ], 8.05, 5.1, 5.0, 2.2, size=12, head="Caveats", head_col="#aa3333", fill="#fdeeee")

    # 6 — panel B budget
    s = new_slide(prs, "Panel B — common test population: which training labels serve all patches?",
                  "Label-budget curves; models trained on B1, B2 or a 50:50 mix, tested separately on B1 and B2 test patches")
    picture(s, pB2, 0.3, 1.05, width=8.7)
    g = lambda f, te, st, b: bud[(f, te, st)].get(b, np.nan)
    bullets(s, [
        f"Left column (B1 test, all-patch-like): up to ~200 labels the three curves overlap within their SD bands. "
        f"At 300 labels CP reaches {g('cp','b1','b1_only',300):.2f} with B1 labels vs {g('cp','b1','b2v2_only',300):.2f} with B2 labels "
        f"(mix {g('cp','b1','mixed',300):.2f}); B2-trained stays flat.",
        f"Right column (B2 test, prototypes): B2-trained {g('cp','b2v2','b2v2_only',300):.2f} vs "
        f"B1-trained {g('cp','b2v2','b1_only',300):.2f} — a clear, large gap in B2's favour on its own population.",
        "Reading rule: compare the three lines within ONE panel. The left column answers the all-patch question.",
        "Supported: B2 labels help mainly on B2-like patches; B1 labels keep improving on hard patches with more labels. "
        "The B1-vs-B2 gap on the left is small at low budgets.",
    ], 9.1, 1.1, 4.0, 4.2, size=12, head="What it shows")
    bullets(s, [
        "A cleaner test is a single fixed test set with labels from one protocol; here each column uses its own annotator's labels.",
        "Curves cap at 300 labels (pool size of each set is ~314/fold).",
    ], 9.1, 5.4, 4.0, 1.85, size=11.5, head="Caveats", head_col="#aa3333", fill="#fdeeee")

    # 7 — panel C score dists
    z1, z2 = zs["union B1 test"], zs["union B2 test"]
    s = new_slide(prs, "Panel C — prediction scores by outcome (TP / TN / FP / FN)",
                  "Union-trained CP-LightGBM; held-out patches, B1 and B2 test populations shown separately")
    picture(s, pC1, 0.3, 1.05, width=8.4)
    rows = [["Population", "n", "Accuracy", "Cover. outside 0.3–0.7", "Acc. outside", "n inside", "Acc. inside"]]
    for k in ["union B1 test", "union B2 test", "union B1+B2 pooled"]:
        z = zs[k]
        lo, hi = wilson(z["k_mid"], z["n_mid"])
        rows.append([k, z["n"], f"{z['acc']:.1%}", f"{z['cov']:.1%}", f"{z['acc_out']:.1%}",
                     z["n_mid"], f"{z['acc_mid']:.0%} ({lo:.0%}–{hi:.0%})"])
    table(s, rows, 0.3, 4.6, 8.5, col_w=[1.9, 0.5, 1.0, 1.5, 1.2, 0.8, 1.6], size=10.5, row_h=0.42)
    bullets(s, [
        "Correct predictions (TP, TN) pile up near 0 and 1; errors (FP, FN) spread toward the middle — "
        "that is why extreme scores are more trustworthy.",
        "Grey band = borderline zone 0.3–0.7. It holds only "
        f"{z1['n_mid']} B1 and {z2['n_mid']} B2 test patches, so its accuracy estimate is very noisy (see CI).",
        "Many errors are still confident: FN/FP with scores near 0/1 exist, i.e. a high score does not guarantee correctness.",
    ], 9.0, 1.1, 4.1, 3.5, size=12, head="What it shows")
    bullets(s, [
        "Score = LightGBM probability, not calibrated for correctness.",
        "Don't pool B1 and B2 test rows into one claim: B2 is easier and has almost no borderline patches.",
    ], 9.0, 4.75, 4.1, 2.5, size=12, head="Caveats", head_col="#aa3333", fill="#fdeeee")

    # 8 — coverage + accuracy
    s = new_slide(prs, "Panel C — report coverage AND accuracy at the 0.3/0.7 boundaries",
                  "Accuracy alone is misleading: it is a trade-off with how many patches you keep")
    picture(s, pC2, 0.3, 1.05, width=8.3)
    rows = [["Population", "Retained (cov.)", "Accuracy retained", "Rejected", "Acc. rejected"]]
    for k in ["union B1 test", "union B2 test", "LOIO image-eff. (CP)"]:
        z = zs[k]
        rows.append([k, f"{z['cov']:.1%}", f"{z['acc_out']:.1%}", f"{z['n_mid']} ({1-z['cov']:.1%})",
                     f"{z['acc_mid']:.0%}"])
    table(s, rows, 0.3, 4.55, 8.3, col_w=[2.4, 1.4, 1.6, 1.5, 1.4], size=11, row_h=0.4)
    bullets(s, [
        "Left: how many patches fall in each score zone. Right: accuracy in each zone with 95% CI.",
        "Retaining only patches outside 0.3–0.7 raises accuracy modestly at a small coverage cost.",
        "The gain depends strongly on the evaluation population: the LOIO (image-held-out) setting rejects "
        f"{1-zs['LOIO image-eff. (CP)']['cov']:.0%} of patches, the 5-fold union setting only {1-zs['union B1+B2 pooled']['cov']:.0%}.",
    ], 8.8, 1.1, 4.3, 3.3, size=12, head="What it shows")
    bullets(s, [
        "State the boundary's coverage in the figure (e.g. 'retains 97%, accuracy 89%').",
        "Rejected-patch accuracy of ~50–60% means they are near coin-flips, but n is small.",
    ], 8.8, 4.55, 4.3, 2.6, size=12, head="Caveats", head_col="#aa3333", fill="#fdeeee")

    # 9 — accuracy vs retained
    s = new_slide(prs, "Panel C — accuracy versus retained fraction",
                  "Tightening the score boundary: accuracy climbs while coverage falls")
    picture(s, pC3, 0.3, 1.05, width=9.0)
    bullets(s, [
        "Left curve: sweep the margin t; keep patches with score ≤ t or ≥ 1−t. Dots mark the 0.3/0.7 boundary.",
        "Right: same sweep against t, solid = accuracy, dashed = coverage.",
        "B2 test curves sit above B1 test curves everywhere — population difficulty again.",
        "LOIO (held-out image) is the harsher, more realistic setting: lower accuracy and more rejected patches.",
    ], 9.4, 1.1, 3.7, 4.0, size=12, head="How to read")
    bullets(s, ["Curves are descriptive; no confidence bands (single prediction per patch)."],
            9.4, 5.4, 3.7, 1.2, size=11.5, head="Caveat", head_col="#aa3333", fill="#fdeeee")

    # 10 — examples of errors
    s = new_slide(prs, "Panel C — representative patches: confident-correct, confident-wrong, ambiguous",
                  "Union-trained CP-LightGBM; number above each patch = prediction score")
    picture(s, pC4, 0.3, 1.05, height=5.95)
    bullets(s, [
        "Rows: TP, TN (confidently correct), FP, FN (confidently wrong), and ambiguous (score 0.3–0.7).",
        "Confident errors are the limiting cases: they look like the opposite class or have disputed labels.",
        "Ambiguous patches may be label-ambiguous, model errors or genuinely mixed appearance — "
        "these images alone cannot tell which.",
    ], 8.6, 1.2, 4.5, 4.5, size=12, head="How to read")

    # 11 — Panel D
    s = new_slide(prs, "Panel D — human agreement versus classifier score",
                  "Does classifier uncertainty coincide with annotator disagreement?")
    picture(s, pD1, 0.3, 1.05, width=7.9)
    rows = [["Independent annotation set", "n", "Agree (binary)", "κ", "Agree (5-class)"]]
    for k, v in agr.items():
        lo, hi = wilson(v["k2"], v["n"])
        rows.append([k, v["n"], f"{v['agree2']:.0%} ({lo:.0%}–{hi:.0%})", f"{v['kappa2']:.2f}",
                     "—" if np.isnan(v["agree5"]) else f"{v['agree5']:.0%}"])
    table(s, rows, 0.3, 4.2, 7.9, col_w=[3.1, 0.5, 1.9, 0.6, 1.8], size=10.5, row_h=0.42)
    st = dstats["stats"]
    bullets(s, [
        f"Only {dstats['n']} DS1 patches carry labels from both annotators ({dstats['n_dis']} disagreements).",
        "Disagreement by score zone: " + "; ".join(f"{z} {k}/{n}" for z, n, k, *_ in st) +
        f".  Score-margin vs disagreement: Spearman ρ = {dstats['rho']:.2f} (p = {dstats['p']:.2f}).",
        "Preliminary signal: disagreement is rarer at extreme scores and the few borderline patches mostly disagree — "
        "but the 0.3–0.7 zone has only 3 patches, so this is suggestive, not a test of the claim.",
    ], 8.4, 1.1, 4.7, 3.2, size=12, head="What it shows")
    bullets(s, [
        "Association is NOT established: n = 53, 7 disagreements, 3 borderline patches, one score source (B1-trained). "
        "Needs a dedicated double-annotated set (≥ ~200 patches, enriched for 0.3–0.7).",
        "Overlap patches were not sampled to be representative, and the score predicts B1's label, not B2's.",
        "Score-coloured embedding (UMAP) can illustrate overlap but is not validation.",
        "Intermediate score ≠ biological continuum: it may be label ambiguity, model error or true mixed appearance.",
    ], 8.4, 4.4, 4.7, 2.85, size=11.5, head="Caveats", head_col="#aa3333", fill="#fdeeee")

    # 12 — verification
    loio_cp = zs["LOIO image-eff. (CP)"]; loio_both = zs["LOIO image-eff. (CP+ilastik pooled)"]
    ag = agr["DS1 B1 vs B2 (full sets)"]
    s = new_slide(prs, "Verification of the reported numbers",
                  "Each claim re-computed from the repository's saved predictions; population stated explicitly")
    rows = [["Reported claim", "Re-computed", "Population / source", "Verdict"],
            ["B1 accuracy ≈ 80%",
             f"{cross_acc[('b1','b1')]:.1%} (within-B1 CV); {z1['acc']:.1%} (union→B1 test)",
             "B1 held-out patches, CP-LGBM, 5-fold", "Confirmed"],
            ["Two-annotator agreement ≈ 80%",
             f"{ag['agree2']:.0%} binary (n={ag['n']}), {ag['agree5']:.0%} 5-class; κ={ag['kappa2']:.2f}",
             "DS1 patches in both B1 and B2", "Not reproduced: 87% binary, 70% 5-class; n small"],
            ["B2 accuracy can exceed 95%",
             f"{cross_acc[('b2v2','b2v2')]:.1%} (within-B2 CV); {z2['acc']:.1%} (union→B2 test)",
             "B2 held-out patches, CP-LGBM", "Confirmed (easier population)"],
            ["≈93% retained outside 0.3–0.7, ≈90% acc.; ≈50% acc. in remaining 7%",
             f"CP only: {loio_cp['cov']:.1%} / {loio_cp['acc_out']:.1%} / {loio_cp['acc_mid']:.0%}.  "
             f"CP+ilastik pooled: {loio_both['cov']:.1%} / {loio_both['acc_out']:.1%} / {loio_both['acc_mid']:.0%}",
             "LOIO, B1+B2 labels, CP-LGBM (image_eff_b1b2v2)", "Confirmed for LOIO only"],
            ["Same, in the union 5-fold setting",
             f"{zs['union B1+B2 pooled']['cov']:.1%} / {zs['union B1+B2 pooled']['acc_out']:.1%} / "
             f"{zs['union B1+B2 pooled']['acc_mid']:.0%} (n_mid={zs['union B1+B2 pooled']['n_mid']})",
             "5-fold union, n=747", "Does NOT match: 97.5% retained, only 19 borderline patches"]]
    table(s, rows, 0.3, 1.15, 12.7, col_w=[3.0, 4.3, 2.9, 2.5], size=11, row_h=0.8)
    bullets(s, [
        "The 93/90/50 numbers come from the leave-one-image-out run, not from the union model in the main panel C — label the population.",
        "'Two annotators agree ≈ 80%' is not reproduced: 87% for binary labels, 70% for 5-class (n = 53, B1 vs B2); Margaret vs herself: 100% binary, 83% 5-class (n = 29).",
    ], 0.3, 5.85, 12.7, 1.4, size=12.5, head="Discrepancies to resolve before writing the figure legend",
        head_col="#aa3333", fill="#fdeeee")

    # 13 — open tests + assembly
    s = new_slide(prs, "Open tests, caveats, and how to assemble the figure",
                  "What the figure can and cannot claim")
    bullets(s, [
        "Score–human-agreement association is not yet established (panel D, n too small).",
        "Scores are not calibrated probabilities of correctness; no calibration curve was computed.",
        "Borderline predictions may reflect label ambiguity, model error, or mixed appearance — not a biological continuum.",
        "Strategy vs annotator is confounded (B1 = Margaret, B2 = Annabel).",
        "B2 as cleaner supervision for representation learning (AE) remains optional and untested.",
    ], 0.3, 1.1, 6.3, 5.9, size=13, head="Open / not claimed", head_col="#aa3333", fill="#fdeeee")
    rows = [["Panel", "PNG to use (results/fig4_panels/)"],
            ["A", "A1_composition.png, A2_examples.png (+ table on slide 3)"],
            ["B", "B1_cross_heatmap.png, B2_budget_common_test.png"],
            ["C", "C1_score_distributions.png, C2_zone_coverage_accuracy.png, C3_acc_vs_retained.png, C4_examples.png"],
            ["D", "D1_agreement_vs_score.png (+ agreement table on slide 11)"]]
    table(s, rows, 6.9, 1.15, 6.15, col_w=[0.8, 5.35], size=12, row_h=0.62)
    bullets(s, [
        "Main-figure suggestion: A = table + A2; B = B1 heatmap + left column of B2 budget; "
        "C = C1 + C3 (mark 0.3/0.7 with coverage); D = D1 only if sample size grows, else move to Supplement.",
        "Re-run: python scripts/make_pptx_fig4_annotation_reliability.py (~1–2 min, needs patch HDF5).",
    ], 6.9, 4.5, 6.15, 2.6, size=12, head="Suggested layout")

    prs.save(OUT)
    print("saved", OUT)
    print({k: {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()} for k, v in zs.items()})
    print(agr)
    print(dstats["stats"], dstats["rho"], dstats["p"], dstats["n"], dstats["n_dis"])
    print(cross_acc)
    print("shared imgs/patches", shared_imgs, shared_patch)


if __name__ == "__main__":
    main()
