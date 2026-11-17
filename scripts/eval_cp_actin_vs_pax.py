#!/usr/bin/env python3
"""
eval_cp_actin_vs_pax.py
========================
Evaluate ad vs no-ad classification using CellProfiler features from
  ch3 actin   (ds1_ch3_s32.csv)
  ch1 paxillin (cellprofiler/ds1.csv)

Design mirrors eval_le_b1b2v2_budget.py:
  - B1 matched set (fold_splits_b1.csv), b1_only strategy
  - 5-fold CV, test on held-out fold
  - LightGBM, budgets 20..340 + 0 (all), 10 repeats

Output:
  eval_results/b1b2v2/le_actin_cp_budget_lgbm.csv
  results/cp_actin_vs_pax_le.pptx

Usage:
  python scripts/eval_cp_actin_vs_pax.py
  python scripts/eval_cp_actin_vs_pax.py --skip-eval   # only remake PPT
"""
from __future__ import annotations

import argparse
import io
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA = Path("/net/projects/CLS/lding/data/fa_data_analysis")
REPO = Path(__file__).resolve().parents[1]

ACTIN_CSV = DATA / "ae_results/features/cp_scale/ds1_ch3_s32.csv"
PAX_CSV   = DATA / "ae_results/features/cellprofiler/ds1.csv"
FS_DIR    = DATA / "labelling/le_b1b2v2_matched"
EVAL_DIR  = REPO / "eval_results/b1b2v2"
OUT_CSV   = EVAL_DIR / "le_actin_cp_budget_lgbm.csv"
OUT_PPT   = REPO / "results/cp_actin_vs_pax_le.pptx"

# Also load the existing paxillin result for comparison
PAX_EXIST_CSV = EVAL_DIR / "le_b1b2v2_budget_cp_lgbm.csv"

FOLDS     = list(range(5))
BUDGETS   = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 300, 340, 0]
N_REPEATS = 10
SLIDE_W, SLIDE_H = 13.33, 7.5


# ── Helpers ───────────────────────────────────────────────────────────────────

def _h2u(uid: str) -> str:
    return uid.replace("-f", "_f", 1)


def _strat_sample(pool: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    groups = {lbl: grp.reset_index(drop=True) for lbl, grp in pool.groupby("label")}
    total  = len(pool)
    alloc  = {}
    for lbl, grp in groups.items():
        alloc[lbl] = max(1, int(round(n * len(grp) / total)))
        alloc[lbl] = min(alloc[lbl], len(grp))
    diff = n - sum(alloc.values())
    if diff != 0:
        largest = max(groups, key=lambda l: len(groups[l]))
        alloc[largest] = min(len(groups[largest]), max(1, alloc[largest] + diff))
    parts = []
    for lbl, grp in groups.items():
        k = alloc[lbl]
        idx = rng.choice(len(grp), size=k, replace=False)
        parts.append(grp.iloc[idx])
    return pd.concat(parts, ignore_index=True)


def _build_clf():
    return LGBMClassifier(
        n_estimators=200, num_leaves=31, learning_rate=0.05,
        min_child_samples=1, class_weight="balanced",
        n_jobs=2, random_state=42, verbose=-1,
    )


def _score(y_true, y_pred) -> dict:
    return dict(
        bal_acc     = round(balanced_accuracy_score(y_true, y_pred), 6),
        recall_ad   = round(recall_score(y_true, y_pred, pos_label=1, zero_division=0), 6),
        recall_noad = round(recall_score(y_true, y_pred, pos_label=0, zero_division=0), 6),
        prec_ad     = round(precision_score(y_true, y_pred, pos_label=1, zero_division=0), 6),
        prec_noad   = round(precision_score(y_true, y_pred, pos_label=0, zero_division=0), 6),
    )


# ── Evaluation ────────────────────────────────────────────────────────────────

def run_eval() -> pd.DataFrame:
    print("Loading features ...", flush=True)
    actin = pd.read_csv(ACTIN_CSV).set_index("filename")
    actin_feat_cols = [c for c in actin.columns
                       if c not in ("condition", "frame_idx", "canvas_cx", "canvas_cy")]
    actin = actin[actin_feat_cols]

    print("Loading fold splits ...", flush=True)
    fs = pd.read_csv(FS_DIR / "fold_splits_b1.csv")
    fs["fn"] = fs["unique_ID"].apply(_h2u)

    master_rng = np.random.default_rng(0)
    rows = []

    for fold in FOLDS:
        pool = fs[fs["fold"] != fold][["fn", "label"]].copy()
        test = fs[fs["fold"] == fold][["fn", "label"]].copy()
        print(f"  fold {fold}: pool={len(pool)} test={len(test)}", flush=True)

        for budget in BUDGETS:
            use_all = (budget == 0)
            reps    = 1 if use_all else N_REPEATS

            for rep in range(reps):
                rng = np.random.default_rng(master_rng.integers(0, 2**32))
                tr_df = pool.copy() if use_all else _strat_sample(pool, budget, rng)
                tr_df = tr_df[tr_df["fn"].isin(actin.index)]

                if len(tr_df) == 0:
                    continue

                X_tr = actin.reindex(tr_df["fn"]).values.astype(np.float64)
                y_tr = np.array([1 if l == "adhesion" else 0 for l in tr_df["label"]])
                ok_tr = ~np.isnan(X_tr).any(axis=1)
                X_tr, y_tr = X_tr[ok_tr], y_tr[ok_tr]

                if len(X_tr) == 0 or len(np.unique(y_tr)) < 2:
                    continue

                sc   = StandardScaler().fit(X_tr)
                X_tr = sc.transform(X_tr)
                clf  = _build_clf()
                clf.fit(X_tr, y_tr)

                te = test[~test["fn"].isin(set(tr_df["fn"]))].copy()
                te = te[te["fn"].isin(actin.index)]
                if len(te) == 0:
                    continue

                X_te = sc.transform(actin.reindex(te["fn"]).values.astype(np.float64))
                y_te = np.array([1 if l == "adhesion" else 0 for l in te["label"]])
                y_pred = clf.predict(X_te)

                row = dict(feat="actin_cp32", clf="lgbm",
                           strategy="b1_only", budget=budget,
                           fold=fold, repeat=rep, test_batch="b1",
                           n_train=int(ok_tr.sum()), n_train_ad=int(y_tr.sum()),
                           n_test=len(y_te))
                row.update(_score(y_te, y_pred))
                rows.append(row)

    df = pd.DataFrame(rows)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved → {OUT_CSV}  ({len(df)} rows)")
    return df


# ── PPT helpers ───────────────────────────────────────────────────────────────

def _hex2rgb(h: str) -> RGBColor:
    h = h.lstrip("#")
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _txt(slide, text, left, top, width, height,
         size=10, bold=False, color="#222222", align=PP_ALIGN.LEFT,
         italic=False, wrap=True):
    tb = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = _hex2rgb(color)
    return tb


def _rect(slide, left, top, width, height, fill_hex, line_hex=None, line_pt=None):
    shape = slide.shapes.add_shape(
        1, Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = _hex2rgb(fill_hex)
    if line_hex:
        shape.line.fill.solid()
        shape.line.fill.fore_color.rgb = _hex2rgb(line_hex)
        if line_pt:
            shape.line.width = Pt(line_pt)
    else:
        shape.line.fill.background()
    return shape


def _paste_pil(slide, img: Image.Image, left, top, width, height):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    slide.shapes.add_picture(buf, Inches(left), Inches(top),
                              Inches(width), Inches(height))


def _fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf).copy()


# ── Figures ───────────────────────────────────────────────────────────────────

C_PAX   = "#1f77b4"   # paxillin — blue
C_ACT   = "#d62728"   # actin — red


def _agg_budget(df, budgets_plot):
    """Aggregate over folds+repeats per budget → (mean, std, n_train_mean)."""
    rows = []
    for b in budgets_plot:
        sub = df[df["budget"] == b]
        if len(sub) == 0:
            continue
        rows.append(dict(
            budget=b,
            mean=sub["bal_acc"].mean(),
            std=sub["bal_acc"].std(ddof=0),
            n_train=sub["n_train"].mean(),
        ))
    return pd.DataFrame(rows)


def fig_le_curves(df_act: pd.DataFrame, df_pax: pd.DataFrame) -> Image.Image:
    """Label efficiency curves: actin vs paxillin."""
    BUDGETS_PLOT = [20, 40, 60, 80, 100, 120, 140, 160, 200, 260, 340]

    agg_act = _agg_budget(df_act, BUDGETS_PLOT)
    agg_pax = _agg_budget(df_pax, BUDGETS_PLOT)

    # "all" point
    def _all_pt(df):
        sub = df[df["budget"] == 0]
        if len(sub) == 0:
            return None, None, None
        return sub["n_train"].mean(), sub["bal_acc"].mean(), sub["bal_acc"].std(ddof=0)

    act_all = _all_pt(df_act)
    pax_all = _all_pt(df_pax)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    fig.patch.set_facecolor("white")

    ax.errorbar(agg_act["n_train"], agg_act["mean"], yerr=agg_act["std"],
                color=C_ACT, marker="^", markersize=5, linewidth=1.8,
                capsize=3, label="CP ch3 actin")
    ax.errorbar(agg_pax["n_train"], agg_pax["mean"], yerr=agg_pax["std"],
                color=C_PAX, marker="o", markersize=5, linewidth=1.8,
                capsize=3, label="CP ch1 paxillin")

    # "all" open markers
    for (n, m, s), color, mk in [(act_all, C_ACT, "^"), (pax_all, C_PAX, "o")]:
        if n is None:
            continue
        ax.plot(n, m, marker=mk, markersize=8, color=color,
                markerfacecolor="white", markeredgewidth=2.0, zorder=5)

    ax.set_xlabel("Training labels (patches)", fontsize=10)
    ax.set_ylabel("Balanced accuracy", fontsize=10)
    ax.set_ylim(0.45, 1.02)
    ax.set_xlim(left=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.axhline(0.5, color="#cccccc", linewidth=0.6, linestyle=":")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.set_title(
        "Ad vs no-ad label efficiency  —  CP features, LightGBM\n"
        "B1 matched set, b1-only strategy, 5-fold CV, 10 repeats",
        fontsize=9, color="#444444", pad=5)

    fig.tight_layout()
    return _fig_to_pil(fig)


def fig_final_bar(df_act: pd.DataFrame, df_pax: pd.DataFrame) -> Image.Image:
    """Bar chart: final bal_acc at budget=340 and all."""
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    fig.patch.set_facecolor("white")

    labels = ["budget=340", "all labels"]
    width  = 0.32
    x      = np.arange(len(labels))

    for i, (bud, lab) in enumerate([(340, "budget=340"), (0, "all labels")]):
        for j, (df, name, color) in enumerate([(df_pax, "paxillin", C_PAX),
                                                (df_act, "actin",    C_ACT)]):
            sub = df[df["budget"] == bud]["bal_acc"]
            m, s = sub.mean(), sub.std(ddof=0)
            xpos = i + (j - 0.5) * width
            ax.bar(xpos, m, width=width * 0.9, color=color, alpha=0.85,
                   label=name if i == 0 else None)
            ax.errorbar(xpos, m, yerr=s, color="black", capsize=3,
                        linewidth=1.2, fmt="none")
            ax.text(xpos, m + s + 0.005, f"{m:.3f}", ha="center",
                    va="bottom", fontsize=7.5, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Balanced accuracy", fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9, frameon=False)
    ax.set_title("Final performance\n(balanced accuracy)", fontsize=9, color="#444444")
    fig.tight_layout()
    return _fig_to_pil(fig)


# ── PPT builder ───────────────────────────────────────────────────────────────

def build_ppt(df_act: pd.DataFrame, df_pax: pd.DataFrame):
    img_le  = fig_le_curves(df_act, df_pax)
    img_bar = fig_final_bar(df_act, df_pax)

    prs = Presentation()
    prs.slide_width  = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _rect(slide, 0, 0, SLIDE_W, SLIDE_H, "#ffffff")
    _txt(slide,
         "Ad vs No-ad classification — CP ch3 actin vs ch1 paxillin  "
         "(DS1, PS32, B1 matched set, LightGBM)",
         0.25, 0.06, SLIDE_W - 0.50, 0.36,
         size=13, bold=True, color="#1a2a3a")
    _rect(slide, 0.25, 0.43, SLIDE_W - 0.50, 0.016, "#cccccc")

    BODY_TOP = 0.50
    curve_w  = SLIDE_W * 0.60
    bar_w    = SLIDE_W - curve_w - 0.55

    _paste_pil(slide, img_le,  0.25, BODY_TOP, curve_w, SLIDE_H - BODY_TOP - 0.15)
    _paste_pil(slide, img_bar, 0.25 + curve_w + 0.05, BODY_TOP, bar_w,
               SLIDE_H - BODY_TOP - 0.15)

    # Summary stats table (text)
    for df, name, color in [(df_pax, "Paxillin ch1", C_PAX),
                             (df_act, "Actin ch3",   C_ACT)]:
        pass  # values are in the bar chart

    prs.save(OUT_PPT)
    print(f"Saved → {OUT_PPT}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-eval", action="store_true",
                    help="Skip evaluation; load existing CSV and just make PPT")
    args = ap.parse_args()

    if args.skip_eval and OUT_CSV.exists():
        print(f"Loading existing results from {OUT_CSV}")
        df_act = pd.read_csv(OUT_CSV)
    else:
        df_act = run_eval()

    # Load paxillin results for comparison (b1_only strategy, test_batch=b1)
    df_pax = pd.read_csv(PAX_EXIST_CSV)
    df_pax = df_pax[(df_pax["strategy"] == "b1_only") &
                    (df_pax["test_batch"] == "b1")].copy()
    print(f"Paxillin comparison: {len(df_pax)} rows loaded")

    build_ppt(df_act, df_pax)


if __name__ == "__main__":
    main()
