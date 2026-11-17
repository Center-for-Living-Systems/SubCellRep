#!/usr/bin/env python3
"""
add_lgbm_importance_slide.py

Append slide 11 to pred_confidence_image_eff_b1b2v2_cp_lgbm.pptx:

  Slide 11 — LightGBM built-in feature importance (gain)
             Side-by-side comparison with Spearman ρ ranking.
             Features appearing in both top-15 lists are highlighted.

Usage:
  python scripts/add_lgbm_importance_slide.py
"""
from __future__ import annotations
import io, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

REPO     = Path(__file__).resolve().parent.parent
PRED_DIR = REPO / "eval_results" / "patch_preds"
RES      = REPO / "results"
CP_CSV   = Path("/net/projects/CLS/lding/data/fa_data_analysis/ae_results/features/cellprofiler/ds1.csv")

PPT_PATH = RES / "pred_confidence_image_eff_b1b2v2_cp_lgbm.pptx"

TOP_N = 15   # features to show per ranking


def _compute_lgbm_importance() -> pd.Series:
    df = pd.read_csv(PRED_DIR / "image_eff_b1b2v2_lgbm.csv")
    df = df.rename(columns={"feature": "feat"})
    df = df[df["feat"] == "cp"].copy()
    agg = df.groupby("fn").agg(y_true=("y_true","first")).reset_index()
    feat_df = pd.read_csv(CP_CSV).set_index("filename")
    merged  = agg.set_index("fn").join(feat_df, how="inner").dropna()
    feat_cols = feat_df.columns.tolist()
    X = merged[feat_cols].values
    y = merged["y_true"].values
    sc  = StandardScaler().fit(X)
    clf = LGBMClassifier(
        n_estimators=200, num_leaves=31, learning_rate=0.05,
        min_child_samples=1, class_weight="balanced",
        n_jobs=4, random_state=42, verbose=-1,
    )
    clf.fit(sc.transform(X), y)
    return pd.Series(clf.feature_importances_, index=feat_cols, name="gain")


def _compute_spearman() -> pd.Series:
    df = pd.read_csv(PRED_DIR / "image_eff_b1b2v2_lgbm.csv")
    df = df.rename(columns={"feature": "feat"})
    df = df[df["feat"] == "cp"].copy()
    agg = (df.groupby("fn")
             .agg(y_true=("y_true","first"),
                  prob_adhesion=("prob_adhesion","median"))
             .reset_index())
    feat_df = pd.read_csv(CP_CSV).set_index("filename")
    merged  = agg.set_index("fn").join(feat_df, how="inner")
    feat_cols = feat_df.columns.tolist()
    rhos = {}
    for col in feat_cols:
        v = merged[col].dropna()
        p = merged.loc[v.index, "prob_adhesion"]
        r, _ = stats.spearmanr(v, p)
        rhos[col] = abs(r)
    return pd.Series(rhos, name="|rho|")


def _shorten(name: str) -> str:
    return (name
            .replace("inverse_difference_moment", "IDM")
            .replace("angular_second_moment",     "ASM")
            .replace("difference_variance",       "diff.var")
            .replace("difference_entropy",        "diff.ent")
            .replace("sum_variance",              "sum.var")
            .replace("sum_average",               "sum.avg")
            .replace("sum_entropy",               "sum.ent")
            .replace("info_meas1",                "IM-1")
            .replace("info_meas2",                "IM-2")
            .replace("correlation",               "corr")
            .replace("contrast",                  "contr")
            .replace("entropy",                   "ent")
            .replace("intensity_",                "int.")
            .replace("_d", " d"))


def _build_figure(gain: pd.Series, rho: pd.Series) -> io.BytesIO:
    top_gain = gain.sort_values(ascending=False).head(TOP_N)
    top_rho  = rho.sort_values(ascending=False).head(TOP_N)

    shared = set(top_gain.index) & set(top_rho.index)

    def _bar_colors(index, shared_set):
        return ["#e05020" if f in shared_set else "#aaaaaa" for f in index]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))

    # ── Left: LGBM gain ───────────────────────────────────────────────────────
    ax = axes[0]
    labels = [_shorten(f) for f in top_gain.index]
    colors = _bar_colors(top_gain.index, shared)
    bars = ax.barh(range(len(top_gain)), top_gain.values, color=colors, height=0.7)
    ax.set_yticks(range(len(top_gain)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Feature importance (gain)", fontsize=10)
    ax.set_title("LightGBM built-in importance\n(total split gain)", fontsize=11,
                 fontweight="bold", pad=8)
    ax.spines[["top","right"]].set_visible(False)
    # Add value labels
    for bar, val in zip(bars, top_gain.values):
        ax.text(val + top_gain.max()*0.01, bar.get_y() + bar.get_height()/2,
                f"{int(val)}", va="center", fontsize=7.5, color="#444444")

    # ── Right: |Spearman ρ| ───────────────────────────────────────────────────
    ax2 = axes[1]
    labels2 = [_shorten(f) for f in top_rho.index]
    colors2  = _bar_colors(top_rho.index, shared)
    bars2 = ax2.barh(range(len(top_rho)), top_rho.values, color=colors2, height=0.7)
    ax2.set_yticks(range(len(top_rho)))
    ax2.set_yticklabels(labels2, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("|Spearman ρ|  with P(adhesion)", fontsize=10)
    ax2.set_title("|Spearman ρ| ranking\n(monotonic correlation with probability)", fontsize=11,
                  fontweight="bold", pad=8)
    ax2.spines[["top","right"]].set_visible(False)
    for bar, val in zip(bars2, top_rho.values):
        ax2.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", fontsize=7.5, color="#444444")

    # Legend
    shared_patch = mpatches.Patch(color="#e05020",
                                   label=f"In both top-{TOP_N} lists ({len(shared)} features)")
    uniq_patch   = mpatches.Patch(color="#aaaaaa",
                                   label=f"Unique to this ranking")
    fig.legend(handles=[shared_patch, uniq_patch], fontsize=9,
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04),
               framealpha=0.9)

    # Bottom note
    fig.text(0.5, -0.10,
             "LGBM gain measures how much a feature reduces impurity across all decision-tree splits.\n"
             "Spearman ρ measures global monotonic correlation with P(adhesion).\n"
             "Features in both lists are robustly discriminating; features unique to one list "
             "capture complementary signal.",
             ha="center", va="top", fontsize=9, color="#444444",
             wrap=True)

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


def main():
    print(f"Loading {PPT_PATH.name} ...")
    prs = Presentation(str(PPT_PATH))
    print(f"  {len(prs.slides)} slides")

    print("Training LightGBM and extracting feature importance ...")
    gain = _compute_lgbm_importance()

    print("Computing Spearman ρ ...")
    rho = _compute_spearman()

    n_shared = len(set(gain.nlargest(TOP_N).index) & set(rho.nlargest(TOP_N).index))
    print(f"Features in both top-{TOP_N}: {n_shared}")

    print("Building slide ...")
    buf = _build_figure(gain, rho)

    # Append slide
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    for ph in list(slide.placeholders):
        ph.element.getparent().remove(ph.element)
    W, H = prs.slide_width, prs.slide_height

    # Title
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    txb = slide.shapes.add_textbox(Inches(0.2), Inches(0.06), W - Inches(0.4), Inches(0.46))
    tf  = txb.text_frame
    tf.text = "LightGBM feature importance vs Spearman ρ ranking"
    tf.paragraphs[0].runs[0].font.size = Pt(20)
    tf.paragraphs[0].runs[0].font.bold = True

    slide.shapes.add_picture(buf, Inches(0.2), Inches(0.56),
                              W - Inches(0.4), H - Inches(0.66))

    prs.save(str(PPT_PATH))
    sz = PPT_PATH.stat().st_size // 1024
    print(f"\nSaved → {PPT_PATH}  ({sz} KB, {len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
