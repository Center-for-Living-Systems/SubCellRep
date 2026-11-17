#!/usr/bin/env python3
"""
make_pptx_pred_confidence.py

Visualize classifier prediction confidence from patch-level prediction CSVs.

Slides:
  1. Title
  2. Probability distribution by outcome group (TP/FP/TN/FN), 2×2 panel (feat × clf)
  3. Patch montage — Group 1: TP, high prob (confident correct adhesion)
  4. Patch montage — Group 2: FP, high prob (confidently wrong, predicted adhesion)
  5. Patch montage — Group 3: TN, low prob  (confident correct no-adhesion)
  6. Patch montage — Group 4: FN, low prob  (missed adhesions, low confidence)
  7. Feature correlation with prob_adhesion (top-20 by |Spearman ρ|)
  8. Top-6 features: violin plots split by TP/FP/TN/FN group

Default experiment: LE cross, train=b1, test=b2v2 (cross-annotator transfer)

Usage:
  python scripts/make_pptx_pred_confidence.py [--source le_cross] [--train_batch b1]
      [--test_batch b2v2] [--feat cp] [--clf lgbm] [--n_patches 12]
"""
from __future__ import annotations
import argparse, io, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

REPO     = Path(__file__).resolve().parent.parent
PRED_DIR = REPO / "eval_results" / "patch_preds"
OUT_DIR  = REPO / "results"

PATCH_ROOT = Path("/net/projects/CLS/lding/data/fa_data_analysis/ae_results/patches/cio/vinc")
CP_CSV     = Path("/net/projects/CLS/lding/data/fa_data_analysis/ae_results/features/cellprofiler/ds1.csv")
IL_CSV     = Path("/net/projects/CLS/lding/data/fa_data_analysis/ae_results/features/ilastik/ds1.csv")

GROUP_COLORS = {
    "TP": "#2166ac",   # blue
    "FP": "#d6604d",   # red-orange
    "TN": "#92c5de",   # light blue
    "FN": "#f4a582",   # light orange
}
GROUP_ORDER = ["TP", "FP", "TN", "FN"]
GROUP_LABELS = {
    "TP": "TP — High prob, correct (adhesion)",
    "FP": "FP — High prob, wrong (false adhesion)",
    "TN": "TN — Low prob, correct (no-adhesion)",
    "FN": "FN — Low prob, wrong (missed adhesion)",
}


# ── helpers ────────────────────────────────────────────────────────────────────

def _fig_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


def _add_img_slide(prs, buf, title_text, layout_idx=6):
    from pptx.util import Inches, Pt
    slide = prs.slides.add_slide(prs.slide_layouts[layout_idx])
    for ph in slide.placeholders:
        ph.element.getparent().remove(ph.element)

    from pptx.util import Emu
    W, H = prs.slide_width, prs.slide_height

    # Title bar
    txBox = slide.shapes.add_textbox(Inches(0.3), Inches(0.1), W - Inches(0.6), Inches(0.5))
    tf = txBox.text_frame
    tf.text = title_text
    tf.paragraphs[0].runs[0].font.size = Pt(20)
    tf.paragraphs[0].runs[0].font.bold = True

    # Image
    margin = Inches(0.3)
    top = Inches(0.65)
    slide.shapes.add_picture(buf, margin, top, W - 2*margin, H - top - Inches(0.1))
    return slide


def _load_pred_csv(source: str, feat: str = None, clf: str = None) -> pd.DataFrame:
    """Load pred CSV(s) for the given source, normalizing column names.

    le_* CSVs: one file per feat×clf → {source}_{feat}_{clf}.csv, column 'feat'
    image_eff_* CSVs: one file per clf (both feats inside) → {source}_{clf}.csv, column 'feature'
    """
    # Try per-feat-clf naming first (le_* sources)
    if feat is not None and clf is not None:
        path = PRED_DIR / f"{source}_{feat}_{clf}.csv"
        if path.exists():
            df = pd.read_csv(path)
            if "feature" in df.columns and "feat" not in df.columns:
                df = df.rename(columns={"feature": "feat"})
            return df

    # Try per-clf naming (image_eff_* sources): {source}_{clf}.csv
    if clf is not None:
        path = PRED_DIR / f"{source}_{clf}.csv"
        if path.exists():
            df = pd.read_csv(path)
            if "feature" in df.columns and "feat" not in df.columns:
                df = df.rename(columns={"feature": "feat"})
            if feat is not None:
                df = df[df["feat"] == feat]
            return df

    # Wildcard load for all variants
    files = sorted(PRED_DIR.glob(f"{source}_*.csv"))
    if not files:
        raise FileNotFoundError(f"No pred CSVs matching {PRED_DIR}/{source}_*.csv")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        if "feature" in df.columns and "feat" not in df.columns:
            df = df.rename(columns={"feature": "feat"})
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def _assign_group(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["group"] = "TN"
    df.loc[(df["y_true"] == 1) & (df["y_pred"] == 1), "group"] = "TP"
    df.loc[(df["y_true"] == 0) & (df["y_pred"] == 1), "group"] = "FP"
    df.loc[(df["y_true"] == 1) & (df["y_pred"] == 0), "group"] = "FN"
    return df


def _dedup_by_median_prob(df: pd.DataFrame) -> pd.DataFrame:
    """When a patch appears across multiple folds/repeats, keep median prob and modal group."""
    agg = (df.groupby("fn")
             .agg(y_true=("y_true","first"),
                  prob_adhesion=("prob_adhesion","median"),
                  y_pred=("y_pred", lambda x: x.mode()[0]))
             .reset_index())
    agg = _assign_group(agg)
    return agg


def _load_patch(fn: str) -> np.ndarray | None:
    cond = fn.split("_")[0]  # "control" or "ycomp"
    path = PATCH_ROOT / cond / "tiff_patches32_mr10" / fn
    if not path.exists():
        return None
    try:
        import tifffile
        img = tifffile.imread(str(path)).astype(np.float32)
    except Exception:
        try:
            from PIL import Image
            img = np.array(Image.open(str(path))).astype(np.float32)
        except Exception:
            return None
    if img.ndim == 3:
        img = img[0]  # take first channel
    lo, hi = np.percentile(img, 1), np.percentile(img, 99)
    if hi > lo:
        img = np.clip((img - lo) / (hi - lo), 0, 1)
    return img


# ── slide 2: probability distribution ─────────────────────────────────────────

def _fig_prob_dist(df_all: pd.DataFrame) -> io.BytesIO:
    feats = sorted(df_all["feat"].unique())
    clfs  = sorted(df_all["clf"].unique())
    nrow, ncol = len(feats), max(len(clfs), 1)

    fig, axes = plt.subplots(nrow, ncol, figsize=(5*ncol, 4*nrow), squeeze=False)

    for ri, feat in enumerate(feats):
        for ci, clf in enumerate(clfs):
            ax = axes[ri][ci]
            sub = df_all[(df_all["feat"] == feat) & (df_all["clf"] == clf)]
            if sub.empty:
                ax.set_visible(False)
                continue
            sub = _dedup_by_median_prob(sub)
            sub = _assign_group(sub)
            from scipy.stats import gaussian_kde
            for grp in GROUP_ORDER:
                vals = sub.loc[sub["group"] == grp, "prob_adhesion"].values
                if len(vals) < 5:
                    continue
                xs = np.linspace(0, 1, 200)
                try:
                    kde = gaussian_kde(vals, bw_method=0.15)
                    ys  = kde(xs)
                except Exception:
                    continue
                ax.plot(xs, ys, color=GROUP_COLORS[grp], lw=2,
                        label=f"{grp} (n={len(vals)})")
                ax.fill_between(xs, ys, alpha=0.15, color=GROUP_COLORS[grp])
            feat_lbl = "CellProfiler" if feat == "cp" else "ilastik"
            clf_lbl  = "LightGBM" if clf == "lgbm" else "LogReg"
            ax.set_title(f"{feat_lbl} / {clf_lbl}", fontsize=13, fontweight="bold")
            ax.set_xlabel("P(adhesion)")
            ax.set_ylabel("Density")
            ax.set_xlim(0, 1)
            ax.legend(fontsize=8, framealpha=0.7)
            ax.spines[["top","right"]].set_visible(False)

    fig.suptitle("Prediction probability distribution by outcome group", fontsize=15, y=1.01)
    fig.tight_layout()
    return _fig_to_img(fig)


# ── slides 3–6: patch montages ─────────────────────────────────────────────────

def _fig_montage(patches_df: pd.DataFrame, group: str, n_patches: int,
                 feat: str, clf: str) -> io.BytesIO:
    """Show top/bottom patches by probability for the given group."""
    sub = patches_df[patches_df["group"] == group].copy()
    if group in ("TP", "FP"):
        sub = sub.nlargest(n_patches * 3, "prob_adhesion")  # oversample, skip missing
    else:
        sub = sub.nsmallest(n_patches * 3, "prob_adhesion")

    imgs, labels = [], []
    for _, row in sub.iterrows():
        img = _load_patch(row["fn"])
        if img is None:
            continue
        imgs.append(img)
        labels.append(f"p={row['prob_adhesion']:.2f}")
        if len(imgs) >= n_patches:
            break

    if not imgs:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.text(0.5, 0.5, "No patches found", ha="center", va="center")
        ax.set_axis_off()
        return _fig_to_img(fig)

    ncols = min(6, len(imgs))
    nrows = int(np.ceil(len(imgs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.6, nrows * 1.8))
    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)

    for idx, (img, lbl) in enumerate(zip(imgs, labels)):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(lbl, fontsize=7, pad=2)
        ax.set_axis_off()
        for spine in ax.spines.values():
            spine.set_visible(False)

    for idx in range(len(imgs), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_axis_off()

    feat_lbl = "CellProfiler" if feat == "cp" else "ilastik"
    clf_lbl  = "LightGBM" if clf == "lgbm" else "LogReg"
    fig.suptitle(f"{GROUP_LABELS[group]}  |  {feat_lbl} / {clf_lbl}", fontsize=12, y=1.01)
    fig.tight_layout(pad=0.3)
    return _fig_to_img(fig)


# ── slide 7: feature correlation ───────────────────────────────────────────────

def _fig_feature_corr(patches_df: pd.DataFrame, feat: str, clf: str,
                      top_k: int = 20) -> io.BytesIO:
    feat_csv = CP_CSV if feat == "cp" else IL_CSV
    df_feat  = pd.read_csv(feat_csv).set_index("filename")
    # patches_df is already de-duped and filtered to this feat/clf
    merged   = patches_df.set_index("fn").join(df_feat, how="inner")
    feat_cols = df_feat.columns.tolist()

    rhos, pvals = [], []
    for col in feat_cols:
        vals = merged[col].values
        prob = merged["prob_adhesion"].values
        ok   = np.isfinite(vals) & np.isfinite(prob)
        if ok.sum() < 10:
            rhos.append(np.nan); pvals.append(np.nan)
            continue
        r, p = stats.spearmanr(vals[ok], prob[ok])
        rhos.append(r); pvals.append(p)

    corr_df = pd.DataFrame({"feature": feat_cols, "rho": rhos, "pval": pvals})
    corr_df = corr_df.dropna().sort_values("rho", key=abs, ascending=False).head(top_k)

    colors = ["#d6604d" if r > 0 else "#2166ac" for r in corr_df["rho"]]

    fig, ax = plt.subplots(figsize=(8, max(4, top_k * 0.35)))
    bars = ax.barh(range(len(corr_df)), corr_df["rho"].values, color=colors, height=0.7)
    ax.set_yticks(range(len(corr_df)))
    ax.set_yticklabels(corr_df["feature"].values, fontsize=9)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Spearman ρ  (with prob_adhesion)", fontsize=10)
    ax.invert_yaxis()
    ax.spines[["top","right"]].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#d6604d", label="positive ρ"),
                        Patch(color="#2166ac", label="negative ρ")],
              fontsize=9, loc="lower right")

    feat_lbl = "CellProfiler" if feat == "cp" else "ilastik"
    clf_lbl  = "LightGBM" if clf == "lgbm" else "LogReg"
    ax.set_title(f"Top-{top_k} features by |Spearman ρ|  |  {feat_lbl} / {clf_lbl}", fontsize=12)
    fig.tight_layout()
    return _fig_to_img(fig)


# ── slide 8: 4-group feature violin ───────────────────────────────────────────

def _fig_feature_violin(patches_df: pd.DataFrame, feat: str, clf: str,
                        top_k: int = 6) -> io.BytesIO:
    feat_csv = CP_CSV if feat == "cp" else IL_CSV
    df_feat  = pd.read_csv(feat_csv).set_index("filename")
    # patches_df is already de-duped and filtered to this feat/clf
    merged   = patches_df.set_index("fn").join(df_feat, how="inner")
    feat_cols = df_feat.columns.tolist()

    # Pick top-k features by ANOVA F-stat across 4 groups
    groups_data = {g: merged[merged["group"] == g] for g in GROUP_ORDER}
    f_stats = {}
    for col in feat_cols:
        vals = [g[col].dropna().values for g in groups_data.values()]
        vals = [v for v in vals if len(v) >= 3]
        if len(vals) < 2:
            continue
        try:
            f, _ = stats.f_oneway(*vals)
            f_stats[col] = f if np.isfinite(f) else 0.0
        except Exception:
            f_stats[col] = 0.0
    top_feats = sorted(f_stats, key=f_stats.get, reverse=True)[:top_k]

    ncols = 3
    nrows = int(np.ceil(len(top_feats) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5), squeeze=False)

    for idx, fcol in enumerate(top_feats):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        parts = ax.violinplot(
            [groups_data[g][fcol].dropna().values for g in GROUP_ORDER],
            positions=range(len(GROUP_ORDER)),
            showmedians=True, showextrema=False,
        )
        for pi, g in enumerate(GROUP_ORDER):
            parts["bodies"][pi].set_facecolor(GROUP_COLORS[g])
            parts["bodies"][pi].set_alpha(0.7)
        parts["cmedians"].set_color("k")
        ax.set_xticks(range(len(GROUP_ORDER)))
        ax.set_xticklabels(GROUP_ORDER, fontsize=9)
        ax.set_title(fcol, fontsize=9, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)

    for idx in range(len(top_feats), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_axis_off()

    feat_lbl = "CellProfiler" if feat == "cp" else "ilastik"
    clf_lbl  = "LightGBM" if clf == "lgbm" else "LogReg"
    fig.suptitle(f"Top-{top_k} discriminating features by ANOVA  |  {feat_lbl} / {clf_lbl}",
                 fontsize=13, y=1.01)
    fig.tight_layout(pad=1.0)
    return _fig_to_img(fig)


# ── main ───────────────────────────────────────────────────────────────────────

def build(args):
    from pptx import Presentation
    from pptx.util import Inches, Pt, Emu
    from pptx.dml.color import RGBColor

    # ── load & filter ─────────────────────────────────────────────────────────
    feat = args.feat
    clf  = args.clf

    df = _load_pred_csv(args.source, feat=feat, clf=clf)

    # For le_cross: filter to requested train/test batch
    if "train_batch" in df.columns and args.train_batch:
        df = df[df["train_batch"] == args.train_batch]
    if "test_batch" in df.columns and args.test_batch:
        df = df[df["test_batch"] == args.test_batch]
    # Ensure feat column exists (image_eff adds it via rename)
    if "feat" not in df.columns:
        df["feat"] = feat
    if "clf" not in df.columns:
        df["clf"] = clf

    print(f"Loaded {len(df)} rows  (feat={feat}, clf={clf})")

    # De-duplicate across folds/images — one row per patch
    df_dedup = _dedup_by_median_prob(df)
    df_dedup = _assign_group(df_dedup)
    print("Group counts (de-duped):")
    print(df_dedup["group"].value_counts().to_string())

    # Load all feats/clfs for the distribution slide (shows 2×2)
    try:
        df_all = _load_pred_csv(args.source, feat=None, clf=None)
        if "train_batch" in df_all.columns and args.train_batch:
            df_all = df_all[df_all["train_batch"] == args.train_batch]
        if "test_batch" in df_all.columns and args.test_batch:
            df_all = df_all[df_all["test_batch"] == args.test_batch]
        if "feat" not in df_all.columns:
            df_all["feat"] = df_all.get("feature", feat)
    except Exception:
        df_all = df.copy()

    # ── build PPT ──────────────────────────────────────────────────────────────
    prs = Presentation()
    prs.slide_width  = Inches(13.33)
    prs.slide_height = Inches(7.5)

    def blank_slide():
        return prs.slides.add_slide(prs.slide_layouts[6])

    # ── Slide 1: Title ────────────────────────────────────────────────────────
    sl = blank_slide()
    from pptx.util import Inches, Pt
    txb = sl.shapes.add_textbox(Inches(1), Inches(2.5), Inches(11), Inches(2))
    tf  = txb.text_frame
    tf.word_wrap = True
    p   = tf.paragraphs[0]
    p.text = "Prediction Confidence Analysis"
    p.runs[0].font.size = Pt(36)
    p.runs[0].font.bold = True
    p.alignment = 2  # PP_ALIGN.CENTER

    p2 = tf.add_paragraph()
    src_lbl = args.source.replace("_", " ").title()
    feat_lbl = "CellProfiler" if feat == "cp" else "ilastik"
    clf_lbl  = "LightGBM" if clf == "lgbm" else "LogReg"
    batch_info = ""
    if args.train_batch and args.test_batch:
        batch_info = f"  |  train={args.train_batch} → test={args.test_batch}"
    p2.text = f"{src_lbl}{batch_info}  |  {feat_lbl} / {clf_lbl}"
    p2.runs[0].font.size = Pt(20)
    p2.alignment = 2

    # ── Slide 2: Probability distribution ────────────────────────────────────
    print("Building slide 2: probability distribution...")
    buf = _fig_prob_dist(df_all)
    _add_img_slide(prs, buf, "Probability distribution by outcome group (TP/FP/TN/FN)")

    # ── Slides 3–6: patch montages ────────────────────────────────────────────
    for grp in GROUP_ORDER:
        short = {"TP":"TP high-prob (correct adhesion)", "FP":"FP high-prob (false adhesion)",
                 "TN":"TN low-prob (correct no-adhesion)", "FN":"FN low-prob (missed adhesion)"}[grp]
        print(f"Building montage slide: {grp}...")
        buf = _fig_montage(df_dedup, grp, args.n_patches, feat, clf)
        _add_img_slide(prs, buf, f"Example patches — {short}")

    # ── Slide 7: feature correlation ──────────────────────────────────────────
    print("Building slide 7: feature correlation...")
    buf = _fig_feature_corr(df_dedup, feat, clf, top_k=20)
    feat_lbl = "CellProfiler" if feat == "cp" else "ilastik"
    _add_img_slide(prs, buf, f"Top-20 {feat_lbl} features correlated with P(adhesion)")

    # ── Slide 8: group violin ─────────────────────────────────────────────────
    print("Building slide 8: feature violin plots...")
    buf = _fig_feature_violin(df_dedup, feat, clf, top_k=6)
    _add_img_slide(prs, buf, f"Top-6 discriminating features across TP/FP/TN/FN groups")

    # ── Save ──────────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    src_tag = args.source
    if args.train_batch and args.test_batch:
        src_tag += f"_{args.train_batch}_to_{args.test_batch}"
    out = OUT_DIR / f"pred_confidence_{src_tag}_{feat}_{clf}.pptx"
    prs.save(str(out))
    print(f"\nSaved → {out}  ({out.stat().st_size//1024} KB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",      default="le_cross",
                    help="Pred CSV prefix: le_cross, le_union, le_budget, image_eff_b1_full, image_eff_b1b2v2")
    ap.add_argument("--train_batch", default="b1",
                    help="Filter train_batch (for le_cross; empty = no filter)")
    ap.add_argument("--test_batch",  default="b2v2",
                    help="Filter test_batch (for le_cross; empty = no filter)")
    ap.add_argument("--feat",   default="cp",   choices=["cp","ilastik"])
    ap.add_argument("--clf",    default="lgbm", choices=["lgbm","logreg"])
    ap.add_argument("--n_patches", type=int, default=12)
    args = ap.parse_args()
    build(args)


if __name__ == "__main__":
    main()
