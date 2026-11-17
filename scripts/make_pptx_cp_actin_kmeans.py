#!/usr/bin/env python3
"""
make_pptx_cp_actin_kmeans.py
==============================
Actin CP features (DS1, PS32, ch3) — UMAP / PCA / KMeans analysis.

Steps
-----
1. Load ds1_ch3_s32.csv  (27637 patches × 50 features, all DS1 patches)
2. Join B1 FA labels (1339/1340 patches matched)
3. Standardise features → UMAP (2D) + PCA (2D)
4. KMeans k=10 on all 27637 patches
5. Find 5 patches closest to each cluster centre (in scaled feature space)
6. Extract 32×32 actin crops for example patches from label H5 images/act

PPT slides
----------
  Slide 1 : UMAP (left) | PCA (right) — coloured by FA class
             Labeled patches only; unlabeled shown faintly in the background.
  Slide 2 : UMAP (left) | PCA (right) — coloured by KMeans cluster
             Example patches highlighted with ring markers.
  Slide 3 : 10-cluster patch gallery
             10 rows × 5 patches, sorted by distance to cluster centre.

Output: results/cp_actin_kmeans.pptx
"""
from __future__ import annotations

import io
import warnings
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgb
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT = Path("/net/projects/CLS/lding/data/fa_data_analysis")
REPO      = Path(__file__).resolve().parents[1]

FEAT_CSV  = DATA_ROOT / "ae_results/features/cp_scale/ds1_ch3_s32.csv"
LAB_CSV   = DATA_ROOT / "labelling/labels_vinc_20260521.csv"
H5_CTRL   = DATA_ROOT / "ae_results/patches/cio_rb/vinc/vinc_control_label.h5"
H5_YCOMP  = DATA_ROOT / "ae_results/patches/cio_rb/vinc/vinc_ycomp_label.h5"
OUT       = REPO / "results/cp_actin_kmeans.pptx"

SLIDE_W, SLIDE_H = 13.33, 7.5

# ── Constants ─────────────────────────────────────────────────────────────────
N_CLUSTERS   = 10
N_EXAMPLES   = 5        # patches per cluster to display
PS           = 32       # patch size
PAD          = 64       # image padding used during extraction
RANDOM_STATE = 42

FA_ORDER = [
    "No adhesion",
    "Nascent Adhesion",
    "focal complex",
    "focal adhesion",
    "fibrillar adhesion",
    "Uncertain",
]
FA_COLORS = {
    "No adhesion":      "#888888",
    "Nascent Adhesion": "#1f77b4",
    "focal complex":    "#ff7f0e",
    "focal adhesion":   "#2ca02c",
    "fibrillar adhesion": "#d62728",
    "Uncertain":        "#bbbbbb",
    "_unlabeled":       "#e8e8e8",
}
CLUSTER_CMAP = "tab10"

FEAT_COLS = [
    "intensity_mean", "intensity_std", "intensity_median",
    "intensity_min",  "intensity_max", "intensity_integrated",
    "intensity_mad",  "intensity_p10", "intensity_p25",
    "intensity_p75",  "intensity_p90",
    "angular_second_moment_d1", "contrast_d1", "correlation_d1",
    "variance_d1", "inverse_difference_moment_d1", "sum_average_d1",
    "sum_variance_d1", "sum_entropy_d1", "entropy_d1",
    "difference_variance_d1", "difference_entropy_d1",
    "info_meas1_d1", "info_meas2_d1",
    "angular_second_moment_d2", "contrast_d2", "correlation_d2",
    "variance_d2", "inverse_difference_moment_d2", "sum_average_d2",
    "sum_variance_d2", "sum_entropy_d2", "entropy_d2",
    "difference_variance_d2", "difference_entropy_d2",
    "info_meas1_d2", "info_meas2_d2",
    "angular_second_moment_d4", "contrast_d4", "correlation_d4",
    "variance_d4", "inverse_difference_moment_d4", "sum_average_d4",
    "sum_variance_d4", "sum_entropy_d4", "entropy_d4",
    "difference_variance_d4", "difference_entropy_d4",
    "info_meas1_d4", "info_meas2_d4",
]

# ── PPTX helpers ──────────────────────────────────────────────────────────────

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


def _panel_label(slide, letter, left, top, size=14):
    _txt(slide, letter, left, top, 0.28, 0.32,
         size=size, bold=True, color="#1a2a3a")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Returns (df, X_scaled, feat_cols_present).
    df has: filename, condition, frame_idx, canvas_cx, canvas_cy,
            fa_label (str or '_unlabeled'), is_labeled (bool), cluster (int)
    """
    print("Loading features ...", flush=True)
    df = pd.read_csv(FEAT_CSV)
    df = df.rename(columns={"condition": "condition"})

    print("Loading labels ...", flush=True)
    lab = pd.read_csv(LAB_CSV)
    lab["feat_key"] = lab["unique_ID"].str.replace("-", "_", n=1)
    lab_map = dict(zip(lab["feat_key"], lab["classification"]))
    df["fa_label"]   = df["filename"].map(lab_map).fillna("_unlabeled")
    df["is_labeled"] = df["fa_label"] != "_unlabeled"

    print(f"Patches: {len(df)} total, {df['is_labeled'].sum()} labeled")
    print("FA class distribution (labeled):")
    print(df[df["is_labeled"]]["fa_label"].value_counts().to_string())

    print("\nScaling features ...", flush=True)
    X = df[FEAT_COLS].values.astype(np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df, X_scaled


def run_umap(X_scaled: np.ndarray, seed=RANDOM_STATE) -> np.ndarray:
    print("Running UMAP ...", flush=True)
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                        metric="euclidean", random_state=seed)
    return reducer.fit_transform(X_scaled)


def run_pca(X_scaled: np.ndarray) -> np.ndarray:
    print("Running PCA ...", flush=True)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    return pca.fit_transform(X_scaled)


def run_kmeans(X_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (labels, centres) in scaled space."""
    print(f"Running KMeans k={N_CLUSTERS} ...", flush=True)
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=RANDOM_STATE)
    labels = km.fit_predict(X_scaled)
    return labels, km.cluster_centers_


def find_examples(df: pd.DataFrame, X_scaled: np.ndarray,
                  km_labels: np.ndarray, centres: np.ndarray) -> list[list[int]]:
    """For each cluster: indices of N_EXAMPLES closest patches to centre."""
    examples = []
    for k in range(N_CLUSTERS):
        mask   = km_labels == k
        idxs   = np.where(mask)[0]
        dists  = np.linalg.norm(X_scaled[idxs] - centres[k], axis=1)
        top    = idxs[np.argsort(dists)[:N_EXAMPLES]]
        examples.append(top.tolist())
    return examples


# ── Actin patch loading ───────────────────────────────────────────────────────

def _load_act_images_and_meta(h5_path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    with h5py.File(h5_path) as f:
        imgs = f["images/act"][:]
        meta = pd.read_csv(io.StringIO(f["meta/csv"][()].decode()))
    return imgs, meta


def build_patch_loader(df: pd.DataFrame):
    """Returns a function patch_fn(row_idx) → 32×32 actin array (float32)."""
    print("Loading actin images from label H5s ...", flush=True)
    ctrl_imgs, ctrl_meta = _load_act_images_and_meta(H5_CTRL)
    ycomp_imgs, ycomp_meta = _load_act_images_and_meta(H5_YCOMP)

    ctrl_lookup = {row.filename: (row.frame_idx, row.canvas_cx, row.canvas_cy)
                   for row in ctrl_meta.itertuples()}
    ycomp_lookup = {row.filename: (row.frame_idx, row.canvas_cx, row.canvas_cy)
                    for row in ycomp_meta.itertuples()}

    # Pad images once
    ctrl_pad  = np.pad(ctrl_imgs,  ((0,0),(PAD,PAD),(PAD,PAD)), mode="edge")
    ycomp_pad = np.pad(ycomp_imgs, ((0,0),(PAD,PAD),(PAD,PAD)), mode="edge")

    half = PS // 2

    def _get(row_idx: int) -> np.ndarray | None:
        fname = df["filename"].iloc[row_idx]
        cond  = df["condition"].iloc[row_idx] if "condition" in df.columns else \
                ("ycomp" if fname.startswith("ycomp") else "control")
        if cond == "control" and fname in ctrl_lookup:
            fi, cx, cy = ctrl_lookup[fname]
            imgs = ctrl_pad
        elif cond == "ycomp" and fname in ycomp_lookup:
            fi, cx, cy = ycomp_lookup[fname]
            imgs = ycomp_pad
        else:
            return None
        cy0 = cy + PAD - half
        cx0 = cx + PAD - half
        return imgs[fi, cy0:cy0+PS, cx0:cx0+PS].copy()

    return _get


def _norm_patch(p: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(p, 1), np.percentile(p, 99)
    return np.clip((p - lo) / (hi - lo + 1e-9), 0, 1)


# ── Scatter plot helpers ───────────────────────────────────────────────────────

def _scatter_fa(ax, emb: np.ndarray, df: pd.DataFrame,
                title: str, subtitle: str = ""):
    """Scatter: unlabeled grey (tiny), labeled colored by FA class."""
    # Background: all unlabeled patches
    mask_unlab = ~df["is_labeled"].values
    ax.scatter(emb[mask_unlab, 0], emb[mask_unlab, 1],
               s=2, c="#dddddd", alpha=0.25, linewidths=0, rasterized=True,
               label=None)

    # Labeled: one series per class
    for cls in FA_ORDER:
        mask = df["fa_label"].values == cls
        if mask.sum() == 0:
            continue
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   s=16, c=FA_COLORS[cls], alpha=0.85, linewidths=0,
                   rasterized=True, label=f"{cls} (n={mask.sum()})")

    ax.set_title(f"{title}\n{subtitle}", fontsize=8.5, color="#333333", pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

    legend = ax.legend(fontsize=7, frameon=False, markerscale=1.5,
                       loc="best", handlelength=1.0)
    for lh in legend.legend_handles:
        lh.set_alpha(1.0)


def _scatter_km(ax, emb: np.ndarray, df: pd.DataFrame,
                km_labels: np.ndarray, examples: list[list[int]],
                title: str, subtitle: str = ""):
    """Scatter: all patches colored by KMeans cluster, examples ringed."""
    cmap = plt.get_cmap(CLUSTER_CMAP)
    colors = [cmap(k / N_CLUSTERS) for k in km_labels]

    ax.scatter(emb[:, 0], emb[:, 1],
               s=4, c=colors, alpha=0.5, linewidths=0, rasterized=True)

    # Highlight example patches (ring markers)
    flat_ex = [idx for cluster in examples for idx in cluster]
    for k, cluster in enumerate(examples):
        ax.scatter(emb[cluster, 0], emb[cluster, 1],
                   s=60, facecolors="none", edgecolors=[cmap(k / N_CLUSTERS)],
                   linewidths=1.4, zorder=5, alpha=1.0)

    # Legend entries per cluster
    handles = [mpatches.Patch(color=cmap(k / N_CLUSTERS),
                               label=f"Cluster {k}") for k in range(N_CLUSTERS)]
    ax.legend(handles=handles, fontsize=6, frameon=False, ncol=2,
              loc="best", handlelength=1.0)

    ax.set_title(f"{title}\n{subtitle}", fontsize=8.5, color="#333333", pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)


# ── Figure generators ─────────────────────────────────────────────────────────

def fig_fa_comparison(emb_umap: np.ndarray, emb_pca: np.ndarray,
                      df: pd.DataFrame) -> Image.Image:
    fig, (ax_u, ax_p) = plt.subplots(1, 2, figsize=(12.5, 5.8))
    fig.patch.set_facecolor("white")
    n_lab = df["is_labeled"].sum()
    _scatter_fa(ax_u, emb_umap, df,
                "UMAP — coloured by FA class",
                f"n_labeled = {n_lab}  ·  n_total = {len(df)}")
    _scatter_fa(ax_p, emb_pca, df,
                "PCA — coloured by FA class",
                f"n_labeled = {n_lab}  ·  n_total = {len(df)}")
    ax_u.set_xlabel("UMAP-1", fontsize=8)
    ax_u.set_ylabel("UMAP-2", fontsize=8)
    ax_p.set_xlabel("PC1", fontsize=8)
    ax_p.set_ylabel("PC2", fontsize=8)
    fig.suptitle("DS1 / ch3 actin / PS32  —  CellProfiler features (50-dim)",
                 fontsize=10, y=1.01, color="#1a2a3a")
    fig.tight_layout()
    return _fig_to_pil(fig)


def fig_km_comparison(emb_umap: np.ndarray, emb_pca: np.ndarray,
                      df: pd.DataFrame, km_labels: np.ndarray,
                      examples: list[list[int]]) -> Image.Image:
    fig, (ax_u, ax_p) = plt.subplots(1, 2, figsize=(12.5, 5.8))
    fig.patch.set_facecolor("white")
    n_ex = N_EXAMPLES * N_CLUSTERS
    _scatter_km(ax_u, emb_umap, df, km_labels, examples,
                "UMAP — coloured by KMeans cluster",
                f"k={N_CLUSTERS}  ·  rings = {n_ex} example patches")
    _scatter_km(ax_p, emb_pca, df, km_labels, examples,
                "PCA — coloured by KMeans cluster",
                f"k={N_CLUSTERS}  ·  rings = {n_ex} example patches")
    ax_u.set_xlabel("UMAP-1", fontsize=8)
    ax_u.set_ylabel("UMAP-2", fontsize=8)
    ax_p.set_xlabel("PC1", fontsize=8)
    ax_p.set_ylabel("PC2", fontsize=8)
    fig.suptitle("DS1 / ch3 actin / PS32  —  CellProfiler features (50-dim)  —  KMeans k=10",
                 fontsize=10, y=1.01, color="#1a2a3a")
    fig.tight_layout()
    return _fig_to_pil(fig)


def fig_gallery(df: pd.DataFrame, examples: list[list[int]],
                get_patch, km_labels: np.ndarray) -> Image.Image:
    """10 rows (clusters) × N_EXAMPLES columns (patches)."""
    cmap_km = plt.get_cmap(CLUSTER_CMAP)
    fig, axes = plt.subplots(N_CLUSTERS, N_EXAMPLES,
                              figsize=(N_EXAMPLES * 1.15, N_CLUSTERS * 1.15))
    fig.patch.set_facecolor("white")

    for k in range(N_CLUSTERS):
        for j, idx in enumerate(examples[k]):
            ax = axes[k, j]
            p = get_patch(idx)
            if p is not None:
                ax.imshow(_norm_patch(p), cmap="gray", vmin=0, vmax=1,
                          interpolation="nearest")
            else:
                ax.set_facecolor("#cccccc")

            # FA label overlay (small coloured dot)
            fa = df["fa_label"].iloc[idx]
            dot_c = FA_COLORS.get(fa, "#aaaaaa")
            ax.plot(2, 2, "o", markersize=4, color=dot_c,
                    markeredgewidth=0.5, markeredgecolor="white")

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(cmap_km(k / N_CLUSTERS))
                spine.set_linewidth(1.5)

        # Row label (cluster number)
        axes[k, 0].set_ylabel(f"C{k}", rotation=0, labelpad=14,
                               fontsize=7, va="center",
                               color=cmap_km(k / N_CLUSTERS))

    # Column title: "Rank 1" ... "Rank N_EXAMPLES"
    for j in range(N_EXAMPLES):
        axes[0, j].set_title(f"#{j+1}", fontsize=7, color="#555555")

    fig.suptitle(
        f"Actin CP32 KMeans k={N_CLUSTERS} — example patches (closest to cluster centre)\n"
        "Border colour = cluster; dot = FA class label",
        fontsize=8.5, y=1.01, color="#1a2a3a")
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    return _fig_to_pil(fig)


# ── PPT slide builders ────────────────────────────────────────────────────────

def _slide_two_panel(prs, img: Image.Image, title: str, subtitle: str):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _rect(slide, 0, 0, SLIDE_W, SLIDE_H, "#ffffff")
    _txt(slide, title, 0.22, 0.05, SLIDE_W - 0.44, 0.36,
         size=12.5, bold=True, color="#1a2a3a")
    _rect(slide, 0.22, 0.40, SLIDE_W - 0.44, 0.016, "#cccccc")
    if subtitle:
        _txt(slide, subtitle, 0.22, 0.42, SLIDE_W - 0.44, 0.25,
             size=8, italic=True, color="#666666")
    top_offset = 0.68 if subtitle else 0.45
    h = SLIDE_H - top_offset - 0.12
    _paste_pil(slide, img, 0.20, top_offset, SLIDE_W - 0.40, h)


def _slide_gallery(prs, img: Image.Image, title: str):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _rect(slide, 0, 0, SLIDE_W, SLIDE_H, "#ffffff")
    _txt(slide, title, 0.22, 0.05, SLIDE_W - 0.44, 0.36,
         size=12.5, bold=True, color="#1a2a3a")
    _rect(slide, 0.22, 0.40, SLIDE_W - 0.44, 0.016, "#cccccc")
    _paste_pil(slide, img, 0.20, 0.44, SLIDE_W - 0.40, SLIDE_H - 0.56)


# ── Legend slide (FA class colours) ──────────────────────────────────────────

def fig_legend() -> Image.Image:
    fig, ax = plt.subplots(figsize=(6, 1.8))
    fig.patch.set_facecolor("white")
    ax.set_visible(False)

    handles = [mpatches.Patch(color=FA_COLORS[c], label=c) for c in FA_ORDER]
    handles += [mpatches.Patch(color=FA_COLORS["_unlabeled"], label="Unlabeled")]
    fig.legend(handles=handles, ncol=4, loc="center",
               fontsize=9, frameon=False)
    fig.tight_layout()
    return _fig_to_pil(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load data and features
    df, X_scaled = load_data()

    # 2. Dimensionality reduction
    emb_umap = run_umap(X_scaled)
    emb_pca  = run_pca(X_scaled)

    # 3. KMeans clustering
    km_labels, centres = run_kmeans(X_scaled)
    df["cluster"] = km_labels

    # 4. Find example patches
    examples = find_examples(df, X_scaled, km_labels, centres)
    print(f"Selected {sum(len(e) for e in examples)} example patches "
          f"({N_EXAMPLES} per cluster × {N_CLUSTERS} clusters)")

    # 5. Patch loader for actin images
    get_patch = build_patch_loader(df)

    # 6. Generate figures
    print("Rendering figures ...", flush=True)
    img_fa  = fig_fa_comparison(emb_umap, emb_pca, df)
    img_km  = fig_km_comparison(emb_umap, emb_pca, df, km_labels, examples)
    img_gal = fig_gallery(df, examples, get_patch, km_labels)
    img_leg = fig_legend()

    # 7. Build PPT
    prs = Presentation()
    prs.slide_width  = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)

    _slide_two_panel(
        prs, img_fa,
        "Actin CP32 features — UMAP & PCA coloured by FA class",
        "DS1 (vinc, U2OS)  ·  ch3 actin  ·  PS32  ·  CellProfiler 50-dim  ·  "
        f"KMeans k={N_CLUSTERS}  ·  B1 labels n≈1339  ·  UMAP: n_neighbors=15 min_dist=0.1"
    )
    _slide_two_panel(
        prs, img_km,
        f"Actin CP32 features — UMAP & PCA coloured by KMeans cluster (k={N_CLUSTERS})",
        f"Rings mark the {N_EXAMPLES} patches closest to each cluster centre  ·  all {len(df)} patches shown"
    )
    _slide_gallery(
        prs, img_gal,
        f"Actin CP32 KMeans k={N_CLUSTERS} — example patches per cluster"
    )

    prs.save(OUT)
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()
