#!/usr/bin/env python3
"""
make_fig4_panels.py

Panel-ready plots for Figure 4 (latent representations of local subcellular patterns).
Four modalities x {CellProfiler, ilastik} (SupCon-AE = placeholders):
  paxillin/FA (DS1 ch1, 5 FA classes), actin (DS1 ch3), microtubules (Ernest ch1), zyxin (DS1 ch2)

Every plot is saved as its own PNG (300 dpi) in results/fig4_panels/<modality>/ and
also placed on slides in results/fig4_panels.pptx.

Inputs (from run_mt_features_clusters.py / fig4_build_fa_patches.py):
  patches32.npy, patches32_index.csv, features_{cp,il}32.csv, clusters_{cp,il}32.csv, umap_{cp,il}32.npy
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "results" / "fig4_panels"
PPTX = REPO / "results" / "fig4_panels.pptx"
AE = Path("/home/lding/lding/fa_data_analysis/ae_results")

MODS = {
    "paxillin": dict(title="Paxillin / FA (DS1, ch1)", dir=AE / "fig4_fa_patches"),
    "actin": dict(title="Actin (DS1, ch3)", dir=AE / "ds1_actin_patches"),
    "mt": dict(title="Microtubules (Ernest U2OS, ch1)", dir=Path("/home/lding/lding/data/Ernest/mt_patches")),
    "zyxin": dict(title="Zyxin (DS1, ch2)", dir=AE / "ds1_zyxin_patches"),
}
METHODS = {"cp": "CellProfiler", "il": "ilastik", "ae": "ConAE"}

FA5 = ["No adhesion", "Nascent Adhesion", "focal complex", "focal adhesion", "fibrillar adhesion"]
FA5_COL = {"No adhesion": "#9467bd", "Nascent Adhesion": "#1f77b4", "focal complex": "#ff7f0e",
           "focal adhesion": "#2ca02c", "fibrillar adhesion": "#d62728"}
COND_COL = {"Control": "#d62728", "Ycomp": "#2ca02c"}
TAB10 = plt.get_cmap("tab10")
TAB20 = plt.get_cmap("tab20")


def ccol(c, k):
    return TAB10(c) if k <= 10 else TAB20(c)
GREY = "#c8c8c8"

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "font.family": "DejaVu Sans", "savefig.dpi": 300})


# ───────────────────────────── data ─────────────────────────────
class Mod:
    def __init__(self, key):
        self.key, self.title, d = key, MODS[key]["title"], MODS[key]["dir"]
        self.P = np.load(d / "patches32.npy", mmap_mode="r")
        self.idx = pd.read_csv(d / "patches32_index.csv")
        self.feat, self.cl, self.umap, self.Z = {}, {}, {}, {}
        self.methods = [m for m in METHODS if (d / f"features_{m}32.csv").exists()]
        self.k = {}
        for m in self.methods:
            f = pd.read_csv(d / f"features_{m}32.csv")
            F = np.nan_to_num(f.drop(columns="name").values.astype(np.float64))
            self.feat[m] = f
            self.Z[m] = QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                                            random_state=0).fit_transform(F)
            self.cl[m] = pd.read_csv(d / f"clusters_{m}32.csv")
            self.k[m] = int(self.cl[m].cluster.max()) + 1
            self.umap[m] = np.load(d / f"umap_{m}32.npy")
        self.intensity = self.feat["cp"]["intensity_mean"].values
        # categorical annotation used for colouring
        if key == "paxillin":
            lab = self.idx.fa_label.fillna("")
            self.labels = lab.where(lab.isin(FA5), "")
            self.label_name = "FA class (annotated)"
        else:
            self.labels = None
            self.label_name = "KMeans cluster"
        # metadata colouring
        if key == "mt":
            self.meta = self.idx.file.str.extract(r"U2OS_DMSO_(\w+?)_scan")[0].fillna("?")
            self.meta_name = "acquisition set"
            ms = sorted(self.meta.unique())
            self.meta_col = {m: TAB10(i) for i, m in enumerate(ms)}
        else:
            self.meta, self.meta_name, self.meta_col = self.idx.file, "condition", COND_COL

    def patch(self, i):
        return np.asarray(self.P[int(i)], dtype=np.float32)


# ───────────────────────────── helpers ─────────────────────────────
def save(fig, mod, name):
    out = FIG / mod
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{name}.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def umap_axes(title):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return fig, ax


def legend(ax, items, ncol=1):
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c, label=l, markersize=5) for l, c in items],
              fontsize=7, frameon=False, loc="best", ncol=ncol, handletextpad=0.2, labelspacing=0.3)


def umap_classes(M, m):
    e = M.umap[m]
    fig, ax = umap_axes(f"{METHODS[m]}: {M.label_name}" + ("" if M.labels is not None else f" (k={M.k[m]})"))
    if M.labels is not None:
        un = (M.labels == "").values
        ax.scatter(*e[un].T, s=1.5, c=GREY, alpha=0.4, rasterized=True, linewidths=0)
        for c in FA5:
            k = (M.labels == c).values
            ax.scatter(*e[k].T, s=9, c=FA5_COL[c], alpha=0.9, linewidths=0, rasterized=True)
        legend(ax, [(f"{c} (n={int((M.labels == c).sum())})", FA5_COL[c]) for c in FA5] + [("unlabelled", GREY)])
    else:
        cl = M.cl[m].cluster.values
        K = M.k[m]
        ax.scatter(*e.T, s=1.5, c=[ccol(c, K) for c in cl], alpha=0.6, rasterized=True, linewidths=0)
        for k in range(K):
            ax.text(*np.median(e[cl == k], 0), str(k), fontsize=10, weight="bold", ha="center", va="center")
    return save(fig, M.key, f"umap_class_{m}")


def umap_meta(M, m):
    e = M.umap[m]
    fig, ax = umap_axes(f"{METHODS[m]}: {M.meta_name}")
    order = np.random.default_rng(0).permutation(len(e))
    ax.scatter(*e[order].T, s=1.5, c=[M.meta_col[v] for v in M.meta.values[order]], alpha=0.5,
               rasterized=True, linewidths=0)
    legend(ax, [(f"{k} (n={int((M.meta == k).sum())})", c) for k, c in M.meta_col.items()])
    return save(fig, M.key, f"umap_{M.meta_name}_{m}")


def umap_intensity(M, m):
    e, v = M.umap[m], M.intensity
    lo, hi = np.percentile(v, [1, 99])
    fig, ax = umap_axes(f"{METHODS[m]}: patch mean intensity")
    sc = ax.scatter(*e.T, s=1.5, c=np.clip(v, lo, hi), cmap="viridis", rasterized=True, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("mean intensity (normalised)", fontsize=8)
    return save(fig, M.key, f"umap_intensity_{m}")


def patch_grid(M, rows, ncol, row_labels, title, name, window=None, col_titles=None, fixed=None):
    """rows: list of lists of patch indices.  window: None => per-patch min-max, else (lo,hi)."""
    nr = len(rows)
    fig, axes = plt.subplots(nr, ncol, figsize=(ncol * 0.62 + 1.1, nr * 0.62 + 0.5), squeeze=False)
    for r, ids in enumerate(rows):
        for c in range(ncol):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if c < len(ids):
                p = M.patch(ids[c])
                lo, hi = (p.min(), p.max()) if window is None else window[r] if isinstance(window, list) else window
                ax.imshow(p, cmap="gray", vmin=lo, vmax=hi + 1e-8, interpolation="nearest")
                if col_titles is not None:
                    ax.set_title(col_titles[r][c], fontsize=5.5, pad=1)
            else:
                ax.axis("off")
        axes[r, 0].set_ylabel(row_labels[r], rotation=0, ha="right", va="center", fontsize=7, labelpad=4)
    fig.suptitle(title, fontsize=9, y=1.0)
    fig.subplots_adjust(wspace=0.04, hspace=0.04)
    return save(fig, M.key, name)


def representative(M, m, n=8):
    Z, cl = M.Z[m], M.cl[m]
    rows, labs = [], []
    if M.labels is not None:
        for c in FA5:
            ids = np.where((M.labels == c).values)[0]
            cen = Z[ids].mean(0)
            near = ids[np.argsort(np.linalg.norm(Z[ids] - cen, axis=1))[:n]]
            rows.append(near); labs.append(f"{c}\n(n={len(ids)})")
    else:
        for k in range(M.k[m]):
            sub = cl[cl.cluster == k].nsmallest(n, "dist")
            rows.append(sub.idx.values); labs.append(f"cluster {k}\n(n={int((cl.cluster == k).sum())})")
    return patch_grid(M, rows, n, labs, f"{METHODS[m]}: patches nearest to {'class' if M.labels is not None else 'cluster'} centre",
                      f"patches_centres_{m}")


def overlap_patches(M, m, n=8, k=15):
    """Labelled patches sitting in another class's neighbourhood (class-overlap regions)."""
    Z = M.Z[m]
    li = np.where((M.labels != "").values)[0]
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z[li])
    _, nb = nn.kneighbors(Z[li])
    lab = M.labels.values[li]
    rows, labs, tit = [], [], []
    for c in FA5:
        sel = np.where(lab == c)[0]
        frac = np.array([(lab[nb[i, 1:]] == c).mean() for i in sel])
        pick = sel[np.argsort(frac)[:n]]                      # least self-consistent neighbourhoods
        rows.append(li[pick])
        labs.append(f"{c}")
        tit.append([_maj(lab, nb[i, 1:])[:7] for i in pick])
    return patch_grid(M, rows, n, labs,
                      f"{METHODS[m]}: overlap-region patches (top: own class, small title: majority class of 15 nearest labelled)",
                      f"patches_overlap_{m}", col_titles=tit)


def _maj(lab, ids):
    v, c = np.unique(lab[ids], return_counts=True)
    return str(v[c.argmax()])


# ───────────────────────────── trajectories ─────────────────────────────
def knn_graph(Z, k):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    d, i = nn.kneighbors(Z)
    r = np.repeat(np.arange(len(Z)), k)
    g = csr_matrix((d[:, 1:].ravel(), (r, i[:, 1:].ravel())), shape=(len(Z),) * 2)
    return g.maximum(g.T)


def shortest(g, a, b):
    dist, pred = dijkstra(g, indices=a, return_predecessors=True)
    if not np.isfinite(dist[b]):
        return None
    p, cur = [b], b
    while cur != a:
        cur = pred[cur]
        p.append(cur)
    return p[::-1]


def medoid(Z, ids):
    return ids[np.argmin(np.linalg.norm(Z[ids] - Z[ids].mean(0), axis=1))]


def trajectory(M, m, nsamp=12):
    Z, cl, e = M.Z[m], M.cl[m], M.umap[m]
    if M.labels is not None:
        anchors = [medoid(Z, np.where((M.labels == c).values)[0]) for c in FA5]
        names = FA5
        note = "chain: " + " → ".join(c.split()[0] if c != "No adhesion" else "No adh." for c in FA5)
    else:
        cen = np.stack([Z[cl.cluster.values == k].mean(0) for k in range(M.k[m])])
        D = np.linalg.norm(cen[:, None] - cen[None], axis=2)
        a, b = np.unravel_index(D.argmax(), D.shape)
        anchors = [medoid(Z, np.where(cl.cluster.values == k)[0]) for k in (a, b)]
        names = [f"cluster {a}", f"cluster {b}"]
        note = f"cluster {a} → cluster {b} (most distant centres)"
    path = None
    for k in (15, 30, 60):
        g = knn_graph(Z, k)
        segs = [shortest(g, anchors[i], anchors[i + 1]) for i in range(len(anchors) - 1)]
        if all(s is not None for s in segs):
            path = [n for i, s in enumerate(segs) for n in (s if i == 0 else s[1:])]
            break
    if path is None:
        return None, None
    path = np.array(path)
    # UMAP with path
    fig, ax = umap_axes(f"{METHODS[m]}: candidate path\n({note})")
    ax.scatter(*e.T, s=1.2, c=GREY, alpha=0.35, rasterized=True, linewidths=0)
    if M.labels is not None:
        for c in FA5:
            kk = (M.labels == c).values
            ax.scatter(*e[kk].T, s=5, c=FA5_COL[c], alpha=0.6, linewidths=0, rasterized=True)
    ax.plot(*e[path].T, "-", color="k", lw=1.0, alpha=0.8)
    pick = np.unique(np.linspace(0, len(path) - 1, nsamp).round().astype(int))
    ax.scatter(*e[path[pick]].T, s=22, c=np.arange(len(pick)), cmap="autumn", edgecolors="k", linewidths=0.5, zorder=5)
    for j, pi in enumerate(path[pick]):
        ax.text(*e[pi], str(j + 1), fontsize=6, ha="left", va="bottom", color="k")
    p1 = save(fig, M.key, f"trajectory_umap_{m}")
    # patch strip
    ids = path[pick]
    t = [[""] * len(ids)]
    if M.labels is not None:
        t = [[(M.labels.values[i] or "·")[:9] for i in ids]]
    p2 = patch_grid(M, [list(ids)], len(ids), [f"path\n({len(path)} nodes)"],
                    f"{METHODS[m]}: patches along the path (1→{len(ids)})", f"trajectory_patches_{m}", col_titles=t)
    return p1, p2


# ───────────────────────────── intensity ─────────────────────────────
def intensity_matched(M, m, ncl=4, n=8):
    cl = M.cl[m].cluster.values
    sizes = np.bincount(cl, minlength=M.k[m])
    rows, labs, wins = [], [], []
    for k in np.argsort(sizes)[::-1][:ncl]:
        ids = np.where(cl == k)[0]
        v = M.intensity[ids]
        q = np.percentile(v, [10, 50, 90])
        pat = np.stack([M.patch(i) for i in ids[::max(1, len(ids) // 400)]])
        win = (float(np.percentile(pat, 1)), float(np.percentile(pat, 99.5)))      # fixed window per cluster
        for lbl, qq in zip(("low", "mid", "high"), q):
            near = ids[np.argsort(np.abs(v - qq))[:n * 3]]
            near = np.random.default_rng(0).choice(near, n, replace=False)
            rows.append(near); labs.append(f"cl {k} {lbl}\n(I={qq:.2f})"); wins.append(win)
    return patch_grid(M, rows, n, labs,
                      f"{METHODS[m]}: same cluster, low/mid/high intensity (fixed display window per cluster)",
                      f"patches_intensity_matched_{m}", window=wins)


def intensity_box(M, m):
    key = M.labels.where(M.labels != "", np.nan) if M.labels is not None else pd.Series(M.cl[m].cluster.values)
    df = pd.DataFrame({"g": key, "I": M.intensity}).dropna()
    order = FA5 if M.labels is not None else list(range(M.k[m]))
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    data = [df.I[df.g == g].values for g in order]
    bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6)
    for i, b in enumerate(bp["boxes"]):
        b.set_facecolor(FA5_COL[order[i]] if M.labels is not None else ccol(i, M.k[m])); b.set_alpha(0.8)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([str(o).replace(" ", "\n") if M.labels is not None else str(o) for o in order],
                       fontsize=6 if M.labels is not None else 8)
    ax.set_ylabel("mean intensity (normalised)")
    ax.set_title(f"{METHODS[m]}: intensity by {'class' if M.labels is not None else f'cluster (k={M.k[m]})'}")
    return save(fig, M.key, f"intensity_box_{m}")


def eta2(v, g):
    g = np.asarray(g)
    m = pd.notna(g) & (g != "")
    v, g = v[m], g[m]
    tot = ((v - v.mean()) ** 2).sum()
    btw = sum(len(v[g == u]) * (v[g == u].mean() - v.mean()) ** 2 for u in np.unique(g))
    return btw / tot


def intensity_summary(mods):
    """Fraction of intensity variance explained by cluster membership (eta^2)."""
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    keys = list(mods)
    w = 0.26
    cols = {"cp": "#4c72b0", "il": "#dd8452", "ae": "#55a868"}
    for j, m in enumerate(METHODS):
        vals = [eta2(mods[k].intensity, mods[k].cl[m].cluster.values) if m in mods[k].methods else np.nan for k in keys]
        ax.bar(np.arange(len(keys)) + (j - 1) * w * 1.1, np.nan_to_num(vals), w, label=METHODS[m], color=cols[m])
        for i, v in enumerate(vals):
            if np.isnan(v):
                ax.bar(i + (j - 1) * w * 1.1, 0.04, w, color="none", edgecolor="k", hatch="//", lw=0.5)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([mods[k].title.split(" (")[0].replace(" / FA", "") for k in keys], fontsize=8)
    ax.set_ylabel("η² (intensity ~ cluster)")
    ax.set_title("How much of patch intensity is explained\nby KMeans cluster membership (hatched = pending)")
    ax.legend(fontsize=7, frameon=False)
    ax.set_ylim(0, 1)
    return save(fig, "summary", "intensity_eta2")


def kcurve(M):
    d = MODS[M.key]["dir"]
    fig, ax = plt.subplots(figsize=(4, 3))
    for m in M.methods:
        f = d / f"kcurve_{m}32.csv"
        if f.exists():
            c = pd.read_csv(f)
            ln, = ax.plot(c.k, c.silhouette, "-o", ms=3, label=METHODS[m])
            ax.plot(M.k[m], c.silhouette[c.k == M.k[m]], "o", ms=9, mfc="none", mec=ln.get_color(), mew=1.5)
    ax.set_xlabel("k"); ax.set_ylabel("silhouette")
    ax.set_title("Choice of k (circle = chosen)")
    ax.legend(fontsize=7, frameon=False)
    return save(fig, M.key, "kcurve")


def placeholder(text, w=4.0, h=4.0, name="placeholder"):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("#f2f2f2")
    for s in ax.spines.values():
        s.set_linestyle("--"); s.set_color("#888")
    ax.text(0.5, 0.5, f"SupCon-AE\n{text}\n(placeholder)", ha="center", va="center", fontsize=10, color="#666",
            transform=ax.transAxes)
    return save(fig, "summary", f"{name}_{w:g}x{h:g}")


# ───────────────────────────── PPT ─────────────────────────────
SW, SH = Inches(13.33), Inches(7.5)
DARK = RGBColor(0x1F, 0x4E, 0x79)


def new_slide(prs, title, sub=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    r = s.shapes.add_shape(1, 0, 0, SW, Inches(0.6))
    r.fill.solid(); r.fill.fore_color.rgb = DARK; r.line.fill.background()
    tb = s.shapes.add_textbox(Inches(0.2), Inches(0.07), Inches(12.9), Inches(0.5))
    tb.text_frame.text = title
    p = tb.text_frame.paragraphs[0]; p.font.size = Pt(20); p.font.bold = True; p.font.color.rgb = RGBColor(255, 255, 255)
    if sub:
        tb = s.shapes.add_textbox(Inches(0.2), Inches(0.62), Inches(12.9), Inches(0.35))
        tb.text_frame.text = sub
        tb.text_frame.paragraphs[0].font.size = Pt(11)
        tb.text_frame.paragraphs[0].font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    return s


def put(slide, path, l, t, w, h, caption=None):
    """Place picture fit inside the (w,h) box, keeping aspect ratio."""
    im = Image.open(path)
    ar = im.width / im.height
    ww, hh = (w, w / ar) if w / h < ar else (h * ar, h)
    slide.shapes.add_picture(str(path), int(l + (w - ww) / 2), int(t), int(ww), int(hh))
    if caption:
        tb = slide.shapes.add_textbox(l, t + h, w, Inches(0.3))
        tb.text_frame.text = caption
        tb.text_frame.paragraphs[0].font.size = Pt(10)


def text_slide(prs, title, lines):
    s = new_slide(prs, title)
    tb = s.shapes.add_textbox(Inches(0.5), Inches(0.9), Inches(12.3), Inches(6.3))
    tf = tb.text_frame; tf.word_wrap = True
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ln; p.font.size = Pt(13)
    return s


def three(prs, title, paths, ph, sub=None, cols=3, stack=False):
    """paths: dict method->path (cp, il) + ph placeholder path."""
    s = new_slide(prs, title, sub)
    items = [(METHODS[m], paths.get(m)) for m in ("cp", "il")]
    items.append(("ConAE (unsupervised)", paths.get("ae")) if paths.get("ae") else ("SupCon-AE (pending)", ph))
    if stack:                         # wide strips: one per row
        h = Inches(1.9)
        for i, (n, p) in enumerate(items):
            if p:
                put(s, p, Inches(0.4), Inches(1.0) + i * (h + Inches(0.2)), Inches(12.5), h, n)
        return s
    w = Inches(4.1)
    for i, (n, p) in enumerate(items):
        if p:
            put(s, p, Inches(0.3) + i * (w + Inches(0.1)), Inches(1.05), w, Inches(5.7), n)
    return s


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    mods = {k: Mod(k) for k in MODS}
    prs = Presentation(); prs.slide_width, prs.slide_height = SW, SH

    s = new_slide(prs, "Figure 4 panels: latent representations of local subcellular patterns")
    tb = s.shapes.add_textbox(Inches(0.5), Inches(1.0), Inches(12.3), Inches(5))
    tb.text_frame.word_wrap = True
    tb.text_frame.text = ("Every plot is also a stand-alone PNG (300 dpi) in results/fig4_panels/<modality>/.\n"
                          "Columns: CellProfiler (50 feat), ilastik (80 feat), ConAE (actin/zyxin/MT, trained locally); grey = pending (paxillin SupCon-AE).\n"
                          "Same population and UMAP settings per modality (quantile-transformed features, UMAP 15 nn / min_dist 0.1 / seed 42).\n"
                          "32 px patches. Modalities: paxillin/FA (DS1 ch1), actin (DS1 ch3), microtubules (Ernest ch1), zyxin (DS1 ch2).")
    for p in tb.text_frame.paragraphs:
        p.font.size = Pt(15)

    ph_umap = placeholder("UMAP", 4, 4, "umap")
    ph_grid = placeholder("patch panel", 4, 5.5, "grid")
    ph_strip = placeholder("patch strip / path", 12, 1.9, "strip")
    ph_box = placeholder("intensity per cluster", 4.2, 3.2, "box")

    summary_stats = []
    for key, M in mods.items():
        P = {m: {} for m in METHODS}
        for m in M.methods:
            P[m]["class"] = umap_classes(M, m)
            P[m]["meta"] = umap_meta(M, m)
            P[m]["int"] = umap_intensity(M, m)
            P[m]["rep"] = representative(M, m)
            P[m]["box"] = intensity_box(M, m)
            P[m]["match"] = intensity_matched(M, m)
            P[m]["tr_u"], P[m]["tr_p"] = trajectory(M, m)
            if M.labels is not None:
                P[m]["overlap"] = overlap_patches(M, m)
            rho = spearmanr(M.intensity, M.umap[m][:, 0])[0], spearmanr(M.intensity, M.umap[m][:, 1])[0]
            summary_stats.append((key, m, eta2(M.intensity, M.cl[m].cluster.values), rho))
        g = lambda k: {m: P[m].get(k) for m in M.methods}
        T = M.title
        three(prs, f"{T}: UMAP by {M.label_name}", g("class"), ph_umap,
              sub=None if M.labels is not None else "k per method chosen by silhouette: " + ", ".join(f"{METHODS[m]} k={M.k[m]}" for m in M.methods))
        three(prs, f"{T}: UMAP by {M.meta_name}", g("meta"), ph_umap)
        three(prs, f"{T}: UMAP by patch intensity", g("int"), ph_umap)
        three(prs, f"{T}: representative patches", g("rep"), ph_grid,
              sub="rows = " + ("annotated classes" if M.labels is not None else "KMeans clusters") + ", 8 patches nearest to the centre")
        if M.labels is not None:
            three(prs, f"{T}: overlap-region patches", g("overlap"), ph_grid,
                  sub="labelled patches whose 15 nearest labelled neighbours are mostly another class")
        three(prs, f"{T}: candidate trajectory (UMAP)", g("tr_u"), ph_umap,
              sub="shortest path in a kNN graph of the feature space; not a temporal / biological transition")
        three(prs, f"{T}: patches along the candidate trajectory", g("tr_p"), ph_strip, stack=True)
        three(prs, f"{T}: intensity vs pattern", g("box"), ph_box)
        if M.labels is None or True:
            sk = new_slide(prs, f"{T}: choice of k for KMeans")
            put(sk, kcurve(M), Inches(0.5), Inches(1.0), Inches(5.5), Inches(4.5))
            tb = sk.shapes.add_textbox(Inches(6.3), Inches(1.2), Inches(6.5), Inches(4)); tb.text_frame.word_wrap = True
            tb.text_frame.text = ("k from silhouette over k = 4..12 (6,000-patch subsample, quantile-transformed features).\n"
                                  "Curves are nearly flat (these are continuous populations), so the argmax is unstable: "
                                  "chosen k = largest k within 0.02 silhouette of the best.\n"
                                  + "\n".join(f"{METHODS[m]}: k={M.k[m]}" for m in M.methods))
            for pp in tb.text_frame.paragraphs:
                pp.font.size = Pt(13)
        three(prs, f"{T}: same pattern, different intensity", g("match"), ph_grid,
              sub="4 largest clusters; low / mid / high intensity (10th / 50th / 90th pct) with a fixed display window")

    p = intensity_summary(mods)
    s = new_slide(prs, "Pattern variation vs intensity-driven separation")
    put(s, p, Inches(0.5), Inches(1.0), Inches(6.2), Inches(5.0))
    tab = pd.DataFrame(summary_stats, columns=["modality", "method", "eta2", "rho"])
    lines = ["Intensity-explained fraction of variance by cluster (η²) and Spearman ρ of intensity with UMAP1 / UMAP2:"]
    for _, r in tab.iterrows():
        lines.append(f"  {r.modality:9s} {METHODS[r.method]:13s} η²={r.eta2:.2f}   ρ(UMAP1)={r.rho[0]:+.2f}  ρ(UMAP2)={r.rho[1]:+.2f}")
    tb = s.shapes.add_textbox(Inches(6.9), Inches(1.2), Inches(6.2), Inches(5))
    tb.text_frame.word_wrap = True
    for i, ln in enumerate(lines):
        pp = tb.text_frame.paragraphs[0] if i == 0 else tb.text_frame.add_paragraph()
        pp.text = ln; pp.font.size = Pt(11)
    tab.to_csv(FIG / "intensity_summary.csv", index=False)

    # checks slide
    fa = mods["paxillin"]
    cnt = fa.labels[fa.labels != ""].value_counts().reindex(FA5)
    unc = int((fa.idx.fa_label == "Uncertain").sum())
    text_slide(prs, "Checks / pending", [
        "Paxillin classes: 5 = No adhesion + 4 FA subclasses (Nascent Adhesion, focal complex, focal adhesion, fibrillar adhesion) — "
        "mapping matches FA5_ORDER in make_pptx_label_overview.py; please confirm.",
        "Counts shown here come from the B1 file labels_vinc_20260521.csv (Margaret + Annabel): "
        + ", ".join(f"{k}={int(v)}" for k, v in cnt.items()) + f"; {unc} 'Uncertain' left unlabelled.",
        "The B2 (Annabel, Sep 2026) label file was not available locally (/mnt/p/...), so B2 counts (1,224 patches) are not plotted.",
        f"Paxillin population: {len(fa.idx)} mr10 patches (control+ycomp); only {int((fa.labels != '').sum())} are annotated.",
        "Same population + UMAP settings across methods within each modality; features quantile-transformed (heavy-tailed CP/IL features).",
        "Intensity = mean of the cell-inside/outside-normalised patch, i.e. relative to the cell, not absolute camera counts.",
        "Actin / zyxin / MT are unannotated: KMeans per method with k chosen by silhouette (see 'choice of k' slides); each UMAP coloured by its own method's clusters.",
        "MT set: two acquisition groups (AcTub_bTub / bTub_AcTub) with possibly different ch1 targets — check before interpreting.",
        "Actin / MT / zyxin results test the workflow with new features, not transfer of the unchanged FA model.",
        "UMAP separation is not classification performance; paths are kNN-graph shortest paths, not temporal or biological transitions.",
        "ConAE (plain contrastive AE, no labels, latent 12): actin / zyxin / MT trained locally (200 epochs); paxillin = existing run contrastive_run/annabel_vinc_conae_combined (same 27,637 patches). SupCon-AE (label-supervised, cluster) is still pending for all modalities.",
    ])
    prs.save(PPTX)
    print(PPTX)


if __name__ == "__main__":
    main()
