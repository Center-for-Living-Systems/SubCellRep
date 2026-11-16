#!/usr/bin/env python3
"""
make_fig1_panels.py

Panel-ready plots for Figure 1 (local image patches as the unit of subcellular analysis).
  A  motivation / examples   B  dataset   C  baseline sampling   D  workflow
Every plot is saved as its own PNG (300 dpi) in results/fig1_panels/ and placed on slides in
results/fig1_panels.pptx.

Inputs: DS1 CZI frames + B1/B2 labels (dsicluster_CLS_rsync_folder), the Figure-4 patch
populations (patches32.npy, features_{cp,ae}32.csv, clusters_ae32.csv) and the Ernest MT TIFFs.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import czifile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.patches import Rectangle
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
FIG = REPO / "results" / "fig1_panels"
PPTX = REPO / "results" / "fig1_panels.pptx"
R = Path("/home/lding/lding/dsicluster_CLS_rsync_folder/data/fa_data_analysis")
AE = Path("/home/lding/lding/fa_data_analysis/ae_results")
PAXD, ACTD = AE / "fig4_fa_patches", AE / "ds1_actin_patches"
MTD = Path("/home/lding/lding/data/Ernest/mt_patches")
MTT = Path("/home/lding/lding/data/Ernest/mt_tiff")

UM_PER_PX = 0.1315305679563492       # DS1 CZI scaling (ScalingX = 1.3153e-7 m)
PS = 32
FA5 = ["No adhesion", "Nascent Adhesion", "focal complex", "focal adhesion", "fibrillar adhesion"]
FA5S = ["No Adh", "NA", "FC", "FA", "Fib"]
FA5_COL = {"No adhesion": "#9467bd", "Nascent Adhesion": "#1f77b4", "focal complex": "#ff7f0e",
           "focal adhesion": "#2ca02c", "fibrillar adhesion": "#d62728"}
COND_COL = {"control": "#d62728", "ycomp": "#2ca02c"}
TAB10, TAB20 = plt.get_cmap("tab10"), plt.get_cmap("tab20")

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "font.family": "DejaVu Sans", "savefig.dpi": 300})
PANELS: list[tuple[str, str, str, Path]] = []      # (panel letter, title, note, png)


def ccol(c, k):
    return TAB10(c) if k <= 10 else TAB20(c)


def save(fig, name, panel, title, note=""):
    FIG.mkdir(parents=True, exist_ok=True)
    p = FIG / f"{name}.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    PANELS.append((panel, title, note, p))
    print("saved", p.name, flush=True)


# ───────────────────────────── data ─────────────────────────────
_cache: dict = {}


def czi_path(cond, stem=None, f=None):
    d = R / "fa_data/031125data" / cond.capitalize()
    if stem is not None:
        return d / f"{stem}.czi"
    return sorted(d.glob("*.czi"))[f]


def frame(cond, stem=None, f=None):
    """(4, H, W) float, ch0 Cy5/vinculin, ch1 AF568/paxillin, ch2 AF488/zyxin, ch3 405/phalloidin(actin)."""
    p = czi_path(cond, stem, f)
    if p not in _cache:
        _cache[p] = czifile.imread(str(p)).squeeze().astype(np.float32) / 65025.0
    return _cache[p]


def disp(img, lo=0.5, hi=99.7, ref=None):
    ref = img if ref is None else ref
    a, b = np.percentile(ref, lo), np.percentile(ref, hi)
    return np.clip((img - a) / (b - a + 1e-9), 0, 1)


def scalebar(ax, um, px_um, x, y, color="white", fs=8, label=None):
    L = um / px_um
    ax.plot([x, x + L], [y, y], color=color, lw=2.5, solid_capstyle="butt")
    ax.text(x + L / 2, y - 6, label or f"{um:g} µm", color=color, ha="center", va="bottom", fontsize=fs)


def parse_pax(name):
    s = name.replace("ps32.tif", "")
    cond, rest = s.split("_f")
    f, x, y = int(rest[:4]), int(rest[5:9]), int(rest[10:14])
    return cond, f, x - 80, y - 80      # patch names carry an 80 px pad offset (verified by template matching)


def load_pax():
    idx = pd.read_csv(PAXD / "patches32_index.csv")
    p = idx.name.map(parse_pax)
    idx["cond"] = [t[0] for t in p]
    idx["f"] = [t[1] for t in p]
    idx["x0"] = [t[2] for t in p]
    idx["y0"] = [t[3] for t in p]
    idx["img"] = idx.cond + "_f" + idx.f.map("{:04d}".format)
    P = np.load(PAXD / "patches32.npy", mmap_mode="r")
    ae = pd.read_csv(PAXD / "features_ae32.csv").set_index("name").loc[idx.name]
    cp = pd.read_csv(PAXD / "features_cp32.csv").set_index("name").loc[idx.name]
    b2 = pd.read_csv(R / "labelling/vinc_combined_label_Annabel_20260911.csv").drop_duplicates("filename", keep="last")
    idx["b2"] = idx.name.map(b2.set_index("filename").label)
    b1 = pd.read_csv(R / "labelling/labels_vinc_20260521.csv")
    b1["name"] = b1.condition + "_" + b1.crop_img_filename
    b1 = b1.drop_duplicates("name", keep="last")
    idx["b1"] = idx.name.map(b1.set_index("name").classification)
    idx["b1_czi"] = idx.name.map(b1.set_index("name").czi_filename)
    return idx, P, ae.values.astype(np.float64), np.nan_to_num(cp.values.astype(np.float64)), b1


PAX, PAXP, AEZ, CPF, B1ALL = load_pax()
B2 = PAX[PAX.b2.notna()].copy()


def patch_strip(ax, patches, ncol, vmin, vmax, pad=2):
    n = len(patches)
    nrow = (n + ncol - 1) // ncol
    H = nrow * (PS + pad) - pad
    W = ncol * (PS + pad) - pad
    canvas = np.ones((H, W)) * np.nan
    for i, p in enumerate(patches):
        r, c = divmod(i, ncol)
        canvas[r * (PS + pad):r * (PS + pad) + PS, c * (PS + pad):c * (PS + pad) + PS] = np.asarray(p, np.float32)
    cm = plt.get_cmap("gray").copy()
    cm.set_bad("white")
    ax.imshow(canvas, cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.axis("off")


def nearest(Z, idxs, centre, n):
    d = np.linalg.norm(Z[idxs] - centre, axis=1)
    return idxs[np.argsort(d)[:n]]


# ───────────────────────────── A ─────────────────────────────
def a_examples():
    """Plain images (no overlays): paxillin, actin and zyxin from three different DS1 cells, microtubules from a large Ernest cell."""
    ai = pd.read_csv(ACTD / "patches32_index.csv")
    ai["stem"] = ai.name.str.rsplit("_y", n=1).str[0]
    cnt = ai[ai.file == "Control"].stem.value_counts()
    stems = list(cnt.index)
    st_pax = czi_path("control", f=int(EXAMPLE.split("_f")[1])).stem
    others = [t for t in stems if t != st_pax]
    st_act, st_zyx = others[3], others[8]
    # largest MT cell (segmented area), fully inside the frame
    import extract_mt_patches as emp
    mi = pd.read_csv(MTD / "patches32_index.csv")
    mi["stem"] = mi.name.str.rsplit("_y", n=1).str[0]
    best, barea = None, 0
    for st in mi.stem.value_counts().index[:80]:
        t = tifffile.imread(str(MTT / mi[mi.stem == st].file.iloc[0] / f"{st}.tif")).astype(np.float32)
        lab_ = emp.segment(t[0])
        if lab_.max() == 0:
            continue
        seg = lab_ == (np.bincount(lab_.ravel())[1:].argmax() + 1)   # largest single cell region
        ys, xs = np.where(seg)
        if len(ys) == 0 or ys.min() < 10 or xs.min() < 10 or ys.max() > seg.shape[0] - 10 or xs.max() > seg.shape[1] - 10:
            continue
        if seg.sum() > barea:
            best, barea, tbest, sbest = st, seg.sum(), t, seg
    ys, xs = np.where(sbest)
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    half = max(ys.max() - ys.min(), xs.max() - xs.min()) // 2 + 40
    y1, x1 = max(0, cy - half), max(0, cx - half)
    win = tbest[1][y1:cy + half, x1:cx + half]
    ds1 = [("Paxillin", frame("control", stem=st_pax)[1], st_pax), ("Actin (phalloidin)", frame("control", stem=st_act)[3], st_act),
           ("Zyxin", frame("control", stem=st_zyx)[2], st_zyx)]
    fig, ax = plt.subplots(1, 4, figsize=(20, 5.2))
    for a_, (t, im, st) in zip(ax, ds1):
        a_.imshow(disp(im), cmap="gray")
        a_.set_title(t + "  (DS1 " + st.rsplit("_", 1)[1] + ")")
        scalebar(a_, 20, UM_PER_PX, 40, 990)
    ax[3].imshow(disp(win, 1, 99.8), cmap="gray")
    ax[3].set_title(f"Microtubules (Ernest U2OS, {best.split('U2OS_')[-1]})")
    for a_ in ax:
        a_.axis("off")
    fig.tight_layout()
    save(fig, "A1_four_modalities", "A", "Paxillin, actin, zyxin and microtubule images",
         f"Paxillin/actin/zyxin: three different control cells of DS1 ({st_pax[-10:]}, {st_act[-10:]}, {st_zyx[-10:]}). "
         f"MT: largest fully-in-frame cell among the Ernest frames ({best}, mask {int(barea)} px; scale not shown). Plain images, autoscaled.")


def a_fa_classes():
    lab = B2[B2.b2.isin(FA5)]
    cen = {c: AEZ[lab[lab.b2 == c].index].mean(0) for c in FA5}
    sd = AEZ[lab.index].std(0) + 1e-9
    Zs = AEZ / sd
    ncol = 12
    allp = []
    rows = []
    for c in FA5:
        ids = lab[lab.b2 == c].index.values
        pick = nearest(Zs, ids, cen[c] / sd, ncol)
        rows.append(pick)
        allp += [PAXP[i] for i in pick]
    vmin, vmax = np.percentile(np.stack(allp).astype(np.float32), [1, 99.7])
    fig, ax = plt.subplots(5, 1, figsize=(8.5, 3.9), gridspec_kw={"hspace": 0.04})
    for a, c, pick in zip(ax, FA5, rows):
        patch_strip(a, [PAXP[i] for i in pick], ncol, vmin, vmax)
        a.text(-0.01, 0.5, c, transform=a.transAxes, ha="right", va="center", color=FA5_COL[c], fontsize=10, fontweight="bold")
    save(fig, "A2_fa_class_examples", "A", "Different FA types are recognisable as local patterns",
         "12 B2-labelled patches per class closest to the class mean in the ConAE latent space (representative examples). Shared intensity scale.")


def cluster_panel(D, title, name, note, per=12):
    P = np.load(D / "patches32.npy", mmap_mode="r")
    cl = pd.read_csv(D / "clusters_ae32.csv")
    k = cl.cluster.max() + 1
    sel = [nearest(np.stack([cl.dist.values] * 1, 1), np.where(cl.cluster == c)[0], np.zeros(1), per) for c in range(k)]
    allp = np.stack([np.asarray(P[i], np.float32) for s in sel for i in s])
    vmin, vmax = np.percentile(allp, [1, 99.7])
    fig, ax = plt.subplots(k, 1, figsize=(8.5, 0.95 * k + 0.6))
    for c, a in enumerate(np.atleast_1d(ax)):
        patch_strip(a, [P[i] for i in sel[c]], per, vmin, vmax)
        a.set_title(f"ConAE cluster {c}  (n={int((cl.cluster == c).sum())})", loc="left",
                    color=ccol(c, k), fontsize=9, fontweight="bold", pad=2)
    fig.tight_layout()
    save(fig, name, "A", title, note)


def a_clusters():
    cluster_panel(ACTD, "Actin: recurring local patterns (ConAE clusters)", "A3_actin_conae_clusters",
                  "DS1 actin (ch3) 32 px patches; k chosen by silhouette; 12 patches closest to each ConAE cluster centre.")
    cluster_panel(MTD, "Microtubules: recurring local patterns (ConAE clusters)", "A4_mt_conae_clusters",
                  "Ernest U2OS DMSO microtubules (ch1) 32 px patches, ConAE trained locally; 12 patches nearest each cluster centre.")


def a_segmentation():
    """Same dense lamella region: object masks change completely with the threshold, patches stay labelled."""
    from skimage.filters import gaussian, threshold_otsu
    from skimage.measure import label as cclabel
    cond, f = EXAMPLE.split("_f")
    f = int(f)
    img = frame(cond, f=f)[1]
    sub = B2[B2.img == EXAMPLE]
    best, score = None, -1
    for y in range(0, 1024 - 192, 32):
        for x in range(0, 1024 - 192, 32):
            m = sub[(sub.x0 >= x) & (sub.x0 + PS <= x + 192) & (sub.y0 >= y) & (sub.y0 + PS <= y + 192)]
            s = m.b2.nunique() * 10 + len(m)
            if s > score:
                best, score = (x, y), s
    x, y = best
    crop = img[y:y + 192, x:x + 192]
    sm = gaussian(crop, 1)
    t0 = threshold_otsu(sm)
    fig, ax = plt.subplots(1, 5, figsize=(18, 3.9))
    ax[0].imshow(disp(crop, ref=img), cmap="gray")
    ax[0].set_title("Paxillin crop (192 px)")
    scalebar(ax[0], 5, UM_PER_PX, 8, 184)
    for a, fct in zip(ax[1:4], (0.6, 1.0, 1.5)):
        m = sm > fct * t0
        lab = cclabel(m)
        a.imshow(disp(crop, ref=img), cmap="gray")
        a.contour(m, [0.5], colors="cyan", linewidths=0.7)
        a.set_title(f"threshold {fct:g} x Otsu -> {lab.max()} objects")
    ax[4].imshow(disp(crop, ref=img), cmap="gray")
    for _, r in sub.iterrows():
        if x <= r.x0 and r.x0 + PS <= x + 192 and y <= r.y0 and r.y0 + PS <= y + 192:
            ax[4].add_patch(Rectangle((r.x0 - x, r.y0 - y), PS, PS, fill=False, ec=FA5_COL[r.b2], lw=1.3))
    ax[4].set_title("32 px patches + class label")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    save(fig, "A5_segmentation_ambiguity", "A",
         "Local appearance is recognisable even when object boundaries are uncertain",
         "Gaussian-smoothed paxillin thresholded at 0.6/1/1.5 x Otsu: object count and shape change with the threshold, "
         "while the patch (right) can still be assigned a class. Illustrative single region.")


# ───────────────────────────── B ─────────────────────────────
def b_multichannel():
    cond_ex = {"control": EXAMPLE.split("_f")[0]}
    yc = B2[B2.cond == "ycomp"].img.value_counts().index[0]
    items = [("control", int(EXAMPLE.split("_f")[1])), ("ycomp", int(yc.split("_f")[1]))]
    names = ["Paxillin (AF568)", "Vinculin (Cy5)", "Zyxin (AF488)", "Actin (phalloidin 405)", "Merge pax/zyxin/actin"]
    fig, ax = plt.subplots(2, 5, figsize=(17, 7))
    for r, (cond, f) in enumerate(items):
        im = frame(cond, f=f)
        chs = [disp(im[1]), disp(im[0]), disp(im[2]), disp(im[3])]
        for c, (a, g) in enumerate(zip(ax[r, :4], chs)):
            a.imshow(g, cmap="gray")
            if r == 0:
                a.set_title(names[c])
        ax[r, 4].imshow(np.dstack([chs[0], chs[2], chs[3]]))
        if r == 0:
            ax[r, 4].set_title(names[4])
        ax[r, 0].text(-0.04, 0.5, {"control": "Control", "ycomp": "Y-compound"}[cond], transform=ax[r, 0].transAxes,
                      rotation=90, va="center", ha="right", fontsize=13, fontweight="bold", color=COND_COL[cond])
        scalebar(ax[r, 0], 20, UM_PER_PX, 40, 990)
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    save(fig, "B1_multichannel_conditions", "B", "Multichannel dataset, two conditions",
         "DS1: 4-channel confocal CZI (1024x1024, 0.1315 µm/px). One example cell per condition; each channel autoscaled (0.5-99.7 pct).")


def b_classes_counts():
    n_img = {c: len(list((R / "fa_data/031125data" / c).glob("*.czi"))) for c in ("Control", "Ycomp")}
    n_pat = PAX.cond.value_counts()
    fig = plt.figure(figsize=(14, 4.6))
    g = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 2.4])
    a0 = fig.add_subplot(g[0])
    a0.bar(["Control", "Y-comp"], [n_img["Control"], n_img["Ycomp"]], color=[COND_COL["control"], COND_COL["ycomp"]])
    for i, v in enumerate([n_img["Control"], n_img["Ycomp"]]):
        a0.text(i, v + 0.5, str(v), ha="center")
    a0.set_title(f"Images ({sum(n_img.values())} total)")
    a0.set_ylabel("cells / images")
    a1 = fig.add_subplot(g[1])
    a1.bar(["Control", "Y-comp"], [n_pat["control"], n_pat["ycomp"]], color=[COND_COL["control"], COND_COL["ycomp"]])
    for i, v in enumerate([n_pat["control"], n_pat["ycomp"]]):
        a1.text(i, v + 200, f"{v:,}", ha="center")
    a1.set_title(f"32 px patches ({len(PAX):,} total)")
    # class tree
    a2 = fig.add_subplot(g[2])
    a2.axis("off")
    a2.set_xlim(0, 10)
    a2.set_ylim(0, 6)
    cnt = B2.b2.value_counts()
    n_ad = int(sum(cnt.get(c, 0) for c in FA5[1:]))
    box = lambda x, y, t, col: a2.text(x, y, t, fontsize=10, fontweight="bold", color=col, ha="center", va="center",
                                       bbox=dict(fc="white", ec=col, boxstyle="round,pad=0.35"))
    box(1.2, 3.0, "B2 labelled\npatches\n(n=%d)" % len(B2[B2.b2.isin(FA5)]), "k")
    box(4.6, 5.0, "No adhesion\nn=%d" % cnt.get("No adhesion", 0), FA5_COL["No adhesion"])
    box(4.6, 2.2, "Adhesion\nn=%d" % n_ad, "k")
    a2.plot([2.1, 3.7], [3.2, 5.0], "k", lw=1)
    a2.plot([2.1, 3.7], [2.8, 2.2], "k", lw=1)
    for i, c in enumerate(FA5[1:]):
        yy = 5.0 - i * 1.4
        box(8.3, yy, "%s\nn=%d" % (c, cnt.get(c, 0)), FA5_COL[c])
        a2.plot([5.5, 7.3], [2.2, yy], "k", lw=0.8)
    a2.set_title("FA label hierarchy (B2)")
    fig.tight_layout()
    save(fig, "B2_dataset_numbers_label_tree", "B", "Benchmark dataset of an FA test case",
         "Images from the CZI folders (control 50, Y-compound 41); patches = non-overlapping 32 px grid patches with >=10 % in the cell mask "
         "(Figure-4 paxillin population). Label tree: no adhesion vs four adhesion types (B2 counts).")


def b_label_tree_b1():
    b1 = B1ALL
    n_unc = int((~b1.classification.isin(FA5)).sum())
    b1 = b1[b1.classification.isin(FA5)]

    def nt(m):
        return "%d\n(%dC / %dY)" % (m.sum(), (m & (b1.condition == "control")).sum(), (m & (b1.condition == "ycomp")).sum())

    fig, a2 = plt.subplots(figsize=(8.2, 4.8))
    a2.axis("off")
    a2.set_xlim(0, 10.6)
    a2.set_ylim(0, 6)
    box = lambda x, y, t, col: a2.text(x, y, t, fontsize=10, fontweight="bold", color=col, ha="center", va="center",
                                       bbox=dict(fc="white", ec=col, boxstyle="round,pad=0.35"))
    cls = b1.classification
    box(1.3, 3.0, "B1 labelled\npatches\n" + nt(cls.isin(FA5)), "k")
    box(4.8, 5.0, "No adhesion\n" + nt(cls == "No adhesion"), FA5_COL["No adhesion"])
    box(4.8, 2.2, "Adhesion\n" + nt(cls.isin(FA5[1:])), "k")
    a2.plot([2.3, 3.8], [3.2, 5.0], "k", lw=1)
    a2.plot([2.3, 3.8], [2.8, 2.2], "k", lw=1)
    for i, c in enumerate(FA5[1:]):
        yy = 5.0 - i * 1.4
        box(9.0, yy, c + "\n" + nt(cls == c), FA5_COL[c])
        a2.plot([5.75, 7.95], [2.2, yy], "k", lw=0.8)
    a2.set_title("FA label hierarchy (B1)   n (control C / Y-compound Y)")
    save(fig, "B3_label_tree_b1", "B", "FA label hierarchy with the B1 label set",
         f"B1 labels excluding {n_unc} 'Uncertain'. Control {int((b1.condition == 'control').sum())}, Y-compound {int((b1.condition == 'ycomp').sum())}.")
    print(pd.crosstab(b1.classification, b1.condition, margins=True), "uncertain", n_unc,
          pd.crosstab(B1ALL.classification, B1ALL.condition, margins=True).loc["All"].to_dict())


def label_table(name, title, cls, cond, note):
    """Hierarchy as a table: No adhesion / Adhesion (4 types, indented) / total, columns Control, Y-compound, Total."""
    cls, cond = pd.Series(cls.values), pd.Series(cond.values)
    ok = cls.isin(FA5)
    cls, cond = cls[ok], cond[ok]

    def row(label, m):
        c, y = int((m & (cond == "control")).sum()), int((m & (cond == "ycomp")).sum())
        return [label, f"{c:,}", f"{y:,}", f"{c + y:,}"]

    rows = [("No adhesion", row("No adhesion", cls == "No adhesion"), FA5_COL["No adhesion"], False),
            ("Adhesion", row("Adhesion (4 types)", cls.isin(FA5[1:])), "k", True)]
    for c in FA5[1:]:
        rows.append((c, row("      " + c, cls == c), FA5_COL[c], False))
    rows.append(("Total", row("All labelled patches", cls.isin(FA5)), "k", True))
    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    ax.axis("off")
    t = ax.table(cellText=[r[1] for r in rows], colLabels=["", "Control", "Y-compound", "Total"], loc="center",
                 cellLoc="center", colWidths=[0.42, 0.18, 0.22, 0.18])
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    t.scale(1, 1.55)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#bbbbbb")
        if r == 0:
            cell.set_facecolor("#1F4E79")
            cell.set_text_props(color="white", fontweight="bold")
            continue
        _, _, col, bold = rows[r - 1]
        if c == 0:
            cell.set_text_props(color=col, fontweight="bold", ha="left")
            cell._loc = "left"
        if bold:
            cell.set_facecolor("#eef2f7")
            cell.set_text_props(fontweight="bold", **({"color": col} if c == 0 else {}))
    ax.set_title(title, fontsize=11, fontweight="bold")
    save(fig, name, "B", title, note)


def b_label_tables():
    b1 = B1ALL
    n_unc = int((~b1.classification.isin(FA5)).sum())
    label_table("B3_label_table_b1", "FA label counts: B1", b1.classification, b1.condition,
                f"B1 labels, excluding {n_unc} 'Uncertain'.")
    label_table("B2_label_table_b2", "FA label counts: B2", B2.b2, B2.cond,
                "B2 labels (Annabel, 11 Sep 2026 file).")


# ───────────────────────────── C ─────────────────────────────
def c_grid_overlay():
    cond, f = EXAMPLE.split("_f")
    f = int(f)
    img = frame(cond, f=f)[1]
    sub = PAX[PAX.img == EXAMPLE]
    fig, ax = plt.subplots(1, 2, figsize=(14, 6.8), gridspec_kw={"width_ratios": [1.5, 1]})
    ax[0].imshow(disp(img), cmap="gray")
    for _, r in sub.iterrows():
        ax[0].add_patch(Rectangle((r.x0, r.y0), PS, PS, fill=False, ec="#ffd400", lw=0.35, alpha=0.9))
    ax[0].set_title(f"Grid sampling: {len(sub)} non-overlapping 32 px patches over the cell (>=10 % in cell mask)")
    scalebar(ax[0], 20, UM_PER_PX, 40, 990)
    x, y, w = 560, 480, 192
    ax[1].imshow(disp(img)[y:y + w, x:x + w], cmap="gray", extent=(x, x + w, y + w, y))
    for _, r in sub.iterrows():
        if x <= r.x0 < x + w and y <= r.y0 < y + w:
            ax[1].add_patch(Rectangle((r.x0, r.y0), PS, PS, fill=False, ec="#ffd400", lw=0.9))
    ax[1].set_title(f"Zoom: 1 patch = 32 px = {PS * UM_PER_PX:.1f} µm")
    ax[0].add_patch(Rectangle((x, y), w, w, fill=False, ec="cyan", lw=1.5))
    ax[1].set_xlim(x, x + w)
    ax[1].set_ylim(y + w, y)
    ax[1].plot([x + 8, x + 8 + 5 / UM_PER_PX], [y + w - 8] * 2, color="white", lw=3)
    ax[1].text(x + 8 + 2.5 / UM_PER_PX, y + w - 12, "5 µm", color="white", ha="center", fontsize=8)
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    save(fig, "C1_grid_sampling_overlay", "C", "Baseline sampling: paxillin with grid sampling region",
         f"{EXAMPLE}. Patch = 32x32 px = {PS * UM_PER_PX:.2f} µm at the DS1 pixel size (0.1315 µm/px). "
         "NOTE: the outline text says 3 µm; the CZI metadata gives 4.2 µm for 32 px.")


def c_grid_left_only():
    cond, f = EXAMPLE.split("_f")
    img = frame(cond, f=int(f))[1]
    sub = PAX[PAX.img == EXAMPLE]
    fig, a = plt.subplots(figsize=(7.5, 7.5))
    a.imshow(disp(img), cmap="gray")
    for _, r in sub.iterrows():
        a.add_patch(Rectangle((r.x0, r.y0), PS, PS, fill=False, ec="#ffd400", lw=0.35, alpha=0.9))
    scalebar(a, 20, UM_PER_PX, 40, 990)
    a.axis("off")
    fig.tight_layout()
    save(fig, "C1_grid_sampling_overlay_left", "C", "Baseline sampling: paxillin with grid sampling region (overview only)",
         f"{EXAMPLE}: {len(sub)} non-overlapping 32 px patches (4.2 µm) with >=10 % in the cell mask.")


def c_orientation():
    """CP feature stability under in-plane rotation (centre 24 px crop of the rotated patch vs the same crop of the original)."""
    import extract_cp_features as cp
    from scipy.ndimage import rotate
    rng = np.random.default_rng(0)
    ids = rng.choice(len(PAX), 500, replace=False)
    nint = len(cp.INTENSITY_NAMES)
    C = slice(4, 28)
    base = [np.asarray(PAXP[i], np.float32) for i in ids]

    def feats(ps):
        return np.stack([np.concatenate([cp.intensity_features(np.ascontiguousarray(p)), cp.haralick_features(np.ascontiguousarray(p))])
                         for p in ps])

    F0 = feats([p[C, C] for p in base])
    angles = [15, 30, 45, 90]
    res = {}
    for ang in angles:
        F1 = feats([rotate(p, ang, reshape=False, order=1, mode="reflect")[C, C] for p in base])
        r = []
        for j in range(F0.shape[1]):
            a_, b_ = F0[:, j], F1[:, j]
            ok = np.isfinite(a_) & np.isfinite(b_)
            r.append(np.corrcoef(a_[ok], b_[ok])[0, 1] if a_[ok].std() > 0 and b_[ok].std() > 0 else np.nan)
        res[ang] = np.array(r)
    ex = ids[0]
    for i in ids:
        if PAX.b2[i] == "focal adhesion":
            ex = i
            break
    p = np.asarray(PAXP[ex], np.float32)
    fig = plt.figure(figsize=(14, 3.6))
    g = fig.add_gridspec(1, 7, width_ratios=[1, 1, 1, 1, 1, 0.25, 3.3])
    tiles = [("original", p)] + [(f"rot {a}°", rotate(p, a, reshape=False, order=1, mode="reflect")) for a in angles]
    for i, (t, q) in enumerate(tiles):
        ax_ = fig.add_subplot(g[i])
        ax_.imshow(q, cmap="gray", vmin=p.min(), vmax=p.max())
        ax_.set_title(t, fontsize=9)
        ax_.axis("off")
    ax_ = fig.add_subplot(g[6])
    for k, ang in enumerate(angles):
        for off, sl, col in ((-0.18, slice(0, nint), "#1f77b4"), (0.18, slice(nint, None), "#ff7f0e")):
            v = res[ang][sl][np.isfinite(res[ang][sl])]
            ax_.boxplot(v, positions=[k + off], widths=0.3, patch_artist=True, showfliers=False,
                        boxprops=dict(facecolor=col, alpha=0.6), medianprops=dict(color="k"))
    ax_.set_xticks(range(len(angles)))
    ax_.set_xticklabels([f"{a}°" for a in angles])
    ax_.set_xlabel("in-plane rotation (bilinear, centre 24 px crop)")
    ax_.set_ylabel("corr(original, rotated)\nper CP feature")
    ax_.set_ylim(0.7, 1.02)
    ax_.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc="#1f77b4", alpha=0.6), plt.Rectangle((0, 0), 1, 1, fc="#ff7f0e", alpha=0.6)],
               labels=[f"intensity ({nint})", f"Haralick ({len(res[15]) - nint})"], loc="lower left", fontsize=8)
    ax_.set_title("Rotation sensitivity of CP features (500 random patches)")
    fig.tight_layout()
    med = {a: (np.nanmedian(res[a][:nint]), np.nanmedian(res[a][nint:])) for a in angles}
    print("orientation medians (intensity, haralick):", med, flush=True)
    save(fig, "C3_orientation_invariance", "C", "Orientation: rotation sensitivity of texture features (not fully resolved)",
         "Per-feature Pearson r between the CP features of a patch centre crop and of the rotated patch. Median (intensity, Haralick): " +
         "; ".join(f"{a}° ({m[0]:.2f}, {m[1]:.2f})" for a, m in med.items()) +
         ". ConAE saw random flips and ±15° rotations in training, so it is only approximately invariant.")


def c_grid_vs_localmax():
    cond, f = EXAMPLE.split("_f")
    f = int(f)
    img = frame(cond, f=f)[1]
    lm = pd.read_csv(R / "ae_results/local_max_sampling/ds1/mask10/local_max_ps32.csv")
    lm = lm[(lm.condition == cond) & (lm.frame_idx == f)]
    gr = PAX[PAX.img == EXAMPLE]
    x, y, w = 480, 420, 320
    fig, ax = plt.subplots(1, 2, figsize=(12, 6.2))
    for a, ttl in zip(ax, (f"Grid: {len(gr)} patches / cell", f"Local maxima: {len(lm)} patches / cell")):
        a.imshow(disp(img)[y:y + w, x:x + w], cmap="gray", extent=(x, x + w, y + w, y))
        a.set_title(ttl)
        a.axis("off")
    for _, r in gr.iterrows():
        if x <= r.x0 < x + w and y <= r.y0 < y + w:
            ax[0].add_patch(Rectangle((r.x0, r.y0), PS, PS, fill=False, ec="#ffd400", lw=0.9))
    for _, r in lm.iterrows():
        if x <= r.x0 < x + w and y <= r.y0 < y + w:
            ax[1].add_patch(Rectangle((r.x0, r.y0), PS, PS, fill=False, ec="cyan", lw=0.9))
            ax[1].plot(r.cx, r.cy, "r.", ms=3)
    for a in ax:
        a.set_xlim(x, x + w)
        a.set_ylim(y + w, y)
    fig.tight_layout()
    save(fig, "C4_grid_vs_local_maximum", "C", "Sampling strategy: grid vs local maximum (not fully resolved)",
         f"{EXAMPLE}, same 320 px window. Local maxima (red dot = peak) from the existing mask10 sampling run (patches may overlap and cluster on bright FAs); "
         "grid = non-overlapping, unbiased towards intensity. Counts in titles are for the whole cell.")


def c_label_hist():
    b1 = B1ALL
    b1 = b1[b1.classification.isin(FA5)]
    b2 = B2[B2.b2.isin(FA5)]
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.2))
    for r, (nm, d, col, imgc) in enumerate((("B1 (Margaret etc., 89 images)", b1, "classification", "czi_filename"),
                                             ("B2 (Annabel, higher quality)", None, "b2", "img"))):
        if d is None:
            d = b2
            cond = d.cond
        else:
            cond = d.condition
        ct = pd.crosstab(d[col], cond).reindex(FA5).fillna(0)
        bot = np.zeros(len(FA5))
        for c in ("control", "ycomp"):
            if c in ct:
                ax[r, 0].bar(range(5), ct[c], bottom=bot, color=COND_COL[c], label={"control": "Control", "ycomp": "Y-comp"}[c])
                bot += ct[c].values
        for i, v in enumerate(bot):
            ax[r, 0].text(i, v + 5, int(v), ha="center", fontsize=8)
        ax[r, 0].set_xticks(range(5))
        ax[r, 0].set_xticklabels(FA5S)
        ax[r, 0].set_ylabel("labelled patches")
        ax[r, 0].set_title(f"{nm}: {len(d):,} patches")
        ax[r, 0].legend(fontsize=8)
        per = d.groupby(imgc).size()
        ax[r, 1].hist(per.values, bins=20, color="#555555")
        ax[r, 1].set_xlabel("labelled patches per image")
        ax[r, 1].set_ylabel("images")
        ax[r, 1].set_title(f"{len(per)} images, median {int(per.median())} / image")
    fig.tight_layout()
    save(fig, "C5_label_histograms", "C", "Labels: class histogram and labels per image",
         "B1 counts exclude 'Uncertain' (30) and are restricted to patches present in the Figure-4 population. "
         "The outline's '1.3k labels on 89 images' matches B1 (1,340 rows incl. Uncertain, 89 images); B2 is 1,739 on 27 images.")


# ───────────────────────────── D ─────────────────────────────
def classify():
    """Image-grouped CV (B2 labels) with LightGBM on ConAE latents and on CP features; predictions for every patch."""
    import lightgbm as lgb
    lab = B2[B2.b2.isin(FA5)]
    y = lab.b2.map({c: i for i, c in enumerate(FA5)}).values
    grp = lab.img.values
    out = {}
    for nm, X in (("ConAE", AEZ), ("CP", CPF)):
        Xl = X[lab.index]
        pred = np.zeros(len(lab), int)
        for tr, te in GroupKFold(5).split(Xl, y, grp):
            m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, class_weight="balanced",
                                   verbose=-1, random_state=0).fit(Xl[tr], y[tr])
            pred[te] = m.predict(Xl[te])
        out[nm] = (balanced_accuracy_score(y, pred), (pred == y).mean())
    best = max(out, key=lambda k: out[k][0])
    X = AEZ if best == "ConAE" else CPF
    full = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, class_weight="balanced",
                              verbose=-1, random_state=0).fit(X[lab.index], y)
    P = full.predict_proba(X)
    # labelled images: fold model that never saw the image
    imgs = lab.img.unique()
    folds = list(GroupKFold(5).split(lab.index, y, grp))
    for tr, te in folds:
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, class_weight="balanced",
                               verbose=-1, random_state=0).fit(X[lab.index[tr]], y[tr])
        held = np.unique(grp[te])
        rows = PAX.index[PAX.img.isin(held)]
        P[rows] = m.predict_proba(X[rows])
    return P.argmax(1), P, out, best


def d_labelled_example():
    cond, f = EXAMPLE.split("_f")
    f = int(f)
    img = frame(cond, f=f)[1]
    sub = B2[B2.img == EXAMPLE]
    fig, ax = plt.subplots(1, 2, figsize=(14, 6.4), gridspec_kw={"width_ratios": [1.5, 1]})
    ax[0].imshow(disp(img), cmap="gray")
    for _, r in sub.iterrows():
        ax[0].add_patch(Rectangle((r.x0, r.y0), PS, PS, fill=False, ec=FA5_COL[r.b2], lw=1.1))
    ax[0].set_title(f"Expert patch labels on one image ({len(sub)} labelled patches)")
    ax[0].legend(handles=[Rectangle((0, 0), 1, 1, fc="none", ec=FA5_COL[c], lw=1.5) for c in FA5], labels=FA5S,
                 loc="lower left", fontsize=8, ncol=5, framealpha=0.8)
    x, y, w = 480, 420, 256
    ax[1].imshow(disp(img)[y:y + w, x:x + w], cmap="gray", extent=(x, x + w, y + w, y))
    for _, r in sub.iterrows():
        if x <= r.x0 < x + w and y <= r.y0 < y + w:
            ax[1].add_patch(Rectangle((r.x0, r.y0), PS, PS, fill=False, ec=FA5_COL[r.b2], lw=1.6))
    ax[1].set_xlim(x, x + w)
    ax[1].set_ylim(y + w, y)
    ax[1].set_title("Zoom")
    ax[0].add_patch(Rectangle((x, y), w, w, fill=False, ec="cyan", lw=1.2))
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    save(fig, "D1_labelled_example", "D", "Workflow step 1: patch labelling (example result)",
         f"{EXAMPLE}, B2 labels. Interface not shown, only the labelling result.")


def d_prediction_map(pred, Pm, accs, best):
    cond, f = EXAMPLE.split("_f")
    f = int(f)
    img = frame(cond, f=f)[1]
    sub = PAX[PAX.img == EXAMPLE]
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(disp(img), cmap="gray")
    ax[0].set_title("Paxillin image")
    ax[1].imshow(disp(img), cmap="gray")
    for i, r in sub.iterrows():
        ax[1].add_patch(Rectangle((r.x0, r.y0), PS, PS, fc=FA5_COL[FA5[pred[i]]], ec="none", alpha=0.55))
    ax[1].set_title(f"Spatial prediction map ({best} + LightGBM, image held out of training)")
    ax[2].imshow(disp(img), cmap="gray")
    n_ok = 0
    for i, r in sub.iterrows():
        if isinstance(r.b2, str):
            ok = FA5[pred[i]] == r.b2
            n_ok += ok
            ax[2].add_patch(Rectangle((r.x0, r.y0), PS, PS, fill=False, ec="#00cc00" if ok else "#ff0000", lw=1.1))
    nl = sub.b2.notna().sum()
    ax[2].set_title(f"Check vs. expert labels: {n_ok}/{nl} correct (green) / wrong (red)")
    ax[1].legend(handles=[Rectangle((0, 0), 1, 1, fc=FA5_COL[c], alpha=0.7) for c in FA5], labels=FA5S, loc="lower left",
                 fontsize=8, ncol=5, framealpha=0.8)
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    save(fig, "D3_spatial_prediction_map", "D", "Workflow step 4: spatial prediction map",
         "Classifier trained on B2 labels from other images (5-fold image-grouped CV; this image's fold model never saw it). "
         f"Group-CV balanced accuracy over all B2 labels: " +
         ", ".join(f"{k} {v[0]:.2f}" for k, v in accs.items()) + f" (5 classes; chance 0.20). Using {best}.")


def d_composition(pred):
    sub = PAX[PAX.img == EXAMPLE]
    fr = np.bincount(pred[sub.index], minlength=5) / len(sub)
    fig = plt.figure(figsize=(15, 4.8))
    g = fig.add_gridspec(1, 3, width_ratios=[1.0, 3.0, 1.2])
    a = fig.add_subplot(g[0])
    a.bar(range(5), fr * 100, color=[FA5_COL[c] for c in FA5])
    a.set_xticks(range(5))
    a.set_xticklabels(FA5S)
    a.set_ylabel("% of patches in the cell")
    a.set_title("Example cell composition")
    # per-image stacked bars
    imgs = PAX.groupby("img").size().index
    comp = pd.DataFrame({im: np.bincount(pred[PAX.index[PAX.img == im]], minlength=5) /
                         (PAX.img == im).sum() for im in imgs}).T
    comp["cond"] = [i.split("_f")[0] for i in comp.index]
    comp = comp.sort_values(["cond", 3, 0])
    b = fig.add_subplot(g[1])
    bot = np.zeros(len(comp))
    for k, c in enumerate(FA5):
        b.bar(range(len(comp)), comp[k].values * 100, bottom=bot, color=FA5_COL[c], width=1.0, label=FA5S[k])
        bot += comp[k].values * 100
    n_c = int((comp.cond == "control").sum())
    b.axvline(n_c - 0.5, color="k", lw=1)
    b.text(n_c / 2, 103, "Control", ha="center", color=COND_COL["control"], fontweight="bold")
    b.text(n_c + (len(comp) - n_c) / 2, 103, "Y-compound", ha="center", color=COND_COL["ycomp"], fontweight="bold")
    b.set_xlim(-0.5, len(comp) - 0.5)
    b.set_ylim(0, 110)
    b.set_xlabel(f"images (n={len(comp)})")
    b.set_ylabel("% of patches")
    b.set_title("Per-cell FA-class composition")
    b.legend(ncol=5, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    c2 = fig.add_subplot(g[2])
    for cond in ("control", "ycomp"):
        m = comp[comp.cond == cond][list(range(5))].mean().values * 100
        c2.plot(range(5), m, "o-", color=COND_COL[cond], label={"control": "Control", "ycomp": "Y-comp"}[cond])
    c2.set_xticks(range(5))
    c2.set_xticklabels(FA5S)
    c2.set_ylabel("mean % per image")
    c2.set_title("Condition mean")
    c2.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "D4_cell_composition", "D", "Workflow step 5: final cell representation = FA-class composition per image",
         "Fractions of predicted classes over all grid patches of an image. Predictions for labelled images are out-of-image; "
         "unlabelled images use the model trained on all B2 labels. Predicted fractions are not independently validated.")


def phate_cp(n=8000):
    """PHATE of standardised CP features; all labelled (B1/B2) patches + random unlabelled up to n. Cached."""
    import phate
    from sklearn.preprocessing import StandardScaler
    cache_i, cache_e = FIG / "_phate_idx.npy", FIG / "_phate_emb.npy"
    if cache_i.exists():
        return np.load(cache_i), np.load(cache_e)
    lab = np.where(PAX.b2.isin(FA5) | PAX.b1.isin(FA5))[0]
    rest = np.setdiff1d(np.arange(len(PAX)), lab)
    rng = np.random.default_rng(0)
    idx = np.sort(np.concatenate([lab, rng.choice(rest, max(0, n - len(lab)), replace=False)]))
    X = StandardScaler().fit_transform(CPF[idx])
    emb = phate.PHATE(n_components=2, random_state=0, n_jobs=8, verbose=0).fit_transform(X)
    np.save(cache_i, idx)
    np.save(cache_e, emb)
    return idx, emb


def d_workflow(pred):
    import extract_cp_features as cp
    from sklearn.feature_selection import f_classif
    cond, f = EXAMPLE.split("_f")
    img = frame(cond, int(f) and None or None, int(f))[1] if False else frame(cond, f=int(f))[1]
    sub = PAX[PAX.img == EXAMPLE]
    lab = B2[B2.b2.isin(FA5)]
    emb = np.load(PAXD / "umap_cp32.npy")
    fig = plt.figure(figsize=(15.5, 3.1))
    g = fig.add_gridspec(1, 11, width_ratios=[1, 0.22, 1, 0.22, 1, 0.22, 1, 0.22, 1, 0.22, 1], wspace=0.05)
    titles = ["1  Image", "2  Patch labelling", "3  Features", "4  Classifier", "5  Prediction map", "6  Cell composition"]
    axs = []
    for c in (0, 2, 6, 8, 10):
        axs.append(fig.add_subplot(g[c]))
    ax_img, ax_lab, ax_umap, ax_map, ax_bar = axs
    sq = {}
    for a_ in (ax_img, ax_lab, ax_umap, ax_map, ax_bar):
        a_.set_box_aspect(1)
    ax_img.imshow(disp(img), cmap="gray")
    ax_lab.imshow(disp(img), cmap="gray")
    for _, r in B2[B2.img == EXAMPLE].iterrows():
        ax_lab.add_patch(Rectangle((r.x0, r.y0), PS, PS, fill=False, ec=FA5_COL[r.b2], lw=0.9))
    ax_map.imshow(disp(img), cmap="gray")
    for i, r in sub.iterrows():
        ax_map.add_patch(Rectangle((r.x0, r.y0), PS, PS, fc=FA5_COL[FA5[pred[i]]], ec="none", alpha=0.55))
    for a_ in (ax_img, ax_lab, ax_map):
        a_.axis("off")
    # 3: feature histograms (3 CP features that best separate the classes) in a sub-grid
    Xl = CPF[lab.index]
    y = lab.b2.map({c: i for i, c in enumerate(FA5)}).values
    Fv, _ = f_classif(Xl, y)
    Fv = np.nan_to_num(Fv)
    top = []
    for j in np.argsort(-Fv):
        if not top or all(abs(np.corrcoef(Xl[:, j], Xl[:, t])[0, 1]) < 0.8 for t in top):
            top.append(j)
        if len(top) == 3:
            break
    ax_f = fig.add_subplot(g[4])
    ax_f.axis("off")
    hs = []
    for k, j in enumerate(top):
        h = fig.add_axes([0, 0, 0.1, 0.1])
        hs.append(h)
        lo, hi = np.percentile(Xl[:, j], [1, 99])
        bins = np.linspace(lo, hi, 25)
        for ci, c in enumerate(FA5):
            h.hist(Xl[y == ci, j], bins=bins, histtype="step", density=True, color=FA5_COL[c], lw=1.1)
        h.set_yticks([])
        h.tick_params(labelsize=6)
        h.set_title(cp.FEATURE_NAMES[j], fontsize=6.5, pad=1)
        for sp in ("top", "right", "left"):
            h.spines[sp].set_visible(False)
    # 4: PHATE of the CP features (whole set, 8000-patch subsample incl. all labelled), B1+B2 labels (B2 wins)
    ph_idx, ph = phate_cp()
    comb = PAX.b2.where(PAX.b2.isin(FA5), PAX.b1.where(PAX.b1.isin(FA5)))
    ax_umap.scatter(ph[:, 0], ph[:, 1], s=1.5, c="#d8d8d8", lw=0, rasterized=True)
    for ci, c in enumerate(FA5):
        m = (comb.values[ph_idx] == c)
        ax_umap.scatter(ph[m, 0], ph[m, 1], s=4, c=FA5_COL[c], lw=0, label=FA5S[ci])
    ax_umap.set_xticks([])
    ax_umap.set_yticks([])
    ax_umap.set_xlabel("PHATE 1", fontsize=7, labelpad=1)
    ax_umap.set_ylabel("PHATE 2", fontsize=7, labelpad=1)
    ax_umap.legend(fontsize=5.5, markerscale=2, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=5, frameon=False,
                   handletextpad=0.05, columnspacing=0.6)
    fr = np.bincount(pred[sub.index], minlength=5) / len(sub) * 100
    ax_bar.bar(range(5), fr, color=[FA5_COL[c] for c in FA5])
    ax_bar.set_xticks(range(5))
    ax_bar.set_xticklabels(FA5S, fontsize=6.5)
    ax_bar.tick_params(axis="y", labelsize=7)
    ax_bar.set_title("% of patches", fontsize=7, pad=2)
    for c in (1, 3, 5, 7, 9):
        a_ = fig.add_subplot(g[c])
        a_.axis("off")
        a_.annotate("", xy=(0.6, 0.5), xytext=(0.0, 0.5), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#1f4e79"))
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=0.2)
    fig.canvas.draw()
    bi, bf = ax_img.get_position(), ax_f.get_position()
    slot = bi.height / 3
    for k, h in enumerate(hs):
        h.set_position([bf.x0 + 0.04 * bf.width, bi.y0 + (2 - k) * slot + 0.2 * slot, 0.92 * bf.width, 0.4 * slot])
    pos = [ax_img, ax_lab, ax_f, ax_umap, ax_map, ax_bar]
    for t, o in zip(titles, pos):
        bb = o.get_position()
        fig.text(bb.x0 + bb.width / 2, 0.97, t, ha="center", va="top", fontsize=11, fontweight="bold")
    save(fig, "D2_workflow_overview", "D", "The full workflow a user can apply",
         "Image -> labelled patches -> features (histograms of the 3 CP features best separating the B2 classes, coloured by class) -> "
         "classifier (PHATE of CP features, 8,000-patch subsample of the 27,637; B1+B2-labelled patches coloured, others grey) -> spatial prediction map -> within-cell class composition. "
         "Maps/histograms use the example image.")


# ───────────────────────────── deck ─────────────────────────────
SW, SH = Inches(13.33), Inches(7.5)
DARK = RGBColor(0x1F, 0x4E, 0x79)


def build_deck():
    prs = Presentation()
    prs.slide_width, prs.slide_height = SW, SH
    blank = prs.slide_layouts[6]

    def tb(s, t, l, tp, w, h, size, bold=False, color=RGBColor(0x22, 0x22, 0x22)):
        b = s.shapes.add_textbox(l, tp, w, h)
        b.text_frame.word_wrap = True
        p = b.text_frame.paragraphs[0]
        p.text = t
        p.font.size, p.font.bold, p.font.color.rgb = Pt(size), bold, color

    def header(s, t):
        r = s.shapes.add_shape(1, 0, 0, SW, Inches(0.6))
        r.fill.solid()
        r.fill.fore_color.rgb = DARK
        r.line.fill.background()
        tb(s, t, Inches(0.2), Inches(0.07), Inches(12.9), Inches(0.5), 20, True, RGBColor(255, 255, 255))

    s = prs.slides.add_slide(blank)
    header(s, "Figure 1 panels: local image patches as a unit for subcellular analysis")
    tb(s, "Panels A-D as separate PNGs (results/fig1_panels/). Scripts: scripts/make_fig1_panels.py\n"
          "A examples (pax / actin / MT, FA types, segmentation ambiguity)   B dataset   C baseline sampling   D workflow\n"
          "DS1 = 91 images (50 control + 41 Y-compound), pixel 0.1315 µm -> a 32 px patch is 4.2 µm.",
       Inches(0.6), Inches(1.2), Inches(12), Inches(2), 16)
    for panel, title, note, png in PANELS:
        s = prs.slides.add_slide(blank)
        header(s, f"Panel {panel}: {title}")
        from PIL import Image
        w, h = Image.open(png).size
        maxw, maxh = Inches(12.9), Inches(5.7)
        sc = min(maxw / w, maxh / h)
        s.shapes.add_picture(str(png), int((SW - w * sc) / 2), Inches(0.7), int(w * sc), int(h * sc))
        tb(s, note, Inches(0.4), Inches(6.5), Inches(12.5), Inches(0.9), 11)
    prs.save(PPTX)
    print(PPTX)


def pick_example():
    """B2-labelled control frame with all five classes and the most labels."""
    g = B2[B2.cond == "control"].groupby("img").b2.agg(["nunique", "size"])
    g = g.sort_values(["nunique", "size"], ascending=False)
    return g.index[0]


EXAMPLE = pick_example()


def main():
    print("example image", EXAMPLE, flush=True)
    a_examples()
    a_fa_classes()
    a_clusters()
    a_segmentation()
    b_multichannel()
    b_classes_counts()
    b_label_tree_b1()
    b_label_tables()
    c_grid_overlay()
    c_grid_left_only()
    c_orientation()
    c_grid_vs_localmax()
    c_label_hist()
    d_labelled_example()
    pred, Pm, accs, best = classify()
    print("cv", accs, best, flush=True)
    d_prediction_map(pred, Pm, accs, best)
    d_composition(pred)
    d_workflow(pred)
    order = {"A": 0, "B": 1, "C": 2, "D": 3}
    PANELS.sort(key=lambda t: (order[t[0]], t[3].name))
    build_deck()


if __name__ == "__main__":
    main()
