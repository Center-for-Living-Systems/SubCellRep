#!/usr/bin/env python3
"""
make_pptx_mt_patches.py

PPT for the Ernest microtubule patch extraction + CP/IL feature clustering.
Slides: pipeline, segmentation/patch displays, patch stats, then for each of
{CP, IL} x {32, 64}: UMAP coloured by cluster + 10 clusters x 16 patches nearest
to the cluster centre.

Run:  python scripts/make_pptx_mt_patches.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

D = Path("/home/lding/lding/data/Ernest/mt_patches")
OUT = Path(__file__).resolve().parents[1] / "results" / "mt_patch_clusters.pptx"
K, N_PANEL = 10, 16

SW, SH = Inches(13.33), Inches(7.5)
DARK = RGBColor(0x1F, 0x4E, 0x79)


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def text(slide, s, l, t, w, h, size=14, bold=False, color=RGBColor(0x22, 0x22, 0x22)):
    tb = slide.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, line in enumerate(s.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.size, p.font.bold, p.font.color.rgb = Pt(size), bold, color
    return tb


def header(slide, title):
    r = slide.shapes.add_shape(1, 0, 0, SW, Inches(0.65))
    r.fill.solid()
    r.fill.fore_color.rgb = DARK
    r.line.fill.background()
    text(slide, title, Inches(0.2), Inches(0.08), Inches(12.9), Inches(0.5), 22, True, RGBColor(255, 255, 255))


def pic(slide, img, l, t, w=None, h=None):
    p = D / "_tmp.png"
    (img if isinstance(img, Image.Image) else Image.open(img)).save(p)
    slide.shapes.add_picture(str(p), l, t, w, h)


def table(slide, df, l, t, w, h, size=11):
    gt = slide.shapes.add_table(len(df) + 1, len(df.columns), l, t, w, h).table
    for j, c in enumerate(df.columns):
        gt.cell(0, j).text = str(c)
    for i, row in enumerate(df.itertuples(index=False), 1):
        for j, v in enumerate(row):
            gt.cell(i, j).text = f"{v:.2f}" if isinstance(v, float) else str(v)
    for r in gt.rows:
        for c in r.cells:
            for p in c.text_frame.paragraphs:
                p.font.size = Pt(size)


def font(sz):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", sz)
    except Exception:
        return ImageFont.load_default()


def panel(patches, title, cell=64, gap=3, cols=4):
    """4x4 grid of per-patch min-max scaled patches with a title bar."""
    rows = (len(patches) + cols - 1) // cols
    W = cols * (cell + gap) - gap
    H = rows * (cell + gap) - gap
    bar = 26
    im = Image.new("L", (W, H + bar), 30)
    ImageDraw.Draw(im).text((4, 4), title, fill=235, font=font(14))
    for i, p in enumerate(patches):
        p = p.astype(np.float32)
        p = (p - p.min()) / (p.max() - p.min() + 1e-8)
        t = Image.fromarray((p * 255).astype(np.uint8)).resize((cell, cell), Image.NEAREST)
        im.paste(t, ((i % cols) * (cell + gap), bar + (i // cols) * (cell + gap)))
    return im.convert("RGB")


def main():
    import argparse
    global D, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=D)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--title", default="Microtubule patches: extraction + CP/IL feature clustering")
    ap.add_argument("--subtitle", default=None)
    ap.add_argument("--seg-ch", default="ch0")
    ap.add_argument("--patch-ch", default="ch1")
    ap.add_argument("--cond-col", default=None, help="index column to tabulate per cluster (e.g. file)")
    ap.add_argument("--kinds", nargs="+", default=["cp", "il"], choices=["cp", "il", "hc"])
    ap.add_argument("--min-cov", type=float, default=0.1)
    ap.add_argument("--cap-note", default="capped at 100 random patches per frame per size (disk space limit)")
    a = ap.parse_args()
    D, OUT = a.dir, a.out
    prs = Presentation()
    prs.slide_width, prs.slide_height = SW, SH
    summ = pd.read_csv(D / "frame_summary.csv")
    summ["file"] = summ["frame"].str.rsplit("_s", n=1).str[0]

    # 1 title
    s = blank(prs)
    header(s, a.title)
    text(s, (a.subtitle or "Ernest U2OS DMSO data (6 ND2 files, 375 series, 3 ch: RFP / GFP / JF646)") + "\n"
            f"{a.seg_ch} -> cell segmentation, {a.patch_ch} -> patches (32 px and 64 px)\n"
            "CellProfiler-style (50) and ilastik-style (80) features, KMeans k=10",
         Inches(0.6), Inches(1.2), Inches(12), Inches(2), 20)

    # 2 pipeline
    s = blank(prs)
    header(s, "Patch extraction pipeline")
    nf = int((summ.n_regions > 0).sum())
    text(s,
         f"Segmentation ({a.seg_ch}): flat-field correction -> 1-99 pct stretch -> threshold 0.1 -> close (disk 11) -> open (disk 3)\n"
         "   -> fill holes -> keep every region >= 40,000 px (all cells kept, not only the central one)\n"
         f"Empty-frame gate: {a.seg_ch} p99.9 / median < 3 means pure noise -> no cells, no patches\n"
         "   (otherwise the percentile stretch amplifies noise into a whole-field 'cell')\n"
         "Frame with no region >= 40,000 px -> zero patches\n\n"
         f"Patches ({a.patch_ch}): cell_insideoutside normalisation using the {a.seg_ch} mask (no rolling ball),\n"
         f"   non-overlapping grid, patch kept if cell-mask coverage >= {a.min_cov:.0%}\n"
         f"   {a.cap_note}\n\n"
         f"Result: {nf} / {len(summ)} frames contain cells; {int(summ.n_ps32.sum())} patches (32 px), {int(summ.n_ps64.sum())} patches (64 px)\n\n"
         "Features: CP = 11 intensity + 39 Haralick; IL = 80 multiscale filter-bank stats\n"
         "Clustering: per-feature quantile transform to N(0,1) (plain z-score gave singleton clusters from outliers),\n"
         "   KMeans k=10, 16 patches closest to each centre shown",
         Inches(0.4), Inches(0.85), Inches(12.5), Inches(6.4), 15)

    # 3 per-file counts
    s = blank(prs)
    header(s, "Frames with cells and patch counts per source group")
    if summ["file"].nunique() > 15:     # per-image files (CZI): group by condition suffix, e.g. "..._control-01"
        summ["file"] = summ["frame"].str.rsplit("_", n=1).str[1].str.rsplit("-", n=1).str[0]
    g = summ.groupby("file").agg(series=("frame", "size"),
                                 with_cells=("n_regions", lambda x: int((x > 0).sum())),
                                 patches_32=("n_ps32", "sum"), patches_64=("n_ps64", "sum")).reset_index()
    table(s, g, Inches(0.6), Inches(1.0), Inches(12), Inches(3), 13)

    # 4+ segmentation displays
    disp = sorted((D / "display").glob("*.png"))
    have = set(summ[summ.n_regions > 0].frame)
    pick = []
    for pref in ("AcTub_bTub", "bTub_AcTub"):
        c = [p for p in disp if pref in p.name and p.stem in have]
        pick += c[len(c) // 4::max(1, len(c) // 2)][:2]
    c = [p for p in disp if p.stem not in have]
    pick += c[:1]
    for i in range(0, len(pick), 2):
        s = blank(prs)
        header(s, f"Segmentation ({a.seg_ch}) and extracted patch locations ({a.patch_ch})")
        for j, p in enumerate(pick[i:i + 2]):
            pic(s, p, Inches(0.2), Inches(0.8 + j * 3.3), w=Inches(12.9))
    # random patch montage
    s = blank(prs)
    header(s, "Random extracted patches (per-patch min-max display)")
    rng = np.random.default_rng(0)
    for j, S in enumerate((32, 64)):
        P = np.load(D / f"patches{S}.npy", mmap_mode="r")
        ids = np.sort(rng.choice(len(P), 32, replace=False))
        pic(s, panel([P[i] for i in ids[:16]], f"{S} px", 100 if S == 32 else 100, cols=4),
            Inches(0.4 + j * 6.5), Inches(0.9), h=Inches(5.5) if False else None, w=Inches(5.9))

    # clusters
    for S in (32, 64):
        P = np.load(D / f"patches{S}.npy", mmap_mode="r")
        for kind, kname in [(k_, {"cp": "CellProfiler (50 feat)", "il": "ilastik (80 feat)", "hc": "Haralick only (39 feat)"}[k_]) for k_ in a.kinds]:
            cl = pd.read_csv(D / f"clusters_{kind}{S}.csv")
            emb = np.load(D / f"umap_{kind}{S}.npy")
            s = blank(prs)
            header(s, f"{kname}, {S} px: UMAP coloured by KMeans k=10")
            fig, ax = plt.subplots(figsize=(7, 5.8))
            sc = ax.scatter(emb[:, 0], emb[:, 1], c=cl.cluster, cmap="tab10", s=3, alpha=0.6)
            for k in range(K):
                m = cl.cluster == k
                ax.text(*np.median(emb[m], 0), str(k), fontsize=13, weight="bold", ha="center")
            ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
            fig.tight_layout()
            fig.canvas.draw()
            pic(s, Image.fromarray(np.asarray(fig.canvas.buffer_rgba())), Inches(0.4), Inches(0.85), h=Inches(6.3))
            plt.close(fig)
            sz = cl.groupby("cluster").size()
            lines = [f"  {k}: {sz[k]}" for k in range(K)]
            if a.cond_col:
                idx = pd.read_csv(D / f"patches{S}_index.csv")
                ct = pd.crosstab(cl.cluster, idx[a.cond_col].values, normalize="index") * 100
                lines = [f"  {k}: {sz[k]}  (" + ", ".join(f"{c} {ct.loc[k, c]:.0f}%" for c in ct.columns) + ")"
                         for k in range(K)]
            text(s, "cluster sizes (+ % by " + (a.cond_col or "-") + "):\n" + "\n".join(lines),
                 Inches(8.6), Inches(1.5), Inches(4.6), Inches(5), 13)

            s = blank(prs)
            header(s, f"{kname}, {S} px: {N_PANEL} patches closest to each cluster centre")
            pw = Inches(2.5)
            for k in range(K):
                sub = cl[cl.cluster == k].nsmallest(N_PANEL, "dist")
                im = panel([P[i] for i in sub.idx], f"Cluster {k} (N={sz[k]})", 64, 3)
                pic(s, im, Inches(0.15) + (k % 5) * Inches(2.6), Inches(0.8) + (k // 5) * Inches(3.35), w=pw)
    OUT.parent.mkdir(exist_ok=True)
    prs.save(OUT)
    (D / "_tmp.png").unlink(missing_ok=True)
    print(OUT)


if __name__ == "__main__":
    main()
