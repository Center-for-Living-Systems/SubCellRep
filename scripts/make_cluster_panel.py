#!/usr/bin/env python3
"""
make_cluster_panel.py

For each setting directory (e.g. full_highdim/, umap2d/, or the flat root),
assemble a single overview PNG: 10 rows (clusters) × 16 columns (examples).

Usage:
  python scripts/make_cluster_panel.py [--root eval_results/actin_kmeans_patches]
                                        [--settings full_highdim umap2d flat]
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT_DEFAULT = (Path(__file__).resolve().parents[1]
                / "eval_results" / "actin_kmeans_patches")

N_CLUSTERS  = 10
N_EXAMPLES  = 16
PS          = 32          # patch pixel size
GAP         = 3           # gap between patches (px)
LABEL_W     = 72          # left label column width (px)
HEADER_H    = 28          # top header row height (px)
BG          = 30          # background grey value
LABEL_COLOR = (220, 220, 220)
TITLE_COLOR = (255, 255, 255)


def load_cluster_pngs(setting_dir: Path) -> list[list[np.ndarray | None]]:
    """Return grid[cluster][example] of uint8 arrays (or None if missing)."""
    grid = []
    for k in range(N_CLUSTERS):
        cdir = setting_dir / f"cluster_{k:02d}"
        row = []
        for ex in range(1, N_EXAMPLES + 1):
            pattern = f"cluster_{k:02d}_ex{ex:02d}_*.png"
            matches = sorted(cdir.glob(pattern)) if cdir.exists() else []
            if matches:
                arr = np.asarray(Image.open(matches[0]).convert("L"))
                row.append(arr)
            else:
                row.append(None)
        grid.append(row)
    return grid


def build_panel(grid: list[list[np.ndarray | None]], title: str) -> Image.Image:
    cols = N_EXAMPLES
    rows = N_CLUSTERS

    canvas_w = LABEL_W + cols * PS + (cols - 1) * GAP
    canvas_h = HEADER_H + rows * PS + (rows - 1) * GAP

    canvas = np.full((canvas_h, canvas_w), BG, dtype=np.uint8)
    img = Image.fromarray(canvas).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        font_ti = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except Exception:
        font_sm = ImageFont.load_default()
        font_ti = font_sm

    # Title
    draw.text((LABEL_W, 4), title, fill=TITLE_COLOR, font=font_ti)

    # Column numbers
    for ex in range(cols):
        x = LABEL_W + ex * (PS + GAP) + PS // 2 - 4
        draw.text((x, 4), str(ex + 1), fill=(160, 160, 160), font=font_sm)

    for k, row in enumerate(grid):
        y0 = HEADER_H + k * (PS + GAP)

        # Row label
        draw.text((2, y0 + PS // 2 - 6), f"C{k:02d}", fill=LABEL_COLOR, font=font_sm)

        for ex, patch in enumerate(row):
            x0 = LABEL_W + ex * (PS + GAP)
            if patch is None:
                # draw an X placeholder
                draw.rectangle([x0, y0, x0 + PS - 1, y0 + PS - 1], fill=(60, 30, 30))
                draw.line([x0, y0, x0 + PS, y0 + PS], fill=(120, 60, 60))
                draw.line([x0 + PS, y0, x0, y0 + PS], fill=(120, 60, 60))
            else:
                # paste greyscale patch as RGB
                p = Image.fromarray(patch).convert("RGB")
                img.paste(p, (x0, y0))

    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT_DEFAULT))
    ap.add_argument("--settings", nargs="+",
                    default=["flat", "full_highdim", "umap2d"])
    args = ap.parse_args()

    root = Path(args.root)

    setting_dirs: dict[str, Path] = {
        "flat":         root,
        "full_highdim": root / "full_highdim",
        "umap2d":       root / "umap2d",
        "pca2d":        root / "pca2d",
    }

    for name in args.settings:
        d = setting_dirs.get(name, root / name)
        if not d.exists():
            print(f"Skip {name}: {d} not found")
            continue

        # For "flat", skip sub-dirs that are named full_highdim/umap2d
        if name == "flat":
            # verify at least one cluster_XX dir exists directly
            if not any(d.glob("cluster_0*")):
                print(f"Skip {name}: no cluster dirs found")
                continue

        print(f"Building panel for {name} ...", flush=True)
        grid = load_cluster_pngs(d)
        found = sum(1 for row in grid for p in row if p is not None)
        print(f"  {found}/{N_CLUSTERS * N_EXAMPLES} patches found", flush=True)

        title_map = {
            "flat":         "AE baseline — actin KMeans k=10",
            "full_highdim": "CP features (50-D) — KMeans k=10",
            "umap2d":       "UMAP 2-D — KMeans k=10",
            "pca2d":        "PCA 2-D — KMeans k=10",
        }
        panel = build_panel(grid, title_map.get(name, name))

        out = root / f"panel_{name}.png"
        panel.save(str(out), format="PNG", compress_level=0)
        print(f"  Saved → {out}  ({panel.width}×{panel.height} px)", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
