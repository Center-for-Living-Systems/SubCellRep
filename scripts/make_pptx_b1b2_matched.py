#!/usr/bin/env python3
"""
make_pptx_b1b2_matched.py

Thin wrapper — builds the B1/B2 matched annotator comparison deck.
All slide logic lives in make_pptx_le_benchmark.py.

Output: results/le_b1b2_matched.pptx

Usage:
  python scripts/make_pptx_b1b2_matched.py [--out results/le_b1b2_matched.pptx]
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_pptx_le_benchmark import build_pptx_b1b2, OUT_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DIR / "le_b1b2_matched.pptx"))
    args = ap.parse_args()
    build_pptx_b1b2(Path(args.out))


if __name__ == "__main__":
    main()
