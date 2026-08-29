"""
Real tile-level oil-pixel distribution across the actual training tile grid
ZenodoTileDataset builds (non-overlapping 512x512 tiles, stride=tile_size).

compute_mask_class_balance.py measured oil-pixel imbalance *within* the
1200 oil-class images (mean 2.98%) and used it to set pos_weight -- but
pos_weight only reweights pixels *within* a tile that already has some
oil in it. It says nothing about how many training tiles have ANY oil
pixels at all, out of the full tile grid that also includes every
no_oil/lookalike image (685+685, zero oil by definition) and every
non-slick tile of the 1200 oil images. If most tiles are entirely
oil-free, most training batches carry zero oil-learning gradient
regardless of how pos_weight is tuned -- a sampling problem pos_weight
cannot fix. This script measures that directly.

Usage:
    venv\\Scripts\\python.exe scripts\\analyze_tile_oil_distribution.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import rasterio

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_MANIFEST = REPO_ROOT / "data" / "processed" / "train_manifest.csv"
TILE_SIZE = 512


def main() -> None:
    if not TRAIN_MANIFEST.exists():
        print("ERROR: train_manifest.csv not found. Run scripts/build_training_pool.py first.")
        return

    rows = list(csv.DictReader(open(TRAIN_MANIFEST, newline="")))
    total_tiles = 0
    oil_tiles = 0  # tiles with >=1 oil pixel
    substantial_tiles = 0  # tiles with >=1% oil pixels (arbitrary "meaningfully present" bar)
    by_label = {"oil": [0, 0], "no_oil": [0, 0], "lookalike": [0, 0]}  # [total_tiles, oil_tiles]

    for row in rows:
        mask_path, label = row["mask_path"], row["label"]
        with rasterio.open(mask_path) as src:
            h, w = src.height, src.width
            arr = src.read(1)
        for y in range(0, h - TILE_SIZE + 1, TILE_SIZE):
            for x in range(0, w - TILE_SIZE + 1, TILE_SIZE):
                tile = arr[y:y + TILE_SIZE, x:x + TILE_SIZE]
                frac = (tile > 0).mean()
                total_tiles += 1
                by_label[label][0] += 1
                if frac > 0:
                    oil_tiles += 1
                    by_label[label][1] += 1
                if frac >= 0.01:
                    substantial_tiles += 1

    print(f"real images in train_manifest.csv: {len(rows)}")
    print(f"real training tiles (512x512, non-overlapping): {total_tiles}")
    print(f"tiles with ANY oil pixel:      {oil_tiles} ({100 * oil_tiles / total_tiles:.1f}%)")
    print(f"tiles with >=1% oil pixels:    {substantial_tiles} ({100 * substantial_tiles / total_tiles:.1f}%)")
    print(f"tiles with ZERO oil pixels:    {total_tiles - oil_tiles} ({100 * (total_tiles - oil_tiles) / total_tiles:.1f}%)")
    print()
    for label, (n, n_oil) in by_label.items():
        print(f"  [{label}] {n} tiles, {n_oil} with any oil ({100 * n_oil / n:.1f}%)" if n else f"  [{label}] 0 tiles")


if __name__ == "__main__":
    main()
