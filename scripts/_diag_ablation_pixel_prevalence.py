"""
Scratch diagnostic (not part of the pipeline): real oil-PIXEL prevalence
within data/processed/train_manifest_ablation.csv's selected tiles, vs.
the whole-image 2.98% mean figure pos_weight=32.6 was calibrated against
(DECISIONS.md line ~151, scripts/compute_mask_class_balance.py).

Reads each selected tile's mask window directly (rasterio), same 512x512
grid ZenodoTileDataset/build_ablation_manifest.py use. Read-only, no
training involved.
"""
import csv
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import rasterio
from rasterio.windows import Window

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "data" / "processed" / "train_manifest_ablation.csv"
TILE_SIZE = 512


def main():
    rows = list(csv.DictReader(open(MANIFEST, newline="")))
    print(f"{len(rows)} selected tiles in {MANIFEST}")

    # Group by mask_path so we only open each mask file once.
    by_mask = defaultdict(list)
    for r in rows:
        by_mask[r["mask_path"]].append((int(r["row_offset"]), int(r["col_offset"]), r["label"]))

    total_pixels = 0
    total_oil_pixels = 0
    per_label_total = defaultdict(int)
    per_label_oil = defaultdict(int)
    n_tiles_with_oil = 0

    for i, (mask_path, tiles) in enumerate(by_mask.items()):
        with rasterio.open(mask_path) as src:
            for y, x, label in tiles:
                window = Window(x, y, TILE_SIZE, TILE_SIZE)
                arr = src.read(1, window=window)
                oil = int((arr > 0).sum())
                n = arr.size
                total_pixels += n
                total_oil_pixels += oil
                per_label_total[label] += n
                per_label_oil[label] += oil
                if oil > 0:
                    n_tiles_with_oil += 1
        if (i + 1) % 500 == 0:
            print(f"  ...{i+1}/{len(by_mask)} mask files processed", file=sys.stderr)

    frac = total_oil_pixels / total_pixels
    print(f"\n=== ablation subset pixel-level oil prevalence ===")
    print(f"total pixels: {total_pixels:,}  oil pixels: {total_oil_pixels:,}")
    print(f"oil-pixel fraction (whole subset): {frac:.4%}")
    print(f"tiles with any oil pixel: {n_tiles_with_oil} / {len(rows)} ({100*n_tiles_with_oil/len(rows):.2f}%)")
    print(f"\nimplied pos_weight ((1-p)/p) for THIS subset: {(1-frac)/frac:.1f}   (config uses pos_weight=32.6)")

    print(f"\n--- by label ---")
    for label in per_label_total:
        t = per_label_total[label]
        o = per_label_oil[label]
        print(f"  {label:12s}: {t:>12,} px, {o:>10,} oil px, frac={o/t:.4%}")

    # Oil-containing tiles only (the has_oil group), pixel-level fraction WITHIN just those tiles
    oil_rows = [r for r in rows if True]  # placeholder, real filter below


if __name__ == "__main__":
    main()
