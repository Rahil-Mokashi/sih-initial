"""
Dataset-wide nodata fraction: what fraction of VALIDATION pixels are exact
0.0 dB in both real SAR bands (see src/detection/preprocess.py's
compute_nodata_mask, and docs/metric_audit.md Finding #12/#11).

The earlier 1.81% figure was a single-image spot-check. This scans every
tile of the actual 512x512 non-overlapping grid the val ZenodoTileDataset
produces (same tile_size/stride, same edge-remainder dropping) -- i.e. the
exact pixel population that ever passes through training/evaluation, not
the full raw images (some border pixels are dropped by the tile grid and
never seen by the model either way).

Usage:
    venv\\Scripts\\python.exe scripts\\compute_nodata_fraction.py
    venv\\Scripts\\python.exe scripts\\compute_nodata_fraction.py --manifest data/processed/train_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from detection.preprocess import compute_nodata_mask  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
TILE_SIZE = 512


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "data" / "processed" / "val_manifest.csv")
    parser.add_argument("--tile-size", type=int, default=TILE_SIZE)
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.manifest, newline="")))
    print(f"scanning {len(rows)} images from {args.manifest} (tile_size={args.tile_size}, non-overlapping grid)")

    total_pixels = 0
    nodata_pixels = 0
    tiles_scanned = 0
    tiles_with_any_nodata = 0
    tiles_fully_nodata = 0
    per_label_total: dict[str, int] = {}
    per_label_nodata: dict[str, int] = {}

    for i, row in enumerate(rows):
        image_path, label = row["image_path"], row["label"]
        with rasterio.open(image_path) as src:
            h, w = src.height, src.width
            for y in range(0, h - args.tile_size + 1, args.tile_size):
                for x in range(0, w - args.tile_size + 1, args.tile_size):
                    window = Window(x, y, args.tile_size, args.tile_size)
                    band1 = src.read(1, window=window).astype(np.float32)
                    band2 = src.read(2, window=window).astype(np.float32)
                    nodata = compute_nodata_mask(band1, band2)

                    n = nodata.size
                    k = int(nodata.sum())
                    total_pixels += n
                    nodata_pixels += k
                    tiles_scanned += 1
                    if k > 0:
                        tiles_with_any_nodata += 1
                    if k == n:
                        tiles_fully_nodata += 1
                    per_label_total[label] = per_label_total.get(label, 0) + n
                    per_label_nodata[label] = per_label_nodata.get(label, 0) + k

        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{len(rows)} images, running fraction so far: {100 * nodata_pixels / total_pixels:.3f}%")

    print(f"\n=== {args.manifest} ===")
    print(f"tiles scanned: {tiles_scanned} ({args.tile_size}x{args.tile_size}, non-overlapping)")
    print(f"total pixels: {total_pixels:,}")
    print(f"nodata pixels (exact 0.0 dB in both bands): {nodata_pixels:,}")
    print(f"nodata fraction: {100 * nodata_pixels / total_pixels:.4f}%")
    print(f"tiles with >=1 nodata pixel: {tiles_with_any_nodata}/{tiles_scanned} "
          f"({100 * tiles_with_any_nodata / tiles_scanned:.2f}%)")
    print(f"tiles that are 100% nodata: {tiles_fully_nodata}/{tiles_scanned} "
          f"({100 * tiles_fully_nodata / tiles_scanned:.2f}%)")
    print("\nper source-image class:")
    for label in sorted(per_label_total):
        t, k = per_label_total[label], per_label_nodata[label]
        print(f"  {label}: {100 * k / t:.4f}% nodata ({k:,}/{t:,} px)")


if __name__ == "__main__":
    main()
