"""
Builds a TILE-level ablation subset manifest for Experiment 01, so a
25-epoch comparative run doesn't have to pay for the full 34,940-tile
training set (82.4% of which is zero-oil, see docs/metric_audit.md
Finding #8) every epoch.

The existing manifest CSV format (data/processed/train_manifest.csv) is
per SOURCE IMAGE, not per tile -- ZenodoTileDataset tiles each listed
image's full grid internally. A tile-level subset ("all oil-containing
tiles, plus an equal count of random zero-oil tiles, plus all lookalike
tiles") can't be expressed in that 3-column format, so this script writes
an EXTENDED 5-column format (image_path,mask_path,label,row_offset,
col_offset -- one row per SELECTED TILE) and
detection.dataset.ZenodoTileDataset gained an explicit_index constructor
param specifically so train.py can consume it (see that file's docstring).
Does NOT touch or overwrite the original train_manifest.csv.

Selection logic, applied over the exact same 512x512 non-overlapping tile
grid ZenodoTileDataset would build from the full train_manifest.csv:
  1. Every tile with any real oil pixel (has_oil=True), regardless of its
     source image's class label.
  2. Every tile from a "lookalike" source image (all zero-oil by
     construction -- see Finding #3 -- and unconditionally included, not
     drawn from the random sample pool).
  3. A random sample (seed=42, WITHOUT replacement), sized to exactly
     match the count from (1), of zero-oil tiles from NON-lookalike
     source images (i.e. drawn from "oil"-labeled images' own empty tiles
     and all "no_oil"-labeled images' tiles) -- "equal count of randomly
     sampled zero-oil tiles."

Usage:
    venv\\Scripts\\python.exe scripts\\build_ablation_manifest.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_MANIFEST = REPO_ROOT / "data" / "processed" / "train_manifest.csv"
OUT_MANIFEST = REPO_ROOT / "data" / "processed" / "train_manifest_ablation.csv"
TILE_SIZE = 512
SEED = 42


def load_manifest(path: Path) -> list[dict]:
    return list(csv.DictReader(open(path, newline="")))


def main() -> None:
    rows = load_manifest(TRAIN_MANIFEST)
    print(f"scanning {len(rows)} training images from {TRAIN_MANIFEST} for real per-tile oil presence "
          f"(tile_size={TILE_SIZE}, non-overlapping -- same grid ZenodoTileDataset builds)")

    oil_tiles = []            # (image_path, mask_path, label, y, x)
    lookalike_tiles = []
    zero_oil_non_lookalike_tiles = []
    total_tiles = 0

    for i, row in enumerate(rows):
        image_path, mask_path, label = row["image_path"], row["mask_path"], row["label"]
        with rasterio.open(image_path) as img_src, rasterio.open(mask_path) as mask_src:
            h, w = img_src.height, img_src.width
            for y in range(0, h - TILE_SIZE + 1, TILE_SIZE):
                for x in range(0, w - TILE_SIZE + 1, TILE_SIZE):
                    window = Window(x, y, TILE_SIZE, TILE_SIZE)
                    mask_tile = mask_src.read(1, window=window)
                    has_oil = bool((mask_tile > 0).any())
                    total_tiles += 1
                    row_out = (image_path, mask_path, label, y, x)
                    if has_oil:
                        oil_tiles.append(row_out)
                    elif label == "lookalike":
                        lookalike_tiles.append(row_out)
                    else:
                        zero_oil_non_lookalike_tiles.append(row_out)
        if (i + 1) % 200 == 0:
            print(f"  ...{i + 1}/{len(rows)} images scanned")

    # Sanity-check the documented Finding #3 assumption ("lookalike" images
    # are all-zero by construction) rather than silently trusting it -- if
    # any lookalike tile actually has oil pixels, it already landed in
    # oil_tiles above (has_oil is checked first), so this is just visibility
    # into whether that ever happened, not a correctness dependency.
    lookalike_with_oil = sum(1 for _p, _m, label, _y, _x in oil_tiles if label == "lookalike")
    if lookalike_with_oil:
        print(f"NOTE: {lookalike_with_oil} lookalike tiles actually had real oil pixels -- "
              f"included in the oil-containing group, not the unconditional lookalike group.")

    n_oil = len(oil_tiles)
    rng = np.random.default_rng(SEED)
    sample_size = min(n_oil, len(zero_oil_non_lookalike_tiles))
    if sample_size < n_oil:
        print(f"WARNING: only {len(zero_oil_non_lookalike_tiles)} zero-oil non-lookalike tiles exist, "
              f"fewer than the {n_oil} oil tiles -- sampling all of them instead of an equal count.")
    sampled_idx = rng.choice(len(zero_oil_non_lookalike_tiles), size=sample_size, replace=False)
    sampled_zero_oil = [zero_oil_non_lookalike_tiles[i] for i in sorted(sampled_idx)]

    selected = oil_tiles + lookalike_tiles + sampled_zero_oil
    selected.sort(key=lambda r: (r[0], r[3], r[4]))  # stable, readable ordering: by image path then position

    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MANIFEST, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "mask_path", "label", "row_offset", "col_offset"])
        for image_path, mask_path, label, y, x in selected:
            writer.writerow([image_path, mask_path, label, y, x])

    print(f"\n=== {OUT_MANIFEST} ===")
    print(f"full training tile population scanned: {total_tiles}")
    print(f"  oil-containing tiles: {n_oil}")
    print(f"  lookalike tiles (unconditional): {len(lookalike_tiles)}")
    print(f"  zero-oil non-lookalike tiles available: {len(zero_oil_non_lookalike_tiles)}")
    print(f"  zero-oil non-lookalike tiles sampled (seed={SEED}): {len(sampled_zero_oil)}")
    print(f"selected total: {len(selected)} tiles ({100 * len(selected) / total_tiles:.2f}% of the full "
          f"{total_tiles}-tile training set)")
    print(f"wrote {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
