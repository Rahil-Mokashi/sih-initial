"""
Compute the real oil-pixel class imbalance across all 1200 real masks from
the Zenodo Sentinel-1 SAR Oil Spill dataset's mask archive.

We don't have the matching images yet (see DECISIONS.md), so these masks
can't be used as image/mask training pairs -- but their pixel statistics
are real and directly useful for calibrating the segmentation loss's
pos_weight (src/detection/losses.py), instead of guessing a weight.

Extracts the mask archive (if not already extracted) and prints/saves the
per-mask oil-pixel fraction distribution.
"""

import os
from pathlib import Path

import numpy as np
import py7zr
from PIL import Image

MASK_ARCHIVE = Path(__file__).resolve().parent.parent / "data" / "raw" / "zenodo_sar_oil_spill" / "01_Train_Val_Oil_Spill_mask.7z"
EXTRACT_DIR = MASK_ARCHIVE.parent / "masks_extracted"
OUT_NPY = Path(__file__).resolve().parent.parent / "data" / "processed" / "zenodo_mask_oil_fractions.npy"


def main() -> None:
    if not MASK_ARCHIVE.exists():
        print(f"ERROR: {MASK_ARCHIVE} not found. Run scripts/download_zenodo_sample.py first.")
        return

    if not EXTRACT_DIR.exists():
        print(f"Extracting {MASK_ARCHIVE.name}...")
        with py7zr.SevenZipFile(MASK_ARCHIVE, mode="r") as z:
            z.extractall(path=EXTRACT_DIR)

    fractions = []
    for root, _, fnames in os.walk(EXTRACT_DIR):
        for fn in fnames:
            if fn.lower().endswith(".tif"):
                arr = np.array(Image.open(os.path.join(root, fn)))
                fractions.append((arr > 0).mean())

    fractions = np.array(fractions)
    print(f"n masks analyzed: {len(fractions)}")
    print(f"mean oil fraction:   {fractions.mean():.4f}")
    print(f"median oil fraction: {np.median(fractions):.4f}")
    print(f"min/max:             {fractions.min():.4f} / {fractions.max():.4f}")
    print(f"masks with 0 oil px: {(fractions == 0).sum()}")

    mean_frac = fractions.mean()
    pos_weight = (1 - mean_frac) / mean_frac
    print(f"\nsuggested BCE pos_weight ((1-p)/p on mean fraction): {pos_weight:.1f}")

    OUT_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_NPY, fractions)
    print(f"saved per-mask fractions to {OUT_NPY}")


if __name__ == "__main__":
    main()
