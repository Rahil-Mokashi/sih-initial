"""
Run the preprocessing pipeline (src/detection/preprocess.py) end to end on
one real downloaded sample image and save a before/after figure.

Uses a PANGAEA sample patch (data/raw/pangaea_med_oil_spill/images/ow-0001.jpg)
rather than a Zenodo image, because the Zenodo images archive is a single
40GB solid .7z with no per-file sampling -- see DECISIONS.md and
scripts/download_zenodo_sample.py. The PANGAEA patch is a real Sentinel-1
quicklook, not a calibrated Sigma0-dB GeoTIFF, so the calibration check is
expected to (correctly) report it as NOT calibrated dB -- that's the check
doing its job, not a bug. Once the full Zenodo archive is downloaded, this
same script should be pointed at one of its GeoTIFFs instead, where the
calibration check is expected to pass.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from detection.preprocess import check_calibration, lee_filter, tile_image  # noqa: E402

SAMPLE_JPG = Path(__file__).resolve().parent.parent / "data" / "raw" / "pangaea_med_oil_spill" / "images" / "ow-0001.jpg"
OUT_PNG = Path(__file__).resolve().parent.parent / "data" / "processed" / "demo_before_after.png"


def load_grayscale_array(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float32)


def main() -> None:
    if not SAMPLE_JPG.exists():
        print(f"ERROR: sample image not found at {SAMPLE_JPG}")
        print("Run scripts/download_pangaea_sample.py first.")
        sys.exit(1)

    raw = load_grayscale_array(SAMPLE_JPG)
    print(f"Loaded {SAMPLE_JPG.name}: shape={raw.shape}, dtype={raw.dtype}")

    calibration = check_calibration(raw)
    print(f"Calibration check: looks_like_db={calibration.looks_like_db}")
    print(f"  min={calibration.min_val:.1f} max={calibration.max_val:.1f} mean={calibration.mean_val:.1f}")
    print(f"  {calibration.reason}")

    despeckled = lee_filter(raw)
    tiles = tile_image(despeckled, tile_size=256)
    print(f"Tiled into {len(tiles)} patches of 256x256 (from a {raw.shape} image)")

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(raw, cmap="gray")
    axes[0].set_title("Raw (PANGAEA quicklook)")
    axes[0].axis("off")

    axes[1].imshow(despeckled, cmap="gray")
    axes[1].set_title("After Lee filter (despeckled)")
    axes[1].axis("off")

    diff = raw - despeckled
    axes[2].imshow(diff, cmap="RdBu")
    axes[2].set_title("Removed speckle (raw - despeckled)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=120)
    print(f"\nSaved before/after figure to {OUT_PNG}")


if __name__ == "__main__":
    main()
