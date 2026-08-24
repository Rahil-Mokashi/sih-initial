"""
SAR image preprocessing: despeckling, Sigma0-dB calibration check, tiling.

Pipeline (per raw image):
  1. load_sar_image  -- read a single-band raster (GeoTIFF or plain image)
  2. check_calibration -- sanity-check that pixel values look like Sigma0
     in decibels (roughly -35 to +5 dB for Sentinel-1 ocean scenes), not
     raw amplitude/power or an 8-bit visualization
  3. lee_filter      -- despeckle with an adaptive Lee filter
  4. tile_image       -- cut into fixed-size patches for training
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import uniform_filter


# Sentinel-1 Sigma0 ocean scenes typically fall in this dB range. Values
# well outside this band mean the image isn't calibrated dB backscatter
# (e.g. it's raw DN, linear power, or an 8-bit quicklook).
EXPECTED_DB_RANGE = (-40.0, 10.0)


@dataclass
class CalibrationReport:
    looks_like_db: bool
    min_val: float
    max_val: float
    mean_val: float
    reason: str


def load_sar_image(path: str | Path) -> np.ndarray:
    """Load a single-band raster as a float32 array."""
    with rasterio.open(path) as src:
        band = src.read(1).astype(np.float32)
    return band


def check_calibration(image: np.ndarray) -> CalibrationReport:
    """
    Heuristic check that `image` looks like Sigma0 in dB.

    This can't prove calibration is correct (that requires the product's
    metadata/LUTs), but it catches the common failure modes: raw 8-bit
    quicklooks (0-255), linear power/amplitude values (>>10), or data that's
    suspiciously uniform.
    """
    min_val, max_val, mean_val = float(image.min()), float(image.max()), float(image.mean())
    lo, hi = EXPECTED_DB_RANGE

    if max_val > 100:
        return CalibrationReport(False, min_val, max_val, mean_val,
                                  f"max value {max_val:.1f} is far above the dB range "
                                  f"({lo} to {hi}) -- looks like linear power/amplitude, not dB")
    if min_val >= 0 and max_val <= 255 and image.dtype != np.float64:
        if not (lo <= mean_val <= hi):
            return CalibrationReport(False, min_val, max_val, mean_val,
                                      f"value range [{min_val:.1f}, {max_val:.1f}] looks like an "
                                      f"8-bit quicklook (0-255), not calibrated dB backscatter")
    if lo <= min_val and max_val <= hi:
        return CalibrationReport(True, min_val, max_val, mean_val,
                                  f"value range [{min_val:.1f}, {max_val:.1f}] dB is within the "
                                  f"expected Sentinel-1 ocean Sigma0 range ({lo} to {hi})")
    return CalibrationReport(False, min_val, max_val, mean_val,
                              f"value range [{min_val:.1f}, {max_val:.1f}] falls outside the "
                              f"expected dB range ({lo} to {hi})")


def lee_filter(image: np.ndarray, window_size: int = 7) -> np.ndarray:
    """
    Adaptive Lee filter for speckle reduction.

    Standard formulation: output = mean + k * (pixel - mean), where k is a
    per-pixel weight derived from local vs. estimated noise variance, so
    flat/homogeneous areas get smoothed hard and edges are preserved.
    """
    image = image.astype(np.float32)
    mean = uniform_filter(image, size=window_size)
    mean_sq = uniform_filter(image ** 2, size=window_size)
    local_var = mean_sq - mean ** 2
    local_var = np.clip(local_var, a_min=1e-6, a_max=None)

    overall_var = float(image.var())
    overall_var = max(overall_var, 1e-6)

    k = local_var / (local_var + overall_var)
    return mean + k * (image - mean)


def tile_image(image: np.ndarray, tile_size: int = 256, stride: int | None = None) -> list[np.ndarray]:
    """
    Cut `image` into tile_size x tile_size patches. Partial edge tiles are
    dropped rather than padded, since padded tiles would need masking later
    to avoid the model learning from artificial borders.
    """
    stride = stride or tile_size
    h, w = image.shape[:2]
    tiles = []
    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            tiles.append(image[y:y + tile_size, x:x + tile_size])
    return tiles


def preprocess_image(path: str | Path, tile_size: int = 256) -> dict:
    """Run the full pipeline on one image and return inputs/outputs for inspection."""
    raw = load_sar_image(path)
    calibration = check_calibration(raw)
    despeckled = lee_filter(raw)
    tiles = tile_image(despeckled, tile_size=tile_size)
    return {
        "raw": raw,
        "despeckled": despeckled,
        "calibration": calibration,
        "tiles": tiles,
    }
