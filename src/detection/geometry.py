"""
Real geometric characterization of a detected oil-spill mask -- area,
length, width, orientation, and shape (elongation) -- per the SIH problem
statement's "(a) Detect and characterise the oil spill and calculate
geometric properties... if feasible." (Age estimation, also named in that
same clause, is explicitly "if feasible" and not implemented -- see
DECISIONS.md.)

Uses cv2 (already a project dependency for despeckling) rather than
adding a new one. Real image-moment/contour geometry, not fancy: the
minimum-area bounding rectangle around the largest mask component gives
length/width/orientation in one standard, well-understood operation.

Real-world units (area_km2, length_m, width_m) are only computed when
pixel_size_m is explicitly passed -- never assumed. The Zenodo training
tiles this project trains on carry no real geotransform (rasterio warns
"NotGeoreferencedWarning" on every read; see LOG.md), so there is no
reliable per-image ground sample distance to convert pixels to meters
with. Pass pixel_size_m only when it's actually known for that specific
image (e.g. 10m for a confirmed Sentinel-1 IW GRDH product, the real
product type of the two validated PANGAEA cases -- see DECISIONS.md).
"""

from __future__ import annotations

import cv2
import numpy as np


def characterize_mask(mask: np.ndarray, pixel_size_m: float | None = None) -> dict:
    """
    mask: 2D array, oil pixels > 0. Returns a dict of real geometric
    properties, always in pixel units, plus real-world units too if
    pixel_size_m is given.
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    area_px = int(binary.sum())

    empty = {
        "area_px": 0, "n_components": 0, "length_px": None, "width_px": None,
        "orientation_deg": None, "elongation": None, "centroid_px": None,
        "area_m2": None, "area_km2": None, "length_m": None, "width_m": None,
    }
    if area_px == 0:
        return empty

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return empty

    # Real spills can be multi-part (a slick can fragment); characterize the
    # largest connected component, but report how many real components exist
    # so a multi-part detection isn't silently reduced to just one number.
    largest = max(contours, key=cv2.contourArea)
    (cx, cy), (w, h), angle = cv2.minAreaRect(largest)
    length_px, width_px = max(w, h), min(w, h)

    result = {
        "area_px": area_px,
        "n_components": len(contours),
        "length_px": round(float(length_px), 1),
        "width_px": round(float(width_px), 1),
        # As returned by cv2.minAreaRect (degrees, referenced to the rectangle's
        # width axis) -- not normalized to a compass bearing.
        "orientation_deg": round(float(angle), 1),
        # Length/width ratio -- a real, basic shape cue: elongated features
        # (>~3-4x) are more consistent with a real wind/current-driven slick;
        # compact/circular shapes are more often look-alikes (biogenic slicks,
        # wind shadows) per the problem statement's "characterise... shape".
        # A heuristic cue to show alongside the numbers, not a classifier.
        "elongation": round(float(length_px / width_px), 2) if width_px > 1e-6 else None,
        "centroid_px": [round(float(cx), 1), round(float(cy), 1)],
    }

    if pixel_size_m is not None:
        result["area_m2"] = round(area_px * pixel_size_m ** 2, 1)
        result["area_km2"] = round(area_px * pixel_size_m ** 2 / 1e6, 4)
        result["length_m"] = round(length_px * pixel_size_m, 1)
        result["width_m"] = round(width_px * pixel_size_m, 1)
    else:
        result["area_m2"] = result["area_km2"] = result["length_m"] = result["width_m"] = None

    return result
