"""
Augmentation for real SAR tile training. Kept deliberately simple and
SAR-appropriate:
  - horizontal/vertical flips and 90-degree rotations: geometry-safe, no
    interpolation artifacts (unlike arbitrary-angle rotation), and SAR
    backscatter has no canonical "up" the way optical imagery has a horizon,
    so these are physically reasonable label-preserving transforms.
  - mild speckle jitter: despeckling (src/detection/preprocess.py's Lee
    filter) reduces but doesn't eliminate speckle. SAR speckle is
    multiplicative in the linear domain, which is roughly additive once
    the image is in dB (log) space -- so jitter is implemented as small
    additive Gaussian noise in dB space, not multiplicative noise in
    linear space, to match how the despeckled input is actually
    represented.
No random-angle rotation, no elastic/perspective warps, no color jitter
(single-channel radar, "color" doesn't apply) -- those would either need
interpolation (softening real edges we want the model to key on) or don't
correspond to anything physically real for SAR.
"""

from __future__ import annotations

import numpy as np


def augment_pair(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator, speckle_jitter_std_db: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        image, mask = np.fliplr(image), np.fliplr(mask)
    if rng.random() < 0.5:
        image, mask = np.flipud(image), np.flipud(mask)

    k = rng.integers(0, 4)  # 0/90/180/270 degree rotation
    if k:
        image, mask = np.rot90(image, k), np.rot90(mask, k)

    if speckle_jitter_std_db > 0:
        image = image + rng.normal(0, speckle_jitter_std_db, size=image.shape).astype(np.float32)

    return np.ascontiguousarray(image), np.ascontiguousarray(mask)
