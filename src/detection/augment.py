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

image may be (H, W) (single-band) or (C, H, W) (e.g. the real VV+VH
dual-pol bands) -- all geometric ops below use axis=-1/-2 (last two axes)
rather than np.fliplr/flipud/rot90's implicit axes 0/1, so the same code
flips/rotates image and mask consistently regardless of whether image
carries a leading channel dim (mask is always (H, W), whose axes 0/1 ARE
its last two axes, so this is a no-op behavior change for the
single-channel case).
"""

from __future__ import annotations

import numpy as np


def augment_pair(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator, speckle_jitter_std_db: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        image, mask = np.flip(image, axis=-1), np.flip(mask, axis=-1)
    if rng.random() < 0.5:
        image, mask = np.flip(image, axis=-2), np.flip(mask, axis=-2)

    k = rng.integers(0, 4)  # 0/90/180/270 degree rotation
    if k:
        image, mask = np.rot90(image, k, axes=(-2, -1)), np.rot90(mask, k, axes=(-2, -1))

    if speckle_jitter_std_db > 0:
        image = image + rng.normal(0, speckle_jitter_std_db, size=image.shape).astype(np.float32)

    return np.ascontiguousarray(image), np.ascontiguousarray(mask)
