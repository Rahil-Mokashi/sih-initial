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


def augment_pair(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    speckle_jitter_std_db: float = 0.5,
    extra_masks: list[np.ndarray] | None = None,
):
    """
    extra_masks, if given (e.g. a nodata-validity mask that must stay
    pixel-aligned with mask/image), is a list of additional (H, W) arrays
    that get the exact same flips/rotation as image/mask -- same RNG draws,
    same order, so alignment is preserved -- but NOT the speckle jitter
    (that's image-signal noise, meaningless for a boolean validity mask).
    Returns (image, mask) as before when extra_masks is None (unchanged
    behavior/RNG consumption for every existing caller); returns
    (image, mask, transformed_extra_masks) when extra_masks is given.
    """
    flip_lr = rng.random() < 0.5
    flip_ud = rng.random() < 0.5
    k = rng.integers(0, 4)  # 0/90/180/270 degree rotation

    def apply_geometry(arr: np.ndarray) -> np.ndarray:
        if flip_lr:
            arr = np.flip(arr, axis=-1)
        if flip_ud:
            arr = np.flip(arr, axis=-2)
        if k:
            arr = np.rot90(arr, k, axes=(-2, -1))
        return np.ascontiguousarray(arr)

    image = apply_geometry(image)
    mask = apply_geometry(mask)

    if speckle_jitter_std_db > 0:
        image = image + rng.normal(0, speckle_jitter_std_db, size=image.shape).astype(np.float32)
    image = np.ascontiguousarray(image)

    if extra_masks is None:
        return image, mask
    return image, mask, [apply_geometry(m) for m in extra_masks]
