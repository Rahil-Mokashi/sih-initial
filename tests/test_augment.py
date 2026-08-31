"""
Tests for src/detection/augment.py's augment_pair, specifically the
extra_masks parameter added so a nodata-validity mask can ride along with
image/mask through the same flips/rotation and stay pixel-aligned -- a
mask transformed independently (e.g. with its own RNG draws) would silently
drift out of alignment with the image it's supposed to describe.
"""

import numpy as np

from detection.augment import augment_pair


def test_extra_mask_stays_aligned_with_mask_through_geometry():
    # A mask and a "nodata" companion array that starts IDENTICAL to it --
    # if both get the same geometric transform, they must stay identical
    # after augmentation, for every RNG draw the test exercises.
    mask = np.zeros((6, 6), dtype=np.float32)
    mask[0, 0] = 1.0
    mask[2, 3] = 1.0
    companion = mask.copy()
    image = np.random.default_rng(0).normal(size=(6, 6)).astype(np.float32)

    for seed in range(20):  # exercise many combinations of flip/flip/rotation
        rng = np.random.default_rng(seed)
        _, aug_mask, extras = augment_pair(image, mask, rng, speckle_jitter_std_db=0.0, extra_masks=[companion])
        assert np.array_equal(aug_mask, extras[0]), f"seed {seed}: companion mask drifted out of alignment"


def test_extra_mask_not_given_returns_two_tuple_unchanged():
    # Backward compatibility: existing callers that don't pass extra_masks
    # must keep getting exactly a 2-tuple back.
    mask = np.zeros((4, 4), dtype=np.float32)
    image = np.zeros((4, 4), dtype=np.float32)
    rng = np.random.default_rng(0)
    result = augment_pair(image, mask, rng, speckle_jitter_std_db=0.0)
    assert len(result) == 2


def test_speckle_jitter_does_not_touch_extra_masks():
    # The extra mask is a boolean-ish validity map, not image signal -- it
    # must NOT receive the additive dB-space speckle noise applied to image.
    mask = np.zeros((4, 4), dtype=np.float32)
    companion = np.array([[1.0, 0.0, 1.0, 0.0]] * 4, dtype=np.float32)
    image = np.zeros((4, 4), dtype=np.float32)
    rng = np.random.default_rng(1)
    _, _, extras = augment_pair(image, mask, rng, speckle_jitter_std_db=5.0, extra_masks=[companion])
    assert set(np.unique(extras[0]).tolist()) <= {0.0, 1.0}  # still exactly binary, no noise added
