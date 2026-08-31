"""Tests for src/detection/preprocess.py's compute_nodata_mask."""

import numpy as np

from detection.preprocess import compute_nodata_mask


def test_exact_zero_in_both_bands_is_nodata():
    band1 = np.array([[0.0, -12.3], [-30.0, 0.0]], dtype=np.float32)
    band2 = np.array([[0.0, -8.1], [-25.0, 0.0]], dtype=np.float32)
    mask = compute_nodata_mask(band1, band2)
    assert mask.tolist() == [[True, False], [False, True]]


def test_zero_in_only_one_band_is_not_nodata():
    # A real-signal pixel can legitimately be near 0.0 dB in one band while
    # the other band is clearly non-zero -- only exact-zero in BOTH bands at
    # the same location counts as nodata (see the function's docstring for
    # why: that's what Phase 0's audit found to be the real nodata
    # signature, not a coincidental low-backscatter pixel).
    band1 = np.array([0.0, -12.3])
    band2 = np.array([-5.0, -8.1])
    mask = compute_nodata_mask(band1, band2)
    assert mask.tolist() == [False, False]


def test_no_nodata_pixels_gives_all_false():
    band1 = np.array([-10.0, -20.0, -30.0])
    band2 = np.array([-5.0, -15.0, -25.0])
    mask = compute_nodata_mask(band1, band2)
    assert not mask.any()


def test_all_nodata_gives_all_true():
    band1 = np.zeros((3, 3), dtype=np.float32)
    band2 = np.zeros((3, 3), dtype=np.float32)
    mask = compute_nodata_mask(band1, band2)
    assert mask.all()
