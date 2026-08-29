"""Tests for src/detection/geometry.py's mask characterization."""

import numpy as np

from detection.geometry import characterize_mask


def test_empty_mask_returns_zeros_not_error():
    mask = np.zeros((100, 100))
    result = characterize_mask(mask)
    assert result["area_px"] == 0
    assert result["length_px"] is None
    assert result["elongation"] is None


def test_area_matches_real_pixel_count():
    mask = np.zeros((100, 100))
    mask[10:20, 10:30] = 1  # 10 x 20 = 200 real "oil" pixels
    result = characterize_mask(mask)
    assert result["area_px"] == 200


def test_elongated_rectangle_has_length_greater_than_width():
    mask = np.zeros((100, 100))
    mask[45:55, 5:95] = 1  # 10 tall x 90 wide -- clearly elongated
    result = characterize_mask(mask)
    assert result["length_px"] > result["width_px"]
    assert result["elongation"] > 5  # ~9x in this synthetic case


def test_square_has_elongation_near_one():
    mask = np.zeros((100, 100))
    mask[40:60, 40:60] = 1  # 20x20 square
    result = characterize_mask(mask)
    assert 0.9 < result["elongation"] < 1.1


def test_two_separate_blobs_counted_as_two_components():
    mask = np.zeros((100, 100))
    mask[10:20, 10:20] = 1
    mask[70:80, 70:80] = 1
    result = characterize_mask(mask)
    assert result["n_components"] == 2


def test_real_world_units_only_computed_when_pixel_size_given():
    mask = np.zeros((100, 100))
    mask[10:20, 10:30] = 1
    no_pixel_size = characterize_mask(mask)
    assert no_pixel_size["area_km2"] is None

    with_pixel_size = characterize_mask(mask, pixel_size_m=10.0)
    assert with_pixel_size["area_km2"] == round(200 * 10.0 ** 2 / 1e6, 4)
