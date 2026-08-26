"""Tests for src/drift/advect.py -- mainly the Ekman deflection rotation."""

import numpy as np

from drift.advect import _deflect_wind


def test_deflection_preserves_magnitude():
    u, v = _deflect_wind(np.array([3.0]), np.array([4.0]), np.array([33.0]))
    mag_before = np.hypot(3.0, 4.0)
    mag_after = np.hypot(u, v)
    assert abs(mag_before - mag_after[0]) < 1e-9


def test_northern_hemisphere_deflects_right_of_wind():
    # Pure eastward wind (u=1, v=0) at a Northern Hemisphere latitude:
    # "right of eastward" is south, so v should come out negative.
    u, v = _deflect_wind(np.array([1.0]), np.array([0.0]), np.array([33.0]))
    assert v[0] < 0
    assert u[0] > 0  # still mostly eastward at a 20deg deflection


def test_southern_hemisphere_deflects_left_of_wind():
    # Same eastward wind south of the equator: "left of eastward" is north.
    u, v = _deflect_wind(np.array([1.0]), np.array([0.0]), np.array([-33.0]))
    assert v[0] > 0
    assert u[0] > 0


def test_zero_wind_stays_zero():
    u, v = _deflect_wind(np.array([0.0]), np.array([0.0]), np.array([33.0]))
    assert u[0] == 0.0
    assert v[0] == 0.0
