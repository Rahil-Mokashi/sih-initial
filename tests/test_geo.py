"""Tests for src/common/geo.py."""

from common.geo import haversine_km


def test_same_point_is_zero():
    p = (33.058, 33.259)
    assert haversine_km(p, p) == 0.0


def test_known_distance_one_degree_longitude_at_equator():
    # 1 degree of longitude at the equator is ~111.2 km
    dist = haversine_km((0.0, 0.0), (1.0, 0.0))
    assert 110.5 < dist < 111.5


def test_known_distance_one_degree_latitude():
    # 1 degree of latitude is ~111.2 km anywhere (meridians converge, parallels don't)
    dist = haversine_km((30.0, 30.0), (30.0, 31.0))
    assert 110.5 < dist < 111.5


def test_symmetric():
    p1, p2 = (32.9503, 33.1981), (33.058, 33.259)
    assert abs(haversine_km(p1, p2) - haversine_km(p2, p1)) < 1e-9
