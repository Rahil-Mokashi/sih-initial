"""Small shared geo utilities. Factored out once needed by both
src/dashboard/build_map.py and src/attribution/ -- kept tiny on purpose."""

from __future__ import annotations

import math


def haversine_km(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Great-circle distance in km between two (lon, lat) points."""
    lon1, lat1 = p1
    lon2, lat2 = p2
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))
