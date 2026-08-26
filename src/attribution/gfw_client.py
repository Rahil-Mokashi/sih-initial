"""
Reusable Global Fishing Watch API client for the attribution stage.
Extracted from scripts/test_gfw_api.py once a second caller (the scoring
module) needed the same request logic -- see DECISIONS.md "GFW API
request format" for how the real request shape was found (geojson as a
plain Polygon object, spatial-resolution as a required param).
"""

from __future__ import annotations

import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

API_BASE = "https://gateway.api.globalfishingwatch.org"
REPORT_ENDPOINT = f"{API_BASE}/v3/4wings/report"

DEFAULT_DATASET = "public-global-presence:latest"


class GFWError(RuntimeError):
    pass


def fetch_vessel_presence(
    bbox: tuple[float, float, float, float],  # (min_lon, min_lat, max_lon, max_lat)
    date_range: tuple[str, str],
    spatial_resolution: str = "LOW",
    temporal_resolution: str = "DAILY",
    dataset: str = DEFAULT_DATASET,
) -> list[dict]:
    """
    Returns a flat list of vessel-presence records (one per vessel per
    date/grid-cell) from the 4Wings report API. Each record includes
    mmsi/imo/shipName/flag/vesselType, date, entryTimestamp/exitTimestamp,
    hours, and lat/lon -- note lat/lon are the *grid cell* center at the
    requested spatial_resolution, not an exact vessel position (see
    DECISIONS.md "Attribution scoring" for why this matters for scoring).
    """
    token = os.environ.get("GFW_API_TOKEN")
    if not token:
        raise GFWError("GFW_API_TOKEN is not set (see .env / DECISIONS.md 'Secrets / .env').")

    min_lon, min_lat, max_lon, max_lat = bbox
    geojson = {
        "type": "Polygon",
        "coordinates": [[
            [min_lon, min_lat], [max_lon, min_lat], [max_lon, max_lat], [min_lon, max_lat], [min_lon, min_lat],
        ]],
    }
    params = {
        "spatial-resolution": spatial_resolution,
        "temporal-resolution": temporal_resolution,
        "group-by": "VESSEL_ID",
        "datasets[0]": dataset,
        "date-range": f"{date_range[0]},{date_range[1]}",
        "format": "JSON",
    }
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    resp = requests.post(REPORT_ENDPOINT, headers=headers, params=params, json={"geojson": geojson}, timeout=60)
    if resp.status_code != 200:
        raise GFWError(f"GFW API returned {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    records = []
    for entry in data.get("entries", []):
        for dataset_key, rows in entry.items():
            records.extend(rows)
    return records
