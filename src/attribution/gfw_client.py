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
EVENTS_ENDPOINT = f"{API_BASE}/v3/events"

DEFAULT_DATASET = "public-global-presence:latest"
GAPS_DATASET = "public-global-gaps-events:latest"


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


MAX_VESSELS_PER_EVENTS_REQUEST = 20  # real, empirically-found cap on v3/events'
# `vessels[]` array param -- 20 succeeds, 21 returns a real 422 ("vessels must be
# an array" / "each value in vessels must be a string", a misleading message for
# what's actually an array-length limit). Not documented anywhere; found by
# binary-searching real requests. See DECISIONS.md "Full-pool behavioral
# rescoring" for why this matters (it's what makes checking the FULL candidate
# pool, not just a pre-filtered top-N, actually affordable).


def fetch_gap_events_batch(vessel_ids: list[str], date_range: tuple[str, str], limit_per_batch: int = 100) -> dict[str, list[dict]]:
    """
    Real AIS-gap ("went dark") events for potentially many vessels at once,
    from GFW's v3/events API -- see scripts/test_gfw_events_api.py for how
    this endpoint's real behavior was found (a NEW test, separate from the
    4wings/report connectivity test above), and DECISIONS.md "Attribution
    scoring: behavioral-anomaly (AIS gap) sub-score" / "Full-pool behavioral
    rescoring" for the full writeup.

    Filtered by vessel ID rather than a geographic bbox: this token's
    permission tier returns 403 "Not authorized by permissions" on
    v3/events' POST-with-geometry spatial filter (confirmed by direct
    test -- unlike v3/4wings/report, which does allow geojson POST with
    this same token), so geographic filtering isn't available here. Vessel
    ID filtering doesn't hit that restriction and is exactly what's needed
    anyway: this runs against already-identified candidate vessels (from
    fetch_vessel_presence's distance+timing pass), not a fresh area search.

    Batches vessel_ids MAX_VESSELS_PER_EVENTS_REQUEST at a time (one real
    GET per batch) rather than one request per vessel -- confirmed via
    direct test that a single request can carry up to 20 vessel IDs and
    return events for all of them together, which is what makes checking
    an entire raw candidate pool (hundreds of vessels) actually cheap:
    ceil(n_vessels / 20) requests total, not n_vessels.

    Returns {vessel_id: [events]} -- vessels with zero real gap events in
    the window are simply absent from the dict (not an error).

    Each real returned event includes a `gap` sub-object with
    intentionalDisabling (bool, GFW's own suspected-deliberate-shutoff
    flag), durationHours, distanceKm, impliedSpeedKnots, and
    on/offPosition -- confirmed by direct real-token test, not assumed
    from docs alone.
    """
    token = os.environ.get("GFW_API_TOKEN")
    if not token:
        raise GFWError("GFW_API_TOKEN is not set (see .env / DECISIONS.md 'Secrets / .env').")

    headers = {"Authorization": f"Bearer {token}"}
    events_by_vessel: dict[str, list[dict]] = {}

    for start in range(0, len(vessel_ids), MAX_VESSELS_PER_EVENTS_REQUEST):
        batch = vessel_ids[start:start + MAX_VESSELS_PER_EVENTS_REQUEST]
        params = [
            ("datasets[0]", GAPS_DATASET),
            ("start-date", date_range[0]),
            ("end-date", date_range[1]),
            ("limit", str(limit_per_batch)),
            ("offset", "0"),
        ]
        params += [(f"vessels[{i}]", vid) for i, vid in enumerate(batch)]

        resp = requests.get(EVENTS_ENDPOINT, headers=headers, params=params, timeout=60)
        if resp.status_code != 200:
            raise GFWError(f"GFW events API returned {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        total = data.get("total") or 0
        if total > limit_per_batch:
            # Real safeguard, not expected to trigger in practice: a batch of
            # <=20 vessels over a case's ~2-4 week window returning more than
            # limit_per_batch real gap events would mean silently missing some
            # rather than paginating for them -- surface it instead of hiding it.
            raise GFWError(
                f"gap events batch returned total={total} > limit_per_batch={limit_per_batch} "
                f"for {len(batch)} vessels -- pagination needed, not implemented, results would be incomplete."
            )
        for event in data.get("entries", []):
            vid = event.get("vessel", {}).get("id")
            if vid:
                events_by_vessel.setdefault(vid, []).append(event)

    return events_by_vessel
