"""
Test connectivity + authentication against the Global Fishing Watch API,
and pull real AIS vessel presence for case ow-0001 (see DECISIONS.md /
LOG.md, drift track): 33.06E, 33.26N, Eastern Mediterranean, around
2019-01-01T03:42:35 UTC.

Docs: https://globalfishingwatch.org/our-apis
API reference (fetched from the underlying markdown, not just the SPA
landing page -- see DECISIONS.md "GFW API request format" for how):
https://globalfishingwatch.org/our-apis/documentation/docs/v3/4wings/report
https://globalfishingwatch.org/our-apis/documentation/docs/examples/report/report-example1
https://globalfishingwatch.org/our-apis/documentation/docs/examples/report/report-example10

Requires a free API token (see DECISIONS.md, "Global Fishing Watch API"
section, for the registration steps: sign up, then generate a token at
https://globalfishingwatch.org/our-apis/tokens). Loaded from a `.env` file
at the project root (GFW_API_TOKEN=... -- see DECISIONS.md "Secrets /
.env" section) via python-dotenv, or from a real environment variable if
you'd rather set it that way -- this script never hardcodes or stores the
token itself:

    Windows (PowerShell):  $env:GFW_API_TOKEN = "your-token-here"
    Windows (Git Bash):    export GFW_API_TOKEN="your-token-here"

This calls the 4Wings Report API (POST /v3/4wings/report) for AIS vessel
presence in a small bounding box around case ow-0001, over a 2-week window
centered on the detection date, as a minimal "can we authenticate and get
real data back" check for the attribution stage.
"""

import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

API_BASE = "https://gateway.api.globalfishingwatch.org"
REPORT_ENDPOINT = f"{API_BASE}/v3/4wings/report"

# ow-0001 case (see DECISIONS.md): real oil detection at 33.06E, 33.26N,
# 2019-01-01T03:42:35 UTC. Bounding box padded ~1 deg around the detection
# point; date range is a 2-week window centered on the detection date (AIS
# presence reports are aggregated, so a single instant wouldn't give much
# to look at).
CASE_LON, CASE_LAT = 33.058, 33.259
PAD_DEG = 1.0
SAMPLE_BBOX = {
    "min_lon": CASE_LON - PAD_DEG,
    "min_lat": CASE_LAT - PAD_DEG,
    "max_lon": CASE_LON + PAD_DEG,
    "max_lat": CASE_LAT + PAD_DEG,
}
SAMPLE_DATE_RANGE = ("2018-12-25", "2019-01-08")

# "public-global-presence:latest" = general vessel presence (any AIS-visible
# vessel), not just fishing activity -- the better match for attribution work.
DATASET = "public-global-presence:latest"


def main() -> None:
    token = os.environ.get("GFW_API_TOKEN")
    if not token:
        print("GFW_API_TOKEN is not set in the environment.")
        print("Register for a free API token at https://globalfishingwatch.org/our-apis")
        print("then generate one at https://globalfishingwatch.org/our-apis/tokens")
        print("and set GFW_API_TOKEN before re-running this script.")
        print("(This is expected to fail right now -- see DECISIONS.md action item.)")
        sys.exit(1)

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    min_lon, min_lat = SAMPLE_BBOX["min_lon"], SAMPLE_BBOX["min_lat"]
    max_lon, max_lat = SAMPLE_BBOX["max_lon"], SAMPLE_BBOX["max_lat"]

    # geojson is a plain Polygon object -- NOT a FeatureCollection, and NOT
    # a stringified JSON blob. (An earlier version of this script got both
    # of those wrong, which is what caused the 422 "body malformed" error --
    # see DECISIONS.md.)
    geojson = {
        "type": "Polygon",
        "coordinates": [[
            [min_lon, min_lat],
            [max_lon, min_lat],
            [max_lon, max_lat],
            [min_lon, max_lat],
            [min_lon, min_lat],
        ]],
    }

    params = {
        "spatial-resolution": "LOW",  # required param, missing before -- see DECISIONS.md
        "temporal-resolution": "DAILY",
        "group-by": "VESSEL_ID",
        "datasets[0]": DATASET,
        "date-range": f"{SAMPLE_DATE_RANGE[0]},{SAMPLE_DATE_RANGE[1]}",
        "format": "JSON",
    }
    body = {"geojson": geojson}

    print(f"POST {REPORT_ENDPOINT}")
    print(f"params: {params}")
    print(f"bbox: {SAMPLE_BBOX}, date range: {SAMPLE_DATE_RANGE}")

    try:
        resp = requests.post(REPORT_ENDPOINT, headers=headers, params=params, json=body, timeout=60)
    except requests.RequestException as e:
        print(f"ERROR: request failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print("SUCCESS -- authenticated and got real data back.")
        entries = data.get("entries", [])
        n_records = sum(len(v) for e in entries for v in e.values()) if entries else 0
        print(f"total: {data.get('total')}, records returned: {n_records}")
        print(json.dumps(data, indent=2)[:2000])
    elif resp.status_code == 401:
        print("AUTH FAILED -- token is missing/invalid/expired.")
        print(resp.text[:500])
        sys.exit(1)
    else:
        print(f"Unexpected response ({resp.status_code}):")
        print(resp.text[:1000])
        sys.exit(1)


if __name__ == "__main__":
    main()
