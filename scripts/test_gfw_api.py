"""
Test connectivity + authentication against the Global Fishing Watch API.

Docs: https://globalfishingwatch.org/our-apis
API reference: https://globalfishingwatch.org/our-apis/documentation/docs/v3/4wings/report

Requires a free API token (see DECISIONS.md, "Global Fishing Watch API"
section, for the registration steps: sign up, then generate a token at
https://globalfishingwatch.org/our-apis/tokens). Set it as an environment
variable before running -- this script never hardcodes or stores it:

    Windows (PowerShell):  $env:GFW_API_TOKEN = "your-token-here"
    Windows (Git Bash):    export GFW_API_TOKEN="your-token-here"

This calls the 4Wings Report API (POST /v3/4wings/report) for vessel
presence in a small bounding box (a slice of the Eastern Mediterranean,
matching the PANGAEA dataset's coverage area) over a short date range, as a
minimal "can we authenticate and get data back" check.
"""

import json
import os
import sys

import requests

API_BASE = "https://gateway.api.globalfishingwatch.org"
REPORT_ENDPOINT = f"{API_BASE}/v3/4wings/report"

# Sample bounding box: a slice of the Eastern Mediterranean (matches the
# PANGAEA Eastern Med dataset coverage: ~30-36E, 31-34.7N)
SAMPLE_BBOX = {
    "min_lon": 32.0,
    "min_lat": 32.0,
    "max_lon": 33.0,
    "max_lat": 33.0,
}
SAMPLE_DATE_RANGE = ("2019-01-01", "2019-01-31")

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

    geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [min_lon, min_lat],
                    [max_lon, min_lat],
                    [max_lon, max_lat],
                    [min_lon, max_lat],
                    [min_lon, min_lat],
                ]],
            },
        }],
    }

    params = {
        "format": "JSON",
        "group-by": "VESSEL_ID",
        "temporal-resolution": "MONTHLY",
        "datasets[0]": DATASET,
        "date-range": f"{SAMPLE_DATE_RANGE[0]},{SAMPLE_DATE_RANGE[1]}",
    }
    # geojson must be sent as a JSON-encoded STRING inside the body, per the API spec.
    body = {"geojson": json.dumps(geojson)}

    print(f"POST {REPORT_ENDPOINT}")
    print(f"params: {params}")
    print(f"bbox: {SAMPLE_BBOX}, date range: {SAMPLE_DATE_RANGE}")

    try:
        resp = requests.post(REPORT_ENDPOINT, headers=headers, params=params, json=body, timeout=30)
    except requests.RequestException as e:
        print(f"ERROR: request failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print("SUCCESS -- authenticated and got a response back.")
        print(json.dumps(data, indent=2)[:1000])
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
