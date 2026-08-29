"""
Test connectivity + real response schema for GFW's v3/events API, against
a real token -- same pattern as test_gfw_api.py (which tests v3/4wings/report,
used for vessel presence). This is a NEW, separate test: score_vessels.py's
attribution scoring currently uses only distance+timing (see DECISIONS.md
"Attribution scoring"); the SIH problem statement also asks for trajectory
and behavioural-anomaly scoring (course deviations, speed changes, AIS
gaps), which would need per-event fields (course/heading/speed, gap
duration) that v3/4wings/report's presence records don't carry. This
script exists to find out, against real data, whether v3/events actually
returns those fields before any scoring code gets built on top of an
assumption.

Docs: https://globalfishingwatch.org/our-apis/documentation/docs/v3/events
Confirmed dataset identifiers (real doc text, not guessed):
    public-global-fishing-events:latest
    public-global-encounters-events:latest
    public-global-loitering-events:latest
    public-global-port-visits-events:latest
    public-global-gaps-events:latest   -- "For AIS off event, use this dataset"
        (confirms GAP is a real supported event type)

Usage:
    venv\\Scripts\\python.exe scripts\\test_gfw_events_api.py
"""

import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

API_BASE = "https://gateway.api.globalfishingwatch.org"
EVENTS_ENDPOINT = f"{API_BASE}/v3/events"

# Same real ow-0001 case window used by test_gfw_api.py, widened slightly
# (gaps/encounters are rarer events than raw presence, so a 2-week window
# might turn up nothing near this specific bbox -- widen if the first real
# call comes back empty rather than assuming failure).
CASE_LON, CASE_LAT = 33.058, 33.259
DATE_RANGE = ("2018-12-01", "2019-01-15")

DATASETS = [
    "public-global-fishing-events:latest",
    "public-global-encounters-events:latest",
    "public-global-loitering-events:latest",
    "public-global-port-visits-events:latest",
    "public-global-gaps-events:latest",
]


def try_dataset(token: str, dataset: str) -> None:
    headers = {"Authorization": f"Bearer {token}"}
    params = [
        ("datasets[0]", dataset),
        ("start-date", DATE_RANGE[0]),
        ("end-date", DATE_RANGE[1]),
        ("limit", "5"),
        ("offset", "0"),
    ]
    print(f"\n--- {dataset} ---")
    print(f"GET {EVENTS_ENDPOINT} params={params}")
    try:
        resp = requests.get(EVENTS_ENDPOINT, headers=headers, params=params, timeout=60)
    except requests.RequestException as e:
        print(f"ERROR: request failed: {e}")
        return

    print(f"status: {resp.status_code}")
    if resp.status_code != 200:
        print(resp.text[:1500])
        return

    data = resp.json()
    entries = data.get("entries", [])
    total = data.get("total")
    print(f"total: {total}, entries returned: {len(entries)}")
    if entries:
        print("first real entry (full, unmodified):")
        print(json.dumps(entries[0], indent=2))
        keys_seen = set()
        for e in entries:
            keys_seen.update(e.keys())
        print(f"top-level keys across returned entries: {sorted(keys_seen)}")
    else:
        print("(no entries in this window/dataset -- not necessarily an API limitation, "
              "could just be no real events of this type near this bbox/date range)")


def main() -> None:
    token = os.environ.get("GFW_API_TOKEN")
    if not token:
        print("GFW_API_TOKEN is not set.")
        sys.exit(1)

    for dataset in DATASETS:
        try_dataset(token, dataset)


if __name__ == "__main__":
    main()
