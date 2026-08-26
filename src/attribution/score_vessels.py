"""
Score and rank candidate vessels against a drift-estimated spill origin.

Per the problem statement, this produces a ranked, evidence-backed list --
not a single verdict. Each candidate's score is broken into its two real
components (proximity, timing) so the ranking is inspectable, not a black
box.

Methodology (first pass -- see DECISIONS.md "Attribution scoring" for the
full writeup, including known limitations):
  - origin estimate = the drift skeleton's centroid (src/drift/) and an
    origin TIME = detection_time - hours_back (the drift window's start).
  - candidate vessels = real AIS presence records from GFW
    (src/attribution/gfw_client.py) in a box around the case.
  - each vessel's evidence row = whichever of its presence records is
    closest in time to the origin estimate (a vessel might have several
    rows across the query window).
  - score = weighted combination of (distance from origin, in km) and
    (time gap from the origin estimate, in hours), both normalized against
    documented scale constants -- NOT validated against any labeled ground
    truth (none exists yet for this project), so treat the ranking as a
    reasonable first-pass ordering to inspect, not a calibrated probability.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from attribution.gfw_client import fetch_vessel_presence  # noqa: E402
from common.geo import haversine_km  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "dashboard"

CASE_IDS = ["ow-0001", "ow-0002"]  # every case with a drift_{case}.json gets scored

# 1 degree padding around the detection point comfortably covers the
# drift-estimated origin too (they're only a few km apart in practice --
# see DECISIONS.md). Date range: a week either side of the origin/detection
# times, generalizing the manually-chosen 2-week window originally
# validated for ow-0001 (see LOG.md "GFW API 422 fixed").
PAD_DEG = 1.0
DATE_WINDOW_DAYS = 7

# First-pass scoring weights/scales -- documented, not tuned. See module docstring.
DISTANCE_SCALE_KM = 50.0   # distance at which the proximity term saturates to 1.0 (fully "far")
TIME_SCALE_HOURS = 24.0    # time gap at which the timing term saturates to 1.0 -- matches the drift window
DISTANCE_WEIGHT = 0.5
TIME_WEIGHT = 0.5

TOP_N = 15


@dataclass
class VesselScore:
    vessel_id: str
    mmsi: str | None
    imo: str | None
    ship_name: str | None
    flag: str | None
    vessel_type: str | None
    distance_km: float
    time_gap_hours: float
    lon: float
    lat: float
    evidence_date: str
    entry_timestamp: str
    exit_timestamp: str
    hours_present: float
    score: float  # lower = more consistent with being the source


def parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def time_gap_hours(origin_time: datetime, entry: datetime, exit: datetime) -> float:
    """0 if origin_time falls within [entry, exit], else the gap to the nearer edge."""
    if entry <= origin_time <= exit:
        return 0.0
    return min(abs((origin_time - entry).total_seconds()), abs((origin_time - exit).total_seconds())) / 3600.0


def score_vessels(origin: tuple[float, float], origin_time: datetime, records: list[dict]) -> list[VesselScore]:
    best_by_vessel: dict[str, VesselScore] = {}

    for r in records:
        vessel_id = r.get("vesselId")
        if not vessel_id:
            continue
        try:
            entry = parse_iso(r["entryTimestamp"])
            exit_ = parse_iso(r["exitTimestamp"])
        except (KeyError, ValueError):
            continue

        dist_km = haversine_km(origin, (r["lon"], r["lat"]))
        gap_h = time_gap_hours(origin_time, entry, exit_)

        norm_dist = min(dist_km / DISTANCE_SCALE_KM, 1.0)
        norm_time = min(gap_h / TIME_SCALE_HOURS, 1.0)
        score = DISTANCE_WEIGHT * norm_dist + TIME_WEIGHT * norm_time

        candidate = VesselScore(
            vessel_id=vessel_id,
            mmsi=r.get("mmsi"), imo=r.get("imo"), ship_name=r.get("shipName"),
            flag=r.get("flag"), vessel_type=r.get("vesselType"),
            distance_km=dist_km, time_gap_hours=gap_h,
            lon=r["lon"], lat=r["lat"],
            evidence_date=r.get("date", ""), entry_timestamp=r.get("entryTimestamp", ""),
            exit_timestamp=r.get("exitTimestamp", ""), hours_present=r.get("hours", 0.0),
            score=score,
        )
        existing = best_by_vessel.get(vessel_id)
        if existing is None or candidate.score < existing.score:
            best_by_vessel[vessel_id] = candidate

    return sorted(best_by_vessel.values(), key=lambda v: v.score)


def score_case(case_id: str) -> None:
    drift_path = DATA_DIR / f"drift_{case_id.replace('-', '')}.json"
    out_path = DATA_DIR / f"vessel_ranking_{case_id.replace('-', '')}.json"
    if not drift_path.exists():
        print(f"SKIP {case_id}: {drift_path} not found (run scripts/export_drift_dashboard_data.py for it first).")
        return
    drift = json.loads(drift_path.read_text())

    era5 = next((s for s in drift["sources"] if s["name"] == "ERA5"), drift["sources"][0])
    origin = tuple(era5["centroid"])
    detection_time = parse_iso(drift["detection_time_utc"])
    origin_time = detection_time - timedelta(hours=drift["hours_back"])

    case_lon, case_lat = drift["detection_point"]
    bbox = (case_lon - PAD_DEG, case_lat - PAD_DEG, case_lon + PAD_DEG, case_lat + PAD_DEG)
    date_range = (
        (origin_time - timedelta(days=DATE_WINDOW_DAYS)).date().isoformat(),
        (detection_time + timedelta(days=DATE_WINDOW_DAYS)).date().isoformat(),
    )

    print(f"\n=== {case_id} ===")
    print(f"origin estimate: {origin[1]:.4f}N, {origin[0]:.4f}E at {origin_time.isoformat()}")
    print(f"fetching GFW vessel presence for bbox={bbox}, date_range={date_range}...")
    # HIGH resolution (finer grid cells) rather than the LOW default used in the
    # initial connectivity test -- confirmed to perform fine (~10s, similar record
    # count) and gives tighter proximity scoring; see DECISIONS.md "Attribution
    # scoring" for the LOW-resolution grid-coarseness limitation this addresses.
    records = fetch_vessel_presence(bbox, date_range, spatial_resolution="HIGH")
    print(f"got {len(records)} presence records")

    ranked = score_vessels(origin, origin_time, records)
    print(f"{len(ranked)} unique candidate vessels\n")

    top = ranked[:TOP_N]
    for i, v in enumerate(top, 1):
        print(f"{i:2d}. {v.ship_name or '(unnamed)':<20} MMSI={v.mmsi:<12} flag={v.flag:<4} "
              f"dist={v.distance_km:6.1f}km  time_gap={v.time_gap_hours:5.1f}h  score={v.score:.3f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "case": drift["case"],
        "origin_estimate": {"lon": origin[0], "lat": origin[1], "time_utc": origin_time.isoformat()},
        "methodology": {
            "distance_scale_km": DISTANCE_SCALE_KM, "time_scale_hours": TIME_SCALE_HOURS,
            "distance_weight": DISTANCE_WEIGHT, "time_weight": TIME_WEIGHT,
            "note": "First-pass heuristic scoring, not calibrated against labeled ground truth. "
                    "Lower score = more consistent with being the spill source.",
        },
        "n_candidates": len(ranked),
        "ranking": [
            {
                "rank": i + 1, "vessel_id": v.vessel_id, "mmsi": v.mmsi, "imo": v.imo,
                "ship_name": v.ship_name, "flag": v.flag, "vessel_type": v.vessel_type,
                "distance_km": round(v.distance_km, 2), "time_gap_hours": round(v.time_gap_hours, 2),
                "lon": v.lon, "lat": v.lat,
                "evidence_date": v.evidence_date, "entry_timestamp": v.entry_timestamp,
                "exit_timestamp": v.exit_timestamp, "hours_present": v.hours_present,
                "score": round(v.score, 4),
                # Derived, not an independently-measured probability -- see
                # methodology.note above. confidence_pct = (1 - score) * 100.
                "confidence_pct": round((1 - min(v.score, 1.0)) * 100, 1),
            }
            for i, v in enumerate(top)
        ],
    }
    out_path.write_text(json.dumps(output, indent=2))
    print(f"wrote {out_path}")


def main() -> None:
    for case_id in CASE_IDS:
        score_case(case_id)


if __name__ == "__main__":
    main()
