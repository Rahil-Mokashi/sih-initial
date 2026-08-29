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

Extended per the SIH problem statement's explicit ask for trajectory and
behavioural-anomaly scoring, not just proximity+timing (see DECISIONS.md
"Attribution scoring: behavioral-anomaly (AIS gap) sub-score" for the
real GFW API test that determined what's actually buildable):
  - trajectory evidence (n_presence_records, closest_approach_km) comes
    from the SAME presence records already fetched for the top-N
    candidates -- not a new API call. score_vessels() above still keeps
    only each vessel's single closest-in-time row; trajectory_evidence()
    below looks at ALL of that vessel's real rows in the query window to
    see whether its track came even closer to the origin at some other
    point, which the single closest-in-time row alone can't show.
  - behavioral-anomaly evidence is a real AIS-gap check via GFW's
    v3/events API (gaps dataset), confirmed by direct real-token test to
    return `intentionalDisabling` (GFW's own suspected-deliberate-shutoff
    flag), duration, and distance per gap -- this is genuinely available
    and implemented as `composite_score`, a SEPARATE field alongside the
    original `score`, not blended into it silently.
  - composite_score is computed against the FULL raw candidate pool (all
    355-1006+ vessels depending on case), not just a distance+timing
    pre-filtered top-N -- this was originally scoped to only the top 15
    (behavioral checks looked like they'd cost one GFW call per vessel),
    but a real, un-documented finding changed that: v3/events' `vessels[]`
    filter accepts up to 20 vessel IDs per request (confirmed by direct
    binary-search test -- 20 succeeds, 21 returns a real 422), so checking
    the entire pool costs ceil(n_candidates / 20) requests, not
    n_candidates -- ~18 for a 355-vessel pool, ~51 for a 1006-vessel one.
    Trivial against the real 50,000/day quota. See gfw_client.py's
    `fetch_gap_events_batch()` and DECISIONS.md "Full-pool behavioral
    rescoring" for the real numbers and why the earlier top-15-only
    version was a real selection-bias risk (a vessel with a genuine AIS
    gap but middling proximity/timing would never have been checked).
  - course/heading and instantaneous per-position speed, which the
    problem statement's "trajectory" language could also suggest, were
    checked directly against real GFW v3/events responses (all 5 event
    types) and are NOT present in any event schema at this API tier --
    a confirmed external constraint, not something this implementation
    skipped.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from attribution.gfw_client import (  # noqa: E402
    MAX_VESSELS_PER_EVENTS_REQUEST, GFWError, fetch_gap_events_batch, fetch_vessel_presence,
)
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

# Behavioral-anomaly (AIS gap) adjustment -- subtracted from `score` to
# produce `composite_score` (lower score = more consistent with being the
# source, same convention as distance/time above). An intentional AIS gap
# is a real, well-recognized "went dark" red flag; an unflagged gap is
# weaker evidence (could be a real receiver blackspot, not deliberate) so
# gets a smaller bonus. Documented, not tuned -- same honesty as the
# distance/time scales above.
GAP_INTENTIONAL_BONUS = 0.15
GAP_ANY_BONUS = 0.05

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


def trajectory_evidence(records: list[dict], vessel_id: str, origin: tuple[float, float]) -> dict:
    """
    Real trajectory-consistency evidence for one vessel, from ALL of its
    presence rows in the query window (not just the single closest-in-time
    row score_vessels() keeps). See module docstring for why this
    substitutes for a true continuous track, which GFW's API doesn't
    expose at this tier.
    """
    distances = [
        haversine_km(origin, (r["lon"], r["lat"]))
        for r in records
        if r.get("vesselId") == vessel_id
    ]
    return {
        "n_presence_records": len(distances),
        "closest_approach_km": round(min(distances), 2) if distances else None,
    }


def behavior_evidence(events: list[dict]) -> dict:
    """
    Real AIS-gap behavioral-anomaly evidence for one vessel, from its real
    gap events (already fetched in bulk for the whole candidate pool by
    gfw_client.fetch_gap_events_batch -- see score_case()). Returns the
    single longest-duration gap event if any exist (a vessel could have
    several; the most sustained one is the most evidentially interesting),
    or a real "no gap" result -- never fabricated.
    """
    if not events:
        return {"ais_gap_count": 0, "ais_gap_intentional": None,
                "ais_gap_duration_hours": None, "ais_gap_distance_km": None}

    longest = max(events, key=lambda e: e.get("gap", {}).get("durationHours", 0) or 0)
    gap = longest.get("gap", {})
    return {
        "ais_gap_count": len(events),
        "ais_gap_intentional": gap.get("intentionalDisabling"),
        "ais_gap_duration_hours": gap.get("durationHours"),
        "ais_gap_distance_km": float(gap["distanceKm"]) if gap.get("distanceKm") is not None else None,
    }


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

    # Behavioral evidence is checked against the FULL raw candidate pool, not
    # just a distance+timing pre-filtered top-N -- a real, un-documented API
    # finding (v3/events' vessels[] filter accepts up to
    # gfw_client.MAX_VESSELS_PER_EVENTS_REQUEST=20 IDs per request, confirmed by
    # direct binary-search test) makes this cheap: ceil(n/20) requests for the
    # whole pool, not n. Checking only a pre-filtered top-N was a real
    # selection-bias risk -- a vessel with a genuine AIS gap but middling
    # proximity/timing would never have been checked at all. See DECISIONS.md
    # "Full-pool behavioral rescoring".
    all_vessel_ids = [v.vessel_id for v in ranked]
    n_batches = -(-len(all_vessel_ids) // MAX_VESSELS_PER_EVENTS_REQUEST)  # ceil div
    print(f"fetching real AIS-gap behavioral evidence for all {len(ranked)} candidates "
          f"({n_batches} batched GFW requests, {MAX_VESSELS_PER_EVENTS_REQUEST} vessels/request)...")
    try:
        gap_events_by_vessel = fetch_gap_events_batch(all_vessel_ids, date_range)
    except GFWError as e:
        print(f"  WARNING: gap-events fetch failed ({e}) -- falling back to no behavioral evidence for this case.")
        gap_events_by_vessel = {}

    enriched = []
    for v in ranked:
        traj = trajectory_evidence(records, v.vessel_id, origin)
        behavior = behavior_evidence(gap_events_by_vessel.get(v.vessel_id, []))
        bonus = GAP_INTENTIONAL_BONUS if behavior.get("ais_gap_intentional") else (
            GAP_ANY_BONUS if behavior.get("ais_gap_count") else 0.0
        )
        composite_score = max(v.score - bonus, 0.0)
        enriched.append((v, traj, behavior, composite_score))

    enriched.sort(key=lambda row: row[3])  # re-rank the FULL pool by composite_score

    n_with_gaps = sum(1 for _, _, behavior, _ in enriched if behavior.get("ais_gap_count"))
    print(f"{n_with_gaps} of {len(enriched)} candidates had a real AIS gap event in the window\n")

    top_enriched = enriched[:TOP_N]  # final displayed ranking, now selected AFTER full-pool behavioral scoring
    for i, (v, traj, behavior, composite_score) in enumerate(top_enriched, 1):
        gap_note = (f"GAP(intentional={behavior['ais_gap_intentional']}, n={behavior['ais_gap_count']})"
                    if behavior.get("ais_gap_count") else "no AIS gap")
        print(f"{i:2d}. {v.ship_name or '(unnamed)':<20} MMSI={v.mmsi:<12} flag={v.flag:<4} "
              f"dist={v.distance_km:6.1f}km  time_gap={v.time_gap_hours:5.1f}h  score={v.score:.3f}  "
              f"composite={composite_score:.3f}  {gap_note}  closest_approach={traj['closest_approach_km']}km")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "case": drift["case"],
        "origin_estimate": {"lon": origin[0], "lat": origin[1], "time_utc": origin_time.isoformat()},
        "methodology": {
            "distance_scale_km": DISTANCE_SCALE_KM, "time_scale_hours": TIME_SCALE_HOURS,
            "distance_weight": DISTANCE_WEIGHT, "time_weight": TIME_WEIGHT,
            "gap_intentional_bonus": GAP_INTENTIONAL_BONUS, "gap_any_bonus": GAP_ANY_BONUS,
            "note": "First-pass heuristic scoring, not calibrated against labeled ground truth. "
                    "Lower score = more consistent with being the spill source. `score` is "
                    "proximity+timing only (the original methodology); `composite_score` additionally "
                    "factors in real AIS-gap behavioral evidence and is what the ranking below is sorted "
                    "by. Behavioral/trajectory evidence is checked against the FULL candidate pool "
                    "(n_candidates below), not just the vessels shown -- see "
                    "src/attribution/score_vessels.py module docstring and DECISIONS.md 'Full-pool "
                    "behavioral rescoring' for the real per-case request counts this took.",
        },
        "n_candidates": len(ranked),
        "n_candidates_with_ais_gap": n_with_gaps,
        "ranking": [
            {
                "rank": i + 1, "vessel_id": v.vessel_id, "mmsi": v.mmsi, "imo": v.imo,
                "ship_name": v.ship_name, "flag": v.flag, "vessel_type": v.vessel_type,
                "distance_km": round(v.distance_km, 2), "time_gap_hours": round(v.time_gap_hours, 2),
                "lon": v.lon, "lat": v.lat,
                "evidence_date": v.evidence_date, "entry_timestamp": v.entry_timestamp,
                "exit_timestamp": v.exit_timestamp, "hours_present": v.hours_present,
                "score": round(v.score, 4),
                "composite_score": round(composite_score, 4),
                # Derived, not an independently-measured probability -- see
                # methodology.note above. Named match_score_pct (not "confidence")
                # deliberately: this is an uncalibrated composite, not a probability
                # of guilt, and "confidence" reads as calibrated to anyone skimming
                # the dashboard. match_score_pct = (1 - composite_score) * 100.
                "match_score_pct": round((1 - min(composite_score, 1.0)) * 100, 1),
                # Trajectory evidence: real, from this vessel's OTHER presence rows in the
                # window (see trajectory_evidence() docstring) -- not a new API call.
                "n_presence_records": traj["n_presence_records"],
                "closest_approach_km": traj["closest_approach_km"],
                # Behavioral-anomaly evidence: real AIS-gap check via GFW v3/events (see
                # behavior_evidence() docstring). ais_gap_intentional is GFW's own
                # suspected-deliberate-shutoff flag, None if no gap event exists at all.
                "ais_gap_count": behavior["ais_gap_count"],
                "ais_gap_intentional": behavior["ais_gap_intentional"],
                "ais_gap_duration_hours": behavior["ais_gap_duration_hours"],
                "ais_gap_distance_km": behavior["ais_gap_distance_km"],
            }
            for i, (v, traj, behavior, composite_score) in enumerate(top_enriched)
        ],
    }
    out_path.write_text(json.dumps(output, indent=2))
    print(f"wrote {out_path}")


def main() -> None:
    for case_id in CASE_IDS:
        score_case(case_id)


if __name__ == "__main__":
    main()
