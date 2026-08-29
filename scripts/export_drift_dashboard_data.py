"""
Re-runs the drift skeleton for a given case (both wind sources) and dumps
real per-particle results to JSON for the dashboard map -- not just the
aggregate centroid/std from scripts/run_drift_skeleton.py's printed
output, but the actual final particle positions, so the map shows real
spread rather than a synthetic circle standing in for it.

CASES holds every real PANGAEA case we have metadata for (see
DECISIONS.md "Pipeline validated on a second real case" for why ow-0002
was added -- checking the pipeline isn't overfit to a single example).
Set CASE_ID below to pick which one runs.

Output: data/processed/dashboard/drift_{case_id}.json
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from drift import advect, wind_era5, wind_ncep  # noqa: E402

CASES = {
    "ow-0001": {
        "detection_time": datetime(2019, 1, 1, 3, 42, 35, tzinfo=timezone.utc),
        "obj_lon_range": (33.0545345548213, 33.0610206397016),
        "obj_lat_range": (33.2547758037478, 33.2637343446692),
        # Real, confirmed via a direct reverse-geocoding lookup (OpenStreetMap
        # Nominatim, zoom 5 and 10 both return "Unable to geocode" -- this point
        # is genuine open water, not attributable to a single country's
        # nearshore area) -- not guessed. Sea name matches what this project's
        # own scripts/run_drift_skeleton.py docstring already documented
        # ("Eastern Mediterranean, between Cyprus and Egypt").
        "geo_context": "Levantine Sea, Eastern Mediterranean — open water between Cyprus and the Nile Delta coast",
    },
    "ow-0002": {
        "detection_time": datetime(2019, 1, 4, 15, 56, 38, tzinfo=timezone.utc),
        "obj_lon_range": (32.0268313998942, 32.0310868011572),
        "obj_lat_range": (31.6795634053252, 31.6913258699428),
        # Real, confirmed via direct reverse-geocoding lookup (Nominatim
        # returns "Dumyat, Egypt" / country_code "eg" for this exact point).
        "geo_context": "Eastern Mediterranean, off Damietta (Dumyat), Egypt",
    },
}

CASE_IDS_TO_RUN = ["ow-0001", "ow-0002"]  # every case here gets (re-)run

HOURS_BACK = 24
HOURS_FORWARD = 24  # same window length as HOURS_BACK -- symmetric hindcast/forecast horizon,
                     # not a physical constraint (see DECISIONS.md "Forward drift forecasting added")
WIND_PADDING_DEG = 2.0
N_PARTICLES = 50

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "dashboard"


def wind_snapshot(wind_ds) -> list[dict]:
    """
    Real (lat, lon, u10, v10) grid points at the time step closest to
    detection (the last one in the fetched window), for the map's
    wind-vector overlay -- an actual snapshot of the field that drove the
    advection, not a decorative arrow grid. ERA5's grid is 16x16 within
    the padded bbox; that's plotted as-is (dense but real).
    """
    last = wind_ds.isel(time=-1)
    lats, lons = last["lat"].values, last["lon"].values
    u, v = last["u10"].values, last["v10"].values
    points = []
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            uu, vv = float(u[i, j]), float(v[i, j])
            if uu == uu and vv == vv:  # skip NaN (can occur near a fetch window edge)
                points.append({"lat": float(lat), "lon": float(lon), "u": uu, "v": vv})
    return points


def run_source(name: str, fetch_wind_window, detection_time, obj_lon_range, obj_lat_range) -> dict:
    print(f"running {name}...")
    lon_range = (obj_lon_range[0] - WIND_PADDING_DEG, obj_lon_range[1] + WIND_PADDING_DEG)
    lat_range = (obj_lat_range[0] - WIND_PADDING_DEG, obj_lat_range[1] + WIND_PADDING_DEG)

    backward_start = detection_time - timedelta(hours=HOURS_BACK)
    wind_ds_back = fetch_wind_window(backward_start, detection_time, lat_range, lon_range)

    lons0, lats0 = advect.seed_particles_in_bbox(*obj_lon_range, *obj_lat_range, n_particles=N_PARTICLES)
    traj = advect.backward_advect(lons0, lats0, detection_time, HOURS_BACK, wind_ds_back, dt_hours=1.0)
    region = advect.origin_region(traj)

    # Forward forecast -- same seed particles (the detection bbox), same
    # physics, opposite time direction (see DECISIONS.md "Forward drift
    # forecasting added"). Needs its own wind fetch: [detection_time,
    # detection_time + HOURS_FORWARD] instead of the backward window.
    # Reuses the same padded lat/lon box as the backward fetch -- at
    # typical drift speeds over a 24h window the forecast track stays
    # well within it, a documented simplification, not a hard limit.
    forward_end = detection_time + timedelta(hours=HOURS_FORWARD)
    wind_ds_fwd = fetch_wind_window(detection_time, forward_end, lat_range, lon_range)
    fwd_traj = advect.forward_advect(lons0, lats0, detection_time, HOURS_FORWARD, wind_ds_fwd, dt_hours=1.0)
    forecast = advect.origin_region(fwd_traj)

    return {
        "name": name,
        "centroid": [region["centroid_lon"], region["centroid_lat"]],
        "lon_std": region["lon_std"],
        "lat_std": region["lat_std"],
        # full backward tracks for every particle, [ [lon,lat], ... ] per particle, detection -> origin
        "tracks": [
            list(zip(traj.lons[p, :].tolist(), traj.lats[p, :].tolist()))
            for p in range(traj.lons.shape[0])
        ],
        "final_positions": list(zip(traj.lons[:, -1].tolist(), traj.lats[:, -1].tolist())),
        "wind_snapshot": wind_snapshot(wind_ds_back),
        # Forward forecast -- same shape/fields as the backward ones above, prefixed
        # "forecast_" so the map layer (build_map.py) can draw it as a distinguishable
        # second track type rather than mixing it into the backward trace.
        "forecast_centroid": [forecast["centroid_lon"], forecast["centroid_lat"]],
        "forecast_lon_std": forecast["lon_std"],
        "forecast_lat_std": forecast["lat_std"],
        "forecast_tracks": [
            list(zip(fwd_traj.lons[p, :].tolist(), fwd_traj.lats[p, :].tolist()))
            for p in range(fwd_traj.lons.shape[0])
        ],
        "forecast_final_positions": list(zip(fwd_traj.lons[:, -1].tolist(), fwd_traj.lats[:, -1].tolist())),
    }


def run_case(case_id: str) -> dict:
    case = CASES[case_id]
    detection_time = case["detection_time"]
    obj_lon_range, obj_lat_range = case["obj_lon_range"], case["obj_lat_range"]
    detection_lon = sum(obj_lon_range) / 2
    detection_lat = sum(obj_lat_range) / 2

    result = {
        "case": case_id,
        "detection_point": [detection_lon, detection_lat],
        # Real labeled-object bbox (PANGAEA metadata) -- the actual detected-slick
        # footprint used to seed advection particles, not just its centroid. Lets
        # the map draw a real slick outline instead of a single point.
        "detection_bbox": {"lon_range": list(obj_lon_range), "lat_range": list(obj_lat_range)},
        "detection_time_utc": detection_time.isoformat(),
        "hours_back": HOURS_BACK,
        "hours_forward": HOURS_FORWARD,
        "n_particles": N_PARTICLES,
        "geo_context": case.get("geo_context", ""),
        "sources": [],
    }

    for name, fetch_fn in [("ERA5", wind_era5.fetch_wind_window), ("NCEP/NCAR", wind_ncep.fetch_wind_window)]:
        try:
            result["sources"].append(run_source(name, fetch_fn, detection_time, obj_lon_range, obj_lat_range))
        except Exception as e:
            # A source being temporarily unreachable (e.g. NOAA PSL's own server
            # being down, observed 2026-08-25 -- unrelated to our network/bandwidth,
            # confirmed via a direct curl test) shouldn't block the whole export --
            # ERA5 is the primary source anyway (see DECISIONS.md); NCEP/NCAR is the
            # documented fallback, not required for the map to be useful.
            print(f"  WARNING: {name} failed ({type(e).__name__}: {e}) -- skipping, "
                  f"map will be built without it.")

    return result


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for case_id in CASE_IDS_TO_RUN:
        print(f"\n=== {case_id} ===")
        result = run_case(case_id)
        out_path = OUT_DIR / f"drift_{case_id.replace('-', '')}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
