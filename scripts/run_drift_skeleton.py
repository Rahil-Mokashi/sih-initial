"""
Drift/hindcast skeleton: backward particle advection for case ow-0001,
comparing wind sources (NCEP/NCAR vs. ERA5) feeding the exact same
advection code -- only fetch_wind_window() changes, per DECISIONS.md.

Case ow-0001 (PANGAEA dataset, Step 0/1): real oil detection at
33.06E, 33.26N, Eastern Mediterranean between Cyprus and Egypt,
2019-01-01T03:42:35 UTC, Sentinel-1B.
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from drift import advect, wind_ncep  # noqa: E402

try:
    from drift import wind_era5
    ERA5_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover - depends on local cdsapi install
    wind_era5 = None
    ERA5_IMPORT_ERROR = e

DETECTION_TIME = datetime(2019, 1, 1, 3, 42, 35, tzinfo=timezone.utc)
# ow-0001's labeled oil object bbox corners (see DECISIONS.md / data_matrix.tab)
OBJ_LON_RANGE = (33.0545345548213, 33.0610206397016)
OBJ_LAT_RANGE = (33.2547758037478, 33.2637343446692)

HOURS_BACK = 24
WIND_PADDING_DEG = 2.0
N_PARTICLES = 50


def run_with_wind_source(name: str, fetch_wind_window) -> dict:
    print(f"\n=== backward advection using {name} wind ===")
    start_time = DETECTION_TIME - timedelta(hours=HOURS_BACK)
    lon_range = (OBJ_LON_RANGE[0] - WIND_PADDING_DEG, OBJ_LON_RANGE[1] + WIND_PADDING_DEG)
    lat_range = (OBJ_LAT_RANGE[0] - WIND_PADDING_DEG, OBJ_LAT_RANGE[1] + WIND_PADDING_DEG)

    wind_ds = fetch_wind_window(start_time, DETECTION_TIME, lat_range, lon_range)
    print(f"wind data: {dict(wind_ds.sizes)}")

    lons0, lats0 = advect.seed_particles_in_bbox(*OBJ_LON_RANGE, *OBJ_LAT_RANGE, n_particles=N_PARTICLES)
    traj = advect.backward_advect(lons0, lats0, DETECTION_TIME, HOURS_BACK, wind_ds, dt_hours=1.0)
    region = advect.origin_region(traj)
    print(
        f"estimated origin region ({HOURS_BACK}h back): "
        f"centroid=({region['centroid_lon']:.4f}, {region['centroid_lat']:.4f}), "
        f"lon_std={region['lon_std']:.4f}, lat_std={region['lat_std']:.4f}"
    )
    return region


def main() -> None:
    results = {}

    try:
        results["NCEP/NCAR"] = run_with_wind_source("NCEP/NCAR", wind_ncep.fetch_wind_window)
    except Exception as e:
        print(f"NCEP/NCAR run FAILED: {type(e).__name__}: {e}")

    if wind_era5 is None:
        print(f"\nERA5 run SKIPPED (import failed: {ERA5_IMPORT_ERROR})")
    else:
        try:
            results["ERA5"] = run_with_wind_source("ERA5", wind_era5.fetch_wind_window)
        except Exception as e:
            print(f"\nERA5 run FAILED: {type(e).__name__}: {e}")

    if "NCEP/NCAR" in results and "ERA5" in results:
        import math
        a, b = results["NCEP/NCAR"], results["ERA5"]
        dist_km = math.hypot(
            (a["centroid_lon"] - b["centroid_lon"]) * 111.320 * math.cos(math.radians(a["centroid_lat"])),
            (a["centroid_lat"] - b["centroid_lat"]) * 111.320,
        )
        print("\n=== comparison ===")
        print(f"NCEP/NCAR centroid: ({a['centroid_lon']:.4f}, {a['centroid_lat']:.4f})")
        print(f"ERA5      centroid: ({b['centroid_lon']:.4f}, {b['centroid_lat']:.4f})")
        print(f"centroid distance: {dist_km:.2f} km")


if __name__ == "__main__":
    main()
