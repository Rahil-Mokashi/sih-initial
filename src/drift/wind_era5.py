"""
ERA5 10m wind via the Copernicus Climate Data Store (CDS) API. Primary
wind source for the drift model (see DECISIONS.md "Wind: ERA5 via
Copernicus CDS") -- 0.25deg, hourly, much finer than the NCEP/NCAR
fallback in wind_ncep.py.

Requires a free CDS account, ~/.cdsapirc with an API key, and the
dataset's Terms of Use accepted on its page (see DECISIONS.md for exact
steps). Results are cached locally (data/processed/era5_cache/) since each
CDS request queues server-side and can take a while to fulfill.

Implements the same fetch_wind_window(...) interface as wind_ncep.py so
the advection code in advect.py doesn't need to change based on which
source is used.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from pathlib import Path

import cdsapi
import xarray as xr

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "era5_cache"


def _cache_key(start_time: datetime, end_time: datetime, lat_range: tuple, lon_range: tuple) -> str:
    raw = f"{start_time.isoformat()}_{end_time.isoformat()}_{lat_range}_{lon_range}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def fetch_wind_window(
    start_time: datetime, end_time: datetime,
    lat_range: tuple[float, float], lon_range: tuple[float, float],
) -> xr.Dataset:
    """
    Same interface/contract as wind_ncep.fetch_wind_window: returns an
    xr.Dataset with dims (time, lat, lon) and variables u10, v10 (m/s),
    covering [start_time, end_time] and the given lat/lon box.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CACHE_DIR / f"era5_wind_{_cache_key(start_time, end_time, lat_range, lon_range)}.nc"

    if not out_path.exists():
        lat_min, lat_max = lat_range
        lon_min, lon_max = lon_range

        days = set()
        t = start_time
        while t <= end_time:
            days.add(t.date())
            t += timedelta(hours=1)
        days = sorted(days)
        years = sorted({f"{d.year}" for d in days})
        months = sorted({f"{d.month:02d}" for d in days})
        day_strs = sorted({f"{d.day:02d}" for d in days})

        client = cdsapi.Client()
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
                "year": years,
                "month": months,
                "day": day_strs,
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": [lat_max, lon_min, lat_min, lon_max],  # N, W, S, E
                "format": "netcdf",
            },
            str(out_path),
        )

    ds = xr.open_dataset(out_path)
    # The modern CDS-Beta download API uses full CF names (latitude/longitude,
    # valid_time) instead of the legacy short names (lat/lon, time) that
    # wind_ncep.py's dataset uses -- normalize here so advect.py can treat
    # both wind sources identically.
    rename_map = {}
    if "u10" not in ds and "10m_u_component_of_wind" in ds:
        rename_map["10m_u_component_of_wind"] = "u10"
    if "v10" not in ds and "10m_v_component_of_wind" in ds:
        rename_map["10m_v_component_of_wind"] = "v10"
    if "valid_time" in ds.dims and "time" not in ds.dims:
        rename_map["valid_time"] = "time"
    if "latitude" in ds.dims and "lat" not in ds.dims:
        rename_map["latitude"] = "lat"
    if "longitude" in ds.dims and "lon" not in ds.dims:
        rename_map["longitude"] = "lon"
    if rename_map:
        ds = ds.rename(rename_map)

    start_naive = start_time.replace(tzinfo=None)
    end_naive = end_time.replace(tzinfo=None)
    ds = ds.sel(time=slice(start_naive, end_naive))
    return ds[["u10", "v10"]].load()
