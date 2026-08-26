"""
NCEP/NCAR Reanalysis 10m wind -- open access, no account, via NOAA PSL's
OPeNDAP server. Coarser than ERA5 (~1.875deg grid, 6-hourly vs. ERA5's
0.25deg/hourly) but real data with zero registration friction. Documented
fallback wind source if ERA5/CDS access ever breaks -- see DECISIONS.md.

Implements the same fetch_wind_window(...) interface as wind_era5.py so
the advection code in advect.py doesn't need to change based on which
source is used.
"""

from __future__ import annotations

from datetime import datetime

import xarray as xr

BASE_URL = "https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/surface_gauss"


def fetch_wind_window(
    start_time: datetime, end_time: datetime,
    lat_range: tuple[float, float], lon_range: tuple[float, float],
) -> xr.Dataset:
    """
    Returns an xr.Dataset with dims (time, lat, lon) and variables u10, v10
    (m/s), covering [start_time, end_time] and the given lat/lon box.
    lat_range/lon_range are (min, max) in degrees, lon in -180..180.
    """
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range
    lon_min360, lon_max360 = lon_min % 360, lon_max % 360

    # NCEP's time coord is tz-naive (values are UTC); strip tzinfo to compare.
    start_naive = start_time.replace(tzinfo=None)
    end_naive = end_time.replace(tzinfo=None)

    years = sorted({start_time.year, end_time.year})
    parts = []
    for year in years:
        u = xr.open_dataset(f"{BASE_URL}/uwnd.10m.gauss.{year}.nc")
        v = xr.open_dataset(f"{BASE_URL}/vwnd.10m.gauss.{year}.nc")
        merged = xr.merge([u, v])
        # Subset + load EACH year's remote OPeNDAP dataset to real in-memory
        # values before concatenating across years -- concatenating the lazy,
        # still-remote datasets first (subsetting only after) silently
        # produces all-zero data, a known pitfall with cross-file OPeNDAP concat.
        subset = merged.sel(
            time=slice(start_naive, end_naive),
            lat=slice(lat_max, lat_min),  # NCEP lat runs 90 -> -90 (descending)
            lon=slice(lon_min360, lon_max360),  # NCEP lon is 0-360
        ).load()
        if subset.sizes.get("time", 0) > 0:
            parts.append(subset)

    ds = xr.concat(parts, dim="time") if len(parts) > 1 else parts[0]
    ds = ds.rename({"uwnd": "u10", "vwnd": "v10"})
    ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby("lon").sortby("lat")
    return ds[["u10", "v10"]]
