"""
Ocean surface currents from HYCOM GLBy0.08 reanalysis -- open access, no
account needed, via tds.hycom.org THREDDS/OPeNDAP (data is marked
"Approved for public release. Distribution unlimited."). See DECISIONS.md
"Ocean currents: HYCOM GLBy0.08 reanalysis".

Note: this mirror only serves barotropic (depth-averaged) velocity via the
"_sur" files, not true 0m surface velocity -- an approximation adequate for
the drift skeleton, flagged in DECISIONS.md as worth upgrading later.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import xarray as xr

BASE_URL = "https://tds.hycom.org/thredds/dodsC/datasets/GLBy0.08/expt_93.0/data/hindcasts"


def cycle_and_tau(target_time: datetime) -> tuple[datetime, int]:
    """HYCOM hindcasts run one 12Z cycle per day, with hourly tau offsets covering the next ~24h."""
    cycle_date: date = target_time.date() if target_time.hour >= 12 else (target_time - timedelta(days=1)).date()
    cycle_time = datetime(cycle_date.year, cycle_date.month, cycle_date.day, 12, tzinfo=timezone.utc)
    tau = round((target_time - cycle_time).total_seconds() / 3600)
    # Rounding a target_time within ~30s of the next cycle's 12Z can round tau
    # up to 24, one past this cycle's valid range (0-23) -- roll to the next
    # cycle's tau=0 instead of requesting a file that doesn't exist.
    if tau >= 24:
        cycle_time += timedelta(days=1)
        tau -= 24
    return cycle_time, tau


def fetch_current_at(
    target_time: datetime, lat_range: tuple[float, float], lon_range: tuple[float, float]
) -> xr.Dataset:
    """
    Returns a single-timestep xr.Dataset with u, v (m/s, barotropic velocity)
    for ocean current at target_time, subset to the given lat/lon box.
    lat_range/lon_range are (min, max) in degrees, lon in -180..180.
    """
    cycle_time, tau = cycle_and_tau(target_time)
    url = (
        f"{BASE_URL}/{cycle_time.year}/"
        f"hycom_GLBy0.08_930_{cycle_time.strftime('%Y%m%d%H')}_t{tau:03d}_sur.nc"
    )
    # decode_times=False: this file's "tau" variable has a non-CF-compliant
    # units string ("hours since analysis") that xarray's default time
    # decoder can't parse. We don't need decoded time values here anyway --
    # cycle_time/tau (above) already pin down which file/timestep this is.
    ds = xr.open_dataset(url, decode_times=False)

    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range
    lon_min360, lon_max360 = lon_min % 360, lon_max % 360

    ds = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min360, lon_max360))
    ds = ds.rename({"u_barotropic_velocity": "u", "v_barotropic_velocity": "v"})
    ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby("lon")
    return ds[["u", "v"]].load()
