"""
Backward particle-advection skeleton: given a detected slick's bounding
box and wind + current fields, seed particles across it and advect them
backward in time to estimate an origin region.

Physics (see DECISIONS.md "Drift model approach" and "Ekman/Coriolis
deflection added"): particle velocity = ocean current + windage-scaled,
Ekman-deflected wind. Windage = 0.03 (the standard "3% of wind speed"
oil-drift rule of thumb). The wind-driven component is additionally
rotated by DEFLECTION_ANGLE_DEG (empirically ~20 deg in most oil-spill
trajectory literature) -- clockwise (to the right) in the Northern
Hemisphere, counter-clockwise (to the left) in the Southern Hemisphere,
per the standard Ekman-layer explanation for why wind-driven surface
drift doesn't point straight downwind. This does NOT touch the ocean
current term (HYCOM's currents already embed whatever real Ekman
transport exists in the modeled ocean state; deflecting them again would
double-count it) -- only the empirical wind-driven correction, which is
the piece a simple "windage %" rule leaves out.

Backward integration still works the same way as before: the deflected
wind is computed as it would act going forward in time, then the whole
(current + deflected wind) forward-time velocity is negated to step the
particle backward -- see backward_advect below.

Integration is explicit Euler in lon/lat space with a small flat-Earth
per-step correction (cos(lat) for longitude spacing); adequate at hourly
steps over a day-scale window at this latitude, not appropriate for
multi-week windows or high latitudes without revisiting.

Forward-integration forecasting (forward_advect) was added per the SIH
problem statement's explicit requirement to "predict the future flow of
the slick" / trace it "backward AND forward" -- see DECISIONS.md
"Forward drift forecasting added". It reuses the exact same physics
(_advect below): same windage/current combination, same Ekman
deflection, same explicit-Euler integration -- the only difference from
backward_advect is the sign of the velocity step and the direction time
moves in, both driven by a single `direction` parameter (+1 forward, -1
backward) rather than a second, duplicated integration loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

from drift import currents as hycom_currents

EARTH_RADIUS_M = 6371000.0
WINDAGE = 0.03  # fraction of wind speed added to the current velocity
DEFLECTION_ANGLE_DEG = 20.0  # empirical Ekman deflection of wind-driven drift from the wind direction


def _deflect_wind(u_wind: np.ndarray, v_wind: np.ndarray, lats: np.ndarray, angle_deg: float = DEFLECTION_ANGLE_DEG) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotates the wind vector by angle_deg: clockwise (to the right) in the
    Northern Hemisphere, counter-clockwise (to the left) in the Southern
    Hemisphere -- per-particle, based on that particle's current latitude
    sign, so this stays correct if a future case straddles the equator.
    """
    sign = np.where(lats >= 0, 1.0, -1.0)  # NH: clockwise (+angle); SH: counter-clockwise (-angle)
    theta = np.radians(angle_deg) * sign
    u_defl = u_wind * np.cos(theta) + v_wind * np.sin(theta)
    v_defl = -u_wind * np.sin(theta) + v_wind * np.cos(theta)
    return u_defl, v_defl


@dataclass
class Trajectories:
    times: list[datetime]  # length n_steps+1, from detection time backward to origin
    lons: np.ndarray       # (n_particles, n_steps+1)
    lats: np.ndarray       # (n_particles, n_steps+1)


def seed_particles_in_bbox(
    lon_min: float, lon_max: float, lat_min: float, lat_max: float,
    n_particles: int = 50, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lons = rng.uniform(lon_min, lon_max, n_particles)
    lats = rng.uniform(lat_min, lat_max, n_particles)
    return lons, lats


def _interp_field(data_array: xr.DataArray, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    result = data_array.interp(
        lat=xr.DataArray(lats, dims="p"), lon=xr.DataArray(lons, dims="p"), method="linear"
    ).values
    return np.nan_to_num(result)


def _interp_wind(wind_ds: xr.Dataset, time: datetime, lons: np.ndarray, lats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # wind_ds sources (NCEP, ERA5) both carry tz-naive time coords (netCDF convention); strip tzinfo to match.
    naive_time = time.replace(tzinfo=None) if time.tzinfo is not None else time
    snap = wind_ds.sel(time=naive_time, method="nearest")
    return _interp_field(snap["u10"], lons, lats), _interp_field(snap["v10"], lons, lats)


def _advect(
    lons0: np.ndarray, lats0: np.ndarray,
    start_time: datetime, hours: int,
    wind_ds: xr.Dataset,
    direction: int,
    dt_hours: float = 1.0,
    current_bbox_padding_deg: float = 1.0,
) -> Trajectories:
    """
    Shared integration core for both backward_advect and forward_advect.
    direction=-1 steps particles backward in time (hindcast); direction=+1
    steps them forward (forecast). Physics (current + windage-scaled,
    Ekman-deflected wind) is identical either way -- only the sign of the
    velocity step and which way `t` moves differ, both driven by
    `direction`, so there is exactly one integration loop to keep correct
    rather than two copies that could drift apart.
    """
    n_steps = int(hours / dt_hours)
    n_particles = len(lons0)

    lons = np.zeros((n_particles, n_steps + 1))
    lats = np.zeros((n_particles, n_steps + 1))
    lons[:, 0], lats[:, 0] = lons0, lats0
    times = [start_time]

    current_cache: dict[tuple, xr.Dataset] = {}
    t = start_time

    for step in range(n_steps):
        cur_lons, cur_lats = lons[:, step], lats[:, step]

        cache_key = hycom_currents.cycle_and_tau(t)
        if cache_key not in current_cache:
            pad = current_bbox_padding_deg
            current_cache[cache_key] = hycom_currents.fetch_current_at(
                t,
                (float(cur_lats.min() - pad), float(cur_lats.max() + pad)),
                (float(cur_lons.min() - pad), float(cur_lons.max() + pad)),
            )
        cur_ds = current_cache[cache_key]
        u_curr = _interp_field(cur_ds["u"], cur_lons, cur_lats)
        v_curr = _interp_field(cur_ds["v"], cur_lons, cur_lats)

        u_wind, v_wind = _interp_wind(wind_ds, t, cur_lons, cur_lats)
        u_wind, v_wind = _deflect_wind(u_wind, v_wind, cur_lats)

        u_total = u_curr + WINDAGE * u_wind
        v_total = v_curr + WINDAGE * v_wind

        dt_sec = dt_hours * 3600
        dlat = direction * v_total * dt_sec / EARTH_RADIUS_M * (180 / np.pi)
        dlon = direction * u_total * dt_sec / (EARTH_RADIUS_M * np.cos(np.radians(cur_lats))) * (180 / np.pi)

        lats[:, step + 1] = cur_lats + dlat
        lons[:, step + 1] = cur_lons + dlon

        t = t + direction * timedelta(hours=dt_hours)
        times.append(t)

    return Trajectories(times=times, lons=lons, lats=lats)


def backward_advect(
    lons0: np.ndarray, lats0: np.ndarray,
    detection_time: datetime, hours_back: int,
    wind_ds: xr.Dataset,
    dt_hours: float = 1.0,
    current_bbox_padding_deg: float = 1.0,
) -> Trajectories:
    return _advect(lons0, lats0, detection_time, hours_back, wind_ds, direction=-1,
                    dt_hours=dt_hours, current_bbox_padding_deg=current_bbox_padding_deg)


def forward_advect(
    lons0: np.ndarray, lats0: np.ndarray,
    detection_time: datetime, hours_forward: int,
    wind_ds: xr.Dataset,
    dt_hours: float = 1.0,
    current_bbox_padding_deg: float = 1.0,
) -> Trajectories:
    """
    Forward-integration forecast of where the slick is headed, from the
    detection point forward in time -- same physics as backward_advect
    (see _advect), opposite time direction. `wind_ds` must cover
    [detection_time, detection_time + hours_forward], not the backward
    window -- the caller fetches that, same as backward_advect's caller
    fetches [detection_time - hours_back, detection_time].
    """
    return _advect(lons0, lats0, detection_time, hours_forward, wind_ds, direction=+1,
                    dt_hours=dt_hours, current_bbox_padding_deg=current_bbox_padding_deg)


def origin_region(traj: Trajectories) -> dict:
    final_lons = traj.lons[:, -1]
    final_lats = traj.lats[:, -1]
    return {
        "centroid_lon": float(np.mean(final_lons)),
        "centroid_lat": float(np.mean(final_lats)),
        "lon_std": float(np.std(final_lons)),
        "lat_std": float(np.std(final_lats)),
        "lon_range": (float(final_lons.min()), float(final_lons.max())),
        "lat_range": (float(final_lats.min()), float(final_lats.max())),
    }
