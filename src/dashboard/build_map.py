"""
Builds the real drift-result map as a standalone Leaflet map (via folium)
-- real basemap, real coordinates, real particle tracks, real candidate
vessel positions. Reads data/processed/dashboard/drift_{case}.json
(scripts/export_drift_dashboard_data.py) and vessel_ranking_{case}.json
(src/attribution/score_vessels.py), writes src/dashboard/output/map.html
(first case) / map_{case}.html (the rest).

Add --anonymize to write to output_anon/ instead, with real ship_name/
mmsi/imo replaced by fictional stand-ins in tooltips/popups -- see
src/common/anonymize.py for why. Must match build_dashboard.py's
--anonymize output directory so index.html's map iframe resolves.

Dark navy/amber/teal theme matching the dashboard shell
(build_dashboard.py) -- see DECISIONS.md "Dashboard visual redesign" for
where this design language came from. Colors:
  - amber (#f4b95a): the drift trace itself (particle tracks, origin
    marker) and the #1-ranked "top suspect" vessel -- the two things the
    whole pipeline is pointing at.
  - steel blue (#5b8fb0): NCEP/NCAR fallback wind source, kept visually
    distinct from amber's ERA5-primary meaning.
  - teal (#4ed4a6): candidate vessels ranked below #1 -- present as
    evidence, not (yet) the leading suspect.
  - alert red (#d9534f): the actual detection point.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import folium
import numpy as np
from folium.plugins import Fullscreen
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.anonymize import anonymize_ranking  # noqa: E402
from common.geo import haversine_km  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "dashboard"
OUT_DIR = Path(__file__).resolve().parent / "output"

CASE_IDS = ["ow-0001", "ow-0002"]
N_VESSEL_MARKERS = 10  # how many top-ranked candidates to plot on the map itself

SOURCE_COLORS = {"ERA5": "#f4b95a", "NCEP/NCAR": "#5b8fb0"}
DETECTION_COLOR = "#d9534f"
TOP_SUSPECT_COLOR = "#f4b95a"
CANDIDATE_COLOR = "#4ed4a6"

# Restyles Leaflet's own chrome (popups, layer control, zoom buttons) to
# match the dashboard's dark navy/amber theme, since folium's defaults are
# a plain white Leaflet look that would otherwise clash inside the iframe.
DARK_CHROME_CSS = """
<style>
  .leaflet-popup-content-wrapper, .leaflet-popup-tip {
    background: #0e1d2c; color: #eaf1f7; border: 1px solid #2c4d68;
  }
  .leaflet-popup-content { font-family: 'IBM Plex Sans', system-ui, sans-serif; font-size: 12.5px; }
  .leaflet-popup-content b { color: #f4b95a; font-family: 'Chivo', system-ui, sans-serif; }
  .leaflet-tooltip {
    background: #0e1d2c; color: #eaf1f7; border: 1px solid #2c4d68;
    font-family: 'IBM Plex Sans', system-ui, sans-serif; font-size: 12px;
    box-shadow: 0 6px 16px -6px rgba(0,0,0,.7);
  }
  .leaflet-tooltip-top:before, .leaflet-tooltip-bottom:before,
  .leaflet-tooltip-left:before, .leaflet-tooltip-right:before { border-top-color: #2c4d68; }
  .leaflet-control-layers, .leaflet-bar {
    background: #0e1d2c !important; border: 1px solid #2c4d68 !important; color: #eaf1f7;
  }
  .leaflet-control-layers-toggle, .leaflet-bar a {
    background-color: #0e1d2c !important; color: #eaf1f7 !important;
  }
  .leaflet-bar a { border-bottom: 1px solid #2c4d68 !important; }
  .leaflet-control-layers-expanded { color: #eaf1f7; }
  .leaflet-control-scale-line { background: rgba(14,29,44,.85) !important; color: #eaf1f7 !important; border-color: #2c4d68 !important; }
</style>
"""


def render_drift_animation(map_var: str, source: dict, hours_back: int) -> str:
    """
    Real backward-advection playback: steps through the actual per-particle
    (lon, lat) positions in source["tracks"] (one point per simulated hour,
    index 0 = detection time, index hours_back = the origin estimate --
    see src/drift/advect.py's Trajectories.lons/lats shape) rather than
    just showing the static start/end markers build_map() already draws.
    Waits for the folium-generated map object to exist on `window` before
    attaching anything, since script placement relative to folium's own
    map-init script isn't guaranteed by insertion order alone.
    """
    tracks = source["tracks"]  # [particle][step] = [lon, lat]
    n_steps = len(tracks[0]) if tracks else 0
    tracks_json = json.dumps(tracks)
    return f"""
<style>
  .drift-anim-ctrl {{
    position:absolute; left:12px; bottom:12px; z-index:1000;
    background:rgba(10,18,28,.92); border:1px solid #2c4d68; border-radius:10px;
    padding:9px 12px; display:flex; align-items:center; gap:10px;
    font-family:'IBM Plex Mono',ui-monospace,monospace; font-size:11.5px; color:#eaf1f7;
    box-shadow:0 10px 24px -10px rgba(0,0,0,.7); backdrop-filter:blur(4px);
  }}
  .drift-anim-ctrl button {{
    background:#e6a23c; color:#1c1204; border:none; border-radius:6px;
    width:26px; height:26px; cursor:pointer; font-size:12px; flex:none;
  }}
  .drift-anim-ctrl input[type=range] {{ width:150px; accent-color:#e6a23c; }}
  .drift-anim-ctrl .hrs {{ min-width:52px; text-align:right; color:#f4b95a; }}
</style>
<div class="drift-anim-ctrl" id="anim-{map_var}">
  <button id="play-{map_var}" title="Play/pause backward-drift animation">&#9654;</button>
  <input type="range" id="scrub-{map_var}" min="0" max="{n_steps - 1}" value="0" step="1">
  <span class="hrs" id="hrs-{map_var}">0h back</span>
</div>
<script>
(function() {{
  var tracks = {tracks_json};
  var hoursBack = {hours_back};
  var nSteps = {n_steps};
  var mapVarName = "{map_var}";

  function init() {{
    var map = window[mapVarName];
    if (!map) {{ setTimeout(init, 50); return; }}

    var markers = tracks.map(function(track) {{
      return L.circleMarker([track[0][1], track[0][0]], {{
        radius: 3.5, color: "#f4b95a", weight: 0, fillColor: "#f4b95a", fillOpacity: 0.95,
      }}).addTo(map);
    }});

    var scrub = document.getElementById("scrub-" + mapVarName);
    var hrsLabel = document.getElementById("hrs-" + mapVarName);
    var playBtn = document.getElementById("play-" + mapVarName);
    var playing = false, timer = null;

    function setStep(step) {{
      step = Math.max(0, Math.min(nSteps - 1, step));
      markers.forEach(function(mk, i) {{
        var pt = tracks[i][step];
        if (pt) mk.setLatLng([pt[1], pt[0]]);
      }});
      var hrsBackNow = Math.round(step / (nSteps - 1) * hoursBack);
      hrsLabel.textContent = hrsBackNow + "h back";
      scrub.value = step;
    }}

    scrub.addEventListener("input", function() {{
      playing = false; playBtn.innerHTML = "&#9654;"; clearInterval(timer);
      setStep(parseInt(scrub.value, 10));
    }});

    playBtn.addEventListener("click", function() {{
      playing = !playing;
      playBtn.innerHTML = playing ? "&#10074;&#10074;" : "&#9654;";
      if (playing) {{
        if (parseInt(scrub.value, 10) >= nSteps - 1) setStep(0);
        timer = setInterval(function() {{
          var next = parseInt(scrub.value, 10) + 1;
          if (next >= nSteps) {{ playing = false; playBtn.innerHTML = "&#9654;"; clearInterval(timer); return; }}
          setStep(next);
        }}, 220);
      }} else {{
        clearInterval(timer);
      }}
    }});

    setStep(0);
  }}
  init();
}})();
</script>
"""


def render_wind_vectors(source: dict) -> folium.FeatureGroup:
    """
    Real ERA5/NCEP wind vectors (u10, v10 in m/s) at the grid points
    actually used to drive the advection -- see
    scripts/export_drift_dashboard_data.py's wind_snapshot(), the time
    step closest to detection. Each arrow's length is proportional to
    real wind speed (capped so a strong outlier doesn't blow out the
    scale for the rest); direction is drawn FROM the grid point (meteorological
    "blowing toward" convention, matching how the advection code itself
    uses u/v).
    """
    color = SOURCE_COLORS.get(source["name"], "#93a8bc")
    group = folium.FeatureGroup(name=f"{source['name']}: wind field ({len(source.get('wind_snapshot', []))} pts)", show=False)

    points = source.get("wind_snapshot", [])
    if not points:
        return group
    max_speed = max((p["u"] ** 2 + p["v"] ** 2) ** 0.5 for p in points) or 1.0
    max_len_deg = 0.18  # longest arrow, in degrees, at max_speed within this snapshot

    for p in points:
        lat, lon, u, v = p["lat"], p["lon"], p["u"], p["v"]
        speed = (u ** 2 + v ** 2) ** 0.5
        scale = (speed / max_speed) * max_len_deg if max_speed > 0 else 0
        # u = eastward (dLon), v = northward (dLat); flat-earth approx is fine at this zoom/extent
        end_lat = lat + (v / speed) * scale if speed > 0 else lat
        end_lon = lon + (u / speed) * scale if speed > 0 else lon
        folium.PolyLine(
            [[lat, lon], [end_lat, end_lon]], color=color, weight=1.4, opacity=0.55,
        ).add_to(group)
        folium.CircleMarker(
            [end_lat, end_lon], radius=1.3, color=color, fill=True, fill_opacity=0.8, weight=0,
            tooltip=f"{speed:.1f} m/s",
        ).add_to(group)
    return group


def drift_cone_latlon(tracks: list) -> list[list[float]] | None:
    """
    Real "drift reconstruction cone" outline: the convex hull of every real
    particle position at every real simulated timestep (not just the
    endpoints), giving a single clean shape that honestly bounds the whole
    swept advection area instead of drawing 50 individual thin lines --
    simplifies the default view without hiding or approximating the
    underlying physics (the raw per-particle tracks stay available as a
    togglable layer, see build_map() below). Returns [[lat,lon], ...] (a
    closed ring, first point repeated at the end) or None if there aren't
    enough distinct points for a real hull (needs >=3).
    """
    points = np.array([[lon, lat] for track in tracks for lon, lat in track])
    if len(points) < 3:
        return None
    try:
        hull = ConvexHull(points)
    except Exception:
        return None  # degenerate (e.g. all points collinear) -- real edge case, not an error
    ring = points[hull.vertices]
    ring = np.vstack([ring, ring[0]])  # close the ring
    return [[lat, lon] for lon, lat in ring]


def ship_icon(color: str) -> folium.Icon:
    """Real vessel positions get a ship glyph, not a generic dot -- see the
    user-supplied reference mockup this map's redesign is matching."""
    return folium.Icon(color=color, icon="ship", prefix="fa")


def fmt_origin_time(detection_time_utc: str, hours_back: int) -> str:
    """Real origin time = detection time - hours_back, formatted for the
    map's persistent origin callout (e.g. '01 Jan 2019 · 03:42 UTC')."""
    dt = datetime.fromisoformat(detection_time_utc.replace("Z", "+00:00")) - timedelta(hours=hours_back)
    return dt.strftime("%d %b %Y · %H:%M UTC")


def build_map(data: dict, ranking: dict | None) -> folium.Map:
    detection = data["detection_point"]
    m = folium.Map(location=[detection[1], detection[0]], zoom_start=11, tiles=None, control_scale=True)
    m.get_root().html.add_child(folium.Element(DARK_CHROME_CSS))

    folium.TileLayer("CartoDB dark_matter", name="Dark basemap").add_to(m)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite",
    ).add_to(m)

    # Bounds are computed explicitly from case content (detection, tracks,
    # vessels) rather than folium's own m.get_bounds() -- that walks the
    # whole object tree regardless of a layer's show=True/False, so the
    # wide 256-point ERA5 wind grid (added below) would otherwise force
    # the initial view to zoom out to fit it even though it starts hidden.
    bounds_points: list[list[float]] = []

    geo_context = data.get("geo_context", "")
    folium.Marker(
        [detection[1], detection[0]],
        tooltip=f"Oil detection: {data['case']}",
        popup=folium.Popup(
            f"<b>Detection ({data['case']})</b><br>"
            f"{detection[1]:.4f}°N, {detection[0]:.4f}°E<br>"
            f"{data['detection_time_utc']}<br>"
            f"Sentinel-1{' · ' + geo_context if geo_context else ''}",
            max_width=280,
        ),
        icon=folium.Icon(color="red", icon="tint", prefix="fa"),
    ).add_to(m)
    bounds_points.append([detection[1], detection[0]])

    # Real detected-slick footprint -- the actual labeled-object bbox (PANGAEA
    # metadata) used to seed advection particles, not a fabricated shape. Tiny
    # (~1km) at this zoom, which is honest: it's a real, small, real-world slick
    # extent, not exaggerated for visual effect.
    bbox = data.get("detection_bbox")
    if bbox:
        lon_min, lon_max = bbox["lon_range"]
        lat_min, lat_max = bbox["lat_range"]
        folium.Rectangle(
            [[lat_min, lon_min], [lat_max, lon_max]],
            color=DETECTION_COLOR, weight=2, fill=True, fill_color=DETECTION_COLOR, fill_opacity=0.35,
            tooltip="Detected slick (real labeled extent)",
        ).add_to(m)
        bounds_points += [[lat_min, lon_min], [lat_max, lon_max]]

    if geo_context:
        m.get_root().html.add_child(folium.Element(f"""
<div style="position:absolute;top:12px;left:12px;z-index:1000;background:rgba(10,18,28,.92);
            border:1px solid #2c4d68;border-radius:8px;padding:8px 12px;max-width:260px;
            font-family:'IBM Plex Sans',system-ui,sans-serif;font-size:12px;color:#eaf1f7;
            box-shadow:0 10px 24px -10px rgba(0,0,0,.7);backdrop-filter:blur(4px);">
  <div style="font-family:'IBM Plex Mono',monospace;font-size:9.5px;color:#93a8bc;
              text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px;">Location</div>
  {geo_context}
</div>"""))

    centroids = {}
    for source in data["sources"]:
        name = source["name"]
        color = SOURCE_COLORS[name]
        centroids[name] = source["centroid"]

        # Default view: one clean "drift reconstruction cone" (real convex hull
        # of every particle position at every timestep -- see
        # drift_cone_latlon()) instead of 50 raw lines. The raw per-particle
        # tracks are still real and still here, just off by default -- a
        # togglable "inspect real tracks" layer for anyone who wants the detail,
        # per the user's request for a simpler default view.
        cone_ring = drift_cone_latlon(source["tracks"])
        if cone_ring:
            cone_group = folium.FeatureGroup(name=f"{name}: drift reconstruction cone", show=(name == "ERA5"))
            folium.Polygon(
                cone_ring, color=color, weight=1.5, opacity=0.7,
                fill=True, fill_color=color, fill_opacity=0.18,
                tooltip=f"{name}: real backward-drift spread envelope ({data['n_particles']} particles, {data['hours_back']}h)",
            ).add_to(cone_group)
            cone_group.add_to(m)
            bounds_points += cone_ring

        tracks_group = folium.FeatureGroup(name=f"{name}: raw particle tracks ({data['n_particles']})", show=False)
        for track in source["tracks"]:
            latlon_track = [[lat, lon] for lon, lat in track]
            folium.PolyLine(latlon_track, color=color, weight=1, opacity=0.4).add_to(tracks_group)
        tracks_group.add_to(m)

        points_group = folium.FeatureGroup(name=f"{name}: final particle positions", show=(name == "ERA5"))
        for lon, lat in source["final_positions"]:
            folium.CircleMarker([lat, lon], radius=2.5, color=color, fill=True, fill_opacity=0.8, weight=0).add_to(points_group)
            bounds_points.append([lat, lon])
        points_group.add_to(m)

        # Forward-forecast track -- same physics/color as the backward hindcast
        # above (src/drift/advect.py forward_advect), drawn dashed rather than
        # solid so the two are visually distinguishable at a glance: solid =
        # reconstructed past (hindcast), dashed = predicted future (forecast).
        # Per the SIH problem statement's "predict the future flow of the
        # slick" -- see DECISIONS.md "Forward drift forecasting added".
        if source.get("forecast_tracks"):
            forecast_cone_ring = drift_cone_latlon(source["forecast_tracks"])
            if forecast_cone_ring:
                forecast_cone_group = folium.FeatureGroup(name=f"{name}: forecast cone (forward)", show=(name == "ERA5"))
                folium.Polygon(
                    forecast_cone_ring, color=color, weight=1.5, opacity=0.6, dash_array="4,6",
                    fill=True, fill_color=color, fill_opacity=0.12,
                    tooltip=f"{name}: real forward-forecast spread envelope ({data['n_particles']} particles, "
                            f"{data.get('hours_forward', '?')}h)",
                ).add_to(forecast_cone_group)
                forecast_cone_group.add_to(m)
                bounds_points += forecast_cone_ring

            forecast_group = folium.FeatureGroup(
                name=f"{name}: raw forecast tracks (forward, {data.get('hours_forward', '?')}h)", show=False
            )
            for track in source["forecast_tracks"]:
                latlon_track = [[lat, lon] for lon, lat in track]
                folium.PolyLine(latlon_track, color=color, weight=1.6, opacity=0.55, dash_array="4,6").add_to(forecast_group)
            forecast_group.add_to(m)

            forecast_points_group = folium.FeatureGroup(name=f"{name}: forecast final positions", show=(name == "ERA5"))
            for lon, lat in source["forecast_final_positions"]:
                folium.CircleMarker(
                    [lat, lon], radius=2.5, color=color, fill=True, fill_opacity=0.35, weight=1.2, dash_array="2,3"
                ).add_to(forecast_points_group)
                bounds_points.append([lat, lon])
            forecast_points_group.add_to(m)

            folium.Marker(
                [source["forecast_centroid"][1], source["forecast_centroid"][0]],
                tooltip=f"{name} forecast (forward {data.get('hours_forward', '?')}h)",
                popup=folium.Popup(
                    f"<b>{name} forward forecast</b><br>"
                    f"{source['forecast_centroid'][1]:.4f}°N, {source['forecast_centroid'][0]:.4f}°E<br>"
                    f"{data.get('hours_forward', '?')}h forward, {data['n_particles']} particles<br>"
                    f"spread: σlon={source['forecast_lon_std']:.4f}°, σlat={source['forecast_lat_std']:.4f}°",
                    max_width=260,
                ),
                icon=folium.Icon(color="orange" if name == "ERA5" else "blue", icon="arrow-right", prefix="fa"),
            ).add_to(m)
            bounds_points.append([source["forecast_centroid"][1], source["forecast_centroid"][0]])

        render_wind_vectors(source).add_to(m)

        origin_time_str = fmt_origin_time(data["detection_time_utc"], data["hours_back"])
        # ERA5 (primary source) gets a permanent, always-visible callout -- matching
        # the user-supplied reference mockup's persistent "Estimated Origin" box --
        # instead of requiring a click. NCEP/NCAR (secondary/toggle source) keeps a
        # click-to-open popup only, so two permanent boxes don't overlap by default.
        folium.Marker(
            [source["centroid"][1], source["centroid"][0]],
            tooltip=(
                folium.Tooltip(
                    f"<b>Estimated Origin</b><br>{origin_time_str}<br>"
                    f"{source['centroid'][1]:.4f}°N, {source['centroid'][0]:.4f}°E",
                    permanent=True, direction="top", sticky=False,
                ) if name == "ERA5" else f"{name} origin estimate"
            ),
            popup=folium.Popup(
                f"<b>{name} origin estimate</b><br>"
                f"{source['centroid'][1]:.4f}°N, {source['centroid'][0]:.4f}°E<br>"
                f"{origin_time_str}<br>"
                f"{data['hours_back']}h backward, {data['n_particles']} particles<br>"
                f"spread: σlon={source['lon_std']:.4f}°, σlat={source['lat_std']:.4f}°",
                max_width=260,
            ),
            icon=folium.Icon(color="orange" if name == "ERA5" else "blue", icon="circle", prefix="fa"),
        ).add_to(m)
        bounds_points.append([source["centroid"][1], source["centroid"][0]])

    if "ERA5" in centroids and "NCEP/NCAR" in centroids:
        dist_km = haversine_km(centroids["ERA5"], centroids["NCEP/NCAR"])
        mid = [(centroids["ERA5"][1] + centroids["NCEP/NCAR"][1]) / 2, (centroids["ERA5"][0] + centroids["NCEP/NCAR"][0]) / 2]
        folium.PolyLine(
            [[centroids["ERA5"][1], centroids["ERA5"][0]], [centroids["NCEP/NCAR"][1], centroids["NCEP/NCAR"][0]]],
            color="#93a8bc", weight=2, opacity=0.8, dash_array="6,6",
        ).add_to(m)
        folium.Marker(
            mid,
            icon=folium.DivIcon(html=(
                f'<div style="background:#0e1d2c;border:1px solid #2c4d68;border-radius:4px;'
                f'padding:2px 6px;font:600 11px \'IBM Plex Mono\',monospace;color:#eaf1f7;white-space:nowrap;'
                f'transform:translate(-50%,-50%);">{dist_km:.2f} km apart</div>'
            )),
        ).add_to(m)

    primary_source = next((s for s in data["sources"] if s["name"] == "ERA5"), data["sources"][0] if data["sources"] else None)
    if primary_source is not None:
        m.get_root().html.add_child(folium.Element(
            render_drift_animation(m.get_name(), primary_source, data["hours_back"])
        ))

    if ranking is not None:
        vessels_group = folium.FeatureGroup(name=f"Candidate vessels (top {min(N_VESSEL_MARKERS, len(ranking['ranking']))})", show=True)
        for v in ranking["ranking"][:N_VESSEL_MARKERS]:
            is_top = v["rank"] == 1
            # folium's built-in Icon only takes named colors (not arbitrary hex) --
            # "orange" and "cadetblue" are the closest real matches to this theme's
            # amber/teal for a real ship glyph (see ship_icon()).
            icon_color = "orange" if is_top else "cadetblue"
            ship_name = v["ship_name"] or "(unnamed)"
            folium.Marker(
                [v["lat"], v["lon"]],
                # Top suspect gets a permanent label (real ship name), matching the
                # user-supplied reference mockup's always-visible top-suspect marker --
                # every other candidate stays hover-only so the default view isn't
                # cluttered with N labels.
                tooltip=(
                    folium.Tooltip(f"★ {ship_name} — top suspect", permanent=True, direction="right")
                    if is_top else f"#{v['rank']} {ship_name} -- {v['match_score_pct']:.0f}% match score"
                ),
                popup=folium.Popup(
                    f"<b>#{v['rank']} {ship_name}</b>{' &mdash; TOP SUSPECT' if is_top else ''}<br>"
                    f"MMSI {v['mmsi'] or '-'} &middot; flag {v['flag'] or '-'}<br>"
                    f"{v['distance_km']:.1f} km from origin, {v['time_gap_hours']:.1f}h time gap<br>"
                    f"match score: {v['match_score_pct']:.0f}%",
                    max_width=260,
                ),
                icon=ship_icon(icon_color),
            ).add_to(vessels_group)
            bounds_points.append([v["lat"], v["lon"]])
        vessels_group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    Fullscreen().add_to(m)
    m.fit_bounds(bounds_points, padding=(30, 30))
    return m


def main() -> None:
    anonymize = "--anonymize" in sys.argv
    out_dir = OUT_DIR.parent / "output_anon" if anonymize else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, case_id in enumerate(CASE_IDS):
        data_path = DATA_DIR / f"drift_{case_id.replace('-', '')}.json"
        ranking_path = DATA_DIR / f"vessel_ranking_{case_id.replace('-', '')}.json"
        if not data_path.exists():
            print(f"SKIP {case_id}: {data_path} not found (run scripts/export_drift_dashboard_data.py for it first).")
            continue
        data = json.loads(data_path.read_text())
        ranking = json.loads(ranking_path.read_text()) if ranking_path.exists() else None
        if anonymize:
            ranking = anonymize_ranking(ranking)
        m = build_map(data, ranking)
        out_name = "map.html" if i == 0 else f"map_{case_id.replace('-', '')}.html"
        out_path = out_dir / out_name
        m.save(str(out_path))
        print(f"wrote {out_path}{' (anonymized)' if anonymize else ''}")


if __name__ == "__main__":
    main()
