"""
Builds src/dashboard/output/tile_locations.html: a small Folium map showing
the REAL geographic location of each of the 10 demo detection tiles (from
web/public/demo/tile_locations.json). Deliberately NO drift cone, NO
vessels, NO time-based computation of any kind -- these tiles have no real
acquisition timestamp (verified, see LOG.md), so nothing beyond plotting
their real embedded coordinates is legitimate here. Same dark basemap
(CartoDB dark_matter) as src/dashboard/build_map.py's real drift map, for
visual consistency -- not a new visual language.

Usage:
    venv\\Scripts\\python.exe scripts\\build_tile_locations_map.py
"""

from __future__ import annotations

import json
from pathlib import Path

import folium

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCATIONS_JSON = REPO_ROOT / "web" / "public" / "demo" / "tile_locations.json"
OUT_HTML = REPO_ROOT / "src" / "dashboard" / "output" / "tile_locations.html"

COLOR_BY_LABEL = {"oil": "#d9534f", "lookalike": "#e6a23c", "no_oil": "#2fb88a"}


def main() -> None:
    locations = json.loads(LOCATIONS_JSON.read_text())

    m = folium.Map(location=[20, 0], zoom_start=2, tiles=None, control_scale=True)
    folium.TileLayer("CartoDB dark_matter", name="Dark basemap").add_to(m)

    for loc in locations:
        color = COLOR_BY_LABEL.get(loc["label"], "#93a8bc")
        folium.CircleMarker(
            location=[loc["lat"], loc["lon"]],
            radius=7,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            weight=2,
            popup=folium.Popup(
                f"<b>{loc['image_id']}</b><br>{loc['label']}<br>{loc['lat']:.3f}, {loc['lon']:.3f}",
                max_width=200,
            ),
            tooltip=loc["image_id"],
        ).add_to(m)

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(OUT_HTML))
    print(f"wrote {OUT_HTML} ({len(locations)} real tile locations)")


if __name__ == "__main__":
    main()
