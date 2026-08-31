"""
Inserts a small "Tile Locations" panel into src/dashboard/output/index.html,
right after the Detection Overlay demo grid and before the geo-strip
section. Shows the real (verified) coordinates of the 10 demo tiles via an
iframe to tile_locations.html (see scripts/build_tile_locations_map.py) --
explicitly labeled to avoid confusion with the real ow-0001/ow-0002 drift
map below. No drift, no vessels, no time-based computation -- see LOG.md
for why (no real acquisition timestamp exists for these tiles).

Backs up index.html before writing, restores it automatically on any
failure so the file is never left in a broken state.

Usage:
    venv\\Scripts\\python.exe scripts\\add_tile_locations_panel.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_HTML = REPO_ROOT / "src" / "dashboard" / "output" / "index.html"
BACKUP_PATH = DASHBOARD_HTML.with_suffix(".html.prelocations_backup")

NEW_CSS = """
    .tile-locations-panel{margin:0 20px 18px;background:var(--navy-app);border:1px solid var(--line-soft);border-radius:10px;padding:14px 16px;}
    .tile-locations-panel .panel-sub{margin-bottom:10px;}
    .tile-locations-map{width:100%;height:280px;border:0;border-radius:8px;display:block;}
"""

NEW_PANEL_HTML = """
    <div class="tile-locations-panel">
      <div class="panel-title">Tile Locations <span style="font-weight:400;color:var(--fog-dim);font-size:11px;">(real coordinates only -- no drift or attribution computed)</span></div>
      <div class="panel-sub">These 10 validation tiles have real embedded geographic coordinates but no acquisition timestamp, so no wind/current/AIS-based reconstruction is possible or shown here -- see docs/metric_audit.md and LOG.md. This is location only, unrelated to the ow-0001/ow-0002 drift case study below.</div>
      <iframe class="tile-locations-map" src="tile_locations.html" title="Real locations of the 10 demo tiles"></iframe>
    </div>
"""


def main() -> None:
    if not DASHBOARD_HTML.exists():
        raise RuntimeError(f"{DASHBOARD_HTML} not found -- aborting.")

    shutil.copy(DASHBOARD_HTML, BACKUP_PATH)
    print(f"backed up to {BACKUP_PATH}")

    try:
        content = DASHBOARD_HTML.read_text(encoding="utf-8")

        anchor = '<div class="geo-strip">'
        anchor_idx = content.find(anchor)
        if anchor_idx == -1:
            raise RuntimeError("Could not find the geo-strip anchor -- aborting without modifying the file.")

        style_close = content.find("</style>")
        if style_close == -1:
            raise RuntimeError("Could not find </style> -- aborting without modifying the file.")

        content = content[:style_close] + NEW_CSS + content[style_close:]
        # re-find anchor_idx since inserting CSS shifted offsets
        anchor_idx = content.find(anchor)
        content = content[:anchor_idx] + NEW_PANEL_HTML.strip() + "\n\n    " + content[anchor_idx:]

        DASHBOARD_HTML.write_text(content, encoding="utf-8")
        print(f"wrote {DASHBOARD_HTML} with the new Tile Locations panel")

    except Exception as e:
        print(f"FAILED ({e}) -- restoring backup")
        shutil.copy(BACKUP_PATH, DASHBOARD_HTML)
        raise


if __name__ == "__main__":
    main()
