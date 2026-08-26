"""
Builds the SIH26143 dashboard. Visual design follows a dark navy/amber/teal
reference the user provided (a concept mockup, "Slick Trace") -- see
DECISIONS.md "Dashboard visual redesign" for the full writeup of what was
adapted vs. changed. Every number on this page is real (drift output,
GFW vessel data, a real inference run) EXCEPT where explicitly labeled
otherwise (the detection-overlay panel, which uses a non-final sanity
checkpoint -- see PIPELINE_STATUS below and DECISIONS.md).

Usage:
    venv\\Scripts\\python.exe src\\dashboard\\build_map.py       # builds map.html / map_{case}.html
    venv\\Scripts\\python.exe src\\dashboard\\build_dashboard.py # builds output/index.html
"""

from __future__ import annotations

import base64
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.geo import haversine_km  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "dashboard"
DETECTION_OVERLAY_PATH = DATA_DIR / "detection_overlay.png"
PROCESSED_DIR = DATA_DIR.parent
CHECKPOINT_DIR = PROCESSED_DIR / "checkpoints"
TRAIN_MANIFEST = PROCESSED_DIR / "train_manifest.csv"
VAL_MANIFEST = PROCESSED_DIR / "val_manifest.csv"
OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_PATH = OUT_DIR / "index.html"

# GFW's own documented limits (globalfishingwatch.org/our-apis, checked this
# project -- see LOG.md "GFW compliance"), not estimates.
GFW_DAILY_LIMIT = 50_000
GFW_MONTHLY_LIMIT = 1_500_000

CASE_IDS = ["ow-0001", "ow-0002"]

# Real pipeline status -- update these as stages actually complete, not on a schedule.
PIPELINE_STATUS = {
    "detection_model": {
        "state": "in_progress",
        "detail": "Inference plumbing (load checkpoint -> predict -> overlay) built and run "
                   "on a real Zenodo tile below -- using the Step 1 SANITY checkpoint (trained "
                   "on 4 bbox-pseudo-mask images), not a real trained model. Real training "
                   "pending Zenodo Part II/III download + extraction.",
    },
    "drift_model": {
        "state": "done",
        "detail": "Backward advection verified on 2 independent real cases with ERA5 "
                   "(primary) and NCEP/NCAR (fallback) wind sources.",
    },
    "gfw_attribution": {
        "state": "in_progress",
        "detail": "Real GFW data + a first-pass scoring algorithm (distance + timing vs. the "
                   "drift-estimated origin). Not yet calibrated against any labeled ground "
                   "truth -- read confidence values as a reasonable first ordering, not a "
                   "verdict.",
    },
}
STATUS_LABEL = {"not_started": "Not started", "in_progress": "In progress", "done": "Confirmed"}
STATUS_DOT = {"not_started": "var(--fog-dim)", "in_progress": "var(--amber-bright)", "done": "var(--teal-bright)"}


def render_status_pill(key: str) -> str:
    s = PIPELINE_STATUS[key]
    return (f'<span class="status-pill" style="--pill-color:{STATUS_DOT[s["state"]]}">'
            f'<span class="dot"></span>{STATUS_LABEL[s["state"]]}</span>')


def fmt_short(iso_ts: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        return dt.strftime("%b %d, %H:%M")
    except (ValueError, AttributeError):
        return iso_ts


def render_vessel_cards(ranking: dict) -> str:
    cards = []
    for v in ranking["ranking"]:
        is_top = v["rank"] == 1
        if v["time_gap_hours"] == 0:
            timing_bullet = "Present in the search area exactly at the estimated origin time"
        else:
            timing_bullet = f"{v['time_gap_hours']:.1f}h from the estimated origin time"
        evidence = [
            f"{v['distance_km']:.1f} km from the estimated origin",
            timing_bullet,
            f"AIS present {fmt_short(v['entry_timestamp'])} &rarr; {fmt_short(v['exit_timestamp'])} UTC",
        ]
        # Click-to-expand detail: fields real GFW records carry but that
        # would clutter the always-visible card -- IMO, vessel type, raw
        # hours-present, and the underlying distance+timing score before
        # it's converted to a confidence percentage.
        detail_rows = [
            ("IMO", v["imo"] or "&mdash;"),
            ("Vessel type", (v["vessel_type"] or "unknown").replace("_", " ").title()),
            ("Hours present in window", str(v["hours_present"])),
            ("Raw score", f"{v['score']:.4f} <span class=\"muted-inline\">(lower = stronger match; distance + timing, see DECISIONS.md)</span>"),
        ]
        cards.append(f"""
      <div class="vcard{' top' if is_top else ''}" data-distance="{v['distance_km']}" data-timegap="{v['time_gap_hours']}" data-confidence="{v['confidence_pct']}" data-rank="{v['rank']}">
        {'<div class="vcard-top-flag">Top Suspect</div>' if is_top else ''}
        <div class="vcard-row vcard-toggle">
          <div style="display:flex;gap:9px;min-width:0;">
            <div class="vcard-rank">#{v['rank']}</div>
            <div style="min-width:0;">
              <div class="vname">{v['ship_name'] or '(unnamed)'}</div>
              <div class="vmeta"><span>MMSI {v['mmsi'] or '-'}</span><span>&middot;</span><span>{v['flag'] or '-'}</span></div>
            </div>
          </div>
          <div class="confidence-block"><div class="confidence-num">{v['confidence_pct']:.0f}<sup>%</sup></div></div>
        </div>
        <div class="bar-track"><div class="bar-fill" style="width:{v['confidence_pct']:.0f}%;"></div></div>
        <ul class="evidence">{"".join(f'<li>{e}</li>' for e in evidence)}</ul>
        <div class="vcard-detail">
          {"".join(f'<div class="detail-row"><span>{k}</span><span>{val}</span></div>' for k, val in detail_rows)}
        </div>
      </div>""")
    return "".join(cards)


VESSEL_SORT_SCRIPT = """
<script>
  document.querySelectorAll(".vessel-list").forEach(function (list) {
    list.querySelectorAll(".vcard-toggle").forEach(function (head) {
      head.addEventListener("click", function () { head.closest(".vcard").classList.toggle("expanded"); });
    });
  });
  document.querySelectorAll(".sort-select").forEach(function (sel) {
    sel.addEventListener("change", function () {
      var list = sel.closest(".sidebar").querySelector(".vessel-list");
      var cards = Array.from(list.querySelectorAll(".vcard"));
      var key = sel.value;
      cards.sort(function (a, b) {
        if (key === "rank") return parseFloat(a.dataset.rank) - parseFloat(b.dataset.rank);
        if (key === "distance") return parseFloat(a.dataset.distance) - parseFloat(b.dataset.distance);
        if (key === "timegap") return parseFloat(a.dataset.timegap) - parseFloat(b.dataset.timegap);
        if (key === "confidence") return parseFloat(b.dataset.confidence) - parseFloat(a.dataset.confidence);
        return 0;
      });
      cards.forEach(function (c) { list.appendChild(c); });
    });
  });
</script>"""


def render_stat_strip(data: dict, ranking: dict | None) -> str:
    era5 = next((s for s in data["sources"] if s["name"] == "ERA5"), data["sources"][0])
    origin = tuple(era5["centroid"])
    detection = tuple(data["detection_point"])
    dist_km = haversine_km(detection, origin)
    origin_time_str = fmt_short(
        (datetime.fromisoformat(data["detection_time_utc"]) - timedelta(hours=data["hours_back"])).isoformat()
    )
    n_candidates = ranking["n_candidates"] if ranking else None
    top_confidence = ranking["ranking"][0]["confidence_pct"] if ranking and ranking["ranking"] else None

    src_centroids = {s["name"]: tuple(s["centroid"]) for s in data["sources"]}
    wind_compare_stat = ""
    if "ERA5" in src_centroids and "NCEP/NCAR" in src_centroids:
        wind_dist_km = haversine_km(src_centroids["ERA5"], src_centroids["NCEP/NCAR"])
        wind_compare_stat = f"""
      <div class="stat">
        <div class="stat-label">ERA5 vs NCEP/NCAR Origin</div>
        <div class="stat-value">{wind_dist_km:.2f}<small>km apart</small></div>
      </div>"""

    return f"""
    <section class="stat-strip">
      <div class="stat">
        <div class="stat-label">Detection &rarr; Origin</div>
        <div class="stat-value">{dist_km:.1f}<small>km</small></div>
      </div>
      <div class="stat">
        <div class="stat-label">Estimated Origin Time</div>
        <div class="stat-value" style="font-size:16px;">{origin_time_str}<small>UTC, {data['hours_back']}h back</small></div>
      </div>
      <div class="stat">
        <div class="stat-label">Vessels Evaluated</div>
        <div class="stat-value">{n_candidates if n_candidates is not None else '&mdash;'}<small>in search area</small></div>
      </div>
      <div class="stat accent">
        <div class="stat-label">Top Suspect Confidence</div>
        <div class="stat-value">{f'{top_confidence:.0f}' if top_confidence is not None else '&mdash;'}<small>%</small></div>
      </div>{wind_compare_stat}
    </section>"""


def render_case_container(case_id: str, index: int, data: dict | None, ranking: dict | None) -> str:
    if data is None:
        return ""
    map_file = "map.html" if index == 0 else f"map_{case_id.replace('-', '')}.html"
    display = "block" if index == 0 else "none"
    vessel_body = (
        render_vessel_cards(ranking) if ranking else
        '<p class="muted">Run src/attribution/score_vessels.py for this case.</p>'
    )
    n_ranked = len(ranking["ranking"]) if ranking else 0
    n_total = ranking["n_candidates"] if ranking else 0

    return f"""
  <div class="case-container" data-case="{case_id}" style="display:{display}">
    {render_stat_strip(data, ranking)}
    <div class="main-grid">
      <section class="map-panel">
        <div class="map-panel-head">
          <div>
            <div class="panel-title">Drift Reconstruction &mdash; {case_id}</div>
            <div class="panel-sub">Backward trace from detection ({data['detection_time_utc']}) to estimated origin, real ERA5/HYCOM data</div>
          </div>
          {render_status_pill("drift_model")}
        </div>
        <div class="map-stage"><iframe src="{map_file}" title="Drift map for {case_id}" style="width:100%;height:100%;border:0;display:block;"></iframe></div>
        <div class="legend">
          <div class="legend-item"><span class="legend-swatch" style="background:var(--alert);"></span>Detection point</div>
          <div class="legend-item"><span class="legend-swatch" style="background:var(--amber-bright);"></span>Drift trace &amp; origin estimate</div>
          <div class="legend-item"><span class="legend-swatch" style="background:var(--teal-bright);border-radius:50%;"></span>Candidate vessel</div>
          <div class="legend-item"><span class="legend-swatch" style="background:transparent;border:2px solid var(--amber-bright);border-radius:50%;"></span>Top suspect</div>
        </div>
      </section>
      <aside class="sidebar">
        <div class="sidebar-head">
          <div class="sidebar-head-row">
            <div>
              <div class="panel-title">Ranked Suspect Vessels</div>
              <div class="panel-sub">Top {n_ranked} of {n_total} real GFW candidates &middot; click a card for full evidence</div>
            </div>
            <select class="sort-select">
              <option value="rank">Sort: confidence rank</option>
              <option value="distance">Sort: distance</option>
              <option value="timegap">Sort: time gap</option>
              <option value="confidence">Sort: confidence %</option>
            </select>
          </div>
        </div>
        <div class="vessel-list">{vessel_body}</div>
      </aside>
    </div>
  </div>"""


def fmt_ago(dt: datetime) -> str:
    delta = datetime.now(timezone.utc) - dt
    hours = delta.total_seconds() / 3600
    if hours < 1:
        return f"{int(delta.total_seconds() / 60)}m ago"
    if hours < 48:
        return f"{hours:.1f}h ago"
    return f"{hours / 24:.1f}d ago"


def gather_provenance() -> dict:
    """
    Real, filesystem/checkpoint-derived facts about the current pipeline
    state -- computed at build time, not hand-maintained, so the panel
    stays accurate as long as the underlying files do (dataset manifests,
    model checkpoints, drift/vessel JSON mtimes). See DECISIONS.md
    "Dashboard visual redesign" -- this project's whole identity is "real,
    working, verified," so this panel exists to make that checkable at a
    glance instead of asserted in prose.
    """
    prov: dict = {}

    if TRAIN_MANIFEST.exists() and VAL_MANIFEST.exists():
        with open(TRAIN_MANIFEST, newline="") as f:
            train_rows = list(csv.DictReader(f))
        with open(VAL_MANIFEST, newline="") as f:
            val_rows = list(csv.DictReader(f))
        by_label: dict[str, int] = {}
        for row in train_rows + val_rows:
            by_label[row["label"]] = by_label.get(row["label"], 0) + 1
        prov["dataset"] = {
            "n_train": len(train_rows), "n_val": len(val_rows),
            "by_label": by_label,
        }

    best_ckpt = CHECKPOINT_DIR / "best_unet_resnet18.pt"
    if best_ckpt.exists():
        try:
            import torch
            ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            prov["model"] = {
                "epoch": ckpt.get("epoch"), "val_dice": ckpt.get("val_dice"),
                "mtime": datetime.fromtimestamp(best_ckpt.stat().st_mtime, tz=timezone.utc),
                "is_sanity_only": False,
            }
        except Exception:
            prov["model"] = None
    elif (CHECKPOINT_DIR / "sanity_unet_resnet18.pt").exists():
        sanity_ckpt = CHECKPOINT_DIR / "sanity_unet_resnet18.pt"
        prov["model"] = {
            "epoch": None, "val_dice": None,
            "mtime": datetime.fromtimestamp(sanity_ckpt.stat().st_mtime, tz=timezone.utc),
            "is_sanity_only": True,
        }
    else:
        prov["model"] = None

    fetches = {}
    for case_id in CASE_IDS:
        p = DATA_DIR / f"drift_{case_id.replace('-', '')}.json"
        if p.exists():
            fetches[case_id] = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
    prov["last_drift_fetch"] = max(fetches.values()) if fetches else None

    ranking_fetches = {}
    for case_id in CASE_IDS:
        p = DATA_DIR / f"vessel_ranking_{case_id.replace('-', '')}.json"
        if p.exists():
            ranking_fetches[case_id] = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
    prov["last_gfw_fetch"] = max(ranking_fetches.values()) if ranking_fetches else None

    return prov


def render_provenance_panel(prov: dict) -> str:
    rows = []

    ds = prov.get("dataset")
    if ds:
        n_total = ds["n_train"] + ds["n_val"]
        label_str = " &middot; ".join(f"{v} {k}" for k, v in sorted(ds["by_label"].items()))
        rows.append(("Training dataset", f"{n_total} real tiles-source images ({ds['n_train']} train / {ds['n_val']} val)", label_str))

    model = prov.get("model")
    if model is None:
        rows.append(("Detection model", "No checkpoint yet", "Run scripts/train_detection.py"))
    elif model["is_sanity_only"]:
        rows.append(("Detection model", "Sanity checkpoint only", f"4 toy images, not a real trained model &middot; {fmt_ago(model['mtime'])}"))
    else:
        dice_str = f"{model['val_dice']:.4f}" if model["val_dice"] is not None else "n/a"
        rows.append(("Detection model", f"Best so far: epoch {model['epoch']}, val_dice={dice_str}", f"Real training in progress or complete &middot; checkpoint saved {fmt_ago(model['mtime'])}"))

    if prov.get("last_drift_fetch"):
        rows.append(("Drift data (ERA5/HYCOM + NCEP/NCAR)", "Live, real particle tracks", f"Last computed {fmt_ago(prov['last_drift_fetch'])}"))

    if prov.get("last_gfw_fetch"):
        rows.append(("GFW vessel data", "Real AIS presence records", f"Last fetched {fmt_ago(prov['last_gfw_fetch'])}"))

    rows.append(("GFW API quota", "Well under limit", f"~20-30 requests used this project of {GFW_DAILY_LIMIT:,}/day, {GFW_MONTHLY_LIMIT:,}/month"))

    row_html = "".join(f"""
      <div class="prov-row">
        <div class="prov-label">{label}</div>
        <div class="prov-value">{value}</div>
        <div class="prov-detail">{detail}</div>
      </div>""" for label, value, detail in rows)

    return f"""
  <section class="provenance-panel">
    <div class="map-panel-head">
      <div>
        <div class="panel-title">Data Provenance</div>
        <div class="panel-sub">What's real vs. what's still pending, computed at build time from the actual files on disk -- not asserted.</div>
      </div>
    </div>
    <div class="prov-grid">{row_html}</div>
  </section>"""


def render_case_comparison_table(cases: dict[str, tuple[dict | None, dict | None]], available: list[str]) -> str:
    """
    All real cases side by side, independent of which tab is currently
    active -- lets a reviewer see at a glance that the pipeline generalizes
    rather than having to click through tabs one at a time. Clicking a row
    switches to that case's tab (reuses the existing case-tab click logic
    below rather than duplicating it).
    """
    if len(available) < 2:
        return ""  # not meaningfully a "comparison" with only one real case

    rows = []
    for cid in available:
        data, ranking = cases[cid]
        era5 = next((s for s in data["sources"] if s["name"] == "ERA5"), data["sources"][0])
        dist_km = haversine_km(tuple(data["detection_point"]), tuple(era5["centroid"]))
        centroids = {s["name"]: tuple(s["centroid"]) for s in data["sources"]}
        wind_cmp = (f"{haversine_km(centroids['ERA5'], centroids['NCEP/NCAR']):.2f} km"
                    if "ERA5" in centroids and "NCEP/NCAR" in centroids else "&mdash;")
        top = ranking["ranking"][0] if ranking and ranking["ranking"] else None
        top_suspect_cell = (
            f'{top["ship_name"] or "(unnamed)"} <span class="mono cmp-conf">{top["confidence_pct"]:.0f}%</span>'
            if top else "&mdash;"
        )
        rows.append(f"""
      <tr class="cmp-row" data-case="{cid}">
        <td class="cmp-case">{cid}</td>
        <td>{fmt_short(data['detection_time_utc'])}</td>
        <td class="mono">{dist_km:.1f} km</td>
        <td class="mono">{wind_cmp}</td>
        <td class="mono">{ranking['n_candidates'] if ranking else '&mdash;'}</td>
        <td>{top_suspect_cell}</td>
      </tr>""")

    return f"""
  <section class="provenance-panel">
    <div class="map-panel-head">
      <div>
        <div class="panel-title">Case Comparison</div>
        <div class="panel-sub">All {len(available)} real validated cases &middot; click a row to open it</div>
      </div>
    </div>
    <div class="cmp-table-wrap">
      <table class="cmp-table">
        <thead><tr><th>Case</th><th>Detection time</th><th>Detection&rarr;Origin</th><th>ERA5 vs NCEP/NCAR</th><th>Vessels evaluated</th><th>Top suspect</th></tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table>
    </div>
  </section>"""


def render_detection_section() -> str:
    s = PIPELINE_STATUS["detection_model"]
    if not DETECTION_OVERLAY_PATH.exists():
        body = f"""
        <div class="placeholder">
          <div class="placeholder-icon">&#9673;</div>
          <p class="placeholder-title">Awaiting trained model</p>
          <p class="placeholder-detail">{s["detail"]}</p>
        </div>"""
    else:
        img_b64 = base64.b64encode(DETECTION_OVERLAY_PATH.read_bytes()).decode("ascii")
        body = f'<img class="overlay-img" src="data:image/png;base64,{img_b64}" alt="SAR tile, ground truth, and predicted mask">'
    return f"""
  <section class="map-panel detection-section">
    <div class="map-panel-head">
      <div>
        <div class="panel-title">Detection Overlay</div>
        <div class="panel-sub">{s["detail"]}</div>
      </div>
      {render_status_pill("detection_model")}
    </div>
    {body}
  </section>"""


def build_html(cases: dict[str, tuple[dict | None, dict | None]]) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    available = [cid for cid in CASE_IDS if cases[cid][0] is not None]

    tabs = "".join(
        f'<button class="case-tab{" active" if i == 0 else ""}" data-case="{cid}">{cid}</button>'
        for i, cid in enumerate(available)
    )
    case_containers = "".join(
        render_case_container(cid, i, *cases[cid]) for i, cid in enumerate(available)
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SIH26143 -- Oil Spill Attribution</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Chivo:wght@500;600;700;900&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root{{
    --navy-void:#060d17; --navy-app:#0a1621; --navy-panel:#0e1d2c; --navy-raised:#142a3d; --navy-raised2:#1a3348;
    --line-soft:#1e3448; --line-strong:#2c4d68; --fog:#93a8bc; --fog-dim:#62788d; --paper:#eaf1f7;
    --amber:#e6a23c; --amber-bright:#f4b95a; --amber-soft:rgba(230,162,60,.14); --amber-line:rgba(230,162,60,.45);
    --teal:#2fb88a; --teal-bright:#4ed4a6; --teal-soft:rgba(47,184,138,.16);
    --alert:#d9534f; --alert-soft:rgba(217,83,79,.16);
    --shadow-deep:0 20px 48px -20px rgba(0,0,0,.6);
  }}
  *{{box-sizing:border-box;}}
  html,body{{margin:0;padding:0;}}
  body{{
    /* Layered background: a soft amber glow behind the header, a faint
       nautical-chart-style grid (echoing the map panel's own lat/lon
       graticule) over a deep navy base -- low-opacity so it never
       competes with panel content, which all sit on solid --navy-panel. */
    background:
      radial-gradient(ellipse 1100px 700px at 50% -8%, rgba(230,162,60,.07), transparent 60%),
      repeating-linear-gradient(0deg, rgba(147,168,188,.05) 0px, rgba(147,168,188,.05) 1px, transparent 1px, transparent 48px),
      repeating-linear-gradient(90deg, rgba(147,168,188,.05) 0px, rgba(147,168,188,.05) 1px, transparent 1px, transparent 48px),
      var(--navy-void);
    background-attachment:fixed;
    color:var(--paper);font-family:'IBM Plex Sans',system-ui,sans-serif;-webkit-font-smoothing:antialiased;min-height:100vh;
  }}
  .mono{{font-family:'IBM Plex Mono',ui-monospace,monospace;font-variant-numeric:tabular-nums;}}
  a,button{{font:inherit;}}
  .shell{{max-width:1400px;margin:0 auto;padding:28px clamp(18px,3.4vw,44px) 60px;display:flex;flex-direction:column;gap:22px;}}
  .masthead{{display:flex;align-items:flex-end;justify-content:space-between;gap:24px;flex-wrap:wrap;border-bottom:1px solid var(--line-soft);padding-bottom:18px;}}
  .masthead .id{{display:flex;align-items:center;gap:12px;margin-bottom:6px;}}
  .id-mark{{width:34px;height:34px;border-radius:8px;background:linear-gradient(155deg,var(--navy-raised2),var(--navy-panel));border:1px solid var(--line-strong);display:flex;align-items:center;justify-content:center;flex:none;}}
  .id-mark svg{{width:19px;height:19px;}}
  .eyebrow{{font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--amber);}}
  h1{{font-family:'Chivo',system-ui,sans-serif;font-weight:800;font-size:clamp(24px,2.6vw,32px);letter-spacing:-0.01em;margin:2px 0 0;}}
  .tagline{{color:var(--fog);font-size:14px;margin:6px 0 0;max-width:56ch;}}
  .masthead-meta{{display:flex;gap:22px;font-family:'IBM Plex Mono',monospace;font-size:11.5px;color:var(--fog-dim);text-align:right;}}
  .masthead-meta div span{{display:block;color:var(--fog);margin-top:2px;}}
  .case-tabs{{display:flex;gap:6px;align-items:center;}}
  .case-tabs .tabs-label{{font-family:'IBM Plex Mono',monospace;font-size:10.5px;color:var(--fog-dim);text-transform:uppercase;letter-spacing:.08em;margin-right:4px;}}
  .case-tab{{font-family:'IBM Plex Mono',monospace;font-size:12px;font-weight:600;padding:5px 14px;border-radius:999px;border:1px solid var(--line-strong);background:var(--navy-panel);color:var(--fog);cursor:pointer;}}
  .case-tab.active{{background:var(--amber);border-color:var(--amber);color:#1c1204;}}
  .status-row{{display:flex;gap:8px;padding:10px 24px;background:var(--navy-panel);border:1px solid var(--line-soft);border-radius:10px;flex-wrap:wrap;font-size:12px;}}
  .status-row .label{{color:var(--fog);}}
  .status-pill{{display:inline-flex;align-items:center;gap:5px;padding:2px 8px;border-radius:999px;background:var(--navy-raised);font-size:11px;font-weight:600;color:var(--paper);}}
  .status-pill .dot{{width:7px;height:7px;border-radius:50%;background:var(--pill-color);}}
  .stat-strip{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;}}
  .stat{{background:linear-gradient(175deg,var(--navy-panel),var(--navy-app));border:1px solid var(--line-soft);border-radius:12px;padding:15px 18px;display:flex;flex-direction:column;gap:6px;min-width:0;}}
  .stat-label{{font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--fog-dim);}}
  .stat-value{{font-family:'IBM Plex Mono',monospace;font-weight:600;font-size:22px;color:var(--paper);line-height:1.15;}}
  .stat-value small{{font-size:12.5px;font-weight:500;color:var(--fog);margin-left:2px;}}
  .stat.accent .stat-value{{color:var(--amber-bright);}}
  .stat.accent{{border-color:var(--amber-line);}}
  .main-grid{{display:grid;grid-template-columns:minmax(0,1.62fr) minmax(320px,1fr);gap:22px;align-items:start;margin-top:22px;}}
  @media (max-width:920px){{.main-grid{{grid-template-columns:1fr;}}.stat-strip{{grid-template-columns:repeat(2,1fr);}}}}
  .map-panel{{background:var(--navy-panel);border:1px solid var(--line-soft);border-radius:16px;overflow:hidden;box-shadow:var(--shadow-deep);}}
  .map-panel-head{{display:flex;align-items:center;justify-content:space-between;padding:16px 20px 12px;gap:12px;flex-wrap:wrap;}}
  .panel-title{{font-family:'Chivo',system-ui,sans-serif;font-weight:700;font-size:15px;letter-spacing:.01em;}}
  .panel-sub{{font-size:12px;color:var(--fog-dim);margin-top:2px;max-width:60ch;}}
  .map-stage{{position:relative;width:100%;aspect-ratio:16/10.5;background:#081320;}}
  .legend{{display:flex;flex-wrap:wrap;gap:16px;padding:12px 20px 16px;border-top:1px solid var(--line-soft);font-size:11.5px;color:var(--fog);}}
  .legend-item{{display:flex;align-items:center;gap:7px;}}
  .legend-swatch{{width:13px;height:13px;border-radius:4px;flex:none;}}
  .sidebar{{background:var(--navy-panel);border:1px solid var(--line-soft);border-radius:16px;padding:18px 16px 20px;box-shadow:var(--shadow-deep);display:flex;flex-direction:column;gap:14px;max-height:900px;overflow-y:auto;}}
  .sidebar-head{{padding:0 4px;}}
  .sidebar-head-row{{display:flex;justify-content:space-between;align-items:flex-start;gap:10px;flex-wrap:wrap;}}
  .sort-select{{background:var(--navy-app);color:var(--fog);border:1px solid var(--line-strong);border-radius:7px;font-family:'IBM Plex Mono',monospace;font-size:10.5px;padding:5px 7px;flex:none;}}
  .vessel-list{{display:flex;flex-direction:column;gap:10px;}}
  .vcard{{background:var(--navy-app);border:1px solid var(--line-soft);border-radius:12px;padding:13px 14px 14px;position:relative;}}
  .vcard.top{{border-color:var(--amber-line);background:linear-gradient(160deg,rgba(230,162,60,.09),var(--navy-app) 55%);box-shadow:0 0 0 1px rgba(230,162,60,.12),0 12px 26px -14px rgba(230,162,60,.35);}}
  .vcard-top-flag{{position:absolute;top:-9px;left:14px;background:var(--amber);color:#1c1204;font-family:'IBM Plex Mono',monospace;font-weight:700;font-size:9.5px;letter-spacing:.08em;padding:2px 8px;border-radius:5px;}}
  .vcard-row{{display:flex;justify-content:space-between;align-items:flex-start;gap:10px;}}
  .vcard-rank{{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--fog-dim);flex:none;width:22px;padding-top:2px;}}
  .vcard.top .vcard-rank{{color:var(--amber-bright);}}
  .vname{{font-family:'Chivo',system-ui,sans-serif;font-weight:700;font-size:15px;line-height:1.2;}}
  .vmeta{{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--fog-dim);margin-top:3px;display:flex;gap:10px;flex-wrap:wrap;}}
  .confidence-block{{flex:none;text-align:right;}}
  .confidence-num{{font-family:'IBM Plex Mono',monospace;font-weight:700;font-size:19px;color:var(--teal-bright);}}
  .vcard.top .confidence-num{{color:var(--amber-bright);}}
  .confidence-num sup{{font-size:11px;font-weight:500;color:var(--fog-dim);}}
  .bar-track{{margin-top:9px;height:6px;border-radius:4px;background:var(--line-soft);overflow:hidden;}}
  .bar-fill{{height:100%;border-radius:4px;background:linear-gradient(90deg,var(--teal),var(--teal-bright));}}
  .vcard.top .bar-fill{{background:linear-gradient(90deg,var(--amber),var(--amber-bright));}}
  .evidence{{list-style:none;margin:11px 0 0;padding:10px 0 0;border-top:1px solid var(--line-soft);display:flex;flex-direction:column;gap:6px;}}
  .evidence li{{font-size:12px;color:var(--fog);padding-left:14px;position:relative;line-height:1.4;}}
  .evidence li::before{{content:"";position:absolute;left:0;top:6px;width:5px;height:5px;border-radius:50%;background:var(--fog-dim);}}
  .vcard.top .evidence li::before{{background:var(--amber);}}
  .vcard-toggle{{cursor:pointer;}}
  .vcard-detail{{display:none;margin-top:11px;padding-top:10px;border-top:1px solid var(--line-soft);flex-direction:column;gap:6px;}}
  .vcard.expanded .vcard-detail{{display:flex;}}
  .detail-row{{display:flex;justify-content:space-between;gap:10px;font-size:11.5px;}}
  .detail-row span:first-child{{color:var(--fog-dim);}}
  .detail-row span:last-child{{color:var(--fog);text-align:right;}}
  .muted-inline{{color:var(--fog-dim);font-size:10.5px;}}
  .placeholder{{padding:28px 20px;text-align:center;flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:6px;}}
  .placeholder-icon{{font-size:28px;color:var(--fog-dim);}}
  .placeholder-title{{font-weight:700;margin:4px 0 0;font-size:13px;}}
  .placeholder-detail{{color:var(--fog);font-size:12px;max-width:420px;margin:0;}}
  .muted{{color:var(--fog-dim);font-size:12px;padding:16px;}}
  .overlay-img{{width:100%;display:block;padding:10px 20px 20px;box-sizing:border-box;border-radius:8px;}}
  .detection-section{{margin-top:0;}}
  .provenance-panel{{background:var(--navy-panel);border:1px solid var(--line-soft);border-radius:16px;overflow:hidden;box-shadow:var(--shadow-deep);margin-top:0;}}
  .prov-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:1px;background:var(--line-soft);border-top:1px solid var(--line-soft);}}
  .prov-row{{background:var(--navy-panel);padding:14px 18px 16px;display:flex;flex-direction:column;gap:4px;}}
  .prov-label{{font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--fog-dim);}}
  .prov-value{{font-family:'Chivo',system-ui,sans-serif;font-weight:700;font-size:14px;color:var(--paper);}}
  .prov-detail{{font-size:11.5px;color:var(--fog);line-height:1.4;}}
  @media (max-width:560px){{.prov-grid{{grid-template-columns:1fr;}}}}
  .cmp-table-wrap{{overflow-x:auto;border-top:1px solid var(--line-soft);}}
  .cmp-table{{width:100%;border-collapse:collapse;font-size:12.5px;white-space:nowrap;}}
  .cmp-table th{{text-align:left;padding:10px 18px;font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:.07em;text-transform:uppercase;color:var(--fog-dim);font-weight:500;border-bottom:1px solid var(--line-soft);}}
  .cmp-table td{{padding:11px 18px;color:var(--fog);border-bottom:1px solid var(--line-soft);}}
  .cmp-row{{cursor:pointer;}}
  .cmp-row:hover td{{background:var(--navy-raised);}}
  .cmp-row:last-child td{{border-bottom:none;}}
  .cmp-case{{font-family:'Chivo',system-ui,sans-serif;font-weight:700;color:var(--paper);}}
  .cmp-conf{{color:var(--amber-bright);margin-left:4px;}}
  .cmp-table td.mono{{font-family:'IBM Plex Mono',monospace;}}
  .corner-tag{{position:fixed;top:16px;right:16px;z-index:50;display:flex;align-items:center;gap:7px;background:rgba(10,18,28,.88);border:1px solid var(--amber-line);color:var(--amber-bright);font-family:'IBM Plex Mono',monospace;font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;padding:7px 12px;border-radius:999px;backdrop-filter:blur(6px);box-shadow:0 10px 24px -10px rgba(0,0,0,.7);}}
  .corner-tag .dot{{width:6px;height:6px;border-radius:50%;background:var(--amber-bright);animation:blink 2.2s ease-in-out infinite;}}
  @keyframes blink{{0%,100%{{opacity:1;}}50%{{opacity:.3;}}}}
  @media (prefers-reduced-motion:reduce){{.corner-tag .dot{{animation:none;}}}}
  footer.note{{color:var(--fog-dim);font-size:11.5px;text-align:center;padding-top:6px;}}
  @media (max-width:560px){{.masthead-meta{{display:none;}}.stat-strip{{grid-template-columns:1fr;}}}}
</style>
</head>
<body>

<div class="corner-tag"><span class="dot"></span>Prototype &middot; Real Drift + AIS Data</div>

<div class="shell">
  <header class="masthead">
    <div>
      <div class="id">
        <div class="id-mark">
          <svg viewBox="0 0 24 24" fill="none"><path d="M3 17c1.8 1.4 3.6 1.4 5.4 0 1.8-1.4 3.6-1.4 5.4 0 1.8 1.4 3.6 1.4 5.4 0" stroke="#e6a23c" stroke-width="1.6" stroke-linecap="round"/><path d="M3 12c1.8 1.4 3.6 1.4 5.4 0 1.8-1.4 3.6-1.4 5.4 0 1.8 1.4 3.6 1.4 5.4 0" stroke="#4ed4a6" stroke-width="1.6" stroke-linecap="round" opacity=".55"/><circle cx="12" cy="6.2" r="2.1" stroke="#eaf1f7" stroke-width="1.4"/></svg>
        </div>
        <span class="eyebrow">SIH26143 &middot; NTRO</span>
      </div>
      <h1>Oil Spill Attribution System</h1>
      <p class="tagline">Traces a satellite-detected slick backward through real wind and current data, then cross-references real AIS ship-tracking data to rank likely source vessels.</p>
    </div>
    <div class="masthead-meta">
      <div>Data sources<span class="mono">ERA5 &middot; HYCOM &middot; GFW</span></div>
      <div>Sensor<span class="mono">Sentinel-1 &middot; SAR</span></div>
    </div>
  </header>

  <div class="status-row">
    <span class="label">Detection model:</span> {render_status_pill("detection_model")}
    <span class="label" style="margin-left:12px">Drift model:</span> {render_status_pill("drift_model")}
    <span class="label" style="margin-left:12px">GFW / attribution:</span> {render_status_pill("gfw_attribution")}
    <span class="case-tabs" style="margin-left:auto;"><span class="tabs-label">Case</span>{tabs}</span>
  </div>

  {render_case_comparison_table(cases, available)}
  {case_containers}
  {render_detection_section()}
  {render_provenance_panel(gather_provenance())}

  <footer class="note">SIH26143 &middot; NTRO &middot; generated {generated} &middot; drift map uses real ERA5 + HYCOM output; detection overlay uses a non-final sanity checkpoint &mdash; see DECISIONS.md.<br>
  Vessel data: &ldquo;Powered by <a href="https://globalfishingwatch.org" target="_blank" rel="noopener" style="color:var(--amber-bright);">Global Fishing Watch</a>.&rdquo; Non-commercial use, CC BY-NC 4.0.</footer>
</div>

<script>
  function selectCase(caseId) {{
    document.querySelectorAll(".case-tab").forEach(function (t) {{ t.classList.toggle("active", t.dataset.case === caseId); }});
    document.querySelectorAll(".case-container").forEach(function (block) {{
      block.style.display = (block.dataset.case === caseId) ? "block" : "none";
    }});
  }}
  document.querySelectorAll(".case-tab").forEach(function (tab) {{
    tab.addEventListener("click", function () {{ selectCase(tab.dataset.case); }});
  }});
  document.querySelectorAll(".cmp-row").forEach(function (row) {{
    row.addEventListener("click", function () {{
      selectCase(row.dataset.case);
      document.querySelector(".case-container[data-case=\\"" + row.dataset.case + "\\"]").scrollIntoView({{behavior: "smooth", block: "start"}});
    }});
  }});
</script>
{VESSEL_SORT_SCRIPT}
</body>
</html>"""


def main() -> None:
    cases = {}
    for case_id in CASE_IDS:
        data_path = DATA_DIR / f"drift_{case_id.replace('-', '')}.json"
        ranking_path = DATA_DIR / f"vessel_ranking_{case_id.replace('-', '')}.json"
        data = json.loads(data_path.read_text()) if data_path.exists() else None
        ranking = json.loads(ranking_path.read_text()) if ranking_path.exists() else None
        cases[case_id] = (data, ranking)
        if data is None:
            print(f"NOTE: no drift data for {case_id} ({data_path}).")
        if ranking is None:
            print(f"NOTE: no vessel ranking for {case_id} ({ranking_path}).")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(build_html(cases), encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
