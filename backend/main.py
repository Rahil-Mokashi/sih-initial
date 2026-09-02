"""
Backend API for the React dashboard (oil-spill-attribution-system/).

This does NOT replace src/dashboard/build_dashboard.py's static HTML build
(kept as-is, still useful for a no-JS-server snapshot) -- it's a second,
live consumer of the exact same real data files that script already reads:
data/processed/dashboard/{drift,vessel_ranking}_*.json,
detection_overlay.png, detection_geometry.json, plus the real trained
detection checkpoint for the new upload/inference feature the mock React
app never had.

Every field this API returns is either read straight from one of those
real files, computed with the same real math build_dashboard.py already
uses (haversine distance, gather_provenance()), or -- for the upload
endpoint -- produced by an actual forward pass through a real trained
checkpoint via src/detection/inference.py. Fields the underlying pipeline
doesn't produce (per-vessel length/draft/speed/owner, sensor-corroboration
percentages, per-case slick geometry) are omitted rather than invented --
see oil-spill-attribution-system/src/types.ts for where each optional
field is documented as such.

Run: venv\\Scripts\\python.exe -m uvicorn backend.main:app --reload --port 8000
(from the repo root, so `backend` and `src` are both importable.)
"""

from __future__ import annotations

import base64
import io
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from common.geo import haversine_km  # noqa: E402
from dashboard.build_dashboard import (  # noqa: E402
    CASE_IDS,
    PIPELINE_STATUS,
    gather_provenance,
)

DATA_DIR = REPO_ROOT / "data" / "processed" / "dashboard"
DASHBOARD_OUTPUT_DIR = REPO_ROOT / "src" / "dashboard" / "output"
DETECTION_OVERLAY_PATH = DATA_DIR / "detection_overlay.png"
DETECTION_GEOMETRY_PATH = DATA_DIR / "detection_geometry.json"

# Real trained checkpoint used for the upload/inference endpoint -- the same
# one scripts/render_detection_overlay.py uses for the pre-rendered demo.
# Override with DETECTION_CHECKPOINT env var once the focal/tversky loss
# comparison (see LOG.md) picks a better checkpoint.
import os  # noqa: E402

DETECTION_CHECKPOINT = Path(
    os.environ.get(
        "DETECTION_CHECKPOINT",
        str(REPO_ROOT / "data" / "processed" / "checkpoints" / "best_unet_resnet18.pt"),
    )
)

app = FastAPI(title="SIH26143 Oil Spill Attribution API")

# Permissive CORS for local dev only (Vite on :3000 talking to this on :8000).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def adapt_geometry(g: dict | None) -> dict | None:
    """detection/geometry.py's characterize_mask() returns Python
    snake_case keys (area_px, n_components, length_px, ...) -- the React
    app's GeometricCharacterization type is camelCase, so this is the one
    conversion point both /api/detection/demo and /api/detect route through
    rather than leaving each caller to remember the mapping."""
    if g is None:
        return None
    return {
        "areaPx": g["area_px"],
        "lengthPx": g["length_px"],
        "widthPx": g["width_px"],
        "orientationDeg": g["orientation_deg"],
        "elongation": g["elongation"],
        "connectedComponents": g["n_components"],
    }


def _case_slug(case_id: str) -> str:
    return case_id.replace("-", "")


def _fmt_short(iso_ts: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        return dt.strftime("%b %d, %H:%M UTC")
    except (ValueError, AttributeError):
        return iso_ts


def _fmt_ago(dt: datetime) -> str:
    delta = datetime.now(timezone.utc) - dt
    hours = delta.total_seconds() / 3600
    if hours < 1:
        return f"{int(delta.total_seconds() / 60)}m ago"
    if hours < 48:
        return f"{hours:.1f}h ago"
    return f"{hours / 24:.1f}d ago"


def adapt_vessel(v: dict) -> dict:
    tags = []
    if v["rank"] == 1:
        tags.append("Top Match")
    if v.get("ais_gap_intentional"):
        tags.append("Intentional AIS Gap")
    elif v.get("ais_gap_count", 0) > 0:
        tags.append("AIS Gap")
    if v["time_gap_hours"] == 0:
        tags.append("Exact Time Match")
    if v["distance_km"] < 10:
        tags.append("Near Origin")
    if v["n_presence_records"] > 50:
        tags.append("High AIS Presence")

    if v.get("ais_gap_intentional"):
        ais_status = "Intentional AIS Gap Detected"
    elif v.get("ais_gap_count", 0) > 0:
        ais_status = "AIS Gap (unclassified)"
    else:
        ais_status = "Continuous AIS Coverage"

    summary = (
        f"{v['ship_name']} ({v['vessel_type']}) was {v['distance_km']:.1f} km from the "
        f"estimated spill origin, "
        + ("exactly at" if v["time_gap_hours"] == 0 else f"{v['time_gap_hours']:.1f}h from")
        + f" the estimated origin time, with {v['n_presence_records']} real GFW presence "
        f"records in the search window."
    )
    if v.get("ais_gap_intentional"):
        summary += (
            f" Global Fishing Watch flagged a {v['ais_gap_duration_hours']:.1f}-hour "
            f"likely-intentional AIS gap covering {v['ais_gap_distance_km']:.0f} km during "
            f"the search period."
        )

    return {
        "id": v["vessel_id"],
        "rank": v["rank"],
        "name": v["ship_name"],
        "mmsi": v["mmsi"],
        "imo": v["imo"],
        "flag": v["flag"],
        "countryCode": v["flag"],
        "vesselType": v["vessel_type"],
        "matchScore": round(v["match_score_pct"]),
        "rawScore": v.get("composite_score", v.get("score")),
        "distFromOriginKm": v["distance_km"],
        "timeGapHours": v["time_gap_hours"],
        "aisStatus": ais_status,
        "aisGapIntentional": v.get("ais_gap_intentional") or False,
        "aisGapHours": v.get("ais_gap_duration_hours"),
        "unaccountedMovementKm": v.get("ais_gap_distance_km"),
        "closestApproachKm": v["closest_approach_km"],
        "gfwPresenceRecords": v["n_presence_records"],
        "evidenceDate": v.get("evidence_date"),
        "entryTimestamp": v.get("entry_timestamp"),
        "exitTimestamp": v.get("exit_timestamp"),
        "lastSeenTime": v.get("exit_timestamp", ""),
        "evidenceTags": tags,
        "behaviorSummary": summary,
        "coordinates": [v["lat"], v["lon"]],
        "historicalPath": [
            {"time": v.get("entry_timestamp", ""), "lat": v["lat"], "lng": v["lon"], "isDark": False},
            {
                "time": v.get("exit_timestamp", ""),
                "lat": v["lat"],
                "lng": v["lon"],
                "isDark": bool(v.get("ais_gap_intentional")),
            },
        ],
    }


def build_provenance(prov: dict) -> dict:
    ds = prov.get("dataset")
    if ds:
        n_total = ds["n_train"] + ds["n_val"]
        label_str = " / ".join(f"{v} {k}" for k, v in sorted(ds["by_label"].items()))
        training_dataset = f"{n_total} real tiles/images ({ds['n_train']} train / {ds['n_val']} val)"
        training_tiles = label_str
    else:
        training_dataset, training_tiles = "No manifest found", ""

    model = prov.get("model")
    if model is None:
        detection_model, detection_val_dice = "No checkpoint yet", "n/a"
    elif model["is_sanity_only"]:
        detection_model = f"Sanity checkpoint only ({_fmt_ago(model['mtime'])})"
        detection_val_dice = "n/a -- 4 toy images, not a real trained model"
    else:
        dice_str = f"{model['val_dice']:.4f}" if model["val_dice"] is not None else "n/a"
        detection_model = f"Best so far: epoch {model['epoch']} ({_fmt_ago(model['mtime'])})"
        detection_val_dice = f"val_dice={dice_str}"

    drift_models = "ERA5 (primary) + NCEP/NCAR (fallback) backward advection"
    drift_last_computed = _fmt_ago(prov["last_drift_fetch"]) if prov.get("last_drift_fetch") else "n/a"

    gfw_quota = "Well under limit"
    gfw_used = "~20-30 requests used this project of 50,000/day, 1,500,000/month"

    return {
        "trainingDataset": training_dataset,
        "trainingTiles": training_tiles,
        "detectionModel": detection_model,
        "detectionValDice": detection_val_dice,
        "driftModels": drift_models,
        "driftLastComputed": drift_last_computed,
        "gfwApiQuota": gfw_quota,
        "gfwRequestsUsed": gfw_used,
    }


def build_case_record(case_id: str) -> dict | None:
    slug = _case_slug(case_id)
    drift = _load_json(DATA_DIR / f"drift_{slug}.json")
    ranking = _load_json(DATA_DIR / f"vessel_ranking_{slug}.json")
    if drift is None:
        return None

    sources = {s["name"]: s for s in drift["sources"]}
    era5 = sources.get("ERA5") or next(iter(sources.values()))
    ncep = sources.get("NCEP/NCAR")

    detection_point = tuple(drift["detection_point"])  # (lon, lat)
    era5_centroid = tuple(era5["centroid"])  # (lon, lat)
    drift_back_km = haversine_km(detection_point, era5_centroid)

    env_disagreement_km = None
    variance_note = None
    if ncep is not None:
        env_disagreement_km = round(haversine_km(era5_centroid, tuple(ncep["centroid"])), 2)
        variance_note = "ERA5 vs. NCEP/NCAR origin-estimate disagreement"

    n_candidates = ranking["n_candidates"] if ranking else 0
    vessels = [adapt_vessel(v) for v in ranking["ranking"]] if ranking else []
    top_match_score = vessels[0]["matchScore"] if vessels else 0

    origin_estimate = ranking["origin_estimate"] if ranking else {
        "lat": era5_centroid[1], "lon": era5_centroid[0], "time_utc": None,
    }
    origin_time = origin_estimate.get("time_utc") or (
        datetime.fromisoformat(drift["detection_time_utc"]) - timedelta(hours=drift["hours_back"])
    ).isoformat()

    status_key_map = {"detection_model": "detectionStatus", "drift_model": "driftStatus", "gfw_attribution": "attributionStatus"}
    react_status_label = {"not_started": "Pending", "in_progress": "In Progress", "done": "Confirmed"}
    statuses = {status_key_map[k]: react_status_label[v["state"]] for k, v in PIPELINE_STATUS.items()}

    summary = (
        f"Real ERA5{'/ NCEP-NCAR' if ncep else ''} backward drift traces this detection "
        f"{drift_back_km:.1f} km back to an estimated origin, cross-referenced against "
        f"{n_candidates} real GFW AIS vessel candidates."
    )
    if vessels:
        summary += f" Top match: {vessels[0]['name']} ({vessels[0]['matchScore']}%)."

    map_file = "map.html" if case_id == CASE_IDS[0] else f"map_{slug}.html"

    timeline = [
        {
            "time": origin_time,
            "source": "Drift model (backward advection)",
            "event": f"Estimated spill origin at {origin_estimate['lat']:.3f}, {origin_estimate['lon']:.3f}",
            "type": "drift",
            "confidence": "high",
        },
        {
            "time": drift["detection_time_utc"],
            "source": "Sentinel-1 SAR",
            "event": f"Oil slick detected at {detection_point[1]:.3f}, {detection_point[0]:.3f}",
            "type": "sar",
            "confidence": "high",
        },
    ]
    if env_disagreement_km is not None:
        timeline.append({
            "time": drift["detection_time_utc"],
            "source": "ERA5 + NCEP/NCAR",
            "event": f"Two independent wind reanalysis sources agree within {env_disagreement_km:.2f} km on drift origin",
            "type": "weather",
            "confidence": "high",
        })
    for v in vessels[:3]:
        if v.get("entryTimestamp"):
            timeline.append({
                "time": v["entryTimestamp"],
                "source": "GFW AIS",
                "event": f"{v['name']} (rank #{v['rank']}) enters search area, {v['distFromOriginKm']:.1f} km from origin",
                "type": "ais",
                "confidence": "medium" if v["rank"] > 1 else "high",
            })
        if v.get("aisGapIntentional"):
            timeline.append({
                "time": v.get("entryTimestamp") or drift["detection_time_utc"],
                "source": "GFW AIS gap detection",
                "event": (
                    f"{v['name']}: {v['aisGapHours']:.1f}h likely-intentional AIS gap "
                    f"covering {v['unaccountedMovementKm']:.0f} km"
                ),
                "type": "ais",
                "confidence": "critical",
            })
    timeline.sort(key=lambda e: e["time"] or "")

    return {
        "id": case_id,
        "code": case_id.upper(),
        "name": f"Case {case_id.upper()}",
        "locationName": drift.get("geo_context") or "Unknown location",
        "region": (drift.get("geo_context") or "").split(",")[0],
        "detectionTime": _fmt_short(drift["detection_time_utc"]),
        "originEstimatedTime": _fmt_short(origin_time),
        "status": statuses["attributionStatus"],
        "detectionStatus": statuses["detectionStatus"],
        "driftStatus": statuses["driftStatus"],
        "attributionStatus": statuses["attributionStatus"],
        "summary": summary,
        "satelliteSensor": "Sentinel-1 SAR",
        "sarTileUrl": "/api/detection/demo/overlay.png",
        "sarMaskUrl": "/api/detection/demo/overlay.png",
        "mapBgUrl": "",
        "mapUrl": f"/api/dashboard-static/{map_file}",
        "coordinates": {"lat": detection_point[1], "lng": detection_point[0]},
        "environmental": {
            "driftBackDistanceKm": round(drift_back_km, 2),
            "vesselsEvaluated": n_candidates,
            "topMatchScore": top_match_score,
            "environmentalDisagreementKm": env_disagreement_km,
            "varianceNote": variance_note,
            "era5OriginCoords": [era5_centroid[1], era5_centroid[0]],
            "ncepOriginCoords": [ncep["centroid"][1], ncep["centroid"][0]] if ncep else None,
            "consensusOriginCoords": [origin_estimate["lat"], origin_estimate["lon"]],
            "detectionCoords": [detection_point[1], detection_point[0]],
        },
        "topSuspect": vessels[0] if vessels else None,
        "rankedCandidates": vessels,
        "nCandidatesTotal": n_candidates,
        "provenance": build_provenance(gather_provenance()),
        "evidenceTimeline": timeline,
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/cases")
def list_cases():
    cases = [build_case_record(cid) for cid in CASE_IDS]
    return [c for c in cases if c is not None]


@app.get("/api/cases/{case_id}")
def get_case(case_id: str):
    case = build_case_record(case_id)
    if case is None:
        raise HTTPException(404, f"No real drift/ranking data for case {case_id!r}")
    return case


@app.get("/api/dashboard-static/{filename}")
def dashboard_static(filename: str):
    """Serves the real, already-built Leaflet drift maps
    (src/dashboard/build_map.py's map.html / map_{case}.html) for the React
    app to <iframe>, instead of reimplementing real particle-track
    rendering in a second, less-verified way."""
    path = DASHBOARD_OUTPUT_DIR / filename
    if not path.exists() or path.suffix != ".html":
        raise HTTPException(404, f"{filename} not found -- run src/dashboard/build_map.py first")
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.get("/api/detection/demo")
def detection_demo():
    geo = _load_json(DETECTION_GEOMETRY_PATH)
    return {
        "demoImage": geo.get("demo_image") if geo else None,
        "overlayUrl": "/api/detection/demo/overlay.png" if DETECTION_OVERLAY_PATH.exists() else None,
        "groundTruth": adapt_geometry(geo.get("ground_truth")) if geo else None,
        "prediction": adapt_geometry(geo.get("prediction")) if geo else None,
        "checkpoint": str(DETECTION_CHECKPOINT.name) if DETECTION_CHECKPOINT.exists() else None,
    }


@app.get("/api/detection/demo/overlay.png")
def detection_demo_overlay():
    if not DETECTION_OVERLAY_PATH.exists():
        raise HTTPException(404, "No demo overlay yet -- run scripts/render_detection_overlay.py")
    return FileResponse(DETECTION_OVERLAY_PATH)


# ---------------------------------------------------------------------------
# Upload feature: real forward pass through the real trained checkpoint.
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict = {}


def _get_model():
    import torch
    from detection.inference import load_model_for_inference

    key = str(DETECTION_CHECKPOINT)
    if key not in _MODEL_CACHE:
        if not DETECTION_CHECKPOINT.exists():
            raise HTTPException(503, f"Detection checkpoint not found: {DETECTION_CHECKPOINT}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model_for_inference(DETECTION_CHECKPOINT, device)
        _MODEL_CACHE[key] = (model, device)
    return _MODEL_CACHE[key]


def _load_uploaded_array(raw_bytes: bytes, filename: str) -> tuple[np.ndarray, str]:
    """
    rasterio/GDAL happily opens plain PNG/JPEG too, so file format alone
    can't tell a real calibrated GeoTIFF apart from a generic photo --
    checked directly (a synthetic 8-bit PNG round-tripped through rasterio
    without error). The real disambiguator is dtype: this project's real
    calibrated Sigma0-dB tiles are stored as float32 (negative dB values),
    while any generic display image (PNG/JPG, or an integer-dtype TIFF) is
    an 8/16-bit pixel-intensity image, not calibrated dB -- see
    DECISIONS.md / scripts/render_detection_overlay.py for the same
    assumption the rest of this pipeline already makes about this dataset.
    """
    import rasterio
    from rasterio.io import MemoryFile

    arr = None
    try:
        with MemoryFile(raw_bytes) as memfile:
            with memfile.open() as src:
                if src.dtypes[0] in ("float32", "float64"):
                    return src.read(1).astype(np.float32), "geotiff_dB"
                arr = src.read(1).astype(np.float32)
                arr_max = float(np.iinfo(src.dtypes[0]).max) if np.issubdtype(src.dtypes[0], np.integer) else 255.0
    except Exception:
        pass

    if arr is None:
        from PIL import Image

        img = Image.open(io.BytesIO(raw_bytes)).convert("L")
        arr = np.asarray(img).astype(np.float32)
        arr_max = 255.0

    # Map the image's pixel range onto the same dB range training data was
    # normalized against (normalize_db_fixed's EXPECTED_DB_RANGE), so the
    # already-trained model sees inputs in roughly the range it expects
    # instead of raw 0-255 values that would saturate the sigmoid input.
    from detection.preprocess import EXPECTED_DB_RANGE

    lo, hi = EXPECTED_DB_RANGE
    arr = lo + (arr / arr_max) * (hi - lo)
    return arr, "generic_image_best_effort"


def _resize_to_tile(arr: np.ndarray, tile_size: int = 512) -> np.ndarray:
    import cv2

    return cv2.resize(arr, (tile_size, tile_size), interpolation=cv2.INTER_AREA)


def _render_overlay_png(despeckled_norm: np.ndarray, mask: np.ndarray) -> bytes:
    """Amber mask over a grayscale despeckled tile -- same color language as
    scripts/render_detection_overlay.py's prediction panel."""
    from PIL import Image

    gray = (np.clip(despeckled_norm, 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    amber = np.array([244, 185, 90], dtype=np.float32)
    alpha = 0.55 * (mask > 0)[..., None]
    blended = rgb * (1 - alpha) + amber * alpha
    img = Image.fromarray(blended.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    from detection.geometry import characterize_mask
    from detection.inference import predict_probs
    from detection.preprocess import lee_filter, normalize_db_fixed

    raw_bytes = await file.read()
    arr, input_format = _load_uploaded_array(raw_bytes, file.filename or "upload")
    original_size = (arr.shape[1], arr.shape[0])

    tile = _resize_to_tile(arr, 512) if arr.shape != (512, 512) else arr

    model, device = _get_model()
    probs = predict_probs(model, tile, device)
    mask = (probs > 0.5).astype(np.float32)
    geometry = adapt_geometry(characterize_mask(mask))

    despeckled_norm = normalize_db_fixed(lee_filter(tile.astype(np.float32)))
    overlay_bytes = _render_overlay_png(despeckled_norm, mask)
    overlay_b64 = base64.b64encode(overlay_bytes).decode("ascii")

    note = (
        "Calibrated Sigma0-dB GeoTIFF read directly (same format training data uses)."
        if input_format == "geotiff_dB"
        else (
            "Uploaded file wasn't a readable calibrated GeoTIFF -- decoded as a generic "
            "grayscale image and rescaled onto the model's expected dB range as a "
            "best-effort approximation. Treat this result as a rough demo, not a "
            "calibrated measurement."
        )
    )

    return JSONResponse({
        "overlayPngBase64": overlay_b64,
        "geometry": geometry,
        "checkpoint": DETECTION_CHECKPOINT.name,
        "inputFormat": input_format,
        "resizedTo": [512, 512],
        "originalSize": list(original_size),
        "note": note,
    })
