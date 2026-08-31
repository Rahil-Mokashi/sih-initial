"""
Generate static demo assets from the epoch-39 baseline checkpoint for a
no-inference, no-crash-possible UI demo (see web/README.md).

Uses the ALREADY-CACHED probability maps from scripts/evaluate.py's earlier
run against latest_unet_resnet18_epoch39_backup.pt
(data/processed/eval_cache/latest_unet_resnet18_epoch39_backup/*.npz) --
no model loading, no inference, no GPU needed, so this can't crash on
anything but a missing cache file.

Picks 6 oil-containing validation images (by highest real GT oil fraction
in their best tile, for visibly clear slicks), 2 lookalike, and 2 no_oil
(by highest predicted probability, so the shown tile is the model's most
"interesting" behavior on a genuinely oil-free scene, not a blank crop).

For each: raw.png (Band 1, normalized), gt_overlay.png, pred_overlay.png
(threshold 0.35, the real best threshold from docs/metric_audit.md), and
metrics.json (IoU/precision/recall/pred-oil%/true-oil%, from
src/detection/metrics.py -- same unit-tested implementation used
throughout Phase 0, not a new computation). Plus one summary.json with the
overall baseline numbers.

Usage:
    venv\\Scripts\\python.exe scripts\\generate_demo_assets.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from detection.metrics import tile_metrics  # noqa: E402
from detection.preprocess import lee_filter, normalize_db_fixed  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
VAL_MANIFEST = REPO_ROOT / "data" / "processed" / "val_manifest.csv"
CACHE_DIR = REPO_ROOT / "data" / "processed" / "eval_cache" / "latest_unet_resnet18_epoch39_backup"
OUT_DIR = REPO_ROOT / "web" / "public" / "demo"
TILE_SIZE = 512
THRESHOLD = 0.35
N_OIL = 6
N_NO_OIL = 2
N_LOOKALIKE = 2


def load_manifest() -> list[dict]:
    return list(csv.DictReader(open(VAL_MANIFEST, newline="")))


def select_images() -> list[dict]:
    by_label: dict[str, list[dict]] = {"oil": [], "no_oil": [], "lookalike": []}
    for row in load_manifest():
        stem = Path(row["image_path"]).stem
        npz_path = CACHE_DIR / f"{stem}.npz"
        if not npz_path.exists():
            continue
        data = np.load(npz_path)
        gt, probs = data["gt"], data["probs"].astype(np.float32)
        n_tiles = gt.shape[0]
        oil_fracs = gt.reshape(n_tiles, -1).mean(axis=1)
        pred_means = probs.reshape(n_tiles, -1).mean(axis=1)
        if row["label"] == "oil":
            tile_idx = int(np.argmax(oil_fracs))
            sort_key = float(oil_fracs[tile_idx])
        else:
            tile_idx = int(np.argmax(pred_means))
            sort_key = float(pred_means[tile_idx])
        by_label[row["label"]].append({**row, "stem": stem, "tile_idx": tile_idx, "sort_key": sort_key})

    for label in by_label:
        by_label[label].sort(key=lambda r: -r["sort_key"])

    selected = by_label["oil"][:N_OIL] + by_label["lookalike"][:N_LOOKALIKE] + by_label["no_oil"][:N_NO_OIL]
    return selected


def tile_origin(tile_idx: int, image_w: int) -> tuple[int, int]:
    """ZenodoTileDataset/evaluate_test_set iterate tiles row-major over a
    non-overlapping TILE_SIZE grid -- reproduce the same (y, x) origin math
    to read the exact same tile evaluate.py cached the probs for."""
    tiles_per_row = image_w // TILE_SIZE
    row, col = divmod(tile_idx, tiles_per_row)
    return row * TILE_SIZE, col * TILE_SIZE


def make_overlay(base_rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    out = base_rgb.copy().astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * color_arr
    return out.astype(np.uint8)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = select_images()
    print(f"selected {len(selected)} images: "
          f"{sum(1 for s in selected if s['label']=='oil')} oil, "
          f"{sum(1 for s in selected if s['label']=='lookalike')} lookalike, "
          f"{sum(1 for s in selected if s['label']=='no_oil')} no_oil")

    per_image_summaries = []
    for i, row in enumerate(selected):
        image_id = f"demo_{i:02d}_{row['label']}_{row['stem']}"
        with rasterio.open(row["image_path"]) as src:
            image_w = src.width
            y, x = tile_origin(row["tile_idx"], image_w)
            band1 = src.read(1, window=Window(x, y, TILE_SIZE, TILE_SIZE)).astype(np.float32)

        cached = np.load(CACHE_DIR / f"{row['stem']}.npz")
        probs = cached["probs"][row["tile_idx"]].astype(np.float32)
        gt = cached["gt"][row["tile_idx"]]

        m = tile_metrics(probs, gt, THRESHOLD)

        norm = normalize_db_fixed(lee_filter(band1))
        gray = (norm * 255).astype(np.uint8)
        rgb = np.stack([gray] * 3, axis=-1)

        Image.fromarray(rgb).save(OUT_DIR / f"{image_id}_raw.png")
        gt_overlay = make_overlay(rgb, gt > 0, color=(220, 40, 40))
        Image.fromarray(gt_overlay).save(OUT_DIR / f"{image_id}_gt.png")
        pred_overlay = make_overlay(rgb, probs > THRESHOLD, color=(40, 140, 220))
        Image.fromarray(pred_overlay).save(OUT_DIR / f"{image_id}_pred.png")

        metrics_out = {
            "image_id": image_id,
            "label": row["label"],
            "source_image": row["image_path"],
            "tile_index": row["tile_idx"],
            "threshold": THRESHOLD,
            "iou": round(m["iou"], 4),
            "dice": round(m["dice"], 4),
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "predicted_oil_pct": round(100 * m["pred_positive_fraction"], 2),
            "true_oil_pct": round(100 * m["gt_positive_fraction"], 2),
        }
        (OUT_DIR / f"{image_id}_metrics.json").write_text(json.dumps(metrics_out, indent=2))
        per_image_summaries.append(metrics_out)
        print(f"  [{i+1}/{len(selected)}] {image_id}: IoU={m['iou']:.3f} pred_oil%={metrics_out['predicted_oil_pct']} true_oil%={metrics_out['true_oil_pct']}")

    summary = {
        "checkpoint": "latest_unet_resnet18_epoch39_backup.pt (epoch 39)",
        "threshold": THRESHOLD,
        "baseline_oil_tiles_only": {
            "iou": 0.1074,
            "dice": 0.1691,
            "precision": 0.139,
            "recall": 0.476,
            "note": "from docs/metric_audit.md's independent re-verification of the real oil-tiles-only IoU comparison, val set",
        },
        "images": per_image_summaries,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT_DIR / 'summary.json'} and {len(selected)*4} per-image files to {OUT_DIR}")


if __name__ == "__main__":
    main()
