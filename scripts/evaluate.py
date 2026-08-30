"""
Phase 0.3 of the metric audit (docs/metric_audit.md): a single, oil-class-
focused evaluation entrypoint that any checkpoint can be pointed at.

Distinct from the existing scripts/evaluate_test_set.py (which evaluates
against the held-out Part III test set and is intentionally touched only
once per real methodology, per DECISIONS.md "Train/val/test methodology")
-- this script targets the VAL set by default, exactly like
scripts/compare_checkpoints_on_val.py, and adds two things neither existing
script has: (1) raw probability maps cached to disk per image, so a
downstream threshold sweep (scripts/sweep_threshold.py) costs zero extra
GPU/CPU inference, and (2) precision/recall alongside IoU/Dice, plus both
predicted-positive and ground-truth-positive pixel fractions, so a
collapsed ("predict everything" or "predict nothing") checkpoint is visible
directly rather than inferred from a suspiciously low/flat Dice number
(see LOG.md's trial_posweight_sched kill -- exactly this ambiguity).

Reports both a per-tile mean (average of each tile's own IoU/Dice/etc, the
convention used everywhere else in this project so far) and a global,
pixel-accumulated version (src/detection/metrics.py's aggregate_global) --
per-tile averaging on a sparse-oil dataset implicitly weights a tile with 1
oil pixel the same as a 90%-oil tile, so the global number is the
cross-check for whether that's distorting the per-tile one.

Usage:
    venv\\Scripts\\python.exe scripts\\evaluate.py --checkpoint data/processed/checkpoints/latest_unet_resnet18_epoch39_backup.pt
    venv\\Scripts\\python.exe scripts\\evaluate.py --checkpoint data/processed/checkpoints/trial_tversky/best_unet_resnet18.pt --threshold 0.35
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from detection.inference import load_model_for_inference, predict_probs  # noqa: E402
from detection.metrics import aggregate_global, tile_metrics  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
VAL_MANIFEST = REPO_ROOT / "data" / "processed" / "val_manifest.csv"
CHECKPOINT_ROOT = REPO_ROOT / "data" / "processed" / "checkpoints"
CACHE_ROOT = REPO_ROOT / "data" / "processed" / "eval_cache"
TILE_SIZE = 512


def checkpoint_tag(checkpoint_path: Path) -> str:
    """A filesystem-safe, unique-per-checkpoint tag, used both as the cache
    directory name and the JSON report filename -- needed because every
    trial directory reuses the same filename ("best_unet_resnet18.pt"), so
    the bare stem alone would collide across trials."""
    try:
        rel = checkpoint_path.resolve().relative_to(CHECKPOINT_ROOT.resolve())
    except ValueError:
        rel = Path(checkpoint_path.stem)
    return str(rel).replace("\\", "__").replace("/", "__").rsplit(".", 1)[0]


def load_manifest(path: Path) -> list[tuple[Path, Path, str]]:
    """Returns (image_path, mask_path, label) -- label is the manifest's own
    "oil" | "no_oil" | "lookalike" source-image category (see
    src/detection/dataset.py's ImageMaskPair), used to break metrics out
    per class, not just per has-oil-pixels tile."""
    import csv
    rows = list(csv.DictReader(open(path, newline="")))
    return [(Path(r["image_path"]), Path(r["mask_path"]), r["label"]) for r in rows]


def cache_probability_maps(model, device, pairs: list[tuple[Path, Path, str]], channels: tuple[int, ...], cache_dir: Path) -> None:
    """One forward pass per tile, per image; writes one .npz per source image
    (probs stacked (n_tiles, H, W) float16 + gt stacked (n_tiles, H, W) uint8)
    to cache_dir. Skips images whose cache file already exists, so a partial
    or interrupted run can resume without re-doing finished images."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    for i, (image_path, mask_path, _label) in enumerate(pairs):
        out_path = cache_dir / f"{image_path.stem}.npz"
        if out_path.exists():
            continue
        probs_list, gt_list = [], []
        with rasterio.open(image_path) as img_src, rasterio.open(mask_path) as mask_src:
            h, w = img_src.height, img_src.width
            for y in range(0, h - TILE_SIZE + 1, TILE_SIZE):
                for x in range(0, w - TILE_SIZE + 1, TILE_SIZE):
                    window = Window(x, y, TILE_SIZE, TILE_SIZE)
                    image_tile = img_src.read(list(channels), window=window).astype(np.float32)
                    if len(channels) == 1:
                        image_tile = image_tile[0]
                    gt_tile = mask_src.read(1, window=window).astype(np.float32)
                    probs = predict_probs(model, image_tile, device)
                    probs_list.append(probs.astype(np.float16))
                    gt_list.append((gt_tile > 0).astype(np.uint8))
        np.savez_compressed(out_path, probs=np.stack(probs_list), gt=np.stack(gt_list))
        if (i + 1) % 25 == 0:
            print(f"  cached {i + 1}/{len(pairs)} images")


def load_cached_tiles(cache_dir: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """Returns a flat list of (probs, gt) tile pairs across every cached image."""
    tiles = []
    for npz_path in sorted(cache_dir.glob("*.npz")):
        data = np.load(npz_path)
        probs, gt = data["probs"], data["gt"]
        for i in range(probs.shape[0]):
            tiles.append((probs[i].astype(np.float32), gt[i]))
    return tiles


def load_cached_tiles_by_label(cache_dir: Path, pairs: list[tuple[Path, Path, str]]) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Same as load_cached_tiles, but tags each tile with its SOURCE IMAGE's
    manifest label (oil/no_oil/lookalike) -- distinct from a tile's own
    has_oil flag, since e.g. a "lookalike" source image is still expected to
    have an all-empty ground-truth mask (lookalikes are a negative class by
    construction; see DECISIONS.md), so "label" and "has_oil" answer
    different questions (what kind of scene is this vs. does this specific
    512x512 crop happen to contain oil pixels)."""
    label_by_stem = {image_path.stem: label for image_path, _mask_path, label in pairs}
    tiles = []
    for npz_path in sorted(cache_dir.glob("*.npz")):
        label = label_by_stem.get(npz_path.stem, "unknown")
        data = np.load(npz_path)
        probs, gt = data["probs"], data["gt"]
        for i in range(probs.shape[0]):
            tiles.append((probs[i].astype(np.float32), gt[i], label))
    return tiles


def _mean_of(key: str, metrics_list: list[dict]) -> float | None:
    vals = [m[key] for m in metrics_list]
    return float(np.mean(vals)) if vals else None


def report_at_threshold(labeled_tiles: list[tuple[np.ndarray, np.ndarray, str]], threshold: float) -> dict:
    tiles = [(probs, gt) for probs, gt, _label in labeled_tiles]
    per_tile = [tile_metrics(probs, gt, threshold) for probs, gt in tiles]
    oil_tiles = [(probs, gt) for (probs, gt), m in zip(tiles, per_tile) if m["has_oil"]]
    oil_metrics = [m for m in per_tile if m["has_oil"]]

    per_class = {}
    for class_label in ("oil", "no_oil", "lookalike"):
        class_pairs = [(probs, gt) for probs, gt, label in labeled_tiles if label == class_label]
        if not class_pairs:
            per_class[class_label] = None
            continue
        class_metrics = [tile_metrics(probs, gt, threshold) for probs, gt in class_pairs]
        per_class[class_label] = {
            "n_tiles": len(class_pairs),
            "pred_positive_fraction": _mean_of("pred_positive_fraction", class_metrics),
            "gt_positive_fraction": _mean_of("gt_positive_fraction", class_metrics),
            # for no_oil/lookalike source images every tile is GT-empty by construction,
            # so IoU/Dice here just report whether the model correctly predicts empty too
            # (1.0) or hallucinates oil on a clean/lookalike scene (drags toward 0.0) --
            # the false-positive-rate lens B.3 asks for, not oil-segmentation quality.
            "iou": _mean_of("iou", class_metrics),
            "dice": _mean_of("dice", class_metrics),
        }

    return {
        "threshold": threshold,
        "n_tiles_total": len(tiles),
        "n_tiles_with_oil": len(oil_tiles),
        "oil_tiles_per_tile_mean": {
            "iou": _mean_of("iou", oil_metrics),
            "dice": _mean_of("dice", oil_metrics),
            "precision": _mean_of("precision", oil_metrics),
            "recall": _mean_of("recall", oil_metrics),
            "pred_positive_fraction": _mean_of("pred_positive_fraction", oil_metrics),
            "gt_positive_fraction": _mean_of("gt_positive_fraction", oil_metrics),
        },
        "oil_tiles_global": aggregate_global(oil_tiles, threshold) if oil_tiles else None,
        "all_tiles_per_tile_mean": {
            "iou": _mean_of("iou", per_tile),
            "dice": _mean_of("dice", per_tile),
            "precision": _mean_of("precision", per_tile),
            "recall": _mean_of("recall", per_tile),
        },
        "all_tiles_global": aggregate_global(tiles, threshold),
        "per_source_class": per_class,
    }


def print_report(report: dict, checkpoint: Path) -> None:
    otm = report["oil_tiles_per_tile_mean"]
    otg = report["oil_tiles_global"]
    print(f"\n=== {checkpoint} @ threshold={report['threshold']} ===")
    print(f"tiles: {report['n_tiles_total']} total, {report['n_tiles_with_oil']} with real oil")
    print("oil tiles only, per-tile mean:")
    print(f"  IoU={otm['iou']:.4f}  Dice={otm['dice']:.4f}  Precision={otm['precision']:.4f}  Recall={otm['recall']:.4f}")
    print(f"  pred-positive-fraction={otm['pred_positive_fraction']:.4f}  gt-positive-fraction={otm['gt_positive_fraction']:.4f}")
    if otg:
        print(f"oil tiles only, global (pixel-accumulated): IoU={otg['iou']:.4f}  Dice={otg['dice']:.4f}  "
              f"Precision={otg['precision']:.4f}  Recall={otg['recall']:.4f}")
    atg = report["all_tiles_global"]
    print(f"all tiles, global: IoU={atg['iou']:.4f}  Dice={atg['dice']:.4f}  Precision={atg['precision']:.4f}  Recall={atg['recall']:.4f}")
    print("per source-image class (false-positive lens -- pred_positive_fraction should be ~0 for no_oil/lookalike):")
    for class_label, stats in report["per_source_class"].items():
        if stats is None:
            print(f"  {class_label}: no images of this class in the manifest")
        else:
            print(f"  {class_label} (n={stats['n_tiles']} tiles): pred_positive_fraction={stats['pred_positive_fraction']:.4f}  "
                  f"gt_positive_fraction={stats['gt_positive_fraction']:.4f}  IoU={stats['iou']:.4f}  Dice={stats['dice']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=VAL_MANIFEST)
    parser.add_argument("--channels", type=str, default="1", help="comma-separated 1-indexed rasterio bands, e.g. '1' or '1,2'")
    parser.add_argument("--threshold", type=float, default=0.5, help="single-threshold report; use scripts/sweep_threshold.py for a full sweep")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--force-recache", action="store_true")
    args = parser.parse_args()

    channels = tuple(int(c) for c in args.channels.split(","))
    tag = checkpoint_tag(args.checkpoint)
    cache_dir = args.cache_dir or (CACHE_ROOT / tag)

    if args.force_recache and cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"checkpoint: {args.checkpoint} (tag={tag}, channels={channels})")

    pairs = load_manifest(args.manifest)
    print(f"manifest: {len(pairs)} real images ({args.manifest})")

    already_cached = cache_dir.exists() and len(list(cache_dir.glob("*.npz"))) >= len(pairs)
    if already_cached:
        print(f"using existing cache at {cache_dir} ({len(pairs)} images already cached)")
    else:
        model = load_model_for_inference(args.checkpoint, device, in_channels=len(channels))
        print(f"caching probability maps to {cache_dir} ...")
        cache_probability_maps(model, device, pairs, channels, cache_dir)

    labeled_tiles = load_cached_tiles_by_label(cache_dir, pairs)
    report = report_at_threshold(labeled_tiles, args.threshold)
    print_report(report, args.checkpoint)

    out_json = cache_dir / "report.json"
    out_json.write_text(json.dumps({"checkpoint": str(args.checkpoint), "channels": channels, **report}, indent=2))
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
