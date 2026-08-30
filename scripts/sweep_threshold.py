"""
Phase 0.4 of the metric audit (docs/metric_audit.md): sweep the binarization
threshold 0.05 -> 0.50 in steps of 0.025 against a checkpoint's CACHED
probability maps (from scripts/evaluate.py -- run that first for a given
checkpoint; this script does zero model inference, only reads .npz files),
so a full threshold sweep across 19 thresholds costs no extra GPU/CPU time
over the single evaluate.py pass that produced the cache.

Emits a table (oil-tiles-only IoU/Dice/Precision/Recall per threshold, plus
the global pixel-accumulated IoU/Dice for the same tile set) and a plot to
reports/threshold_sweep_<tag>.png.

Usage:
    venv\\Scripts\\python.exe scripts\\sweep_threshold.py --checkpoint data/processed/checkpoints/latest_unet_resnet18_epoch39_backup.pt
    venv\\Scripts\\python.exe scripts\\sweep_threshold.py --cache-dir data/processed/eval_cache/trial_tversky__best_unet_resnet18 --tag trial_tversky
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from detection.metrics import aggregate_global, tile_metrics  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import CACHE_ROOT, checkpoint_tag, load_cached_tiles  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
THRESHOLDS = [round(0.05 + 0.025 * i, 3) for i in range(19)]  # 0.05 .. 0.50 inclusive


def sweep(tiles: list[tuple[np.ndarray, np.ndarray]]) -> list[dict]:
    oil_tiles = [(probs, gt) for probs, gt in tiles if (gt > 0).any()]
    rows = []
    for t in THRESHOLDS:
        per_tile = [tile_metrics(probs, gt, t) for probs, gt in oil_tiles]
        global_metrics = aggregate_global(oil_tiles, t)
        rows.append({
            "threshold": t,
            "per_tile_mean_iou": float(np.mean([m["iou"] for m in per_tile])) if per_tile else None,
            "per_tile_mean_dice": float(np.mean([m["dice"] for m in per_tile])) if per_tile else None,
            "per_tile_mean_precision": float(np.mean([m["precision"] for m in per_tile])) if per_tile else None,
            "per_tile_mean_recall": float(np.mean([m["recall"] for m in per_tile])) if per_tile else None,
            "global_iou": global_metrics["iou"],
            "global_dice": global_metrics["dice"],
        })
    return rows


def plot(rows: list[dict], out_path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thresholds = [r["threshold"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, [r["per_tile_mean_iou"] for r in rows], marker="o", label="IoU (per-tile mean)")
    ax.plot(thresholds, [r["per_tile_mean_dice"] for r in rows], marker="o", label="Dice (per-tile mean)")
    ax.plot(thresholds, [r["per_tile_mean_precision"] for r in rows], marker="o", label="Precision")
    ax.plot(thresholds, [r["per_tile_mean_recall"] for r in rows], marker="o", label="Recall")
    ax.plot(thresholds, [r["global_iou"] for r in rows], marker="x", linestyle="--", label="IoU (global, pixel-accumulated)")
    ax.set_xlabel("threshold")
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=None, help="derive cache dir + tag from this checkpoint path")
    parser.add_argument("--cache-dir", type=Path, default=None, help="explicit cache dir (from a prior scripts/evaluate.py run)")
    parser.add_argument("--tag", type=str, default=None, help="output filename tag; derived from --checkpoint if omitted")
    args = parser.parse_args()

    if args.cache_dir is None:
        if args.checkpoint is None:
            print("ERROR: pass --checkpoint or --cache-dir")
            sys.exit(1)
        tag = checkpoint_tag(args.checkpoint)
        cache_dir = CACHE_ROOT / tag
    else:
        cache_dir = args.cache_dir
        tag = args.tag or cache_dir.name

    if not cache_dir.exists():
        print(f"ERROR: no cache at {cache_dir}. Run scripts/evaluate.py for this checkpoint first.")
        sys.exit(1)

    tiles = load_cached_tiles(cache_dir)
    print(f"loaded {len(tiles)} cached tiles from {cache_dir}")

    rows = sweep(tiles)
    print(f"\n{'threshold':>10} {'IoU(tile)':>10} {'Dice(tile)':>11} {'Precision':>10} {'Recall':>8} {'IoU(global)':>12}")
    for r in rows:
        print(f"{r['threshold']:>10.3f} {r['per_tile_mean_iou']:>10.4f} {r['per_tile_mean_dice']:>11.4f} "
              f"{r['per_tile_mean_precision']:>10.4f} {r['per_tile_mean_recall']:>8.4f} {r['global_iou']:>12.4f}")

    best = max(rows, key=lambda r: r["per_tile_mean_iou"] or 0)
    print(f"\nbest threshold by per-tile-mean oil IoU: {best['threshold']} (IoU={best['per_tile_mean_iou']:.4f}, "
          f"global IoU at same threshold={best['global_iou']:.4f})")

    out_json = cache_dir / "threshold_sweep.json"
    out_json.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out_json}")

    plot(rows, REPORTS_DIR / f"threshold_sweep_{tag}.png", title=f"threshold sweep: {tag}")


if __name__ == "__main__":
    main()
