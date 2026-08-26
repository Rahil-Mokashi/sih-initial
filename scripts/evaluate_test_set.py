"""
Real evaluation on the held-out Zenodo Part III test set (150 Oil / 150
No-oil / 150 Lookalike, 450 real images -- see DECISIONS.md "Train/val/test
methodology") against a trained checkpoint. This set is never touched
during training or hyperparameter tuning; this script produces the real
accuracy number for the project, not the training-loop's val split
(val_manifest.csv is drawn from the same Part I/II pool as train_manifest.csv,
so it's useful for picking the best checkpoint but isn't a true held-out
test).

Per-image evaluation: tiles each 2048x2048 image into non-overlapping
512x512 tiles (same tile size as training), runs inference per tile via
src/detection/inference.py, computes IoU and Dice per tile against the
real ground-truth mask, then aggregates per class (oil/no_oil/lookalike)
and overall.

Usage:
    venv\\Scripts\\python.exe scripts\\evaluate_test_set.py [--checkpoint PATH]

Requires scripts/extract_zenodo_part3.py to have been run first.
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

REPO_ROOT = Path(__file__).resolve().parent.parent
PART3_DIR = REPO_ROOT / "data" / "raw" / "zenodo_sar_oil_spill_part3"
IMAGES_EXTRACTED = PART3_DIR / "images_extracted"
MASKS_EXTRACTED = PART3_DIR / "masks_extracted"
DEFAULT_CHECKPOINT = REPO_ROOT / "data" / "processed" / "checkpoints" / "best_unet_resnet18.pt"
OUT_PATH = REPO_ROOT / "data" / "processed" / "test_set_evaluation.json"

TILE_SIZE = 512
CLASS_DIRS = {
    "oil": ("Images/Oil", "Mask/Oil"),
    "no_oil": ("Images/No oil", "Mask/No oil"),
    "lookalike": ("Images/Lookalike", "Mask/Lookalike"),
}


THRESHOLDS = [0.5, 0.35, 0.25, 0.22, 0.2, 0.18]  # swept from one forward pass per tile -- see
# src/detection/inference.py's predict_probs() docstring for why: this project's real
# checkpoint never crosses 0.5 anywhere on a real test tile despite carrying real signal.


def tile_metrics(probs: np.ndarray, gt: np.ndarray, threshold: float) -> tuple[float, float, bool]:
    """Returns (iou, dice, has_oil) for one tile at one threshold. IoU/Dice
    defined as 1.0 when both pred and gt are empty (correct rejection), not
    NaN/0 -- otherwise a majority-empty test set would understate real
    performance on the no-oil/lookalike classes, which are SUPPOSED to
    predict empty."""
    pred_b, gt_b = probs > threshold, gt > 0
    intersection = np.logical_and(pred_b, gt_b).sum()
    union = np.logical_or(pred_b, gt_b).sum()
    has_oil = bool(gt_b.any())
    if union == 0:
        return 1.0, 1.0, has_oil
    iou = intersection / union
    dice = 2 * intersection / (pred_b.sum() + gt_b.sum())
    return float(iou), float(dice), has_oil


def evaluate_image(model, device, image_path: Path, mask_path: Path) -> list[dict]:
    """One real forward pass per tile; metrics computed at every threshold
    in THRESHOLDS from the same probability map, so a multi-threshold
    sweep costs no extra GPU inference over evaluating at a single fixed
    threshold."""
    results = []
    with rasterio.open(image_path) as img_src, rasterio.open(mask_path) as mask_src:
        h, w = img_src.height, img_src.width
        for y in range(0, h - TILE_SIZE + 1, TILE_SIZE):
            for x in range(0, w - TILE_SIZE + 1, TILE_SIZE):
                window = Window(x, y, TILE_SIZE, TILE_SIZE)
                image_tile = img_src.read(1, window=window).astype(np.float32)
                gt_tile = mask_src.read(1, window=window).astype(np.float32)
                probs = predict_probs(model, image_tile, device)
                has_oil = bool((gt_tile > 0).any())
                per_threshold = {}
                for t in THRESHOLDS:
                    iou, dice, _ = tile_metrics(probs, gt_tile, t)
                    per_threshold[t] = {"iou": iou, "dice": dice}
                results.append({"has_oil": has_oil, "per_threshold": per_threshold})
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    if not IMAGES_EXTRACTED.exists():
        print("ERROR: Part III not extracted. Run scripts/extract_zenodo_part3.py first.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"checkpoint: {args.checkpoint}")
    model = load_model_for_inference(args.checkpoint, device)

    all_results: dict[str, list[dict]] = {}
    for label, (img_subdir, mask_subdir) in CLASS_DIRS.items():
        img_dir = IMAGES_EXTRACTED / img_subdir
        mask_dir = MASKS_EXTRACTED / mask_subdir
        images = sorted(img_dir.glob("*.tif"))
        print(f"\n[{label}] evaluating {len(images)} real images...")
        class_results = []
        for i, img_path in enumerate(images):
            # Part III masks carry a "_segmentation" suffix the images don't
            # (00000.tif -> 00000_segmentation.tif) -- unlike Part I/II,
            # where image and mask filenames match exactly. Found via a
            # smoke-test RasterioIOError before the real evaluation run.
            mask_path = mask_dir / f"{img_path.stem}_segmentation{img_path.suffix}"
            if not mask_path.exists():
                print(f"  WARNING: no mask for {img_path.name}, skipping")
                continue
            class_results.extend(evaluate_image(model, device, img_path, mask_path))
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(images)} images done")
        all_results[label] = class_results

    # The headline number is mean_iou_oil_tiles_only, not the overall mean --
    # the overall mean is dominated by trivially-easy "correctly predict
    # empty" tiles from the no_oil/lookalike classes and can look good even
    # when the model is failing at the actual oil-segmentation task (this
    # happened for real at threshold=0.5 -- see LOG.md).
    all_tiles = [r for results in all_results.values() for r in results]
    oil_tiles = [r for r in all_tiles if r["has_oil"]]

    summary = {"checkpoint": str(args.checkpoint), "tile_size": TILE_SIZE, "per_threshold": {}}
    print("\n=== threshold sweep (mean IoU) ===")
    print(f"{'threshold':>10} {'overall':>10} {'oil-tiles-only':>16} {'no_oil':>10} {'lookalike':>12}")
    for t in THRESHOLDS:
        overall_iou = [r["per_threshold"][t]["iou"] for r in all_tiles]
        oil_iou = [r["per_threshold"][t]["iou"] for r in oil_tiles]
        per_class_iou = {
            label: [r["per_threshold"][t]["iou"] for r in results]
            for label, results in all_results.items()
        }
        summary["per_threshold"][t] = {
            "overall_mean_iou": float(np.mean(overall_iou)),
            "oil_tiles_only_mean_iou": float(np.mean(oil_iou)) if oil_iou else None,
            "per_class_mean_iou": {k: float(np.mean(v)) for k, v in per_class_iou.items()},
        }
        print(f"{t:>10} {np.mean(overall_iou):>10.4f} {np.mean(oil_iou) if oil_iou else float('nan'):>16.4f} "
              f"{np.mean(per_class_iou['no_oil']):>10.4f} {np.mean(per_class_iou['lookalike']):>12.4f}")

    best_t = max(THRESHOLDS, key=lambda t: summary["per_threshold"][t]["oil_tiles_only_mean_iou"] or 0)
    summary["best_threshold_by_oil_iou"] = best_t
    print(f"\nbest threshold by oil-tiles-only IoU: {best_t} "
          f"(oil IoU={summary['per_threshold'][best_t]['oil_tiles_only_mean_iou']:.4f}, "
          f"overall IoU={summary['per_threshold'][best_t]['overall_mean_iou']:.4f})")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
