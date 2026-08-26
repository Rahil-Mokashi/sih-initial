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
from detection.inference import load_model_for_inference, predict_mask  # noqa: E402

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


def tile_metrics(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, bool]:
    """Returns (iou, dice, has_oil) for one tile. IoU/Dice defined as 1.0
    when both pred and gt are empty (correct rejection), not NaN/0 --
    otherwise a majority-empty test set would understate real performance
    on the no-oil/lookalike classes, which are SUPPOSED to predict empty."""
    pred_b, gt_b = pred > 0.5, gt > 0
    intersection = np.logical_and(pred_b, gt_b).sum()
    union = np.logical_or(pred_b, gt_b).sum()
    has_oil = bool(gt_b.any())
    if union == 0:
        return 1.0, 1.0, has_oil
    iou = intersection / union
    dice = 2 * intersection / (pred_b.sum() + gt_b.sum())
    return float(iou), float(dice), has_oil


def evaluate_image(model, device, image_path: Path, mask_path: Path) -> list[dict]:
    results = []
    with rasterio.open(image_path) as img_src, rasterio.open(mask_path) as mask_src:
        h, w = img_src.height, img_src.width
        for y in range(0, h - TILE_SIZE + 1, TILE_SIZE):
            for x in range(0, w - TILE_SIZE + 1, TILE_SIZE):
                window = Window(x, y, TILE_SIZE, TILE_SIZE)
                image_tile = img_src.read(1, window=window).astype(np.float32)
                gt_tile = mask_src.read(1, window=window).astype(np.float32)
                pred_tile = predict_mask(model, image_tile, device)
                iou, dice, has_oil = tile_metrics(pred_tile, gt_tile)
                results.append({"iou": iou, "dice": dice, "has_oil": has_oil})
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

    summary = {"checkpoint": str(args.checkpoint), "tile_size": TILE_SIZE, "per_class": {}}
    overall_iou, overall_dice = [], []
    print("\n=== results ===")
    for label, results in all_results.items():
        ious = [r["iou"] for r in results]
        dices = [r["dice"] for r in results]
        oil_results = [r for r in results if r["has_oil"]]
        oil_ious = [r["iou"] for r in oil_results]
        summary["per_class"][label] = {
            "n_tiles": len(results),
            "mean_iou": float(np.mean(ious)) if ious else None,
            "mean_dice": float(np.mean(dices)) if dices else None,
            "n_tiles_with_oil": len(oil_results),
            "mean_iou_oil_tiles_only": float(np.mean(oil_ious)) if oil_ious else None,
        }
        print(f"{label}: {len(results)} tiles, mean IoU={np.mean(ious):.4f}, mean Dice={np.mean(dices):.4f} "
              f"({len(oil_results)} tiles actually contain oil, mean IoU on those={np.mean(oil_ious) if oil_ious else float('nan'):.4f})")
        overall_iou.extend(ious)
        overall_dice.extend(dices)

    summary["overall"] = {
        "n_tiles": len(overall_iou),
        "mean_iou": float(np.mean(overall_iou)),
        "mean_dice": float(np.mean(overall_dice)),
    }
    print(f"\nOVERALL: {len(overall_iou)} tiles, mean IoU={np.mean(overall_iou):.4f}, mean Dice={np.mean(overall_dice):.4f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
