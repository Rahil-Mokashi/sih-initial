"""
Renders a real SAR tile / ground-truth mask / predicted mask comparison,
using the real trained checkpoint (best_unet_resnet18.pt, from
scripts/train_detection.py -- see LOG.md for its real val_dice and which
epoch). Picks a real Zenodo Part III image (the held-out test set, never
touched during training) with a decent amount of real oil and a 512x512
window that actually contains it, so the demo isn't just blank water and
isn't an image the model may have already seen during training.

Output: data/processed/dashboard/detection_overlay.png,
        data/processed/dashboard/detection_geometry.json

Usage: venv\\Scripts\\python.exe scripts\\render_detection_overlay.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from detection.geometry import characterize_mask  # noqa: E402
from detection.inference import load_model_for_inference, predict_mask  # noqa: E402
from detection.preprocess import lee_filter, normalize_db_fixed  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = REPO_ROOT / "data" / "raw" / "zenodo_sar_oil_spill_part3" / "images_extracted" / "Images" / "Oil"
MASK_DIR = REPO_ROOT / "data" / "raw" / "zenodo_sar_oil_spill_part3" / "masks_extracted" / "Mask" / "Oil"
CHECKPOINT_PATH = REPO_ROOT / "data" / "processed" / "checkpoints" / "best_unet_resnet18.pt"
OUT_PATH = REPO_ROOT / "data" / "processed" / "dashboard" / "detection_overlay.png"
GEOMETRY_OUT_PATH = REPO_ROOT / "data" / "processed" / "dashboard" / "detection_geometry.json"

# No pixel_size_m is passed to characterize_mask() below -- checked directly
# against the Zenodo dataset's own record page (DOI 10.5281/zenodo.8346860)
# and it does not state a Sentinel-1 product type or ground resolution for
# these images, unlike the PANGAEA drift cases (confirmed IW GRDH product
# IDs -- see DECISIONS.md). Reporting pixel-unit geometry only here rather
# than assuming a resolution this dataset never actually documented.

TILE_SIZE = 512
MIN_OIL_FRACTION = 0.05  # pick an image with at least 5% oil for a meaningful demo


def find_demo_image() -> tuple[Path, Path, Window]:
    """Scans a handful of Part III (held-out test set) masks, picks one
    with enough oil, and a tile window covering it. Mask filenames carry
    a "_segmentation" suffix the images don't (see
    scripts/evaluate_test_set.py, which hit this as a real bug first)."""
    for mask_path in sorted(MASK_DIR.glob("*.tif"))[:200]:
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
        frac = (mask > 0).mean()
        if frac < MIN_OIL_FRACTION:
            continue
        ys, xs = np.where(mask > 0)
        cy, cx = int(ys.mean()), int(xs.mean())
        h, w = mask.shape
        y0 = min(max(cy - TILE_SIZE // 2, 0), h - TILE_SIZE)
        x0 = min(max(cx - TILE_SIZE // 2, 0), w - TILE_SIZE)
        image_name = mask_path.name.replace("_segmentation", "")
        image_path = IMG_DIR / image_name
        if image_path.exists():
            return image_path, mask_path, Window(x0, y0, TILE_SIZE, TILE_SIZE)
    raise RuntimeError(f"No image with oil fraction >= {MIN_OIL_FRACTION} found in the first 200 masks scanned.")


def main() -> None:
    if not CHECKPOINT_PATH.exists():
        print(f"ERROR: {CHECKPOINT_PATH} not found. Run scripts/train_detection.py first.")
        sys.exit(1)

    image_path, mask_path, window = find_demo_image()
    print(f"demo tile: {image_path.name}, window={window}")

    with rasterio.open(image_path) as src:
        raw_tile = src.read(1, window=window).astype(np.float32)
    with rasterio.open(mask_path) as src:
        gt_tile = src.read(1, window=window).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_for_inference(CHECKPOINT_PATH, device)
    pred_tile = predict_mask(model, raw_tile, device)

    gt_geometry = characterize_mask(gt_tile)
    pred_geometry = characterize_mask(pred_tile)
    print(f"real ground-truth geometry: {gt_geometry}")
    print(f"model prediction geometry: {pred_geometry}")
    GEOMETRY_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    GEOMETRY_OUT_PATH.write_text(json.dumps(
        {"demo_image": image_path.name, "ground_truth": gt_geometry, "prediction": pred_geometry}, indent=2
    ))
    print(f"wrote {GEOMETRY_OUT_PATH}")

    despeckled_display = normalize_db_fixed(lee_filter(raw_tile))

    # Dark navy figure styling to match the dashboard theme (see
    # src/dashboard/build_dashboard.py) -- red for the real ground truth
    # (matches "detection point" elsewhere: a real, confirmed thing), amber
    # for the model's prediction (matches "drift trace & origin estimate":
    # a model-derived output, not raw ground truth).
    navy_panel, paper, alert, amber = "#0e1d2c", "#eaf1f7", "#d9534f", "#f4b95a"
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), facecolor=navy_panel)
    for ax in axes:
        ax.set_facecolor(navy_panel)

    axes[0].imshow(despeckled_display, cmap="gray")
    axes[0].set_title(f"SAR tile ({image_path.name})", color=paper, fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(despeckled_display, cmap="gray")
    axes[1].imshow(gt_tile, cmap="Reds", alpha=0.55 * (gt_tile > 0))
    axes[1].set_title("Real ground truth (Zenodo mask)", color=alert, fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(despeckled_display, cmap="gray")
    axes[2].imshow(pred_tile, cmap="Wistia", alpha=0.55 * (pred_tile > 0))
    axes[2].set_title("Model prediction (real trained checkpoint,\nheld-out Part III test image, see LOG.md)", color=amber, fontsize=12)
    axes[2].axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=110, facecolor=navy_panel)
    print(f"wrote {OUT_PATH}")

    iou = (
        (gt_tile.astype(bool) & pred_tile.astype(bool)).sum()
        / max((gt_tile.astype(bool) | pred_tile.astype(bool)).sum(), 1)
    )
    print(f"IoU vs. ground truth on this tile (real trained model, held-out test image): {iou:.4f}")


if __name__ == "__main__":
    main()
