"""
Phase 0 Gate A.1 diagnostic: tiny overfit smoke test.

If a model with real capacity cannot memorize a handful of fixed,
un-augmented training tiles, something upstream of any hyperparameter
(mask alignment, tensor shape, loss sign, label polarity, normalization,
dataloader, model output shape) is broken -- no amount of tuning on the
full dataset would be meaningful until this passes. This is checked BEFORE
trusting the metric audit or any of this project's prior training-tuning
conclusions (see LOG.md lines ~1363-1408 and docs/metric_audit.md).

Deliberately selects OIL-CONTAINING tiles only (not a random sample) --
an all-background tile is trivially "solved" by predicting all-zero and
wouldn't exercise the label/mask/loss plumbing this test exists to check.

Usage:
    venv\\Scripts\\python.exe scripts\\overfit_smoke_test.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from detection.dataset import ImageMaskPair, ZenodoTileDataset  # noqa: E402
from detection.losses import DiceBCELoss  # noqa: E402
from detection.metrics import tile_metrics  # noqa: E402
from detection.model import build_model  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_MANIFEST = REPO_ROOT / "data" / "processed" / "train_manifest.csv"
TILE_SIZE = 512
N_SAMPLES = 12
N_ITERS = 300
LR = 1e-3
POS_WEIGHT = 32.6  # matches production DiceBCELoss default -- see src/detection/losses.py


def load_manifest(path: Path) -> list[ImageMaskPair]:
    pairs = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            pairs.append(ImageMaskPair(Path(row["image_path"]), Path(row["mask_path"]), row["label"]))
    return pairs


def pick_oil_tiles(dataset: ZenodoTileDataset, n: int) -> list[int]:
    picked = []
    for i, (pair_idx, y, x) in enumerate(dataset.index):
        mask_path = dataset.pairs[pair_idx].mask_path
        with rasterio.open(mask_path) as src:
            tile = src.read(1, window=Window(x, y, dataset.tile_size, dataset.tile_size))
        if (tile > 0).any():
            picked.append(i)
        if len(picked) >= n:
            break
    return picked


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    pairs = load_manifest(TRAIN_MANIFEST)
    full_dataset = ZenodoTileDataset(pairs, tile_size=TILE_SIZE, augment=False)  # augment=False: isolate memorization from augmentation variance
    indices = pick_oil_tiles(full_dataset, N_SAMPLES)
    print(f"selected {len(indices)} fixed oil-containing tiles out of {len(full_dataset)} total tiles")
    if len(indices) < N_SAMPLES:
        print(f"WARNING: only found {len(indices)} oil-containing tiles scanning from the start of the manifest")
    if not indices:
        print("FAIL: no oil-containing tiles found at all -- cannot run this test")
        sys.exit(1)

    subset = torch.utils.data.Subset(full_dataset, indices)
    images, masks = zip(*[subset[i] for i in range(len(subset))])
    images = torch.stack(images).to(device)
    masks = torch.stack(masks).to(device)
    print(f"batch shape: images={tuple(images.shape)} masks={tuple(masks.shape)}")
    print(f"mask oil-pixel fraction per sample: {[round(float(m.mean()), 4) for m in masks]}")

    model = build_model(in_channels=images.shape[1]).to(device)
    loss_fn = DiceBCELoss(pos_weight=POS_WEIGHT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    first_loss = None
    for it in range(1, N_ITERS + 1):
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()
        if first_loss is None:
            first_loss = loss.item()
        if it % 25 == 0 or it == 1:
            print(f"iter {it}/{N_ITERS}  loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()

    print(f"\n=== per-sample metrics on the SAME {len(indices)} memorized tiles, threshold=0.5 ===")
    ious, dices = [], []
    for i in range(len(indices)):
        m = tile_metrics(probs[i, 0], masks[i, 0].cpu().numpy(), threshold=0.5)
        ious.append(m["iou"]); dices.append(m["dice"])
        print(f"  sample {i}: IoU={m['iou']:.4f} Dice={m['dice']:.4f} Precision={m['precision']:.4f} "
              f"Recall={m['recall']:.4f} pred_pos={m['pred_positive_fraction']:.4f} gt_pos={m['gt_positive_fraction']:.4f}")

    mean_iou, mean_dice = float(np.mean(ious)), float(np.mean(dices))
    print(f"\nfirst-iter loss={first_loss:.4f}  final loss={loss.item():.4f}")
    print(f"mean IoU={mean_iou:.4f}  mean Dice={mean_dice:.4f}  (on the {len(indices)} memorized training tiles)")

    verdict = "PASS" if mean_iou > 0.8 else ("PARTIAL" if mean_iou > 0.3 else "FAIL")
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
