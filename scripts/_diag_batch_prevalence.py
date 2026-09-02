"""
Scratch diagnostic (not part of the pipeline): checks batch COMPOSITION for
the ablation subset's actual training DataLoader -- exp01a/b/c and the
scratch focal/tversky variants all use sampling_strategy: none, which
train.py (src/detection/train.py's DataLoader construction) turns into a
plain shuffle=True DataLoader with no oil-aware resampling.

If a large fraction of real training batches happen to contain literally
zero oil-positive pixels anywhere in the whole batch, the easiest
loss-minimizing solution for those steps is just "predict the background
rate" regardless of which loss function is in use -- this would point at
batch composition/sampling as a real contributor to the ~0.40-everywhere
collapse, separate from the loss-formula question the focal/tversky
scratch variants are testing.

No GPU/model involved -- pure data loading, CPU only. Read-only.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train import load_manifest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from detection.dataset import ZenodoTileDataset  # noqa: E402
from detection.preprocess import normalize_db_fixed  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG = REPO_ROOT / "configs" / "exp01a_band1_perband.yaml"
TRAIN_MANIFEST = REPO_ROOT / "data" / "processed" / "train_manifest_ablation.csv"
N_BATCHES = 20
SEED = 42


def main():
    config = yaml.safe_load(CONFIG.read_text())
    batch_size = config["batch_size"]
    channels = tuple(config.get("channels", [1]))

    train_pairs, train_explicit_index = load_manifest(TRAIN_MANIFEST)
    print(f"ablation manifest: {len(train_pairs)} source images, {len(train_explicit_index)} selected tiles")
    print(f"batch_size={batch_size} (from {CONFIG.name}), sampling_strategy={config.get('sampling_strategy')!r} "
          f"-> real DataLoader uses shuffle=True, no oil-aware resampling (src/detection/train.py)")

    # normalize_fn doesn't matter for this check (only mask oil-pixel presence
    # is examined, not the image tensor) -- default normalize_db_fixed is fine.
    ds = ZenodoTileDataset(train_pairs, tile_size=config.get("tile_size", 512), augment=False,
                            channels=channels, normalize_fn=normalize_db_fixed,
                            explicit_index=train_explicit_index)

    g = torch.Generator().manual_seed(SEED)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, generator=g)

    n_batches_all_zero_oil = 0
    n_batches_checked = 0
    oil_pixel_fracs_per_batch = []
    n_oil_tiles_per_batch = []

    for images, masks in loader:
        if n_batches_checked >= N_BATCHES:
            break
        n_batches_checked += 1
        masks_np = masks.numpy()  # (B, 1, H, W)
        oil_pixels_per_tile = (masks_np > 0).reshape(masks_np.shape[0], -1).sum(axis=1)
        total_oil_pixels = oil_pixels_per_tile.sum()
        total_pixels = masks_np.size
        n_oil_tiles = int((oil_pixels_per_tile > 0).sum())

        oil_pixel_fracs_per_batch.append(total_oil_pixels / total_pixels)
        n_oil_tiles_per_batch.append(n_oil_tiles)
        if total_oil_pixels == 0:
            n_batches_all_zero_oil += 1

        print(f"  batch {n_batches_checked:2d}: {n_oil_tiles}/{masks_np.shape[0]} tiles have >=1 oil pixel, "
              f"batch oil-pixel fraction = {total_oil_pixels/total_pixels:.4%}")

    oil_pixel_fracs_per_batch = np.array(oil_pixel_fracs_per_batch)
    n_oil_tiles_per_batch = np.array(n_oil_tiles_per_batch)

    print(f"\n=== summary over {n_batches_checked} real sampled batches (batch_size={batch_size}) ===")
    print(f"batches with ZERO oil-positive pixels anywhere in the batch: "
          f"{n_batches_all_zero_oil}/{n_batches_checked} ({100*n_batches_all_zero_oil/n_batches_checked:.1f}%)")
    print(f"mean oil-pixel fraction per batch: {oil_pixel_fracs_per_batch.mean():.4%} "
          f"(min={oil_pixel_fracs_per_batch.min():.4%}, max={oil_pixel_fracs_per_batch.max():.4%})")
    print(f"mean # tiles-with-any-oil per batch: {n_oil_tiles_per_batch.mean():.2f} / {batch_size} "
          f"(min={n_oil_tiles_per_batch.min()}, max={n_oil_tiles_per_batch.max()})")


if __name__ == "__main__":
    main()
