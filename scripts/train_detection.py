"""
Real detection-model training run. See DECISIONS.md "Train/val/test
methodology" for why the data is split the way it is, and "Detection model
architecture" for the U-Net+ResNet18 choice.

Requires data/processed/train_manifest.csv and val_manifest.csv to exist
(run scripts/build_training_pool.py first). Part III (test set) is never
read by this script -- intentionally, see DECISIONS.md.

Before committing to a long run, this script first probes batch size
(tries 16, falls back to 8, then 4) with a couple of real forward/backward
passes, logging actual VRAM at each size, given the large headroom found
in Step 1 (under 15% used at batch_size=2, tile_size=512).

Usage:
    venv\\Scripts\\python.exe scripts\\train_detection.py
"""

import csv
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from detection.dataset import ImageMaskPair, ZenodoTileDataset  # noqa: E402
from detection.losses import DiceBCELoss  # noqa: E402
from detection.model import build_model  # noqa: E402
from detection.train import train  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_MANIFEST = REPO_ROOT / "data" / "processed" / "train_manifest.csv"
VAL_MANIFEST = REPO_ROOT / "data" / "processed" / "val_manifest.csv"
CHECKPOINT_DIR = REPO_ROOT / "data" / "processed" / "checkpoints"

TILE_SIZE = 512
POS_WEIGHT = 32.6  # real empirical value from scripts/compute_mask_class_balance.py
EPOCHS = 30
CANDIDATE_BATCH_SIZES = [16, 8, 4]  # tried in order; first one that doesn't OOM on a real batch is used
NUM_WORKERS = 6  # parallel tile reads (rasterio) -- epoch 1 at num_workers=0 was disk-I/O-bound,
                  # not GPU-bound (93min/epoch vs. ~64min of pure compute time from the batch probe)


def load_manifest(path: Path) -> list[ImageMaskPair]:
    pairs = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            pairs.append(ImageMaskPair(Path(row["image_path"]), Path(row["mask_path"]), row["label"]))
    return pairs


def probe_batch_size(model, loss_fn, dataset, device) -> int:
    print("\n=== batch size probe ===")
    for bs in CANDIDATE_BATCH_SIZES:
        try:
            torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
            loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
            images, masks = next(iter(loader))
            images, masks = images.to(device), masks.to(device)

            start = time.perf_counter()
            use_amp = device.type == "cuda"
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = loss_fn(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            elapsed = time.perf_counter() - start

            peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else None
            print(f"  batch_size={bs}: OK, {elapsed:.2f}s/step, peak_vram={peak_mb:.0f}MB" if peak_mb else
                  f"  batch_size={bs}: OK, {elapsed:.2f}s/step (CPU)")
            return bs
        except torch.cuda.OutOfMemoryError:
            print(f"  batch_size={bs}: OOM, trying smaller")
            torch.cuda.empty_cache()
            continue
    raise RuntimeError("Even the smallest candidate batch size OOM'd -- reduce tile size or candidates further.")


def main() -> None:
    if not TRAIN_MANIFEST.exists() or not VAL_MANIFEST.exists():
        print("ERROR: manifests not found. Run scripts/build_training_pool.py first.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_pairs = load_manifest(TRAIN_MANIFEST)
    val_pairs = load_manifest(VAL_MANIFEST)
    print(f"train: {len(train_pairs)} source images, val: {len(val_pairs)} source images")

    train_dataset = ZenodoTileDataset(train_pairs, tile_size=TILE_SIZE, augment=True)
    val_dataset = ZenodoTileDataset(val_pairs, tile_size=TILE_SIZE, augment=False)
    print(f"train tiles: {len(train_dataset)}, val tiles: {len(val_dataset)}")

    model = build_model()
    loss_fn = DiceBCELoss(pos_weight=POS_WEIGHT)
    model.to(device)

    batch_size = probe_batch_size(model, loss_fn, train_dataset, device)

    # Rebuild a fresh model -- the probe step already did one optimizer step on it.
    model = build_model()

    print(f"\n=== real training run: {EPOCHS} epochs, batch_size={batch_size}, tile_size={TILE_SIZE} ===")
    result = train(
        model, train_dataset, loss_fn, device,
        epochs=EPOCHS, batch_size=batch_size,
        checkpoint_path=CHECKPOINT_DIR / "final_unet_resnet18.pt",
        val_dataset=val_dataset,
        best_checkpoint_path=CHECKPOINT_DIR / "best_unet_resnet18.pt",
        num_workers=NUM_WORKERS,
        latest_checkpoint_path=CHECKPOINT_DIR / "latest_unet_resnet18.pt",
    )

    print(f"\nbest val_dice={result.best_val_dice:.4f} at epoch {result.best_epoch}")


if __name__ == "__main__":
    main()
