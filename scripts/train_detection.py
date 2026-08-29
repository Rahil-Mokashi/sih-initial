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

Loss/pos_weight/LR-scheduler are CLI-configurable (not hardcoded) -- added
after the real 60-epoch run plateaued *below* the trivial "predict all
oil" Dice baseline (val_dice ~0.022-0.0235 vs. ~0.058 at the real 2.98%
oil fraction), pointing at pos_weight=32.6 fighting Dice rather than just
missing LR decay (see LOG.md). Use --tag to sandbox trial-run checkpoints
under data/processed/checkpoints/<tag>/ instead of the top-level files
evaluate_test_set.py and render_detection_overlay.py read directly, and
--fresh to force a from-scratch start even if a checkpoint exists at that
path (trials should NOT resume from the plateaued epoch-39 checkpoint,
which was trained under the old loss/pos_weight).

    # loss-function trial (isolates the loss change -- no LR scheduler)
    venv\\Scripts\\python.exe scripts\\train_detection.py --loss tversky --epochs 10 --tag trial_tversky --fresh

    # pos_weight + LR-scheduler trial (keeps DiceBCELoss)
    venv\\Scripts\\python.exe scripts\\train_detection.py --pos-weight 8 --epochs 10 --tag trial_posweight_sched \\
        --fresh --use-lr-scheduler --lr-monitor val_dice --lr-patience 4 --lr-factor 0.5 --lr-min 1e-6
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from detection.dataset import ImageMaskPair, ZenodoTileDataset, compute_oil_tile_weights  # noqa: E402
from detection.losses import DiceBCELoss, TverskyLoss  # noqa: E402
from detection.model import build_model  # noqa: E402
from detection.train import train  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_MANIFEST = REPO_ROOT / "data" / "processed" / "train_manifest.csv"
VAL_MANIFEST = REPO_ROOT / "data" / "processed" / "val_manifest.csv"
CHECKPOINT_DIR = REPO_ROOT / "data" / "processed" / "checkpoints"

TILE_SIZE = 512
POS_WEIGHT = 32.6  # real empirical value from scripts/compute_mask_class_balance.py
EPOCHS = 60  # extended from 30 after the real held-out eval showed the model was still
             # improving (training loss hadn't plateaued) and genuinely undertrained --
             # see LOG.md "threshold sweep" entry. Resumes automatically via
             # latest_checkpoint_path, not a restart from scratch.
CANDIDATE_BATCH_SIZES = [16, 8, 4]  # tried in order; first one that doesn't OOM on a real batch is used
NUM_WORKERS = 6  # parallel tile reads (rasterio) -- epoch 1 at num_workers=0 was disk-I/O-bound,
                  # not GPU-bound (93min/epoch vs. ~64min of pure compute time from the batch probe)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--loss", choices=["dicebce", "tversky"], default="dicebce")
    p.add_argument("--pos-weight", type=float, default=POS_WEIGHT, help="DiceBCELoss only")
    p.add_argument("--tversky-alpha", type=float, default=0.3, help="false-positive weight, TverskyLoss only")
    p.add_argument("--tversky-beta", type=float, default=0.7, help="false-negative weight, TverskyLoss only")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--tag", type=str, default="", help="sandbox checkpoints under checkpoints/<tag>/ instead of the top-level files")
    p.add_argument("--fresh", action="store_true", help="ignore any existing checkpoint at this run's path and start from epoch 1")
    p.add_argument("--use-lr-scheduler", action="store_true")
    p.add_argument("--lr-monitor", choices=["val_loss", "val_dice"], default="val_loss")
    p.add_argument("--lr-patience", type=int, default=3)
    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--lr-min", type=float, default=1e-6)
    p.add_argument("--oversample-oil-tiles", action="store_true",
                    help="WeightedRandomSampler targeting --target-oil-fraction oil-containing tiles per epoch "
                         "(see scripts/analyze_tile_oil_distribution.py -- 82.4%% of real training tiles have zero "
                         "oil pixels, which pos_weight/loss choice cannot fix)")
    p.add_argument("--target-oil-fraction", type=float, default=0.5)
    return p.parse_args()


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
    args = parse_args()

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

    if args.loss == "tversky":
        loss_fn = TverskyLoss(alpha=args.tversky_alpha, beta=args.tversky_beta)
        loss_desc = f"TverskyLoss(alpha={args.tversky_alpha}, beta={args.tversky_beta})"
    else:
        loss_fn = DiceBCELoss(pos_weight=args.pos_weight)
        loss_desc = f"DiceBCELoss(pos_weight={args.pos_weight})"

    model = build_model()
    model.to(device)

    batch_size = probe_batch_size(model, loss_fn, train_dataset, device)

    # Rebuild a fresh model -- the probe step already did one optimizer step on it.
    model = build_model()

    checkpoint_dir = (CHECKPOINT_DIR / args.tag) if args.tag else CHECKPOINT_DIR

    sampler = None
    if args.oversample_oil_tiles:
        weights = compute_oil_tile_weights(train_dataset, target_oil_fraction=args.target_oil_fraction)
        sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(train_dataset), replacement=True)

    print(f"\n=== real training run: {args.epochs} epochs, batch_size={batch_size}, tile_size={TILE_SIZE}, "
          f"loss={loss_desc}, oversample_oil_tiles={args.oversample_oil_tiles}, checkpoint_dir={checkpoint_dir} ===")
    result = train(
        model, train_dataset, loss_fn, device,
        epochs=args.epochs, batch_size=batch_size,
        checkpoint_path=checkpoint_dir / "final_unet_resnet18.pt",
        val_dataset=val_dataset,
        best_checkpoint_path=checkpoint_dir / "best_unet_resnet18.pt",
        num_workers=NUM_WORKERS,
        latest_checkpoint_path=checkpoint_dir / "latest_unet_resnet18.pt",
        resume=not args.fresh,
        use_lr_scheduler=args.use_lr_scheduler,
        lr_monitor=args.lr_monitor,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        lr_min=args.lr_min,
        sampler=sampler,
    )

    print(f"\nbest val_dice={result.best_val_dice:.4f} at epoch {result.best_epoch}")


if __name__ == "__main__":
    main()
