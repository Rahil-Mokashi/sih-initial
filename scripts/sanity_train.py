"""
Step 1 sanity check: prove the training loop runs end to end on real (if
limited) data on this GPU. NOT a real training run -- see DECISIONS.md and
LOG.md for why the real Zenodo images aren't available yet.

Data used: the 5 real PANGAEA image+annotation pairs (despeckled image +
bbox-rasterized pseudo-mask -- see src/detection/dataset.py for why these
are pseudo-masks, not true pixel masks). Tiled at two candidate tile sizes
(256 and, if that trains stably with VRAM headroom, 512) so we know which
is safe once the full Zenodo dataset is on disk.

Prints per-epoch loss/time/VRAM, saves a checkpoint, then reloads it to
confirm save/load round-trips correctly.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from detection.dataset import SARTileDataset, bbox_to_mask  # noqa: E402
from detection.losses import DiceBCELoss  # noqa: E402
from detection.model import build_model  # noqa: E402
from detection.train import load_checkpoint, train  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = REPO_ROOT / "data" / "raw" / "pangaea_med_oil_spill" / "images"
ANNOTATIONS_DIR = REPO_ROOT / "data" / "raw" / "pangaea_med_oil_spill" / "annotations"
CHECKPOINT_PATH = REPO_ROOT / "data" / "processed" / "checkpoints" / "sanity_unet_resnet18.pt"

# Real empirical class balance from all 1200 Zenodo masks (see
# scripts/compute_mask_class_balance.py): mean oil fraction ~0.0298 ->
# pos_weight = (1 - 0.0298) / 0.0298 ~= 32.6
POS_WEIGHT = 32.6

EPOCHS = 5
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 2  # effective batch size 4


def load_pairs() -> list[tuple[np.ndarray, np.ndarray]]:
    pairs = []
    for jpg_path in sorted(IMAGES_DIR.glob("*.jpg")):
        xml_path = ANNOTATIONS_DIR / (jpg_path.stem + ".xml")
        if not xml_path.exists():
            continue
        image = np.array(Image.open(jpg_path).convert("L"), dtype=np.float32)
        mask = bbox_to_mask(xml_path, image.shape)
        pairs.append((image, mask))
    return pairs


def try_tile_size(tile_size: int, device: torch.device, pairs) -> bool:
    print(f"\n=== sanity pass at tile_size={tile_size} ===")
    dataset = SARTileDataset(pairs, tile_size=tile_size)
    print(f"dataset has {len(dataset)} tiles from {len(pairs)} source images")
    if len(dataset) == 0:
        print(f"SKIP: no tiles produced at tile_size={tile_size} (source images too small)")
        return False

    model = build_model()
    loss_fn = DiceBCELoss(pos_weight=POS_WEIGHT)

    try:
        result = train(
            model, dataset, loss_fn, device,
            epochs=EPOCHS, batch_size=BATCH_SIZE, grad_accum_steps=GRAD_ACCUM_STEPS,
            checkpoint_path=CHECKPOINT_PATH if tile_size == 256 else None,
        )
    except torch.cuda.OutOfMemoryError:
        print(f"OOM at tile_size={tile_size}, batch_size={BATCH_SIZE} -- not safe on this GPU.")
        return False

    losses = [s.loss for s in result.history]
    print(f"loss trajectory: {[f'{l:.4f}' for l in losses]}")
    decreased = losses[-1] < losses[0]
    print(f"loss decreased over {EPOCHS} epochs: {decreased}")
    peak = max((s.peak_vram_mb or 0) for s in result.history)
    print(f"peak VRAM across epochs: {peak:.0f}MB")
    return True


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    pairs = load_pairs()
    print(f"loaded {len(pairs)} real image+pseudo-mask pairs from PANGAEA samples")
    if not pairs:
        print("ERROR: no PANGAEA samples found. Run scripts/download_pangaea_sample.py first.")
        sys.exit(1)

    ok_256 = try_tile_size(256, device, pairs)
    ok_512 = try_tile_size(512, device, pairs)

    print("\n=== checkpoint save/reload check ===")
    if CHECKPOINT_PATH.exists():
        model = build_model()
        load_checkpoint(model, CHECKPOINT_PATH, device)
        print(f"successfully reloaded checkpoint from {CHECKPOINT_PATH}")
    else:
        print("no checkpoint was saved (256 pass did not complete) -- nothing to reload.")

    print("\n=== summary ===")
    print(f"256x256 tile_size safe: {ok_256}")
    print(f"512x512 tile_size safe: {ok_512}")


if __name__ == "__main__":
    main()
