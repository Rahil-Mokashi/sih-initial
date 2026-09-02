"""
Scratch diagnostic (not part of the pipeline): checks whether the RAW INPUT
itself carries a usable learning signal for exp01a, independent of which
loss function is used (the focal/tversky scratch variants -- see
configs/_scratch_exp01a_{focal,tversky}.yaml -- are testing the loss;
this instead asks whether the input could explain the collapse regardless
of loss).

Loads exp01a's real best.pt checkpoint (in_channels=1, Band 1 alone, its
actual trained architecture/weights) and runs ONE forward+backward pass on
a real batch from the ablation subset manifest, using exp01a's own loss
(DiceBCE, pos_weight=32.6) and normalization. Reports:
  - gradient norm at the first conv layer (encoder.conv1) and the final
    output layer (segmentation_head.0) after that one backward pass
  - per-band input tensor stats (mean/std/min/max) for a real batch -- BOTH
    real SAR bands read directly off disk (channels=(1,2)), even though
    exp01a's architecture only consumes band 1, so band 2's raw signal can
    be compared too
  - which channels/dimensions of the (band-1-only, as fed to the model)
    input have near-zero variance across the batch (a per-pixel-position
    std across the batch dim near 0 would mean the model sees an
    almost-constant image at that position regardless of sample)

Read-only / inference-plus-one-backward-pass only. Does not save any
checkpoint or optimizer state, does not modify best.pt.

--fresh: use a freshly-constructed model (build_model() with its default
ImageNet-pretrained encoder, randomly-initialized decoder/segmentation
head -- i.e. exactly what exp01a itself started from before its 15 real
epochs, since resume_from was null) INSTEAD of loading best.pt. Same real
batch (same seed=42 sample), same loss, same everything else -- the only
difference is trained-and-collapsed vs. never-trained weights. Added to
rule out "the collapsed checkpoint sits in a flat local minimum, so of
course its gradients look small" as an alternative explanation for the
plain-best.pt run's gradient-norm numbers.
"""
import argparse
import sys
from pathlib import Path
from functools import partial

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train import load_manifest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from detection.dataset import ZenodoTileDataset  # noqa: E402
from detection.model import build_model  # noqa: E402
from detection.preprocess import normalize_db_per_channel  # noqa: E402
from detection.losses import DiceBCELoss  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG = REPO_ROOT / "configs" / "exp01a_band1_perband.yaml"
CHECKPOINT = REPO_ROOT / "data" / "processed" / "checkpoints" / "exp01a_band1_perband" / "best.pt"
TRAIN_MANIFEST = REPO_ROOT / "data" / "processed" / "train_manifest_ablation.csv"
BATCH_SIZE = 16
POS_WEIGHT = 32.6

# Both real SAR bands' own empirical [p1,p99] ranges (exp01c_both_perband.yaml),
# used ONLY for the both-bands raw-input-stats report below -- the
# forward/backward pass itself still uses exp01a's actual band-1-only
# normalization (config's own fixed_range), since that's the real trained model.
BOTH_BAND_RANGES = [(-43.710, -12.894), (-33.862, -6.324)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fresh", action="store_true",
                    help="use a freshly-constructed model instead of loading exp01a's best.pt")
    return p.parse_args()


def main():
    args = parse_args()
    config = yaml.safe_load(CONFIG.read_text())
    band1_range = [tuple(r) for r in config["normalization"]["fixed_range"]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- either exp01a's real trained-and-collapsed weights, or a fresh,
    # never-trained-on-this-task model (same architecture) ---
    model = build_model(in_channels=config.get("in_channels", 1))
    if args.fresh:
        print("using a FRESH model (ImageNet-pretrained encoder, randomly-initialized "
              "decoder/segmentation head) -- best.pt NOT loaded")
    else:
        ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"loaded {CHECKPOINT}: epoch={ckpt.get('epoch')} val_loss={ckpt.get('val_loss')} val_dice={ckpt.get('val_dice')}")
    model.to(device)
    model.train()  # backward pass, not eval-mode inference

    train_pairs, train_explicit_index = load_manifest(TRAIN_MANIFEST)
    print(f"ablation manifest: {len(train_pairs)} source images, {len(train_explicit_index)} selected tiles")

    # --- dataset reading BOTH bands (for the raw input-stats report), band-1
    # normalized the same way exp01a's config does, band-2 normalized with
    # its own empirical range purely for reporting purposes ---
    normalize_fn = partial(normalize_db_per_channel, ranges=BOTH_BAND_RANGES)
    ds = ZenodoTileDataset(train_pairs, tile_size=config.get("tile_size", 512), augment=False,
                            channels=(1, 2), normalize_fn=normalize_fn, return_nodata_mask=True,
                            explicit_index=train_explicit_index)

    rng = np.random.default_rng(42)
    idxs = rng.choice(len(ds), size=BATCH_SIZE, replace=False)
    images, masks, valids = zip(*(ds[int(i)] for i in idxs))
    images_both = torch.stack(images)   # (B, 2, H, W), both bands
    masks_t = torch.stack(masks)        # (B, 1, H, W)
    valids_t = torch.stack(valids)      # (B, 1, H, W)
    print(f"\nreal batch: {images_both.shape[0]} tiles, tile_size={images_both.shape[-1]}")

    # --- input tensor stats, BOTH real bands ---
    print("\n=== input tensor stats, both real SAR bands (real batch, after per-band [0,1] normalization) ===")
    band_names = ["Band 1 (VV)", "Band 2 (VH)"]
    for c in range(2):
        band = images_both[:, c]
        print(f"{band_names[c]}: mean={band.mean():.6f} std={band.std():.6f} "
              f"min={band.min():.6f} max={band.max():.6f}")

    # --- near-zero-variance check, per-pixel-position across the batch,
    # done separately for each band ---
    print("\n=== near-zero-variance check (std across the batch dim, per pixel position) ===")
    for c in range(2):
        band = images_both[:, c]  # (B, H, W)
        per_pixel_std = band.std(dim=0)  # (H, W)
        frac_near_zero = (per_pixel_std < 1e-4).float().mean().item()
        print(f"{band_names[c]}: per-pixel-position std across batch -- "
              f"mean={per_pixel_std.mean():.6f} min={per_pixel_std.min():.6f} max={per_pixel_std.max():.6f}; "
              f"fraction of pixel positions with std < 1e-4 across the {BATCH_SIZE}-tile batch: {frac_near_zero:.4%}")
    # whole-band-is-constant check (a single scalar std over the entire band, all pixels+samples)
    for c in range(2):
        band = images_both[:, c]
        print(f"{band_names[c]}: whole-band std (all pixels x all samples pooled): {band.flatten().std():.6f}")

    # --- forward + backward pass using exp01a's ACTUAL band-1-only input,
    # its actual loss (DiceBCE, pos_weight=32.6), on this same real batch ---
    images_band1 = images_both[:, :1].clone()  # (B, 1, H, W), exp01a's real input
    # re-normalize band 1 with exp01a's OWN range (not BOTH_BAND_RANGES' copy,
    # which happens to be identical for band 1, but be explicit/correct anyway)
    del band1_range  # (identical to BOTH_BAND_RANGES[0]; already applied above)

    images_band1 = images_band1.to(device)
    masks_t = masks_t.to(device)
    valids_t = valids_t.to(device)

    loss_fn = DiceBCELoss(pos_weight=POS_WEIGHT, dice_weight=1.0, bce_weight=1.0).to(device)

    model.zero_grad(set_to_none=True)
    logits = model(images_band1)
    loss = loss_fn(logits, masks_t, mask=valids_t)
    loss.backward()

    model_desc = "FRESH untrained model" if args.fresh else "exp01a's real trained-and-collapsed checkpoint"
    print(f"\n=== forward+backward pass ({model_desc}, band-1 input, DiceBCE pos_weight={POS_WEIGHT}) ===")
    print(f"loss on this real batch: {loss.item():.6f}")

    first_conv = model.encoder.conv1
    final_conv = model.segmentation_head[0]
    print(f"first conv layer ({type(first_conv).__name__}, encoder.conv1): "
          f"weight grad norm = {first_conv.weight.grad.norm().item():.6e}")
    print(f"final output layer ({type(final_conv).__name__}, segmentation_head.0): "
          f"weight grad norm = {final_conv.weight.grad.norm().item():.6e}")

    # total grad norm across all parameters, for context
    total_norm = torch.sqrt(sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None))
    n_params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_params_total = sum(1 for _ in model.parameters())
    print(f"total grad norm across all parameters: {total_norm.item():.6e} "
          f"({n_params_with_grad}/{n_params_total} params received a grad)")


if __name__ == "__main__":
    main()
