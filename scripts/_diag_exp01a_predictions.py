"""
Scratch diagnostic (not part of the pipeline): loads exp01a's already-trained
best.pt checkpoint and runs real forward passes (no training) over a sample
of real oil-labeled validation tiles, to check whether predictions are
collapsing to near-all-background (near-0 everywhere) or near-all-foreground
(near-1 everywhere), vs. actually discriminating oil pixels.

Read-only / inference-only. Does not modify any checkpoint or write new
training state.
"""
import argparse
import csv
import sys
from pathlib import Path
from functools import partial

import numpy as np
import rasterio
import torch
import yaml
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from detection.dataset import ImageMaskPair, ZenodoTileDataset  # noqa: E402
from detection.model import build_model  # noqa: E402
from detection.preprocess import normalize_db_per_channel  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
VAL_MANIFEST = REPO_ROOT / "data" / "processed" / "val_manifest.csv"
N_OIL_IMAGES = 15
N_NOOIL_IMAGES = 10


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "exp01a_band1_perband.yaml")
    p.add_argument("--checkpoint", type=Path,
                   default=REPO_ROOT / "data" / "processed" / "checkpoints" / "exp01a_band1_perband" / "best.pt")
    return p.parse_args()


def load_manifest(path):
    with open(path, newline="") as f:
        return [ImageMaskPair(Path(r["image_path"]), Path(r["mask_path"]), r["label"]) for r in csv.DictReader(f)]


def main():
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    channels = tuple(config.get("channels", [1]))
    ranges = [tuple(r) for r in config["normalization"]["fixed_range"]]
    normalize_fn = partial(normalize_db_per_channel, ranges=ranges)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(in_channels=config.get("in_channels", 1))
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"loaded {args.checkpoint}: epoch={ckpt.get('epoch')} val_loss={ckpt.get('val_loss')} val_dice={ckpt.get('val_dice')}")

    pairs = load_manifest(VAL_MANIFEST)
    oil_pairs = [p for p in pairs if p.label == "oil"][:N_OIL_IMAGES]
    noil_pairs = [p for p in pairs if p.label != "oil"][:N_NOOIL_IMAGES]
    sample_pairs = oil_pairs + noil_pairs
    print(f"sampling {len(oil_pairs)} oil-labeled + {len(noil_pairs)} non-oil-labeled val images "
          f"(full 512x512 tile grid each)")

    ds = ZenodoTileDataset(sample_pairs, tile_size=512, augment=False, channels=channels,
                            normalize_fn=normalize_fn, return_nodata_mask=True)
    print(f"{len(ds)} tiles total")

    all_pred_means = []
    oil_tile_pred_means = []
    oil_tile_true_fracs = []
    nooil_tile_pred_means = []
    frac_above_035_per_tile = []
    n_pred_effectively_zero = 0  # tile-mean pred prob < 1e-3
    n_pred_effectively_one = 0   # tile-mean pred prob > 0.999

    with torch.no_grad():
        for i in range(len(ds)):
            image_t, mask_t, valid_t = ds[i]
            image_t = image_t.unsqueeze(0).to(device)
            logits = model(image_t)
            probs = torch.sigmoid(logits).cpu().numpy()[0, 0]  # (H, W)
            valid = valid_t.numpy()[0]  # (1,H,W) -> (H,W)
            mask = mask_t.numpy()[0]    # (1,H,W) -> (H,W)

            valid_mask = valid > 0.5
            if valid_mask.sum() == 0:
                continue
            pred_mean = probs[valid_mask].mean()
            true_frac = (mask[valid_mask] > 0).mean()
            frac_above = (probs[valid_mask] > 0.35).mean()

            all_pred_means.append(pred_mean)
            frac_above_035_per_tile.append(frac_above)
            if pred_mean < 1e-3:
                n_pred_effectively_zero += 1
            if pred_mean > 0.999:
                n_pred_effectively_one += 1

            if true_frac > 0:
                oil_tile_pred_means.append(pred_mean)
                oil_tile_true_fracs.append(true_frac)
            else:
                nooil_tile_pred_means.append(pred_mean)

    all_pred_means = np.array(all_pred_means)
    frac_above_035_per_tile = np.array(frac_above_035_per_tile)
    oil_tile_pred_means = np.array(oil_tile_pred_means)
    oil_tile_true_fracs = np.array(oil_tile_true_fracs)
    nooil_tile_pred_means = np.array(nooil_tile_pred_means)

    print(f"\n=== overall predicted-probability stats across {len(all_pred_means)} tiles ===")
    print(f"mean predicted prob (per-tile mean, averaged): {all_pred_means.mean():.6f}")
    print(f"min/max per-tile mean pred prob: {all_pred_means.min():.6f} / {all_pred_means.max():.6f}")
    print(f"tiles with mean pred prob < 0.001 (~all-background): {n_pred_effectively_zero}/{len(all_pred_means)}")
    print(f"tiles with mean pred prob > 0.999 (~all-foreground): {n_pred_effectively_one}/{len(all_pred_means)}")
    print(f"mean fraction of pixels > 0.35 threshold, per tile (averaged): {frac_above_035_per_tile.mean():.6f}")

    print(f"\n=== tiles WITH real oil pixels ({len(oil_tile_pred_means)} tiles) ===")
    if len(oil_tile_pred_means):
        print(f"mean predicted prob: {oil_tile_pred_means.mean():.6f}  (true oil-pixel frac mean: {oil_tile_true_fracs.mean():.6f})")
        print(f"correlation(pred_mean, true_frac) across these tiles: {np.corrcoef(oil_tile_pred_means, oil_tile_true_fracs)[0,1]:.4f}")

    print(f"\n=== tiles with ZERO real oil pixels ({len(nooil_tile_pred_means)} tiles) ===")
    if len(nooil_tile_pred_means):
        print(f"mean predicted prob: {nooil_tile_pred_means.mean():.6f}")

    print(f"\n=== a few individual oil tiles (pred_mean vs true_frac) ===")
    for pm, tf in list(zip(oil_tile_pred_means, oil_tile_true_fracs))[:15]:
        print(f"  pred_mean={pm:.5f}  true_frac={tf:.5f}")


if __name__ == "__main__":
    main()
