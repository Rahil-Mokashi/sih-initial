"""
Real threshold-swept IoU comparison across multiple trained checkpoints,
on the VALIDATION set (val_manifest.csv) -- NOT Part III. Part III is
touched exactly once, at the very end, per DECISIONS.md "Train/val/test
methodology"; comparing/selecting among candidate checkpoints is exactly
what the val split exists for.

Built to answer a real question raised mid-training: three different
interventions (pos_weight change, oil-tile oversampling, Tversky loss)
all left val_dice flat (~0.0226-0.0235) during training. But val_dice is
computed on RAW, unthresholded sigmoid probabilities (see
src/detection/train.py's evaluate()) -- if a checkpoint's real oil-pixel
probabilities are low-but-improving (e.g. drifting from 0.15 to 0.25),
raw dice can stay flat while a threshold-swept IoU (which only cares
about relative separation, not absolute confidence) would show real
movement. This script checks that directly rather than assuming flat
val_dice means flat learning.

Usage:
    venv\\Scripts\\python.exe scripts\\compare_checkpoints_on_val.py \\
        data/processed/checkpoints/best_unet_resnet18.pt \\
        data/processed/checkpoints/trial_oversample/best_unet_resnet18.pt \\
        data/processed/checkpoints/trial_tversky/best_unet_resnet18.pt
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from detection.inference import load_model_for_inference  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_test_set import THRESHOLDS, evaluate_image  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
VAL_MANIFEST = REPO_ROOT / "data" / "processed" / "val_manifest.csv"


def load_val_pairs() -> list[tuple[Path, Path]]:
    rows = list(csv.DictReader(open(VAL_MANIFEST, newline="")))
    return [(Path(r["image_path"]), Path(r["mask_path"])) for r in rows]


def evaluate_checkpoint(checkpoint_path: Path, pairs: list[tuple[Path, Path]], device: torch.device) -> dict:
    model = load_model_for_inference(checkpoint_path, device)
    all_tiles = []
    for image_path, mask_path in pairs:
        all_tiles.extend(evaluate_image(model, device, image_path, mask_path))
    oil_tiles = [r for r in all_tiles if r["has_oil"]]

    per_threshold = {}
    for t in THRESHOLDS:
        oil_iou = [r["per_threshold"][t]["iou"] for r in oil_tiles]
        per_threshold[t] = float(np.mean(oil_iou)) if oil_iou else None
    best_t = max(THRESHOLDS, key=lambda t: per_threshold[t] or 0)
    return {"per_threshold_oil_iou": per_threshold, "best_threshold": best_t, "best_oil_iou": per_threshold[best_t],
            "n_oil_tiles": len(oil_tiles)}


def main() -> None:
    checkpoint_paths = [Path(p) for p in sys.argv[1:]]
    if not checkpoint_paths:
        print("Usage: compare_checkpoints_on_val.py <checkpoint1> [checkpoint2] ...")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    pairs = load_val_pairs()
    print(f"val set: {len(pairs)} real images (val_manifest.csv)\n")

    results = {}
    for ckpt_path in checkpoint_paths:
        if not ckpt_path.exists():
            print(f"SKIP {ckpt_path}: not found")
            continue
        print(f"=== {ckpt_path} ===")
        result = evaluate_checkpoint(ckpt_path, pairs, device)
        results[str(ckpt_path)] = result
        for t in THRESHOLDS:
            print(f"  threshold={t:.2f}  oil-tiles-only IoU={result['per_threshold_oil_iou'][t]:.4f}")
        print(f"  best: threshold={result['best_threshold']}, oil IoU={result['best_oil_iou']:.4f} "
              f"({result['n_oil_tiles']} real oil tiles in val set)\n")

    print("=== summary: best real oil-tiles-only IoU per checkpoint (val set) ===")
    for path, r in sorted(results.items(), key=lambda kv: -(kv[1]["best_oil_iou"] or 0)):
        print(f"  {r['best_oil_iou']:.4f} (threshold {r['best_threshold']})  {path}")


if __name__ == "__main__":
    main()
