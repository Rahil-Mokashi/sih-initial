"""
Inference: load a trained checkpoint and run it on real SAR imagery,
producing a predicted oil mask. Built now against the Step 1 sanity
checkpoint (trained on 4 PANGAEA images with rectangular bbox
pseudo-masks -- NOT a meaningful model, see DECISIONS.md) purely to prove
the load -> predict -> overlay plumbing works, so that swapping in the
real trained checkpoint (once scripts/train_detection.py finishes) is a
one-line change, not new code written under time pressure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from detection.model import build_model
from detection.preprocess import lee_filter, normalize_db_fixed


def load_model_for_inference(checkpoint_path: str | Path, device: torch.device) -> torch.nn.Module:
    model = build_model(encoder_weights=None)  # weights come from the checkpoint, not ImageNet
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_mask(model: torch.nn.Module, image_tile: np.ndarray, device: torch.device, threshold: float = 0.5) -> np.ndarray:
    """
    image_tile: raw (despeckled or not) single-band array in real dB units
    (calibrated) OR already-normalized [0,1] -- pass raw dB and this
    despeckles + normalizes consistently with training. Returns a binary
    mask (0/1) at the same resolution as the input tile.
    """
    despeckled = lee_filter(image_tile.astype(np.float32))
    normalized = normalize_db_fixed(despeckled)

    x = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).float().to(device)
    logits = model(x)
    probs = torch.sigmoid(logits)
    mask = (probs > threshold).float().cpu().numpy()[0, 0]
    return mask
