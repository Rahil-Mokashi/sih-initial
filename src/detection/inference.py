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
from typing import Callable

import numpy as np
import torch

from detection.model import build_model
from detection.preprocess import lee_filter, normalize_db_fixed


def load_model_for_inference(checkpoint_path: str | Path, device: torch.device, in_channels: int = 1) -> torch.nn.Module:
    """in_channels must match the architecture the checkpoint was trained with -- the
    dashboard/render_detection_overlay.py production path always uses the default (1),
    since the real PANGAEA case-study images it runs on are single-band JPGs. Pass
    in_channels=2 only for a checkpoint actually trained on both real SAR bands."""
    model = build_model(encoder_weights=None, in_channels=in_channels)  # weights come from the checkpoint, not ImageNet
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_probs(model: torch.nn.Module, image_tile: np.ndarray, device: torch.device,
                   normalize_fn: Callable[[np.ndarray], np.ndarray] | None = None) -> np.ndarray:
    """
    image_tile: raw (despeckled or not) array in real dB units (calibrated)
    OR already-normalized [0,1] -- pass raw dB and this despeckles +
    normalizes consistently with training. Shape (H, W) for the production
    single-band path, or (C, H, W) for a checkpoint trained on multiple real
    SAR bands (see src/detection/model.py's build_model docstring). Returns
    per-pixel oil probabilities in [0, 1] (sigmoid output, not yet
    thresholded) -- see predict_mask() for the thresholded version, and
    scripts/evaluate_test_set.py for why raw probabilities matter: this
    project's real trained checkpoint (see LOG.md) never crosses 0.5
    anywhere on a real test tile despite carrying genuine, weaker-than-0.5
    signal, so evaluating at multiple thresholds from one probability map
    is how the real operating point gets found instead of guessed.

    normalize_fn, if given, replaces the default normalize_db_fixed (the
    original global [-40, 10] range) -- required for any checkpoint trained
    via train.py's per-band normalization (configs/exp01*, the Focal/Tversky
    scratch checkpoints), since feeding those a differently-scaled input
    than they were trained on produces meaningless predictions, not just
    slightly-off ones. None reproduces the exact prior hardcoded behavior
    for every existing caller (the original epoch-39/44 checkpoints, which
    predate per-band normalization and were trained on normalize_db_fixed).
    """
    despeckled = lee_filter(image_tile.astype(np.float32))  # per-channel if image_tile is (C, H, W)
    normalized = (normalize_fn or normalize_db_fixed)(despeckled)

    if normalized.ndim == 2:
        x = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).float().to(device)
    else:
        x = torch.from_numpy(normalized).unsqueeze(0).float().to(device)
    logits = model(x)
    probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
    return probs


@torch.no_grad()
def predict_mask(model: torch.nn.Module, image_tile: np.ndarray, device: torch.device, threshold: float = 0.5) -> np.ndarray:
    """Thresholded binary mask (0/1) -- see predict_probs() for the raw probabilities this is built from."""
    probs = predict_probs(model, image_tile, device)
    return (probs > threshold).astype(np.float32)
