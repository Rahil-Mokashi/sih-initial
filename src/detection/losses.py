"""
Loss function for binary oil-spill segmentation.

Oil pixels are a small minority of any scene -- empirically ~3.0% mean,
~1.8% median across the 1200 real masks from the Zenodo Sentinel-1 dataset
(see scripts/compute_mask_class_balance.py and DECISIONS.md). Plain
per-pixel accuracy, or unweighted BCE, would let the model collapse to
"predict all background" and still score >97% accuracy. Two things address
this:
  - Dice loss directly optimizes region overlap (intersection over union of
    predicted vs. true foreground), which is insensitive to the background
    pixel count dominating the loss.
  - BCE gets a positive-class weight (`pos_weight`) derived from the real
    class imbalance above, so misclassifying an oil pixel costs proportionally
    more than misclassifying a background pixel.
The two are summed so the model gets both a region-overlap signal (Dice)
and a stable per-pixel gradient (weighted BCE) from the start of training.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


class DiceBCELoss(nn.Module):
    """Dice loss + weighted BCE-with-logits. dice_weight/bce_weight balance the two terms."""

    def __init__(self, pos_weight: float = 1.0, dice_weight: float = 1.0, bce_weight: float = 1.0):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(pos_weight))
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * dice_loss(logits, targets) + self.bce_weight * self.bce(logits, targets)
