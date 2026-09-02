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


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    """
    mask, if given, is a same-shape (or broadcastable) 1.0=valid/0.0=invalid
    tensor (see detection.dataset.ZenodoTileDataset's return_nodata_mask) --
    invalid (nodata) pixels are excluded from both the intersection and the
    union sums, not just zeroed in isolation. This is safe here specifically
    because dice is a ratio of SUMS, not a per-element mean: zeroing `probs`
    and `targets` together at an invalid pixel removes that pixel's
    contribution from every term (intersection, union) identically, so the
    ratio is exactly what it would be if the pixel had never existed. (BCE
    below needs a different fix -- see DiceBCELoss -- because its usual
    'mean' reduction divides by the total element count regardless of value,
    so simply zeroing inputs would silently dilute the loss with fake
    "correct" pixels instead of excluding them.)
    """
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    targets = targets.flatten(1)
    if mask is not None:
        mask = mask.flatten(1)
        probs = probs * mask
        targets = targets * mask
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
        # reduction="none": BCEWithLogitsLoss's default 'mean' divides by the
        # TOTAL element count, including any nodata pixels we want excluded --
        # zeroing logits/targets at those pixels would NOT zero their BCE
        # contribution (BCE(logit=0, target=0) = log(2) != 0) and the fixed
        # divisor would still silently dilute the loss. Computing per-pixel
        # and dividing by the actual valid-pixel count (below) is the correct
        # masked mean; with mask=None this reduces to the exact same value as
        # the old default-'mean' BCE (mean over all elements either way).
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        dice = dice_loss(logits, targets, mask=mask)
        bce_per_pixel = self.bce(logits, targets)
        if mask is not None:
            bce_per_pixel = bce_per_pixel.flatten(1) * mask.flatten(1)
            valid_counts = mask.flatten(1).sum(dim=1).clamp(min=1.0)
            bce = (bce_per_pixel.sum(dim=1) / valid_counts).mean()
        else:
            bce = bce_per_pixel.mean()
        return self.dice_weight * dice + self.bce_weight * bce


class FocalLoss(nn.Module):
    """
    Binary focal loss (Lin et al. 2017): down-weights easy (already
    well-classified) pixels by (1-p_t)^gamma so hard/rare pixels dominate
    the gradient, instead of pos_weight's flat per-class multiplier.

    alpha (default 0.25, the paper's own default) balances positive vs.
    negative class weight independently of gamma's easy-example
    down-weighting -- this is a deliberately gentler imbalance correction
    than DiceBCELoss's pos_weight=32.6, added specifically to test whether
    pos_weight itself (not Dice, not the architecture) is what's pushing
    exp01a's training toward the near-constant ~0.40-everywhere prediction
    diagnosed in LOG.md, since focal loss's imbalance handling is additive/
    multiplicative rather than a single large scalar reweight.

    Masking follows DiceBCELoss.bce's approach, not dice_loss's zero-and-sum
    trick: per-pixel focal loss has no fixed-count 'mean' reduction to
    silently dilute (there's no nn.BCEWithLogitsLoss call here), but a
    masked-out logit still produces a well-defined (nonzero) per-pixel
    focal value that must be explicitly excluded before averaging, same
    reasoning as bce_per_pixel there.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        probs = torch.sigmoid(logits).flatten(1)
        targets = targets.flatten(1)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        per_pixel = -alpha_t * (1 - p_t).clamp(min=0).pow(self.gamma) * torch.log(p_t.clamp(min=self.eps))
        if mask is not None:
            mask = mask.flatten(1)
            per_pixel = per_pixel * mask
            valid_counts = mask.sum(dim=1).clamp(min=1.0)
            return (per_pixel.sum(dim=1) / valid_counts).mean()
        return per_pixel.mean()


class TverskyLoss(nn.Module):
    """
    Generalizes Dice with independent false-positive/false-negative weights
    (alpha, beta) instead of Dice's implicit 0.5/0.5 -- added as a
    DiceBCELoss alternative after the real 60-epoch run plateaued well
    *below* the trivial "predict all oil" baseline (val_dice ~0.022-0.0235
    vs. ~0.058 at the real 2.98% oil fraction), pointing at pos_weight=32.6
    double-correcting for class imbalance on top of Dice and pushing the
    model toward a low-confidence degenerate solution (see LOG.md). beta >
    alpha (default 0.3/0.7) weights false negatives more than false
    positives -- the direction a 3%-positive, currently under-predicting
    class needs.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # Same masking approach as dice_loss (see its docstring): zeroing
        # `probs` and `targets` together at an invalid pixel makes it
        # contribute 0 to tp, fp, AND fn alike (fp = probs*(1-targets) = 0
        # since probs=0; fn = (1-probs)*targets = 0 since targets=0), which
        # is safe because tversky, like dice, is a ratio of sums with no
        # separate total-element-count divisor.
        probs = torch.sigmoid(logits).flatten(1)
        targets = targets.flatten(1)
        if mask is not None:
            mask = mask.flatten(1)
            probs = probs * mask
            targets = targets * mask
        tp = (probs * targets).sum(dim=1)
        fp = (probs * (1 - targets)).sum(dim=1)
        fn = ((1 - probs) * targets).sum(dim=1)
        tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        return 1 - tversky.mean()
