"""
Tests for src/detection/losses.py's optional nodata `mask` support
(dice_loss, DiceBCELoss, TverskyLoss). Hand-computed cases: construct a
tiny synthetic tile with known nodata pixels, confirm the masked loss
matches a value hand-computed with those pixels excluded, and confirm it
differs from the unmasked loss on the same inputs -- i.e. the mask isn't
just being silently ignored.
"""

import math

import torch

from detection.losses import DiceBCELoss, TverskyLoss, dice_loss


def test_dice_loss_matches_hand_computation_with_nodata_excluded():
    # 1x4 "tile". Pixel 3 (index 3) is nodata: prediction says confidently
    # oil (prob 0.9) but ground truth says not-oil -- if masking is
    # ignored, this pixel drags the score down; if masking works, it's
    # excluded entirely and the score is computed over pixels 0-2 only.
    logits = torch.logit(torch.tensor([[0.9, 0.1, 0.8, 0.9]]), eps=1e-6)
    targets = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])  # pixel 3 excluded

    probs = torch.tensor([0.9, 0.1, 0.8])  # pixels 0-2 only, hand-computed
    tg = torch.tensor([1.0, 0.0, 1.0])
    eps = 1e-6
    intersection = float((probs * tg).sum())
    union = float(probs.sum() + tg.sum())
    expected_dice = (2 * intersection + eps) / (union + eps)
    expected_loss = 1 - expected_dice

    masked_loss = dice_loss(logits, targets, mask=mask).item()
    assert math.isclose(masked_loss, expected_loss, rel_tol=1e-4)

    unmasked_loss = dice_loss(logits, targets).item()
    assert not math.isclose(masked_loss, unmasked_loss, rel_tol=1e-3)


def test_dice_loss_all_nodata_tile_is_treated_as_perfect():
    # Every pixel excluded -> both intersection and union sums are 0 ->
    # eps/eps = 1.0 dice (loss 0), matching the "nothing to score" convention
    # used elsewhere in this project (see metrics.py's union==0 case).
    logits = torch.logit(torch.tensor([[0.9, 0.1]]), eps=1e-6)
    targets = torch.tensor([[1.0, 0.0]])
    mask = torch.tensor([[0.0, 0.0]])
    loss = dice_loss(logits, targets, mask=mask).item()
    assert math.isclose(loss, 0.0, abs_tol=1e-4)


def test_dicebce_loss_bce_term_excludes_nodata_not_just_zeros_it():
    # A naive "zero the inputs at nodata pixels" fix would still divide by
    # the FULL pixel count in BCE's mean reduction, silently diluting the
    # loss instead of excluding the pixel. Confirm the masked value matches
    # hand-computing BCE as a mean over ONLY the valid pixels.
    logits = torch.logit(torch.tensor([[0.9, 0.1, 0.8, 0.5]]), eps=1e-6)
    targets = torch.tensor([[1.0, 0.0, 1.0, 1.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])  # pixel 3 (a confident-wrong pixel) excluded
    pos_weight = 1.0

    loss_fn = DiceBCELoss(pos_weight=pos_weight, dice_weight=0.0, bce_weight=1.0)  # isolate the BCE term
    masked = loss_fn(logits, targets, mask=mask).item()

    bce_elem = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=torch.tensor(pos_weight), reduction="none"
    )
    expected_bce = float(bce_elem[0, :3].mean())  # hand-computed mean over the 3 valid pixels only
    assert math.isclose(masked, expected_bce, rel_tol=1e-4)

    unmasked = loss_fn(logits, targets).item()
    assert not math.isclose(masked, unmasked, rel_tol=1e-3)


def test_dicebce_loss_mask_none_reproduces_original_unmasked_value():
    # Backward compatibility: switching BCEWithLogitsLoss to reduction="none"
    # + a manual .mean() must give the exact same number the old default
    # reduction="mean" gave, for every existing (mask=None) caller.
    logits = torch.logit(torch.tensor([[0.9, 0.1, 0.8, 0.3]]), eps=1e-6)
    targets = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    loss_fn = DiceBCELoss(pos_weight=2.0, dice_weight=1.0, bce_weight=1.0)

    original_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=torch.tensor(2.0), reduction="mean"
    )
    expected_dice = dice_loss(logits, targets)
    expected = float(expected_dice + original_bce)

    got = loss_fn(logits, targets).item()
    assert math.isclose(got, expected, rel_tol=1e-5)


def test_tversky_loss_masked_matches_hand_computation():
    logits = torch.logit(torch.tensor([[0.9, 0.1, 0.8, 0.9]]), eps=1e-6)
    targets = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    alpha, beta, eps = 0.3, 0.7, 1e-6

    probs = torch.tensor([0.9, 0.1, 0.8])
    tg = torch.tensor([1.0, 0.0, 1.0])
    tp = float((probs * tg).sum())
    fp = float((probs * (1 - tg)).sum())
    fn = float(((1 - probs) * tg).sum())
    expected = 1 - (tp + eps) / (tp + alpha * fp + beta * fn + eps)

    loss_fn = TverskyLoss(alpha=alpha, beta=beta, eps=eps)
    masked = loss_fn(logits, targets, mask=mask).item()
    assert math.isclose(masked, expected, rel_tol=1e-4)

    unmasked = loss_fn(logits, targets).item()
    assert not math.isclose(masked, unmasked, rel_tol=1e-3)
