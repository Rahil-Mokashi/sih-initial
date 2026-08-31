"""
Core binary segmentation metrics (oil vs. not-oil), factored out on their own
so they can be unit-tested independently of any model/dataset/checkpoint
plumbing -- see docs/metric_audit.md (Phase 0 metric audit) for why this
mattered: the training loop's `val_dice` (src/detection/train.py's
evaluate(), via losses.dice_loss) and the "real IoU" used to compare
checkpoints (scripts/evaluate_test_set.py's tile_metrics) are two different,
easily-conflated notions of "how well is the model doing," and neither had
a unit test before this.

Convention: when both prediction and ground truth are empty (no oil pixels
at all), IoU/Dice/precision/recall are all defined as 1.0 -- a model that
correctly predicts "no oil" on a genuinely oil-free tile scored a perfect
result, not penalized (0.0) or hidden (NaN). This matches
scripts/evaluate_test_set.py's existing tile_metrics convention for IoU/Dice;
precision/recall get the same treatment here for consistency, since an
empty/empty tile has no false positives or false negatives to speak of.
"""

from __future__ import annotations

import numpy as np


def binarize(probs: np.ndarray, threshold: float) -> np.ndarray:
    return probs > threshold


def iou_dice(pred_binary: np.ndarray, gt_binary: np.ndarray, valid_mask: np.ndarray | None = None) -> tuple[float, float]:
    """IoU and Dice between two same-shape boolean arrays.

    valid_mask, if given, is a same-shape boolean array (True = valid pixel,
    e.g. from ~preprocess.compute_nodata_mask) -- invalid pixels are dropped
    via boolean indexing BEFORE any sum is taken, i.e. actually excluded from
    both arrays, not zeroed in place (zeroing a boolean "is this pixel oil"
    array by definition means "not oil," which would misrepresent an unknown
    nodata pixel as a confirmed true negative)."""
    if valid_mask is not None:
        pred_binary = pred_binary[valid_mask]
        gt_binary = gt_binary[valid_mask]
    intersection = int(np.logical_and(pred_binary, gt_binary).sum())
    union = int(np.logical_or(pred_binary, gt_binary).sum())
    if union == 0:
        return 1.0, 1.0
    iou = intersection / union
    denom = int(pred_binary.sum()) + int(gt_binary.sum())
    dice = (2 * intersection / denom) if denom > 0 else 1.0
    return float(iou), float(dice)


def precision_recall(pred_binary: np.ndarray, gt_binary: np.ndarray, valid_mask: np.ndarray | None = None) -> tuple[float, float]:
    """Precision/recall for the positive (oil) class. valid_mask: see iou_dice."""
    if valid_mask is not None:
        pred_binary = pred_binary[valid_mask]
        gt_binary = gt_binary[valid_mask]
    tp = int(np.logical_and(pred_binary, gt_binary).sum())
    fp = int(np.logical_and(pred_binary, ~gt_binary).sum())
    fn = int(np.logical_and(~pred_binary, gt_binary).sum())
    if tp + fp == 0 and tp + fn == 0:
        return 1.0, 1.0
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    return float(precision), float(recall)


def tile_metrics(probs: np.ndarray, gt: np.ndarray, threshold: float, valid_mask: np.ndarray | None = None) -> dict:
    """Full metric set for one tile at one threshold.

    probs: raw sigmoid probabilities in [0, 1] (NOT yet thresholded).
    gt: raw ground-truth mask array (oil = any value > 0).
    valid_mask: same-shape boolean array, True = valid pixel (see iou_dice).
    Defaults to None (every pixel valid), which reproduces the exact prior
    behavior of this function for every existing caller.
    """
    pred_b = binarize(probs, threshold)
    gt_b = gt > 0
    iou, dice = iou_dice(pred_b, gt_b, valid_mask=valid_mask)
    precision, recall = precision_recall(pred_b, gt_b, valid_mask=valid_mask)
    if valid_mask is not None:
        pred_b_valid = pred_b[valid_mask]
        gt_b_valid = gt_b[valid_mask]
    else:
        pred_b_valid = pred_b
        gt_b_valid = gt_b
    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "has_oil": bool(gt_b_valid.any()),
        "pred_positive_fraction": float(pred_b_valid.mean()) if pred_b_valid.size else 0.0,
        "gt_positive_fraction": float(gt_b_valid.mean()) if gt_b_valid.size else 0.0,
    }


def aggregate_global(pairs: list[tuple[np.ndarray, np.ndarray]], threshold: float, valid_masks: list[np.ndarray] | None = None) -> dict:
    """Pixel-accumulated ("global") IoU/Dice/precision/recall over a whole set of
    (probs, gt) tile pairs: sums TP/FP/FN/TN across ALL tiles first, then computes
    each ratio once -- as opposed to averaging per-tile ratios (see tile_metrics
    used per-tile then meaned by a caller). This is the cross-check for whether
    per-tile averaging, which implicitly weights a tile with 1 oil pixel the same
    as a tile that's 90% oil, is distorting a per-tile-mean headline number.

    valid_masks, if given, is a list parallel to `pairs` (same length, same
    order) of same-shape boolean arrays, True = valid pixel (see iou_dice).
    Defaults to None (every pixel in every tile valid), reproducing the
    exact prior behavior for every existing caller.
    """
    tp = fp = fn = tn = 0
    for i, (probs, gt) in enumerate(pairs):
        pred_b = binarize(probs, threshold)
        gt_b = gt > 0
        vm = valid_masks[i] if valid_masks is not None else None
        if vm is not None:
            pred_b = pred_b[vm]
            gt_b = gt_b[vm]
        tp += int(np.logical_and(pred_b, gt_b).sum())
        fp += int(np.logical_and(pred_b, ~gt_b).sum())
        fn += int(np.logical_and(~pred_b, gt_b).sum())
        tn += int(np.logical_and(~pred_b, ~gt_b).sum())

    union = tp + fp + fn
    iou = 1.0 if union == 0 else tp / union
    dice_denom = 2 * tp + fp + fn
    dice = 1.0 if dice_denom == 0 else 2 * tp / dice_denom
    precision = 1.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 1.0 if (tp + fn) == 0 else tp / (tp + fn)
    return {
        "iou": float(iou), "dice": float(dice),
        "precision": float(precision), "recall": float(recall),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
