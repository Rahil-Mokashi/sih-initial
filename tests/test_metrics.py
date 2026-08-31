"""
Unit tests for src/detection/metrics.py -- Phase 0.2 of the metric audit
(docs/metric_audit.md). Hand-computed cases, checked against the documented
convention rather than assumed.
"""

import math

import numpy as np

from detection.metrics import aggregate_global, iou_dice, precision_recall, tile_metrics


def test_perfect_overlap_gives_iou_and_dice_1():
    pred = np.array([[True, True], [False, False]])
    gt = np.array([[True, True], [False, False]])
    iou, dice = iou_dice(pred, gt)
    assert iou == 1.0
    assert dice == 1.0


def test_zero_overlap_both_nonempty_gives_0():
    pred = np.array([[True, False], [False, False]])
    gt = np.array([[False, True], [False, False]])
    iou, dice = iou_dice(pred, gt)
    assert iou == 0.0
    assert dice == 0.0


def test_half_overlap_hand_computed():
    # 4x4 grid: pred is the left half (8 px), gt is the top half (8 px).
    # Overlap (top-left quadrant) = 4 px. Union = 8 + 8 - 4 = 12.
    pred = np.zeros((4, 4), dtype=bool)
    pred[:, :2] = True
    gt = np.zeros((4, 4), dtype=bool)
    gt[:2, :] = True
    iou, dice = iou_dice(pred, gt)
    assert iou == 4 / 12
    assert dice == (2 * 4) / (8 + 8)


def test_empty_prediction_nonempty_gt_gives_0():
    pred = np.zeros((4, 4), dtype=bool)
    gt = np.zeros((4, 4), dtype=bool)
    gt[0, 0] = True
    iou, dice = iou_dice(pred, gt)
    assert iou == 0.0
    assert dice == 0.0
    precision, recall = precision_recall(pred, gt)
    assert precision == 0.0
    assert recall == 0.0


def test_empty_prediction_empty_gt_gives_documented_convention_of_1():
    pred = np.zeros((4, 4), dtype=bool)
    gt = np.zeros((4, 4), dtype=bool)
    iou, dice = iou_dice(pred, gt)
    assert iou == 1.0
    assert dice == 1.0
    precision, recall = precision_recall(pred, gt)
    assert precision == 1.0
    assert recall == 1.0


def test_tile_metrics_thresholds_before_scoring():
    probs = np.array([[0.9, 0.1], [0.4, 0.6]])
    gt = np.array([[1.0, 0.0], [0.0, 1.0]])
    result = tile_metrics(probs, gt, threshold=0.5)
    # pred_binary = probs > 0.5 -> [[T, F], [F, T]], matches gt exactly
    assert result["iou"] == 1.0
    assert result["has_oil"] is True
    assert result["pred_positive_fraction"] == 0.5
    assert result["gt_positive_fraction"] == 0.5


def test_tile_metrics_no_oil_tile_flagged_correctly():
    probs = np.array([[0.01, 0.02], [0.01, 0.01]])
    gt = np.zeros((2, 2))
    result = tile_metrics(probs, gt, threshold=0.5)
    assert result["has_oil"] is False
    assert result["iou"] == 1.0  # correct rejection


def test_iou_dice_excludes_nodata_pixels_and_differs_from_unmasked():
    # 2x2 tile. pred[1,1] is a confident false positive that would drag IoU
    # down to 0 if scored -- mark it nodata and confirm it's excluded, then
    # confirm the unmasked score (same inputs, no mask) really is different,
    # i.e. this isn't just silently ignoring the mask argument.
    pred = np.array([[True, False], [False, True]])
    gt = np.array([[True, False], [False, False]])
    valid_mask = np.array([[True, True], [True, False]])  # bottom-right is nodata

    iou, dice = iou_dice(pred, gt, valid_mask=valid_mask)
    # Hand-computed over the 3 valid pixels only: pred=[T,F,F], gt=[T,F,F] -> perfect match
    assert iou == 1.0
    assert dice == 1.0

    unmasked_iou, unmasked_dice = iou_dice(pred, gt)
    assert unmasked_iou != iou
    assert unmasked_dice != dice


def test_precision_recall_excludes_nodata_pixels():
    pred = np.array([True, True, False])
    gt = np.array([True, False, False])
    valid_mask = np.array([True, False, True])  # drop pred[1]=True/gt[1]=False, a false positive

    precision, recall = precision_recall(pred, gt, valid_mask=valid_mask)
    # Hand-computed over pixels 0, 2 only: pred=[T,F], gt=[T,F] -> tp=1, fp=0, fn=0
    assert precision == 1.0
    assert recall == 1.0

    unmasked_precision, _ = precision_recall(pred, gt)
    assert unmasked_precision != precision  # the excluded false positive would otherwise cost precision


def test_tile_metrics_valid_mask_affects_positive_fractions_too():
    probs = np.array([[0.9, 0.9], [0.1, 0.1]])
    gt = np.array([[1.0, 1.0], [0.0, 0.0]])
    # Mark one of the two predicted-oil pixels as nodata.
    valid_mask = np.array([[True, False], [True, True]])

    result = tile_metrics(probs, gt, threshold=0.5, valid_mask=valid_mask)
    # Hand-computed over the 3 valid pixels: pred=[T,F,F], gt=[T,F,F] -> 1/3 predicted positive
    assert math.isclose(result["pred_positive_fraction"], 1 / 3)
    assert math.isclose(result["gt_positive_fraction"], 1 / 3)

    unmasked = tile_metrics(probs, gt, threshold=0.5)
    assert unmasked["pred_positive_fraction"] != result["pred_positive_fraction"]


def test_aggregate_global_valid_masks_excludes_nodata_across_tiles():
    probs1 = np.array([[0.9, 0.9]])
    gt1 = np.array([[1.0, 0.0]])  # pixel 1 is a false positive
    valid1 = np.array([[True, False]])  # exclude that false positive

    probs2 = np.array([[0.1]])
    gt2 = np.array([[0.0]])
    valid2 = np.array([[True]])

    result = aggregate_global([(probs1, gt1), (probs2, gt2)], threshold=0.5, valid_masks=[valid1, valid2])
    assert result["tp"] == 1
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["tn"] == 1

    unmasked = aggregate_global([(probs1, gt1), (probs2, gt2)], threshold=0.5)
    assert unmasked["fp"] == 1  # the excluded pixel counts as a false positive when not masked


def test_iou_dice_fully_invalid_tile_gives_perfect_score_not_nan():
    # A tile that's 100% nodata (e.g. an edge-of-swath tile -- 74/6176 real
    # validation tiles are exactly this, see scripts/compute_nodata_fraction.py)
    # boolean-indexes down to two EMPTY arrays. union==0 for two empty
    # arrays, which the existing convention already treats as "nothing to
    # score, call it perfect" -- confirm this holds, and specifically that
    # it's a real float, not NaN (np.array([]).sum() is 0, not nan, but
    # confirm the whole path end to end rather than assuming).
    pred = np.array([[True, False], [False, True]])
    gt = np.array([[False, True], [True, False]])
    valid_mask = np.zeros((2, 2), dtype=bool)  # every pixel invalid

    iou, dice = iou_dice(pred, gt, valid_mask=valid_mask)
    assert iou == 1.0 and not math.isnan(iou)
    assert dice == 1.0 and not math.isnan(dice)

    precision, recall = precision_recall(pred, gt, valid_mask=valid_mask)
    assert precision == 1.0 and not math.isnan(precision)
    assert recall == 1.0 and not math.isnan(recall)


def test_tile_metrics_fully_invalid_tile_no_nan_anywhere():
    probs = np.array([[0.9, 0.1], [0.4, 0.6]])
    gt = np.array([[1.0, 0.0], [0.0, 1.0]])
    valid_mask = np.zeros((2, 2), dtype=bool)

    result = tile_metrics(probs, gt, threshold=0.5, valid_mask=valid_mask)
    for key in ("iou", "dice", "precision", "recall", "pred_positive_fraction", "gt_positive_fraction"):
        assert not math.isnan(result[key]), f"{key} is NaN for a fully-invalid tile"
    assert result["has_oil"] is False  # nothing valid to call "has oil"


def test_aggregate_global_fully_invalid_tile_contributes_nothing_not_nan():
    # A fully-invalid tile mixed in with a normal one must not corrupt the
    # running tp/fp/fn/tn accumulation (e.g. via an empty-array sum
    # producing something other than a clean 0).
    probs1 = np.array([[0.9, 0.9]])
    gt1 = np.array([[1.0, 0.0]])
    valid1 = np.ones((1, 2), dtype=bool)

    probs2 = np.array([[0.9, 0.1]])
    gt2 = np.array([[1.0, 0.0]])
    valid2 = np.zeros((1, 2), dtype=bool)  # fully invalid -- must contribute exactly nothing

    result = aggregate_global([(probs1, gt1), (probs2, gt2)], threshold=0.5, valid_masks=[valid1, valid2])
    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 0
    assert result["tn"] == 0
    for key in ("iou", "dice", "precision", "recall"):
        assert not math.isnan(result[key])


def test_aggregate_global_matches_hand_computed_confusion_counts():
    # Two tiles: first is the half-overlap case above (tp=4, fp=4, fn=4, tn=4),
    # second is a perfect-empty tile (tp=fp=fn=0, tn=16).
    probs1 = np.zeros((4, 4))
    probs1[:, :2] = 0.9  # binarizes to the same "left half" prediction
    gt1 = np.zeros((4, 4))
    gt1[:2, :] = 1.0

    probs2 = np.zeros((4, 4))
    gt2 = np.zeros((4, 4))

    result = aggregate_global([(probs1, gt1), (probs2, gt2)], threshold=0.5)
    assert result["tp"] == 4
    assert result["fp"] == 4
    assert result["fn"] == 4
    assert result["tn"] == 4 + 16
    assert result["iou"] == 4 / (4 + 4 + 4)
