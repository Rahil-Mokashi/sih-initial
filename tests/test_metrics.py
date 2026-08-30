"""
Unit tests for src/detection/metrics.py -- Phase 0.2 of the metric audit
(docs/metric_audit.md). Hand-computed cases, checked against the documented
convention rather than assumed.
"""

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
