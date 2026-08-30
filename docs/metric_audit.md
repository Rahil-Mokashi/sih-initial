# Metric audit (Phase 0, Gate B)

Question this answers: is the real oil-tiles-only IoU number (0.1057, from
`scripts/compare_checkpoints_on_val.py`) trustworthy, and were the three
tuning trials (`oversample`, `tversky`, `posweight_sched`) genuinely worse,
or is the measurement itself broken?

There are **two distinct metrics** in this codebase, easy to conflate. Only
one of them is what actually produced the 0.1057 comparison.

## Metric 1: the training-loop's raw `val_dice` (src/detection/train.py's `evaluate()`)

This is the number printed every epoch during training and used to decide
when to save `best_unet_resnet18.pt`. It calls `1 - dice_loss(logits, masks)`
from `src/detection/losses.py`, where `dice_loss`:

- Operates on **raw sigmoid probabilities**, never thresholded.
- Is computed **per-tile** (flattened over all pixels in that tile,
  foreground+background together — not oil-pixels-only), then averaged
  over the batch.
- Includes **every tile**, including the ~82% of tiles with zero
  ground-truth oil pixels (see Gate C.4). There is no special-casing for
  empty-GT tiles; the formula `(2*intersection+eps)/(union+eps)` is applied
  uniformly.

**This has a real, previously-undiagnosed mathematical flaw.** For a tile
with zero ground-truth oil, `intersection = 0` and `union = probs.sum()`
(background probabilities only), so `dice = eps/(probs.sum()+eps)`. This
collapses toward 0 for almost *any* nonzero background probability — I
verified this numerically:

| mean background probability on an empty-GT tile | soft-dice score |
|---|---|
| 0.001 | 0.0000 |
| 0.01 | 0.0000 |
| 0.05 | 0.0000 |
| 0.1 | 0.0000 |
| near-zero (sigmoid(-20)) | 0.0018 (not 1.0) |

So an empty-GT tile can **never** score meaningfully close to 1.0 under
this metric, regardless of how "correct" the prediction is — only a
near-perfectly-zero prediction gets even close, and even then it tops out
around 0.002, not 1.0. With ~82% of every batch being empty-GT tiles, the
reported `val_dice` is dominated by a near-zero floor that has nothing to
do with real detection quality. This is why raw `val_dice` looked flat
(~0.022–0.028) across the ORIGINAL checkpoint, `oversample`, `tversky`, AND
`posweight_sched` alike — the metric is structurally incapable of showing
real differences at that scale. **This part of the project's prior
diagnosis ("val_dice looked broken") was right, and now has a precise,
verified mechanism, not just a "wrong baseline units" explanation.**

**Consequence for checkpoint selection**: `best_unet_resnet18.pt` is saved
whenever this broken `val_dice` improves. That means the "best epoch"
selected during training may not correspond to the best epoch by real
oil-IoU either — there's no way to check retroactively since only
best/latest/final checkpoints were kept historically (not one per epoch).
This is a related, separate risk worth keeping in mind, not something this
audit can fully resolve after the fact.

## Metric 2: the "real IoU" (`scripts/evaluate_test_set.py` / `compare_checkpoints_on_val.py`, and now `scripts/evaluate.py`)

This is what actually produced 0.1057 vs. 0.0828 vs. 0.0800. Audited
against the four specific questions:

1. **Class handling**: the headline `oil_tiles_only_mean_iou` filters to
   tiles whose ground truth has ANY oil pixel, then computes standard
   binary IoU (oil vs. not-oil) over the whole tile. This is the right
   scope for the headline number — not diluted by background pixels within
   an oil tile, and not including trivially-empty tiles.
2. **Threshold**: swept across `[0.5, 0.35, 0.25, 0.22, 0.2, 0.18]`
   historically (now extended to 0.05–0.50 in steps of 0.025 via
   `scripts/sweep_threshold.py`) — never hardcoded to 0.5. The reported
   0.1057 is the swept best (threshold 0.35), an intentional, disclosed
   choice.
3. **Empty-mask behavior**: both-empty is explicitly defined as IoU=Dice=1.0
   (`union == 0` branch). This is a defensible, disclosed convention
   ("correct rejection is perfect"), and it does NOT affect the
   `oil_tiles_only` headline number since that number excludes empty-GT
   tiles by construction. It does affect the separate `no_oil`/`lookalike`
   per-class numbers.
4. **Aggregation**: per-tile mean (arithmetic mean of each tile's own IoU),
   not global pixel accumulation. This is a legitimate open question — a
   1-pixel oil tile and a 90%-oil tile get equal weight in the mean. I
   added `src/detection/metrics.py::aggregate_global` specifically to
   cross-check this and re-ran it against the real checkpoints (see
   below): **the two aggregation methods agree on the same ranking and the
   same rough threshold neighborhood (0.35–0.45)**, so per-tile averaging
   is not the thing driving the original conclusion, though the *exact*
   optimal threshold shifts slightly (0.35 per-tile vs. ~0.425–0.45
   global).

**Verdict: 0.1057 is a trustworthy, real number.** I independently
reproduced it (0.1074, using the exact epoch-39 checkpoint weights
preserved in `latest_unet_resnet18_epoch39_backup.pt` — see note below on
why that distinction matters) using a from-scratch reimplementation
(`src/detection/metrics.py`, unit-tested — see `tests/test_metrics.py`),
not just re-running the original code.

### Important side-finding: precision is very low even at the best threshold

Re-running the baseline with the new evaluator (which reports
precision/recall, which the old scripts didn't) shows the 0.1057-class
number comes from a **low-precision, over-triggering** model, not a
precise one:

- At threshold 0.35 (the "best" one): Precision=0.139, Recall=0.476,
  predicted-positive-fraction=0.290 vs. ground-truth-positive-fraction=0.079
  on oil tiles (predicting ~3.7x more oil pixels than actually exist).
- On `no_oil` source images (should be 0% predicted positive):
  predicted-positive-fraction=0.160.
- On `lookalike` source images (should also be 0%):
  predicted-positive-fraction=0.224.

Precision never exceeds ~0.17 anywhere across the full 0.05–0.50 sweep.
This means the IoU=0.1057 checkpoint is a real, above-trivial-baseline
detector, but a broadly over-triggering one, not a precise one — this
project never had a precision/recall view before (only IoU/Dice), so this
specific characterization is new.

## Important correction to the checkpoint used for comparison

The current `data/processed/checkpoints/best_unet_resnet18.pt` is **epoch
44** (from the LR-scheduler run that has continued since the 2026-08-27
pivot), not epoch 39. The real epoch-39 weights only survive in
`latest_unet_resnet18_epoch39_backup.pt`. All "baseline" numbers in this
audit and in Gate B's re-scoring table use that backup file specifically —
using the current `best_unet_resnet18.pt` would silently compare against a
different (later, not yet independently IoU-verified) checkpoint.

## Re-scoring the three dead trials (Gate B.5)

All four checkpoints evaluated with the identical new pipeline
(`scripts/evaluate.py` + `scripts/sweep_threshold.py`), val set, best
threshold per checkpoint (not 0.5):

| checkpoint | best threshold | oil IoU (per-tile) | oil Dice | Precision | Recall | pred-pos-frac (oil tiles) | gt-pos-frac | failure mode |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| epoch-39 baseline (`latest_unet_resnet18_epoch39_backup.pt`) | 0.35 | **0.1074** | 0.1691 | 0.139 | 0.476 | 0.290 | 0.079 | none (graded response), but low precision |
| `trial_oversample` | 0.475 | 0.0852 | 0.1414 | 0.0995 | 0.668 | 0.961 (at 0.35) | 0.079 | collapsed toward predicting almost everything as oil (pred-pos 0.66–0.96 across ALL classes incl. no_oil/lookalike) |
| `trial_tversky` | 0.5 (**flat at 0.0797 across the ENTIRE 0.05–0.50 sweep**) | 0.0797 | 0.1302 | 0.0797 | 0.9999 | 0.94–0.99 | 0.079 | fully collapsed — recall≈1.0 and IoU identical to 4 decimal places at every threshold tested, i.e. the raw probability output is saturated near 1 almost everywhere |
| `trial_posweight_sched` | 0.10 | 0.0805 | 0.1319 | 0.0817 | 0.964 (at 0.10) | 0.042 (at threshold 0.35) | 0.079 | opposite collapse from the other two — under-predicts (at the baseline's own operating threshold of 0.35, predicts about half the true oil-pixel rate); only trained 1 epoch before being killed, so this comparison is not fully fair to it |

**All three trials score below the epoch-39 baseline at their own best
threshold, not just at 0.5 or at the baseline's threshold.** The original
LOG.md conclusion holds up under a from-scratch, unit-tested, precision/
recall-aware re-implementation of the metric.

On the specific question of what `posweight_sched`'s poor score was made
of: **at the threshold that makes it look best (0.10), it's over-predicting
like the other two** (recall 0.964, meaning most pixels above a very low
bar get flagged) — its raw probabilities are just generally lower-confidence
across the board (consistent with pos_weight being turned down, which
directly reduces the confidence push on the positive class), not
qualitatively "predicting nothing." At threshold 0.35 (matching the other
three), it visibly under-predicts (pred-pos 0.042 vs. gt 0.079). Given it
only got 1 epoch of training before being killed, it's the least
conclusive of the three trials, but nothing here suggests it would have
beaten the baseline with more epochs — it never established the kind of
graded, discriminating threshold response the baseline shows.

## Aggregation cross-check (per-tile mean vs. global pixel-accumulated)

Re-running the epoch-39 sweep with `aggregate_global` alongside the
per-tile mean:

| threshold | IoU (per-tile mean) | IoU (global) |
|---:|---:|---:|
| 0.35 | 0.1074 | 0.1342 |
| 0.40 | 0.1012 | 0.1436 |
| 0.425 | 0.0979 | 0.1441 |
| 0.45 | 0.0936 | 0.1439 |

Global IoU is consistently higher than per-tile mean (expected — a few
large, well-segmented oil tiles pull the pixel-weighted number up more than
they pull the tile-count-weighted mean up), and its optimum sits slightly
higher (~0.425–0.45) than the per-tile optimum (0.35), but **both methods
agree the baseline is meaningfully better than all three trials** — this
was checked, not assumed.
