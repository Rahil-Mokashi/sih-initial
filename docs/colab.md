# Running training on Colab

See `colab_bootstrap.ipynb` for the actual notebook. This page is the
one-page reference for the parts that aren't self-explanatory in the
notebook itself.

## Expected epoch wall-time

Local (RTX 4050 Laptop GPU, 6GB): ~20-25 min/epoch at batch_size=16,
tile_size=512, num_workers=6 (see LOG.md). A Colab T4 is roughly comparable
in raw throughput for a model this size; an A100 (if assigned) should be
several times faster. Always check the actual per-epoch time printed by
`train.py` for the specific GPU Colab assigns (`nvidia-smi` in cell 1) --
don't assume it matches local without checking.

## Where checkpoints land

`configs/<experiment>.yaml`'s `output_dir` controls this. Point it under
`/content/drive/MyDrive/sih26143-oil-spill-attribution/experiments/<name>/`
(not `/content/repo/...`) so it survives a disconnect -- Colab's local disk
is wiped when the runtime recycles, Drive is not. Inside `output_dir` you
get:

- `epochs/epoch_NNNN.pt` -- full resumable state (model, optimizer,
  scheduler, epoch, RNG state), pruned to the config's `keep_last_n` most
  recent
- `best.pt` -- lightweight (model weights only), saved whenever val_dice
  improves (see docs/metric_audit.md before trusting that number alone --
  always re-check with `scripts/evaluate.py` + `scripts/sweep_threshold.py`)
- `metrics.jsonl` -- one flushed JSON line per epoch; survives even if the
  session dies before any checkpoint write completes
- `run_manifest.json` -- git SHA, dirty-tree state, full resolved config,
  Python/torch/CUDA versions, GPU name, timestamp -- written once at the
  start of the run

## Resuming after a disconnect

Just re-run cell 6 with the same config (same `output_dir`). `train.py`
auto-detects existing `epochs/epoch_*.pt` files in `output_dir` and resumes
from the newest one, including RNG state -- nothing to configure manually.
If cell 6's config also needs data or the protected checkpoint, re-run
cells 3-5 first (a fresh Colab runtime has nothing on local disk).

## Pulling `metrics.jsonl` back for local analysis

```
# from a local terminal, after mounting/syncing the same Drive folder,
# or via `drive.mount` + download in a Colab cell:
cp /content/drive/MyDrive/sih26143-oil-spill-attribution/experiments/<name>/metrics.jsonl .
```

It's plain JSON-lines, one epoch per line -- load with
`[json.loads(l) for l in open('metrics.jsonl')]` or `pandas.read_json(path, lines=True)`.

## Version mismatch = invalid comparison

`requirements-colab.txt` pins exact versions matching the local environment
that produced the epoch-39 baseline. If Colab can't satisfy a pin (unlikely
for CPU-side packages, more plausible for the CUDA-specific torch wheel),
record the actual installed versions in that run's `run_manifest.json`
(already automatic) and say so explicitly in the LOG.md entry for that
experiment -- don't present a mismatched-version run as directly comparable
to 0.1057 without that caveat.

## Dataset: downloaded fresh each session, not staged via Drive

Cell 5 re-downloads the Zenodo dataset directly (public, no auth needed)
onto Colab's local disk every session, rather than requiring you to
pre-upload a ~38GB archive to Drive. This trades a slower cell-5 (however
long the Zenodo download takes) for zero one-time setup burden. If repeated
downloads become the bottleneck, the next step would be caching the
already-tiled/extracted data on Drive and reading from there instead --
not built yet, since Phase 1 doesn't require it and it's easy to add later
without touching anything else in this notebook.
