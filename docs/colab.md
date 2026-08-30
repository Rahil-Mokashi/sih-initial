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

## Dataset: cached on Drive after the first download, timed either way

Cell 5 downloads the Zenodo dataset directly (public, no auth needed) the
FIRST time it runs, then tars the extracted result (per-image GeoTIFFs +
`train_manifest.csv`/`val_manifest.csv` -- there's no separate "tiles"
artifact to cache, since `ZenodoTileDataset` reads 512x512 windows on the
fly at train time rather than materializing them to disk) and writes it to
`DRIVE_PROJECT_DIR/dataset_cache/zenodo_extracted.tar`. Every subsequent
run (including after a Colab disconnect kills the session mid-training)
checks that path first and extracts from there instead of re-downloading.

**Set `FORCE_FRESH_DOWNLOAD = True`** in cell 0 to bypass the cache
entirely -- e.g. if the download/extract scripts change, or you suspect the
cached archive is stale or corrupt. There is no automatic staleness
check; this flag is the only invalidation mechanism.

**Real cost is measured, not assumed** -- cell 5 prints elapsed wall time
and data volume for whichever path it actually took (`source=Drive cache`
or `source=Zenodo (fresh download)`), plus a per-step breakdown for the
fresh-download path. Record the first real number of each kind here once
you have it:

| path | elapsed | data volume |
|---|---|---|
| fresh download + extract + build + write cache | *(fill in after first real run)* | *(fill in)* |
| Drive cache hit (copy + extract) | *(fill in after first real run)* | *(fill in)* |

**Before relying on this, know the real constraint it runs into**: the
combined Part I+II+III extracted dataset is on the order of ~94GB
(README's documented ~38GB Part I images archive alone, plus Part II's
~23GB+23GB and Part III's ~10GB). A tar of the extracted (uncompressed
float32 GeoTIFF) result is comparable in size, not smaller -- SAR
backscatter data doesn't compress well, which is also why cell 5 tars
without gzip (`tarfile.open(..., 'w')` not `'w:gz'`): compression would
just cost CPU time for negligible size reduction. **A free Google Drive
account only gets 15GB of storage** -- nowhere near enough to hold this
cache. If your Drive is on the free tier, `FORCE_FRESH_DOWNLOAD` effectively
has to stay `True` (the cache write in cell 5 will fail or fill your Drive
long before finishing), and the fresh-download path's own timing numbers
above are what actually matters for planning around Colab disconnects --
not the cache. This wasn't sized down further (e.g. caching only Part I)
without being asked to, since which parts you actually need depends on the
experiment.
