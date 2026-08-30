# SIH26143 — Oil Spill Detection & Vessel Attribution

**Problem statement (NTRO):** detect oil spills in Sentinel-1 SAR satellite imagery, trace
the spill backward through wind and ocean current data to estimate where and when it
started, then cross-reference AIS vessel tracking data to produce a **ranked,
evidence-backed list of suspect vessels — not a single verdict.**

> For day-to-day project state, read [`LOG.md`](LOG.md) (what was done, session by
> session) and [`DECISIONS.md`](DECISIONS.md) (why, with tradeoffs) — this README is
> the "how do I run this" reference, those two are the "what's actually true right now"
> reference. They're kept current; this file may lag behind the latest session.

## Pipeline overview

```
SAR image  ──▶  detection model  ──▶  slick mask  ──▶  geometric characterization
                                          │              (area/length/width/orientation/shape)
                                          ▼
                              backward drift/advection (hindcast)
                          (ERA5 wind + HYCOM currents + Ekman)         forward drift/advection (forecast)
                                          │                                        │
                                          ▼                                        ▼
                              estimated origin region + time          predicted future slick position
                                          │
                                          ▼
                AIS vessel presence + trajectory + AIS-gap behavior
                          (Global Fishing Watch, full candidate pool)
                                          │
                                          ▼
                ranked, evidence-backed suspect list (proximity, timing,
                    trajectory, behavioral anomaly all shown separately)
```

Each stage is real and independently runnable — see below.

## Setup

```
# 1. Create/activate the venv (already created at venv/ if you're continuing this project)
python -m venv venv

# 2. Install PyTorch FIRST with the CUDA wheel for your GPU (plain `pip install torch`
#    gives you a CPU-only build)
venv\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Everything else
venv\Scripts\pip.exe install -r requirements.txt
```

Verify: `venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"`

### API keys (`.env`)

Copy real values into a `.env` file at the repo root (gitignored):

```
GFW_API_TOKEN=your-global-fishing-watch-token
```

- **Global Fishing Watch**: register at https://globalfishingwatch.org/our-apis, generate
  a token at https://globalfishingwatch.org/our-apis/tokens.
- **ERA5 / Copernicus CDS**: register at https://cds.climate.copernicus.eu, set up
  `~/.cdsapirc` per [DECISIONS.md](DECISIONS.md#wind-era5-via-copernicus-cds--requires-a-free-account-pending-like-gfw)
  (must be an actual **file**, not a folder — a real mistake hit during this project),
  and accept the dataset's license on its page before the API will serve data.
- **HYCOM** (ocean currents): no account needed, fully open.

## Repo structure

```
/data/raw            untouched downloaded datasets (gitignored)
/data/processed       despeckled/tiled data, checkpoints, dashboard JSON (gitignored)
/src/detection         SAR preprocessing + segmentation model + training loop + mask geometry
/src/drift              backward (hindcast) + forward (forecast) drift/advection (wind + currents)
/src/attribution        AIS scoring (proximity, timing, trajectory, AIS-gap behavior) and ranking
/src/dashboard           builds the real dashboard (map + panels)
/src/common              tiny shared utilities (geo math, vessel-identity anonymization)
/scripts                 one-off scripts: downloads, sanity checks, orchestration
/tests                   pytest suite (pure-logic pieces -- see "Tests" below)
/notebooks                exploration / debugging (currently empty)
/docs                     (currently empty)
```

## Running each stage

**Detection (SAR → oil mask)**
```
venv\Scripts\python.exe scripts\download_zenodo_sample.py          # Zenodo Part I (oil-positive)
venv\Scripts\python.exe scripts\download_zenodo_part2_parallel.py  # Part II Lookalike images (fast, resumable)
venv\Scripts\python.exe scripts\download_zenodo_part2_part3.py     # Part II remainder + Part III (test set)
venv\Scripts\python.exe scripts\extract_zenodo_part2.py            # extract Part II (No_oil + Lookalike)
venv\Scripts\python.exe scripts\extract_zenodo_part3.py            # extract Part III (held-out test set)
venv\Scripts\python.exe scripts\build_training_pool.py             # verify + build train/val manifests
venv\Scripts\python.exe scripts\train_detection.py                 # real training run
venv\Scripts\python.exe scripts\evaluate_test_set.py               # real held-out accuracy (IoU/Dice) on Part III
venv\Scripts\python.exe scripts\render_detection_overlay.py        # overlay image + real mask geometry (area/length/width/orientation)
```
Architecture: U-Net + ResNet18 encoder (ImageNet-pretrained), Dice + weighted-BCE loss
(or `--loss tversky`, see below) by default, mixed precision, `num_workers>0` parallel
tile loading (single-threaded reads left the GPU starved on I/O — see DECISIONS.md).

**Config-driven training (Phase 1 infra) + Colab**: `train.py --config configs/<name>.yaml`
is a newer, config-driven entrypoint (seeding, `run_manifest.json`, per-epoch/RNG-state
resume) built for running experiments on Colab — see `docs/colab.md` and
`colab_bootstrap.ipynb`. It wraps the same `src/detection/train.py` loop `scripts/train_detection.py`
above uses, so both stay valid. **A dependency-version mismatch against `requirements-colab.txt`'s
pins invalidates any direct comparison to the epoch-39 baseline** (real oil-tiles-only IoU
0.1057, re-verified as 0.1074 in `docs/metric_audit.md`) — always check `run_manifest.json`'s
recorded versions before comparing two runs.
Zenodo downloads use a 4-connection resumable downloader
(`scripts/_zenodo_download_utils.py`'s `download_parallel`) rather than one slow
single-stream connection — see DECISIONS.md "Corrupted archive diagnosis and the
parallel-download-safety fix" for why, including a real corruption bug it was written to
fix. `train_detection.py` also resumes from the last completed epoch automatically if
interrupted (crash, sleep, etc.) rather than restarting from scratch.

`train_detection.py` is CLI-configurable, not hardcoded, for real experiment trials
(`--loss {dicebce,tversky}`, `--pos-weight`, `--oversample-oil-tiles`
`--target-oil-fraction`, `--use-lr-scheduler`, `--tag` to sandbox a trial's checkpoints
under `checkpoints/<tag>/` instead of the real top-level files, `--fresh` to force
epoch-1 rather than resuming) — see DECISIONS.md for why: the original 60-epoch run
plateaued *below* a trivial "predict all oil" baseline, and `scripts\
analyze_tile_oil_distribution.py` found 82.4% of real training tiles have zero oil
pixels at all, a signal-sparsity issue `--oversample-oil-tiles` addresses via a
`WeightedRandomSampler`.

Real geometric characterization (`src/detection/geometry.py`, `cv2` contour/
`minAreaRect`) computes area, length, width, orientation, and elongation from a mask —
pixel units only; real-world km²/m conversion is deliberately not applied since the
Zenodo dataset's own record doesn't document a Sentinel-1 product type or ground
resolution (checked directly, not assumed — see DECISIONS.md).

**Drift (backward hindcast + forward forecast advection)**
```
venv\Scripts\python.exe scripts\run_drift_skeleton.py               # prints origin estimate, ERA5 vs NCEP/NCAR
venv\Scripts\python.exe scripts\export_drift_dashboard_data.py      # dumps real particle tracks (both directions) to JSON for the dashboard
```
Edit `CASE_ID` / the `CASES` dict at the top of `export_drift_dashboard_data.py` to run a
different real case — validated on two so far (ow-0001, ow-0002) with zero methodology
changes between them. `src/drift/advect.py` runs the SAME physics both directions
(`backward_advect`/`forward_advect` share one `_advect()` core, differing only in a
`direction` sign) — backward reconstructs where the slick came from (hindcast), forward
predicts where it's headed (forecast), per the SIH problem statement's "trace... backward
AND forward" ask.

**Attribution (vessel ranking: proximity, timing, trajectory, behavioral anomaly)**
```
venv\Scripts\python.exe scripts\test_gfw_api.py            # confirm GFW auth + 4wings/report request format
venv\Scripts\python.exe scripts\test_gfw_events_api.py     # confirm v3/events real schema (course/speed/gap fields)
venv\Scripts\python.exe src\attribution\score_vessels.py   # real ranked vessel list for every case with a drift_{case}.json
```
Scoring has four real, separately-visible dimensions per vessel (not a black-box score):
distance + timing vs. the drift-estimated origin (`score`), trajectory (real presence
records across the whole query window, not just the closest one), and behavioral anomaly
(real AIS-gap check via GFW's v3/events, including GFW's own "intentional disabling" flag)
folded into a separate `composite_score` the ranking is sorted by. Behavioral/trajectory
checks run against the FULL raw candidate pool (hundreds to 1000+ vessels), not a
pre-filtered subset — GFW's `vessels[]` filter batches up to 20 IDs per request, so this
costs `ceil(n/20)` real requests, not one per vessel (confirmed by direct test — see
DECISIONS.md "Full-pool behavioral rescoring"). Course/heading and instantaneous
per-position speed are confirmed NOT available at this API tier (checked directly against
real responses, not assumed) — see DECISIONS.md. Not calibrated against labeled ground
truth; none exists for this problem, a permanent limitation, not a gap more tuning fixes.

**Dashboard**
```
venv\Scripts\python.exe src\dashboard\build_map.py           # real Leaflet map from drift JSON (backward + forward tracks)
venv\Scripts\python.exe src\dashboard\build_dashboard.py     # full page (map + status + panels + geometry + report export)
```
Add `--anonymize` to either command to build to `output_anon/` instead, with real vessel
names/MMSI/IMO replaced by fictional stand-ins (`src/common/anonymize.py`) — for a deck,
recording, or screenshot that may leave the room. The default (no flag) `output/` build
keeps real identities and is meant to be shown live, not shared onward — see DECISIONS.md.

Open `src/dashboard/output/index.html` in a real browser (`map.html` alongside it, same
folder) — the map's basemap tiles are loaded from external servers, so a sandboxed
preview (e.g. a chat tool's inline HTML viewer) may not render them. The dashboard's
"Export Report (PDF)" button uses the browser's native print-to-PDF (no server, no new
dependency) — enable "background graphics" in the print dialog to keep the dark theme.

## Tests

```
venv\Scripts\python.exe -m pytest tests\ -v
```

Covers the pure-logic pieces (geo math, GFW scoring math, bbox-to-mask rasterization, Ekman
deflection rotation, mask geometry) and specifically regression-tests `cycle_and_tau`'s
tau=24 rounding edge case, which was a real bug hit during development — see
DECISIONS.md "Drift model approach". Doesn't cover the network-dependent code (data
downloads, HYCOM/ERA5/GFW fetches) or model training — those are exercised for real by
the scripts above instead. 28 tests as of the last update (22 original + 6 for
`src/detection/geometry.py`).

## Current status

Real, working, and verified on real data: environment/GPU, all three raw data sources
(2570 real training images across oil/no_oil/lookalike, plus a 450-image held-out test
set), the drift model in BOTH directions (Ekman-corrected backward hindcast and forward
forecast, validated on two independent real cases against both ERA5 and NCEP/NCAR wind),
real geometric mask characterization, GFW API integration with real proximity + timing +
trajectory + AIS-gap-behavioral scoring against the FULL candidate pool (not a
pre-filtered subset), a 28-test pytest suite, an anonymized-build mode for anything that
leaves the room, a print-to-PDF report export, and a dashboard with real drift-animation
playback (both track directions), wind-vector overlay, data-provenance panel,
sortable/expandable vessel ranking, geometric-characterization panel, and a multi-case
comparison table. The one piece still weak: the detection model's real accuracy (see
LOG.md for the current state — a 60-epoch run plateaued below a trivial baseline;
diagnosis pointed at signal sparsity and loss/pos_weight interaction, and CLI-configurable
trial runs to fix it are documented there, not yet concluded as of the last update). Once
a real fix lands, `scripts/evaluate_test_set.py` gives the real held-out accuracy number
and the dashboard's detection panel gets the improved checkpoint swapped in. Full detail
— including bugs hit and fixed, and what's still open — is in LOG.md and DECISIONS.md.

## Known limitations (see DECISIONS.md for full writeups)

- Drift physics: simplified windage model (3% of wind speed) plus an empirical 20-degree
  Ekman/Coriolis deflection — real and hemisphere-aware, but still a simplification, not
  a full ocean-physics model. The forward forecast shares this same simplification (same
  physics, opposite time direction), so forecast uncertainty compounds the same way
  hindcast uncertainty does.
- Ocean currents: HYCOM barotropic (depth-averaged) velocity, not true 0m surface velocity.
- Attribution scoring weights (proximity, timing, and the AIS-gap behavioral bonus) are a
  documented first pass, not calibrated against any labeled ground truth — no such labeled
  "this vessel caused this spill" dataset exists for this problem, so this is a permanent,
  accepted limitation rather than something more training or tuning resolves.
- Vessel course/heading and instantaneous per-position speed are confirmed NOT available
  from GFW's API at this access tier (checked directly against real v3/events responses,
  all 5 event types) — a real external constraint on how far "trajectory" scoring can go,
  not an implementation shortcut.
- Geometric characterization is pixel-units only — the Zenodo training dataset's own
  record doesn't document a Sentinel-1 product type or ground resolution, so no km²/m
  conversion is applied (checked directly against the dataset's record page, not assumed).
- Detection model accuracy is genuinely weak and the active subject of tuning trials —
  read LOG.md for the current real numbers rather than trusting this file, which may lag.
