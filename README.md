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
SAR image  ──▶  detection model  ──▶  slick mask
                                          │
                                          ▼
                              backward drift/advection
                          (ERA5 wind + HYCOM currents)
                                          │
                                          ▼
                              estimated origin region + time
                                          │
                                          ▼
                    AIS vessel presence (Global Fishing Watch)
                                          │
                                          ▼
                      ranked, evidence-backed suspect list
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
/src/detection         SAR preprocessing + segmentation model + training loop
/src/drift              backward drift/advection model (wind + currents)
/src/attribution        AIS scoring and vessel ranking
/src/dashboard           builds the real dashboard (map + panels)
/src/common              tiny shared utilities (geo math)
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
```
Architecture: U-Net + ResNet18 encoder, Dice + weighted-BCE loss, mixed precision,
`num_workers>0` parallel tile loading (single-threaded reads left the GPU starved on I/O
— see DECISIONS.md). Zenodo downloads use a 4-connection resumable downloader
(`scripts/_zenodo_download_utils.py`'s `download_parallel`) rather than one slow
single-stream connection — see DECISIONS.md "Corrupted archive diagnosis and the
parallel-download-safety fix" for why, including a real corruption bug it was written to
fix. `train_detection.py` also resumes from the last completed epoch automatically if
interrupted (crash, sleep, etc.) rather than restarting from scratch.

**Drift (backward advection)**
```
venv\Scripts\python.exe scripts\run_drift_skeleton.py               # prints origin estimate, ERA5 vs NCEP/NCAR
venv\Scripts\python.exe scripts\export_drift_dashboard_data.py      # dumps real particle tracks to JSON for the dashboard
```
Edit `CASE_ID` / the `CASES` dict at the top of `export_drift_dashboard_data.py` to run a
different real case — validated on two so far (ow-0001, ow-0002) with zero methodology
changes between them.

**Attribution (vessel ranking)**
```
venv\Scripts\python.exe scripts\test_gfw_api.py       # confirm GFW auth + request format
venv\Scripts\python.exe src\attribution\score_vessels.py   # real ranked vessel list for CASE_ID
```
Scoring = distance + timing vs. the drift-estimated origin, both terms visible per vessel
(not a black-box score) — see DECISIONS.md "Attribution scoring" for the exact formula and
its known limitations (not calibrated against labeled ground truth; none exists yet).

**Dashboard**
```
venv\Scripts\python.exe src\dashboard\build_map.py         # real Leaflet map from drift JSON
venv\Scripts\python.exe src\dashboard\build_dashboard.py   # full page (map + status + panels)
```
Open `src/dashboard/output/index.html` in a real browser (`map.html` alongside it, same
folder) — the map's basemap tiles are loaded from external servers, so a sandboxed
preview (e.g. a chat tool's inline HTML viewer) may not render them.

## Tests

```
venv\Scripts\python.exe -m pytest tests\ -v
```

Covers the pure-logic pieces (geo math, GFW scoring math, bbox-to-mask rasterization, Ekman
deflection rotation) and specifically regression-tests `cycle_and_tau`'s tau=24 rounding
edge case, which was a real bug hit during development — see DECISIONS.md "Drift model
approach". Doesn't cover the network-dependent code (data downloads, HYCOM/ERA5/GFW
fetches) or model training — those are exercised for real by the scripts above instead.

## Current status

Real, working, and verified on real data: environment/GPU, all three raw data sources
(2570 real training images across oil/no_oil/lookalike, plus a 450-image held-out test
set), the drift model (Ekman-corrected, validated on two independent real cases against
both ERA5 and NCEP/NCAR wind), GFW API integration, first-pass attribution scoring, a
22-test pytest suite, and a dashboard with a real drift-animation playback, wind-vector
overlay, data-provenance panel, sortable/expandable vessel ranking, and a multi-case
comparison table. The one piece still in progress: the detection model's real training
run (see LOG.md for the current epoch / loss / val_dice — as of the last update, loss
was decreasing steadily and val_dice climbing after an initial slow start). Once it
finishes, `scripts/evaluate_test_set.py` gives the real held-out accuracy number and the
dashboard's detection panel gets the real checkpoint swapped in, replacing the current
sanity-checkpoint placeholder. Full detail — including bugs hit and fixed, and what's
still open — is in LOG.md and DECISIONS.md.

## Known limitations (see DECISIONS.md for full writeups)

- Drift physics: simplified windage model (3% of wind speed) plus an empirical 20-degree
  Ekman/Coriolis deflection — real and hemisphere-aware, but still a simplification, not
  a full ocean-physics model.
- Ocean currents: HYCOM barotropic (depth-averaged) velocity, not true 0m surface velocity.
- Attribution scoring weights are a documented first pass, not calibrated against any
  labeled ground truth — no such labeled "this vessel caused this spill" dataset exists
  for this problem, so this is a permanent, accepted limitation rather than something
  more training or tuning resolves.
- Detection model accuracy is not yet known for real — training is in progress; read
  LOG.md for the current state rather than trusting this file, which may lag.
