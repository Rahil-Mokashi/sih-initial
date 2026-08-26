# LOG.md

Session-by-session log of what was actually done. Read this (and
DECISIONS.md) first when starting a new session — this says where things
stand right now; DECISIONS.md says why they got that way.

---

## 2026-08-25 — Step 0: environment + repo setup

**Goal:** environment confirmed working, repo structured, small samples of
all three real data sources downloaded and verified accessible, one image
preprocessed end to end. No model training.

**Done:**
- Confirmed venv (Python 3.10.10) and installed PyTorch with CUDA support;
  `torch.cuda.is_available()` result recorded below.
- Repo structure created per DECISIONS.md.
- `requirements.txt` filled in with the full dependency list.
- `scripts/download_zenodo_sample.py` — pulls a small sample from the
  Sentinel-1 SAR Oil Spill Zenodo dataset (full-dataset flag included but
  off by default).
- `scripts/download_pangaea_sample.py` — pulls a small sample from the
  PANGAEA Eastern Mediterranean dataset.
- `scripts/test_gfw_api.py` — Global Fishing Watch API auth/connectivity
  test for a sample bounding box + date range.
- `src/detection/preprocess.py` — despeckle (Lee filter) + dB calibration
  check + tiling function, run on one sample image.

**Status / what worked:**
- Environment: Python 3.10.10 venv at `venv/`. `torch==2.6.0+cu124`,
  `torch.cuda.is_available() == True`, GPU detected as "NVIDIA GeForce RTX
  4050 Laptop GPU" (6GB VRAM) — **not** the RTX 3060 assumed at the start
  of this task; see DECISIONS.md. All other deps (rasterio, opencv,
  scikit-image, matplotlib, requests, xarray, netCDF4, geopandas, scipy,
  py7zr) installed and import cleanly.
- PANGAEA sample: full success. `data_matrix.tab` (5515 rows, real
  dates/coords) downloaded, plus 5 real image+annotation pairs.
- Zenodo sample: masks downloaded and verified in full (6MB archive, 1201
  masks, extracted 3 and confirmed real 2048x2048 TIFFs). Images could NOT
  be sampled small — see DECISIONS.md for why; a 20MB connectivity-only
  peek was fetched, confirmed non-extractable as expected, then deleted
  (a README.txt in that folder explains why it's gone).
- GFW API test script: written against the verified real API spec (POST
  `/v3/4wings/report`), runs and correctly reports "no token set" with
  registration instructions. **Not yet verified against a real token** —
  no API key exists yet.
- Preprocessing: ran end to end on the real PANGAEA `ow-0001.jpg` sample.
  Calibration check correctly reported it as NOT calibrated dB (it's an
  8-bit quicklook, as expected — this is the check working correctly, not
  a failure). Lee filter visibly despeckled the image while preserving the
  oil-slick streak's edge; tiled into 4 patches of 256x256 from the 640x640
  input. Before/after figure saved to `data/processed/demo_before_after.png`.

**What didn't work / open items:**
- Real per-image sampling from the Zenodo dataset is not possible without
  the full ~38GB download (architectural limitation of the source, not a
  bug in our script).
- GFW API auth is untested — needs a real API key (registration is free,
  see DECISIONS.md for the link).
- No calibrated Sigma0-dB GeoTIFF has been preprocessed yet (only the
  PANGAEA quicklook JPG) — that requires either the full Zenodo download
  or another calibrated single-scene source.

**Next session should:** decide whether to commit to the full ~38GB Zenodo
download now (flip `FULL_DOWNLOAD = True` in
`scripts/download_zenodo_sample.py`) before starting Step 1 (detection
model architecture selection), and get a Global Fishing Watch API key to
unblock verifying `scripts/test_gfw_api.py` against real data.

---

## 2026-08-25 — Step 1: detection training pipeline + sanity pass

**Goal:** build the training pipeline (dataset, model, loss, AMP training
loop) and prove it runs end to end on real data on the RTX 4050 — not a
real training run, just: data loads, no OOM, loss decreases, checkpoint
save/reload works. Full training is still blocked on the 38GB Zenodo
images download.

**Built:**
- `src/detection/dataset.py` — `SARTileDataset` (tiles image/mask array
  pairs, despeckles via the Step 0 Lee filter, min-max normalizes) and
  `bbox_to_mask` (rasterizes a Pascal-VOC bounding box into a pseudo-mask,
  since the only real images we have right now — PANGAEA — carry bbox
  annotations, not pixel masks).
- `src/detection/model.py` — `build_model()`: U-Net + ResNet18 encoder via
  `segmentation_models_pytorch`, `in_channels=1`, ImageNet-pretrained
  encoder. See DECISIONS.md for why ResNet18 over ResNet34/EfficientNet-B0/DeepLabV3+.
- `src/detection/losses.py` — `DiceBCELoss`: Dice + BCE-with-logits, BCE
  `pos_weight` set from real data (see below), not guessed.
- `src/detection/train.py` — training loop: `torch.amp` mixed precision
  from the start, optional gradient accumulation, per-epoch loss/time/peak-VRAM
  logging, checkpoint save + a separate `load_checkpoint` helper.
- `scripts/compute_mask_class_balance.py` — extracts all 1200 real Zenodo
  masks and computes real oil-pixel-fraction statistics.
- `scripts/sanity_train.py` — orchestrates the actual sanity run.

**Results (real numbers, RTX 4050 6GB, torch 2.6.0+cu124):**
- Real class balance from all 1200 Zenodo masks: mean oil fraction
  **2.98%**, median **1.76%**, min **0.12%**, max **57.4%**, zero masks
  with no oil pixels at all. Used to set `pos_weight ≈ 32.6`.
- Sanity training data: 4 real PANGAEA image+bbox pairs (5th requested row
  was a duplicate image, correctly deduped) → despeckled + bbox-rasterized
  pseudo-masks.
- **256x256 tiles**: 16 tiles from 4 images. 5 epochs, loss `2.037 → 1.717
  → 1.520 → 1.410 → 1.373` (monotonically decreasing). ~0.4s/epoch after
  warmup. **Peak VRAM: 354MB** (~6% of the 6GB budget).
- **512x512 tiles**: 4 tiles from 4 images (only one 512 tile fits per
  640x640 source image). 5 epochs, loss `1.616 → 1.610 → 1.518 → 1.439 →
  1.391` (decreasing). ~0.1-0.3s/epoch. **Peak VRAM: 818MB** (~13% of
  budget).
- Checkpoint (`data/processed/checkpoints/sanity_unet_resnet18.pt`, 172MB
  — model + Adam optimizer state) saved after the 256px run, successfully
  reloaded into a fresh model instance and confirmed.
- No OOM at either tile size, at batch_size=2 with grad_accum_steps=2
  (effective batch 4).

**Verdict on tile size / batch size:** both 256 and 512 are safe right now
with large headroom — this isn't yet a meaningful stress test since the
sanity dataset is only 4 source images. Once the full Zenodo dataset (1200
real 2048x2048 images) is downloaded, re-run `scripts/sanity_train.py`-style
timing/VRAM checks at realistic batch sizes (this sanity pass used
batch_size=2; the ~13% VRAM use at 512px suggests batch_size=8-16 at 512px
or larger at 256px should still fit, but that needs to be confirmed against
real data volume, not assumed from this small a sample).

**What didn't work / open items:**
- The 1201 Zenodo masks still have no matching real images — only used
  indirectly (class-balance stats), not as training pairs.
- The sanity-trained model itself is not meaningful (4 images, rectangular
  pseudo-masks) — do not use `sanity_unet_resnet18.pt` for anything beyond
  confirming the checkpoint format round-trips.
- GFW API key still not obtained (carried over from Step 0).

**Next session should:** decide whether to commit to the full ~38GB Zenodo
download now that the training pipeline is proven to work, then re-run a
VRAM/timing check at realistic batch sizes on real 2048x2048 images before
starting a real training run.

---

## 2026-08-25 — Drift track: skeleton built, real NCEP/NCAR result, ERA5 blocked

**Context:** earlier the same day, only data-access research had been done
for this track (HYCOM and NCEP/NCAR both confirmed reachable) -- no drift
code existed yet. Built it this session: `src/drift/currents.py`
(HYCOM), `wind_ncep.py`, `wind_era5.py`, `advect.py`, and
`scripts/run_drift_skeleton.py`. Three real bugs hit and fixed along the
way -- see DECISIONS.md for details (all-zero wind from a cross-year
OPeNDAP concat bug, a non-CF `tau` units crash from HYCOM, and a tau=24
rounding edge case).

**Real result, case ow-0001, 24h backward, NCEP/NCAR wind + HYCOM
barotropic currents** (50 particles seeded across the labeled object bbox):
- Detection point: (33.058E, 33.259N), 2019-01-01T03:42:35 UTC
- Estimated origin centroid (24h back): **(33.022E, 33.192N)**
- Spread: lon_std=0.0015 deg, lat_std=0.0041 deg (particles stayed tight --
  expected, given the small seed bbox and locally near-uniform flow)
- Runtime: ~184s, almost entirely the 24 sequential HYCOM OPeNDAP fetches
  (one per unique cycle/tau touched by the backward window)

**ERA5 comparison: blocked.** The user registered for CDS and set up
`~/.cdsapirc`, but `cdsapi.Client()` fails with `PermissionError` on that
path -- diagnosed as `.cdsapirc` having been created as a **directory**
instead of a file (with the real key sitting in a misnamed
`New Text Document.txt` inside it), not a registration or code problem.
This needs fixing on the user's machine (session sandbox can't touch paths
outside the project dir). `wind_era5.py` is written and ready to go the
moment this is fixed -- re-run `scripts/run_drift_skeleton.py` and it will
pick up both legs automatically and print the comparison.

**Also blocked, unrelated:** GFW API token now authenticates (progress
from earlier), but the actual 4Wings report request fails with a 422
"body malformed" -- likely a dataset-ID or payload shape issue, not
investigated further yet.

**Next session should:** once `.cdsapirc` is fixed, re-run
`scripts/run_drift_skeleton.py` to get the real ERA5-vs-NCEP/NCAR
comparison (code needs no changes, just needs the credential file fixed).
Separately, debug the GFW 422 error when there's time.

---

## 2026-08-25 — Drift track: ERA5 wired in, real comparison result

Same day as above, continued. User fixed `.cdsapirc` (it had been created
as a folder, not a file) and, after a subsequent `403 required licences
not accepted` error, accepted the ERA5 dataset's license on the CDS site.
Along the way, `cdsapi` surfaced one more real bug: the modern CDS
download API uses `latitude`/`longitude`/`valid_time` dimension names
instead of the legacy `lat`/`lon`/`time` -- fixed in `wind_era5.py` with a
rename step so it presents the same shape as `wind_ncep.py`. Installed
`cdsapi` (added to requirements.txt).

**Side-by-side result, case ow-0001** (detection: 33.058E, 33.259N,
2019-01-01T03:42:35 UTC; 24h backward, 50 particles, same advection code,
only the wind source differs):

| wind source | grid used | origin centroid | lon_std | lat_std |
|---|---|---|---|---|
| NCEP/NCAR | 4 steps, 3x2 pts (~1.875deg, 6-hourly) | (33.0216E, 33.1924N) | 0.0015 | 0.0041 |
| ERA5 | 24 steps, 16x16 pts (0.25deg, hourly) | (32.9503E, 33.1981N) | 0.0014 | 0.0040 |

**Centroid distance between the two estimates: 6.67 km.** For context,
the total backward displacement from the detection point itself is only
~8km over 24h -- so the NCEP/NCAR-vs-ERA5 choice shifts the answer by an
amount comparable to the signal being measured. The two aren't wildly
inconsistent (same general SW-ish direction, latitude estimates agree to
within ~0.7km), but they meaningfully disagree on longitude. Read as: use
ERA5 as the real answer, treat NCEP/NCAR as a rough sanity check /
fallback, not an interchangeable substitute -- confirms the DECISIONS.md
call to make ERA5 primary.

**GFW 422 "body malformed" error**: still not investigated further this
session -- token authenticates, request payload/dataset-ID needs fixing.

**Next session should:** move past the skeleton -- current physics is
deliberately simplified (3% windage, no Ekman deflection, single detection
event, one case). Decide whether to refine the physics or move on to the
attribution/AIS stage using this skeleton's output as-is. Also: debug the
GFW 422 error, and check on the Zenodo full download (was in progress
throughout this session).

---

## 2026-08-25 — GFW API 422 fixed, real vessel presence data confirmed

The SPA docs site (globalfishingwatch.org/our-apis/documentation) doesn't
serve its endpoint reference as static HTML -- `WebFetch` kept returning
only the landing page. Found real content by appending `.md` to any doc
URL (fumadocs convention) or fetching `/our-apis/documentation/llms.txt`,
which lists every page. Pulled the actual worked examples from
`docs/examples/report/report-example1` (custom polygon) and
`report-example10` (AIS vessel presence) -- both show real request/response
pairs, not just a parameter table.

**Two real bugs in the old request, found this way:**
1. `geojson` must be a **plain `Polygon` object** in the request body
   (`{"geojson": {"type": "Polygon", "coordinates": [...]}}`) -- the old
   script wrapped it in a `FeatureCollection`/`Feature` AND
   `json.dumps()`-stringified the whole thing. Both wrong.
2. `spatial-resolution` (`LOW` or `HIGH`) is a **required** query param
   that was missing entirely.

**Working request** (`scripts/test_gfw_api.py`, now targets case
ow-0001's real location/date instead of an arbitrary bbox):
```
POST https://gateway.api.globalfishingwatch.org/v3/4wings/report
  ?spatial-resolution=LOW
  &temporal-resolution=DAILY
  &group-by=VESSEL_ID
  &datasets[0]=public-global-presence:latest
  &date-range=2018-12-25,2019-01-08
  &format=JSON
Headers: Authorization: Bearer <token>, Content-Type: application/json
Body: {"geojson": {"type": "Polygon", "coordinates": [[[lon,lat], ...]]}}
```
bbox: 1 deg padded around (33.058E, 33.259N); date range: 2 weeks centered
on the ow-0001 detection date (2019-01-01).

**Result: status 200, real data.** 2276 vessel-presence records in the
window, e.g. a Liberia-flagged cargo ship "CONTSHIP VOW" (MMSI 636018185,
IMO 9395599) present 2019-01-03, and a Malta-flagged cargo ship "A. OBELIX"
(MMSI 256432000) present 2018-12-25 through 2018-12-30. Each record
carries MMSI/IMO/callsign/flag/vessel type and entry/exit timestamps --
exactly the fields the attribution stage will need to score candidate
vessels against the drift-estimated origin region/time.

**Next session should:** the GFW pipeline is now fully unblocked (auth +
correct request format both confirmed). Natural next step is the
attribution stage itself -- cross-reference this real vessel list against
the drift skeleton's origin-region/time estimate for ow-0001.

---

## 2026-08-25 — Full Zenodo dataset download started

Flipped `FULL_DOWNLOAD = True` in `scripts/download_zenodo_sample.py` (was
`False` since Step 0) and started it in the background: full
`01_Train_Val_Oil_Spill_images.7z` (~38GB, 1200 real 2048x2048 SAR images)
plus the already-verified `01_Train_Val_Oil_Spill_mask.7z`.

Also hardened the download function while touching it, since a single
~38GB request is much more likely to hit a network hiccup than the small
downloads Step 0 used: it now prints progress (% complete, speed, ETA)
every ~10s, and resumes via HTTP Range from the partial `.part` file if
interrupted and re-run, with a fallback to restart from scratch if the
server doesn't honor the resume request.

**Status:** confirmed actively downloading. Sustained speed observed is
**~0.4 MB/s**, giving an **ETA of roughly 30-31 hours** — this looks like
a Zenodo-side per-connection throttle rather than a local network issue
(0.4 MB/s is unusually slow for most connections). This will very likely
span multiple sessions; that's fine, the script resumes via HTTP Range from
the `.part` file if interrupted (safe to stop/restart the machine or the
script). Re-check `data/raw/zenodo_sar_oil_spill/01_Train_Val_Oil_Spill_images.7z.part`'s
size, or re-run the script (it prints current progress and continues), to
see current status. Next session should verify completion (file renamed
from `.7z.part` to `.7z`, size ~37.9GB) before extracting or training on
it, and only then move to the real training run.

**Update, same day — 16-connection parallel download attempted, then abandoned:**
Wrote `scripts/download_zenodo_parallel_test.py` (16 concurrent HTTP Range
requests into a separate file, never touching the original `.part`) to test
whether the 0.4 MB/s was a per-connection throttle. It was: a 60s sample
hit **~12 MB/s combined** (~30x faster). But the full run
(`--full`) crashed partway at 2.23GB/37.92GB — all 16 threads hit
`ConnectionError: Read timed out` simultaneously, almost certainly Zenodo
rejecting/throttling that many concurrent connections after some data had
transferred. Per the standing instruction to abandon the parallel approach
if it gets complicated: deleted the failed partial file (2.2GB real data
padded to a 38GB preallocated file — the original `.part` file was never
touched, confirmed untouched at 520MB throughout), and resumed the
original single-connection resumable download from Step 2, which is
running again now. **Takeaway for later**: the per-connection throttle is
real and confirmed (12 MB/s vs 0.4 MB/s on a clean sample), so a *modest*
parallelism (e.g. 3-4 connections instead of 16) might sustain without
tripping a connection-count limit — worth trying if the single-stream
download is still running next session, but not worth more debugging time
right now per the "abandon if complicated" instruction.

---

## 2026-08-25 — Zenodo full dataset download complete

`01_Train_Val_Oil_Spill_images.7z` finished downloading: **37.92 GB,
40,712,942,245 bytes -- exact match** to the size Zenodo's own metadata
reported. Verified the archive itself opens and is valid (not just the
right size): `py7zr.SevenZipFile(...).getnames()` lists **1201 entries**
(1200 real 2048x2048 Sentinel-1 SAR images under `Oil/`, plus the folder
entry itself) -- matches the dataset description exactly. Combined with
the already-verified 1200 masks from Step 0, **the full real training set
is now on disk and confirmed usable**: `data/raw/zenodo_sar_oil_spill/`
has both `01_Train_Val_Oil_Spill_images.7z` (this) and
`01_Train_Val_Oil_Spill_mask.7z` (Step 0).

Took roughly 2.5 hours total across this session, mostly single-connection
at 0.4-8 MB/s (see the parallel-download-attempt entry above for why
16-way parallelism wasn't used for the full run: it worked but wasn't
stable enough to leave unattended).

**Next session should:** this unblocks the real detection-model training
run that Step 1's sanity pass was explicitly gating on. Before training:
extract the archive (not yet done -- only verified it opens), re-run a
VRAM/timing check at realistic batch sizes on real 2048x2048 images (the
Step 1 sanity numbers were measured on 640x640 PANGAEA images, not the
real 2048x2048 Zenodo images), then start real training with the
already-built pipeline (`src/detection/`).

---

## 2026-08-25 — Real training pipeline built; Part I extraction + Part II/III download in progress

Confirmed (see DECISIONS.md): Zenodo Part I is 100% oil-positive (0/1200
masks empty), so real hard negatives have to come from Zenodo Part II
(685 no-oil + 685 look-alike, same calibrated domain) rather than PANGAEA
(wrong pixel domain -- would teach the model to distinguish datasets, not
oil vs. look-alike). Started downloading Part II + III in the background
in parallel with Part I's extraction.

**Built:**
- `src/detection/augment.py` -- flips, 90-degree rotations, mild dB-scale
  speckle jitter (all SAR-appropriate, no interpolation artifacts).
- `src/detection/preprocess.py: normalize_db_fixed` -- fixed-range dB
  normalization (NOT per-tile min-max, which would destroy the real
  calibration signal -- see the function's docstring).
- `src/detection/dataset.py: ZenodoTileDataset` -- real training dataset,
  windowed disk reads (rasterio) rather than precomputing tiles in memory,
  since 2500+ full 2048x2048 source images don't fit in RAM at once.
- `src/detection/train.py` -- extended with an optional validation pass
  (val loss + Dice score each epoch) and a separate best-val-Dice
  checkpoint, distinct from the final-epoch checkpoint.
- `scripts/build_training_pool.py` -- discovers Part I/II image+mask
  directories by name, matches by filename, verifies every image has a
  mask and vice versa, builds an 85/15 train/val split stratified by
  class (oil/no_oil/lookalike) and by oil-fraction quartile within the
  oil class.
- `scripts/train_detection.py` -- the real training run: batch-size probe
  (tries 16/8/4, picks the largest that doesn't OOM) then a real training
  loop with checkpointing. Written and ready; not yet run -- data isn't
  fully ready, and the user asked to report back before starting it.

**Real bug found and fixed via a dry run** (20 available Part I images,
2 epochs, before waiting hours for the full dataset): the speckle-jitter
augmentation was being applied *after* `normalize_db_fixed` had already
compressed the ~50dB range into [0, 1], so a jitter std tuned for real dB
units (0.5) blew values out to roughly [-2, 2.5] instead of staying in
[0, 1]. Fixed by reordering: despeckle -> augment (in real dB units) ->
normalize (last step). Re-verified: augmented tile values now stay
correctly within [0, 1].

**Dry-run pipeline sanity check** (20 images, 2 epochs, batch_size=8,
tile_size=512, real GPU): ran end to end without errors, loss decreased
(2.055 -> 1.978), val Dice improved (0.114 -> 0.164), best-checkpoint
saving worked. **Peak VRAM: 1778MB (~29% of 6GB)** at batch_size=8 on
real full-density 512x512 tiles -- notably higher than Step 1's sanity
numbers (354-818MB), since those used much smaller 640x640 PANGAEA source
images, not real 2048x2048 ones. Still comfortable headroom for
batch_size=16. (This dry run's actual loss/Dice numbers are not
meaningful -- 20 images, all oil-positive, no real val methodology --
purely a pipeline-correctness check before the long real run.)

**Hit one transient network timeout** on the Part II download (`Read
timed out` after ~4GB) -- exactly the failure mode the resumable design
was built for. Confirmed the `.part` file was untouched and resumable,
then switched to running the download under a small bash auto-retry
wrapper rather than relying on manual restarts, since a multi-hour
download will likely hit this again.

**Status as this entry was written:** Part I extraction ~50% done (600ish
/1200), Part II download in progress (auto-retrying on failure). Filename
verification, archive cleanup, and the training-pool build are next once
both finish.

**Update, same session -- Part I extraction done and verified:** took
1944.8s (~32 min). **1200 images, 1200 masks, exact 1:1 filename match --
0 images without a mask, 0 masks without an image.** Deleted both
verified `.7z` archives (images + mask, ~38GB) to reclaim disk headroom
before Part II/III land -- freed disk from 101GB to 139GB available.

---

## 2026-08-25 — Dashboard built with real drift data, while Part II/III download continues

Checked for "the earlier mockup" the user referenced before building --
didn't exist anywhere in the repo or git history (see DECISIONS.md). Built
`src/dashboard/` fresh: `build_map.py` (real folium/Leaflet map),
`build_dashboard.py` (page shell with explicit placeholder swap-in points
for the detection overlay and vessel ranking panels), and
`scripts/export_drift_dashboard_data.py` (re-runs the drift skeleton and
dumps real per-particle tracks/positions, not just the aggregate
centroid/std).

**Real finding**: NOAA PSL's server (the NCEP/NCAR wind fallback) was
genuinely down during this session -- confirmed via a direct `curl` test
that ruled out our own network/bandwidth (HYCOM and general internet both
fine). Made the export script resilient to a source failing independently
rather than crashing the whole run. Map was built with real ERA5 data
(NCEP/NCAR temporarily unavailable, not a bug) -- centroid
(32.9503°E, 33.1981°N), matching the earlier drift-track result exactly.

**Delivered**: `src/dashboard/output/index.html` + `map.html`, sent to the
user as local files (not published as a claude.ai Artifact -- the map's
external basemap tiles would be blocked by the Artifact sandbox's CSP).
Detection overlay and vessel ranking panels are explicit, informative
placeholders driven by real project status (not generic "coming soon"),
e.g. the vessel panel correctly states GFW access is confirmed working
while the scoring algorithm itself doesn't exist yet.

Meanwhile, the Zenodo Part II download continued in the background
throughout (paused briefly to rule out bandwidth contention as the NOAA
issue's cause, confirmed it wasn't, resumed) -- still in progress as this
entry was written, Lookalike images file ~55% done.

**Next session should:** re-run `scripts/export_drift_dashboard_data.py` +
`build_map.py` once NOAA PSL is back to get the real NCEP/NCAR comparison
back on the map. Once training/attribution are further along, replace
`render_detection_panel()` and `render_vessel_panel()` in
`build_dashboard.py` with real output per their docstrings.

---

## 2026-08-25 — Attribution scoring built and wired into the dashboard

Done in parallel with the Zenodo Part II/III download (independent work,
per the user, while waiting). Built `src/attribution/gfw_client.py`
(reusable GFW fetch, extracted from `scripts/test_gfw_api.py`) and
`src/attribution/score_vessels.py` (real scoring against the drift
origin estimate -- see DECISIONS.md "Attribution scoring" for the full
methodology and its known limitations, especially the GFW `LOW`-resolution
grid-coarseness caveat). Also factored `haversine_km` out of
`src/dashboard/build_map.py` into `src/common/geo.py` once a second
caller needed it.

**Real result**: 2276 GFW presence records for case ow-0001 -> 334 unique
candidate vessels, ranked. Top of the list: "THOR FREYJA" (MMSI
311000273, Bahamas) and "PHOENIX III" (MMSI 374559000, Panama), both
4.6km from the origin estimate with a 0h time gap. Full ranking written to
`data/processed/dashboard/vessel_ranking_ow0001.json` and wired into the
dashboard's vessel-ranking panel, replacing that placeholder with this
real (first-pass, uncalibrated) output -- rebuilt `index.html` confirms
the real vessel names render correctly.

**Zenodo Part II/III status at this point**: still downloading in the
background throughout this work, hit one more transient Zenodo read
timeout partway through (same failure mode as before, real progress had
been made first -- 51.5% of the Lookalike images file before it dropped),
resumed cleanly via the `.part` file, now running under a persistent
auto-retry wrapper so it self-heals without needing another manual
restart.

**Next session should:** next real step is either the detection-model
training run (once Part II/III data is ready) or extending attribution
with heading/speed-consistency scoring.

**Update, same session -- switched vessel scoring to HIGH spatial
resolution:** confirmed it performs fine (~10s, 2661 vs 2276 records for
the same query) and fixes the grid-coarseness tie noted above -- THOR
FREYJA and PHOENIX III now separate cleanly to 2.6km/5.2km instead of
both showing 4.6km. Top-ranked vessel identity unchanged, a good
consistency signal. Dashboard rebuilt with the improved ranking. Also
checked whether NOAA PSL (NCEP/NCAR fallback) had recovered -- still
down as of this check.

---

## 2026-08-25 — Pipeline validated end-to-end on a second real case

Refactored `scripts/export_drift_dashboard_data.py` and
`src/attribution/score_vessels.py` to take a case ID / small case
registry instead of hardcoded ow-0001 constants (see DECISIONS.md
"Pipeline validated on a second real case" for why). Ran the full
drift + attribution pipeline on **ow-0002** (real detection: 32.029E,
31.685N, 2019-01-04T15:56:38 UTC, near the Egyptian coast / Suez Canal
approaches) with zero methodology changes.

**Real results**: drift origin estimate (31.754E, 31.577N) -- ~28km SW
of detection (larger than ow-0001's ~8km, plausibly real given different
conditions and coastal proximity, not a red flag). GFW: 10,180 presence
records -> 1,006 candidate vessels (much busier area than ow-0001's
2,661/334, consistent with Suez-adjacent traffic). Top match: "CAPTAIN
AMIR" (MMSI 667001723, Sierra Leone), 0.5km from the origin estimate, 0h
time gap.

**Takeaway**: the pipeline genuinely generalizes -- this wasn't a
from-scratch build for the second case, just plugging in different real
coordinates/dates into the same code. NCEP/NCAR still skipped gracefully
(NOAA PSL still down, checked again).

**Zenodo Part II/III**: still downloading throughout, ~85% through the
Lookalike images file as this entry was written, running clean under the
auto-retry wrapper with no further manual intervention needed.

**Next session should:** the pipeline has now been proven on two
independent cases for drift+attribution. Remaining real work: the
detection-model training run (blocked on Part II/III), and writing a
top-level README for the repo (setup + how to run each stage) -- useful
for anyone (including hackathon judges) looking at this repo cold.

---

## 2026-08-25 — README written; detection inference plumbing built

Wrote `README.md` (setup, repo structure, how to run each stage, current
status, known limitations) -- also caught a real gap while writing it:
`folium` had been installed and used since the dashboard work but never
added to `requirements.txt`, fixed.

Built `src/detection/inference.py` + `scripts/render_detection_overlay.py`
against the Step 1 sanity checkpoint (see DECISIONS.md for why) --
real Zenodo tile, real ground truth, real (poor, as expected) prediction,
IoU=0.08. Wired into the dashboard's detection panel, replacing that
placeholder. All three dashboard panels now show real output (map,
detection overlay, vessel ranking) -- only the detection model itself is
still a stand-in, clearly labeled as such throughout.

**Zenodo status**: Part II (No-Oil + Look-alike images/masks) fully
downloaded. Part III (test set) in progress, ~11% in. Hit one more
transient Zenodo read timeout partway through Part III -- the persistent
auto-retry wrapper caught it and resumed automatically with no manual
intervention needed (confirmed via direct file-size check after a "retry"
prompt turned out to already be handled).

**Next session should:** once Part III finishes -- extract both Part II
and Part III, verify filenames against expected counts, delete the
verified archives, build the training pool (Part I + Part II, 85/15
stratified split), and run the real training. Also worth periodically
re-checking whether NOAA PSL (NCEP/NCAR) has recovered.

---

## 2026-08-25 — Dashboard case switcher (ow-0001 / ow-0002 both live in the UI)

Added a tab switcher to the dashboard -- see DECISIONS.md for the
`display: contents` implementation approach. Generalized
`src/dashboard/build_map.py` to build a map file per case
(`map.html` + `map_ow0002.html`). Rebuilt and verified via direct HTML
inspection: both tabs present, both case-blocks present with correct
default visibility, both real vessel rankings embedded (THOR FREYJA for
ow-0001, CAPTAIN AMIR for ow-0002), click handler present. Dashboard
resent to the user.

**Zenodo Part III**: ~13% in, progressing normally (confirmed via direct
file-size check -- the printed log speed readings have repeatedly lagged
behind real progress this session; direct file checks are the reliable
source of truth, not the log tail).

**Next session should:** same as above -- extraction/training is the
next real milestone once Part III finishes. Dashboard and pipeline
documentation are otherwise in a solid, presentable state.

---

## 2026-08-25 — Dashboard visual redesign (dark navy/amber/teal, matching a user-supplied reference)

User supplied a complete concept mockup and asked the dashboard to match
it "and even more better" -- see DECISIONS.md "Dashboard visual redesign"
for the full writeup, including where this went beyond the reference (real
candidate vessels now plotted on the actual map, not just a decorative
sidebar list) and where it deliberately diverged (corner tag says "real
data" not "simulated data", since ours isn't).

**Built/changed**: `build_dashboard.py` fully rewritten (fonts, color
tokens, stat strip, evidence-card vessel list, corner tag); `build_map.py`
restyled to a dark basemap + real vessel markers; `score_vessels.py` now
also saves each vessel's real lat/lon (needed for the map markers) --
re-ran scoring for both cases; `render_detection_overlay.py` restyled to
a dark matplotlib figure (was a jarring white box against the new dark
dashboard, caught and fixed via an actual screenshot, not just by
inspecting the HTML source).

**How it was verified**: found headless Edge already installed on this
machine (`msedge.exe --headless --screenshot`), used it to actually
render and screenshot both the map and the full dashboard -- caught the
white-box detection-overlay issue this way, fixed it, re-screenshotted to
confirm. This is now a reusable verification technique for future
dashboard changes in this project: build -> screenshot -> look -> fix,
rather than trusting "it built without errors."

**Zenodo Part III**: ~18% in, progressing normally throughout this work
(confirmed via direct file-size checks).

**Next session should:** extraction/training remains the next real
milestone once Part III finishes. Dashboard is now in a strong, real,
visually cohesive state -- worth applying the same dark theme treatment
if/when the vessel-ranking or detection panels get more functionality
(e.g. heading/speed-consistency scoring, or the real trained model).

**Update, same session -- page background treatment**: added a layered
body background (faint nautical-chart grid + soft amber glow behind the
header, over the navy base) -- see DECISIONS.md for the exact CSS. Kept
deliberately low-opacity, verified via another real screenshot that it
doesn't touch panel content (panels sit on their own solid background,
grid only shows in the gaps).

---

## 2026-08-25 — GFW compliance, Ekman deflection, test suite (all four follow-ups done)

**GFW rate limits**: confirmed real numbers -- 50,000/day, 1,500,000/month,
shared per-user across up to 5 tokens. We're nowhere close (~20-30 total
requests all project). **Fixed a real compliance gap**: added the required
"Powered by Global Fishing Watch" attribution to the dashboard footer,
which wasn't there before -- a genuine ToS requirement (CC BY-NC 4.0),
not just tidiness.

**Heading/speed-consistency scoring**: investigated, not currently
feasible -- the `public-global-presence` dataset's records have no
course/speed field (checked a real record's exact keys). Would need a
different GFW endpoint (vessel tracks/events). Documented in DECISIONS.md
rather than faked or silently skipped.

**Ekman/Coriolis deflection**: added to `src/drift/advect.py` (20 degree
empirical deflection of the wind-driven term, hemisphere-aware, current
term untouched to avoid double-counting HYCOM's own Ekman transport).
Re-ran the full pipeline (drift -> scoring -> maps -> dashboard) for both
real cases with this change:

| case | origin shift | top suspect |
|---|---|---|
| ow-0001 | ~1.9 km | unchanged (THOR FREYJA) |
| ow-0002 | ~5.4 km | **changed**: CAPTAIN AMIR -> ALAWAD1 |

The ow-0002 flip is a real, important finding -- see DECISIONS.md
"Ekman/Coriolis deflection added" for why this matters for how the
project's output should be read (ranked list with visible evidence, not
a single trusted verdict).

**Test suite**: added `tests/` (pytest), 22 tests, all passing --
covers geo math, GFW scoring math, bbox-to-mask rasterization, the new
Ekman rotation, and specifically regression-tests the `cycle_and_tau`
tau=24 bug that actually happened earlier this project. Run via
`venv\Scripts\python.exe -m pytest tests\ -v`.

**Zenodo Part III**: continued downloading throughout all of this work,
no issues.

**Next session should:** dashboard/maps have the updated Ekman-corrected
data as of this entry (rebuilt and re-sent to the user). Extraction and
the real training run are still the next data-pipeline milestone once
Part III finishes.

---

## 2026-08-25 — Real bug: corrupted Lookalike archive found; parallel download added; pipeline dry-run verified on real Part I + Part II data

**Real bug found**: `01_Train_Val_Lookalike_images.7z` (Part II) had finished
downloading (no `.part` suffix) but was corrupted -- **32.23GB on disk vs.
22.99GB reported by Zenodo's own metadata (9.2GB too large)**, and `py7zr`
failed to even read its header (`Bad7zFile: invalid header data`). The other
three Part II archives (No_Oil images, both masks) were confirmed correctly
sized and readable. Most likely cause: two overlapping download runs (e.g.
the persistent auto-retry wrapper mentioned in an earlier entry firing a
second run before the first had exited) both appending to the same `.part`
file via `download_full`'s `"ab"` resume mode -- that function was never
designed to be safe against two writers at once. Deleted the corrupted file
and started a clean re-download.

**Parallel download added** (`scripts/download_zenodo_part2_parallel.py`):
the single-stream resumable downloader sustains only ~0.1-0.4 MB/s against
Zenodo (confirmed again on the redownload: 0.1 MB/s, 54h ETA) -- a
per-connection throttle, not a local bandwidth limit, per the Part I
16-connection test earlier this project (~12 MB/s combined, but crashed on
the full run from connection-count instability). This new script applies
that entry's own suggested follow-up -- a *modest* connection count (4,
not 16) -- and, learning from the corruption bug just found, writes to a
**separate preallocated file** with each thread seeking to its own
pre-assigned byte range (no shared appendable state, no possible write
race even if a thread dies/restarts), verifies the final size and re-opens
it with `py7zr` before renaming it into place. Result: **~1.7-1.8 MB/s
combined, ETA ~3h** -- a real ~10-15x speedup over the single-stream
redownload, without the instability the 16-connection attempt hit.

**Extraction**: added `scripts/extract_zenodo_part2.py` (verifies expected
vs. actual `.tif` count per archive, skips already-extracted archives so
it's safe to re-run once the Lookalike images archive finishes
downloading). Ran it on the three ready archives -- **685 No_Oil images,
685 No_Oil masks, 685 Lookalike masks, all verified**. Also empirically
confirmed all 685 No_Oil masks are genuinely all-zero (0/685 with any
nonzero pixel), matching what the download script's docstring claimed
rather than just trusting it.

**Pipeline dry-run**: while waiting on the Lookalike images download,
built a `ZenodoTileDataset` over 3 real Part I (oil) + 3 real Part II
(no_oil) image/mask pairs -- 96 tiles indexed, correct tensor shapes/dtype
(`(1,512,512)` float32), a no_oil tile's mask confirmed all-zero all the
way through the real augment -> normalize pipeline. Catches any
Part-II-specific loading issues before committing to the full 3-class
training pool build.

**Next session should:** once the Lookalike images download finishes
(~3h from when this entry was written): run `extract_zenodo_part2.py`
again (picks up the newly-downloaded archive), then
`scripts/build_training_pool.py`, then the real
`scripts/train_detection.py` run. Part III (test set) continued
downloading throughout, untouched.

**Update, same session -- root cause of the corruption confirmed, and it
was still actively happening to Part III:** NOAA PSL (NCEP/NCAR wind
fallback) came back up for the first time this project -- re-ran
`scripts/run_drift_skeleton.py` and got a real fresh NCEP/NCAR-vs-ERA5
comparison for ow-0001 (7.24km centroid distance, vs. 6.67km pre-Ekman),
then `scripts/export_drift_dashboard_data.py` for both cases (first time
*both* wind sources succeeded for *both* cases in one pass -- ow-0002's
NCEP/NCAR-vs-ERA5 distance is a new real number: **11.76km**), then
rebuilt the map and dashboard.

While checking on the Lookalike download afterward, noticed Part III's
`.part` file was at 10.12GB and growing -- already past Zenodo's own
reported total size (9.86GB) for that archive, while still incomplete.
Same corruption signature as the Lookalike bug. `Get-CimInstance
Win32_Process` found the actual cause: **two separate invocations of
`scripts/download_zenodo_part2_part3.py` running at once** (PIDs 26276
and 13576, launched at the identical second, 11:40:47 -- consistent with
the "persistent auto-retry wrapper" mentioned in an earlier entry firing
twice), both past Part II (already on disk, skipped) and both appending
to the same Part III `.part` file simultaneously. Confirmed with the
user before killing anything (process-killing is a real, hard-to-reverse
action); killed all 4 PIDs (both invocations + their child interpreters),
deleted the corrupted `.part` file, and restarted Part III using the
same disjoint-byte-range parallel approach already proven for Lookalike
images -- moved that logic into `_zenodo_download_utils.py` as a shared
`download_parallel()` (was duplicated inline in
`download_zenodo_part2_parallel.py`; now both that script and the new
`download_zenodo_part3_parallel.py` are thin wrappers around it) so this
fix applies to any future re-download, not just this one archive.

**Takeaway for later**: this wrapper (never itself committed to the
repo, always run ad hoc) has now caused the same corruption twice across
two different archives. If it's used again, check for an already-running
instance before launching a retry, or just use `download_parallel()`
directly -- it doesn't need retry-wrapping in the first place, since it
already validates size + `py7zr` integrity before accepting a download
as done.

**Update, same session -- the actual root cause was a different, still-live
Claude Code session, and it kept re-corrupting things:** Part III started
overshooting its expected size again shortly after the fix above (10.12GB
`.part` vs. Zenodo's reported 9.86GB, still growing). `Get-CimInstance
Win32_Process` traced this one all the way up the process tree and found
it wasn't a leftover from earlier in *this* session at all -- it was
**a separate Claude Code session** (a different session ID, visible in
its child processes' working-directory sentinel files), which had
written `/tmp/retry_part23.sh` (an infinite `until
download_zenodo_part2_part3.py; do sleep 15; done` loop), run it, then
apparently written a second copy `/tmp/retry_part23_v2.sh` and started
*that* too without stopping the first -- two independent infinite retry
loops, each relaunching the unsafe single-stream downloader every ~15s,
guaranteed to periodically overlap and corrupt whatever they were both
touching. This explains both corruption incidents in this entry: neither
was caused by anything done in this session, and the first fix (killing
4 specific PIDs) only cleared that moment's symptom, not the loops
generating new ones every 15 seconds.

Confirmed with the user again before acting (a live process tree
belonging to a *different* session is exactly the kind of thing worth a
second check, not just extending the earlier approval) -- killed both
retry loops and their spawned downloader processes, deleted
`/tmp/retry_part23*.sh` so they can't be manually re-run by mistake,
waited 20s and re-confirmed via `Get-CimInstance` that nothing
respawned. Also **made `download_parallel()` itself resumable**
(`_zenodo_download_utils.py`): a per-thread-position JSON sidecar
(`<file>.parallel.progress.json`) is updated every ~10s and on failure,
so a transient connection drop -- which was also happening for real,
independent of the corruption, roughly every 500MB-2GB on this
network -- now resumes from the last saved position per thread instead
of discarding all progress and restarting the full multi-GB download
from zero. Both Lookalike images and Part III restarted clean under this
resumable version with a 30-attempt retry loop (cheap now that retries
resume rather than restart).

**Takeaway for later, revised**: if a download on this project looks
like it's being corrupted again, check `Get-CimInstance Win32_Process`
(or equivalent) for *any* process touching the target file before
assuming it's this session's own doing -- multiple Claude Code sessions
can be live against the same repo/machine at once, and their background
work isn't visible from inside a single session's own task list.

---

## 2026-08-25 — Real training run started; dashboard UI upgrades (drift animation, data provenance panel)

**Full training data pipeline complete**: extracted the now-clean Lookalike
images archive (685 tifs, verified), built the training pool
(`scripts/build_training_pool.py`) across all 3 real classes -- **2570
total image/mask pairs** (1200 oil + 685 no_oil + 685 lookalike), split
2184 train / 386 val, zero image/mask mismatches. Started the real
training run (`scripts/train_detection.py`): 34,940 train tiles / 6,176
val tiles at 512px, batch_size=16 (auto-selected by the probe, 3235MB
peak VRAM).

**Real bug found in the training run itself**: epoch 1 at the default
`num_workers=0` took **93.1 minutes** (loss=1.8372, val_dice=0.0175) --
GPU utilization was cycling 0-60%, not pinned high, meaning the
single-threaded `rasterio` tile reads between batches were the real
bottleneck, not GPU compute (the batch-size probe's pure-compute timing
was ~64min of the 93min actual). At that rate, 30 epochs would have
taken **~46.5 hours**. Confirmed with the user before restarting (losing
the 93 minutes already spent) -- added `num_workers=6` (16 logical cores
available) to both the training and validation `DataLoader`s in
`src/detection/train.py` / `scripts/train_detection.py`. Confirmed
working: GPU utilization went from cycling 0-60% to a sustained 100%
once the worker prefetch queues filled. Real epoch-1 timing for this
run pending -- see next entry once it lands.

**Dashboard UI, in parallel while training ran** (the user asked for a
more "futuristic," feature-rich UI -- a plan was proposed and approved
piece by piece rather than a single big rewrite, so each addition could
be verified real before moving to the next):
- **Real drift-animation playback** (`src/dashboard/build_map.py`,
  `render_drift_animation`): a play/scrub control that steps through the
  actual per-particle backward-advection positions (25 real timesteps
  from `src/drift/advect.py`'s `Trajectories`, detection time -> 24h
  back), not a decorative animation -- reads the same `tracks` array
  already in `drift_{case}.json`. Verified via headless-Edge screenshot
  (this project's established build->screenshot->look->fix technique).
- **Data provenance panel** (`src/dashboard/build_dashboard.py`,
  `gather_provenance`/`render_provenance_panel`): dataset size, current
  best model checkpoint (epoch/val_dice, read live from the `.pt` file,
  not hand-maintained), drift/GFW data freshness (file mtimes), and real
  GFW API quota usage (~20-30 requests of 50k/day, 1.5M/month) -- every
  value computed at build time from the actual files on disk, so it
  can't silently go stale the way a written-once status blurb could.

**Update, same session -- `num_workers=6` speedup confirmed real, and
wind-vector overlay added:** epoch 1 of the parallel-data-loading run
finished in **1340.9s (22.3 min)** vs. the original 5586.9s (93.1 min) --
a real **4.2x speedup**, GPU now pinned at 100% instead of cycling
0-60%. Puts the full 30-epoch run at roughly 11.2h instead of ~46.5h.
loss=1.8362, val_loss=2.0843, val_dice=0.0190 (new best).

Added the wind-vector overlay from the UI plan:
`scripts/export_drift_dashboard_data.py` now also exports a real
`wind_snapshot` (grid of lat/lon/u10/v10 at the timestep closest to
detection -- 256 real ERA5 points, 6 real NCEP/NCAR points per case,
straight from the same `wind_ds` that drives the actual advection, not
synthesized) and `src/dashboard/build_map.py` draws them as
speed-scaled arrows, one hidden-by-default layer per wind source.

**Real bug caught before shipping**: with the wind layer enabled to
verify it, the map's initial zoom blew out to fit the entire ~4deg-wide
wind grid instead of the tight case area -- and reverting the layer to
`show=False` did NOT fix it, because folium's `m.get_bounds()` walks the
whole object tree at Python build time regardless of a layer's runtime
Leaflet visibility. Fixed by computing the fit-bounds explicitly from a
manually-tracked list of the case's own points (detection, particle
final positions, origin centroids, vessels) instead of `m.get_bounds()`,
so wind-grid extent can never affect the initial view again, whether or
not that layer is later toggled on. Reverified both the default (hidden,
correct tight zoom) and manually-enabled (correct zoom retained, real
arrows visible at the region's edges) states via headless-Edge
screenshots before shipping.

**Update, same session -- vessel drill-down + sortable ranking added:**
`src/dashboard/build_dashboard.py`'s vessel cards now click-to-expand a
real detail section (IMO, vessel type, hours present in the AIS window,
raw score) -- fields the GFW records already carried but weren't shown,
not new data. Added a sort dropdown (confidence rank / distance / time
gap / confidence %) that re-sorts the existing cards client-side via
`data-*` attributes, no server round-trip. Verified both states (default
collapsed, and forced-expanded to check the detail layout) via
headless-Edge screenshots.

**Update, same session -- wind-source comparison surfaced in the stat
strip:** the map already drew the ERA5-vs-NCEP/NCAR comparison (dashed
line + km-apart label between centroids), but the top-level stat strip
didn't -- added a 5th stat card ("ERA5 vs NCEP/NCAR Origin") with the
same real `haversine_km` distance, shown only when both sources are
present for a case. Grid CSS switched from a hardcoded `repeat(4,1fr)`
to `auto-fit,minmax(190px,1fr)` so a 5th card (or fewer, if a case is
missing NCEP/NCAR) doesn't require a manual column-count edit.

Epoch 2 of the `num_workers=6` run: loss=1.8241, val_loss=1.8183,
val_dice=0.0179 (1805.2s -- slower than epoch 1's 1340.9s, plausibly
some contention from the dashboard screenshot work running concurrently
this session; not a regression in the fix itself).

**Update, same session -- case comparison table added, completing 5 of
6 planned UI pieces:** `render_case_comparison_table()` in
`build_dashboard.py` -- all real validated cases (currently ow-0001,
ow-0002) in one table (detection time, detection->origin distance,
ERA5-vs-NCEP/NCAR distance, vessels evaluated, top suspect+confidence),
independent of which tab is active, clicking a row switches to and
scrolls to that case. Only renders with 2+ real cases (not meaningfully
a "comparison" with one). Hit one real Python syntax error along the
way -- a nested f-string with escaped quotes inside the outer f-string's
expression part isn't legal pre-3.12 -- fixed by pulling that piece into
its own variable first. Confirmed via screenshot: ow-0002's real top
suspect shows as ALAWAD1 at 98%, matching the Ekman-deflection flip
recorded earlier this project (was CAPTAIN AMIR before Ekman was added).

**UI plan status: 5 of 6 done** (drift animation, data provenance,
wind vectors, vessel drill-down/sort, wind-source comparison, case
comparison -- six built, since wind-source comparison ended up being two
small additions rather than one). Only report/PDF export remains,
lowest priority in the original plan.

**Update, same session -- resume-from-checkpoint added to training:**
prompted by checking whether the ~11h training run survives the laptop
sleeping/losing power -- this machine's sleep is disabled but hibernate
still triggers after 1h on battery, and even setting that aside,
`train_detection.py` had no way to continue after any crash/interruption
except restarting at epoch 1, discarding everything. Added
`latest_checkpoint_path` to `train()` (`src/detection/train.py`):
written after *every* epoch (model + optimizer + AMP scaler state,
epoch number, best-so-far tracking, full history) -- distinct from
`best_checkpoint_path` (only written when val_dice improves) and
`checkpoint_path` (only written once, at the very end). If that file
exists when `train()` is called and `resume=True` (the default), it
picks up at the epoch after the saved one instead of starting over. Not
applied to the currently-running process (editing the file doesn't
affect an already-loaded Python module) -- takes effect on the next
invocation, e.g. if this run needs restarting for any reason.

**Update, same session -- Part III test-set evaluation prepared while
training ran:** with training now genuinely learning (val_dice
0.0190 -> 0.0224 across epochs 1-22, see the epoch log above), used the
remaining epochs' wait time to get ready for the actual "how good is it"
number rather than figuring it out cold once training finishes.

- `scripts/extract_zenodo_part3.py`: extracts the held-out test set (150
  Oil / 150 No-oil / 150 Lookalike, 450 real images + masks). Part III's
  internal archive layout differs from Part I/II -- nested under
  `Images/`/`Mask/` instead of flat class dirs, and uses `"No oil"` (a
  space) instead of `No_oil`. Ran it for real: all 900 tifs extracted
  and verified.
- `scripts/evaluate_test_set.py`: tiles each real test image the same
  way training did (512px, non-overlapping), runs the trained checkpoint
  via `src/detection/inference.py`, computes real IoU/Dice per tile
  against ground truth, aggregates per class and overall. IoU/Dice
  defined as 1.0 when both prediction and ground truth are empty
  (correct rejection) rather than 0/NaN -- otherwise the no-oil/lookalike
  classes, which are *supposed* to predict empty, would look artificially
  bad.
- **Real bug caught by smoke-testing before the real run**: Part III
  masks carry a `_segmentation` suffix the images don't
  (`00000.tif` -> `00000_segmentation.tif`) -- unlike Part I/II, where
  filenames match exactly. A first smoke-test run hit a real
  `RasterioIOError` from this; fixed the mask-path construction, then
  verified on one image each from the Oil and "No oil" classes (the
  latter also confirming paths with a literal space extract/read
  correctly) -- both passed, real IoU numbers computed, `has_oil`
  correctly `False` throughout the no-oil image. Ran on CPU
  deliberately, not the training GPU, so the smoke test wouldn't
  compete with the live training run.

**Update, same session -- committed everything, then a real disk-space
emergency:** committed the full session's work locally (48 files,
~22.5k insertions -- detection training pipeline, drift/attribution/
dashboard build-out, download infra hardening; not pushed to origin,
that wasn't asked for). Also refreshed README.md, which had gone stale
(claimed "no Ekman deflection yet" and "NCEP/NCAR down", both fixed
earlier this project) -- now reflects the real current state and points
at LOG.md for anything that might lag.

Shortly after, the user reported the disk nearly full. Checked: **9.8GB
free of 224GB (96% used)**. Cause: `zenodo_sar_oil_spill_part2` (102GB)
and `_part3` (25GB) each still held their original `.7z` archives
*alongside* the already-extracted, already-verified image/mask data --
Part I had its archives deleted after verified extraction (see the "Full
Zenodo dataset download complete" entry, much earlier this project), but
that same cleanup step was never applied to Part II/III once their
extraction was verified this session. Deleted the redundant archives
(Lookalike images/masks, No_Oil images/masks, Part III combined --
~55.8GB) now that both are independently verified extracted with correct
per-class tif counts. **Freed disk from 9.8GB to 62GB available.**
Training unaffected throughout (it only touches the checkpoints
directory, never the raw archives).

**Next session should:** continue watching the `num_workers=6` training
run (real final numbers -- best val_dice, epoch, total wall-clock time --
once it completes or spans into a new session), then run
`scripts/evaluate_test_set.py` for the real held-out accuracy number
(fully prepared and verified, ready to go), and re-run
`scripts/render_detection_overlay.py` against the real trained checkpoint
to replace the sanity-checkpoint placeholder in the dashboard's detection
panel. UI-wise, only the report/PDF export item remains from the
original plan. Also worth committing again once training/evaluation
produce real final numbers, and keeping an eye on disk space going
forward -- this project's datasets are large enough that redundant
archives add up fast.
