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

---

## 2026-08-26 — Training finished at 30 epochs; real eval revealed a
threshold problem, not a dead model; extended to 60 epochs

**Training finished**: 30/30 epochs, best val_dice=0.0233 at epoch 27,
final-epoch checkpoint also saved. `scripts/evaluate_test_set.py` (built
and smoke-tested last session) ran for real against all 450 held-out
Part III images.

**First real result looked bad, then turned out to be a threshold
problem**: at the training-time default threshold (0.5), mean IoU on the
1072 tiles that actually contain real oil was **0.0110** -- essentially
zero overlap -- while the "OVERALL" aggregate showed 0.6544, entirely
because the no_oil/lookalike classes are trivially easy to get right
(their ground truth is genuinely empty, so predicting empty scores well
there regardless of whether the model learned anything about oil).
Rendered the real detection overlay (`scripts/render_detection_overlay.py`,
updated to use the real checkpoint and a genuinely held-out Part III demo
image instead of Part I) and it visually confirmed this: the model
predicted a completely empty mask on a real oil tile.

**Diagnosed rather than assumed dead**: checked raw sigmoid probabilities
directly. Max probability anywhere on the demo tile was 0.40 -- never
crosses 0.5, so threshold=0.5 predicts empty on literally every tile,
always, regardless of content. But mean probability *inside* the real
oil region (0.22) was measurably higher than *outside* it (0.17) -- a
real, if weak, learned signal existed underneath the threshold. Added
`predict_probs()` to `src/detection/inference.py` (raw sigmoid output,
`predict_mask()` now built on top of it) and reworked
`scripts/evaluate_test_set.py` to sweep 6 thresholds from one forward
pass per tile (no extra GPU cost over a single threshold) rather than
evaluating at a hardcoded 0.5.

**Real threshold-sweep result** (all 450 images):

| threshold | overall IoU | oil-tiles-only IoU | no_oil IoU | lookalike IoU |
|---|---|---|---|---|
| 0.5 | 0.6544 | 0.0110 | 0.8462 | 0.7179 |
| 0.35 | 0.3352 | 0.0449 | 0.5400 | 0.2913 |
| 0.25 | 0.1333 | 0.0947 | 0.2621 | 0.0808 |
| 0.22 | 0.1170 | 0.1158 | 0.2275 | 0.0608 |
| 0.20 | 0.0984 | 0.1356 | 0.1692 | 0.0546 |
| 0.18 | 0.0881 | **0.1645** | 0.1283 | 0.0517 |

Honest conclusion: no single threshold fixes an undertrained model --
lowering it trades oil-detection IoU against no_oil/lookalike false
positives, and even the best real oil-tiles-only IoU (0.1645 at
threshold=0.18) is still low in absolute terms. Training loss was still
decreasing at epoch 30 with no sign of plateauing, so this reads as
genuinely undertrained rather than architecturally incapable.

**Decision, with the user**: extend training rather than stop on this
result or just re-threshold and call it done -- asked the user directly
rather than deciding unilaterally, given it's another several hours of
compute. Bumped `EPOCHS` in `scripts/train_detection.py` from 30 to 60.

**Real bug hit using the resume-from-checkpoint support added earlier
this project**: the run that had just finished was launched *before*
that resume code was added to `src/detection/train.py`, so (as
documented at the time) it never wrote a `latest_unet_resnet18.pt` --
resuming would have silently restarted from epoch 1, discarding all 30
epochs. Reconstructed one manually from `final_unet_resnet18.pt` (model +
optimizer state, real) plus the full 30-epoch history transcribed from
the actual training log (loss/time/vram/val_loss/val_dice per epoch,
real numbers, not estimated) and `best_val_dice=0.0233`/`best_epoch=27`.
Verified it loads correctly (model + optimizer state_dict) before
trusting it.

Hit one more real bug getting the resumed run started: the reconstructed
checkpoint's `scaler_state_dict` was `None` (no real AMP scaler state
existed to reconstruct), but `train()`'s resume check was
`if use_amp and "scaler_state_dict" in ckpt` -- true even when the value
is `None`, since the key exists -- causing `GradScaler.load_state_dict(None)`
to crash with `TypeError: object of type 'NoneType' has no len()`. Fixed
to `if use_amp and ckpt.get("scaler_state_dict")`, which is also just a
more correct check in general (handles any checkpoint missing real
scaler state, not only this reconstructed one). Relaunched, confirmed
via the real startup log: `"Resumed from ...: continuing at epoch
31/60 (best_val_dice=0.0233 at epoch 27)"` -- correct, no lost progress.

**Next session should:** watch the extended run (epochs 31-60) for real
signs of improvement in oil-tiles-only IoU, not just the misleading
overall aggregate -- re-run `scripts/evaluate_test_set.py`'s threshold
sweep once it finishes or spans into a new session. If oil-tiles-only
IoU is still weak by ~epoch 45-50, worth reconsidering pos_weight or
loss formulation rather than just running to epoch 60. Once a
genuinely-improved checkpoint exists, re-run
`scripts/render_detection_overlay.py` and rebuild the dashboard.

---

## 2026-08-26 — Training crash + resume; "confidence" -> "match score";
anonymized dashboard build

**Training crashed, resumed cleanly.** The extended 60-epoch run (see
previous entry) died at epoch 37 with a Windows-specific PyTorch
DataLoader bug (`RuntimeError: Couldn't open shared event ... DataLoader
worker exited unexpectedly`, `num_workers=6` in `scripts/
train_detection.py`) -- a known flaky Windows multiprocessing issue, not
a code bug. The resume mechanism worked exactly as designed:
`latest_unet_resnet18.pt` had saved through epoch 37 (best_val_dice
0.0235 at epoch 33), and relaunching picked up at epoch 38 with no lost
progress. Worth flagging: loss/val_dice have been essentially flat since
~epoch 30 (loss 1.712->1.703, val_dice ~0.022), which is the plateau the
"next session should" note above was watching for -- if it's still flat
by epoch 45-50, stop extending and tune pos_weight/loss formulation
instead of running to epoch 60 for no gain.

**"Confidence" relabeled to "match score" everywhere user-facing**, and
an anonymized dashboard build mode added for anything that leaves the
judging room -- both raised by the user as review feedback, not found
independently. Full reasoning in DECISIONS.md's entry for today
("'Confidence' relabeled to 'match score'; anonymized dashboard build
added for public-facing use"). Concretely: `confidence_pct` renamed to
`match_score_pct` in `score_vessels.py`'s output (existing ranking JSONs
updated in place, not re-queried from GFW); `build_dashboard.py` and
`build_map.py` both gained `--anonymize`, writing to `output_anon/` with
real ship_name/mmsi/imo replaced by fictional stand-ins via the new
`src/common/anonymize.py`, while the real `output/` build (shown live to
judges) keeps real identities. Verified both builds render correctly and
that no "confidence" or real vessel name leaked into the anonymized
output.

**Stopped the LR-only fix, diagnosed the real cause, launched two proper
trials.** The user caught that my LR-decay fix targeted the wrong thing:
val_dice (~0.022-0.0235) has been sitting *below* the trivial "predict
all oil" Dice baseline (~0.058 at the real 2.98% oil fraction) since
~epoch 30, which points at `pos_weight=32.6` fighting Dice loss rather
than a missing LR schedule. Stopped that run (epoch-39 checkpoint
verified intact, backed up separately), added `TverskyLoss` to
`src/detection/losses.py` (alpha=0.3/beta=0.7, keeping `DiceBCELoss`),
made `train.py`'s LR scheduler opt-in/configurable instead of always-on
(so a loss-function trial isolates that one variable), and gave
`train_detection.py` CLI flags (`--loss`, `--pos-weight`, `--tag`,
`--fresh`, `--use-lr-scheduler`, ...) so trial runs sandbox their
checkpoints under `checkpoints/<tag>/` without touching the real
`best_unet_resnet18.pt`/`latest_unet_resnet18.pt` that
`evaluate_test_set.py` and `render_detection_overlay.py` read directly.
Launched two 10-epoch trials sequentially in the background: Tversky
loss (no scheduler, isolates the loss change) then, if time permits,
DiceBCELoss with pos_weight dropped to 8 plus a val_dice-monitored
ReduceLROnPlateau. Real results not in yet -- next session (or later
this one) should report the actual val_dice trajectory against the
0.0233 floor before picking a direction for the next full run.

**Report/PDF export added, closing the last open UI-plan item** (full
reasoning in DECISIONS.md's entry for today). Asked directly what could
honestly be called 100% done while the training trials run -- the
detection model can't be (gated on real results), but this could be, so
it was built rather than left open. `window.print()` + a `@media print`
stylesheet in `build_dashboard.py`, no new dependency. Verified the
rebuilt `output/`/`output_anon/` HTML is well-formed and the export
button/print header render with real data (methodology note, real
generated timestamp).

**Found a real, previously-unaddressed lever: tile-level oil-sampling
imbalance.** Asked whether there was a better way to fix detection
beyond the two running trials. Checked the pipeline for other gaps
first -- encoder already ImageNet-pretrained, augmentation already
SAR-appropriate, neither was it. Measured the actual training tile grid
for real (`scripts/analyze_tile_oil_distribution.py`, new): of 34,940
real tiles, **82.4% have zero oil pixels** (0% in the 685+685 no_oil/
lookalike images, and only 37.7% even within the 1200 oil images touch
the real slick). `pos_weight` can't fix this -- it only reweights pixels
inside a tile that already has oil, doing nothing for the ~1-in-5
batches (at batch_size~8) that have none at all. Full reasoning in
DECISIONS.md's entry for today.

Added `compute_oil_tile_weights()` (`src/detection/dataset.py`) and a
`sampler` param on `train()` to support a `WeightedRandomSampler`
targeting ~50% oil-tile representation per epoch, plus
`--oversample-oil-tiles`/`--target-oil-fraction` on `train_detection.py`.
22/22 tests still pass. Reordered the trial queue: killed the Tversky
trial after its real epoch 1 (val_dice=0.0227, checkpoint kept, not
discarded) to run the oil-tile-oversampling trial first (original
DiceBCE/pos_weight=32.6, isolating the sampling variable alone), then
Tversky resuming from its saved epoch-1 checkpoint, then the
pos_weight+scheduler trial as before. No real oversampling-trial numbers
yet -- next update should report its actual val_dice trajectory before
concluding anything.

---

## 2026-08-27 — Audited and closed the real gaps against the official
SIH26143 problem statement

User supplied the actual official PS text and asked for a line-by-line
audit against the real repo before any new work -- reported back
first (per instruction), confirming each of 5 items against real code
citations rather than assumption:

1. Geometric characterization -- confirmed MISSING (pipeline stopped at
   the raw mask/probability array).
2. Age estimation -- confirmed MISSING (correctly absent; PS says "if
   feasible").
3. Forward drift -- confirmed MISSING (`advect.py` had only
   `backward_advect()`).
4. Attribution scoring -- confirmed distance+timing ONLY, verified
   directly in `score_vessels.py`'s code (not the user's recollection).
5. `step3-attribution-design.md` design doc -- confirmed does not exist
   anywhere in the repo (docs/ is empty); its core open question (was
   GFW's events endpoint ever tested with a real token) was real and
   unanswered.

User confirmed: close everything. Full technical reasoning and real
numbers for all of the below are in DECISIONS.md's entry for today
("Closing the real gaps against the official SIH26143 problem
statement").

**Forward drift**: added `forward_advect()` to `src/drift/advect.py`,
sharing one physics core with `backward_advect()` (a `direction`
parameter, not a second copy of the integration loop) -- confirmed
`backward_advect()`'s behavior is unchanged (all 22 pre-existing tests
still pass). Ran for real on both validated cases, both wind sources,
via `scripts/export_drift_dashboard_data.py` (new `HOURS_FORWARD=24`) --
real detection-to-forecast distances range 12-29km depending on
case/source (full table in DECISIONS.md). Wired into `build_map.py` as a
dashed second track layer, distinguishable from the solid backward
trace; dashboard legend updated.

**Attribution -- real GFW v3/events test, first**
(`scripts/test_gfw_events_api.py`, new): called all 5 real event dataset
types against a real token. Real findings: GAP (AIS "went dark") is a
real, working event type with a genuine `intentionalDisabling` flag,
duration, and distance; course/heading and instantaneous speed are
confirmed NOT present in any event schema at this API tier (an external
constraint, not a shortcut); spatial (bbox/geometry) filtering on
v3/events is confirmed forbidden for this token (`403 Not authorized by
permissions`, even on an empty POST body); a `vessels[0]=<id>` GET filter
works and needs no spatial filter, which also turns out to be the right
design (checking already-identified candidates, not searching a fresh
area).

**Built on those real findings**: `gfw_client.fetch_vessel_gap_events()`
(new), `score_vessels.py`'s `trajectory_evidence()` (from the presence
records already fetched -- no new API call) and `behavior_evidence()`
(one real GFW call per top-15 candidate). New `composite_score` field,
kept genuinely separate from the original `score` (both shown, never
silently blended) -- the ranking is now sorted by `composite_score`.
**Real result: ow-0001's top suspect flipped from THOR FREYJA to SANCO
SEA**, driven by a real confirmed intentional AIS gap in the origin
window (not a code change to the underlying distance/timing math). Noted
honestly in DECISIONS.md that this pushes composite_score to exactly
0.000 -- a real but slightly blunt consequence of the bonus size, worth a
future look. ow-0002 had no real AIS gaps among its top 15 -- ALAWAD1
stays #1, an honest null result. Dashboard vessel cards now show the AIS
gap evidence as an always-visible bullet (not buried), plus
composite/proximity scores and trajectory evidence (presence-record
count, closest approach) in the expandable detail.

**Geometric characterization**: added `src/detection/geometry.py`
(`characterize_mask()` -- cv2 contour/minAreaRect, both already-installed
deps, no new one) with 6 new passing tests. Real output on the current
demo tile: ground truth 247,402px across 5 components; current model
prediction is empty (0px, consistent with the already-known
threshold-miscalibration issue). Checked the Zenodo dataset's actual
record page directly and confirmed it does NOT document a Sentinel-1
product type or ground resolution, so real-world km² conversion is
deliberately not applied (pixel units only) -- would otherwise have been
fabricated precision. Wired into `build_dashboard.py`'s Detection
Overlay panel as a new geometry row.

All 28 tests pass (22 original + 6 new geometry tests). Both real and
anonymized dashboard/map builds regenerated and spot-checked (forecast
tracks present, AIS-gap evidence rendering, no real vessel names leaked
into the anonymized build despite the ranking reorder).

---

## 2026-08-27 — Full-pool behavioral rescoring: real cost audit, real
negative result

User flagged a real problem with the previous entry's behavioral
scoring: it only ran against the top 15 candidates, which is a genuine
selection-bias risk (a vessel with a real AIS gap but middling
proximity/timing would never get checked). Asked for a real cost audit
before doing anything -- full reasoning and numbers in DECISIONS.md's
"Full-pool behavioral rescoring" entry for today.

**Real audit result**: GFW's v3/events `vessels[]` filter batches up to
**20 vessel IDs per request** (found by binary-searching the live API --
21 IDs returns a real 422). Full-pool coverage: **18 requests for
ow-0001 (355 candidates), 51 for ow-0002 (1006)** -- 69 total, trivial
against the 50,000/day quota. Confirmed affordable before spending
anything.

**Re-ran both cases against the full pool, not just top-15**:
- ow-0001: 5 of 355 candidates had a real AIS gap. Ranking **unchanged**
  -- SANCO SEA still #1 (its real intentional gap).
- ow-0002: 14 of 1006 candidates had a real AIS gap. Ranking
  **unchanged** -- ALAWAD1 still #1, no gap among its top candidates.

**Honest conclusion: a real negative finding, not padding.** Widening to
the full pool did not surface a stronger candidate in either validated
case -- the earlier top-15 cutoff happened not to introduce bias here,
but that's now a confirmed fact, not an assumption. `gfw_client.py`'s
`fetch_gap_events_batch()` (batched) replaces the old one-vessel-per-call
version; `score_vessels.py` now scores the full pool before selecting
the displayed top-N. New `n_candidates_with_ais_gap` field in the output
JSON keeps the real denominator visible.

Confirmed output stays auditable (proximity/timing/trajectory/AIS-gap
still separate fields -- real sample row for SANCO SEA:
`distance_km=6.67, time_gap_hours=0.0, n_presence_records=57,
closest_approach_km=6.67, ais_gap_count=1, ais_gap_intentional=true,
ais_gap_duration_hours=167.1, ais_gap_distance_km=310.89`, alongside
`score=0.0667` and `composite_score=0.0`). Both dashboard builds
regenerated fresh and re-checked -- zero real-name leaks into the
anonymized build (checked fresh, not assumed from the earlier
spot-check). 28/28 tests still pass.

---

## 2026-08-27 — Map redesign for clarity, still 100% real data

User shared a reference mockup (clean drift cone, ship icons, persistent
"Estimated Origin" callout) and asked the real map to look that simple.
Full reasoning in DECISIONS.md's entry for today.

Real work done: reverse-geocoded both cases via OpenStreetMap Nominatim
(ow-0001 confirmed genuine open water -- "Unable to geocode" even at
zoom 5; ow-0002 confirmed Damietta, Egypt) and added the result as a real
`geo_context` field, shown on the map and in the dashboard panel.
Added the real detected-slick bbox (already used to seed particles,
now also drawn as a rectangle). Replaced the default view's 50 raw
particle-track lines with a single real convex-hull "drift cone"
(`scipy.spatial.ConvexHull` over every real particle position at every
timestep) for both backward and forward tracks -- raw tracks kept as a
togglable layer, not deleted. Vessels now render as real ship icons
instead of dots; the ERA5 origin and the #1 vessel get permanent
always-visible labels instead of click-only popups.

Verified: anonymized build's permanent top-suspect label correctly reads
"Vessel A", not the real name; real map still shows real names; 28/28
tests pass.

---

## 2026-08-27 — Reversed the whole detection-tuning direction: the
original checkpoint was best all along

Built a real threshold-swept IoU comparison (`scripts/
compare_checkpoints_on_val.py`, on the val set, not touching Part III
again) to check whether the three trials' flat `val_dice` was hiding real
progress the metric might be blind to. Real result: it wasn't hiding
progress -- it was hiding real *regression*. The ORIGINAL untouched
checkpoint (pos_weight=32.6, plain DiceBCE) scored real oil-tiles-only
IoU **0.1057** with a genuine graded threshold response; `trial_oversample`
scored 0.0828 with a flat, saturated-looking response; `trial_tversky`
scored exactly **0.0800 at every threshold from 0.18 to 0.50** -- zero
variation, a real sign of a collapsed, non-discriminating output. Full
reasoning in DECISIONS.md's entry for today.

The original checkpoint was the best one from the start. All three
tuning experiments this session made real performance worse while their
training-loop metric looked the same as each other, giving no warning.
Root cause: the original "below trivial baseline" alarm compared the
wrong units (raw dice vs. a dice-equivalent baseline of ~0.058), when
the real IoU-equivalent trivial baseline is ~0.03 and the original
checkpoint's real IoU (0.1057) was already well clear of it the whole
time.

Killed `trial_posweight_sched` after 1 epoch (val_dice=0.0136, already
the lowest of any trial -- same failure signature) rather than let a 4th
experiment run on the same bad assumption. Resumed training from the
real best checkpoint (epoch 39, original loss/pos_weight, unmodified)
with `--use-lr-scheduler` added -- the one real lever not yet tested
against the setup that actually works. Cleaned up several orphaned
monitor processes along the way (confirmed `TaskStop` doesn't reliably
kill the underlying tail/grep pipeline in this environment -- second
time this session; worth remembering to verify manually going forward).

**Training update, same session**: `trial_tversky` at epoch 3/10, also
not showing a breakout so far -- val_dice 0.0227/0.0228/0.0228 (flat),
loss barely moving at all (0.9683/0.9682/0.9680). Not conclusive yet at
3 epochs, but not encouraging alongside oversampling's confirmed-flat
10-epoch result either. One epoch (2) took ~67min instead of the usual
~20min -- explained by real CPU contention from this session's own
concurrent work (Nominatim lookups, drift re-export), not a system
stall. Separately fixed a real stall cause found earlier: this machine's
hibernate-after-1h-on-battery was still enabled (a known risk this
project's LOG.md already flagged once before) and caused a real ~6-hour
gap at epoch 4 of the oversampling trial -- disabled hibernate/sleep on
battery entirely so it can't recur.

---

## 2026-08-31 — Phase 0 metric audit + Phase 1 Colab/reproducibility infra

Before any new training: audited whether the three tuning trials above
were real failures or a broken metric, per a structured diagnostics plan.
Full writeup in `docs/metric_audit.md`; short version:

- **Gate A (cheap, invalidating)**: model overfits 12 fixed oil-containing
  tiles (mean IoU 0.9841, loss 2.37->0.024) -- core pipeline (mask
  alignment, label polarity, loss, normalization) is sound. Confirmed 2
  real SAR bands exist in every Zenodo part (distinct dB stats, not a
  duplicate) with only band 1 consumed by production. Confirmed
  `"lookalike"` is a real, explicit manifest label (582 train / 103 val),
  not a proxy.
- **Gate B (metric trust)**: the training loop's raw `val_dice` has a real,
  now-diagnosed flaw -- any tile with zero ground-truth oil scores ~0
  under it for almost any nonzero prediction (verified numerically), so
  with ~82% of tiles empty, it's structurally near-uninformative. That
  explains why it looked equally flat across all four configs. Separately,
  the "real IoU" used for the actual 0.1057 comparison
  (`scripts/evaluate_test_set.py`/`compare_checkpoints_on_val.py`) was
  independently reproduced from scratch (0.1074, unit-tested, using the
  exact epoch-39 weights preserved in
  `latest_unet_resnet18_epoch39_backup.pt` -- the top-level
  `best_unet_resnet18.pt` has since moved on to epoch 44+ under the
  LR-scheduler continuation and is NOT epoch 39 anymore). All three trials
  re-scored below baseline at their own best threshold, each with a
  distinct, now-quantified failure signature (`oversample`/`tversky`:
  predict-almost-everything, precision ~0.08-0.10, pred-positive-fraction
  0.66-0.99 across every class including no_oil/lookalike;
  `posweight_sched`: generally under-confident probabilities, but only 1
  epoch of training). New finding the old scripts never surfaced: even the
  baseline's best-threshold precision is only ~0.14, with 16-22%
  false-positive pixel rates on clean/lookalike scenes -- 0.1057 describes
  a broadly over-triggering detector, not a precise one.
- **Gate C (data integrity)**: independently re-verified 82.38% of 34,940
  training tiles have zero oil pixels (matches the 82.4% claimed
  previously). Visual audit (`scripts/visualize_dataset.py`, real montages
  in `reports/dataset_visual_audit/`) surfaced an unplanned finding: Band
  1's normalized values are visibly and numerically compressed (median
  0.13, 31% of pixels below 0.10) under the current shared -40/10 dB fixed
  normalization range, while Band 2 sits well-centered (median 0.40) --
  likely, not yet proven, a contributing cause of the weak precision
  above. Train/val leakage: split is at the whole-image level (no
  same-file leakage, confirmed in code); 0 exact-duplicate files found
  across the split via content hash; scene/acquisition-level leakage
  UNPROVEN (no such metadata exists in the files).

**Phase 1**: built `train.py` (config-driven entrypoint wrapping the
existing `src/detection/train.py` loop, unchanged for
`scripts/train_detection.py`), `configs/baseline.yaml` (epoch-39's config
reconstructed field-by-field, `seed` marked UNKNOWN since none was ever
set), per-band normalization (`normalize_db_per_channel`, per Amendment 2),
per-epoch checkpointing + RNG-state true resume + `metrics.jsonl`, a
`run_manifest.json` per run, and `colab_bootstrap.ipynb` +
`requirements-colab.txt` for running experiments on Colab. Protected the
epoch-39 checkpoint at `checkpoints/baseline_epoch39/` (sha256 recorded in
its `MANIFEST.json`).

Real bugs caught by actually running things, not just reading code:
`ZenodoTileDataset`'s `channels` param (added earlier this session) was
never wired into `__getitem__` -- the two-channel read path was
non-functional until fixed (commit `17a8845`, NOT the earlier `c70349a`
which introduced the param). `torch.set_rng_state`/
`torch.cuda.set_rng_state_all` both require CPU tensors, which
`map_location=device` had moved to GPU -- resume crashed until fixed, then
verified working end-to-end on a tiny manifest (correct epoch continuation,
correct loss trajectory, correct keep_last_n pruning).

**Experiment 01 (channel ablation) must pin to commit `17a8845`** (Group 3:
"Add config-driven training entrypoint + Colab reproducibility infra"),
not `c70349a` (Group 1: "Add optional multi-channel support to the
detection preprocessing path") -- the latter's dual-channel path did not
actually work.

---

## 2026-08-31 — Nodata masking, a real eval cache-collision bug found and
fixed, corrected baseline, AMP/channels_last flags, Experiment 01 ablation
subset

**Nodata masking (Finding #12, previously diagnosed but never fixed).**
Exact-0.0-dB-in-both-bands pixels (the real nodata mask from the per-band
normalization work above) were being trained on and scored as if they were
real signal, in both loss and metrics, with no way to exclude them.
Fixed properly, not by zeroing inputs (that's wrong for BCE's mean
reduction -- see `src/detection/losses.py`'s docstrings for why dice/
Tversky can be zeroed safely but BCE needs an explicit masked-mean instead):
`src/detection/preprocess.py::compute_nodata_mask`, threaded through
`ZenodoTileDataset(return_nodata_mask=True)` (computed from RAW pre-despeckle
band values, kept spatially aligned through flip/rotation augmentation --
`augment.py`'s `augment_pair` now accepts `extra_masks`), `losses.py`'s
`dice_loss`/`DiceBCELoss`/`TverskyLoss`, and `metrics.py`'s `iou_dice`/
`precision_recall`/`tile_metrics`/`aggregate_global`, all via an optional
`mask`/`valid_mask` parameter (`None` reproduces the exact prior behavior
for every existing caller). Wired into `train.py`'s config-driven entrypoint
unconditionally (every future run through it is masked, not a config
toggle) -- `configs/baseline.yaml` now documents this as one more permanent
divergence from epoch-39's original run, on top of the already-documented
missing seed.

Real dataset-wide fraction (`scripts/compute_nodata_fraction.py`, full
386-image/6,176-tile validation set, not the earlier 1.81% single-image
estimate): **2.96% of validation pixels are nodata** (47.86M/1.619B); 6.40%
of tiles have any nodata pixel, 1.20% (74 tiles) are entirely nodata.
`no_oil`/`lookalike` images run ~4.1-4.25% nodata vs. `oil` images at 1.55%.
Confirmed the 74 fully-nodata tiles don't produce NaN/divide-by-zero in
`aggregate_global`/`tile_metrics` -- they boolean-index down to empty
arrays, which the existing `union==0`/`tp+fp==0` "nothing to score, call it
perfect" convention already catches for free (checked on the real 74 tiles,
not just synthetic cases -- zero NaN).

**Real bug found while re-running `scripts/evaluate.py` for a fair exp01
baseline: a cache-key collision silently dropped 42/386 validation images
(10.9%) from every report this script has ever produced.** Zenodo's oil/
no_oil/lookalike source folders each number their images independently, so
the same filename stem (e.g. `00098.tif`) exists in more than one class (41
of 386 val stems collide) -- the cache was keyed by bare stem only, so a
later image silently overwrote an earlier one's cache file. This affected
**every historical number from this script**, confirmed directly: all four
existing cache directories (`latest_unet_resnet18_epoch39_backup`, and all
three `trial_*` caches from the earlier tuning-trial comparison) have
exactly 344 npz files, not 386. Fixed by keying cache files
`{label}__{stem}.npz` instead (`scripts/evaluate.py::_cache_filename`),
plus a count cross-check that now warns loudly if this ever recurs.
exp01's own pre-flight check is NOT affected (it never went through this
caching path); `scripts/evaluate_test_set.py` (the separate Part III
test-set script) doesn't use this per-stem caching pattern either, so it
isn't exposed to the same bug -- checked, not assumed, though only via a
targeted grep for the same pattern, not a full independent audit.

**Corrected, fair baseline for exp01 comparison** (`checkpoints/baseline_epoch39/`,
threshold 0.35, full 386-image/6,176-tile validation set, both fixes
applied): **oil IoU 0.1056, Dice 0.1666, Precision 0.1383, Recall 0.4800**
-- supersedes the original 0.1074/0.169/0.139/0.476 (which was both
missing 42 images AND unmasked). Decomposed the two effects on the same
complete corrected cache to isolate them: unmasked-but-complete gives IoU
0.105635 vs. masked-and-complete 0.105625 -- nodata masking's own effect on
this headline metric is negligible (4th-5th decimal place, only 22/1,089
real oil tiles contain any nodata pixel at all); essentially the entire
shift from the original number is the cache-collision fix, not the masking
work. The three historical trial-comparison numbers in `docs/metric_audit.md`'s
Gate B.5 table are very likely affected by the same collision bug too (not
yet re-run) -- the gaps there (precision 0.08-0.10 vs. baseline's ~0.14)
are large enough that the ranking is unlikely to flip, but that's an
assumption, not verified.

**`amp`/`channels_last` config flags** (both opt-in, default off, zero
effect on any caller that doesn't set them): `_resolve_amp(amp, device)` in
`src/detection/train.py` replaces the old hardcoded `device.type=="cuda"`
AMP toggle with an explicit override, still gated to only take effect on
cuda. Verified real, not just wired: `scripts/verify_amp.py` ran 5 real
batches on 4 real oil-class images, identical initial weights and batch
order, `amp=True` vs `amp=False` -- losses close but not identical (max abs
diff 0.0302, mean 0.0079 across the 5 batches), as expected from fp16
reduced precision. `channels_last` moves the model and image batches (not
masks/valid, which it doesn't meaningfully affect) to `torch.channels_last`
memory format the same way.

**Experiment 01 ablation subset manifest** (`scripts/build_ablation_manifest.py`
-- the plain per-image manifest CSV format can't express a tile-level
subset, so `ZenodoTileDataset` gained an `explicit_index` param and
`train.py`'s `load_manifest` now detects an extended 5-column
`row_offset`/`col_offset` format to use it). Scanned the real per-tile
oil presence across all 34,940 real training tiles: **6,155 oil-containing
tiles + 9,308 lookalike tiles (unconditional, all zero-oil by construction)
+ 6,155 randomly-sampled (seed=42) zero-oil non-lookalike tiles = 21,618
tiles selected, 61.87% of the full 34,940-tile training set** -- written to
`data/processed/train_manifest_ablation.csv` (gitignored, regenerable),
original `train_manifest.csv` untouched. Verified end-to-end: `load_manifest`
correctly resolves 2,183 unique source images + exactly 21,618 explicit
tile positions, `ZenodoTileDataset(explicit_index=...)` produces exactly
that many items with correct shapes; `val_manifest.csv` still resolves
`explicit_index=None` (unaffected, full grid). `configs/exp01{a,b,c}`
updated: `epochs: 25` (comparative ablation, not the 60-epoch baseline
run), `amp: true`, `channels_last: true`, `train_manifest` pointed at the
subset, `val_manifest` left at the full 386-image set so results stay
directly comparable to the corrected baseline above. No training run yet.

Test suite: 56/56 passing (36 pre-existing + 20 added this session across
`test_preprocess.py`, `test_losses.py`, `test_augment.py`, `test_metrics.py`,
`test_dataset.py`).
