# DECISIONS.md

This file logs every significant technical choice made on this project, in
order, with the reasoning and tradeoffs considered. Read this first when
resuming work — each session starts with no memory of prior ones, and this
is how continuity is maintained. Append, don't rewrite history.

---

## 2026-08-25 — Project started: SIH26143 oil spill attribution pipeline

Problem statement: SIH26143 (NTRO). Build a system that (1) detects oil
spills in SAR satellite imagery, (2) traces the spill backward through wind
and ocean current data to estimate where/when it started, and (3)
cross-references AIS vessel tracking data to produce a ranked, evidence-
backed list of suspect vessels — not a single verdict.

### Dev machine / GPU

- GPU is an **NVIDIA RTX 4050 (6GB VRAM)**, not the RTX 3060 originally
  assumed. Noted here because 6GB is a real constraint on batch size / patch
  size for the detection model later — flagging now so a future session
  doesn't plan training assuming more headroom than exists.
- Driver reports CUDA 13.1 (backward compatible). Installed PyTorch's
  **cu124** wheel build (see below) since that's the newest stable prebuilt
  index at setup time and is compatible with the 13.1 driver.

### Environment

- Using an existing Python **3.10.10** virtualenv at `venv/` (not 3.13, which
  is the system-wide Python — chosen because PyTorch CUDA wheels and the geo
  stack (rasterio/geopandas) have more reliable prebuilt wheel support on
  3.10 than on 3.13 at this time).
- Install order matters: PyTorch must be installed first with an explicit
  CUDA index URL, *then* the rest of `requirements.txt`, because plain
  `pip install torch` pulls a CPU-only build by default.
  ```
  venv\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  venv\Scripts\pip.exe install -r requirements.txt
  ```

### Repo structure

```
/data/raw          untouched downloaded datasets
/data/processed     despeckled, tiled, ready-to-train data
/src/detection      SAR preprocessing + segmentation model code
/src/drift          backward drift/advection model code
/src/attribution    AIS scoring and ranking code
/src/dashboard       interface layer (built later)
/notebooks           exploration / debugging
/docs
/scripts             one-off / setup scripts (downloads, env checks)
```

### Data sources chosen (Step 0)

1. **Sentinel-1 SAR Oil Spill dataset** — Zenodo, DOI
   `10.5281/zenodo.8346860` ("Part I": train/val). Primary training data for
   the detection model (SAR images + segmentation masks, 1201 pairs,
   2048x2048, Sigma0 in dB per the record's description).
   **Important constraint discovered**: this record publishes exactly two
   files, not per-image files — `01_Train_Val_Oil_Spill_images.7z` (~38GB)
   and `01_Train_Val_Oil_Spill_mask.7z` (~6MB). Both are solid 7z archives
   whose file index lives at the END of the archive, so there is no way to
   cherry-pick "a handful of images" via HTTP Range requests — confirmed
   this experimentally (a truncated 20MB peek of the images archive fails
   to open with `Bad7zFile: invalid header data`). A true small sample of
   real *images* from this specific dataset is not obtainable without
   downloading the full ~38GB archive; there's no smaller preview on
   Zenodo. The mask archive, by contrast, IS genuinely small (6MB, all 1201
   masks) and was downloaded and extraction-verified in full.
   `scripts/download_zenodo_sample.py` reflects this: it always pulls the
   masks in full, and either does a truncated connectivity-only peek of the
   images archive (default) or the full download (`FULL_DOWNLOAD = True`).
   **Action item for a future session**: when ready to actually train,
   flip `FULL_DOWNLOAD = True` and budget the time/disk for ~38GB.
2. **PANGAEA / ESSD Eastern Mediterranean oil spill dataset** — DOI
   `10.1594/PANGAEA.980773`. Used as a real-world validation set: 5515
   annotated patches (oil / look-alike / no-oil), each with a real
   acquisition date/time, Sentinel-1 product ID, and lon/lat corner
   coordinates — exactly what the drift and attribution stages need. Unlike
   the Zenodo record, files here ARE individually downloadable (no account
   needed for single files): `data_matrix.tab` (the full metadata table,
   2.4MB, always downloaded in full) plus per-patch `<tag>.jpg` (640x640
   grayscale quicklook) and `<tag>.xml` (Pascal-VOC-style bounding box
   annotation). `scripts/download_pangaea_sample.py` downloads the full
   table plus a small sample (default 5) of image/annotation pairs — raise
   `SAMPLE_COUNT` for more, or use the account-gated `allfiles.zip` bulk
   download for everything. **Caveat**: these JPGs are quicklook
   visualizations (0-255), not calibrated Sigma0-dB GeoTIFFs like the
   Zenodo set — confirmed by running the calibration check against one
   (see Step 0 log entry). They're real SAR-derived imagery with real
   speckle and a real oil slick visible, useful for geolocation/date
   metadata and for sanity-checking preprocessing code, but not a
   substitute for the Zenodo training data.
3. **Global Fishing Watch API** — vessel presence / AIS data for the
   attribution stage. Requires a free API key: register at
   https://globalfishingwatch.org/our-apis, then generate a token at
   https://globalfishingwatch.org/our-apis/tokens. Verified the real
   endpoint shape via the official docs: `POST
   https://gateway.api.globalfishingwatch.org/v3/4wings/report`, with
   query params (`format`, `group-by`, `temporal-resolution`,
   `datasets[0]`, `date-range`) and a JSON body of `{"geojson": "<geojson
   as a JSON-encoded string>"}`. `scripts/test_gfw_api.py` targets the
   `public-global-presence:latest` dataset (general vessel presence, not
   just fishing activity) over a bbox matching the PANGAEA coverage area.
   **Resolved**: token obtained, request format debugged and confirmed
   working with real data — see "GFW API request format" below and the
   corresponding LOG.md entry.

### GFW API request format (fixed a 422 "body malformed" error)

The SPA docs site doesn't serve endpoint references as static HTML —
`WebFetch` only ever saw the landing page. Real content is available by
appending `.md` to any doc URL (e.g.
`docs/examples/report/report-example1.md`) or via
`/our-apis/documentation/llms.txt`, which lists every page — this is how
the real request format below was found, from actual worked examples
rather than the (JS-rendered, not fetchable) parameter reference.

Two bugs in the original request: (1) `geojson` must be a **plain
`Polygon` object** in the body, not a `FeatureCollection` and not
`json.dumps()`-stringified; (2) `spatial-resolution` (`LOW`/`HIGH`) is a
**required** query param that was missing. Full worked example with real
returned vessel data in LOG.md.

### Preprocessing (Step 0)

`src/detection/preprocess.py` implements: `load_sar_image` (rasterio,
single-band), `check_calibration` (heuristic — flags values outside a
Sentinel-1 ocean Sigma0-dB range of roughly -40 to +10 dB, catching 8-bit
quicklooks and raw linear power as the two common non-calibrated failure
modes; it cannot *prove* correct calibration, only catch obviously wrong
input), `lee_filter` (adaptive Lee despeckling via local-variance
weighting, `scipy.ndimage.uniform_filter`), and `tile_image` (fixed-size
tiling, drops partial edge tiles rather than padding them). Demoed end to
end in `scripts/demo_preprocess.py` against a real PANGAEA sample
(`ow-0001.jpg`) — see the Step 0 LOG.md entry for the result. `scipy` and
`py7zr` were added to `requirements.txt` (scipy for the Lee filter; py7zr
to extract the Zenodo `.7z` archives) beyond the originally-specified list.

## 2026-08-25 — Step 1: detection model training pipeline + sanity pass

### Sanity training data: the 1201 Zenodo masks and the 5 PANGAEA pairs are NOT directly usable together

The Zenodo masks (1200 real, pixel-accurate) have no matching real images
yet (the 38GB images archive is still undownloaded — see Step 0). They
cannot form image/mask training pairs on their own. What they ARE useful
for right now: `scripts/compute_mask_class_balance.py` extracts all 1200
and computes the real oil-pixel fraction per mask — **mean 2.98%, median
1.76%, min 0.12%, max 57.4%, zero masks with no oil pixels at all**. That
real number (not a guess) sets the loss's BCE `pos_weight = (1 -
0.0298)/0.0298 ≈ 32.6` in `src/detection/losses.py`.

The 5 PANGAEA rows requested for sampling produced only **4 unique
images** (`ow-0001`–`ow-0004`; the 5th table row was a second object on
the same `ow-0004.jpg`, so the download script's dedup correctly skipped
re-fetching it). These have real images but only bounding-box annotations,
not pixel masks. `src/detection/dataset.py:bbox_to_mask` rasterizes each
box into a rectangular pseudo-mask. This is the real (if approximate)
data the sanity pass in `scripts/sanity_train.py` actually trains on:
despeckled PANGAEA image + bbox-rasterized pseudo-mask. It is NOT a
substitute for real segmentation training — the mask shape is a rectangle,
not the true irregular slick outline — it exists purely to prove the loop
works on real (not synthetic) images before the full Zenodo data arrives.

### Secrets / .env

Scripts that need an API key (currently `scripts/test_gfw_api.py`) load it
from a `.env` file at the project root via `python-dotenv`'s `load_dotenv()`,
called at the top of the script before `os.environ.get(...)` is used. `.env`
itself is gitignored (confirmed) and contains real secrets locally only —
it currently has one placeholder line, `GFW_API_TOKEN=paste_your_token_here`,
to be replaced with a real token once GFW registration is done. **Any new
secret key added to this project (the upcoming CDS/ERA5 API key included)
should follow this same pattern**: add a `KEY_NAME=placeholder` line to
`.env`, read it via `os.environ.get("KEY_NAME")` after a `load_dotenv()`
call at the top of whatever script needs it, and never hardcode or commit
the real value.

### Detection model architecture: U-Net + ResNet18 encoder

Built with `segmentation_models_pytorch` (added to requirements.txt, pulls
in `timm`) rather than hand-rolled, to avoid re-implementing a
well-tested encoder/decoder and to get ImageNet-pretrained encoder weights
for free (loaded via Hugging Face Hub on first run — confirmed working,
no auth needed, just an "unauthenticated requests" rate-limit warning).

Chose **U-Net decoder + ResNet18 encoder** over the other options
mentioned:
- vs. ResNet34: ResNet18 has fewer layers/params (14.3M total for the full
  U-Net+ResNet18 vs. more for ResNet34), and both use the same plain
  conv-block design, so ResNet18 is the strictly more memory-efficient of
  the two with no architectural downside beyond raw capacity — the right
  call while the training set is still tiny and there's no evidence more
  capacity is the bottleneck.
- vs. EfficientNet-B0: fewer raw parameters (5.3M) but its inverted
  residual / depthwise-separable blocks produce more intermediate
  activation tensors per layer, which is what actually drives VRAM use
  during training (not parameter count). ResNet18's plain residual blocks
  are more memory-predictable.
- vs. DeepLabV3+: its ASPP head keeps several parallel dilated-conv
  feature maps in memory at once, which is a worse fit for 6GB than
  U-Net's single skip-connection decoder path for the same encoder.
- in_channels=1 (single-band SAR, not RGB) is supported directly by smp's
  `Unet(in_channels=1)` — confirmed working.

Empirical result (see LOG.md Step 1 entry) validates this choice: peak
VRAM was 354MB at 256x256 tiles and 818MB at 512x512 tiles, both far under
the 6GB budget — there's substantial headroom to raise batch size and/or
move to ResNet34 once real training data volume justifies it.

### Loss function: Dice + weighted BCE

`src/detection/losses.py: DiceBCELoss` sums Dice loss (region-overlap,
insensitive to background pixel count dominating the signal) and
BCE-with-logits weighted by the real empirical `pos_weight ≈ 32.6` above.
Plain accuracy or unweighted BCE would let the model collapse to
"predict all background" and still score >97% given how rare oil pixels
are in a typical scene.

### Training loop: AMP + optional gradient accumulation

`src/detection/train.py` uses `torch.amp.autocast`/`GradScaler` from the
first training step (not added later), given the 6GB VRAM ceiling.
`grad_accum_steps` lets a larger effective batch size be simulated on
small physical batches once physical batch size becomes the constraint —
not yet needed at the current data volume (see sanity-pass VRAM numbers
above) but wired in for when it is.

## 2026-08-25 — Drift/hindcast track started (parallel to Zenodo download)

### Working case for this track

`ow-0001` from the PANGAEA dataset (already downloaded in Step 0): real
oil detection at **33.06°E, 33.26°N**, Eastern Mediterranean between
Cyprus and Egypt, **2019-01-01T03:42:35 UTC**, Sentinel-1B product
`S1B_IW_GRDH_1SDV_20190101T034300_20190101T034325_014295_01A97E_39B8`.
Chosen because we already have the real image, bbox annotation, and exact
lat/lon corners for it from Step 0/1.

### Ocean currents: HYCOM GLBy0.08 reanalysis — open access, confirmed working

`tds.hycom.org` THREDDS server, no account/API key needed (data is marked
"Approved for public release. Distribution unlimited."). Confirmed via a
real OPeNDAP metadata query. Structure: one 12Z cycle per day, hourly
τ-offsets covering roughly the next 24-36h from each cycle (e.g. our
2019-01-01T03:42 case needs the **2018-12-31 12Z cycle at τ≈16h**).
**Caveat**: this specific mirror only serves `_sur` files, which contain
**barotropic (depth-averaged) velocity**, not true 0m surface velocity —
an approximation adequate for a skeleton but worth upgrading to a proper
near-surface product (e.g. CMEMS Mediterranean reanalysis) later.

### Wind: ERA5 via Copernicus CDS — requires a free account (pending, like GFW)

Confirmed the real requirements: register at
https://cds.climate.copernicus.eu, obtain a personal API token from your
profile page, save it to `~/.cdsapirc` as:
```
url: https://cds.climate.copernicus.eu/api
key: <PERSONAL-ACCESS-TOKEN>
```
then `pip install "cdsapi>=0.7.2"`. You must also manually accept the
dataset's Terms of Use on its page (e.g.
https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels)
before the API will serve data — this can't be done via the API itself.
**Action item, exactly like the GFW key: register and accept the ToS,
then this can be wired in** — account creation needs your email
verification, so this isn't something done on your behalf.

**Update, same day**: registration is done and `~/.cdsapirc` was set up,
but `cdsapi.Client()` fails with `PermissionError: Permission denied:
'C:\Users\Asus/.cdsapirc'` — a `PermissionError` (not `FileNotFoundError`)
on Windows when opening a path that's actually a **directory** is a strong
signal that `C:\Users\Asus\.cdsapirc` got created as a folder instead of a
file (confirmed: `ls` on that path listed a directory containing a
misnamed `New Text Document.txt`). This needs to be fixed on the user's
machine directly — the session sandbox can't read/write outside the
project directory to fix it. `src/drift/wind_era5.py` is written and
ready; blocked purely on this local file-vs-folder mixup, not on code or
CDS access itself.

**Update, same day**: `.cdsapirc` fixed, `cdsapi.Client()` now connects
and submits real requests. Next blocker, exactly as anticipated above:
`403 Forbidden -- required licences not accepted`. Fix is manual, at
https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download#manage-licences
(while logged in) — accept the dataset's license there, then re-run
`scripts/run_drift_skeleton.py` with no code changes needed.

**Update, same day — working end to end**: license accepted, real ERA5
data now downloads successfully (24 hourly, 0.25deg steps for our window,
cached in `data/processed/era5_cache/`). One more real bug found: the
modern CDS-Beta download API names dimensions `latitude`/`longitude` and
`valid_time` instead of the legacy short names (`lat`/`lon`/`time`) that
`wind_ncep.py`'s NCEP/NCAR dataset uses — fixed with a rename step in
`wind_era5.py` so both sources present the exact same `lat`/`lon`/`time`
shape to `advect.py`. **ERA5 is now the primary wind source; NCEP/NCAR is
kept as a documented fallback** (`wind_ncep.py`, unchanged) for if ERA5/CDS
access ever breaks — both implement the same `fetch_wind_window(...)`
interface, so switching is a one-line change in whatever script picks the
source (see `scripts/run_drift_skeleton.py`, which already runs both).

**Real comparison result** (case ow-0001, 24h backward): NCEP/NCAR and
ERA5 origin estimates are **6.67 km apart** — see LOG.md for full numbers.
That's roughly as large as the total ~8km backward displacement itself,
so the wind source choice materially affects the answer here: not wildly
inconsistent (same general direction/magnitude, similar latitude), but a
real, non-negligible divergence concentrated in longitude, consistent with
ERA5's 0.25deg grid resolving mesoscale wind structure that NCEP/NCAR's
1.875deg grid smooths away. This is a concrete argument for ERA5 as the
default in later stages, with NCEP/NCAR treated as a degraded-but-workable
fallback, not an equivalent substitute.

**Unblocking wind data now, before ERA5 access exists**: NCEP/NCAR
Reanalysis 10m wind via NOAA PSL's OPeNDAP server
(`psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/surface_gauss/`) —
confirmed open, no account, real 2019 data. Coarser than ERA5 (~1.875°
grid, 6-hourly vs. ERA5's 0.25°/hourly) but adequate for a skeleton. Swap
for ERA5 once CDS access exists.

### Drift model approach: windage + backward Euler, skeleton built and working

Implemented in `src/drift/`: `currents.py` (HYCOM, see above),
`wind_ncep.py` / `wind_era5.py` (common `fetch_wind_window(start, end,
lat_range, lon_range) -> xr.Dataset` interface, so `advect.py` never knows
which wind source it's using), and `advect.py` (particle seeding +
backward integration). Physics: `particle_velocity = ocean_current +
0.03 * wind` (the standard "3% of wind speed" oil-drift rule of thumb),
**no Ekman/Coriolis deflection angle yet** — a known simplification,
flagged for later refinement, not full accuracy. Integration is explicit
Euler in lon/lat space at hourly steps, with a flat-Earth cos(lat)
correction for longitude spacing — adequate for a ~24h/small-area
skeleton, not validated for longer windows or higher latitudes.

Three real bugs were found and fixed while getting this to run against
real data (worth knowing about if touching this code):
1. **NCEP/NCAR**: concatenating two years' *lazy* remote OPeNDAP datasets
   across a year boundary before subsetting silently produced all-zero
   wind data (no error, just wrong numbers) — fixed by subsetting +
   `.load()`-ing each year's data into memory first, then concatenating
   the small in-memory pieces.
2. **HYCOM**: the `tau` variable's units (`"hours since analysis"`) aren't
   valid CF time units, so `xarray`'s default time decoder crashes opening
   the file — fixed with `decode_times=False` (we don't need decoded time
   values from that file; the cycle/tau are already computed independently).
3. **HYCOM cycle/tau math**: rounding a target time within ~30s of the
   next day's 12Z cycle could round tau up to exactly 24, one past the
   valid 0-23 range for that cycle, requesting a file that doesn't exist
   — fixed by rolling over to the next cycle's tau=0 when this happens.

**First real result** (`scripts/run_drift_skeleton.py`, case ow-0001, 24h
backward, NCEP/NCAR wind): see LOG.md for the full numbers side by side
with the ERA5 run once that's unblocked.

## 2026-08-25 — Training data: Part I is oil-only; look-alike negatives come from Zenodo Part II/III, not PANGAEA

**Confirmed from the actual files** (not just the record description):
Zenodo Part I (`zenodo.8346860`, already downloaded) is genuinely
oil-positive-only — all 1200 masks have at least one oil pixel (0 masks
are empty; verified via `scripts/compute_mask_class_balance.py`'s saved
per-mask fractions, min fraction 0.12%). This matches the record's own
description ("This part contains only the training and validation images
for Oil spill.") There is no look-alike or oil-free class anywhere in
Part I.

**PANGAEA look-alikes are NOT a good source of hard negatives for this
training set — rejected.** The PANGAEA images are 8-bit JPG quicklooks
(confirmed back in Step 0); Zenodo Part I is calibrated Sigma0-dB
GeoTIFFs. Mixing them would mean **100% of negative examples come from one
pixel domain (JPG) and 100% of positive examples from another (calibrated
dB)** — since Part I has zero true negatives of its own, there'd be no
same-domain negatives to dilute this confound. A model trained on that mix
would have an easy shortcut available (8-bit quantization / JPEG
artifacts / value-range differences) and could hit high accuracy while
having learned "which dataset is this from" rather than "is there oil
here" — this would not be a model that honestly rejects look-alikes.

**Better option found instead**: the Zenodo record's own metadata pointed
to two more parts of the same dataset family, not previously noticed:
- **Part II** (`10.5281/zenodo.8253899`): `01_Train_Val_No_Oil_Images`
  (685 oil-free scenes) + `01_Train_Val_Lookalike_images` (685 look-alike
  scenes), each with a matching all-zero mask ("Since the focus was on
  oil spills, the value of the ground truth of the look-alike images is
  only 0" — i.e. look-alike masks are deliberately zeroed, exactly the
  hard-negative signal we want: real look-alike SAR signatures labeled
  "not oil"). Same calibrated Sigma0-dB format as Part I. ~22.9GB (no-oil)
  + ~23.0GB (look-alike) = **~46GB**.
- **Part III** (`10.5281/zenodo.13761290`): a proper held-out **test**
  set, balanced 150 Oil / 150 No-oil / 150 Look-alike images+masks, same
  calibrated format. **~9.9GB**.

This is the right fix: same-domain (calibrated dB), genuinely labeled
look-alike negatives, from the same dataset family and pipeline as Part
I, rather than a cross-domain JPG substitute. **Cost: ~56GB more
download**, on top of the 38GB Part I already on disk (~94GB total) — a
real time/disk commitment comparable to the original Part I decision, so
this was reported back before downloading rather than assumed.

## 2026-08-25 — Train/val/test methodology for the real training run

**Training pool**: Part I (1200 oil-positive images) + Part II (685
no-oil + 685 look-alike, same calibrated domain, all-zero masks) = 2570
source images. Split **85/15 train/val**, stratified by class (oil /
no-oil / look-alike) so val isn't accidentally skewed toward one category
by chance, plus a secondary stratum on oil-fraction quartile within the
oil-positive class (using the real per-mask fractions computed in Step 1)
so val also sees a representative spread of "barely any oil" vs "lots of
oil" scenes, not just a random subset that happens to cluster at one end.

**Test set**: Part III (150/150/150 balanced Oil/No-oil/Look-alike) is
kept **completely separate and untouched** until a final, one-time
evaluation after training and any hyperparameter tuning are fully decided.
This is standard practice, but worth stating explicitly and holding to:
if Part III gets used even once to pick a checkpoint or tune a
hyperparameter, it stops being a valid measure of generalization and
starts being an indirect part of the validation set — the number reported
at the end would then be an overestimate of real-world performance, not
because of any single reuse being "cheating" but because repeated
peeking, even indirectly through decisions it influenced, leaks
information about the held-out set into the model-selection process. So:
Part III is read exactly once, at the very end, to report the number
that goes in front of anyone judging this project.

## 2026-08-25 — Dashboard: built fresh (no prior mockup existed), real drift data wired in

The user referenced "the earlier mockup" when asking for dashboard work;
checked for one first rather than assuming -- `src/dashboard/` had been
empty since Step 0, git history has one commit (Step 0 only) with no
dashboard files, and the working tree had nothing dashboard-related,
tracked or untracked. There was no earlier mockup. Built one fresh,
designed to be the reference structure going forward, per the same
"placeholder swap-in" contract the user asked for.

**Structure** (`src/dashboard/`):
- `build_map.py`: real Leaflet map via `folium`, reading
  `data/processed/dashboard/drift_ow0001.json` (produced by
  `scripts/export_drift_dashboard_data.py`, which re-runs the drift
  skeleton and additionally dumps per-particle final positions and full
  backward tracks -- not just the aggregate centroid/std that
  `scripts/run_drift_skeleton.py` prints -- so the map shows real particle
  spread, not a synthetic circle standing in for it). Detection marker,
  per-source origin markers, toggleable track/scatter layers, a
  distance line between source estimates with the real km value labeled,
  satellite/light basemap toggle, fullscreen control.
- `build_dashboard.py`: the page shell. `render_detection_panel()` and
  `render_vessel_panel()` are explicit placeholder functions with a
  docstring stating exactly what real output should replace them with and
  where -- swapping in real data later means replacing one function body
  each, not restructuring the page. A `PIPELINE_STATUS` dict drives status
  pills in the header from the project's actual real state (not
  hardcoded "coming soon" text) -- e.g. the vessel-ranking placeholder
  correctly says GFW access is confirmed working (2276 real records for
  this case) while the ranking *algorithm* itself doesn't exist yet
  (`src/attribution/` is still empty), rather than a generic "not ready".
- Colors follow the project's validated categorical palette (dataviz
  skill): ERA5 (primary wind source) = slot 1 blue, NCEP/NCAR (fallback)
  = slot 2 orange -- the same pairing already used elsewhere to talk about
  "primary vs fallback", now carried into the map.

**Not published as a claude.ai Artifact**: the map's basemap tiles load
from external servers (CartoDB, Esri) at render time, which the
Artifact sandbox's CSP blocks (only Google Fonts are allowed as an
external origin) -- the map would render blank/broken there. Sent as
local files instead; open `index.html` in a real browser (with `map.html`
alongside it) for the working interactive map.

**Real finding while building this**: NCEP/NCAR (the documented wind
fallback) was itself unreachable -- `psl.noaa.gov` timed out completely on
a direct `curl` test, while HYCOM and general internet connectivity were
both fine, ruling out our own bandwidth/network as the cause. This is a
genuine NOAA-side outage, observed live. Made
`scripts/export_drift_dashboard_data.py` resilient to exactly this: each
wind source is now tried independently and a failure is logged and
skipped rather than crashing the whole export -- since ERA5 is already
the primary source (see above), losing NCEP/NCAR temporarily doesn't
block the map. The map was built with ERA5 data only as a result; will
show both once NOAA's service is back and the export is re-run.

## 2026-08-25 — Attribution scoring: first-pass real implementation

Built while Zenodo Part II/III downloaded in the background (independent
work, per the user). `src/attribution/`:
- `gfw_client.py`: `fetch_vessel_presence(bbox, date_range, ...)`,
  extracted from `scripts/test_gfw_api.py` once a second caller (scoring)
  needed the same request logic. Same request shape already verified
  working (see "GFW API request format" above).
- `score_vessels.py`: scores every real candidate vessel from GFW against
  the drift-estimated origin (centroid + estimated origin time =
  detection_time - hours_back, from `src/drift/`).

**Methodology**: each vessel's *evidence row* is whichever of its GFW
presence records is closest in time to the origin estimate (a vessel can
have several rows across the query window). Score = `0.5 * min(distance_km
/ 50, 1) + 0.5 * min(time_gap_hours / 24, 1)` -- distance and time gap
each normalized against a documented scale (50km, 24h -- the latter
matching the drift window itself) and equally weighted. **These weights
and scales are a first-pass heuristic, not calibrated against any labeled
ground truth** -- none exists for this project (there's no known-correct
answer for "which real vessel caused this spill" to tune against). Treat
the ranking as a reasonable, inspectable first ordering -- exactly why
every row keeps its raw distance_km and time_gap_hours alongside the
combined score, not just the score alone.

**Known limitation surfaced by the real output, then fixed**: GFW's
`LOW` spatial-resolution presence data reports each vessel's position as
a grid cell center, not an exact location -- the real top-ranked vessels
for ow-0001 initially included pairs sharing the *exact* same distance
value (e.g. two different ships both at 4.6km), which was this grid
coarseness showing up in the data, not a bug. Switched
`score_vessels.py` to request `HIGH` spatial resolution instead --
confirmed it performs fine (~10s for the same bbox/date-range, similar
record count: 2661 vs 2276) and fixed the tie: the same two ships now
separate cleanly to 2.6km and 5.2km. The top-ranked vessel identity
(THOR FREYJA) stayed the same across both resolutions, which is a good
consistency signal for the ranking, not just noise from the grid change.

**Real result, case ow-0001** (origin estimate: 33.1981°N, 32.9503°E at
2018-12-31T03:42:35 UTC, from the ERA5 drift run; `HIGH` spatial
resolution): 2661 GFW presence records -> 334 unique candidate vessels.
Top of the ranked list: "THOR FREYJA" (MMSI 311000273, Bahamas flag,
2.6km from the origin estimate) and "PHOENIX III" (MMSI 374559000, Panama
flag, 5.2km), both with a 0h time gap (i.e. both vessels were present
exactly when the origin estimate says the spill likely started). Full
ranking in
`data/processed/dashboard/vessel_ranking_ow0001.json`, wired into the
dashboard's vessel-ranking panel (`src/dashboard/build_dashboard.py`),
replacing that placeholder with this real (if uncalibrated) output.

## 2026-08-25 — Pipeline validated on a second real case (ow-0002)

Everything up to this point (drift + attribution) had only ever been run
on ow-0001. Ran the full pipeline against a second, independent real
PANGAEA case to check it wasn't accidentally tuned to the first one --
**no methodology code changed**, only the case's real coordinates/date
were swapped in (refactored `scripts/export_drift_dashboard_data.py` and
`src/attribution/score_vessels.py` to take a `CASE_ID`/case registry
instead of hardcoded ow-0001 constants, so this was a config change, not a
rewrite).

**Case ow-0002**: real detection at 32.029°E, 31.685°N,
2019-01-04T15:56:38 UTC (near the Egyptian coast, close to the Suez Canal
approaches -- a much busier shipping area than ow-0001's location).

- **Drift**: origin estimate (31.754°E, 31.577°N) -- a ~28km SW shift,
  notably larger than ow-0001's ~8km shift, which is plausible (different
  date, different real wind/current conditions, closer to the coast where
  currents can run stronger) rather than a red flag. Particle spread
  stayed tight (σlon=0.0015°, σlat=0.0027°), same order of magnitude as
  ow-0001 -- consistent behavior across cases is a good sign the physics
  code itself isn't case-specific.
- **Attribution**: 10,180 GFW presence records (vs. ow-0001's 2,661 --
  this area has much more real traffic, consistent with being near Suez)
  -> 1,006 unique candidate vessels. Top match: "CAPTAIN AMIR" (MMSI
  667001723, Sierra Leone flag), 0.5km from the origin estimate, 0h time
  gap -- a materially tighter top match than ow-0001's 2.6km, though
  that's more likely explained by this being a denser-traffic area (more
  candidates means a better chance one lines up closely) than by the
  scoring method being "more accurate" here -- worth keeping in mind
  before reading too much into score magnitudes across cases.

**Conclusion**: the drift + attribution pipeline runs end-to-end on real
data for a second case without any code changes beyond swapping in that
case's real parameters -- a genuine generalization check, not just a
demo that only works for one hand-tuned example.

## 2026-08-25 — Detection inference plumbing built (using the Step 1 sanity checkpoint)

The dashboard's detection-overlay panel was a placeholder because no real
model exists yet -- but the *inference code itself* (load checkpoint ->
predict -> overlay) didn't exist either, and writing it only once real
training finishes would mean doing it under time pressure. Built it now
against the Step 1 sanity checkpoint instead, mirroring the same pattern
already used for drift and attribution: prove the plumbing works on real
data now, swap in the real artifact later with no code changes.

`src/detection/inference.py`: `load_model_for_inference` (loads a
checkpoint's `model_state_dict` into a fresh `build_model()`, no ImageNet
weights since the checkpoint supplies its own) and `predict_mask` (runs
the same despeckle -> `normalize_db_fixed` preprocessing used in training,
so inference sees exactly the same input distribution the model was
shown). `scripts/render_detection_overlay.py`: picks a real Zenodo Part I
image with a decent amount of real oil (>=5% of a 512x512 tile, found by
scanning real masks rather than picking an image at random and risking an
all-water tile), runs inference, renders a 3-panel real SAR tile / real
ground truth / predicted mask comparison.

**Result, run on a real Zenodo tile (`00039.tif`)**: IoU against the real
ground truth = **0.08** -- correctly poor, and expected to be poor: the
Step 1 checkpoint was trained on 4 PANGAEA images with rectangular
bbox-shaped pseudo-masks, never on real Zenodo data or real irregular
slick shapes. The rendered comparison makes this honestly visible -- the
real ground truth shows an irregular slick outline, the prediction is a
blob-ish rectangle, visibly wrong in the expected way, not a fabricated
"looks plausible" result. Wired into the dashboard
(`render_detection_panel` in `build_dashboard.py`, PNG embedded as a
base64 data URI so `index.html` stays self-contained) -- clearly labeled
throughout as the sanity checkpoint, not a real model.

**Swap-in point for the real model**: once `scripts/train_detection.py`
finishes, point `CHECKPOINT_PATH` in `render_detection_overlay.py` at the
real best checkpoint and re-run it -- the dashboard panel picks it up
automatically, no HTML/layout changes needed.

## 2026-08-25 — Dashboard: case switcher (ow-0001 / ow-0002 both live)

The ow-0002 validation (drift + attribution) only existed in JSON files
and log entries -- the actual dashboard deliverable still only showed
ow-0001. Added a tab switcher so both real, independently-validated cases
are visible in the UI itself, not just documented.

**How it's implemented** (`src/dashboard/build_dashboard.py`): each
case's map-panel + vessel-ranking-panel pair is wrapped in a `<div
class="case-block" data-case="..." style="display:contents">`. `display:
contents` makes the wrapper invisible to CSS Grid layout -- its children
participate directly in `main`'s existing two-column grid as if they were
its own children, so multiple cases' panels can share the same two grid
slots without a layout rewrite. Only the active case's block is
`display: contents`; every other case is `display: none` (fully removed
from rendering, so there's never more than one map-panel/vessel-panel
pair actually occupying the grid at once). A small vanilla-JS click
handler toggles which case-block is `contents` vs `none` and updates the
tab's `active` class -- no framework, no build step, consistent with the
rest of this dashboard.

`src/dashboard/build_map.py` was generalized the same way as the drift
and attribution scripts earlier (a `CASE_IDS` list instead of a single
hardcoded case) -- writes `map.html` for the first case and
`map_{case}.html` for the rest.

The detection-overlay panel is deliberately NOT per-case -- it's
demonstrating the inference plumbing on a Zenodo training tile, not a
result for either specific case, so it stays outside the case-blocks and
is always visible regardless of which tab is selected.

Verified via direct HTML inspection (not just "it built without
errors"): both tabs render with the correct `data-case` values, both
case-blocks are present with the right default visibility (ow-0001
`contents`, ow-0002 `none`), both real vessel rankings (THOR FREYJA /
CAPTAIN AMIR) are embedded, and the click handler is present.

## 2026-08-25 — Dashboard visual redesign (dark navy/amber/teal theme)

The user supplied a complete concept mockup ("Slick Trace") -- a dark
navy/amber/teal design with a stat strip, evidence-card vessel list, and
a decorative canvas-drawn map -- and asked for our dashboard to match it
"and even more better." Rebuilt the dashboard's visual language to match
(fonts: Chivo/IBM Plex Sans/IBM Plex Mono via Google Fonts; the navy/amber/
teal/alert color tokens; stat strip; evidence-bullet vessel cards with a
confidence bar; corner tag), keeping every number wired to real data --
the reference's map, vessel names, and stats were all fictional/simulated
("Illustrative scenario... vessel names, IMO numbers and positions are
fictional"), ours are not, so the corner tag says "Prototype - Real Drift
+ AIS Data" rather than copying the reference's "Simulated Data" wording,
which would misrepresent this project's actual state.

**Where this goes further than the reference, not just matches it**: the
reference's map was a decorative canvas drawing with fabricated ship
icons at fixed fractional coordinates -- ours is the real Leaflet map
(already built), now also plotting the top 10 real candidate vessels at
their real GFW-reported (lon, lat) grid-cell positions, color-coded (top
suspect = amber, matching the drift trace's color; others = teal, matching
the sidebar cards) with real popups. This required adding `lon`/`lat` to
`src/attribution/score_vessels.py`'s output (they were computed internally
for scoring but not previously saved) -- re-ran scoring for both cases to
regenerate the ranking JSONs with this field.

**Real stat strip** (`render_stat_strip` in `build_dashboard.py`), all
four numbers real and case-dependent:
- Detection -> Origin: real haversine distance between the detection
  point and the ERA5 origin centroid.
- Estimated Origin Time: `detection_time - hours_back`, real.
- Vessels Evaluated: real GFW candidate count for that case.
- Top Suspect Confidence: `(1 - score) * 100` for the #1-ranked vessel --
  a derived, not independently-measured, number (same caveat as the
  ranking score itself, carried into the stat tile's meaning, not hidden).
No stat was fabricated to fill the reference's 4-tile layout; the two
tiles that didn't have an obvious real equivalent (the reference's
"Spill Area Detected" in km2, which needs a trained detection model we
don't have) were replaced with different real numbers rather than
invented.

**Vessel cards**: each vessel's 3 evidence bullets are all real, derived
directly from its GFW record: distance from the origin estimate, time gap
phrasing (explicit "present exactly at the origin time" when the gap is
zero, which is common and worth stating plainly rather than just showing
"0.0h"), and the real AIS entry/exit timestamps formatted compactly.

**Detection overlay restyled to match**: `scripts/render_detection_overlay.py`
now renders on a dark navy matplotlib figure instead of a white one (was
a visible white box against the dark dashboard before this pass) -- red
for the real ground truth mask (matches "detection point" elsewhere: a
real, confirmed thing), amber for the model's prediction (matches
"drift trace & origin estimate": a model-derived output).

Verified visually via a real headless-browser screenshot (Edge, found
already installed on the machine, `--headless --screenshot`), not just
by inspecting the generated HTML source -- caught and fixed the white-box
issue this way before calling it done, per the dataviz skill's "render
it and look at it" step.

**Follow-up, same day -- page background treatment**: added a layered
`body` background: a faint repeating-grid pattern (48px spacing, ~5%
opacity fog color, both axes) echoing the map panel's own lat/lon
graticule styling, plus a soft amber radial glow behind the header, over
the solid `--navy-void` base. Deliberately low-opacity and placed only on
`body` (not on any `.panel`, which all keep their own solid
`--navy-panel` background) so it reads as ambient texture in the gaps
between panels, never competing with panel content or hurting text
contrast. Verified via another real screenshot, not just by reasoning
about the CSS values.

## 2026-08-25 — GFW compliance check, heading/speed scoring investigated (not feasible yet), Ekman deflection added

Three follow-up items the user asked for together; results below.

**1. GFW rate limits, checked for real** (previously never verified,
just assumed headroom): confirmed via the actual docs page --
**50,000 requests/day, 1,500,000/month**, shared across up to 5 tokens
per user, enforced per-user (not per-token), returns `429` when
exceeded. We've made on the order of 20-30 total requests this entire
project. No scaling concern at current usage; worth remembering if
attribution ever runs across many cases in a loop.

**Real compliance gap found and fixed**: GFW's Terms of Use (CC BY-NC
4.0, non-commercial only) require attribution -- "Powered by Global
Fishing Watch" with a link -- on anything displaying their data. The
dashboard didn't have this. Added to the footer in
`build_dashboard.py`, alongside a plain-language non-commercial-use
note. This is a real ToS requirement, not cosmetic.

**2. Heading/speed-consistency scoring: investigated, not currently
feasible.** Checked the actual fields in a real GFW
`public-global-presence` report record: `callsign, dataset, date,
entryTimestamp, exitTimestamp, firstTransmissionDate, flag, geartype,
hours, imo, lastTransmissionDate, lat, lon, mmsi, shipName, vesselId,
vesselType` -- no course/heading/speed field. This dataset reports
presence intervals per grid cell per day, not individual AIS pings, so
there's no per-position kinematic data to score against. Getting
heading/speed would mean integrating a different GFW endpoint (their
vessel tracks/events API) -- real new API surface, not a quick addition
to the existing scoring function. Documented here rather than
implemented with fabricated data or silently dropped.

**3. Ekman/Coriolis deflection added to the drift model** (this had been
an openly-flagged simplification since the skeleton was first built --
see "Drift model approach" below). `src/drift/advect.py`: the wind-driven
component of particle velocity is now rotated by `DEFLECTION_ANGLE_DEG =
20.0` (a commonly-cited empirical value in oil-spill trajectory
literature) -- clockwise (to the right) in the Northern Hemisphere,
counter-clockwise (to the left) in the Southern Hemisphere, applied
per-particle by that particle's current-step latitude sign (so it stays
correct if a future case straddles the equator, even though neither of
our current Mediterranean cases needs that). Deliberately only applied to
the wind term, not the HYCOM current term -- the current field already
embeds whatever real Ekman transport exists in that ocean model's state,
so deflecting it again would double-count the effect. Verified the
rotation math directly before trusting it in the full pipeline: magnitude
preserved, NH deflects right of the wind direction, SH deflects left
(`tests/test_advect.py`).

**Real result of adding this** (re-ran the drift skeleton, then attribution
scoring, for both real cases):

| case | origin (no deflection) | origin (20 deg Ekman deflection) | shift |
|---|---|---|---|
| ow-0001 | 32.9503E, 33.1981N | 32.9318E, 33.2097N | ~1.9 km |
| ow-0002 | 31.7540E, 31.5772N | 31.7095E, 31.6108N | ~5.4 km |

**Consequential, not cosmetic**: for ow-0001, the top-ranked suspect
stayed the same (THOR FREYJA, distance shifted from 2.6km to 2.9km) --
reassuring stability. But for **ow-0002, the top-ranked suspect actually
changed** -- "CAPTAIN AMIR" (previously #1 at 0.5km) dropped out of the
top spot, replaced by "ALAWAD1" (now #1 at 1.6km, was previously ranked
lower). This is worth stating plainly: a physics refinement that sounds
like a minor correction (20 degrees on a 3%-of-wind-speed term) was
large enough to flip which real vessel the pipeline names as the leading
suspect in a real case. That's exactly why the project reports a ranked
list with visible evidence rather than a single verdict -- the "answer"
is sensitive to modeling choices that are each individually defensible
but not beyond question, and a reader should be able to see the runner-up
candidates and their evidence, not just trust whichever vessel currently
sits at #1.

## 2026-08-25 — Test suite added

No unit tests existed anywhere in this project before now. Added
`tests/` (pytest, `requirements.txt` updated), covering the pure-logic
functions -- not network calls, not model training, those are exercised
for real by the scripts elsewhere in this project:
- `test_geo.py`: `haversine_km` against known distances (1 degree
  latitude/longitude ~= 111km) and symmetry.
- `test_currents.py`: `cycle_and_tau`, specifically regression-testing
  the tau=24 rounding bug that actually happened during this project
  (a target time 23h59m31s past a cycle's 12Z used to round tau up to
  24 and request a HYCOM file that doesn't exist) -- plus a sweep of
  every hour of a day confirming tau never leaves [0, 23].
- `test_advect.py`: the new Ekman deflection rotation (magnitude
  preservation, correct hemisphere-dependent direction).
- `test_score_vessels.py`: `time_gap_hours`'s three cases (inside
  window, before, after), that closer vessels score better, that a
  missing `vesselId` is skipped rather than crashing, and that a vessel
  appearing in multiple GFW rows keeps only its best-scoring row.
- `test_dataset.py`: `bbox_to_mask` rasterizes Pascal-VOC boxes
  correctly (single box, multiple boxes, no boxes).

All 22 tests pass. Run via `venv\Scripts\python.exe -m pytest tests\ -v`
(also documented in README.md).

### Attribution/ranking approach — first-pass scoring decided and implemented (see above)

Distance + timing vs. the drift-estimated origin, see the entry above this
section for the full methodology, weights, and known limitations.
Heading/speed-consistency scoring (mentioned as a possibility when this
section was first written) not implemented yet -- a real refinement to
consider once the current distance+timing baseline has been checked
against more cases.

---

## 2026-08-25 — Corrupted archive diagnosis and the parallel-download-safety fix

**Why the archive was almost certainly corrupted by a concurrent writer,
not a bad connection**: a single interrupted download leaves the `.part`
file short (resumable, harmless -- this is the normal/expected failure
mode this project has hit repeatedly). What was found instead was the
opposite: the *finished* file (already renamed from `.part`, matching
`dest.exists()` in `download_full`) was 9.2GB *larger* than Zenodo's own
reported size. The only way `download_full`'s logic produces a file bigger
than expected is if two independent runs of it were both opened in
append (`"ab"`) mode against the same `.part` file at the same time, each
unaware of the other's writes -- interleaving/duplicating bytes into a
corrupt result once renamed. Given an earlier session logged "running the
download under a small bash auto-retry wrapper... now running under a
persistent auto-retry wrapper" for exactly this file's record, a wrapper
that doesn't correctly detect a still-alive prior run before retrying is
the leading explanation. Not fixed in `download_full` itself this
session (that would mean adding lock-file machinery to a function that's
otherwise proven itself across three large archives) -- instead, the
one corrupted file was deleted and the redownload used a design that
can't have this failure mode at all (see below).

**Why 4 connections, and why a separate preallocated file, not a shared
`.part`**: the Part I 16-connection parallel-download experiment (see the
"Full Zenodo dataset download started" LOG entry) already answered the
speed question -- Zenodo throttles per-connection, not per-account/IP, so
parallelism helps a lot (~12 MB/s at 16 connections vs ~0.4 MB/s at 1) --
but crashed on the full run from connection-count instability, and that
session's own conclusion was "a *modest* parallelism (e.g. 3-4
connections instead of 16) might sustain... worth trying." This session
did exactly that at N=4, getting ~1.7-1.8 MB/s combined (a real ~10-15x
speedup over single-stream) without hitting the instability. Each of the
4 threads seeks to and writes only its own pre-assigned, non-overlapping
byte range of a preallocated file (`scripts/download_zenodo_part2_parallel.py`)
-- structurally incapable of the interleaved-write corruption diagnosed
above, since no two threads (and no accidental second script run) can
ever write to the same byte range. Final size and `py7zr` header
integrity are both checked before the file is renamed into place, so a
bad download is caught immediately rather than silently reused later
(as the original corrupted file was).

## 2026-08-26 — "Confidence" relabeled to "match score"; anonymized
dashboard build added for public-facing use

**"Confidence" was the wrong word.** The attribution score is an
uncalibrated composite of distance + timing (explicitly documented
above, "Attribution scoring", as not calibrated against any labeled
ground truth -- no such dataset exists for this problem). But the
dashboard displayed it as "97% confidence" / "98% confidence", which
reads as a calibrated probability of guilt to anyone not reading the
methodology fine print, including a judge skimming the screen. Relabeled
everywhere user-facing -- stat tile ("Top Suspect Match Score"), vessel
cards, sort dropdown, comparison table, map tooltips/popups -- and
renamed the underlying field itself (`confidence_pct` ->
`match_score_pct` in `src/attribution/score_vessels.py`'s output, not
just the display strings) so the code's vocabulary matches what's shown,
rather than leaving a "confidence" field internally that now displays as
something else. Existing `vessel_ranking_*.json` files were updated
in place (key renamed, values untouched) rather than re-querying GFW.
Same substance, same distance/timing breakdown next to it as before --
just no implied precision the methodology doesn't back up.

**Real vessel identity is a reputational exposure, not just a technical
one.** THOR FREYJA and ALAWAD1 (the real top suspects in ow-0001/
ow-0002) are real, identifiable ships. Naming them as "suspects" in a
pitch that may get recorded, screenshotted, or shared past the judging
room is a genuine exposure for a real company's real vessel, even
though the methodology is explicit that this is a ranked lead and not a
verdict. Decided with the user to anonymize deliberately rather than let
it happen by omission: real vessel identity stays in the live dashboard
shown directly to judges (proof the pipeline runs on real GFW data, not
a mockup), but anything built for a deck, recording, or screenshot uses
`--anonymize` (added to `build_dashboard.py` and `build_map.py`, writing
to `output_anon/` instead of `output/`). `src/common/anonymize.py`
replaces `ship_name`/`mmsi`/`imo` with fictional stand-ins ("Vessel A",
"REDACTED") while leaving distance, timing, score, flag, and
vessel_type real -- those are evidence, not identity. Scoped narrowly to
the fields the user actually flagged; lon/lat marker positions are left
real since stripping them would break the map visualization and wasn't
what was asked.

## 2026-08-26 — Report/PDF export added, closing the UI plan's last item

Asked directly what could honestly be pushed to 100% completion while the
detection-model loss/pos_weight trials run. The detection model itself
can't be -- its real accuracy is gated on those trial results, not a
to-do list Claude can just clear. But the UI plan (see "case comparison
table" entry above, "5 of 6 done ... only report/PDF export remains")
had exactly one open item, and it's a real, boundable piece of scope:
built it rather than leave the last checklist item permanently open.

Kept it a static, client-side export rather than adding a PDF-generation
dependency (reportlab/weasyprint) the rest of the codebase doesn't use --
an "Export Report (PDF)" button (`window.print()`) plus a `@media print`
stylesheet in `build_dashboard.py`: hides interactive-only chrome (sort
dropdown, case tabs, the corner tag, the button itself), force-expands
every vessel card's click-to-reveal detail block (a browser can't click
during print, so the IMO/vessel-type/raw-score evidence would otherwise
be silently dropped from the PDF), force-shows both case containers
(normally one is `display:none` for the tab switcher) so the exported
report covers everything rather than whatever tab happened to be open,
and drops the map panel (a live Leaflet/OSM iframe needs network tile
loads at print time, which isn't guaranteed -- the vessel evidence table
is the report's real content anyway). Added `print-color-adjust: exact`
so the dark navy/amber/teal theme survives into the PDF instead of
browsers stripping backgrounds by default. A print-only header (title,
real generation timestamp, the same "not calibrated against labeled
ground truth" methodology line already on the live dashboard) makes the
exported PDF self-contained and exactly as honest as the on-screen
version -- same substance, no less caveat.

## 2026-08-26 — Tile-level oil-sampling imbalance found and measured;
weighted sampler added as a third, higher-priority trial

Asked directly whether there was a better lever than the two loss/
pos_weight trials already running. Rather than guess, checked the
actual training pipeline end to end (model.py: encoder is already
ImageNet-pretrained; augment.py: flips/rotation/dB-speckle-jitter are
already SAR-appropriate -- neither was the gap) and found one:
`ZenodoTileDataset` tiles every training image into a non-overlapping
512x512 grid and samples uniformly across all of it, but 685 `no_oil` +
685 `lookalike` images are 100% oil-free by construction, and most
tiles even within the 1200 real oil images miss the slick entirely.
`pos_weight` in `DiceBCELoss` only reweights pixels *inside* a tile that
already has some oil -- it cannot inject signal into a batch that has
none.

Measured it for real rather than assuming (`scripts/
analyze_tile_oil_distribution.py`, reading every real training mask's
actual tile grid): of 34,940 real training tiles, **82.4% have zero oil
pixels** (0% within `no_oil`/`lookalike`, and even within `oil`-labeled
images only 37.7% of their own tiles touch the slick). At batch_size~8,
that puts roughly 1 in 5 batches at zero oil-learning gradient
regardless of loss/pos_weight choice -- a signal-sparsity problem
neither running trial addresses.

Added `compute_oil_tile_weights()` to `src/detection/dataset.py`:
reads each indexed tile's real mask once, solves for the oil-tile
weight multiplier that hits a target per-epoch oil-tile representation
(default 50%) under `WeightedRandomSampler`, kept as a standalone
function rather than baked into `ZenodoTileDataset` so it stays an
independently-testable variable -- `train()` gained a `sampler` param
(replacing `shuffle=True` when given, since DataLoader disallows both),
and `train_detection.py` gained `--oversample-oil-tiles` /
`--target-oil-fraction`. Validation is intentionally left unsampled
(real class distribution) so val_dice stays an honest, unbiased metric.

Reordered the trial queue rather than running all three losing time to
GPU/disk contention on a single 6GB card: killed the just-started
Tversky trial (its real epoch-1 result, val_dice=0.0227, kept via its
saved checkpoint -- not discarded, just deprioritized) and requeued
oil-tile oversampling (original DiceBCE/pos_weight=32.6, isolating the
sampling variable alone) first, then Tversky resuming from its real
epoch-1 checkpoint, then the pos_weight+scheduler trial as originally
planned.

## 2026-08-27 — Closing the real gaps against the official SIH26143
problem statement (forward drift, attribution trajectory/behavior,
geometric characterization)

The user supplied the actual official problem-statement text and asked
for a line-by-line audit against the real repo (not an assumption-based
one) before any new work, then to close every real gap found. Full audit
results were reported back before starting (see that turn); this entry
covers what was actually built afterward, with real numbers, per the
audit's five items.

### Forward drift forecasting added

The PS names this twice ("predict the future flow," "backward AND
forward") and it was a total, confirmed gap: `src/drift/advect.py` had
only `backward_advect()`. Added `forward_advect()` sharing one physics
core (`_advect()`, new) with `backward_advect()` via a `direction`
parameter (-1 backward, +1 forward) -- same windage/current/Ekman
physics either way, one integration loop to keep correct instead of two
copies that could silently drift apart. `backward_advect()`'s own
behavior is bit-for-bit unchanged (direction=-1 reproduces the original
sign exactly); all 22 pre-existing tests plus the new ones still pass.

Wired into `scripts/export_drift_dashboard_data.py` (`HOURS_FORWARD=24`,
same window length as `HOURS_BACK` -- a documented symmetric choice, not
a physical constraint) and run for real on both validated cases, both
wind sources -- real detection-to-forecast distances:

| case | source | detection&rarr;backward-origin | detection&rarr;forward-forecast |
|---|---|---|---|
| ow-0001 | ERA5 | 12.95 km | 12.16 km |
| ow-0001 | NCEP/NCAR | 10.44 km | 16.30 km |
| ow-0002 | ERA5 | 31.36 km | 24.56 km |
| ow-0002 | NCEP/NCAR | 23.21 km | 29.35 km |

Wired into `src/dashboard/build_map.py` as a second, visually distinct
track layer -- same color per wind source as the backward trace, but
dashed (`dash_array`) rather than solid, with its own toggleable Leaflet
layer and popup, plus a forward-arrow marker at the forecast centroid.
Legend updated (`build_dashboard.py`) to read "Backward trace (hindcast)"
vs. "Forward trace (forecast)" rather than the previous single "Drift
trace" label, now that there are two.

### Attribution: trajectory + behavioral-anomaly scoring

**Confirmed via direct code read** (not the user's recollection) that the
existing `score_vessels()` used only distance+timing, exactly as
described -- see the audit turn for the file/line citations.

**Real GFW v3/events API test** (`scripts/test_gfw_events_api.py`, new --
separate from `test_gfw_api.py`, which only tests v3/4wings/report):
called all 5 real event dataset types
(`public-global-{fishing,encounters,loitering,port-visits,gaps}-events:latest`)
against a real token. Real, confirmed findings:
- **GAP is a real, working event type** with rich real fields: `gap.intentionalDisabling`
  (GFW's own suspected-deliberate-AIS-shutoff flag), `durationHours`,
  `distanceKm`, `impliedSpeedKnots`, `on`/`offPosition`.
- **Course/heading and instantaneous per-position speed are NOT present**
  in any of the 5 real event schemas returned -- checked directly, not
  assumed from docs. `fishing`/`loitering`/`encounter` events do carry an
  *averaged* speed over the whole (often multi-day) event, not an
  instantaneous heading/speed at a point in time. This is a real,
  confirmed external constraint: a literal "course deviation" score isn't
  buildable at this API tier.
- **Spatial (bbox/geometry) filtering on v3/events is forbidden for this
  token's tier**: `POST /v3/events` with a `geometry` body returns real
  `403 {"message":"Not authorized by permissions"}` -- confirmed by
  direct test, and confirmed NOT specific to the geometry field (an empty
  POST body gets the same 403; the whole POST path is restricted). A
  named-region filter (`region-ids[0]`/`region-datasets[0]`, real
  kebab-case params -- GFW's own docs show them in camelCase, which
  actually 422s) works via GET, but a named region (EEZ/RFMO/etc.) is far
  too coarse for this project's few-degree case bboxes.
- **Real workaround found and used**: `vessels[0]=<vessel_id>` on GET
  works with this token and needs no spatial filter at all -- confirmed
  with a real vessel_id already known from the presence-record step
  (`fetch_vessel_presence`'s `vesselId` field turns out to be the same ID
  space as `vessel.id` in events responses). This is also the *right*
  design regardless of the permission issue: gap events are checked per
  already-identified candidate vessel, not via a fresh area search.

**Implemented, given those findings** (`src/attribution/gfw_client.py`'s
`fetch_vessel_gap_events()`, `src/attribution/score_vessels.py`'s
`trajectory_evidence()`/`behavior_evidence()`):
- **Behavioral-anomaly sub-score**: real AIS-gap check per top-N
  candidate. `GAP_INTENTIONAL_BONUS=0.15` / `GAP_ANY_BONUS=0.05`
  (documented, not tuned -- same honesty as `DISTANCE_SCALE_KM`/
  `TIME_SCALE_HOURS`) subtracted from `score` to produce `composite_score`,
  a genuinely separate field shown alongside the original `score` (never
  silently blended) -- both are in the output JSON and the dashboard's
  vessel-card detail rows, and the AIS-gap evidence itself (duration,
  distance, GFW's intentional flag) is an always-visible card bullet, not
  buried in the expandable detail.
- **Trajectory evidence**: `n_presence_records` and `closest_approach_km`,
  computed from the SAME presence records already fetched for the
  distance+timing pass (no new API call) -- looks at ALL of a candidate
  vessel's real rows in the query window, not just the single
  closest-in-time one `score_vessels()` keeps, to see whether its track
  came closer to the origin at some other point. This substitutes for a
  true continuous track (which the API doesn't expose at this tier) with
  something real and auditable rather than fabricated.
- **Documented limitation, not hidden**: behavioral/trajectory checks only
  run for the top `TOP_N=15` candidates (already narrowed by
  distance+timing, which is free/already-fetched) -- each vessel checked
  costs one real GFW call, so this doesn't re-scan the full 355-1006 raw
  candidates per case for a hidden AIS-gap outlier ranked outside the
  top 15. A real engineering tradeoff, stated in the output JSON's
  `methodology.note`, not silently assumed away.

**Real result: this changed an actual ranking.** ow-0001's top suspect
flipped from THOR FREYJA (distance-only score 0.029) to **SANCO SEA**
(distance-only score 0.067, but a real confirmed intentional AIS gap in
the origin window drops its composite_score to 0.000, clipped at the
floor) -- a real, substantive change driven by genuine new evidence, not
a cosmetic one. Worth flagging honestly: composite_score hitting exactly
0.000 (match_score_pct=100%) is a real, if slightly blunt, consequence of
`GAP_INTENTIONAL_BONUS` (0.15) exceeding SANCO SEA's original score
(0.067) -- a future session might reconsider whether an intentional gap
should be able to fully zero out an otherwise-real distance/timing
mismatch, rather than just discount it. Flagging this rather than quietly
shipping a suspiciously round number. ow-0002 had no real AIS gaps among
its top 15 -- ALAWAD1 stays #1, an honest null result, not forced signal.

### Geometric characterization added

Confirmed via direct search of `src/` that nothing computed area/length/
width/orientation/shape from a mask -- the pipeline stopped at the raw
probability array (`detection/inference.py`). Added
`src/detection/geometry.py`'s `characterize_mask()`: `cv2.findContours` +
`cv2.minAreaRect` on the largest connected component (both already
project dependencies, no new one added) -- area, length, width,
orientation, elongation (length/width ratio, a basic real-vs-look-alike
shape cue per the PS's "characterise... shape"), and component count (a
real spill can fragment into several pieces). 6 new pytest tests
(synthetic masks with known geometry), all passing alongside the
existing 22.

**Real-world units (km²/m) are deliberately NOT computed for the demo
tile.** Checked the actual Zenodo dataset record page
(`10.5281/zenodo.8346860`) directly rather than assuming a resolution --
it documents pixel dimensions (2048x2048) and calibration (Sigma0-dB) but
explicitly does NOT state a Sentinel-1 product type or ground sample
distance. Reporting a km² number without a real documented GSD would be
exactly the kind of fabricated precision this project's `DECISIONS.md`
and `LOG.md` have consistently avoided elsewhere (see "Confidence
relabeled to match score"). `characterize_mask()` accepts an optional
`pixel_size_m` and only computes real-world fields when it's given
explicitly -- never assumed. (The two validated PANGAEA drift cases DO
have a confirmed real Sentinel-1 IW GRDH product ID with a known 10m
spec, but geometric characterization isn't wired to those images in this
pass, only to the Zenodo Part III demo tile shown in the dashboard's
Detection Overlay panel.)

Real output on the current demo tile (Part III, `00000.tif`): ground
truth 247,402 px across 5 real components (the largest component's
bounding box spans nearly the full 512px tile -- elongation reads as
1.0, a real but somewhat blunt consequence of `minAreaRect` on a
large/irregular multi-part shape, not a bug); the current trained
model's prediction on this tile is empty (0 px, consistent with the
already-documented threshold-miscalibration issue, not a new problem).
Wired into `build_dashboard.py`'s Detection Overlay panel as a new
`.geo-strip` row showing both ground-truth and prediction geometry side
by side, explicitly labeled "pixel units only" with the reason why.

### Design doc (item 5) confirmed absent, not partially present

`docs/` is empty and no file matching `step3-attribution-design.md` (or
the `S_prox`/`S_time`/`S_course`/`F_gap`/`S_type` model it allegedly
contained) exists anywhere in the repo or in LOG.md/DECISIONS.md history.
Reported as genuinely absent rather than guessed at. Its core open
question -- whether GFW's events endpoint was ever tested against a real
token -- is answered directly above: it had not been, until this session.

## 2026-08-27 — Full-pool behavioral rescoring (closing a real
selection-bias gap from the previous entry)

The previous entry's behavioral/trajectory scoring only ran against the
top 15 candidates by proximity+timing, with a stated reason (one GFW call
per vessel, so checking the full 355-1006-vessel pool looked expensive).
The user correctly flagged this as a real selection-bias risk, not just a
caveat: a vessel with a genuine AIS gap but middling proximity/timing
would never surface, since the behavioral check never ran on it. Asked to
audit the real cost before doing anything, per this project's own
"verify against real behavior" pattern rather than assuming.

**Real cost audit, done before committing to anything**: tested whether
`v3/events`' `vessels[]` filter can batch multiple vessel IDs into one
request (it can -- confirmed with a real 5-vessel batch first), then
binary-searched the real batch-size limit against the live API (not
documented anywhere by GFW): **20 vessel IDs succeeds, 21 fails with a
real, slightly misleading 422** (`"each value in vessels must be a
string"` / `"vessels must be an array"` -- actually an array-length cap,
not a type error). That makes full-pool coverage `ceil(n/20)` requests,
not `n`: **18 requests for ow-0001's 355 candidates, 51 for ow-0002's
1006** -- 69 total, trivial against the real 50,000/day quota. Confirmed
affordable before spending any GPU/API time on it, per the user's
explicit ask.

**Implemented**: `gfw_client.fetch_gap_events_batch()` replaces the
previous one-vessel-at-a-time `fetch_vessel_gap_events()`, batching
`MAX_VESSELS_PER_EVENTS_REQUEST=20` IDs per real request. `score_case()`
now runs behavioral+trajectory scoring against the ENTIRE `ranked` pool,
re-sorts the full pool by `composite_score`, and only then slices to
`TOP_N` for the displayed ranking -- selection happens after full-pool
evidence, not before.

**Real result: a genuine negative finding, reported honestly rather than
padded.** Re-running both cases against their full pools found real AIS
gaps in **5 of ow-0001's 355 candidates** and **14 of ow-0002's 1006**,
but in both cases none of the additional gap-flagged vessels were close
or timely enough to break into the top 15 -- **both rankings are
unchanged** from the top-15-only version (SANCO SEA still #1 for
ow-0001 via its real intentional gap; ALAWAD1 still #1 for ow-0002, no
gap at all among its top candidates). The earlier top-15 restriction did
not, in these two specific validated cases, hide a stronger candidate --
but that's a fact now confirmed by checking, not something that could
have been assumed from the restriction's existence alone. New
`n_candidates_with_ais_gap` field added to each case's output JSON so
this real denominator (5/355, 14/1006) stays visible, not just the ones
that made the top 15.

Confirmed the output stays fully auditable after this change (proximity,
timing, trajectory, and AIS-gap fields are still separate, un-blended
JSON fields -- see a real sample row in LOG.md). Both dashboard builds
(real + anonymized) regenerated fresh from the full-pool results and
re-checked for real-name leakage -- zero leaks, confirmed by grep, not
assumed to still hold from the earlier spot-check.

## 2026-08-27 — Map redesign: simplified default view, still 100% real data

User supplied a reference mockup (clean drift cone, labeled ship icons,
a persistent "Estimated Origin" callout, visible coastline/geography) and
asked the real map to look that clean and understandable. Same tension as
the earlier "Dashboard visual redesign" entry: match the reference's
clarity without copying its actual approach, which uses fabricated ship
icons at fixed positions -- everything here stays wired to real data.

**Real geographic context, verified rather than guessed**: reverse-
geocoded both cases' real coordinates via OpenStreetMap Nominatim.
ow-0001 (33.06E, 33.26N) returns "Unable to geocode" even at zoom 5 --
confirmed genuine open water, not attributable to one country -- so it's
labeled "Levantine Sea, Eastern Mediterranean, open water between Cyprus
and the Nile Delta coast" (consistent with `scripts/run_drift_skeleton.py`'s
existing docstring). ow-0002 (32.03E, 31.68N) returns a real match:
Damietta (Dumyat), Egypt. Added `geo_context` to `CASES` in
`export_drift_dashboard_data.py`, exported to the drift JSON, shown as a
fixed label on the map and in the dashboard's panel subtitle.

**Real detected-slick footprint**: added `detection_bbox` to the exported
JSON -- the actual labeled-object bbox (PANGAEA metadata) already used to
seed advection particles, now also drawn as a real (if small, ~1km)
rectangle on the map, instead of only a point marker.

**Drift reconstruction cone replaces 50 raw lines as the default view**:
`build_map.py`'s new `drift_cone_latlon()` computes a real convex hull
(`scipy.spatial.ConvexHull`, already an installed dependency via
`scipy`/`geopandas` -- no new dependency added) over every real particle
position at every real simulated timestep, both backward and forward, and
draws that single translucent polygon as the default-visible layer. The
raw per-particle tracks are NOT deleted or hidden permanently -- they're
still real, still there, just moved to a togglable "raw particle tracks"
layer (`show=False` by default) for anyone who wants to inspect individual
trajectories. Simplifies the default view without losing or approximating
the underlying physics output.

**Ship icons instead of dots**: candidate vessels now render with
`folium.Icon(icon="ship", prefix="fa")` at their real GFW positions.
folium's built-in Icon only accepts named colors, not arbitrary hex, so
"orange"/"cadetblue" (closest real matches to this theme's amber/teal)
are used for the icon background rather than trying to force exact hex --
a deliberate, minor concession to using folium's built-in icon system
instead of hand-rolling SVG markers.

**Persistent callouts, matching the reference's always-visible info
boxes**: the ERA5 (primary) origin marker now shows a permanent Leaflet
tooltip with the real origin date/time/coordinates, and the #1-ranked
vessel gets a permanent "★ {real ship name} — top suspect" label, instead
of requiring a click for either. NCEP/NCAR (secondary/toggle source) and
non-top candidates keep click/hover-only info so the default view doesn't
end up with a dozen overlapping permanent boxes. `DARK_CHROME_CSS` gained
`.leaflet-tooltip` styling (previously only popups/controls were dark-
themed) so these new permanent tooltips match the theme instead of
showing Leaflet's default white box.

Verified after rebuilding both real and anonymized maps/dashboards fresh:
the new permanent top-suspect label correctly shows "Vessel A" (not the
real name) in the anonymized build, real names only appear in the real
build, and 28/28 tests still pass.

## 2026-08-27 — The whole detection-tuning diagnosis was chasing a
misleading metric; reversed course

Three genuinely different fixes (pos_weight reduction, oil-tile
oversampling, a full Tversky loss swap) all left the training loop's
`val_dice` flat in the same ~0.020-0.023 band. Rather than accept that at
face value, built `scripts/compare_checkpoints_on_val.py` -- a real
threshold-swept oil-tiles-only IoU comparison on the validation set
(reusing `evaluate_test_set.py`'s real logic, deliberately NOT touching
Part III test set again, per this project's own "read it exactly once"
rule) -- to check whether the flat raw metric was hiding real learned
discrimination the metric itself is structurally blind to (it's computed
on raw, unthresholded sigmoid probabilities -- see `train.py`'s
`evaluate()` -- which heavily penalizes low absolute confidence
regardless of real relative separation between oil and background).

**Real result, and it overturned the whole diagnosis**: the ORIGINAL,
untouched checkpoint (pos_weight=32.6, plain DiceBCE, no oversampling)
scored the best real oil-tiles-only IoU of the three -- **0.1057 at
threshold 0.35**, with a real, smooth threshold-sensitivity curve
(0.089-0.106 across the sweep, meaning its probabilities carry genuine
graded confidence). `trial_oversample`'s best checkpoint scored **0.0828**,
flat at exactly 0.0799 across thresholds 0.18-0.35 -- a real sign of a
saturated/degenerate probability distribution, not graded confidence.
`trial_tversky`'s best checkpoint scored **0.0800 at every single
threshold from 0.18 to 0.50, with zero variation** -- an extremely strong
signal the model collapsed to a near-binary output that doesn't
discriminate at any operating point in that range.

**The original checkpoint was the best one the entire time.** All three
loss/sampling interventions this session made REAL, properly-evaluated
performance worse, not better -- even though their raw training-loop
metric looked similarly flat to each other, giving no hint of this.

**Root cause of the misdiagnosis**: the original alarm ("val_dice
plateaued below a trivial predict-all-oil baseline") compared raw
unthresholded dice (~0.0233) against a dice-equivalent trivial baseline
(~0.058) -- but the equivalent trivial baseline in real IoU terms is only
~0.03, and the original checkpoint's real thresholded IoU (0.1057) is
already well above that. The raw metric made an already-reasonable model
look broken, and three real experiments got spent chasing that false
signal, each one trading away real calibrated confidence for a lower
raw-metric number that looked the same as before regardless.

**Reversed course rather than run a 4th loss variant blind**: killed
`trial_posweight_sched` after 1 epoch (its epoch-1 val_dice, 0.0136, was
already the lowest of any trial -- consistent with the same failure
mode) and resumed training from the REAL best checkpoint
(`data/processed/checkpoints/latest_unet_resnet18.pt`, epoch 39,
unmodified pos_weight=32.6/DiceBCE -- the one lineage now demonstrated to
actually work) with `--use-lr-scheduler` added on top -- the one lever
never tested against the setup that's real evidence shows performs best,
rather than against a loss/sampling change that was actively hurting it.

**Lesson for the rest of this project, not just this run**: raw
`val_dice` (unthresholded) is not a reliable signal for model-selection
or go/no-go decisions in this class-imbalance regime -- a real
threshold-swept IoU check should run before trusting it for any future
tuning decision, not after three experiments have already been spent.
