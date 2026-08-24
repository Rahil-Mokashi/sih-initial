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
   **Action item: get an API key and set the `GFW_API_TOKEN` env var
   before running `scripts/test_gfw_api.py`** — the script correctly
   detects the missing token and exits with instructions; the actual
   authenticated call is still unverified.

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

### Detection model architecture — not yet decided

No model architecture has been picked yet. Step 0 is environment + data
plumbing only. This section will be filled in when we start Step 1
(detection model).

### Drift model approach — not yet decided

Backward advection through wind + ocean current fields (likely ERA5 for
wind, a reanalysis product like HYCOM or CMEMS for currents) — approach and
specific data products not yet chosen. To be decided in Step 2.

### Attribution/ranking approach — not yet decided

Will score candidate vessels from AIS tracks against the estimated
spill origin time/location. Scoring method (proximity, heading consistency,
speed profile, etc.) not yet designed. To be decided in Step 3.
