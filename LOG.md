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
