# Presentation status — read-only audit

Independently verified on 2026-08-31, immediately before the presentation. Every claim below was checked against files on disk, code, or a live run/screenshot during this pass — not recalled from a prior write-up. Git HEAD: `89eff00`, working tree dirty (uncommitted files listed at the end; nothing was committed, built, or trained during this audit).

A near-identical draft of this file already existed (untracked) before this audit started, apparently from an earlier pass earlier tonight. I did not trust it — I re-derived every claim below independently (ran pytest, started both servers and curled them, screenshotted both UIs and the drift/tile-locations maps with headless Edge, loaded the checkpoint files in torch, diffed the configs, read the JSON outputs directly). Where I found something the draft got wrong or missed, it's flagged explicitly below.

---

## 1. DETECTION

**Current best checkpoint, metrics, threshold**

Two checkpoints both look like "the best one" — they are not the same, and only one has a trustworthy metric:

- `checkpoints/baseline_epoch39/unet_resnet18_epoch39.pt` (epoch 39) — **oil-tiles-only IoU = 0.1074, Dice = 0.1691, precision = 0.139, recall = 0.476, at threshold 0.35**, on the validation set. Verified directly: `MANIFEST.json` records this, and it matches `docs/metric_audit.md`'s independent, unit-tested re-derivation.
- `data/processed/checkpoints/best_unet_resnet18.pt` — the file the older eval scripts read by default. I loaded it directly with `torch.load`: **epoch field = 44**, `val_dice = 0.0238` (the same value baked into a second file, `_epoch45_check_best.pt`, confirming this is the LR-scheduler continuation run, not epoch 39). `val_dice` is the same structurally-broken metric documented below — it has never been scored with real IoU/precision/recall. **Nobody knows if epoch 44 is better or worse than epoch 39.**

**Status: WORKS-BUT-STATIC.** The epoch-39 number is real, reproducible, and I re-confirmed the file it's computed from at `data/processed/checkpoints/latest_unet_resnet18_epoch39_backup.pt` (sha256-pinned in the MANIFEST). Do not call epoch 44 "current best" — there is no real evidence either way.

**What epoch-39 can and cannot detect, plainly**

Verified visually via `web/index.html` (screenshotted live, see §2):
- On real oil tiles it shows genuine, graded signal: IoU ranged 0.286–0.781 across the 6 oil examples shown, with visibly correct (if imprecise) overlap between predicted and true slick shape.
- It over-triggers badly. Precision is only 0.139 overall. On the 2 lookalike examples it predicted **100% of the tile as oil** on scenes with zero real oil. On the 2 clean-ocean examples it predicted 92.3% and 79.1% oil coverage. **In plain terms: it cannot reliably tell oil apart from lookalikes or clean ocean** — it is strongly biased toward calling everything oil.
- Do not say "high confidence" to a judge about this model's precision. The real number is ~14%.

**Experiment 01 status**

- Three configs exist: `configs/exp01{a,b,c}_band{1,2,both}_perband.yaml`.
- I diffed the uncommitted changes myself: the only edit tonight was fixing the per-band normalization range (`[-43.673, 0.0]` → `[-43.710, -12.894]`) to exclude the 1.81%-of-pixels nodata plateau that was sitting at the old range's upper bound — a real, disclosed fix, not a new feature. `epochs: 60` is unchanged, no `amp`/`channels_last` keys were added.
- **Zero training has happened.** No checkpoint, log, or timing number exists for any exp01 run. I found no evidence of even a killed-partial run tonight (no exp01-tagged files under `data/processed/checkpoints/` or `logs/`).

**Status: PARTIAL.** Configs exist and (per LOG.md) were pre-flight-verified to load and shape-check correctly. No training has been run.

---

## 2. THE TWO UIs

### `src/dashboard/output/index.html`

Started it myself (`python -m http.server 8001 --directory src/dashboard/output`), curled every asset (200s), and rendered it with headless Edge (`msedge.exe --headless --screenshot`).

| Panel | Status | Notes |
|---|---|---|
| Header / KPI strip | WORKS-BUT-STATIC | Confirmed live in screenshot: 12.9km, 355 vessels, 100% top suspect, all real precomputed numbers for ow-0001 |
| Case Comparison table (ow-0001/ow-0002) | WORKS-BUT-STATIC | Confirmed live in screenshot |
| Drift Reconstruction map (iframed `map.html`) | **RISK — basemap tiles are unreliable, not confirmed clean** | See finding below. The particle tracks, vessel markers, and origin estimate all render correctly (confirmed by screenshot). The basemap underneath is a different story — see next section. |
| Ranked Suspect Vessels | WORKS-BUT-STATIC | Confirmed live: SANCO SEA #1 at 100%, real AIS-gap flag text visible ("AIS gap: 167.1h, 311km drift while dark (GFW-flagged intentional disabling)") |
| Detection Overlay (10 epoch-39 tiles) | WORKS-BUT-STATIC | Footer text confirms: "detection overlay uses the real epoch-39 checkpoint (oil IoU 0.1074)" |
| Geometric characterization strip | **NOT WORKING / STALE** | Read `data/processed/dashboard/detection_geometry.json` directly: ground truth `area_px = 247402` (real), prediction `area_px = 0`, `n_components = 0` — leftover from an old, non-trained checkpoint, never regenerated against epoch-39. Directly contradicts the Detection Overlay panel above it, which shows real nonzero predictions for the same kind of tile. If a judge scrolls down, the numbers won't agree. |
| Tile Locations map (`tile_locations.html`) | **NOT WORKING — confirmed by screenshot** | Markers plot at correct coordinates, but the map underneath is the CartoDB "API KEY REQUIRED" watermark tiled across the whole view, not a real basemap. I reproduced this myself with a headless screenshot just now (image saved during this session). |
| Data Provenance panel | WORKS-BUT-STATIC | Computed at build time from real files |

**New finding this pass, not in the earlier draft — basemap reliability is a systemic risk, not just a Tile Locations bug.** `src/dashboard/build_map.py:259` uses folium's built-in `"CartoDB dark_matter"` preset (no API key configured anywhere in the repo — grepped `.env`, confirmed absent). CARTO's free anonymous tile service is known to intermittently serve the "API KEY REQUIRED" watermark. I screenshotted the **main dashboard's Drift Reconstruction panel** (the one the draft called clean) and it also showed patches of "API KEY REQUIRED" watermark text bleeding through at the edges of the visible map area — inconsistent with a fully clean render. A separate standalone screenshot of the same `map.html` (full window, not iframed) showed no watermark but also no visible basemap tiles at all (blank navy background) — i.e. it did not clearly succeed either. **This means the Drift Reconstruction map's basemap may or may not render correctly live on your machine tonight — it is not a settled "works" the way the vessel/particle overlays are.** Recommendation: reload `map.html` a few times right before presenting and watch for the watermark; if it's flaky, the markers/tracks/tooltips (the actual content) still render fine on a blank background, so it's presentable even if the basemap tile layer fails.

**To start it**:
```
cd src\dashboard\output
..\..\..\venv\Scripts\python.exe -m http.server 8001
```
Then open `http://localhost:8001/index.html`. Confirmed working (HTTP 200 on all assets, rendered correctly via headless-browser screenshot) during this audit; I stopped the server afterward so port 8001 is free for you.

### `web/`

Static, self-contained page (`web/index.html`, identical backup `web/fallback.html`). Shows the same 10 real epoch-39 tiles, no dependency on the rest of the dashboard, no external network calls.

**Status: DEMOABLE.** Started it myself (`python -m http.server 8000 --directory web`), curled `index.html`, `fallback.html`, and `public/demo/summary.json` (all 200), and rendered `index.html` with headless Edge — every tile, metric, and label (IoU/precision/recall/predicted-oil-% per tile) matched the numbers above exactly. This is the most reliable thing in the repo to show live, precisely because it has zero moving parts.

**To start it**:
```
cd web
..\venv\Scripts\python.exe -m http.server 8000
```
Then open `http://localhost:8000/`. Stopped the server afterward so port 8000 is free.

---

## 3. ATTRIBUTION PIPELINE

**What exists and runs**: `src/drift/advect.py` (particle advection: ERA5 primary + NCEP/NCAR fallback wind, HYCOM currents, Ekman/Coriolis deflection, shared backward/forward physics), `src/attribution/gfw_client.py` + `score_vessels.py` (real GFW `v3/events`/`4wings/report` queries, proximity + time-gap + AIS-gap/behavioral-anomaly scoring into a composite match score), `src/dashboard/build_map.py` (Folium map + drift-animation + wind-vector overlay generation), `src/detection/geometry.py` (cv2-based area/length/width/orientation/elongation, has a dedicated passing test).

**What ow-0001/ow-0002 are built from**: I did not re-run the GFW/ERA5/HYCOM API calls myself tonight (out of scope for a read-only audit, and would risk quota/rate-limit or take real wall-clock time). Based on direct reading of `LOG.md`'s session-by-session entries (which document specific real results as they happened, including real bugs hit and fixed — e.g. a cross-year OPeNDAP concat bug, a non-CF `tau`-units crash, a GFW `422` fixed by discovering the correct request shape) — this is a real pipeline run on real inputs: Sentinel-1/PANGAEA-sourced detection coordinates, real ERA5 wind reanalysis, real HYCOM currents, and a real GFW AIS vessel pool (2,661 candidates → ow-0001, 10,180 → 1,006 candidates → ow-0002). Specific numbers I can independently confirm from the live dashboard render: ow-0001 top suspect SANCO SEA at 100%, 355 vessels evaluated, 12.9km detection→origin distance — these match what's on screen right now, not just what a doc claims.

**Explicitly asked: anything synthetic/placeholder shown as real.** What I found:
1. The geometry strip's `area_px = 0` prediction (§2) — stale data from an old, untrained checkpoint, inconsistent with the real numbers next to it. This is the clearest "looks wrong if you look closely" item in the repo.
2. The lookalike/no_oil demo tiles were **deliberately selected as the model's worst false-positive cases** — I confirmed this directly in `scripts/generate_demo_assets.py`: tiles are sorted by `-pred_positive_fraction` (descending) within each non-oil class, i.e. the highest-false-positive examples are picked on purpose, not randomly. This is real model output, not fabricated — but if a judge asks "is this representative," the honest answer is "no, it's the worst case, shown on purpose for honesty."
3. The Drift Reconstruction map's basemap reliability issue (§2, new finding this pass) — not fabricated data, but a live-rendering risk that could make a real panel look broken on stage through no fault of the underlying data.

I found no fabricated numbers being presented as real anywhere in this repo's current live output.

---

## 4. OTHER COMPONENTS

- `src/detection/geometry.py` (83 lines) — real, has `tests/test_geometry.py`, passing. Finished.
- `src/common/anonymize.py` (34 lines) — real, produces `src/dashboard/output_anon/` (identity-swapped copy for public sharing). Finished.
- `src/drift/` (`advect.py` 203 lines, `currents.py` 61, `wind_era5.py` 98, plus `wind_ncep.py`) — real physics, has `tests/test_advect.py` and `tests/test_currents.py`, both passing. Explicitly documented in the project's own notes as a first-pass, non-ground-truth-calibrated methodology — that's a stated scope limitation, not unfinished code.
- `src/attribution/score_vessels.py` (352 lines) + `gfw_client.py` (159 lines) — real, has `tests/test_score_vessels.py`, passing.
- Test suite: **36/36 passing**, confirmed by running `pytest` myself just now (18.08s, zero failures, zero skips).

---

## 5. VERIFIED FINDINGS (Phase 0), for accurate quoting

Cross-checked directly against `docs/metric_audit.md` and `LOG.md`'s 2026-08-31 Phase 0 entry (Gates A/B/C):

1. The model can memorize 12 fixed oil-containing tiles (mean IoU 0.9841, loss 2.37→0.024) — confirms the core pipeline (mask alignment, loss, normalization) is structurally sound, not broken.
2. Two real, statistically distinct SAR bands exist in every Zenodo image; the production model consumes only Band 1.
3. `"lookalike"` is a real, explicit manifest label (582 train / 103 val) — not a derived or proxy category.
4. The training loop's raw `val_dice` metric has a proven mathematical flaw: for any tile with zero ground-truth oil (≈82% of all tiles), the dice formula collapses toward 0 for almost any nonzero prediction and can never approach 1.0 — verified numerically in `docs/metric_audit.md` with a worked table. This is why `val_dice` looked flat across every historical experiment.
5. The real oil-tiles-only IoU was independently reproduced from scratch (0.1074 at threshold 0.35) using a new, unit-tested metric implementation (`src/detection/metrics.py`) — the number is trustworthy, not just re-run of old code.
6. At that same threshold, precision is only 0.139 — the checkpoint is a broadly over-triggering detector, a characterization no prior script in this project had surfaced (precision/recall were never reported before this audit).
7. All three historical tuning trials (`oversample`, `tversky`, `posweight_sched`) score below the epoch-39 baseline at their own best threshold, each with a distinct, now-quantified failure mode: the first two collapse toward predicting almost everything as oil (66–99% of pixels flagged even on clean scenes); the third is generally under-confident and was only trained 1 epoch before being killed.
8. 82.38% of the 34,940 real training tiles have zero oil pixels — independently re-verified, matches the figure already recorded in `LOG.md`.
9. Band 1's normalized values are compressed under the shared [-40, +10] dB range (median 0.13, 31% of pixels below 0.10) versus Band 2 (median 0.40, 0.24% below 0.10) — a likely, not-yet-proven contributor to the weak precision above.
10. Per-band [p1, p99] normalization narrows this gap but does not close it (Band 1 median 0.331 vs. Band 2 0.505 after excluding nodata) — suggests a real information-content difference between the two bands, not just a preprocessing artifact.
11. Both bands' original p99 landed at exactly 0.0 dB because 1.81% of pixels are exactly 0.0 dB at identical spatial locations in both bands — a real nodata mask (edge-of-swath), not sensor clipping.
12. Those nodata pixels are included, unmasked, in both the training loss and every IoU/precision/recall calculation today — confirmed by direct code search, no masking exists anywhere. They make up 2.97% of the full validation set.
13. The train/val split is at the whole-image level (0 exact-duplicate files across the split, content-hash verified); scene/acquisition-level leakage is unproven either way — no such metadata exists in the source files to check it.
14. Tonight's per-band normalization config fix (exp01a/b/c) excludes those same nodata pixels before computing percentiles, moving the fixed range from `[-43.673, 0.0]` to `[-43.710, -12.894]` — I confirmed this via direct diff, it's the only change in those three files.

---

## 6. KNOWN GAPS — what a judge could reasonably ask that isn't answered

- "Why is real precision only ~14%?" — Documented (#6), root cause only partially diagnosed (band info-content and normalization are suspects, not proven).
- "Have you tried both SAR channels?" — Configs exist, nothing has been trained. This is a planned next step, not a result.
- "Is your current best checkpoint actually your best?" — No. `best_unet_resnet18.pt` (epoch 44) has never been scored with the real IoU/precision/recall metric; only epoch 39 has a verified number.
- "Is the attribution methodology validated against ground truth?" — No, and the project's own documentation says so explicitly: first-pass, uncalibrated, a stated permanent limitation.
- "Can detection feed directly into drift/attribution, live, end to end?" — Not demonstrated. The one place they sit adjacent on the same page (the geometry strip) currently shows a stale, zero-pixel prediction from an untrained checkpoint.
- "Why does the model do worse on lookalikes than on oil?" — Open question; precision problem is documented, cause is not fully isolated.
- "Does the nodata-pixel contamination affect your reported numbers?" — Diagnosed (#12), not corrected. Honest answer: "somewhat, direction and magnitude unquantified."
- "Can I see the Tile Locations map?" — It's visibly broken (basemap watermark) — confirmed by screenshot tonight. Don't show it live.
- New tonight: the main Drift Reconstruction map's basemap tile layer showed the same "API KEY REQUIRED" artifact in one of two screenshots taken minutes apart — reload it a few times before presenting and be ready for the basemap (not the data) to fail.

---

## Uncommitted changes at time of this audit

```
 M configs/exp01a_band1_perband.yaml
 M configs/exp01b_band2_perband.yaml
 M configs/exp01c_both_perband.yaml
 M src/dashboard/output/index.html
?? scripts/add_tile_locations_panel.py
?? scripts/build_tile_locations_map.py
?? scripts/generate_demo_assets.py
?? scripts/integrate_demo_into_dashboard.py
?? src/dashboard/output/index.html.prelocations_backup
?? src/dashboard/output/tile_locations.html
?? web/
```

Nothing was committed, built, or trained during this audit. Both preview servers (ports 8000, 8001) I started for verification were stopped afterward — ports are free for you tonight.
