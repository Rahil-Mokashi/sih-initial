# SIH26143 — Oil Spill Detection & Vessel Attribution

**Problem statement (NTRO, Smart India Hackathon):** Detect and characterise oil spills in
Sentinel-1 SAR satellite imagery, trace the slick backward through wind/ocean current data
to estimate where and when it started AND forward to predict where it's headed, then
cross-reference AIS vessel tracking data (proximity, trajectory, behavioural anomalies) to
produce a **ranked, evidence-backed list of suspect vessels — not a single verdict.**

## Pipeline

```
SAR image → detection model → oil mask → geometric characterization
                                  ↓         (area/length/width/orientation/shape)
              backward hindcast ←――――――――→ forward forecast
        (ERA5/NCEP wind + HYCOM currents + Ekman deflection, same physics both ways)
                                  ↓
                       estimated origin region + time
                                  ↓
     AIS vessel presence + trajectory + AIS-gap behavior (Global Fishing Watch,
                    checked against the FULL raw candidate pool)
                                  ↓
    ranked, evidence-backed suspect list (proximity/timing/trajectory/behavior
                         all shown as separate fields)
                                  ↓
      interactive dashboard (real + anonymized builds, print-to-PDF export)
```

## What's real and working

- **Data**: 2,570 real training images (Zenodo Sentinel-1 SAR dataset: 1200 oil-positive,
  685 no-oil, 685 look-alike negatives) + 450-image held-out test set. All real, verified,
  no synthetic data anywhere.
- **Drift model, both directions**: real particle advection using ERA5 wind (primary) +
  NCEP/NCAR (fallback) + HYCOM ocean currents, with an empirical Ekman/Coriolis deflection
  term — backward (hindcast, validated on two independent real cases, ERA5 vs NCEP/NCAR
  origin estimates agree within ~7-12km) and **forward (forecast)**, sharing one physics
  core (`_advect()`, a `direction` sign flips which way it integrates) rather than two
  separate implementations. Real detection→forecast distances: 12-29km depending on
  case/wind source.
- **Geometric characterization**: real area/length/width/orientation/elongation from a
  detection mask (`cv2` contour + `minAreaRect`) — pixel units only, since the Zenodo
  dataset's own record doesn't document a Sentinel-1 product type or ground resolution
  (checked directly against the record page, not assumed).
- **Attribution, four separately-visible dimensions**: real Global Fishing Watch AIS data,
  scored by proximity + timing vs. the drift-estimated origin (`score`), trajectory (a
  candidate's OTHER real presence rows across the window, not just the closest one), and
  behavioral anomaly (a real AIS-gap check via GFW's v3/events API, including GFW's own
  "intentional disabling" flag) — folded into a separate `composite_score`, never silently
  blended with the original `score`. Checked against the **full raw candidate pool**
  (355-1006+ vessels depending on case), not a pre-filtered subset — GFW's `vessels[]`
  filter batches up to 20 IDs per request (found by direct test), so this costs
  `ceil(n/20)` requests, not one per vessel. Course/heading and instantaneous per-position
  speed are confirmed NOT available at this API tier (checked directly against real
  responses) — a real external constraint, not a shortcut. Explicitly documented as a
  first-pass methodology, not calibrated against labeled ground truth (a permanent,
  domain-inherent limitation). Displayed as a **"match score,"** not "confidence" — the
  latter implies a calibrated probability the methodology can't back up. A separate
  anonymized build (`--anonymize`) swaps real ship names/MMSI/IMO for fictional stand-ins
  for anything that leaves the judging room; the live dashboard keeps real identities as
  proof it runs on real GFW data.
- **Dashboard**: interactive, real data throughout — a redesigned map (real convex-hull
  "drift cone" instead of 50 raw lines by default, real ship icons at real GFW positions,
  a permanent "Estimated Origin" callout, a real detected-slick bbox, and a real
  reverse-geocoded location label — "Levantine Sea, Eastern Mediterranean" / "off
  Damietta, Egypt", verified via OpenStreetMap, not guessed), drift-animation playback
  (both backward and forward tracks, visually distinct), real wind-vector overlay,
  geometric characterization panel, a data provenance panel (computed live from disk),
  sortable/expandable vessel ranking cards (with AIS-gap evidence as an always-visible
  flag, not buried), a multi-case comparison table, and a print-to-PDF report export
  (browser-native, no new dependency).
- **Test suite**: 28 passing pytest tests covering the pure-logic pieces (geo math, GFW
  scoring, bbox rasterization, Ekman rotation, mask geometry, a real regression test for a
  rounding bug hit during development).

## What's still weak (the honest gap)

- **Detection model** (SAR image → oil mask): a U-Net + ResNet18 encoder (ImageNet-
  pretrained). Real story, including a real mid-course correction: raw training-loop
  `val_dice` plateaued near 0.022-0.0235, which looked like a broken model (below a
  dice-equivalent trivial baseline), so three fixes were tried in isolation — reduced
  `pos_weight`, oil-tile oversampling (`scripts/analyze_tile_oil_distribution.py` found
  82.4% of real training tiles have zero oil pixels), and a full Tversky loss swap. All
  three left `val_dice` similarly flat, with no way to tell from that metric alone whether
  any of them helped.

  **Built a real threshold-swept IoU comparison (`scripts/compare_checkpoints_on_val.py`,
  on the validation set) to check, rather than assume — and it reversed the whole
  diagnosis.** The ORIGINAL, untouched checkpoint (pos_weight=32.6, plain DiceBCE) scored
  the best real oil-tiles-only IoU of all four configurations tested (**0.1057** at
  threshold 0.35, with a real graded threshold response) — all three "fixes" had actually
  made real performance *worse* (oversampling: 0.0828, flat/saturated-looking; Tversky:
  exactly 0.0800 at every threshold tested, a real sign of a collapsed, non-discriminating
  output), even though their raw training metric looked the same as each other. Root
  cause of the original false alarm: the dice-equivalent trivial baseline (~0.058) isn't
  the right comparison — the real IoU-equivalent trivial baseline is ~0.03, and the
  original checkpoint's real IoU (0.1057) was already well clear of it the whole time.

  Reversed course: killed the misleading trials and resumed training from the real best
  checkpoint with `--use-lr-scheduler` added (the one real lever never tested against the
  config that actually works). As of this summary: epoch 43/60, real threshold-swept IoU
  re-check planned for epoch 45 before trusting raw `val_dice` again. Next real untried
  lever if this plateaus too: a bigger encoder (ResNet34) — real unused VRAM headroom
  (~1.3GB+) that no trial so far has touched, addressing model capacity rather than
  optimization dynamics.

## Key numbers as of this summary

- Detection: best real held-out oil-tiles-only IoU **0.1057** (val set, threshold 0.35)
  from the original/best checkpoint — also 0.1645 on the actual Part III test set at an
  earlier evaluation (threshold 0.18). Three loss/sampling variants tried and confirmed
  *worse* by real threshold-swept evaluation, not just guessed at. Currently retraining
  the real best config with LR decay, epoch 43/60.
- Two validated real cases, full-pool attribution (5 of 355 / 14 of 1006 candidates had a
  real AIS gap): ow-0001 (12.9km detection→origin distance, top suspect **SANCO SEA** at
  100% match score — flipped from THOR FREYJA once real AIS-gap behavioral evidence was
  added, driven by a real GFW-flagged intentional AIS disabling event) and ow-0002 (31.4km,
  top suspect **ALAWAD1** at 98.4% match score, unchanged — no real AIS gap among its top
  candidates)
- GFW API usage: still well under 200 total requests this project (69 alone for the
  full-pool behavioral rescoring across both cases), far under the 50,000/day limit

## Architecture / tech stack

Python, PyTorch (U-Net/ResNet18 via segmentation_models_pytorch), rasterio (GeoTIFF I/O),
OpenCV + scikit-image (despeckling, mask geometry), xarray (NetCDF wind/current data),
Global Fishing Watch REST API (4wings/report + v3/events), Folium/Leaflet for mapping, a
hand-built HTML/CSS/JS dashboard (no frontend framework). Real GPU training on an RTX 4050
Laptop GPU (6GB VRAM).

## Where the full detail lives

The project repo maintains three docs kept current every session:
- `README.md` — how to run each stage
- `LOG.md` — session-by-session real history: what was done, what broke, what was fixed,
  real numbers throughout (not a sanitized summary — includes dead ends and bugs)
- `DECISIONS.md` — why things were built the way they were, tradeoffs considered

This summary is a snapshot; those three files are the living source of truth and may be
further ahead by the time you're reading this.
