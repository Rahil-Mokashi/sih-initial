"""
Phase 0 Gate C.3: visual dataset audit. Generates montage PNGs so a human
can directly confirm the model receives sensible, correctly-aligned inputs
-- this script only reads data and writes images, it does not touch any
training code or model.

Per example, shows: raw Band 1, raw Band 2, normalized Band 1 (what the
model actually receives), normalized Band 2, ground-truth mask, and a
crop-location thumbnail (the full 2048x2048 source image with the sampled
512x512 tile's location outlined) -- so misalignment between image and
mask, or a crop drawn from the wrong location, would be visible directly.

Preprocessing shown for "normalized" columns matches production val-time
preprocessing (src/detection/preprocess.py's lee_filter + a normalize_fn,
augment=False) -- NOT the augmented train-time version, since the point is
to see what the network structurally receives, not one random augmentation
draw. Defaults to normalize_db_fixed (the shared -40/10 dB range); pass
--per-band-ranges to use normalize_db_per_channel instead (see Amendment 2
/ docs/metric_audit.md's Gate C finding) -- e.g. to verify whether per-band
ranges actually close the Band 1 vs. Band 2 compression gap that finding
measured.

Usage:
    venv\\Scripts\\python.exe scripts\\visualize_dataset.py
    venv\\Scripts\\python.exe scripts\\visualize_dataset.py --per-band-ranges "-43.673:0.0,-33.821:0.0" \\
        --out-dir reports/dataset_visual_audit_per_band
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from functools import partial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import rasterio
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from detection.preprocess import lee_filter, normalize_db_fixed, normalize_db_per_channel  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_MANIFEST = REPO_ROOT / "data" / "processed" / "train_manifest.csv"
OUT_DIR = REPO_ROOT / "reports" / "dataset_visual_audit"
TILE_SIZE = 512
N_PER_CLASS = 10
SEED = 0


def load_manifest(path: Path) -> list[dict]:
    return list(csv.DictReader(open(path, newline="")))


def pick_tile(image_path: Path, mask_path: Path, rng: random.Random, require_oil: bool) -> tuple[int, int] | None:
    """Returns (y, x) of a valid 512x512 tile origin, or None if require_oil
    and no oil-containing tile exists in this image."""
    with rasterio.open(image_path) as src:
        h, w = src.height, src.width
    ys = list(range(0, h - TILE_SIZE + 1, TILE_SIZE))
    xs = list(range(0, w - TILE_SIZE + 1, TILE_SIZE))
    coords = [(y, x) for y in ys for x in xs]
    rng.shuffle(coords)

    if not require_oil:
        return coords[0] if coords else None

    with rasterio.open(mask_path) as src:
        for y, x in coords:
            tile = src.read(1, window=Window(x, y, TILE_SIZE, TILE_SIZE))
            if (tile > 0).any():
                return (y, x)
    return None


def render_example(ax_row, image_path: Path, mask_path: Path, y: int, x: int, label: str,
                    normalize_b1, normalize_b2) -> None:
    with rasterio.open(image_path) as src:
        full_h, full_w = src.height, src.width
        b1_full_thumb = src.read(1, out_shape=(256, 256))
        b1 = src.read(1, window=Window(x, y, TILE_SIZE, TILE_SIZE)).astype(np.float32)
        b2 = src.read(2, window=Window(x, y, TILE_SIZE, TILE_SIZE)).astype(np.float32)
    with rasterio.open(mask_path) as src:
        gt = src.read(1, window=Window(x, y, TILE_SIZE, TILE_SIZE))

    b1_norm = normalize_b1(lee_filter(b1))
    b2_norm = normalize_b2(lee_filter(b2))

    ax_row[0].imshow(b1, cmap="gray")
    ax_row[0].set_title("Band 1 (raw dB)", fontsize=8)
    ax_row[1].imshow(b2, cmap="gray")
    ax_row[1].set_title("Band 2 (raw dB)", fontsize=8)
    ax_row[2].imshow(b1_norm, cmap="gray", vmin=0, vmax=1)
    ax_row[2].set_title("Band 1 (normalized)", fontsize=8)
    ax_row[3].imshow(b2_norm, cmap="gray", vmin=0, vmax=1)
    ax_row[3].set_title("Band 2 (normalized)", fontsize=8)
    ax_row[4].imshow(gt, cmap="Reds", vmin=0, vmax=1)
    ax_row[4].set_title(f"GT mask ({label}, oil_frac={100 * (gt > 0).mean():.2f}%)", fontsize=8)

    ax_row[5].imshow(b1_full_thumb, cmap="gray")
    scale = 256 / full_w
    rect = patches.Rectangle((x * scale, y * scale), TILE_SIZE * scale, TILE_SIZE * scale,
                              linewidth=1.5, edgecolor="lime", facecolor="none")
    ax_row[5].add_patch(rect)
    ax_row[5].set_title(f"crop @ (y={y}, x={x})", fontsize=8)

    for ax in ax_row:
        ax.axis("off")


def build_montage(rows: list[tuple[Path, Path, int, int, str]], out_path: Path, title: str,
                   normalize_b1, normalize_b2) -> None:
    n = len(rows)
    fig, axes = plt.subplots(n, 6, figsize=(18, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    for i, (image_path, mask_path, y, x, label) in enumerate(rows):
        render_example(axes[i], image_path, mask_path, y, x, label, normalize_b1, normalize_b2)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--per-band-ranges", type=str, default=None,
                   help="'lo1:hi1,lo2:hi2' -- use normalize_db_per_channel with these two ranges "
                        "(Band 1, Band 2) instead of the default shared normalize_db_fixed(-40, 10)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.per_band_ranges:
        r1_str, r2_str = args.per_band_ranges.split(",")
        lo1, hi1 = (float(v) for v in r1_str.split(":"))
        lo2, hi2 = (float(v) for v in r2_str.split(":"))
        normalize_b1 = partial(normalize_db_per_channel, ranges=[(lo1, hi1)])
        normalize_b2 = partial(normalize_db_per_channel, ranges=[(lo2, hi2)])
        norm_desc = f"per-band: Band1={lo1}:{hi1}, Band2={lo2}:{hi2}"
    else:
        normalize_b1 = normalize_db_fixed
        normalize_b2 = normalize_db_fixed
        norm_desc = "shared normalize_db_fixed(-40, 10)"
    print(f"normalization: {norm_desc}")
    print(f"output directory: {args.out_dir}")

    rng = random.Random(SEED)
    manifest = load_manifest(TRAIN_MANIFEST)
    by_label: dict[str, list[dict]] = {"oil": [], "no_oil": [], "lookalike": []}
    for row in manifest:
        by_label.setdefault(row["label"], []).append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for label, rows_for_label in by_label.items():
        rng.shuffle(rows_for_label)
        selected = []
        for row in rows_for_label:
            image_path, mask_path = Path(row["image_path"]), Path(row["mask_path"])
            require_oil = (label == "oil")
            coord = pick_tile(image_path, mask_path, rng, require_oil=require_oil)
            if coord is None:
                continue
            y, x = coord
            selected.append((image_path, mask_path, y, x, label))
            if len(selected) >= N_PER_CLASS:
                break
        print(f"[{label}] selected {len(selected)}/{N_PER_CLASS} examples")
        if selected:
            build_montage(selected, args.out_dir / f"{label}_examples.png",
                           f"{label} -- {len(selected)} real examples (train manifest) -- {norm_desc}",
                           normalize_b1, normalize_b2)


if __name__ == "__main__":
    main()
