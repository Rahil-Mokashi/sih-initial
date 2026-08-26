"""
Build the real training pool: Part I (oil-positive) + Part II (no-oil +
look-alike negatives), verified and matched by filename, split 85/15
train/val stratified by class (and by oil-fraction quartile within the
oil class). Part III (test set) is intentionally NOT touched here -- see
DECISIONS.md "Train/val/test methodology".

Usage:
    venv\\Scripts\\python.exe scripts\\build_training_pool.py

Writes data/processed/train_manifest.csv and val_manifest.csv, each row:
    image_path,mask_path,label

Also verifies: every image has a matching mask (by filename) and vice
versa, per source, reporting any mismatches rather than silently dropping
them.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import rasterio

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"

VAL_FRACTION = 0.15
SPLIT_SEED = 0


def find_dirs_with_tifs(root: Path) -> list[Path]:
    """Any directory under root that directly contains .tif files."""
    if not root.exists():
        return []
    dirs = set()
    for p in root.rglob("*.tif"):
        dirs.add(p.parent)
    return sorted(dirs)


def classify_dir(d: Path) -> tuple[str, str] | None:
    """
    Returns (kind, label) where kind is "image" or "mask", label is
    "oil"/"no_oil"/"lookalike", based on the directory name. Returns None
    if the directory name doesn't clearly indicate what it is -- caller
    should investigate rather than guess.
    """
    name = d.name.lower()
    is_mask = "mask" in name
    if "lookalike" in name or "look_alike" in name or "look-alike" in name:
        label = "lookalike"
    elif "no_oil" in name or "nooil" in name or "no-oil" in name:
        label = "no_oil"
    elif "oil" in name:  # must check after lookalike/no_oil since both also contain "oil"
        label = "oil"
    else:
        return None
    return ("mask" if is_mask else "image", label)


def match_by_filename(image_dir: Path, mask_dir: Path) -> tuple[list[tuple[Path, Path]], list[str], list[str]]:
    images = {p.name: p for p in image_dir.glob("*.tif")}
    masks = {p.name: p for p in mask_dir.glob("*.tif")}
    common = sorted(set(images) & set(masks))
    only_images = sorted(set(images) - set(masks))
    only_masks = sorted(set(masks) - set(images))
    pairs = [(images[name], masks[name]) for name in common]
    return pairs, only_images, only_masks


def discover_label_dirs(root: Path) -> dict[str, dict[str, Path]]:
    """{label: {"image": dir, "mask": dir}} for whatever's found under root."""
    result: dict[str, dict[str, Path]] = {}
    for d in find_dirs_with_tifs(root):
        classified = classify_dir(d)
        if classified is None:
            print(f"  WARNING: could not classify directory {d} by name -- skipping. "
                  f"Check it manually if it should have been included.")
            continue
        kind, label = classified
        result.setdefault(label, {})[kind] = d
    return result


def oil_fraction(mask_path: Path) -> float:
    with rasterio.open(mask_path) as src:
        arr = src.read(1)
    return float((arr > 0).mean())


def stratum_key(label: str, frac: float | None) -> str:
    if label != "oil" or frac is None:
        return label
    # Quartile buckets within the oil class, using the real fraction distribution
    # from Step 1 (mean 2.98%, median 1.76%, max 57.4%) to pick sensible edges.
    if frac < 0.01:
        return "oil_q1"
    elif frac < 0.02:
        return "oil_q2"
    elif frac < 0.05:
        return "oil_q3"
    else:
        return "oil_q4"


def main() -> None:
    print("Discovering Part I (oil-positive)...")
    part1_root = DATA_RAW / "zenodo_sar_oil_spill"
    part1_dirs = discover_label_dirs(part1_root)

    print("Discovering Part II (no-oil + look-alike)...")
    part2_root = DATA_RAW / "zenodo_sar_oil_spill_part2"
    part2_dirs = discover_label_dirs(part2_root)

    all_dirs = {**part1_dirs, **part2_dirs}
    for label in ("oil", "no_oil", "lookalike"):
        if label not in all_dirs or "image" not in all_dirs[label] or "mask" not in all_dirs[label]:
            print(f"ERROR: missing image or mask directory for label '{label}'. Found so far: {all_dirs}")
            sys.exit(1)

    all_pairs: list[tuple[Path, Path, str]] = []
    for label, dirs in all_dirs.items():
        pairs, only_images, only_masks = match_by_filename(dirs["image"], dirs["mask"])
        print(f"\n[{label}] {len(pairs)} matched image/mask pairs "
              f"({dirs['image']} <-> {dirs['mask']})")
        if only_images:
            print(f"  WARNING: {len(only_images)} images with no matching mask: {only_images[:10]}"
                  f"{' ...' if len(only_images) > 10 else ''}")
        if only_masks:
            print(f"  WARNING: {len(only_masks)} masks with no matching image: {only_masks[:10]}"
                  f"{' ...' if len(only_masks) > 10 else ''}")
        for img_path, mask_path in pairs:
            all_pairs.append((img_path, mask_path, label))

    print(f"\nTotal training pool: {len(all_pairs)} image/mask pairs")

    print("Computing oil-fraction strata for the oil class (for stratified split)...")
    strata = []
    for img_path, mask_path, label in all_pairs:
        frac = oil_fraction(mask_path) if label == "oil" else None
        strata.append(stratum_key(label, frac))

    rng = np.random.default_rng(SPLIT_SEED)
    train_rows, val_rows = [], []
    for stratum in sorted(set(strata)):
        idxs = [i for i, s in enumerate(strata) if s == stratum]
        rng.shuffle(idxs)
        n_val = max(1, round(len(idxs) * VAL_FRACTION)) if len(idxs) > 1 else 0
        val_idxs = set(idxs[:n_val])
        for i in idxs:
            row = all_pairs[i]
            (val_rows if i in val_idxs else train_rows).append(row)
        print(f"  stratum '{stratum}': {len(idxs)} total -> {len(idxs) - n_val} train / {n_val} val")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, rows in [("train_manifest.csv", train_rows), ("val_manifest.csv", val_rows)]:
        out_path = DATA_PROCESSED / name
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "mask_path", "label"])
            for img_path, mask_path, label in rows:
                writer.writerow([str(img_path), str(mask_path), label])
        print(f"wrote {out_path} ({len(rows)} rows)")

    print(f"\nFinal: {len(train_rows)} train / {len(val_rows)} val "
          f"({len(val_rows) / (len(train_rows) + len(val_rows)) * 100:.1f}% val)")


if __name__ == "__main__":
    main()
