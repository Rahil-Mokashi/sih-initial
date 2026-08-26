"""
Extract the Zenodo Part III archive (held-out balanced test set: 150 Oil /
150 No-oil / 150 Look-alike, real images + ground-truth masks -- see
scripts/download_zenodo_part2_part3.py). Mirrors
scripts/extract_zenodo_part2.py's verify-before-skip pattern.

Internal archive layout differs from Part I/II (which used flat
Oil/No_oil/Lookalike + Mask_oil/Mask_no_oil/Mask_lookalike dirs): Part III
nests everything under Images/ and Mask/, and uses "No oil" (a space, not
an underscore) for that class.
"""

from pathlib import Path

import py7zr

PART3_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "zenodo_sar_oil_spill_part3"
ARCHIVE = PART3_DIR / "02_Test_images_and_ground_truth.7z"
IMAGES_EXTRACTED = PART3_DIR / "images_extracted"
MASKS_EXTRACTED = PART3_DIR / "masks_extracted"

# (archive-internal top-level dir, real class label, extraction root)
SUBDIRS = [
    ("Images/Oil", "oil", IMAGES_EXTRACTED),
    ("Images/No oil", "no_oil", IMAGES_EXTRACTED),
    ("Images/Lookalike", "lookalike", IMAGES_EXTRACTED),
    ("Mask/Oil", "oil", MASKS_EXTRACTED),
    ("Mask/No oil", "no_oil", MASKS_EXTRACTED),
    ("Mask/Lookalike", "lookalike", MASKS_EXTRACTED),
]


def main() -> None:
    if not ARCHIVE.exists():
        print(f"ERROR: {ARCHIVE} not found. Run scripts/download_zenodo_part3_parallel.py first.")
        return

    with py7zr.SevenZipFile(ARCHIVE, mode="r") as z:
        names = z.getnames()

    expected_counts = {}
    for archive_dir, _, _ in SUBDIRS:
        expected_counts[archive_dir] = sum(1 for n in names if n.startswith(archive_dir + "/") and n.lower().endswith(".tif"))

    already_done = all(
        (extract_root / archive_dir).exists()
        and len(list((extract_root / archive_dir).glob("*.tif"))) == expected_counts[archive_dir]
        for archive_dir, _, extract_root in SUBDIRS
    )
    if already_done:
        print("SKIP: Part III already extracted and verified.")
        for archive_dir, label, extract_root in SUBDIRS:
            print(f"  {archive_dir}: {expected_counts[archive_dir]} tifs OK")
        return

    print(f"Extracting {ARCHIVE.name} ({sum(expected_counts.values())} tif files expected)...")
    IMAGES_EXTRACTED.mkdir(parents=True, exist_ok=True)
    MASKS_EXTRACTED.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(ARCHIVE, mode="r") as z:
        z.extractall(path=PART3_DIR / "_raw_extract")

    raw = PART3_DIR / "_raw_extract"
    for archive_dir, label, extract_root in SUBDIRS:
        src = raw / archive_dir
        dest = extract_root / archive_dir
        dest.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in src.glob("*.tif"):
            f.rename(dest / f.name)
            n += 1
        expected = expected_counts[archive_dir]
        status = "OK" if n == expected else f"MISMATCH (expected {expected})"
        print(f"  {archive_dir} -> {dest}: {n} tifs {status}")

    import shutil
    shutil.rmtree(raw, ignore_errors=True)


if __name__ == "__main__":
    main()
