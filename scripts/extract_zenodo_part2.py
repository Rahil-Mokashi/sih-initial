"""
Extract Zenodo Part II archives (No-Oil + Look-alike training negatives,
see scripts/download_zenodo_part2_part3.py) into the same
images_extracted/masks_extracted layout scripts/build_training_pool.py
expects Part I to already be in.

Skips any archive whose extraction target directory already exists (so
it's safe to re-run once the still-downloading Lookalike images archive
finishes). Verifies each extraction's file count against the archive's
own entry count before declaring it done.
"""

from pathlib import Path

import py7zr

PART2_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "zenodo_sar_oil_spill_part2"
IMAGES_EXTRACTED = PART2_DIR / "images_extracted"
MASKS_EXTRACTED = PART2_DIR / "masks_extracted"

ARCHIVES = [
    ("01_Train_Val_No_Oil_Images.7z", IMAGES_EXTRACTED),
    ("01_Train_Val_No_Oil_mask.7z", MASKS_EXTRACTED),
    ("01_Train_Val_Lookalike_images.7z", IMAGES_EXTRACTED),
    ("01_Train_Val_Lookalike_mask.7z", MASKS_EXTRACTED),
]


def extract_one(archive_name: str, extract_root: Path) -> None:
    archive_path = PART2_DIR / archive_name
    if not archive_path.exists():
        print(f"SKIP {archive_name}: not downloaded yet")
        return

    with py7zr.SevenZipFile(archive_path, mode="r") as z:
        names = z.getnames()
        n_expected = sum(1 for n in names if n.lower().endswith(".tif"))
        # top-level folder name inside the archive, e.g. "No_oil" or "Mask_lookalike"
        top_dirs = {n.split("/")[0] for n in names}

    already_done = all((extract_root / d).exists() and
                        len(list((extract_root / d).glob("*.tif"))) == n_expected
                        for d in top_dirs)
    if already_done:
        print(f"SKIP {archive_name}: already extracted and verified ({n_expected} tifs)")
        return

    print(f"Extracting {archive_name} ({n_expected} tif files expected)...")
    with py7zr.SevenZipFile(archive_path, mode="r") as z:
        z.extractall(path=extract_root)

    n_actual = sum(len(list((extract_root / d).glob("*.tif"))) for d in top_dirs)
    if n_actual != n_expected:
        print(f"  ERROR: expected {n_expected} tif files, found {n_actual} after extraction")
    else:
        print(f"  done: {n_actual} tif files verified under {[str(extract_root / d) for d in top_dirs]}")


def main() -> None:
    for archive_name, extract_root in ARCHIVES:
        extract_one(archive_name, extract_root)


if __name__ == "__main__":
    main()
