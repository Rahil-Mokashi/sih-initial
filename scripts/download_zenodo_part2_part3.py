"""
Download Zenodo Part II and Part III of the Sentinel-1 SAR Oil Spill
dataset family (see DECISIONS.md "Training data: Part I is oil-only" for
why these are needed and how they were found).

Part II (DOI 10.5281/zenodo.8253899): same-domain (calibrated Sigma0-dB)
hard negatives for training -- 685 oil-free scenes + 685 look-alike scenes,
each with an all-zero mask. Combined with Part I's 1200 oil-positive
images, this is the real training pool.
  - 01_Train_Val_No_Oil_Images.7z    (~22.9GB, 685 images)
  - 01_Train_Val_No_Oil_mask.7z      (~0.4MB, 685 masks, all-zero)
  - 01_Train_Val_Lookalike_images.7z (~23.0GB, 685 images)
  - 01_Train_Val_Lookalike_mask.7z   (~0.4MB, 685 masks, all-zero)

Part III (DOI 10.5281/zenodo.13761290): held-out TEST set, balanced 150
Oil / 150 No-oil / 150 Look-alike, same calibrated format. Reserved --
never touched during training or hyperparameter tuning; this is what
produces the real accuracy number reported at the end.
  - 02_Test_images_and_ground_truth.7z (~9.9GB)

Same resumable/progress-reporting approach as Part I
(scripts/download_zenodo_sample.py): safe to interrupt and re-run, resumes
via HTTP Range from each file's .part.
"""

import sys
from pathlib import Path

from _zenodo_download_utils import download_full, fetch_record_metadata

PART_II_RECORD_ID = "8253899"
PART_III_RECORD_ID = "13761290"

OUT_DIR_PART_II = Path(__file__).resolve().parent.parent / "data" / "raw" / "zenodo_sar_oil_spill_part2"
OUT_DIR_PART_III = Path(__file__).resolve().parent.parent / "data" / "raw" / "zenodo_sar_oil_spill_part3"


def download_record(record_id: str, out_dir: Path) -> None:
    print(f"\nFetching Zenodo record {record_id} metadata...")
    try:
        meta = fetch_record_metadata(record_id)
    except Exception as e:
        print(f"ERROR: could not reach Zenodo API for record {record_id}: {e}", file=sys.stderr)
        sys.exit(1)

    files = meta.get("files", [])
    if not files:
        print(f"ERROR: no files listed on Zenodo record {record_id}.", file=sys.stderr)
        sys.exit(1)

    print(f"Record {record_id} has {len(files)} file(s):")
    for f in files:
        print(f"  - {f['key']} ({f['size'] / (1024**3):.2f} GB)")

    for f in files:
        download_full(f["links"]["self"], out_dir / f["key"])


def main() -> None:
    print("=== Part II: No-Oil + Look-alike training negatives ===")
    download_record(PART_II_RECORD_ID, OUT_DIR_PART_II)

    print("\n=== Part III: held-out balanced test set ===")
    download_record(PART_III_RECORD_ID, OUT_DIR_PART_III)

    print("\nDone. Part II files in", OUT_DIR_PART_II)
    print("Part III files in", OUT_DIR_PART_III)


if __name__ == "__main__":
    main()
