"""
Download 02_Test_images_and_ground_truth.7z (Zenodo Part III, record
13761290) using scripts/_zenodo_download_utils.py's download_parallel().

Written after finding this exact file corrupted by two duplicate
`download_zenodo_part2_part3.py` processes writing to the same .part
file at once (see LOG.md "Real bug: corrupted Lookalike archive found..."
and DECISIONS.md "Corrupted archive diagnosis and the
parallel-download-safety fix" for the full incident) -- this script's
disjoint-byte-range design can't have that failure mode.

Usage:
    venv\\Scripts\\python.exe scripts\\download_zenodo_part3_parallel.py
"""

from pathlib import Path

from _zenodo_download_utils import download_parallel

RECORD_ID = "13761290"
FILE_KEY = "02_Test_images_and_ground_truth.7z"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "zenodo_sar_oil_spill_part3"


def main() -> None:
    download_parallel(RECORD_ID, FILE_KEY, OUT_DIR / FILE_KEY, n_connections=4)


if __name__ == "__main__":
    main()
