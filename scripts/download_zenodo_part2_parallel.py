"""
Download 01_Train_Val_Lookalike_images.7z (Zenodo Part II, record 8253899)
using scripts/_zenodo_download_utils.py's download_parallel() -- a modest
(default 4) concurrent-connection downloader, safe against the
double-writer corruption a shared .part file is vulnerable to. See that
function's docstring, and DECISIONS.md "Corrupted archive diagnosis and
the parallel-download-safety fix", for the full why.

Usage:
    venv\\Scripts\\python.exe scripts\\download_zenodo_part2_parallel.py
"""

from pathlib import Path

from _zenodo_download_utils import download_parallel

RECORD_ID = "8253899"
FILE_KEY = "01_Train_Val_Lookalike_images.7z"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "zenodo_sar_oil_spill_part2"


def main() -> None:
    download_parallel(RECORD_ID, FILE_KEY, OUT_DIR / FILE_KEY, n_connections=4)


if __name__ == "__main__":
    main()
