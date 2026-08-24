"""
Download a small sample from the Sentinel-1 SAR Oil Spill dataset (Zenodo).

DOI: 10.5281/zenodo.8346860 ("Part I": train/val images + masks)

IMPORTANT — read before running:
This record does NOT publish individual per-image files. It publishes
exactly two files:
  - 01_Train_Val_Oil_Spill_images.7z   (~40 GB, all 1200 SAR images, solid archive)
  - 01_Train_Val_Oil_Spill_mask.7z     (~6 MB,  all 1200 masks)
7z is a solid/compressed archive format with its file index stored at the
END of the file, so there is no way to cherry-pick "a few images" via HTTP
Range requests without downloading effectively the whole archive anyway.
There is no smaller "preview" or per-file listing available from Zenodo for
this dataset.

So this script does the honest thing instead of pretending to sample:
  - SAMPLE_MODE (default): downloads the small masks archive in full (~6MB,
    all 1200 masks — genuinely small) so you can inspect ground-truth
    structure/format, and downloads only the FIRST `SAMPLE_BYTES` of the
    images archive as a truncated .7z (useful for confirming the endpoint
    is reachable and for eyeballing header bytes) but note: a truncated .7z
    generally CANNOT be extracted -- see NOTE below.
  - FULL_DOWNLOAD = True: downloads both files in full (~40GB total,
    will take a long time and a lot of disk).

NOTE on getting real sample *images* to test the preprocessing pipeline
right now, without the 40GB wait: use one of the small sample images we
already saved to data/raw/manual_samples/ (see docs/DECISIONS.md, Step 0
entry) or wait for the full download. This script alone cannot produce a
handful of extractable real images from this dataset.

To pull the FULL dataset later, flip FULL_DOWNLOAD to True below.
"""

import sys
from pathlib import Path

import requests

ZENODO_RECORD_ID = "8346860"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

# --- flip this to True to download both files in full (~40GB) ---
FULL_DOWNLOAD = False
SAMPLE_BYTES = 20 * 1024 * 1024  # 20MB truncated peek of the images archive

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "zenodo_sar_oil_spill"


def fetch_record_metadata() -> dict:
    resp = requests.get(ZENODO_API_URL, timeout=30)
    resp.raise_for_status()
    return resp.json()


def download_full(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  already have {dest.name}, skipping")
        return
    print(f"  downloading {dest.name} in full...")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
        tmp.rename(dest)
    print(f"  done: {dest}")


def download_partial(url: str, dest: Path, n_bytes: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  already have {dest.name}, skipping")
        return
    print(f"  downloading first {n_bytes / 1024 / 1024:.0f} MB of {dest.name} (peek only, not extractable)...")
    headers = {"Range": f"bytes=0-{n_bytes - 1}"}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
    print(f"  done: {dest} ({dest.stat().st_size} bytes)")


def main() -> None:
    print(f"Fetching Zenodo record {ZENODO_RECORD_ID} metadata...")
    try:
        meta = fetch_record_metadata()
    except requests.RequestException as e:
        print(f"ERROR: could not reach Zenodo API: {e}", file=sys.stderr)
        sys.exit(1)

    files = {f["key"]: f for f in meta.get("files", [])}
    if not files:
        print("ERROR: no files listed on this Zenodo record.", file=sys.stderr)
        sys.exit(1)

    images_key = next((k for k in files if "images" in k.lower()), None)
    mask_key = next((k for k in files if "mask" in k.lower()), None)

    if mask_key:
        f = files[mask_key]
        print(f"Mask archive: {f['key']} ({f['size'] / 1024 / 1024:.1f} MB)")
        download_full(f["links"]["self"], OUT_DIR / f["key"])

    if images_key:
        f = files[images_key]
        print(f"Image archive: {f['key']} ({f['size'] / 1024 / 1024 / 1024:.1f} GB)")
        if FULL_DOWNLOAD:
            download_full(f["links"]["self"], OUT_DIR / f["key"])
        else:
            download_partial(f["links"]["self"], OUT_DIR / f["key"], SAMPLE_BYTES)
            print(
                "  NOTE: this is a truncated file for connectivity-testing only. "
                "It will NOT extract with 7z (the file index lives at the end of "
                "the archive). Set FULL_DOWNLOAD=True to actually get usable images."
            )

    print("\nDone. Files are in", OUT_DIR)


if __name__ == "__main__":
    main()
