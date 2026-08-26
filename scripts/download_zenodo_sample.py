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
  - FULL_DOWNLOAD = True (current default, as of Step 2 -- the sanity
    training pass in Step 1 proved the pipeline works, so we committed to
    the real download): downloads the masks archive (~6MB) and the full
    images archive (~38GB) with resumable, progress-reporting download
    (prints % complete / speed / ETA every ~10s; safe to interrupt and
    re-run -- it resumes via HTTP Range from wherever the .part file left
    off, with a fallback to restart if the server doesn't honor the resume
    request).
  - FULL_DOWNLOAD = False: downloads the masks in full (still small) and
    only a truncated 20MB connectivity-test peek of the images archive,
    which is NOT extractable (the 7z index lives at the end of the file)
    -- this was the Step 0 default, useful only for confirming the
    endpoint is reachable before committing to the full download.
"""

import sys
import time
from pathlib import Path

import requests

ZENODO_RECORD_ID = "8346860"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

# --- flip this to True to download both files in full (~40GB) ---
FULL_DOWNLOAD = True
SAMPLE_BYTES = 20 * 1024 * 1024  # 20MB truncated peek of the images archive (only used when FULL_DOWNLOAD is False)

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "zenodo_sar_oil_spill"

PROGRESS_INTERVAL_SEC = 10  # how often to print a progress line during a full download


def fetch_record_metadata() -> dict:
    resp = requests.get(ZENODO_API_URL, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _format_eta(seconds: float) -> str:
    if seconds != seconds or seconds in (float("inf"), float("-inf")):  # NaN/inf guard
        return "unknown"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def download_full(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  already have {dest.name}, skipping")
        return

    tmp = dest.with_suffix(dest.suffix + ".part")
    resume_from = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
    mode = "ab" if resume_from else "wb"
    if resume_from:
        print(f"  resuming {dest.name} from {resume_from / (1024**3):.2f} GB...")
    else:
        print(f"  downloading {dest.name} in full...")

    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        if resume_from and r.status_code != 206:
            print("  WARNING: server did not honor the resume request (no 206 Partial Content) -- "
                  "restarting this file from scratch to avoid corrupting it.")
            resume_from = 0
            mode = "wb"
        total_size = resume_from + int(r.headers.get("Content-Length", 0))
        downloaded = resume_from
        start = time.perf_counter()
        last_print = start

        with open(tmp, mode) as f:
            for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)

                now = time.perf_counter()
                if now - last_print >= PROGRESS_INTERVAL_SEC:
                    elapsed = now - start
                    speed_mbps = (downloaded - resume_from) / elapsed / (1024 ** 2) if elapsed > 0 else 0
                    pct = downloaded / total_size * 100 if total_size else 0
                    remaining_bytes = total_size - downloaded
                    eta = remaining_bytes / (speed_mbps * 1024 ** 2) if speed_mbps > 0 else float("nan")
                    print(
                        f"  {downloaded / (1024**3):.2f} / {total_size / (1024**3):.2f} GB "
                        f"({pct:.1f}%)  {speed_mbps:.1f} MB/s  ETA {_format_eta(eta)}",
                        flush=True,
                    )
                    last_print = now

        tmp.rename(dest)
    print(f"  done: {dest} ({dest.stat().st_size / (1024**3):.2f} GB)")


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
