"""
Speed test: download the same Zenodo images archive using N concurrent
HTTP Range requests instead of one single-stream download, to check
whether the ~0.4 MB/s observed in scripts/download_zenodo_sample.py is a
Zenodo-side per-connection throttle (parallel connections should help a
lot) or a real local bandwidth ceiling (parallel connections won't help).

SAFETY: this writes to a completely separate file
(01_Train_Val_Oil_Spill_images.7z.parallel_test) and never touches
01_Train_Val_Oil_Spill_images.7z.part, the resumable single-stream
download's in-progress file. Only call replace_original_if_better() --
manually, after eyeballing the result -- to swap it in, and only then.

Usage:
    venv\\Scripts\\python.exe scripts\\download_zenodo_parallel_test.py            # run the timed test (default: 60s sample)
    venv\\Scripts\\python.exe scripts\\download_zenodo_parallel_test.py --full     # run to completion instead of just sampling
"""

import sys
import threading
import time
from pathlib import Path

import requests

ZENODO_RECORD_ID = "8346860"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
N_CONNECTIONS = 16

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "zenodo_sar_oil_spill"
ORIGINAL_PART = OUT_DIR / "01_Train_Val_Oil_Spill_images.7z.part"
FINAL_TARGET = OUT_DIR / "01_Train_Val_Oil_Spill_images.7z"
TEST_FILE = OUT_DIR / "01_Train_Val_Oil_Spill_images.7z.parallel_test"

progress_lock = threading.Lock()
total_downloaded = 0
stop_flag = threading.Event()


def fetch_image_url() -> tuple[str, int]:
    resp = requests.get(ZENODO_API_URL, timeout=30)
    resp.raise_for_status()
    files = {f["key"]: f for f in resp.json()["files"]}
    key = next(k for k in files if "images" in k.lower())
    f = files[key]
    return f["links"]["self"], f["size"]


def download_range(url: str, start: int, end: int, fh, chunk_size: int = 4 * 1024 * 1024) -> None:
    """Download bytes [start, end] (inclusive) and write them at the right offset in fh."""
    global total_downloaded
    headers = {"Range": f"bytes={start}-{end}"}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        pos = start
        for chunk in r.iter_content(chunk_size=chunk_size):
            if stop_flag.is_set():
                return
            with progress_lock:
                fh.seek(pos)
                fh.write(chunk)
            pos += len(chunk)
            with progress_lock:
                total_downloaded += len(chunk)


def main() -> None:
    sample_only_seconds = None if "--full" in sys.argv else 60

    print(f"Fetching record metadata...")
    url, total_size = fetch_image_url()
    print(f"Image archive: {total_size / (1024**3):.2f} GB, using {N_CONNECTIONS} connections")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Preallocate the file so each thread can seek+write its own region independently.
    with open(TEST_FILE, "wb") as f:
        f.truncate(total_size)

    chunk_size = total_size // N_CONNECTIONS
    ranges = []
    for i in range(N_CONNECTIONS):
        start = i * chunk_size
        end = (start + chunk_size - 1) if i < N_CONNECTIONS - 1 else total_size - 1
        ranges.append((start, end))

    fh = open(TEST_FILE, "r+b")
    threads = [
        threading.Thread(target=download_range, args=(url, start, end, fh), daemon=True)
        for start, end in ranges
    ]

    start_time = time.perf_counter()
    for t in threads:
        t.start()

    last_print = start_time
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(1)
            now = time.perf_counter()
            if now - last_print >= 5:
                with progress_lock:
                    downloaded = total_downloaded
                elapsed = now - start_time
                speed_mbps = downloaded / elapsed / (1024 ** 2) if elapsed > 0 else 0
                pct = downloaded / total_size * 100
                print(f"  {downloaded / (1024**3):.2f} / {total_size / (1024**3):.2f} GB "
                      f"({pct:.1f}%)  {speed_mbps:.1f} MB/s combined  [{N_CONNECTIONS} connections]", flush=True)
                last_print = now

            if sample_only_seconds and (now - start_time) >= sample_only_seconds:
                print(f"\n{sample_only_seconds}s sample window complete, stopping test threads...")
                stop_flag.set()
                break
    except KeyboardInterrupt:
        stop_flag.set()

    for t in threads:
        t.join(timeout=30)
    fh.close()

    elapsed = time.perf_counter() - start_time
    with progress_lock:
        downloaded = total_downloaded
    speed_mbps = downloaded / elapsed / (1024 ** 2) if elapsed > 0 else 0
    print(f"\n=== result ===")
    print(f"downloaded {downloaded / (1024**3):.2f} GB in {elapsed:.1f}s -> {speed_mbps:.1f} MB/s combined")

    if stop_flag.is_set() and sample_only_seconds:
        print(f"(sample only -- test file left in place at {TEST_FILE} for a possible --full continuation,")
        print(f" or delete it manually: this script never touches {ORIGINAL_PART.name})")
    elif downloaded >= total_size:
        actual_size = TEST_FILE.stat().st_size
        print(f"Download complete. File size: {actual_size / (1024**3):.2f} GB (expected {total_size / (1024**3):.2f} GB)")
        if actual_size == total_size:
            print("Size matches exactly. Safe to consider replacing the original with this file.")
        else:
            print("WARNING: size mismatch -- do NOT use this file, something went wrong.")


if __name__ == "__main__":
    main()
