"""
Shared resumable-download helper for Zenodo records, used by
download_zenodo_sample.py (Part I) and download_zenodo_part2_part3.py
(Part II/III). Factored out rather than duplicated once a second script
needed the same resume/progress logic.

Also holds download_parallel(), a modest-connection-count (default 4)
concurrent downloader -- see DECISIONS.md "Corrupted archive diagnosis and
the parallel-download-safety fix" for why this exists: the single-stream
download_full() above sustains only ~0.1-0.4 MB/s against Zenodo (a real
per-connection throttle, not a local bandwidth limit -- confirmed via a
16-connection test that hit ~12 MB/s combined but crashed on the full run
from connection-count instability), and why it writes to a separate
preallocated file with each thread owning a disjoint byte range rather than
sharing appendable state with anything else that might be downloading the
same file.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import py7zr
import requests

PROGRESS_INTERVAL_SEC = 10


def fetch_record_metadata(record_id: str) -> dict:
    resp = requests.get(f"https://zenodo.org/api/records/{record_id}", timeout=30)
    resp.raise_for_status()
    return resp.json()


def format_eta(seconds: float) -> str:
    if seconds != seconds or seconds in (float("inf"), float("-inf")):  # NaN/inf guard
        return "unknown"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def download_full(url: str, dest: Path) -> None:
    """Resumable download with progress reporting. Safe to interrupt and re-run."""
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
                        f"  [{dest.name}] {downloaded / (1024**3):.2f} / {total_size / (1024**3):.2f} GB "
                        f"({pct:.1f}%)  {speed_mbps:.1f} MB/s  ETA {format_eta(eta)}",
                        flush=True,
                    )
                    # (flush=True above should already prevent buffering; if progress reads still
                    # lag the real file size, check the file's actual size directly as ground truth.)
                    last_print = now

        tmp.rename(dest)
    print(f"  done: {dest} ({dest.stat().st_size / (1024**3):.2f} GB)")


def download_parallel(record_id: str, file_key: str, dest: Path, n_connections: int = 4) -> None:
    """
    Downloads one Zenodo record file using n_connections concurrent HTTP
    Range requests into a preallocated temp file, each thread owning a
    disjoint byte range (no shared appendable state -- see module
    docstring for why). Verifies final size and py7zr header integrity
    before renaming into place.

    Resumable across runs: a small JSON sidecar (dest.name + ".progress.json")
    records each thread's current write position, updated in lockstep with
    every write. A failed attempt leaves both the partial data and this
    sidecar in place; the next call reads it back and re-requests only the
    unwritten tail of each thread's byte range (Range: bytes={saved_pos}-{end}),
    rather than re-downloading from scratch. Ranges are recomputed from
    total_size/n_connections each call, so n_connections must stay the same
    across retries of the same download for resume to line up correctly --
    if it's ever changed, the mismatch is detected and the download restarts
    from scratch rather than silently resuming into the wrong byte ranges.
    """
    if dest.exists():
        print(f"  already have {dest.name}, skipping")
        return

    meta = fetch_record_metadata(record_id)
    files = {f["key"]: f for f in meta["files"]}
    f = files[file_key]
    url, total_size = f["links"]["self"], f["size"]

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".parallel")
    progress_path = tmp.with_suffix(tmp.suffix + ".progress.json")

    chunk_ranges = []
    base_chunk = total_size // n_connections
    for i in range(n_connections):
        start = i * base_chunk
        end = (start + base_chunk - 1) if i < n_connections - 1 else total_size - 1
        chunk_ranges.append((start, end))

    resuming = tmp.exists() and progress_path.exists()
    resume_positions = [start for start, _ in chunk_ranges]
    if resuming:
        saved = json.loads(progress_path.read_text())
        if saved.get("total_size") == total_size and saved.get("n_connections") == n_connections \
                and len(saved.get("positions", [])) == n_connections:
            resume_positions = saved["positions"]
            done_gb = sum(p - s for p, (s, _) in zip(resume_positions, chunk_ranges)) / (1024 ** 3)
            print(f"  resuming {file_key} from {done_gb:.2f} GB already written")
        else:
            print("  stale/mismatched progress file, restarting from scratch")
            resuming = False

    if not resuming:
        with open(tmp, "wb") as fh:
            fh.truncate(total_size)

    print(f"  {file_key}: {total_size / (1024**3):.2f} GB, using {n_connections} connections")

    progress_lock = threading.Lock()
    positions = list(resume_positions)
    thread_errors: list[Exception] = []
    last_save = time.perf_counter()

    def save_progress() -> None:
        progress_path.write_text(json.dumps({
            "total_size": total_size, "n_connections": n_connections, "positions": positions,
        }))

    def download_range(idx: int, start: int, end: int, fh, chunk_size: int = 4 * 1024 * 1024) -> None:
        nonlocal last_save
        if start > end:
            return  # already fully downloaded on a prior attempt
        try:
            with requests.get(url, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=60) as r:
                r.raise_for_status()
                pos = start
                for chunk in r.iter_content(chunk_size=chunk_size):
                    with progress_lock:
                        fh.seek(pos)
                        fh.write(chunk)
                        pos += len(chunk)
                        positions[idx] = pos
                        now = time.perf_counter()
                        if now - last_save >= PROGRESS_INTERVAL_SEC:
                            save_progress()
                            last_save = now
        except Exception as e:
            thread_errors.append(e)

    fh = open(tmp, "r+b")
    threads = [
        threading.Thread(target=download_range, args=(i, resume_positions[i], end, fh), daemon=True)
        for i, (_, end) in enumerate(chunk_ranges)
    ]

    start_time = time.perf_counter()
    already_done = sum(p - s for p, (s, _) in zip(resume_positions, chunk_ranges))
    for t in threads:
        t.start()

    last_print = start_time
    while any(t.is_alive() for t in threads):
        time.sleep(1)
        now = time.perf_counter()
        if now - last_print >= PROGRESS_INTERVAL_SEC:
            with progress_lock:
                downloaded = sum(positions[i] - chunk_ranges[i][0] for i in range(n_connections))
            elapsed = now - start_time
            speed_mbps = (downloaded - already_done) / elapsed / (1024 ** 2) if elapsed > 0 else 0
            total_progress = sum(positions) - sum(s for s, _ in chunk_ranges)
            pct = total_progress / total_size * 100
            eta_h = (total_size - total_progress) / (speed_mbps * 1024 ** 2) / 3600 if speed_mbps > 0 else float("nan")
            print(f"  [{file_key}] {total_progress / (1024**3):.2f} / {total_size / (1024**3):.2f} GB "
                  f"({pct:.1f}%)  {speed_mbps:.1f} MB/s combined  ETA {eta_h:.1f}h", flush=True)
            last_print = now

    for t in threads:
        t.join(timeout=30)
    fh.close()
    with progress_lock:
        save_progress()

    if thread_errors:
        print(f"  ERROR: {len(thread_errors)} thread(s) failed: {thread_errors[0]}")
        print(f"  progress saved -- re-run to resume from {(sum(positions) - sum(s for s, _ in chunk_ranges)) / (1024**3):.2f} GB")
        raise RuntimeError(f"parallel download of {file_key} failed")

    actual_size = tmp.stat().st_size
    if actual_size != total_size:
        tmp.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)
        raise RuntimeError(f"size mismatch for {file_key}: got {actual_size}, expected {total_size}")

    with py7zr.SevenZipFile(tmp, "r") as z:
        names = z.getnames()
    print(f"  archive OK: {len(names)} entries")

    tmp.rename(dest)
    progress_path.unlink(missing_ok=True)
    print(f"  done: {dest} ({actual_size / (1024**3):.2f} GB)")
