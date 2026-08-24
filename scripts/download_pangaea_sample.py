"""
Download a small sample from the PANGAEA / ESSD Eastern Mediterranean oil
spill dataset.

DOI: 10.1594/PANGAEA.980773
"Oil slicks, look-alikes and other remarkable SAR signatures in Sentinel-1
imagery in the Eastern Mediterranean Sea in 2019" (Yang & Singha, 2025)

Unlike the Zenodo dataset, this one publishes individual per-patch files
(no monolithic archive), so real sampling is straightforward:
  - data_matrix.tab: one tab-separated table, ALL 5563 rows (~2.4MB) --
    real acquisition date/time, Sentinel-1 product ID, and lat/lon corner
    coordinates for every labeled patch (oil / look-alike / no-oil). Small
    enough to always download in full.
  - <tag>.jpg / <tag>.xml per patch (e.g. ow-0001.jpg + ow-0001.xml):
    640x640 single-channel JPEG quicklook + a Pascal-VOC-style XML with the
    bounding box of the labeled object. These ARE individually downloadable,
    no account needed. This script pulls a small sample of N patches.

NOTE: these JPGs are quicklook visualizations, not calibrated Sigma0-dB
GeoTIFFs like the Zenodo training set -- see DECISIONS.md. They're useful
for: verifying real dates/coords are attached to each detection (needed for
the drift/attribution stages), and as a real-world image to sanity-check
the preprocessing code's tiling logic. They are NOT a substitute for the
Zenodo training data.

To pull every patch's jpg/xml instead of a small sample, raise SAMPLE_COUNT
below (there are ~3655 oil-set + 2290 no-oil-set patches; check the table
for exact counts), or use the "allfiles.zip"/"allfiles.tar" bulk download
linked in the table header (requires a free PANGAEA account).
"""

import sys
from pathlib import Path

import requests

TAB_URL = "https://doi.pangaea.de/10.1594/PANGAEA.980773?format=textfile"
FILES_BASE_URL = "https://download.pangaea.de/dataset/980773/files/"

SAMPLE_COUNT = 5  # how many jpg/xml patch pairs to pull

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "pangaea_med_oil_spill"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  already have {dest.name}, skipping")
        return
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    print(f"  downloaded {dest.name} ({len(resp.content)} bytes)")


def parse_data_matrix(tab_path: Path) -> list[dict]:
    """Parse the PANGAEA .tab file into row dicts, skipping the header block."""
    lines = tab_path.read_text(encoding="utf-8").splitlines()
    header_idx = next(i for i, line in enumerate(lines) if line.startswith("*/")) + 1
    columns = lines[header_idx].split("\t")
    rows = []
    for line in lines[header_idx + 1:]:
        if not line.strip():
            continue
        values = line.split("\t")
        rows.append(dict(zip(columns, values)))
    return rows


def main() -> None:
    print("Downloading full data_matrix.tab (real dates/coords per detection)...")
    tab_dest = OUT_DIR / "data_matrix.tab"
    try:
        download(TAB_URL, tab_dest)
    except requests.RequestException as e:
        print(f"ERROR: could not download the PANGAEA data table: {e}", file=sys.stderr)
        sys.exit(1)

    rows = parse_data_matrix(tab_dest)
    print(f"Table has {len(rows)} annotated patches.")

    jpg_col = "IMAGE (jpg_file)"
    xml_col = "Binary (xml_file)"
    sample_rows = rows[:SAMPLE_COUNT]

    print(f"Downloading {len(sample_rows)} sample image+annotation pairs...")
    for row in sample_rows:
        jpg_name = row.get(jpg_col)
        xml_name = row.get(xml_col)
        print(f"- {jpg_name} (patch {row.get('ID (patch_name)')}, "
              f"start {row.get('Date/Time (start_time)')}, "
              f"Sentinel product {row.get('ID (Sentinel_ID)')})")
        if jpg_name:
            download(FILES_BASE_URL + jpg_name, OUT_DIR / "images" / jpg_name)
        if xml_name:
            download(FILES_BASE_URL + xml_name, OUT_DIR / "annotations" / xml_name)

    print("\nDone. Files are in", OUT_DIR)
    print("Each row in data_matrix.tab has real start/end acquisition time and")
    print("lon/lat corner coordinates for the patch, plus the object's pixel")
    print("bounding box -- this is what later stages will use as ground truth")
    print("for the drift back-trace and AIS cross-reference.")


if __name__ == "__main__":
    main()
