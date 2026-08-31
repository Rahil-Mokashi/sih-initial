"""Tests for src/detection/dataset.py's bbox_to_mask (Pascal-VOC XML -> rectangular mask)
and ZenodoTileDataset's explicit_index (tile-level ablation subset support)."""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

from detection.dataset import ImageMaskPair, ZenodoTileDataset, bbox_to_mask


def _write_voc_xml(path: Path, boxes: list[tuple[int, int, int, int]]) -> None:
    objects = "".join(
        f"<object><bndbox><xmin>{xmin}</xmin><ymin>{ymin}</ymin>"
        f"<xmax>{xmax}</xmax><ymax>{ymax}</ymax></bndbox></object>"
        for xmin, ymin, xmax, ymax in boxes
    )
    path.write_text(f"<annotation>{objects}</annotation>")


def test_single_box_rasterized_correctly(tmp_path):
    xml_path = tmp_path / "one.xml"
    _write_voc_xml(xml_path, [(10, 20, 30, 40)])
    mask = bbox_to_mask(xml_path, (100, 100))

    assert mask.shape == (100, 100)
    assert mask.sum() == (40 - 20) * (30 - 10)  # box area
    assert mask[20:40, 10:30].all()
    assert not mask[0:20, :].any()
    assert not mask[:, 0:10].any()


def test_multiple_boxes_all_rasterized(tmp_path):
    xml_path = tmp_path / "two.xml"
    _write_voc_xml(xml_path, [(0, 0, 5, 5), (50, 50, 60, 60)])
    mask = bbox_to_mask(xml_path, (100, 100))

    assert mask[0:5, 0:5].all()
    assert mask[50:60, 50:60].all()
    assert mask.sum() == 5 * 5 + 10 * 10


def test_no_objects_gives_empty_mask(tmp_path):
    xml_path = tmp_path / "empty.xml"
    _write_voc_xml(xml_path, [])
    mask = bbox_to_mask(xml_path, (50, 50))
    assert mask.sum() == 0


def _write_geotiff(path: Path, array: np.ndarray, dtype: str = "float32") -> None:
    """array: (C, H, W)."""
    count, h, w = array.shape
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=count,
                        dtype=dtype, transform=Affine.identity()) as dst:
        dst.write(array)


def test_zenodo_dataset_explicit_index_selects_exact_tiles(tmp_path):
    # Two 512x512 tiles side by side: (y=0,x=0) and (y=0,x=512). Mark the
    # SECOND tile all-oil and the first all-empty, so reading the wrong
    # tile is immediately obvious rather than silently passing.
    h, w = 512, 1024
    rng = np.random.default_rng(0)
    band1 = rng.uniform(-30, -10, size=(h, w)).astype(np.float32)
    band2 = rng.uniform(-30, -10, size=(h, w)).astype(np.float32)
    mask = np.zeros((h, w), dtype=np.float32)
    mask[:, 512:] = 1.0

    image_path = tmp_path / "img.tif"
    mask_path = tmp_path / "mask.tif"
    _write_geotiff(image_path, np.stack([band1, band2]))
    _write_geotiff(mask_path, mask[np.newaxis, :, :])

    pair = ImageMaskPair(image_path, mask_path, "oil")

    full = ZenodoTileDataset([pair], tile_size=512, channels=(1,))
    assert len(full) == 2  # default behavior unchanged: both tiles auto-indexed

    # explicit_index selects ONLY the second tile (pair_idx=0, y=0, x=512)
    subset = ZenodoTileDataset([pair], tile_size=512, channels=(1,), explicit_index=[(0, 0, 512)])
    assert len(subset) == 1
    _, mask_t = subset[0]
    assert mask_t.squeeze(0).numpy().tolist() == mask[:, 512:1024].tolist()  # got the oil tile, not the empty one
