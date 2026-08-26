"""Tests for src/detection/dataset.py's bbox_to_mask (Pascal-VOC XML -> rectangular mask)."""

from pathlib import Path

import numpy as np

from detection.dataset import bbox_to_mask


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
