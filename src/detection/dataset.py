"""
PyTorch Dataset classes for tiled SAR image/mask segmentation training.

Two datasets here, for two different purposes -- see DECISIONS.md:
  - `SARTileDataset`: the Step 1 sanity-check dataset. Precomputes all
    tiles in memory (fine at the tiny scale it was used at: a handful of
    640x640 PANGAEA quicklooks with bbox-rasterized pseudo-masks). Not
    used for real training -- kept for the sanity-pass script.
  - `ZenodoTileDataset`: the real training dataset, over the full Part I
    (1200 oil-positive) + Part II (685 no-oil + 685 look-alike) Zenodo
    images -- real calibrated Sigma0-dB GeoTIFFs with real pixel-accurate
    masks. Reads tiles as windowed disk reads rather than precomputing,
    since the full set doesn't fit in memory.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import Dataset

from detection.augment import augment_pair
from detection.preprocess import compute_nodata_mask, lee_filter, normalize_db_fixed, tile_image


def bbox_to_mask(xml_path: str | Path, image_shape: tuple[int, int]) -> np.ndarray:
    """Rasterize all <object><bndbox> boxes in a Pascal-VOC-style XML into a binary mask."""
    tree = ET.parse(xml_path)
    mask = np.zeros(image_shape, dtype=np.float32)
    for obj in tree.findall("object"):
        box = obj.find("bndbox")
        xmin = int(float(box.find("xmin").text))
        ymin = int(float(box.find("ymin").text))
        xmax = int(float(box.find("xmax").text))
        ymax = int(float(box.find("ymax").text))
        mask[ymin:ymax, xmin:xmax] = 1.0
    return mask


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Min-max normalize a despeckled image to [0, 1] for the network input."""
    lo, hi = float(image.min()), float(image.max())
    if hi - lo < 1e-6:
        return np.zeros_like(image, dtype=np.float32)
    return ((image - lo) / (hi - lo)).astype(np.float32)


class SARTileDataset(Dataset):
    """
    Tiles a list of (image, mask) array pairs into fixed-size patches and
    serves them as (1, tile, tile) float tensors. Tiling is precomputed at
    init since the current sample count is tiny.
    """

    def __init__(self, pairs: list[tuple[np.ndarray, np.ndarray]], tile_size: int = 256, stride: int | None = None):
        self.tile_size = tile_size
        self.tiles: list[tuple[np.ndarray, np.ndarray]] = []
        for image, mask in pairs:
            if image.shape != mask.shape:
                raise ValueError(f"image/mask shape mismatch: {image.shape} vs {mask.shape}")
            despeckled = lee_filter(image)
            image_tiles = tile_image(despeckled, tile_size=tile_size, stride=stride)
            mask_tiles = tile_image(mask, tile_size=tile_size, stride=stride)
            for img_tile, mask_tile in zip(image_tiles, mask_tiles):
                self.tiles.append((normalize_image(img_tile), mask_tile))

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.tiles[idx]
        image_t = torch.from_numpy(image).unsqueeze(0).float()
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        return image_t, mask_t


@dataclass
class ImageMaskPair:
    image_path: Path
    mask_path: Path
    label: str  # "oil" | "no_oil" | "lookalike" -- source category, for stratified splitting


class ZenodoTileDataset(Dataset):
    """
    Real training dataset over the full Part I + Part II Zenodo images
    (2048x2048 calibrated Sigma0-dB GeoTIFFs). Unlike SARTileDataset above,
    this does NOT precompute/hold all tiles in memory -- with 2500+
    source images at 2048x2048, that's tens of GB. Instead it builds a
    lightweight index of (pair_idx, row, col) tile coordinates up front and
    reads each tile as a windowed read directly off disk via rasterio when
    requested, so memory use stays bounded regardless of dataset size.

    Despeckling (Lee filter) is applied per-tile rather than on the full
    image before cropping -- avoids loading the full 2048x2048 array just
    to filter it, and the Lee filter's window (7px default) is tiny
    relative to a 512px tile so the edge-of-tile approximation error is
    negligible.
    """

    def __init__(
        self,
        pairs: list[ImageMaskPair],
        tile_size: int = 512,
        stride: int | None = None,
        augment: bool = False,
        seed: int = 0,
        channels: tuple[int, ...] = (1,),
        normalize_fn=None,
        return_nodata_mask: bool = False,
        explicit_index: list[tuple[int, int, int]] | None = None,
    ):
        """
        channels selects which 1-indexed rasterio bands to read, e.g. (1,)
        for the original single-band (VV) setup or (1, 2) for both real SAR
        bands the Zenodo GeoTIFFs carry (confirmed via direct inspection --
        distinct dB ranges per band, not a duplicate). Defaults to (1,) so
        existing callers (sanity checks, anything not passing this
        explicitly) keep the exact prior single-channel behavior;
        scripts/train_detection.py's --channels flag is what actually opts
        into dual-channel training.

        normalize_fn, if given, replaces the default normalize_db_fixed
        call with any callable(image) -> image in [0,1] -- e.g.
        functools.partial(normalize_db_per_channel, ranges=[...]) for the
        per-band ranges Gate C's audit found necessary (see
        docs/metric_audit.md). Defaults to None, which keeps the exact
        original normalize_db_fixed(image) behavior.

        return_nodata_mask, if True, makes __getitem__ return a THIRD
        tensor: a (1, H, W) float mask, 1.0 = valid pixel, 0.0 = nodata
        (see preprocess.compute_nodata_mask -- exact 0.0 dB in both real
        SAR bands, ~1.8% of pixels, confirmed unmasked in loss/metrics
        before this was added; see docs/metric_audit.md Finding #12).
        Always reads both bands to compute it regardless of `channels`
        (one extra cheap windowed read when channels != (1, 2)), and
        computes it from the RAW pre-despeckle values -- lee_filter's local
        averaging would blur exact-zero nodata pixels into non-zero
        neighbors, destroying the exact-zero signature this detection
        depends on. Defaults to False: every existing caller (SARTileDataset
        is unaffected entirely; old scripts/tests using ZenodoTileDataset
        without this flag) keeps the exact original 2-tuple return.

        explicit_index, if given, is a list of (pair_idx, row_off, col_off)
        tuples used AS-IS instead of the auto-generated full non-overlapping
        grid below -- lets a caller train on a specific TILE-level subset
        (e.g. an oil/zero-oil-balanced ablation manifest -- see
        scripts/build_ablation_manifest.py) even though `pairs`/the manifest
        CSV format is only ever per-SOURCE-IMAGE, not per-tile. Defaults to
        None, which keeps the exact original "every tile of every listed
        image" behavior for every existing caller.
        """
        self.tile_size = tile_size
        self.stride = stride or tile_size
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.pairs = pairs
        self.channels = channels
        self.normalize_fn = normalize_fn or normalize_db_fixed
        self.return_nodata_mask = return_nodata_mask

        if explicit_index is not None:
            self.index: list[tuple[int, int, int]] = list(explicit_index)
        else:
            self.index = []  # (pair_idx, row_off, col_off)
            for i, pair in enumerate(pairs):
                with rasterio.open(pair.image_path) as src:
                    h, w = src.height, src.width
                for y in range(0, h - tile_size + 1, self.stride):
                    for x in range(0, w - tile_size + 1, self.stride):
                        self.index.append((i, y, x))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pair_idx, y, x = self.index[idx]
        pair = self.pairs[pair_idx]
        window = Window(x, y, self.tile_size, self.tile_size)

        with rasterio.open(pair.image_path) as src:
            image = src.read(list(self.channels), window=window).astype(np.float32)  # (C, H, W)
            if len(self.channels) == 1:
                image = image[0]  # (H, W) -- keeps the exact original single-channel shape/behavior
            if self.return_nodata_mask:
                # RAW, pre-despeckle reads -- must happen before lee_filter touches
                # `image`, since despeckling blurs exact-zero nodata pixels into
                # non-zero neighbors and would destroy the signal this depends on.
                raw_band1 = src.read(1, window=window).astype(np.float32)
                raw_band2 = src.read(2, window=window).astype(np.float32)
        with rasterio.open(pair.mask_path) as src:
            mask = src.read(1, window=window).astype(np.float32)

        if self.return_nodata_mask:
            valid = (~compute_nodata_mask(raw_band1, raw_band2)).astype(np.float32)  # 1.0 = valid, 0.0 = nodata

        image = lee_filter(image)  # per-channel automatically if image is (C, H, W)

        # Augment (flips/rotation/dB-scale speckle jitter) BEFORE normalizing:
        # the speckle jitter's std is calibrated in real dB units, so it must
        # be applied while the image is still in dB, not after normalize_db_fixed
        # has compressed the ~50dB range down to [0, 1] (applying it after would
        # inject ~50x too much noise relative to what the std was tuned for).
        if self.augment:
            if self.return_nodata_mask:
                image, mask, extras = augment_pair(image, mask, self.rng, extra_masks=[valid])
                valid = extras[0]
            else:
                image, mask = augment_pair(image, mask, self.rng)

        image = self.normalize_fn(image)

        image_t = torch.from_numpy(image).float()
        if image_t.ndim == 2:
            image_t = image_t.unsqueeze(0)  # (H, W) -> (1, H, W); already (C, H, W) otherwise
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()

        if self.return_nodata_mask:
            valid_t = torch.from_numpy(np.ascontiguousarray(valid)).unsqueeze(0).float()
            return image_t, mask_t, valid_t
        return image_t, mask_t


def compute_oil_tile_weights(dataset: "ZenodoTileDataset", target_oil_fraction: float = 0.5) -> np.ndarray:
    """
    Per-tile sample weights for a WeightedRandomSampler, so oil-containing
    tiles make up ~target_oil_fraction of what a training epoch actually
    samples (with replacement) instead of whatever their real frequency
    happens to be.

    Added after scripts/analyze_tile_oil_distribution.py measured the real
    tile grid: 82.4% of all 34,940 training tiles have zero oil pixels
    (no_oil/lookalike images are 100% zero by construction, and most tiles
    even within the 1200 oil images miss the slick). pos_weight in
    DiceBCELoss only reweights pixels *inside* a tile that already has some
    oil -- it does nothing for the ~4/5 of batches that are entirely
    oil-free, which is a signal-sparsity problem no loss-function choice
    can fix. This is a separate lever from loss/pos_weight, kept as its
    own dataset-level function (not baked into ZenodoTileDataset itself)
    so a loss-function trial and a sampling trial stay independently
    testable.

    Reads only the mask band (cheap) for every indexed tile once, up
    front -- real per-tile ground truth, not an estimate from the whole
    image's overall oil fraction.
    """
    is_oil = np.zeros(len(dataset.index), dtype=bool)
    for i, (pair_idx, y, x) in enumerate(dataset.index):
        mask_path = dataset.pairs[pair_idx].mask_path
        with rasterio.open(mask_path) as src:
            tile = src.read(1, window=Window(x, y, dataset.tile_size, dataset.tile_size))
        is_oil[i] = (tile > 0).any()

    n_oil, n_total = int(is_oil.sum()), len(is_oil)
    n_non = n_total - n_oil
    if n_oil == 0 or n_non == 0:
        return np.ones(n_total, dtype=np.float32)

    # Solve for oil_weight such that oil tiles make up target_oil_fraction
    # of total sample weight, holding non-oil weight at 1.0:
    #   target = (n_oil * oil_weight) / (n_oil * oil_weight + n_non)
    oil_weight = target_oil_fraction * n_non / (n_oil * (1 - target_oil_fraction))
    weights = np.where(is_oil, oil_weight, 1.0).astype(np.float32)
    print(f"compute_oil_tile_weights: {n_oil}/{n_total} tiles have oil ({100 * n_oil / n_total:.1f}%), "
          f"weighting oil tiles {oil_weight:.2f}x to target {target_oil_fraction:.0%} representation per epoch")
    return weights
