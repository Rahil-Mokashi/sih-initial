"""
Config-driven training entrypoint (Phase 1 of the diagnostics/
reproducibility plan -- see docs/colab.md and
checkpoints/baseline_epoch39/MANIFEST.json for the checkpoint this
infrastructure is meant to let future runs be fairly compared against).

Wraps the existing src/detection/train.py::train() loop -- which
scripts/train_detection.py still uses unchanged -- with config parsing,
explicit seeding, a run_manifest.json, and the per-epoch/RNG-state
persistence needed to survive a Colab disconnect.

Usage:
    venv\\Scripts\\python.exe train.py --config configs/baseline.yaml

Config fields not in the schema literally specified by the Phase 1 plan,
added because the training system cannot function without them: tile_size,
channels, dataset (train_manifest/val_manifest paths), num_workers,
grad_accum_steps, keep_last_n, deterministic. See configs/baseline.yaml for
a fully-annotated example, and docs/colab.md for the overall workflow.

Known limitation, stated rather than silently worked around: `optimizer`
only supports "adam" -- src/detection/train.py's loop has only ever
constructed torch.optim.Adam directly, and no other optimizer has been
used or tested in this project, so the config does not pretend to offer a
generality that doesn't exist yet.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import random
import subprocess
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from detection.dataset import ImageMaskPair, ZenodoTileDataset, compute_oil_tile_weights  # noqa: E402
from detection.losses import DiceBCELoss, TverskyLoss  # noqa: E402
from detection.model import build_model  # noqa: E402
from detection.preprocess import normalize_db_per_channel  # noqa: E402
from detection.train import train  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent


def git_state() -> tuple[str, bool]:
    """Returns (commit_sha, is_dirty). Treats a git failure as dirty+unknown
    (fail safe) rather than silently proceeding as if the tree were clean."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True)
        return sha, bool(status.strip())
    except Exception as e:
        print(f"WARNING: could not determine git state ({e}) -- treating tree as dirty")
        return "UNKNOWN", True


def set_seed(seed: int | None, deterministic: bool) -> None:
    if seed is None:
        print("WARNING: no seed set in config -- this run will not be reproducible "
              "(see checkpoints/baseline_epoch39/MANIFEST.json for why this matters: "
              "the original epoch-39 run had this exact gap).")
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_manifest(path: Path) -> tuple[list[ImageMaskPair], list[tuple[int, int, int]] | None]:
    """Returns (pairs, explicit_index).

    explicit_index is None for the plain 3-column (image_path,mask_path,
    label) manifest format -- exact original behavior, ZenodoTileDataset
    auto-tiles every listed image's full grid.

    For the EXTENDED 5-column tile-level format (image_path,mask_path,
    label,row_offset,col_offset -- written by
    scripts/build_ablation_manifest.py, e.g. Experiment 01's ablation
    subset), returns the deduplicated per-image `pairs` plus an explicit
    (pair_idx, row_offset, col_offset) index so ZenodoTileDataset trains on
    exactly the selected TILES, not every tile of every listed image."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if "row_offset" in fieldnames and "col_offset" in fieldnames:
        pair_index_by_path: dict[str, int] = {}
        pairs: list[ImageMaskPair] = []
        explicit_index: list[tuple[int, int, int]] = []
        for row in rows:
            key = row["image_path"]
            if key not in pair_index_by_path:
                pair_index_by_path[key] = len(pairs)
                pairs.append(ImageMaskPair(Path(row["image_path"]), Path(row["mask_path"]), row["label"]))
            explicit_index.append((pair_index_by_path[key], int(row["row_offset"]), int(row["col_offset"])))
        return pairs, explicit_index

    pairs = [ImageMaskPair(Path(row["image_path"]), Path(row["mask_path"]), row["label"]) for row in rows]
    return pairs, None


def build_normalize_fn(config: dict):
    """Returns (callable, description). mode: "fixed_range" (default) takes
    one (lo, hi) range per channel -- a single range broadcasts to all
    channels, which is what reproduces normalize_db_fixed's original shared
    behavior exactly when set to [-40, 10] (see configs/baseline.yaml).
    Amendment 2 / docs/metric_audit.md's Gate C finding is why per-band
    ranges exist as a first-class option here rather than a single global
    constant."""
    norm_cfg = config.get("normalization", {}) or {}
    mode = norm_cfg.get("mode", "fixed_range")
    if mode == "fixed_range":
        ranges = [tuple(r) for r in norm_cfg.get("fixed_range", [[-40.0, 10.0]])]
        return partial(normalize_db_per_channel, ranges=ranges), f"fixed_range {ranges}"
    if mode == "percentile":
        raise NotImplementedError("normalization.mode: percentile is in the schema but not implemented yet -- "
                                   "use mode: fixed_range for now (see Amendment 2).")
    raise ValueError(f"unknown normalization.mode: {mode!r}")


def build_loss(config: dict):
    loss_name = config.get("loss", "dicebce")
    params = config.get("loss_parameters", {}) or {}
    if loss_name == "dicebce":
        return DiceBCELoss(pos_weight=config.get("pos_weight", 1.0),
                            dice_weight=params.get("dice_weight", 1.0),
                            bce_weight=params.get("bce_weight", 1.0))
    if loss_name == "tversky":
        return TverskyLoss(alpha=params.get("alpha", 0.3), beta=params.get("beta", 0.7))
    raise NotImplementedError(f"loss {loss_name!r} is not implemented in src/detection/losses.py -- "
                               f"supported: 'dicebce', 'tversky'")


def write_run_manifest(config: dict, config_path: Path, output_dir: Path, device: torch.device,
                        git_sha: str, is_dirty: bool) -> None:
    manifest = {
        "git_commit_sha": git_sha,
        "git_tree_dirty": is_dirty,
        "config_path": str(config_path),
        "resolved_config": config,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "run_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())

    git_sha, is_dirty = git_state()
    allow_dirty = bool(config.get("allow_dirty", False))
    if is_dirty and not allow_dirty:
        print("ERROR: working tree is dirty and config does not set allow_dirty: true.")
        print("A result trained from a dirty tree can't be tied to the exact code that produced it. "
              "Commit your changes, or set allow_dirty: true in the config if you deliberately mean to "
              "train from uncommitted code (e.g. quick local iteration).")
        sys.exit(1)

    if config.get("optimizer", "adam") != "adam":
        print(f"ERROR: optimizer {config.get('optimizer')!r} requested, but only 'adam' is implemented "
              f"(src/detection/train.py hardcodes torch.optim.Adam) -- see this file's module docstring.")
        sys.exit(1)

    seed = config.get("seed")
    set_seed(seed, bool(config.get("deterministic", False)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    output_dir = REPO_ROOT / config["output_dir"]
    write_run_manifest(config, args.config, output_dir, device, git_sha, is_dirty)

    dataset_cfg = config.get("dataset", {}) or {}
    train_pairs, train_explicit_index = load_manifest(REPO_ROOT / dataset_cfg.get("train_manifest", "data/processed/train_manifest.csv"))
    val_pairs, val_explicit_index = load_manifest(REPO_ROOT / dataset_cfg.get("val_manifest", "data/processed/val_manifest.csv"))
    if train_explicit_index is not None:
        print(f"train_manifest is a tile-level subset ({len(train_pairs)} unique source images, "
              f"{len(train_explicit_index)} selected tiles) -- see scripts/build_ablation_manifest.py")

    channels = tuple(config.get("channels", [1]))
    normalize_fn, norm_desc = build_normalize_fn(config)
    tile_size = config.get("tile_size", 512)
    augment_enabled = bool((config.get("augmentation") or {}).get("enabled", True))

    # return_nodata_mask=True unconditionally: exact-0.0-dB nodata pixels
    # (~1.8% of pixels, see docs/metric_audit.md Finding #12) were being
    # trained on and scored as if they were real ocean/oil signal, in both
    # loss and metrics, with no config knob to turn it off -- this is a
    # correctness fix for every future run through this entrypoint, not an
    # experimental option (see LOG.md's nodata-masking entry).
    train_dataset = ZenodoTileDataset(train_pairs, tile_size=tile_size, augment=augment_enabled,
                                       seed=seed or 0, channels=channels, normalize_fn=normalize_fn,
                                       return_nodata_mask=True, explicit_index=train_explicit_index)
    val_dataset = ZenodoTileDataset(val_pairs, tile_size=tile_size, augment=False,
                                     channels=channels, normalize_fn=normalize_fn,
                                     return_nodata_mask=True, explicit_index=val_explicit_index)
    print(f"train tiles: {len(train_dataset)}, val tiles: {len(val_dataset)}, "
          f"channels={channels}, normalization={norm_desc}")

    sampling_strategy = config.get("sampling_strategy", "none")
    sampler = None
    if sampling_strategy == "oil_aware":
        weights = compute_oil_tile_weights(train_dataset, target_oil_fraction=config.get("oil_tile_fraction", 0.4))
        sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(train_dataset), replacement=True)
    elif sampling_strategy not in (None, "none"):
        raise ValueError(f"unknown sampling_strategy: {sampling_strategy!r}")

    loss_fn = build_loss(config)
    model = build_model(in_channels=config.get("in_channels", 1))

    scheduler_cfg = config.get("scheduler", "none")
    use_lr_scheduler = scheduler_cfg not in (None, "none")
    sched_params = config.get("scheduler_parameters", {}) or {}

    resume_from = config.get("resume_from")
    if resume_from:
        ckpt = torch.load(REPO_ROOT / resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"initialized model weights from {resume_from} "
              f"(a warm start, NOT a training-state resume -- an interrupted run of THIS experiment "
              f"still resumes from output_dir/epochs/ automatically if present)")

    result = train(
        model, train_dataset, loss_fn, device,
        epochs=config.get("epochs", 60),
        batch_size=config.get("batch_size", 16),
        grad_accum_steps=config.get("grad_accum_steps", 1),
        lr=config.get("learning_rate", 1e-3),
        weight_decay=config.get("weight_decay", 0.0),
        val_dataset=val_dataset,
        best_checkpoint_path=output_dir / "best.pt",
        checkpoint_path=output_dir / "final.pt",
        num_workers=config.get("num_workers", 6),
        sampler=sampler,
        use_lr_scheduler=use_lr_scheduler,
        lr_monitor=sched_params.get("monitor", "val_loss"),
        lr_patience=sched_params.get("patience", 3),
        lr_factor=sched_params.get("factor", 0.5),
        lr_min=sched_params.get("min_lr", 1e-6),
        save_every_epoch_dir=output_dir / "epochs",
        keep_last_n=config.get("keep_last_n", 3),
        metrics_jsonl_path=output_dir / "metrics.jsonl",
        save_rng_state=True,
        resume=True,
        amp=config.get("amp"),  # None (key absent) = exact old hardcoded "AMP iff cuda" behavior
        channels_last=bool(config.get("channels_last", False)),
    )

    print(f"\nbest val_dice={result.best_val_dice:.4f} at epoch {result.best_epoch} "
          f"(raw val_dice -- see docs/metric_audit.md before trusting this number; "
          f"run scripts/evaluate.py + scripts/sweep_threshold.py against {output_dir / 'best.pt'} for the real metric)")


if __name__ == "__main__":
    main()
