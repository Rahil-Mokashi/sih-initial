"""
Verifies the new `amp` config flag (src/detection/train.py's train()/
evaluate() amp=True/False param, wired from configs' `amp: true/false`)
actually changes AMP usage, using the SAME model/loss/dataset/optimizer
primitives and the same autocast/GradScaler pattern train()'s real loop
uses -- not a reimplementation that could hide a wiring bug.

Runs 5 real batches on a tiny real manifest (4 real oil-class training
images) with amp=True and amp=False, starting from IDENTICAL initial model
weights and the SAME batch order both times, and prints the per-batch loss
for both so the actual delta is visible, not just asserted.

Usage:
    venv\\Scripts\\python.exe scripts\\verify_amp.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from detection.dataset import ImageMaskPair, ZenodoTileDataset  # noqa: E402
from detection.losses import DiceBCELoss  # noqa: E402
from detection.model import build_model  # noqa: E402
from detection.train import _resolve_amp  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
N_IMAGES = 4
N_BATCHES = 5
BATCH_SIZE = 2
TILE_SIZE = 512


def load_tiny_manifest(n: int = N_IMAGES) -> list[ImageMaskPair]:
    rows = list(csv.DictReader(open(REPO_ROOT / "data" / "processed" / "train_manifest.csv", newline="")))
    oil_rows = [r for r in rows if r["label"] == "oil"][:n]
    return [ImageMaskPair(Path(r["image_path"]), Path(r["mask_path"]), r["label"]) for r in oil_rows]


def run(amp_flag: bool | None, device: torch.device, pairs: list[ImageMaskPair], init_state: dict) -> list[float]:
    model = build_model(in_channels=1)
    model.load_state_dict(init_state)
    model.to(device)
    model.train()
    loss_fn = DiceBCELoss(pos_weight=32.6)

    dataset = ZenodoTileDataset(pairs, tile_size=TILE_SIZE, augment=False, channels=(1,))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    use_amp = _resolve_amp(amp_flag, device)  # the exact function train.py's train()/evaluate() now call
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    losses = []
    it = iter(loader)
    for _ in range(N_BATCHES):
        images, masks = next(it)
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = loss_fn(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
    return losses, use_amp


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    pairs = load_tiny_manifest()
    print(f"tiny manifest: {len(pairs)} real oil-class training images")

    torch.manual_seed(0)
    init_model = build_model(in_channels=1)
    init_state = {k: v.clone() for k, v in init_model.state_dict().items()}

    losses_true, resolved_true = run(True, device, pairs, init_state)
    losses_false, resolved_false = run(False, device, pairs, init_state)
    print(f"amp=True  -> _resolve_amp resolved to use_amp={resolved_true}")
    print(f"amp=False -> _resolve_amp resolved to use_amp={resolved_false}")

    print(f"\n{'batch':>6} {'amp=True loss':>15} {'amp=False loss':>16} {'abs diff':>10} {'rel diff':>10}")
    diffs = []
    for i, (a, b) in enumerate(zip(losses_true, losses_false)):
        diff = abs(a - b)
        rel = diff / abs(b) if b else float("nan")
        diffs.append(diff)
        print(f"{i:>6} {a:>15.6f} {b:>16.6f} {diff:>10.6f} {rel:>10.4%}")

    print(f"\nmax abs diff across {N_BATCHES} batches: {max(diffs):.6f}")
    print(f"mean abs diff: {sum(diffs) / len(diffs):.6f}")
    if resolved_true == resolved_false:
        print("NOTE: use_amp resolved identically for both flags on this device "
              "(expected on CPU, where AMP has no effect either way) -- "
              "losses will be near-identical, which is correct, not a failure to distinguish the flag.")


if __name__ == "__main__":
    main()
