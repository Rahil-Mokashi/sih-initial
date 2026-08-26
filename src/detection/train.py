"""
Training loop for the oil-spill segmentation model.

Uses torch.cuda.amp mixed precision from the start (VRAM is the binding
constraint on the 6GB RTX 4050 -- see DECISIONS.md), with optional gradient
accumulation to simulate a larger effective batch size on small physical
batches. Tracks loss, wall-clock time, and peak VRAM per epoch so future
sessions know exactly how much headroom exists before scaling up further.

Extended (real training run, see DECISIONS.md "Train/val/test
methodology") with an optional validation pass: after each epoch, if
`val_dataset` is given, evaluates val loss + Dice score (no grad, eval
mode) and saves a *separate* "best" checkpoint whenever val Dice improves
-- distinct from the final-epoch checkpoint, since the last epoch isn't
necessarily the best one.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from detection.losses import dice_loss


@dataclass
class EpochStats:
    epoch: int
    loss: float
    seconds: float
    peak_vram_mb: float | None
    val_loss: float | None = None
    val_dice: float | None = None


@dataclass
class TrainResult:
    history: list[EpochStats] = field(default_factory=list)
    best_val_dice: float | None = None
    best_epoch: int | None = None


@torch.no_grad()
def evaluate(model: torch.nn.Module, dataset, loss_fn: torch.nn.Module, device: torch.device, batch_size: int = 8, num_workers: int = 0) -> tuple[float, float]:
    """Returns (avg_loss, avg_dice_score) over the dataset. Dice score = 1 - dice_loss, so higher is better."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    total_loss, total_dice, n_batches = 0.0, 0.0, 0
    use_amp = device.type == "cuda"

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = loss_fn(logits, masks)
            dice_score = 1 - dice_loss(logits, masks)
        total_loss += loss.item()
        total_dice += dice_score.item()
        n_batches += 1

    model.train()
    return total_loss / max(n_batches, 1), total_dice / max(n_batches, 1)


def train(
    model: torch.nn.Module,
    dataset,
    loss_fn: torch.nn.Module,
    device: torch.device,
    epochs: int = 5,
    batch_size: int = 2,
    grad_accum_steps: int = 1,
    lr: float = 1e-3,
    checkpoint_path: str | Path | None = None,
    val_dataset=None,
    best_checkpoint_path: str | Path | None = None,
    num_workers: int = 0,
    latest_checkpoint_path: str | Path | None = None,
    resume: bool = True,
) -> TrainResult:
    """
    latest_checkpoint_path, if given, is overwritten after every epoch
    (model + optimizer + scaler state, epoch number, best-so-far
    tracking, full history) -- a resume point distinct from
    best_checkpoint_path (only written when val_dice improves) and
    checkpoint_path (only written once, at the very end). If `resume` is
    true and that file already exists when train() is called, training
    picks up from the epoch after the saved one instead of starting over
    -- added so a long run (multi-GB VRAM, hours per epoch) surviving a
    laptop sleep/hibernate/crash costs at most one epoch, not everything
    already done. See LOG.md for why this was added.
    """
    model.to(device)
    model.train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    result = TrainResult()
    start_epoch = 1

    if resume and latest_checkpoint_path is not None and Path(latest_checkpoint_path).exists():
        ckpt = torch.load(latest_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if use_amp and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        result.best_val_dice = ckpt.get("best_val_dice")
        result.best_epoch = ckpt.get("best_epoch")
        result.history = [EpochStats(**h) for h in ckpt.get("history", [])]
        print(f"Resumed from {latest_checkpoint_path}: continuing at epoch {start_epoch}/{epochs} "
              f"(best_val_dice={result.best_val_dice} at epoch {result.best_epoch})")

    for epoch in range(start_epoch, epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()

        epoch_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for step, (images, masks) in enumerate(loader):
            images, masks = images.to(device), masks.to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = loss_fn(logits, masks) / grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * grad_accum_steps
            n_batches += 1

        elapsed = time.perf_counter() - start
        avg_loss = epoch_loss / max(n_batches, 1)
        peak_vram_mb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else None
        )

        val_loss, val_dice = (None, None)
        if val_dataset is not None:
            val_loss, val_dice = evaluate(model, val_dataset, loss_fn, device, batch_size=batch_size, num_workers=num_workers)

        stats = EpochStats(epoch=epoch, loss=avg_loss, seconds=elapsed, peak_vram_mb=peak_vram_mb,
                            val_loss=val_loss, val_dice=val_dice)
        result.history.append(stats)
        vram_str = f"{peak_vram_mb:.0f}MB" if peak_vram_mb is not None else "n/a"
        val_str = f"  val_loss={val_loss:.4f}  val_dice={val_dice:.4f}" if val_dataset is not None else ""
        print(f"epoch {epoch}/{epochs}  loss={avg_loss:.4f}  time={elapsed:.1f}s  peak_vram={vram_str}{val_str}")

        if val_dataset is not None and (result.best_val_dice is None or val_dice > result.best_val_dice):
            result.best_val_dice = val_dice
            result.best_epoch = epoch
            if best_checkpoint_path is not None:
                best_checkpoint_path = Path(best_checkpoint_path)
                best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                }, best_checkpoint_path)
                print(f"  new best val_dice={val_dice:.4f} at epoch {epoch} -> saved to {best_checkpoint_path}")

        if latest_checkpoint_path is not None:
            latest_checkpoint_path = Path(latest_checkpoint_path)
            latest_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if use_amp else None,
                "epoch": epoch,
                "best_val_dice": result.best_val_dice,
                "best_epoch": result.best_epoch,
                "history": [asdict(h) for h in result.history],
            }, latest_checkpoint_path)

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "final_loss": result.history[-1].loss if result.history else None,
        }, checkpoint_path)
        print(f"saved final-epoch checkpoint to {checkpoint_path}")

    return result


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
