"""
train.py  —  RT-Gesture3D
==========================
Full training pipeline for GestureCNN3D / GestureST_CNN.

Usage
-----
    python src/training/train.py                         # default (st_cnn)
    python src/training/train.py --arch cnn3d --epochs 40
    python src/training/train.py --arch st_cnn --epochs 50 --batch 8 --lr 1e-3

Output
------
    models/best_<arch>.pt           ← best val-accuracy checkpoint
    models/last_<arch>.pt           ← last epoch checkpoint
    logs/train_<arch>_<time>.csv    ← per-epoch metrics
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.training.dataset import make_loaders, GESTURE_CLASSES
from src.training.models  import build_model


# ── helpers ──────────────────────────────────────────────────────────────────

def _device() -> torch.device:
    if torch.cuda.is_available():
        d = torch.device("cuda")
    elif torch.backends.mps.is_available():
        d = torch.device("mps")
    else:
        d = torch.device("cpu")
    print(f"🔧 Device: {d}")
    return d


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean())


# ── one epoch ────────────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    train: bool,
) -> tuple[float, float]:
    """Returns (mean_loss, mean_accuracy)."""
    model.train(train)
    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    ctx = torch.enable_grad if train else torch.no_grad

    with ctx():
        for clips, labels in loader:
            clips  = clips.to(device)
            labels = labels.to(device)

            logits = model(clips)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            total_acc  += _accuracy(logits, labels)
            n_batches  += 1

    if n_batches == 0:
        return 0.0, 0.0
    return total_loss / n_batches, total_acc / n_batches


# ── training loop ─────────────────────────────────────────────────────────────

def train(args):
    device = _device()
    data_dir = ROOT / "data" / "raw"

    if not data_dir.exists() or not any(data_dir.iterdir()):
        print("⚠️  No data found at", data_dir)
        print("   Run:  python src/capture/collect_data.py   first.")
        return

    print("📂 Loading dataset …")
    train_loader, val_loader, test_loader = make_loaders(
        data_dir,
        batch_size=args.batch,
        num_workers=args.workers,
    )

    print(f"🏗  Building model: {args.arch}")
    model = build_model(arch=args.arch, num_classes=len(GESTURE_CLASSES), device=str(device))

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"   Parameters: {n_params:.2f}M")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── logging setup ────────────────────────────────────────────────────────
    logs_dir = ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    run_tag  = time.strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"train_{args.arch}_{run_tag}.csv"
    best_path = models_dir / f"best_{args.arch}.pt"
    last_path = models_dir / f"last_{args.arch}.pt"

    log_file = open(log_path, "w", newline="")
    writer   = csv.writer(log_file)
    writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    best_val_acc = 0.0
    print(f"\n🚀 Training for {args.epochs} epochs …")
    print(f"{'Epoch':>6} {'TrLoss':>8} {'TrAcc':>7} {'VaLoss':>8} {'VaAcc':>7}  LR")
    print("─" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion, None,      device, train=False)

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        elapsed = time.time() - t0
        print(f"{epoch:>6}  {tr_loss:>8.4f}  {tr_acc:>6.2%}  "
              f"{va_loss:>8.4f}  {va_acc:>6.2%}  {lr:.2e}  ({elapsed:.0f}s)")

        writer.writerow([epoch, f"{tr_loss:.4f}", f"{tr_acc:.4f}",
                         f"{va_loss:.4f}", f"{va_acc:.4f}", f"{lr:.2e}"])
        log_file.flush()

        # save best
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), best_path)
            print(f"         💾 New best  val_acc={va_acc:.2%}  → {best_path.name}")

    # save last
    torch.save(model.state_dict(), last_path)
    log_file.close()

    # ── test evaluation ───────────────────────────────────────────────────────
    print("\n📊 Test evaluation (best checkpoint) …")
    model.load_state_dict(torch.load(best_path, map_location=device))
    te_loss, te_acc = run_epoch(model, test_loader, criterion, None, device, train=False)
    print(f"   Test Loss : {te_loss:.4f}")
    print(f"   Test Acc  : {te_acc:.2%}")
    print(f"\n✅ Done.  Best val acc: {best_val_acc:.2%}")
    print(f"   Model    : {best_path}")
    print(f"   Log      : {log_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="RT-Gesture3D trainer")
    p.add_argument("--arch",    default="st_cnn", choices=["cnn3d", "st_cnn"])
    p.add_argument("--epochs",  type=int,   default=40)
    p.add_argument("--batch",   type=int,   default=8)
    p.add_argument("--lr",      type=float, default=1e-3)
    p.add_argument("--workers", type=int,   default=0)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
