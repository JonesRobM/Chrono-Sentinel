#!/usr/bin/env python3
"""
Training script for the time-series anomaly detection transformer.

Trains on a set of NAB series and validates on a disjoint set, so the
selected checkpoint is the one that generalises to unseen series rather than
the one that best fits the training series.

Model selection is on validation **average precision**, not loss and not
ROC-AUC. With a ~3% positive rate, loss is dominated by the negative class, so
a model that collapses to the prior can post a respectable loss while being
useless. ROC-AUC is better but still rewards ranking across the vast negative
majority; average precision tracks how well the few positives are surfaced,
which is what a rare-event detector is for. A sweep over learning rate, width,
depth and weight decay picked the defaults below on validation AP.

Usage:
    python scripts/train.py
    python scripts/train.py --epochs 80 --batch-size 64
    python scripts/train.py --no-features        # sequence-only ablation
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

from threatsim.data import (
    DEFAULT_ANOMALY_WINDOW_FRAC,
    DEFAULT_TEST_SERIES,
    DEFAULT_TRAIN_SERIES,
    DEFAULT_VAL_SERIES,
    get_dataloaders,
)
from threatsim.features import get_feature_names
from threatsim.models import create_model
from threatsim.utils import (
    EarlyStopping,
    get_device,
    plot_training_history,
    save_model,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train anomaly detection transformer on NAB data"
    )
    parser.add_argument("--epochs", type=int, default=80, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--window-size", type=int, default=50, help="Sliding window size")
    parser.add_argument("--step-size", type=int, default=10, help="Step between windows")
    parser.add_argument(
        "--window-frac",
        type=float,
        default=DEFAULT_ANOMALY_WINDOW_FRAC,
        help="Fraction of each series labelled anomalous (NAB scoring uses 0.10)",
    )
    parser.add_argument("--d-model", type=int, default=64, help="Transformer model dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    parser.add_argument(
        "--patience",
        type=int,
        default=80,
        help=(
            "Early stopping patience. Defaults to the epoch budget, i.e. off. "
            "Validation AP on this data is noisy and non-monotonic: a run that "
            "peaks at epoch 3, dips, and peaks higher at epoch 30 is normal, "
            "and a tighter patience truncates the better optimum."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "auto"],
        help=(
            "Training device. Defaults to cpu: the model is small, cpu is the "
            "deployment target, and mps/cuda numerics shift results enough to "
            "make a sweep run and a training run disagree."
        ),
    )
    parser.add_argument(
        "--scheduler",
        action="store_true",
        help=(
            "Enable ReduceLROnPlateau. Off by default: validation AP is noisy "
            "here, so plateau detection halves the learning rate during dips "
            "the run would otherwise recover from. Measured on seed 42, "
            "enabling it dropped best validation AP from 0.335 to 0.227."
        ),
    )
    parser.add_argument(
        "--no-features",
        action="store_true",
        help="Ablation: train on the z-scored sequence only, without the feature branch",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs", help="Directory to save outputs"
    )
    parser.add_argument(
        "--checkpoint-name", type=str, default="best_model.pt", help="Checkpoint filename"
    )
    return parser.parse_args()


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimiser: torch.optim.Optimizer = None,
) -> tuple:
    """
    Runs one pass over a dataloader, training if an optimiser is supplied.

    Args:
        model: The neural network model.
        loader: DataLoader to iterate.
        criterion: Loss function operating on logits.
        device: Device to run on.
        optimiser: Optimiser for parameter updates, or None to evaluate.

    Returns:
        Tuple of (mean loss, labels array, predicted probability array).
    """
    training = optimiser is not None
    model.train(training)

    total_loss = 0.0
    num_batches = 0
    all_labels, all_probs = [], []

    with torch.set_grad_enabled(training):
        for batch in loader:
            windows, features, labels = batch
            windows = windows.to(device)
            features = features.to(device)
            labels = labels.to(device)

            if training:
                optimiser.zero_grad()

            logits = model(windows, features if model.feature_dim > 0 else None)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimiser.step()

            total_loss += loss.item()
            num_batches += 1
            all_labels.append(labels.detach().cpu().numpy())
            all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())

    return (
        total_loss / max(num_batches, 1),
        np.concatenate(all_labels),
        np.concatenate(all_probs),
    )


def safe_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """ROC-AUC that returns NaN rather than raising when a split is one-class."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probs))


def safe_ap(labels: np.ndarray, probs: np.ndarray) -> float:
    """Average precision that returns NaN rather than raising on a one-class split."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, probs))


def main() -> None:
    """Main training loop."""
    args = parse_args()

    set_seed(args.seed)
    device = torch.device("cpu") if args.device == "cpu" else get_device()
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading NAB data (grouped split, disjoint series per split)...")
    train_loader, val_loader, test_loader, pos_weight, scaler = get_dataloaders(
        window_size=args.window_size,
        step_size=args.step_size,
        batch_size=args.batch_size,
        window_frac=args.window_frac,
    )
    print(f"  train series: {len(DEFAULT_TRAIN_SERIES)}  windows: {len(train_loader.dataset)}")
    print(f"  val   series: {len(DEFAULT_VAL_SERIES)}  windows: {len(val_loader.dataset)}")
    print(f"  test  series: {len(DEFAULT_TEST_SERIES)}  windows: {len(test_loader.dataset)}")
    print(f"  positive-class weight: {pos_weight.item():.2f}")

    feature_dim = 0 if args.no_features else len(get_feature_names())
    model = create_model(
        window_size=args.window_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        feature_dim=feature_dim,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}  (feature branch: {'off' if feature_dim == 0 else 'on'})")

    # BCEWithLogitsLoss fuses the sigmoid into the loss for numerical
    # stability and supports pos_weight directly.
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="max", factor=0.5, patience=8
        )
        if args.scheduler
        else None
    )
    early_stopping = EarlyStopping(patience=args.patience, mode="max")

    train_losses, val_losses, val_aucs, val_aps = [], [], [], []
    best_val_ap = -float("inf")
    best_val_auc = float("nan")
    best_epoch = 0

    config = {
        "window_size": args.window_size,
        "step_size": args.step_size,
        "window_frac": args.window_frac,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "feature_dim": feature_dim,
        "train_series": DEFAULT_TRAIN_SERIES,
        "val_series": DEFAULT_VAL_SERIES,
        "test_series": DEFAULT_TEST_SERIES,
        "feature_scaler": scaler.to_dict(),
    }

    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, device, optimiser)
        val_loss, val_labels, val_probs = run_epoch(model, val_loader, criterion, device)
        val_auc = safe_auc(val_labels, val_probs)
        val_ap = safe_ap(val_labels, val_probs)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)
        val_aps.append(val_ap)
        if scheduler is not None:
            scheduler.step(val_ap)

        marker = ""
        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_val_auc = val_auc
            best_epoch = epoch
            marker = "  <- best"
            save_model(
                model,
                str(output_dir / args.checkpoint_name),
                config=config,
                metrics={
                    "val_loss": val_loss,
                    "val_auc": val_auc,
                    "val_ap": val_ap,
                    "epoch": epoch,
                },
            )

        print(
            f"Epoch {epoch:3d}/{args.epochs} | Train {train_loss:.4f} | "
            f"Val {val_loss:.4f} | AUC {val_auc:.4f} | AP {val_ap:.4f}{marker}"
        )

        if early_stopping(val_ap):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    print(
        f"\nBest validation AP {best_val_ap:.4f} (AUC {best_val_auc:.4f}) "
        f"at epoch {best_epoch}"
    )

    plot_training_history(
        train_losses, val_losses, save_path=str(output_dir / "training_history.png")
    )

    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_aucs": val_aucs,
        "val_aps": val_aps,
        "best_val_ap": best_val_ap,
        "best_val_auc": best_val_auc,
        "best_epoch": best_epoch,
        "config": {
            **{k: v for k, v in config.items() if k != "feature_scaler"},
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        },
    }
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Checkpoint: {output_dir / args.checkpoint_name}")
    print("Now run: python scripts/evaluate.py")


if __name__ == "__main__":
    main()
