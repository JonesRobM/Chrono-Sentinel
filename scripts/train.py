#!/usr/bin/env python3
"""
Training script for the time-series anomaly detection transformer.

This script provides a complete training pipeline with early stopping,
model checkpointing, and training visualisation.

Usage:
    python scripts/train.py
    python scripts/train.py --epochs 50 --batch-size 64
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from tqdm import tqdm

from threatsim.data import get_dataloaders
from threatsim.models import create_model
from threatsim.utils import (
    set_seed,
    get_device,
    save_model,
    EarlyStopping,
    plot_training_history,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train anomaly detection transformer on NAB data"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument(
        "--window-size", type=int, default=50, help="Sliding window size"
    )
    parser.add_argument(
        "--d-model", type=int, default=64, help="Transformer model dimension"
    )
    parser.add_argument(
        "--num-layers", type=int, default=2, help="Number of transformer layers"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.2, help="Dropout probability"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save outputs",
    )
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Trains the model for one epoch.

    Args:
        model: The neural network model.
        train_loader: DataLoader for training data.
        criterion: Loss function.
        optimiser: Optimiser for parameter updates.
        device: Device to train on.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in train_loader:
        windows, labels = batch
        windows = windows.to(device)
        labels = labels.to(device)

        optimiser.zero_grad()
        predictions = model(windows)
        loss = criterion(predictions, labels)
        loss.backward()
        optimiser.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Validates the model on the validation set.

    Args:
        model: The neural network model.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to run on.

    Returns:
        Average validation loss.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            windows, labels = batch
            windows = windows.to(device)
            labels = labels.to(device)

            predictions = model(windows)
            loss = criterion(predictions, labels)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def main():
    """Main training loop."""
    args = parse_args()

    # Set up reproducibility
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading NAB data...")
    train_loader, val_loader, test_loader, class_weight = get_dataloaders(
        window_size=args.window_size,
        batch_size=args.batch_size,
    )
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Class weight (anomaly): {class_weight.item():.2f}")

    # Create model
    model = create_model(
        window_size=args.window_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Loss function with class weighting
    criterion = nn.BCELoss(reduction="none")

    def weighted_bce_loss(predictions, labels):
        """Binary cross-entropy with class weighting for imbalanced data."""
        loss = criterion(predictions, labels)
        weights = torch.where(labels == 1, class_weight.to(device), torch.ones_like(labels))
        return (loss * weights).mean()

    # Optimiser
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=5, verbose=True
    )

    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, mode="min")

    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    # Training loop
    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, weighted_bce_loss, optimiser, device
        )
        train_losses.append(train_loss)

        # Validate
        val_loss = validate(model, val_loader, weighted_bce_loss, device)
        val_losses.append(val_loss)

        # Update learning rate
        scheduler.step(val_loss)

        # Print progress
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            config = {
                "window_size": args.window_size,
                "d_model": args.d_model,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
            }
            save_model(
                model,
                str(output_dir / "best_model.pt"),
                config=config,
                metrics={"val_loss": val_loss, "epoch": epoch},
            )

        # Check early stopping
        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    # Plot training history
    plot_training_history(
        train_losses,
        val_losses,
        save_path=str(output_dir / "training_history.png"),
    )

    # Save training history
    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "window_size": args.window_size,
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "seed": args.seed,
        },
    }
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete!")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Model saved to: {output_dir / 'best_model.pt'}")
    print(f"  Training history saved to: {output_dir / 'training_history.json'}")
    print(f"\nRun evaluation with: python scripts/evaluate.py")


if __name__ == "__main__":
    main()
