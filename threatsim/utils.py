"""
Utility functions for the Chrono-Sentinel project.

This module provides helper functions for reproducibility, model persistence,
and visualisation.
"""

import random
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import torch
import matplotlib.pyplot as plt


def set_seed(seed: int = 42) -> None:
    """
    Sets random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Returns the best available device (CUDA, MPS, or CPU).

    Returns:
        torch.device for computation.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_model(
    model: torch.nn.Module,
    path: str,
    config: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    """
    Saves model checkpoint with optional configuration and metrics.

    Args:
        model: PyTorch model to save.
        path: Path to save the checkpoint.
        config: Optional dictionary of model configuration.
        metrics: Optional dictionary of evaluation metrics.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
    }
    if config is not None:
        checkpoint["config"] = config
    if metrics is not None:
        checkpoint["metrics"] = metrics

    # Create directory if it doesn't exist
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_model(
    model: torch.nn.Module,
    path: str,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Loads model checkpoint and returns metadata.

    Args:
        model: PyTorch model to load weights into.
        path: Path to the checkpoint file.
        device: Device to load the model to.

    Returns:
        Dictionary containing config and metrics if they were saved.
    """
    if device is None:
        device = get_device()

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return {
        "config": checkpoint.get("config"),
        "metrics": checkpoint.get("metrics"),
    }


class EarlyStopping:
    """
    Early stopping handler to prevent overfitting.

    Monitors a validation metric and stops training if no improvement
    is seen for a specified number of epochs.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping.
            min_delta: Minimum change to qualify as an improvement.
            mode: 'min' for loss (lower is better), 'max' for accuracy.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation score.

        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def plot_training_history(
    train_losses: list,
    val_losses: list,
    save_path: Optional[str] = None,
) -> None:
    """
    Plots training and validation loss curves.

    Args:
        train_losses: List of training losses per epoch.
        val_losses: List of validation losses per epoch.
        save_path: Optional path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "b-", label="Training Loss")
    ax.plot(epochs, val_losses, "r-", label="Validation Loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training History")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)

    plt.close()


def plot_predictions_with_uncertainty(
    timestamps: np.ndarray,
    true_labels: np.ndarray,
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Predictions with Uncertainty",
) -> None:
    """
    Plots predictions with uncertainty bands overlaid on true labels.

    Args:
        timestamps: Array of timestamps or indices.
        true_labels: Ground truth binary labels.
        predictions: Mean predicted probabilities.
        uncertainties: Standard deviation of predictions.
        save_path: Optional path to save the figure.
        title: Plot title.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Upper plot: predictions with uncertainty
    ax1.fill_between(
        timestamps,
        predictions - 2 * uncertainties,
        predictions + 2 * uncertainties,
        alpha=0.3,
        color="blue",
        label="95% CI",
    )
    ax1.plot(timestamps, predictions, "b-", linewidth=1.5, label="Prediction")
    ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Threshold")
    ax1.set_ylabel("Anomaly Probability")
    ax1.set_title(title)
    ax1.legend(loc="upper right")
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True, alpha=0.3)

    # Lower plot: true labels
    ax2.fill_between(timestamps, 0, true_labels, alpha=0.5, color="red", label="True Anomaly")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("True Label")
    ax2.legend(loc="upper right")
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)

    plt.close()


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """
    Formats a dictionary of metrics as a readable string.

    Args:
        metrics: Dictionary of metric names to values.
        precision: Number of decimal places.

    Returns:
        Formatted string.
    """
    lines = []
    for name, value in metrics.items():
        lines.append(f"  {name}: {value:.{precision}f}")
    return "\n".join(lines)
