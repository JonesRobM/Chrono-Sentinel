#!/usr/bin/env python3
"""
Evaluation script with Monte Carlo Dropout uncertainty quantification.

This script loads a trained model and performs comprehensive evaluation
including standard metrics and uncertainty analysis.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --model-path outputs/best_model.pt --mc-samples 50
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)

from threatsim.data import get_dataloaders
from threatsim.models import TimeSeriesTransformer, mc_dropout_predict
from threatsim.utils import get_device, load_model, format_metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained model with MC Dropout uncertainty"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="outputs/best_model.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=30,
        help="Number of MC Dropout forward passes",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save outputs",
    )
    return parser.parse_args()


def collect_predictions_with_uncertainty(
    model: TimeSeriesTransformer,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    n_samples: int = 30,
) -> tuple:
    """
    Collects predictions with MC Dropout uncertainty for entire dataset.

    Args:
        model: Trained model.
        dataloader: DataLoader to evaluate.
        device: Device to run on.
        n_samples: Number of MC Dropout samples.

    Returns:
        Tuple of (all_labels, all_means, all_stds) as numpy arrays.
    """
    all_labels = []
    all_means = []
    all_stds = []

    for batch in dataloader:
        windows, labels = batch
        windows = windows.to(device)

        mean, std = mc_dropout_predict(model, windows, n_samples=n_samples)

        all_labels.append(labels.numpy())
        all_means.append(mean.cpu().numpy())
        all_stds.append(std.cpu().numpy())

    return (
        np.concatenate(all_labels),
        np.concatenate(all_means),
        np.concatenate(all_stds),
    )


def compute_classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Computes standard classification metrics.

    Args:
        labels: Ground truth binary labels.
        predictions: Predicted probabilities.
        threshold: Classification threshold.

    Returns:
        Dictionary of metrics.
    """
    binary_preds = (predictions >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(labels, binary_preds),
        "precision": precision_score(labels, binary_preds, zero_division=0),
        "recall": recall_score(labels, binary_preds, zero_division=0),
        "f1": f1_score(labels, binary_preds, zero_division=0),
    }

    # ROC-AUC only if we have both classes
    if len(np.unique(labels)) > 1:
        metrics["roc_auc"] = roc_auc_score(labels, predictions)
    else:
        metrics["roc_auc"] = float("nan")

    return metrics


def compute_calibration_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Computes calibration and uncertainty quality metrics.

    Args:
        labels: Ground truth binary labels.
        predictions: Mean predicted probabilities.
        uncertainties: Standard deviation of predictions.
        n_bins: Number of bins for calibration.

    Returns:
        Dictionary of calibration metrics.
    """
    # Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = labels[in_bin].mean()
            avg_confidence_in_bin = predictions[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    # Uncertainty vs Error correlation
    # High uncertainty should correlate with errors
    binary_preds = (predictions >= 0.5).astype(int)
    errors = (binary_preds != labels).astype(float)

    if uncertainties.std() > 0 and errors.std() > 0:
        uncertainty_error_corr = np.corrcoef(uncertainties, errors)[0, 1]
    else:
        uncertainty_error_corr = 0.0

    # Average uncertainty for correct vs incorrect predictions
    correct_mask = binary_preds == labels
    avg_uncertainty_correct = uncertainties[correct_mask].mean() if correct_mask.sum() > 0 else 0
    avg_uncertainty_incorrect = uncertainties[~correct_mask].mean() if (~correct_mask).sum() > 0 else 0

    # Prediction interval coverage (approximate 95% CI using 2*std)
    lower_bound = predictions - 2 * uncertainties
    upper_bound = predictions + 2 * uncertainties
    # For binary classification, check if true probability (0 or 1) is within interval
    coverage = ((labels >= lower_bound) & (labels <= upper_bound)).mean()

    return {
        "expected_calibration_error": ece,
        "uncertainty_error_correlation": uncertainty_error_corr,
        "avg_uncertainty_correct": avg_uncertainty_correct,
        "avg_uncertainty_incorrect": avg_uncertainty_incorrect,
        "prediction_interval_coverage": coverage,
        "avg_uncertainty": uncertainties.mean(),
    }


def plot_roc_curve(
    labels: np.ndarray,
    predictions: np.ndarray,
    save_path: str,
) -> None:
    """Plots and saves ROC curve."""
    if len(np.unique(labels)) < 2:
        print("Warning: Cannot plot ROC curve with single class")
        return

    fpr, tpr, _ = roc_curve(labels, predictions)
    roc_auc = roc_auc_score(labels, predictions)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_precision_recall_curve(
    labels: np.ndarray,
    predictions: np.ndarray,
    save_path: str,
) -> None:
    """Plots and saves Precision-Recall curve."""
    if len(np.unique(labels)) < 2:
        print("Warning: Cannot plot PR curve with single class")
        return

    precision, recall, _ = precision_recall_curve(labels, predictions)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(recall, precision, "b-", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_calibration_curve(
    labels: np.ndarray,
    predictions: np.ndarray,
    save_path: str,
    n_bins: int = 10,
) -> None:
    """Plots reliability diagram (calibration curve)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
        if in_bin.sum() > 0:
            bin_accuracies.append(labels[in_bin].mean())
            bin_confidences.append(predictions[in_bin].mean())
            bin_counts.append(in_bin.sum())
        else:
            bin_accuracies.append(0)
            bin_confidences.append((bin_lower + bin_upper) / 2)
            bin_counts.append(0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={"height_ratios": [3, 1]})

    # Reliability diagram
    bin_centers = (bin_lowers + bin_uppers) / 2
    ax1.bar(bin_centers, bin_accuracies, width=0.08, alpha=0.7, label="Actual")
    ax1.plot([0, 1], [0, 1], "k--", linewidth=2, label="Perfectly Calibrated")
    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("Fraction of Positives")
    ax1.set_title("Calibration Plot (Reliability Diagram)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)

    # Histogram of predictions
    ax2.bar(bin_centers, bin_counts, width=0.08, alpha=0.7, color="gray")
    ax2.set_xlabel("Mean Predicted Probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Prediction Distribution")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_uncertainty_histogram(
    labels: np.ndarray,
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    save_path: str,
) -> None:
    """Plots histogram of uncertainties for correct vs incorrect predictions."""
    binary_preds = (predictions >= 0.5).astype(int)
    correct_mask = binary_preds == labels

    fig, ax = plt.subplots(figsize=(10, 6))

    if correct_mask.sum() > 0:
        ax.hist(
            uncertainties[correct_mask],
            bins=30,
            alpha=0.6,
            label=f"Correct (n={correct_mask.sum()})",
            color="green",
        )
    if (~correct_mask).sum() > 0:
        ax.hist(
            uncertainties[~correct_mask],
            bins=30,
            alpha=0.6,
            label=f"Incorrect (n={(~correct_mask).sum()})",
            color="red",
        )

    ax.set_xlabel("Uncertainty (Std Dev)")
    ax.set_ylabel("Count")
    ax.set_title("Uncertainty Distribution: Correct vs Incorrect Predictions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_predictions_timeline(
    labels: np.ndarray,
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    save_path: str,
    max_points: int = 500,
) -> None:
    """Plots predictions with uncertainty bands over time."""
    # Subsample if too many points
    if len(labels) > max_points:
        indices = np.linspace(0, len(labels) - 1, max_points, dtype=int)
        labels = labels[indices]
        predictions = predictions[indices]
        uncertainties = uncertainties[indices]

    x = np.arange(len(labels))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Predictions with uncertainty
    ax1.fill_between(
        x,
        np.clip(predictions - 2 * uncertainties, 0, 1),
        np.clip(predictions + 2 * uncertainties, 0, 1),
        alpha=0.3,
        color="blue",
        label="95% CI",
    )
    ax1.plot(x, predictions, "b-", linewidth=1, label="Prediction")
    ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Threshold")
    ax1.set_ylabel("Anomaly Probability")
    ax1.set_title("Predictions with MC Dropout Uncertainty")
    ax1.legend(loc="upper right")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)

    # True labels
    ax2.fill_between(x, 0, labels, alpha=0.5, color="red", label="True Anomaly")
    ax2.set_xlabel("Sample Index")
    ax2.set_ylabel("True Label")
    ax2.legend(loc="upper right")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    """Main evaluation function."""
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model checkpoint
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    config = checkpoint.get("config", {})

    # Recreate model with saved config
    model = TimeSeriesTransformer(
        input_dim=1,
        d_model=config.get("d_model", 64),
        nhead=4,
        num_encoder_layers=config.get("num_layers", 2),
        dim_feedforward=config.get("d_model", 64) * 2,
        dropout=config.get("dropout", 0.2),
        max_seq_len=config.get("window_size", 50) + 10,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"  Model config: {config}")

    # Load test data
    print("Loading test data...")
    _, _, test_loader, _ = get_dataloaders(
        window_size=config.get("window_size", 50),
        batch_size=32,
    )
    print(f"  Test batches: {len(test_loader)}")

    # Collect predictions with MC Dropout
    print(f"Running MC Dropout evaluation with {args.mc_samples} samples...")
    labels, predictions, uncertainties = collect_predictions_with_uncertainty(
        model, test_loader, device, n_samples=args.mc_samples
    )
    print(f"  Total test samples: {len(labels)}")
    print(f"  Anomaly samples: {labels.sum()} ({100 * labels.mean():.1f}%)")

    # Compute metrics
    print("\nComputing metrics...")
    class_metrics = compute_classification_metrics(labels, predictions, args.threshold)
    calib_metrics = compute_calibration_metrics(labels, predictions, uncertainties)

    all_metrics = {**class_metrics, **calib_metrics}

    print("\n=== Classification Metrics ===")
    print(format_metrics(class_metrics))

    print("\n=== Uncertainty & Calibration Metrics ===")
    print(format_metrics(calib_metrics))

    # Generate plots
    print("\nGenerating visualisations...")

    plot_roc_curve(labels, predictions, str(output_dir / "roc_curve.png"))
    print("  Saved: roc_curve.png")

    plot_precision_recall_curve(labels, predictions, str(output_dir / "pr_curve.png"))
    print("  Saved: pr_curve.png")

    plot_calibration_curve(labels, predictions, str(output_dir / "calibration_curve.png"))
    print("  Saved: calibration_curve.png")

    plot_uncertainty_histogram(
        labels, predictions, uncertainties, str(output_dir / "uncertainty_histogram.png")
    )
    print("  Saved: uncertainty_histogram.png")

    plot_predictions_timeline(
        labels, predictions, uncertainties, str(output_dir / "predictions_timeline.png")
    )
    print("  Saved: predictions_timeline.png")

    # Save metrics to JSON
    metrics_path = output_dir / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
