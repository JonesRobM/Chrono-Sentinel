#!/usr/bin/env python3
"""
Evaluation with Monte Carlo Dropout uncertainty and classical baselines.

Evaluates the trained checkpoint on the held-out series and reports it
alongside three baselines. The baselines are the point of this script as much
as the model is: a transformer that cannot beat logistic regression on ten
statistical features is not worth serving, and reporting the model without
that comparison hides the only number that makes its result meaningful.

Metrics are chosen for a ~3% positive rate:
  * average precision is the headline; ROC-AUC is reported alongside it
  * precision/recall/F1 are reported at the threshold that maximises F1 on
    the validation split, never one tuned on test
  * uncertainty is judged on whether it separates correct from incorrect
    predictions, not on whether it merely exists. error_detection_auc is the
    number that matters: 0.5 means the interval is decorative.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --mc-samples 50
"""

import argparse
import json
import sys
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from threatsim.data import prepare_grouped_splits
from threatsim.models import create_model, mc_dropout_predict
from threatsim.utils import format_metrics, set_seed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate the trained model with MC Dropout uncertainty"
    )
    parser.add_argument("--model-path", type=str, default="outputs/best_model.pt")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Seed for the MC Dropout draws. Evaluation is stochastic: on an "
            "identical checkpoint, unseeded runs moved error_detection_auc "
            "between 0.43 and 0.56, because the F1 threshold is chosen from "
            "sampled validation scores. Seeded so published numbers reproduce."
        ),
    )
    parser.add_argument("--mc-samples", type=int, default=30)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Batch size for MC inference"
    )
    return parser.parse_args()


def load_scorer(model_path: Path):
    """Loads the checkpoint and returns (model, config)."""
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    config = payload["config"]
    model = create_model(
        window_size=config["window_size"],
        d_model=config["d_model"],
        nhead=config.get("nhead", 4),
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        feature_dim=config.get("feature_dim", 0),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, config


def mc_predict_split(model, windows, features, n_samples, batch_size):
    """
    Runs MC Dropout over a whole split in batches.

    Args:
        model: Trained model.
        windows: Normalised windows array.
        features: Scaled features array.
        n_samples: MC passes per window.
        batch_size: Windows per batch.

    Returns:
        Tuple of (mean probabilities, standard deviations).
    """
    means, stds = [], []
    use_features = model.feature_dim > 0
    for start in range(0, len(windows), batch_size):
        stop = start + batch_size
        w = torch.from_numpy(windows[start:stop]).float()
        f = torch.from_numpy(features[start:stop]).float() if use_features else None
        mean, std = mc_dropout_predict(model, w, f, n_samples=n_samples, batched=True)
        means.append(mean.numpy())
        stds.append(std.numpy())
    return np.concatenate(means), np.concatenate(stds)


def best_f1_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    """Finds the probability threshold maximising F1 on the given split."""
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    # precision_recall_curve returns one more point than thresholds.
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(precision[:-1]),
        where=(precision[:-1] + recall[:-1]) > 0,
    )
    if len(f1) == 0:
        return 0.5
    return float(thresholds[int(np.argmax(f1))])


def classification_metrics(labels, probs, threshold):
    """Standard classification metrics at a fixed threshold."""
    predicted = (probs >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(labels, probs)),
        "average_precision": float(average_precision_score(labels, probs)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "positive_rate": float(labels.mean()),
        "n": len(labels),
    }


def expected_calibration_error(labels, probs, n_bins: int = 10) -> float:
    """Expected Calibration Error over equal-width probability bins."""
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lower, upper in pairwise(edges):
        in_bin = (probs > lower) & (probs <= upper)
        weight = in_bin.mean()
        if weight > 0:
            ece += abs(probs[in_bin].mean() - labels[in_bin].mean()) * weight
    return float(ece)


def uncertainty_metrics(labels, probs, stds, threshold):
    """
    Judges whether the MC Dropout spread carries information.

    The headline is error_detection_auc: treating |error| as the target and
    the standard deviation as the score, can uncertainty rank which
    predictions are wrong? 0.5 means the uncertainty is decorative.
    """
    predicted = (probs >= threshold).astype(int)
    correct = predicted == labels
    errors = (~correct).astype(int)

    metrics = {
        "mean_uncertainty": float(stds.mean()),
        "uncertainty_std_across_inputs": float(stds.std()),
        "mean_uncertainty_correct": float(stds[correct].mean())
        if correct.any()
        else float("nan"),
        "mean_uncertainty_incorrect": float(stds[~correct].mean())
        if (~correct).any()
        else float("nan"),
    }
    if len(np.unique(errors)) > 1:
        metrics["error_detection_auc"] = float(roc_auc_score(errors, stds))
        metrics["uncertainty_error_correlation"] = float(
            np.corrcoef(stds, errors)[0, 1]
        )
    else:
        metrics["error_detection_auc"] = float("nan")
        metrics["uncertainty_error_correlation"] = float("nan")
    return metrics


def deterministic_probs(model, windows, features, batch_size):
    """Single deterministic forward pass with dropout disabled."""
    use_features = model.feature_dim > 0
    out = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            stop = start + batch_size
            w = torch.from_numpy(windows[start:stop]).float()
            f = torch.from_numpy(features[start:stop]).float() if use_features else None
            out.append(torch.sigmoid(model(w, f)).numpy())
    return np.concatenate(out)


def mc_vs_deterministic(model, splits, mc_samples, batch_size):
    """
    Compares MC-averaged predictions against a single deterministic pass.

    MC Dropout exists here for the uncertainty interval, not for accuracy.
    This records what averaging actually costs or buys in ranking quality, so
    the ~n_samples-fold latency is a documented trade rather than an assumed win.
    """
    comparison = {}
    for split_name in ("val", "test"):
        windows = splits[split_name]["windows"]
        features = splits[split_name]["features"]
        labels = splits[split_name]["labels"]

        det = deterministic_probs(model, windows, features, batch_size)
        mc, _ = mc_predict_split(model, windows, features, mc_samples, batch_size)

        comparison[split_name] = {
            "deterministic": {
                "roc_auc": float(roc_auc_score(labels, det)),
                "average_precision": float(average_precision_score(labels, det)),
            },
            f"mc_dropout_{mc_samples}": {
                "roc_auc": float(roc_auc_score(labels, mc)),
                "average_precision": float(average_precision_score(labels, mc)),
            },
        }
    return comparison


def run_baselines(splits):
    """Fits the classical baselines and scores them on validation and test."""
    train_y = splits["train"]["labels"]
    results = {}

    def score_model(name, fit_x, val_x, test_x):
        model = LogisticRegression(max_iter=5000, class_weight="balanced")
        model.fit(fit_x, train_y)
        for split_name, x in (("val", val_x), ("test", test_x)):
            y = splits[split_name]["labels"]
            probs = model.predict_proba(x)[:, 1]
            results.setdefault(name, {})[split_name] = {
                "roc_auc": float(roc_auc_score(y, probs)),
                "average_precision": float(average_precision_score(y, probs)),
            }

    score_model(
        "logistic_regression_features",
        splits["train"]["features"],
        splits["val"]["features"],
        splits["test"]["features"],
    )
    score_model(
        "logistic_regression_sequence",
        splits["train"]["windows"],
        splits["val"]["windows"],
        splits["test"]["windows"],
    )

    # Always-positive predictor: the degenerate solution the previous
    # checkpoint collapsed to. Its AP equals the positive rate by definition.
    for split_name in ("val", "test"):
        y = splits[split_name]["labels"]
        results.setdefault("always_anomaly", {})[split_name] = {
            "roc_auc": 0.5,
            "average_precision": float(y.mean()),
        }

    return results


def make_plots(labels, probs, stds, threshold, output_dir: Path):
    """Writes the ROC, precision-recall, calibration and uncertainty figures."""
    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(labels, probs):.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC, held-out series")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=120)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(
        recall, precision, label=f"AP = {average_precision_score(labels, probs):.3f}"
    )
    ax.axhline(
        labels.mean(),
        ls="--",
        c="k",
        alpha=0.4,
        label=f"base rate = {labels.mean():.3f}",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall, held-out series")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "pr_curve.png", dpi=120)
    plt.close(fig)

    edges = np.linspace(0, 1, 11)
    centres, observed = [], []
    for lower, upper in pairwise(edges):
        in_bin = (probs > lower) & (probs <= upper)
        if in_bin.sum() > 0:
            centres.append(probs[in_bin].mean())
            observed.append(labels[in_bin].mean())
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect calibration")
    ax.plot(centres, observed, "o-", label="model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed anomaly frequency")
    ax.set_title("Reliability diagram")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "calibration_curve.png", dpi=120)
    plt.close(fig)

    predicted = (probs >= threshold).astype(int)
    correct = predicted == labels
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0, max(stds.max(), 1e-6), 40)
    ax.hist(stds[correct], bins=bins, alpha=0.6, density=True, label="correct")
    ax.hist(stds[~correct], bins=bins, alpha=0.6, density=True, label="incorrect")
    ax.set_xlabel("MC Dropout standard deviation")
    ax.set_ylabel("Density")
    ax.set_title("Uncertainty, correct vs incorrect predictions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "uncertainty_histogram.png", dpi=120)
    plt.close(fig)


def main() -> None:
    """Evaluate the checkpoint and write metrics and figures."""
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, config = load_scorer(Path(args.model_path))
    print(f"Loaded {args.model_path}  (feature_dim={config.get('feature_dim', 0)})")

    splits, _ = prepare_grouped_splits(
        window_size=config["window_size"],
        step_size=config.get("step_size", 10),
        window_frac=config.get("window_frac", 0.02),
    )

    print(f"Running MC Dropout with {args.mc_samples} samples...")
    val_probs, _ = mc_predict_split(
        model,
        splits["val"]["windows"],
        splits["val"]["features"],
        args.mc_samples,
        args.batch_size,
    )
    test_probs, test_stds = mc_predict_split(
        model,
        splits["test"]["windows"],
        splits["test"]["features"],
        args.mc_samples,
        args.batch_size,
    )

    val_labels = splits["val"]["labels"]
    test_labels = splits["test"]["labels"]

    # Threshold chosen on validation, applied to test. Choosing it on test
    # would report a number no deployment could reproduce.
    threshold = best_f1_threshold(val_labels, val_probs)

    results = {
        "model": {
            "val": classification_metrics(val_labels, val_probs, threshold),
            "test": classification_metrics(test_labels, test_probs, threshold),
        },
        "calibration": {
            "expected_calibration_error_test": expected_calibration_error(
                test_labels, test_probs
            ),
        },
        "uncertainty": uncertainty_metrics(
            test_labels, test_probs, test_stds, threshold
        ),
        "mc_vs_deterministic": mc_vs_deterministic(
            model, splits, args.mc_samples, args.batch_size
        ),
        "baselines": run_baselines(splits),
        "config": {
            "seed": args.seed,
            "mc_samples": args.mc_samples,
            "threshold_selected_on": "validation split, max F1",
            "window_size": config["window_size"],
            "window_frac": config.get("window_frac"),
            "test_series": config.get("test_series"),
        },
    }

    with open(output_dir / "evaluation_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    make_plots(test_labels, test_probs, test_stds, threshold, output_dir)

    print("\n=== Held-out test series ===")
    print(format_metrics(results["model"]["test"]))
    print("\n=== Uncertainty quality ===")
    print(format_metrics(results["uncertainty"]))
    print("\n=== Comparison (average precision / ROC-AUC) ===")
    print(f"{'method':<34} {'val AP':>7} {'val AUC':>8} {'test AP':>8} {'test AUC':>9}")
    row = results["model"]
    print(
        f"{'transformer + MC dropout':<34} {row['val']['average_precision']:>7.3f} "
        f"{row['val']['roc_auc']:>8.3f} {row['test']['average_precision']:>8.3f} "
        f"{row['test']['roc_auc']:>9.3f}"
    )
    for name, scores in results["baselines"].items():
        print(
            f"{name:<34} {scores['val']['average_precision']:>7.3f} "
            f"{scores['val']['roc_auc']:>8.3f} {scores['test']['average_precision']:>8.3f} "
            f"{scores['test']['roc_auc']:>9.3f}"
        )
    print("\n=== MC Dropout vs a single deterministic pass ===")
    print(f"{'split':<6} {'mode':<20} {'AUC':>7} {'AP':>7}")
    for split_name, modes in results["mc_vs_deterministic"].items():
        for mode, scores in modes.items():
            print(
                f"{split_name:<6} {mode:<20} {scores['roc_auc']:>7.3f} "
                f"{scores['average_precision']:>7.3f}"
            )

    print(f"\nWritten to {output_dir / 'evaluation_metrics.json'}")


if __name__ == "__main__":
    main()
