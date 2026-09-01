#!/usr/bin/env python3
"""
Builds the population-drift reference profile served alongside the model.

The profile records how the model's input features and output scores were
distributed on the data it was trained on. The service compares live traffic
against it and exposes the divergence as chrono_drift_psi, which is how the
deployment notices it has gone stale.

The reference is built from the **training split only**. Using test data would
leak; using live traffic would let the reference drift along with the data and
report stability while the model quietly goes stale, defeating the purpose.

Usage:
    python scripts/build_reference.py
    python scripts/build_reference.py --model-path outputs/best_model.pt --n-bins 20
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from threatsim.data import prepare_grouped_splits
from threatsim.models import create_model, mc_dropout_predict
from threatsim.reference import DEFAULT_N_BINS, ReferenceProfile


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build the drift reference profile from the training split"
    )
    parser.add_argument("--model-path", type=str, default="outputs/best_model.pt")
    parser.add_argument("--output", type=str, default="outputs/reference.json")
    parser.add_argument(
        "--n-bins", type=int, default=DEFAULT_N_BINS, help="Quantile bins per feature"
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=30,
        help="MC passes used to build the reference score distribution. Should "
        "match the service default so live and reference scores are comparable.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    """Score the training split and write the reference profile."""
    args = parse_args()
    model_path = Path(args.model_path)

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

    splits, scaler = prepare_grouped_splits(
        window_size=config["window_size"],
        step_size=config.get("step_size", 10),
        window_frac=config.get("window_frac", 0.02),
    )
    train = splits["train"]
    print(f"Reference source: {len(train['labels'])} training windows")

    use_features = model.feature_dim > 0
    scores = []
    for start in range(0, len(train["windows"]), args.batch_size):
        stop = start + args.batch_size
        windows = torch.from_numpy(train["windows"][start:stop]).float()
        features = (
            torch.from_numpy(train["features"][start:stop]).float()
            if use_features
            else None
        )
        mean, _ = mc_dropout_predict(
            model, windows, features, n_samples=args.mc_samples, batched=True
        )
        scores.append(mean.numpy())
    scores = np.concatenate(scores)

    profile = ReferenceProfile.build(
        features=train["features"],
        feature_names=scaler.feature_names,
        scores=scores,
        n_bins=args.n_bins,
        source={
            "model_path": str(model_path),
            "model_version": payload.get("metrics", {}).get("epoch", ""),
            "train_series": config.get("train_series", []),
            "window_size": config["window_size"],
            "window_frac": config.get("window_frac"),
            "mc_samples": args.mc_samples,
            "split": "train",
        },
    )

    output = Path(args.output)
    profile.save(output)

    print(
        f"Score distribution: min {scores.min():.4f}  median "
        f"{np.median(scores):.4f}  max {scores.max():.4f}"
    )
    print(
        f"Features profiled: {len(profile.feature_names)} + score, "
        f"{profile.n_bins} bins each"
    )
    print(f"Written to {output}")


if __name__ == "__main__":
    main()
