#!/usr/bin/env python3
"""
Exports the model for the browser, plus golden vectors to verify the JS port.

The .npz the container uses is a zip of .npy files, which is awkward to parse
in a browser. This writes two files instead:

    docs/assets/model.bin    every parameter concatenated as little-endian
                             float32, 297 KB, fetched as one ArrayBuffer
    docs/assets/model.json   config, feature scaler, and a manifest giving
                             each tensor's shape and offset into that buffer

It also writes tests/web/golden.json: raw windows with the intermediate values
and final logits that Python produces for them. The JS implementation is a
*third* rendering of the same forward pass, and without a fixture to check it
against there is nothing stopping it drifting. Each case carries the
intermediate stages too, so a mismatch says which stage broke rather than just
that the answer is wrong.

Usage:
    python scripts/export_web_model.py
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from threatsim.features import extract_window_features, get_feature_names
from threatsim.serving.forward import NumpyModel
from threatsim.serving.inference import AnomalyScorer

WEB_DIR = Path(__file__).parent.parent / "docs" / "assets"
GOLDEN_PATH = Path(__file__).parent.parent / "tests" / "web" / "golden.json"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Export the model for the browser")
    parser.add_argument("--model-path", type=str, default="outputs/model.npz")
    parser.add_argument(
        "--golden-cases", type=int, default=12, help="Number of golden vectors"
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--mc-reference-cases",
        type=int,
        default=4,
        help="Cases to record Monte Carlo reference statistics for.",
    )
    parser.add_argument("--mc-repeats", type=int, default=250)
    return parser.parse_args()


def build_binary(model: NumpyModel) -> tuple[bytes, list[dict]]:
    """
    Packs every tensor into one little-endian float32 buffer.

    Returns:
        Tuple of (buffer bytes, manifest entries in buffer order).
    """
    manifest = []
    chunks = []
    offset = 0

    # Sorted so the layout is deterministic and a diff of model.json is stable.
    for name in sorted(model.weights):
        array = np.ascontiguousarray(model.weights[name], dtype="<f4")
        flat = array.reshape(-1)
        manifest.append(
            {
                "name": name,
                "shape": list(array.shape),
                "offset": offset,
                "length": int(flat.size),
            }
        )
        chunks.append(flat.tobytes())
        offset += int(flat.size)

    return b"".join(chunks), manifest


def golden_windows(count: int, window_size: int, rng) -> list[np.ndarray]:
    """A spread of window shapes, so the fixture exercises more than one regime."""
    windows = []
    for index in range(count):
        kind = index % 6
        if kind == 0:
            window = np.full(window_size, 85.0)
        elif kind == 1:
            window = 85 + rng.normal(0, rng.uniform(0.5, 12), window_size)
        elif kind == 2:
            window = np.linspace(rng.uniform(0, 40), rng.uniform(60, 180), window_size)
        elif kind == 3:
            half = window_size // 2
            window = np.r_[
                np.full(half, 85.0), np.full(window_size - half, rng.uniform(0, 60))
            ]
        elif kind == 4:
            window = 85 + rng.uniform(2, 20) * np.sin(
                np.arange(window_size) * rng.uniform(0.1, 1.5)
            )
        else:
            window = np.full(window_size, 85.0)
            window[rng.integers(0, window_size)] = rng.uniform(150, 400)
        windows.append(window.astype(np.float64))
    return windows


def main() -> None:
    """Write the browser bundle and the golden fixture."""
    args = parse_args()

    scorer = AnomalyScorer.from_weights(Path(args.model_path))
    model = scorer.model

    WEB_DIR.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)

    buffer, manifest = build_binary(model)
    (WEB_DIR / "model.bin").write_bytes(buffer)

    scaler = scorer.scaler
    metadata = {
        "modelVersion": model.model_version,
        "windowSize": model.window_size,
        "dModel": model.d_model,
        "numHeads": model.num_heads,
        "numLayers": model.num_layers,
        "dropout": model.dropout_p,
        "featureDim": model.feature_dim,
        "featureNames": get_feature_names(),
        "scaler": {
            "mean": [float(v) for v in scaler.mean],
            "std": [float(v) for v in scaler.std],
        }
        if scaler is not None
        else None,
        "totalFloats": sum(entry["length"] for entry in manifest),
        "tensors": manifest,
    }
    (WEB_DIR / "model.json").write_text(json.dumps(metadata, indent=2) + "\n")

    # Golden vectors. Each records every stage, so a JS mismatch localises.
    rng = np.random.default_rng(args.seed)
    mc_rng = np.random.default_rng(args.seed + 1)
    cases = []
    for index, window in enumerate(
        golden_windows(args.golden_cases, model.window_size, rng)
    ):
        sequence, features = scorer.preprocess(window)
        raw_features = extract_window_features(window.astype(np.float32))
        logit = float(model.logits(sequence, features, training=False)[0])
        case = {
            "window": [float(v) for v in window],
            "sequence": [float(v) for v in sequence[0]],
            "rawFeatures": [float(v) for v in raw_features],
            "scaledFeatures": [float(v) for v in features[0]],
            "logit": logit,
            "probability": float(1.0 / (1.0 + np.exp(-logit))),
        }

        # Monte Carlo reference for the first few cases. The deterministic
        # check above cannot catch a missing or misplaced dropout site: with
        # dropout off, every site is the identity. Only the spread reveals it,
        # which is exactly how the fused-fast-path bug in the torch backend
        # was found.
        if index < args.mc_reference_cases:
            means, sigmas = [], []
            for _ in range(args.mc_repeats):
                mean, sigma = model.mc_dropout_predict(
                    sequence, features, n_samples=30, rng=mc_rng
                )
                means.append(float(mean[0]))
                sigmas.append(float(sigma[0]))
            case["mcReference"] = {
                "nSamples": 30,
                "repeats": args.mc_repeats,
                "meanOfMeans": float(np.mean(means)),
                "meanOfSigmas": float(np.mean(sigmas)),
            }

        cases.append(case)

    GOLDEN_PATH.write_text(
        json.dumps(
            {
                "modelVersion": model.model_version,
                "windowSize": model.window_size,
                "note": (
                    "Generated by scripts/export_web_model.py. Each case carries "
                    "the intermediate stages so a JS mismatch identifies which "
                    "stage diverged."
                ),
                "cases": cases,
            },
            indent=2,
        )
        + "\n"
    )

    print(
        f"  {WEB_DIR / 'model.bin'}   {len(buffer) / 1024:.0f} KB, "
        f"{metadata['totalFloats']:,} floats"
    )
    print(f"  {WEB_DIR / 'model.json'}  {len(manifest)} tensors")
    print(f"  {GOLDEN_PATH}  {len(cases)} golden cases")


if __name__ == "__main__":
    main()
