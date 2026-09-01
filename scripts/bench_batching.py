#!/usr/bin/env python3
"""
Measures the serving forward pass, in isolation from HTTP.

Two questions:

  1. **Batching.** MC Dropout can run its n_samples passes as a Python loop or
     as one forward pass over replicated inputs. How much does folding them
     together save?
  2. **Backend.** The container runs numpy rather than torch. What does that
     cost or save at batch 1, which is the shape the service actually sees?

Distributional equivalence between the two backends is not checked here; it is
pinned by tests/test_forward_parity.py, which is where a regression should
fail.

Usage:
    python scripts/bench_batching.py
    python scripts/bench_batching.py --repeats 500 --threads 4
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from threatsim.serving.forward import NumpyModel
from threatsim.serving.inference import AnomalyScorer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark the serving forward pass")
    parser.add_argument("--model-path", type=str, default="outputs/model.npz")
    parser.add_argument(
        "--torch-checkpoint",
        type=str,
        default="outputs/best_model.pt",
        help="Used only for the backend comparison; skipped if torch is absent.",
    )
    parser.add_argument("--repeats", type=int, default=300, help="Timed iterations")
    parser.add_argument("--warmup", type=int, default=50, help="Discarded iterations")
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="torch intra-op threads for the comparison. The service runs numpy.",
    )
    parser.add_argument(
        "--mc-samples", type=str, default="10,30,100", help="Comma-separated counts"
    )
    parser.add_argument("--output", type=str, default="benchmarks/batching.json")
    return parser.parse_args()


def time_calls(fn, repeats: int, warmup: int) -> dict:
    """Times a callable and returns latency percentiles in milliseconds."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        began = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - began) * 1000.0)
    return {
        "p50": round(float(np.percentile(samples, 50)), 3),
        "p99": round(float(np.percentile(samples, 99)), 3),
        "mean": round(float(np.mean(samples)), 3),
    }


def benchmark_backends(args, sequence, features, results) -> None:
    """Times numpy against torch at batch 1, if torch is installed."""
    try:
        import torch

        from threatsim.models import create_model, mc_dropout_predict
    except ImportError:
        print("\n(torch not installed; skipping the backend comparison)")
        return

    checkpoint = Path(args.torch_checkpoint)
    if not checkpoint.exists():
        print(f"\n({checkpoint} not found; skipping the backend comparison)")
        return

    torch.set_num_threads(args.threads)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
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

    tensor_sequence = torch.from_numpy(sequence)
    tensor_features = torch.from_numpy(features)

    torch_ms = time_calls(
        lambda: mc_dropout_predict(
            model, tensor_sequence, tensor_features, n_samples=30, batched=True
        ),
        args.repeats,
        args.warmup,
    )
    numpy_ms = results["speedup"]["30"]["batched_ms"]
    speedup = torch_ms["p50"] / numpy_ms["p50"]
    results["backend"] = {
        "torch_ms": torch_ms,
        "numpy_ms": numpy_ms,
        "numpy_speedup_p50": round(speedup, 2),
        "torch_threads": args.threads,
    }

    print(f"\nBackend comparison at batch 1, mc_samples=30, {args.threads} thread(s):")
    print(f"  torch  p50 {torch_ms['p50']:>7.2f} ms   p99 {torch_ms['p99']:>7.2f} ms")
    print(f"  numpy  p50 {numpy_ms['p50']:>7.2f} ms   p99 {numpy_ms['p99']:>7.2f} ms")
    print(f"  numpy is {speedup:.2f}x the torch p50")


def main() -> None:
    """Run the benchmarks and write the results."""
    args = parse_args()

    scorer = AnomalyScorer.from_weights(Path(args.model_path))
    model: NumpyModel = scorer.model
    rng = np.random.default_rng(0)
    window = 85 + rng.normal(0, 1, scorer.window_size)
    sequence, features = scorer.preprocess(window)

    results = {
        "repeats": args.repeats,
        "warmup_discarded": args.warmup,
        "window_size": scorer.window_size,
        "model_version": scorer.model_version,
        "speedup": {},
    }

    print(f"repeats={args.repeats}  warmup={args.warmup} (discarded)  backend=numpy")
    print(
        f"\n{'mc':>5} {'sequential p50':>15} {'batched p50':>12} {'speedup':>8} "
        f"{'seq p99':>9} {'bat p99':>9}"
    )

    draw = np.random.default_rng(1)
    for mc_samples in (int(v) for v in args.mc_samples.split(",")):

        def run(batched: bool, n: int = mc_samples):
            return model.mc_dropout_predict(
                sequence, features, n_samples=n, rng=draw, batched=batched
            )

        sequential = time_calls(lambda: run(False), args.repeats, args.warmup)
        batched = time_calls(lambda: run(True), args.repeats, args.warmup)
        speedup = sequential["p50"] / batched["p50"]
        results["speedup"][str(mc_samples)] = {
            "sequential_ms": sequential,
            "batched_ms": batched,
            "speedup_p50": round(speedup, 2),
        }
        print(
            f"{mc_samples:>5} {sequential['p50']:>15.2f} {batched['p50']:>12.2f} "
            f"{speedup:>7.1f}x {sequential['p99']:>9.2f} {batched['p99']:>9.2f}"
        )

    benchmark_backends(args, sequence, features, results)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nWritten to {output}")


if __name__ == "__main__":
    main()
