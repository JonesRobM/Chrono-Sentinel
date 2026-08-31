#!/usr/bin/env python3
"""
Measures the MC Dropout batching optimisation, in isolation from HTTP.

`mc_dropout_predict` can run its n_samples stochastic passes either as a
Python loop or as a single forward pass over a batch of replicated inputs.
The batched form is faster, but it is only a legitimate substitute if it is
distributionally equivalent -- a faster function that changes the uncertainty
semantics is a bug. This reports both the speedup and the equivalence check,
so the README can cite a measured number rather than an assumed win.

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
import torch

from threatsim.models import mc_dropout_predict
from threatsim.serving.inference import AnomalyScorer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark MC Dropout batching")
    parser.add_argument("--model-path", type=str, default="outputs/best_model.pt")
    parser.add_argument("--repeats", type=int, default=300, help="Timed iterations")
    parser.add_argument("--warmup", type=int, default=50, help="Discarded iterations")
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="torch intra-op threads. The service runs with 1.",
    )
    parser.add_argument(
        "--mc-samples", type=str, default="10,30,100", help="Comma-separated sample counts"
    )
    parser.add_argument(
        "--equivalence-repeats",
        type=int,
        default=300,
        help="Draws used for the distributional equivalence check",
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


def main() -> None:
    """Run the benchmark and the equivalence check."""
    args = parse_args()
    torch.set_num_threads(args.threads)

    scorer = AnomalyScorer.from_checkpoint(Path(args.model_path))
    rng = np.random.default_rng(0)
    window = 85 + rng.normal(0, 1, scorer.window_size)
    sequence, features, _ = scorer.preprocess(window)

    results = {
        "torch_threads": args.threads,
        "repeats": args.repeats,
        "warmup_discarded": args.warmup,
        "window_size": scorer.window_size,
        "model_version": scorer.model_version,
        "speedup": {},
        "equivalence": {},
    }

    print(f"torch threads={args.threads}  repeats={args.repeats}  "
          f"warmup={args.warmup} (discarded)")
    print(f"\n{'mc':>5} {'sequential p50':>15} {'batched p50':>12} {'speedup':>8} "
          f"{'seq p99':>9} {'bat p99':>9}")

    for mc_samples in (int(v) for v in args.mc_samples.split(",")):
        sequential = time_calls(
            lambda: mc_dropout_predict(
                scorer.model, sequence, features,
                n_samples=mc_samples, batched=False, manage_mode=False,
            ),
            args.repeats, args.warmup,
        )
        batched = time_calls(
            lambda: mc_dropout_predict(
                scorer.model, sequence, features,
                n_samples=mc_samples, batched=True, manage_mode=False,
            ),
            args.repeats, args.warmup,
        )
        speedup = sequential["p50"] / batched["p50"]
        results["speedup"][str(mc_samples)] = {
            "sequential_ms": sequential,
            "batched_ms": batched,
            "speedup_p50": round(speedup, 2),
        }
        print(f"{mc_samples:>5} {sequential['p50']:>15.2f} {batched['p50']:>12.2f} "
              f"{speedup:>7.1f}x {sequential['p99']:>9.2f} {batched['p99']:>9.2f}")

    # Equivalence: the two forms draw independent dropout masks, so they cannot
    # agree sample-for-sample. They must agree on the estimates they produce.
    print(f"\nDistributional equivalence, n=30, {args.equivalence_repeats} draws:")
    for label, batched_flag in (("sequential", False), ("batched", True)):
        means, stds = [], []
        for seed in range(args.equivalence_repeats):
            torch.manual_seed(seed)
            mean, std = mc_dropout_predict(
                scorer.model, sequence, features,
                n_samples=30, batched=batched_flag, manage_mode=False,
            )
            means.append(mean.item())
            stds.append(std.item())
        results["equivalence"][label] = {
            "mean_of_means": round(float(np.mean(means)), 5),
            "mean_of_sigmas": round(float(np.mean(stds)), 5),
            "sd_of_means": round(float(np.std(means)), 5),
        }
        print(f"  {label:<11} mean {np.mean(means):.5f}  sigma {np.mean(stds):.5f}  "
              f"sd of means {np.std(means):.5f}")

    sequential_mean = results["equivalence"]["sequential"]["mean_of_means"]
    batched_mean = results["equivalence"]["batched"]["mean_of_means"]
    sequential_sigma = results["equivalence"]["sequential"]["mean_of_sigmas"]
    batched_sigma = results["equivalence"]["batched"]["mean_of_sigmas"]
    results["equivalence"]["mean_absolute_difference"] = round(
        abs(sequential_mean - batched_mean), 5
    )
    results["equivalence"]["sigma_relative_difference"] = round(
        abs(sequential_sigma - batched_sigma) / sequential_sigma, 4
    )
    print(f"\n  mean differs by {results['equivalence']['mean_absolute_difference']:.5f} "
          f"absolute; sigma by "
          f"{100 * results['equivalence']['sigma_relative_difference']:.1f}% relative")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nWritten to {output}")


if __name__ == "__main__":
    main()
