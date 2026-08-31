#!/usr/bin/env python3
"""
Load test for the scoring service. Produces the numbers in the README table.

Reporting rules this script enforces, so the published figures mean something:

  * A warm-up phase runs first and is excluded. The first requests include
    lazy allocation and cache warming and would otherwise dominate the tail.
  * Percentiles are reported with the sample count beside them. A p99 from
    100 requests is a single request and is not a percentile.
  * Every run records its conditions -- concurrency, request count, window
    size, mc_samples, target URL -- into the JSON output. A latency figure
    without them is not reproducible and should not be published.
  * Client-side latency and the server's own reported inference time are both
    recorded, so the gap between them (framework, serialisation, network) is
    visible rather than assumed.

Usage:
    python scripts/loadtest.py
    python scripts/loadtest.py --concurrency 16 --requests 2000
    python scripts/loadtest.py --url https://<space>.hf.space --concurrency 4
    python scripts/loadtest.py --sweep-mc 1,10,30,100
"""

import argparse
import asyncio
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Load test the scoring service")
    parser.add_argument("--url", default="http://127.0.0.1:8077", help="Base URL of the service")
    parser.add_argument("--requests", type=int, default=1000, help="Measured requests")
    parser.add_argument("--concurrency", type=int, default=8, help="In-flight requests")
    parser.add_argument("--warmup", type=int, default=100, help="Warm-up requests, discarded")
    parser.add_argument("--mc-samples", type=int, default=None, help="MC passes per request")
    parser.add_argument(
        "--sweep-mc",
        type=str,
        default=None,
        help="Comma-separated mc_samples values to test in sequence, e.g. 1,10,30,100",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the synthetic windows")
    parser.add_argument(
        "--output", type=str, default="benchmarks/results.json", help="Where to write results"
    )
    parser.add_argument("--label", type=str, default="local", help="Run label, e.g. local or spaces")
    return parser.parse_args()


def percentile(values: List[float], q: float) -> float:
    """Linear-interpolated percentile, in the same units as the input."""
    return float(np.percentile(values, q)) if values else float("nan")


async def discover_window_size(client: httpx.AsyncClient, base_url: str) -> int:
    """
    Asks the service what window size it expects.

    Hard-coding 50 here would silently send malformed requests against a model
    trained with a different window, and the run would measure the 422 path.
    """
    response = await client.get(f"{base_url}/readyz")
    response.raise_for_status()
    payload = response.json()
    if not payload.get("ready"):
        raise SystemExit(f"Service is not ready: {payload.get('detail')}")
    return int(payload["window_size"])


def make_windows(count: int, window_size: int, seed: int) -> List[List[float]]:
    """
    Builds a pool of synthetic windows spanning several shapes.

    A single repeated window would let any caching along the path flatter the
    result, and would exercise one region of the score distribution only.
    """
    rng = np.random.default_rng(seed)
    pool = []
    for index in range(count):
        kind = index % 4
        if kind == 0:
            window = 85 + rng.normal(0, 1, window_size)
        elif kind == 1:
            window = 85 + rng.normal(0, 8, window_size)
        elif kind == 2:
            window = np.linspace(80, 110, window_size) + rng.normal(0, 1, window_size)
        else:
            half = window_size // 2
            window = np.r_[np.full(half, 85.0), np.full(window_size - half, 20.0)]
            window = window + rng.normal(0, 1, window_size)
        pool.append([float(v) for v in window])
    return pool


async def worker(
    client: httpx.AsyncClient,
    base_url: str,
    queue: asyncio.Queue,
    mc_samples: Optional[int],
    client_latencies: List[float],
    server_latencies: List[float],
    failures: Dict[str, int],
) -> None:
    """Consumes windows from the queue and records per-request timings."""
    while True:
        try:
            window = queue.get_nowait()
        except asyncio.QueueEmpty:
            return

        body: Dict[str, object] = {"values": window}
        if mc_samples is not None:
            body["mc_samples"] = mc_samples

        began = time.perf_counter()
        try:
            response = await client.post(f"{base_url}/score", json=body)
            elapsed = time.perf_counter() - began
            if response.status_code == 200:
                client_latencies.append(elapsed * 1000.0)
                server_latencies.append(float(response.json()["inference_ms"]))
            else:
                failures[str(response.status_code)] = failures.get(str(response.status_code), 0) + 1
        except Exception as exc:  # noqa: BLE001 - counted, not raised
            elapsed = time.perf_counter() - began
            failures[type(exc).__name__] = failures.get(type(exc).__name__, 0) + 1
        finally:
            queue.task_done()


async def run_phase(
    base_url: str,
    windows: List[List[float]],
    concurrency: int,
    mc_samples: Optional[int],
    timeout: float,
) -> Dict[str, object]:
    """Runs one measured phase and returns its timings."""
    queue: asyncio.Queue = asyncio.Queue()
    for window in windows:
        queue.put_nowait(window)

    client_latencies: List[float] = []
    server_latencies: List[float] = []
    failures: Dict[str, int] = {}

    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        began = time.perf_counter()
        await asyncio.gather(
            *[
                worker(client, base_url, queue, mc_samples,
                       client_latencies, server_latencies, failures)
                for _ in range(concurrency)
            ]
        )
        wall_seconds = time.perf_counter() - began

    return {
        "client_latencies_ms": client_latencies,
        "server_latencies_ms": server_latencies,
        "failures": failures,
        "wall_seconds": wall_seconds,
    }


def summarise(phase: Dict[str, object], concurrency: int) -> Dict[str, object]:
    """Reduces a phase's raw timings to the reported statistics."""
    client = phase["client_latencies_ms"]
    server = phase["server_latencies_ms"]
    wall = phase["wall_seconds"]
    successes = len(client)

    return {
        "successful_requests": successes,
        "failures": phase["failures"],
        "wall_seconds": round(wall, 3),
        "throughput_rps": round(successes / wall, 1) if wall > 0 else None,
        "concurrency": concurrency,
        "client_ms": {
            "p50": round(percentile(client, 50), 2),
            "p90": round(percentile(client, 90), 2),
            "p95": round(percentile(client, 95), 2),
            "p99": round(percentile(client, 99), 2),
            "max": round(max(client), 2) if client else None,
            "mean": round(statistics.fmean(client), 2) if client else None,
        },
        "server_inference_ms": {
            "p50": round(percentile(server, 50), 2),
            "p99": round(percentile(server, 99), 2),
            "mean": round(statistics.fmean(server), 2) if server else None,
        },
    }


async def main_async(args: argparse.Namespace) -> None:
    """Runs warm-up then one measured phase per mc_samples value."""
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        window_size = await discover_window_size(client, args.url)
    print(f"Target {args.url}  window_size={window_size}")

    mc_values: List[Optional[int]] = (
        [int(v) for v in args.sweep_mc.split(",")] if args.sweep_mc else [args.mc_samples]
    )

    phases = {}
    for mc in mc_values:
        label = f"mc_samples={mc if mc is not None else 'server default'}"

        warmup_windows = make_windows(args.warmup, window_size, args.seed)
        await run_phase(args.url, warmup_windows, args.concurrency, mc, args.timeout)

        windows = make_windows(args.requests, window_size, args.seed + 1)
        phase = await run_phase(args.url, windows, args.concurrency, mc, args.timeout)
        summary = summarise(phase, args.concurrency)
        phases[label] = summary

        print(
            f"  {label:<32} n={summary['successful_requests']:<5} "
            f"p50 {summary['client_ms']['p50']:>7.2f} ms  "
            f"p99 {summary['client_ms']['p99']:>7.2f} ms  "
            f"{summary['throughput_rps']:>7.1f} rps"
        )
        if summary["failures"]:
            print(f"    failures: {summary['failures']}")

    results = {
        "label": args.label,
        "target": args.url,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "conditions": {
            "measured_requests": args.requests,
            "warmup_requests_discarded": args.warmup,
            "concurrency": args.concurrency,
            "window_size": window_size,
            "timeout_seconds": args.timeout,
            "client_platform": f"{platform.system()} {platform.machine()}",
            "client_python": platform.python_version(),
        },
        "phases": phases,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nWritten to {output}")


def main() -> None:
    """Entry point."""
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except httpx.ConnectError:
        raise SystemExit(
            f"Could not connect to {args.url}. Start the service first:\n"
            "  uvicorn threatsim.serving.app:app --port 8077"
        )


if __name__ == "__main__":
    main()
