#!/usr/bin/env python3
"""
Regenerates the example request payloads in examples/.

The payloads double as API documentation and as the input to the terminal
recording in docs/. Keeping them in files rather than inlining a long list of
numbers into every curl example is what makes both readable.

Window length is read from the trained checkpoint, so the examples cannot
drift out of step with the model the service actually loads.

Usage:
    python scripts/make_examples.py
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
DEFAULT_MC_SAMPLES = 30


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Regenerate example payloads")
    parser.add_argument("--model-path", type=str, default="outputs/best_model.pt")
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Override the window size instead of reading it from the checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def window_size_from_checkpoint(path: Path) -> int:
    """Reads the window size the served model expects."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload["config"]["window_size"])


def build_cases(window_size: int, seed: int) -> dict[str, tuple[str, list[float]]]:
    """Returns {filename: (description, values)} for each example window."""
    rng = np.random.default_rng(seed)
    half = window_size // 2

    return {
        "flat_window.json": (
            "constant 85",
            [85.0] * window_size,
        ),
        "noisy_window.json": (
            "high variance, no level shift",
            [round(float(v), 3) for v in 85 + rng.normal(0, 8, window_size)],
        ),
        "step_change.json": (
            "85 then 20",
            [85.0] * half + [20.0] * (window_size - half),
        ),
    }


def main() -> None:
    """Write the payloads and their index."""
    args = parse_args()
    window_size = args.window_size or window_size_from_checkpoint(Path(args.model_path))
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    cases = build_cases(window_size, args.seed)
    for name, (_, values) in cases.items():
        body = {"values": values, "mc_samples": DEFAULT_MC_SAMPLES}
        (EXAMPLES_DIR / name).write_text(json.dumps(body) + "\n")
        print(f"  {name:<22} {len(values)} points")

    rows = "\n".join(
        f"| `{name}` | {shape} |" for name, (shape, _) in sorted(cases.items())
    )
    (EXAMPLES_DIR / "README.md").write_text(
        "# Example request payloads\n\n"
        "Ready-to-POST bodies for `/score`. The window length must match the\n"
        f"loaded model, which is {window_size} points; query `/readyz` to confirm.\n\n"
        "```bash\n"
        "curl -s localhost:7860/score -H 'Content-Type: application/json' \\\n"
        "  -d @examples/step_change.json | jq\n"
        "```\n\n"
        "| File | Shape |\n| --- | --- |\n"
        f"{rows}\n\n"
        "A level shift is the machine-temperature failure signature and should\n"
        "score high; the noisy window checks the detector is not merely\n"
        "reacting to variance.\n\n"
        "Regenerate with `python scripts/make_examples.py`.\n"
    )
    print(f"  {'README.md':<22} index for {len(cases)} payloads")


if __name__ == "__main__":
    main()
