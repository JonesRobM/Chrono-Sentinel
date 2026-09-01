#!/usr/bin/env python3
"""
Exports a trained checkpoint to a framework-free .npz the service can load.

`best_model.pt` is a pickled torch object, so reading it requires torch. The
serving container doesn't have torch: it runs the forward pass in numpy
(threatsim/serving/forward.py). This script is the bridge, and it runs in the
training environment where torch is available.

Everything the service needs goes into one file: the parameter arrays, the
architecture config, and the feature scaler.

Usage:
    python scripts/export_weights.py
    python scripts/export_weights.py --model-path outputs/best_model.pt --output outputs/model.npz
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

CONFIG_KEY = "__config__"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Export a checkpoint to .npz")
    parser.add_argument("--model-path", type=str, default="outputs/best_model.pt")
    parser.add_argument("--output", type=str, default="outputs/model.npz")
    return parser.parse_args()


def main() -> None:
    """Read the checkpoint and write the portable artefact."""
    args = parse_args()
    model_path = Path(args.model_path)

    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    config = dict(payload["config"])
    state = payload["model_state_dict"]

    arrays = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in state.items()}

    # The service identifies itself by the hash of the checkpoint it serves, so
    # carry the same value across rather than hashing the .npz. That keeps
    # model_version stable whether the torch or the numpy backend is running.
    config["source_checkpoint"] = model_path.name
    config["model_version"] = hashlib.sha256(model_path.read_bytes()).hexdigest()[:12]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays, **{CONFIG_KEY: np.array(json.dumps(config))})

    total = sum(v.size for v in arrays.values())
    print(f"Exported {len(arrays)} tensors, {total:,} parameters")
    print(f"  model_version: {config['model_version']}")
    print(f"  window_size:   {config['window_size']}")
    print(f"  feature_dim:   {config.get('feature_dim', 0)}")
    print(f"  {output}  ({output.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
