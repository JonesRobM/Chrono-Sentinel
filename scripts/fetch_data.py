#!/usr/bin/env python3
"""
Fetches the NAB (Numenta Anomaly Benchmark) series this project trains on.

The upstream README suggested cloning the whole NAB repository, which pulls
~100 MB for the handful of CSVs actually used. This downloads only the series
listed in threatsim.data.DEFAULT_DATASETS (or those named on the command line)
plus the shared label file, into the NAB_temp/ layout the loader expects.

Usage:
    python scripts/fetch_data.py
    python scripts/fetch_data.py --datasets realKnownCause/ambient_temperature_system_failure.csv
    python scripts/fetch_data.py --all-defaults --force
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from threatsim.data import DEFAULT_DATASETS, get_nab_root

NAB_RAW_BASE = "https://raw.githubusercontent.com/numenta/NAB/master"
LABELS_PATH = "labels/combined_labels.json"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download NAB series used by this project"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="NAB dataset paths to fetch. Defaults to threatsim.data.DEFAULT_DATASETS.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files that already exist locally.",
    )
    parser.add_argument(
        "--timeout", type=int, default=60, help="Per-request timeout in seconds"
    )
    return parser.parse_args()


def download(url: str, destination: Path, timeout: int, force: bool) -> bool:
    """
    Downloads a single file, creating parent directories as needed.

    Args:
        url: Source URL.
        destination: Local path to write to.
        timeout: Request timeout in seconds.
        force: Overwrite an existing file.

    Returns:
        True if a download occurred, False if the file was already present.
    """
    if destination.exists() and not force:
        print(f"  exists  {destination}")
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"HTTP {exc.code} fetching {url}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Network error fetching {url}: {exc.reason}") from exc

    destination.write_bytes(payload)
    print(f"  fetched {destination}  ({len(payload):,} bytes)")
    return True


def main() -> None:
    """Fetch the label file and each requested series."""
    args = parse_args()
    datasets = args.datasets if args.datasets is not None else DEFAULT_DATASETS
    nab_root = get_nab_root()

    print(f"NAB root: {nab_root}")

    download(
        f"{NAB_RAW_BASE}/{LABELS_PATH}",
        nab_root / LABELS_PATH,
        args.timeout,
        args.force,
    )

    for dataset in datasets:
        download(
            f"{NAB_RAW_BASE}/data/{dataset}",
            nab_root / "data" / dataset,
            args.timeout,
            args.force,
        )

    # Fail loudly now rather than at training time if a series has no labels.
    labels = json.loads((nab_root / LABELS_PATH).read_text())
    missing = [d for d in datasets if d not in labels]
    if missing:
        raise SystemExit(
            "These datasets have no entry in combined_labels.json: "
            + ", ".join(missing)
        )

    print(f"\n{len(datasets)} dataset(s) ready.")


if __name__ == "__main__":
    main()
