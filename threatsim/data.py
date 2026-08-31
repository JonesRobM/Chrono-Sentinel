"""
Data loading and preprocessing for NAB time-series anomaly detection.

This module loads NAB (Numenta Anomaly Benchmark) series, labels them using
NAB's own anomaly-window convention, cuts them into sliding windows, and
builds PyTorch DataLoaders.

Splitting is **grouped by series**: train, validation and test draw from
disjoint sets of series. This measures whether the detector generalises to an
asset it has never seen, and it keeps the positive-class rate comparable
across splits. A temporal within-series split does neither, because NAB
anomalies cluster in the later part of most series, which pushes nearly all
the positives into the test set.

Preprocessing is a pure function of a single window (per-window
z-scoring for the sequence, plus a globally scaled statistical feature
vector), so the serving path can reproduce it exactly from an incoming
window with no series identity.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from threatsim.features import extract_features, get_feature_names
from threatsim.scaling import FeatureScaler


# Series held out entirely for validation and test. Chosen to span both NAB
# collections and several value scales, so the test split is a genuine
# transfer setting rather than a rerun of the training distribution.
DEFAULT_TRAIN_SERIES = [
    "realKnownCause/ambient_temperature_system_failure.csv",
    "realKnownCause/cpu_utilization_asg_misconfiguration.csv",
    "realKnownCause/ec2_request_latency_system_failure.csv",
    "realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv",
    "realAWSCloudwatch/ec2_cpu_utilization_ac20cd.csv",
    "realAWSCloudwatch/rds_cpu_utilization_e47b3b.csv",
    "realAWSCloudwatch/ec2_network_in_5abac7.csv",
]

DEFAULT_VAL_SERIES = [
    "realKnownCause/nyc_taxi.csv",
    "realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv",
]

DEFAULT_TEST_SERIES = [
    "realKnownCause/machine_temperature_system_failure.csv",
    "realAWSCloudwatch/rds_cpu_utilization_cc0c53.csv",
    "realAWSCloudwatch/elb_request_count_8c0756.csv",
]

# Every series the project touches; used by scripts/fetch_data.py.
DEFAULT_DATASETS = DEFAULT_TRAIN_SERIES + DEFAULT_VAL_SERIES + DEFAULT_TEST_SERIES

# Fraction of a series' length treated as anomalous, distributed across that
# series' labelled anomalies. NAB's own scoring convention uses 0.10; we
# default narrower so the positive class reflects the acute anomaly rather
# than a broad neighbourhood around it. scripts/train.py exposes this as a
# flag, and evaluation reports its sensitivity.
DEFAULT_ANOMALY_WINDOW_FRAC = 0.02


def get_nab_root() -> Path:
    """Returns the path to the NAB data directory."""
    return Path(__file__).parent.parent / "NAB_temp"


def load_nab_labels(nab_root: Optional[Path] = None) -> Dict[str, List[str]]:
    """
    Loads anomaly labels from the NAB combined_labels.json file.

    Args:
        nab_root: Path to NAB directory. Uses default if None.

    Returns:
        Dictionary mapping dataset names to lists of anomaly timestamps.
    """
    if nab_root is None:
        nab_root = get_nab_root()

    labels_path = nab_root / "labels" / "combined_labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"{labels_path} not found. Run: python scripts/fetch_data.py"
        )
    with open(labels_path, "r") as f:
        return json.load(f)


def load_nab_data(
    dataset_name: str = "realKnownCause/machine_temperature_system_failure.csv",
    nab_root: Optional[Path] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Loads a dataset from the Numenta Anomaly Benchmark (NAB).

    Args:
        dataset_name: Relative path to the dataset within NAB data directory.
        nab_root: Path to NAB directory. Uses default if None.

    Returns:
        Tuple of (DataFrame with timestamp and value columns, list of anomaly timestamps).
    """
    if nab_root is None:
        nab_root = get_nab_root()

    data_path = nab_root / "data" / dataset_name
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Run: python scripts/fetch_data.py"
        )

    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    labels = load_nab_labels(nab_root)
    anomaly_timestamps = labels.get(dataset_name, [])

    return df, anomaly_timestamps


def nab_anomaly_mask(
    df: pd.DataFrame,
    anomaly_timestamps: Sequence[str],
    window_frac: float = DEFAULT_ANOMALY_WINDOW_FRAC,
) -> np.ndarray:
    """
    Creates a binary anomaly mask using NAB's anomaly-window convention.

    NAB annotates anomalies as single timestamps, but a detector that fires
    slightly early or late is still correct. NAB therefore scores against a
    window around each annotation whose total length is a fixed fraction of
    the series, divided evenly across that series' anomalies. This applies the
    same rule to produce classification labels.

    Args:
        df: DataFrame with a timestamp column.
        anomaly_timestamps: Anomaly timestamp strings for this series.
        window_frac: Fraction of the series length that is anomalous in total.

    Returns:
        Binary numpy array where 1 indicates anomaly.
    """
    n = len(df)
    mask = np.zeros(n, dtype=np.int64)
    if not anomaly_timestamps or n == 0:
        return mask

    half_width = max(1, int((window_frac * n) / len(anomaly_timestamps) / 2))
    # searchsorted rather than an index lookup: several NAB series contain
    # duplicate timestamps, which makes a DatetimeIndex non-unique.
    stamps = df["timestamp"].values.astype("datetime64[ns]")

    for timestamp in anomaly_timestamps:
        centre = int(np.searchsorted(stamps, np.datetime64(timestamp, "ns")))
        centre = min(max(centre, 0), n - 1)
        mask[max(0, centre - half_width) : min(n, centre + half_width + 1)] = 1

    return mask


def create_anomaly_mask(
    df: pd.DataFrame,
    anomaly_timestamps: List[str],
    window_minutes: int = 30,
) -> np.ndarray:
    """
    Creates a binary anomaly mask using a fixed wall-clock window.

    Retained for the exploratory notebooks. Prefer nab_anomaly_mask for
    training and evaluation: at NAB's 5-minute sampling this marks only
    +/-6 samples around each annotation, which leaves too few positive
    windows to learn from.

    Args:
        df: DataFrame with timestamp column.
        anomaly_timestamps: List of anomaly timestamp strings.
        window_minutes: Minutes around each anomaly to mark as anomalous.

    Returns:
        Binary numpy array where 1 indicates anomaly.
    """
    mask = np.zeros(len(df), dtype=np.int64)

    if not anomaly_timestamps:
        return mask

    anomaly_times = pd.to_datetime(anomaly_timestamps)
    window_delta = pd.Timedelta(minutes=window_minutes)

    for anomaly_time in anomaly_times:
        start_time = anomaly_time - window_delta
        end_time = anomaly_time + window_delta
        mask[(df["timestamp"] >= start_time) & (df["timestamp"] <= end_time)] = 1

    return mask


def create_windows(
    values: np.ndarray,
    labels: np.ndarray,
    window_size: int = 50,
    step_size: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates sliding windows from time-series data.

    Args:
        values: 1D array of time-series values.
        labels: 1D array of binary labels (0=normal, 1=anomaly).
        window_size: Number of timesteps per window.
        step_size: Step between consecutive windows.

    Returns:
        Tuple of (windows array of shape (n_windows, window_size),
                  window_labels array of shape (n_windows,)).
    """
    n_samples = len(values)
    if n_samples < window_size:
        return (
            np.zeros((0, window_size), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    n_windows = (n_samples - window_size) // step_size + 1

    windows = np.zeros((n_windows, window_size), dtype=np.float32)
    window_labels = np.zeros(n_windows, dtype=np.int64)

    for i in range(n_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        windows[i] = values[start_idx:end_idx]
        # Window is anomalous if it contains any anomaly
        window_labels[i] = 1 if labels[start_idx:end_idx].sum() > 0 else 0

    return windows, window_labels


def normalise_windows(windows: np.ndarray) -> np.ndarray:
    """
    Normalises each window to zero mean and unit variance.

    This is deliberately a pure function of a single window, so the serving
    path can apply it to an incoming request with no series context. It
    discards absolute level and scale; the statistical feature vector
    produced by extract_features carries that information instead.

    Args:
        windows: Array of shape (n_windows, window_size).

    Returns:
        Normalised windows array.
    """
    mean = windows.mean(axis=1, keepdims=True)
    std = windows.std(axis=1, keepdims=True)
    # Avoid division by zero
    std = np.where(std == 0, 1.0, std)
    return (windows - mean) / std


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time-series windows and their feature vectors."""

    def __init__(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        features: Optional[np.ndarray] = None,
    ):
        """
        Args:
            windows: Array of shape (n_windows, window_size), already normalised.
            labels: Array of shape (n_windows,).
            features: Optional scaled features of shape (n_windows, n_features).
        """
        self.windows = torch.from_numpy(np.ascontiguousarray(windows)).float()
        self.labels = torch.from_numpy(np.ascontiguousarray(labels)).float()
        self.features = (
            torch.from_numpy(np.ascontiguousarray(features)).float()
            if features is not None
            else None
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        if self.features is not None:
            return self.windows[idx], self.features[idx], self.labels[idx]
        return self.windows[idx], self.labels[idx]


def temporal_train_val_test_split(
    windows: np.ndarray,
    labels: np.ndarray,
    features: Optional[np.ndarray] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Dict[str, Tuple[np.ndarray, ...]]:
    """
    Splits a single series temporally (no shuffling) into train/val/test sets.

    Retained for the notebooks and for single-series experiments. Do not
    apply it to a concatenation of several series: the split boundaries land
    inside whichever series is longest, so entire series end up confined to
    one split. Use prepare_grouped_splits instead.

    Args:
        windows: Array of shape (n_windows, window_size).
        labels: Array of shape (n_windows,).
        features: Optional features array.
        train_ratio: Proportion of data for training.
        val_ratio: Proportion of data for validation.

    Returns:
        Dictionary with 'train', 'val', 'test' keys.
    """
    n_samples = len(windows)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    splits = {}
    bounds = {
        "train": slice(0, train_end),
        "val": slice(train_end, val_end),
        "test": slice(val_end, n_samples),
    }
    for name, sl in bounds.items():
        if features is not None:
            splits[name] = (windows[sl], labels[sl], features[sl])
        else:
            splits[name] = (windows[sl], labels[sl])

    return splits


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Computes the positive-class weight for handling imbalanced data.

    Args:
        labels: Binary labels array.

    Returns:
        Scalar tensor weighting the positive (anomaly) class, suitable for
        the pos_weight argument of BCEWithLogitsLoss.
    """
    n_normal = (labels == 0).sum()
    n_anomaly = (labels == 1).sum()

    if n_anomaly == 0:
        return torch.tensor(1.0)

    return torch.tensor(n_normal / n_anomaly, dtype=torch.float32)


def build_series_windows(
    dataset_names: Sequence[str],
    window_size: int,
    step_size: int,
    window_frac: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the named series and cuts them all into windows.

    Args:
        dataset_names: NAB series paths.
        window_size: Timesteps per window.
        step_size: Step between consecutive windows.
        window_frac: Anomaly window fraction, see nab_anomaly_mask.

    Returns:
        Tuple of (raw windows, labels, series index per window).
    """
    all_windows, all_labels, all_origin = [], [], []

    for series_idx, dataset_name in enumerate(dataset_names):
        df, anomaly_timestamps = load_nab_data(dataset_name)
        values = df["value"].values.astype(np.float32)
        mask = nab_anomaly_mask(df, anomaly_timestamps, window_frac)

        windows, labels = create_windows(values, mask, window_size, step_size)
        all_windows.append(windows)
        all_labels.append(labels)
        all_origin.append(np.full(len(windows), series_idx, dtype=np.int64))

    return (
        np.concatenate(all_windows),
        np.concatenate(all_labels),
        np.concatenate(all_origin),
    )


def prepare_grouped_splits(
    train_series: Optional[Sequence[str]] = None,
    val_series: Optional[Sequence[str]] = None,
    test_series: Optional[Sequence[str]] = None,
    window_size: int = 50,
    step_size: int = 10,
    window_frac: float = DEFAULT_ANOMALY_WINDOW_FRAC,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], FeatureScaler]:
    """
    Builds train/val/test arrays from disjoint sets of NAB series.

    The feature scaler is fitted on the training split only and returned so
    it can be persisted with the model.

    Args:
        train_series: Series to train on. Defaults to DEFAULT_TRAIN_SERIES.
        val_series: Series for validation. Defaults to DEFAULT_VAL_SERIES.
        test_series: Series for test. Defaults to DEFAULT_TEST_SERIES.
        window_size: Timesteps per window.
        step_size: Step between consecutive windows.
        window_frac: Anomaly window fraction, see nab_anomaly_mask.

    Returns:
        Tuple of (splits, feature_scaler). Each split is a dict with keys
        'windows' (normalised), 'features' (scaled), 'labels' and 'origin'.
    """
    groups = {
        "train": list(train_series if train_series is not None else DEFAULT_TRAIN_SERIES),
        "val": list(val_series if val_series is not None else DEFAULT_VAL_SERIES),
        "test": list(test_series if test_series is not None else DEFAULT_TEST_SERIES),
    }

    overlap = set(groups["train"]) & (set(groups["val"]) | set(groups["test"]))
    if overlap:
        raise ValueError(
            "Series appear in both the training split and a held-out split, "
            f"which leaks: {sorted(overlap)}"
        )

    raw: Dict[str, Dict[str, np.ndarray]] = {}
    for name, series in groups.items():
        windows, labels, origin = build_series_windows(
            series, window_size, step_size, window_frac
        )
        raw[name] = {"raw_windows": windows, "labels": labels, "origin": origin}

    # Fit the feature scaler on training windows only.
    train_features = extract_features(raw["train"]["raw_windows"])
    scaler = FeatureScaler.fit(train_features)

    splits: Dict[str, Dict[str, np.ndarray]] = {}
    for name, payload in raw.items():
        features = (
            train_features
            if name == "train"
            else extract_features(payload["raw_windows"])
        )
        splits[name] = {
            "windows": normalise_windows(payload["raw_windows"]),
            "features": scaler.transform(features),
            "labels": payload["labels"],
            "origin": payload["origin"],
            "raw_windows": payload["raw_windows"],
        }

    return splits, scaler


def get_dataloaders(
    train_series: Optional[Sequence[str]] = None,
    val_series: Optional[Sequence[str]] = None,
    test_series: Optional[Sequence[str]] = None,
    window_size: int = 50,
    step_size: int = 10,
    batch_size: int = 32,
    window_frac: float = DEFAULT_ANOMALY_WINDOW_FRAC,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor, FeatureScaler]:
    """
    Creates train/val/test DataLoaders from disjoint sets of NAB series.

    Args:
        train_series: Series to train on. Defaults to DEFAULT_TRAIN_SERIES.
        val_series: Series for validation. Defaults to DEFAULT_VAL_SERIES.
        test_series: Series for test. Defaults to DEFAULT_TEST_SERIES.
        window_size: Timesteps per window.
        step_size: Step between consecutive windows.
        batch_size: Batch size for DataLoaders.
        window_frac: Anomaly window fraction, see nab_anomaly_mask.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, pos_weight, scaler).
    """
    splits, scaler = prepare_grouped_splits(
        train_series, val_series, test_series, window_size, step_size, window_frac
    )

    class_weight = compute_class_weights(splits["train"]["labels"])

    loaders = []
    for name in ("train", "val", "test"):
        dataset = TimeSeriesDataset(
            splits[name]["windows"],
            splits[name]["labels"].astype(np.float32),
            splits[name]["features"],
        )
        loaders.append(
            DataLoader(dataset, batch_size=batch_size, shuffle=(name == "train"))
        )

    return loaders[0], loaders[1], loaders[2], class_weight, scaler
