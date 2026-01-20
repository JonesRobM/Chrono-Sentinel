"""
Data loading and preprocessing for NAB time-series anomaly detection.

This module provides utilities to load NAB (Numenta Anomaly Benchmark) datasets,
create sliding windows, and prepare PyTorch DataLoaders for training.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# Default datasets with clear anomalies, good for demonstration
DEFAULT_DATASETS = [
    "realKnownCause/machine_temperature_system_failure.csv",
    "realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv",
]


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

    # Load data
    data_path = nab_root / "data" / dataset_name
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Load labels
    labels = load_nab_labels(nab_root)
    anomaly_timestamps = labels.get(dataset_name, [])

    return df, anomaly_timestamps


def create_anomaly_mask(
    df: pd.DataFrame,
    anomaly_timestamps: List[str],
    window_minutes: int = 30,
) -> np.ndarray:
    """
    Creates a binary mask indicating anomaly regions.

    NAB anomalies are point annotations, but we treat a window around each
    anomaly timestamp as anomalous for more robust labelling.

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
    """PyTorch Dataset for time-series windows."""

    def __init__(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        features: Optional[np.ndarray] = None,
    ):
        """
        Args:
            windows: Array of shape (n_windows, window_size).
            labels: Array of shape (n_windows,).
            features: Optional precomputed features of shape (n_windows, n_features).
        """
        self.windows = torch.from_numpy(windows).float()
        self.labels = torch.from_numpy(labels).float()
        self.features = (
            torch.from_numpy(features).float() if features is not None else None
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        window = self.windows[idx]
        label = self.labels[idx]

        if self.features is not None:
            # Concatenate raw window with features
            features = self.features[idx]
            return window, features, label

        return window, label


def temporal_train_val_test_split(
    windows: np.ndarray,
    labels: np.ndarray,
    features: Optional[np.ndarray] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Dict[str, Tuple[np.ndarray, ...]]:
    """
    Splits data temporally (no shuffling) into train/val/test sets.

    Args:
        windows: Array of shape (n_windows, window_size).
        labels: Array of shape (n_windows,).
        features: Optional features array.
        train_ratio: Proportion of data for training.
        val_ratio: Proportion of data for validation.

    Returns:
        Dictionary with 'train', 'val', 'test' keys, each containing
        tuple of (windows, labels) or (windows, labels, features).
    """
    n_samples = len(windows)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    splits = {}

    if features is not None:
        splits["train"] = (windows[:train_end], labels[:train_end], features[:train_end])
        splits["val"] = (windows[train_end:val_end], labels[train_end:val_end], features[train_end:val_end])
        splits["test"] = (windows[val_end:], labels[val_end:], features[val_end:])
    else:
        splits["train"] = (windows[:train_end], labels[:train_end])
        splits["val"] = (windows[train_end:val_end], labels[train_end:val_end])
        splits["test"] = (windows[val_end:], labels[val_end:])

    return splits


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Computes class weights for handling imbalanced data.

    Args:
        labels: Binary labels array.

    Returns:
        Tensor with weight for the positive (anomaly) class.
    """
    n_normal = (labels == 0).sum()
    n_anomaly = (labels == 1).sum()

    if n_anomaly == 0:
        return torch.tensor(1.0)

    # Weight anomalies higher to compensate for imbalance
    weight = n_normal / n_anomaly
    return torch.tensor(weight, dtype=torch.float32)


def get_dataloaders(
    dataset_names: Optional[List[str]] = None,
    window_size: int = 50,
    step_size: int = 10,
    batch_size: int = 32,
    normalise: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Creates train/val/test DataLoaders from NAB datasets.

    Args:
        dataset_names: List of NAB dataset names. Uses defaults if None.
        window_size: Number of timesteps per window.
        step_size: Step between consecutive windows.
        batch_size: Batch size for DataLoaders.
        normalise: Whether to normalise windows.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_weight).
    """
    if dataset_names is None:
        dataset_names = DEFAULT_DATASETS

    all_windows = []
    all_labels = []

    for dataset_name in dataset_names:
        df, anomaly_timestamps = load_nab_data(dataset_name)
        values = df["value"].values.astype(np.float32)
        labels = create_anomaly_mask(df, anomaly_timestamps)

        windows, window_labels = create_windows(
            values, labels, window_size=window_size, step_size=step_size
        )

        if normalise:
            windows = normalise_windows(windows)

        all_windows.append(windows)
        all_labels.append(window_labels)

    # Concatenate all datasets
    windows = np.concatenate(all_windows, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # Split temporally
    splits = temporal_train_val_test_split(windows, labels)

    # Compute class weights from training data
    class_weight = compute_class_weights(splits["train"][1])

    # Create datasets
    train_dataset = TimeSeriesDataset(splits["train"][0], splits["train"][1])
    val_dataset = TimeSeriesDataset(splits["val"][0], splits["val"][1])
    test_dataset = TimeSeriesDataset(splits["test"][0], splits["test"][1])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, class_weight
