"""
Feature extraction for time-series windows.

This module provides statistical feature extraction routines for
time-series anomaly detection. Features are computed per window
and can be used alongside raw sequences in the transformer model.
"""

import numpy as np
from typing import Tuple


def compute_slope(window: np.ndarray) -> float:
    """
    Computes the linear trend (slope) of a window using least squares.

    Args:
        window: 1D array of values.

    Returns:
        Slope coefficient.
    """
    n = len(window)
    x = np.arange(n)
    # Least squares slope: sum((x - x_mean)(y - y_mean)) / sum((x - x_mean)^2)
    x_mean = x.mean()
    y_mean = window.mean()
    numerator = ((x - x_mean) * (window - y_mean)).sum()
    denominator = ((x - x_mean) ** 2).sum()
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_zero_crossings(window: np.ndarray) -> int:
    """
    Counts the number of zero crossings (sign changes) in a window.

    Useful for detecting oscillatory behaviour.

    Args:
        window: 1D array of values.

    Returns:
        Number of zero crossings.
    """
    # Centre around mean to detect crossings relative to average
    centred = window - window.mean()
    signs = np.sign(centred)
    # Count sign changes (excluding zeros)
    crossings = np.sum(signs[1:] != signs[:-1])
    return int(crossings)


def compute_autocorrelation(window: np.ndarray, lag: int = 1) -> float:
    """
    Computes autocorrelation at a given lag.

    Args:
        window: 1D array of values.
        lag: Lag for autocorrelation computation.

    Returns:
        Autocorrelation coefficient at the specified lag.
    """
    if len(window) <= lag:
        return 0.0

    y1 = window[:-lag]
    y2 = window[lag:]

    mean = window.mean()
    var = window.var()

    if var == 0:
        return 0.0

    return ((y1 - mean) * (y2 - mean)).mean() / var


def extract_window_features(window: np.ndarray) -> np.ndarray:
    """
    Extracts statistical features from a single window.

    Features computed:
    - mean: Average value
    - std: Standard deviation
    - min: Minimum value
    - max: Maximum value
    - range: Max - Min
    - slope: Linear trend
    - skewness: Asymmetry of distribution
    - kurtosis: Tailedness of distribution
    - zero_crossings: Number of sign changes around mean
    - autocorr_1: Lag-1 autocorrelation

    Args:
        window: 1D array of values.

    Returns:
        1D array of features.
    """
    mean = window.mean()
    std = window.std()
    min_val = window.min()
    max_val = window.max()
    range_val = max_val - min_val
    slope = compute_slope(window)
    zero_crossings = compute_zero_crossings(window)
    autocorr = compute_autocorrelation(window, lag=1)

    # Skewness: measure of asymmetry
    if std > 0:
        skewness = ((window - mean) ** 3).mean() / (std ** 3)
    else:
        skewness = 0.0

    # Kurtosis: measure of tailedness (excess kurtosis, so normal = 0)
    if std > 0:
        kurtosis = ((window - mean) ** 4).mean() / (std ** 4) - 3.0
    else:
        kurtosis = 0.0

    features = np.array([
        mean,
        std,
        min_val,
        max_val,
        range_val,
        slope,
        skewness,
        kurtosis,
        zero_crossings,
        autocorr,
    ], dtype=np.float32)

    return features


def extract_features(windows: np.ndarray) -> np.ndarray:
    """
    Extracts features from multiple windows.

    Args:
        windows: Array of shape (n_windows, window_size).

    Returns:
        Array of shape (n_windows, n_features).
    """
    n_windows = windows.shape[0]
    # Get number of features from a sample extraction
    sample_features = extract_window_features(windows[0])
    n_features = len(sample_features)

    features = np.zeros((n_windows, n_features), dtype=np.float32)

    for i in range(n_windows):
        features[i] = extract_window_features(windows[i])

    return features


def normalise_features(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalises features to zero mean and unit variance.

    Args:
        features: Array of shape (n_samples, n_features).

    Returns:
        Tuple of (normalised_features, mean, std) for later inverse transform.
    """
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    # Avoid division by zero
    std = np.where(std == 0, 1.0, std)

    normalised = (features - mean) / std

    return normalised, mean, std


def get_feature_names() -> list:
    """
    Returns the names of extracted features in order.

    Returns:
        List of feature names.
    """
    return [
        "mean",
        "std",
        "min",
        "max",
        "range",
        "slope",
        "skewness",
        "kurtosis",
        "zero_crossings",
        "autocorr_1",
    ]
