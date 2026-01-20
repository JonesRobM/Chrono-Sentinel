"""
Chrono-Sentinel: Time-series anomaly detection with transformers and uncertainty quantification.

This package provides tools for:
- Loading and preprocessing NAB (Numenta Anomaly Benchmark) data
- Extracting statistical features from time-series windows
- Training transformer-based anomaly classifiers
- Uncertainty quantification via Monte Carlo Dropout
"""

from threatsim.data import (
    load_nab_data,
    create_windows,
    get_dataloaders,
    TimeSeriesDataset,
)
from threatsim.features import (
    extract_features,
    extract_window_features,
    get_feature_names,
)
from threatsim.models import (
    TimeSeriesTransformer,
    PositionalEncoding,
    mc_dropout_predict,
    create_model,
)
from threatsim.utils import (
    set_seed,
    get_device,
    save_model,
    load_model,
    EarlyStopping,
)

__version__ = "0.1.0"
__all__ = [
    # Data
    "load_nab_data",
    "create_windows",
    "get_dataloaders",
    "TimeSeriesDataset",
    # Features
    "extract_features",
    "extract_window_features",
    "get_feature_names",
    # Models
    "TimeSeriesTransformer",
    "PositionalEncoding",
    "mc_dropout_predict",
    "create_model",
    # Utils
    "set_seed",
    "get_device",
    "save_model",
    "load_model",
    "EarlyStopping",
]
