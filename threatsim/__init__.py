"""
Chrono-Sentinel: Time-series anomaly detection with transformers and uncertainty quantification.

This package provides tools for:
- Loading and preprocessing NAB (Numenta Anomaly Benchmark) data
- Extracting statistical features from time-series windows
- Training transformer-based anomaly classifiers
- Uncertainty quantification via Monte Carlo Dropout

The serving layer lives in threatsim.serving and is imported separately, so
this package stays importable without FastAPI installed.
"""

from threatsim.data import (
    DEFAULT_ANOMALY_WINDOW_FRAC,
    DEFAULT_DATASETS,
    DEFAULT_TEST_SERIES,
    DEFAULT_TRAIN_SERIES,
    DEFAULT_VAL_SERIES,
    FeatureScaler,
    TimeSeriesDataset,
    create_windows,
    get_dataloaders,
    load_nab_data,
    nab_anomaly_mask,
    normalise_windows,
    prepare_grouped_splits,
)
from threatsim.features import (
    extract_features,
    extract_window_features,
    get_feature_names,
)
from threatsim.models import (
    PositionalEncoding,
    TimeSeriesTransformer,
    create_model,
    mc_dropout_predict,
)
from threatsim.utils import (
    EarlyStopping,
    get_device,
    load_model,
    save_model,
    set_seed,
)

__version__ = "0.2.0"
__all__ = [
    # Data
    "DEFAULT_ANOMALY_WINDOW_FRAC",
    "DEFAULT_DATASETS",
    "DEFAULT_TRAIN_SERIES",
    "DEFAULT_VAL_SERIES",
    "DEFAULT_TEST_SERIES",
    "FeatureScaler",
    "TimeSeriesDataset",
    "create_windows",
    "get_dataloaders",
    "load_nab_data",
    "nab_anomaly_mask",
    "normalise_windows",
    "prepare_grouped_splits",
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
