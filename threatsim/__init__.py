"""
Chrono-Sentinel: Time-series anomaly detection with transformers and uncertainty quantification.

This package provides tools for:
- Loading and preprocessing NAB (Numenta Anomaly Benchmark) data
- Extracting statistical features from time-series windows
- Training transformer-based anomaly classifiers
- Uncertainty quantification via Monte Carlo Dropout

Attributes are resolved lazily (PEP 562). `from threatsim import create_model`
works exactly as before, but importing a submodule no longer executes every
other one. That matters for the container: `threatsim.data` imports pandas and
`threatsim.utils` imports matplotlib, neither of which the scoring service
uses, and eagerly importing them here put roughly 300 MB of the research stack
into the serving image. The serving layer imports `threatsim.scaling`,
`threatsim.features`, `threatsim.models` and `threatsim.reference`, which
between them need only numpy and torch.

The serving layer itself lives in threatsim.serving and is never imported
here, so this package stays importable without FastAPI installed.
"""

import importlib
from typing import Any

__version__ = "0.2.2"

# Public name -> submodule that defines it.
_EXPORTS = {
    # Data
    "DEFAULT_ANOMALY_WINDOW_FRAC": "threatsim.data",
    "DEFAULT_DATASETS": "threatsim.data",
    "DEFAULT_TRAIN_SERIES": "threatsim.data",
    "DEFAULT_VAL_SERIES": "threatsim.data",
    "DEFAULT_TEST_SERIES": "threatsim.data",
    "TimeSeriesDataset": "threatsim.data",
    "create_windows": "threatsim.data",
    "get_dataloaders": "threatsim.data",
    "load_nab_data": "threatsim.data",
    "nab_anomaly_mask": "threatsim.data",
    "normalise_windows": "threatsim.data",
    "prepare_grouped_splits": "threatsim.data",
    # Scaling (dependency-light, shared with the serving layer)
    "FeatureScaler": "threatsim.scaling",
    # Features
    "extract_features": "threatsim.features",
    "extract_window_features": "threatsim.features",
    "get_feature_names": "threatsim.features",
    # Models
    "TimeSeriesTransformer": "threatsim.models",
    "PositionalEncoding": "threatsim.models",
    "mc_dropout_predict": "threatsim.models",
    "create_model": "threatsim.models",
    # Drift
    "ReferenceProfile": "threatsim.reference",
    "population_stability_index": "threatsim.reference",
    # Utils
    "set_seed": "threatsim.utils",
    "get_device": "threatsim.utils",
    "save_model": "threatsim.utils",
    "load_model": "threatsim.utils",
    "EarlyStopping": "threatsim.utils",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Imports the defining submodule on first access (PEP 562)."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value  # cache, so later lookups skip __getattr__
    return value


def __dir__() -> list:
    """Makes tab completion and dir() show the lazy exports."""
    return sorted(set(__all__) | {"__version__"})
