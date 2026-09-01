"""
Feature scaling, kept free of heavy dependencies.

`FeatureScaler` lives here rather than in `threatsim.data` because the serving
container needs it and needs nothing else from that module. `data.py` imports
pandas and `torch.utils.data`; `utils.py` imports matplotlib. Importing the
scaler from either would drag the whole research stack into the image for a
transform that is ten means and ten standard deviations.

This module depends on numpy and `threatsim.features` only.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from threatsim.features import get_feature_names


@dataclass
class FeatureScaler:
    """
    Zero-mean, unit-variance scaler for the statistical feature vector.

    Fitted on the training split only and persisted in the model checkpoint,
    so training and serving apply exactly the same transform.
    """

    mean: np.ndarray
    std: np.ndarray
    feature_names: list[str]

    @classmethod
    def fit(cls, features: np.ndarray) -> "FeatureScaler":
        """Fits the scaler to a (n_samples, n_features) array."""
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std = np.where(std == 0, 1.0, std)
        return cls(mean=mean, std=std, feature_names=get_feature_names())

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Applies the fitted transform."""
        return ((features - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        """Serialises to plain lists for checkpointing."""
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "feature_names": self.feature_names,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeatureScaler":
        """Restores a scaler from its serialised form."""
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
            feature_names=list(payload["feature_names"]),
        )
