"""
Model loading and scoring for the HTTP service.

No torch. The forward pass runs in numpy (threatsim.serving.forward), loaded
from the artefact `scripts/export_weights.py` writes. torch is 635 MB of image
for a 76k-parameter model whose serving path needs nothing but a forward pass,
and `tests/test_forward_parity.py` holds the two implementations to the same
numbers.

Preprocessing must match training exactly and must be a pure function of the
incoming window: the service receives a bare list of values with no series
identity, so anything fitted per-series at training time would be
unreproducible here. The two transforms are

  1. per-window z-scoring of the sequence  (pure function of the window)
  2. the statistical feature vector, scaled by the FeatureScaler exported
     alongside the weights  (fitted on the training split, fixed thereafter)

`AnomalyScorer.preprocess` is the single place that contract is expressed.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from threatsim.features import extract_window_features, get_feature_names
from threatsim.scaling import FeatureScaler
from threatsim.serving.forward import NumpyModel

DEFAULT_WEIGHTS = Path("outputs/model.npz")

# Bounds on the client-supplied MC sample count. An unbounded value is a
# trivial denial of service: cost is linear in the number of passes.
MIN_MC_SAMPLES = 2
MAX_MC_SAMPLES = 200


@dataclass
class ScoreResult:
    """Outcome of scoring one window."""

    score: float
    uncertainty_std: float
    interval_lower: float
    interval_upper: float
    mc_samples: int
    scaled_features: np.ndarray
    inference_seconds: float


class AnomalyScorer:
    """
    Loads exported weights and scores windows with MC Dropout uncertainty.

    Safe to share across requests and threads. Nothing here mutates model
    state: the numpy backend takes an explicit `training` flag and a random
    generator per call, so concurrent requests cannot interfere. The torch
    backend needed care on exactly this point, because toggling dropout
    modules is global to the model.
    """

    def __init__(
        self,
        model: NumpyModel,
        scaler: FeatureScaler | None,
        default_mc_samples: int = 30,
        seed: int | None = None,
    ):
        """
        Args:
            model: Loaded numpy model.
            scaler: Feature scaler exported with the weights, or None.
            default_mc_samples: Passes used when a request does not specify.
            seed: Optional seed for the dropout generator. Leave None in
                production; set it where reproducible scores are wanted.
        """
        self.model = model
        self.scaler = scaler
        self.default_mc_samples = default_mc_samples
        self.window_size: int = model.window_size
        self.feature_dim: int = model.feature_dim
        self.model_version: str = model.model_version
        self.feature_names: list[str] = (
            scaler.feature_names if scaler is not None else get_feature_names()
        )
        self._rng = np.random.default_rng(seed)

    @classmethod
    def from_weights(
        cls,
        path: Path = DEFAULT_WEIGHTS,
        default_mc_samples: int = 30,
        seed: int | None = None,
    ) -> "AnomalyScorer":
        """
        Builds a scorer from the exported .npz.

        Args:
            path: Path to the artefact from scripts/export_weights.py.
            default_mc_samples: Passes used when a request does not specify.
            seed: Optional seed for the dropout generator.

        Returns:
            A ready AnomalyScorer.
        """
        model = NumpyModel.from_npz(Path(path))
        scaler_payload = model.config.get("feature_scaler")
        scaler = FeatureScaler.from_dict(scaler_payload) if scaler_payload else None
        return cls(model, scaler, default_mc_samples, seed)

    def preprocess(self, values) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Turns a raw window into the two model inputs.

        Args:
            values: Raw time-series values, exactly window_size of them.

        Returns:
            Tuple of (normalised sequence, scaled feature vector or None),
            both shaped (1, ...).

        Raises:
            ValueError: If the window is the wrong length or not finite.
        """
        window = np.asarray(values, dtype=np.float32)

        if window.ndim != 1:
            raise ValueError("values must be a flat list of numbers")
        if window.shape[0] != self.window_size:
            raise ValueError(
                f"expected exactly {self.window_size} values, got {window.shape[0]}"
            )
        if not np.all(np.isfinite(window)):
            raise ValueError("values must all be finite (no NaN or infinity)")

        # Per-window z-score: a pure function of this window.
        std = window.std()
        sequence = (window - window.mean()) / (std if std > 0 else 1.0)

        scaled_features = None
        if self.feature_dim > 0:
            raw = extract_window_features(window)
            scaled_features = (
                self.scaler.transform(raw.reshape(1, -1))
                if self.scaler is not None
                else raw.reshape(1, -1).astype(np.float32)
            )

        return sequence.reshape(1, -1), scaled_features

    def score(
        self,
        values,
        mc_samples: int | None = None,
        interval_sigma: float = 2.0,
    ) -> ScoreResult:
        """
        Scores one window with a Monte Carlo Dropout uncertainty interval.

        Args:
            values: Raw time-series values, exactly window_size of them.
            mc_samples: Number of stochastic forward passes.
            interval_sigma: Half-width of the reported interval, in standard
                deviations. 2.0 is roughly a 95% interval on the MC samples.

        Returns:
            A ScoreResult.

        Raises:
            ValueError: On an invalid window or an out-of-range sample count.
        """
        samples = self.default_mc_samples if mc_samples is None else int(mc_samples)
        if not MIN_MC_SAMPLES <= samples <= MAX_MC_SAMPLES:
            raise ValueError(
                f"mc_samples must be between {MIN_MC_SAMPLES} and {MAX_MC_SAMPLES}"
            )

        sequence, features = self.preprocess(values)

        # Timed around the forward pass only, so the reported figure excludes
        # validation and JSON handling.
        began = time.perf_counter()
        mean, std = self.model.mc_dropout_predict(
            sequence, features, n_samples=samples, rng=self._rng
        )
        elapsed = time.perf_counter() - began

        score = float(mean[0])
        uncertainty = float(std[0])

        return ScoreResult(
            score=score,
            uncertainty_std=uncertainty,
            interval_lower=max(0.0, score - interval_sigma * uncertainty),
            interval_upper=min(1.0, score + interval_sigma * uncertainty),
            mc_samples=samples,
            scaled_features=(
                features[0]
                if features is not None
                else np.zeros(len(self.feature_names), dtype=np.float32)
            ),
            inference_seconds=elapsed,
        )

    def info(self) -> dict[str, str]:
        """Identifying metadata for /readyz and the metrics endpoint."""
        config: dict[str, Any] = self.model.config
        return {
            "model_version": self.model_version,
            "backend": "numpy",
            "window_size": str(self.window_size),
            "feature_dim": str(self.feature_dim),
            "d_model": str(config.get("d_model", "")),
            "num_layers": str(config.get("num_layers", "")),
            "dropout": str(config.get("dropout", "")),
        }
