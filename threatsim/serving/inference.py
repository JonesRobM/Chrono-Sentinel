"""
Model loading and scoring for the HTTP service.

Preprocessing here must match training exactly, and must be a pure function of
the incoming window: the service receives a bare list of values with no series
identity, so anything fitted per-series at training time would be
unreproducible at serve time. The two transforms are therefore

  1. per-window z-scoring of the sequence  (pure function of the window)
  2. the statistical feature vector, scaled by the FeatureScaler persisted in
     the checkpoint  (fitted on the training split, fixed thereafter)

Both are applied by `AnomalyScorer.preprocess`, which is the single place the
contract is expressed.
"""

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from threatsim.scaling import FeatureScaler
from threatsim.features import extract_window_features, get_feature_names
from threatsim.models import TimeSeriesTransformer, create_model, mc_dropout_predict

DEFAULT_CHECKPOINT = Path("outputs/best_model.pt")

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
    Loads a trained checkpoint and scores windows with MC Dropout uncertainty.

    Instances are safe to share across requests and across threads. Dropout is
    switched on once at load and left on, so scoring never mutates module
    state; the forward pass then reads parameters only, and PyTorch releases
    the GIL inside its kernels, so concurrent requests genuinely overlap.

    The alternative -- toggling dropout on and off around each prediction --
    would need a lock, because one request's exit would disable dropout inside
    another request's forward pass and silently collapse its uncertainty to
    zero. That lock would serialise all inference.
    """

    def __init__(
        self,
        model: TimeSeriesTransformer,
        scaler: Optional[FeatureScaler],
        config: Dict[str, Any],
        model_version: str,
        default_mc_samples: int = 30,
    ):
        """
        Args:
            model: Loaded model in eval mode.
            scaler: Feature scaler from the checkpoint, or None for a
                sequence-only model.
            config: Checkpoint config dictionary.
            model_version: Short content hash identifying the checkpoint.
            default_mc_samples: Passes used when a request does not specify.
        """
        self.model = model
        self.scaler = scaler
        self.config = config
        self.model_version = model_version
        self.default_mc_samples = default_mc_samples
        self.window_size: int = int(config["window_size"])
        self.feature_dim: int = int(config.get("feature_dim", 0))
        self.feature_names: List[str] = (
            scaler.feature_names if scaler is not None else get_feature_names()
        )
        # Enable dropout once, permanently. Every prediction this class makes
        # is a Monte Carlo one, so there is no deterministic path to protect.
        self.model.enable_mc_dropout()

    @classmethod
    def from_checkpoint(
        cls,
        path: Path = DEFAULT_CHECKPOINT,
        device: str = "cpu",
        default_mc_samples: int = 30,
        num_threads: Optional[int] = None,
    ) -> "AnomalyScorer":
        """
        Builds a scorer from a saved checkpoint.

        Args:
            path: Checkpoint path.
            device: Torch device string. CPU is the deployment target.
            default_mc_samples: Passes used when a request does not specify.
            num_threads: torch intra-op thread count. On a small shared
                container the default oversubscribes and adds latency; the
                service sets this explicitly.

        Returns:
            A ready AnomalyScorer.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint {path} not found. Train one with scripts/train.py"
            )

        if num_threads is not None:
            torch.set_num_threads(num_threads)

        payload = torch.load(path, map_location=device, weights_only=False)
        config = payload.get("config")
        if config is None:
            raise ValueError(
                f"Checkpoint {path} has no config block; it predates the "
                "serving layer and cannot be served. Retrain with scripts/train.py."
            )

        model = create_model(
            window_size=config["window_size"],
            d_model=config["d_model"],
            nhead=config.get("nhead", 4),
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            feature_dim=config.get("feature_dim", 0),
        )
        model.load_state_dict(payload["model_state_dict"])
        model.eval()

        scaler_payload = config.get("feature_scaler")
        scaler = FeatureScaler.from_dict(scaler_payload) if scaler_payload else None

        version = hashlib.sha256(path.read_bytes()).hexdigest()[:12]

        return cls(model, scaler, config, version, default_mc_samples)

    def preprocess(self, values: Sequence[float]) -> tuple:
        """
        Turns a raw window into the two model inputs.

        Args:
            values: Raw time-series values, exactly window_size of them.

        Returns:
            Tuple of (normalised sequence tensor, scaled feature tensor or None,
            scaled feature vector as a numpy array).

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

        # Per-window z-score: pure function of this window.
        mean = window.mean()
        std = window.std()
        sequence = (window - mean) / (std if std > 0 else 1.0)

        scaled_features = None
        feature_tensor = None
        if self.feature_dim > 0:
            raw_features = extract_window_features(window)
            scaled_features = (
                self.scaler.transform(raw_features.reshape(1, -1))[0]
                if self.scaler is not None
                else raw_features
            )
            feature_tensor = torch.from_numpy(
                np.ascontiguousarray(scaled_features.reshape(1, -1))
            ).float()

        sequence_tensor = torch.from_numpy(
            np.ascontiguousarray(sequence.reshape(1, -1))
        ).float()

        return sequence_tensor, feature_tensor, scaled_features

    def score(
        self,
        values: Sequence[float],
        mc_samples: Optional[int] = None,
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

        sequence, features, scaled_features = self.preprocess(values)

        # perf_counter around the forward pass only, so the reported
        # inference time excludes validation and JSON handling.
        began = time.perf_counter()
        mean, std = mc_dropout_predict(
            self.model,
            sequence,
            features,
            n_samples=samples,
            batched=True,
            manage_mode=False,
        )
        elapsed = time.perf_counter() - began

        score = float(mean.item())
        uncertainty = float(std.item())

        return ScoreResult(
            score=score,
            uncertainty_std=uncertainty,
            interval_lower=max(0.0, score - interval_sigma * uncertainty),
            interval_upper=min(1.0, score + interval_sigma * uncertainty),
            mc_samples=samples,
            scaled_features=(
                scaled_features
                if scaled_features is not None
                else np.zeros(len(self.feature_names), dtype=np.float32)
            ),
            inference_seconds=elapsed,
        )

    def info(self) -> Dict[str, str]:
        """Returns identifying metadata for /healthz and the metrics endpoint."""
        return {
            "model_version": self.model_version,
            "window_size": str(self.window_size),
            "feature_dim": str(self.feature_dim),
            "d_model": str(self.config.get("d_model", "")),
            "num_layers": str(self.config.get("num_layers", "")),
            "dropout": str(self.config.get("dropout", "")),
        }
