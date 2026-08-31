"""
Prometheus instrumentation for the scoring service.

Four things are exposed, matching the four questions an operator actually
asks of a deployed model:

  how much traffic   chrono_requests_total
  how fast           chrono_request_latency_seconds, chrono_inference_latency_seconds
  what it is saying  chrono_anomaly_score, chrono_uncertainty_std
  is it still valid  chrono_drift_psi

Label cardinality is kept deliberately small. Endpoint and status code are
bounded sets; the drift gauge is labelled by feature name, of which there are
eleven. Nothing is labelled by anything client-controlled, because a
per-request or per-client label would grow the series count without bound.

Latency histogram buckets are tuned for a CPU forward pass repeated
`mc_samples` times, which lands in the single-digit-to-tens-of-milliseconds
range. The default prometheus_client buckets start at 5 ms and would put
almost every observation in one or two buckets, making the percentiles
useless.
"""

import threading
from collections import deque
from typing import Deque, Dict, Optional

import numpy as np
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, Info

from threatsim.reference import ReferenceProfile, classify_psi

LATENCY_BUCKETS = (
    0.001, 0.0025, 0.005, 0.0075,
    0.01, 0.025, 0.05, 0.075,
    0.1, 0.25, 0.5, 1.0, 2.5,
    float("inf"),
)

SCORE_BUCKETS = tuple(np.round(np.linspace(0.0, 1.0, 11), 2).tolist()) + (float("inf"),)

UNCERTAINTY_BUCKETS = (
    0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5,
    float("inf"),
)

# How many recent requests the drift statistic is computed over. PSI on a
# handful of observations is dominated by binning noise, so the gauge stays
# unset until the buffer holds at least MIN_DRIFT_SAMPLES.
DEFAULT_DRIFT_BUFFER = 2000
MIN_DRIFT_SAMPLES = 50


class ServiceMetrics:
    """
    Owns the collectors and the rolling window of recent requests.

    A dedicated registry rather than the global default, so tests can build
    an isolated instance without duplicate-timeseries errors.
    """

    def __init__(
        self,
        registry: Optional[CollectorRegistry] = None,
        drift_buffer_size: int = DEFAULT_DRIFT_BUFFER,
        min_drift_samples: int = MIN_DRIFT_SAMPLES,
    ):
        """
        Args:
            registry: Collector registry to register into. A fresh one is
                created if omitted.
            drift_buffer_size: Number of recent requests retained for drift.
            min_drift_samples: Observations required before drift is reported.
        """
        self.registry = registry if registry is not None else CollectorRegistry()
        self.drift_buffer_size = drift_buffer_size
        self.min_drift_samples = min_drift_samples

        self.requests_total = Counter(
            "chrono_requests_total",
            "Requests handled, by endpoint and HTTP status class.",
            ["endpoint", "status"],
            registry=self.registry,
        )
        self.request_latency = Histogram(
            "chrono_request_latency_seconds",
            "End-to-end request handling time, including validation and serialisation.",
            ["endpoint"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.inference_latency = Histogram(
            "chrono_inference_latency_seconds",
            "Model forward-pass time only, excluding request handling.",
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.anomaly_score = Histogram(
            "chrono_anomaly_score",
            "Distribution of returned anomaly scores.",
            buckets=SCORE_BUCKETS,
            registry=self.registry,
        )
        self.uncertainty_std = Histogram(
            "chrono_uncertainty_std",
            "Distribution of Monte Carlo Dropout standard deviations.",
            buckets=UNCERTAINTY_BUCKETS,
            registry=self.registry,
        )
        self.mc_samples_total = Counter(
            "chrono_mc_forward_passes_total",
            "Cumulative stochastic forward passes executed.",
            registry=self.registry,
        )
        self.drift_psi = Gauge(
            "chrono_drift_psi",
            "Population Stability Index of recent traffic against the training "
            "reference. Below 0.1 stable, 0.1-0.25 moderate, above 0.25 significant.",
            ["feature"],
            registry=self.registry,
        )
        self.drift_buffer_fill = Gauge(
            "chrono_drift_buffer_observations",
            "Observations currently in the rolling drift buffer.",
            registry=self.registry,
        )
        self.drift_max_psi = Gauge(
            "chrono_drift_max_psi",
            "Largest PSI across all features, for a single-number staleness alert.",
            registry=self.registry,
        )
        self.model_info = Info(
            "chrono_model",
            "Identity of the served checkpoint.",
            registry=self.registry,
        )
        self.reference_info = Info(
            "chrono_reference",
            "Identity of the loaded drift reference profile.",
            registry=self.registry,
        )

        self._lock = threading.Lock()
        self._feature_buffer: Deque[np.ndarray] = deque(maxlen=drift_buffer_size)
        self._score_buffer: Deque[float] = deque(maxlen=drift_buffer_size)
        self._reference: Optional[ReferenceProfile] = None

    def set_reference(self, reference: Optional[ReferenceProfile]) -> None:
        """Attaches the reference profile that drift is measured against."""
        self._reference = reference
        if reference is not None:
            self.reference_info.info(
                {
                    "created_at": reference.created_at,
                    "n_reference_windows": str(reference.n_reference_windows),
                    "n_bins": str(reference.n_bins),
                    "model_version": str(reference.source.get("model_version", "")),
                }
            )

    def set_model_info(self, info: Dict[str, str]) -> None:
        """Records the served checkpoint's identity."""
        self.model_info.info(info)

    def observe_request(self, endpoint: str, status: int, seconds: float) -> None:
        """Records one handled request."""
        self.requests_total.labels(endpoint=endpoint, status=str(status)).inc()
        self.request_latency.labels(endpoint=endpoint).observe(seconds)

    def observe_score(
        self,
        score: float,
        uncertainty: float,
        mc_samples: int,
        inference_seconds: float,
        scaled_features: np.ndarray,
    ) -> None:
        """
        Records one scored window and adds it to the drift buffer.

        Args:
            score: Mean anomaly probability.
            uncertainty: MC Dropout standard deviation.
            mc_samples: Passes executed.
            inference_seconds: Forward-pass time.
            scaled_features: The scaled feature vector fed to the model.
        """
        self.inference_latency.observe(inference_seconds)
        self.anomaly_score.observe(score)
        self.uncertainty_std.observe(uncertainty)
        self.mc_samples_total.inc(mc_samples)

        with self._lock:
            self._feature_buffer.append(np.asarray(scaled_features, dtype=np.float32))
            self._score_buffer.append(float(score))

    def refresh_drift(self) -> Dict[str, float]:
        """
        Recomputes PSI over the rolling buffer and updates the gauges.

        Called at scrape time rather than per request: PSI over a few thousand
        observations costs a handful of histograms, which is wasteful to
        repeat on every request but negligible once per scrape.

        Returns:
            Mapping of feature name to PSI, empty if drift is not yet
            computable.
        """
        with self._lock:
            fill = len(self._score_buffer)
            self.drift_buffer_fill.set(fill)

            if self._reference is None or fill < self.min_drift_samples:
                return {}

            features = np.vstack(self._feature_buffer)
            scores = np.asarray(self._score_buffer, dtype=np.float64)

        psi = self._reference.drift(features, scores)
        for name, value in psi.items():
            self.drift_psi.labels(feature=name).set(value)
        self.drift_max_psi.set(max(psi.values()) if psi else 0.0)

        return psi

    def drift_summary(self) -> Dict[str, Dict[str, object]]:
        """Returns the current PSI values with their verbal classification."""
        psi = self.refresh_drift()
        return {
            name: {"psi": round(value, 4), "status": classify_psi(value)}
            for name, value in sorted(psi.items(), key=lambda kv: -kv[1])
        }

    @property
    def buffer_size(self) -> int:
        """Number of observations currently in the rolling drift buffer."""
        with self._lock:
            return len(self._score_buffer)

    def reset_buffer(self) -> None:
        """Clears the rolling drift buffer. Used by tests."""
        with self._lock:
            self._feature_buffer.clear()
            self._score_buffer.clear()
