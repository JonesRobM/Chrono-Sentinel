"""
Population-drift reference for the scoring service.

A deployed detector degrades silently: the model keeps returning scores long
after the traffic stops resembling anything it was trained on. This module
builds a reference profile from the **training split only** and provides the
statistic used to compare live traffic against it.

The statistic is the Population Stability Index (PSI). For a quantity binned
into k bins:

    PSI = sum_i (live_i - ref_i) * ln(live_i / ref_i)

where live_i and ref_i are the proportions of observations falling in bin i.
It is symmetric, zero when the distributions match, and grows without bound as
they separate. The conventional reading, which the service exposes as
thresholds rather than baking into an alert:

    PSI < 0.10   no meaningful shift
    0.10 - 0.25  moderate shift, worth investigating
    PSI > 0.25   significant shift, the model is likely stale

The reference is never rebuilt from live traffic. Doing so would let the
profile drift along with the data and report stability while the model
quietly goes stale, which is the exact failure this is meant to catch.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# Proportions below this are floored before the log, so an empty bin gives a
# large but finite contribution rather than an infinity.
EPSILON = 1e-6

DEFAULT_N_BINS = 10

PSI_MODERATE_THRESHOLD = 0.10
PSI_SIGNIFICANT_THRESHOLD = 0.25


def quantile_bin_edges(values: np.ndarray, n_bins: int = DEFAULT_N_BINS) -> np.ndarray:
    """
    Computes quantile bin edges for a 1D reference sample.

    Edges are open at both ends (-inf, +inf) so live values outside the
    reference range still land in a bin instead of being dropped. Duplicate
    interior edges, which occur when a feature is heavily tied, are collapsed.

    Args:
        values: 1D reference sample.
        n_bins: Target number of bins.

    Returns:
        Monotonically increasing edge array of length <= n_bins + 1.
    """
    quantiles = np.linspace(0, 100, n_bins + 1)[1:-1]
    interior = np.unique(np.percentile(values, quantiles))
    return np.concatenate([[-np.inf], interior, [np.inf]])


def bin_proportions(values: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """
    Computes the proportion of values falling in each bin.

    Args:
        values: 1D sample to bin.
        bin_edges: Edges from quantile_bin_edges.

    Returns:
        Array of proportions summing to 1, of length len(bin_edges) - 1.
    """
    counts, _ = np.histogram(values, bins=bin_edges)
    total = counts.sum()
    if total == 0:
        return np.full(len(counts), 1.0 / len(counts))
    return counts / total


def population_stability_index(
    live_proportions: np.ndarray, reference_proportions: np.ndarray
) -> float:
    """
    Computes PSI between a live and a reference distribution.

    Args:
        live_proportions: Proportions of live observations per bin.
        reference_proportions: Proportions of reference observations per bin.

    Returns:
        The PSI value. Zero means identical distributions.
    """
    live = np.clip(live_proportions, EPSILON, None)
    reference = np.clip(reference_proportions, EPSILON, None)
    return float(np.sum((live - reference) * np.log(live / reference)))


def classify_psi(psi: float) -> str:
    """Returns the conventional verbal reading of a PSI value."""
    if psi < PSI_MODERATE_THRESHOLD:
        return "stable"
    if psi < PSI_SIGNIFICANT_THRESHOLD:
        return "moderate"
    return "significant"


@dataclass
class ReferenceProfile:
    """
    Binned reference distributions for each model input feature and the score.

    Built by scripts/build_reference.py from the training split, persisted to
    JSON, and loaded by the serving layer at startup.
    """

    feature_names: List[str]
    feature_bin_edges: Dict[str, List[float]]
    feature_proportions: Dict[str, List[float]]
    score_bin_edges: List[float]
    score_proportions: List[float]
    n_reference_windows: int
    n_bins: int = DEFAULT_N_BINS
    created_at: str = ""
    source: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        features: np.ndarray,
        feature_names: Sequence[str],
        scores: np.ndarray,
        n_bins: int = DEFAULT_N_BINS,
        source: Optional[Dict[str, Any]] = None,
    ) -> "ReferenceProfile":
        """
        Builds a profile from training-split features and model scores.

        Args:
            features: Scaled feature matrix of shape (n_windows, n_features).
            feature_names: Names matching the feature columns.
            scores: Model anomaly scores for the same windows.
            n_bins: Number of quantile bins.
            source: Provenance metadata recorded in the artefact.

        Returns:
            A populated ReferenceProfile.
        """
        if features.shape[1] != len(feature_names):
            raise ValueError(
                f"features has {features.shape[1]} columns but "
                f"{len(feature_names)} names were given"
            )

        edges: Dict[str, List[float]] = {}
        proportions: Dict[str, List[float]] = {}
        for index, name in enumerate(feature_names):
            column = features[:, index]
            column_edges = quantile_bin_edges(column, n_bins)
            edges[name] = column_edges.tolist()
            proportions[name] = bin_proportions(column, column_edges).tolist()

        score_edges = quantile_bin_edges(scores, n_bins)

        return cls(
            feature_names=list(feature_names),
            feature_bin_edges=edges,
            feature_proportions=proportions,
            score_bin_edges=score_edges.tolist(),
            score_proportions=bin_proportions(scores, score_edges).tolist(),
            n_reference_windows=int(features.shape[0]),
            n_bins=n_bins,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            source=source or {},
        )

    def drift(self, live_features: np.ndarray, live_scores: np.ndarray) -> Dict[str, float]:
        """
        Computes PSI for every feature and for the score distribution.

        Args:
            live_features: Scaled feature matrix from recent requests.
            live_scores: Anomaly scores from the same requests.

        Returns:
            Mapping of feature name (plus "__score__") to PSI.
        """
        result: Dict[str, float] = {}

        for index, name in enumerate(self.feature_names):
            live = bin_proportions(
                live_features[:, index], np.asarray(self.feature_bin_edges[name])
            )
            result[name] = population_stability_index(
                live, np.asarray(self.feature_proportions[name])
            )

        live_score = bin_proportions(live_scores, np.asarray(self.score_bin_edges))
        result["__score__"] = population_stability_index(
            live_score, np.asarray(self.score_proportions)
        )

        return result

    def to_dict(self) -> Dict[str, Any]:
        """Serialises the profile to a JSON-compatible dict."""
        return {
            "feature_names": self.feature_names,
            "feature_bin_edges": self.feature_bin_edges,
            "feature_proportions": self.feature_proportions,
            "score_bin_edges": self.score_bin_edges,
            "score_proportions": self.score_proportions,
            "n_reference_windows": self.n_reference_windows,
            "n_bins": self.n_bins,
            "created_at": self.created_at,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ReferenceProfile":
        """Restores a profile from its serialised form."""
        return cls(**payload)

    def save(self, path: Path) -> None:
        """Writes the profile to a JSON file, creating parent directories."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "ReferenceProfile":
        """Reads a profile from a JSON file."""
        return cls.from_dict(json.loads(Path(path).read_text()))
