"""
Tests for the population-drift reference and the PSI statistic.

The drift number is the repo's claim to "knowing when the model has gone
stale", so it has to be right in the two directions that matter: near zero
when nothing has changed, and large when the distribution has moved. A
statistic that is merely always small is indistinguishable from a broken one.
"""

import json

import numpy as np
import pytest

from threatsim.reference import (
    PSI_MODERATE_THRESHOLD,
    PSI_SIGNIFICANT_THRESHOLD,
    ReferenceProfile,
    bin_proportions,
    classify_psi,
    population_stability_index,
    quantile_bin_edges,
)

FEATURE_NAMES = [f"f{i}" for i in range(4)]


def make_profile(rng, n=4000, n_bins=10) -> ReferenceProfile:
    """Builds a reference from standard-normal features and uniform scores."""
    features = rng.normal(0, 1, (n, len(FEATURE_NAMES))).astype(np.float32)
    scores = rng.uniform(0, 1, n)
    return ReferenceProfile.build(features, FEATURE_NAMES, scores, n_bins=n_bins)


class TestBinning:
    """Bin edges must cover the whole real line and survive ties."""

    def test_edges_are_open_ended(self):
        edges = quantile_bin_edges(np.random.default_rng(0).normal(0, 1, 1000))
        assert edges[0] == -np.inf
        assert edges[-1] == np.inf

    def test_values_outside_the_reference_range_still_bin(self):
        reference = np.random.default_rng(0).normal(0, 1, 1000)
        edges = quantile_bin_edges(reference)
        proportions = bin_proportions(np.array([-1e9, 1e9]), edges)
        assert proportions.sum() == pytest.approx(1.0)
        assert proportions[0] == 0.5 and proportions[-1] == 0.5

    def test_heavily_tied_feature_collapses_duplicate_edges(self):
        """A near-constant feature must not produce zero-width bins."""
        tied = np.concatenate([np.zeros(900), np.ones(100)])
        edges = quantile_bin_edges(tied, n_bins=10)
        assert np.all(np.diff(edges) > 0)

    def test_proportions_sum_to_one(self):
        rng = np.random.default_rng(1)
        edges = quantile_bin_edges(rng.normal(0, 1, 1000))
        assert bin_proportions(rng.normal(0, 1, 500), edges).sum() == pytest.approx(1.0)

    def test_empty_sample_gives_uniform_proportions(self):
        edges = quantile_bin_edges(np.random.default_rng(0).normal(0, 1, 100))
        proportions = bin_proportions(np.array([]), edges)
        assert proportions.sum() == pytest.approx(1.0)


class TestPSI:
    """The statistic itself."""

    def test_identical_distributions_give_zero(self):
        p = np.full(10, 0.1)
        assert population_stability_index(p, p) == pytest.approx(0.0, abs=1e-9)

    def test_psi_is_non_negative(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            a = rng.dirichlet(np.ones(10))
            b = rng.dirichlet(np.ones(10))
            assert population_stability_index(a, b) >= 0

    def test_psi_is_symmetric(self):
        rng = np.random.default_rng(1)
        a, b = rng.dirichlet(np.ones(10)), rng.dirichlet(np.ones(10))
        assert population_stability_index(a, b) == pytest.approx(
            population_stability_index(b, a)
        )

    def test_psi_grows_with_separation(self):
        reference = np.full(10, 0.1)
        near = np.array([0.12, 0.11, 0.1, 0.1, 0.1, 0.1, 0.1, 0.09, 0.09, 0.09])
        far = np.array([0.8, 0.1, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert population_stability_index(near, reference) < population_stability_index(
            far, reference
        )

    def test_empty_bin_stays_finite(self):
        """A zero proportion must not produce an infinity."""
        live = np.array([0.5, 0.5] + [0.0] * 8)
        assert np.isfinite(population_stability_index(live, np.full(10, 0.1)))

    def test_classification_thresholds(self):
        assert classify_psi(0.05) == "stable"
        assert classify_psi(PSI_MODERATE_THRESHOLD + 0.01) == "moderate"
        assert classify_psi(PSI_SIGNIFICANT_THRESHOLD + 0.01) == "significant"


class TestReferenceProfile:
    """End-to-end behaviour of the profile."""

    def test_same_distribution_reads_as_stable(self):
        rng = np.random.default_rng(0)
        profile = make_profile(rng)
        live_features = rng.normal(0, 1, (2000, len(FEATURE_NAMES))).astype(np.float32)
        live_scores = rng.uniform(0, 1, 2000)
        drift = profile.drift(live_features, live_scores)
        for name, psi in drift.items():
            assert psi < PSI_MODERATE_THRESHOLD, f"{name} drifted without cause: {psi}"

    def test_shifted_distribution_reads_as_significant(self):
        rng = np.random.default_rng(0)
        profile = make_profile(rng)
        # Mean shifted by three standard deviations.
        live_features = rng.normal(3, 1, (2000, len(FEATURE_NAMES))).astype(np.float32)
        live_scores = rng.uniform(0, 1, 2000)
        drift = profile.drift(live_features, live_scores)
        for name in FEATURE_NAMES:
            assert drift[name] > PSI_SIGNIFICANT_THRESHOLD, f"{name} missed a 3-sigma shift"

    def test_score_drift_is_reported_separately(self):
        rng = np.random.default_rng(0)
        profile = make_profile(rng)
        live_features = rng.normal(0, 1, (2000, len(FEATURE_NAMES))).astype(np.float32)
        # Inputs unchanged, outputs collapsed to one end.
        live_scores = rng.uniform(0.9, 1.0, 2000)
        drift = profile.drift(live_features, live_scores)
        assert drift["__score__"] > PSI_SIGNIFICANT_THRESHOLD
        assert max(drift[name] for name in FEATURE_NAMES) < PSI_MODERATE_THRESHOLD

    def test_variance_change_alone_is_detected(self):
        """A shift need not move the mean to matter."""
        rng = np.random.default_rng(0)
        profile = make_profile(rng)
        live_features = rng.normal(0, 5, (2000, len(FEATURE_NAMES))).astype(np.float32)
        drift = profile.drift(live_features, rng.uniform(0, 1, 2000))
        assert max(drift[name] for name in FEATURE_NAMES) > PSI_MODERATE_THRESHOLD

    def test_round_trip_through_json(self, tmp_path):
        rng = np.random.default_rng(0)
        profile = make_profile(rng)
        path = tmp_path / "reference.json"
        profile.save(path)

        # Must be plain JSON: the artefact ships in the container image.
        json.loads(path.read_text())

        restored = ReferenceProfile.load(path)
        live = rng.normal(0, 1, (500, len(FEATURE_NAMES))).astype(np.float32)
        scores = rng.uniform(0, 1, 500)
        original = profile.drift(live, scores)
        reloaded = restored.drift(live, scores)
        for name in original:
            assert original[name] == pytest.approx(reloaded[name])

    def test_mismatched_feature_names_are_rejected(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="columns"):
            ReferenceProfile.build(
                rng.normal(0, 1, (100, 4)), ["a", "b"], rng.uniform(0, 1, 100)
            )
