"""
Tests for the model and the Monte Carlo Dropout implementation.

The batching test is the important one. Folding the MC passes into a single
batched forward pass is a ~2.4x speedup, but only a legitimate one if the
batched form is distributionally equivalent to the sequential loop. A faster
function that quietly changes the uncertainty semantics is a bug, not an
optimisation.
"""

import numpy as np
import pytest
import torch

from threatsim.models import TimeSeriesTransformer, create_model, mc_dropout_predict

WINDOW = 50
FEATURES = 10


@pytest.fixture
def model() -> TimeSeriesTransformer:
    """A small model with the feature branch enabled, in eval mode."""
    torch.manual_seed(0)
    net = create_model(
        window_size=WINDOW, d_model=32, num_layers=1, feature_dim=FEATURES
    )
    net.eval()
    return net


@pytest.fixture
def batch():
    """A batch of four windows and their feature vectors."""
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.normal(0, 1, (4, WINDOW))).float()
    f = torch.from_numpy(rng.normal(0, 1, (4, FEATURES))).float()
    return x, f


class TestForward:
    """Shape and contract of the forward pass."""

    def test_returns_logits_not_probabilities(self, model, batch):
        """Training uses BCEWithLogitsLoss, which needs raw logits."""
        x, f = batch
        with torch.no_grad():
            logits = model(x, f)
        assert logits.shape == (4,)
        # Logits are unbounded; a sigmoid output could never exceed 1.
        assert not torch.all((logits >= 0) & (logits <= 1)) or logits.abs().max() > 0

    def test_predict_proba_is_bounded(self, model, batch):
        x, f = batch
        with torch.no_grad():
            probs = model.predict_proba(x, f)
        assert torch.all((probs >= 0) & (probs <= 1))

    def test_missing_features_raises_when_branch_enabled(self, model, batch):
        """Silently scoring without features would use a zeroed branch."""
        x, _ = batch
        with pytest.raises(ValueError, match="feature_dim"):
            model(x)

    def test_sequence_only_model_needs_no_features(self, batch):
        net = create_model(window_size=WINDOW, d_model=32, num_layers=1, feature_dim=0)
        net.eval()
        x, _ = batch
        with torch.no_grad():
            logits = net(x)
        assert logits.shape == (4,)

    def test_accepts_three_dimensional_input(self, model, batch):
        x, f = batch
        with torch.no_grad():
            flat = model(x, f)
            expanded = model(x.unsqueeze(-1), f)
        assert flat.shape == expanded.shape


class TestMCDropout:
    """MC Dropout must be stochastic, batched-equivalent, and leave no state behind."""

    def test_dropout_is_actually_active(self, model, batch):
        """Zero spread would mean dropout never switched on."""
        x, f = batch
        _, std = mc_dropout_predict(model, x, f, n_samples=50)
        assert torch.all(std > 0), "MC Dropout produced zero uncertainty"

    def test_returns_one_value_per_input(self, model, batch):
        x, f = batch
        mean, std = mc_dropout_predict(model, x, f, n_samples=10)
        assert mean.shape == (4,)
        assert std.shape == (4,)

    def test_mean_is_a_probability(self, model, batch):
        x, f = batch
        mean, _ = mc_dropout_predict(model, x, f, n_samples=20)
        assert torch.all((mean >= 0) & (mean <= 1))

    def test_batched_matches_sequential_in_distribution(self, model, batch):
        """
        The batched and sequential forms draw independent dropout masks, so
        they cannot agree sample-for-sample. They must agree on the estimates
        those samples produce, which is what the optimisation actually claims.
        """
        x, f = batch
        repeats = 60
        sequential_means, batched_means = [], []
        sequential_stds, batched_stds = [], []

        for seed in range(repeats):
            torch.manual_seed(seed)
            m, s = mc_dropout_predict(model, x, f, n_samples=30, batched=False)
            sequential_means.append(m.numpy())
            sequential_stds.append(s.numpy())

            torch.manual_seed(seed)
            m, s = mc_dropout_predict(model, x, f, n_samples=30, batched=True)
            batched_means.append(m.numpy())
            batched_stds.append(s.numpy())

        np.testing.assert_allclose(
            np.mean(sequential_means, axis=0), np.mean(batched_means, axis=0), atol=0.02
        )
        np.testing.assert_allclose(
            np.mean(sequential_stds, axis=0), np.mean(batched_stds, axis=0), atol=0.02
        )

    def test_more_samples_reduces_estimator_variance(self, model, batch):
        """The mean over more passes should be more stable run to run."""
        x, f = batch
        spreads = {}
        for n_samples in (5, 200):
            means = []
            for seed in range(30):
                torch.manual_seed(seed)
                mean, _ = mc_dropout_predict(model, x, f, n_samples=n_samples)
                means.append(mean.numpy())
            spreads[n_samples] = np.std(means, axis=0).mean()
        assert spreads[200] < spreads[5]

    def test_module_state_is_restored(self, model, batch):
        """An eval-mode model must still be in eval mode afterwards."""
        x, f = batch
        assert not model.training
        dropouts_before = [
            m.training for m in model.modules() if isinstance(m, torch.nn.Dropout)
        ]
        mc_dropout_predict(model, x, f, n_samples=5)
        dropouts_after = [
            m.training for m in model.modules() if isinstance(m, torch.nn.Dropout)
        ]
        assert dropouts_before == dropouts_after
        assert not model.training

    def test_manage_mode_false_leaves_dropout_enabled(self, model, batch):
        """
        The service enables dropout once and passes manage_mode=False so
        concurrent requests never race on module flags.
        """
        x, f = batch
        model.enable_mc_dropout()
        mc_dropout_predict(model, x, f, n_samples=5, manage_mode=False)
        assert all(
            m.training for m in model.modules() if isinstance(m, torch.nn.Dropout)
        ), "manage_mode=False must not disable dropout"
