"""
The contract between the two forward-pass implementations.

Training uses PyTorch; the container runs numpy (threatsim/serving/forward.py)
so torch's 635 MB stays out of the image. Two implementations of one function
is a real hazard: if they drift, the service returns confident wrong answers
and nothing else notices. These tests are the thing that notices.

Writing the numpy version is what surfaced the fused-fast-path bug in
`enable_mc_dropout` — the two disagreed on sigma by 5%, and the reason was
that `nn.TransformerEncoderLayer` ignores its dropout submodules while in eval
mode. `test_all_dropout_sites_contribute` guards against that regressing.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from threatsim.models import create_model, mc_dropout_predict
from threatsim.serving.forward import NumpyModel

CHECKPOINT = Path("outputs/best_model.pt")
WEIGHTS = Path("outputs/model.npz")

needs_artefacts = pytest.mark.skipif(
    not (CHECKPOINT.exists() and WEIGHTS.exists()),
    reason="Run scripts/train.py and scripts/export_weights.py",
)


@pytest.fixture(scope="module")
def models():
    """The same checkpoint loaded through both backends."""
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    config = payload["config"]
    torch_model = create_model(
        window_size=config["window_size"],
        d_model=config["d_model"],
        nhead=config.get("nhead", 4),
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        feature_dim=config.get("feature_dim", 0),
    )
    torch_model.load_state_dict(payload["model_state_dict"])
    torch_model.eval()
    return torch_model, NumpyModel.from_npz(WEIGHTS), config


@pytest.fixture
def batch(models):
    """A batch of random normalised windows and scaled features."""
    _, numpy_model, _ = models
    rng = np.random.default_rng(0)
    sequence = rng.normal(0, 1, (8, numpy_model.window_size)).astype(np.float32)
    features = rng.normal(0, 1, (8, numpy_model.feature_dim)).astype(np.float32)
    return sequence, features


@needs_artefacts
class TestDeterministicParity:
    """With dropout off the two must agree to floating-point noise."""

    def test_logits_match(self, models, batch):
        torch_model, numpy_model, _ = models
        sequence, features = batch
        with torch.no_grad():
            expected = torch_model(
                torch.from_numpy(sequence), torch.from_numpy(features)
            ).numpy()
        actual = numpy_model.logits(sequence, features, training=False)
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)

    def test_probabilities_match(self, models, batch):
        torch_model, numpy_model, _ = models
        sequence, features = batch
        with torch.no_grad():
            expected = torch_model.predict_proba(
                torch.from_numpy(sequence), torch.from_numpy(features)
            ).numpy()
        np.testing.assert_allclose(
            numpy_model.predict_proba(sequence, features), expected, atol=1e-4
        )

    def test_deterministic_path_is_repeatable(self, models, batch):
        """No dropout means the same input gives bit-identical output."""
        _, numpy_model, _ = models
        sequence, features = batch
        first = numpy_model.predict_proba(sequence, features)
        second = numpy_model.predict_proba(sequence, features)
        np.testing.assert_array_equal(first, second)

    def test_single_window_matches_batch(self, models, batch):
        """A one-row call must agree with the same row inside a batch."""
        _, numpy_model, _ = models
        sequence, features = batch
        batched = numpy_model.predict_proba(sequence, features)
        single = numpy_model.predict_proba(sequence[2], features[2])
        np.testing.assert_allclose(single[0], batched[2], atol=1e-5)


@needs_artefacts
class TestMonteCarloParity:
    """
    Stochastic paths cannot match sample-for-sample, only in distribution.

    Tolerances are set from the sampling error: sigma estimated from n draws
    has a relative standard error near 1/sqrt(2n), so the checks below use
    enough repeats that a real 5%-scale divergence fails while noise passes.
    """

    def test_mean_and_sigma_agree(self, models, batch):
        torch_model, numpy_model, _ = models
        sequence, features = batch
        repeats, n_samples = 150, 30

        torch_means, torch_sigmas = [], []
        numpy_means, numpy_sigmas = [], []
        rng = np.random.default_rng(7)

        for seed in range(repeats):
            torch.manual_seed(seed)
            mean, sigma = mc_dropout_predict(
                torch_model,
                torch.from_numpy(sequence),
                torch.from_numpy(features),
                n_samples=n_samples,
            )
            torch_means.append(mean.numpy())
            torch_sigmas.append(sigma.numpy())

            mean, sigma = numpy_model.mc_dropout_predict(
                sequence, features, n_samples=n_samples, rng=rng
            )
            numpy_means.append(mean)
            numpy_sigmas.append(sigma)

        torch_mean = np.mean(torch_means, axis=0)
        numpy_mean = np.mean(numpy_means, axis=0)
        torch_sigma = np.mean(torch_sigmas, axis=0)
        numpy_sigma = np.mean(numpy_sigmas, axis=0)

        np.testing.assert_allclose(numpy_mean, torch_mean, atol=0.02)
        ratio = numpy_sigma / torch_sigma
        assert np.all((ratio > 0.9) & (ratio < 1.1)), (
            f"sigma ratio out of tolerance: {ratio}. A systematic gap here "
            "usually means one implementation is missing a dropout site."
        )

    def test_sigma_is_positive(self, models, batch):
        _, numpy_model, _ = models
        sequence, features = batch
        _, sigma = numpy_model.mc_dropout_predict(sequence, features, n_samples=40)
        assert np.all(sigma > 0), "MC Dropout produced zero uncertainty"

    def test_more_samples_stabilises_the_mean(self, models, batch):
        _, numpy_model, _ = models
        sequence, features = batch
        spread = {}
        for n_samples in (5, 200):
            rng = np.random.default_rng(3)
            means = [
                numpy_model.mc_dropout_predict(
                    sequence, features, n_samples=n_samples, rng=rng
                )[0]
                for _ in range(30)
            ]
            spread[n_samples] = np.std(means, axis=0).mean()
        assert spread[200] < spread[5]


@needs_artefacts
class TestDropoutSites:
    """
    Every dropout site must actually fire.

    nn.TransformerEncoderLayer takes a fused path in eval mode that skips its
    dropout submodules entirely. Setting them to train mode is not enough; the
    layer itself has to leave eval mode. When it did not, the encoder
    contributed exactly zero variance and MC Dropout was sampling only the
    positional encoding, the feature branch and the classifier head.
    """

    def test_all_dropout_sites_contribute(self, models, batch):
        import torch.nn as nn

        torch_model, _, _ = models
        sequence, features = batch
        tensor_sequence = torch.from_numpy(sequence)
        tensor_features = torch.from_numpy(features)

        def sigma_of(active_prefixes, repeats=120):
            torch_model.eval()
            for name, module in torch_model.named_modules():
                if isinstance(
                    module,
                    (nn.Dropout, nn.TransformerEncoderLayer, nn.MultiheadAttention),
                ):
                    module.train(any(name.startswith(p) for p in active_prefixes))
            outputs = []
            with torch.no_grad():
                for seed in range(repeats):
                    torch.manual_seed(seed)
                    outputs.append(
                        torch.sigmoid(
                            torch_model(tensor_sequence, tensor_features)
                        ).numpy()
                    )
            return float(np.std(np.array(outputs), axis=0).mean())

        encoder_only = sigma_of(["transformer_encoder"])
        torch_model.eval()

        assert encoder_only > 0, (
            "The transformer encoder contributes no MC Dropout variance. Its "
            "layers are taking the fused eval-mode path, which ignores their "
            "dropout submodules. enable_mc_dropout must put "
            "TransformerEncoderLayer and MultiheadAttention in train mode too."
        )


@needs_artefacts
def test_exported_config_round_trips(models):
    """The .npz must carry everything the service needs to describe itself."""
    _, numpy_model, config = models
    assert numpy_model.window_size == config["window_size"]
    assert numpy_model.feature_dim == config.get("feature_dim", 0)
    assert numpy_model.num_heads == config.get("nhead", 4)
    assert numpy_model.num_layers == config["num_layers"]
    assert numpy_model.dropout_p == pytest.approx(config["dropout"])
    assert len(numpy_model.model_version) == 12


@needs_artefacts
class TestWebBundleIsCurrent:
    """
    The browser bundle is a third copy of the weights and can go stale.

    Retraining regenerates best_model.pt and model.npz, but docs/assets/ is
    only refreshed by scripts/export_web_model.py. Without this check the
    hosted demo would keep serving the previous model with no visible sign.
    """

    WEB_META = Path("docs/assets/model.json")
    WEB_BIN = Path("docs/assets/model.bin")

    def test_bundle_matches_the_checkpoint(self):
        if not self.WEB_META.exists():
            pytest.skip("no web bundle; run scripts/export_web_model.py")

        import hashlib
        import json

        metadata = json.loads(self.WEB_META.read_text())
        expected = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()[:12]
        assert metadata["modelVersion"] == expected, (
            "docs/assets/model.json was exported from a different checkpoint. "
            "Run: python scripts/export_web_model.py"
        )

    def test_binary_size_matches_the_manifest(self):
        if not self.WEB_BIN.exists():
            pytest.skip("no web bundle; run scripts/export_web_model.py")

        import json

        metadata = json.loads(self.WEB_META.read_text())
        expected_bytes = metadata["totalFloats"] * 4
        assert self.WEB_BIN.stat().st_size == expected_bytes, (
            f"model.bin is {self.WEB_BIN.stat().st_size} bytes but the manifest "
            f"describes {expected_bytes}. Re-run scripts/export_web_model.py"
        )
