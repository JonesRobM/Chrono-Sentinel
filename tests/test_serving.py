"""
Tests for the HTTP scoring service.

Covers the request contract, the failure modes that matter for a public
endpoint (oversized input, wrong window length, unbounded work), and the
metrics surface. The input-validation tests are security tests as much as
correctness ones: both the window length and the MC sample count multiply
into inference cost, so an unbounded value is a denial of service.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from threatsim.serving.inference import MAX_MC_SAMPLES, MIN_MC_SAMPLES

CHECKPOINT = Path("outputs/best_model.pt")

needs_checkpoint = pytest.mark.skipif(
    not CHECKPOINT.exists(),
    reason="No trained checkpoint; run scripts/train.py",
)


@pytest.fixture(scope="module")
def client():
    """A TestClient with the app's startup hooks run."""
    from threatsim.serving.app import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def window_size(client) -> int:
    """The window length the loaded model expects."""
    payload = client.get("/readyz").json()
    if not payload.get("ready"):
        pytest.skip(f"service not ready: {payload.get('detail')}")
    return int(payload["window_size"])


def make_window(window_size: int, seed: int = 0) -> list:
    """A plausible window of sensor-like values."""
    rng = np.random.default_rng(seed)
    return [float(v) for v in 85 + rng.normal(0, 1, window_size)]


def _diverse_window(window_size: int, index: int, rng) -> list:
    """One of six window shape families, cycled by index."""
    family = index % 6
    if family == 0:
        window = np.full(window_size, 85.0)
    elif family == 1:
        window = 85 + rng.normal(0, rng.uniform(0.2, 15), window_size)
    elif family == 2:
        window = np.linspace(rng.uniform(0, 50), rng.uniform(60, 200), window_size)
    elif family == 3:
        half = window_size // 2
        window = np.r_[
            np.full(half, 85.0), np.full(window_size - half, rng.uniform(0, 60))
        ]
    elif family == 4:
        window = 85 + rng.uniform(2, 20) * np.sin(
            np.arange(window_size) * rng.uniform(0.1, 1.5)
        )
    else:
        window = np.full(window_size, 85.0)
        window[rng.integers(0, window_size)] = rng.uniform(150, 400)
    return [float(v) for v in window]


class TestHealth:
    """Liveness and readiness are distinct probes."""

    def test_healthz_does_not_depend_on_the_model(self, client):
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    @needs_checkpoint
    def test_readyz_reports_what_is_loaded(self, client):
        response = client.get("/readyz")
        assert response.status_code == 200
        body = response.json()
        assert body["ready"] is True
        assert body["model_loaded"] is True
        assert body["window_size"] > 0
        assert body["model_version"]


@needs_checkpoint
class TestScoring:
    """The /score contract."""

    def test_valid_window_scores(self, client, window_size):
        response = client.post("/score", json={"values": make_window(window_size)})
        assert response.status_code == 200
        body = response.json()
        assert 0.0 <= body["score"] <= 1.0
        assert body["uncertainty"]["std"] >= 0.0
        assert body["mc_samples"] > 0
        assert body["model_version"]
        assert body["inference_ms"] > 0

    def test_interval_brackets_the_score_and_stays_in_range(self, client, window_size):
        body = client.post("/score", json={"values": make_window(window_size)}).json()
        interval = body["uncertainty"]
        assert 0.0 <= interval["lower"] <= body["score"] <= interval["upper"] <= 1.0

    def test_uncertainty_is_not_a_constant(self, client, window_size):
        """
        A flat sigma across very different inputs is the failure the previous
        checkpoint had: it returned sigma ~= 0.021 for everything, so the
        interval was decorative. This asserts the served model's uncertainty
        actually responds to its input.

        The threshold is a coefficient of variation of 0.10 over 48 windows
        spanning six shape families. The current checkpoint measures 0.29, so
        this fails on a genuine collapse rather than on sampling noise.
        """
        rng = np.random.default_rng(7)
        sigmas = []
        for index in range(48):
            sigmas.append(
                client.post(
                    "/score", json={"values": _diverse_window(window_size, index, rng)}
                ).json()["uncertainty"]["std"]
            )
        sigmas = np.array(sigmas)
        coefficient_of_variation = sigmas.std() / sigmas.mean()
        assert coefficient_of_variation > 0.10, (
            f"sigma is nearly constant (CV={coefficient_of_variation:.3f}); "
            "the uncertainty interval has collapsed"
        )

    def test_scores_differ_across_window_shapes(self, client, window_size):
        flat = client.post("/score", json={"values": [85.0] * window_size}).json()[
            "score"
        ]
        step = client.post(
            "/score",
            json={
                "values": [85.0] * (window_size // 2)
                + [20.0] * (window_size - window_size // 2)
            },
        ).json()["score"]
        assert abs(flat - step) > 0.05

    def test_mc_samples_is_honoured(self, client, window_size):
        body = client.post(
            "/score", json={"values": make_window(window_size), "mc_samples": 7}
        ).json()
        assert body["mc_samples"] == 7


@needs_checkpoint
class TestInputValidation:
    """Rejections must be explicit; nothing is silently reshaped."""

    def test_wrong_window_length_is_rejected(self, client, window_size):
        response = client.post("/score", json={"values": [1.0, 2.0, 3.0]})
        assert response.status_code == 422
        assert str(window_size) in response.json()["detail"]

    def test_oversized_payload_is_rejected_by_the_schema(self, client):
        response = client.post("/score", json={"values": [1.0] * 100_000})
        assert response.status_code == 422

    def test_excessive_mc_samples_is_rejected(self, client, window_size):
        response = client.post(
            "/score",
            json={"values": make_window(window_size), "mc_samples": MAX_MC_SAMPLES + 1},
        )
        assert response.status_code == 422

    def test_too_few_mc_samples_is_rejected(self, client, window_size):
        """A single pass has no standard deviation to report."""
        response = client.post(
            "/score",
            json={"values": make_window(window_size), "mc_samples": MIN_MC_SAMPLES - 1},
        )
        assert response.status_code == 422

    @pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e400"])
    def test_non_finite_values_are_rejected(self, client, window_size, literal):
        """
        Sent as a raw body: NaN and Infinity are not legal JSON, so a
        conforming client cannot encode them, but a hand-rolled one can and
        Python's parser accepts them. They must not reach the model.
        """
        body = (
            '{"values": [' + ",".join([literal] + ["85.0"] * (window_size - 1)) + "]}"
        )
        response = client.post(
            "/score", content=body, headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_missing_body_is_rejected(self, client):
        assert client.post("/score", json={}).status_code == 422

    def test_non_numeric_values_are_rejected(self, client, window_size):
        values = make_window(window_size)
        values[0] = "not a number"
        assert client.post("/score", json={"values": values}).status_code == 422


@needs_checkpoint
class TestMetrics:
    """The Prometheus surface."""

    def test_metrics_exposes_the_four_families(self, client, window_size):
        client.post("/score", json={"values": make_window(window_size)})
        text = client.get("/metrics").text
        for family in (
            "chrono_requests_total",
            "chrono_request_latency_seconds",
            "chrono_inference_latency_seconds",
            "chrono_anomaly_score",
            "chrono_uncertainty_std",
            "chrono_drift_buffer_observations",
        ):
            assert family in text, f"{family} missing from /metrics"

    def test_request_counter_increments(self, client, window_size):
        def count() -> float:
            for line in client.get("/metrics").text.splitlines():
                if line.startswith(
                    'chrono_requests_total{endpoint="/score",status="200"}'
                ):
                    return float(line.rsplit(" ", 1)[1])
            return 0.0

        before = count()
        client.post("/score", json={"values": make_window(window_size)})
        assert count() > before

    def test_failed_requests_are_counted_separately(self, client):
        client.post("/score", json={"values": [1.0, 2.0]})
        text = client.get("/metrics").text
        assert 'status="422"' in text

    def test_no_unbounded_label_from_an_unknown_route(self, client):
        """
        A 404 path is client-controlled. If it reached the endpoint label it
        would let any caller mint unbounded time series.
        """
        client.get("/this-route-does-not-exist-" + "x" * 50)
        text = client.get("/metrics").text
        assert "this-route-does-not-exist" not in text
        assert 'endpoint="unmatched"' in text

    def test_metrics_carries_no_request_payload(self, client, window_size):
        """Scores are histogrammed, never echoed."""
        sentinel = 12345.6789
        values = make_window(window_size)
        values[0] = sentinel
        client.post("/score", json={"values": values})
        assert str(sentinel) not in client.get("/metrics").text


@needs_checkpoint
class TestDrift:
    """The /drift view over the same statistic /metrics exposes."""

    def test_drift_reports_its_state(self, client):
        body = client.get("/drift").json()
        assert "features" in body
        assert "reference_loaded" in body
        assert body["minimum_observations"] > 0

    def test_drift_is_withheld_until_enough_observations(self, client, window_size):
        """PSI on a handful of points is binning noise, not a signal."""
        from threatsim.serving.app import state

        state.metrics.reset_buffer()
        body = client.get("/drift").json()
        if body["reference_loaded"]:
            assert body["features"] == {}

    def test_drift_appears_once_the_buffer_fills(self, client, window_size):
        from threatsim.serving.app import state

        if state.reference is None:
            pytest.skip("no reference profile; run scripts/build_reference.py")

        state.metrics.reset_buffer()
        rng = np.random.default_rng(3)
        for _ in range(state.metrics.min_drift_samples + 10):
            client.post(
                "/score",
                json={"values": [float(v) for v in 85 + rng.normal(0, 1, window_size)]},
            )

        body = client.get("/drift").json()
        assert body["observations_in_buffer"] >= state.metrics.min_drift_samples
        assert body["features"], "drift should be reported once the buffer fills"
        for entry in body["features"].values():
            assert entry["status"] in {"stable", "moderate", "significant"}
            assert entry["psi"] >= 0


class TestImportFootprint:
    """
    The serving import chain must stay light.

    pandas, matplotlib, scikit-learn, scipy and seaborn are training and
    evaluation dependencies. When `threatsim/__init__.py` imported everything
    eagerly, importing the app pulled matplotlib and pandas in, and the
    container had to install the whole research stack: 1.87 GB versus 1.35 GB.
    A module-level heavy import anywhere in the serving chain silently undoes
    that, so it is asserted rather than assumed.

    Run in a subprocess because the rest of this suite imports pandas, so
    sys.modules in-process says nothing about what the service alone needs.
    """

    FORBIDDEN = ("pandas", "matplotlib", "sklearn", "scipy", "seaborn")

    def test_service_does_not_import_the_research_stack(self):
        probe = (
            "import sys; import threatsim.serving.app; "
            f"bad=[m for m in {self.FORBIDDEN!r} if m in sys.modules]; "
            "print(','.join(bad))"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr[-2000:]
        leaked = [m for m in result.stdout.strip().split(",") if m]
        assert not leaked, (
            f"the serving import chain now pulls in {leaked}, which adds "
            "hundreds of MB to the container image"
        )

    def test_scaler_is_importable_without_pandas_or_torch_dataloaders(self):
        """FeatureScaler must stay in the dependency-light module."""
        probe = (
            "import sys; from threatsim.scaling import FeatureScaler; "
            "print('pandas' in sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert result.stdout.strip() == "False"
