"""
FastAPI application exposing the Chrono-Sentinel anomaly detector.

Endpoints:
    POST /score    score one window, with an MC Dropout uncertainty interval
    GET  /healthz  liveness; deliberately does not touch the model
    GET  /readyz   readiness; reports what is actually loaded
    GET  /drift    human-readable view of the same PSI values /metrics exposes
    GET  /metrics  Prometheus text exposition

Liveness and readiness are separate on purpose. Loading the checkpoint takes
long enough that a combined probe would report a crash loop during a slow
cold start, which on a free-tier host is exactly when it happens.

Configuration is by environment variable so the container needs no arguments:
    CHRONO_MODEL_PATH      checkpoint to serve      (default outputs/best_model.pt)
    CHRONO_REFERENCE_PATH  drift reference profile  (default outputs/reference.json)
    CHRONO_MC_SAMPLES      default MC passes        (default 30)
    CHRONO_TORCH_THREADS   torch intra-op threads   (default 1)
    CHRONO_DRIFT_BUFFER    rolling drift window     (default 2000)
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from threatsim.reference import ReferenceProfile
from threatsim.serving.inference import AnomalyScorer
from threatsim.serving.metrics import ServiceMetrics
from threatsim.serving.schemas import (
    HealthResponse,
    ReadyResponse,
    ScoreRequest,
    ScoreResponse,
    UncertaintyInterval,
)

logger = logging.getLogger("chrono_sentinel")

SERVICE_NAME = "chrono-sentinel"
SERVICE_VERSION = "0.2.0"


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        logger.warning("%s is not an integer; using default %s", name, default)
        return default


class ServiceState:
    """Holds what the process loaded at startup."""

    def __init__(self) -> None:
        self.scorer: Optional[AnomalyScorer] = None
        self.reference: Optional[ReferenceProfile] = None
        self.metrics = ServiceMetrics(
            drift_buffer_size=_env_int("CHRONO_DRIFT_BUFFER", 2000)
        )
        self.load_error: Optional[str] = None

    @property
    def ready(self) -> bool:
        """The service can score only once the model is loaded."""
        return self.scorer is not None


state = ServiceState()


def load_resources() -> None:
    """
    Loads the checkpoint and drift reference into module state.

    A missing or unreadable checkpoint leaves the service live but not ready,
    with the reason recorded, rather than crashing the process: a readiness
    probe reporting a clear cause is more useful than a restart loop.
    """
    model_path = _env_path("CHRONO_MODEL_PATH", "outputs/best_model.pt")
    reference_path = _env_path("CHRONO_REFERENCE_PATH", "outputs/reference.json")

    try:
        state.scorer = AnomalyScorer.from_checkpoint(
            model_path,
            default_mc_samples=_env_int("CHRONO_MC_SAMPLES", 30),
            num_threads=_env_int("CHRONO_TORCH_THREADS", 1),
        )
        state.metrics.set_model_info(state.scorer.info())
        logger.info(
            "Loaded model %s (window_size=%d)",
            state.scorer.model_version,
            state.scorer.window_size,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced through /readyz
        state.load_error = f"model: {exc}"
        logger.error("Could not load model from %s: %s", model_path, exc)
        return

    if reference_path.exists():
        try:
            state.reference = ReferenceProfile.load(reference_path)
            state.metrics.set_reference(state.reference)
            logger.info(
                "Loaded drift reference built from %d windows",
                state.reference.n_reference_windows,
            )
        except Exception as exc:  # noqa: BLE001 - drift is optional
            logger.error("Could not load reference from %s: %s", reference_path, exc)
    else:
        logger.warning(
            "No drift reference at %s; PSI metrics will not be reported. "
            "Build one with scripts/build_reference.py",
            reference_path,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads resources on startup."""
    load_resources()
    yield


app = FastAPI(
    title="Chrono-Sentinel Scoring Service",
    description=(
        "Time-series anomaly detection with Monte Carlo Dropout uncertainty, "
        "instrumented with latency, throughput, score-distribution and "
        "population-drift metrics."
    ),
    version=SERVICE_VERSION,
    lifespan=lifespan,
)


@app.middleware("http")
async def record_request_metrics(request: Request, call_next):
    """Times every request and records it against a bounded endpoint label."""
    began = time.perf_counter()
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:
        state.metrics.observe_request(
            request.url.path, 500, time.perf_counter() - began
        )
        raise

    # request.url.path is client-controlled, so an unmatched route would let a
    # caller mint unbounded label values. Fall back to a constant.
    route = request.scope.get("route")
    endpoint = getattr(route, "path", None) or "unmatched"
    state.metrics.observe_request(endpoint, status, time.perf_counter() - began)
    return response


@app.get("/healthz", response_model=HealthResponse, tags=["operations"])
async def healthz() -> HealthResponse:
    """Liveness. Returns 200 as long as the process is serving."""
    return HealthResponse(status="ok", service=SERVICE_NAME, version=SERVICE_VERSION)


@app.get("/readyz", response_model=ReadyResponse, tags=["operations"])
async def readyz() -> JSONResponse:
    """Readiness. 503 until the model is loaded."""
    payload = ReadyResponse(
        ready=state.ready,
        model_loaded=state.scorer is not None,
        reference_loaded=state.reference is not None,
        model_version=state.scorer.model_version if state.scorer else None,
        window_size=state.scorer.window_size if state.scorer else None,
        default_mc_samples=state.scorer.default_mc_samples if state.scorer else None,
        detail=state.load_error,
    )
    return JSONResponse(
        status_code=200 if state.ready else 503,
        content=payload.model_dump(),
    )


@app.post("/score", response_model=ScoreResponse, tags=["scoring"])
def score(request: ScoreRequest) -> ScoreResponse:
    """
    Scores one window of time-series points.

    Deliberately a synchronous endpoint. The forward pass is CPU-bound and
    blocking; declaring it `async def` would run it directly on the event
    loop and stall every other request, including health probes, for its
    duration. As a sync endpoint FastAPI dispatches it to a worker thread,
    and because AnomalyScorer holds no lock, concurrent scores overlap.

    The window must contain exactly the number of points the loaded model was
    trained on; /readyz reports that number. A mismatch is rejected rather
    than padded or truncated, because silently reshaping the input would
    return a confident score for something the caller did not ask about.
    """
    if state.scorer is None:
        raise HTTPException(
            status_code=503,
            detail=state.load_error or "model not loaded",
        )

    try:
        result = state.scorer.score(request.values, request.mc_samples)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    state.metrics.observe_score(
        score=result.score,
        uncertainty=result.uncertainty_std,
        mc_samples=result.mc_samples,
        inference_seconds=result.inference_seconds,
        scaled_features=result.scaled_features,
    )

    return ScoreResponse(
        score=result.score,
        uncertainty=UncertaintyInterval(
            std=result.uncertainty_std,
            lower=result.interval_lower,
            upper=result.interval_upper,
        ),
        mc_samples=result.mc_samples,
        model_version=state.scorer.model_version,
        inference_ms=result.inference_seconds * 1000.0,
    )


@app.get("/drift", tags=["operations"])
async def drift() -> dict:
    """
    Human-readable view of the population-drift statistic.

    The same numbers appear in /metrics as chrono_drift_psi; this endpoint
    exists so the staleness story is legible without a Prometheus scrape.
    """
    summary = state.metrics.drift_summary()
    return {
        "reference_loaded": state.reference is not None,
        "reference_windows": (
            state.reference.n_reference_windows if state.reference else None
        ),
        "reference_created_at": state.reference.created_at if state.reference else None,
        "observations_in_buffer": state.metrics.buffer_size,
        "minimum_observations": state.metrics.min_drift_samples,
        "features": summary,
        "note": (
            "PSI below 0.1 is stable, 0.1-0.25 moderate, above 0.25 significant. "
            "'__score__' is drift in the output distribution; the rest are input "
            "features. Empty until the buffer reaches the minimum."
        ),
    }


@app.get("/metrics", tags=["operations"])
async def metrics() -> Response:
    """Prometheus text exposition. Drift is recomputed at scrape time."""
    state.metrics.refresh_drift()
    return Response(
        content=generate_latest(state.metrics.registry),
        media_type=CONTENT_TYPE_LATEST,
    )
