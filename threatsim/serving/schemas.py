"""
Request and response models for the scoring API.

Validation limits here are load-bearing, not cosmetic. Both the window length
and the MC sample count multiply directly into inference cost, so an
unbounded value from a client is a denial of service. The window length is
additionally checked against the checkpoint at request time, because the
correct length is a property of the loaded model rather than of the schema.
"""

from pydantic import BaseModel, Field

# Absolute ceiling independent of the loaded model. The per-request check
# against the checkpoint's window_size is stricter; this exists so an
# oversized payload is rejected before any array is allocated.
MAX_WINDOW_VALUES = 4096


class ScoreRequest(BaseModel):
    """A window of time-series points to score."""

    values: list[float] = Field(
        ...,
        min_length=2,
        max_length=MAX_WINDOW_VALUES,
        description=(
            "Raw time-series values, most recent last. Must contain exactly "
            "the window size the loaded model was trained with; query /healthz "
            "for that value."
        ),
    )
    mc_samples: int | None = Field(
        None,
        ge=2,
        le=200,
        description=(
            "Monte Carlo Dropout forward passes. More passes give a smoother "
            "uncertainty estimate at proportionally higher latency. Defaults "
            "to the server's configured value."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"values": [42.1, 41.8, 43.0, 44.7], "mc_samples": 30}]
        }
    }


class UncertaintyInterval(BaseModel):
    """Monte Carlo Dropout spread around the point estimate."""

    std: float = Field(..., description="Standard deviation across MC samples")
    lower: float = Field(
        ..., description="Point estimate minus two standard deviations, clipped to 0"
    )
    upper: float = Field(
        ..., description="Point estimate plus two standard deviations, clipped to 1"
    )


class ScoreResponse(BaseModel):
    """Anomaly score with its uncertainty interval and provenance."""

    score: float = Field(..., description="Mean anomaly probability across MC samples")
    uncertainty: UncertaintyInterval
    mc_samples: int = Field(..., description="Number of stochastic passes actually run")
    model_version: str = Field(
        ..., description="Short content hash of the served checkpoint"
    )
    inference_ms: float = Field(
        ..., description="Server-side forward-pass time in milliseconds"
    )


class HealthResponse(BaseModel):
    """Liveness payload. Deliberately does not touch the model."""

    status: str
    service: str
    version: str


class ReadyResponse(BaseModel):
    """Readiness payload, reporting what the service has actually loaded."""

    ready: bool
    model_loaded: bool
    reference_loaded: bool
    model_version: str | None = None
    window_size: int | None = None
    default_mc_samples: int | None = None
    detail: str | None = None


class ErrorResponse(BaseModel):
    """Structured error body."""

    detail: str
