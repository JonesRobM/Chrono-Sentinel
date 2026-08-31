"""
HTTP scoring service for the Chrono-Sentinel anomaly detector.

Importing this subpackage requires the serving extras (FastAPI, uvicorn,
prometheus-client). The research package `threatsim` itself does not, so the
training path stays lightweight:

    pip install -e ".[serve]"
"""

from threatsim.serving.inference import AnomalyScorer, ScoreResult

__all__ = ["AnomalyScorer", "ScoreResult"]
