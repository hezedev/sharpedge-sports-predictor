"""Evaluation package exports."""

from src.evaluation.metrics import MetricsCalculator

try:
    from src.evaluation.backtester import Backtester
except Exception:  # pragma: no cover - keep package import resilient to legacy issues
    Backtester = None  # type: ignore[assignment]

__all__ = ["Backtester", "MetricsCalculator"]
