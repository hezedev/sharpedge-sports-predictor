"""
Lightweight basketball side-model helpers.

This gives NBA sides a small structural anchor using the feature cache signals
we already trust most: Elo, form, rest, and travel burden. The goal is to
reduce classifier overconfidence before a fuller possession / efficiency model
exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BasketballProbabilityView:
    home: float
    away: float

    def as_tuple(self) -> tuple[float, float]:
        return self.home, self.away


@dataclass(frozen=True)
class BasketballBlendDiagnostics:
    classifier: BasketballProbabilityView
    structural: Optional[BasketballProbabilityView]
    combined: BasketballProbabilityView
    disagreement: float
    model_weight: float
    structural_weight: float
    regime: str


def _safe_float(value: object, default: float = np.nan) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def normalize_two_probs(p_home: float, p_away: float) -> BasketballProbabilityView:
    home = max(0.0, float(p_home))
    away = max(0.0, float(p_away))
    total = home + away
    if total <= 0:
        return BasketballProbabilityView(0.5, 0.5)
    return BasketballProbabilityView(home / total, away / total)


class BasketballSideModel:
    def __init__(
        self,
        *,
        low_disagreement_model_weight: float = 0.68,
        medium_disagreement_model_weight: float = 0.54,
        high_disagreement_model_weight: float = 0.38,
    ) -> None:
        self.low_disagreement_model_weight = low_disagreement_model_weight
        self.medium_disagreement_model_weight = medium_disagreement_model_weight
        self.high_disagreement_model_weight = high_disagreement_model_weight

    def structural_probs_from_snapshot(self, snapshot: Optional[pd.Series]) -> Optional[BasketballProbabilityView]:
        if snapshot is None:
            return None

        base = _safe_float(snapshot.get("elo_win_prob"), np.nan)
        if not np.isfinite(base):
            return None

        form_diff = _safe_float(snapshot.get("form_diff"), 0.0)
        rest_diff = _safe_float(snapshot.get("rest_diff"), 0.0)
        away_travel_bucket = _safe_float(snapshot.get("away_travel_bucket"), 0.0)
        away_cross_country = _safe_float(snapshot.get("away_cross_country"), 0.0)
        away_crossed_2tz = _safe_float(snapshot.get("away_crossed_2tz"), 0.0)
        away_travel_tz_shift = _safe_float(snapshot.get("away_travel_tz_shift"), 0.0)
        pace_edge = _safe_float(snapshot.get("home_pace_vs_avg"), 0.0) - _safe_float(snapshot.get("away_pace_vs_avg"), 0.0)

        adjustment = (
            (form_diff * 0.10)
            + (rest_diff * 0.018)
            + (away_travel_bucket * 0.018)
            + (away_cross_country * 0.016)
            + (away_crossed_2tz * 0.014)
            + (away_travel_tz_shift * 0.006)
            + (pace_edge * 0.0008)
        )
        adjustment = float(np.clip(adjustment, -0.16, 0.16))
        home = float(np.clip(base + adjustment, 0.08, 0.92))
        away = 1.0 - home
        return normalize_two_probs(home, away)

    def combine_with_classifier_diagnostics(
        self,
        classifier_probs: tuple[float, float] | BasketballProbabilityView,
        structural_probs: Optional[tuple[float, float] | BasketballProbabilityView],
    ) -> BasketballBlendDiagnostics:
        if isinstance(classifier_probs, BasketballProbabilityView):
            classifier = classifier_probs
        else:
            classifier = normalize_two_probs(*classifier_probs)

        if structural_probs is None:
            return BasketballBlendDiagnostics(
                classifier=classifier,
                structural=None,
                combined=classifier,
                disagreement=0.0,
                model_weight=1.0,
                structural_weight=0.0,
                regime="classifier_only",
            )

        if isinstance(structural_probs, BasketballProbabilityView):
            structural = structural_probs
        else:
            structural = normalize_two_probs(*structural_probs)

        disagreement = abs(classifier.home - structural.home) + abs(classifier.away - structural.away)
        if disagreement <= 0.10:
            model_weight = self.low_disagreement_model_weight
        elif disagreement <= 0.18:
            model_weight = self.medium_disagreement_model_weight
        else:
            model_weight = self.high_disagreement_model_weight
        structural_weight = 1.0 - model_weight

        if disagreement <= 0.08:
            regime = "aligned"
        elif disagreement <= 0.16:
            regime = "balanced"
        elif structural_weight >= 0.50:
            regime = "structural_override"
        else:
            regime = "classifier_lean"

        combined = normalize_two_probs(
            (classifier.home * model_weight) + (structural.home * structural_weight),
            (classifier.away * model_weight) + (structural.away * structural_weight),
        )
        return BasketballBlendDiagnostics(
            classifier=classifier,
            structural=structural,
            combined=combined,
            disagreement=float(disagreement),
            model_weight=float(model_weight),
            structural_weight=float(structural_weight),
            regime=regime,
        )
