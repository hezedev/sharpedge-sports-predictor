"""
Lightweight NHL side-model helpers.

This gives NHL sides a small structural anchor using feature-cache signals we
already trust most: Elo, xG form, special teams, rest, and travel burden.
The goal is to reduce classifier overconfidence before a fuller goalie/xG
model exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NHLProbabilityView:
    home: float
    away: float

    def as_tuple(self) -> tuple[float, float]:
        return self.home, self.away


@dataclass(frozen=True)
class NHLBlendDiagnostics:
    classifier: NHLProbabilityView
    structural: Optional[NHLProbabilityView]
    combined: NHLProbabilityView
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


def normalize_two_probs(p_home: float, p_away: float) -> NHLProbabilityView:
    home = max(0.0, float(p_home))
    away = max(0.0, float(p_away))
    total = home + away
    if total <= 0:
        return NHLProbabilityView(0.5, 0.5)
    return NHLProbabilityView(home / total, away / total)


class NHLSideModel:
    def __init__(
        self,
        *,
        low_disagreement_model_weight: float = 0.66,
        medium_disagreement_model_weight: float = 0.52,
        high_disagreement_model_weight: float = 0.36,
    ) -> None:
        self.low_disagreement_model_weight = low_disagreement_model_weight
        self.medium_disagreement_model_weight = medium_disagreement_model_weight
        self.high_disagreement_model_weight = high_disagreement_model_weight

    def structural_probs_from_snapshot(self, snapshot: Optional[pd.Series]) -> Optional[NHLProbabilityView]:
        if snapshot is None:
            return None

        base = _safe_float(snapshot.get("elo_win_prob"), np.nan)
        if not np.isfinite(base):
            return None

        xg_form_edge = _safe_float(snapshot.get("home_xg_diff_10"), 0.0) - _safe_float(snapshot.get("away_xg_diff_10"), 0.0)
        xgf_edge = _safe_float(snapshot.get("home_xgf_pg_10"), 0.0) - _safe_float(snapshot.get("away_xgf_pg_10"), 0.0)
        xga_edge = _safe_float(snapshot.get("away_xga_pg_10"), 0.0) - _safe_float(snapshot.get("home_xga_pg_10"), 0.0)
        pp_edge = _safe_float(snapshot.get("home_pp_pct_10"), 0.0) - _safe_float(snapshot.get("away_pp_pct_10"), 0.0)
        pk_edge = _safe_float(snapshot.get("home_pk_pct_10"), 0.0) - _safe_float(snapshot.get("away_pk_pct_10"), 0.0)
        rest_edge = _safe_float(snapshot.get("home_rest_days"), 3.0) - _safe_float(snapshot.get("away_rest_days"), 3.0)
        travel_edge = _safe_float(snapshot.get("away_travel_bucket"), 0.0) + (_safe_float(snapshot.get("away_travel_tz_shift"), 0.0) * 0.5)
        shots_edge = _safe_float(snapshot.get("home_shots"), 0.0) - _safe_float(snapshot.get("away_shots"), 0.0)

        adjustment = (
            (xg_form_edge * 0.020)
            + (xgf_edge * 0.060)
            + (xga_edge * 0.050)
            + (pp_edge * 0.0018)
            + (pk_edge * 0.0014)
            + (rest_edge * 0.014)
            + (travel_edge * 0.010)
            + (shots_edge * 0.0025)
        )
        adjustment = float(np.clip(adjustment, -0.16, 0.16))
        home = float(np.clip(base + adjustment, 0.08, 0.92))
        away = 1.0 - home
        return normalize_two_probs(home, away)

    def combine_with_classifier_diagnostics(
        self,
        classifier_probs: tuple[float, float] | NHLProbabilityView,
        structural_probs: Optional[tuple[float, float] | NHLProbabilityView],
    ) -> NHLBlendDiagnostics:
        if isinstance(classifier_probs, NHLProbabilityView):
            classifier = classifier_probs
        else:
            classifier = normalize_two_probs(*classifier_probs)

        if structural_probs is None:
            return NHLBlendDiagnostics(
                classifier=classifier,
                structural=None,
                combined=classifier,
                disagreement=0.0,
                model_weight=1.0,
                structural_weight=0.0,
                regime="classifier_only",
            )

        if isinstance(structural_probs, NHLProbabilityView):
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
        return NHLBlendDiagnostics(
            classifier=classifier,
            structural=structural,
            combined=combined,
            disagreement=float(disagreement),
            model_weight=float(model_weight),
            structural_weight=float(structural_weight),
            regime=regime,
        )
