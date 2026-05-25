"""
Structural soccer score-model helpers.

This module exposes a lightweight probability engine built from the
already-engineered Dixon-Coles style features in the soccer feature cache.
It does not retrain a separate artifact yet; instead it provides a stable,
testable home for the score-distribution logic that can later grow into a
full soccer score model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SoccerProbabilityView:
    home: float
    draw: float
    away: float

    def as_tuple(self) -> tuple[float, float, float]:
        return self.home, self.draw, self.away

    def as_dict(self) -> dict[str, float]:
        return {
            "home": round(float(self.home), 4),
            "draw": round(float(self.draw), 4),
            "away": round(float(self.away), 4),
        }


@dataclass(frozen=True)
class SoccerBlendDiagnostics:
    classifier: SoccerProbabilityView
    structural: Optional[SoccerProbabilityView]
    combined: SoccerProbabilityView
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


def normalize_three_probs(p_home: float, p_draw: float, p_away: float) -> SoccerProbabilityView:
    vals = np.array(
        [max(0.0, float(p_home)), max(0.0, float(p_draw)), max(0.0, float(p_away))],
        dtype=float,
    )
    total = float(vals.sum())
    if total <= 0:
        return SoccerProbabilityView(1 / 3, 1 / 3, 1 / 3)
    vals /= total
    return SoccerProbabilityView(float(vals[0]), float(vals[1]), float(vals[2]))


class SoccerScoreModel:
    """
    Lightweight structural soccer probability model.

    Current responsibilities:
    - read structural 1X2 probabilities from Dixon-Coles engineered features
    - derive those probabilities from expected goals when needed
    - combine the structural view with the trained classifier output
    """

    def __init__(
        self,
        *,
        low_disagreement_model_weight: float = 0.70,
        medium_disagreement_model_weight: float = 0.56,
        high_disagreement_model_weight: float = 0.40,
    ) -> None:
        self.low_disagreement_model_weight = low_disagreement_model_weight
        self.medium_disagreement_model_weight = medium_disagreement_model_weight
        self.high_disagreement_model_weight = high_disagreement_model_weight

    def normalize_three_probs(self, p_home: float, p_draw: float, p_away: float) -> SoccerProbabilityView:
        return normalize_three_probs(p_home, p_draw, p_away)

    def structural_probs_from_snapshot(
        self,
        snapshot: Optional[pd.Series],
    ) -> Optional[SoccerProbabilityView]:
        if snapshot is None:
            return None

        home_prob = _safe_float(snapshot.get("home_dc_win_prob"), np.nan)
        draw_prob = _safe_float(snapshot.get("dc_draw_prob"), np.nan)
        away_prob = _safe_float(snapshot.get("away_dc_win_prob"), np.nan)
        if np.isfinite(home_prob) and np.isfinite(draw_prob) and np.isfinite(away_prob):
            return normalize_three_probs(home_prob, draw_prob, away_prob)

        home_xg = _safe_float(snapshot.get("home_dc_xg"), np.nan)
        away_xg = _safe_float(snapshot.get("away_dc_xg"), np.nan)
        if not (np.isfinite(home_xg) and np.isfinite(away_xg) and home_xg > 0 and away_xg > 0):
            return None

        try:
            from scipy.stats import poisson as _poisson

            k_max = 8
            ph = np.array([_poisson.pmf(k, home_xg) for k in range(k_max + 1)])
            pa = np.array([_poisson.pmf(k, away_xg) for k in range(k_max + 1)])
            mat = np.outer(ph, pa)
            return normalize_three_probs(
                float(np.tril(mat, -1).sum()),
                float(np.diag(mat).sum()),
                float(np.triu(mat, 1).sum()),
            )
        except Exception:
            return None

    def combine_with_classifier(
        self,
        classifier_probs: tuple[float, float, float],
        structural_probs: Optional[tuple[float, float, float] | SoccerProbabilityView],
    ) -> SoccerProbabilityView:
        return self.combine_with_classifier_diagnostics(classifier_probs, structural_probs).combined

    def combine_with_classifier_diagnostics(
        self,
        classifier_probs: tuple[float, float, float],
        structural_probs: Optional[tuple[float, float, float] | SoccerProbabilityView],
    ) -> SoccerBlendDiagnostics:
        classifier = normalize_three_probs(*classifier_probs)
        if structural_probs is None:
            return SoccerBlendDiagnostics(
                classifier=classifier,
                structural=None,
                combined=classifier,
                disagreement=0.0,
                model_weight=1.0,
                structural_weight=0.0,
                regime="classifier_only",
            )

        if isinstance(structural_probs, SoccerProbabilityView):
            structural = structural_probs
        else:
            structural = normalize_three_probs(*structural_probs)

        disagreement = (
            abs(classifier.home - structural.home)
            + abs(classifier.draw - structural.draw)
            + abs(classifier.away - structural.away)
        )

        if disagreement <= 0.18:
            model_weight = self.low_disagreement_model_weight
        elif disagreement <= 0.30:
            model_weight = self.medium_disagreement_model_weight
        else:
            model_weight = self.high_disagreement_model_weight
        struct_weight = 1.0 - model_weight
        if disagreement <= 0.12:
            regime = "aligned"
        elif disagreement <= 0.24:
            regime = "balanced"
        elif struct_weight >= 0.50:
            regime = "structural_override"
        else:
            regime = "classifier_lean"

        combined = normalize_three_probs(
            (classifier.home * model_weight) + (structural.home * struct_weight),
            (classifier.draw * model_weight) + (structural.draw * struct_weight),
            (classifier.away * model_weight) + (structural.away * struct_weight),
        )
        return SoccerBlendDiagnostics(
            classifier=classifier,
            structural=structural,
            combined=combined,
            disagreement=float(disagreement),
            model_weight=float(model_weight),
            structural_weight=float(struct_weight),
            regime=regime,
        )
