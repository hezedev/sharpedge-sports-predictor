"""
Lightweight MLB side-model helpers.

This module gives MLB a small structural probability anchor using existing
feature-cache signals that already capture team strength, pitcher quality,
recent form, and schedule density. It is intentionally modest: the goal is to
reduce classifier overconfidence before a fuller run-environment model exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MLBProbabilityView:
    home: float
    away: float

    def as_tuple(self) -> tuple[float, float]:
        return self.home, self.away


@dataclass(frozen=True)
class MLBBlendDiagnostics:
    classifier: MLBProbabilityView
    structural: Optional[MLBProbabilityView]
    combined: MLBProbabilityView
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


def normalize_two_probs(p_home: float, p_away: float) -> MLBProbabilityView:
    home = max(0.0, float(p_home))
    away = max(0.0, float(p_away))
    total = home + away
    if total <= 0:
        return MLBProbabilityView(0.5, 0.5)
    return MLBProbabilityView(home / total, away / total)


class MLBSideModel:
    def __init__(
        self,
        *,
        low_disagreement_model_weight: float = 0.70,
        medium_disagreement_model_weight: float = 0.58,
        high_disagreement_model_weight: float = 0.42,
    ) -> None:
        self.low_disagreement_model_weight = low_disagreement_model_weight
        self.medium_disagreement_model_weight = medium_disagreement_model_weight
        self.high_disagreement_model_weight = high_disagreement_model_weight

    def structural_probs_from_snapshot(self, snapshot: Optional[pd.Series]) -> Optional[MLBProbabilityView]:
        if snapshot is None:
            return None

        base = _safe_float(snapshot.get("elo_win_prob"), np.nan)
        if not np.isfinite(base):
            return None

        sp_era_diff = _safe_float(snapshot.get("sp_era_diff"), 0.0)
        sp_whip_diff = _safe_float(snapshot.get("sp_whip_diff"), 0.0)
        sp_k9_diff = _safe_float(snapshot.get("sp_k9_diff"), 0.0)
        form_diff = _safe_float(snapshot.get("home_win_pct_10"), 0.5) - _safe_float(snapshot.get("away_win_pct_10"), 0.5)
        run_diff = _safe_float(snapshot.get("home_run_diff_10"), 0.0) - _safe_float(snapshot.get("away_run_diff_10"), 0.0)
        density_diff = _safe_float(snapshot.get("density_diff"), 0.0)
        home_games_l3d = _safe_float(snapshot.get("home_games_L3D"), 0.0)
        away_games_l3d = _safe_float(snapshot.get("away_games_L3D"), 0.0)
        home_b2b = _safe_float(snapshot.get("home_b2b"), 0.0)
        away_b2b = _safe_float(snapshot.get("away_b2b"), 0.0)
        bullpen_load_edge = (
            (away_games_l3d - home_games_l3d) * 0.012
            + (away_b2b - home_b2b) * 0.018
        )

        adjustment = (
            (-sp_era_diff * 0.018)
            + (-sp_whip_diff * 0.10)
            + (sp_k9_diff * 0.008)
            + (form_diff * 0.10)
            + (run_diff * 0.015)
            + (-density_diff * 0.04)
            + bullpen_load_edge
        )
        adjustment = float(np.clip(adjustment, -0.18, 0.18))
        home = float(np.clip(base + adjustment, 0.08, 0.92))
        away = 1.0 - home
        return normalize_two_probs(home, away)

    def combine_with_classifier_diagnostics(
        self,
        classifier_probs: tuple[float, float] | MLBProbabilityView,
        structural_probs: Optional[tuple[float, float] | MLBProbabilityView],
    ) -> MLBBlendDiagnostics:
        if isinstance(classifier_probs, MLBProbabilityView):
            classifier = classifier_probs
        else:
            classifier = normalize_two_probs(*classifier_probs)

        if structural_probs is None:
            return MLBBlendDiagnostics(
                classifier=classifier,
                structural=None,
                combined=classifier,
                disagreement=0.0,
                model_weight=1.0,
                structural_weight=0.0,
                regime="classifier_only",
            )

        if isinstance(structural_probs, MLBProbabilityView):
            structural = structural_probs
        else:
            structural = normalize_two_probs(*structural_probs)

        disagreement = abs(classifier.home - structural.home) + abs(classifier.away - structural.away)
        if disagreement <= 0.12:
            model_weight = self.low_disagreement_model_weight
        elif disagreement <= 0.22:
            model_weight = self.medium_disagreement_model_weight
        else:
            model_weight = self.high_disagreement_model_weight
        structural_weight = 1.0 - model_weight

        if disagreement <= 0.08:
            regime = "aligned"
        elif disagreement <= 0.18:
            regime = "balanced"
        elif structural_weight >= 0.50:
            regime = "structural_override"
        else:
            regime = "classifier_lean"

        combined = normalize_two_probs(
            (classifier.home * model_weight) + (structural.home * structural_weight),
            (classifier.away * model_weight) + (structural.away * structural_weight),
        )
        return MLBBlendDiagnostics(
            classifier=classifier,
            structural=structural,
            combined=combined,
            disagreement=float(disagreement),
            model_weight=float(model_weight),
            structural_weight=float(structural_weight),
            regime=regime,
        )
