"""
Structural soccer score-model helpers.

This module exposes a lightweight probability engine built from the
already-engineered Dixon-Coles style features in the soccer feature cache.
It does not retrain a separate artifact yet; instead it provides a stable,
testable home for the score-distribution logic that can later grow into a
full soccer score model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import exp, factorial
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.markets.engine import MarketEngine, MarketOutcomeInput


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


@dataclass(frozen=True)
class SoccerGoalModelReport:
    expected_home_goals: float
    expected_away_goals: float
    score_matrix: list[list[float]]
    home_win_probability: float
    draw_probability: float
    away_win_probability: float
    over_probabilities: dict[str, float] = field(default_factory=dict)
    under_probabilities: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SoccerLineupAdjustment:
    confidence_multiplier: float
    home_goal_delta: float = 0.0
    away_goal_delta: float = 0.0
    warnings: tuple[str, ...] = ()
    lineup_quality_score: float | None = None
    confirmed_lineup_delta: float | None = None


@dataclass(frozen=True)
class SoccerMarketOutcomeValue:
    outcome: str
    model_probability: float
    market_implied_probability: float
    no_vig_market_probability: float
    offered_odds: float
    edge: float
    expected_value: float
    recommended_action: str = "pass"
    decision_reason: str = ""
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SoccerValueReport:
    goal_model: SoccerGoalModelReport
    market_values: list[SoccerMarketOutcomeValue]
    confidence: float
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["goal_model"] = self.goal_model.as_dict()
        payload["market_values"] = [value.as_dict() for value in self.market_values]
        return payload


def _safe_float(value: object, default: float = np.nan) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _timestamp_after(context: dict[str, Any], timestamp_key: str, reference_key: str = "prediction_time") -> bool:
    if timestamp_key not in context or reference_key not in context:
        return False
    timestamp = pd.to_datetime(context.get(timestamp_key), errors="coerce", utc=True)
    reference = pd.to_datetime(context.get(reference_key), errors="coerce", utc=True)
    if pd.isna(timestamp) or pd.isna(reference):
        return False
    return bool(timestamp > reference)


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

    @staticmethod
    def _poisson_pmf(lam: float, max_goals: int) -> np.ndarray:
        lam = max(0.05, min(8.0, float(lam)))
        probs = np.array([(lam ** k) * exp(-lam) / factorial(k) for k in range(max_goals + 1)], dtype=float)
        remainder = max(0.0, 1.0 - float(probs.sum()))
        probs[-1] += remainder
        return probs

    def score_distribution_from_xg(
        self,
        expected_home_goals: float,
        expected_away_goals: float,
        *,
        max_goals: int = 8,
        draw_correlation: float = 0.04,
    ) -> np.ndarray:
        """
        Build a bounded exact-score matrix from expected goals.

        The small diagonal boost is a lightweight Dixon-Coles-style correction:
        low-scoring draws are slightly more common than independent Poisson
        assumptions imply. The matrix is always renormalized to sum to one.
        """
        home_pmf = self._poisson_pmf(expected_home_goals, max_goals)
        away_pmf = self._poisson_pmf(expected_away_goals, max_goals)
        matrix = np.outer(home_pmf, away_pmf)

        corr = max(0.0, min(0.12, float(draw_correlation)))
        for score in range(min(3, max_goals) + 1):
            matrix[score, score] *= 1.0 + corr
        matrix /= matrix.sum()
        return matrix

    def probability_report_from_expected_goals(
        self,
        expected_home_goals: float,
        expected_away_goals: float,
        *,
        max_goals: int = 8,
    ) -> SoccerGoalModelReport:
        matrix = self.score_distribution_from_xg(
            expected_home_goals,
            expected_away_goals,
            max_goals=max_goals,
        )
        home_win = float(np.tril(matrix, -1).sum())
        draw = float(np.diag(matrix).sum())
        away_win = float(np.triu(matrix, 1).sum())

        totals = np.add.outer(np.arange(max_goals + 1), np.arange(max_goals + 1))
        over_probabilities = {
            "over_1_5": round(float(matrix[totals > 1.5].sum()), 4),
            "over_2_5": round(float(matrix[totals > 2.5].sum()), 4),
            "over_3_5": round(float(matrix[totals > 3.5].sum()), 4),
        }
        under_probabilities = {
            key.replace("over", "under"): round(1.0 - value, 4)
            for key, value in over_probabilities.items()
        }

        return SoccerGoalModelReport(
            expected_home_goals=round(float(expected_home_goals), 4),
            expected_away_goals=round(float(expected_away_goals), 4),
            score_matrix=np.round(matrix, 6).tolist(),
            home_win_probability=round(home_win, 4),
            draw_probability=round(draw, 4),
            away_win_probability=round(away_win, 4),
            over_probabilities=over_probabilities,
            under_probabilities=under_probabilities,
        )

    def lineup_adjustment_from_context(
        self,
        context: Optional[dict[str, Any]],
    ) -> SoccerLineupAdjustment:
        if not context:
            return SoccerLineupAdjustment(
                confidence_multiplier=0.88,
                warnings=("lineup context unavailable; confidence reduced",),
            )

        warnings: list[str] = []
        confidence = 1.0
        home_delta = 0.0
        away_delta = 0.0

        home_confirmed = bool(context.get("home_lineup_confirmed"))
        away_confirmed = bool(context.get("away_lineup_confirmed"))
        if _timestamp_after(context, "lineup_timestamp") or _timestamp_after(context, "lineup_as_of"):
            home_confirmed = False
            away_confirmed = False
            warnings.append("lineup timestamp is after prediction time")
        if not (home_confirmed and away_confirmed):
            confidence *= 0.9
            warnings.append("confirmed starting XIs unavailable")

        for side in ("home", "away"):
            side_delta = 0.0
            if context.get(f"{side}_missing_goalkeeper"):
                side_delta -= 0.16
                warnings.append(f"{side} missing starting goalkeeper")
            if context.get(f"{side}_missing_center_back"):
                side_delta -= 0.08
                warnings.append(f"{side} missing center-back")
            if context.get(f"{side}_missing_central_midfielder"):
                side_delta -= 0.06
                warnings.append(f"{side} missing central midfielder")
            if context.get(f"{side}_missing_striker"):
                side_delta -= 0.10
                warnings.append(f"{side} missing striker")
            if context.get(f"{side}_suspension_flag"):
                side_delta -= 0.05
                warnings.append(f"{side} suspension flag present")

            if side == "home":
                home_delta += side_delta
                away_delta += -side_delta * 0.35
            else:
                away_delta += side_delta
                home_delta += -side_delta * 0.35

            if side_delta < 0:
                confidence *= max(0.78, 1.0 - abs(side_delta) * 0.4)

        projected_count = _safe_float(context.get("projected_starters_count"), np.nan)
        if np.isfinite(projected_count) and projected_count < 20:
            confidence *= 0.94
            warnings.append("projected starting XI coverage is incomplete")

        confirmed_delta = _safe_float(context.get("confirmed_vs_expected_lineup_delta"), np.nan)
        if np.isfinite(confirmed_delta) and abs(confirmed_delta) >= 0.15:
            confidence *= 0.92
            warnings.append("confirmed lineup materially differs from expected lineup")

        lineup_quality = _safe_float(context.get("lineup_quality_score"), np.nan)
        if np.isfinite(lineup_quality) and lineup_quality < 0.75:
            confidence *= 0.93
            warnings.append("lineup quality score is weak")

        return SoccerLineupAdjustment(
            confidence_multiplier=round(max(0.55, min(1.0, confidence)), 3),
            home_goal_delta=round(home_delta, 4),
            away_goal_delta=round(away_delta, 4),
            warnings=tuple(dict.fromkeys(warnings)),
            lineup_quality_score=round(float(lineup_quality), 4) if np.isfinite(lineup_quality) else None,
            confirmed_lineup_delta=round(float(confirmed_delta), 4) if np.isfinite(confirmed_delta) else None,
        )

    def build_value_report(
        self,
        *,
        expected_home_goals: float,
        expected_away_goals: float,
        odds_1x2: dict[str, float],
        model_probabilities: Optional[dict[str, float]] = None,
        lineup_context: Optional[dict[str, Any]] = None,
        base_confidence: float = 0.64,
        prediction_time: object | None = None,
        signal_odds: Optional[dict[str, float]] = None,
        closing_odds: Optional[dict[str, float]] = None,
        closing_odds_timestamp: object | None = None,
    ) -> SoccerValueReport:
        context = dict(lineup_context or {})
        if prediction_time is None:
            prediction_time = context.get("prediction_time")
        lineup = self.lineup_adjustment_from_context(lineup_context)
        adjusted_home_xg = max(0.05, expected_home_goals + lineup.home_goal_delta)
        adjusted_away_xg = max(0.05, expected_away_goals + lineup.away_goal_delta)
        goal_report = self.probability_report_from_expected_goals(adjusted_home_xg, adjusted_away_xg)

        outcomes = ("home", "draw", "away")
        model_probs = {
            "home": goal_report.home_win_probability,
            "draw": goal_report.draw_probability,
            "away": goal_report.away_win_probability,
        }
        if model_probabilities:
            override = normalize_three_probs(
                float(model_probabilities.get("home", model_probs["home"])),
                float(model_probabilities.get("draw", model_probs["draw"])),
                float(model_probabilities.get("away", model_probs["away"])),
            )
            model_probs = {
                "home": override.home,
                "draw": override.draw,
                "away": override.away,
            }

        engine_decisions = MarketEngine().evaluate_market(
            sport="soccer",
            market="1x2",
            event="soccer probability/value report",
            prediction_time=prediction_time,
            signal_odds=signal_odds,
            closing_odds=closing_odds,
            closing_odds_timestamp=closing_odds_timestamp,
            outcomes=[
                MarketOutcomeInput(outcome=outcome, odds=float(odds_1x2[outcome]), model_probability=float(model_probs[outcome]))
                for outcome in outcomes
            ],
        )
        values: list[SoccerMarketOutcomeValue] = []
        for decision in engine_decisions:
            values.append(
                SoccerMarketOutcomeValue(
                    outcome=decision.outcome,
                    model_probability=round(decision.model_probability, 4),
                    market_implied_probability=round(decision.raw_implied_probability, 4),
                    no_vig_market_probability=round(decision.market_no_vig_probability, 4),
                    offered_odds=round(decision.decimal_odds, 3),
                    edge=round(decision.edge, 4),
                    expected_value=round(decision.expected_value, 4),
                    recommended_action=decision.recommended_action,
                    decision_reason=decision.reason,
                    warnings=decision.warnings,
                )
            )

        return SoccerValueReport(
            goal_model=goal_report,
            market_values=values,
            confidence=round(max(0.05, min(0.95, base_confidence * lineup.confidence_multiplier)), 3),
            warnings=lineup.warnings,
        )

    @staticmethod
    def validate_feature_timestamps(
        features: pd.DataFrame,
        *,
        event_time_col: str = "date",
        timestamp_suffix: str = "_as_of",
    ) -> list[str]:
        """
        Return names of feature timestamp columns that are later than kickoff.
        This is a guardrail for lineup/injury/goalkeeper data, which can easily
        leak if post-match snapshots are joined back into historical rows.
        """
        if event_time_col not in features.columns:
            return []
        event_times = pd.to_datetime(features[event_time_col], errors="coerce")
        unsafe: list[str] = []
        for col in features.columns:
            if not col.endswith(timestamp_suffix):
                continue
            as_of = pd.to_datetime(features[col], errors="coerce")
            if bool((as_of > event_times).fillna(False).any()):
                unsafe.append(col)
        return unsafe

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

        report = self.probability_report_from_expected_goals(home_xg, away_xg)
        return normalize_three_probs(
            report.home_win_probability,
            report.draw_probability,
            report.away_win_probability,
        )

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
