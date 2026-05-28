"""
Lightweight basketball side-model helpers.

This gives NBA sides a small structural anchor using the feature cache signals
we already trust most: Elo, form, rest, and travel burden. The goal is to
reduce classifier overconfidence before a fuller possession / efficiency model
exists.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import erf, exp, sqrt
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.markets.engine import MarketEngine, MarketOutcomeInput


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


@dataclass(frozen=True)
class BasketballAvailabilityAdjustment:
    net_rating_delta: float
    expected_margin_delta: float
    confidence_multiplier: float
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BasketballProjectionReport:
    home_win_probability: float
    away_win_probability: float
    expected_margin: float
    projected_home_score: float
    projected_away_score: float
    projected_total: float
    possessions_projection: float
    spread_cover_probability: Optional[float] = None
    total_over_probability: Optional[float] = None
    total_under_probability: Optional[float] = None
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BasketballMarketOutcomeValue:
    outcome: str
    model_probability: float
    market_implied_probability: float
    no_vig_market_probability: float
    offered_odds: float
    edge: float
    expected_value: float
    market_type: str = "moneyline"
    line: Optional[float] = None
    recommended_action: str = "pass"
    decision_reason: str = ""
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BasketballValueReport:
    projection: BasketballProjectionReport
    market_values: list[BasketballMarketOutcomeValue]
    availability_adjustment: BasketballAvailabilityAdjustment
    confidence: float
    clv_status: str = "pending"
    late_line_movement_flag: bool = False
    main_reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["projection"] = self.projection.as_dict()
        payload["market_values"] = [value.as_dict() for value in self.market_values]
        payload["availability_adjustment"] = self.availability_adjustment.as_dict()
        return payload


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


def _normal_cdf(x: float, mean: float = 0.0, sigma: float = 1.0) -> float:
    sigma = max(1e-6, float(sigma))
    z = (float(x) - mean) / sigma
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def _timestamp_after(context: dict[str, Any], timestamp_key: str, reference_key: str = "prediction_time") -> bool:
    if timestamp_key not in context or reference_key not in context:
        return False
    timestamp = pd.to_datetime(context.get(timestamp_key), errors="coerce", utc=True)
    reference = pd.to_datetime(context.get(reference_key), errors="coerce", utc=True)
    if pd.isna(timestamp) or pd.isna(reference):
        return False
    return bool(timestamp > reference)


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

    def availability_adjustment_from_context(self, context: Optional[dict[str, Any]]) -> BasketballAvailabilityAdjustment:
        ctx = dict(context or {})
        warnings: list[str] = []
        confidence = 1.0

        home_lineup_confirmed = bool(ctx.get("home_lineup_confirmed"))
        away_lineup_confirmed = bool(ctx.get("away_lineup_confirmed"))
        if _timestamp_after(ctx, "injury_report_timestamp") or _timestamp_after(ctx, "injury_as_of"):
            warnings.append("injury report timestamp is after prediction time")
            confidence *= 0.82
        if not (home_lineup_confirmed and away_lineup_confirmed):
            warnings.append("projected or unconfirmed starters")
            confidence *= 0.88

        home_star_out = bool(ctx.get("home_star_player_missing") or ctx.get("home_star_out"))
        away_star_out = bool(ctx.get("away_star_player_missing") or ctx.get("away_star_out"))
        if home_star_out or away_star_out:
            warnings.append("star player availability impacts roster strength")
            confidence *= 0.90

        home_minutes_lost = _safe_float(ctx.get("home_expected_minutes_lost"), 0.0)
        away_minutes_lost = _safe_float(ctx.get("away_expected_minutes_lost"), 0.0)
        home_onoff = _safe_float(ctx.get("home_player_impact_missing"), 0.0)
        away_onoff = _safe_float(ctx.get("away_player_impact_missing"), 0.0)
        replacement_gap = _safe_float(ctx.get("home_replacement_quality_gap"), 0.0) - _safe_float(
            ctx.get("away_replacement_quality_gap"),
            0.0,
        )

        questionable = (
            _safe_float(ctx.get("home_questionable_count"), 0.0)
            + _safe_float(ctx.get("away_questionable_count"), 0.0)
            + _safe_float(ctx.get("home_doubtful_count"), 0.0) * 1.5
            + _safe_float(ctx.get("away_doubtful_count"), 0.0) * 1.5
        )
        if questionable > 0:
            warnings.append("late injury statuses still uncertain")
            confidence *= max(0.78, 1.0 - min(0.16, questionable * 0.025))

        home_continuity = _safe_float(ctx.get("home_top8_rotation_continuity"), 1.0)
        away_continuity = _safe_float(ctx.get("away_top8_rotation_continuity"), 1.0)
        continuity_edge = np.clip(home_continuity - away_continuity, -0.35, 0.35)

        # Positive means home strength improves relative to away.
        net_delta = (
            ((away_minutes_lost - home_minutes_lost) / 48.0) * 1.6
            + (away_onoff - home_onoff)
            - replacement_gap
            + (continuity_edge * 2.0)
        )
        if home_star_out:
            net_delta -= 2.2
        if away_star_out:
            net_delta += 2.2

        expected_margin_delta = net_delta * 0.92
        return BasketballAvailabilityAdjustment(
            net_rating_delta=round(float(np.clip(net_delta, -8.0, 8.0)), 4),
            expected_margin_delta=round(float(np.clip(expected_margin_delta, -7.0, 7.0)), 4),
            confidence_multiplier=round(float(np.clip(confidence, 0.50, 1.0)), 3),
            warnings=tuple(dict.fromkeys(warnings)),
        )

    def projection_from_snapshot(
        self,
        snapshot: Optional[pd.Series],
        *,
        availability_context: Optional[dict[str, Any]] = None,
        spread_line: Optional[float] = None,
        total_line: Optional[float] = None,
        model_probabilities: Optional[dict[str, float]] = None,
    ) -> tuple[BasketballProjectionReport, BasketballAvailabilityAdjustment]:
        warnings: list[str] = []
        snap = snapshot if snapshot is not None else pd.Series(dtype=float)
        availability = self.availability_adjustment_from_context(availability_context)
        warnings.extend(availability.warnings)

        possessions = _safe_float(snap.get("possessions_projection"), np.nan)
        if not np.isfinite(possessions):
            possessions = _safe_float(snap.get("expected_matchup_pace"), np.nan)
        if not np.isfinite(possessions):
            expected_points_pace = _safe_float(snap.get("expected_pace"), 220.0)
            possessions = expected_points_pace / 2.20
            warnings.append("true possession data unavailable; using pace proxy")
        possessions = float(np.clip(possessions, 88.0, 108.0))

        home_ortg = _safe_float(snap.get("home_off_rating_per_100"), np.nan)
        away_ortg = _safe_float(snap.get("away_off_rating_per_100"), np.nan)
        home_drtg = _safe_float(snap.get("home_def_rating_per_100"), np.nan)
        away_drtg = _safe_float(snap.get("away_def_rating_per_100"), np.nan)
        if not np.isfinite(home_ortg):
            home_ortg = _safe_float(snap.get("home_adj_ortg"), _safe_float(snap.get("home_ortg"), 113.0))
        if not np.isfinite(away_ortg):
            away_ortg = _safe_float(snap.get("away_adj_ortg"), _safe_float(snap.get("away_ortg"), 113.0))
        if not np.isfinite(home_drtg):
            home_drtg = _safe_float(snap.get("home_adj_drtg"), _safe_float(snap.get("home_drtg"), 113.0))
        if not np.isfinite(away_drtg):
            away_drtg = _safe_float(snap.get("away_adj_drtg"), _safe_float(snap.get("away_drtg"), 113.0))

        home_eff = (0.58 * home_ortg) + (0.42 * away_drtg)
        away_eff = (0.58 * away_ortg) + (0.42 * home_drtg)
        home_score = (home_eff * possessions) / 100.0
        away_score = (away_eff * possessions) / 100.0
        margin = (home_score - away_score) + 1.6 + availability.expected_margin_delta

        net_diff = _safe_float(snap.get("injury_adjusted_net_rating_diff"), np.nan)
        if not np.isfinite(net_diff):
            net_diff = _safe_float(snap.get("opponent_adjusted_net_rating_diff"), _safe_float(snap.get("net_rtg_diff"), np.nan))
        if np.isfinite(net_diff):
            margin = (0.62 * margin) + (0.38 * ((net_diff + availability.net_rating_delta) * possessions / 100.0 + 1.6))

        home_prob = 1.0 / (1.0 + exp(-margin / 6.7))
        away_prob = 1.0 - home_prob
        if model_probabilities:
            normalized = normalize_two_probs(
                float(model_probabilities.get("home", home_prob)),
                float(model_probabilities.get("away", away_prob)),
            )
            home_prob, away_prob = normalized.as_tuple()

        projected_total = max(160.0, min(280.0, home_score + away_score))
        adjusted_home_score = (projected_total + margin) / 2.0
        adjusted_away_score = projected_total - adjusted_home_score

        spread_cover_probability = None
        if spread_line is not None:
            spread_cover_probability = round(1.0 - _normal_cdf(0.0, mean=margin + float(spread_line), sigma=12.0), 4)

        total_over_probability = None
        total_under_probability = None
        if total_line is not None:
            total_over_probability = round(1.0 - _normal_cdf(float(total_line), mean=projected_total, sigma=16.0), 4)
            total_under_probability = round(1.0 - total_over_probability, 4)

        if not bool((availability_context or {}).get("garbage_time_filtered_available")):
            warnings.append("garbage-time filtered efficiency unavailable")

        report = BasketballProjectionReport(
            home_win_probability=round(float(home_prob), 4),
            away_win_probability=round(float(away_prob), 4),
            expected_margin=round(float(margin), 3),
            projected_home_score=round(float(adjusted_home_score), 2),
            projected_away_score=round(float(adjusted_away_score), 2),
            projected_total=round(float(projected_total), 2),
            possessions_projection=round(float(possessions), 2),
            spread_cover_probability=spread_cover_probability,
            total_over_probability=total_over_probability,
            total_under_probability=total_under_probability,
            warnings=tuple(dict.fromkeys(warnings)),
        )
        return report, availability

    def build_value_report(
        self,
        *,
        snapshot: Optional[pd.Series],
        odds_moneyline: dict[str, float],
        availability_context: Optional[dict[str, Any]] = None,
        spread_line: Optional[float] = None,
        spread_odds: Optional[dict[str, float]] = None,
        total_line: Optional[float] = None,
        total_odds: Optional[dict[str, float]] = None,
        model_probabilities: Optional[dict[str, float]] = None,
        base_confidence: float = 0.64,
        clv_status: str = "pending",
        prediction_time: object | None = None,
        signal_odds_moneyline: Optional[dict[str, float]] = None,
        closing_odds_moneyline: Optional[dict[str, float]] = None,
        closing_odds_timestamp: object | None = None,
    ) -> BasketballValueReport:
        ctx = dict(availability_context or {})
        if prediction_time is None:
            prediction_time = ctx.get("prediction_time")
        projection, availability = self.projection_from_snapshot(
            snapshot,
            availability_context=availability_context,
            spread_line=spread_line,
            total_line=total_line,
            model_probabilities=model_probabilities,
        )
        values: list[BasketballMarketOutcomeValue] = []
        ml_probs = {"home": projection.home_win_probability, "away": projection.away_win_probability}
        engine = MarketEngine()
        ml_decisions = engine.evaluate_market(
            sport="basketball",
            market="moneyline",
            event="basketball probability/value report",
            prediction_time=prediction_time,
            signal_odds=signal_odds_moneyline,
            closing_odds=closing_odds_moneyline,
            closing_odds_timestamp=closing_odds_timestamp,
            outcomes=[
                MarketOutcomeInput(outcome=outcome, odds=float(odds_moneyline[outcome]), model_probability=float(ml_probs[outcome]))
                for outcome in ("home", "away")
            ],
        )
        for decision in ml_decisions:
            values.append(
                BasketballMarketOutcomeValue(
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

        if spread_line is not None and spread_odds and projection.spread_cover_probability is not None:
            home_cover = float(projection.spread_cover_probability)
            away_cover = 1.0 - home_cover
            spread_decisions = engine.evaluate_market(
                sport="basketball",
                market="spread",
                event="basketball probability/value report",
                outcomes=[
                    MarketOutcomeInput(outcome="home_spread", odds=float(spread_odds["home"]), model_probability=home_cover),
                    MarketOutcomeInput(outcome="away_spread", odds=float(spread_odds["away"]), model_probability=away_cover),
                ],
            )
            for decision in spread_decisions:
                values.append(
                    BasketballMarketOutcomeValue(
                        outcome=decision.outcome,
                        market_type="spread",
                        model_probability=round(decision.model_probability, 4),
                        market_implied_probability=round(decision.raw_implied_probability, 4),
                    no_vig_market_probability=round(decision.market_no_vig_probability, 4),
                    offered_odds=round(decision.decimal_odds, 3),
                    edge=round(decision.edge, 4),
                    expected_value=round(decision.expected_value, 4),
                    line=round(float(spread_line), 2),
                    recommended_action=decision.recommended_action,
                    decision_reason=decision.reason,
                    warnings=decision.warnings,
                )
            )

        if total_line is not None and total_odds and projection.total_over_probability is not None:
            total_decisions = engine.evaluate_market(
                sport="basketball",
                market="total",
                event="basketball probability/value report",
                outcomes=[
                    MarketOutcomeInput(outcome="over", odds=float(total_odds["over"]), model_probability=float(projection.total_over_probability)),
                    MarketOutcomeInput(outcome="under", odds=float(total_odds["under"]), model_probability=float(projection.total_under_probability or 0.0)),
                ],
            )
            for decision in total_decisions:
                values.append(
                    BasketballMarketOutcomeValue(
                        outcome=decision.outcome,
                        market_type="total",
                        model_probability=round(decision.model_probability, 4),
                        market_implied_probability=round(decision.raw_implied_probability, 4),
                        no_vig_market_probability=round(decision.market_no_vig_probability, 4),
                        offered_odds=round(decision.decimal_odds, 3),
                        edge=round(decision.edge, 4),
                        expected_value=round(decision.expected_value, 4),
                        line=round(float(total_line), 2),
                        recommended_action=decision.recommended_action,
                        decision_reason=decision.reason,
                        warnings=decision.warnings,
                    )
                )

        main_reasons = []
        snap = snapshot if snapshot is not None else pd.Series(dtype=float)
        if np.isfinite(_safe_float(snap.get("opponent_adjusted_net_rating_diff"), np.nan)):
            main_reasons.append("opponent-adjusted net rating")
        if projection.possessions_projection:
            main_reasons.append("matchup pace projection")
        if availability.warnings:
            main_reasons.append("player availability adjustment")
        if ctx.get("late_line_movement_flag") or ctx.get("injury_line_movement_flag"):
            main_reasons.append("late line movement")

        warnings = tuple(dict.fromkeys((*projection.warnings, *availability.warnings)))
        return BasketballValueReport(
            projection=projection,
            market_values=values,
            availability_adjustment=availability,
            confidence=round(float(np.clip(base_confidence * availability.confidence_multiplier, 0.05, 0.95)), 3),
            clv_status=clv_status,
            late_line_movement_flag=bool(ctx.get("late_line_movement_flag") or ctx.get("injury_line_movement_flag")),
            main_reasons=tuple(dict.fromkeys(main_reasons)),
            warnings=warnings,
        )

    @staticmethod
    def validate_feature_timestamps(
        features: pd.DataFrame,
        *,
        event_time_col: str = "date",
        timestamp_suffix: str = "_as_of",
    ) -> list[str]:
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
