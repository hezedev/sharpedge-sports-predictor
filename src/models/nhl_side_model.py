"""
Lightweight NHL side-model helpers.

This gives NHL sides a small structural anchor using feature-cache signals we
already trust most: Elo, xG form, special teams, rest, and travel burden.
The goal is to reduce classifier overconfidence before a fuller goalie/xG
model exists.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import exp, factorial
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.markets.engine import MarketEngine, MarketOutcomeInput


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


@dataclass(frozen=True)
class NHLProjectedGoalsReport:
    expected_home_goals: float
    expected_away_goals: float
    regulation_home_probability: float
    regulation_tie_probability: float
    regulation_away_probability: float
    overtime_probability: float
    full_game_home_probability: float
    full_game_away_probability: float
    score_matrix: list[list[float]]
    over_probabilities: dict[str, float] = field(default_factory=dict)
    under_probabilities: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NHLGoalieScenario:
    name: str
    home_goalie_status: str
    away_goalie_status: str
    expected_home_goals: float
    expected_away_goals: float
    full_game_home_probability: float
    full_game_away_probability: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NHLMarketOutcomeValue:
    outcome: str
    model_probability: float
    market_implied_probability: float
    no_vig_market_probability: float
    offered_odds: float
    edge: float
    expected_value: float
    market_type: str = "moneyline"
    recommended_action: str = "pass"
    decision_reason: str = ""
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NHLValueReport:
    projected_goals: NHLProjectedGoalsReport
    market_values: list[NHLMarketOutcomeValue]
    goalie_status: str
    goalie_scenarios: list[NHLGoalieScenario]
    goalie_sensitivity_home_prob: float
    confidence: float
    clv_status: str = "pending"
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["projected_goals"] = self.projected_goals.as_dict()
        payload["market_values"] = [value.as_dict() for value in self.market_values]
        payload["goalie_scenarios"] = [scenario.as_dict() for scenario in self.goalie_scenarios]
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

    @staticmethod
    def _poisson_pmf(lam: float, max_goals: int) -> np.ndarray:
        lam = max(0.05, min(9.0, float(lam)))
        probs = np.array([(lam ** k) * exp(-lam) / factorial(k) for k in range(max_goals + 1)], dtype=float)
        probs[-1] += max(0.0, 1.0 - float(probs.sum()))
        return probs

    def projected_goals_from_snapshot(self, snapshot: Optional[pd.Series]) -> tuple[float, float]:
        if snapshot is None:
            return 3.05, 2.85

        home_xgf = _safe_float(snapshot.get("home_5v5_xgf_pg_10"), np.nan)
        away_xgf = _safe_float(snapshot.get("away_5v5_xgf_pg_10"), np.nan)
        home_xga = _safe_float(snapshot.get("home_5v5_xga_pg_10"), np.nan)
        away_xga = _safe_float(snapshot.get("away_5v5_xga_pg_10"), np.nan)
        if not np.isfinite(home_xgf):
            home_xgf = _safe_float(snapshot.get("home_xgf_pg_10"), 2.65)
        if not np.isfinite(away_xgf):
            away_xgf = _safe_float(snapshot.get("away_xgf_pg_10"), 2.50)
        if not np.isfinite(home_xga):
            home_xga = _safe_float(snapshot.get("home_xga_pg_10"), 2.50)
        if not np.isfinite(away_xga):
            away_xga = _safe_float(snapshot.get("away_xga_pg_10"), 2.65)

        league_total = _safe_float(snapshot.get("nhl_league_goal_environment"), 5.95)
        baseline = max(4.8, min(7.2, league_total)) / 2.0
        special_home = (
            (_safe_float(snapshot.get("home_pp_pct_10"), 0.20) - 0.20) * 0.9
            + (_safe_float(snapshot.get("away_pk_pct_10"), 0.80) - 0.80) * -0.55
        )
        special_away = (
            (_safe_float(snapshot.get("away_pp_pct_10"), 0.20) - 0.20) * 0.9
            + (_safe_float(snapshot.get("home_pk_pct_10"), 0.80) - 0.80) * -0.55
        )

        home_goals = (0.48 * home_xgf) + (0.35 * away_xga) + (0.17 * baseline) + 0.10 + special_home
        away_goals = (0.48 * away_xgf) + (0.35 * home_xga) + (0.17 * baseline) - 0.04 + special_away
        return float(np.clip(home_goals, 1.4, 5.2)), float(np.clip(away_goals, 1.3, 5.0))

    def apply_goalie_adjustment(
        self,
        expected_home_goals: float,
        expected_away_goals: float,
        goalie_context: Optional[dict[str, Any]],
    ) -> tuple[float, float, tuple[str, ...], str, float]:
        if not goalie_context:
            return expected_home_goals, expected_away_goals, ("starting goalie context unavailable",), "unconfirmed", 0.86

        warnings: list[str] = []
        status = str(goalie_context.get("goalie_status") or "").strip().lower() or "unknown"
        home_confirmed = bool(goalie_context.get("home_goalie_confirmed"))
        away_confirmed = bool(goalie_context.get("away_goalie_confirmed"))
        confidence = 1.0
        if _timestamp_after(goalie_context, "goalie_confirmation_timestamp") or _timestamp_after(goalie_context, "goalie_as_of"):
            home_confirmed = False
            away_confirmed = False
            warnings.append("goalie confirmation timestamp is after prediction time")
        if not (home_confirmed and away_confirmed):
            confidence *= 0.86
            warnings.append("starting goalie is unconfirmed")
            status = "unconfirmed"
        else:
            status = "confirmed"

        def _shrunk_gsax(side: str) -> float:
            long_term = _safe_float(goalie_context.get(f"{side}_goalie_gsax_long_term"), np.nan)
            recent = _safe_float(goalie_context.get(f"{side}_goalie_recent_gsax"), np.nan)
            if not np.isfinite(long_term):
                long_term = _safe_float(goalie_context.get(f"{side}_goalie_gsax"), 0.0)
            if not np.isfinite(recent):
                recent = long_term
            starts = max(0.0, _safe_float(goalie_context.get(f"{side}_goalie_recent_starts"), 5.0))
            weight = min(0.45, starts / 24.0)
            return (long_term * (1.0 - weight)) + (recent * weight)

        home_gsax = _shrunk_gsax("home")
        away_gsax = _shrunk_gsax("away")
        home_backup_gap = max(0.0, _safe_float(goalie_context.get("home_goalie_quality_gap"), 0.0))
        away_backup_gap = max(0.0, _safe_float(goalie_context.get("away_goalie_quality_gap"), 0.0))
        if goalie_context.get("home_backup_goalie_flag"):
            warnings.append("home backup goalie projected")
            expected_away_goals += 0.14 + (home_backup_gap * 0.04)
            confidence *= 0.94
        if goalie_context.get("away_backup_goalie_flag"):
            warnings.append("away backup goalie projected")
            expected_home_goals += 0.14 + (away_backup_gap * 0.04)
            confidence *= 0.94

        expected_away_goals -= np.clip(home_gsax * 0.035, -0.22, 0.22)
        expected_home_goals -= np.clip(away_gsax * 0.035, -0.22, 0.22)

        if goalie_context.get("home_goalie_b2b") or goalie_context.get("away_goalie_b2b"):
            warnings.append("goalie rest risk on consecutive starts")
            confidence *= 0.96

        return (
            float(np.clip(expected_home_goals, 1.0, 6.2)),
            float(np.clip(expected_away_goals, 1.0, 6.2)),
            tuple(dict.fromkeys(warnings)),
            status,
            round(max(0.50, min(1.0, confidence)), 3),
        )

    def projected_goals_report(
        self,
        expected_home_goals: float,
        expected_away_goals: float,
        *,
        max_goals: int = 9,
        home_ot_strength: float = 0.5,
        away_ot_strength: float = 0.5,
    ) -> NHLProjectedGoalsReport:
        home_pmf = self._poisson_pmf(expected_home_goals, max_goals)
        away_pmf = self._poisson_pmf(expected_away_goals, max_goals)
        matrix = np.outer(home_pmf, away_pmf)
        matrix /= matrix.sum()
        reg_home = float(np.tril(matrix, -1).sum())
        reg_tie = float(np.diag(matrix).sum())
        reg_away = float(np.triu(matrix, 1).sum())
        ot_home_share = float(np.clip(home_ot_strength / max(1e-9, home_ot_strength + away_ot_strength), 0.38, 0.62))
        full_home = reg_home + (reg_tie * ot_home_share)
        full_away = reg_away + (reg_tie * (1.0 - ot_home_share))
        total_probs = np.add.outer(np.arange(max_goals + 1), np.arange(max_goals + 1))
        over = {
            "over_5_5": round(float(matrix[total_probs > 5.5].sum()), 4),
            "over_6_5": round(float(matrix[total_probs > 6.5].sum()), 4),
        }
        under = {key.replace("over", "under"): round(1.0 - value, 4) for key, value in over.items()}
        return NHLProjectedGoalsReport(
            expected_home_goals=round(float(expected_home_goals), 4),
            expected_away_goals=round(float(expected_away_goals), 4),
            regulation_home_probability=round(reg_home, 4),
            regulation_tie_probability=round(reg_tie, 4),
            regulation_away_probability=round(reg_away, 4),
            overtime_probability=round(reg_tie, 4),
            full_game_home_probability=round(full_home, 4),
            full_game_away_probability=round(full_away, 4),
            score_matrix=np.round(matrix, 6).tolist(),
            over_probabilities=over,
            under_probabilities=under,
        )

    def goalie_scenarios(
        self,
        expected_home_goals: float,
        expected_away_goals: float,
        goalie_context: Optional[dict[str, Any]],
    ) -> list[NHLGoalieScenario]:
        ctx = dict(goalie_context or {})
        starter_home_xg, starter_away_xg, _, starter_status, _ = self.apply_goalie_adjustment(
            expected_home_goals,
            expected_away_goals,
            ctx,
        )
        starter_report = self.projected_goals_report(starter_home_xg, starter_away_xg)
        backup_ctx = {
            **ctx,
            "home_backup_goalie_flag": True,
            "away_backup_goalie_flag": True,
            "home_goalie_confirmed": False,
            "away_goalie_confirmed": False,
        }
        backup_home_xg, backup_away_xg, _, backup_status, _ = self.apply_goalie_adjustment(
            expected_home_goals,
            expected_away_goals,
            backup_ctx,
        )
        backup_report = self.projected_goals_report(backup_home_xg, backup_away_xg)
        return [
            NHLGoalieScenario("expected_starters", starter_status, starter_status, round(starter_home_xg, 4), round(starter_away_xg, 4), starter_report.full_game_home_probability, starter_report.full_game_away_probability),
            NHLGoalieScenario("backup_goalies", backup_status, backup_status, round(backup_home_xg, 4), round(backup_away_xg, 4), backup_report.full_game_home_probability, backup_report.full_game_away_probability),
        ]

    def build_value_report(
        self,
        *,
        snapshot: Optional[pd.Series],
        odds_moneyline: dict[str, float],
        goalie_context: Optional[dict[str, Any]] = None,
        model_probabilities: Optional[dict[str, float]] = None,
        base_confidence: float = 0.62,
        clv_status: str = "pending",
        prediction_time: object | None = None,
        signal_odds_moneyline: Optional[dict[str, float]] = None,
        closing_odds_moneyline: Optional[dict[str, float]] = None,
        closing_odds_timestamp: object | None = None,
    ) -> NHLValueReport:
        ctx = dict(goalie_context or {})
        if prediction_time is None:
            prediction_time = ctx.get("prediction_time")
        base_home_xg, base_away_xg = self.projected_goals_from_snapshot(snapshot)
        home_xg, away_xg, warnings, goalie_status, goalie_confidence = self.apply_goalie_adjustment(
            base_home_xg,
            base_away_xg,
            goalie_context,
        )
        report = self.projected_goals_report(
            home_xg,
            away_xg,
            home_ot_strength=_safe_float((goalie_context or {}).get("home_3v3_ot_strength"), 0.5),
            away_ot_strength=_safe_float((goalie_context or {}).get("away_3v3_ot_strength"), 0.5),
        )
        model_home = report.full_game_home_probability
        model_away = report.full_game_away_probability
        if model_probabilities:
            normalized = normalize_two_probs(
                float(model_probabilities.get("home", model_home)),
                float(model_probabilities.get("away", model_away)),
            )
            model_home, model_away = normalized.as_tuple()

        market_decisions = MarketEngine().evaluate_market(
            sport="nhl",
            market="moneyline",
            event="nhl probability/value report",
            prediction_time=prediction_time,
            signal_odds=signal_odds_moneyline,
            closing_odds=closing_odds_moneyline,
            closing_odds_timestamp=closing_odds_timestamp,
            outcomes=[
                MarketOutcomeInput(outcome="home", odds=float(odds_moneyline["home"]), model_probability=float(model_home)),
                MarketOutcomeInput(outcome="away", odds=float(odds_moneyline["away"]), model_probability=float(model_away)),
            ],
        )
        values = [
            NHLMarketOutcomeValue(
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
            for decision in market_decisions
        ]
        scenarios = self.goalie_scenarios(base_home_xg, base_away_xg, goalie_context)
        sensitivity = 0.0
        if len(scenarios) >= 2:
            sensitivity = abs(scenarios[0].full_game_home_probability - scenarios[1].full_game_home_probability)
        return NHLValueReport(
            projected_goals=report,
            market_values=values,
            goalie_status=goalie_status,
            goalie_scenarios=scenarios,
            goalie_sensitivity_home_prob=round(float(sensitivity), 4),
            confidence=round(max(0.05, min(0.95, base_confidence * goalie_confidence)), 3),
            clv_status=clv_status,
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
