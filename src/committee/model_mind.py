from __future__ import annotations

from typing import Any

from config import settings

from .contracts import ModelMindDecision, ModelVerdict


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class QuantModelMind:
    """
    Thin adapter around the existing pricing math.

    This does not replace or re-run the production model pipeline. It simply
    normalizes an existing candidate/bet payload into a stable Model Mind
    contract for the future committee architecture.
    """

    def __init__(self, *, min_edge: float | None = None) -> None:
        risk_cfg = settings.get("risk", {}).get("kelly", {})
        self.min_edge = float(min_edge if min_edge is not None else risk_cfg.get("min_edge", 0.03) or 0.03)

    def evaluate(self, candidate: dict[str, Any]) -> ModelMindDecision:
        model_probability = _coerce_float(candidate.get("ml_prob"))
        market_implied_probability = _coerce_float(candidate.get("market_implied_prob"))
        vig_free_market_probability = _coerce_float(candidate.get("vig_free_implied_prob"))
        fair_odds = _coerce_float(candidate.get("fair_odds"))
        minimum_acceptable_odds = _coerce_float(candidate.get("minimum_acceptable_odds"))
        current_odds = _coerce_float(candidate.get("odds"))
        estimated_edge = _coerce_float(candidate.get("edge"))
        confidence_low = _coerce_float(candidate.get("confidence_range_low"))
        confidence_high = _coerce_float(candidate.get("confidence_range_high"))
        lower_bound_passed = candidate.get("lower_bound_passed")

        current_market = str(candidate.get("market", "") or "")
        suggested_market = str(candidate.get("recommended_market", current_market) or current_market)

        reasons: list[str] = []
        verdict = ModelVerdict.BET

        if not self._has_market_data(
            model_probability=model_probability,
            market_implied_probability=market_implied_probability,
            vig_free_market_probability=vig_free_market_probability,
            current_odds=current_odds,
        ):
            verdict = ModelVerdict.HOLD
            reasons.append("market data is missing or incomplete")
        else:
            if model_probability is not None and vig_free_market_probability is not None:
                if model_probability <= vig_free_market_probability:
                    verdict = ModelVerdict.NO_BET
                    reasons.append("model probability does not beat the vig-free market probability")

            if estimated_edge is None:
                verdict = ModelVerdict.HOLD
                reasons.append("estimated edge is missing")
            elif estimated_edge < self.min_edge and verdict == ModelVerdict.BET:
                verdict = ModelVerdict.NO_BET
                reasons.append("edge is below the configured threshold")

            if (
                verdict == ModelVerdict.BET
                and current_odds is not None
                and minimum_acceptable_odds is not None
                and current_odds < minimum_acceptable_odds
            ):
                verdict = ModelVerdict.NO_BET
                reasons.append("current odds are below the minimum acceptable odds")

            if lower_bound_passed is False and verdict == ModelVerdict.BET:
                verdict = ModelVerdict.HOLD
                reasons.append("confidence interval lower bound does not support the pick")

        if verdict == ModelVerdict.BET:
            reasons.append("model pricing clears the threshold and supports the current price")

        confidence_interval = (confidence_low, confidence_high)
        risk_tier = self._derive_risk_tier(
            verdict=verdict,
            estimated_edge=estimated_edge,
            confidence_low=confidence_low,
            confidence_high=confidence_high,
        )
        parlay_suitability = self._derive_parlay_suitability(
            verdict=verdict,
            risk_tier=risk_tier,
            market=suggested_market,
        )

        return ModelMindDecision(
            model_verdict=verdict,
            model_probability=model_probability,
            market_implied_probability=market_implied_probability,
            vig_free_market_probability=vig_free_market_probability,
            fair_odds=fair_odds,
            minimum_acceptable_odds=minimum_acceptable_odds,
            current_odds=current_odds,
            estimated_edge=estimated_edge,
            confidence_interval=confidence_interval,
            risk_tier=risk_tier,
            suggested_market=suggested_market,
            parlay_suitability=parlay_suitability,
            reasons=tuple(reasons),
            metadata={
                "sport": str(candidate.get("sport", "") or ""),
                "market": current_market,
                "selection": str(candidate.get("team", "") or ""),
                "min_edge_threshold": self.min_edge,
            },
        )

    @staticmethod
    def _has_market_data(
        *,
        model_probability: float | None,
        market_implied_probability: float | None,
        vig_free_market_probability: float | None,
        current_odds: float | None,
    ) -> bool:
        return (
            model_probability is not None
            and market_implied_probability is not None
            and vig_free_market_probability is not None
            and current_odds is not None
            and current_odds > 1.0
        )

    def _derive_risk_tier(
        self,
        *,
        verdict: ModelVerdict,
        estimated_edge: float | None,
        confidence_low: float | None,
        confidence_high: float | None,
    ) -> str:
        if verdict != ModelVerdict.BET:
            return "avoid"

        if confidence_low is None or confidence_high is None or estimated_edge is None:
            return "medium"

        interval_width = max(0.0, confidence_high - confidence_low)
        if estimated_edge >= max(self.min_edge * 2.0, 0.08) and interval_width <= 0.08:
            return "low"
        if estimated_edge >= self.min_edge and interval_width <= 0.12:
            return "medium"
        return "high"

    @staticmethod
    def _derive_parlay_suitability(
        *,
        verdict: ModelVerdict,
        risk_tier: str,
        market: str,
    ) -> str:
        if verdict != ModelVerdict.BET:
            return "avoid"

        market_key = market.lower()
        if risk_tier == "low" and market_key in {"moneyline", "double_chance", "draw_no_bet", "totals", "spreads"}:
            return "good_leg"
        if risk_tier == "medium":
            return "small_parlay_only"
        return "avoid"
