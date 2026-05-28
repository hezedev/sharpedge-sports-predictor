from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.markets.engine import (
    edge_vs_market,
    expected_value_per_unit,
    implied_probability,
)


@dataclass
class PredictionFactor:
    name: str
    category: str
    value: float
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrueProbabilityEstimate:
    sport: str
    market: str
    selection: str
    base_prob: float
    adjusted_prob: float
    confidence: float
    confidence_low: float
    confidence_high: float
    factors: list[PredictionFactor]
    adjustments: list[PredictionFactor]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["factors"] = [factor.to_dict() for factor in self.factors]
        payload["adjustments"] = [factor.to_dict() for factor in self.adjustments]
        return payload


@dataclass
class PricingDecision:
    true_prob: float
    market_prob: float
    fair_prob: float
    vig_free_implied_prob: float
    fair_odds: float
    offered_odds: float
    edge: float
    minimum_acceptable_odds: float
    lower_bound_prob: float
    lower_bound_edge: float
    lower_bound_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SPORT_CONTEXT_CAP = {
    "soccer": 0.035,
    "basketball": 0.03,
    "tennis": 0.025,
    "tennis_wta": 0.015,
    "mlb": 0.025,
    "nhl": 0.025,
}

_SPORT_CONFIDENCE_BAND = {
    "soccer": 0.04,
    "basketball": 0.035,
    "tennis": 0.03,
    "tennis_wta": 0.028,
    "mlb": 0.04,
    "nhl": 0.035,
}


def _clip_prob(value: float) -> float:
    return max(0.01, min(0.99, value))


def derive_confidence_range(*, sport: str, probability: float, confidence: float) -> tuple[float, float]:
    base_band = _SPORT_CONFIDENCE_BAND.get(sport, 0.06)
    # Higher confidence narrows the range, but never collapses it to zero.
    band = max(0.01, base_band * (1.05 - max(0.05, min(0.95, confidence))))
    low = _clip_prob(probability - band)
    high = _clip_prob(probability + band)
    return round(low, 4), round(high, 4)


def estimate_true_probability(
    *,
    sport: str,
    market: str,
    selection: str,
    base_prob: float,
    factors: list[PredictionFactor] | None = None,
    adjustments: list[PredictionFactor] | None = None,
    confidence: float = 0.6,
) -> TrueProbabilityEstimate:
    factors = factors or []
    adjustments = adjustments or []
    cap = _SPORT_CONTEXT_CAP.get(sport, 0.03)
    context_shift = sum(factor.value for factor in adjustments)
    context_shift = max(-cap, min(cap, context_shift))
    adjusted_prob = _clip_prob(base_prob + context_shift)
    confidence_low, confidence_high = derive_confidence_range(
        sport=sport,
        probability=adjusted_prob,
        confidence=confidence,
    )
    return TrueProbabilityEstimate(
        sport=sport,
        market=market,
        selection=selection,
        base_prob=round(base_prob, 4),
        adjusted_prob=round(adjusted_prob, 4),
        confidence=round(confidence, 3),
        confidence_low=confidence_low,
        confidence_high=confidence_high,
        factors=factors,
        adjustments=adjustments,
    )


def build_pricing_decision(
    *,
    true_prob: float,
    offered_odds: float,
    fair_prob: float,
    min_edge: float = 0.0,
    lower_bound_prob: float | None = None,
) -> PricingDecision:
    market_prob = implied_probability(offered_odds) if offered_odds > 1.0 else 0.0
    fair_odds = (1.0 / true_prob) if true_prob > 0 else 0.0
    edge = expected_value_per_unit(true_prob, offered_odds) if offered_odds > 1.0 else -1.0
    minimum_acceptable_odds = ((1.0 + min_edge) / true_prob) if true_prob > 0 else 0.0
    lower_prob = _clip_prob(lower_bound_prob if lower_bound_prob is not None else true_prob)
    lower_bound_edge = expected_value_per_unit(lower_prob, offered_odds) if offered_odds > 1.0 else -1.0
    lower_bound_passed = edge_vs_market(lower_prob, fair_prob) > 0
    return PricingDecision(
        true_prob=round(true_prob, 4),
        market_prob=round(market_prob, 4),
        fair_prob=round(fair_prob, 4),
        vig_free_implied_prob=round(fair_prob, 4),
        fair_odds=round(fair_odds, 3) if fair_odds else 0.0,
        offered_odds=round(offered_odds, 3),
        edge=round(edge, 4),
        minimum_acceptable_odds=round(minimum_acceptable_odds, 3) if minimum_acceptable_odds else 0.0,
        lower_bound_prob=round(lower_prob, 4),
        lower_bound_edge=round(lower_bound_edge, 4),
        lower_bound_passed=lower_bound_passed,
    )
