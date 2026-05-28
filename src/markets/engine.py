from __future__ import annotations

from dataclasses import asdict, dataclass, field
from fractions import Fraction
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd


OddsFormat = Literal["decimal", "american", "fractional"]
VigMethod = Literal["proportional", "shin"]
StakeMode = Literal["paper", "flat", "fractional_kelly"]
MarketAction = Literal["bet", "pass", "monitor"]


@dataclass(frozen=True)
class MarketOutcomeInput:
    outcome: str
    odds: float | int | str
    model_probability: float
    bookmaker: str = "unknown"
    source: str = "unknown"
    odds_format: OddsFormat = "decimal"
    confidence: float = 1.0
    stale_data_risk: float = 0.0
    liquidity_score: float | None = None


@dataclass(frozen=True)
class MarketEngineConfig:
    min_edge: float = 0.03
    min_ev: float = 0.02
    min_confidence: float = 0.55
    max_stale_data_risk: float = 0.35
    min_liquidity_score: float = 0.0
    stake_mode: StakeMode = "paper"
    flat_stake_units: float = 1.0
    kelly_fraction: float = 0.25
    max_stake_units: float = 0.05
    daily_loss_cap_units: float = 0.05
    drawdown_protection: float = 0.20


@dataclass(frozen=True)
class MarketPrice:
    outcome: str
    decimal_odds: float
    raw_implied_probability: float
    no_vig_probability: float
    market_overround: float
    bookmaker: str
    source: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CLVReport:
    signal_decimal_odds: float | None = None
    bet_decimal_odds: float | None = None
    closing_decimal_odds: float | None = None
    closing_no_vig_probability: float | None = None
    clv_percentage: float | None = None
    sport: str = ""
    market_type: str = ""
    model_version: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MarketDecision:
    sport: str
    market: str
    event: str
    outcome: str
    model_probability: float
    market_no_vig_probability: float
    raw_implied_probability: float
    market_overround: float
    edge: float
    decimal_odds: float
    expected_value: float
    break_even_probability: float
    bookmaker: str
    source: str
    recommended_action: MarketAction
    reason: str
    warnings: tuple[str, ...] = ()
    confidence: float = 1.0
    stake_units: float = 0.0
    paper_trade: bool = True
    clv: CLVReport = field(default_factory=CLVReport)
    model_version: str = ""

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["clv"] = self.clv.as_dict()
        return payload


def convert_odds_to_decimal(odds: float | int | str, odds_format: OddsFormat = "decimal") -> float:
    if odds_format == "decimal":
        value = float(odds)
    elif odds_format == "american":
        american = float(odds)
        if american > 0:
            value = 1.0 + (american / 100.0)
        elif american < 0:
            value = 1.0 + (100.0 / abs(american))
        else:
            raise ValueError("American odds cannot be zero")
    elif odds_format == "fractional":
        if isinstance(odds, str):
            fraction = Fraction(odds.strip())
            value = 1.0 + (fraction.numerator / fraction.denominator)
        else:
            value = 1.0 + float(odds)
    else:
        raise ValueError(f"Unsupported odds format: {odds_format}")

    if value <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.0")
    return float(value)


def implied_probability(decimal_odds: float) -> float:
    if decimal_odds <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.0")
    return 1.0 / decimal_odds


def calculate_overround(raw_probabilities: Sequence[float]) -> float:
    return float(sum(raw_probabilities) - 1.0)


def remove_vig_probabilities(
    raw_probabilities: Sequence[float],
    *,
    method: VigMethod = "proportional",
) -> list[float]:
    probs = [max(0.0, float(p)) for p in raw_probabilities]
    total = sum(probs)
    if total <= 0:
        return probs

    if method == "proportional":
        return [p / total for p in probs]

    if method == "shin":
        # Practical bounded Shin-style shrink. For two-way markets this stays
        # close to proportional; for larger overrounds it redistributes a bit
        # more margin away from very short favorites.
        overround = max(0.0, total - 1.0)
        z = min(0.25, overround / max(1.0, len(probs)))
        adjusted = [max(1e-9, (p * (1.0 - z)) / max(1e-9, 1.0 - (z * p))) for p in probs]
        adj_total = sum(adjusted)
        return [p / adj_total for p in adjusted]

    raise ValueError(f"Unsupported vig-removal method: {method}")


def expected_value_per_unit(model_probability: float, decimal_odds: float) -> float:
    return (float(model_probability) * float(decimal_odds)) - 1.0


def edge_vs_market(model_probability: float, no_vig_probability: float) -> float:
    return float(model_probability) - float(no_vig_probability)


def fractional_kelly_stake(
    *,
    model_probability: float,
    decimal_odds: float,
    fraction: float,
    max_stake_units: float,
) -> float:
    if model_probability <= 0 or model_probability >= 1 or decimal_odds <= 1:
        return 0.0
    b = decimal_odds - 1.0
    q = 1.0 - model_probability
    full_kelly = ((model_probability * b) - q) / b
    if full_kelly <= 0:
        return 0.0
    return round(float(min(full_kelly * fraction, max_stake_units)), 6)


def calculate_clv(
    *,
    bet_decimal_odds: float,
    closing_decimal_odds: float,
) -> float:
    if bet_decimal_odds <= 1.0 or closing_decimal_odds <= 1.0:
        return 0.0
    return float((bet_decimal_odds / closing_decimal_odds) - 1.0)


def _timestamp_not_after(value: object, reference: object) -> bool:
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    reference_time = pd.to_datetime(reference, errors="coerce", utc=True)
    if pd.isna(timestamp) or pd.isna(reference_time):
        return False
    return bool(timestamp <= reference_time)


class MarketEngine:
    def __init__(self, config: MarketEngineConfig | None = None) -> None:
        self.config = config or MarketEngineConfig()

    def price_market(
        self,
        outcomes: Sequence[MarketOutcomeInput],
        *,
        vig_method: VigMethod = "proportional",
    ) -> list[MarketPrice]:
        decimals = [convert_odds_to_decimal(item.odds, item.odds_format) for item in outcomes]
        raw = [implied_probability(odds) for odds in decimals]
        no_vig = remove_vig_probabilities(raw, method=vig_method)
        overround = calculate_overround(raw)
        return [
            MarketPrice(
                outcome=item.outcome,
                decimal_odds=round(decimals[idx], 4),
                raw_implied_probability=round(raw[idx], 6),
                no_vig_probability=round(no_vig[idx], 6),
                market_overround=round(overround, 6),
                bookmaker=item.bookmaker,
                source=item.source,
            )
            for idx, item in enumerate(outcomes)
        ]

    def evaluate_market(
        self,
        *,
        sport: str,
        market: str,
        event: str,
        outcomes: Sequence[MarketOutcomeInput],
        vig_method: VigMethod = "proportional",
        model_version: str = "",
        prediction_time: object | None = None,
        signal_odds: Mapping[str, float | int | str] | None = None,
        closing_odds: Mapping[str, float | int | str] | None = None,
        closing_odds_timestamp: object | None = None,
        closing_vig_method: VigMethod = "proportional",
        current_daily_loss_units: float = 0.0,
        current_drawdown: float = 0.0,
    ) -> list[MarketDecision]:
        prices = self.price_market(outcomes, vig_method=vig_method)
        closing_prices: dict[str, MarketPrice] = {}
        if closing_odds:
            closing_inputs = [
                MarketOutcomeInput(
                    outcome=item.outcome,
                    odds=closing_odds[item.outcome],
                    model_probability=item.model_probability,
                    bookmaker=item.bookmaker,
                    source="closing",
                    odds_format=item.odds_format,
                    confidence=item.confidence,
                    stale_data_risk=item.stale_data_risk,
                    liquidity_score=item.liquidity_score,
                )
                for item in outcomes
                if item.outcome in closing_odds
            ]
            closing_prices = {
                price.outcome: price
                for price in self.price_market(closing_inputs, vig_method=closing_vig_method)
            }

        decisions: list[MarketDecision] = []
        for item, price in zip(outcomes, prices):
            model_probability = float(np.clip(item.model_probability, 0.0, 1.0))
            ev = expected_value_per_unit(model_probability, price.decimal_odds)
            edge = edge_vs_market(model_probability, price.no_vig_probability)
            warnings: list[str] = []
            if (
                closing_odds
                and closing_odds_timestamp is not None
                and prediction_time is not None
                and _timestamp_not_after(closing_odds_timestamp, prediction_time)
            ):
                warnings.append("closing odds timestamp is not after prediction time")

            if item.confidence < self.config.min_confidence:
                warnings.append("model confidence below threshold")
            if item.stale_data_risk > self.config.max_stale_data_risk:
                warnings.append("stale-data risk above threshold")
            if item.liquidity_score is not None and item.liquidity_score < self.config.min_liquidity_score:
                warnings.append("book or market liquidity below threshold")
            if current_daily_loss_units >= self.config.daily_loss_cap_units:
                warnings.append("daily loss cap reached")
            if current_drawdown >= self.config.drawdown_protection:
                warnings.append("drawdown protection active")

            if edge >= self.config.min_edge and ev >= self.config.min_ev and not warnings:
                action: MarketAction = "bet"
                reason = "edge and EV clear configured thresholds"
            elif edge > 0 and ev > 0 and not warnings:
                action = "monitor"
                reason = "positive value but at least one threshold is not met"
            else:
                action = "pass"
                reason = "negative EV or risk filters failed"

            stake = self._stake_units(action, model_probability, price.decimal_odds)
            signal_decimal = None
            if signal_odds and item.outcome in signal_odds:
                signal_decimal = convert_odds_to_decimal(signal_odds[item.outcome], item.odds_format)
            closing = closing_prices.get(item.outcome)
            clv = CLVReport(
                signal_decimal_odds=round(signal_decimal, 4) if signal_decimal else None,
                bet_decimal_odds=price.decimal_odds,
                closing_decimal_odds=closing.decimal_odds if closing else None,
                closing_no_vig_probability=closing.no_vig_probability if closing else None,
                clv_percentage=round(calculate_clv(bet_decimal_odds=price.decimal_odds, closing_decimal_odds=closing.decimal_odds), 6)
                if closing
                else None,
                sport=sport,
                market_type=market,
                model_version=model_version,
            )

            decisions.append(
                MarketDecision(
                    sport=sport,
                    market=market,
                    event=event,
                    outcome=item.outcome,
                    model_probability=round(model_probability, 6),
                    market_no_vig_probability=price.no_vig_probability,
                    raw_implied_probability=price.raw_implied_probability,
                    market_overround=price.market_overround,
                    edge=round(edge, 6),
                    decimal_odds=price.decimal_odds,
                    expected_value=round(ev, 6),
                    break_even_probability=round(implied_probability(price.decimal_odds), 6),
                    bookmaker=price.bookmaker,
                    source=price.source,
                    recommended_action=action,
                    reason=reason,
                    warnings=tuple(warnings),
                    confidence=round(float(item.confidence), 4),
                    stake_units=stake,
                    paper_trade=self.config.stake_mode == "paper",
                    clv=clv,
                    model_version=model_version,
                )
            )
        return decisions

    def _stake_units(self, action: MarketAction, model_probability: float, decimal_odds: float) -> float:
        if action != "bet":
            return 0.0
        if self.config.stake_mode == "paper":
            return 0.0
        if self.config.stake_mode == "flat":
            return round(min(self.config.flat_stake_units, self.config.max_stake_units), 6)
        if self.config.stake_mode == "fractional_kelly":
            return fractional_kelly_stake(
                model_probability=model_probability,
                decimal_odds=decimal_odds,
                fraction=self.config.kelly_fraction,
                max_stake_units=self.config.max_stake_units,
            )
        raise ValueError(f"Unsupported stake mode: {self.config.stake_mode}")
