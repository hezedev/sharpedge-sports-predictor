import pytest

from src.markets.engine import (
    MarketEngine,
    MarketEngineConfig,
    MarketOutcomeInput,
    calculate_clv,
    calculate_overround,
    convert_odds_to_decimal,
    edge_vs_market,
    expected_value_per_unit,
    implied_probability,
    remove_vig_probabilities,
)


def test_odds_conversion_supports_decimal_american_and_fractional() -> None:
    assert convert_odds_to_decimal(2.5, "decimal") == pytest.approx(2.5)
    assert convert_odds_to_decimal("+150", "american") == pytest.approx(2.5)
    assert convert_odds_to_decimal("-200", "american") == pytest.approx(1.5)
    assert convert_odds_to_decimal("3/2", "fractional") == pytest.approx(2.5)
    assert implied_probability(2.5) == pytest.approx(0.4)


def test_overround_and_no_vig_proportional_normalization() -> None:
    raw = [1 / 1.91, 1 / 1.91]

    assert calculate_overround(raw) == pytest.approx(0.04712, abs=0.0001)
    no_vig = remove_vig_probabilities(raw)

    assert sum(no_vig) == pytest.approx(1.0)
    assert no_vig == pytest.approx([0.5, 0.5])


def test_shin_no_vig_is_available_and_normalized() -> None:
    raw = [0.58, 0.30, 0.20]

    no_vig = remove_vig_probabilities(raw, method="shin")

    assert sum(no_vig) == pytest.approx(1.0)
    assert all(0 < p < 1 for p in no_vig)


def test_ev_and_edge_calculations() -> None:
    assert expected_value_per_unit(0.55, 2.10) == pytest.approx(0.155)
    assert edge_vs_market(0.55, 0.50) == pytest.approx(0.05)


def test_market_engine_outputs_bet_decision_with_no_vig_and_paper_mode() -> None:
    engine = MarketEngine(MarketEngineConfig(min_edge=0.03, min_ev=0.02, min_confidence=0.60))

    decisions = engine.evaluate_market(
        sport="basketball",
        market="moneyline",
        event="A vs B",
        outcomes=[
            MarketOutcomeInput("home", 2.05, 0.56, bookmaker="Book A", source="odds_api", confidence=0.72),
            MarketOutcomeInput("away", 1.85, 0.44, bookmaker="Book A", source="odds_api", confidence=0.72),
        ],
        model_version="nba_v1",
    )

    home = next(decision for decision in decisions if decision.outcome == "home")
    assert home.recommended_action == "bet"
    assert home.paper_trade is True
    assert home.stake_units == 0.0
    assert home.market_no_vig_probability < home.model_probability
    assert home.expected_value == pytest.approx(0.148, abs=0.001)
    assert home.bookmaker == "Book A"
    assert home.source == "odds_api"


def test_clv_calculation_and_report_fields() -> None:
    engine = MarketEngine()

    decision = engine.evaluate_market(
        sport="soccer",
        market="1x2",
        event="A vs B",
        outcomes=[
            MarketOutcomeInput("home", 2.10, 0.52),
            MarketOutcomeInput("draw", 3.40, 0.27),
            MarketOutcomeInput("away", 3.80, 0.21),
        ],
        signal_odds={"home": 2.20, "draw": 3.50, "away": 3.70},
        closing_odds={"home": 1.95, "draw": 3.60, "away": 4.10},
        model_version="soccer_v1",
    )[0]

    assert calculate_clv(bet_decimal_odds=2.10, closing_decimal_odds=1.95) == pytest.approx(0.076923, abs=1e-6)
    assert decision.clv.signal_decimal_odds == pytest.approx(2.20)
    assert decision.clv.closing_decimal_odds == pytest.approx(1.95)
    assert decision.clv.clv_percentage == pytest.approx(0.076923)
    assert decision.clv.sport == "soccer"
    assert decision.clv.market_type == "1x2"
    assert decision.clv.model_version == "soccer_v1"


def test_fractional_kelly_stake_respects_cap() -> None:
    engine = MarketEngine(
        MarketEngineConfig(
            min_edge=0.01,
            min_ev=0.01,
            min_confidence=0.50,
            stake_mode="fractional_kelly",
            kelly_fraction=1.0,
            max_stake_units=0.03,
        )
    )

    decision = engine.evaluate_market(
        sport="nhl",
        market="moneyline",
        event="A vs B",
        outcomes=[
            MarketOutcomeInput("home", 2.20, 0.60, confidence=0.80),
            MarketOutcomeInput("away", 1.75, 0.40, confidence=0.80),
        ],
    )[0]

    assert decision.recommended_action == "bet"
    assert decision.paper_trade is False
    assert decision.stake_units == pytest.approx(0.03)


def test_negative_ev_and_risk_filters_pass_logic() -> None:
    engine = MarketEngine(MarketEngineConfig(min_edge=0.03, min_ev=0.02, min_confidence=0.65, max_stale_data_risk=0.20))

    decisions = engine.evaluate_market(
        sport="basketball",
        market="moneyline",
        event="A vs B",
        outcomes=[
            MarketOutcomeInput("home", 1.80, 0.50, confidence=0.80),
            MarketOutcomeInput("away", 2.05, 0.50, confidence=0.50, stale_data_risk=0.60),
        ],
    )

    home = next(decision for decision in decisions if decision.outcome == "home")
    away = next(decision for decision in decisions if decision.outcome == "away")
    assert home.recommended_action == "pass"
    assert home.expected_value < 0
    assert away.recommended_action == "pass"
    assert "model confidence below threshold" in away.warnings
    assert "stale-data risk above threshold" in away.warnings
