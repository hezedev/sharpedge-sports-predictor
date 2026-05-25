from __future__ import annotations

from src.committee import ModelMindDecision, ModelVerdict, QuantModelMind


def _base_candidate() -> dict:
    return {
        "sport": "soccer",
        "market": "moneyline",
        "team": "Alpha FC",
        "ml_prob": 0.58,
        "market_implied_prob": 0.5263,
        "vig_free_implied_prob": 0.51,
        "fair_odds": 1.724,
        "minimum_acceptable_odds": 1.78,
        "odds": 1.9,
        "edge": 0.102,
        "confidence_range_low": 0.54,
        "confidence_range_high": 0.62,
        "lower_bound_passed": True,
        "recommended_market": "moneyline",
    }


def test_model_mind_holds_when_market_data_is_missing() -> None:
    candidate = _base_candidate()
    candidate["odds"] = None

    decision = QuantModelMind(min_edge=0.03).evaluate(candidate)

    assert decision.model_verdict == ModelVerdict.HOLD
    assert "market data is missing or incomplete" in decision.reasons


def test_model_mind_rejects_edge_below_threshold() -> None:
    candidate = _base_candidate()
    candidate["edge"] = 0.02

    decision = QuantModelMind(min_edge=0.03).evaluate(candidate)

    assert decision.model_verdict == ModelVerdict.NO_BET
    assert "edge is below the configured threshold" in decision.reasons


def test_model_mind_rejects_when_odds_are_too_short() -> None:
    candidate = _base_candidate()
    candidate["odds"] = 1.7

    decision = QuantModelMind(min_edge=0.03).evaluate(candidate)

    assert decision.model_verdict == ModelVerdict.NO_BET
    assert "current odds are below the minimum acceptable odds" in decision.reasons


def test_model_mind_holds_when_confidence_interval_fails() -> None:
    candidate = _base_candidate()
    candidate["lower_bound_passed"] = False

    decision = QuantModelMind(min_edge=0.03).evaluate(candidate)

    assert decision.model_verdict == ModelVerdict.HOLD
    assert "confidence interval lower bound does not support the pick" in decision.reasons


def test_model_mind_rejects_when_model_does_not_beat_vig_free_market() -> None:
    candidate = _base_candidate()
    candidate["ml_prob"] = 0.5
    candidate["vig_free_implied_prob"] = 0.51

    decision = QuantModelMind(min_edge=0.03).evaluate(candidate)

    assert decision.model_verdict == ModelVerdict.NO_BET
    assert "model probability does not beat the vig-free market probability" in decision.reasons


def test_model_mind_returns_bet_for_strong_valid_candidate() -> None:
    decision = QuantModelMind(min_edge=0.03).evaluate(_base_candidate())

    assert decision.model_verdict == ModelVerdict.BET
    assert decision.model_probability == 0.58
    assert decision.market_implied_probability == 0.5263
    assert decision.vig_free_market_probability == 0.51
    assert decision.current_odds == 1.9
    assert decision.estimated_edge == 0.102
    assert decision.confidence_interval == (0.54, 0.62)
    assert decision.risk_tier in {"low", "medium", "high"}
    assert decision.suggested_market == "moneyline"
    assert decision.parlay_suitability in {"good_leg", "small_parlay_only", "avoid"}


def test_model_mind_decision_serializes_to_json_safe_dict() -> None:
    decision = ModelMindDecision(
        model_verdict=ModelVerdict.BET,
        model_probability=0.58,
        market_implied_probability=0.5263,
        vig_free_market_probability=0.51,
        fair_odds=1.724,
        minimum_acceptable_odds=1.78,
        current_odds=1.9,
        estimated_edge=0.102,
        confidence_interval=(0.54, 0.62),
        risk_tier="medium",
        suggested_market="moneyline",
        parlay_suitability="small_parlay_only",
        reasons=("positive edge", "lower bound passed"),
        metadata={"sport": "soccer"},
    )

    payload = decision.to_dict()

    assert payload["model_verdict"] == "BET"
    assert payload["confidence_interval"] == [0.54, 0.62]
    assert payload["reasons"] == ["positive edge", "lower bound passed"]
    assert payload["metadata"]["sport"] == "soccer"
