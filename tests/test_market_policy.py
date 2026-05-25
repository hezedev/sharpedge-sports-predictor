from __future__ import annotations

from src.markets.policy import annotate_bet, filter_and_rank_bets, get_market_policy


def test_get_market_policy_disables_nhl_totals() -> None:
    policy = get_market_policy("nhl", "totals")

    assert policy["status"] == "disabled"
    assert policy["parlay_allowed"] is False


def test_filter_and_rank_bets_suppresses_disabled_and_promotes_preferred() -> None:
    bets = [
        {"sport": "nhl", "market": "totals", "edge": 0.12, "ml_prob": 0.6},
        {"sport": "mlb", "market": "moneyline", "edge": 0.06, "ml_prob": 0.6},
        {"sport": "soccer", "market": "moneyline", "edge": 0.09, "ml_prob": 0.55},
    ]

    ranked = filter_and_rank_bets(bets)

    assert len(ranked) == 2
    assert all(bet["production_allowed"] for bet in ranked)
    assert ranked[0]["sport"] == "soccer"
    assert ranked[0]["prediction_lane_status"] == "controlled"
    assert ranked[1]["sport"] == "mlb"
    assert ranked[1]["prediction_lane_status"] == "primary"
    assert ranked[1]["market_policy_label"] == "Limited"


def test_soccer_double_chance_and_dnb_are_preferred() -> None:
    double_chance = get_market_policy("soccer", "double_chance_home_or_draw")
    draw_no_bet = get_market_policy("soccer", "away_draw_no_bet")

    assert double_chance["status"] == "preferred"
    assert draw_no_bet["status"] == "preferred"
    assert double_chance["score"] > draw_no_bet["score"] > 80


def test_annotate_bet_adds_policy_fields() -> None:
    annotated = annotate_bet({"sport": "tennis", "market": "moneyline", "edge": 0.1, "ml_prob": 0.7})

    assert annotated["market_status"] == "preferred"
    assert annotated["market_priority_score"] >= 80
    assert annotated["market_policy_reason"]
    assert annotated["production_allowed"] is True


def test_annotate_bet_allows_limited_experimental_market_with_reduced_stake() -> None:
    annotated = annotate_bet({"sport": "mlb", "market": "moneyline", "edge": 0.08, "ml_prob": 0.58})

    assert annotated["market_status"] == "experimental"
    assert annotated["market_policy_label"] == "Limited"
    assert annotated["prediction_lane_status"] == "primary"
    assert annotated["prediction_focus_allowed"] is True
    assert annotated["production_allowed"] is True
    assert annotated["parlay_allowed"] is False
    assert annotated["stake_multiplier"] == 0.4


def test_nonfocused_nhl_moneyline_is_analysis_only() -> None:
    annotated = annotate_bet({"sport": "nhl", "market": "moneyline", "edge": 0.06, "ml_prob": 0.57})

    assert annotated["market_status"] == "experimental"
    assert annotated["market_policy_label"] == "Limited"
    assert annotated["prediction_lane_status"] == "disabled"
    assert annotated["prediction_focus_allowed"] is False
    assert annotated["production_allowed"] is False
    assert annotated["parlay_allowed"] is False


def test_nonfocused_basketball_moneyline_is_analysis_only() -> None:
    annotated = annotate_bet({"sport": "basketball", "market": "moneyline", "edge": 0.08, "ml_prob": 0.56})

    assert annotated["market_status"] == "experimental"
    assert annotated["market_policy_label"] == "Limited"
    assert annotated["prediction_lane_status"] == "disabled"
    assert annotated["prediction_focus_allowed"] is False
    assert annotated["production_allowed"] is False
    assert annotated["parlay_allowed"] is False


def test_nonfocused_mlb_spreads_are_analysis_only() -> None:
    annotated = annotate_bet({"sport": "mlb", "market": "spreads", "edge": 0.08, "ml_prob": 0.56})

    assert annotated["market_status"] == "experimental"
    assert annotated["market_policy_label"] == "Limited"
    assert annotated["prediction_lane_status"] == "disabled"
    assert annotated["prediction_focus_allowed"] is False
    assert annotated["production_allowed"] is False
    assert annotated["parlay_allowed"] is False


def test_limited_soccer_moneyline_is_controlled_focus() -> None:
    annotated = annotate_bet({"sport": "soccer", "market": "moneyline", "edge": 0.08, "ml_prob": 0.56})

    assert annotated["market_status"] == "experimental"
    assert annotated["market_policy_label"] == "Limited"
    assert annotated["prediction_lane_status"] == "controlled"
    assert annotated["prediction_focus_allowed"] is True
    assert annotated["production_allowed"] is True
    assert annotated["parlay_allowed"] is False
    assert annotated["stake_multiplier"] == 0.25


def test_annotate_bet_normalizes_soccer_market_families() -> None:
    annotated = annotate_bet({"sport": "soccer", "market": "double_chance_home_or_away", "edge": 0.08, "ml_prob": 0.7})

    assert annotated["market"] == "double_chance"
    assert annotated["market_status"] == "preferred"
    assert annotated["prediction_lane_status"] == "controlled"


def test_soccer_goals_lane_includes_totals_and_team_totals() -> None:
    totals = annotate_bet({"sport": "soccer", "market": "goals", "edge": 0.06, "ml_prob": 0.57})
    team_total = annotate_bet({"sport": "soccer", "market": "team_total", "edge": 0.06, "ml_prob": 0.57})

    assert totals["market"] == "totals"
    assert totals["prediction_lane_status"] == "controlled"
    assert totals["production_allowed"] is True
    assert team_total["market"] == "team_total"
    assert team_total["prediction_lane_status"] == "controlled"
    assert team_total["prediction_focus_allowed"] is True
