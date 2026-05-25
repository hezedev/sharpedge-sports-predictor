"""Smoke tests for the manual game analyst."""

from src.analysis.manual_analyst import ManualGameAnalyst


def test_resolve_selection_from_team_names() -> None:
    analyst = ManualGameAnalyst()
    selection = analyst._resolve_selection(
        selection=None,
        bet="Arsenal moneyline",
        home_team="Arsenal",
        away_team="Chelsea",
    )
    assert selection == "home"


def test_selection_market_label_for_h2h() -> None:
    analyst = ManualGameAnalyst()
    label = analyst._selection_market_label(
        market="h2h",
        selection="away",
        home_team="Arsenal",
        away_team="Chelsea",
    )
    assert label == "Chelsea"
