from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.soccer_features import SoccerFeatureEngineer
from src.models.soccer_score_model import SoccerScoreModel


def test_soccer_goal_report_has_explicit_three_way_probabilities() -> None:
    model = SoccerScoreModel()

    report = model.probability_report_from_expected_goals(1.55, 1.05)

    total = (
        report.home_win_probability
        + report.draw_probability
        + report.away_win_probability
    )
    assert total == pytest.approx(1.0, abs=0.0015)
    assert report.draw_probability > 0.0
    assert len(report.score_matrix) == 9
    assert len(report.score_matrix[0]) == 9
    assert np.asarray(report.score_matrix).sum() == pytest.approx(1.0, abs=1e-5)
    assert "over_2_5" in report.over_probabilities
    assert "under_2_5" in report.under_probabilities


def test_soccer_value_report_compares_against_no_vig_market_probability() -> None:
    model = SoccerScoreModel()

    report = model.build_value_report(
        expected_home_goals=1.65,
        expected_away_goals=0.95,
        odds_1x2={"home": 2.2, "draw": 3.35, "away": 3.55},
        lineup_context={"home_lineup_confirmed": 1, "away_lineup_confirmed": 1},
    )

    market_total = sum(value.no_vig_market_probability for value in report.market_values)
    assert market_total == pytest.approx(1.0, abs=0.0015)

    home = next(value for value in report.market_values if value.outcome == "home")
    assert home.no_vig_market_probability < home.market_implied_probability
    assert home.edge == pytest.approx(
        home.model_probability - home.no_vig_market_probability,
        abs=0.0001,
    )
    assert home.expected_value == pytest.approx(
        (home.model_probability * home.offered_odds) - 1.0,
        abs=0.0001,
    )
    assert home.recommended_action in {"bet", "monitor", "pass"}
    assert home.decision_reason


def test_soccer_value_report_can_use_final_blended_three_way_probabilities() -> None:
    model = SoccerScoreModel()

    report = model.build_value_report(
        expected_home_goals=1.2,
        expected_away_goals=1.2,
        odds_1x2={"home": 2.0, "draw": 3.4, "away": 4.2},
        model_probabilities={"home": 0.52, "draw": 0.27, "away": 0.21},
    )

    probs = {value.outcome: value.model_probability for value in report.market_values}
    assert probs["home"] == pytest.approx(0.52, abs=0.0001)
    assert probs["draw"] == pytest.approx(0.27, abs=0.0001)
    assert probs["away"] == pytest.approx(0.21, abs=0.0001)


def test_soccer_lineup_uncertainty_reduces_confidence_and_flags_goalkeeper() -> None:
    model = SoccerScoreModel()

    clean = model.build_value_report(
        expected_home_goals=1.4,
        expected_away_goals=1.1,
        odds_1x2={"home": 2.3, "draw": 3.2, "away": 3.1},
        lineup_context={"home_lineup_confirmed": 1, "away_lineup_confirmed": 1},
    )
    risky = model.build_value_report(
        expected_home_goals=1.4,
        expected_away_goals=1.1,
        odds_1x2={"home": 2.3, "draw": 3.2, "away": 3.1},
        lineup_context={
            "home_lineup_confirmed": 0,
            "away_lineup_confirmed": 1,
            "home_missing_goalkeeper": 1,
        },
    )

    assert risky.confidence < clean.confidence
    assert any("goalkeeper" in warning for warning in risky.warnings)


def test_soccer_feature_timestamp_safety_detects_post_kickoff_context() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-20 15:00", "2026-05-21 18:00"]),
            "lineup_as_of": pd.to_datetime(["2026-05-20 14:30", "2026-05-21 18:05"]),
            "injury_as_of": pd.to_datetime(["2026-05-20 10:00", "2026-05-21 08:00"]),
        }
    )

    unsafe = SoccerScoreModel.validate_feature_timestamps(frame)

    assert unsafe == ["lineup_as_of"]


def test_soccer_specific_features_are_prematch_and_league_adjusted() -> None:
    engineer = SoccerFeatureEngineer()
    rows = []
    teams = ["A", "B", "C", "D"]
    for i in range(28):
        home = teams[i % 4]
        away = teams[(i + 1) % 4]
        rows.append(
            {
                "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i * 3),
                "home_team": home,
                "away_team": away,
                "home_goals": 1 + (i % 3 == 0),
                "away_goals": int(i % 4 == 0),
                "home_np_xg": 1.1 + (i % 3) * 0.15,
                "away_np_xg": 0.8 + (i % 2) * 0.2,
                "result": "home_win" if i % 3 else "draw",
                "competition": "League A" if i < 14 else "League B",
                "round": "Regular Season",
                "season": 2026,
                "elo_diff": 25.0,
            }
        )
    df = pd.DataFrame(rows)

    featured = engineer._compute_soccer_xg_features(df.copy())
    featured = engineer._compute_exponential_form_features(featured)
    featured = engineer._compute_schedule_fatigue_features(featured)

    assert {
        "home_np_xg_for_rolling",
        "away_np_xg_against_rolling",
        "opponent_adjusted_xg_diff",
        "xg_adjusted_elo_diff",
        "home_exp_decay_form",
        "home_matches_last_7d",
        "league_scoring_environment",
    }.issubset(featured.columns)
    assert pd.isna(featured.loc[0, "home_np_xg_for_rolling"])
    assert featured["league_scoring_environment"].iloc[-1] > 0
