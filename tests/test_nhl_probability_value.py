from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.nhl_features import NHLFeatureEngineer
from src.models.nhl_side_model import NHLSideModel


def _snapshot() -> pd.Series:
    return pd.Series(
        {
            "home_5v5_xgf_pg_10": 3.05,
            "away_5v5_xgf_pg_10": 2.65,
            "home_5v5_xga_pg_10": 2.40,
            "away_5v5_xga_pg_10": 2.95,
            "home_pp_pct_10": 0.24,
            "away_pp_pct_10": 0.19,
            "home_pk_pct_10": 0.83,
            "away_pk_pct_10": 0.76,
            "nhl_league_goal_environment": 6.05,
        }
    )


def test_nhl_projected_goals_report_separates_regulation_and_full_game() -> None:
    model = NHLSideModel()

    report = model.projected_goals_report(3.2, 2.7)

    reg_total = (
        report.regulation_home_probability
        + report.regulation_tie_probability
        + report.regulation_away_probability
    )
    full_total = report.full_game_home_probability + report.full_game_away_probability
    assert reg_total == pytest.approx(1.0, abs=0.0015)
    assert full_total == pytest.approx(1.0, abs=0.0015)
    assert report.overtime_probability == report.regulation_tie_probability
    assert report.regulation_tie_probability > 0.0
    assert np.asarray(report.score_matrix).sum() == pytest.approx(1.0, abs=1e-5)


def test_nhl_value_report_uses_no_vig_moneyline_and_ev() -> None:
    model = NHLSideModel()

    report = model.build_value_report(
        snapshot=_snapshot(),
        odds_moneyline={"home": 1.91, "away": 2.02},
        goalie_context={"home_goalie_confirmed": 1, "away_goalie_confirmed": 1},
        model_probabilities={"home": 0.56, "away": 0.44},
    )

    market_total = sum(value.no_vig_market_probability for value in report.market_values)
    assert market_total == pytest.approx(1.0, abs=0.0015)
    home = next(value for value in report.market_values if value.outcome == "home")
    assert home.no_vig_market_probability < home.market_implied_probability
    assert home.edge == pytest.approx(0.56 - home.no_vig_market_probability, abs=0.0001)
    assert home.expected_value == pytest.approx((0.56 * 1.91) - 1.0, abs=0.0001)
    assert home.recommended_action in {"bet", "monitor", "pass"}
    assert home.decision_reason
    assert report.goalie_status == "confirmed"


def test_unconfirmed_goalie_outputs_scenarios_and_reduces_confidence() -> None:
    model = NHLSideModel()

    confirmed = model.build_value_report(
        snapshot=_snapshot(),
        odds_moneyline={"home": 1.9, "away": 2.05},
        goalie_context={
            "home_goalie_confirmed": 1,
            "away_goalie_confirmed": 1,
            "home_goalie_gsax_long_term": 6.0,
            "away_goalie_gsax_long_term": -3.0,
        },
    )
    uncertain = model.build_value_report(
        snapshot=_snapshot(),
        odds_moneyline={"home": 1.9, "away": 2.05},
        goalie_context={
            "home_goalie_confirmed": 0,
            "away_goalie_confirmed": 0,
            "home_goalie_gsax_long_term": 6.0,
            "away_goalie_gsax_long_term": -3.0,
            "home_goalie_quality_gap": 2.0,
            "away_goalie_quality_gap": 3.0,
        },
    )

    assert uncertain.confidence < confirmed.confidence
    assert uncertain.goalie_status == "unconfirmed"
    assert len(uncertain.goalie_scenarios) == 2
    assert uncertain.goalie_sensitivity_home_prob >= 0.0
    assert any("goalie" in warning for warning in uncertain.warnings)


def test_nhl_feature_timestamp_safety_detects_goalie_leakage() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-02-01 19:00", "2026-02-02 19:00"]),
            "goalie_as_of": pd.to_datetime(["2026-02-01 18:00", "2026-02-02 19:03"]),
            "odds_as_of": pd.to_datetime(["2026-02-01 17:00", "2026-02-02 18:45"]),
        }
    )

    assert NHLSideModel.validate_feature_timestamps(frame) == ["goalie_as_of"]


def test_nhl_feature_engineer_adds_5v5_goalie_and_schedule_hooks() -> None:
    engineer = NHLFeatureEngineer()
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=4, freq="2D"),
            "home_team": ["A", "B", "A", "B"],
            "away_team": ["B", "A", "B", "A"],
            "home_xgf_pg_10": [3.0, 2.7, 3.2, 2.8],
            "away_xgf_pg_10": [2.6, 3.1, 2.7, 3.0],
            "home_xga_pg_10": [2.5, 2.9, 2.4, 2.8],
            "away_xga_pg_10": [2.9, 2.4, 3.0, 2.5],
            "home_cf_pct_10": [0.53, 0.49, 0.55, 0.50],
            "away_cf_pct_10": [0.48, 0.54, 0.47, 0.52],
            "home_ff_pct_10": [0.52, 0.50, 0.54, 0.51],
            "away_ff_pct_10": [0.49, 0.53, 0.48, 0.52],
            "home_b2b": [0, 1, 0, 1],
            "away_b2b": [1, 0, 1, 0],
            "home_rest_days": [3, 1, 4, 1],
            "away_rest_days": [1, 4, 1, 3],
            "home_games_L3D": [0, 1, 0, 1],
            "away_games_L3D": [1, 0, 1, 0],
            "home_games_L5D": [1, 3, 1, 3],
            "away_games_L5D": [3, 1, 3, 1],
            "home_games_L7D": [2, 3, 2, 3],
            "away_games_L7D": [3, 2, 3, 2],
            "home_games_L10D": [3, 4, 3, 4],
            "away_games_L10D": [4, 3, 4, 3],
        }
    )

    featured = engineer._add_5v5_strength_features(frame.copy())
    featured = engineer._add_goalie_features(featured)
    featured = engineer._add_nhl_schedule_context(featured)

    assert {
        "home_5v5_xgf_pg_10",
        "away_5v5_xg_share_10",
        "home_goalie_gsax",
        "goalie_gsax_gap",
        "home_three_in_four",
        "goalie_rotation_likelihood_b2b",
    }.issubset(featured.columns)
    assert featured["home_goalie_save_pct"].iloc[0] == pytest.approx(0.910)
