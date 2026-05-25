"""
Tests for feature engineering modules.

Uses synthetic match data to validate feature computations
without requiring API access.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.soccer_features import SoccerFeatureEngineer
from src.features.basketball_features import BasketballFeatureEngineer
from src.features.tennis_features import TennisFeatureEngineer


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def make_soccer_data(n: int = 50) -> pd.DataFrame:
    """Generate synthetic soccer match data."""
    np.random.seed(42)
    teams = ["Arsenal", "Chelsea", "Liverpool", "Man City", "Tottenham"]

    rows = []
    for i in range(n):
        home = teams[i % len(teams)]
        away = teams[(i + 2) % len(teams)]
        hg = np.random.poisson(1.4)
        ag = np.random.poisson(1.1)

        if hg > ag:
            result = "home_win"
        elif hg < ag:
            result = "away_win"
        else:
            result = "draw"

        rows.append({
            "match_id": i + 1,
            "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i * 3),
            "competition": "PL",
            "season": "2024",
            "matchday": (i % 38) + 1,
            "home_team": home,
            "home_team_id": teams.index(home) + 1,
            "away_team": away,
            "away_team_id": teams.index(away) + 1,
            "home_goals": hg,
            "away_goals": ag,
            "result": result,
            "home_ht": max(0, hg - 1),
            "away_ht": max(0, ag - 1),
        })
    return pd.DataFrame(rows)


def make_basketball_data(n: int = 50) -> pd.DataFrame:
    """Generate synthetic basketball game data."""
    np.random.seed(42)
    teams = ["Lakers", "Celtics", "Warriors", "Nets", "Heat"]

    rows = []
    for i in range(n):
        home = teams[i % len(teams)]
        away = teams[(i + 2) % len(teams)]
        hs = np.random.normal(108, 12)
        as_ = np.random.normal(106, 12)
        hs, as_ = int(max(80, hs)), int(max(80, as_))

        # No draws in basketball
        if hs == as_:
            hs += 1

        result = "home_win" if hs > as_ else "away_win"

        rows.append({
            "match_id": i + 1,
            "date": pd.Timestamp("2024-10-01") + pd.Timedelta(days=i * 2),
            "league_id": 12,
            "home_team": home,
            "home_team_id": teams.index(home) + 1,
            "away_team": away,
            "away_team_id": teams.index(away) + 1,
            "home_score": hs,
            "away_score": as_,
            "result": result,
            "home_q1": int(hs * 0.25),
            "home_q2": int(hs * 0.27),
            "home_q3": int(hs * 0.24),
            "home_q4": hs - int(hs * 0.25) - int(hs * 0.27) - int(hs * 0.24),
            "away_q1": int(as_ * 0.26),
            "away_q2": int(as_ * 0.25),
            "away_q3": int(as_ * 0.25),
            "away_q4": as_ - int(as_ * 0.26) - int(as_ * 0.25) - int(as_ * 0.25),
            "home_ot": 0,
            "away_ot": 0,
        })
    return pd.DataFrame(rows)


def make_tennis_data(n: int = 50) -> pd.DataFrame:
    """Generate synthetic tennis match data."""
    np.random.seed(42)
    players = ["Djokovic", "Alcaraz", "Sinner", "Medvedev", "Zverev"]
    surfaces = ["Hard", "Clay", "Grass"]

    rows = []
    for i in range(n):
        p1 = players[i % len(players)]
        p2 = players[(i + 2) % len(players)]
        p1_sets = np.random.choice([2, 3], p=[0.6, 0.4])
        p2_sets = np.random.choice([0, 1], p=[0.5, 0.5]) if p1_sets == 2 else 2

        result = "player1_win" if p1_sets > p2_sets else "player2_win"

        rows.append({
            "match_id": i + 1,
            "date": pd.Timestamp("2024-01-10") + pd.Timedelta(days=i * 4),
            "tournament": "Test Open",
            "tournament_id": 100,
            "surface": surfaces[i % 3],
            "round": "Round of 32",
            "player1_name": p1,
            "player1_id": players.index(p1) + 1,
            "player1_rank": players.index(p1) + 1,
            "player1_rank_pts": 8000 - players.index(p1) * 500,
            "player1_seed": players.index(p1) + 1,
            "player1_age": 24 + players.index(p1),
            "player1_ht": 185 + players.index(p1),
            "player2_name": p2,
            "player2_id": players.index(p2) + 1,
            "player2_rank": players.index(p2) + 1,
            "player2_rank_pts": 7800 - players.index(p2) * 500,
            "player2_seed": players.index(p2) + 1,
            "player2_age": 24 + players.index(p2),
            "player2_ht": 185 + players.index(p2),
            "player1_sets": p1_sets,
            "player2_sets": p2_sets,
            "p1_svpt": 60 + (i % 10),
            "p1_1stIn": 38 + (i % 6),
            "p1_ace": 6 + (i % 4),
            "p1_bpSaved": 4 + (i % 3),
            "p1_bpFaced": 6 + (i % 3),
            "p2_svpt": 58 + (i % 10),
            "p2_1stIn": 36 + (i % 6),
            "p2_ace": 5 + (i % 4),
            "p2_bpSaved": 3 + (i % 3),
            "p2_bpFaced": 6 + (i % 3),
            "round_num": 3,
            "best_of": 3,
            "tourney_level_name": "ATP250/500",
            "result": result,
            "set_scores": "6-4|7-5",
            "status": "Finished",
            "season": "2024",
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Soccer Feature Tests
# ------------------------------------------------------------------


class TestSoccerFeatures:
    """Test suite for SoccerFeatureEngineer."""

    def test_engineer_features_produces_expected_columns(self) -> None:
        """Feature engineering should add ELO, form, and goal features."""
        df = make_soccer_data(50)
        engineer = SoccerFeatureEngineer()
        featured = engineer.engineer_features(df)

        expected_cols = [
            "home_elo", "away_elo", "elo_diff",
            "home_win_form", "away_win_form", "form_diff",
            "home_goals_scored_avg", "away_goals_scored_avg",
            "h2h_home_win_rate", "h2h_meetings",
            "home_rest_days", "away_rest_days",
            "target",
        ]
        for col in expected_cols:
            assert col in featured.columns, f"Missing column: {col}"

    def test_elo_initialization(self) -> None:
        """All teams should start at the configured initial ELO."""
        df = make_soccer_data(10)
        engineer = SoccerFeatureEngineer()
        featured = engineer.engineer_features(df)

        # First match for each team should have initial ELO
        first_home_elo = featured.iloc[0]["home_elo"]
        assert first_home_elo == 1500

    def test_target_encoding(self) -> None:
        """Target should be encoded as integers."""
        df = make_soccer_data(30)
        engineer = SoccerFeatureEngineer()
        featured = engineer.engineer_features(df)

        assert "target" in featured.columns
        assert featured["target"].dtype in [np.int64, np.float64, int, float]
        unique_targets = featured["target"].dropna().unique()
        assert len(unique_targets) <= 3  # home_win, draw, away_win

    def test_no_lookahead_in_rolling(self) -> None:
        """Rolling features should not include the current row's data."""
        df = make_soccer_data(20)
        engineer = SoccerFeatureEngineer()
        featured = engineer.engineer_features(df)

        # First match for a team should have NaN rolling stats
        # (no previous data to compute from)
        first_row = featured.iloc[0]
        assert pd.isna(first_row["home_goals_scored_avg"])

    def test_home_features_include_prior_away_matches(self) -> None:
        """A team's history should include matches played on either side."""
        df = pd.DataFrame(
            [
                {
                    "match_id": 1,
                    "date": pd.Timestamp("2024-01-01"),
                    "competition": "PL",
                    "season": "2024",
                    "matchday": 1,
                    "home_team": "B",
                    "home_team_id": 2,
                    "away_team": "A",
                    "away_team_id": 1,
                    "home_goals": 0,
                    "away_goals": 2,
                    "result": "away_win",
                    "home_ht": 0,
                    "away_ht": 1,
                },
                {
                    "match_id": 2,
                    "date": pd.Timestamp("2024-01-08"),
                    "competition": "PL",
                    "season": "2024",
                    "matchday": 2,
                    "home_team": "A",
                    "home_team_id": 1,
                    "away_team": "C",
                    "away_team_id": 3,
                    "home_goals": 1,
                    "away_goals": 1,
                    "result": "draw",
                    "home_ht": 1,
                    "away_ht": 0,
                },
            ]
        )
        engineer = SoccerFeatureEngineer()
        featured = engineer.engineer_features(df)

        row = featured.iloc[1]
        assert row["home_goals_scored_avg"] == 2.0
        assert row["home_win_form"] == 1.0
        assert row["home_rest_days"] == 7.0


# ------------------------------------------------------------------
# Basketball Feature Tests
# ------------------------------------------------------------------


class TestBasketballFeatures:
    """Test suite for BasketballFeatureEngineer."""

    def test_engineer_features_shape(self) -> None:
        """Feature engineering should produce more columns than input."""
        df = make_basketball_data(50)
        engineer = BasketballFeatureEngineer()
        featured = engineer.engineer_features(df)

        assert featured.shape[1] > df.shape[1]
        assert "home_ppg" in featured.columns
        assert "home_net_rtg" in featured.columns
        assert "home_b2b" in featured.columns

    def test_binary_target(self) -> None:
        """Basketball should have binary target (no draws)."""
        df = make_basketball_data(50)
        engineer = BasketballFeatureEngineer()
        featured = engineer.engineer_features(df)

        unique_results = featured["result"].dropna().unique()
        assert "draw" not in unique_results

    def test_fatigue_computation(self) -> None:
        """Fatigue index should be non-negative."""
        df = make_basketball_data(50)
        engineer = BasketballFeatureEngineer()
        featured = engineer.engineer_features(df)

        assert (featured["home_fatigue"].fillna(0) >= 0).all()
        assert (featured["away_fatigue"].fillna(0) >= 0).all()


# ------------------------------------------------------------------
# Tennis Feature Tests
# ------------------------------------------------------------------


class TestTennisFeatures:
    """Test suite for TennisFeatureEngineer."""

    def test_surface_encoding(self) -> None:
        """Surface should be encoded via one-hot features."""
        df = make_tennis_data(30)
        engineer = TennisFeatureEngineer()
        featured = engineer.engineer_features(df)

        assert "surface_hard" in featured.columns
        assert "surface_clay" in featured.columns
        assert "surface_grass" in featured.columns

    def test_round_encoding(self) -> None:
        """Round should be encoded as ordinal numeric."""
        df = make_tennis_data(30)
        engineer = TennisFeatureEngineer()
        featured = engineer.engineer_features(df)

        assert "round_num" in featured.columns
        assert (featured["round_num"] >= 0).all()

    def test_h2h_features(self) -> None:
        """H2H features should be present and bounded."""
        df = make_tennis_data(50)
        engineer = TennisFeatureEngineer()
        featured = engineer.engineer_features(df)

        assert "h2h_p1_win_rate" in featured.columns
        assert featured["h2h_p1_win_rate"].between(0, 1).all()

    def test_form_quality_features_exist(self) -> None:
        """Opponent-quality-weighted form features should be engineered."""
        df = make_tennis_data(50)
        engineer = TennisFeatureEngineer()
        featured = engineer.engineer_features(df)

        for col in ["p1_form_quality", "p2_form_quality", "form_quality_diff"]:
            assert col in featured.columns, f"Missing column: {col}"
        assert featured["form_quality_diff"].notna().all()
        assert "_p1_quality_result" not in featured.columns
        assert "_p2_quality_result" not in featured.columns

    def test_return_pressure_features_exist(self) -> None:
        """Serve/return balance features should be present and finite."""
        df = make_tennis_data(50)
        engineer = TennisFeatureEngineer()
        featured = engineer.engineer_features(df)

        expected = [
            "roll_p1_return_pressure", "roll_p2_return_pressure",
            "roll_p1_break_conv", "roll_p2_break_conv",
            "return_pressure_diff", "break_conv_diff", "serve_balance_diff",
        ]
        for col in expected:
            assert col in featured.columns, f"Missing column: {col}"
            assert np.isfinite(featured[col].fillna(0)).all(), f"Non-finite values in {col}"

    def test_target_ready_for_training(self) -> None:
        """Feature output should include a numeric target and many numeric features."""
        df = make_tennis_data(50)
        engineer = TennisFeatureEngineer()
        featured = engineer.engineer_features(df)

        numeric_cols = featured.select_dtypes(include=[np.number]).columns
        assert "target" in numeric_cols
        assert len(numeric_cols) > 10
