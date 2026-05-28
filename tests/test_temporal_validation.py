import pandas as pd
import pytest

from src.evaluation.validation_metrics import (
    basketball_spread_validation,
    multiclass_brier_score,
    nhl_goalie_status_performance,
    probability_validation_report,
)
from src.features.feature_store import FeatureStore
from src.models.basketball_side_model import BasketballSideModel
from src.models.nhl_side_model import NHLSideModel
from src.models.soccer_score_model import SoccerScoreModel
from src.validation.temporal import audit_temporal_frame, chronological_split_indices, walk_forward_splits


def test_temporal_audit_flags_future_evidence_and_late_prediction() -> None:
    frame = pd.DataFrame(
        {
            "game_start_time": ["2026-05-01T20:00:00Z", "2026-05-02T20:00:00Z"],
            "prediction_time": ["2026-05-01T19:00:00Z", "2026-05-02T21:00:00Z"],
            "injury_report_timestamp": ["2026-05-01T18:30:00Z", "2026-05-02T21:30:00Z"],
            "odds_timestamp": ["2026-05-01T18:00:00Z", "2026-05-02T19:30:00Z"],
        }
    )

    result = audit_temporal_frame(frame)

    assert result.passed is False
    assert "prediction_time" in result.unsafe_columns
    assert "injury_report_timestamp" in result.unsafe_columns


def test_walk_forward_splits_are_chronological() -> None:
    splits = walk_forward_splits(12, initial_train_size=5, test_size=3, step_size=2)

    assert splits
    for train_idx, test_idx in splits:
        assert max(train_idx) < min(test_idx)
        assert train_idx == list(range(0, len(train_idx)))

    train, val, test = chronological_split_indices(20, train_ratio=0.6, val_ratio=0.2)
    assert train == slice(0, 12)
    assert val == slice(12, 16)
    assert test == slice(16, 20)


def test_feature_store_basketball_extras_respect_as_of_cutoff() -> None:
    rows = []
    for i, q4 in enumerate([20, 22, 24, 36, 38, 40]):
        rows.append(
            {
                "date": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=i),
                "home_team": "Boston Celtics",
                "away_team": "Miami Heat",
                "home_q1": 25,
                "home_q2": 25,
                "home_q3": 20,
                "home_q4": q4,
                "home_ot": 0,
            }
        )
    store = FeatureStore(pd.DataFrame(rows), sport="basketball", window=10)

    early = store.get_basketball_extras("Boston Celtics", "Miami Heat", before_time="2026-01-04T00:00:00Z")
    late = store.get_basketball_extras("Boston Celtics", "Miami Heat", before_time="2026-01-08T00:00:00Z")

    assert early["home_q4_avg"] < late["home_q4_avg"]
    assert early["home_q4_avg"] == pytest.approx(22.0)


def test_sport_context_timestamps_downgrade_future_lineup_injury_and_goalie_data() -> None:
    soccer = SoccerScoreModel().build_value_report(
        expected_home_goals=1.5,
        expected_away_goals=1.1,
        odds_1x2={"home": 2.0, "draw": 3.4, "away": 4.0},
        lineup_context={
            "home_lineup_confirmed": True,
            "away_lineup_confirmed": True,
            "lineup_timestamp": "2026-05-01T19:30:00Z",
            "prediction_time": "2026-05-01T18:00:00Z",
        },
    )
    assert soccer.confidence < 0.64
    assert "lineup timestamp is after prediction time" in soccer.warnings

    basketball = BasketballSideModel().build_value_report(
        snapshot=pd.Series({"possessions_projection": 100.0}),
        odds_moneyline={"home": 1.9, "away": 1.9},
        availability_context={
            "home_lineup_confirmed": True,
            "away_lineup_confirmed": True,
            "injury_report_timestamp": "2026-05-01T19:30:00Z",
            "prediction_time": "2026-05-01T18:00:00Z",
        },
    )
    assert basketball.confidence < 0.64
    assert "injury report timestamp is after prediction time" in basketball.warnings

    nhl = NHLSideModel().build_value_report(
        snapshot=pd.Series({"home_xgf_pg_10": 3.0, "away_xgf_pg_10": 2.7}),
        odds_moneyline={"home": 1.95, "away": 1.95},
        goalie_context={
            "home_goalie_confirmed": True,
            "away_goalie_confirmed": True,
            "goalie_confirmation_timestamp": "2026-05-01T19:30:00Z",
            "prediction_time": "2026-05-01T18:00:00Z",
        },
    )
    assert nhl.goalie_status == "unconfirmed"
    assert "goalie confirmation timestamp is after prediction time" in nhl.warnings


def test_validation_metrics_cover_multiclass_brier_roi_clv_and_sport_slices() -> None:
    y_true = [0, 1, 2]
    y_proba = pd.DataFrame(
        [[0.70, 0.20, 0.10], [0.20, 0.65, 0.15], [0.10, 0.25, 0.65]]
    ).to_numpy()

    report = probability_validation_report(
        y_true=y_true,
        y_proba=y_proba,
        returns=[0.1, -1.0, 0.2],
        clv=[0.02, -0.01, 0.03],
        edges=[0.04, 0.01, 0.05],
    )

    assert report.brier_score == pytest.approx(multiclass_brier_score(y_true, y_proba), abs=1e-6)
    assert report.log_loss > 0
    assert report.max_drawdown is not None
    assert report.avg_clv == pytest.approx(0.013333, abs=1e-6)

    spread = basketball_spread_validation(actual_margins=[5, -3, 8], projected_margins=[4, -1, 10], spread_lines=[-3, 2, -7])
    assert spread["spread_mae"] == pytest.approx(1.666667, abs=1e-6)
    assert "ats_accuracy" in spread

    goalie = nhl_goalie_status_performance(
        pd.DataFrame(
            {
                "goalie_status": ["confirmed", "confirmed", "unconfirmed"],
                "won": [1, 0, 0],
                "edge": [0.04, 0.01, -0.02],
                "clv": [0.02, 0.00, -0.03],
            }
        )
    )
    assert goalie["confirmed"]["count"] == 2
    assert goalie["unconfirmed"]["avg_clv"] == pytest.approx(-0.03)
