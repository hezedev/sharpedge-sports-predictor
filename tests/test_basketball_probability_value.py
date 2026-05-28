import pandas as pd
import pytest

from src.features.basketball_features import BasketballFeatureEngineer
from src.models.basketball_side_model import BasketballSideModel


def _snapshot(**overrides):
    base = {
        "home_off_rating_per_100": 118.0,
        "away_off_rating_per_100": 111.0,
        "home_def_rating_per_100": 110.0,
        "away_def_rating_per_100": 116.0,
        "opponent_adjusted_net_rating_diff": 5.5,
        "possessions_projection": 100.0,
    }
    base.update(overrides)
    return pd.Series(base)


def test_basketball_projection_outputs_probability_margin_and_total() -> None:
    model = BasketballSideModel()

    report, availability = model.projection_from_snapshot(
        _snapshot(),
        availability_context={"home_lineup_confirmed": True, "away_lineup_confirmed": True, "garbage_time_filtered_available": True},
        spread_line=-4.5,
        total_line=224.5,
    )

    assert report.home_win_probability + report.away_win_probability == pytest.approx(1.0, abs=1e-4)
    assert report.expected_margin > 0
    assert report.projected_total > 200
    assert 0 <= report.spread_cover_probability <= 1
    assert report.total_over_probability + report.total_under_probability == pytest.approx(1.0)
    assert availability.confidence_multiplier == pytest.approx(1.0)


def test_basketball_pace_projection_changes_projected_total() -> None:
    model = BasketballSideModel()

    slow, _ = model.projection_from_snapshot(_snapshot(possessions_projection=92.0))
    fast, _ = model.projection_from_snapshot(_snapshot(possessions_projection=106.0))

    assert fast.projected_total > slow.projected_total
    assert fast.possessions_projection > slow.possessions_projection


def test_basketball_availability_adjustment_lowers_confidence_and_margin() -> None:
    model = BasketballSideModel()

    clean, _ = model.projection_from_snapshot(
        _snapshot(),
        availability_context={"home_lineup_confirmed": True, "away_lineup_confirmed": True, "garbage_time_filtered_available": True},
    )
    injured, adjustment = model.projection_from_snapshot(
        _snapshot(),
        availability_context={
            "home_lineup_confirmed": False,
            "away_lineup_confirmed": True,
            "home_star_player_missing": True,
            "home_expected_minutes_lost": 38,
            "home_questionable_count": 1,
        },
    )

    assert injured.expected_margin < clean.expected_margin
    assert adjustment.confidence_multiplier < 1.0
    assert "projected or unconfirmed starters" in adjustment.warnings


def test_basketball_value_report_uses_no_vig_market_and_ev() -> None:
    model = BasketballSideModel()

    report = model.build_value_report(
        snapshot=_snapshot(),
        odds_moneyline={"home": 1.95, "away": 1.95},
        availability_context={"home_lineup_confirmed": True, "away_lineup_confirmed": True, "garbage_time_filtered_available": True},
        model_probabilities={"home": 0.56, "away": 0.44},
    )

    home_value = next(value for value in report.market_values if value.outcome == "home")
    assert home_value.no_vig_market_probability == pytest.approx(0.5, abs=0.001)
    assert home_value.edge == pytest.approx(0.06, abs=0.001)
    assert home_value.expected_value == pytest.approx(0.092, abs=0.002)
    assert home_value.recommended_action in {"bet", "monitor", "pass"}
    assert home_value.decision_reason
    assert report.projection.home_win_probability == pytest.approx(0.56, abs=0.0001)


def test_basketball_spread_and_total_value_markets_are_supported() -> None:
    model = BasketballSideModel()

    report = model.build_value_report(
        snapshot=_snapshot(),
        odds_moneyline={"home": 1.91, "away": 1.91},
        spread_line=-3.5,
        spread_odds={"home": 1.91, "away": 1.91},
        total_line=222.5,
        total_odds={"over": 1.91, "under": 1.91},
    )

    market_types = {value.market_type for value in report.market_values}
    assert {"moneyline", "spread", "total"}.issubset(market_types)
    assert report.projection.spread_cover_probability is not None
    assert report.projection.total_over_probability is not None


def test_basketball_feature_timestamp_guard_flags_postgame_availability() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-01 19:00", "2026-05-02 19:00"]),
            "injury_as_of": pd.to_datetime(["2026-05-01 18:00", "2026-05-02 20:00"]),
        }
    )

    unsafe = BasketballSideModel.validate_feature_timestamps(df)

    assert unsafe == ["injury_as_of"]


def test_basketball_feature_hooks_add_possession_four_factors_and_drop_leaks() -> None:
    rows = []
    for i in range(24):
        rows.append(
            {
                "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                "home_team": "Denver Nuggets" if i % 2 == 0 else "Boston Celtics",
                "away_team": "Miami Heat" if i % 2 == 0 else "Chicago Bulls",
                "home_score": 112 + (i % 5),
                "away_score": 105 + (i % 4),
                "result": "home_win",
                "home_q1": 28,
                "home_q2": 29,
                "home_q3": 27,
                "home_q4": 28,
                "home_ot": 0,
                "away_q1": 25,
                "away_q2": 26,
                "away_q3": 27,
                "away_q4": 27,
                "away_ot": 0,
                "home_fga": 88,
                "home_fgm": 43,
                "home_fg3m": 13,
                "home_fta": 22,
                "home_tov": 12,
                "home_orb": 10,
                "home_drb": 34,
                "away_fga": 86,
                "away_fgm": 40,
                "away_fg3m": 11,
                "away_fta": 20,
                "away_tov": 14,
                "away_orb": 9,
                "away_drb": 32,
            }
        )
    df = pd.DataFrame(rows)
    engineer = BasketballFeatureEngineer()

    features = engineer.engineer_features(df)

    for col in ["possessions_projection", "four_factors_diff", "injury_adjusted_net_rating_diff", "garbage_time_warning"]:
        assert col in features.columns
    for leaked in ["home_half_ratio", "away_half_ratio", "went_to_ot"]:
        assert leaked not in features.columns
