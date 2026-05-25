from __future__ import annotations

import pandas as pd
import numpy as np

from src.evaluation.market_backtest import (
    MarketSpec,
    _evaluate_predictions,
    _build_replay_event_rows,
    _handicap_target,
    _under_total_target,
    _parse_tennis_games,
    _parse_tennis_set_wins,
    _soccer_btts_target,
    _over_total_target,
    _result_binary_target,
)


def test_soccer_market_targets() -> None:
    df = pd.DataFrame(
        {
            "home_goals": [2, 1, 0],
            "away_goals": [1, 0, 0],
            "result": ["home_win", "draw", "away_win"],
        }
    )

    totals = _over_total_target("home_goals", "away_goals", 2.5)(df)
    under = _under_total_target("home_goals", "away_goals", 2.5)(df)
    btts = _soccer_btts_target(df)
    home_dnb = _result_binary_target({"home_win"}, exclude={"draw"})(df)
    dc_home_draw = _result_binary_target({"home_win", "draw"})(df)
    home_plus = _handicap_target("home_goals", "away_goals", 1.5)(df)
    away_minus = _handicap_target("away_goals", "home_goals", -1.5)(df)

    assert totals.tolist() == [1.0, 0.0, 0.0]
    assert under.tolist() == [0.0, 1.0, 1.0]
    assert btts.tolist() == [1.0, 0.0, 0.0]
    assert pd.isna(home_dnb.iloc[1])
    assert home_dnb.iloc[0] == 1.0
    assert home_dnb.iloc[2] == 0.0
    assert dc_home_draw.tolist() == [1.0, 1.0, 0.0]
    assert home_plus.tolist() == [1.0, 1.0, 1.0]
    assert away_minus.tolist() == [0.0, 0.0, 0.0]


def test_tennis_score_parsers() -> None:
    assert _parse_tennis_games("6-4 7-6(4)") == 23
    assert _parse_tennis_set_wins("6-4 7-6(4)") == (2, 0)
    assert _parse_tennis_set_wins("4-6 7-5 6-3") == (2, 1)


def test_evaluate_predictions_handles_single_class_window() -> None:
    y_true = np.array([1, 1, 1])
    y_proba = np.array(
        [
            [0.30, 0.70],
            [0.45, 0.55],
            [0.20, 0.80],
        ]
    )

    metrics = _evaluate_predictions(y_true, y_proba)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["log_loss"] > 0.0


def test_build_replay_event_rows_includes_identity_and_metrics() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-04-01", "2026-04-02"]),
            "home_team": ["A", "C"],
            "away_team": ["B", "D"],
        },
        index=[10, 11],
    )
    spec = MarketSpec(
        key="moneyline",
        sport="soccer",
        market_type="moneyline",
        target_builder=lambda frame: pd.Series([0, 1], index=frame.index),
    )
    y_test = pd.Series([1, 0], index=[10, 11])
    proba = np.array([[0.2, 0.8], [0.7, 0.3]])
    pred = np.array([1, 0])

    rows = _build_replay_event_rows(
        sport="soccer",
        spec=spec,
        df=df,
        test_idx=pd.Index([10, 11]),
        y_test=y_test,
        proba=proba,
        pred=pred,
        period=2,
        train_end=pd.Timestamp("2026-03-31"),
        test_end=pd.Timestamp("2026-04-30"),
    )

    assert rows["match_id"].tolist() == ["A vs B", "C vs D"]
    assert rows["correct"].tolist() == [1, 1]
    assert rows["market"].tolist() == ["moneyline", "moneyline"]
    assert "pred_prob_0" in rows.columns
    assert "pred_prob_1" in rows.columns
