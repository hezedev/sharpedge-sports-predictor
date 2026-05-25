from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from src.utils import results_tracker


def test_settle_prediction_stores_clv_and_mistake_classification(tmp_path, monkeypatch) -> None:
    tracker_dir = tmp_path / "tracker"
    monkeypatch.setattr(results_tracker, "_TRACKER_DIR", tracker_dir)
    monkeypatch.setattr(results_tracker, "_PRED_FILE", tracker_dir / "predictions.parquet")
    monkeypatch.setattr(results_tracker, "_SETTLED_FILE", tracker_dir / "settled.parquet")
    monkeypatch.setattr(results_tracker, "_SUMMARY_FILE", tracker_dir / "summary.parquet")
    monkeypatch.setattr(results_tracker, "_PARLAY_FILE", tracker_dir / "parlays.parquet")

    pred_id = results_tracker.record_prediction(
        sport="soccer",
        match_id="Home vs Away",
        team_or_player="Home",
        commence_time=datetime(2026, 5, 5, 18, 0, tzinfo=timezone.utc),
        market="moneyline",
        ml_prob=0.68,
        fair_prob=0.58,
        bet_odds=1.72,
        bookmaker="Book",
        edge=0.17,
        kelly_stake_pct=0.02,
        stake_units=0.02,
        lower_bound_passed=False,
    )

    settled = results_tracker.settle_prediction(
        pred_id,
        "away_win",
        closing_odds=1.88,
        won=False,
    )

    assert settled is not None
    assert round(float(settled["clv"]), 6) == round((1.72 / 1.88) - 1.0, 6)
    assert settled["mistake_classification"] == "odds/value error"


def test_record_prediction_keeps_probability_diagnostics_for_post_result_audit(tmp_path, monkeypatch) -> None:
    tracker_dir = tmp_path / "tracker"
    monkeypatch.setattr(results_tracker, "_TRACKER_DIR", tracker_dir)
    monkeypatch.setattr(results_tracker, "_PRED_FILE", tracker_dir / "predictions.parquet")
    monkeypatch.setattr(results_tracker, "_SETTLED_FILE", tracker_dir / "settled.parquet")
    monkeypatch.setattr(results_tracker, "_SUMMARY_FILE", tracker_dir / "summary.parquet")
    monkeypatch.setattr(results_tracker, "_PARLAY_FILE", tracker_dir / "parlays.parquet")

    pred_id = results_tracker.record_prediction(
        sport="mlb",
        match_id="Home vs Away",
        team_or_player="Home",
        commence_time=datetime(2026, 5, 5, 18, 0, tzinfo=timezone.utc),
        market="moneyline",
        ml_prob=0.57,
        fair_prob=0.54,
        bet_odds=1.95,
        probability_debug={
            "structural_available": True,
            "structural_weight": 0.35,
            "context_probability_adjustment": {
                "applied": True,
                "reasons": ["pitcher_context_uncertain", "bullpen_load"],
            },
        },
    )

    predictions = pd.read_parquet(tracker_dir / "predictions.parquet")
    row = predictions[predictions["pred_id"] == pred_id].iloc[0]
    assert bool(row["probability_context_applied"]) is True
    assert json.loads(row["probability_context_reasons"]) == ["pitcher_context_uncertain", "bullpen_load"]
    assert bool(row["structural_available"]) is True
    assert float(row["structural_weight"]) == 0.35

    settled = results_tracker.settle_prediction(pred_id, "away_win", won=False)
    assert settled is not None
    assert bool(settled["probability_context_applied"]) is True
    assert json.loads(settled["probability_debug_json"])["structural_weight"] == 0.35


def test_mistake_report_includes_daily_and_weekly_counts(tmp_path, monkeypatch) -> None:
    tracker_dir = tmp_path / "tracker"
    tracker_dir.mkdir(parents=True)
    monkeypatch.setattr(results_tracker, "_TRACKER_DIR", tracker_dir)
    monkeypatch.setattr(results_tracker, "_PRED_FILE", tracker_dir / "predictions.parquet")
    monkeypatch.setattr(results_tracker, "_SETTLED_FILE", tracker_dir / "settled.parquet")
    monkeypatch.setattr(results_tracker, "_SUMMARY_FILE", tracker_dir / "summary.parquet")
    monkeypatch.setattr(results_tracker, "_PARLAY_FILE", tracker_dir / "parlays.parquet")

    settled_df = pd.DataFrame(
        [
            {
                "pred_id": "p1",
                "settled_at": "2026-05-05T20:00:00Z",
                "sport": "soccer",
                "match_id": "A vs B",
                "team_or_player": "A",
                "commence_time": "2026-05-05T18:00:00Z",
                "recorded_at": "2026-05-05T10:00:00Z",
                "market": "moneyline",
                "market_status": "preferred",
                "tier": "Preferred",
                "bet_odds": 1.68,
                "bookmaker": "Book",
                "edge": 0.09,
                "ml_prob": 0.66,
                "fair_prob": 0.57,
                "stake_units": 1.0,
                "kelly_stake_pct": 0.02,
                "is_parlay_leg": False,
                "actual_result": "away_win",
                "won": False,
                "profit_units": -1.0,
                "closing_odds": 1.85,
                "clv": -0.0919,
                "status": "lost",
            },
            {
                "pred_id": "p2",
                "settled_at": "2026-05-03T20:00:00Z",
                "sport": "soccer",
                "match_id": "C vs D",
                "team_or_player": "C",
                "commence_time": "2026-05-03T18:00:00Z",
                "recorded_at": "2026-05-03T10:00:00Z",
                "market": "moneyline",
                "market_status": "preferred",
                "tier": "Preferred",
                "bet_odds": 2.55,
                "bookmaker": "Book",
                "edge": 0.05,
                "ml_prob": 0.44,
                "fair_prob": 0.39,
                "stake_units": 1.0,
                "kelly_stake_pct": 0.02,
                "is_parlay_leg": False,
                "actual_result": "away_win",
                "won": False,
                "profit_units": -1.0,
                "closing_odds": 2.45,
                "clv": 0.0408,
                "status": "lost",
            },
        ]
    )
    settled_df.to_parquet(tracker_dir / "settled.parquet", index=False)

    parlay_df = pd.DataFrame(
        [
            {
                "parlay_id": "par-1",
                "recorded_at": "2026-05-05T10:00:00Z",
                "tier": "value",
                "bracket": "10x",
                "n_legs": 6,
                "combined_odds": 8.2,
                "combined_prob": 0.16,
                "ev": 1.31,
                "edge": 0.31,
                "kelly_stake_pct": 1.0,
                "stake_units": 0.01,
                "legs_json": "[]",
                "risk_tier": "high-risk",
                "build_verdict": "BUILD",
                "weakest_leg_json": "{}",
                "version_snapshot": "",
                "status": "lost",
                "settled_at": "2026-05-05T22:00:00Z",
                "won": False,
                "profit_units": -0.01,
                "mistake_classification": "parlay-construction error",
            }
        ]
    )
    parlay_df.to_parquet(tracker_dir / "parlays.parquet", index=False)

    report = results_tracker.mistake_report("2026-05-05")

    assert report["daily"]["losses"] == 2
    assert report["daily"]["categories"]["odds/value error"] == 1
    assert report["daily"]["categories"]["parlay-construction error"] == 1
    assert report["weekly"]["categories"]["underdog-resistance error"] == 1


def test_classify_mistake_identifies_rotation_and_overconfidence_signals() -> None:
    mistake = results_tracker._classify_mistake(
        {
            "status": "lost",
            "market": "moneyline",
            "recommended_market": "moneyline",
            "market_suitable": True,
            "fixture_verified": True,
            "context_factor_names": "rotation_risk rotation_uncertainty",
            "decision_reason": "Cup rotation risk was present pre-match",
            "ml_prob": 0.72,
            "edge": 0.14,
            "bet_odds": 1.68,
            "lower_bound_passed": False,
            "freshness_check": "fresh",
            "odds_freshness": "fresh",
            "lineup_freshness": "fresh",
            "injury_news_freshness": "fresh",
            "standings_freshness": "fresh",
        }
    )

    assert mistake == "rotation error"


def test_classify_mistake_identifies_wrong_conversion_error() -> None:
    mistake = results_tracker._classify_mistake(
        {
            "status": "lost",
            "market": "draw_no_bet",
            "recommended_market": "spreads",
            "market_suitable": False,
            "fixture_verified": True,
            "ml_prob": 0.42,
            "edge": 0.03,
            "bet_odds": 2.05,
            "lower_bound_passed": True,
            "freshness_check": "fresh",
            "odds_freshness": "fresh",
            "lineup_freshness": "fresh",
            "injury_news_freshness": "fresh",
            "standings_freshness": "fresh",
        }
    )

    assert mistake == "wrong-conversion error"


def test_classify_mistake_identifies_normal_variance_when_pick_beats_closing_line() -> None:
    mistake = results_tracker._classify_mistake(
        {
            "status": "lost",
            "market": "moneyline",
            "recommended_market": "moneyline",
            "market_suitable": True,
            "fixture_verified": True,
            "context_factor_names": "",
            "decision_reason": "",
            "ml_prob": 0.57,
            "edge": 0.05,
            "bet_odds": 2.2,
            "clv": 0.041,
            "lower_bound_passed": True,
            "freshness_check": "fresh",
            "odds_freshness": "fresh",
            "lineup_freshness": "fresh",
            "injury_news_freshness": "fresh",
            "standings_freshness": "fresh",
        }
    )

    assert mistake == "normal variance"
