from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import daily_scan
from src.features.feature_store import TeamResolver, build_entity_alias_map, resolve_canonical_name
from src.markets.decision_layer import (
    DECISION_AVOID,
    DECISION_BET,
    DECISION_BET_SUBSTITUTE,
    DECISION_HOLD,
    DECISION_NO_BET,
    DECISION_WAIT_FOR_LINEUPS,
)


class _FakeModel:
    feature_names_in_ = np.array([
        "p1_surface_win",
        "p2_surface_win",
        "surface_hard",
        "surface_clay",
        "surface_grass",
    ])

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return np.array([[0.4, 0.6]])


class _FakeTrainer:
    def __init__(self, sport: str) -> None:
        self.sport = sport
        self.trained_models = {"rf": _FakeModel()}
        self.ensemble_model = None

    def load_models(self, tag: str) -> None:
        return None


class _FakeSoccerModel:
    feature_names_in_ = np.array([
        "home_dc_win_prob",
        "dc_draw_prob",
        "away_dc_win_prob",
        "home_dc_xg",
        "away_dc_xg",
        "elo_diff",
        "dc_xg_diff",
        "xg_diff",
        "form_diff",
        "away_form_diff",
    ])

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return np.array([[0.2, 0.2, 0.6]])


class _FakeSoccerTrainer:
    def __init__(self, sport: str) -> None:
        self.sport = sport
        self.trained_models = {"rf": _FakeSoccerModel()}
        self.ensemble_model = None

    def load_models(self, tag: str) -> None:
        return None


class _FakeMlbModel:
    feature_names_in_ = np.array([
        "elo_win_prob",
        "sp_era_diff",
        "sp_whip_diff",
        "sp_k9_diff",
        "home_win_pct_10",
        "away_win_pct_10",
        "home_run_diff_10",
        "away_run_diff_10",
        "density_diff",
        "home_rest_days",
        "away_rest_days",
    ])

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return np.array([[0.4, 0.6]])


class _FakeMlbTrainer:
    def __init__(self, sport: str) -> None:
        self.sport = sport
        self.trained_models = {"rf": _FakeMlbModel()}
        self.ensemble_model = None

    def load_models(self, tag: str) -> None:
        return None


class _FakeBasketballModel:
    feature_names_in_ = np.array([
        "elo_win_prob",
        "form_diff",
        "rest_diff",
        "away_travel_bucket",
        "away_cross_country",
        "away_crossed_2tz",
        "home_pace_vs_avg",
        "away_pace_vs_avg",
    ])

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return np.array([[0.38, 0.62]])


class _FakeBasketballTrainer:
    def __init__(self, sport: str) -> None:
        self.sport = sport
        self.trained_models = {"rf": _FakeBasketballModel()}
        self.ensemble_model = None

    def load_models(self, tag: str) -> None:
        return None


def test_history_adjusted_alpha_discounts_sparse_history() -> None:
    sparse_alpha = daily_scan._history_adjusted_alpha(
        "soccer",
        0.72,
        home_rows=6,
        away_rows=7,
        min_rows=6,
    )
    deep_alpha = daily_scan._history_adjusted_alpha(
        "soccer",
        0.72,
        home_rows=30,
        away_rows=28,
        min_rows=6,
    )

    assert sparse_alpha < 0.72
    assert sparse_alpha >= daily_scan._CLV_ALPHA_MIN
    assert deep_alpha == 0.72


def test_market_health_adjusted_alpha_penalizes_weak_lane_and_disagreement(monkeypatch) -> None:
    monkeypatch.setattr(
        daily_scan,
        "_MARKET_HEALTH_BY_MARKET",
        {
            ("soccer", "moneyline"): {
                "bets": 22,
                "clv_coverage_pct": 100.0,
                "action": "pause",
                "clv_signal": "weak",
                "roi_pct": -12.0,
                "avg_clv_pct": -18.0,
            }
        },
    )

    adjusted = daily_scan._market_health_adjusted_alpha(
        "soccer",
        "moneyline",
        0.72,
        disagreement_pp=14.0,
    )

    assert adjusted < 0.72
    assert adjusted >= daily_scan._CLV_ALPHA_MIN


class _FakeNhlModel:
    feature_names_in_ = np.array([
        "elo_win_prob",
        "home_xg_diff_10",
        "away_xg_diff_10",
        "home_xgf_pg_10",
        "away_xgf_pg_10",
        "home_xga_pg_10",
        "away_xga_pg_10",
        "home_pp_pct_10",
        "away_pp_pct_10",
        "home_pk_pct_10",
        "away_pk_pct_10",
        "home_rest_days",
        "away_rest_days",
        "away_travel_bucket",
        "away_travel_tz_shift",
        "home_shots",
        "away_shots",
    ])

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return np.array([[0.39, 0.61]])


class _FakeNhlTrainer:
    def __init__(self, sport: str) -> None:
        self.sport = sport
        self.trained_models = {"rf": _FakeNhlModel()}
        self.ensemble_model = None

    def load_models(self, tag: str) -> None:
        return None


def test_current_api_min_gap_seconds_scales_with_quota_health() -> None:
    assert daily_scan._current_api_min_gap_seconds(remaining=450, daily_allowance=20, used_today=1) == 2
    assert daily_scan._current_api_min_gap_seconds(remaining=300, daily_allowance=20, used_today=1) == 5
    assert daily_scan._current_api_min_gap_seconds(remaining=180, daily_allowance=20, used_today=1) == 10
    assert daily_scan._current_api_min_gap_seconds(remaining=120, daily_allowance=20, used_today=16) == 20
    assert daily_scan._current_api_min_gap_seconds(remaining=80, daily_allowance=2, used_today=2) == daily_scan._API_MIN_GAP_SECONDS


def test_apply_decision_labels_assigns_all_phase1_verdicts() -> None:
    published, review, suppressed = daily_scan._apply_decision_labels(
        [{"team": "Team A", "publish_ready": True}],
        [
            {"team": "Team B", "review_reason": "availability or starter uncertainty still needs human review"},
            {"team": "Team C", "review_reason": "best price looked stale relative to the wider market snapshot"},
        ],
        [
            {"team": "Team D", "suppression_reason": "same-game side correlation guardrail kept only the strongest thesis"},
            {"team": "Team E", "suppression_reason": "stake reduced to zero by production staking rules"},
        ],
    )

    assert published[0]["decision_status"] == DECISION_BET
    assert review[0]["decision_status"] == DECISION_WAIT_FOR_LINEUPS
    assert review[1]["decision_status"] == DECISION_HOLD
    assert suppressed[0]["decision_status"] == DECISION_AVOID
    assert suppressed[1]["decision_status"] == DECISION_NO_BET


def test_load_recent_empty_disk_cache_reuses_fresh_empty_slate(monkeypatch) -> None:
    monkeypatch.setattr(daily_scan, "_disk_cache_age_hours", lambda _sport_key: 2.0)
    monkeypatch.setattr(daily_scan, "_load_disk_cache", lambda _sport_key: [])

    assert daily_scan._load_recent_empty_disk_cache("soccer_test") == []


def test_load_recent_empty_disk_cache_ignores_nonempty_or_old_cache(monkeypatch) -> None:
    monkeypatch.setattr(daily_scan, "_disk_cache_age_hours", lambda _sport_key: 8.0)
    monkeypatch.setattr(daily_scan, "_load_disk_cache", lambda _sport_key: [])
    assert daily_scan._load_recent_empty_disk_cache("soccer_test") is None

    monkeypatch.setattr(daily_scan, "_disk_cache_age_hours", lambda _sport_key: 2.0)
    monkeypatch.setattr(daily_scan, "_load_disk_cache", lambda _sport_key: [{"id": "g1"}])
    assert daily_scan._load_recent_empty_disk_cache("soccer_test") is None


def test_fetch_odds_force_fresh_bypasses_cache_layers(monkeypatch) -> None:
    monkeypatch.setattr(daily_scan, "ODDS_KEY", "test-key")
    monkeypatch.setattr(daily_scan, "_FORCE_FRESH_ODDS", True)
    monkeypatch.setattr(daily_scan, "_OFFLINE_ODDS_ONLY", False)
    monkeypatch.setattr(daily_scan, "_odds_cache", {"soccer_test": (daily_scan.datetime.now(), [{"id": "cached"}])})
    monkeypatch.setattr(daily_scan, "_disk_cache_age_hours", lambda _sport_key: 1.0)
    monkeypatch.setattr(
        daily_scan,
        "_load_disk_cache_bundle",
        lambda _sport_key: (_ for _ in ()).throw(AssertionError("disk cache should be bypassed in force-fresh mode")),
    )
    monkeypatch.setattr(daily_scan, "_load_recent_empty_disk_cache", lambda _sport_key: [])
    monkeypatch.setattr(daily_scan, "_check_budget", lambda priority="high": True)
    monkeypatch.setattr(daily_scan, "_fetch_odds_raw", lambda sport_key: [{"id": "fresh", "commence_time": "2099-01-01T12:00:00Z", "bookmakers": []}])
    monkeypatch.setattr(daily_scan, "_future_windowed_games", lambda games: games)

    games = daily_scan.fetch_odds("soccer_test")

    assert games == [{"id": "fresh", "commence_time": "2099-01-01T12:00:00Z", "bookmakers": []}]


def test_fetch_odds_force_fresh_reuses_current_run_live_payload(monkeypatch) -> None:
    now = daily_scan.datetime.now()
    live_payload = [
        {
            "id": "fresh",
            "commence_time": "2099-01-01T12:00:00Z",
            "bookmakers": [{"markets": [{"key": "spreads", "outcomes": []}]}],
        }
    ]

    monkeypatch.setattr(daily_scan, "ODDS_KEY", "test-key")
    monkeypatch.setattr(daily_scan, "_FORCE_FRESH_ODDS", True)
    monkeypatch.setattr(daily_scan, "_OFFLINE_ODDS_ONLY", False)
    monkeypatch.setattr(daily_scan, "_odds_cache", {"baseball_mlb": (now, live_payload)})
    monkeypatch.setattr(daily_scan, "_force_fresh_live_fetch_sports", {"baseball_mlb"})
    monkeypatch.setattr(
        daily_scan,
        "_fetch_odds_raw",
        lambda _sport_key: (_ for _ in ()).throw(AssertionError("current-run live payload should be reused")),
    )
    monkeypatch.setattr(daily_scan, "_future_windowed_games", lambda games: games)

    games = daily_scan.fetch_odds("baseball_mlb", markets="spreads")

    assert len(games) == 1
    assert games[0]["id"] == "fresh"
    assert games[0]["_odds_source_status"] == "fresh_run_cache"
    assert games[0]["_odds_source_reason"] == "force_fresh_reused_current_run_fetch"


def test_fetch_odds_force_fresh_reports_why_fallback_was_used(monkeypatch) -> None:
    sample_game = {
        "id": "match-1",
        "commence_time": "2099-01-01T12:00:00Z",
        "bookmakers": [
            {
                "title": "Book A",
                "last_update": "2099-01-01T08:00:00Z",
                "markets": [{"key": "h2h", "last_update": "2099-01-01T08:00:00Z", "outcomes": []}],
            }
        ],
    }
    loaded_at = datetime(2099, 1, 1, 9, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(daily_scan, "ODDS_KEY", "test-key")
    monkeypatch.setattr(daily_scan, "_FORCE_FRESH_ODDS", True)
    monkeypatch.setattr(daily_scan, "_OFFLINE_ODDS_ONLY", False)
    monkeypatch.setattr(daily_scan, "_odds_cache", {})
    monkeypatch.setattr(daily_scan, "_check_budget", lambda priority="high": False)
    monkeypatch.setattr(
        daily_scan,
        "_load_disk_cache_bundle",
        lambda _sport_key: ([sample_game], {"cache_loaded_at": loaded_at, "cache_age_hours": 15.5, "cache_kind": "stale_fallback"}),
    )
    monkeypatch.setattr(daily_scan, "_future_windowed_games", lambda games: games)

    games = daily_scan.fetch_odds("soccer_test")

    assert games[0]["_odds_source_status"] == "stale_fallback"
    assert games[0]["_odds_fallback_used"] is True
    assert games[0]["_odds_force_fresh_requested"] is True
    assert "force_fresh_requested" in games[0]["_odds_source_reason"]
    assert games[0]["_odds_source_detail"] == "fallback_odds_used_because_force_fresh_fetch_could_not_complete"


def test_fetch_odds_force_fresh_falls_back_to_saved_odds_on_live_fetch_error(monkeypatch) -> None:
    sample_game = {
        "id": "match-1",
        "commence_time": "2099-01-01T12:00:00Z",
        "bookmakers": [{"markets": [{"key": "h2h", "outcomes": []}]}],
    }
    loaded_at = datetime(2099, 1, 1, 9, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(daily_scan, "ODDS_KEY", "test-key")
    monkeypatch.setattr(daily_scan, "_FORCE_FRESH_ODDS", True)
    monkeypatch.setattr(daily_scan, "_OFFLINE_ODDS_ONLY", False)
    monkeypatch.setattr(daily_scan, "_odds_cache", {})
    monkeypatch.setattr(daily_scan, "_force_fresh_live_fetch_sports", set())
    monkeypatch.setattr(daily_scan, "_check_budget", lambda priority="high": True)
    monkeypatch.setattr(daily_scan, "_fetch_odds_raw", lambda _sport_key: (_ for _ in ()).throw(RuntimeError("dns failed")))
    monkeypatch.setattr(
        daily_scan,
        "_load_disk_cache_bundle",
        lambda _sport_key: ([sample_game], {"cache_loaded_at": loaded_at, "cache_age_hours": 2.5, "cache_kind": "stable"}),
    )
    monkeypatch.setattr(daily_scan, "_disk_cache_age_hours", lambda _sport_key: 2.5)
    monkeypatch.setattr(daily_scan, "_future_windowed_games", lambda games: games)

    games = daily_scan.fetch_odds("baseball_mlb")

    assert games[0]["id"] == "match-1"
    assert games[0]["_odds_source_status"] == "stale_fallback"
    assert games[0]["_odds_fallback_used"] is True
    assert games[0]["_odds_force_fresh_requested"] is True
    assert games[0]["_odds_source_reason"] == "force_fresh_requested_but_live_fetch_failed"


def test_write_report_merges_single_sport_summary_without_erasing_other_sports(tmp_path, monkeypatch) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    summary_path = reports_dir / f"summary_{daily_scan.TODAY}.json"
    summary_path.write_text(json.dumps({
        "date": daily_scan.TODAY,
        "single_bets": {
            "total": 1,
            "review_total": 1,
            "suppressed_total": 0,
            "bankroll_blocked_total": 0,
            "bets": [{"sport": "soccer", "team": "Arsenal", "edge": 0.04}],
            "review_bets": [{"sport": "soccer", "team": "Chelsea", "review_reason": "lineups"}],
            "suppressed_bets": [],
            "bankroll_blocked_bets": [],
        },
        "soccer_games": [{"sport": "soccer", "home": "A", "away": "B"}],
        "other_games": [{"sport": "nhl", "home": "Rangers", "away": "Bruins"}],
        "sport_pipeline_diagnostics": {
            "by_sport": {"soccer": {"scanned_games": 1}, "nhl": {"scanned_games": 1}},
            "totals": {"scanned_games": 2},
        },
    }))
    monkeypatch.setattr(daily_scan, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(daily_scan, "_soccer_full_games", [])
    monkeypatch.setattr(daily_scan, "_other_sport_games", [{"sport": "mlb", "home": "Yankees", "away": "Red Sox"}])
    monkeypatch.setattr(daily_scan, "_scan_runtime_notes", [])

    daily_scan.write_report(
        [{"sport": "mlb", "team": "Yankees", "home": "Yankees", "away": "Red Sox", "edge": 0.05, "odds": 1.9, "ml_prob": 0.56, "fair_prob": 0.51, "kelly_stake_pct": 1.0}],
        review_bets=[],
        suppressed_bets=[],
        bankroll_blocked_bets=[],
        bankroll=1000,
        scan_sport="mlb",
    )

    merged = json.loads(summary_path.read_text())
    assert {bet["sport"] for bet in merged["single_bets"]["bets"]} == {"soccer", "mlb"}
    assert merged["soccer_games"] == [{"sport": "soccer", "home": "A", "away": "B"}]
    assert {game["sport"] for game in merged["other_games"]} == {"nhl", "mlb"}
    assert "soccer" in merged["sport_pipeline_diagnostics"]["by_sport"]
    assert "mlb" in merged["sport_pipeline_diagnostics"]["by_sport"]


def test_with_candidate_odds_diagnostics_inherits_source_timestamp_for_synthetic_market() -> None:
    game = {
        "_odds_source_status": "live_api",
        "_odds_fetched_at": "2026-05-06T12:00:00+00:00",
        "_odds_bookmaker_last_update": "2026-05-06T09:00:00+00:00",
        "_odds_snapshot_age_hours": 3.0,
        "_odds_force_fresh_requested": True,
        "_odds_cache_used": False,
        "_odds_fallback_used": False,
        "bookmakers": [
            {
                "title": "Book A",
                "last_update": "2026-05-06T09:00:00Z",
                "markets": [{"key": "h2h", "last_update": "2026-05-06T09:00:00Z", "outcomes": []}],
            }
        ],
    }

    enriched = daily_scan._with_candidate_odds_diagnostics(
        {"team": "Alpha FC or Draw"},
        game=game,
        market_key="h2h",
        selection_name="Alpha FC or Draw",
        bookmaker_name="synthetic_1x2",
    )

    assert enriched["bookmaker_last_update"] == "2026-05-06T09:00:00+00:00"
    assert enriched["selected_bookmaker"] == "synthetic_1x2"
    assert enriched["odds_source_status"] == "live_api"


def test_stale_price_diagnostics_exposes_force_fresh_and_committee_fields() -> None:
    rows = daily_scan._stale_price_diagnostics(
        [
            {
                "home": "Alpha FC",
                "away": "Beta FC",
                "market": "moneyline",
                "team": "Alpha FC",
                "odds": 1.95,
                "bookmaker": "Book A",
                "odds_source_status": "stale_fallback",
                "odds_source_detail": "fallback_odds_used_because_force_fresh_fetch_could_not_complete",
                "odds_fetched_at": "2026-05-06T12:00:00+00:00",
                "bookmaker_last_update": "2026-05-05T20:00:00+00:00",
                "cache_loaded_at": "2026-05-06T11:50:00+00:00",
                "candidate_created_at": "2026-05-06T12:00:01+00:00",
                "computed_odds_age_hours": 16.0,
                "force_fresh_odds_active": True,
                "odds_cache_used": True,
                "odds_fallback_used": True,
                "review_reason": "best price looked stale relative to the wider market snapshot",
                "committee_final_decision": "HOLD",
                "committee_veto_flags": ["STALE_ODDS"],
            }
        ]
    )

    assert rows[0]["force_fresh_odds_active"] is True
    assert rows[0]["fallback_used"] is True
    assert rows[0]["committee_final_decision"] == "HOLD"
    assert rows[0]["arbiter_veto_flags"] == ["STALE_ODDS"]


def test_select_odds_api_key_prefers_highest_tracked_remaining(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ODDS_API_KEY", "primary-key-11111111")
    monkeypatch.setenv("ODDS_API_KEYS", "primary-key-11111111,backup-key-22222222")
    monkeypatch.setattr(daily_scan, "_ODDS_KEY_POOL_FILE", tmp_path / "odds_key_pool.json")
    daily_scan._save_odds_key_pool_usage({
        "11111111": {"fingerprint": "11111111", "remaining": 120},
        "22222222": {"fingerprint": "22222222", "remaining": 420},
    })

    selected = daily_scan._select_odds_api_key()

    assert selected == "backup-key-22222222"


def test_select_odds_api_key_picks_high_remaining_when_in_env_and_pool(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ODDS_API_KEYS", "key-a-11111111,key-b-22222222,key-c-33333333")
    monkeypatch.setenv("ODDS_API_KEY", "key-a-11111111")
    monkeypatch.setattr(daily_scan, "_ODDS_KEY_POOL_FILE", tmp_path / "odds_key_pool.json")
    daily_scan._save_odds_key_pool_usage({
        "11111111": {"fingerprint": "11111111", "remaining": 26},
        "22222222": {"fingerprint": "22222222", "remaining": 332},
        "33333333": {"fingerprint": "33333333", "remaining": 14},
    })

    selected = daily_scan._select_odds_api_key()
    payload = daily_scan._load_odds_key_pool_usage()

    assert selected == "key-b-22222222"
    assert payload["_meta"]["last_selected_fingerprint"] == "22222222"
    assert payload["_meta"]["runtime_loaded_fingerprints"] == ["11111111", "22222222", "33333333"]


def test_load_odds_key_pool_normalizes_quotes_spaces_and_invalid_entries(monkeypatch) -> None:
    monkeypatch.setenv("ODDS_API_KEY", " 'primary-key-11111111' ")
    monkeypatch.setenv(
        "ODDS_API_KEYS",
        ' "primary-key-11111111, backup-key-22222222, bad key, , third-key-33333333" ',
    )

    keys = daily_scan._load_odds_key_pool()

    assert keys == [
        "primary-key-11111111",
        "backup-key-22222222",
        "third-key-33333333",
    ]


def test_select_odds_api_key_avoids_low_quota_when_healthier_key_exists(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ODDS_API_KEY", "primary-key-11111111")
    monkeypatch.setenv(
        "ODDS_API_KEYS",
        "primary-key-11111111,low-key-33333333,healthy-key-22222222",
    )
    monkeypatch.setattr(daily_scan, "_ODDS_KEY_POOL_FILE", tmp_path / "odds_key_pool.json")
    daily_scan._save_odds_key_pool_usage({
        "11111111": {"fingerprint": "11111111", "remaining": 14},
        "33333333": {"fingerprint": "33333333", "remaining": 26},
        "22222222": {"fingerprint": "22222222", "remaining": 332},
    })

    selected = daily_scan._select_odds_api_key()
    payload = daily_scan._load_odds_key_pool_usage()

    assert selected == "healthy-key-22222222"
    assert payload["_meta"]["last_selected_fingerprint"] == "22222222"
    assert payload["_meta"]["last_selected_reason"] == "selected highest remaining runtime-available key"


def test_select_odds_api_key_excludes_exhausted_runtime_key(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ODDS_API_KEY", "empty-key-11111111")
    monkeypatch.setenv("ODDS_API_KEYS", "empty-key-11111111,backup-key-22222222")
    monkeypatch.setattr(daily_scan, "_ODDS_KEY_POOL_FILE", tmp_path / "odds_key_pool.json")
    daily_scan._save_odds_key_pool_usage({
        "11111111": {"fingerprint": "11111111", "remaining": 0},
        "22222222": {"fingerprint": "22222222", "remaining": 120},
    })

    selected = daily_scan._select_odds_api_key()
    payload = daily_scan._load_odds_key_pool_usage()

    assert selected == "backup-key-22222222"
    assert payload["_meta"]["last_selected_fingerprint"] == "22222222"
    assert any(item["reason"] == "quota_exhausted" for item in payload["_meta"]["excluded_details"])


def test_select_odds_api_key_excludes_tracked_key_missing_raw_runtime_key(tmp_path, monkeypatch, caplog) -> None:
    monkeypatch.setenv("ODDS_API_KEY", "runtime-key-33333333")
    monkeypatch.setenv("ODDS_API_KEYS", "runtime-key-33333333,runtime-key-11111111")
    monkeypatch.setattr(daily_scan, "_ODDS_KEY_POOL_FILE", tmp_path / "odds_key_pool.json")
    daily_scan._save_odds_key_pool_usage({
        "11111111": {"fingerprint": "11111111", "remaining": 26},
        "33333333": {"fingerprint": "33333333", "remaining": 14},
        "22222222": {"fingerprint": "22222222", "remaining": 332},
    })

    with caplog.at_level("INFO", logger="daily_scan"):
        selected = daily_scan._select_odds_api_key()

    payload = daily_scan._load_odds_key_pool_usage()
    assert selected == "runtime-key-11111111"
    assert payload["_meta"]["last_selected_fingerprint"] == "11111111"
    assert payload["_meta"]["excluded_fingerprints"] == ["22222222"]
    assert payload["_meta"]["runtime_parse_excluded"] == [
        {"fingerprint": "33333333", "index": 0, "reason": "duplicate", "source": "ODDS_API_KEYS"}
    ]
    assert payload["_meta"]["selected_below_low_threshold"] is True
    assert "tracked key exists but raw key not available at runtime (…22222222)" in caplog.text
    assert "excluded key …22222222 (raw_key_missing)" in caplog.text


def test_select_odds_api_key_logs_invalid_runtime_env_entries(tmp_path, monkeypatch, caplog) -> None:
    monkeypatch.setenv("ODDS_API_KEY", "primary-key-11111111")
    monkeypatch.setenv("ODDS_API_KEYS", "primary-key-11111111, bad key , short, backup-key-22222222")
    monkeypatch.setattr(daily_scan, "_ODDS_KEY_POOL_FILE", tmp_path / "odds_key_pool.json")
    daily_scan._save_odds_key_pool_usage({
        "11111111": {"fingerprint": "11111111", "remaining": 26},
        "22222222": {"fingerprint": "22222222", "remaining": 332},
    })

    with caplog.at_level("INFO", logger="daily_scan"):
        selected = daily_scan._select_odds_api_key()

    payload = daily_scan._load_odds_key_pool_usage()
    assert selected == "backup-key-22222222"
    assert len(payload["_meta"]["runtime_parse_excluded"]) == 3
    assert "excluded runtime env key …bad key (invalid_format from ODDS_API_KEYS)" in caplog.text
    assert "excluded runtime env key …short (invalid_format from ODDS_API_KEYS)" in caplog.text


def test_select_odds_api_key_reports_stale_metadata_without_silent_exclusion(tmp_path, monkeypatch, caplog) -> None:
    monkeypatch.setenv("ODDS_API_KEY", "runtime-key-11111111")
    monkeypatch.setenv("ODDS_API_KEYS", "runtime-key-11111111,runtime-key-22222222")
    monkeypatch.setattr(daily_scan, "_ODDS_KEY_POOL_FILE", tmp_path / "odds_key_pool.json")
    daily_scan._save_odds_key_pool_usage({
        "11111111": {"fingerprint": "11111111", "remaining": 120, "updated_at": "2026-05-06T11:00:00+00:00"},
        "22222222": {"fingerprint": "22222222", "remaining": 332, "updated_at": "2026-05-04T11:00:00+00:00"},
    })

    with caplog.at_level("INFO", logger="daily_scan"):
        selected = daily_scan._select_odds_api_key()

    payload = daily_scan._load_odds_key_pool_usage()
    assert selected == "runtime-key-22222222"
    assert payload["_meta"]["last_selected_fingerprint"] == "22222222"
    assert "selected highest remaining runtime-available key; metadata is stale" in payload["_meta"]["last_selected_reason"]
    assert "selected key …22222222" in caplog.text


def test_select_odds_api_key_excludes_quarantined_auth_failure_keys_across_runs(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ODDS_API_KEY", "runtime-key-11111111")
    monkeypatch.setenv("ODDS_API_KEYS", "runtime-key-11111111,runtime-key-22222222")
    monkeypatch.setattr(daily_scan, "_ODDS_KEY_POOL_FILE", tmp_path / "odds_key_pool.json")
    future = (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat()
    daily_scan._save_odds_key_pool_usage({
        "11111111": {
            "fingerprint": "11111111",
            "remaining": 420,
            "auth_quarantined_until": future,
            "auth_quarantine_reason": "http_401",
        },
        "22222222": {"fingerprint": "22222222", "remaining": 120},
    })

    selected = daily_scan._select_odds_api_key()
    payload = daily_scan._load_odds_key_pool_usage()

    assert selected == "runtime-key-22222222"
    assert payload["_meta"]["excluded_fingerprints"] == ["11111111"]
    assert any(item["reason"] == "http_401" for item in payload["_meta"]["excluded_details"])


def test_selector_reason_never_claims_highest_tracked_without_exclusion_reporting(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ODDS_API_KEY", "runtime-key-11111111")
    monkeypatch.setenv("ODDS_API_KEYS", "runtime-key-11111111")
    monkeypatch.setattr(daily_scan, "_ODDS_KEY_POOL_FILE", tmp_path / "odds_key_pool.json")
    daily_scan._save_odds_key_pool_usage({
        "11111111": {"fingerprint": "11111111", "remaining": 26},
        "22222222": {"fingerprint": "22222222", "remaining": 332},
    })

    selected = daily_scan._select_odds_api_key()
    payload = daily_scan._load_odds_key_pool_usage()

    assert selected == "runtime-key-11111111"
    assert payload["_meta"]["last_selected_reason"] == "selected highest remaining runtime-available key; higher tracked keys were excluded with explicit reasons"
    assert payload["_meta"]["excluded_fingerprints"] == ["22222222"]


def test_check_budget_rotates_to_healthier_pool_key(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ODDS_API_KEY", "primary-key-11111111")
    monkeypatch.setenv("ODDS_API_KEYS", "primary-key-11111111,backup-key-22222222")
    monkeypatch.setattr(daily_scan, "_ODDS_KEY_POOL_FILE", tmp_path / "odds_key_pool.json")
    daily_scan._save_odds_key_pool_usage({
        "11111111": {"fingerprint": "11111111", "remaining": 20},
        "22222222": {"fingerprint": "22222222", "remaining": 420},
    })
    monkeypatch.setattr(daily_scan, "ODDS_KEY", "primary-key-11111111")
    monkeypatch.setattr(
        daily_scan,
        "_load_api_usage",
        lambda: {"key_fingerprint": "11111111", "odds_remaining": 20, "odds_requests_used_today": 0},
    )
    monkeypatch.setattr(
        daily_scan,
        "get_odds_budget_status",
        lambda api_key, **_kwargs: {"remaining": 20, "daily_allowance": 5, "used_today": 0}
        if api_key.endswith("11111111")
        else {"remaining": 420, "daily_allowance": 5, "used_today": 0},
    )
    monkeypatch.setattr(daily_scan, "save_odds_api_usage", lambda **_kwargs: None)
    monkeypatch.setattr(daily_scan, "_odds_api_auth_failed", False)

    assert daily_scan._check_budget() is True
    assert daily_scan.ODDS_KEY == "backup-key-22222222"

    payload = daily_scan._load_odds_key_pool_usage()
    assert payload["_meta"]["last_selected_fingerprint"] == "22222222"
    assert payload["_meta"]["last_selected_selector"] == "low_quota_rotation"


def test_check_budget_pool_mode_ignores_daily_allowance_for_optional_requests(monkeypatch) -> None:
    monkeypatch.setenv("ODDS_API_KEY", "primary-key-11111111")
    monkeypatch.setenv("ODDS_API_KEYS", "primary-key-11111111,backup-key-22222222")
    monkeypatch.setattr(daily_scan, "ODDS_KEY", "primary-key-11111111")
    monkeypatch.setattr(
        daily_scan,
        "_load_api_usage",
        lambda: {"key_fingerprint": "11111111", "odds_remaining": 400, "odds_requests_used_today": 5},
    )
    monkeypatch.setattr(
        daily_scan,
        "get_odds_budget_status",
        lambda *_args, **_kwargs: {"remaining": 400, "daily_allowance": 5, "used_today": 5},
    )
    monkeypatch.setattr(daily_scan, "_odds_api_auth_failed", False)

    assert daily_scan._check_budget(priority="optional") is True


def test_handle_odds_api_auth_failure_rotates_to_fallback_key(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ODDS_API_KEY", "primary-key-11111111")
    monkeypatch.setenv("ODDS_API_KEYS", "primary-key-11111111,backup-key-22222222")
    monkeypatch.setattr(daily_scan, "_ODDS_KEY_POOL_FILE", tmp_path / "odds_key_pool.json")
    daily_scan._save_odds_key_pool_usage({
        "11111111": {"fingerprint": "11111111", "remaining": 20},
        "22222222": {"fingerprint": "22222222", "remaining": 420},
    })
    monkeypatch.setattr(daily_scan, "ODDS_KEY", "primary-key-11111111")
    monkeypatch.setattr(daily_scan, "_odds_api_auth_failed", False)
    monkeypatch.setattr(daily_scan, "_odds_api_failed_fingerprints", set())
    monkeypatch.setattr(daily_scan, "_scan_runtime_notes", [])

    assert daily_scan._handle_odds_api_auth_failure(sport_key="soccer_epl", status_code=401) is True
    assert daily_scan.ODDS_KEY == "backup-key-22222222"
    assert daily_scan._odds_api_auth_failed is False
    assert "11111111" in daily_scan._odds_api_failed_fingerprints
    assert any(note.get("type") == "odds_api_auth_recovered" for note in daily_scan._scan_runtime_notes)


def test_handle_odds_api_auth_failure_enters_degraded_mode_without_fallback(monkeypatch) -> None:
    monkeypatch.setenv("ODDS_API_KEY", "primary-key-11111111")
    monkeypatch.delenv("ODDS_API_KEYS", raising=False)
    monkeypatch.setattr(daily_scan, "ODDS_KEY", "primary-key-11111111")
    monkeypatch.setattr(daily_scan, "_odds_api_auth_failed", False)
    monkeypatch.setattr(daily_scan, "_odds_api_failed_fingerprints", set())
    monkeypatch.setattr(daily_scan, "_scan_runtime_notes", [])

    assert daily_scan._handle_odds_api_auth_failure(sport_key="soccer_epl", status_code=403) is False
    assert daily_scan._odds_api_auth_failed is True
    assert "11111111" in daily_scan._odds_api_failed_fingerprints
    assert any(note.get("type") == "odds_api_degraded_mode" for note in daily_scan._scan_runtime_notes)


def test_select_soccer_live_scope_defers_cold_review_leagues_when_quota_is_tight(monkeypatch) -> None:
    monkeypatch.setenv("SCAN_CONSERVE_SOCCER_SCOPE", "1")
    monkeypatch.setattr(
        daily_scan,
        "get_odds_budget_status",
        lambda *_args, **_kwargs: {"remaining": 120, "daily_allowance": 10, "used_today": 7},
    )
    monkeypatch.setattr(
        daily_scan,
        "get_capability_profile",
        lambda **kwargs: type("P", (), {"review_only": kwargs["sport_key"] != "soccer_epl"})(),
    )
    monkeypatch.setattr(
        daily_scan,
        "_recent_cache_event_state",
        lambda sport_key, max_age_hours=12.0: "nonempty" if sport_key == "soccer_finland_veikkausliiga" else "empty",
    )

    selected, deferred = daily_scan._select_soccer_live_scope(
        ["soccer_epl", "soccer_finland_veikkausliiga", "soccer_mexico_ligamx"]
    )

    assert selected == ["soccer_epl", "soccer_finland_veikkausliiga"]
    assert deferred == ["soccer_mexico_ligamx"]


def test_select_soccer_live_scope_keeps_review_leagues_when_quota_is_healthy(monkeypatch) -> None:
    monkeypatch.setenv("SCAN_CONSERVE_SOCCER_SCOPE", "1")
    monkeypatch.setattr(
        daily_scan,
        "get_odds_budget_status",
        lambda *_args, **_kwargs: {"remaining": 420, "daily_allowance": 20, "used_today": 2},
    )
    monkeypatch.setattr(
        daily_scan,
        "get_capability_profile",
        lambda **kwargs: type("P", (), {"review_only": kwargs["sport_key"] != "soccer_epl"})(),
    )
    monkeypatch.setattr(
        daily_scan,
        "_recent_cache_event_state",
        lambda sport_key, max_age_hours=12.0: "empty" if sport_key == "soccer_finland_veikkausliiga" else "stale_or_missing",
    )

    selected, deferred = daily_scan._select_soccer_live_scope(
        ["soccer_epl", "soccer_finland_veikkausliiga", "soccer_mexico_ligamx"]
    )

    assert selected == ["soccer_epl", "soccer_mexico_ligamx"]
    assert deferred == ["soccer_finland_veikkausliiga"]


def test_select_soccer_live_scope_defaults_to_all_active_leagues(monkeypatch) -> None:
    monkeypatch.delenv("SCAN_CONSERVE_SOCCER_SCOPE", raising=False)
    monkeypatch.delenv("SCAN_FULL_SOCCER_SCOPE", raising=False)

    selected, deferred = daily_scan._select_soccer_live_scope(
        ["soccer_epl", "soccer_finland_veikkausliiga", "soccer_mexico_ligamx"]
    )

    assert selected == ["soccer_epl", "soccer_finland_veikkausliiga", "soccer_mexico_ligamx"]
    assert deferred == []


def test_select_soccer_live_scope_full_override_keeps_all_leagues(monkeypatch) -> None:
    monkeypatch.setenv("SCAN_FULL_SOCCER_SCOPE", "1")

    selected, deferred = daily_scan._select_soccer_live_scope(
        ["soccer_epl", "soccer_finland_veikkausliiga", "soccer_mexico_ligamx"]
    )

    assert selected == ["soccer_epl", "soccer_finland_veikkausliiga", "soccer_mexico_ligamx"]
    assert deferred == []


def test_active_soccer_match_keys_include_unregistered_active_leagues(monkeypatch) -> None:
    monkeypatch.setattr(daily_scan, "soccer_scanable_keys", lambda: ["soccer_epl"])

    selected, skipped, discovered = daily_scan._active_soccer_match_keys({
        "soccer_epl",
        "soccer_chile_campeonato",
        "soccer_fifa_world_cup_winner",
        "baseball_mlb",
    })

    assert selected == ["soccer_chile_campeonato", "soccer_epl"]
    assert skipped == ["soccer_fifa_world_cup_winner"]
    assert discovered == ["soccer_chile_campeonato"]


def test_sport_pipeline_diagnostics_tracks_no_candidate_reasons() -> None:
    diagnostics = daily_scan._sport_pipeline_diagnostics(
        soccer_games=[
            {
                "sport": "soccer",
                "home": "A",
                "away": "B",
                "commence": "2026-05-13T18:00:00Z",
                "league_key": "soccer_usa_mls",
                "model_available": False,
            },
            {
                "sport": "soccer",
                "home": "C",
                "away": "D",
                "commence": "2026-05-13T19:00:00Z",
                "model_available": True,
                "abstain": True,
            },
            {
                "sport": "soccer",
                "home": "E",
                "away": "F",
                "commence": "2026-05-13T20:00:00Z",
                "model_available": True,
                "outcomes": [{"has_value": False}],
            },
        ],
        other_games=[],
        published_bets=[],
        review_bets=[],
        suppressed_bets=[],
    )

    soccer = diagnostics["by_sport"]["soccer"]
    assert soccer["no_candidate_games"] == 3
    assert soccer["no_candidate_reason_breakdown"]["league_not_model_covered"] == 1
    assert soccer["no_candidate_reason_breakdown"]["line_movement_or_abstain"] == 1
    assert soccer["no_candidate_reason_breakdown"]["no_edge"] == 1


def test_load_features_cached_uses_stale_cache_without_live_refresh(tmp_path) -> None:
    cache_path = tmp_path / "soccer_features.parquet"
    meta_path = Path(str(cache_path) + ".meta.json")
    df = pd.DataFrame(
        {
            "result": ["home_win", "away_win"],
            "target": [2, 0],
            "home_team": ["A", "C"],
            "away_team": ["B", "D"],
        }
    )
    df.to_parquet(cache_path)
    meta_path.write_text(json.dumps(daily_scan._feature_cache_scope_signature("soccer")))

    old_mtime = (datetime.now() - timedelta(hours=daily_scan._FEATURE_CACHE_TTL_HOURS + 4)).timestamp()
    cache_path.touch()
    import os
    os.utime(cache_path, (old_mtime, old_mtime))

    called = {"fetch": 0}

    class _FakeEngineer:
        def __init__(self) -> None:
            self.label_map = {}

        def encode_target(self, incoming: pd.DataFrame, target_col: str = "result"):
            self.label_map = {0: "away_win", 2: "home_win"}
            return incoming, self.label_map

    class _FakeFetcher:
        def fetch_all_seasons(self):
            called["fetch"] += 1
            raise AssertionError("live refresh should not run")

    features, engineer = daily_scan._load_features_cached(
        "soccer",
        str(cache_path),
        _FakeFetcher,
        _FakeEngineer,
        allow_live_refresh=False,
    )

    assert called["fetch"] == 0
    assert len(features) == 2
    assert engineer.label_map == {0: "away_win", 2: "home_win"}
    assert daily_scan._feature_cache_meta["soccer"]["source_status"] == "stale_feature_cache"


def test_load_features_cached_does_not_bootstrap_live_refresh_when_disabled(tmp_path) -> None:
    cache_path = tmp_path / "missing_soccer_features.parquet"
    called = {"fetch": 0}

    class _FakeEngineer:
        def __init__(self) -> None:
            self.label_map = {}

    class _FakeFetcher:
        def fetch_all_seasons(self):
            called["fetch"] += 1
            return pd.DataFrame()

    features, _engineer = daily_scan._load_features_cached(
        "soccer",
        str(cache_path),
        _FakeFetcher,
        _FakeEngineer,
        allow_live_refresh=False,
    )

    assert features.empty
    assert called["fetch"] == 0
    assert daily_scan._feature_cache_meta["soccer"]["source_status"] == "missing_feature_cache"


def test_load_features_cached_rebuilds_from_local_raw_snapshot_when_cache_missing(tmp_path) -> None:
    cache_path = tmp_path / "soccer_features.parquet"
    raw_dir = tmp_path / "raw" / "soccer"
    raw_dir.mkdir(parents=True)
    raw_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-08"]),
            "home_team": ["A", "C"],
            "away_team": ["B", "D"],
            "home_goals": [2, 0],
            "away_goals": [1, 1],
            "result": ["home_win", "away_win"],
            "competition": ["soccer_test", "soccer_test"],
            "season": ["2025", "2025"],
        }
    )
    raw_df.to_parquet(raw_dir / "soccer_all_seasons.parquet")

    class _FakeFetcher:
        def __init__(self) -> None:
            self._raw_dir = raw_dir

        def fetch_all_seasons(self):
            raise AssertionError("live refresh should not run")

    features, _engineer = daily_scan._load_features_cached(
        "soccer",
        str(cache_path),
        _FakeFetcher,
        daily_scan.SoccerFeatureEngineer,
        allow_live_refresh=False,
    )

    assert not features.empty
    assert cache_path.exists()
    assert daily_scan._feature_cache_meta["soccer"]["source_status"] == "rebuilt_from_local_raw"


def test_run_tennis_uses_game_for_context_flags(monkeypatch) -> None:
    fdf = pd.DataFrame(
        {
            "player1_name": ["Player A", "Player A", "Player A"],
            "player2_name": ["Player B", "Player B", "Player B"],
            "surface": ["Hard", "Hard", "Hard"],
            "p1_surface_win": [0.6, 0.65, 0.7],
            "p2_surface_win": [0.45, 0.5, 0.55],
            "surface_hard": [1.0, 1.0, 1.0],
            "surface_clay": [0.0, 0.0, 0.0],
            "surface_grass": [0.0, 0.0, 0.0],
            "target": [1, 1, 1],
        }
    )
    live_game = {
        "home_team": "Player A",
        "away_team": "Player B",
        "commence_time": "2026-04-22T12:00:00Z",
        "_window": "today",
        "description": "ATP Finals Match",
    }
    calls: list[dict] = []

    monkeypatch.setattr(daily_scan, "_load_features_cached", lambda *args, **kwargs: (fdf.copy(), object()))
    monkeypatch.setattr(daily_scan, "ModelTrainer", _FakeTrainer)
    monkeypatch.setattr(daily_scan, "_SoftVotingWrapper", lambda **kwargs: _FakeModel())
    monkeypatch.setattr(daily_scan, "get_current_model_tag", lambda *args, **kwargs: "test_tag")
    monkeypatch.setattr(daily_scan, "calibrator_path_for_tag", lambda *args, **kwargs: Path("unused.joblib"))
    monkeypatch.setattr(daily_scan.EnsembleCalibrator, "load", staticmethod(lambda *args, **kwargs: None))
    monkeypatch.setattr(daily_scan, "_prefetch_active_sports", lambda: ["tennis_atp_finals"])
    monkeypatch.setattr(daily_scan, "fetch_odds", lambda *args, **kwargs: [live_game])
    monkeypatch.setattr(daily_scan, "best_odds", lambda *args, **kwargs: (2.0, "book", False))
    monkeypatch.setattr(daily_scan, "vig_free_prob", lambda *args, **kwargs: (0.5, 0.5))
    monkeypatch.setattr(daily_scan, "_winsorize_prob", lambda prob, *_args, **_kwargs: prob)
    monkeypatch.setattr(daily_scan, "_blend", lambda prob, *_args, **_kwargs: prob)
    monkeypatch.setattr(daily_scan, "_effective_alpha", lambda *args, **kwargs: 0.65)
    monkeypatch.setattr(daily_scan, "to_parlay_leg", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        daily_scan,
        "build_value_bet",
        lambda team, ml_prob, team_odds, fair_prob, sport, **kwargs: {
            "team": team,
            "edge": ml_prob - fair_prob,
            "odds": team_odds,
        },
    )

    def _fake_detect_contextual_flags(game: dict, sport: str) -> dict:
        calls.append(game)
        return {"is_playoff": 1, "rest_advantage": 0}

    monkeypatch.setattr(daily_scan, "_detect_contextual_flags", _fake_detect_contextual_flags)
    daily_scan._other_sport_games = []

    bets, parlay_legs = daily_scan.run_tennis(dry_run=False)

    assert len(calls) == 1
    assert calls[0] is live_game
    assert len(bets) == 2
    assert all(bet["is_playoff"] == 1 for bet in bets)
    assert parlay_legs == []


def test_run_tennis_uses_cross_side_player_history(monkeypatch) -> None:
    fdf = pd.DataFrame(
        {
            "player1_name": ["Player A", "Other", "Other", "Player B"],
            "player2_name": ["Player B", "Player A", "Player B", "Player A"],
            "surface": ["Hard", "Hard", "Hard", "Hard"],
            "p1_surface_win": [0.62, 0.42, 0.43, 0.58],
            "p2_surface_win": [0.48, 0.61, 0.49, 0.52],
            "surface_hard": [1.0, 1.0, 1.0, 1.0],
            "surface_clay": [0.0, 0.0, 0.0, 0.0],
            "surface_grass": [0.0, 0.0, 0.0, 0.0],
            "target": [1, 0, 0, 1],
        }
    )
    live_game = {
        "home_team": "Player A",
        "away_team": "Player B",
        "commence_time": "2026-04-22T12:00:00Z",
        "_window": "today",
        "description": "ATP Cross-Side Test",
    }

    monkeypatch.setattr(daily_scan, "_load_features_cached", lambda *args, **kwargs: (fdf.copy(), object()))
    monkeypatch.setattr(daily_scan, "ModelTrainer", _FakeTrainer)
    monkeypatch.setattr(daily_scan, "_SoftVotingWrapper", lambda **kwargs: _FakeModel())
    monkeypatch.setattr(daily_scan, "get_current_model_tag", lambda *args, **kwargs: "test_tag")
    monkeypatch.setattr(daily_scan, "calibrator_path_for_tag", lambda *args, **kwargs: Path("unused.joblib"))
    monkeypatch.setattr(daily_scan.EnsembleCalibrator, "load", staticmethod(lambda *args, **kwargs: None))
    monkeypatch.setattr(daily_scan, "_prefetch_active_sports", lambda: ["tennis_atp_finals"])
    monkeypatch.setattr(daily_scan, "fetch_odds", lambda *args, **kwargs: [live_game])
    monkeypatch.setattr(daily_scan, "best_odds", lambda *args, **kwargs: (2.0, "book", False))
    monkeypatch.setattr(daily_scan, "vig_free_prob", lambda *args, **kwargs: (0.5, 0.5))
    monkeypatch.setattr(daily_scan, "_winsorize_prob", lambda prob, *_args, **_kwargs: prob)
    monkeypatch.setattr(daily_scan, "_blend", lambda prob, *_args, **_kwargs: prob)
    monkeypatch.setattr(daily_scan, "_effective_alpha", lambda *args, **kwargs: 0.65)
    monkeypatch.setattr(daily_scan, "to_parlay_leg", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        daily_scan,
        "build_value_bet",
        lambda team, ml_prob, team_odds, fair_prob, sport, **kwargs: {
            "team": team,
            "edge": ml_prob - fair_prob,
            "odds": team_odds,
        },
    )
    monkeypatch.setattr(daily_scan, "_detect_contextual_flags", lambda *_args, **_kwargs: {})
    daily_scan._other_sport_games = []

    bets, _ = daily_scan.run_tennis(dry_run=False)

    assert len(bets) == 2


def test_run_tennis_passes_market_anchored_probability_into_value_bet(monkeypatch) -> None:
    fdf = pd.DataFrame(
        {
            "player1_name": ["Player A", "Player A", "Player A"],
            "player2_name": ["Player B", "Player B", "Player B"],
            "surface": ["Hard", "Hard", "Hard"],
            "p1_surface_win": [0.6, 0.65, 0.7],
            "p2_surface_win": [0.45, 0.5, 0.55],
            "surface_hard": [1.0, 1.0, 1.0],
            "surface_clay": [0.0, 0.0, 0.0],
            "surface_grass": [0.0, 0.0, 0.0],
            "target": [1, 1, 1],
        }
    )
    live_game = {
        "home_team": "Player A",
        "away_team": "Player B",
        "commence_time": "2026-04-22T12:00:00Z",
        "_window": "today",
    }
    seen_raw_probs: list[float] = []

    monkeypatch.setattr(daily_scan, "_load_features_cached", lambda *args, **kwargs: (fdf.copy(), object()))
    monkeypatch.setattr(daily_scan, "ModelTrainer", _FakeTrainer)
    monkeypatch.setattr(daily_scan, "_SoftVotingWrapper", lambda **kwargs: _FakeModel())
    monkeypatch.setattr(daily_scan, "get_current_model_tag", lambda *args, **kwargs: "test_tag")
    monkeypatch.setattr(daily_scan, "calibrator_path_for_tag", lambda *args, **kwargs: Path("unused.joblib"))
    monkeypatch.setattr(daily_scan.EnsembleCalibrator, "load", staticmethod(lambda *args, **kwargs: None))
    monkeypatch.setattr(daily_scan, "_prefetch_active_sports", lambda: ["tennis_atp_finals"])
    monkeypatch.setattr(daily_scan, "fetch_odds", lambda *args, **kwargs: [live_game])
    monkeypatch.setattr(daily_scan, "best_odds", lambda *args, **kwargs: (2.0, "book", False))
    monkeypatch.setattr(daily_scan, "vig_free_prob", lambda *args, **kwargs: (0.5, 0.5))
    monkeypatch.setattr(daily_scan, "_winsorize_prob", lambda prob, *_args, **_kwargs: prob)
    monkeypatch.setattr(daily_scan, "_blend", lambda prob, *_args, **_kwargs: prob - 0.1)
    monkeypatch.setattr(daily_scan, "_effective_alpha", lambda *args, **kwargs: 0.65)
    monkeypatch.setattr(daily_scan, "to_parlay_leg", lambda *args, **kwargs: None)
    monkeypatch.setattr(daily_scan, "_detect_contextual_flags", lambda *_args, **_kwargs: {})

    def _fake_build_value_bet(team, ml_prob, team_odds, fair_prob, sport, **kwargs):
        seen_raw_probs.append(kwargs["raw_model_prob"])
        return {"team": team, "edge": ml_prob - fair_prob, "odds": team_odds}

    monkeypatch.setattr(daily_scan, "build_value_bet", _fake_build_value_bet)
    daily_scan._other_sport_games = []

    daily_scan.run_tennis(dry_run=False)

    assert len(seen_raw_probs) == 2
    assert sorted(round(prob, 3) for prob in seen_raw_probs) == [0.375, 0.625]


def test_run_soccer_passes_structural_and_market_anchored_probability_into_value_bet(monkeypatch) -> None:
    rows = 18
    fdf = pd.DataFrame(
        {
            "home_team": ["Team A"] * rows,
            "away_team": ["Team B"] * rows,
            "target": [2] * rows,
            "home_dc_win_prob": [0.46] * rows,
            "dc_draw_prob": [0.28] * rows,
            "away_dc_win_prob": [0.26] * rows,
            "home_dc_xg": [1.55] * rows,
            "away_dc_xg": [1.1] * rows,
            "elo_diff": [40.0] * rows,
            "dc_xg_diff": [0.45] * rows,
            "xg_diff": [0.35] * rows,
            "form_diff": [0.12] * rows,
            "away_form_diff": [-0.12] * rows,
        }
    )
    live_game = {
        "id": "soccer-1",
        "home_team": "Team A",
        "away_team": "Team B",
        "commence_time": "2026-05-13T18:00:00Z",
        "_window": "today",
        "bookmakers": [
            {
                "title": "Book A",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Team A", "price": 2.1},
                            {"name": "Draw", "price": 3.4},
                            {"name": "Team B", "price": 3.6},
                        ],
                    }
                ],
            }
        ],
    }
    seen_raw_probs: dict[str, float] = {}

    class _FakeEngineer:
        label_map = {0: "away_win", 1: "draw", 2: "home_win"}

    monkeypatch.setattr(daily_scan, "_MIN_SNAPSHOT_ROWS", 1)
    monkeypatch.setattr(daily_scan, "_load_features_cached", lambda *args, **kwargs: (fdf.copy(), _FakeEngineer()))
    monkeypatch.setattr(daily_scan, "ModelTrainer", _FakeSoccerTrainer)
    monkeypatch.setattr(daily_scan, "_SoftVotingWrapper", lambda **kwargs: _FakeSoccerModel())
    monkeypatch.setattr(daily_scan, "get_current_model_tag", lambda *args, **kwargs: "test_tag")
    monkeypatch.setattr(daily_scan, "calibrator_path_for_tag", lambda *args, **kwargs: Path("unused.joblib"))
    monkeypatch.setattr(daily_scan.EnsembleCalibrator, "load", staticmethod(lambda *args, **kwargs: None))
    monkeypatch.setattr(daily_scan, "soccer_scanable_keys", lambda: ["soccer_epl"])
    monkeypatch.setattr(daily_scan, "fetch_odds", lambda *args, **kwargs: [live_game])
    monkeypatch.setattr(daily_scan, "best_odds", lambda game, team: (2.1 if team == "Team A" else 3.6, "book", False))
    monkeypatch.setattr(daily_scan, "median_odds", lambda *_args, **_kwargs: 2.0)
    monkeypatch.setattr(daily_scan, "vig_free_prob", lambda *args, **kwargs: (0.4, 0.3, 0.3))
    monkeypatch.setattr(daily_scan, "_blend", lambda prob, *_args, **_kwargs: prob)
    monkeypatch.setattr(daily_scan, "_effective_alpha", lambda *args, **kwargs: 0.72)
    monkeypatch.setattr(daily_scan, "_detect_contextual_flags", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(daily_scan, "build_availability_context", lambda *args, **kwargs: {})
    monkeypatch.setattr(daily_scan, "build_environment_context", lambda *args, **kwargs: {})
    monkeypatch.setattr(daily_scan, "to_parlay_leg", lambda *args, **kwargs: None)
    monkeypatch.setattr(daily_scan, "build_refund_value_bet", lambda *args, **kwargs: None)
    monkeypatch.setattr(daily_scan, "enrich_with_capability", lambda bet, **_kwargs: dict(bet))
    monkeypatch.setattr(daily_scan, "_with_candidate_odds_diagnostics", lambda bet, **_kwargs: dict(bet))

    def _fake_build_value_bet(team, ml_prob, team_odds, fair_prob, sport, **kwargs):
        seen_raw_probs[team] = kwargs["raw_model_prob"]
        return {"team": team, "edge": ml_prob - fair_prob, "odds": team_odds}

    monkeypatch.setattr(daily_scan, "build_value_bet", _fake_build_value_bet)
    daily_scan._soccer_full_games = []
    daily_scan._odds_cache = {}

    daily_scan.run_soccer(dry_run=False)

    assert round(seen_raw_probs["Team A"], 3) == 0.512
    assert round(seen_raw_probs["Draw"], 3) == 0.251
    assert round(seen_raw_probs["Team B"], 3) == 0.237
    assert seen_raw_probs["Team A"] > seen_raw_probs["Team B"]


def test_run_mlb_passes_structural_and_market_anchored_probability_into_value_bet(monkeypatch) -> None:
    rows = 18
    fdf = pd.DataFrame(
        {
            "home_team": ["New York Yankees"] * rows,
            "away_team": ["Boston Red Sox"] * rows,
            "target": [1] * rows,
            "elo_win_prob": [0.57] * rows,
            "sp_era_diff": [-0.7] * rows,
            "sp_whip_diff": [-0.12] * rows,
            "sp_k9_diff": [1.1] * rows,
            "home_win_pct_10": [0.6] * rows,
            "away_win_pct_10": [0.4] * rows,
            "home_run_diff_10": [0.7] * rows,
            "away_run_diff_10": [-0.1] * rows,
            "density_diff": [0.0] * rows,
            "home_rest_days": [1.0] * rows,
            "away_rest_days": [1.0] * rows,
        }
    )
    live_game = {
        "id": "mlb-1",
        "home_team": "New York Yankees",
        "away_team": "Boston Red Sox",
        "commence_time": "2026-05-13T18:00:00Z",
        "_window": "today",
        "bookmakers": [
            {
                "title": "Book A",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Yankees", "price": 1.91},
                            {"name": "Red Sox", "price": 1.99},
                        ],
                    }
                ],
            }
        ],
    }
    seen_raw_probs: dict[str, float] = {}

    class _FakeEngineer:
        label_map = {0: "away_win", 1: "home_win"}

    def _fake_fetch_odds(sport_key, markets="h2h", **_kwargs):
        return [live_game] if markets == "h2h" else []

    monkeypatch.setattr(daily_scan, "_MIN_SNAPSHOT_ROWS", 1)
    monkeypatch.setattr(daily_scan, "_load_features_cached", lambda *args, **kwargs: (fdf.copy(), _FakeEngineer()))
    monkeypatch.setattr(daily_scan, "ModelTrainer", _FakeMlbTrainer)
    monkeypatch.setattr(daily_scan, "_SoftVotingWrapper", lambda **kwargs: _FakeMlbModel())
    monkeypatch.setattr(daily_scan, "get_current_model_tag", lambda *args, **kwargs: "test_tag")
    monkeypatch.setattr(daily_scan, "calibrator_path_for_tag", lambda *args, **kwargs: Path("unused.joblib"))
    monkeypatch.setattr(daily_scan.EnsembleCalibrator, "load", staticmethod(lambda *args, **kwargs: None))
    monkeypatch.setattr(daily_scan.TotalsTrainer, "load", staticmethod(lambda *_args, **_kwargs: None))
    monkeypatch.setattr(daily_scan, "fetch_odds", _fake_fetch_odds)
    monkeypatch.setattr(daily_scan, "best_odds", lambda game, team: (1.91 if team == "New York Yankees" else 1.99, "book", False))
    monkeypatch.setattr(daily_scan, "median_odds", lambda *_args, **_kwargs: 1.95)
    monkeypatch.setattr(daily_scan, "vig_free_prob", lambda *args, **kwargs: (0.51, 0.49))
    monkeypatch.setattr(daily_scan, "_blend", lambda prob, *_args, **_kwargs: prob)
    monkeypatch.setattr(daily_scan, "_effective_alpha", lambda *args, **kwargs: 0.60)
    monkeypatch.setattr(daily_scan, "_detect_contextual_flags", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(daily_scan, "build_availability_context", lambda *args, **kwargs: {})
    monkeypatch.setattr(daily_scan, "build_environment_context", lambda *args, **kwargs: {})
    monkeypatch.setattr(daily_scan, "to_parlay_leg", lambda *args, **kwargs: None)
    monkeypatch.setattr(daily_scan, "enrich_with_capability", lambda bet, **_kwargs: dict(bet))
    monkeypatch.setattr(daily_scan, "_with_candidate_odds_diagnostics", lambda bet, **_kwargs: dict(bet))

    def _fake_build_value_bet(team, ml_prob, team_odds, fair_prob, sport, **kwargs):
        seen_raw_probs[team] = kwargs["raw_model_prob"]
        return {"team": team, "edge": ml_prob - fair_prob, "odds": team_odds}

    monkeypatch.setattr(daily_scan, "build_value_bet", _fake_build_value_bet)
    daily_scan._other_sport_games = []
    daily_scan._odds_cache = {}

    daily_scan.run_mlb(dry_run=False)

    assert "New York Yankees" in seen_raw_probs
    assert "Boston Red Sox" in seen_raw_probs
    assert round(seen_raw_probs["New York Yankees"], 3) != 0.600
    assert round(seen_raw_probs["Boston Red Sox"], 3) != 0.400
    assert seen_raw_probs["New York Yankees"] > seen_raw_probs["Boston Red Sox"]


def test_mlb_context_probability_adjustment_compresses_uncertain_pitcher_context() -> None:
    home, away, debug = daily_scan._apply_mlb_context_probability_adjustment(
        0.62,
        0.38,
        {
            "home_pitcher_changed": True,
            "home_starter_confirmed": None,
            "away_starter_confirmed": None,
            "home_likely_starters_count": 0,
            "away_likely_starters_count": 0,
            "weather_risk": 1,
            "home_games_L3D": 3,
            "away_games_L3D": 1,
        },
    )

    assert home < 0.62
    assert away > 0.38
    assert debug["applied"] is True
    assert "pitcher_change" in debug["reasons"]
    assert "starter_uncertainty" in debug["reasons"]
    assert "lineup_uncertainty" in debug["reasons"]


def test_basketball_context_probability_adjustment_uses_availability_and_travel() -> None:
    home, away, debug = daily_scan._apply_basketball_context_probability_adjustment(
        0.58,
        0.42,
        {
            "home_projected_starters_count": 2,
            "away_projected_starters_count": 5,
            "home_priority_absences_count": 1,
            "away_priority_absences_count": 0,
            "home_questionable_count": 1,
            "away_questionable_count": 0,
            "rest_advantage": -1,
            "away_cross_country": True,
            "away_crossed_2tz": True,
            "away_travel_bucket": 3,
            "fixture_congestion_risk": True,
        },
    )

    assert home < 0.58
    assert away > 0.42
    assert debug["applied"] is True
    assert "lineup_uncertainty" in debug["reasons"]
    assert "star_status_uncertainty" in debug["reasons"]
    assert "away_travel_load" in debug["reasons"]


def test_nhl_context_probability_adjustment_uses_goalie_and_special_teams() -> None:
    home, away, debug = daily_scan._apply_nhl_context_probability_adjustment(
        0.61,
        0.39,
        {
            "home_goalie_confirmed": False,
            "away_goalie_confirmed": False,
            "goalie_status": "unconfirmed",
            "rest_advantage": -1,
            "away_cross_country": True,
            "away_crossed_2tz": True,
            "away_travel_bucket": 3,
            "fixture_congestion_risk": True,
            "home_pp_pct_10": 0.18,
            "away_pp_pct_10": 0.28,
            "home_pk_pct_10": 0.72,
            "away_pk_pct_10": 0.82,
        },
    )

    assert home < 0.61
    assert away > 0.39
    assert debug["applied"] is True
    assert "goalie_uncertainty" in debug["reasons"]
    assert "goalie_status_uncertain" in debug["reasons"]
    assert "special_teams_edge" in debug["reasons"]


def test_soccer_context_probability_adjustment_compresses_to_draw_when_uncertain() -> None:
    home, draw, away, debug = daily_scan._apply_soccer_context_probability_adjustment(
        0.60,
        0.23,
        0.17,
        {
            "home_likely_starters_count": 0,
            "away_likely_starters_count": 0,
            "home_priority_absences_count": 1,
            "away_priority_absences_count": 0,
            "home_questionable_count": 1,
            "away_questionable_count": 0,
            "fixture_congestion_risk": 1,
            "cup_rotation_risk": 1,
            "nothing_to_play_for": 1,
        },
    )

    assert home < 0.60
    assert draw > 0.23
    assert away > 0.17
    assert debug["applied"] is True
    assert "lineup_uncertainty" in debug["reasons"]
    assert "availability_context" in debug["reasons"]
    assert "rotation_or_congestion" in debug["reasons"]


def test_tennis_context_probability_adjustment_uses_fatigue_and_surface() -> None:
    snapshot = pd.Series({
        "load_diff": 2.0,
        "surface_win_diff": -0.18,
        "serve_balance_diff": -0.08,
    })
    p1, p2, debug = daily_scan._apply_tennis_context_probability_adjustment(
        0.60,
        0.40,
        snapshot,
        {
            "injury_concern": True,
            "fatigue_risk": True,
            "travel_required": True,
        },
    )

    assert p1 < 0.60
    assert p2 > 0.40
    assert debug["applied"] is True
    assert "injury_or_retirement_risk" in debug["reasons"]
    assert "fatigue_load" in debug["reasons"]
    assert "surface_context" in debug["reasons"]


def test_prediction_diagnostics_for_tracker_extracts_sport_probability_debug() -> None:
    diagnostics = daily_scan._prediction_diagnostics_for_tracker({
        "sport": "soccer",
        "soccer_probability_debug": {
            "structural_available": True,
            "structural_weight": 0.45,
            "context_probability_adjustment": {
                "applied": True,
                "reasons": ["lineup_uncertain", "fixture_congestion"],
            },
        },
    })

    assert diagnostics["probability_context_applied"] is True
    assert json.loads(diagnostics["probability_context_reasons"]) == ["lineup_uncertain", "fixture_congestion"]
    assert diagnostics["structural_available"] is True
    assert diagnostics["structural_weight"] == 0.45


def test_run_basketball_passes_structural_and_market_anchored_probability_into_value_bet(monkeypatch) -> None:
    rows = 18
    fdf = pd.DataFrame(
        {
            "home_team": ["Boston Celtics"] * rows,
            "away_team": ["Miami Heat"] * rows,
            "target": [1] * rows,
            "elo_win_prob": [0.56] * rows,
            "form_diff": [0.18] * rows,
            "rest_diff": [1.0] * rows,
            "away_travel_bucket": [2.0] * rows,
            "away_cross_country": [1.0] * rows,
            "away_crossed_2tz": [1.0] * rows,
            "home_pace_vs_avg": [3.0] * rows,
            "away_pace_vs_avg": [-2.0] * rows,
        }
    )
    live_game = {
        "id": "nba-1",
        "home_team": "Boston Celtics",
        "away_team": "Miami Heat",
        "commence_time": "2026-05-13T18:00:00Z",
        "_window": "today",
        "bookmakers": [
            {
                "title": "Book A",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Boston Celtics", "price": 1.80},
                            {"name": "Miami Heat", "price": 2.05},
                        ],
                    }
                ],
            }
        ],
    }
    seen_raw_probs: dict[str, float] = {}

    class _FakeEngineer:
        label_map = {0: "away_win", 1: "home_win"}

    def _fake_fetch_odds(sport_key, markets="h2h", **_kwargs):
        return [live_game] if markets == "h2h" else []

    monkeypatch.setattr(daily_scan, "_MIN_SNAPSHOT_ROWS", 1)
    monkeypatch.setattr(daily_scan, "_load_features_cached", lambda *args, **kwargs: (fdf.copy(), _FakeEngineer()))
    monkeypatch.setattr(daily_scan, "ModelTrainer", _FakeBasketballTrainer)
    monkeypatch.setattr(daily_scan, "_SoftVotingWrapper", lambda **kwargs: _FakeBasketballModel())
    monkeypatch.setattr(daily_scan, "get_current_model_tag", lambda *args, **kwargs: "test_tag")
    monkeypatch.setattr(daily_scan, "calibrator_path_for_tag", lambda *args, **kwargs: Path("unused.joblib"))
    monkeypatch.setattr(daily_scan.EnsembleCalibrator, "load", staticmethod(lambda *args, **kwargs: None))
    monkeypatch.setattr(daily_scan, "fetch_odds", _fake_fetch_odds)
    monkeypatch.setattr(daily_scan, "best_odds", lambda game, team: (1.80 if team == "Boston Celtics" else 2.05, "book", False))
    monkeypatch.setattr(daily_scan, "median_odds", lambda *_args, **_kwargs: 1.95)
    monkeypatch.setattr(daily_scan, "vig_free_prob", lambda *args, **kwargs: (0.53, 0.47))
    monkeypatch.setattr(daily_scan, "_blend", lambda prob, *_args, **_kwargs: prob)
    monkeypatch.setattr(daily_scan, "_effective_alpha", lambda *args, **kwargs: 0.50)
    monkeypatch.setattr(daily_scan, "_detect_contextual_flags", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(daily_scan, "build_availability_context", lambda *args, **kwargs: {})
    monkeypatch.setattr(daily_scan, "build_environment_context", lambda *args, **kwargs: {})
    monkeypatch.setattr(daily_scan, "to_parlay_leg", lambda *args, **kwargs: None)
    monkeypatch.setattr(daily_scan, "enrich_with_capability", lambda bet, **_kwargs: dict(bet))
    monkeypatch.setattr(daily_scan, "_with_candidate_odds_diagnostics", lambda bet, **_kwargs: dict(bet))

    def _fake_build_value_bet(team, ml_prob, team_odds, fair_prob, sport, **kwargs):
        seen_raw_probs[team] = kwargs["raw_model_prob"]
        return {"team": team, "edge": ml_prob - fair_prob, "odds": team_odds}

    monkeypatch.setattr(daily_scan, "build_value_bet", _fake_build_value_bet)
    daily_scan._other_sport_games = []
    daily_scan._odds_cache = {}

    daily_scan.run_basketball(dry_run=False)

    assert "Boston Celtics" in seen_raw_probs
    assert "Miami Heat" in seen_raw_probs
    assert round(seen_raw_probs["Boston Celtics"], 3) != 0.620
    assert round(seen_raw_probs["Miami Heat"], 3) != 0.380
    assert seen_raw_probs["Boston Celtics"] > seen_raw_probs["Miami Heat"]


def test_run_nhl_passes_structural_and_market_anchored_probability_into_value_bet(monkeypatch) -> None:
    rows = 18
    fdf = pd.DataFrame(
        {
            "home_team": ["Rangers"] * rows,
            "away_team": ["Penguins"] * rows,
            "target": [1] * rows,
            "elo_win_prob": [0.55] * rows,
            "home_xg_diff_10": [0.4] * rows,
            "away_xg_diff_10": [-0.2] * rows,
            "home_xgf_pg_10": [3.0] * rows,
            "away_xgf_pg_10": [2.5] * rows,
            "home_xga_pg_10": [2.3] * rows,
            "away_xga_pg_10": [2.8] * rows,
            "home_pp_pct_10": [24.0] * rows,
            "away_pp_pct_10": [18.0] * rows,
            "home_pk_pct_10": [82.0] * rows,
            "away_pk_pct_10": [77.0] * rows,
            "home_rest_days": [2.0] * rows,
            "away_rest_days": [1.0] * rows,
            "away_travel_bucket": [2.0] * rows,
            "away_travel_tz_shift": [1.0] * rows,
            "home_shots": [31.0] * rows,
            "away_shots": [28.0] * rows,
        }
    )
    live_game = {
        "id": "nhl-1",
        "home_team": "New York Rangers",
        "away_team": "Pittsburgh Penguins",
        "commence_time": "2026-05-13T18:00:00Z",
        "_window": "today",
        "bookmakers": [
            {
                "title": "Book A",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Rangers", "price": 1.87},
                            {"name": "Penguins", "price": 2.00},
                        ],
                    }
                ],
            }
        ],
    }
    seen_raw_probs: dict[str, float] = {}

    class _FakeEngineer:
        label_map = {0: "away_win", 1: "home_win"}

    def _fake_fetch_odds(sport_key, markets="h2h", **_kwargs):
        return [live_game] if markets == "h2h" else []

    monkeypatch.setattr(daily_scan, "_MIN_SNAPSHOT_ROWS", 1)
    monkeypatch.setattr(daily_scan, "_load_features_cached", lambda *args, **kwargs: (fdf.copy(), _FakeEngineer()))
    monkeypatch.setattr(daily_scan, "ModelTrainer", _FakeNhlTrainer)
    monkeypatch.setattr(daily_scan, "_SoftVotingWrapper", lambda **kwargs: _FakeNhlModel())
    monkeypatch.setattr(daily_scan, "get_current_model_tag", lambda *args, **kwargs: "test_tag")
    monkeypatch.setattr(daily_scan, "calibrator_path_for_tag", lambda *args, **kwargs: Path("unused.joblib"))
    monkeypatch.setattr(daily_scan.EnsembleCalibrator, "load", staticmethod(lambda *args, **kwargs: None))
    monkeypatch.setattr(daily_scan.TotalsTrainer, "load", staticmethod(lambda *_args, **_kwargs: None))
    monkeypatch.setattr(daily_scan, "fetch_odds", _fake_fetch_odds)
    monkeypatch.setattr(daily_scan, "best_odds", lambda game, team: (1.87 if team == "Rangers" else 2.00, "book", False))
    monkeypatch.setattr(daily_scan, "median_odds", lambda *_args, **_kwargs: 1.94)
    monkeypatch.setattr(daily_scan, "vig_free_prob", lambda *args, **kwargs: (0.52, 0.48))
    monkeypatch.setattr(daily_scan, "_blend", lambda prob, *_args, **_kwargs: prob)
    monkeypatch.setattr(daily_scan, "_effective_alpha", lambda *args, **kwargs: 0.45)
    monkeypatch.setattr(daily_scan, "_detect_contextual_flags", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(daily_scan, "build_availability_context", lambda *args, **kwargs: {})
    monkeypatch.setattr(daily_scan, "build_environment_context", lambda *args, **kwargs: {})
    monkeypatch.setattr(daily_scan, "to_parlay_leg", lambda *args, **kwargs: None)
    monkeypatch.setattr(daily_scan, "enrich_with_capability", lambda bet, **_kwargs: dict(bet))
    monkeypatch.setattr(daily_scan, "_with_candidate_odds_diagnostics", lambda bet, **_kwargs: dict(bet))

    def _fake_build_value_bet(team, ml_prob, team_odds, fair_prob, sport, **kwargs):
        seen_raw_probs[team] = kwargs["raw_model_prob"]
        return {"team": team, "edge": ml_prob - fair_prob, "odds": team_odds}

    monkeypatch.setattr(daily_scan, "build_value_bet", _fake_build_value_bet)
    daily_scan._other_sport_games = []
    daily_scan._odds_cache = {}

    daily_scan.run_nhl(dry_run=False)

    assert "New York Rangers" in seen_raw_probs
    assert "Pittsburgh Penguins" in seen_raw_probs
    assert round(seen_raw_probs["New York Rangers"], 3) != 0.610
    assert round(seen_raw_probs["Pittsburgh Penguins"], 3) != 0.390
    assert seen_raw_probs["New York Rangers"] > seen_raw_probs["Pittsburgh Penguins"]


def test_run_tennis_includes_wta_when_wta_models_exist(monkeypatch) -> None:
    fdf = pd.DataFrame(
        {
            "player1_name": ["Player A", "Player A", "Player A"],
            "player2_name": ["Player B", "Player B", "Player B"],
            "surface": ["Hard", "Hard", "Hard"],
            "p1_surface_win": [0.6, 0.65, 0.7],
            "p2_surface_win": [0.45, 0.5, 0.55],
            "surface_hard": [1.0, 1.0, 1.0],
            "surface_clay": [0.0, 0.0, 0.0],
            "surface_grass": [0.0, 0.0, 0.0],
            "target": [1, 1, 1],
        }
    )

    monkeypatch.setattr(daily_scan, "_load_features_cached", lambda *args, **kwargs: (fdf.copy(), object()))
    monkeypatch.setattr(daily_scan, "ModelTrainer", _FakeTrainer)
    monkeypatch.setattr(daily_scan, "_SoftVotingWrapper", lambda **kwargs: _FakeModel())
    monkeypatch.setattr(daily_scan, "get_current_model_tag", lambda *args, **kwargs: "test_tag")
    monkeypatch.setattr(daily_scan, "calibrator_path_for_tag", lambda *args, **kwargs: "unused.joblib")
    monkeypatch.setattr(daily_scan.EnsembleCalibrator, "load", staticmethod(lambda *args, **kwargs: None))
    monkeypatch.setattr(daily_scan, "_prefetch_active_sports", lambda: ["tennis_atp_finals", "tennis_wta_madrid_open"])
    calls = []

    def _fake_fetch(sport_key, *args, **kwargs):
        calls.append(sport_key)
        return []

    monkeypatch.setattr(daily_scan, "fetch_odds", _fake_fetch)
    monkeypatch.setattr(daily_scan, "_effective_alpha", lambda *args, **kwargs: 0.65)
    monkeypatch.setattr(daily_scan, "_detect_contextual_flags", lambda *_args, **_kwargs: {})

    daily_scan.run_tennis(dry_run=False)

    assert "tennis_atp_finals" in calls
    assert "tennis_wta_madrid_open" in calls


def test_run_tennis_uses_tennis_wta_alpha_bucket_for_wta(monkeypatch) -> None:
    fdf = pd.DataFrame(
        {
            "player1_name": ["Player A", "Player A", "Player A"],
            "player2_name": ["Player B", "Player B", "Player B"],
            "surface": ["Hard", "Hard", "Hard"],
            "p1_surface_win": [0.6, 0.65, 0.7],
            "p2_surface_win": [0.45, 0.5, 0.55],
            "surface_hard": [1.0, 1.0, 1.0],
            "surface_clay": [0.0, 0.0, 0.0],
            "surface_grass": [0.0, 0.0, 0.0],
            "target": [1, 1, 1],
        }
    )
    seen: list[tuple[str, bool]] = []

    monkeypatch.setattr(daily_scan, "_load_features_cached", lambda *args, **kwargs: (fdf.copy(), object()))
    monkeypatch.setattr(daily_scan, "ModelTrainer", _FakeTrainer)
    monkeypatch.setattr(daily_scan, "_SoftVotingWrapper", lambda **kwargs: _FakeModel())
    monkeypatch.setattr(daily_scan, "get_current_model_tag", lambda *args, **kwargs: "test_tag")
    monkeypatch.setattr(daily_scan, "calibrator_path_for_tag", lambda *args, **kwargs: "unused.joblib")
    monkeypatch.setattr(daily_scan.EnsembleCalibrator, "load", staticmethod(lambda *args, **kwargs: None))
    monkeypatch.setattr(daily_scan, "_prefetch_active_sports", lambda: ["tennis_atp_finals", "tennis_wta_madrid_open"])
    monkeypatch.setattr(daily_scan, "fetch_odds", lambda *args, **kwargs: [])
    monkeypatch.setattr(daily_scan, "_detect_contextual_flags", lambda *_args, **_kwargs: {})

    def _fake_effective_alpha(sport: str, calibrated: bool) -> float:
        seen.append((sport, calibrated))
        return 0.5

    monkeypatch.setattr(daily_scan, "_effective_alpha", _fake_effective_alpha)

    daily_scan.run_tennis(dry_run=False)

    assert ("tennis", False) in seen
    assert ("tennis_wta", False) in seen


def test_run_tennis_uses_tennis_wta_sport_key_for_wta_bets(monkeypatch) -> None:
    fdf = pd.DataFrame(
        {
            "player1_name": ["Player A", "Player A", "Player A"],
            "player2_name": ["Player B", "Player B", "Player B"],
            "surface": ["Hard", "Hard", "Hard"],
            "result": ["player1_win", "player1_win", "player1_win"],
            "p1_surface_win": [0.6, 0.65, 0.7],
            "p2_surface_win": [0.45, 0.5, 0.55],
            "p1_form": [0.6, 0.65, 0.7],
            "p2_form": [0.4, 0.35, 0.3],
            "roll_p1_ace_rate": [0.07, 0.08, 0.09],
            "roll_p2_ace_rate": [0.04, 0.05, 0.06],
            "roll_p1_bp_save": [0.55, 0.57, 0.59],
            "roll_p2_bp_save": [0.49, 0.50, 0.51],
            "p1_load": [2.0, 2.0, 2.0],
            "p2_load": [1.0, 1.0, 1.0],
            "surface_hard": [1.0, 1.0, 1.0],
            "surface_clay": [0.0, 0.0, 0.0],
            "surface_grass": [0.0, 0.0, 0.0],
            "target": [1, 1, 1],
        }
    )
    seen_sports: list[str] = []

    monkeypatch.setattr(daily_scan, "_load_features_cached", lambda *args, **kwargs: (fdf.copy(), object()))
    monkeypatch.setattr(daily_scan, "ModelTrainer", _FakeTrainer)
    monkeypatch.setattr(daily_scan, "_SoftVotingWrapper", lambda **kwargs: _FakeModel())
    monkeypatch.setattr(daily_scan, "get_current_model_tag", lambda *args, **kwargs: "test_tag")
    monkeypatch.setattr(daily_scan, "calibrator_path_for_tag", lambda *args, **kwargs: "unused.joblib")
    monkeypatch.setattr(daily_scan.EnsembleCalibrator, "load", staticmethod(lambda *args, **kwargs: None))
    monkeypatch.setattr(daily_scan, "_prefetch_active_sports", lambda: ["tennis_wta_madrid_open"])
    monkeypatch.setattr(daily_scan, "fetch_odds", lambda *args, **kwargs: [{
        "home_team": "Player A",
        "away_team": "Player B",
        "commence_time": "2026-04-22T12:00:00Z",
        "_window": "today",
    }])
    monkeypatch.setattr(daily_scan, "best_odds", lambda *args, **kwargs: (2.0, "book", False))
    monkeypatch.setattr(daily_scan, "vig_free_prob", lambda *args, **kwargs: (0.5, 0.5))
    monkeypatch.setattr(daily_scan, "_winsorize_prob", lambda prob, *_args, **_kwargs: prob)
    monkeypatch.setattr(daily_scan, "_blend", lambda prob, *_args, **_kwargs: prob)
    monkeypatch.setattr(daily_scan, "_effective_alpha", lambda *args, **kwargs: 0.35)
    monkeypatch.setattr(daily_scan, "to_parlay_leg", lambda *args, **kwargs: None)
    monkeypatch.setattr(daily_scan, "_detect_contextual_flags", lambda *_args, **_kwargs: {})

    def _fake_build_value_bet(team, ml_prob, team_odds, fair_prob, sport, **kwargs):
        seen_sports.append(sport)
        return {"team": team, "edge": ml_prob - fair_prob, "odds": team_odds}

    monkeypatch.setattr(daily_scan, "build_value_bet", _fake_build_value_bet)
    daily_scan._other_sport_games = []

    daily_scan.run_tennis(dry_run=False)

    assert seen_sports == ["tennis_wta", "tennis_wta"]


def test_tennis_alpha_and_uncalibrated_penalty_are_split_by_tour() -> None:
    assert daily_scan._effective_alpha("tennis", calibrated=True) == 0.55
    assert daily_scan._effective_alpha("tennis_wta", calibrated=False) == 0.35


def test_tennis_winsorization_is_tighter_for_wta_than_atp() -> None:
    market_prob = 0.1173
    raw_model_prob = 0.5891

    atp_prob = daily_scan._winsorize_prob(raw_model_prob, market_prob, "tennis")
    wta_prob = daily_scan._winsorize_prob(raw_model_prob, market_prob, "tennis_wta")

    assert abs(atp_prob - (market_prob + 0.08)) < 1e-9
    assert abs(wta_prob - (market_prob + 0.055)) < 1e-9
    assert wta_prob < atp_prob


def test_wta_rank_gap_damping_pulls_probability_toward_market() -> None:
    snapshot = pd.Series(
        {
            "rank_log_ratio": 2.0,
            "rank_pts_log_ratio": 2.4,
            "seed_advantage": 1.0,
        }
    )

    market_prob = 0.15
    blended_prob = 0.30

    damped_wta = daily_scan._apply_tennis_rank_gap_damping(
        blended_prob,
        market_prob,
        snapshot,
        "tennis_wta",
    )
    damped_atp = daily_scan._apply_tennis_rank_gap_damping(
        blended_prob,
        market_prob,
        snapshot,
        "tennis",
    )

    assert damped_wta < blended_prob
    assert damped_wta > market_prob
    assert damped_atp == blended_prob


def test_record_wta_audit_match_captures_best_side_and_reason() -> None:
    daily_scan._wta_audit_rows = []

    daily_scan._record_wta_audit_match(
        home="Marta Kostyuk",
        away="Linda Noskova",
        league="Wta Madrid Open",
        commence="2026-04-29T12:00:00Z",
        home_odds=1.58,
        away_odds=2.66,
        home_prob=0.6504,
        away_prob=0.3496,
        review_only=True,
        min_edge=0.03,
    )

    assert len(daily_scan._wta_audit_rows) == 1
    row = daily_scan._wta_audit_rows[0]
    assert row["team"] == "Marta Kostyuk"
    assert row["edge_pct"] == 2.76
    assert "below min edge" in row["audit_reason"]


def test_refresh_tennis_matchup_snapshot_recomputes_pairwise_diffs() -> None:
    snapshot = pd.Series(
        {
            "p1_surface_win": 0.745,
            "p2_surface_win": 0.4567,
            "p1_form": 0.72,
            "p2_form": 0.20,
            "p1_form_quality": 0.31,
            "p2_form_quality": -0.08,
            "roll_p1_ace_rate": 0.081,
            "roll_p2_ace_rate": 0.052,
            "roll_p1_bp_save": 0.611,
            "roll_p2_bp_save": 0.482,
            "roll_p1_return_pressure": 0.121,
            "roll_p2_return_pressure": 0.094,
            "roll_p1_break_conv": 0.463,
            "roll_p2_break_conv": 0.391,
            "p1_recent_load": 6.1,
            "p2_recent_load": 3.8,
            "surface_win_diff": -0.0982,
            "form_diff": -0.12,
            "form_quality_diff": -0.4,
            "serve_diff": -0.02,
            "bp_save_diff": -0.03,
            "return_pressure_diff": -0.01,
            "break_conv_diff": -0.02,
            "serve_balance_diff": -0.03,
            "load_diff": -0.1,
            "rank_diff": 0.0,
            "rank_log_ratio": 0.0,
            "rank_pts_log_ratio": 0.0,
            "age_diff": 0.0,
            "height_diff": 0.0,
            "seed_advantage": 0.0,
            "h2h_p1_wins": 0.0,
            "h2h_total": 0.0,
            "h2h_p1_win_rate": 0.0,
        }
    )

    refreshed = daily_scan._refresh_tennis_matchup_snapshot(
        snapshot,
        p1_profile={"rank": 1.0, "rank_pts": 10500.0, "age": 26.0, "height": 182.0, "seeded": True},
        p2_profile={"rank": 40.0, "rank_pts": 1400.0, "age": 23.0, "height": 175.0, "seeded": False},
        h2h={"p1_wins": 2.0, "total": 3.0},
    )

    assert abs(refreshed["surface_win_diff"] - 0.2883) < 1e-9
    assert abs(refreshed["form_diff"] - 0.52) < 1e-9
    assert abs(refreshed["form_quality_diff"] - 0.39) < 1e-9
    assert abs(refreshed["serve_diff"] - 0.029) < 1e-9
    assert abs(refreshed["bp_save_diff"] - 0.129) < 1e-9
    assert abs(refreshed["return_pressure_diff"] - 0.027) < 1e-9
    assert abs(refreshed["break_conv_diff"] - 0.072) < 1e-9
    assert abs(refreshed["serve_balance_diff"] - 0.09695) < 1e-9
    assert abs(refreshed["load_diff"] - 2.3) < 1e-9
    assert refreshed["rank_diff"] == 39.0
    assert refreshed["age_diff"] == 3.0
    assert refreshed["height_diff"] == 7.0
    assert refreshed["seed_advantage"] == 1.0
    assert refreshed["h2h_p1_wins"] == 2.0
    assert refreshed["h2h_total"] == 3.0
    assert abs(refreshed["h2h_p1_win_rate"] - (2.0 / 3.0)) < 1e-9


def test_entity_alias_map_handles_soccer_style_names() -> None:
    canonical = {
        "FC Barcelona",
        "Club Atlético de Madrid",
        "Feyenoord Rotterdam",
    }
    aliases = build_entity_alias_map(canonical)

    assert resolve_canonical_name("Barcelona", canonical, alias_map=aliases) == "FC Barcelona"
    assert resolve_canonical_name("Atletico Madrid", canonical, alias_map=aliases) == "Club Atlético de Madrid"
    assert resolve_canonical_name("Feyenoord", canonical, alias_map=aliases) == "Feyenoord Rotterdam"


def test_resolve_soccer_team_name_prefers_exact_matches_and_targeted_aliases() -> None:
    canonical = {
        "Barcelona SC",
        "Machida Zelvia",
        "Urawa",
        "Al-Hazm",
        "Al-Qadisiyah FC",
        "Al Taawon",
        "CR Flamengo",
        "SE Palmeiras",
    }
    aliases = build_entity_alias_map(canonical, extra_aliases=daily_scan._SOCCER_ALIAS_OVERRIDES)
    resolver = TeamResolver("soccer")

    assert daily_scan._resolve_soccer_team_name("Barcelona SC", canonical, aliases, resolver) == "Barcelona SC"
    assert daily_scan._resolve_soccer_team_name("FC Machida Zelvia", canonical, aliases, resolver) == "Machida Zelvia"
    assert daily_scan._resolve_soccer_team_name("Urawa Red Diamonds", canonical, aliases, resolver) == "Urawa"
    assert daily_scan._resolve_soccer_team_name("Al-Hazem", canonical, aliases, resolver) == "Al-Hazm"
    assert daily_scan._resolve_soccer_team_name("Al-Qadsiah", canonical, aliases, resolver) == "Al-Qadisiyah FC"
    assert daily_scan._resolve_soccer_team_name("Al-Taawoun", canonical, aliases, resolver) == "Al Taawon"
    assert daily_scan._resolve_soccer_team_name("Flamengo-RJ", canonical, aliases, resolver) == "CR Flamengo"
    assert daily_scan._resolve_soccer_team_name("Palmeiras-SP", canonical, aliases, resolver) == "SE Palmeiras"
    assert daily_scan._resolve_soccer_team_name("UCV FC", canonical, aliases, resolver) == "UCV FC"


def test_entity_alias_map_handles_tennis_casing_and_particles() -> None:
    canonical = {
        "Alex De Minaur",
        "Botic Van De Zandschulp",
        "Carlota Martinez Cirez",
    }
    aliases = build_entity_alias_map(canonical)

    assert resolve_canonical_name("Alex de Minaur", canonical, alias_map=aliases) == "Alex De Minaur"
    assert resolve_canonical_name("Botic van de Zandschulp", canonical, alias_map=aliases) == "Botic Van De Zandschulp"
    assert resolve_canonical_name("Carlota Martínez Círez", canonical, alias_map=aliases) == "Carlota Martinez Cirez"


def test_build_refund_value_bet_handles_draw_no_bet() -> None:
    bet = daily_scan.build_refund_value_bet(
        "Home DNB",
        win_prob=0.42,
        refund_prob=0.25,
        team_odds=1.9,
        fair_prob=0.6,
        min_edge=0.01,
        sport="soccer",
    )

    assert bet is not None
    assert bet["market"] == "draw_no_bet"
    assert bet["push_prob"] == 0.25
    assert bet["edge"] > 0


def test_sanitize_features_for_parquet_handles_mixed_object_columns() -> None:
    df = pd.DataFrame(
        {
            "player1_seed": [1, None, "3"],
            "player2_hand": ["R", None, "L"],
            "numeric_feature": [0.1, 0.2, 0.3],
        }
    )

    sanitized = daily_scan._sanitize_features_for_parquet(df)

    assert str(sanitized["player1_seed"].dtype).startswith(("float", "int"))
    assert str(sanitized["player2_hand"].dtype) == "string"
    assert sanitized["numeric_feature"].equals(df["numeric_feature"])


def test_build_refund_value_bet_rejects_negative_ev() -> None:
    bet = daily_scan.build_refund_value_bet(
        "Away DNB",
        win_prob=0.25,
        refund_prob=0.25,
        team_odds=1.5,
        fair_prob=0.4,
        min_edge=0.01,
        sport="soccer",
    )

    assert bet is None


def test_build_value_bet_rejects_weak_edge_instead_of_forcing_pick() -> None:
    bet = daily_scan.build_value_bet(
        "Marginal Team",
        ml_prob=0.53,
        team_odds=1.9,
        fair_prob=0.5,
        min_edge=0.03,
        sport="soccer",
        raw_model_prob=0.53,
    )

    assert bet is None


def test_build_value_bet_prefers_raw_model_probability_over_market_blend() -> None:
    bet = daily_scan.build_value_bet(
        "Home Team",
        ml_prob=0.54,
        team_odds=1.7,
        fair_prob=0.55,
        min_edge=0.01,
        sport="soccer",
        raw_model_prob=0.62,
    )

    assert bet is not None
    assert bet["model_prob_raw"] == 0.62
    assert bet["ml_prob"] >= 0.60
    assert bet["market_implied_prob"] == round(1 / 1.7, 4)
    assert bet["true_probability"]["base_prob"] == 0.62


def test_build_value_bet_can_route_moderate_outlier_to_review() -> None:
    bet = daily_scan.build_value_bet(
        "Home Team",
        ml_prob=0.54,
        team_odds=2.0,
        fair_prob=0.5,
        min_edge=0.01,
        sport="soccer",
        raw_model_prob=0.56,
    )

    assert bet is not None
    assert bet["edge_outlier_review"] is True
    assert "EV cap" in bet["edge_outlier_reason"]


def test_build_value_bet_carries_availability_summary() -> None:
    bet = daily_scan.build_value_bet(
        "Home Team",
        ml_prob=0.58,
        team_odds=1.85,
        fair_prob=0.52,
        min_edge=0.01,
        sport="soccer",
        raw_model_prob=0.58,
        availability_context={
            "home_injuries_count": 2,
            "home_suspensions_count": 1,
            "availability_source": "api_football",
        },
    )

    assert bet is not None
    assert "Availability context" in bet["availability_summary"]
    assert bet["availability_source"] == "api_football"


def test_build_value_bet_sets_minimum_acceptable_odds_contract() -> None:
    bet = daily_scan.build_value_bet(
        "Home Team",
        ml_prob=0.58,
        team_odds=1.83,
        fair_prob=0.52,
        min_edge=0.05,
        sport="soccer",
        raw_model_prob=0.58,
    )

    assert bet is not None
    assert bet["minimum_acceptable_odds"] is not None
    assert bet["odds_recheck_status"] == "pending"
    assert bet["odds_recheck_odds"] == 1.83
    assert bet["odds_recheck_delta"] > 0


def test_build_value_bet_exposes_probability_edge_and_confidence_contract() -> None:
    bet = daily_scan.build_value_bet(
        "Home Team",
        ml_prob=0.58,
        team_odds=1.9,
        fair_prob=0.53,
        min_edge=0.03,
        sport="soccer",
        raw_model_prob=0.58,
    )

    assert bet is not None
    assert bet["ml_prob"] == bet["true_probability"]["adjusted_prob"]
    assert bet["model_prob_raw"] == bet["true_probability"]["base_prob"]
    assert bet["market_implied_prob"] == round(1 / 1.9, 4)
    assert bet["vig_free_implied_prob"] == 0.53
    assert bet["fair_odds"] == round(1 / bet["ml_prob"], 3)
    assert bet["minimum_acceptable_odds"] == round((1.0 + 0.03) / bet["ml_prob"], 3)
    assert bet["confidence_range_low"] == bet["true_probability"]["confidence_low"]
    assert bet["confidence_range_high"] == bet["true_probability"]["confidence_high"]
    assert bet["confidence_range"][0] == bet["confidence_range_low"]
    assert bet["confidence_range"][1] == bet["confidence_range_high"]
    assert bet["lower_bound_prob"] == bet["confidence_range_low"]
    assert isinstance(bet["lower_bound_passed"], bool)


def test_build_value_bet_rejects_when_confidence_lower_bound_does_not_clear_vig_free_market() -> None:
    bet = daily_scan.build_value_bet(
        "Thin Edge",
        ml_prob=0.56,
        team_odds=1.9,
        fair_prob=0.55,
        min_edge=0.01,
        sport="soccer",
        raw_model_prob=0.56,
    )

    assert bet is None


def test_build_value_bet_rejects_when_model_probability_does_not_clear_vig_free_market() -> None:
    bet = daily_scan.build_value_bet(
        "Not Enough Edge",
        ml_prob=0.49,
        team_odds=2.1,
        fair_prob=0.5,
        min_edge=0.01,
        sport="soccer",
        raw_model_prob=0.49,
    )

    assert bet is None


def test_context_adjustments_apply_mlb_starter_uncertainty_to_true_probability() -> None:
    snapshot = pd.Series(
        {
            "sp_era_diff": -0.8,
            "sp_whip_diff": -0.12,
            "home_sp_unknown": 1,
            "away_sp_unknown": 0,
        }
    )
    adjustments = daily_scan._context_adjustments("mlb", "home", snapshot, {"rest_advantage": 1})

    names = {item.name for item in adjustments}
    assert "starter_quality" in names
    assert "starter_uncertainty" in names
    assert "rest_advantage" in names

    bet = daily_scan.build_value_bet(
        "Home Team",
        ml_prob=0.53,
        team_odds=2.05,
        fair_prob=0.49,
        min_edge=0.01,
        sport="mlb",
        raw_model_prob=0.53,
        context_adjustments=adjustments,
    )

    assert bet is not None
    assert bet["ml_prob"] > 0.50
    assert any(item["name"] == "starter_uncertainty" for item in bet["context_adjustments"])


def test_context_adjustments_capture_back_to_back_schedule_spot() -> None:
    snapshot = pd.Series({"home_b2b": 1, "away_b2b": 0})
    adjustments = daily_scan._context_adjustments("basketball", "home", snapshot, {"rest_advantage": 0})

    assert any(item.name == "back_to_back" for item in adjustments)
    assert any(item.value < 0 for item in adjustments if item.name == "back_to_back")


def test_context_adjustments_apply_travel_fatigue_signal() -> None:
    snapshot = pd.Series(
        {
            "away_travel_bucket": 3,
            "away_cross_country": 1,
            "away_crossed_2tz": 1,
            "away_travel_tz_shift": 3,
        }
    )
    adjustments = daily_scan._context_adjustments("mlb", "home", snapshot, {"rest_advantage": 0})

    assert any(item.name == "travel_fatigue" and item.value > 0 for item in adjustments)


def test_context_adjustments_apply_soccer_tactical_matchup_signal() -> None:
    snapshot = pd.Series(
        {
            "dc_xg_diff": 0.8,
            "xg_diff": 0.6,
            "home_goals_scored_avg": 1.9,
            "away_goals_conceded_avg": 1.5,
            "away_goals_scored_avg": 0.9,
            "home_goals_conceded_avg": 0.8,
            "h2h_home_win_rate": 0.67,
            "h2h_matches_count": 4,
            "home_season_pts_rate": 0.95,
            "away_season_pts_rate": 1.02,
        }
    )
    adjustments = daily_scan._context_adjustments("soccer", "home", snapshot, {"rest_advantage": 0})

    assert any(item.name == "tactical_matchup" and item.value > 0 for item in adjustments)
    assert any(item.name == "style_clash" and item.value > 0 for item in adjustments)
    assert any(item.name == "h2h_tactical_history" and item.value > 0 for item in adjustments)
    assert any(item.name == "table_pressure" for item in adjustments)


def test_detect_contextual_flags_capture_end_of_season_and_rotation_signals() -> None:
    flags = daily_scan._detect_contextual_flags(
        {
            "description": "Relegation Round Final Day - UEFA Europa League Playoff",
            "sport_title": "Soccer",
            "_league": "Domestic Cup",
            "home_rest_days": 2.0,
            "away_rest_days": 3.0,
        },
        "soccer",
    )

    assert flags["is_playoff"] == 1
    assert flags["playoff_motivation"] == 1
    assert flags["relegation_context"] == 1
    assert flags["final_day_volatility"] == 1
    assert flags["cup_rotation_risk"] == 1
    assert flags["european_rotation_risk"] == 1
    assert flags["fixture_congestion_risk"] == 1


def test_context_adjustments_apply_soccer_motivation_and_rotation_risk_signals() -> None:
    snapshot = pd.Series(
        {
            "home_season_pts_rate": 0.92,
            "away_season_pts_rate": 1.37,
            "home_pts_rate_vs_away": -0.45,
            "home_rest_days": 2.0,
            "away_rest_days": 5.0,
        }
    )
    adjustments = daily_scan._context_adjustments(
        "soccer",
        "home",
        snapshot,
        {
            "rest_advantage": 0,
            "final_day_volatility": 1,
            "fixture_congestion_risk": 1,
            "cup_rotation_risk": 1,
            "european_rotation_risk": 1,
            "home_lineup_confirmed": 0,
            "away_lineup_confirmed": 1,
            "home_likely_starters_count": 8,
            "away_likely_starters_count": 11,
        },
    )

    assert any(item.name == "relegation_motivation" and item.value > 0 for item in adjustments)
    assert any(item.name == "fixture_congestion" and item.value < 0 for item in adjustments)
    assert any(item.name == "final_day_volatility" for item in adjustments)
    assert any(item.name == "rotation_risk" for item in adjustments)
    assert any(item.name == "rotation_uncertainty" and item.value < 0 for item in adjustments)
    assert any(item.name == "late_lineup_risk" and item.value < 0 for item in adjustments)


def test_context_adjustments_penalize_soccer_team_with_nothing_to_play_for() -> None:
    snapshot = pd.Series(
        {
            "home_season_pts_rate": 1.34,
            "away_season_pts_rate": 2.08,
            "home_pts_rate_vs_away": -0.74,
        }
    )
    adjustments = daily_scan._context_adjustments("soccer", "home", snapshot, {"rest_advantage": 0})

    assert any(item.name == "nothing_to_play_for" and item.value < 0 for item in adjustments)
    assert any(item.name == "opponent_title_motivation" and item.value < 0 for item in adjustments)


def test_context_adjustments_apply_basketball_pace_control_signal() -> None:
    snapshot = pd.Series(
        {
            "expected_pace": 232.0,
            "home_pace_vs_avg": 4.0,
            "away_pace_vs_avg": -3.5,
            "home_home_wpct_20": 0.74,
            "away_away_wpct_20": 0.42,
            "home_q4_avg": 29.0,
            "away_q4_avg": 24.0,
            "home_half_ratio": 1.12,
            "away_half_ratio": 0.93,
        }
    )
    adjustments = daily_scan._context_adjustments("basketball", "home", snapshot, {"rest_advantage": 0})

    assert any(item.name == "pace_control" and item.value > 0 for item in adjustments)
    assert any(item.name == "venue_comfort" and item.value > 0 for item in adjustments)
    assert any(item.name == "closing_execution" and item.value > 0 for item in adjustments)


def test_context_adjustments_apply_mlb_matchup_and_motivation_proxies() -> None:
    snapshot = pd.Series(
        {
            "sp_era_diff": -0.5,
            "sp_whip_diff": -0.08,
            "sp_k9_diff": 1.1,
            "sp_bb9_diff": -0.7,
            "home_run_diff_10": 1.6,
            "away_run_diff_10": -0.9,
            "home_home_wpct_20": 0.68,
            "away_away_wpct_20": 0.41,
            "home_streak": 5,
            "away_streak": -3,
        }
    )
    adjustments = daily_scan._context_adjustments("mlb", "home", snapshot, {"rest_advantage": 0})

    assert any(item.name == "pitcher_command" and item.value > 0 for item in adjustments)
    assert any(item.name == "starter_form_synergy" and item.value > 0 for item in adjustments)
    assert any(item.name == "venue_split_edge" and item.value > 0 for item in adjustments)
    assert any(item.name == "streak_pressure" and item.value > 0 for item in adjustments)


def test_context_adjustments_apply_mlb_bullpen_and_park_context() -> None:
    snapshot = pd.Series(
        {
            "home_games_L3D": 1,
            "away_games_L3D": 3,
            "home_run_diff_10": 1.8,
            "away_run_diff_10": -0.4,
        }
    )
    adjustments = daily_scan._context_adjustments(
        "mlb",
        "home",
        snapshot,
        {
            "home_games_L3D": 1,
            "away_games_L3D": 3,
            "bullpen_fatigue_risk": True,
            "park_factor_proxy": 1.08,
            "park_run_environment": "hitter_friendly",
        },
    )
    names = {item.name for item in adjustments}

    assert "bullpen_workload" in names
    assert "bullpen_fatigue_risk" in names
    assert "park_factor" in names
    assert any(item.name == "bullpen_workload" and item.value > 0 for item in adjustments)


def test_context_adjustments_apply_soccer_availability_signal() -> None:
    adjustments = daily_scan._context_adjustments(
        "soccer",
        "home",
        pd.Series(dtype=float),
        {
            "home_injuries_count": 2,
            "home_suspensions_count": 1,
            "away_injuries_count": 0,
            "away_suspensions_count": 0,
        },
    )

    assert any(item.name == "availability_edge" and item.value < 0 for item in adjustments)
    assert any(item.name == "availability_uncertainty" for item in adjustments)


def test_context_adjustments_weight_soccer_priority_absences() -> None:
    adjustments = daily_scan._context_adjustments(
        "soccer",
        "home",
        pd.Series(dtype=float),
        {
            "home_injuries_count": 1,
            "home_suspensions_count": 1,
            "home_absence_severity": 4.0,
            "home_priority_absences_count": 2,
            "away_injuries_count": 0,
            "away_suspensions_count": 0,
            "away_absence_severity": 0.0,
            "away_priority_absences_count": 0,
        },
    )

    assert any(item.name == "availability_edge" and item.value < 0 for item in adjustments)
    assert any(item.name == "availability_uncertainty" for item in adjustments)


def test_context_adjustments_apply_soccer_lineup_confirmation_signal() -> None:
    adjustments = daily_scan._context_adjustments(
        "soccer",
        "home",
        pd.Series(dtype=float),
        {
            "home_lineup_confirmed": 1,
            "away_lineup_confirmed": 0,
            "home_likely_starters_count": 11,
            "away_likely_starters_count": 0,
        },
    )

    assert any(item.name == "lineup_confirmation" and item.value > 0 for item in adjustments)


def test_context_adjustments_apply_soccer_spine_absence_signal() -> None:
    adjustments = daily_scan._context_adjustments(
        "soccer",
        "home",
        pd.Series(dtype=float),
        {
            "home_spine_absences_count": 2,
            "away_spine_absences_count": 0,
            "home_absence_severity": 2.5,
            "away_absence_severity": 0.0,
        },
    )

    assert any(item.name == "availability_edge" and item.value < 0 for item in adjustments)


def test_context_adjustments_apply_nhl_goalie_uncertainty() -> None:
    adjustments = daily_scan._context_adjustments(
        "nhl",
        "home",
        pd.Series({"home_b2b": 0, "away_b2b": 0}),
        {"home_goalie_confirmed": 0, "availability_source": "feature_snapshot"},
    )

    assert any(item.name == "goalie_uncertainty" and item.value < 0 for item in adjustments)


def test_context_adjustments_apply_nhl_goalie_stability() -> None:
    adjustments = daily_scan._context_adjustments(
        "nhl",
        "home",
        pd.Series({"home_b2b": 0, "away_b2b": 0}),
        {
            "home_goalie_confirmed": 1,
            "away_goalie_confirmed": 0,
            "home_goalie_name": "Frederik Andersen",
            "availability_source": "feature_snapshot",
        },
    )

    assert any(item.name == "goalie_stability" and item.value > 0 for item in adjustments)


def test_context_adjustments_apply_nhl_goalie_quality_signal() -> None:
    adjustments = daily_scan._context_adjustments(
        "nhl",
        "home",
        pd.Series(
            {
                "home_b2b": 0,
                "away_b2b": 0,
                "home_pp_pct_10": 0.27,
                "away_pp_pct_10": 0.18,
                "home_pk_pct_10": 0.84,
                "away_pk_pct_10": 0.77,
                "home_home_wpct_20": 0.69,
                "away_away_wpct_20": 0.45,
                "home_xg_diff_10": 0.42,
                "away_xg_diff_10": -0.11,
            }
        ),
        {
            "home_goalie_name": "Frederik Andersen",
            "away_goalie_name": "Backup Goalie",
            "home_goalie_save_pct": 0.924,
            "away_goalie_save_pct": 0.901,
            "home_goalie_gaa": 2.18,
            "away_goalie_gaa": 2.87,
        },
    )

    assert any(item.name == "goalie_quality" and item.value > 0 for item in adjustments)
    assert any(item.name == "special_teams_edge" and item.value > 0 for item in adjustments)
    assert any(item.name == "system_stability" and item.value > 0 for item in adjustments)
    assert any(item.name == "xg_structure" and item.value > 0 for item in adjustments)


def test_publish_guardrails_can_promote_moderate_flagged_bet_with_context() -> None:
    bets = [
        {
            "sport": "soccer",
            "market": "double_chance",
            "team": "Home or Draw",
            "home": "Home",
            "away": "Away",
            "window": "today",
            "odds": 1.55,
            "ml_prob": 0.70,
            "fair_prob": 0.62,
            "edge": 0.085,
            "kelly_stake_pct": 2.0,
            "market_status": "preferred",
            "production_allowed": True,
            "flagged": True,
            "context_adjustments": [
                {"name": "tactical_matchup", "category": "matchup", "value": 0.01, "summary": "Strong matchup"},
                {"name": "travel_fatigue", "category": "environment", "value": 0.008, "summary": "Travel edge"},
            ],
        }
    ]
    published, review, suppressed = daily_scan.apply_publish_guardrails(bets, bankroll=1000)

    assert len(published) == 1
    assert not review
    assert not suppressed
    assert published[0]["flagged_promoted"] is True


def test_team_resolver_normalized_fuzzy_resolution() -> None:
    resolver = TeamResolver("nhl")
    resolver._normalized_canonical_map["montreal canadiens"] = "Canadiens"
    resolved = resolver.resolve("Montréal Canadiens FC")
    assert resolved == "Canadiens"


def test_soccer_team_resolver_uses_registry_aliases() -> None:
    resolver = TeamResolver("soccer")
    assert resolver.resolve("Atletico Madrid") == "Club Atlético de Madrid"
    assert resolver.resolve("Wolves") == "Wolverhampton Wanderers FC"


def test_build_value_bet_carries_goalie_summary() -> None:
    bet = daily_scan.build_value_bet(
        "Carolina Hurricanes",
        ml_prob=0.55,
        team_odds=1.9,
        fair_prob=0.52,
        min_edge=0.01,
        sport="nhl",
        raw_model_prob=0.55,
        availability_context={
            "home_goalie_confirmed": 0,
            "home_goalie_name": "Frederik Andersen",
            "availability_source": "feature_snapshot",
            "home_team_name": "Carolina Hurricanes",
            "away_team_name": "New York Rangers",
        },
    )

    assert bet is not None
    assert bet["availability_summary"] == "Home goalie not confirmed: Frederik Andersen"


def test_context_adjustments_apply_basketball_injury_report_signal() -> None:
    adjustments = daily_scan._context_adjustments(
        "basketball",
        "home",
        pd.Series(dtype=float),
        {
            "home_injuries_count": 2,
            "home_questionable_count": 1,
            "home_rotation_absence_severity": 2.7,
            "home_priority_absences_count": 2,
            "away_injuries_count": 0,
            "away_questionable_count": 0,
            "away_rotation_absence_severity": 0.0,
            "away_priority_absences_count": 0,
            "availability_source": "api_sports_basketball",
        },
    )

    assert any(item.name == "injury_report_edge" and item.value < 0 for item in adjustments)
    assert any(item.name == "rotation_quality_edge" and item.value < 0 for item in adjustments)
    assert any(item.name == "lineup_uncertainty" for item in adjustments)


def test_build_value_bet_carries_basketball_availability_summary() -> None:
    bet = daily_scan.build_value_bet(
        "Boston Celtics",
        ml_prob=0.6,
        team_odds=1.74,
        fair_prob=0.55,
        min_edge=0.01,
        sport="basketball",
        raw_model_prob=0.6,
        availability_context={
            "home_injuries_count": 1,
            "home_questionable_count": 2,
            "availability_source": "api_sports_basketball",
            "home_team_name": "Boston Celtics",
            "away_team_name": "Miami Heat",
        },
    )

    assert bet is not None
    assert bet["availability_summary"] == "Home absences: 1 inj, 2 q"


def test_build_value_bet_carries_mlb_starter_summary() -> None:
    bet = daily_scan.build_value_bet(
        "New York Yankees",
        ml_prob=0.57,
        team_odds=1.8,
        fair_prob=0.53,
        min_edge=0.01,
        sport="mlb",
        raw_model_prob=0.57,
        availability_context={
            "home_starter_confirmed": 1,
            "home_starter_name": "Gerrit Cole",
            "availability_source": "mlb_stats_api",
            "home_team_name": "New York Yankees",
            "away_team_name": "Boston Red Sox",
        },
    )

    assert bet is not None
    assert bet["availability_summary"] == "Home starter confirmed: Gerrit Cole"


def test_build_value_bet_carries_soccer_lineup_summary() -> None:
    bet = daily_scan.build_value_bet(
        "Atletico Madrid",
        ml_prob=0.56,
        team_odds=1.82,
        fair_prob=0.54,
        min_edge=0.01,
        sport="soccer",
        raw_model_prob=0.56,
        availability_context={
            "away_lineup_confirmed": 1,
            "away_likely_starters_count": 11,
            "availability_source": "api_football",
            "home_team_name": "Elche CF",
            "away_team_name": "Atletico Madrid",
        },
    )

    assert bet is not None
    assert bet["availability_summary"] == "Away XI posted: 11 starters"


def test_audit_candidate_freshness_blocks_live_match() -> None:
    now = datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc)
    audit = daily_scan.audit_candidate_freshness(
        {
            "sport": "soccer",
            "home": "Home",
            "away": "Away",
            "commence": "2026-05-05T11:30:00Z",
        },
        now=now,
    )

    assert audit.match_status == "live"
    assert "already live" in audit.suppression_reason


def test_audit_candidate_freshness_blocks_finished_match() -> None:
    now = datetime(2026, 5, 5, 16, 30, tzinfo=timezone.utc)
    audit = daily_scan.audit_candidate_freshness(
        {
            "sport": "soccer",
            "home": "Home",
            "away": "Away",
            "commence": "2026-05-05T11:30:00Z",
        },
        now=now,
    )

    assert audit.match_status == "finished"
    assert "already finished" in audit.suppression_reason


def test_audit_candidate_freshness_holds_stale_odds_snapshot() -> None:
    now = datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc)
    audit = daily_scan.audit_candidate_freshness(
        {
            "sport": "soccer",
            "home": "Arsenal",
            "away": "Chelsea",
            "commence": "2026-05-05T16:00:00Z",
            "odds_snapshot_age_hours": 30,
            "scraped_context": {
                "home_team_name": "Arsenal",
                "away_team_name": "Chelsea",
                "availability_source": "api_football",
            },
        },
        now=now,
    )

    assert audit.odds_freshness == "stale"
    assert "odds snapshot" in audit.review_reason


def test_tag_odds_payload_uses_bookmaker_timestamp_for_fresh_force_fetched_odds() -> None:
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    games = [
        {
            "id": "match-1",
            "commence_time": "2026-05-06T18:00:00Z",
            "bookmakers": [
                {
                    "title": "Book A",
                    "last_update": "2026-05-06T10:30:00Z",
                    "markets": [
                        {
                            "key": "h2h",
                            "last_update": "2026-05-06T10:30:00Z",
                            "outcomes": [{"name": "Alpha FC", "price": 2.0}],
                        }
                    ],
                }
            ],
        }
    ]

    tagged = daily_scan._tag_odds_payload(
        games,
        source_status="live_api",
        fetched_at=now,
        force_fresh=True,
        source_reason="test_live_fetch",
    )

    assert tagged[0]["_odds_source_status"] == "live_api"
    assert tagged[0]["_odds_source_detail"] == "fresh_odds_used"
    assert tagged[0]["_odds_fetched_at"] == now.isoformat()
    assert tagged[0]["_odds_bookmaker_last_update"] == "2026-05-06T10:30:00+00:00"
    assert tagged[0]["_odds_snapshot_age_hours"] == 1.5

    audit = daily_scan.audit_candidate_freshness(
        {
            "sport": "soccer",
            "home": "Alpha FC",
            "away": "Beta FC",
            "commence": "2026-05-06T18:00:00Z",
            "odds_source_status": tagged[0]["_odds_source_status"],
            "bookmaker_last_update": tagged[0]["_odds_bookmaker_last_update"],
            "odds_snapshot_age_hours": tagged[0]["_odds_snapshot_age_hours"],
        },
        now=now,
    )

    assert audit.odds_freshness == "fresh"


def test_tag_odds_payload_does_not_confuse_fetch_time_with_bookmaker_last_update() -> None:
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    games = [
        {
            "id": "match-1",
            "commence_time": "2026-05-06T18:00:00Z",
            "bookmakers": [
                {
                    "title": "Book A",
                    "last_update": "2026-05-05T20:00:00Z",
                    "markets": [
                        {
                            "key": "h2h",
                            "last_update": "2026-05-05T20:00:00Z",
                            "outcomes": [{"name": "Alpha FC", "price": 2.0}],
                        }
                    ],
                }
            ],
        }
    ]

    tagged = daily_scan._tag_odds_payload(games, source_status="live_api", fetched_at=now)

    assert tagged[0]["_odds_fetched_at"] == now.isoformat()
    assert tagged[0]["_odds_bookmaker_last_update"] == "2026-05-05T20:00:00+00:00"
    assert tagged[0]["_odds_snapshot_age_hours"] == 16.0


def test_audit_candidate_freshness_treats_missing_bookmaker_timestamp_as_unverified() -> None:
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    audit = daily_scan.audit_candidate_freshness(
        {
            "sport": "soccer",
            "home": "Alpha FC",
            "away": "Beta FC",
            "commence": "2026-05-06T18:00:00Z",
            "odds_source_status": "live_api",
            "odds_fetched_at": now.isoformat(),
        },
        now=now,
    )

    assert audit.odds_freshness == "unknown"


def test_audit_candidate_freshness_marks_stale_fallback_cache_as_stale() -> None:
    now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    audit = daily_scan.audit_candidate_freshness(
        {
            "sport": "soccer",
            "home": "Alpha FC",
            "away": "Beta FC",
            "commence": "2026-05-06T18:00:00Z",
            "odds_source_status": "stale_fallback",
            "odds_source_detail": "loaded_from_stale_cache_fallback",
            "odds_snapshot_age_hours": 28.0,
        },
        now=now,
    )

    assert audit.odds_freshness == "stale"


def test_audit_candidate_freshness_holds_stale_injury_news_near_kickoff() -> None:
    now = datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc)
    audit = daily_scan.audit_candidate_freshness(
        {
            "sport": "soccer",
            "home": "Arsenal",
            "away": "Chelsea",
            "commence": "2026-05-05T14:00:00Z",
            "scraped_context": {
                "home_team_name": "Arsenal",
                "away_team_name": "Chelsea",
                "availability_source": "api_football",
                "availability_fetched_at": "2026-05-04T23:00:00Z",
                "home_lineup_confirmed": 1,
                "away_lineup_confirmed": 1,
            },
        },
        now=now,
    )

    assert audit.injury_news_freshness == "stale"
    assert "injury/news freshness check failed" in audit.review_reason


def test_audit_candidate_freshness_holds_fixture_mismatch() -> None:
    now = datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc)
    audit = daily_scan.audit_candidate_freshness(
        {
            "sport": "soccer",
            "home": "Arsenal",
            "away": "Chelsea",
            "commence": "2026-05-05T16:00:00Z",
            "scraped_context": {
                "home_team_name": "Arsenal",
                "away_team_name": "Tottenham",
                "availability_source": "api_football",
            },
        },
        now=now,
    )

    assert audit.fixture_verified is False
    assert "fixture verification mismatch" in audit.review_reason


def test_apply_publish_guardrails_waits_for_lineups_near_kickoff() -> None:
    now = datetime.now(timezone.utc)
    commence = (now + timedelta(minutes=70)).isoformat().replace("+00:00", "Z")
    bets = [
        {
            "sport": "soccer",
            "market": "moneyline",
            "team": "Atletico Madrid",
            "home": "Elche CF",
            "away": "Atletico Madrid",
            "window": "today",
            "commence": commence,
            "odds": 1.82,
            "ml_prob": 0.58,
            "fair_prob": 0.54,
            "edge": 0.04,
            "kelly_stake_pct": 2.0,
            "scraped_context": {
                "home_team_name": "Elche CF",
                "away_team_name": "Atletico Madrid",
                "availability_source": "api_football",
            },
        }
    ]

    published, review, suppressed = daily_scan.apply_publish_guardrails(bets, bankroll=1000.0)

    assert published == []
    assert suppressed == []
    assert len(review) == 1
    assert "lineup freshness check failed" in review[0]["review_reason"]


def test_apply_publish_guardrails_holds_stale_force_fresh_fallback_odds() -> None:
    now = datetime.now(timezone.utc)
    commence = (now + timedelta(hours=4)).isoformat().replace("+00:00", "Z")
    bets = [
        {
            "sport": "soccer",
            "market": "moneyline",
            "team": "Alpha FC",
            "home": "Alpha FC",
            "away": "Beta FC",
            "window": "today",
            "commence": commence,
            "odds": 1.95,
            "ml_prob": 0.57,
            "fair_prob": 0.53,
            "edge": 0.04,
            "kelly_stake_pct": 2.0,
            "odds_source_status": "stale_fallback",
            "odds_source_detail": "fallback_odds_used_because_force_fresh_fetch_could_not_complete",
            "force_fresh_odds_active": True,
            "odds_fallback_used": True,
            "bookmaker_last_update": (now - timedelta(hours=30)).isoformat(),
            "odds_snapshot_age_hours": 30.0,
            "scraped_context": {
                "home_team_name": "Alpha FC",
                "away_team_name": "Beta FC",
                "availability_source": "api_football",
                "availability_fetched_at": now.isoformat(),
            },
        }
    ]

    published, review, suppressed = daily_scan.apply_publish_guardrails(bets, bankroll=1000.0)
    _, labelled_review, _ = daily_scan._apply_decision_labels([], review, [])

    assert published == []
    assert suppressed == []
    assert len(review) == 1
    assert review[0]["publication_outcome"] == "review"
    assert review[0]["odds_source_status"] == "stale_fallback"
    assert "stale" in review[0]["review_reason"].lower()
    assert labelled_review[0]["decision_status"] == DECISION_HOLD


def test_apply_publish_guardrails_moves_experimental_markets_to_review_and_blocks_same_game_duplicates() -> None:
    bets = [
        {
            "sport": "soccer",
            "market": "double_chance",
            "team": "Home or Draw",
            "home": "Home",
            "away": "Away",
            "window": "today",
            "odds": 1.6,
            "ml_prob": 0.74,
            "fair_prob": 0.66,
            "edge": 0.05,
            "kelly_stake_pct": 2.0,
        },
        {
            "sport": "soccer",
            "market": "draw_no_bet",
            "team": "Home DNB",
            "home": "Home",
            "away": "Away",
            "window": "today",
            "odds": 1.8,
            "ml_prob": 0.68,
            "fair_prob": 0.6,
            "edge": 0.04,
            "kelly_stake_pct": 1.5,
        },
        {
            "sport": "mlb",
            "market": "moneyline",
            "team": "Blue Jays",
            "home": "Blue Jays",
            "away": "Yankees",
            "window": "today",
            "odds": 2.3,
            "ml_prob": 0.5,
            "fair_prob": 0.43,
            "edge": 0.06,
            "kelly_stake_pct": 2.4,
        },
    ]

    published, review, suppressed = daily_scan.apply_publish_guardrails(bets, bankroll=1000.0)

    assert len(published) == 2
    assert len(review) == 0
    assert {item["market"] for item in published} == {"double_chance", "moneyline"}
    reasons = {item["suppression_reason"] for item in suppressed}
    assert any("same-game side" in reason for reason in reasons)


def test_apply_publish_guardrails_rejects_contradictory_minus_handicap_price(monkeypatch) -> None:
    monkeypatch.setattr(daily_scan, "annotate_bet", lambda bet: dict(bet))
    bets = [
        {
            "sport": "mlb",
            "market": "moneyline",
            "team": "Blue Jays",
            "home": "Blue Jays",
            "away": "Red Sox",
            "window": "today",
            "odds": 2.39,
            "ml_prob": 0.47,
            "fair_prob": 0.42,
            "edge": 0.05,
            "kelly_stake_pct": 1.8,
            "production_allowed": True,
        },
        {
            "sport": "mlb",
            "market": "spreads",
            "team": "Blue Jays -1.5",
            "home": "Blue Jays",
            "away": "Red Sox",
            "window": "today",
            "odds": 1.5,
            "ml_prob": 0.62,
            "fair_prob": 0.55,
            "edge": 0.04,
            "kelly_stake_pct": 1.2,
            "production_allowed": True,
        },
    ]

    published, review, suppressed = daily_scan.apply_publish_guardrails(bets, bankroll=1000.0)

    assert len(published) == 1
    assert len(review) == 0
    assert published[0]["market"] == "moneyline"
    assert any("minus-handicap price is shorter" in item["suppression_reason"] for item in suppressed)


def test_apply_publish_guardrails_moves_flagged_and_uncertain_bets_to_review_queue(monkeypatch) -> None:
    monkeypatch.setattr(daily_scan, "annotate_bet", lambda bet: dict(bet))
    bets = [
        {
            "sport": "mlb",
            "market": "moneyline",
            "team": "Blue Jays",
            "home": "Blue Jays",
            "away": "Yankees",
            "window": "today",
            "odds": 2.1,
            "ml_prob": 0.58,
            "fair_prob": 0.49,
            "edge": 0.06,
            "kelly_stake_pct": 1.8,
            "flagged": True,
            "production_allowed": True,
            "context_adjustments": [
                {"name": "starter_uncertainty", "category": "lineup", "value": -0.018},
            ],
        }
    ]

    published, review, suppressed = daily_scan.apply_publish_guardrails(bets, bankroll=1000.0)

    assert published == []
    assert suppressed == []
    assert len(review) == 1
    assert review[0]["recommendation_type"] == "manual_review"
    assert "review" in review[0]["review_reason"].lower()


def test_apply_publish_guardrails_uses_lane_governor_to_hold_weak_preferred_lane(monkeypatch) -> None:
    monkeypatch.setattr(daily_scan, "annotate_bet", lambda bet: dict(bet))
    monkeypatch.setattr(
        daily_scan,
        "_MARKET_HEALTH_BY_MARKET",
        {
            ("soccer", "moneyline"): {
                "bets": 18,
                "clv_coverage_pct": 100.0,
                "action": "pause",
                "clv_signal": "weak",
                "roi_pct": -9.0,
                "avg_clv_pct": -14.0,
            }
        },
    )
    bets = [
        {
            "sport": "soccer",
            "market": "moneyline",
            "team": "Team A",
            "home": "Team A",
            "away": "Team B",
            "window": "today",
            "odds": 2.1,
            "ml_prob": 0.56,
            "fair_prob": 0.48,
            "edge": 0.05,
            "kelly_stake_pct": 1.8,
            "production_allowed": True,
            "market_status": "preferred",
            "market_policy_label": "Preferred",
        }
    ]

    published, review, suppressed = daily_scan.apply_publish_guardrails(bets, bankroll=1000.0)

    assert published == []
    assert suppressed == []
    assert len(review) == 1
    assert "lane performance governor" in review[0]["review_reason"]


def test_apply_publish_guardrails_tightens_stake_for_soft_weak_lane(monkeypatch) -> None:
    monkeypatch.setattr(daily_scan, "annotate_bet", lambda bet: dict(bet))
    monkeypatch.setattr(
        daily_scan,
        "_MARKET_HEALTH_BY_MARKET",
        {
            ("mlb", "moneyline"): {
                "bets": 16,
                "clv_coverage_pct": 100.0,
                "action": "tighten",
                "clv_signal": "variance",
                "roi_pct": -1.0,
                "avg_clv_pct": 2.5,
            }
        },
    )
    bets = [
        {
            "sport": "mlb",
            "market": "moneyline",
            "team": "Blue Jays",
            "home": "Blue Jays",
            "away": "Yankees",
            "window": "today",
            "odds": 2.3,
            "ml_prob": 0.5,
            "fair_prob": 0.43,
            "edge": 0.06,
            "kelly_stake_pct": 2.4,
            "production_allowed": True,
            "market_status": "preferred",
            "market_policy_label": "Preferred",
        },
    ]

    published, review, suppressed = daily_scan.apply_publish_guardrails(bets, bankroll=1000.0)

    assert review == []
    assert suppressed == []
    assert len(published) == 1
    assert published[0]["stake_multiplier"] == 0.5
    assert published[0]["kelly_stake_pct"] == 1.2


def test_apply_publish_guardrails_blocks_bets_below_minimum_acceptable_odds() -> None:
    bets = [
        {
            "sport": "soccer",
            "market": "double_chance",
            "team": "Team A or Draw",
            "home": "Team B",
            "away": "Team A",
            "window": "today",
            "odds": 1.38,
            "minimum_acceptable_odds": 1.45,
            "edge": 0.06,
            "ml_prob": 0.72,
            "production_allowed": True,
            "market_status": "preferred",
            "context_adjustments": [],
            "kelly_stake_pct": 1.0,
        }
    ]

    published, review, suppressed = daily_scan.apply_publish_guardrails(bets, bankroll=1000.0)

    assert published == []
    assert review == []
    assert len(suppressed) == 1
    assert suppressed[0]["odds_recheck_status"] == "failed"
    assert "minimum acceptable price" in suppressed[0]["suppression_reason"]


def test_market_suitability_prefers_double_chance_over_soccer_moneyline() -> None:
    bets = [
        {
            "sport": "soccer",
            "market": "double_chance",
            "team": "Away FC or Draw",
            "home": "Home FC",
            "away": "Away FC",
            "window": "today",
            "odds": 1.55,
            "ml_prob": 0.77,
            "fair_prob": 0.68,
            "edge": 0.05,
            "kelly_stake_pct": 1.6,
        },
        {
            "sport": "soccer",
            "market": "moneyline",
            "team": "Away FC",
            "home": "Home FC",
            "away": "Away FC",
            "window": "today",
            "odds": 2.45,
            "ml_prob": 0.54,
            "fair_prob": 0.43,
            "edge": 0.04,
            "kelly_stake_pct": 1.0,
        },
        {
            "sport": "soccer",
            "market": "moneyline",
            "team": "Home FC",
            "home": "Home FC",
            "away": "Away FC",
            "window": "today",
            "odds": 2.9,
            "ml_prob": 0.49,
            "fair_prob": 0.36,
            "edge": 0.03,
            "kelly_stake_pct": 0.8,
        },
    ]

    published, review, suppressed = daily_scan.apply_publish_guardrails(bets, bankroll=1000.0)

    assert {item["market"] for item in published} == {"double_chance"}
    moneyline = next(item for item in suppressed if item["market"] == "moneyline" and item["team"] == "Away FC")
    assert moneyline["recommended_market"] == "double_chance"
    assert "preferred double chance" in moneyline["suppression_reason"].lower()


def test_market_suitability_prefers_positive_handicap_over_soccer_dnb() -> None:
    bets = [
        {
            "sport": "soccer",
            "market": "spreads",
            "team": "Roaders +0.5",
            "home": "Hosts",
            "away": "Roaders",
            "window": "today",
            "odds": 1.78,
            "ml_prob": 0.61,
            "fair_prob": 0.55,
            "edge": 0.04,
            "kelly_stake_pct": 1.3,
        },
        {
            "sport": "soccer",
            "market": "draw_no_bet",
            "team": "Roaders DNB",
            "home": "Hosts",
            "away": "Roaders",
            "window": "today",
            "odds": 2.02,
            "ml_prob": 0.41,
            "fair_prob": 0.35,
            "edge": 0.03,
            "kelly_stake_pct": 0.9,
        },
        {
            "sport": "soccer",
            "market": "moneyline",
            "team": "Roaders",
            "home": "Hosts",
            "away": "Roaders",
            "window": "today",
            "odds": 3.05,
            "ml_prob": 0.41,
            "fair_prob": 0.31,
            "edge": 0.025,
            "kelly_stake_pct": 0.5,
        },
        {
            "sport": "soccer",
            "market": "moneyline",
            "team": "Hosts",
            "home": "Hosts",
            "away": "Roaders",
            "window": "today",
            "odds": 2.35,
            "ml_prob": 0.53,
            "fair_prob": 0.43,
            "edge": 0.03,
            "kelly_stake_pct": 0.7,
        },
    ]

    published, review, suppressed = daily_scan.apply_publish_guardrails(bets, bankroll=1000.0)

    assert any(item["market"] == "spreads" for item in published)
    dnb = next(item for item in suppressed if item["market"] == "draw_no_bet")
    assert dnb["recommended_market"] == "spreads"
    assert "positive handicap" in dnb["suppression_reason"].lower()


def test_market_suitability_rejection_is_labeled_avoid() -> None:
    bets = [
        {
            "sport": "soccer",
            "market": "double_chance",
            "team": "Away FC or Draw",
            "home": "Home FC",
            "away": "Away FC",
            "window": "today",
            "odds": 1.55,
            "ml_prob": 0.77,
            "fair_prob": 0.68,
            "edge": 0.05,
            "kelly_stake_pct": 1.6,
            "decision_status": "BET",
        },
        {
            "sport": "soccer",
            "market": "moneyline",
            "team": "Away FC",
            "home": "Home FC",
            "away": "Away FC",
            "window": "today",
            "odds": 2.45,
            "ml_prob": 0.54,
            "fair_prob": 0.43,
            "edge": 0.04,
            "kelly_stake_pct": 1.0,
        },
        {
            "sport": "soccer",
            "market": "moneyline",
            "team": "Home FC",
            "home": "Home FC",
            "away": "Away FC",
            "window": "today",
            "odds": 2.9,
            "ml_prob": 0.49,
            "fair_prob": 0.36,
            "edge": 0.03,
            "kelly_stake_pct": 0.8,
        },
    ]

    published, review, suppressed = daily_scan.apply_publish_guardrails(bets, bankroll=1000.0)
    labelled_published, labelled_review, labelled_suppressed = daily_scan._apply_decision_labels(published, review, suppressed)

    assert labelled_published
    moneyline = next(item for item in labelled_suppressed if item["market"] == "moneyline" and item["team"] == "Away FC")
    assert moneyline["decision_status"] == DECISION_AVOID


def test_market_suitability_does_not_flip_bad_dnb_into_opposite_side_moneyline() -> None:
    peer_bets = [
        {
            "sport": "soccer",
            "market": "draw_no_bet",
            "team": "Roaders DNB",
            "home": "Hosts",
            "away": "Roaders",
            "ml_prob": 0.41,
            "fair_prob": 0.35,
            "edge": 0.03,
        },
        {
            "sport": "soccer",
            "market": "spreads",
            "team": "Roaders +1.0",
            "home": "Hosts",
            "away": "Roaders",
            "ml_prob": 0.59,
            "fair_prob": 0.53,
            "edge": 0.03,
        },
        {
            "sport": "soccer",
            "market": "moneyline",
            "team": "Roaders",
            "home": "Hosts",
            "away": "Roaders",
            "ml_prob": 0.41,
            "fair_prob": 0.31,
            "edge": 0.025,
        },
        {
            "sport": "soccer",
            "market": "moneyline",
            "team": "Hosts",
            "home": "Hosts",
            "away": "Roaders",
            "ml_prob": 0.53,
            "fair_prob": 0.43,
            "edge": 0.03,
        },
    ]

    decision = daily_scan.evaluate_market_suitability(peer_bets[0], peer_bets)

    assert decision.recommended_market == "spreads"
    assert decision.recommended_market != "moneyline"


def test_market_suitability_prefers_totals_for_balanced_volatile_match() -> None:
    peer_bets = [
        {
            "sport": "soccer",
            "market": "moneyline",
            "team": "Away FC",
            "home": "Home FC",
            "away": "Away FC",
            "ml_prob": 0.53,
            "fair_prob": 0.47,
            "edge": 0.03,
        },
        {
            "sport": "soccer",
            "market": "moneyline",
            "team": "Home FC",
            "home": "Home FC",
            "away": "Away FC",
            "ml_prob": 0.50,
            "fair_prob": 0.45,
            "edge": 0.02,
        },
        {
            "sport": "soccer",
            "market": "totals",
            "team": "Over 2.5",
            "home": "Home FC",
            "away": "Away FC",
            "ml_prob": 0.58,
            "fair_prob": 0.52,
            "edge": 0.04,
        },
    ]

    decision = daily_scan.evaluate_market_suitability(peer_bets[0], peer_bets)

    assert decision.recommended_market == "totals"


def test_context_referee_moves_published_bet_to_review(monkeypatch) -> None:
    published = [
        {
            "sport": "soccer",
            "market": "double_chance",
            "team": "Team A or Draw",
            "home": "Team B",
            "away": "Team A",
            "edge": 0.06,
            "ml_prob": 0.72,
            "odds": 1.55,
            "market_priority_score": 5,
            "production_allowed": True,
        }
    ]

    monkeypatch.setattr(
        daily_scan,
        "_run_context_referee",
        lambda _bet: {
            "provider": "openrouter",
            "model": "openrouter/free",
            "content": {"decision": "REVIEW", "reasoning": "Late uncertainty around lineup stability."},
        },
    )

    kept, review, suppressed = daily_scan.apply_context_referee(
        published,
        [],
        [],
        enabled=True,
        max_candidates=4,
    )

    assert kept == []
    assert suppressed == []
    assert len(review) == 1
    assert review[0]["context_referee_decision"] == "REVIEW"
    assert "lineup stability" in review[0]["review_reason"]


def test_publish_guardrails_move_review_only_league_to_manual_review() -> None:
    bets = [
        {
            "sport": "soccer",
            "league": "MLS",
            "league_key": "soccer_usa_mls",
            "market": "moneyline",
            "team": "Seattle Sounders",
            "home": "Seattle Sounders",
            "away": "LA Galaxy",
            "edge": 0.06,
            "ml_prob": 0.58,
            "odds": 1.95,
            "production_allowed": True,
            "market_status": "preferred",
            "context_adjustments": [],
            "kelly_stake_pct": 1.0,
        }
    ]

    published, review, suppressed = daily_scan.apply_publish_guardrails(bets, bankroll=1000.0)

    assert published == []
    assert suppressed == []
    assert len(review) == 1
    assert review[0]["publishable"] is False
    assert review[0]["launch_label"] == "Review"
    assert "review-only" in review[0]["review_reason"].lower() or "held out" in review[0]["review_reason"].lower()


def test_publish_guardrails_suppress_failed_prediction_quality_before_review() -> None:
    bets = [
        {
            "sport": "soccer",
            "league": "Premier League",
            "league_key": "soccer_epl",
            "market": "moneyline",
            "team": "Alpha FC",
            "home": "Alpha FC",
            "away": "Beta FC",
            "edge": 0.07,
            "ml_prob": 0.58,
            "odds": 1.95,
            "lower_bound_passed": False,
            "production_allowed": True,
            "market_status": "preferred",
            "context_adjustments": [],
            "kelly_stake_pct": 1.0,
        }
    ]

    published, review, suppressed = daily_scan.apply_publish_guardrails(bets, bankroll=1000.0)

    assert published == []
    assert review == []
    assert len(suppressed) == 1
    assert suppressed[0]["prediction_quality_ok"] is False
    assert "prediction quality failed" in suppressed[0]["suppression_reason"]


def test_publish_guardrails_suppress_large_model_disagreement_before_review() -> None:
    bets = [
        {
            "sport": "basketball",
            "league": "NBA",
            "league_key": "basketball_nba",
            "market": "moneyline",
            "team": "Alpha Hoops",
            "home": "Alpha Hoops",
            "away": "Beta Hoops",
            "edge": 0.08,
            "ml_prob": 0.59,
            "odds": 1.95,
            "lower_bound_passed": True,
            "lower_bound_edge": 0.03,
            "basketball_probability_debug": {
                "disagreement_pp": 24.0,
                "history_rows": {"home": 30, "away": 30},
            },
            "production_allowed": True,
            "market_status": "preferred",
            "context_adjustments": [],
            "kelly_stake_pct": 1.0,
        }
    ]

    published, review, suppressed = daily_scan.apply_publish_guardrails(bets, bankroll=1000.0)

    assert published == []
    assert review == []
    assert len(suppressed) == 1
    assert suppressed[0]["prediction_quality_ok"] is False
    assert "disagree too much" in suppressed[0]["suppression_reason"]


def test_publish_guardrails_suppress_nonfocused_lane_before_review() -> None:
    bets = [
        {
            "sport": "soccer",
            "league": "Premier League",
            "league_key": "soccer_epl",
            "market": "btts",
            "team": "Team A +0.5",
            "home": "Team A",
            "away": "Team B",
            "window": "today",
            "odds": 1.91,
            "ml_prob": 0.56,
            "fair_prob": 0.52,
            "edge": 0.05,
            "kelly_stake_pct": 1.0,
            "odds_snapshot_age_hours": 99.0,
        }
    ]

    published, review, suppressed = daily_scan.apply_publish_guardrails(bets, bankroll=1000.0)

    assert published == []
    assert review == []
    assert len(suppressed) == 1
    assert suppressed[0]["prediction_focus_allowed"] is False
    assert "focused prediction lanes" in suppressed[0]["suppression_reason"]


def test_context_referee_vetoes_published_bet(monkeypatch) -> None:
    published = [
        {
            "sport": "mlb",
            "market": "moneyline",
            "team": "Home Team",
            "home": "Home Team",
            "away": "Away Team",
            "edge": 0.08,
            "ml_prob": 0.59,
            "odds": 1.95,
            "market_priority_score": 5,
            "production_allowed": True,
        }
    ]

    monkeypatch.setattr(
        daily_scan,
        "_run_context_referee",
        lambda _bet: {
            "provider": "openrouter",
            "model": "openrouter/free",
            "content": {"decision": "VETO", "reasoning": "Confirmed starter scratch invalidated the edge."},
        },
    )

    kept, review, suppressed = daily_scan.apply_context_referee(
        published,
        [],
        [],
        enabled=True,
        max_candidates=4,
    )

    assert kept == []
    assert review == []
    assert len(suppressed) == 1
    assert suppressed[0]["context_referee_decision"] == "VETO"
    assert "context referee vetoed" in suppressed[0]["suppression_reason"]


def _committee_candidate(
    *,
    edge: float = 0.08,
    odds: float = 1.92,
    commence_delta_hours: float = 6.0,
) -> dict:
    commence = (datetime.now(timezone.utc) + timedelta(hours=commence_delta_hours)).isoformat()
    return {
        "sport": "soccer",
        "market": "moneyline",
        "team": "Alpha FC",
        "home": "Alpha FC",
        "away": "Beta FC",
        "edge": edge,
        "odds": odds,
        "ml_prob": 0.60,
        "fair_prob": 0.52,
        "market_implied_prob": 0.54,
        "vig_free_implied_prob": 0.51,
        "minimum_acceptable_odds": 1.78,
        "confidence_range_low": 0.55,
        "confidence_range_high": 0.64,
        "lower_bound_passed": True,
        "publish_ready": True,
        "production_allowed": True,
        "market_status": "preferred",
        "market_policy_label": "Preferred",
        "market_policy_reason": "Preferred lane",
        "window": "today",
        "commence": commence,
        "commence_time": commence,
        "odds_snapshot_age_hours": 1.0,
        "standings_snapshot_age_hours": 3.0,
        "scraped_context": {
            "home_team_name": "Alpha FC",
            "away_team_name": "Beta FC",
            "availability_source": "api_football",
            "availability_fetched_at": datetime.now(timezone.utc).isoformat(),
            "home_lineup_confirmed": 1,
            "away_lineup_confirmed": 1,
            "home_likely_starters_count": 11,
            "away_likely_starters_count": 11,
        },
        "scraped_context_sources": ["api_football"],
        "scraped_context_highlights": ["Availability context fetched from api_football"],
        "context_adjustments": [],
        "kelly_stake_pct": 1.0,
        "stake_abs": 10.0,
    }


def test_committee_pipeline_passes_real_scan_candidate_before_publication() -> None:
    published, review, suppressed, entries = daily_scan.run_committee_pipeline(
        published=[_committee_candidate()],
        review=[],
        suppressed=[],
    )

    assert len(entries) == 1
    assert len(published) == 1
    assert review == []
    assert suppressed == []
    assert published[0]["committee_final_decision"] == "BET"
    assert published[0]["decision_status"] == DECISION_BET
    assert "committee" in published[0]


def test_committee_pipeline_rejects_arbiter_no_bet_from_real_publish_flow() -> None:
    published, review, suppressed, _ = daily_scan.run_committee_pipeline(
        published=[_committee_candidate(edge=0.01)],
        review=[],
        suppressed=[],
    )

    assert published == []
    assert review == []
    assert len(suppressed) == 1
    assert suppressed[0]["committee_final_decision"] == "NO_BET"
    assert suppressed[0]["decision_status"] == DECISION_NO_BET


def test_committee_pipeline_keeps_precommittee_suppressed_candidates_suppressed() -> None:
    candidate = _committee_candidate()
    candidate["suppression_reason"] = "sport/market is outside the focused prediction lanes"

    published, review, suppressed, _ = daily_scan.run_committee_pipeline(
        published=[],
        review=[],
        suppressed=[candidate],
    )

    assert published == []
    assert review == []
    assert len(suppressed) == 1
    assert suppressed[0]["decision_status"] == DECISION_NO_BET
    assert "focused prediction lanes" in suppressed[0]["suppression_reason"]


def test_committee_pipeline_surfaces_wait_hold_and_avoid_states() -> None:
    wait_candidate = _committee_candidate(commence_delta_hours=0.75)
    wait_candidate["scraped_context"]["home_lineup_confirmed"] = 0
    wait_candidate["scraped_context"]["away_lineup_confirmed"] = 0
    wait_candidate["scraped_context"]["home_likely_starters_count"] = 0
    wait_candidate["scraped_context"]["away_likely_starters_count"] = 0

    hold_candidate = _committee_candidate()
    hold_candidate["stale_line"] = True

    avoid_candidate = _committee_candidate()
    avoid_candidate["context_referee_decision"] = "VETO"
    avoid_candidate["context_referee_reason"] = "Critical negative team news."

    published, review, suppressed, _ = daily_scan.run_committee_pipeline(
        published=[wait_candidate, hold_candidate, avoid_candidate],
        review=[],
        suppressed=[],
    )

    assert published == []
    assert {bet["committee_final_decision"] for bet in review} == {"WAIT_FOR_LINEUPS", "HOLD"}
    assert len(suppressed) == 1
    assert suppressed[0]["committee_final_decision"] == "AVOID"
    assert review[0]["decision_status"] in {DECISION_WAIT_FOR_LINEUPS, DECISION_HOLD}


def test_committee_parlay_gate_rejects_non_bet_and_accepts_explicit_substitute(monkeypatch) -> None:
    monkeypatch.setattr(daily_scan, "committee_required_for_parlays", lambda: True)

    blocked = daily_scan._committee_to_parlay_leg({"committee_final_decision": "HOLD"})
    substitute = daily_scan._committee_to_parlay_leg(
        {
            **_committee_candidate(),
            "team": "Alpha FC or Draw",
            "market": "double_chance",
            "committee_final_decision": "BET_SUBSTITUTE",
            "published_from_substitute": True,
        }
    )

    assert blocked is None
    assert substitute is not None
    assert substitute.team == "Alpha FC or Draw"
    assert substitute.market == "double_chance"


def test_write_report_includes_committee_sections(tmp_path, monkeypatch) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    monkeypatch.setattr(daily_scan, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(daily_scan, "_soccer_full_games", [])
    monkeypatch.setattr(daily_scan, "_other_sport_games", [])
    monkeypatch.setattr(daily_scan, "_scan_runtime_notes", [])
    monkeypatch.setattr(daily_scan, "show_committee_details", lambda: False)

    published, _, _, _ = daily_scan.run_committee_pipeline(
        published=[_committee_candidate()],
        review=[],
        suppressed=[],
    )
    report_path = daily_scan.write_report(
        published,
        review_bets=[],
        suppressed_bets=[],
        bankroll=1000.0,
    )

    text = report_path.read_text()
    assert "Research Mind:" in text
    assert "Model Mind:" in text
    assert "Arbiter:" in text
    assert "BET" in text
    assert "Evidence status:" in text
    assert "Concrete info score:" in text
    assert "Source count:" in text
    assert "Lineup:" in text
    assert "Injury:" in text
    assert "Motivation:" in text
    assert "Rotation:" in text


def test_write_report_preserves_existing_meaningful_summary_when_new_run_is_empty(tmp_path, monkeypatch) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    monkeypatch.setattr(daily_scan, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(daily_scan, "_soccer_full_games", [])
    monkeypatch.setattr(daily_scan, "_other_sport_games", [])
    monkeypatch.setattr(daily_scan, "_scan_runtime_notes", [])
    monkeypatch.setattr(daily_scan, "show_committee_details", lambda: False)

    today = daily_scan.TODAY
    report_path = reports_dir / f"value_bets_{today}.md"
    summary_path = reports_dir / f"summary_{today}.json"
    report_path.write_text("KEEP ME")
    summary_path.write_text(json.dumps({
        "date": today,
        "single_bets": {
            "total": 0,
            "review_total": 2,
            "suppressed_total": 0,
            "bets": [],
            "review_bets": [{"team": "Held Pick"}],
            "suppressed_bets": [],
        },
        "sport_pipeline_diagnostics": {
            "totals": {"scanned_games": 3},
            "by_sport": {"soccer": {"scanned_games": 3}},
        },
    }))

    out = daily_scan.write_report(
        [],
        review_bets=[],
        suppressed_bets=[],
        bankroll=1000.0,
    )

    assert out == report_path
    assert report_path.read_text() == "KEEP ME"
    preserved = json.loads(summary_path.read_text())
    assert preserved["single_bets"]["review_total"] == 2


def test_soccer_market_clamp_tightens_synthetic_or_shallow_history() -> None:
    probs, meta = daily_scan._soccer_market_clamp_three_way(
        (0.68, 0.12, 0.20),
        (0.42, 0.28, 0.30),
        home_rows=0,
        away_rows=4,
        home_synthetic=True,
        away_synthetic=False,
    )

    assert meta["gap"] <= 0.055
    assert probs[0] < 0.50
    assert round(sum(probs), 6) == 1.0


def test_write_report_includes_bankroll_blocked_section(tmp_path, monkeypatch) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    monkeypatch.setattr(daily_scan, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(daily_scan, "_soccer_full_games", [])
    monkeypatch.setattr(daily_scan, "_other_sport_games", [])
    monkeypatch.setattr(daily_scan, "_scan_runtime_notes", [])
    monkeypatch.setattr(daily_scan, "show_committee_details", lambda: False)

    blocked = [{
        "sport": "soccer",
        "market": "moneyline",
        "team": "Alpha FC",
        "home": "Alpha FC",
        "away": "Beta FC",
        "decision_status": "NO BET",
        "decision_reason": "Max concurrent bets (1) reached",
        "suppression_reason": "Max concurrent bets (1) reached",
        "market_priority_score": 1.0,
        "edge": 0.04,
    }]

    report_path = daily_scan.write_report(
        [],
        review_bets=[],
        suppressed_bets=[],
        bankroll_blocked_bets=blocked,
        bankroll=1000.0,
    )

    text = report_path.read_text()
    summary = json.loads((reports_dir / f"summary_{daily_scan.TODAY}.json").read_text())
    assert "Blocked By Bankroll" in text
    assert "Max concurrent bets (1) reached" in text
    assert summary["single_bets"]["bankroll_blocked_total"] == 1


def test_build_value_bet_carries_scraper_context_payload(monkeypatch) -> None:
    class _Estimate:
        adjusted_prob = 0.68
        base_prob = 0.66
        factors = []
        adjustments = []

        @staticmethod
        def to_dict():
            return {"adjusted_prob": 0.68, "base_prob": 0.66}

    class _Pricing:
        edge = 0.054
        market_prob = 0.6452
        fair_odds = 1.471

    monkeypatch.setattr(daily_scan, "_edge_outlier_decision", lambda **_kwargs: ("ok", ""))
    monkeypatch.setattr(daily_scan, "_sanity_check", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(daily_scan, "estimate_true_probability", lambda **_kwargs: _Estimate())
    monkeypatch.setattr(daily_scan, "build_pricing_decision", lambda **_kwargs: _Pricing())
    bet = daily_scan.build_value_bet(
        team="Atletico Madrid",
        ml_prob=0.68,
        team_odds=1.55,
        fair_prob=0.68,
        min_edge=0.03,
        sport="soccer",
        market="moneyline",
        raw_model_prob=0.66,
        prediction_factors=[],
        context_adjustments=[],
        availability_context={
            "availability_source": "api_football",
            "home_team_name": "Elche CF",
            "away_team_name": "Atletico Madrid",
            "away_lineup_confirmed": 1,
            "away_likely_starters_count": 11,
            "away_priority_absences_count": 0,
        },
    )

    assert bet is not None
    assert bet["scraped_context"]["availability_source"] == "api_football"
    assert bet["scraped_context"]["away_lineup_confirmed"] == 1
    assert any("Away XI posted" in item for item in bet["scraped_context_highlights"])
    assert "api_football" in bet["scraped_context_sources"]


def test_context_referee_packet_includes_scraper_context() -> None:
    packet = daily_scan._context_referee_packet(
        {
            "sport": "mlb",
            "market": "moneyline",
            "team": "New York Yankees",
            "home": "New York Yankees",
            "away": "Boston Red Sox",
            "odds": 1.91,
            "minimum_acceptable_odds": 1.84,
            "odds_recheck_status": "passed",
            "edge": 0.051,
            "fair_prob": 0.57,
            "market_implied_prob": 0.5236,
            "fair_odds": 1.754,
            "true_probability": {"adjusted_prob": 0.57},
            "prediction_factors": [{"name": "starter_form", "summary": "Starter edge"}],
            "context_adjustments": [{"name": "starter_uncertainty", "summary": "Starter not fully confirmed"}],
            "scraped_context": {"availability_source": "mlb_stats_api", "home_starter_name": "Gerrit Cole"},
            "scraped_context_highlights": ["Home starter confirmed: Gerrit Cole"],
            "scraped_context_sources": ["mlb_stats_api"],
        }
    )

    assert packet["candidate"]["true_probability"]["adjusted_prob"] == 0.57
    assert packet["scraper_context"]["availability"]["home_starter_name"] == "Gerrit Cole"
    assert packet["scraper_context"]["sources"] == ["mlb_stats_api"]
    assert "Home starter confirmed" in packet["analyst_report"]["warnings"][0]


def test_review_edge_threshold_has_mlb_specific_overrides() -> None:
    assert daily_scan._review_edge_threshold("mlb", "moneyline") == 0.13
    assert daily_scan._review_edge_threshold("mlb", "spreads") == 0.125
    assert daily_scan._review_edge_threshold("mlb", "totals") == 0.07


def test_review_edge_threshold_has_nhl_moneyline_override() -> None:
    assert daily_scan._review_edge_threshold("nhl", "moneyline") == 0.14
    assert daily_scan._review_edge_threshold("nhl", "spreads") == 0.10


def test_review_edge_threshold_has_basketball_moneyline_override() -> None:
    assert daily_scan._review_edge_threshold("basketball", "moneyline") == 0.085
    assert daily_scan._review_edge_threshold("basketball", "totals") == 0.10
