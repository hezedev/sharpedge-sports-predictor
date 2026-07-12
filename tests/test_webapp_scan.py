from __future__ import annotations

import io
import json
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd

import webapp.app as webapp_app
import daily_scan
from src.utils import results_tracker


class _ImmediateThread:
    def __init__(self, target=None, daemon=None):
        self._target = target

    def start(self):
        if self._target:
            self._target()


class _FakeProc:
    def __init__(self, cmd, **kwargs):
        self.cmd = cmd
        self.stdout = io.StringIO("")
        self.returncode = 0

    def wait(self):
        return self.returncode


class _FakeHTTPResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_scan_start_forwards_offline_odds(monkeypatch):
    captured = {}

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeProc(cmd, **kwargs)

    monkeypatch.setattr(webapp_app.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(webapp_app.subprocess, "Popen", _fake_popen)
    webapp_app._scan_running = False
    webapp_app._scan_proc = None
    webapp_app._scan_log = []

    client = webapp_app.app.test_client()
    response = client.post(
        "/api/scan/start",
        json={
            "sport": "tennis",
            "market": "all",
            "retrain": False,
            "offline_odds": True,
        },
    )

    assert response.status_code == 200
    assert "--offline-odds" in captured["cmd"]


def test_scan_start_forwards_force_fresh_odds(monkeypatch):
    captured = {}

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeProc(cmd, **kwargs)

    monkeypatch.setattr(webapp_app.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(webapp_app.subprocess, "Popen", _fake_popen)
    webapp_app._scan_running = False
    webapp_app._scan_proc = None
    webapp_app._scan_log = []

    client = webapp_app.app.test_client()
    response = client.post(
        "/api/scan/start",
        json={
            "sport": "soccer",
            "market": "all",
            "retrain": False,
            "force_fresh_odds": True,
        },
    )

    assert response.status_code == 200
    assert "--force-fresh-odds" in captured["cmd"]


def test_scan_start_runs_multiple_targeted_sports_with_force_fresh(monkeypatch):
    captured = []

    def _fake_popen(cmd, **kwargs):
        captured.append(cmd)
        return _FakeProc(cmd, **kwargs)

    monkeypatch.setattr(webapp_app.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(webapp_app.subprocess, "Popen", _fake_popen)
    webapp_app._scan_running = False
    webapp_app._scan_proc = None
    webapp_app._scan_log = []

    client = webapp_app.app.test_client()
    response = client.post(
        "/api/scan/start",
        json={
            "sports": ["mlb", "basketball"],
            "market": "all",
            "force_fresh_odds": True,
        },
    )

    payload = response.get_json()
    assert response.status_code == 200
    assert payload["sports"] == ["mlb", "basketball"]
    assert len(captured) == 2
    assert ["--sport", "mlb"] == captured[0][captured[0].index("--sport"):captured[0].index("--sport") + 2]
    assert ["--sport", "basketball"] == captured[1][captured[1].index("--sport"):captured[1].index("--sport") + 2]
    assert all("--force-fresh-odds" in cmd for cmd in captured)
    assert "--focused-lanes" not in captured[0]
    assert "--focused-lanes" not in captured[1]


def test_scan_start_blocks_broad_force_fresh_by_default(monkeypatch):
    captured = {}

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeProc(cmd, **kwargs)

    monkeypatch.setattr(webapp_app.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(webapp_app.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(
        webapp_app,
        "_odds_key_pool_summary",
        lambda: {"tracked_unavailable_count": 2, "runtime_loaded_count": 1},
    )
    webapp_app._scan_running = False
    webapp_app._scan_proc = None
    webapp_app._scan_log = []

    client = webapp_app.app.test_client()
    response = client.post(
        "/api/scan/start",
        json={
            "sport": "all",
            "market": "all",
            "force_fresh_odds": True,
            "focused_lanes": True,
        },
    )

    payload = response.get_json()
    assert response.status_code == 200
    assert "--force-fresh-odds" not in captured["cmd"]
    assert any("force-fresh odds disabled" in note for note in payload["safety_notes"])
    assert any("not loaded at runtime" in line for line in webapp_app._scan_log)


def test_scan_start_all_defaults_to_full_sport_scope(monkeypatch):
    captured = {}

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeProc(cmd, **kwargs)

    monkeypatch.setattr(webapp_app.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(webapp_app.subprocess, "Popen", _fake_popen)
    webapp_app._scan_running = False
    webapp_app._scan_proc = None
    webapp_app._scan_log = []

    client = webapp_app.app.test_client()
    response = client.post(
        "/api/scan/start",
        json={
            "sport": "all",
            "market": "all",
        },
    )

    assert response.status_code == 200
    assert ["--sport", "all"] == captured["cmd"][captured["cmd"].index("--sport"):captured["cmd"].index("--sport") + 2]
    assert "--focused-lanes" not in captured["cmd"]


def test_scan_start_allows_explicit_broad_force_fresh_override(monkeypatch):
    captured = {}

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeProc(cmd, **kwargs)

    monkeypatch.setattr(webapp_app.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(webapp_app.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(
        webapp_app,
        "_odds_key_pool_summary",
        lambda: {"tracked_unavailable_count": 0, "runtime_loaded_count": 2},
    )
    webapp_app._scan_running = False
    webapp_app._scan_proc = None
    webapp_app._scan_log = []

    client = webapp_app.app.test_client()
    response = client.post(
        "/api/scan/start",
        json={
            "sport": "all",
            "market": "all",
            "force_fresh_odds": True,
            "allow_broad_force_fresh": True,
            "focused_lanes": True,
        },
    )

    assert response.status_code == 200
    assert "--force-fresh-odds" in captured["cmd"]


def test_scan_start_forwards_lean_context(monkeypatch):
    captured = {}

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeProc(cmd, **kwargs)

    monkeypatch.setattr(webapp_app.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(webapp_app.subprocess, "Popen", _fake_popen)
    webapp_app._scan_running = False
    webapp_app._scan_proc = None
    webapp_app._scan_log = []

    client = webapp_app.app.test_client()
    response = client.post(
        "/api/scan/start",
        json={
            "sport": "soccer",
            "market": "all",
            "retrain": False,
            "lean_context": True,
        },
    )

    assert response.status_code == 200
    assert "--lean-context" in captured["cmd"]


def test_api_parlay_calculate_returns_risk_verdict_and_weakest_leg() -> None:
    client = webapp_app.app.test_client()
    response = client.post(
        "/api/parlay/calculate",
        json={
            "tier": "value",
            "legs": [
                {"team": "A", "match": "A vs B", "sport": "soccer", "market": "moneyline", "odds": 1.9, "ml_prob": 0.61, "edge": 0.06},
                {"team": "C", "match": "C vs D", "sport": "soccer", "market": "moneyline", "odds": 1.95, "ml_prob": 0.58, "edge": 0.05},
                {"team": "E", "match": "E vs F", "sport": "soccer", "market": "moneyline", "odds": 2.05, "ml_prob": 0.53, "edge": 0.04},
            ],
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["build_verdict"] == "BUILD"
    assert payload["combined_probability"] > 0
    assert payload["weakest_leg"]["team"] == "E"
    assert payload["risk_tier"] in {"medium-risk", "high-risk"}


def test_api_results_exposes_mistake_reports(tmp_path, monkeypatch) -> None:
    tracker_dir = tmp_path / "data" / "tracker"
    tracker_dir.mkdir(parents=True)
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()

    monkeypatch.setattr(webapp_app, "BASE", tmp_path)
    monkeypatch.setattr(webapp_app, "_MANUAL_PARLAYS_FILE", tracker_dir / "manual_parlays.json")
    monkeypatch.setattr(results_tracker, "_TRACKER_DIR", tracker_dir)
    monkeypatch.setattr(results_tracker, "_PRED_FILE", tracker_dir / "predictions.parquet")
    monkeypatch.setattr(results_tracker, "_SETTLED_FILE", tracker_dir / "settled.parquet")
    monkeypatch.setattr(results_tracker, "_SUMMARY_FILE", tracker_dir / "summary.parquet")
    monkeypatch.setattr(results_tracker, "_PARLAY_FILE", tracker_dir / "parlays.parquet")
    (tracker_dir / "manual_parlays.json").write_text("[]")
    pd.DataFrame([], columns=["pred_id"]).to_parquet(tracker_dir / "predictions.parquet", index=False)
    pd.DataFrame([
        {
            "pred_id": "x1",
            "settled_at": "2026-05-05T20:00:00Z",
            "sport": "soccer",
            "match_id": "A vs B",
            "team_or_player": "A",
            "commence_time": "2026-05-05T18:00:00Z",
            "recorded_at": "2026-05-05T10:00:00Z",
            "market": "moneyline",
            "market_status": "preferred",
            "tier": "Preferred",
            "bet_odds": 1.7,
            "bookmaker": "Book",
            "edge": 0.09,
            "ml_prob": 0.66,
            "fair_prob": 0.57,
            "stake_units": 1.0,
            "kelly_stake_pct": 0.02,
            "actual_result": "away_win",
            "won": False,
            "profit_units": -1.0,
            "closing_odds": 1.88,
            "clv": -0.0957,
            "status": "lost",
            "is_parlay_leg": False,
        }
    ]).to_parquet(tracker_dir / "settled.parquet", index=False)
    pd.DataFrame([], columns=["parlay_id"]).to_parquet(tracker_dir / "parlays.parquet", index=False)

    client = webapp_app.app.test_client()
    response = client.get("/api/results?date=2026-05-05")

    assert response.status_code == 200
    payload = response.get_json()
    assert "mistake_reports" in payload
    assert payload["mistake_reports"]["daily"]["categories"]["odds/value error"] == 1


def test_scan_start_forwards_context_referee(monkeypatch):
    captured = {}

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeProc(cmd, **kwargs)

    monkeypatch.setattr(webapp_app.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(webapp_app.subprocess, "Popen", _fake_popen)
    webapp_app._scan_running = False
    webapp_app._scan_proc = None
    webapp_app._scan_log = []

    client = webapp_app.app.test_client()
    response = client.post(
        "/api/scan/start",
        json={
            "sport": "soccer",
            "market": "all",
            "retrain": False,
            "context_referee": True,
        },
    )

    assert response.status_code == 200
    assert "--context-referee" in captured["cmd"]


def test_scan_start_forwards_full_soccer_scope(monkeypatch):
    captured = {}

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeProc(cmd, **kwargs)

    monkeypatch.setattr(webapp_app.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(webapp_app.subprocess, "Popen", _fake_popen)
    webapp_app._scan_running = False
    webapp_app._scan_proc = None
    webapp_app._scan_log = []

    client = webapp_app.app.test_client()
    response = client.post(
        "/api/scan/start",
        json={
            "sport": "soccer",
            "market": "all",
            "full_soccer_scope": True,
        },
    )

    assert response.status_code == 200
    assert "--full-soccer-scope" in captured["cmd"]


def test_scan_start_world_cup_chip_runs_scoped_soccer_scan(monkeypatch):
    captured = {}

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeProc(cmd, **kwargs)

    monkeypatch.setattr(webapp_app.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(webapp_app.subprocess, "Popen", _fake_popen)
    webapp_app._scan_running = False
    webapp_app._scan_proc = None
    webapp_app._scan_log = []

    client = webapp_app.app.test_client()
    response = client.post(
        "/api/scan/start",
        json={
            "sports": ["soccer_fifa_world_cup"],
            "market": "all",
            "force_fresh_odds": True,
        },
    )

    payload = response.get_json()
    assert response.status_code == 200
    assert payload["sports"] == ["soccer"]
    assert payload["soccer_leagues"] == ["soccer_fifa_world_cup"]
    assert ["--sport", "soccer"] == captured["cmd"][captured["cmd"].index("--sport"):captured["cmd"].index("--sport") + 2]
    assert ["--soccer-league", "soccer_fifa_world_cup"] == captured["cmd"][captured["cmd"].index("--soccer-league"):captured["cmd"].index("--soccer-league") + 2]
    assert "--force-fresh-odds" in captured["cmd"]


def test_api_picks_returns_market_policy(tmp_path, monkeypatch):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    summary_path = reports_dir / f"summary_{today}.json"
    summary_path.write_text(json.dumps({
        "scan_notes": [
            {
                "type": "deferred_leagues",
                "sport": "soccer",
                "count": 2,
                "reason": "low-value review-only leagues were deferred to conserve live quota and improve scan speed",
                "leagues": ["Liga MX", "Allsvenskan"],
            }
        ],
        "single_bets": {
            "bets": [
                {
                    "sport": "tennis",
                    "team": "Player A",
                    "home": "Player A",
                    "away": "Player B",
                    "market": "moneyline",
                    "edge": 0.08,
                    "odds": 1.9,
                    "window": "today",
                }
            ],
            "review_bets": [
                {
                    "sport": "soccer",
                    "team": "Team C or Draw",
                    "home": "Team D",
                    "away": "Team C",
                    "market": "double_chance",
                    "edge": 0.05,
                    "odds": 1.7,
                    "window": "today",
                }
            ],
        }
    }))

    monkeypatch.setattr(webapp_app, "BASE", tmp_path)

    client = webapp_app.app.test_client()
    response = client.get("/api/picks")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["bets"][0]["market_status"] == "preferred"
    assert payload["review_bets"][0]["market"] == "double_chance"
    assert payload["scan_notes"][0]["type"] == "deferred_leagues"
    assert "market_policy" in payload


def test_api_picks_reclassifies_review_only_leagues(tmp_path, monkeypatch):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / f"summary_{today}.json").write_text(json.dumps({
        "single_bets": {
            "bets": [
                {
                    "sport": "soccer",
                    "team": "Seattle Sounders",
                    "home": "Seattle Sounders",
                    "away": "LA Galaxy",
                    "league": "MLS",
                    "league_key": "soccer_usa_mls",
                    "market": "moneyline",
                    "edge": 0.08,
                    "odds": 1.9,
                    "window": "today",
                }
            ],
            "review_bets": [],
        }
    }))

    monkeypatch.setattr(webapp_app, "BASE", tmp_path)

    client = webapp_app.app.test_client()
    response = client.get("/api/picks")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["bets"] == []
    assert payload["review_total"] == 1
    assert payload["review_bets"][0]["launch_label"] == "Review"
    assert payload["review_bets"][0]["publishable"] is False
    assert payload["review_bets"][0]["decision_status"] == "HOLD"
    assert payload["review_bets"][0]["decision_reason"]


def test_api_picks_exposes_sport_scan_counts(tmp_path, monkeypatch):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / f"summary_{today}.json").write_text(json.dumps({
        "soccer_games": [
            {"home": "A", "away": "B"},
            {"home": "C", "away": "D"},
        ],
        "other_games": [
            {"sport": "mlb", "home": "Yankees", "away": "Red Sox"},
            {"sport": "tennis_wta", "home": "Player A", "away": "Player B"},
        ],
        "single_bets": {"bets": [], "review_bets": []},
    }))

    monkeypatch.setattr(webapp_app, "BASE", tmp_path)

    client = webapp_app.app.test_client()
    response = client.get("/api/picks")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["sport_scan_counts"]["soccer"] == 2
    assert payload["sport_scan_counts"]["mlb"] == 1
    assert payload["sport_scan_counts"]["tennis_wta"] == 1
    assert any(note.get("type") == "sport_scan_counts" for note in payload["scan_notes"])


def test_api_picks_accepts_multiple_sport_filters(tmp_path, monkeypatch):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / f"summary_{today}.json").write_text(json.dumps({
        "single_bets": {
            "bets": [
                {"sport": "mlb", "team": "Yankees", "market": "moneyline", "edge": 0.06, "odds": 1.9},
                {"sport": "nhl", "team": "Rangers", "market": "moneyline", "edge": 0.05, "odds": 2.0},
                {"sport": "soccer", "team": "Arsenal", "market": "moneyline", "edge": 0.04, "odds": 1.8},
            ],
            "review_bets": [
                {"sport": "basketball", "team": "Knicks", "market": "moneyline", "edge": 0.03, "odds": 1.95},
                {"sport": "mlb", "team": "Dodgers", "market": "moneyline", "edge": 0.03, "odds": 1.85},
            ],
        }
    }))

    monkeypatch.setattr(webapp_app, "BASE", tmp_path)

    client = webapp_app.app.test_client()
    response = client.get("/api/picks?sport=mlb&sport=nhl")

    assert response.status_code == 200
    payload = response.get_json()
    assert {bet["sport"] for bet in payload["bets"]} == {"mlb", "nhl"}
    assert {bet["sport"] for bet in payload["review_bets"]} == {"mlb"}


def test_api_games_accepts_multiple_sport_filters(tmp_path, monkeypatch):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / f"summary_{today}.json").write_text(json.dumps({
        "soccer_games": [{"home": "A", "away": "B", "outcomes": []}],
        "other_games": [
            {"sport": "mlb", "home": "Yankees", "away": "Red Sox"},
            {"sport": "nhl", "home": "Rangers", "away": "Bruins"},
            {"sport": "basketball", "home": "Knicks", "away": "Celtics"},
        ],
        "single_bets": {"bets": [], "review_bets": []},
    }))

    monkeypatch.setattr(webapp_app, "BASE", tmp_path)

    client = webapp_app.app.test_client()
    response = client.get("/api/games?sport=mlb&sport=nhl")

    assert response.status_code == 200
    payload = response.get_json()
    assert {game["sport"] for game in payload["games"]} == {"mlb", "nhl"}


def test_api_games_explains_why_production_game_is_not_on_pick_board(tmp_path, monkeypatch):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / f"summary_{today}.json").write_text(json.dumps({
        "other_games": [
            {
                "sport": "mlb",
                "home": "Yankees",
                "away": "Red Sox",
                "league": "MLB",
                "model_available": True,
                "model_pick": "Yankees",
                "home_odds": 2.02,
                "away_odds": 1.90,
                "mlb_probability_debug": {
                    "final_probs": {"home": 0.53, "away": 0.47},
                    "market_probs": {"home": 0.50, "away": 0.50},
                },
            }
        ],
        "single_bets": {
            "bets": [],
            "review_bets": [
                {
                    "sport": "mlb",
                    "home": "Yankees",
                    "away": "Red Sox",
                    "team": "Yankees",
                    "market": "moneyline",
                    "edge": 0.03,
                    "ml_prob": 0.53,
                    "fair_prob": 0.50,
                    "review_reason": "availability or starter uncertainty still needs human review",
                    "research_mind_missing_evidence": ["confirmed lineup"],
                }
            ],
            "suppressed_bets": [],
        },
    }))

    monkeypatch.setattr(webapp_app, "BASE", tmp_path)

    client = webapp_app.app.test_client()
    response = client.get("/api/games?sport=mlb")

    assert response.status_code == 200
    game = response.get_json()["games"][0]
    assert game["board_status"] == "review"
    assert "availability" in game["board_reason"]
    assert "confirmed lineup" in game["missing_to_promote"]
    assert game["best_candidate"]["team"] == "Yankees"


def test_api_games_marks_passed_production_game_with_best_edge_gap(tmp_path, monkeypatch):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / f"summary_{today}.json").write_text(json.dumps({
        "other_games": [
            {
                "sport": "mlb",
                "home": "Yankees",
                "away": "Red Sox",
                "league": "MLB",
                "model_available": True,
                "model_pick": "Yankees",
                "home_odds": 2.02,
                "away_odds": 1.90,
                "mlb_probability_debug": {
                    "final_probs": {"home": 0.515, "away": 0.485},
                    "market_probs": {"home": 0.50, "away": 0.50},
                },
            }
        ],
        "single_bets": {"bets": [], "review_bets": [], "suppressed_bets": []},
    }))

    monkeypatch.setattr(webapp_app, "BASE", tmp_path)

    client = webapp_app.app.test_client()
    response = client.get("/api/games?sport=mlb")

    assert response.status_code == 200
    game = response.get_json()["games"][0]
    assert game["board_status"] == "no_candidate"
    assert "below the 3.0pp candidate threshold" in game["board_reason"]
    assert "edge/EV above threshold after no-vig market comparison" in game["missing_to_promote"]


def test_api_adds_odds_api_keys_to_existing_pool(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("ODDS_API_KEY=existing-key-11111111\nODDS_API_KEYS=existing-key-11111111\n")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "odds_key_pool.json").write_text("{}")
    monkeypatch.setattr(webapp_app, "BASE", tmp_path)
    monkeypatch.setenv("ODDS_API_KEY", "existing-key-11111111")
    monkeypatch.setenv("ODDS_API_KEYS", "existing-key-11111111")

    client = webapp_app.app.test_client()
    response = client.post(
        "/api/apis/odds-keys/add",
        json={"value": "new-key-22222222\nbad key"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["added_count"] == 1
    assert "22222222" in payload["added_fingerprints"]
    assert "existing-key-11111111,new-key-22222222" in env_path.read_text()
    assert any(item["reason"] == "invalid_format" for item in payload["excluded"])


def test_api_prunes_exhausted_runtime_odds_keys(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("ODDS_API_KEY=empty-key-11111111\nODDS_API_KEYS=empty-key-11111111,backup-key-22222222\n")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "odds_key_pool.json").write_text(json.dumps({
        "11111111": {"fingerprint": "11111111", "remaining": 0},
        "22222222": {"fingerprint": "22222222", "remaining": 120},
        "_meta": {
            "last_selected_fingerprint": "11111111",
            "runtime_loaded_fingerprints": ["11111111", "22222222"],
            "usable_fingerprints": ["11111111", "22222222"],
        },
    }))
    monkeypatch.setattr(webapp_app, "BASE", tmp_path)
    monkeypatch.setenv("ODDS_API_KEY", "empty-key-11111111")
    monkeypatch.setenv("ODDS_API_KEYS", "empty-key-11111111,backup-key-22222222")

    client = webapp_app.app.test_client()
    response = client.post("/api/apis/odds-keys/prune-exhausted", json={})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["removed_fingerprints"] == ["11111111"]
    text = env_path.read_text()
    assert "empty-key-11111111" not in text
    assert "backup-key-22222222" in text


def test_api_picks_exposes_sport_funnel_diagnostics(tmp_path, monkeypatch):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / f"summary_{today}.json").write_text(json.dumps({
        "single_bets": {"bets": [], "review_bets": []},
        "sport_pipeline_diagnostics": {
            "by_sport": {
                "mlb": {
                    "scanned_games": 12,
                    "model_available_games": 11,
                    "abstained_games": 1,
                    "candidate_games": 4,
                    "published_games": 1,
                    "review_games": 2,
                    "suppressed_games": 1,
                    "no_candidate_games": 8,
                    "no_candidate_reason_breakdown": {"no_edge": 6, "insufficient_history_or_model": 2},
                    "review_reason_breakdown": {"availability_risk": 2},
                    "suppression_reason_breakdown": {"market_fit": 1},
                }
            },
            "totals": {
                "scanned_games": 12,
                "published_games": 1,
            },
        },
    }))

    monkeypatch.setattr(webapp_app, "BASE", tmp_path)

    client = webapp_app.app.test_client()
    response = client.get("/api/picks")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["sport_funnel"]["mlb"]["scanned_games"] == 12
    assert payload["sport_funnel"]["mlb"]["review_games"] == 2
    assert payload["sport_funnel"]["mlb"]["no_candidate_games"] == 8
    assert payload["sport_funnel"]["mlb"]["no_candidate_reason_breakdown"]["no_edge"] == 6
    assert any(note.get("type") == "sport_funnel" for note in payload["scan_notes"])


def test_api_picks_today_filter_keeps_played_items_visible(tmp_path, monkeypatch):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / f"summary_{today}.json").write_text(json.dumps({
        "single_bets": {
            "bets": [
                {
                    "sport": "mlb",
                    "team": "Detroit Tigers",
                    "home": "Detroit Tigers",
                    "away": "Texas Rangers",
                    "market": "moneyline",
                    "edge": 0.08,
                    "odds": 1.9,
                    "window": "today",
                    "commence": "2026-04-29T18:00:00Z",
                }
            ],
            "review_bets": [],
        }
    }))

    monkeypatch.setattr(webapp_app, "BASE", tmp_path)

    client = webapp_app.app.test_client()
    response = client.get("/api/picks?window=today")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["total"] == 1
    assert payload["bets"][0]["team"] == "Detroit Tigers"
    assert payload["bets"][0]["status"] == "played"
    assert payload["bets"][0]["window"] == "past"


def test_api_dashboard_returns_process_summary(tmp_path, monkeypatch):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / f"summary_{today}.json").write_text(json.dumps({"single_bets": {"total": 2, "by_sport": {"soccer": 2}}}))

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "api_usage.json").write_text(json.dumps({"odds_remaining": 400, "odds_remaining_start": 500, "odds_requests_used_total": 10, "odds_requests_used_today": 3}))

    settled = pd.DataFrame(
        [
            {"sport": "soccer", "status": "won", "clv": 0.04, "edge": 0.08, "profit_units": 0.9, "stake_units": 1.0},
            {"sport": "soccer", "status": "lost", "clv": -0.01, "edge": 0.05, "profit_units": -1.0, "stake_units": 1.0},
        ]
    )

    monkeypatch.setattr(webapp_app, "BASE", tmp_path)
    monkeypatch.setattr(webapp_app, "compute_summary", lambda: {"n_bets": 2, "avg_clv": 0.015, "clv_positive_pct": 0.5})
    monkeypatch.setattr(webapp_app, "get_settled", lambda: settled)
    monkeypatch.setattr(webapp_app, "quota_bridge", None)

    client = webapp_app.app.test_client()
    response = client.get("/api/dashboard")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["process_summary"]["avg_clv"] == 0.015
    assert payload["process_by_sport"]["soccer"]["bets"] == 2
    assert payload["process_by_sport"]["soccer"]["clv_positive_pct"] == 0.5


def test_api_dashboard_includes_odds_key_pool_summary(tmp_path, monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    monkeypatch.delenv("ODDS_API_KEYS", raising=False)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / f"summary_{today}.json").write_text(json.dumps({"single_bets": {"total": 0, "by_sport": {}}}))

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "api_usage.json").write_text(json.dumps({"odds_remaining": 420, "odds_remaining_start": 500, "odds_requests_used_total": 12, "odds_requests_used_today": 4}))
    (data_dir / "odds_key_pool.json").write_text(json.dumps({
        "_meta": {
            "last_selected_fingerprint": "bbbb2222",
            "last_selected_at": "2026-04-30T08:00:00+00:00",
            "last_selected_reason": "selected pooled key with the highest tracked remaining quota",
            "tracked_fingerprints": ["aaaa1111", "bbbb2222"],
            "runtime_loaded_fingerprints": ["bbbb2222"],
            "usable_fingerprints": ["bbbb2222"],
        },
        "aaaa1111": {
            "fingerprint": "aaaa1111",
            "remaining": 340,
            "updated_at": "2026-04-30T07:55:00+00:00",
        },
        "bbbb2222": {
            "fingerprint": "bbbb2222",
            "remaining": 480,
            "updated_at": "2026-04-30T08:00:00+00:00",
        },
    }))

    monkeypatch.setattr(webapp_app, "BASE", tmp_path)
    monkeypatch.setattr(webapp_app, "compute_summary", lambda: {})
    monkeypatch.setattr(webapp_app, "get_settled", lambda: pd.DataFrame())
    monkeypatch.setattr(webapp_app, "quota_bridge", None)

    client = webapp_app.app.test_client()
    response = client.get("/api/dashboard")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["odds_key_pool"]["enabled"] is True
    assert payload["odds_key_pool"]["count"] == 2
    assert payload["odds_key_pool"]["total_remaining"] == 820
    assert payload["odds_key_pool"]["last_selected_fingerprint"] == "bbbb2222"
    assert payload["odds_key_pool"]["last_selected_reason"] == "selected pooled key with the highest tracked remaining quota"
    assert payload["odds_key_pool"]["keys"][0]["fingerprint"] == "bbbb2222"
    assert payload["odds_key_pool"]["keys"][0]["selected"] is True
    assert payload["odds_key_pool"]["active_remaining"] == 480
    assert payload["odds_key_pool"]["runtime_loaded_count"] == 1
    assert payload["odds_key_pool"]["usable_count"] == 1
    assert payload["odds_key_pool"]["keys"][1]["status"] == "raw_key_missing"


def test_api_dashboard_prefers_selected_pool_key_for_health_display(tmp_path, monkeypatch):
    monkeypatch.delenv("ODDS_API_KEYS", raising=False)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / f"summary_{today}.json").write_text(json.dumps({"single_bets": {"total": 0, "by_sport": {}}}))

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "api_usage.json").write_text(json.dumps({
        "key_fingerprint": "cccc3333",
        "odds_remaining": 14,
        "odds_remaining_start": 500,
        "odds_requests_used_total": 486,
        "odds_requests_used_today": 8,
    }))
    (data_dir / "odds_key_pool.json").write_text(json.dumps({
        "_meta": {
            "last_selected_fingerprint": "bbbb2222",
            "last_selected_at": "2026-04-30T08:00:00+00:00",
            "last_selected_reason": "selected highest tracked remaining healthy key above 50 while avoiding lower-quota pool keys",
            "low_remaining_threshold": 50,
            "tracked_fingerprints": ["bbbb2222", "aaaa1111"],
            "runtime_loaded_fingerprints": ["bbbb2222"],
            "usable_fingerprints": ["bbbb2222"],
        },
        "bbbb2222": {
            "fingerprint": "bbbb2222",
            "remaining": 26,
            "updated_at": "2026-04-30T08:00:00+00:00",
        },
        "aaaa1111": {
            "fingerprint": "aaaa1111",
            "remaining": 332,
            "updated_at": "2026-04-30T07:55:00+00:00",
        },
    }))

    monkeypatch.setattr(webapp_app, "BASE", tmp_path)
    monkeypatch.setattr(webapp_app, "compute_summary", lambda: {})
    monkeypatch.setattr(webapp_app, "get_settled", lambda: pd.DataFrame())
    monkeypatch.setattr(webapp_app, "quota_bridge", None)
    monkeypatch.setenv("ODDS_API_KEY", "env-key-cccc3333")

    client = webapp_app.app.test_client()
    response = client.get("/api/dashboard")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["odds_remaining"] == 26
    assert payload["odds_display_key_fingerprint"] == "bbbb2222"
    assert payload["odds_usage_key_fingerprint"] == "cccc3333"
    assert payload["odds_display_source"] == "key_pool_selected"
    assert payload["odds_usage_sync_status"] == "pool_selected_differs_from_webapp_usage_file"
    assert payload["odds_key_pool"]["active_remaining"] == 26


def test_api_dashboard_distinguishes_tracked_runtime_usable_and_stale_keys(tmp_path, monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    monkeypatch.delenv("ODDS_API_KEYS", raising=False)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / f"summary_{today}.json").write_text(json.dumps({"single_bets": {"total": 0, "by_sport": {}}}))

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "api_usage.json").write_text(json.dumps({"key_fingerprint": "bbbb2222", "odds_remaining": 26, "odds_remaining_start": 500}))
    (data_dir / "odds_key_pool.json").write_text(json.dumps({
        "_meta": {
            "last_selected_fingerprint": "bbbb2222",
            "last_selected_at": "2026-04-30T08:00:00+00:00",
            "last_selected_reason": "selected highest remaining runtime-available key; higher tracked keys were excluded with explicit reasons",
            "tracked_fingerprints": ["aaaa1111", "bbbb2222", "cccc3333", "dddd4444", "eeee5555"],
            "runtime_loaded_fingerprints": ["bbbb2222", "cccc3333", "dddd4444", "eeee5555"],
            "usable_fingerprints": ["bbbb2222", "cccc3333", "dddd4444", "eeee5555"],
            "excluded_fingerprints": ["aaaa1111"],
            "low_remaining_threshold": 50,
            "metadata_stale_threshold_hours": 24.0,
        },
        "aaaa1111": {"fingerprint": "aaaa1111", "remaining": 332, "updated_at": "2026-04-30T07:55:00+00:00"},
        "bbbb2222": {"fingerprint": "bbbb2222", "remaining": 26, "updated_at": "2026-04-30T08:00:00+00:00"},
        "cccc3333": {"fingerprint": "cccc3333", "remaining": 120, "updated_at": "2026-04-30T08:00:00+00:00"},
        "dddd4444": {"fingerprint": "dddd4444", "remaining": 14, "updated_at": "2026-04-30T08:00:00+00:00"},
        "eeee5555": {"fingerprint": "eeee5555", "remaining": 8, "updated_at": "2026-05-06T11:00:00+00:00"},
    }))

    monkeypatch.setattr(webapp_app, "BASE", tmp_path)
    monkeypatch.setattr(webapp_app, "compute_summary", lambda: {})
    monkeypatch.setattr(webapp_app, "get_settled", lambda: pd.DataFrame())
    monkeypatch.setattr(webapp_app, "quota_bridge", None)

    client = webapp_app.app.test_client()
    response = client.get("/api/dashboard")

    assert response.status_code == 200
    payload = response.get_json()
    pool = payload["odds_key_pool"]
    assert pool["count"] == 5
    assert pool["runtime_loaded_count"] == 4
    assert pool["usable_count"] == 4
    by_fp = {row["fingerprint"]: row for row in pool["keys"]}
    assert by_fp["aaaa1111"]["status"] == "raw_key_missing"
    assert by_fp["bbbb2222"]["status"] == "low"
    assert by_fp["cccc3333"]["status"] == "stale_metadata"
    assert by_fp["dddd4444"]["status"] == "low"
    assert by_fp["eeee5555"]["status"] == "low"


def test_api_dashboard_shows_runtime_env_keys_even_when_pool_meta_is_stale(tmp_path, monkeypatch):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / f"summary_{today}.json").write_text(json.dumps({"single_bets": {"total": 0, "by_sport": {}}}))

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "api_usage.json").write_text(json.dumps({"odds_remaining": 420, "odds_remaining_start": 500}))
    (data_dir / "odds_key_pool.json").write_text(json.dumps({
        "_meta": {
            "tracked_fingerprints": ["aaaa1111"],
            "runtime_loaded_fingerprints": [],
            "usable_fingerprints": [],
        },
        "aaaa1111": {
            "fingerprint": "aaaa1111",
            "remaining": 340,
            "updated_at": "2026-04-30T07:55:00+00:00",
        },
    }))

    monkeypatch.setattr(webapp_app, "BASE", tmp_path)
    monkeypatch.setattr(webapp_app, "compute_summary", lambda: {})
    monkeypatch.setattr(webapp_app, "get_settled", lambda: pd.DataFrame())
    monkeypatch.setattr(webapp_app, "quota_bridge", None)
    monkeypatch.setenv("ODDS_API_KEY", "  'env-primary-11111111'  ")
    monkeypatch.setenv("ODDS_API_KEYS", ' "env-backup-22222222, env-third-33333333" ')

    client = webapp_app.app.test_client()
    response = client.get("/api/dashboard")

    assert response.status_code == 200
    pool = response.get_json()["odds_key_pool"]
    assert pool["runtime_loaded_fingerprints"] == ["11111111", "22222222", "33333333"]
    assert pool["runtime_loaded_count"] == 3
    assert pool["usable_count"] == 3
    by_fp = {row["fingerprint"]: row for row in pool["keys"]}
    assert by_fp["11111111"]["status"] == "runtime_only"
    assert by_fp["22222222"]["status"] == "runtime_only"
    assert by_fp["33333333"]["status"] == "runtime_only"
    assert by_fp["aaaa1111"]["status"] == "raw_key_missing"


def test_api_dashboard_marks_auth_quarantined_pool_keys(tmp_path, monkeypatch):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / f"summary_{today}.json").write_text(json.dumps({"single_bets": {"total": 0, "by_sport": {}}}))

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    future = (datetime.now(timezone.utc) + timedelta(hours=12)).isoformat()
    (data_dir / "odds_key_pool.json").write_text(json.dumps({
        "_meta": {
            "tracked_fingerprints": ["aaaa1111", "bbbb2222"],
            "runtime_loaded_fingerprints": ["aaaa1111", "bbbb2222"],
            "usable_fingerprints": ["bbbb2222"],
        },
        "aaaa1111": {
            "fingerprint": "aaaa1111",
            "remaining": 332,
            "updated_at": "2026-04-30T07:55:00+00:00",
            "auth_quarantined_until": future,
            "auth_quarantine_reason": "http_401",
        },
        "bbbb2222": {
            "fingerprint": "bbbb2222",
            "remaining": 120,
            "updated_at": "2026-04-30T08:00:00+00:00",
        },
    }))

    monkeypatch.setattr(webapp_app, "BASE", tmp_path)
    monkeypatch.setattr(webapp_app, "compute_summary", lambda: {})
    monkeypatch.setattr(webapp_app, "get_settled", lambda: pd.DataFrame())
    monkeypatch.setattr(webapp_app, "quota_bridge", None)

    client = webapp_app.app.test_client()
    response = client.get("/api/dashboard")

    assert response.status_code == 200
    pool = response.get_json()["odds_key_pool"]
    by_fp = {row["fingerprint"]: row for row in pool["keys"]}
    assert by_fp["aaaa1111"]["status"] == "auth_quarantined"
    assert by_fp["aaaa1111"]["exclusion_reason"] == "http_401"


def test_webapp_and_scan_share_same_runtime_key_parser(tmp_path, monkeypatch):
    monkeypatch.setattr(webapp_app, "BASE", tmp_path)
    monkeypatch.setenv("ODDS_API_KEY", " 'env-primary-11111111' ")
    monkeypatch.setenv("ODDS_API_KEYS", ' "env-primary-11111111, env-backup-22222222, bad key, env-third-33333333" ')

    pool_summary = webapp_app._odds_key_pool_summary()
    scan_keys = daily_scan._load_odds_key_pool()

    assert scan_keys == [
        "env-primary-11111111",
        "env-backup-22222222",
        "env-third-33333333",
    ]
    assert pool_summary["runtime_loaded_fingerprints"] == ["11111111", "22222222", "33333333"]
    assert any(item["reason"] == "invalid_format" for item in pool_summary["runtime_parse_excluded"])


def test_api_status_includes_odds_key_pool_summary(tmp_path, monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    monkeypatch.delenv("ODDS_API_KEYS", raising=False)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "odds_key_pool.json").write_text(json.dumps({
        "_meta": {"last_selected_fingerprint": "pool2222"},
        "pool1111": {"fingerprint": "pool1111", "remaining": 300},
        "pool2222": {"fingerprint": "pool2222", "remaining": 450},
    }))

    monkeypatch.setattr(webapp_app, "BASE", tmp_path)
    monkeypatch.setitem(sys.modules, "api_status", SimpleNamespace(
        check_odds_api=lambda: {"name": "Odds API", "status": "ok", "detail": "stub"},
        check_football_data=lambda: {"name": "Football-Data", "status": "ok", "detail": "stub"},
        check_balldontlie=lambda: {"name": "BallDontLie", "status": "ok", "detail": "stub"},
        check_mlb_api=lambda: {"name": "MLB API", "status": "ok", "detail": "stub"},
        check_nhl_api=lambda: {"name": "NHL API", "status": "ok", "detail": "stub"},
        check_telegram=lambda: {"name": "Telegram", "status": "ok", "detail": "stub"},
    ))

    client = webapp_app.app.test_client()
    response = client.get("/api/apis/status")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["odds_key_pool"]["enabled"] is True
    assert payload["odds_key_pool"]["count"] == 2
    assert payload["odds_key_pool"]["last_selected_fingerprint"] == "pool2222"


def test_api_update_allows_odds_api_keys(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("ODDS_API_KEY=seed\n")

    monkeypatch.setattr(webapp_app, "BASE", tmp_path)

    client = webapp_app.app.test_client()
    response = client.post("/api/apis/update", json={
        "var": "ODDS_API_KEYS",
        "value": "key1\nkey2\nkey3",
    })

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    updated = env_path.read_text()
    assert "ODDS_API_KEYS=" in updated
    assert "key1\nkey2\nkey3" in updated


def test_api_update_allows_news_api_key(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("")

    monkeypatch.setattr(webapp_app, "BASE", tmp_path)

    client = webapp_app.app.test_client()
    response = client.post("/api/apis/update", json={
        "var": "NEWS_API_KEY",
        "value": "news-key",
    })

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert "NEWS_API_KEY=" in env_path.read_text()


def test_reasoning_candidates_include_review_metadata(tmp_path, monkeypatch):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    summary_path = reports_dir / f"summary_{today}.json"
    summary_path.write_text(json.dumps({
        "single_bets": {
            "bets": [],
            "review_bets": [
                {
                    "sport": "soccer",
                    "team": "Team C or Draw",
                    "home": "Team D",
                    "away": "Team C",
                    "market": "double_chance",
                    "edge": 0.05,
                    "odds": 1.7,
                    "window": "today",
                    "availability_summary": "Away XI posted: 11 starters",
                    "context_adjustments": [{"name": "lineup_confirmation", "category": "lineup", "summary": "Selected side has a published starting XI while the opponent does not."}],
                    "review_required": True,
                    "review_reason": "availability or starter uncertainty still needs human review",
                }
            ],
        }
    }))

    monkeypatch.setattr(webapp_app, "BASE", tmp_path)

    client = webapp_app.app.test_client()
    response = client.get("/api/reasoning/candidates")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["total"] == 1
    assert payload["candidates"][0]["review_required"] is True
    assert "human review" in payload["candidates"][0]["review_reason"]
    assert payload["candidates"][0]["availability_summary"] == "Away XI posted: 11 starters"
    assert payload["candidates"][0]["context_adjustments"][0]["name"] == "lineup_confirmation"
    assert payload["candidates"][0]["decision_status"] == "WAIT FOR LINEUPS"
    assert "WAIT FOR LINEUPS" in payload["candidates"][0]["display"]


def test_reasoning_candidates_include_suppressed_decision_items(tmp_path, monkeypatch):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    summary_path = reports_dir / f"summary_{today}.json"
    summary_path.write_text(json.dumps({
        "single_bets": {
            "bets": [],
            "review_bets": [],
            "suppressed_bets": [
                {
                    "sport": "soccer",
                    "team": "Team X",
                    "home": "Team X",
                    "away": "Team Y",
                    "market": "moneyline",
                    "edge": 0.03,
                    "odds": 1.95,
                    "window": "today",
                    "decision_status": "AVOID",
                    "decision_reason": "same-game side correlation guardrail kept only the strongest thesis",
                    "suppression_reason": "same-game side correlation guardrail kept only the strongest thesis",
                }
            ],
        }
    }))
    monkeypatch.setattr(webapp_app, "BASE", tmp_path)

    client = webapp_app.app.test_client()
    response = client.get("/api/reasoning/candidates")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["total"] == 1
    assert payload["candidates"][0]["decision_status"] == "AVOID"
    assert "AVOID" in payload["candidates"][0]["display"]


def test_api_results_separates_value_bets_and_parlays_by_event_date(tmp_path, monkeypatch):
    tracker_dir = tmp_path / "data" / "tracker"
    tracker_dir.mkdir(parents=True)
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    (reports_dir / f"summary_{today}.json").write_text(json.dumps({
        "version_snapshot": {
            "scan_date": "2026-04-22",
            "policy_hash": "abc123def456",
            "models": {
                "mlb": {"tag": "mlb_2024_25", "calibrator_present": True, "artifact_sport": "mlb"},
                "tennis": {"tag": "atp_2022_25", "calibrator_present": True, "artifact_sport": "tennis"},
            },
        }
    }))

    predictions = pd.DataFrame([
        {
            "pred_id": "p1",
            "recorded_at": "2026-04-22T08:00:00Z",
            "commence_time": "2026-04-22T19:00:00Z",
            "sport": "soccer",
            "match_id": "A vs B",
            "team_or_player": "A or Draw",
            "bet_odds": 1.6,
            "edge": 0.08,
            "stake_units": 1.0,
            "status": "pending",
            "is_parlay_leg": False,
            "ml_prob": 0.7,
            "version_snapshot": json.dumps({"scan_date": "2026-04-22", "policy_hash": "abc123def456"}),
        },
        {
            "pred_id": "leg1",
            "recorded_at": "2026-04-22T08:00:00Z",
            "commence_time": "2026-04-22T20:00:00Z",
            "sport": "nhl",
            "match_id": "C vs D",
            "team_or_player": "C -1.5",
            "bet_odds": 2.1,
            "edge": 0.09,
            "stake_units": 0.5,
            "status": "pending",
            "is_parlay_leg": True,
            "parlay_id": "sys1",
            "ml_prob": 0.58,
            "version_snapshot": json.dumps({"scan_date": "2026-04-22", "policy_hash": "abc123def456"}),
        },
    ])
    predictions.to_parquet(tracker_dir / "predictions.parquet", index=False)

    settled = pd.DataFrame([
        {
            "pred_id": "s1",
            "settled_at": "2026-04-23T08:00:00Z",
            "recorded_at": "2026-04-22T07:00:00Z",
            "commence_time": "2026-04-22T18:00:00Z",
            "sport": "tennis",
            "match_id": "P1 vs P2",
            "team_or_player": "Over 22.5",
            "market": "totals",
            "market_status": "preferred",
            "tier": "Preferred",
            "bookmaker": "Betfair",
            "bet_odds": 1.9,
            "edge": 0.11,
            "stake_units": 1.0,
            "won": True,
            "profit_units": 0.9,
            "version_snapshot": json.dumps({"scan_date": "2026-04-22", "policy_hash": "abc123def456"}),
            "status": "won",
            "is_parlay_leg": False,
        },
        {
            "pred_id": "s2",
            "settled_at": "2026-04-23T09:00:00Z",
            "recorded_at": "2026-04-22T07:00:00Z",
            "commence_time": "2026-04-22T12:00:00Z",
            "sport": "parlay",
            "match_id": "2-leg Parlay",
            "team_or_player": "2-leg Parlay",
            "bet_odds": 3.5,
            "edge": 0.2,
            "stake_units": 0.5,
            "won": False,
            "profit_units": -0.5,
            "status": "lost",
            "is_parlay_leg": False,
        },
    ])
    settled.to_parquet(tracker_dir / "settled.parquet", index=False)

    manual_parlays = [
        {"id": "m1", "name": "2-leg Parlay", "date": "2026-04-22", "status": "pending", "legs": []}
    ]
    (tracker_dir / "manual_parlays.json").write_text(json.dumps(manual_parlays))

    monkeypatch.setattr(webapp_app, "BASE", tmp_path)
    monkeypatch.setattr(webapp_app, "_MANUAL_PARLAYS_FILE", tracker_dir / "manual_parlays.json")

    client = webapp_app.app.test_client()
    response = client.get("/api/results?date=2026-04-22")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["segments"]["value_bets"]["bets"] == 1
    assert payload["segments"]["parlays"]["bets"] == 1
    assert payload["performance_matrix"][0]["sport"] == "tennis"
    assert payload["performance_matrix"][0]["market"] == "totals"
    assert payload["performance_matrix"][0]["tier"] == "Preferred"
    assert payload["performance_matrix"][0]["clv_covered"] == 0
    assert payload["performance_matrix"][0]["settlement_coverage_pct"] == 100.0
    assert payload["performance_matrix"][0]["clv_signal"] == "missing"
    assert len(payload["settled"]) == 1
    assert payload["settled"][0]["sport"] == "tennis"
    assert payload["settled"][0]["tier"] == "Preferred"
    assert payload["settled"][0]["bookmaker"] == "Betfair"
    assert payload["settled"][0]["version_snapshot"]["policy_hash"] == "abc123def456"
    assert payload["pending"][0]["version_snapshot"]["scan_date"] == "2026-04-22"
    assert payload["version_summary"]["latest"]["policy_hash"] == "abc123def456"
    assert payload["version_summary"]["pending_with_snapshot"] == 2
    assert payload["version_summary"]["settled_with_snapshot"] == 1
    assert payload["settlement_reliability"]["summary"]["tracked_total"] == 3
    assert payload["settlement_reliability"]["summary"]["pending_total"] == 2
    assert any(row["sport"] == "soccer" for row in payload["settlement_reliability"]["rows"])
    assert payload["manual_parlays"][0]["status"] == "lost"


def test_governor_recommendations_include_promote_and_demote() -> None:
    matrix_rows = [
        {
            "sport": "mlb",
            "market": "moneyline",
            "tier": "Limited",
            "tier_status": "experimental",
            "bets": 14,
            "pending_count": 0,
            "tracked_total": 14,
            "settlement_coverage_pct": 100.0,
            "roi": 5.2,
            "avg_clv": 2.1,
            "clv_covered": 12,
            "clv_coverage_pct": 85.7,
            "clv_signal": "confirmed",
        },
        {
            "sport": "nhl",
            "market": "spreads",
            "tier": "Preferred",
            "tier_status": "preferred",
            "bets": 16,
            "pending_count": 0,
            "tracked_total": 16,
            "settlement_coverage_pct": 100.0,
            "roi": -7.5,
            "avg_clv": -1.4,
            "clv_covered": 14,
            "clv_coverage_pct": 87.5,
            "clv_signal": "weak",
        },
    ]

    recs = webapp_app._governor_recommendations(matrix_rows)

    actions = {(r["sport"], r["market"]): r["action"] for r in recs}
    assert actions[("mlb", "moneyline")] == "promote"
    assert actions[("nhl", "spreads")] == "demote"


def test_governor_recommendations_downgrade_low_settlement_or_clv_coverage() -> None:
    matrix_rows = [
        {
            "sport": "soccer",
            "market": "moneyline",
            "tier": "Limited",
            "tier_status": "experimental",
            "bets": 6,
            "pending_count": 6,
            "tracked_total": 12,
            "settlement_coverage_pct": 50.0,
            "roi": 8.0,
            "avg_clv": 3.0,
            "clv_covered": 6,
            "clv_coverage_pct": 100.0,
            "clv_signal": "confirmed",
        },
        {
            "sport": "mlb",
            "market": "moneyline",
            "tier": "Limited",
            "tier_status": "experimental",
            "bets": 12,
            "pending_count": 0,
            "tracked_total": 12,
            "settlement_coverage_pct": 100.0,
            "roi": 6.0,
            "avg_clv": 2.1,
            "clv_covered": 4,
            "clv_coverage_pct": 33.3,
            "clv_signal": "confirmed",
        },
    ]

    recs = webapp_app._governor_recommendations(matrix_rows)

    by_lane = {(r["sport"], r["market"]): r for r in recs}
    assert by_lane[("soccer", "moneyline")]["action"] == "watch"
    assert by_lane[("soccer", "moneyline")]["confidence"] == "low"
    assert "settled" in by_lane[("soccer", "moneyline")]["reason"].lower()
    assert by_lane[("mlb", "moneyline")]["action"] == "watch"
    assert by_lane[("mlb", "moneyline")]["confidence"] == "low"
    assert "clv coverage" in by_lane[("mlb", "moneyline")]["reason"].lower()


def test_retrain_triggers_flag_drift_but_not_variance() -> None:
    matrix_rows = [
        {
            "sport": "mlb",
            "market": "moneyline",
            "tier": "Preferred",
            "tier_status": "preferred",
            "bets": 14,
            "tracked_total": 16,
            "pending_count": 2,
            "settlement_coverage_pct": 87.5,
            "roi": -6.5,
            "avg_clv": -1.8,
            "clv_covered": 12,
            "clv_coverage_pct": 85.7,
            "clv_signal": "weak",
        },
        {
            "sport": "mlb",
            "market": "spreads",
            "tier": "Limited",
            "tier_status": "experimental",
            "bets": 10,
            "tracked_total": 12,
            "pending_count": 2,
            "settlement_coverage_pct": 83.3,
            "roi": -5.2,
            "avg_clv": -1.1,
            "clv_covered": 8,
            "clv_coverage_pct": 80.0,
            "clv_signal": "weak",
        },
        {
            "sport": "soccer",
            "market": "moneyline",
            "tier": "Preferred",
            "tier_status": "preferred",
            "bets": 18,
            "tracked_total": 20,
            "pending_count": 2,
            "settlement_coverage_pct": 90.0,
            "roi": -4.0,
            "avg_clv": 1.3,
            "clv_covered": 15,
            "clv_coverage_pct": 83.3,
            "clv_signal": "variance",
        },
    ]
    version_summary = {
        "latest": {"scan_date": "2026-04-22"},
        "model_rows": [
            {"sport": "mlb", "tag": "mlb_2024_25"},
            {"sport": "soccer", "tag": "pl_2024_25"},
        ],
    }

    triggers = webapp_app._retrain_trigger_rows(matrix_rows, version_summary)
    actions = {row["sport"]: row["action"] for row in triggers["rows"]}

    assert actions["mlb"] == "retrain"
    assert actions["soccer"] == "hold"


def test_api_settle_all_backlog_mode_scans_overdue_dates(tmp_path, monkeypatch) -> None:
    tracker_dir = tmp_path / "data" / "tracker"
    tracker_dir.mkdir(parents=True)
    pred = pd.DataFrame([
        {
            "pred_id": "old1",
            "recorded_at": "2026-04-10T08:00:00Z",
            "commence_time": "2026-04-10T18:00:00Z",
            "sport": "soccer",
            "match_id": "Old Home vs Old Away",
            "team_or_player": "Old Home",
            "bet_odds": 2.0,
            "edge": 0.08,
            "stake_units": 1.0,
            "kelly_stake_pct": 0.02,
            "status": "pending",
            "is_parlay_leg": False,
            "ml_prob": 0.55,
            "fair_prob": 0.52,
        }
    ])
    pred.to_parquet(tracker_dir / "predictions.parquet", index=False)

    monkeypatch.setattr(webapp_app, "BASE", tmp_path)
    monkeypatch.setattr(webapp_app, "_load_my_selections", lambda: [])
    monkeypatch.setattr(webapp_app, "_load_manual_parlays", lambda: [])

    def _fake_get(url, *args, **kwargs):
        if "football-data.org" in url:
            return _FakeHTTPResponse({
                "matches": [
                    {
                        "utcDate": "2026-04-10T18:00:00Z",
                        "homeTeam": {"name": "Old Home"},
                        "awayTeam": {"name": "Old Away"},
                        "score": {"fullTime": {"home": 2, "away": 1}},
                    }
                ]
            })
        return _FakeHTTPResponse({"events": []})

    monkeypatch.setattr(webapp_app.requests, "get", _fake_get)
    client = webapp_app.app.test_client()
    response = client.post("/api/settle-all", json={"mode": "backlog"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["mode"] == "backlog"
    assert payload["bets_settled"] == 1
    assert payload["target_dates_scanned"] >= 1


def test_api_settle_all_reports_unresolved_reason_summary(tmp_path, monkeypatch) -> None:
    tracker_dir = tmp_path / "data" / "tracker"
    tracker_dir.mkdir(parents=True)
    pred = pd.DataFrame([
        {
            "pred_id": "old2",
            "recorded_at": "2026-04-10T08:00:00Z",
            "commence_time": "2026-04-10T18:00:00Z",
            "sport": "soccer",
            "match_id": "Alpha FC vs Beta FC",
            "team_or_player": "Alpha FC",
            "bet_odds": 2.0,
            "edge": 0.08,
            "stake_units": 1.0,
            "kelly_stake_pct": 0.02,
            "status": "pending",
            "is_parlay_leg": False,
            "ml_prob": 0.55,
            "fair_prob": 0.52,
        }
    ])
    pred.to_parquet(tracker_dir / "predictions.parquet", index=False)

    monkeypatch.setattr(webapp_app, "BASE", tmp_path)
    monkeypatch.setattr(webapp_app, "_load_my_selections", lambda: [])
    monkeypatch.setattr(webapp_app, "_load_manual_parlays", lambda: [])

    def _fake_get(url, *args, **kwargs):
        if "football-data.org" in url:
            return _FakeHTTPResponse({
                "matches": [
                    {
                        "utcDate": "2026-04-10T18:00:00Z",
                        "homeTeam": {"name": "Gamma Club"},
                        "awayTeam": {"name": "Delta Club"},
                        "score": {"fullTime": {"home": 2, "away": 1}},
                    }
                ]
            })
        return _FakeHTTPResponse({"events": []})

    monkeypatch.setattr(webapp_app.requests, "get", _fake_get)
    client = webapp_app.app.test_client()
    response = client.post("/api/settle-all", json={"mode": "backlog"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["bets_settled"] == 0
    assert any(item["reason"] == "team_mismatch" for item in payload["unresolved_summary"])
    assert payload["unresolved_by_reason"][0]["reason"] == "team_mismatch"
    assert payload["unresolved_by_sport"][0]["sport"] == "soccer"
    assert payload["unresolved_samples"][0]["scope"] in {"system_book", "tracked"}


def test_api_settle_all_reports_date_mismatch_when_teams_match_on_wrong_day(tmp_path, monkeypatch) -> None:
    tracker_dir = tmp_path / "data" / "tracker"
    tracker_dir.mkdir(parents=True)

    my_selections_path = tracker_dir / "my_selections.json"
    my_selections_path.write_text(json.dumps([
        {
            "id": "sel-date-mismatch",
            "sport": "soccer",
            "team": "Alpha FC",
            "match": "Alpha FC vs Beta FC",
            "odds": 2.10,
            "commence": "2026-04-23T18:00:00Z",
            "date": "2026-04-23",
            "result": None,
            "profit": None,
        },
    ], indent=2))
    manual_parlays_path = tracker_dir / "manual_parlays.json"
    manual_parlays_path.write_text("[]")

    monkeypatch.setattr(webapp_app, "BASE", tmp_path)
    monkeypatch.setattr(webapp_app, "_MY_SELECTIONS_FILE", my_selections_path)
    monkeypatch.setattr(webapp_app, "_MANUAL_PARLAYS_FILE", manual_parlays_path)
    monkeypatch.setattr(webapp_app, "_live_scores_cache", {
        "payload": {
            "events": [
                {
                    "sport": "soccer",
                    "home": "Alpha FC",
                    "away": "Beta FC",
                    "status_type": "finished",
                    "home_score": 2,
                    "away_score": 1,
                    "commence_time": "2026-04-24T18:00:00Z",
                }
            ]
        },
        "ts": 10**12,
    })
    monkeypatch.setattr(webapp_app.requests, "get", lambda *args, **kwargs: _FakeHTTPResponse({"events": []}))

    client = webapp_app.app.test_client()
    response = client.post("/api/settle-all", json={})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["bets_settled"] == 0
    assert any(item["reason"] == "date_mismatch" for item in payload["unresolved_summary"])
    assert payload["unresolved_samples"][0]["reason"] == "date_mismatch"


def test_results_api_exposes_settlement_pending_samples(tmp_path, monkeypatch) -> None:
    tracker_dir = tmp_path / "data" / "tracker"
    tracker_dir.mkdir(parents=True)

    predictions = pd.DataFrame([
        {
            "pred_id": "p-pending-1",
            "recorded_at": "2026-04-20T07:00:00Z",
            "commence_time": "2026-04-20T12:00:00Z",
            "sport": "soccer",
            "match_id": "Alpha FC vs Beta FC",
            "team_or_player": "Alpha FC",
            "market": "moneyline",
            "tier": "Limited",
            "bookmaker": "Betfair",
            "bet_odds": 2.0,
            "edge": 0.05,
            "stake_units": 1.0,
            "status": "pending",
            "is_parlay_leg": False,
        }
    ])
    predictions.to_parquet(tracker_dir / "predictions.parquet", index=False)
    pd.DataFrame([], columns=["pred_id"]).to_parquet(tracker_dir / "settled.parquet", index=False)

    monkeypatch.setattr(webapp_app, "BASE", tmp_path)
    monkeypatch.setattr(webapp_app, "_load_manual_parlays", lambda: [])

    client = webapp_app.app.test_client()
    response = client.get("/api/results")

    assert response.status_code == 200
    payload = response.get_json()
    pending_samples = payload["settlement_reliability"]["pending_samples"]
    assert len(pending_samples) == 1
    assert pending_samples[0]["match"] == "Alpha FC vs Beta FC"
    assert pending_samples[0]["pick"] == "Alpha FC"


def test_replay_validation_can_downgrade_promote_to_watch() -> None:
    recommendations = [
        {
            "lane": "mlb:moneyline:Limited",
            "sport": "mlb",
            "market": "moneyline",
            "tier": "Limited",
            "action": "promote",
            "confidence": "medium",
            "reason": "Live lane looks strong.",
        }
    ]
    replay_support = {
        ("mlb", "moneyline"): {
            "support_level": "weak",
            "rank_within_sport": 4,
            "avg_log_loss": 0.6702,
            "avg_accuracy": 0.5942,
            "avg_ece": 0.0255,
            "games_scored": 2952,
        }
    }

    validated = webapp_app._apply_replay_validation(recommendations, replay_support)

    assert validated[0]["action"] == "watch"
    assert validated[0]["confidence"] == "low"
    assert validated[0]["replay_support"] == "weak"
    assert "historical replay" in validated[0]["reason"].lower()


def test_rebuild_candidates_surface_retrain_and_policy_actions() -> None:
    performance_matrix = [
        {
            "sport": "mlb",
            "market": "moneyline",
            "tier": "Limited",
            "tier_status": "experimental",
            "bets": 18,
            "tracked_total": 20,
            "settlement_coverage_pct": 90.0,
            "clv_coverage_pct": 88.0,
            "roi": -6.5,
            "avg_clv": -1.8,
            "clv_signal": "weak",
        },
        {
            "sport": "soccer",
            "market": "double_chance",
            "tier": "Preferred",
            "tier_status": "preferred",
            "bets": 16,
            "tracked_total": 16,
            "settlement_coverage_pct": 100.0,
            "clv_coverage_pct": 81.0,
            "roi": -5.5,
            "avg_clv": -1.4,
            "clv_signal": "weak",
        },
    ]
    retrain_triggers = {
        "rows": [
            {
                "sport": "mlb",
                "action": "retrain",
                "confidence": "high",
                "roi": -6.5,
                "avg_clv": -1.8,
                "weak_lanes": 2,
                "lanes": 3,
            }
        ]
    }
    governor_recommendations = [
        {
            "sport": "soccer",
            "market": "double_chance",
            "tier": "Preferred",
            "action": "demote",
            "confidence": "medium",
            "reason": "Lane is over-promoted.",
        }
    ]
    replay_support = {
        ("mlb", "moneyline"): {"support_level": "weak", "games_scored": 120},
        ("soccer", "double_chance"): {"support_level": "mixed", "games_scored": 340},
    }
    version_summary = {
        "model_rows": [
            {"sport": "mlb", "tag": "mlb_2024_25"},
            {"sport": "soccer", "tag": "pl_2024_25"},
        ]
    }

    payload = webapp_app._rebuild_candidates(
        performance_matrix,
        retrain_triggers,
        governor_recommendations,
        replay_support,
        version_summary,
    )

    rows = payload["rows"]
    assert rows[0]["action"] == "retrain"
    assert rows[0]["draft_command"] == ".venv/bin/python retrain_and_calibrate.py --sport mlb"
    policy_row = next(row for row in rows if row["action"] == "policy_tighten")
    assert policy_row["policy_template"]["status"] == "experimental"
    assert policy_row["policy_template"]["stake_multiplier"] <= 0.25


def test_replay_support_rows_are_exposed_in_results_api(tmp_path, monkeypatch) -> None:
    tracker_dir = tmp_path / "data" / "tracker"
    tracker_dir.mkdir(parents=True)

    predictions = pd.DataFrame([
        {
            "pred_id": "p1",
            "recorded_at": "2026-04-22T07:00:00Z",
            "commence_time": "2026-04-22T12:00:00Z",
            "sport": "mlb",
            "match_id": "A vs B",
            "team_or_player": "A",
            "bet_odds": 2.1,
            "edge": 0.05,
            "stake_units": 1.0,
            "status": "won",
            "won": True,
            "profit_units": 1.1,
            "market": "moneyline",
            "market_status": "experimental",
            "tier": "Limited",
            "bookmaker": "Betfair",
            "is_parlay_leg": False,
        }
    ])
    predictions.to_parquet(tracker_dir / "predictions.parquet", index=False)

    settled = pd.DataFrame([
        {
            "pred_id": "p1",
            "settled_at": "2026-04-23T09:00:00Z",
            "recorded_at": "2026-04-22T07:00:00Z",
            "commence_time": "2026-04-22T12:00:00Z",
            "sport": "mlb",
            "match_id": "A vs B",
            "team_or_player": "A",
            "bet_odds": 2.1,
            "edge": 0.05,
            "stake_units": 1.0,
            "won": True,
            "profit_units": 1.1,
            "status": "won",
            "market": "moneyline",
            "market_status": "experimental",
            "tier": "Limited",
            "bookmaker": "Betfair",
            "is_parlay_leg": False,
        }
    ])
    settled.to_parquet(tracker_dir / "settled.parquet", index=False)

    monkeypatch.setattr(webapp_app, "BASE", tmp_path)
    monkeypatch.setattr(
        webapp_app,
        "_load_replay_market_support",
        lambda: {
            ("mlb", "moneyline"): {
                "sport": "mlb",
                "market": "moneyline",
                "support_level": "strong",
                "rank_within_sport": 1,
                "spec_count": 3,
                "games_scored": 2952,
                "avg_accuracy": 0.5942,
                "avg_log_loss": 0.6702,
                "avg_ece": 0.0255,
            }
        },
    )

    client = webapp_app.app.test_client()
    response = client.get("/api/results?date=2026-04-22")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["replay_support_matrix"][0]["sport"] == "mlb"
    assert payload["replay_support_matrix"][0]["market"] == "moneyline"
    assert payload["replay_support_matrix"][0]["support_level"] == "strong"
    assert payload["replay_support_matrix"][0]["rank_within_sport"] == 1


def test_replay_slates_are_exposed_in_results_api(tmp_path, monkeypatch) -> None:
    tracker_dir = tmp_path / "data" / "tracker"
    tracker_dir.mkdir(parents=True)
    reports_dir = tmp_path / "reports" / "backtests" / "markets"
    reports_dir.mkdir(parents=True)

    pd.DataFrame([
        {
            "pred_id": "p1",
            "recorded_at": "2026-04-22T07:00:00Z",
            "commence_time": "2026-04-22T12:00:00Z",
            "sport": "mlb",
            "match_id": "A vs B",
            "team_or_player": "A",
            "bet_odds": 2.1,
            "edge": 0.05,
            "stake_units": 1.0,
            "status": "pending",
            "market": "moneyline",
            "market_status": "experimental",
            "tier": "Limited",
            "is_parlay_leg": False,
        }
    ]).to_parquet(tracker_dir / "predictions.parquet", index=False)

    pd.DataFrame([
        {
            "pred_id": "s1",
            "settled_at": "2026-04-23T09:00:00Z",
            "recorded_at": "2026-04-22T07:00:00Z",
            "commence_time": "2026-04-22T12:00:00Z",
            "sport": "mlb",
            "match_id": "A vs B",
            "team_or_player": "A",
            "bet_odds": 2.1,
            "edge": 0.05,
            "stake_units": 1.0,
            "won": True,
            "profit_units": 1.1,
            "status": "won",
            "market": "moneyline",
            "market_status": "experimental",
            "tier": "Limited",
            "bookmaker": "Betfair",
            "is_parlay_leg": False,
        }
    ]).to_parquet(tracker_dir / "settled.parquet", index=False)

    pd.DataFrame([
        {
            "date": pd.Timestamp("2026-04-20"),
            "sport": "mlb",
            "market": "moneyline",
            "market_type": "moneyline",
            "match_id": "A vs B",
            "correct": 1,
            "event_log_loss": 0.22,
        },
        {
            "date": pd.Timestamp("2026-04-20"),
            "sport": "mlb",
            "market": "spreads",
            "market_type": "spreads",
            "match_id": "C vs D",
            "correct": 0,
            "event_log_loss": 0.71,
        },
    ]).to_parquet(reports_dir / "mlb_moneyline_events.parquet", index=False)

    monkeypatch.setattr(webapp_app, "BASE", tmp_path)

    client = webapp_app.app.test_client()
    response = client.get("/api/results")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["replay_slates"]["summary"]["dates"] == 1
    assert payload["replay_slates"]["summary"]["events"] == 2
    assert payload["replay_slates"]["rows"][0]["date"] == "2026-04-20"
    assert payload["replay_slates"]["rows"][0]["published_events"] >= 1
    assert payload["replay_slate_events"][0]["date"] == "2026-04-20"
    assert payload["replay_slate_events"][0]["policy_bucket"] in {"preferred_live", "limited_live", "held_out"}
    assert payload["replay_slate_events"][0]["publish_decision"] in {"publish", "review", "hold_out"}
    assert payload["replay_publish_audit"]["summary"]["dates"] == 1
    assert payload["replay_publish_audit"]["rows"][0]["date"] == "2026-04-20"


def test_parlay_performance_matrix_is_exposed_in_results_api(tmp_path, monkeypatch) -> None:
    tracker_dir = tmp_path / "data" / "tracker"
    tracker_dir.mkdir(parents=True)

    pd.DataFrame([
        {
            "pred_id": "s1",
            "settled_at": "2026-04-23T09:00:00Z",
            "recorded_at": "2026-04-22T07:00:00Z",
            "commence_time": "2026-04-22T12:00:00Z",
            "sport": "mlb",
            "match_id": "A vs B",
            "team_or_player": "A",
            "bet_odds": 2.1,
            "edge": 0.05,
            "stake_units": 1.0,
            "won": True,
            "profit_units": 1.1,
            "status": "won",
            "market": "moneyline",
            "market_status": "experimental",
            "tier": "Limited",
            "bookmaker": "Betfair",
            "is_parlay_leg": False,
        }
    ]).to_parquet(tracker_dir / "settled.parquet", index=False)

    parlay_df = pd.DataFrame([
        {
            "parlay_id": "p-value",
            "recorded_at": pd.Timestamp("2026-04-22T07:00:00Z"),
            "tier": "value",
            "bracket": "5x",
            "n_legs": 2,
            "combined_odds": 4.8,
            "combined_prob": 0.24,
            "ev": 1.152,
            "edge": 0.152,
            "kelly_stake_pct": 1.0,
            "stake_units": 0.01,
            "legs_json": json.dumps([
                {"sport": "soccer", "match_id": "A vs B", "team": "A"},
                {"sport": "soccer", "match_id": "C vs D", "team": "C"},
            ]),
            "status": "won",
        },
        {
            "parlay_id": "p-long",
            "recorded_at": pd.Timestamp("2026-04-22T08:00:00Z"),
            "tier": "speculative",
            "bracket": "20x",
            "n_legs": 4,
            "combined_odds": 22.0,
            "combined_prob": 0.05,
            "ev": 1.1,
            "edge": 0.1,
            "kelly_stake_pct": 0.5,
            "stake_units": 0.005,
            "legs_json": json.dumps([
                {"sport": "mlb", "match_id": "E vs F", "team": "E"},
                {"sport": "nhl", "match_id": "G vs H", "team": "G"},
            ]),
            "status": "lost",
        },
    ])
    parlay_df.to_parquet(tracker_dir / "parlays.parquet", index=False)

    pd.DataFrame([
        {
            "pred_id": "p1",
            "recorded_at": "2026-04-22T07:00:00Z",
            "commence_time": "2026-04-22T12:00:00Z",
            "sport": "mlb",
            "match_id": "A vs B",
            "team_or_player": "A",
            "bet_odds": 2.1,
            "edge": 0.05,
            "stake_units": 1.0,
            "status": "pending",
            "market": "moneyline",
            "market_status": "experimental",
            "tier": "Limited",
            "is_parlay_leg": False,
        }
    ]).to_parquet(tracker_dir / "predictions.parquet", index=False)

    monkeypatch.setattr(webapp_app, "BASE", tmp_path)
    monkeypatch.setattr(webapp_app, "_parlay_breakdown", lambda: parlay_df.assign(
        won=parlay_df["status"] == "won",
        profit_units=parlay_df.apply(
            lambda r: (r["combined_odds"] - 1) * r["stake_units"] if r["status"] == "won" else -r["stake_units"],
            axis=1,
        ),
    ))
    monkeypatch.setattr(webapp_app, "_load_manual_parlays", lambda: [
        {
            "id": "m1",
            "name": "Manual Custom 3-Leg",
            "date": "2026-04-22",
            "saved_at": "2026-04-22T10:00:00Z",
            "type": "manual",
            "status": "won",
            "n_legs": 3,
            "combined_odds": 7.5,
            "ev": 1.2,
            "edge": 20.0,
            "kelly_pct": 0.8,
            "legs": [
                {"sport": "soccer", "match": "I vs J", "team": "I"},
                {"sport": "nhl", "match": "K vs L", "team": "K"},
            ],
        },
        {
            "id": "a1",
            "name": "AI Longshot 4-Leg",
            "date": "2026-04-22",
            "saved_at": "2026-04-22T11:00:00Z",
            "type": "ai_longshot",
            "status": "lost",
            "n_legs": 4,
            "combined_odds": 18.5,
            "ev": 1.05,
            "edge": 5.0,
            "kelly_pct": 0.4,
            "legs": [
                {"sport": "mlb", "match": "M vs N", "team": "M"},
                {"sport": "basketball", "match": "O vs P", "team": "O"},
            ],
        },
    ])

    client = webapp_app.app.test_client()
    response = client.get("/api/results")

    assert response.status_code == 200
    payload = response.get_json()
    assert len(payload["parlay_performance"]["summary_cards"]) >= 4
    styles = {row["style"] for row in payload["parlay_performance"]["matrix_rows"]}
    assert {"Value", "Longshot", "Custom"}.issubset(styles)
    sources = {row["source"] for row in payload["parlay_performance"]["matrix_rows"]}
    assert {"System", "AI", "Manual"}.issubset(sources)
    assert any(row["bracket"] == "20x" for row in payload["parlay_performance"]["matrix_rows"])
    assert any(row["source"] == "Manual" for row in payload["parlay_performance"]["detail_rows"])


def test_governor_change_preview_builds_market_policy_draft() -> None:
    recommendations = [
        {
            "sport": "mlb",
            "market": "moneyline",
            "tier": "Limited",
            "action": "promote",
            "confidence": "medium",
            "reason": "Live lane looks strong.",
            "replay_support": "strong",
        },
        {
            "sport": "nhl",
            "market": "spreads",
            "tier": "Preferred",
            "action": "watch",
            "confidence": "low",
            "reason": "No action yet.",
            "replay_support": "strong",
        },
    ]

    preview = webapp_app._governor_change_preview(recommendations)

    assert len(preview) == 1
    assert preview[0]["sport"] == "mlb"
    assert preview[0]["market"] == "moneyline"
    assert preview[0]["action"] == "promote"
    assert preview[0]["draft"]["status"] == "preferred"
    assert preview[0]["draft"]["parlay_allowed"] is True
    assert "status" in preview[0]["changed_fields"]


def test_replay_policy_audit_flags_alignment_gaps() -> None:
    replay_support = {
        ("mlb", "moneyline"): {
            "sport": "mlb",
            "market": "moneyline",
            "support_level": "strong",
            "rank_within_sport": 1,
            "games_scored": 2952,
            "avg_accuracy": 0.5942,
            "avg_log_loss": 0.6702,
            "avg_ece": 0.0255,
        },
        ("nhl", "totals"): {
            "sport": "nhl",
            "market": "totals",
            "support_level": "weak",
            "rank_within_sport": 4,
            "games_scored": 1800,
            "avg_accuracy": 0.5210,
            "avg_log_loss": 0.7110,
            "avg_ece": 0.0410,
        },
    }

    audit = webapp_app._replay_policy_audit(replay_support)
    by_lane = {(row["sport"], row["market"]): row for row in audit}

    assert by_lane[("mlb", "moneyline")]["alignment"] == "underpromoted"
    assert by_lane[("mlb", "moneyline")]["recommended_status"] == "preferred"
    assert by_lane[("nhl", "totals")]["alignment"] == "aligned"
    assert by_lane[("nhl", "totals")]["recommended_status"] == "disabled"


def test_replay_portfolio_simulation_summarizes_current_policy_buckets() -> None:
    replay_support = {
        ("mlb", "moneyline"): {
            "sport": "mlb",
            "market": "moneyline",
            "support_level": "strong",
            "games_scored": 1000,
            "avg_accuracy": 0.58,
            "avg_log_loss": 0.67,
            "avg_ece": 0.02,
        },
        ("soccer", "double_chance"): {
            "sport": "soccer",
            "market": "double_chance",
            "support_level": "strong",
            "games_scored": 2000,
            "avg_accuracy": 0.69,
            "avg_log_loss": 0.58,
            "avg_ece": 0.02,
        },
        ("nhl", "totals"): {
            "sport": "nhl",
            "market": "totals",
            "support_level": "weak",
            "games_scored": 1500,
            "avg_accuracy": 0.52,
            "avg_log_loss": 0.71,
            "avg_ece": 0.04,
        },
    }

    portfolio = webapp_app._replay_portfolio_simulation(replay_support)

    assert portfolio["preferred_live"]["games_scored"] == 2000
    assert portfolio["limited_live"]["games_scored"] == 1000
    assert portfolio["held_out"]["games_scored"] == 1500
    assert portfolio["published_total"]["games_scored"] == 3000
    assert len(portfolio["lane_rows"]) == 3


def test_replay_policy_scenarios_compare_current_vs_aligned_policy() -> None:
    replay_support = {
        ("mlb", "moneyline"): {
            "sport": "mlb",
            "market": "moneyline",
            "support_level": "strong",
            "games_scored": 1000,
            "avg_accuracy": 0.58,
            "avg_log_loss": 0.67,
            "avg_ece": 0.02,
        },
        ("soccer", "double_chance"): {
            "sport": "soccer",
            "market": "double_chance",
            "support_level": "strong",
            "games_scored": 2000,
            "avg_accuracy": 0.69,
            "avg_log_loss": 0.58,
            "avg_ece": 0.02,
        },
        ("nhl", "totals"): {
            "sport": "nhl",
            "market": "totals",
            "support_level": "weak",
            "games_scored": 1500,
            "avg_accuracy": 0.52,
            "avg_log_loss": 0.71,
            "avg_ece": 0.04,
        },
    }

    scenarios = webapp_app._replay_policy_scenarios(replay_support)

    assert scenarios["current_policy"]["games_scored"] == 3000
    assert scenarios["replay_aligned_policy"]["games_scored"] == 3000
    assert scenarios["delta_games"] == 0
    assert scenarios["promoted_by_alignment"] == 0
    assert scenarios["held_out_by_alignment"] == 0


def test_reasoning_candidates_only_include_today_supported_value_bets(tmp_path, monkeypatch):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / f"summary_{today}.json").write_text(json.dumps({
        "single_bets": {
            "bets": [
                {
                    "sport": "tennis",
                    "team": "Player A",
                    "home": "Player A",
                    "away": "Player B",
                    "market": "moneyline",
                    "edge": 0.08,
                    "odds": 1.9,
                    "window": "today",
                },
                {
                    "sport": "soccer",
                    "team": "Team C or Draw",
                    "home": "Team D",
                    "away": "Team C",
                    "market": "double_chance",
                    "edge": 0.05,
                    "odds": 1.7,
                    "window": "today",
                },
                {
                    "sport": "basketball",
                    "team": "Over 225.5",
                    "home": "Team E",
                    "away": "Team F",
                    "market": "totals",
                    "edge": 0.07,
                    "odds": 1.95,
                    "window": "tomorrow",
                },
            ]
        }
    }))
    monkeypatch.setattr(webapp_app, "BASE", tmp_path)

    client = webapp_app.app.test_client()
    response = client.get("/api/reasoning/candidates")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["total"] == 3
    selections = {candidate["selection"] for candidate in payload["candidates"]}
    assert "Player A" in selections
    assert "Team C or Draw" in selections
    assert "Over 225.5" in selections


def test_reasoning_scan_requires_today_candidate(tmp_path, monkeypatch):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    bet = {
        "sport": "tennis",
        "team": "Player A",
        "home": "Player A",
        "away": "Player B",
        "market": "moneyline",
        "edge": 0.08,
        "odds": 1.9,
        "window": "today",
    }
    (reports_dir / f"summary_{today}.json").write_text(json.dumps({"single_bets": {"bets": [bet]}}))
    monkeypatch.setattr(webapp_app, "BASE", tmp_path)

    class _FakeReport:
        def to_dict(self):
            return {"verdict": "lean", "confidence": 0.66, "signals": [], "warnings": [], "unknowns": []}

        def to_markdown(self):
            return "ok"

    class _FakeAnalyst:
        def analyze_game(self, **kwargs):
            return _FakeReport()

    monkeypatch.setattr(webapp_app, "ManualGameAnalyst", _FakeAnalyst)

    client = webapp_app.app.test_client()
    valid_id = webapp_app._reasoning_candidate_id(webapp_app.annotate_bet(bet))
    ok_response = client.post("/api/reasoning/scan", json={"candidate_id": valid_id})
    bad_response = client.post("/api/reasoning/scan", json={"candidate_id": "not|real"})

    assert ok_response.status_code == 200
    assert ok_response.get_json()["ok"] is True
    assert bad_response.status_code == 404


def test_reasoning_scan_includes_openrouter_layer_when_configured(tmp_path, monkeypatch):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    bet = {
        "sport": "tennis",
        "team": "Player A",
        "home": "Player A",
        "away": "Player B",
        "market": "moneyline",
        "edge": 0.08,
        "odds": 1.9,
        "window": "today",
    }
    (reports_dir / f"summary_{today}.json").write_text(json.dumps({"single_bets": {"bets": [bet]}}))
    monkeypatch.setattr(webapp_app, "BASE", tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    class _FakeReport:
        def to_dict(self):
            return {"verdict": "lean", "confidence": 0.66, "signals": [], "warnings": [], "unknowns": []}

        def to_markdown(self):
            return "ok"

    class _FakeAnalyst:
        def analyze_game(self, **kwargs):
            return _FakeReport()

    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "model": "openrouter/free",
                "choices": [{"message": {"content": json.dumps({
                    "decision": "APPROVE",
                    "recommendation": "Lean with caution",
                    "reasoning": "No critical negative context was found.",
                    "why_for": "Model edge is positive.",
                    "why_against": "Tennis volatility remains high.",
                    "biggest_risk": "Serve variance.",
                    "stake_guidance": "Keep stake modest.",
                    "critical_factors": ["volatility"],
                    "only_context_based": True,
                })}}],
            }

    monkeypatch.setattr(webapp_app, "ManualGameAnalyst", _FakeAnalyst)
    monkeypatch.setattr(webapp_app.requests, "post", lambda *args, **kwargs: _FakeResponse())

    client = webapp_app.app.test_client()
    candidate_id = webapp_app._reasoning_candidate_id(webapp_app.annotate_bet(bet))
    response = client.post("/api/reasoning/scan", json={"candidate_id": candidate_id})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["llm_reasoning"]["provider"] == "openrouter"
    assert payload["llm_reasoning"]["content"]["decision"] == "APPROVE"
    assert payload["llm_reasoning"]["content"]["recommendation"] == "Lean with caution"
    assert payload["referee_system_decision"] == "BET"
    assert any("Context referee:" in item for item in payload["report"]["warnings"])


def test_reasoning_scan_accepts_soccer_double_chance_candidate(tmp_path, monkeypatch):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    bet = {
        "sport": "soccer",
        "team": "Atletico Madrid or Draw",
        "home": "Elche CF",
        "away": "Atletico Madrid",
        "market": "double_chance",
        "edge": 0.0528,
        "odds": 1.6,
        "fair_prob": 0.63,
        "window": "today",
    }
    (reports_dir / f"summary_{today}.json").write_text(json.dumps({"single_bets": {"bets": [bet]}}))
    monkeypatch.setattr(webapp_app, "BASE", tmp_path)

    monkeypatch.setattr(webapp_app, "_attach_fresh_news_context", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(webapp_app, "_openrouter_reasoning_layer", lambda *_args, **_kwargs: None)

    client = webapp_app.app.test_client()
    candidate_id = webapp_app._reasoning_candidate_id(webapp_app.annotate_bet(bet))
    response = client.post("/api/reasoning/scan", json={"candidate_id": candidate_id})
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["report"]["market"] == "double_chance"
    assert payload["report"]["selection"] == "away_or_draw"
    assert payload["report"]["fair_prob"] == 0.63


def test_reasoning_scan_surfaces_system_context_in_report(tmp_path, monkeypatch):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    bet = {
        "sport": "soccer",
        "team": "Atletico Madrid or Draw",
        "home": "Elche CF",
        "away": "Atletico Madrid",
        "market": "double_chance",
        "edge": 0.0528,
        "odds": 1.6,
        "minimum_acceptable_odds": 1.54,
        "odds_recheck_status": "passed",
        "odds_recheck_delta": 0.06,
        "fair_prob": 0.63,
        "window": "today",
        "availability_summary": "Away XI posted: 11 starters",
        "context_adjustments": [
            {"name": "tactical_matchup", "category": "matchup", "summary": "Soccer matchup edge from expected-goals profile and chance-quality shape."},
            {"name": "travel_fatigue", "category": "environment", "summary": "Travel burden adjustment from distance and timezone shift on the away side."},
        ],
        "prediction_factors": [
            {"name": "market_edge", "category": "pricing", "summary": "True probability cleared the synthetic price."},
        ],
        "true_probability": {"adjusted_prob": 0.63},
        "scraped_context": {"availability_source": "api_football", "away_lineup_confirmed": 1},
        "scraped_context_highlights": ["Away lineup posted (11 likely starters)"],
        "scraped_context_sources": ["api_football"],
    }
    (reports_dir / f"summary_{today}.json").write_text(json.dumps({"single_bets": {"bets": [bet]}}))
    monkeypatch.setattr(webapp_app, "BASE", tmp_path)

    class _FakeReport:
        def __init__(self):
            self.warnings = []

        def to_dict(self):
            return {"verdict": "lean", "confidence": 0.66, "signals": [], "warnings": self.warnings, "unknowns": []}

        def to_markdown(self):
            return "ok"

    class _FakeAnalyst:
        def analyze_game(self, **kwargs):
            return _FakeReport()

    monkeypatch.setattr(webapp_app, "ManualGameAnalyst", _FakeAnalyst)
    monkeypatch.setattr(webapp_app, "_openrouter_reasoning_layer", lambda *_args, **_kwargs: None)

    client = webapp_app.app.test_client()
    candidate_id = webapp_app._reasoning_candidate_id(webapp_app.annotate_bet(bet))
    response = client.post("/api/reasoning/scan", json={"candidate_id": candidate_id})

    assert response.status_code == 200
    candidate = response.get_json()["candidate"]
    assert candidate["minimum_acceptable_odds"] == 1.54
    assert candidate["odds_recheck_status"] == "passed"
    assert candidate["scraped_context"]["away_lineup_confirmed"] == 1
    assert candidate["scraped_context_sources"] == ["api_football"]
    warnings = response.get_json()["report"]["warnings"]
    assert any("System context:" in item for item in warnings)
    assert any("System scraper:" in item for item in warnings)
    assert any("System availability:" in item for item in warnings)


def test_reasoning_scan_returns_decision_status_fields(tmp_path, monkeypatch):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    bet = {
        "sport": "soccer",
        "team": "Team C or Draw",
        "home": "Team D",
        "away": "Team C",
        "market": "double_chance",
        "edge": 0.05,
        "odds": 1.7,
        "window": "today",
        "review_required": True,
        "review_reason": "availability or starter uncertainty still needs human review",
        "decision_status": "WAIT FOR LINEUPS",
        "decision_reason": "availability or starter uncertainty still needs human review",
    }
    (reports_dir / f"summary_{today}.json").write_text(json.dumps({"single_bets": {"review_bets": [bet]}}))
    monkeypatch.setattr(webapp_app, "BASE", tmp_path)
    monkeypatch.setattr(webapp_app, "_attach_fresh_news_context", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(webapp_app, "_openrouter_reasoning_layer", lambda *_args, **_kwargs: None)

    client = webapp_app.app.test_client()
    candidate_id = webapp_app._reasoning_candidate_id(webapp_app.annotate_bet(bet))
    response = client.post("/api/reasoning/scan", json={"candidate_id": candidate_id})

    assert response.status_code == 200
    candidate = response.get_json()["candidate"]
    assert candidate["decision_status"] == "WAIT FOR LINEUPS"
    assert "starter uncertainty" in candidate["decision_reason"]


def test_api_picks_exposes_committee_fields(tmp_path, monkeypatch):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    bet = {
        "sport": "soccer",
        "team": "Alpha FC",
        "home": "Alpha FC",
        "away": "Beta FC",
        "market": "moneyline",
        "edge": 0.08,
        "odds": 1.92,
        "window": "today",
        "decision_status": "BET",
        "decision_reason": "Committee approved for publication.",
        "committee_final_decision": "BET",
        "committee_agreement_status": "FULL_AGREEMENT",
        "committee_veto_flags": [],
        "committee_reason": "Research and model both cleared the pick.",
        "committee_better_substitute": "",
        "committee_parlay_suitability": "small_parlay_only",
        "committee": {
            "research_mind": {
                "sport": "soccer",
                "verdict": "AGREE",
                "evidence_status": "ACCEPTABLE",
                "concrete_info_score": 78,
                "source_count": 2,
                "source_quality_summary": "strong",
                "fixture_verified": True,
                "odds_age_minutes": 19,
                "odds_freshness_status": "acceptable",
                "market_availability_status": "available",
                "lineup_status": "monitor",
                "injury_status": "checked_fresh",
                "motivation_status": "checked",
                "rotation_status": "checked",
                "missing_evidence": [],
                "sport_specific_missing_evidence": [],
                "conflicting_evidence": [],
                "main_risks": ["No major risks detected from available evidence"],
            },
            "model_mind": {"verdict": "BET"},
            "arbiter": {"final_decision": "BET", "parlay_suitability": "small_parlay_only"},
        },
        "committee_details_text": "Game: Alpha FC vs Beta FC",
    }
    (reports_dir / f"summary_{today}.json").write_text(json.dumps({"single_bets": {"bets": [bet], "review_bets": []}}))
    monkeypatch.setattr(webapp_app, "BASE", tmp_path)

    client = webapp_app.app.test_client()
    response = client.get("/api/picks")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["bets"][0]["committee_final_decision"] == "BET"
    assert payload["bets"][0]["committee_agreement_status"] == "FULL_AGREEMENT"
    assert payload["bets"][0]["committee"]["arbiter"]["final_decision"] == "BET"
    assert payload["bets"][0]["committee"]["research_mind"]["evidence_status"] == "ACCEPTABLE"
    assert payload["bets"][0]["committee"]["research_mind"]["concrete_info_score"] == 78
    assert payload["bets"][0]["committee"]["research_mind"]["source_count"] == 2
    assert payload["bets"][0]["committee"]["research_mind"]["sport"] == "soccer"
    assert payload["bets"][0]["committee"]["research_mind"]["market_availability_status"] == "available"
    assert payload["bets"][0]["committee"]["research_mind"]["lineup_status"] == "monitor"


def test_api_picks_exposes_soccer_committee_blocker_summary_for_review_bets(tmp_path, monkeypatch):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    bet = {
        "sport": "soccer",
        "team": "Team C or Draw",
        "home": "Team D",
        "away": "Team C",
        "market": "double_chance",
        "edge": 0.05,
        "odds": 1.7,
        "window": "today",
        "decision_status": "HOLD",
        "decision_reason": "Committee review completed.",
        "committee_final_decision": "HOLD",
        "committee_veto_flags": ["MISSING_SPORT_CRITICAL_EVIDENCE"],
        "research_mind_evidence_status": "INSUFFICIENT",
        "research_mind_source_count": 1,
        "research_mind_source_quality_summary": "weak",
        "research_mind_lineup_status": "unknown",
        "research_mind_injury_status": "not_checked",
        "research_mind_motivation_status": "not_checked",
        "research_mind_rotation_status": "not_checked",
        "research_mind_missing_evidence": ["injury/team news", "rotation context", "motivation context"],
        "committee_enrichment": {
            "triggered": True,
            "sources_found": ["bookmaker"],
            "lineup_status": "not_found",
            "injury_status": "provider_failed",
            "rotation_status": "not_checked",
            "motivation_status": "not_checked",
            "fixture_congestion_status": "not_checked",
            "remaining_missing_evidence": ["injury/team news", "rotation context", "motivation context"],
        },
    }
    (reports_dir / f"summary_{today}.json").write_text(json.dumps({"single_bets": {"bets": [], "review_bets": [bet]}}))
    monkeypatch.setattr(webapp_app, "BASE", tmp_path)

    client = webapp_app.app.test_client()
    response = client.get("/api/picks")

    assert response.status_code == 200
    payload = response.get_json()
    review_bet = payload["review_bets"][0]
    assert "committee_blocker_summary" in review_bet
    assert "committee_blockers" in review_bet
    assert "Evidence is still insufficient" in review_bet["committee_blocker_summary"]
    assert any("Injury and team-news context" in item for item in review_bet["committee_blockers"])
    assert any("Rotation risk" in item for item in review_bet["committee_blockers"])


def test_api_picks_falls_back_to_latest_available_summary_when_today_missing(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    previous_day = "2026-05-06"
    bet = {
        "sport": "soccer",
        "team": "Late Slate FC",
        "home": "Late Slate FC",
        "away": "Carryover United",
        "market": "moneyline",
        "edge": 0.05,
        "odds": 1.8,
        "window": "today",
    }
    (reports_dir / f"summary_{previous_day}.json").write_text(json.dumps({"single_bets": {"bets": [bet], "review_bets": []}}))
    monkeypatch.setattr(webapp_app, "BASE", tmp_path)

    client = webapp_app.app.test_client()
    response = client.get("/api/picks")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["summary_date"] == previous_day
    assert payload["bets"][0]["team"] == "Late Slate FC"
    assert previous_day in payload["available_dates"]


def test_api_picks_skips_empty_latest_summary_and_uses_latest_meaningful_summary(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    previous_day = "2026-05-06"
    (reports_dir / f"summary_{today}.json").write_text(json.dumps({
        "single_bets": {"total": 0, "review_total": 0, "bets": [], "review_bets": [], "suppressed_bets": []},
        "sport_pipeline_diagnostics": {"by_sport": {}, "totals": {"scanned_games": 0}},
    }))
    bet = {
        "sport": "soccer",
        "team": "Recovered FC",
        "home": "Recovered FC",
        "away": "Fallback United",
        "market": "moneyline",
        "edge": 0.06,
        "odds": 1.9,
        "window": "today",
    }
    (reports_dir / f"summary_{previous_day}.json").write_text(json.dumps({
        "single_bets": {"bets": [bet], "review_bets": [], "total": 1, "review_total": 0},
        "sport_pipeline_diagnostics": {"by_sport": {"soccer": {"scanned_games": 1}}, "totals": {"scanned_games": 1}},
    }))
    monkeypatch.setattr(webapp_app, "BASE", tmp_path)

    client = webapp_app.app.test_client()
    response = client.get("/api/picks")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["summary_date"] == previous_day
    assert payload["bets"][0]["team"] == "Recovered FC"


def test_api_picks_exposes_market_coverage_from_summary(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    (reports_dir / f"summary_{today}.json").write_text(json.dumps({
        "soccer_games": [{
            "sport": "soccer",
            "home": "Alpha FC",
            "away": "Beta FC",
            "available_market_keys": ["h2h", "double_chance", "spreads"],
            "outcomes": [],
        }],
        "other_games": [{
            "sport": "basketball",
            "home": "Detroit Pistons",
            "away": "Cleveland Cavaliers",
            "available_market_keys": ["h2h", "totals"],
        }],
        "single_bets": {"bets": [], "review_bets": [], "suppressed_bets": [], "total": 0, "review_total": 0},
    }))
    monkeypatch.setattr(webapp_app, "BASE", tmp_path)

    client = webapp_app.app.test_client()
    response = client.get("/api/picks")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["market_coverage"]["soccer"] == ["h2h", "double_chance", "spreads"]
    assert payload["market_coverage"]["basketball"] == ["h2h", "totals"]
    market_note = next(note for note in payload["scan_notes"] if note.get("type") == "market_coverage")
    assert market_note["by_sport"]["soccer"] == ["h2h", "double_chance", "spreads"]


def test_api_picks_respects_explicit_date_without_falling_back(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "summary_2026-05-06.json").write_text(json.dumps({"single_bets": {"bets": [], "review_bets": []}}))
    monkeypatch.setattr(webapp_app, "BASE", tmp_path)

    client = webapp_app.app.test_client()
    response = client.get("/api/picks?date=2026-05-05")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["bets"] == []
    assert payload["summary_date"] == "2026-05-05"
    assert "No scan found for 2026-05-05." in payload["error"]


def test_api_games_falls_back_to_latest_available_summary_when_today_missing(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    previous_day = "2026-05-06"
    (reports_dir / f"summary_{previous_day}.json").write_text(json.dumps({
        "soccer_games": [{
            "home": "Fallback FC",
            "away": "Latest Town",
            "league": "Soccer",
            "commence": "2026-05-06T19:00:00Z",
            "window": "today",
            "outcomes": [{"label": "Home Win", "ml_prob": 0.55, "odds": 1.9, "has_value": True}],
        }],
        "other_games": [],
        "single_bets": {"bets": [], "review_bets": []},
    }))
    monkeypatch.setattr(webapp_app, "BASE", tmp_path)

    client = webapp_app.app.test_client()
    response = client.get("/api/games")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["summary_date"] == previous_day
    assert payload["games"][0]["home"] == "Fallback FC"


def test_api_settle_all_checks_real_pending_dates_and_resolves_soccer_dnb_and_double_chance(tmp_path, monkeypatch):
    tracker_dir = tmp_path / "data" / "tracker"
    tracker_dir.mkdir(parents=True)

    my_selections_path = tracker_dir / "my_selections.json"
    my_selections_path.write_text(json.dumps([
        {
            "id": "sel-1",
            "sport": "soccer",
            "team": "Team C or Draw",
            "match": "Team D vs Team C",
            "odds": 1.75,
            "commence": "2026-04-23T18:00:00Z",
            "date": "2026-04-23",
            "result": None,
            "profit": None,
        },
        {
            "id": "sel-2",
            "sport": "soccer",
            "team": "Team E DNB",
            "match": "Team E vs Team F",
            "odds": 1.90,
            "commence": "2026-04-23T20:00:00Z",
            "date": "2026-04-23",
            "result": None,
            "profit": None,
        },
    ], indent=2))
    manual_parlays_path = tracker_dir / "manual_parlays.json"
    manual_parlays_path.write_text("[]")

    monkeypatch.setattr(webapp_app, "BASE", tmp_path)
    monkeypatch.setattr(webapp_app, "_MY_SELECTIONS_FILE", my_selections_path)
    monkeypatch.setattr(webapp_app, "_MANUAL_PARLAYS_FILE", manual_parlays_path)
    monkeypatch.setattr(webapp_app, "_live_scores_cache", {"payload": None, "ts": 0.0})

    requested_urls = []

    class _FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def _fake_get(url, headers=None, timeout=None):
        requested_urls.append(url)
        if "/sport/football/scheduled-events/2026-04-23" in url:
            return _FakeResponse({
                "events": [
                    {
                        "status": {"type": "finished"},
                        "homeTeam": {"name": "Team D"},
                        "awayTeam": {"name": "Team C"},
                        "homeScore": {"current": 1},
                        "awayScore": {"current": 2},
                    },
                    {
                        "status": {"type": "finished"},
                        "homeTeam": {"name": "Team E"},
                        "awayTeam": {"name": "Team F"},
                        "homeScore": {"current": 2},
                        "awayScore": {"current": 1},
                    },
                ]
            })
        return _FakeResponse({"events": []})

    monkeypatch.setattr(webapp_app.requests, "get", _fake_get)

    client = webapp_app.app.test_client()
    response = client.post("/api/settle-all", json={})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["bets_settled"] == 2
    assert any("2026-04-23" in url for url in requested_urls)

    updated = json.loads(my_selections_path.read_text())
    by_id = {item["id"]: item for item in updated}
    assert by_id["sel-1"]["result"] == "won"
    assert by_id["sel-2"]["result"] == "won"


def test_api_settle_all_matches_finished_games_even_when_home_away_order_is_swapped(tmp_path, monkeypatch):
    tracker_dir = tmp_path / "data" / "tracker"
    tracker_dir.mkdir(parents=True)

    my_selections_path = tracker_dir / "my_selections.json"
    my_selections_path.write_text(json.dumps([
        {
            "id": "sel-swap",
            "sport": "nhl",
            "team": "Pittsburgh Penguins",
            "match": "Pittsburgh Penguins vs Philadelphia Flyers",
            "odds": 2.10,
            "commence": "2026-04-27T23:00:00Z",
            "date": "2026-04-27",
            "result": None,
            "profit": None,
        },
    ], indent=2))
    manual_parlays_path = tracker_dir / "manual_parlays.json"
    manual_parlays_path.write_text("[]")

    monkeypatch.setattr(webapp_app, "BASE", tmp_path)
    monkeypatch.setattr(webapp_app, "_MY_SELECTIONS_FILE", my_selections_path)
    monkeypatch.setattr(webapp_app, "_MANUAL_PARLAYS_FILE", manual_parlays_path)
    monkeypatch.setattr(webapp_app, "_live_scores_cache", {"payload": None, "ts": 0.0})

    class _FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def _fake_get(url, headers=None, timeout=None):
        if "/sport/ice-hockey/scheduled-events/2026-04-27" in url:
            return _FakeResponse({
                "events": [
                    {
                        "status": {"type": "finished"},
                        "homeTeam": {"name": "Philadelphia Flyers"},
                        "awayTeam": {"name": "Pittsburgh Penguins"},
                        "homeScore": {"current": 2},
                        "awayScore": {"current": 4},
                    },
                ]
            })
        return _FakeResponse({"events": []})

    monkeypatch.setattr(webapp_app.requests, "get", _fake_get)

    client = webapp_app.app.test_client()
    response = client.post("/api/settle-all", json={})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["bets_settled"] == 1

    updated = json.loads(my_selections_path.read_text())
    assert updated[0]["result"] == "won"


def test_api_settle_all_matches_common_team_abbreviations(tmp_path, monkeypatch):
    tracker_dir = tmp_path / "data" / "tracker"
    tracker_dir.mkdir(parents=True)

    my_selections_path = tracker_dir / "my_selections.json"
    my_selections_path.write_text(json.dumps([
        {
            "id": "sel-abbrev",
            "sport": "soccer",
            "team": "Man Utd",
            "match": "Man Utd vs Newcastle Utd",
            "odds": 1.95,
            "commence": "2026-04-27T19:00:00Z",
            "date": "2026-04-27",
            "result": None,
            "profit": None,
        },
    ], indent=2))
    manual_parlays_path = tracker_dir / "manual_parlays.json"
    manual_parlays_path.write_text("[]")

    monkeypatch.setattr(webapp_app, "BASE", tmp_path)
    monkeypatch.setattr(webapp_app, "_MY_SELECTIONS_FILE", my_selections_path)
    monkeypatch.setattr(webapp_app, "_MANUAL_PARLAYS_FILE", manual_parlays_path)
    monkeypatch.setattr(webapp_app, "_live_scores_cache", {"payload": None, "ts": 0.0})

    class _FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def _fake_get(url, headers=None, timeout=None):
        if "/sport/football/scheduled-events/2026-04-27" in url:
            return _FakeResponse({
                "events": [
                    {
                        "status": {"type": "finished"},
                        "homeTeam": {"name": "Manchester United"},
                        "awayTeam": {"name": "Newcastle United"},
                        "homeScore": {"current": 2},
                        "awayScore": {"current": 0},
                    },
                ]
            })
        return _FakeResponse({"events": []})

    monkeypatch.setattr(webapp_app.requests, "get", _fake_get)

    client = webapp_app.app.test_client()
    response = client.post("/api/settle-all", json={})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["bets_settled"] == 1

    updated = json.loads(my_selections_path.read_text())
    assert updated[0]["result"] == "won"


def test_api_settle_all_matches_common_soccer_nicknames(tmp_path, monkeypatch):
    tracker_dir = tmp_path / "data" / "tracker"
    tracker_dir.mkdir(parents=True)

    my_selections_path = tracker_dir / "my_selections.json"
    my_selections_path.write_text(json.dumps([
        {
            "id": "sel-psg",
            "sport": "soccer",
            "team": "PSG",
            "match": "PSG vs Spurs",
            "odds": 2.05,
            "commence": "2026-04-27T20:00:00Z",
            "date": "2026-04-27",
            "result": None,
            "profit": None,
        },
    ], indent=2))
    manual_parlays_path = tracker_dir / "manual_parlays.json"
    manual_parlays_path.write_text("[]")

    monkeypatch.setattr(webapp_app, "BASE", tmp_path)
    monkeypatch.setattr(webapp_app, "_MY_SELECTIONS_FILE", my_selections_path)
    monkeypatch.setattr(webapp_app, "_MANUAL_PARLAYS_FILE", manual_parlays_path)
    monkeypatch.setattr(webapp_app, "_live_scores_cache", {"payload": None, "ts": 0.0})

    class _FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def _fake_get(url, headers=None, timeout=None):
        if "/sport/football/scheduled-events/2026-04-27" in url:
            return _FakeResponse({
                "events": [
                    {
                        "status": {"type": "finished"},
                        "homeTeam": {"name": "Paris Saint Germain"},
                        "awayTeam": {"name": "Tottenham Hotspur"},
                        "homeScore": {"current": 3},
                        "awayScore": {"current": 1},
                    },
                ]
            })
        return _FakeResponse({"events": []})

    monkeypatch.setattr(webapp_app.requests, "get", _fake_get)

    client = webapp_app.app.test_client()
    response = client.post("/api/settle-all", json={})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["bets_settled"] == 1

    updated = json.loads(my_selections_path.read_text())
    assert updated[0]["result"] == "won"


def test_api_settle_all_falls_back_to_odds_scores_when_sofascore_is_empty(tmp_path, monkeypatch):
    tracker_dir = tmp_path / "data" / "tracker"
    tracker_dir.mkdir(parents=True)

    my_selections_path = tracker_dir / "my_selections.json"
    my_selections_path.write_text(json.dumps([
        {
            "id": "sel-fallback",
            "sport": "nhl",
            "team": "Pittsburgh Penguins",
            "match": "Pittsburgh Penguins vs Philadelphia Flyers",
            "odds": 2.10,
            "commence": "2026-04-27T23:00:00Z",
            "date": "2026-04-27",
            "result": None,
            "profit": None,
        },
    ], indent=2))
    manual_parlays_path = tracker_dir / "manual_parlays.json"
    manual_parlays_path.write_text("[]")

    monkeypatch.setattr(webapp_app, "BASE", tmp_path)
    monkeypatch.setattr(webapp_app, "_MY_SELECTIONS_FILE", my_selections_path)
    monkeypatch.setattr(webapp_app, "_MANUAL_PARLAYS_FILE", manual_parlays_path)
    monkeypatch.setattr(webapp_app, "_live_scores_cache", {"payload": None, "ts": 0.0})

    class _FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def _fake_get(url, headers=None, params=None, timeout=None):
        if "api.sofascore.com" in url:
            return _FakeResponse({"events": []})
        if "api.the-odds-api.com" in url:
            return _FakeResponse([
                {
                    "completed": True,
                    "home_team": "Philadelphia Flyers",
                    "away_team": "Pittsburgh Penguins",
                    "commence_time": "2026-04-27T23:00:00Z",
                    "scores": [
                        {"name": "Philadelphia Flyers", "score": "2"},
                        {"name": "Pittsburgh Penguins", "score": "4"},
                    ],
                }
            ])
        raise AssertionError(f"Unexpected URL {url}")

    monkeypatch.setattr(webapp_app.requests, "get", _fake_get)

    client = webapp_app.app.test_client()
    response = client.post("/api/settle-all", json={})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["bets_settled"] == 1
    assert "odds_api" in payload["score_sources"]

    updated = json.loads(my_selections_path.read_text())
    assert updated[0]["result"] == "won"


def test_api_settle_all_falls_back_to_espn_for_tennis(tmp_path, monkeypatch):
    tracker_dir = tmp_path / "data" / "tracker"
    tracker_dir.mkdir(parents=True)

    my_selections_path = tracker_dir / "my_selections.json"
    my_selections_path.write_text(json.dumps([
        {
            "id": "sel-tennis",
            "sport": "tennis",
            "team": "Jiri Lehecka",
            "match": "Jiri Lehecka vs Lorenzo Musetti",
            "odds": 2.46,
            "commence": "2026-04-28T09:00:00Z",
            "date": "2026-04-28",
            "result": None,
            "profit": None,
        },
    ], indent=2))
    manual_parlays_path = tracker_dir / "manual_parlays.json"
    manual_parlays_path.write_text("[]")

    monkeypatch.setattr(webapp_app, "_MY_SELECTIONS_FILE", my_selections_path)
    monkeypatch.setattr(webapp_app, "_MANUAL_PARLAYS_FILE", manual_parlays_path)
    monkeypatch.setattr(webapp_app, "_live_scores_cache", {"payload": None, "ts": 0.0})

    class _FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def _fake_get(url, headers=None, params=None, timeout=None):
        if "api.sofascore.com" in url:
            return _FakeResponse({"events": []})
        if "site.api.espn.com" in url and "/sports/tennis/atp/scoreboard" in url:
            return _FakeResponse({
                "events": [
                    {
                        "status": {"type": {"name": "STATUS_FINAL"}},
                        "competitions": [
                            {
                                "competitors": [
                                    {
                                        "athlete": {"displayName": "Jiri Lehecka"},
                                        "homeAway": "home",
                                        "score": "2",
                                        "winner": True,
                                    },
                                    {
                                        "athlete": {"displayName": "Lorenzo Musetti"},
                                        "homeAway": "away",
                                        "score": "1",
                                        "winner": False,
                                    },
                                ]
                            }
                        ],
                    }
                ]
            })
        if "api.the-odds-api.com" in url:
            return _FakeResponse([])
        raise AssertionError(f"Unexpected URL {url}")

    monkeypatch.setattr(webapp_app.requests, "get", _fake_get)

    client = webapp_app.app.test_client()
    response = client.post("/api/settle-all", json={})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["bets_settled"] == 1
    assert "espn" in payload["score_sources"]

    updated = json.loads(my_selections_path.read_text())
    assert updated[0]["result"] == "won"


def test_api_settle_all_also_settles_pending_system_book_rows(tmp_path, monkeypatch):
    import settle as settle_mod

    tracker_dir = tmp_path / "data" / "tracker"
    tracker_dir.mkdir(parents=True)

    (tracker_dir / "my_selections.json").write_text("[]")
    (tracker_dir / "manual_parlays.json").write_text("[]")

    predictions = pd.DataFrame([
        {
            "pred_id": "sys-1",
            "recorded_at": "2026-04-27T20:00:00Z",
            "commence_time": "2026-04-27T23:00:00Z",
            "sport": "nhl",
            "match_id": "Pittsburgh Penguins vs Philadelphia Flyers",
            "team_or_player": "Pittsburgh Penguins",
            "market": "moneyline",
            "market_status": "experimental",
            "tier": "Limited",
            "bookmaker": "Betfair",
            "bet_odds": 2.10,
            "edge": 0.09,
            "stake_units": 1.0,
            "status": "pending",
            "is_parlay_leg": False,
            "ml_prob": 0.58,
            "fair_prob": 0.49,
            "kelly_stake_pct": 1.2,
        }
    ])
    predictions.to_parquet(tracker_dir / "predictions.parquet", index=False)

    monkeypatch.setattr(webapp_app, "BASE", tmp_path)
    monkeypatch.setattr(webapp_app, "_MY_SELECTIONS_FILE", tracker_dir / "my_selections.json")
    monkeypatch.setattr(webapp_app, "_MANUAL_PARLAYS_FILE", tracker_dir / "manual_parlays.json")
    monkeypatch.setattr(webapp_app, "_live_scores_cache", {"payload": None, "ts": 0.0})
    monkeypatch.setattr(settle_mod, "_fetch_closing_odds", lambda *args, **kwargs: 1.95)

    class _FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def _fake_get(url, headers=None, params=None, timeout=None):
        if "api.sofascore.com" in url and "/sport/ice-hockey/scheduled-events/2026-04-27" in url:
            return _FakeResponse({
                "events": [
                    {
                        "status": {"type": "finished"},
                        "homeTeam": {"name": "Philadelphia Flyers"},
                        "awayTeam": {"name": "Pittsburgh Penguins"},
                        "homeScore": {"current": 2},
                        "awayScore": {"current": 4},
                    },
                ]
            })
        if "api.sofascore.com" in url:
            return _FakeResponse({"events": []})
        raise AssertionError(f"Unexpected URL {url}")

    monkeypatch.setattr(webapp_app.requests, "get", _fake_get)

    client = webapp_app.app.test_client()
    response = client.post("/api/settle-all", json={})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["bets_settled"] == 1
    assert payload["still_pending"] == 0

    updated_pred = pd.read_parquet(tracker_dir / "predictions.parquet")
    assert updated_pred.empty

    settled = pd.read_parquet(tracker_dir / "settled.parquet")
    assert list(settled["pred_id"]) == ["sys-1"]
    assert bool(settled.iloc[0]["won"]) is True
    assert settled.iloc[0]["closing_odds"] == 1.95
    assert round(float(settled.iloc[0]["clv"]), 6) == round((2.10 / 1.95) - 1.0, 6)
    assert settled.iloc[0]["tier"] == "Limited"
