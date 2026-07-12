from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from src.data.source_registry import source_status_summary
from src.markets import environment
from src.markets.availability import _MLBLiveAvailabilityEnricher, build_availability_context


def test_mlb_stats_api_enricher_parses_pitchers_stats_and_lineups(monkeypatch) -> None:
    enricher = _MLBLiveAvailabilityEnricher()

    def fake_get_json(path, params):
        if path == "schedule":
            return {
                "dates": [
                    {
                        "games": [
                            {
                                "gamePk": 12345,
                                "gameDate": "2026-06-01T23:05:00Z",
                                "status": {"detailedState": "Scheduled"},
                                "venue": {"name": "Yankee Stadium"},
                                "teams": {
                                    "home": {
                                        "team": {"name": "New York Yankees"},
                                        "probablePitcher": {
                                            "id": 10,
                                            "fullName": "Gerrit Cole",
                                            "pitchHand": {"code": "R"},
                                        },
                                    },
                                    "away": {
                                        "team": {"name": "Boston Red Sox"},
                                        "probablePitcher": {
                                            "id": 20,
                                            "fullName": "Brayan Bello",
                                            "pitchHand": {"code": "R"},
                                        },
                                    },
                                },
                            }
                        ]
                    }
                ]
            }
        if path == "people/10/stats":
            return {"stats": [{"splits": [{"stat": {"era": "2.91", "whip": "0.97", "strikeoutsPer9Inn": "10.2", "walksPer9Inn": "2.1", "inningsPitched": "86.1", "gamesStarted": "14"}}]}]}
        if path == "people/20/stats":
            return {"stats": [{"splits": [{"stat": {"era": "3.84", "whip": "1.18", "strikeoutsPer9Inn": "8.8", "walksPer9Inn": "2.8", "inningsPitched": "75.0", "gamesStarted": "13"}}]}]}
        if path == "game/12345/feed/live":
            players = {f"ID{i}": {"person": {"fullName": f"Home Batter {i}"}} for i in range(1, 10)}
            players.update({f"ID{i}": {"person": {"fullName": f"Away Batter {i}"}} for i in range(11, 20)})
            return {
                "liveData": {
                    "boxscore": {
                        "teams": {
                            "home": {"battingOrder": list(range(1, 10)), "players": players},
                            "away": {"battingOrder": list(range(11, 20)), "players": players},
                        }
                    }
                }
            }
        raise AssertionError(f"unexpected MLB path {path}")

    monkeypatch.setattr(enricher, "_get_json", fake_get_json)

    context = enricher.fetch_match_availability(
        home_team="NY Yankees",
        away_team="Boston Red Sox",
        commence="2026-06-01T23:05:00Z",
    )

    assert context["availability_source"] == "mlb_stats_api"
    assert context["venue_name"] == "Yankee Stadium"
    assert context["home_starter_name"] == "Gerrit Cole"
    assert context["home_starter_hand"] == "R"
    assert context["home_starter_era"] == 2.91
    assert context["away_starter_whip"] == 1.18
    assert context["home_lineup_confirmed"] == 1
    assert context["away_likely_starters_count"] == 9


def test_mlb_availability_context_detects_probable_pitcher_change(monkeypatch) -> None:
    class FakeEnricher:
        def fetch_match_availability(self, home_team, away_team, commence):
            return {
                "availability_source": "mlb_stats_api",
                "home_starter_confirmed": 1,
                "away_starter_confirmed": 1,
                "home_starter_name": "Gerrit Cole",
                "away_starter_name": "Brayan Bello",
            }

    monkeypatch.setattr("src.markets.availability._mlb_availability", lambda: FakeEnricher())
    monkeypatch.delenv("SCAN_LEAN_CONTEXT", raising=False)

    context = build_availability_context(
        "mlb",
        {"home_team": "New York Yankees", "away_team": "Boston Red Sox", "commence_time": "2026-06-01T23:05:00Z"},
        pd.Series({"home_starter_name": "Carlos Rodon", "away_starter_name": "Brayan Bello"}),
    )

    assert context["pitcher_change_detected"] == 1
    assert context["home_pitcher_changed"] == 1
    assert "Carlos Rodon" in context["pitcher_change_note"]


def test_open_meteo_keyless_weather_fallback(monkeypatch) -> None:
    match_dt = datetime.now(timezone.utc) + timedelta(days=2)
    monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)
    monkeypatch.setattr(environment, "_lookup", lambda team, sport: ("Example Park", 40.0, -73.0))

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            hour = match_dt.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M")
            return {
                "hourly": {
                    "time": [hour],
                    "temperature_2m": [91.0],
                    "wind_speed_10m": [16.2],
                    "precipitation": [0.1],
                }
            }

    monkeypatch.setattr(environment.requests, "get", lambda *args, **kwargs: FakeResponse())

    context = environment.build_environment_context("mlb", "New York Yankees", "Boston Red Sox", match_dt.isoformat())

    assert context["outdoor_weather_source"] == "open_meteo"
    assert context["temperature_f"] == 91.0
    assert context["weather_risk"] == 1


def test_mlb_environment_adds_static_park_factor_without_weather(monkeypatch) -> None:
    monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)
    monkeypatch.setattr(environment, "_lookup", lambda team, sport: None)

    context = environment.build_environment_context(
        "mlb",
        "Cincinnati Reds",
        "Chicago Cubs",
        (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
    )

    assert context["park_factor_source"] == "static_mlb_park_proxy"
    assert context["park_factor_proxy"] == 1.08
    assert context["park_run_environment"] == "hitter_friendly"


def test_source_registry_marks_keyless_mlb_sources_configured() -> None:
    summary = source_status_summary(env={})
    providers = {item["key"]: item for item in summary["providers"]}

    assert providers["mlb_stats_api"]["configured"] is True
    assert providers["open_meteo"]["configured"] is True
    assert "mlb" not in summary["missing_critical"]
