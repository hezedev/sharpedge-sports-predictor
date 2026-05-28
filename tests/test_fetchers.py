"""
Tests for data fetcher modules.

Uses the `responses` library to mock HTTP requests, so no
actual API calls are made during testing.
"""

import json
from datetime import datetime

import pandas as pd
import pytest
import responses

from src.data.soccer_fetcher import SoccerFetcher
from src.data.basketball_fetcher import BasketballFetcher
from src.data.tennis_fetcher import TennisFetcher
from src.data.odds_fetcher import OddsFetcher
from src.data.api_football_enricher import APIFootballEnricher


# ------------------------------------------------------------------
# Soccer Fetcher Tests
# ------------------------------------------------------------------


class TestSoccerFetcher:
    """Test suite for SoccerFetcher."""

    MOCK_MATCHES = {
        "matches": [
            {
                "id": 1,
                "utcDate": "2024-09-15T15:00:00Z",
                "matchday": 5,
                "status": "FINISHED",
                "homeTeam": {"id": 100, "name": "Arsenal"},
                "awayTeam": {"id": 200, "name": "Chelsea"},
                "score": {
                    "fullTime": {"home": 2, "away": 1},
                    "halfTime": {"home": 1, "away": 0},
                },
                "season": {"startDate": "2024-08-01"},
                "competition": {"code": "PL"},
            },
            {
                "id": 2,
                "utcDate": "2024-09-16T19:30:00Z",
                "matchday": 5,
                "status": "FINISHED",
                "homeTeam": {"id": 300, "name": "Liverpool"},
                "awayTeam": {"id": 400, "name": "Man City"},
                "score": {
                    "fullTime": {"home": 1, "away": 1},
                    "halfTime": {"home": 0, "away": 1},
                },
                "season": {"startDate": "2024-08-01"},
                "competition": {"code": "PL"},
            },
        ]
    }

    MOCK_API_SPORTS_LEAGUES = {
        "response": [
            {
                "league": {"id": 98, "name": "J-League"},
                "country": {"name": "Japan"},
                "seasons": [{"year": 2026}],
            }
        ]
    }

    MOCK_API_SPORTS_FIXTURES = {
        "response": [
            {
                "fixture": {"id": 5001, "date": "2026-04-29T10:00:00+00:00"},
                "league": {"id": 98, "name": "J-League", "season": 2026, "round": "Regular Season - 12"},
                "teams": {
                    "home": {"id": 1, "name": "Tokyo Verdy"},
                    "away": {"id": 2, "name": "Kashima Antlers"},
                },
                "goals": {"home": 1, "away": 0},
                "score": {"halftime": {"home": 1, "away": 0}},
            }
        ]
    }

    class _MockResponse:
        def __init__(self, payload: dict, status_code: int = 200) -> None:
            self._payload = payload
            self.status_code = status_code

        def json(self) -> dict:
            return self._payload

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"mock http error {self.status_code}")

    def test_fetch_matches_parses_correctly(self, monkeypatch) -> None:
        """Test that matches are fetched and parsed into correct DataFrame."""
        fetcher = SoccerFetcher(cache_expire_hours=0)
        monkeypatch.setattr(
            fetcher._cache,
            "get",
            lambda *args, **kwargs: self._MockResponse(self.MOCK_MATCHES),
        )
        df = fetcher.fetch_matches(season="2024", competition="PL")

        assert len(df) == 2
        assert "match_id" in df.columns
        assert "result" in df.columns
        assert df.iloc[0]["result"] == "home_win"
        assert df.iloc[1]["result"] == "draw"
        assert df.iloc[0]["home_goals"] == 2
        assert df.iloc[0]["away_goals"] == 1

    def test_fetch_standings(self, monkeypatch) -> None:
        """Test standings parsing."""
        mock_standings = {
            "standings": [
                {
                    "type": "TOTAL",
                    "table": [
                        {
                            "position": 1,
                            "team": {"id": 100, "name": "Arsenal"},
                            "playedGames": 10,
                            "won": 8,
                            "draw": 1,
                            "lost": 1,
                            "goalsFor": 25,
                            "goalsAgainst": 8,
                            "goalDifference": 17,
                            "points": 25,
                        }
                    ],
                }
            ]
        }

        fetcher = SoccerFetcher(cache_expire_hours=0)
        monkeypatch.setattr(
            fetcher._cache,
            "get",
            lambda *args, **kwargs: self._MockResponse(mock_standings),
        )
        df = fetcher.fetch_standings(season="2024", competition="PL")

        assert len(df) == 1
        assert df.iloc[0]["team_name"] == "Arsenal"
        assert df.iloc[0]["points"] == 25

    def test_fetch_matches_supports_api_sports_extra_competition(self, monkeypatch) -> None:
        fetcher = SoccerFetcher(cache_expire_hours=0)
        monkeypatch.setattr(fetcher, "_api_sports_key", "test-key")
        monkeypatch.setattr(
            fetcher._cache,
            "get",
            lambda *args, **kwargs: self._MockResponse(self.MOCK_API_SPORTS_FIXTURES),
        )
        df = fetcher.fetch_matches(season="2025", competition="soccer_japan_j_league")

        assert len(df) == 1
        assert df.iloc[0]["competition"] == "soccer_japan_j_league"
        assert df.iloc[0]["season"] == "2026"
        assert df.iloc[0]["home_team"] == "Tokyo Verdy"
        assert df.iloc[0]["result"] == "home_win"

    def test_fetch_matches_pauses_api_sports_after_429(self, monkeypatch) -> None:
        import requests

        fetcher = SoccerFetcher(cache_expire_hours=0)
        monkeypatch.setattr(fetcher, "_api_sports_key", "test-key")
        monkeypatch.setattr(fetcher, "_resolve_api_sports_league_id", lambda *args, **kwargs: 123)

        class _TooManyRequests:
            status_code = 429

            def __init__(self) -> None:
                self.response = type("Resp", (), {"status_code": 429})()

        def _raise_429(*args, **kwargs):
            raise requests.exceptions.HTTPError("429 Rate Limited", response=_TooManyRequests().response)

        monkeypatch.setattr(fetcher, "_get", _raise_429)

        df = fetcher.fetch_matches(season="2025", competition="soccer_japan_j_league")

        assert df.empty
        assert fetcher._api_sports_is_disabled() is True
        assert "429" in fetcher._api_sports_disabled_reason

    def test_fetch_matches_skips_api_sports_when_temporarily_disabled(self, monkeypatch) -> None:
        fetcher = SoccerFetcher(cache_expire_hours=0)
        monkeypatch.setattr(fetcher, "_api_sports_key", "test-key")
        fetcher._disable_api_sports(hours=2, reason="429 Too Many Requests from API-Sports fixtures")

        called = {"count": 0}

        def _should_not_call(*args, **kwargs):
            called["count"] += 1
            return 123

        monkeypatch.setattr(fetcher, "_resolve_api_sports_league_id", _should_not_call)

        df = fetcher.fetch_matches(season="2025", competition="soccer_japan_j_league")

        assert df.empty
        assert called["count"] == 0


class TestAPIFootballEnricher:
    def test_api_football_prefers_direct_api_sports_key_when_both_exist(self, monkeypatch) -> None:
        monkeypatch.setenv("API_SPORTS_KEY", "api-sports-key")
        monkeypatch.setenv("RAPIDAPI_KEY", "rapidapi-key")

        enricher = APIFootballEnricher(cache_expire_hours=0)

        assert enricher.api_key == "api-sports-key"
        assert enricher.provider == "api_sports"
        assert enricher.base_url == "https://v3.football.api-sports.io"
        assert enricher.headers["x-apisports-key"] == "api-sports-key"

    def test_api_football_falls_back_to_direct_api_sports_key(self, monkeypatch) -> None:
        monkeypatch.setenv("API_SPORTS_KEY", "api-sports-key")
        monkeypatch.delenv("RAPIDAPI_KEY", raising=False)

        enricher = APIFootballEnricher(cache_expire_hours=0)

        assert enricher.api_key == "api-sports-key"
        assert enricher.provider == "api_sports"
        assert enricher.base_url == "https://v3.football.api-sports.io"
        assert enricher.headers["x-apisports-key"] == "api-sports-key"

    def test_api_football_falls_back_to_rapidapi_key(self, monkeypatch) -> None:
        monkeypatch.delenv("API_SPORTS_KEY", raising=False)
        monkeypatch.setenv("RAPIDAPI_KEY", "rapidapi-key")

        enricher = APIFootballEnricher(cache_expire_hours=0)

        assert enricher.api_key == "rapidapi-key"
        assert enricher.provider == "rapidapi"
        assert enricher.base_url == "https://api-football-v1.p.rapidapi.com/v3"
        assert enricher.headers["X-RapidAPI-Key"] == "rapidapi-key"

    def test_team_lookup_skips_network_when_temporarily_disabled(self, monkeypatch) -> None:
        enricher = APIFootballEnricher(cache_expire_hours=0)
        monkeypatch.setattr(enricher, "api_key", "test-key")
        enricher._temporarily_disable(hours=1, reason="403 Forbidden from API-Football")

        called = {"count": 0}

        def _should_not_call(*args, **kwargs):
            called["count"] += 1
            raise AssertionError("team lookup should not hit the network while disabled")

        monkeypatch.setattr(enricher._cache, "get", _should_not_call)

        assert enricher._get_team_id("Bayern Munich") is None
        assert called["count"] == 0

    def test_candidate_fixture_lookup_stops_after_provider_pause(self, monkeypatch) -> None:
        enricher = APIFootballEnricher(cache_expire_hours=0)
        monkeypatch.setattr(enricher, "api_key", "test-key")

        calls = {"teams": 0, "fixtures": 0}

        def _fake_get_json(path, params=None):
            if path == "teams":
                calls["teams"] += 1
            if path == "fixtures":
                calls["fixtures"] += 1
                enricher._temporarily_disable(hours=1, reason="429 Too Many Requests from API-Football")
                raise RuntimeError("429 Too Many Requests from API-Football")
            return {"response": []}

        monkeypatch.setattr(enricher, "_get_json", _fake_get_json)

        fixtures = enricher._candidate_fixtures("Bayern Munich", datetime.fromisoformat("2026-05-22T19:00:00+00:00"))

        assert fixtures == []
        assert calls["fixtures"] == 1
        assert calls["teams"] == 0


# ------------------------------------------------------------------
# Basketball Fetcher Tests
# ------------------------------------------------------------------


class TestBasketballFetcher:
    """Test suite for BasketballFetcher."""

    MOCK_GAMES = {
        "response": [
            {
                "id": 1001,
                "date": "2024-11-01T00:00:00+00:00",
                "status": {"long": "Game Finished"},
                "league": {"id": 12, "name": "NBA", "season": "2024-2025"},
                "teams": {
                    "home": {"id": 50, "name": "Lakers"},
                    "away": {"id": 60, "name": "Celtics"},
                },
                "scores": {
                    "home": {
                        "quarter_1": 28, "quarter_2": 30,
                        "quarter_3": 25, "quarter_4": 22,
                        "over_time": None, "total": 105,
                    },
                    "away": {
                        "quarter_1": 30, "quarter_2": 27,
                        "quarter_3": 28, "quarter_4": 25,
                        "over_time": None, "total": 110,
                    },
                },
            }
        ]
    }

    @responses.activate
    def test_fetch_games(self) -> None:
        """Test basketball game fetching."""
        responses.add(
            responses.GET,
            "https://v1.basketball.api-sports.io/games",
            json=self.MOCK_GAMES,
            status=200,
        )

        fetcher = BasketballFetcher(cache_expire_hours=0)
        df = fetcher.fetch_matches(season="2024-2025", league_id=12)

        assert len(df) == 1
        assert df.iloc[0]["result"] == "away_win"
        assert df.iloc[0]["home_score"] == 105
        assert df.iloc[0]["away_score"] == 110


# ------------------------------------------------------------------
# Odds Fetcher Tests
# ------------------------------------------------------------------


class TestOddsFetcher:
    """Test suite for OddsFetcher."""

    MOCK_ODDS = [
        {
            "id": "event_1",
            "commence_time": "2024-12-01T15:00:00Z",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "bookmakers": [
                {
                    "title": "Bet365",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Arsenal", "price": 1.85},
                                {"name": "Draw", "price": 3.50},
                                {"name": "Chelsea", "price": 4.20},
                            ],
                        }
                    ],
                },
                {
                    "title": "Pinnacle",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Arsenal", "price": 1.90},
                                {"name": "Draw", "price": 3.40},
                                {"name": "Chelsea", "price": 4.00},
                            ],
                        }
                    ],
                },
            ],
        }
    ]

    @responses.activate
    def test_fetch_and_best_odds(self) -> None:
        """Test odds fetching and best price selection."""
        responses.add(
            responses.GET,
            "https://api.the-odds-api.com/v4/sports/soccer_epl/odds",
            json=self.MOCK_ODDS,
            status=200,
        )

        fetcher = OddsFetcher(sport="soccer", cache_expire_hours=0)
        df = fetcher.fetch_odds(sport_key="soccer_epl")

        assert len(df) == 6  # 3 outcomes × 2 bookmakers
        assert "price" in df.columns

        best = fetcher.get_best_odds(df)
        arsenal_best = best[best["outcome"] == "Arsenal"]
        assert arsenal_best.iloc[0]["price"] == 1.90  # Pinnacle has better odds

    @responses.activate
    def test_consensus_odds(self) -> None:
        """Test consensus odds calculation with vig removal."""
        responses.add(
            responses.GET,
            "https://api.the-odds-api.com/v4/sports/soccer_epl/odds",
            json=self.MOCK_ODDS,
            status=200,
        )

        fetcher = OddsFetcher(sport="soccer", cache_expire_hours=0)
        df = fetcher.fetch_odds(sport_key="soccer_epl")
        consensus = fetcher.get_consensus_odds(df)

        assert "fair_prob" in consensus.columns
        # Fair probabilities should sum close to 1.0
        fair_sum = consensus["fair_prob"].sum()
        assert abs(fair_sum - 1.0) < 0.01


def test_api_football_fixture_lookup_uses_date_search_and_preserves_league(monkeypatch) -> None:
    enricher = APIFootballEnricher()
    enricher.api_key = "test-key"
    monkeypatch.setattr(enricher._rate_limiter, "allow_request", lambda: True)
    monkeypatch.setattr(enricher, "_get_team_id", lambda team_name: None)

    def _fake_get_json(path, params=None):
        assert path == "fixtures"
        if params and params.get("date") == "2026-05-23":
            return {
                "response": [
                    {
                        "fixture": {
                            "id": 42,
                            "date": "2026-05-23T18:00:00+00:00",
                            "status": {"short": "NS"},
                        },
                        "teams": {"home": {"name": "Alpha FC"}, "away": {"name": "Beta FC"}},
                        "league": {"name": "Bundesliga", "round": "Regular Season - 34"},
                    }
                ]
            }
        return {"response": []}

    monkeypatch.setattr(enricher, "_get_json", _fake_get_json)

    fixture = enricher._find_fixture("Alpha FC", "Beta FC", "2026-05-23T18:00:00Z")

    assert fixture is not None
    assert fixture["id"] == 42
    assert fixture["league"]["name"] == "Bundesliga"
    assert fixture["teams"]["away"]["name"] == "Beta FC"
