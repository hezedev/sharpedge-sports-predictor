"""
Basketball data fetcher using BallDontLie API v1.

BallDontLie provides free NBA game data with cursor-based pagination.
Free tier is strictly rate-limited (~1 req/s), so we:
  - Sleep 1.5 s between every page (not just on 429)
  - Use a persistent disk cache per season so partial results survive restarts
  - On 429: exponential backoff starting at 10 s (up to 5 retries)
  - Save what we have if retries are exhausted rather than discarding

API docs: https://www.balldontlie.io/
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_BASE_URL        = "https://api.balldontlie.io/v1"
_DEFAULT_SEASONS = [2022, 2023, 2024, 2025]   # 4 seasons including current 2025-26
_PER_PAGE        = 100
_PAGE_SLEEP      = 2.0     # seconds between every page request (free tier ~1 req/s)
_RETRY_BASE      = 15.0    # first 429 backoff in seconds
_MAX_RETRIES     = 5
# Disk cache: stores completed (season → list[dict]) so we never re-fetch finished seasons
_CACHE_DIR       = Path("data/cache/bdl_seasons")


class BallDontLieFetcher:
    """
    Fetches NBA historical game data from BallDontLie API.

    Produces the same column schema as the original BasketballFetcher:
        match_id, date, league_id, league_name, season,
        home_team, home_team_id, away_team, away_team_id,
        home_score, away_score, result,
        home_q1..q4, home_ot, away_q1..q4, away_ot, status
    """

    def __init__(
        self,
        seasons: Optional[List[int]] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self._seasons = seasons or _DEFAULT_SEASONS
        self._api_key = api_key or os.environ.get("BALLDONTLIE_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "BALLDONTLIE_API_KEY not set. "
                "Get a free key at https://www.balldontlie.io/"
            )
        self._session = requests.Session()
        self._session.headers.update({"Authorization": self._api_key})
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_all_seasons(self) -> pd.DataFrame:
        """
        Fetch all finished NBA games for configured seasons.
        Seasons already on disk are loaded from cache; only missing ones hit the API.
        """
        all_games: List[dict] = []
        for season in self._seasons:
            cache_file = _CACHE_DIR / f"season_{season}.json"

            if cache_file.exists():
                try:
                    games = json.loads(cache_file.read_text())
                    logger.info(
                        "[BallDontLie] Season %d → loaded %d games from disk cache",
                        season, len(games),
                    )
                    all_games.extend(games)
                    continue
                except Exception:
                    logger.warning("[BallDontLie] Corrupted cache for season %d — re-fetching", season)

            logger.info("[BallDontLie] Fetching season %d from API …", season)
            games = self._fetch_season(season)
            logger.info("[BallDontLie] Season %d → %d finished games", season, len(games))

            if games:
                # Only cache if we got a reasonably complete season (> 400 games)
                if len(games) >= 400:
                    cache_file.write_text(json.dumps(games))
                    logger.info("[BallDontLie] Season %d saved to disk cache", season)
                else:
                    logger.warning(
                        "[BallDontLie] Season %d only has %d games (expected ~1230) — "
                        "not caching, will retry next run",
                        season, len(games),
                    )

            all_games.extend(games)

        if not all_games:
            logger.warning("[BallDontLie] No games fetched.")
            return pd.DataFrame()

        df = pd.DataFrame(all_games)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        logger.info("[BallDontLie] Total games: %d across %d seasons", len(df), len(self._seasons))
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_season(self, season: int) -> List[dict]:
        """Paginate through all games for a single NBA season."""
        games: List[dict] = []
        cursor: Optional[int] = None

        while True:
            params: dict = {"seasons[]": season, "per_page": _PER_PAGE}
            if cursor is not None:
                params["cursor"] = cursor

            data = self._get_with_retry(f"{_BASE_URL}/games", params, season, cursor)
            if data is None:
                # Rate limit exhausted — save what we have and stop
                logger.warning(
                    "[BallDontLie] Stopping season %d early — keeping %d games so far",
                    season, len(games),
                )
                break

            for game in data.get("data", []):
                parsed = self._parse_game(game)
                if parsed:
                    games.append(parsed)

            meta        = data.get("meta", {})
            next_cursor = meta.get("next_cursor")

            if not next_cursor or next_cursor == cursor:
                break  # last page reached

            cursor = next_cursor
            time.sleep(_PAGE_SLEEP)   # be polite to the free tier

        return games

    def _get_with_retry(
        self,
        url: str,
        params: dict,
        season: int,
        cursor: Optional[int],
        max_retries: int = _MAX_RETRIES,
    ) -> Optional[dict]:
        """GET with exponential backoff on 429.  First wait = _RETRY_BASE seconds."""
        for attempt in range(max_retries):
            try:
                resp = self._session.get(url, params=params, timeout=20)
                if resp.status_code == 429:
                    wait = _RETRY_BASE * (2 ** attempt)
                    logger.warning(
                        "[BallDontLie] Rate limited (season=%d, cursor=%s) — "
                        "waiting %.0fs (attempt %d/%d)",
                        season, cursor, wait, attempt + 1, max_retries,
                    )
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.HTTPError as exc:
                logger.error("[BallDontLie] HTTP error (season=%d): %s", season, exc)
                return None
            except Exception as exc:
                logger.error("[BallDontLie] Request error (season=%d): %s", season, exc)
                return None

        logger.error("[BallDontLie] Gave up after %d retries (season=%d)", max_retries, season)
        return None

    def _parse_game(self, game: dict) -> Optional[dict]:
        """
        Parse a single BallDontLie game dict into our standard schema.

        BallDontLie uses 'visitor_team' where we use 'away_team'.
        Only "Final" or overtime-final games are included.
        """
        status = game.get("status", "")
        # Accept "Final" and OT variants e.g. "Final/OT"
        if not (status == "Final" or (isinstance(status, str) and status.startswith("Final"))):
            return None

        home = game.get("home_team", {})
        away = game.get("visitor_team", {})

        home_score = game.get("home_team_score")
        away_score = game.get("visitor_team_score")

        if home_score is None or away_score is None:
            return None

        try:
            home_score = int(home_score)
            away_score = int(away_score)
        except (ValueError, TypeError):
            return None

        result = "home_win" if home_score > away_score else "away_win"

        # Overtime: sum ot1+ot2+ot3 if present
        def _ot_total(prefix: str) -> Optional[int]:
            vals = [game.get(f"{prefix}_ot{i}") for i in range(1, 4)]
            valid = [v for v in vals if v is not None]
            return sum(valid) if valid else None

        home_ot = _ot_total("home")
        away_ot = _ot_total("visitor")

        return {
            "match_id":     game.get("id"),
            "date":         game.get("date"),
            "league_id":    1,                       # NBA
            "league_name":  "NBA",
            "season":       game.get("season"),
            "home_team":    home.get("full_name"),
            "home_team_id": home.get("id"),
            "away_team":    away.get("full_name"),
            "away_team_id": away.get("id"),
            "home_score":   home_score,
            "away_score":   away_score,
            "result":       result,
            "home_q1":      game.get("home_q1"),
            "home_q2":      game.get("home_q2"),
            "home_q3":      game.get("home_q3"),
            "home_q4":      game.get("home_q4"),
            "home_ot":      home_ot,
            "away_q1":      game.get("visitor_q1"),
            "away_q2":      game.get("visitor_q2"),
            "away_q3":      game.get("visitor_q3"),
            "away_q4":      game.get("visitor_q4"),
            "away_ot":      away_ot,
            "status":       status,
        }
