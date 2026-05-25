"""
Basketball data fetcher using API-Sports (v1.basketball).

Fetches NBA game results, standings, and team/player statistics.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from config import settings
from src.data.base_fetcher import BaseFetcher
from src.utils.helpers import RateLimiter, parse_date

logger = logging.getLogger(__name__)


class BasketballFetcher(BaseFetcher):
    """
    Fetcher for basketball game data from API-Sports.

    Free tier: 100 requests/day with caching to maximize coverage.
    """

    def __init__(self, cache_expire_hours: int = 24) -> None:
        super().__init__(sport="basketball", cache_expire_hours=cache_expire_hours)

        api_cfg = settings.get("apis", {}).get("api_sports", {})
        self._base_url = api_cfg.get(
            "basketball_url", "https://v1.basketball.api-sports.io"
        )
        self._api_key = os.environ.get("API_SPORTS_KEY", "")
        self._leagues = self._sport_cfg.get("leagues", [12])  # 12 = NBA
        self._rate_limiter = RateLimiter(
            max_calls=api_cfg.get("rate_limit_per_day", 100),
            period_seconds=86400,
        )

        if not self._api_key:
            logger.warning("API_SPORTS_KEY not set. Basketball API calls will fail.")

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "x-apisports-key": self._api_key,
            "x-rapidapi-host": "v1.basketball.api-sports.io",
        }

    # ------------------------------------------------------------------
    # Core fetch methods
    # ------------------------------------------------------------------

    def fetch_matches(
        self,
        season: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        league_id: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch basketball game results.

        Parameters
        ----------
        season : str, optional
            Season year (e.g. '2024-2025' or '2024').
        date_from : str, optional
            Fetch games on this specific date (YYYY-MM-DD).
        date_to : str, optional
            Not directly supported; use date_from for day-by-day.
        league_id : int, optional
            Specific league ID. Defaults to configured leagues.

        Returns
        -------
        pd.DataFrame
            Standardized game data.
        """
        leagues = [league_id] if league_id else self._leagues
        all_games: List[pd.DataFrame] = []

        for lid in leagues:
            logger.info("Fetching basketball games: league=%d, season=%s", lid, season)

            params: Dict[str, Any] = {"league": lid}
            if season:
                # API-Sports basketball uses season format like "2024-2025"
                params["season"] = season
            if date_from:
                params["date"] = date_from

            try:
                url = f"{self._base_url}/games"
                raw = self._get(
                    url=url,
                    headers=self._headers,
                    params=params,
                    rate_limiter=self._rate_limiter,
                )

                games_raw = raw.get("response", [])
                if not games_raw:
                    logger.info("No games found for league %d season %s", lid, season)
                    continue

                df = self._parse_matches(games_raw, league_id=lid)
                all_games.append(df)
                logger.info("Fetched %d games for league %d", len(df), lid)

            except Exception as exc:
                logger.error("Error fetching league %d: %s", lid, exc)
                continue

        if not all_games:
            return pd.DataFrame()

        return pd.concat(all_games, ignore_index=True)

    def fetch_standings(
        self,
        season: Optional[str] = None,
        league_id: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch basketball league standings.

        Parameters
        ----------
        season : str, optional
            Season identifier.
        league_id : int, optional
            League ID (default: first configured league).

        Returns
        -------
        pd.DataFrame
            Standings with team, wins, losses, conference, etc.
        """
        lid = league_id or self._leagues[0]
        params: Dict[str, Any] = {"league": lid, "stage": "NBA"}
        if season:
            params["season"] = season

        url = f"{self._base_url}/standings"
        raw = self._get(
            url=url,
            headers=self._headers,
            params=params,
            rate_limiter=self._rate_limiter,
        )

        standings_raw = raw.get("response", [])
        if not standings_raw:
            return pd.DataFrame()

        rows = []
        for group in standings_raw:
            for entry in group if isinstance(group, list) else [group]:
                team = entry.get("team", {})
                games = entry.get("games", {})
                win_data = games.get("win", {})
                lose_data = games.get("lose", {})
                points = entry.get("points", {})

                rows.append({
                    "position": entry.get("position"),
                    "team_id": team.get("id"),
                    "team_name": team.get("name"),
                    "group": entry.get("group", {}).get("name"),
                    "games_played": games.get("played"),
                    "wins": win_data.get("total"),
                    "win_pct": win_data.get("percentage"),
                    "losses": lose_data.get("total"),
                    "points_for": points.get("for"),
                    "points_against": points.get("against"),
                    "league_id": lid,
                    "season": season,
                })

        return pd.DataFrame(rows)

    def fetch_team_statistics(
        self,
        team_id: int,
        season: str,
        league_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Fetch aggregated team statistics for a season.

        Parameters
        ----------
        team_id : int
            API-Sports team ID.
        season : str
            Season identifier.
        league_id : int, optional
            League ID.

        Returns
        -------
        dict
            Team statistics (PPG, rebounds, assists, etc.).
        """
        lid = league_id or self._leagues[0]
        url = f"{self._base_url}/statistics"
        params = {"team": team_id, "season": season, "league": lid}

        raw = self._get(
            url=url,
            headers=self._headers,
            params=params,
            rate_limiter=self._rate_limiter,
        )
        return raw.get("response", {})

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_matches(
        self,
        raw_data: Any,
        league_id: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Parse raw API-Sports basketball response into standardized DataFrame.

        Parameters
        ----------
        raw_data : list[dict]
            List of game dictionaries.
        league_id : int, optional
            League ID to tag rows with.

        Returns
        -------
        pd.DataFrame
            Columns: match_id, date, league_id, season, home_team,
                     home_team_id, away_team, away_team_id, home_score,
                     away_score, result, home_q1..q4, away_q1..q4, status.
        """
        rows = []
        for game in raw_data:
            status_info = game.get("status", {})
            status = status_info.get("long", "")

            # Only include finished games
            if status not in ("Game Finished", "After Over Time"):
                continue

            teams = game.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})
            scores = game.get("scores", {})
            home_scores = scores.get("home", {})
            away_scores = scores.get("away", {})

            home_total = home_scores.get("total")
            away_total = away_scores.get("total")

            # Basketball has no draw
            if home_total is not None and away_total is not None:
                result = "home_win" if home_total > away_total else "away_win"
            else:
                result = None

            league_info = game.get("league", {})

            rows.append({
                "match_id": game.get("id"),
                "date": parse_date(game.get("date")),
                "league_id": league_id or league_info.get("id"),
                "league_name": league_info.get("name"),
                "season": league_info.get("season"),
                "home_team": home.get("name"),
                "home_team_id": home.get("id"),
                "away_team": away.get("name"),
                "away_team_id": away.get("id"),
                "home_score": home_total,
                "away_score": away_total,
                "result": result,
                "home_q1": home_scores.get("quarter_1"),
                "home_q2": home_scores.get("quarter_2"),
                "home_q3": home_scores.get("quarter_3"),
                "home_q4": home_scores.get("quarter_4"),
                "home_ot": home_scores.get("over_time"),
                "away_q1": away_scores.get("quarter_1"),
                "away_q2": away_scores.get("quarter_2"),
                "away_q3": away_scores.get("quarter_3"),
                "away_q4": away_scores.get("quarter_4"),
                "away_ot": away_scores.get("over_time"),
                "status": status,
            })

        df = pd.DataFrame(rows)
        if "date" in df.columns and not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)

        return df
