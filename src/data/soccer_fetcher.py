"""
Soccer data fetcher using football-data.org API (v4).

Fetches match results, standings, and team statistics for
configured European competitions.
"""

import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from config import settings
from src.data.base_fetcher import BaseFetcher
from src.utils.helpers import RateLimiter, parse_date

logger = logging.getLogger(__name__)


def _coerce_matchday(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return int(value)
        except Exception:
            return None
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"(\d+)(?!.*\d)", text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


class SoccerFetcher(BaseFetcher):
    """
    Fetcher for soccer match data from football-data.org.

    Uses the free tier (10 requests/minute) with automatic
    rate limiting and caching.
    """

    def __init__(self, cache_expire_hours: int = 24) -> None:
        super().__init__(sport="soccer", cache_expire_hours=cache_expire_hours)

        apis_cfg = settings.get("apis", {})
        api_cfg = apis_cfg.get("football_data", {})
        self._base_url = api_cfg.get("base_url", "https://api.football-data.org/v4")
        self._api_key = os.environ.get("FOOTBALL_DATA_API_KEY", "")
        self._competitions = api_cfg.get("competitions", ["PL"])
        self._rate_limiter = RateLimiter(
            max_calls=api_cfg.get("rate_limit_per_minute", 10),
            period_seconds=60,
        )
        api_sports_cfg = apis_cfg.get("api_sports", {})
        self._api_sports_base_url = api_sports_cfg.get("base_url", "https://v3.football.api-sports.io")
        self._api_sports_key = os.environ.get("API_SPORTS_KEY", "")
        self._api_sports_rate_limiter = RateLimiter(
            max_calls=api_sports_cfg.get("rate_limit_per_day", 100),
            period_seconds=86400,
        )
        self._api_sports_competitions = {
            str(item.get("key")): dict(item)
            for item in self._sport_cfg.get("api_sports_competitions", [])
            if item.get("key")
        }
        self._api_sports_league_cache: dict[str, Optional[int]] = {}
        self._api_sports_disabled_until: Optional[datetime] = None
        self._api_sports_disabled_reason: str = ""

        if not self._api_key:
            logger.warning(
                "FOOTBALL_DATA_API_KEY not set. API calls will fail or be limited."
            )
        if self._api_sports_competitions and not self._api_sports_key:
            logger.warning("API_SPORTS_KEY not set. Extended soccer competitions will be unavailable.")

    def _disable_api_sports(self, *, hours: int, reason: str) -> None:
        self._api_sports_disabled_until = datetime.now(timezone.utc) + timedelta(hours=hours)
        self._api_sports_disabled_reason = reason
        logger.warning("API-Sports soccer fetches paused for %dh: %s", hours, reason)

    def _api_sports_is_disabled(self) -> bool:
        if self._api_sports_disabled_until is None:
            return False
        if datetime.now(timezone.utc) >= self._api_sports_disabled_until:
            self._api_sports_disabled_until = None
            self._api_sports_disabled_reason = ""
            return False
        return True

    @property
    def _headers(self) -> Dict[str, str]:
        return {"X-Auth-Token": self._api_key}

    @property
    def _api_sports_headers(self) -> Dict[str, str]:
        return {
            "x-apisports-key": self._api_sports_key,
            "x-rapidapi-host": "v3.football.api-sports.io",
        }

    def _competition_source(self, competition: Optional[str]) -> str:
        key = str(competition or "").strip()
        if key and key in self._api_sports_competitions:
            return "api_sports"
        return "football_data"

    def _api_sports_season_value(self, competition_key: str, season: Optional[str]) -> int:
        cfg = self._api_sports_competitions.get(competition_key, {})
        mode = str(cfg.get("season_mode", "calendar")).strip().lower()
        now = pd.Timestamp.now()
        if season:
            season_year = int(str(season))
        else:
            season_year = now.year if now.month >= 8 else now.year - 1
        if mode == "calendar":
            if season is not None:
                return season_year + 1 if now.month < 8 else season_year
            return now.year
        return season_year

    def _resolve_api_sports_league_id(self, competition_key: str, season: Optional[str]) -> Optional[int]:
        if competition_key in self._api_sports_league_cache:
            return self._api_sports_league_cache[competition_key]

        if not self._api_sports_key or self._api_sports_is_disabled():
            return None

        cfg = self._api_sports_competitions.get(competition_key, {})
        if not cfg:
            return None

        if cfg.get("league_id") is not None:
            league_id = int(cfg["league_id"])
            self._api_sports_league_cache[competition_key] = league_id
            return league_id

        params: Dict[str, Any] = {"name": cfg.get("name")}
        country = cfg.get("country")
        if country:
            params["country"] = country
        api_season = self._api_sports_season_value(competition_key, season)
        if api_season:
            params["season"] = api_season

        try:
            raw = self._get(
                url=f"{self._api_sports_base_url}/leagues",
                headers=self._api_sports_headers,
                params=params,
                rate_limiter=self._api_sports_rate_limiter,
            )
            candidates = raw.get("response", [])
        except Exception as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code == 429:
                self._disable_api_sports(hours=2, reason="429 Too Many Requests from API-Sports leagues lookup")
            elif status_code == 403:
                self._disable_api_sports(hours=6, reason="403 Forbidden from API-Sports leagues lookup")
            logger.error("Error resolving API-Sports league id for %s: %s", competition_key, exc)
            self._api_sports_league_cache[competition_key] = None
            return None

        target_name = str(cfg.get("name") or "").strip().lower()
        for item in candidates:
            league = item.get("league", {})
            league_name = str(league.get("name") or "").strip().lower()
            if target_name and (league_name == target_name or target_name in league_name or league_name in target_name):
                try:
                    league_id = int(league.get("id"))
                except Exception:
                    continue
                self._api_sports_league_cache[competition_key] = league_id
                return league_id

        if candidates:
            try:
                league_id = int(candidates[0].get("league", {}).get("id"))
                self._api_sports_league_cache[competition_key] = league_id
                return league_id
            except Exception:
                pass

        logger.warning("Could not resolve API-Sports league id for %s", competition_key)
        self._api_sports_league_cache[competition_key] = None
        return None

    # ------------------------------------------------------------------
    # Core fetch methods
    # ------------------------------------------------------------------

    def fetch_matches(
        self,
        season: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        competition: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch match results for one or all configured competitions.

        Parameters
        ----------
        season : str, optional
            Season year (e.g. '2024'). Defaults to current year.
        date_from : str, optional
            Start date in YYYY-MM-DD format.
        date_to : str, optional
            End date in YYYY-MM-DD format.
        competition : str, optional
            Specific competition code. If None, fetches all configured.

        Returns
        -------
        pd.DataFrame
            Standardized match data.
        """
        if competition:
            competitions = [competition]
        else:
            competitions = list(self._competitions) + list(self._api_sports_competitions.keys())
        all_matches: List[pd.DataFrame] = []

        for comp in competitions:
            source = self._competition_source(comp)
            logger.info("Fetching soccer matches: comp=%s, season=%s, source=%s", comp, season, source)

            if source == "api_sports":
                df = self._fetch_api_sports_matches(season=season, competition_key=str(comp))
                if not df.empty:
                    all_matches.append(df)
                continue

            params: Dict[str, Any] = {"status": "FINISHED"}
            if season:
                params["season"] = season
            if date_from:
                params["dateFrom"] = date_from
            if date_to:
                params["dateTo"] = date_to

            try:
                url = f"{self._base_url}/competitions/{comp}/matches"
                raw = self._get(
                    url=url,
                    headers=self._headers,
                    params=params,
                    rate_limiter=self._rate_limiter,
                )
                matches_raw = raw.get("matches", [])
                if not matches_raw:
                    logger.info("No matches found for %s season %s", comp, season)
                    continue

                df = self._parse_matches(matches_raw, competition_code=comp)
                all_matches.append(df)
                logger.info("Fetched %d matches for %s", len(df), comp)

            except Exception as exc:
                logger.error("Error fetching %s matches: %s", comp, exc)
                continue

        if not all_matches:
            return pd.DataFrame()

        combined = pd.concat(all_matches, ignore_index=True)
        return combined

    def _fetch_api_sports_matches(
        self,
        *,
        season: Optional[str],
        competition_key: str,
    ) -> pd.DataFrame:
        if not self._api_sports_key:
            logger.warning("Skipping %s: API_SPORTS_KEY not configured", competition_key)
            return pd.DataFrame()
        if self._api_sports_is_disabled():
            logger.info(
                "Skipping %s: API-Sports temporarily paused (%s)",
                competition_key,
                self._api_sports_disabled_reason or "provider cooling off",
            )
            return pd.DataFrame()

        league_id = self._resolve_api_sports_league_id(competition_key, season)
        if league_id is None:
            return pd.DataFrame()

        api_season = self._api_sports_season_value(competition_key, season)
        params = {
            "league": league_id,
            "season": api_season,
            "status": "FT",
        }
        try:
            raw = self._get(
                url=f"{self._api_sports_base_url}/fixtures",
                headers=self._api_sports_headers,
                params=params,
                rate_limiter=self._api_sports_rate_limiter,
            )
        except Exception as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code == 429:
                self._disable_api_sports(hours=2, reason="429 Too Many Requests from API-Sports fixtures")
            elif status_code == 403:
                self._disable_api_sports(hours=6, reason="403 Forbidden from API-Sports fixtures")
            logger.error("Error fetching API-Sports matches for %s: %s", competition_key, exc)
            return pd.DataFrame()

        fixtures_raw = raw.get("response", [])
        if not fixtures_raw:
            logger.info("No API-Sports fixtures found for %s season %s", competition_key, api_season)
            return pd.DataFrame()

        df = self._parse_api_sports_matches(fixtures_raw, competition_code=competition_key)
        logger.info("Fetched %d API-Sports soccer matches for %s", len(df), competition_key)
        return df

    def fetch_standings(
        self,
        season: Optional[str] = None,
        league_id: Optional[int] = None,
        competition: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch league standings for a competition.

        Parameters
        ----------
        season : str, optional
            Season year.
        league_id : int, optional
            Not used for football-data.org (uses competition code).
        competition : str, optional
            Competition code (e.g. 'PL').

        Returns
        -------
        pd.DataFrame
            Standings table with team, points, goal difference, etc.
        """
        comp = competition or self._competitions[0]
        params: Dict[str, Any] = {}
        if season:
            params["season"] = season

        url = f"{self._base_url}/competitions/{comp}/standings"
        raw = self._get(
            url=url,
            headers=self._headers,
            params=params,
            rate_limiter=self._rate_limiter,
        )

        standings_raw = raw.get("standings", [])
        if not standings_raw:
            return pd.DataFrame()

        # Extract the total standings (index 0 is TOTAL, 1 is HOME, 2 is AWAY)
        total = standings_raw[0].get("table", []) if standings_raw else []

        rows = []
        for entry in total:
            team = entry.get("team", {})
            rows.append({
                "position": entry.get("position"),
                "team_id": team.get("id"),
                "team_name": team.get("name"),
                "played": entry.get("playedGames"),
                "won": entry.get("won"),
                "drawn": entry.get("draw"),
                "lost": entry.get("lost"),
                "goals_for": entry.get("goalsFor"),
                "goals_against": entry.get("goalsAgainst"),
                "goal_difference": entry.get("goalDifference"),
                "points": entry.get("points"),
                "competition": comp,
                "season": season,
            })

        return pd.DataFrame(rows)

    def fetch_team_stats(self, team_id: int) -> Dict[str, Any]:
        """
        Fetch detailed statistics for a specific team.

        Parameters
        ----------
        team_id : int
            Football-data.org team ID.

        Returns
        -------
        dict
            Team details and statistics.
        """
        url = f"{self._base_url}/teams/{team_id}"
        return self._get(
            url=url,
            headers=self._headers,
            rate_limiter=self._rate_limiter,
        )

    # ------------------------------------------------------------------
    # Head-to-head
    # ------------------------------------------------------------------

    def fetch_h2h(self, match_id: int) -> pd.DataFrame:
        """
        Fetch head-to-head record for a specific match.

        Parameters
        ----------
        match_id : int
            Match ID from football-data.org.

        Returns
        -------
        pd.DataFrame
            Historical H2H matches between the two teams.
        """
        url = f"{self._base_url}/matches/{match_id}/head2head"
        raw = self._get(
            url=url,
            headers=self._headers,
            rate_limiter=self._rate_limiter,
        )

        matches = raw.get("matches", [])
        if not matches:
            return pd.DataFrame()

        return self._parse_matches(matches)

    def fetch_all_seasons(self) -> pd.DataFrame:
        """
        Fetch configured football-data and API-Sports soccer competitions.

        Football-data competitions use start-year season labels (e.g. 2025/26 -> 2025).
        Some API-Sports competitions use calendar-year seasons instead, so we resolve
        seasons per competition before fetching.
        """
        seasons_to_fetch = self._sport_cfg.get("seasons_to_fetch", 2)
        now = pd.Timestamp.now()
        football_current = now.year if now.month >= 8 else now.year - 1
        all_frames: List[pd.DataFrame] = []

        for comp in self._competitions:
            for offset in range(seasons_to_fetch):
                season = str(football_current - offset)
                try:
                    df = self.fetch_matches(season=season, competition=comp)
                except Exception as exc:
                    logger.error("Failed to fetch soccer comp %s season %s: %s", comp, season, exc)
                    continue
                if df is not None and not df.empty:
                    all_frames.append(df)

        for comp in self._api_sports_competitions:
            cfg = self._api_sports_competitions.get(comp, {})
            configured_years = [int(year) for year in cfg.get("season_years", []) if year is not None]
            if configured_years:
                season_inputs = [str(year) for year in configured_years]
            else:
                season_inputs = [str(football_current - offset) for offset in range(seasons_to_fetch)]
            seen_api_seasons: set[int] = set()
            for season in season_inputs:
                api_season = self._api_sports_season_value(comp, season)
                if api_season in seen_api_seasons:
                    continue
                seen_api_seasons.add(api_season)
                try:
                    df = self.fetch_matches(season=season, competition=comp)
                except Exception as exc:
                    logger.error("Failed to fetch soccer comp %s season %s: %s", comp, season, exc)
                    continue
                if df is not None and not df.empty:
                    all_frames.append(df)

        if not all_frames:
            logger.warning("No data fetched for soccer")
            return pd.DataFrame()

        combined = pd.concat(all_frames, ignore_index=True)
        if "season" in combined.columns:
            combined["season"] = combined["season"].astype(str)
        if "matchday" in combined.columns:
            combined["matchday"] = pd.to_numeric(combined["matchday"], errors="coerce")
        combined = combined.drop_duplicates(subset=["match_id"]).sort_values(
            "date", ascending=True,
        ).reset_index(drop=True)

        self.save_raw(combined, f"{self.sport}_all_seasons")
        logger.info("Combined soccer data: %d matches across configured competitions", len(combined))
        return combined

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_matches(
        self,
        raw_data: Any,
        competition_code: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Parse raw football-data.org match JSON into a standardized DataFrame.

        Parameters
        ----------
        raw_data : list[dict]
            List of match dictionaries from the API.
        competition_code : str, optional
            Competition code to tag rows with.

        Returns
        -------
        pd.DataFrame
            Columns: match_id, date, competition, season, matchday,
                     home_team, home_team_id, away_team, away_team_id,
                     home_goals, away_goals, result, home_ht, away_ht.
        """
        rows = []
        for match in raw_data:
            home = match.get("homeTeam", {})
            away = match.get("awayTeam", {})
            score = match.get("score", {})
            full_time = score.get("fullTime", {})
            half_time = score.get("halfTime", {})
            season_info = match.get("season", {})

            home_goals = full_time.get("home")
            away_goals = full_time.get("away")

            # Determine result
            if home_goals is not None and away_goals is not None:
                if home_goals > away_goals:
                    result = "home_win"
                elif home_goals < away_goals:
                    result = "away_win"
                else:
                    result = "draw"
            else:
                result = None

            comp = competition_code
            if not comp:
                comp_info = match.get("competition", {})
                comp = comp_info.get("code", "UNK")

            rows.append({
                "match_id": match.get("id"),
                "date": parse_date(match.get("utcDate")),
                "competition": comp,
                "season": str(season_info.get("startDate", "")[:4]),
                "matchday": _coerce_matchday(match.get("matchday")),
                "home_team": home.get("name"),
                "home_team_id": home.get("id"),
                "away_team": away.get("name"),
                "away_team_id": away.get("id"),
                "home_goals": home_goals,
                "away_goals": away_goals,
                "result": result,
                "home_ht": half_time.get("home"),
                "away_ht": half_time.get("away"),
            })

        df = pd.DataFrame(rows)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
            df = df.sort_values("date").reset_index(drop=True)

        return df

    def _parse_api_sports_matches(
        self,
        raw_data: Any,
        competition_code: Optional[str] = None,
    ) -> pd.DataFrame:
        rows = []
        for match in raw_data:
            fixture = match.get("fixture", {})
            league = match.get("league", {})
            teams = match.get("teams", {})
            goals = match.get("goals", {})
            score = match.get("score", {})
            halftime = score.get("halftime", {}) if isinstance(score, dict) else {}

            home_goals = goals.get("home")
            away_goals = goals.get("away")
            if home_goals is not None and away_goals is not None:
                if home_goals > away_goals:
                    result = "home_win"
                elif home_goals < away_goals:
                    result = "away_win"
                else:
                    result = "draw"
            else:
                result = None

            rows.append({
                "match_id": fixture.get("id"),
                "date": parse_date(fixture.get("date")),
                "competition": competition_code or league.get("name") or "UNK",
                "season": str(league.get("season")),
                "matchday": _coerce_matchday(league.get("round")),
                "home_team": (teams.get("home") or {}).get("name"),
                "home_team_id": (teams.get("home") or {}).get("id"),
                "away_team": (teams.get("away") or {}).get("name"),
                "away_team_id": (teams.get("away") or {}).get("id"),
                "home_goals": home_goals,
                "away_goals": away_goals,
                "result": result,
                "home_ht": halftime.get("home"),
                "away_ht": halftime.get("away"),
            })

        df = pd.DataFrame(rows)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
            df = df.sort_values("date").reset_index(drop=True)
        return df
