"""
MLB data fetcher using the free MLB Stats API (statsapi.mlb.com).

No API key required. Fetches game results, team stats, and starting
pitcher stats for the current and recent MLB seasons.

Endpoints used:
    schedule:        https://statsapi.mlb.com/api/v1/schedule
    people stats:    https://statsapi.mlb.com/api/v1/people/{id}/stats
    standings:       https://statsapi.mlb.com/api/v1/standings
"""

import json
import logging
import re
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

from src.data.base_fetcher import BaseFetcher

logger = logging.getLogger(__name__)

_BASE = "https://statsapi.mlb.com/api/v1"
_SPORT_ID = 1         # MLB
_LEAGUE_IDS = [103, 104]  # American League, National League

# Pitcher stats cache file — persisted so we don't re-fetch every run
_PITCHER_CACHE_FILE = Path("data/cache/mlb_pitcher_stats.json")


class MLBFetcher(BaseFetcher):
    """
    Fetches MLB game results from the free MLB Stats API.

    Produces a DataFrame with columns compatible with MLBFeatureEngineer:
        date, home_team, away_team, home_score, away_score, result,
        home_hits, away_hits, home_errors, away_errors,
        home_left_on_base, away_left_on_base,
        home_innings, away_innings, season, game_pk
    """

    def __init__(self, cache_expire_hours: int = 6) -> None:
        super().__init__(sport="mlb", cache_expire_hours=cache_expire_hours)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "sports-predictor/1.0"})
        # In-memory pitcher stats cache: {pitcher_id: {season: stats_dict}}
        self._pitcher_cache: Dict[int, Dict[int, dict]] = self._load_pitcher_cache()

    # ------------------------------------------------------------------
    # Pitcher stats cache (disk + memory)
    # ------------------------------------------------------------------

    def _load_pitcher_cache(self) -> Dict[int, Dict[int, dict]]:
        if _PITCHER_CACHE_FILE.exists():
            try:
                raw = json.loads(_PITCHER_CACHE_FILE.read_text())
                # JSON keys are strings — convert back to int
                return {int(pid): {int(s): v for s, v in seasons.items()}
                        for pid, seasons in raw.items()}
            except Exception:
                pass
        return {}

    def _save_pitcher_cache(self) -> None:
        _PITCHER_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _PITCHER_CACHE_FILE.write_text(json.dumps(self._pitcher_cache, indent=2))

    def fetch_pitcher_stats(self, pitcher_id: int, season: int) -> dict:
        """
        Fetch season pitching stats for a single pitcher.
        Returns dict with era, whip, k_per9, bb_per9, h_per9, games_started,
        wins, losses, innings_pitched. Returns empty dict on failure.
        Caches results to disk to avoid repeated API calls.
        """
        if pitcher_id in self._pitcher_cache:
            if season in self._pitcher_cache[pitcher_id]:
                return self._pitcher_cache[pitcher_id][season]

        try:
            r = self._session.get(
                f"{_BASE}/people/{pitcher_id}/stats",
                params={"stats": "season", "group": "pitching", "season": str(season)},
                timeout=10,
            )
            r.raise_for_status()
            splits = r.json().get("stats", [{}])[0].get("splits", [])
            if not splits:
                return {}
            s = splits[0].get("stat", {})

            def _f(key: str, default: float = 0.0) -> float:
                v = s.get(key)
                try:
                    return float(v) if v is not None else default
                except (TypeError, ValueError):
                    return default

            stats = {
                "era":             _f("era", 4.50),
                "whip":            _f("whip", 1.30),
                "k_per9":          _f("strikeoutsPer9Inn", 8.0),
                "bb_per9":         _f("walksPer9Inn", 3.0),
                "h_per9":          _f("hitsPer9Inn", 9.0),
                "games_started":   int(_f("gamesStarted")),
                "wins":            int(_f("wins")),
                "losses":          int(_f("losses")),
                "innings_pitched": _f("inningsPitched"),
            }
            self._pitcher_cache.setdefault(pitcher_id, {})[season] = stats
            return stats
        except Exception as exc:
            logger.debug("Could not fetch pitcher %d stats for %d: %s", pitcher_id, season, exc)
            return {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict = None) -> dict:
        url = f"{_BASE}/{path}"
        try:
            r = self._session.get(url, params=params or {}, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            logger.error("MLB API error %s: %s", path, exc)
            return {}

    def _season_dates(self, season: int):
        """Return (start_date, end_date) strings for an MLB season."""
        # Regular season typically Apr 1 – Sep 30
        return f"{season}-03-20", f"{season}-10-05"

    def fetch_todays_probable_pitchers(
        self, home_team: str, away_team: str
    ) -> Optional[dict]:
        """
        Return probable starting pitchers for today's matchup.

        Calls the schedule API for today only (no score required — game can be
        upcoming). Returns a dict with pitcher names and season stats, or None
        if no matching game / pitchers not yet announced.
        """
        import difflib
        from datetime import date as _date

        today = _date.today().isoformat()
        season = int(today[:4])
        data = self._get("schedule", {
            "sportId":   _SPORT_ID,
            "startDate": today,
            "endDate":   today,
            "gameType":  "R,P",          # regular season + postseason
            "hydrate":   "probablePitcher",
        })

        def _norm(s: str) -> str:
            return re.sub(r"[^a-z0-9]", " ", s.lower()).strip()

        for date_entry in data.get("dates", []):
            for game in date_entry.get("games", []):
                teams = game.get("teams", {})
                api_home = teams.get("home", {}).get("team", {}).get("name", "")
                api_away = teams.get("away", {}).get("team", {}).get("name", "")
                home_score = difflib.SequenceMatcher(None, _norm(api_home), _norm(home_team)).ratio()
                away_score = difflib.SequenceMatcher(None, _norm(api_away), _norm(away_team)).ratio()
                if home_score < 0.6 or away_score < 0.6:
                    continue

                _default = {"era": 4.50, "whip": 1.30, "k_per9": 8.0,
                            "bb_per9": 3.0, "h_per9": 9.0,
                            "games_started": 0, "innings_pitched": 0.0}
                home_p = teams["home"].get("probablePitcher") or {}
                away_p = teams["away"].get("probablePitcher") or {}
                home_id = home_p.get("id")
                away_id = away_p.get("id")
                home_ps = {**_default, **(self.fetch_pitcher_stats(home_id, season) if home_id else {})}
                away_ps = {**_default, **(self.fetch_pitcher_stats(away_id, season) if away_id else {})}
                return {
                    "home_pitcher_name": home_p.get("fullName", "TBD"),
                    "away_pitcher_name": away_p.get("fullName", "TBD"),
                    "home_sp_era":  home_ps["era"],
                    "home_sp_whip": home_ps["whip"],
                    "home_sp_k9":   home_ps["k_per9"],
                    "home_sp_gs":   home_ps["games_started"],
                    "away_sp_era":  away_ps["era"],
                    "away_sp_whip": away_ps["whip"],
                    "away_sp_k9":   away_ps["k_per9"],
                    "away_sp_gs":   away_ps["games_started"],
                }
        return None

    # ------------------------------------------------------------------
    # Public interface (matches BaseFetcher contract)
    # ------------------------------------------------------------------

    def fetch_all_seasons(self) -> pd.DataFrame:
        """Override base to avoid match_id dedup (MLB uses game_pk)."""
        df = self.fetch_matches()          # fetch_matches handles multi-season
        if df.empty:
            return df
        df = df.drop_duplicates(subset=["game_pk"]).sort_values("date").reset_index(drop=True)
        self._save_pitcher_cache()         # persist pitcher stats to disk
        return df

    def fetch_matches(
        self,
        season: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Fetch completed MLB games.  Iterates over a date range in
        60-day chunks to stay within API limits.
        """
        current_year = date.today().year
        if season is None:
            # Fetch the current season plus two prior seasons
            seasons = [current_year - 2, current_year - 1, current_year]
        else:
            seasons = [int(season)]

        all_games: List[dict] = []
        for yr in seasons:
            s_from, s_to = self._season_dates(yr)
            if date_from:
                s_from = max(s_from, date_from)
            if date_to:
                s_to = min(s_to, date_to)
            logger.info("Fetching MLB season %d  (%s → %s)", yr, s_from, s_to)
            all_games.extend(self._fetch_range(s_from, s_to))

        if not all_games:
            return pd.DataFrame()

        df = pd.DataFrame(all_games)
        df = df.sort_values("date").reset_index(drop=True)
        return df

    def _fetch_range(self, start: str, end: str) -> List[dict]:
        """Fetch games day-by-day between start and end (inclusive)."""
        records: List[dict] = []
        d = date.fromisoformat(start)
        end_d = date.fromisoformat(end)

        while d <= end_d:
            batch_end = min(d + timedelta(days=29), end_d)  # 30-day window
            date_str_from = d.isoformat()
            date_str_to   = batch_end.isoformat()

            data = self._get("schedule", {
                "sportId":   _SPORT_ID,
                "startDate": date_str_from,
                "endDate":   date_str_to,
                "gameType":  "R",          # Regular season only
                "hydrate":   "linescore,probablePitcher",
            })

            for date_entry in data.get("dates", []):
                for game in date_entry.get("games", []):
                    if game.get("status", {}).get("abstractGameState") != "Final":
                        continue
                    rec = self._parse_game(game, date_entry.get("date", ""))
                    if rec:
                        records.append(rec)

            d = batch_end + timedelta(days=1)
            time.sleep(0.2)  # be polite to the free API

        return records

    # ------------------------------------------------------------------
    # Abstract method stubs (not used by MLB pipeline)
    # ------------------------------------------------------------------

    def fetch_standings(self, season=None, league_id=None) -> pd.DataFrame:
        """Not used — MLB uses game-level data only."""
        return pd.DataFrame()

    def _parse_matches(self, raw_data) -> pd.DataFrame:
        """Not used — parsing handled in _parse_game."""
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Game parsing
    # ------------------------------------------------------------------

    def _parse_game(self, game: dict, game_date: str) -> Optional[dict]:
        """Extract relevant fields from a schedule game entry."""
        try:
            teams  = game.get("teams", {})
            home   = teams.get("home", {})
            away   = teams.get("away", {})

            home_team  = home.get("team", {}).get("name", "")
            away_team  = away.get("team", {}).get("name", "")
            home_score = home.get("score")
            away_score = away.get("score")

            if not home_team or not away_team:
                return None
            if home_score is None or away_score is None:
                return None

            home_score = int(home_score)
            away_score = int(away_score)

            if home_score > away_score:
                result = "home_win"
            elif away_score > home_score:
                result = "away_win"
            else:
                result = "draw"  # extremely rare but possible (tie game called)

            # Linescore for innings
            linescore = game.get("linescore", {})
            innings    = linescore.get("innings", [])
            home_innings = [inn.get("home", {}).get("runs", 0) or 0 for inn in innings]
            away_innings = [inn.get("away", {}).get("runs", 0) or 0 for inn in innings]

            # Hits / errors from linescore teams
            ls_teams = linescore.get("teams", {})
            home_hits   = (ls_teams.get("home") or {}).get("hits", 0) or 0
            away_hits   = (ls_teams.get("away") or {}).get("hits", 0) or 0
            home_errors = (ls_teams.get("home") or {}).get("errors", 0) or 0
            away_errors = (ls_teams.get("away") or {}).get("errors", 0) or 0

            season = int(game_date[:4]) if game_date else 0

            # --- Starting pitcher stats ---
            home_pitcher = home.get("probablePitcher", {})
            away_pitcher = away.get("probablePitcher", {})
            home_pitcher_id = home_pitcher.get("id")
            away_pitcher_id = away_pitcher.get("id")

            _default_pitcher = {
                "era": 4.50, "whip": 1.30, "k_per9": 8.0,
                "bb_per9": 3.0, "h_per9": 9.0, "games_started": 0,
                "wins": 0, "losses": 0, "innings_pitched": 0.0,
            }
            home_p_stats = self.fetch_pitcher_stats(home_pitcher_id, season) if home_pitcher_id else {}
            away_p_stats = self.fetch_pitcher_stats(away_pitcher_id, season) if away_pitcher_id else {}

            # Prefix and merge with defaults
            home_ps = {**_default_pitcher, **home_p_stats}
            away_ps = {**_default_pitcher, **away_p_stats}

            rec = {
                "game_pk":           game.get("gamePk"),
                "date":              game_date,
                "season":            season,
                "home_team":         home_team,
                "away_team":         away_team,
                "home_score":        home_score,
                "away_score":        away_score,
                "result":            result,
                "home_hits":         int(home_hits),
                "away_hits":         int(away_hits),
                "home_errors":       int(home_errors),
                "away_errors":       int(away_errors),
                "home_innings":      sum(home_innings),
                "away_innings":      sum(away_innings),
                # Home starter
                "home_sp_era":       home_ps["era"],
                "home_sp_whip":      home_ps["whip"],
                "home_sp_k9":        home_ps["k_per9"],
                "home_sp_bb9":       home_ps["bb_per9"],
                "home_sp_h9":        home_ps["h_per9"],
                "home_sp_gs":        home_ps["games_started"],
                "home_sp_ip":        home_ps["innings_pitched"],
                # Away starter
                "away_sp_era":       away_ps["era"],
                "away_sp_whip":      away_ps["whip"],
                "away_sp_k9":        away_ps["k_per9"],
                "away_sp_bb9":       away_ps["bb_per9"],
                "away_sp_h9":        away_ps["h_per9"],
                "away_sp_gs":        away_ps["games_started"],
                "away_sp_ip":        away_ps["innings_pitched"],
            }
            return rec
        except Exception as exc:
            logger.debug("Could not parse MLB game: %s", exc)
            return None
