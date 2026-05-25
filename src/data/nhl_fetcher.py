"""
NHL data fetcher using the free NHL Stats API (api-web.nhle.com).

No API key required. Fetches game results for the current and
recent NHL seasons.

Endpoints used:
    schedule:  https://api-web.nhle.com/v1/schedule/{date}
    standings: https://api-web.nhle.com/v1/standings/{date}

Note: The NHL web API (api-web.nhle.com) is the current production API
as of the 2023-24 season. The older statsapi.web.nhl.com is deprecated.
"""

import json
import logging
import math
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

from src.data.base_fetcher import BaseFetcher

# Disk cache for per-game shot stats (avoids re-fetching PBP)
_SHOT_CACHE_FILE = Path("data/cache/nhl_shot_stats.json")

logger = logging.getLogger(__name__)

_BASE = "https://api-web.nhle.com/v1"


class NHLFetcher(BaseFetcher):
    """
    Fetches NHL game results from the free NHL web API.

    Produces a DataFrame with columns compatible with NHLFeatureEngineer:
        date, home_team, away_team, home_score, away_score, result,
        home_shots, away_shots, home_pp_goals, away_pp_goals,
        went_to_ot, season, game_id
    """

    def __init__(self, cache_expire_hours: int = 6) -> None:
        super().__init__(sport="nhl", cache_expire_hours=cache_expire_hours)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "sports-predictor/1.0"})
        self._shot_cache: Dict[int, dict] = self._load_shot_cache()

    # ------------------------------------------------------------------
    # Shot / xG cache helpers
    # ------------------------------------------------------------------

    def _load_shot_cache(self) -> Dict[int, dict]:
        if _SHOT_CACHE_FILE.exists():
            try:
                raw = json.loads(_SHOT_CACHE_FILE.read_text())
                return {int(k): v for k, v in raw.items()}
            except Exception:
                pass
        return {}

    def _save_shot_cache(self) -> None:
        _SHOT_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _SHOT_CACHE_FILE.write_text(json.dumps(self._shot_cache, indent=2))

    # ------------------------------------------------------------------
    # xG model: distance + angle from net
    # NHL rink: net at x=89, goal crease centered at y=0
    # Shot coordinates are in feet from centre ice
    # ------------------------------------------------------------------

    @staticmethod
    def _shot_xg(x: float, y: float, shot_type: str = "") -> float:
        """
        Simple distance/angle xG model calibrated to NHL shot data.
        Returns probability [0,1] that a shot results in a goal.

        Based on publicly available NHL shot data research:
        - Average shot xG ~5.5% (league scoring rate on shots)
        - Slot shots (close, low angle) ~15-25%
        - Point shots (blue line, high angle) ~2-5%
        """
        # Net is at x=89 (or x=-89 for other end), y=0
        net_x = 89.0
        dx = abs(abs(x) - net_x)
        dy = abs(y)
        distance = math.sqrt(dx**2 + dy**2)
        # Angle from centre of net (degrees)
        angle = math.degrees(math.atan2(dy, dx)) if dx > 0 else 90.0

        # Base xG from distance (logistic decay)
        base_xg = 0.16 * math.exp(-distance / 18.0)

        # Angle penalty: shots from extreme angles score less
        angle_mult = 1.0 - 0.5 * (angle / 90.0) ** 2

        # Shot type multiplier (tip/deflection and wrap-around are higher quality)
        type_mult = {
            "tip-in": 1.6, "deflected": 1.5, "wrap-around": 1.3,
            "snap": 1.1, "wrist": 1.0, "backhand": 0.85,
            "slap": 0.9, "between-legs": 1.2,
        }.get(shot_type.lower(), 1.0)

        xg = base_xg * angle_mult * type_mult
        return max(0.001, min(0.99, xg))

    def fetch_game_shot_stats(self, game_id: int) -> dict:
        """
        Fetch play-by-play for a game and compute shot-based stats.
        Returns dict with:
            home_corsi, away_corsi       (shot attempts for/against at ES)
            home_fenwick, away_fenwick   (unblocked shot attempts at ES)
            home_sog, away_sog           (shots on goal)
            home_xg, away_xg             (expected goals from shot model)
            home_pp_goals, away_pp_goals (power play goals)
            home_pp_opp, away_pp_opp     (power play opportunities)
        """
        if game_id in self._shot_cache:
            return self._shot_cache[game_id]

        empty = {
            "home_corsi": 0, "away_corsi": 0,
            "home_fenwick": 0, "away_fenwick": 0,
            "home_sog": 0, "away_sog": 0,
            "home_xg": 0.0, "away_xg": 0.0,
            "home_pp_goals": 0, "away_pp_goals": 0,
            "home_pp_opp": 0, "away_pp_opp": 0,
        }

        try:
            r = self._session.get(
                f"{_BASE}/gamecenter/{game_id}/play-by-play", timeout=15
            )
            if r.status_code != 200:
                return empty
            data = r.json()
        except Exception:
            return empty

        home_id = data.get("homeTeam", {}).get("id")
        away_id = data.get("awayTeam", {}).get("id")
        if not home_id or not away_id:
            return empty

        stats = {k: 0 for k in empty}
        stats["home_xg"] = 0.0
        stats["away_xg"] = 0.0

        for play in data.get("plays", []):
            evt = play.get("typeDescKey", "")
            details = play.get("details", {})
            owner_id = details.get("eventOwnerTeamId")
            situation = play.get("situationCode", "")

            # Even-strength: situation codes like 1551, 1541 etc
            # First digit = away skaters, last digit = home skaters, middle = goalies
            # ES = 5v5: situationCode[0]=='1' and [3]=='1' and [1]=='5' and [2]=='5'
            try:
                is_es = (len(situation) == 4 and
                         situation[1] == '5' and situation[2] == '5')
            except Exception:
                is_es = False

            is_home = (owner_id == home_id)
            key = "home" if is_home else "away"
            opp_key = "away" if is_home else "home"

            if evt in ("shot-on-goal", "goal", "missed-shot", "blocked-shot"):
                if is_es:
                    stats[f"{key}_corsi"] += 1
                    stats[f"{opp_key}_corsi"] += 0  # only count for team taking shot
                if evt in ("shot-on-goal", "goal", "missed-shot") and is_es:
                    stats[f"{key}_fenwick"] += 1
                if evt in ("shot-on-goal", "goal"):
                    stats[f"{key}_sog"] += 1
                    # Compute xG from coordinates
                    x = details.get("xCoord", 0) or 0
                    y = details.get("yCoord", 0) or 0
                    shot_type = details.get("shotType", "") or ""
                    stats[f"{key}_xg"] += self._shot_xg(float(x), float(y), shot_type)

            elif evt == "goal":
                # Check if power play goal (situationCode has skater imbalance)
                try:
                    if len(situation) == 4:
                        home_sk = int(situation[2])
                        away_sk = int(situation[1])
                        if is_home and home_sk > away_sk:
                            stats["home_pp_goals"] += 1
                        elif not is_home and away_sk > home_sk:
                            stats["away_pp_goals"] += 1
                except Exception:
                    pass

            elif evt == "penalty":
                # Count PP opportunities for the non-penalised team
                pen_team = details.get("eventOwnerTeamId")
                if pen_team == home_id:
                    stats["away_pp_opp"] += 1
                elif pen_team == away_id:
                    stats["home_pp_opp"] += 1

        self._shot_cache[game_id] = stats
        return stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, path: str) -> dict:
        url = f"{_BASE}/{path}"
        try:
            r = self._session.get(url, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            logger.debug("NHL API error %s: %s", path, exc)
            return {}

    def _season_range(self, season_year: int):
        """
        Return (start, end) dates for an NHL season.
        NHL seasons run Oct of previous year to Apr of season_year.
        """
        return f"{season_year - 1}-10-01", f"{season_year}-04-30"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch_all_seasons(self) -> pd.DataFrame:
        """Override base to avoid match_id dedup (NHL uses game_id)."""
        df = self.fetch_matches()          # fetch_matches handles multi-season
        if df.empty:
            return df
        df = df.drop_duplicates(subset=["game_id"]).sort_values("date").reset_index(drop=True)
        self._save_shot_cache()
        return df

    def fetch_matches(
        self,
        season: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        current_year = date.today().year
        if season is None:
            seasons = [current_year - 2, current_year - 1, current_year]
        else:
            seasons = [int(season)]

        all_games: List[dict] = []
        for yr in seasons:
            s_from, s_to = self._season_range(yr)
            if date_from:
                s_from = max(s_from, date_from)
            if date_to:
                s_to = min(s_to, date_to)
            logger.info("Fetching NHL season %d-%d  (%s → %s)", yr - 1, yr, s_from, s_to)
            all_games.extend(self._fetch_range(s_from, s_to))

        if not all_games:
            return pd.DataFrame()

        df = pd.DataFrame(all_games)
        df = df.sort_values("date").reset_index(drop=True)
        return df

    def _fetch_range(self, start: str, end: str) -> List[dict]:
        """Walk day-by-day through the schedule endpoint."""
        records: List[dict] = []
        d = date.fromisoformat(start)
        end_d = date.fromisoformat(end)

        while d <= end_d:
            data = self._get(f"schedule/{d.isoformat()}")
            game_week = data.get("gameWeek", [])
            for day_entry in game_week:
                for game in day_entry.get("games", []):
                    # gameState: 7=Final, 6=Official
                    if game.get("gameState") not in ("OFF", "FINAL"):
                        continue
                    rec = self._parse_game(game, day_entry.get("date", d.isoformat()))
                    if rec:
                        records.append(rec)

            # The schedule endpoint returns a 7-day week — jump 7 days
            d += timedelta(days=7)
            time.sleep(0.15)

        return records

    # ------------------------------------------------------------------
    # Abstract method stubs (not used by NHL pipeline)
    # ------------------------------------------------------------------

    def fetch_standings(self, season=None, league_id=None) -> pd.DataFrame:
        """Not used — NHL uses game-level data only."""
        return pd.DataFrame()

    def _parse_matches(self, raw_data) -> pd.DataFrame:
        """Not used — parsing handled in _parse_game."""
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Game parsing
    # ------------------------------------------------------------------

    def _parse_game(self, game: dict, game_date: str) -> Optional[dict]:
        try:
            home = game.get("homeTeam", {})
            away = game.get("awayTeam", {})

            home_team = home.get("commonName", {}).get("default", "") or home.get("abbrev", "")
            away_team = away.get("commonName", {}).get("default", "") or away.get("abbrev", "")

            home_score = home.get("score")
            away_score = away.get("score")

            if not home_team or not away_team:
                return None
            if home_score is None or away_score is None:
                return None

            home_score = int(home_score)
            away_score = int(away_score)

            period_descriptor = game.get("periodDescriptor", {})
            went_to_ot = int(period_descriptor.get("periodType", "REG") in ("OT", "SO")
                             or (period_descriptor.get("number", 3) or 3) > 3)

            if home_score > away_score:
                result = "home_win"
            elif away_score > home_score:
                result = "away_win"
            else:
                result = "draw"  # shouldn't happen in NHL but safety net

            season_str = str(game.get("season", ""))
            season = int(season_str[:4]) if len(season_str) >= 4 else int(game_date[:4])

            game_id = game.get("id")
            # Fetch shot-based stats from play-by-play (cached after first fetch)
            shot_stats = self.fetch_game_shot_stats(game_id) if game_id else {}

            return {
                "game_id":         game_id,
                "date":            game_date,
                "season":          season,
                "home_team":       home_team,
                "away_team":       away_team,
                "home_score":      home_score,
                "away_score":      away_score,
                "result":          result,
                "went_to_ot":      went_to_ot,
                "home_shots":      shot_stats.get("home_sog", 0),
                "away_shots":      shot_stats.get("away_sog", 0),
                "home_corsi":      shot_stats.get("home_corsi", 0),
                "away_corsi":      shot_stats.get("away_corsi", 0),
                "home_fenwick":    shot_stats.get("home_fenwick", 0),
                "away_fenwick":    shot_stats.get("away_fenwick", 0),
                "home_xg":         shot_stats.get("home_xg", 0.0),
                "away_xg":         shot_stats.get("away_xg", 0.0),
                "home_pp_goals":   shot_stats.get("home_pp_goals", 0),
                "away_pp_goals":   shot_stats.get("away_pp_goals", 0),
                "home_pp_opp":     shot_stats.get("home_pp_opp", 0),
                "away_pp_opp":     shot_stats.get("away_pp_opp", 0),
            }
        except Exception as exc:
            logger.debug("Could not parse NHL game: %s", exc)
            return None
