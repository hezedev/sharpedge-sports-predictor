"""
API-Football Enrichment Module

Enriches soccer odds snapshots with granular match statistics:
- Team form (recent results)
- Expected Goals (xG)
- Corner stats
- Head-to-head records
- Injury/suspension information

Usage:
    enricher = APIFootballEnricher()
    enriched_df = enricher.enrich_soccer_odds(odds_df)

Cost: 1 request per match (100 req/day limit on free tier)
Recommendation: Use selectively for matches with pending bets only
"""

import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.utils.cache import DiskCache
from src.utils.helpers import RateLimiter

logger = logging.getLogger(__name__)

_TEAM_STOPWORDS = {
    "fc", "cf", "ac", "sc", "afc", "cfc", "club", "de", "atletico", "athletic",
}


class APIFootballEnricher:
    """
    Enriches soccer odds data with API-Football stats.

    Adds columns:
    - home_form: Recent home results (e.g., "WWL")
    - away_form: Recent away results
    - home_xg: Expected goals for home
    - away_xg: Expected goals for away
    - corners_home_avg: Average corners (home)
    - corners_away_avg: Average corners (away)
    - h2h_record: Historical head-to-head
    """

    def __init__(self, cache_expire_hours: int = 24) -> None:
        rapidapi_key = os.environ.get("RAPIDAPI_KEY", "")
        api_sports_key = os.environ.get("API_SPORTS_KEY", "")
        self.api_key = rapidapi_key or api_sports_key
        self.provider = "rapidapi" if rapidapi_key else "api_sports"
        self.base_url = (
            "https://api-football-v1.p.rapidapi.com/v3"
            if rapidapi_key
            else "https://v3.football.api-sports.io"
        )
        self.cache_expire_hours = cache_expire_hours

        self.headers = (
            {
                "X-RapidAPI-Key": self.api_key,
                "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com",
            }
            if rapidapi_key
            else {"x-apisports-key": self.api_key}
        )

        # Rate limiter: 100 req/day free tier
        self._rate_limiter = RateLimiter(max_calls=100, period_seconds=86400)
        self._cache = DiskCache(cache_name="api_football_enrichment", expire_hours=cache_expire_hours)
        self._disabled_until: Optional[datetime] = None
        self._disabled_reason: str = ""

        if not self.api_key:
            logger.warning("Neither RAPIDAPI_KEY nor API_SPORTS_KEY is set. API-Football enrichment will be unavailable.")

    def _temporarily_disable(self, *, hours: int, reason: str) -> None:
        self._disabled_until = datetime.now(timezone.utc) + timedelta(hours=hours)
        self._disabled_reason = reason
        logger.warning("API-Football enrichment paused for %dh: %s", hours, reason)

    def _is_temporarily_disabled(self) -> bool:
        if self._disabled_until is None:
            return False
        if datetime.now(timezone.utc) >= self._disabled_until:
            self._disabled_until = None
            self._disabled_reason = ""
            return False
        return True

    def enrich_soccer_odds(
        self,
        odds_df: pd.DataFrame,
        enrich_fields: Optional[List[str]] = None,
        dry_run: bool = False,
    ) -> pd.DataFrame:
        """
        Enrich soccer odds DataFrame with match statistics.

        Parameters
        ----------
        odds_df : pd.DataFrame
            Odds data from HybridOddsFetcher.
        enrich_fields : list[str], optional
            Which fields to enrich. Default: all available
            Options: ["form", "xg", "corners", "h2h", "injuries"]
        dry_run : bool, optional
            Don't make API calls.

        Returns
        -------
        pd.DataFrame
            Original odds_df with added enrichment columns.
        """
        if odds_df.empty:
            return odds_df

        if dry_run:
            logger.info("[DRY-RUN] Would enrich %d matches", len(odds_df))
            return odds_df

        if not self.api_key:
            logger.warning("API_SPORTS_KEY not configured. Skipping enrichment.")
            return odds_df

        enrich_fields = enrich_fields or ["form", "xg", "corners", "h2h"]
        enriched = odds_df.copy()

        # Enrich each unique match
        unique_matches = odds_df.groupby(["home_team", "away_team"]).first().reset_index()

        for _, match in unique_matches.iterrows():
            home_team = match["home_team"]
            away_team = match["away_team"]
            commence = match.get("commence_time")

            logger.info(f"Enriching: {home_team} vs {away_team}")

            enrichment = self._fetch_match_enrichment(
                home_team,
                away_team,
                commence,
                enrich_fields,
            )

            # Apply enrichment to all rows for this match
            match_mask = (enriched["home_team"] == home_team) & (enriched["away_team"] == away_team)
            for col, value in enrichment.items():
                enriched.loc[match_mask, col] = value

        return enriched

    def _fetch_match_enrichment(
        self,
        home_team: str,
        away_team: str,
        commence: Optional[datetime] = None,
        fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Fetch enrichment data for a specific match.

        Returns dict with form, xG, corners, h2h data.
        """
        enrichment = {}
        fields = fields or ["form", "xg", "corners", "h2h"]

        try:
            # Get fixture ID first
            fixture_id = self._find_fixture_id(home_team, away_team, commence)
            if not fixture_id:
                logger.warning(f"Could not find fixture for {home_team} vs {away_team}")
                return enrichment

            if "form" in fields:
                enrichment.update(self._fetch_form(home_team, away_team))

            if "xg" in fields:
                enrichment.update(self._fetch_xg(fixture_id))

            if "corners" in fields:
                enrichment.update(self._fetch_corners(home_team, away_team))

            if "h2h" in fields:
                enrichment.update(self._fetch_h2h(fixture_id))

            if "injuries" in fields:
                enrichment.update(self._fetch_injuries(fixture_id))

        except Exception as exc:
            logger.error(f"Enrichment error for {home_team} vs {away_team}: {exc}")

        return enrichment

    def fetch_match_availability(
        self,
        home_team: str,
        away_team: str,
        commence: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Fetch injury / suspension summary for a match.

        Returns dict with counts for both teams. Safe to call when the API key
        is absent; it will simply return an empty dict.
        """
        if not self.api_key:
            return {}

        try:
            fixture = self._find_fixture(home_team, away_team, commence)
            if not fixture:
                return {}
            fixture_id = fixture.get("id")
            if not fixture_id:
                return {}
            home_name = fixture.get("teams", {}).get("home", {}).get("name", home_team)
            away_name = fixture.get("teams", {}).get("away", {}).get("name", away_team)
            payload = self._fetch_injuries(int(fixture_id), home_name=home_name, away_name=away_name)
            payload.update(self._fetch_lineups(int(fixture_id), home_name=home_name, away_name=away_name))
            return payload
        except Exception as exc:
            logger.error("Availability fetch error for %s vs %s: %s", home_team, away_team, exc)
            return {}

    def _get_json(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self._is_temporarily_disabled():
            raise RuntimeError(self._disabled_reason or "API-Football enrichment temporarily paused")
        url = f"{self.base_url}/{path.lstrip('/')}"
        resp = self._cache.get(url, headers=self.headers, params=params or {}, timeout=10)
        if resp.status_code == 403:
            self._temporarily_disable(hours=6, reason="403 Forbidden from API-Football")
        elif resp.status_code == 429:
            self._temporarily_disable(hours=2, reason="429 Too Many Requests from API-Football")
        resp.raise_for_status()
        return resp.json()

    def _find_fixture(
        self,
        home_team: str,
        away_team: str,
        commence: Optional[datetime] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Return the matching fixture payload, not just the fixture id.
        """
        try:
            if self._is_temporarily_disabled():
                return None
            if not self._rate_limiter.allow_request():
                logger.warning("API-Football rate limit reached")
                return None

            home_norm = self._normalize_team_name(home_team)
            away_norm = self._normalize_team_name(away_team)
            home_tokens = set(home_norm.split())
            away_tokens = set(away_norm.split())
            commence_dt = self._coerce_datetime(commence)
            fixtures = self._candidate_fixtures(home_team, commence_dt)

            best_fixture = None
            best_score = -1.0

            for fixture in fixtures:
                f_home = fixture.get("teams", {}).get("home", {}).get("name", "")
                f_away = fixture.get("teams", {}).get("away", {}).get("name", "")
                f_home_norm = self._normalize_team_name(f_home)
                f_away_norm = self._normalize_team_name(f_away)
                if not f_home_norm or not f_away_norm:
                    continue

                score = self._team_match_score(home_norm, f_home_norm, home_tokens)
                score += self._team_match_score(away_norm, f_away_norm, away_tokens)
                if commence_dt is not None:
                    fixture_dt = self._coerce_datetime((fixture.get("fixture") or {}).get("date"))
                    if fixture_dt is not None:
                        diff_hours = abs((fixture_dt - commence_dt).total_seconds()) / 3600.0
                        score -= min(diff_hours / 24.0, 1.5)
                if score > best_score:
                    best_score = score
                    best_fixture = fixture

            if best_fixture and best_score >= 1.2:
                return {
                    **(best_fixture.get("fixture", {}) or {}),
                    "teams": best_fixture.get("teams", {}) or {},
                    "league": best_fixture.get("league", {}) or {},
                }
            return None
        except Exception as exc:
            logger.error(f"Error finding fixture payload: {exc}")
            return None

    def _candidate_fixtures(self, home_team: str, commence_dt: Optional[datetime]) -> List[Dict[str, Any]]:
        fixtures: List[Dict[str, Any]] = []
        seen: set[int | str] = set()

        def add_rows(rows: Any) -> None:
            if not isinstance(rows, list):
                return
            for row in rows:
                if not isinstance(row, dict):
                    continue
                fixture_meta = row.get("fixture") or {}
                key = fixture_meta.get("id") or f"{fixture_meta.get('date')}:{row.get('teams')}"
                if key in seen:
                    continue
                seen.add(key)
                fixtures.append(row)

        if commence_dt is not None:
            for offset in (0, -1, 1):
                target_date = (commence_dt + timedelta(days=offset)).date().isoformat()
                try:
                    add_rows(self._get_json("fixtures", params={"date": target_date}).get("response", []))
                except Exception:
                    if self._is_temporarily_disabled():
                        break
                    continue

        if self._is_temporarily_disabled():
            return fixtures

        team_id = self._get_team_id(home_team)
        if team_id:
            for params in (
                {"team": team_id, "season": datetime.now(timezone.utc).year, "next": 20},
                {"team": team_id, "season": datetime.now(timezone.utc).year, "last": 5},
            ):
                try:
                    add_rows(self._get_json("fixtures", params=params).get("response", []))
                except Exception:
                    continue

        if not fixtures and not self._is_temporarily_disabled():
            try:
                add_rows(self._get_json("fixtures", params={"team": home_team, "season": datetime.now(timezone.utc).year, "next": 10}).get("response", []))
            except Exception:
                pass

        return fixtures

    def _find_fixture_id(
        self,
        home_team: str,
        away_team: str,
        commence: Optional[datetime] = None,
    ) -> Optional[int]:
        """
        Find fixture ID for a match by team names and date.

        Returns fixture ID or None.
        """
        try:
            if not self._rate_limiter.allow_request():
                logger.warning("API-Football rate limit reached")
                return None

            fixture = self._find_fixture(home_team, away_team, commence)
            return fixture.get("id") if fixture else None

        except Exception as exc:
            logger.error(f"Error finding fixture ID: {exc}")
            return None

    def _fetch_form(self, home_team: str, away_team: str) -> Dict[str, str]:
        """Fetch recent form (last 5 results)."""
        try:
            if not self._rate_limiter.allow_request():
                return {}

            # Get team ID first
            team_id = self._get_team_id(home_team)
            if not team_id:
                return {}

            payload = self._get_json("fixtures", params={
                "team": team_id,
                "season": datetime.now(timezone.utc).year,
                "last": 5,
            })
            fixtures = payload.get("response", [])

            form_str = ""
            for fixture in fixtures:
                status = fixture.get("fixture", {}).get("status", {}).get("short", "")
                goals_home = fixture.get("goals", {}).get("home")
                goals_away = fixture.get("goals", {}).get("away")

                if status == "FT" and goals_home is not None and goals_away is not None:
                    if goals_home > goals_away:
                        form_str += "W"
                    elif goals_home < goals_away:
                        form_str += "L"
                    else:
                        form_str += "D"

            return {"home_form": form_str[:5]}  # Last 5 games

        except Exception as exc:
            logger.error(f"Error fetching form: {exc}")
            return {}

    def _fetch_xg(self, fixture_id: int) -> Dict[str, float]:
        """Fetch expected goals (xG) if available."""
        try:
            if not self._rate_limiter.allow_request():
                return {}

            payload = self._get_json("fixtures/statistics", params={"fixture": fixture_id})
            stats = payload.get("response", [])

            result = {}
            for stat_group in stats:
                team_type = stat_group.get("team", {}).get("name", "")
                for stat in stat_group.get("statistics", []):
                    if stat.get("type") == "Expected Goals":
                        value = stat.get("value")
                        if team_type.lower() == "home":
                            result["home_xg"] = float(value) if value else 0.0
                        else:
                            result["away_xg"] = float(value) if value else 0.0

            return result

        except Exception as exc:
            logger.error(f"Error fetching xG: {exc}")
            return {}

    def _fetch_corners(self, home_team: str, away_team: str) -> Dict[str, float]:
        """Fetch average corner stats."""
        try:
            if not self._rate_limiter.allow_request():
                return {}

            # Get season stats for corner averages
            team_id = self._get_team_id(home_team)
            if not team_id:
                return {}

            payload = self._get_json("teams/statistics", params={
                "team": team_id,
                "season": datetime.now(timezone.utc).year,
            })
            stats = payload.get("response", {})
            fixtures = stats.get("fixtures", {})

            corners_avg = fixtures.get("corners", {}).get("avg", 0.0)

            return {"home_corners_avg": float(corners_avg) if corners_avg else 0.0}

        except Exception as exc:
            logger.error(f"Error fetching corners: {exc}")
            return {}

    def _fetch_h2h(self, fixture_id: int) -> Dict[str, str]:
        """Fetch head-to-head record."""
        try:
            if not self._rate_limiter.allow_request():
                return {}

            payload = self._get_json("fixtures/headtohead", params={"fixture": fixture_id, "h2h": 10})
            h2h_fixtures = payload.get("response", [])

            home_wins = sum(1 for f in h2h_fixtures
                           if f.get("goals", {}).get("home", 0) > f.get("goals", {}).get("away", 0))
            away_wins = sum(1 for f in h2h_fixtures
                           if f.get("goals", {}).get("away", 0) > f.get("goals", {}).get("home", 0))
            draws = sum(1 for f in h2h_fixtures
                       if f.get("goals", {}).get("home", 0) == f.get("goals", {}).get("away", 0))

            record = f"{home_wins}W-{away_wins}L-{draws}D"

            return {"h2h_record": record}

        except Exception as exc:
            logger.error(f"Error fetching h2h: {exc}")
            return {}

    def _fetch_injuries(
        self,
        fixture_id: int,
        home_name: Optional[str] = None,
        away_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch injury / suspension summary for a fixture."""
        try:
            if not self._rate_limiter.allow_request():
                return {}

            payload = self._get_json("injuries", params={"fixture": fixture_id})
            injuries = payload.get("response", [])
            result: Dict[str, Any] = {
                "home_injuries_count": 0,
                "away_injuries_count": 0,
                "home_suspensions_count": 0,
                "away_suspensions_count": 0,
                "home_absence_severity": 0.0,
                "away_absence_severity": 0.0,
                "home_priority_absences_count": 0,
                "away_priority_absences_count": 0,
                "home_spine_absences_count": 0,
                "away_spine_absences_count": 0,
                "availability_source": "api_football",
            }

            home_norm = self._normalize_team_name(home_name or "")
            away_norm = self._normalize_team_name(away_name or "")
            home_tokens = set(home_norm.split())
            away_tokens = set(away_norm.split())

            for item in injuries:
                team_name = str(((item.get("team") or {}).get("name")) or "")
                team_norm = self._normalize_team_name(team_name)
                reason = str(item.get("reason", "") or "").lower()
                kind = str(item.get("type", "") or "").lower()
                player = item.get("player") or {}
                position = " ".join(
                    str(player.get(field) or item.get(field) or "").lower()
                    for field in ("position", "pos")
                )
                captain_like = bool(player.get("captain") or item.get("captain"))
                side = None
                if home_norm and self._team_match_score(home_norm, team_norm, home_tokens) >= 0.6:
                    side = "home"
                elif away_norm and self._team_match_score(away_norm, team_norm, away_tokens) >= 0.6:
                    side = "away"
                if side is None:
                    continue

                suspension_like = any(token in reason or token in kind for token in ("suspend", "red card", "ban"))
                questionable_like = any(
                    token in reason or token in kind
                    for token in ("question", "doubt", "late fitness", "fitness test", "probable")
                )
                severe_injury_like = any(
                    token in reason or token in kind
                    for token in (
                        "injur", "out", "surgery", "acl", "hamstring", "knee",
                        "ankle", "muscle", "groin", "achilles",
                    )
                )
                if suspension_like:
                    result[f"{side}_suspensions_count"] += 1
                else:
                    result[f"{side}_injuries_count"] += 1
                role_weight = 1.4 if "goal" in position or position == "gk" else 1.25 if any(tok in position for tok in ("forward", "striker", "wing", "fw")) else 1.12 if any(tok in position for tok in ("mid", "dm", "am", "mf")) else 1.0
                if any(tok in position for tok in ("goal", "gk", "mid", "forward", "striker", "fw")):
                    result[f"{side}_spine_absences_count"] += 1
                if captain_like:
                    role_weight += 0.2
                weight = (2.5 if suspension_like else 0.75 if questionable_like else 1.25 if severe_injury_like else 1.0) * role_weight
                result[f"{side}_absence_severity"] += weight
                if suspension_like or severe_injury_like:
                    result[f"{side}_priority_absences_count"] += 1

            return result
        except Exception as exc:
            logger.error(f"Error fetching injuries: {exc}")
            return {}

    def _fetch_lineups(
        self,
        fixture_id: int,
        home_name: Optional[str] = None,
        away_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch lineup publication context for a fixture."""
        try:
            if not self._rate_limiter.allow_request():
                return {}

            payload = self._get_json("fixtures/lineups", params={"fixture": fixture_id})
            lineups = payload.get("response", [])
            result: Dict[str, Any] = {
                "home_lineup_confirmed": 0,
                "away_lineup_confirmed": 0,
                "home_likely_starters_count": 0,
                "away_likely_starters_count": 0,
                "home_lineup_spine_count": 0,
                "away_lineup_spine_count": 0,
                "home_lineup_goalkeeper_named": 0,
                "away_lineup_goalkeeper_named": 0,
                "lineup_source": "api_football",
            }

            home_norm = self._normalize_team_name(home_name or "")
            away_norm = self._normalize_team_name(away_name or "")
            home_tokens = set(home_norm.split())
            away_tokens = set(away_norm.split())

            for item in lineups:
                team_name = str(((item.get("team") or {}).get("name")) or "")
                team_norm = self._normalize_team_name(team_name)
                side = None
                if home_norm and self._team_match_score(home_norm, team_norm, home_tokens) >= 0.6:
                    side = "home"
                elif away_norm and self._team_match_score(away_norm, team_norm, away_tokens) >= 0.6:
                    side = "away"
                if side is None:
                    continue

                start_xi = item.get("startXI") or item.get("startXi") or []
                starters_count = len(start_xi) if isinstance(start_xi, list) else 0
                result[f"{side}_likely_starters_count"] = starters_count
                spine_count = 0
                goalkeeper_named = 0
                for starter in start_xi if isinstance(start_xi, list) else []:
                    player = starter.get("player") or {}
                    pos = str(player.get("pos") or player.get("position") or starter.get("position") or "").lower()
                    if "goal" in pos or pos == "gk":
                        goalkeeper_named = 1
                        spine_count += 1
                    elif any(tok in pos for tok in ("mid", "mf", "forward", "striker", "fw")):
                        spine_count += 1
                result[f"{side}_lineup_spine_count"] = spine_count
                result[f"{side}_lineup_goalkeeper_named"] = goalkeeper_named
                if starters_count >= 10:
                    result[f"{side}_lineup_confirmed"] = 1

            return result
        except Exception as exc:
            logger.debug("Error fetching lineups: %s", exc)
            return {}

    @staticmethod
    def _normalize_team_name(name: str) -> str:
        cleaned = re.sub(r"[^a-z0-9\s]", " ", (name or "").lower())
        tokens = [tok for tok in cleaned.split() if tok and tok not in _TEAM_STOPWORDS]
        return " ".join(tokens)

    @staticmethod
    def _team_match_score(target_norm: str, candidate_norm: str, target_tokens: set[str]) -> float:
        if not target_norm or not candidate_norm:
            return 0.0
        if target_norm == candidate_norm:
            return 1.0
        candidate_tokens = set(candidate_norm.split())
        overlap = len(target_tokens & candidate_tokens)
        if overlap == 0:
            return 0.0
        coverage = overlap / max(len(target_tokens), 1)
        containment = 0.2 if target_norm in candidate_norm or candidate_norm in target_norm else 0.0
        return coverage + containment

    @staticmethod
    def _coerce_datetime(value: Any) -> Optional[datetime]:
        try:
            if value is None:
                return None
            if isinstance(value, datetime):
                return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
            text = str(value).replace("Z", "+00:00")
            dt = datetime.fromisoformat(text)
            return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None

    def _get_team_id(self, team_name: str) -> Optional[int]:
        """Get team ID by name."""
        try:
            if not self.api_key or self._is_temporarily_disabled():
                return None
            if not self._rate_limiter.allow_request():
                return None

            payload = self._get_json("teams", params={"name": team_name})
            teams = payload.get("response", [])
            if teams:
                return teams[0].get("team", {}).get("id")

            return None

        except Exception as exc:
            logger.error(f"Error getting team ID: {exc}")
            return None
