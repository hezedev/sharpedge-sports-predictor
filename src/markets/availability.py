from __future__ import annotations

import difflib
import json
import logging
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests

from config import settings

from src.data.api_football_enricher import APIFootballEnricher
from src.data.provider_health import provider_quota_low, record_provider_response
from src.utils.cache import DiskCache
from src.utils.helpers import RateLimiter

logger = logging.getLogger(__name__)

_SOCCER_ENRICHER: APIFootballEnricher | None = None
_BASKETBALL_AVAILABILITY: "_BasketballAvailabilityEnricher | None" = None
_MLB_AVAILABILITY: "_MLBLiveAvailabilityEnricher | None" = None
_NHL_AVAILABILITY: "_NHLGoalieAvailabilityEnricher | None" = None


def _lean_context_mode() -> bool:
    return os.environ.get("SCAN_LEAN_CONTEXT", "").strip().lower() in {"1", "true", "yes", "on"}


def _normalize_name(name: str) -> str:
    return " ".join(str(name or "").lower().replace(".", " ").split())


class _BasketballAvailabilityEnricher:
    def __init__(self, cache_expire_hours: int = 6) -> None:
        api_cfg = settings.get("apis", {}).get("api_sports", {})
        self.api_key = os.environ.get("API_SPORTS_KEY", "")
        self.base_url = api_cfg.get("basketball_url", "https://v1.basketball.api-sports.io")
        self.headers = {
            "x-apisports-key": self.api_key,
            "x-rapidapi-host": "v1.basketball.api-sports.io",
        }
        self._rate_limiter = RateLimiter(
            max_calls=api_cfg.get("rate_limit_per_day", 100),
            period_seconds=86400,
        )
        self._cache = DiskCache(cache_name="api_basketball_availability", expire_hours=cache_expire_hours)
        self._aliases = self._load_aliases()

    @staticmethod
    def _load_aliases() -> dict[str, str]:
        try:
            path = Path("data/team_ids.json")
            raw = json.loads(path.read_text())
            aliases = ((raw.get("nba") or {}).get("_aliases") or {})
            return {_normalize_name(k): v for k, v in aliases.items()}
        except Exception:
            return {}

    def _canonical_team_key(self, name: str) -> str:
        norm = _normalize_name(name)
        return self._aliases.get(norm, norm)

    def _coerce_date(self, commence: Any) -> str:
        if commence is None:
            return datetime.now(timezone.utc).date().isoformat()
        try:
            if isinstance(commence, datetime):
                dt = commence.astimezone(timezone.utc) if commence.tzinfo else commence.replace(tzinfo=timezone.utc)
            else:
                dt = datetime.fromisoformat(str(commence).replace("Z", "+00:00"))
                dt = dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            return dt.date().isoformat()
        except Exception:
            return datetime.now(timezone.utc).date().isoformat()

    def _get_json(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        if provider_quota_low("api_sports_basketball"):
            raise RuntimeError("api_sports_basketball quota is low; skipping live availability")
        url = f"{self.base_url}/{path.lstrip('/')}"
        resp = self._cache.get(url, headers=self.headers, params=params, timeout=10)
        if not getattr(resp, "from_cache", False):
            record_provider_response("api_sports_basketball", resp)
        resp.raise_for_status()
        return resp.json()

    def fetch_match_availability(self, home_team: str, away_team: str, commence: Any) -> dict[str, Any]:
        if not self.api_key or not self._rate_limiter.allow_request():
            return {}
        params = {
            "league": 12,
            "date": self._coerce_date(commence),
        }
        try:
            payload = self._get_json("injuries", params=params)
            rows = payload.get("response", [])
        except Exception as exc:
            logger.debug("Basketball availability fetch failed: %s", exc)
            return {}

        home_key = self._canonical_team_key(home_team)
        away_key = self._canonical_team_key(away_team)
        result = {
            "home_injuries_count": 0,
            "away_injuries_count": 0,
            "home_questionable_count": 0,
            "away_questionable_count": 0,
            "home_rotation_absence_severity": 0.0,
            "away_rotation_absence_severity": 0.0,
            "home_priority_absences_count": 0,
            "away_priority_absences_count": 0,
            "availability_source": "api_sports_basketball",
            "lineup_source": "api_sports_basketball",
        }
        for item in rows:
            team = item.get("team") or {}
            player = item.get("player") or {}
            team_name = str(team.get("name") or player.get("team") or item.get("team_name") or "")
            team_key = self._canonical_team_key(team_name)
            if team_key == home_key:
                side = "home"
            elif team_key == away_key:
                side = "away"
            else:
                continue
            status = " ".join(
                str(item.get(field) or "").lower()
                for field in ("status", "reason", "type", "description")
            )
            position = " ".join(
                str(player.get(field) or item.get(field) or "").lower()
                for field in ("position", "pos")
            )
            position_weight = 1.15 if "guard" in position or position in {"g", "pg", "sg"} else 1.1 if "center" in position or position in {"c"} else 1.0
            if any(token in status for token in ("question", "probable", "gtd", "day to day", "day-to-day")):
                result[f"{side}_questionable_count"] += 1
                result[f"{side}_rotation_absence_severity"] += 0.5 * position_weight
            else:
                result[f"{side}_injuries_count"] += 1
                result[f"{side}_rotation_absence_severity"] += 1.0 * position_weight
                result[f"{side}_priority_absences_count"] += 1
        return result


class _NHLGoalieAvailabilityEnricher:
    def __init__(self, cache_expire_hours: int = 12) -> None:
        self.base_url = "https://api-web.nhle.com/v1"
        self._cache = DiskCache(cache_name="nhl_goalie_availability", expire_hours=cache_expire_hours)
        self._aliases = self._load_aliases()

    @staticmethod
    def _load_aliases() -> dict[str, str]:
        try:
            path = Path("data/team_ids.json")
            raw = json.loads(path.read_text())
            aliases = ((raw.get("nhl") or {}).get("_aliases") or {})
            return {_normalize_name(k): v for k, v in aliases.items()}
        except Exception:
            return {}

    def _team_abbrev(self, team_name: str) -> str:
        key = self._aliases.get(_normalize_name(team_name), "")
        if key.startswith("NHL_"):
            return key.split("_", 1)[1]
        return ""

    def _get_json(self, path: str) -> dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        resp = self._cache.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _full_name(player: Any) -> str:
        if not isinstance(player, dict):
            return ""
        if isinstance(player.get("fullName"), dict):
            return str(player.get("fullName", {}).get("default") or "")
        if player.get("fullName"):
            return str(player.get("fullName"))
        first = player.get("firstName")
        last = player.get("lastName")
        if isinstance(first, dict):
            first = first.get("default")
        if isinstance(last, dict):
            last = last.get("default")
        return " ".join(part for part in (str(first or "").strip(), str(last or "").strip()) if part).strip()

    @staticmethod
    def _search_metric(payload: Any, keys: tuple[str, ...]) -> Optional[float]:
        if isinstance(payload, dict):
            for key, value in payload.items():
                if key in keys:
                    try:
                        return float(value)
                    except Exception:
                        pass
                found = _NHLGoalieAvailabilityEnricher._search_metric(value, keys)
                if found is not None:
                    return found
        elif isinstance(payload, list):
            for item in payload:
                found = _NHLGoalieAvailabilityEnricher._search_metric(item, keys)
                if found is not None:
                    return found
        return None

    def _find_goalie_id(self, team_name: str, goalie_name: str) -> Optional[int]:
        abbrev = self._team_abbrev(team_name)
        if not abbrev or not goalie_name:
            return None
        try:
            roster = self._get_json(f"roster/{abbrev}/current")
        except Exception as exc:
            logger.debug("NHL roster lookup failed for %s: %s", team_name, exc)
            return None
        target = _normalize_name(goalie_name)
        goalies = roster.get("goalies", []) if isinstance(roster, dict) else []
        for goalie in goalies:
            name = _normalize_name(self._full_name(goalie))
            if name and (name == target or target in name or name in target):
                try:
                    return int(goalie.get("id"))
                except Exception:
                    return None
        return None

    def fetch_goalie_metrics(self, team_name: str, goalie_name: str) -> dict[str, Any]:
        goalie_id = self._find_goalie_id(team_name, goalie_name)
        if not goalie_id:
            return {}
        try:
            payload = self._get_json(f"player/{goalie_id}/landing")
        except Exception as exc:
            logger.debug("NHL goalie landing lookup failed for %s: %s", goalie_name, exc)
            return {}
        save_pct = self._search_metric(payload, ("savePctg", "savePct", "savePercentage"))
        gaa = self._search_metric(payload, ("gaa", "goalsAgainstAverage"))
        games = self._search_metric(payload, ("gamesPlayed",))
        result: dict[str, Any] = {}
        if save_pct is not None:
            if save_pct > 1.0:
                save_pct = save_pct / 100.0
            result["save_pct"] = round(save_pct, 4)
        if gaa is not None:
            result["gaa"] = round(float(gaa), 3)
        if games is not None:
            result["games_played"] = int(games)
        return result


class _MLBLiveAvailabilityEnricher:
    def __init__(self, cache_expire_hours: int = 4) -> None:
        self.base_url = "https://statsapi.mlb.com/api/v1"
        self._cache = DiskCache(cache_name="mlb_live_availability", expire_hours=cache_expire_hours)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "sports-predictor/1.0"})

    @staticmethod
    def _coerce_date(commence: Any) -> str:
        if commence is None:
            return datetime.now(timezone.utc).date().isoformat()
        try:
            if isinstance(commence, datetime):
                dt = commence.astimezone(timezone.utc) if commence.tzinfo else commence.replace(tzinfo=timezone.utc)
            else:
                dt = datetime.fromisoformat(str(commence).replace("Z", "+00:00"))
                dt = dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            return dt.date().isoformat()
        except Exception:
            return datetime.now(timezone.utc).date().isoformat()

    def _get_json(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        resp = self._cache.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _match_score(left: str, right: str) -> float:
        l_norm = _normalize_name(left)
        r_norm = _normalize_name(right)
        if not l_norm or not r_norm:
            return 0.0
        if l_norm == r_norm:
            return 1.0
        # Same-city rivals (e.g. "New York Yankees" vs "New York Mets", "Los
        # Angeles Angels" vs "Los Angeles Dodgers") can score deceptively high
        # on generic string similarity, so require the mascot/nickname (last
        # token) to match before accepting a fuzzy match at all.
        l_tokens = l_norm.split()
        r_tokens = r_norm.split()
        if l_tokens and r_tokens and l_tokens[-1] != r_tokens[-1]:
            return 0.0
        if l_norm in r_norm or r_norm in l_norm:
            return 0.92
        return difflib.SequenceMatcher(None, l_norm, r_norm).ratio()

    @staticmethod
    def _pitcher_hand(player: dict[str, Any]) -> str:
        hand = player.get("pitchHand") or {}
        if isinstance(hand, dict):
            return str(hand.get("code") or hand.get("description") or "").strip()
        return str(hand or "").strip()

    def _pitcher_stats(self, pitcher_id: Any, season: int) -> dict[str, Any]:
        if not pitcher_id:
            return {}
        try:
            payload = self._get_json(
                f"people/{int(pitcher_id)}/stats",
                {"stats": "season", "group": "pitching", "season": str(season)},
            )
        except Exception as exc:
            logger.debug("MLB pitcher stat lookup failed for %s: %s", pitcher_id, exc)
            return {}
        stats_list = payload.get("stats") or [{}]
        splits = (stats_list[0] or {}).get("splits", [])
        stat = (splits[0].get("stat") or {}) if splits else {}
        return {
            "era": _as_float(stat.get("era"), 4.50),
            "whip": _as_float(stat.get("whip"), 1.30),
            "k9": _as_float(stat.get("strikeoutsPer9Inn"), 8.0),
            "bb9": _as_float(stat.get("walksPer9Inn"), 3.0),
            "ip": _as_float(stat.get("inningsPitched"), 0.0),
            "games_started": int(_as_float(stat.get("gamesStarted"), 0.0)),
        }

    def _game_feed(self, game_pk: Any) -> dict[str, Any]:
        if not game_pk:
            return {}
        try:
            return self._get_json(f"game/{int(game_pk)}/feed/live", {})
        except Exception as exc:
            logger.debug("MLB live game feed failed for %s: %s", game_pk, exc)
            return {}

    @staticmethod
    def _extract_batting_order(feed: dict[str, Any], side: str) -> list[str]:
        box = (((feed.get("liveData") or {}).get("boxscore") or {}).get("teams") or {}).get(side) or {}
        batting_order = box.get("battingOrder") or []
        players = box.get("players") or {}
        names: list[str] = []
        for raw_id in batting_order:
            key = f"ID{raw_id}"
            player = (players.get(key) or {}).get("person") or {}
            name = str(player.get("fullName") or "").strip()
            if name:
                names.append(name)
        return names

    @staticmethod
    def _snapshot_starter_name(snapshot: Any, side: str) -> str:
        if snapshot is None:
            return ""
        for key in (f"{side}_starter_name", f"{side}_pitcher_name"):
            try:
                value = snapshot.get(key)
            except Exception:
                value = None
            if value:
                return str(value).strip()
        return ""

    def fetch_match_availability(self, home_team: str, away_team: str, commence: Any) -> dict[str, Any]:
        target_date = self._coerce_date(commence)
        try_dates = [target_date]
        try:
            dt = date.fromisoformat(target_date)
            try_dates.extend([(dt - timedelta(days=1)).isoformat(), (dt + timedelta(days=1)).isoformat()])
        except Exception:
            pass

        home_norm = _normalize_name(home_team)
        away_norm = _normalize_name(away_team)
        for date_str in try_dates:
            try:
                payload = self._get_json(
                    "schedule",
                    {
                        "sportId": 1,
                        "startDate": date_str,
                        "endDate": date_str,
                        "hydrate": "probablePitcher,venue",
                    },
                )
            except Exception as exc:
                logger.debug("MLB live availability fetch failed for %s: %s", date_str, exc)
                continue

            for date_entry in payload.get("dates", []):
                for game in date_entry.get("games", []):
                    teams = game.get("teams", {})
                    home = teams.get("home", {})
                    away = teams.get("away", {})
                    game_home = _normalize_name(((home.get("team") or {}).get("name")) or "")
                    game_away = _normalize_name(((away.get("team") or {}).get("name")) or "")
                    if self._match_score(game_home, home_norm) < 0.72 or self._match_score(game_away, away_norm) < 0.72:
                        continue
                    home_pitcher = home.get("probablePitcher") or {}
                    away_pitcher = away.get("probablePitcher") or {}
                    game_pk = game.get("gamePk")
                    season = int(str(date_str)[:4])
                    home_pitcher_id = home_pitcher.get("id")
                    away_pitcher_id = away_pitcher.get("id")
                    home_stats = self._pitcher_stats(home_pitcher_id, season)
                    away_stats = self._pitcher_stats(away_pitcher_id, season)
                    feed = self._game_feed(game_pk)
                    home_lineup = self._extract_batting_order(feed, "home")
                    away_lineup = self._extract_batting_order(feed, "away")
                    venue = game.get("venue") or {}
                    result = {
                        "game_pk": game_pk,
                        "game_status": str((game.get("status") or {}).get("detailedState") or ""),
                        "game_start_time": str(game.get("gameDate") or ""),
                        "venue_name": str(venue.get("name") or ""),
                        "home_starter_confirmed": 1 if home_pitcher.get("id") else 0,
                        "away_starter_confirmed": 1 if away_pitcher.get("id") else 0,
                        "home_starter_name": str(home_pitcher.get("fullName") or home_pitcher.get("full_name") or ""),
                        "away_starter_name": str(away_pitcher.get("fullName") or away_pitcher.get("full_name") or ""),
                        "home_starter_hand": self._pitcher_hand(home_pitcher),
                        "away_starter_hand": self._pitcher_hand(away_pitcher),
                        "home_starter_era": home_stats.get("era"),
                        "home_starter_whip": home_stats.get("whip"),
                        "home_starter_k9": home_stats.get("k9"),
                        "home_starter_bb9": home_stats.get("bb9"),
                        "home_starter_ip": home_stats.get("ip"),
                        "home_starter_games_started": home_stats.get("games_started"),
                        "away_starter_era": away_stats.get("era"),
                        "away_starter_whip": away_stats.get("whip"),
                        "away_starter_k9": away_stats.get("k9"),
                        "away_starter_bb9": away_stats.get("bb9"),
                        "away_starter_ip": away_stats.get("ip"),
                        "away_starter_games_started": away_stats.get("games_started"),
                        "home_likely_starters_count": len(home_lineup),
                        "away_likely_starters_count": len(away_lineup),
                        "home_lineup_confirmed": 1 if len(home_lineup) >= 9 else 0,
                        "away_lineup_confirmed": 1 if len(away_lineup) >= 9 else 0,
                        "home_lineup_players": home_lineup[:9],
                        "away_lineup_players": away_lineup[:9],
                        "availability_source": "mlb_stats_api",
                        "lineup_source": "mlb_stats_api",
                    }
                    return {k: v for k, v in result.items() if v not in (None, "")}
        return {}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _soccer_enricher() -> APIFootballEnricher | None:
    global _SOCCER_ENRICHER
    if _SOCCER_ENRICHER is None:
        try:
            _SOCCER_ENRICHER = APIFootballEnricher(cache_expire_hours=12)
        except Exception as exc:
            logger.debug("Could not initialize soccer availability enricher: %s", exc)
            _SOCCER_ENRICHER = None
    return _SOCCER_ENRICHER


def _basketball_availability() -> _BasketballAvailabilityEnricher | None:
    global _BASKETBALL_AVAILABILITY
    if _BASKETBALL_AVAILABILITY is None:
        try:
            _BASKETBALL_AVAILABILITY = _BasketballAvailabilityEnricher(cache_expire_hours=6)
        except Exception as exc:
            logger.debug("Could not initialize basketball availability enricher: %s", exc)
            _BASKETBALL_AVAILABILITY = None
    return _BASKETBALL_AVAILABILITY


def _mlb_availability() -> _MLBLiveAvailabilityEnricher | None:
    global _MLB_AVAILABILITY
    if _MLB_AVAILABILITY is None:
        try:
            _MLB_AVAILABILITY = _MLBLiveAvailabilityEnricher(cache_expire_hours=4)
        except Exception as exc:
            logger.debug("Could not initialize MLB live availability enricher: %s", exc)
            _MLB_AVAILABILITY = None
    return _MLB_AVAILABILITY


def _nhl_availability() -> _NHLGoalieAvailabilityEnricher | None:
    global _NHL_AVAILABILITY
    if _NHL_AVAILABILITY is None:
        try:
            _NHL_AVAILABILITY = _NHLGoalieAvailabilityEnricher(cache_expire_hours=12)
        except Exception as exc:
            logger.debug("Could not initialize NHL goalie availability enricher: %s", exc)
            _NHL_AVAILABILITY = None
    return _NHL_AVAILABILITY


def build_availability_context(
    sport: str,
    game: Optional[dict],
    snapshot: Optional[pd.Series] = None,
) -> dict[str, Any]:
    """
    Return structured availability context for the live scan.

    This keeps external injury / starter signals separate from the adjustment
    math so the true-probability layer can evolve without duplicating fetch
    logic inside each sport runner.
    """
    sport = (sport or "").lower()
    game = game or {}
    snapshot = snapshot if snapshot is not None else pd.Series(dtype=float)
    lean_mode = _lean_context_mode()
    fetched_at = datetime.now(timezone.utc).isoformat()

    if sport == "soccer":
        enricher = _soccer_enricher()
        if not lean_mode and enricher and enricher.api_key:
            payload = enricher.fetch_match_availability(
                home_team=str(game.get("home_team") or game.get("home") or ""),
                away_team=str(game.get("away_team") or game.get("away") or ""),
                commence=game.get("commence_time"),
            )
            payload["home_team_name"] = str(game.get("home_team") or game.get("home") or "")
            payload["away_team_name"] = str(game.get("away_team") or game.get("away") or "")
            payload["availability_fetched_at"] = fetched_at
            return payload
        return {
            "home_team_name": str(game.get("home_team") or game.get("home") or ""),
            "away_team_name": str(game.get("away_team") or game.get("away") or ""),
            "availability_fetched_at": fetched_at,
        }

    if sport == "mlb":
        home_unknown = int(_as_float(snapshot.get("home_sp_unknown"), 0))
        away_unknown = int(_as_float(snapshot.get("away_sp_unknown"), 0))
        context = {
            "home_starter_confirmed": 0 if home_unknown else 1,
            "away_starter_confirmed": 0 if away_unknown else 1,
            "availability_source": "feature_snapshot",
            "availability_fetched_at": fetched_at,
            "home_team_name": str(game.get("home_team") or game.get("home") or ""),
            "away_team_name": str(game.get("away_team") or game.get("away") or ""),
        }
        enricher = _mlb_availability()
        if not lean_mode and enricher:
            live_context = enricher.fetch_match_availability(
                home_team=context["home_team_name"],
                away_team=context["away_team_name"],
                commence=game.get("commence_time"),
            )
            if live_context:
                context.update({k: v for k, v in live_context.items() if v not in (None, "")})
        for side in ("home", "away"):
            expected = _MLBLiveAvailabilityEnricher._snapshot_starter_name(snapshot, side)
            live = str(context.get(f"{side}_starter_name") or "").strip()
            if expected and live and _MLBLiveAvailabilityEnricher._match_score(expected, live) < 0.75:
                context[f"{side}_pitcher_changed"] = 1
                context["pitcher_change_detected"] = 1
                context.setdefault("pitcher_change_note", f"{side} starter changed from {expected} to {live}")
        return context

    if sport in {"basketball", "nhl"}:
        context = {"availability_source": "feature_snapshot", "availability_fetched_at": fetched_at}
        context["home_team_name"] = str(game.get("home_team") or game.get("home") or "")
        context["away_team_name"] = str(game.get("away_team") or game.get("away") or "")
        if sport == "basketball":
            enricher = _basketball_availability()
            if not lean_mode and enricher and enricher.api_key:
                live_context = enricher.fetch_match_availability(
                    home_team=context["home_team_name"],
                    away_team=context["away_team_name"],
                    commence=game.get("commence_time"),
                )
                context.update(live_context)
        if sport == "nhl":
            home_goalie = str(game.get("home_goalie_name") or "").strip()
            away_goalie = str(game.get("away_goalie_name") or "").strip()
            if "home_goalie_confirmed" in game:
                context["home_goalie_confirmed"] = int(bool(game.get("home_goalie_confirmed")))
            if "away_goalie_confirmed" in game:
                context["away_goalie_confirmed"] = int(bool(game.get("away_goalie_confirmed")))
            if home_goalie:
                context["home_goalie_name"] = home_goalie
            if away_goalie:
                context["away_goalie_name"] = away_goalie
            if (
                "home_goalie_confirmed" in context
                or "away_goalie_confirmed" in context
                or home_goalie
                or away_goalie
            ):
                context["lineup_source"] = "nhl_api"
            enricher = _nhl_availability()
            if not lean_mode and enricher:
                if home_goalie:
                    home_metrics = enricher.fetch_goalie_metrics(context["home_team_name"], home_goalie)
                    if home_metrics:
                        if "save_pct" in home_metrics:
                            context["home_goalie_save_pct"] = home_metrics["save_pct"]
                        if "gaa" in home_metrics:
                            context["home_goalie_gaa"] = home_metrics["gaa"]
                        if "games_played" in home_metrics:
                            context["home_goalie_games_played"] = home_metrics["games_played"]
                if away_goalie:
                    away_metrics = enricher.fetch_goalie_metrics(context["away_team_name"], away_goalie)
                    if away_metrics:
                        if "save_pct" in away_metrics:
                            context["away_goalie_save_pct"] = away_metrics["save_pct"]
                        if "gaa" in away_metrics:
                            context["away_goalie_gaa"] = away_metrics["gaa"]
                        if "games_played" in away_metrics:
                            context["away_goalie_games_played"] = away_metrics["games_played"]
                if context.get("lineup_source"):
                    context["availability_source"] = "nhl_api"
        return context

    return {}
