"""
Fetch live tennis odds from The Odds API.

Fetches three markets per tournament:
  h2h     — match winner (moneyline)
  spreads — games handicap (e.g. Zverev -3.5 games)
  totals  — total games over/under (e.g. Over 22.5)
"""

import logging
import os
import time
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

_ODDS_BASE = "https://api.the-odds-api.com/v4/sports"
_TENNIS_SPORT_KEYS = [
    "tennis_atp_french_open",
    "tennis_atp_wimbledon",
    "tennis_atp_us_open",
    "tennis_atp_australian_open",
    "tennis_atp_madrid_open",
    "tennis_atp_rome",
    "tennis_atp_monte_carlo_masters",
    "tennis_atp_canadian_open",
    "tennis_atp_cincinnati",
    "tennis_atp_paris_masters",
    "tennis_wta_french_open",
    "tennis_wta_wimbledon",
    "tennis_wta_us_open",
    "tennis_wta_australian_open",
    "tennis_wta_madrid_open",
]

# Cache per (sport_key, market): (ts, list[game_dict])
_odds_cache: Dict[str, tuple] = {}
_ODDS_TTL = 60  # seconds


def fetch_live_tennis_odds(sport_keys: Optional[List[str]] = None) -> List[dict]:
    """
    Return all live odds entries (h2h + spreads + totals) for active tournaments.

    Each entry has: sport_key, market, p1_name, p2_name,
    plus market-specific fields (p1_odds/p2_odds for h2h;
    line/over_odds/under_odds for totals; p1_spread/p1_spread_odds etc for spreads).
    """
    keys = sport_keys or _TENNIS_SPORT_KEYS
    api_key = os.environ.get("ODDS_API_KEY", "")
    if not api_key:
        logger.warning("ODDS_API_KEY not set — live odds unavailable")
        return []

    results = []
    now = time.monotonic()

    for sport_key in keys:
        for market in ("h2h", "spreads", "totals"):
            cache_key = f"{sport_key}:{market}"
            cached = _odds_cache.get(cache_key)
            if cached and now - cached[0] < _ODDS_TTL:
                results.extend(cached[1])
                continue
            games = _fetch_market(sport_key, market, api_key)
            _odds_cache[cache_key] = (now, games)
            results.extend(games)

    return results


def _fetch_market(sport_key: str, market: str, api_key: str) -> List[dict]:
    params = {
        "apiKey":     api_key,
        "regions":    "uk,eu,us,au",
        "markets":    market,
        "oddsFormat": "decimal",
        "inplay":     "true",
    }
    try:
        r = requests.get(f"{_ODDS_BASE}/{sport_key}/odds/", params=params, timeout=10)
        if r.status_code in (422, 404):
            params.pop("inplay", None)
            r = requests.get(f"{_ODDS_BASE}/{sport_key}/odds/", params=params, timeout=10)
        if r.status_code == 404:
            return []
        r.raise_for_status()
        out = []
        for game in r.json():
            parsed = _parse_game(game, sport_key, market)
            if parsed:
                out.append(parsed)
        return out
    except Exception as exc:
        logger.debug("Odds fetch failed %s/%s: %s", sport_key, market, exc)
        return []


def _parse_game(game: dict, sport_key: str, market: str) -> Optional[dict]:
    try:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        if not home or not away:
            return None

        base = {
            "sport_key":     sport_key,
            "market":        market,
            "p1_name":       home,
            "p2_name":       away,
            "commence_time": game.get("commence_time", ""),
        }

        if market == "h2h":
            return {**base, **_best_h2h(game, home, away)}
        elif market == "totals":
            return {**base, **_best_totals(game)}
        elif market == "spreads":
            return {**base, **_best_spreads(game, home, away)}
        return None
    except Exception:
        return None


def _best_h2h(game: dict, home: str, away: str) -> dict:
    best = {"p1_odds": None, "p2_odds": None, "bookmaker": None}
    for bk in game.get("bookmakers", []):
        for mkt in bk.get("markets", []):
            if mkt["key"] != "h2h":
                continue
            om = {o["name"]: o["price"] for o in mkt["outcomes"]}
            h, a = om.get(home), om.get(away)
            if h and a and (best["p1_odds"] is None or h > best["p1_odds"]):
                best = {"p1_odds": h, "p2_odds": a, "bookmaker": bk["key"]}
    return best


def _best_totals(game: dict) -> dict:
    """Return the best-priced Over and Under line."""
    best_over = {"line": None, "over_odds": None, "under_odds": None, "bookmaker": None}
    for bk in game.get("bookmakers", []):
        for mkt in bk.get("markets", []):
            if mkt["key"] != "totals":
                continue
            om = {o["name"]: (o["price"], o.get("point")) for o in mkt["outcomes"]}
            over = om.get("Over")
            under = om.get("Under")
            if over and under:
                if best_over["over_odds"] is None or over[0] > best_over["over_odds"]:
                    best_over = {
                        "line":       over[1],
                        "over_odds":  over[0],
                        "under_odds": under[0],
                        "bookmaker":  bk["key"],
                    }
    return best_over


def _best_spreads(game: dict, home: str, away: str) -> dict:
    """Return the best spread line for the home player (p1)."""
    best = {"p1_spread": None, "p1_spread_odds": None, "p2_spread": None,
            "p2_spread_odds": None, "bookmaker": None}
    for bk in game.get("bookmakers", []):
        for mkt in bk.get("markets", []):
            if mkt["key"] != "spreads":
                continue
            om = {o["name"]: (o["price"], o.get("point")) for o in mkt["outcomes"]}
            h = om.get(home)
            a = om.get(away)
            if h and a:
                if best["p1_spread_odds"] is None or h[0] > best["p1_spread_odds"]:
                    best = {
                        "p1_spread":       h[1],
                        "p1_spread_odds":  h[0],
                        "p2_spread":       a[1],
                        "p2_spread_odds":  a[0],
                        "bookmaker":       bk["key"],
                    }
    return best


def build_odds_index(odds_list: List[dict]) -> Dict[str, List[dict]]:
    """
    Index: normalised_player_name → list of odds entries (one per market).
    """
    index: Dict[str, List[dict]] = {}
    for entry in odds_list:
        for field in ("p1_name", "p2_name"):
            key = _norm(entry[field])
            index.setdefault(key, []).append(entry)
    return index


def _norm(name: str) -> str:
    return name.lower().strip()
