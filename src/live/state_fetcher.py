"""
Fetch live tennis match state from SofaScore (free, no API key).

Returns structured match states including sets, games, server, and
recent momentum — everything the edge model needs.
"""

import logging
import time
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

_SS_BASE = "https://api.sofascore.com/api/v1"
_SS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.sofascore.com/",
    "Origin": "https://www.sofascore.com",
}
_LIVE_STATUSES = {"inprogress", "live", "halftime"}

# Simple in-memory cache: event_id → (ts, detail_dict)
_detail_cache: Dict[int, tuple] = {}
_DETAIL_TTL = 30  # seconds


def _get(url: str) -> dict:
    try:
        r = requests.get(url, headers=_SS_HEADERS, timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logger.debug("SofaScore fetch failed %s: %s", url, exc)
        return {}


def _event_detail(event_id: int) -> dict:
    """Fetch and cache individual event detail (game score, server, stats)."""
    now = time.monotonic()
    cached = _detail_cache.get(event_id)
    if cached and now - cached[0] < _DETAIL_TTL:
        return cached[1]

    data = _get(f"{_SS_BASE}/event/{event_id}")
    event = data.get("event") or {}

    # Try to get statistics (ace, df, first serve %)
    stats_data = _get(f"{_SS_BASE}/event/{event_id}/statistics")
    stats = _parse_live_stats(stats_data)

    result = {**event, "_live_stats": stats}
    _detail_cache[event_id] = (now, result)
    return result


def _parse_live_stats(stats_data: dict) -> dict:
    """Extract first serve %, aces, double faults from SofaScore statistics payload."""
    out = {}
    for period in stats_data.get("statistics", []):
        if period.get("period") != "ALL":
            continue
        for group in period.get("groups", []):
            for item in group.get("statisticsItems", []):
                name = item.get("name", "").lower()
                home_val = item.get("home")
                away_val = item.get("away")
                if "first serve" in name and "%" in name:
                    out["p1_1st_pct"] = _pct(home_val)
                    out["p2_1st_pct"] = _pct(away_val)
                elif name == "aces":
                    out["p1_aces"] = _num(home_val)
                    out["p2_aces"] = _num(away_val)
                elif "double fault" in name:
                    out["p1_df"] = _num(home_val)
                    out["p2_df"] = _num(away_val)
                elif "break point" in name and "converted" in name:
                    out["p1_bp_conv"] = _pct(home_val)
                    out["p2_bp_conv"] = _pct(away_val)
    return out


def _pct(val) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(str(val).replace("%", "")) / 100
    except (ValueError, TypeError):
        return None


def _num(val) -> Optional[int]:
    if val is None:
        return None
    try:
        return int(float(str(val)))
    except (ValueError, TypeError):
        return None


def _parse_score(score_obj: dict, n_sets: int) -> List[int]:
    """Extract per-set game counts from a homeScore/awayScore object."""
    games = []
    for i in range(1, n_sets + 2):
        v = score_obj.get(f"period{i}")
        if v is not None:
            try:
                games.append(int(v))
            except (ValueError, TypeError):
                games.append(0)
    return games


def fetch_live_tennis() -> List[dict]:
    """
    Return a list of live tennis matches with full state.

    Each dict contains:
        event_id, p1_name, p2_name,
        p1_sets, p2_sets,
        p1_games_in_set, p2_games_in_set,  (current set)
        p1_games_history, p2_games_history, (games per completed set)
        current_set (1-indexed),
        server (1=p1, 2=p2, None=unknown),
        status, description,
        live_stats (dict of 1st serve %, aces etc. if available),
        recent_games_p1, recent_games_p2  (last 5 games won/lost)
    """
    from datetime import date
    today = date.today().isoformat()
    data = _get(f"{_SS_BASE}/sport/tennis/scheduled-events/{today}")
    events = data.get("events") or []

    live_matches = []
    for ev in events:
        status_obj = ev.get("status") or {}
        st = status_obj.get("type", "notstarted")
        if st not in _LIVE_STATUSES:
            continue

        event_id = ev.get("id")
        p1_name = (ev.get("homeTeam") or {}).get("name", "")
        p2_name = (ev.get("awayTeam") or {}).get("name", "")
        if not p1_name or not p2_name or not event_id:
            continue

        home_score = ev.get("homeScore") or {}
        away_score = ev.get("awayScore") or {}

        p1_sets = int(home_score.get("current") or 0)
        p2_sets = int(away_score.get("current") or 0)
        current_set = p1_sets + p2_sets + 1

        p1_games_history = _parse_score(home_score, current_set - 1)
        p2_games_history = _parse_score(away_score, current_set - 1)

        p1_games_in_set = int(home_score.get(f"period{current_set}") or 0)
        p2_games_in_set = int(away_score.get(f"period{current_set}") or 0)

        # Fetch detail for server and live stats (cached 30s)
        detail = _event_detail(event_id)
        serving_team = detail.get("serving")  # 1=home, 2=away per SofaScore
        live_stats = detail.get("_live_stats") or {}

        # Recent games: flatten per-set game sequences into last 5 game outcomes
        recent = _recent_game_sequence(p1_games_history, p2_games_history,
                                        p1_games_in_set, p2_games_in_set)

        live_matches.append({
            "event_id":          event_id,
            "p1_name":           p1_name,
            "p2_name":           p2_name,
            "p1_sets":           p1_sets,
            "p2_sets":           p2_sets,
            "current_set":       current_set,
            "p1_games_in_set":   p1_games_in_set,
            "p2_games_in_set":   p2_games_in_set,
            "p1_games_history":  p1_games_history,
            "p2_games_history":  p2_games_history,
            "server":            serving_team,
            "status":            st,
            "description":       status_obj.get("description", ""),
            "live_stats":        live_stats,
            "recent_p1":         recent["p1"],
            "recent_p2":         recent["p2"],
        })

    return live_matches


def _recent_game_sequence(p1_hist: List[int], p2_hist: List[int],
                           p1_cur: int, p2_cur: int,
                           window: int = 5) -> dict:
    """
    Reconstruct which player won each game from per-set totals.
    Returns count of games won by each player in last `window` games.
    """
    games_p1, games_p2 = [], []
    for s1, s2 in zip(p1_hist, p2_hist):
        total = s1 + s2
        # Approximate: assume p1 won s1 and p2 won s2 games in this set
        for _ in range(s1):
            games_p1.append(1)
            games_p2.append(0)
        for _ in range(s2):
            games_p1.append(0)
            games_p2.append(1)
    # Add current set partial games
    for _ in range(p1_cur):
        games_p1.append(1)
        games_p2.append(0)
    for _ in range(p2_cur):
        games_p1.append(0)
        games_p2.append(1)

    return {
        "p1": sum(games_p1[-window:]),
        "p2": sum(games_p2[-window:]),
    }
