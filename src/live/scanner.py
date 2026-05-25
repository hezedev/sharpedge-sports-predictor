"""
Live tennis edge scanner.

Runs a full scan across all live matches and all markets
(h2h, totals, spreads) plus model-only signals (set winner,
next game, tiebreak). Results cached for 30 seconds.
"""

import logging
import re
import time
from typing import List

from src.live.state_fetcher import fetch_live_tennis
from src.live.odds_fetcher import fetch_live_tennis_odds, build_odds_index
from src.live.tennis_edge import compute_edges, LiveEdge

logger = logging.getLogger(__name__)

_cache: dict = {"ts": 0.0, "edges": [], "matches": []}
_CACHE_TTL = 30


def scan(force: bool = False) -> dict:
    now = time.monotonic()
    if not force and now - _cache["ts"] < _CACHE_TTL:
        return _build_result(_cache["edges"], _cache["matches"])

    try:
        matches = fetch_live_tennis()
    except Exception as exc:
        logger.error("Live state fetch failed: %s", exc)
        matches = []

    try:
        odds_list = fetch_live_tennis_odds()
    except Exception as exc:
        logger.error("Live odds fetch failed: %s", exc)
        odds_list = []

    # Group odds by normalised match key
    match_odds = _group_odds_by_match(odds_list)

    all_edges: List[LiveEdge] = []
    for match in matches:
        entries = _find_odds_entries(match, match_odds, odds_list)
        try:
            edges = compute_edges(match, entries)
            all_edges.extend(edges)
        except Exception as exc:
            logger.debug("Edge compute failed %s vs %s: %s",
                         match["p1_name"], match["p2_name"], exc)

    # Sort: real edges first (by edge desc), then signals
    real   = sorted([e for e in all_edges if not e.is_signal],
                    key=lambda e: e.edge, reverse=True)
    signals = [e for e in all_edges if e.is_signal]

    ordered = real + signals

    _cache.update({"ts": now, "edges": ordered, "matches": matches})
    logger.info("Live scan: %d matches, %d edges, %d signals",
                len(matches), len(real), len(signals))
    return _build_result(ordered, matches)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


def _group_odds_by_match(odds_list: list) -> dict:
    """Group all market entries by (p1_norm, p2_norm) key."""
    groups: dict = {}
    for entry in odds_list:
        k = (_norm(entry["p1_name"]), _norm(entry["p2_name"]))
        groups.setdefault(k, []).append(entry)
    return groups


def _find_odds_entries(match: dict, match_odds: dict, odds_list: list) -> list:
    p1 = _norm(match["p1_name"])
    p2 = _norm(match["p2_name"])

    # Exact key
    for key in [(p1, p2), (p2, p1)]:
        if key in match_odds:
            return match_odds[key]

    # Surname substring match
    p1s = p1[-7:] if len(p1) > 7 else p1
    p2s = p2[-7:] if len(p2) > 7 else p2
    for (k1, k2), entries in match_odds.items():
        if (p1s in k1 or p1s in k2) and (p2s in k1 or p2s in k2):
            return entries

    return []  # No odds found — model-only signals still generated


def _build_result(edges: List[LiveEdge], matches: list) -> dict:
    return {
        "edges":   [e.to_dict() for e in edges],
        "matches": [_match_summary(m) for m in matches],
        "scanned": len(matches),
        "ts":      time.time(),
    }


def _match_summary(m: dict) -> dict:
    return {
        "p1_name":     m["p1_name"],
        "p2_name":     m["p2_name"],
        "p1_sets":     m["p1_sets"],
        "p2_sets":     m["p2_sets"],
        "p1_games":    m["p1_games_in_set"],
        "p2_games":    m["p2_games_in_set"],
        "current_set": m["current_set"],
        "server":      m.get("server"),
        "description": m.get("description", ""),
        "recent_p1":   m.get("recent_p1", 0),
        "recent_p2":   m.get("recent_p2", 0),
    }
