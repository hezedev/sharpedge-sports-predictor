"""
Opening odds store — persists the first-seen best h2h odds for every event,
enabling line-movement signals at scan time.

Storage: data/cache/opening_odds_store.json
Key:     event_id (from The Odds API)
Value:   {first_seen, sport, home, away, best_odds: {outcome_name: decimal_odds}}

Line movement interpretation:
    move_raw > 0  (odds shortened) → sharp/informed money came in on this side
    move_raw < 0  (odds drifted)   → public/recreational money faded this side
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_STORE_PATH = Path(__file__).resolve().parents[2] / "data" / "cache" / "opening_odds_store.json"
_MAX_AGE_DAYS = 7


def load() -> dict:
    """Load the opening odds store from disk. Returns empty dict on first run."""
    if not _STORE_PATH.exists():
        return {}
    try:
        return json.loads(_STORE_PATH.read_text())
    except Exception as exc:
        logger.warning("Failed to load opening odds store: %s", exc)
        return {}


def save(store: dict) -> None:
    """Persist the store to disk."""
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        _STORE_PATH.write_text(json.dumps(store, indent=2))
    except Exception as exc:
        logger.warning("Failed to save opening odds store: %s", exc)


def _best_h2h_odds(game: dict) -> Dict[str, float]:
    """Extract best (highest) h2h decimal odds per outcome across all bookmakers."""
    best: Dict[str, float] = {}
    for bk in game.get("bookmakers", []):
        for mkt in bk.get("markets", []):
            if mkt.get("key") != "h2h":
                continue
            for oc in mkt.get("outcomes", []):
                name = oc.get("name", "")
                price = float(oc.get("price", 0))
                if price > best.get(name, 0):
                    best[name] = price
    return best


def record_games(store: dict, games: List[dict]) -> int:
    """
    For any event not yet in the store, save current odds as the opening line.
    Returns the number of new events recorded.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    added = 0
    for g in games:
        eid = g.get("id")
        if not eid or eid in store:
            continue
        best_odds = _best_h2h_odds(g)
        if not best_odds:
            continue
        store[eid] = {
            "first_seen": now_iso,
            "sport": g.get("sport_key", ""),
            "home": g.get("home_team", ""),
            "away": g.get("away_team", ""),
            "best_odds": best_odds,
        }
        added += 1
    return added


def get_movement(
    store: dict,
    game_id: str,
    outcome_name: str,
    current_odds: float,
) -> Optional[dict]:
    """
    Compare current odds against the stored opening line for an outcome.

    Returns a dict with movement info, or None if no opening line exists.

    Keys returned:
        opening_odds  — decimal odds when first seen
        current_odds  — decimal odds now
        move_raw      — opening - current (positive = shortened = sharp signal)
        move_pct      — move_raw / opening
        direction     — "shortened" | "drifted" | "unchanged"
    """
    entry = store.get(game_id)
    if not entry:
        return None
    opening = entry["best_odds"].get(outcome_name)
    if not opening or opening <= 1.0 or current_odds <= 1.0:
        return None

    move_raw = opening - current_odds
    move_pct = move_raw / opening
    if abs(move_pct) < 0.002:
        direction = "unchanged"
    elif move_raw > 0:
        direction = "shortened"
    else:
        direction = "drifted"

    return {
        "opening_odds": round(opening, 3),
        "current_odds": round(current_odds, 3),
        "move_raw": round(move_raw, 3),
        "move_pct": round(move_pct, 4),
        "direction": direction,
    }


def purge_old(store: dict) -> int:
    """Remove entries older than _MAX_AGE_DAYS to keep the file lean."""
    cutoff = datetime.now(timezone.utc).timestamp() - _MAX_AGE_DAYS * 86400
    to_remove = [
        eid for eid, entry in store.items()
        if _entry_age_ok(entry, cutoff)
    ]
    for eid in to_remove:
        del store[eid]
    return len(to_remove)


def _entry_age_ok(entry: dict, cutoff_ts: float) -> bool:
    try:
        ts = datetime.fromisoformat(entry["first_seen"]).timestamp()
        return ts < cutoff_ts
    except Exception:
        return False
