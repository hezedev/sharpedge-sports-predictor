"""
fetch_closing_odds.py
=====================
Fetches current odds for all sports that have pending bets today and saves
them as timestamped snapshots to data/cache/odds/.

Run this ~1 hour before games start (closing line) and again just after
midnight (for overnight settlement). The snapshots are picked up automatically
by settle.py's disk-cache CLV lookup.

Usage:
    python fetch_closing_odds.py              # fetch for all pending sports
    python fetch_closing_odds.py --sport mlb  # specific sport only
    python fetch_closing_odds.py --dry-run    # show what would be fetched, no API calls

Cost:  1 request per sport_key with pending bets today (typically 3-6 requests).
       The daily scan already caches all sports, so this only costs extra if
       run AFTER the scan's disk-cache has expired (next calendar day).

Workflow (add to cron):
    08:00  python daily_scan.py --record-bets --notify   # morning picks + opening odds
    17:00  python fetch_closing_odds.py                  # closing lines (pre-game)
    02:00  python settle.py                              # overnight settlement + CLV
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fetch_closing")
sys.path.insert(0, ".")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

_ODDS_API_KEY  = os.environ.get("ODDS_API_KEY", "")
_ODDS_API_BASE = "https://api.the-odds-api.com/v4"
_CACHE_DIR     = Path("data/cache/odds")
_TRACKER_DIR   = Path("data/tracker")

# Map our internal sport names → Odds API sport keys
_SPORT_KEYS = {
    "soccer": [
        "soccer_epl", "soccer_germany_bundesliga", "soccer_italy_serie_a",
        "soccer_spain_la_liga", "soccer_france_ligue_one",
        "soccer_uefa_champs_league", "soccer_uefa_europa_league",
        "soccer_portugal_primeira_liga", "soccer_netherlands_eredivisie",
        "soccer_belgium_first_div", "soccer_turkey_super_league",
        "soccer_france_ligue_two",
    ],
    "basketball": ["basketball_nba"],
    "mlb":        ["baseball_mlb"],
    "nhl":        ["icehockey_nhl"],
    "tennis": [
        "tennis_atp_french_open", "tennis_atp_wimbledon",
        "tennis_atp_us_open", "tennis_atp_australian_open",
        "tennis_atp_barcelona_open",
    ],
}

_MARKETS     = "h2h,spreads,totals"
_REGIONS     = "eu,uk,us,au"
_ODDS_FORMAT = "decimal"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pending_sport_keys() -> List[str]:
    """
    Return the specific Odds API sport keys needed to cover all pending bets
    in the next 48h. For non-soccer sports this is one key each. For soccer
    we narrow down by checking which league keys have a cached file whose games
    include one of our pending teams — falling back to all soccer keys if unsure.
    """
    pred_path = _TRACKER_DIR / "predictions.parquet"
    if not pred_path.exists():
        return []

    try:
        import pandas as pd
        df = pd.read_parquet(pred_path)
    except Exception as exc:
        logger.warning("Cannot read predictions: %s", exc)
        return []

    if df.empty or "status" not in df.columns:
        return []

    now = datetime.now(tz=timezone.utc)
    window_end = now + timedelta(hours=48)
    df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True, errors="coerce")
    pending = df[
        (df["status"] == "pending") &
        (df["commence_time"] >= now - timedelta(hours=3)) &
        (df["commence_time"] <= window_end)
    ]

    if pending.empty:
        return []

    sports_needed = list(pending["sport"].unique())
    logger.info("Pending bets in next 48h: %d bets across %s", len(pending), sports_needed)

    # For non-soccer sports, map directly
    result_keys: List[str] = []
    for sport in sports_needed:
        if sport == "soccer":
            continue
        result_keys.extend(_SPORT_KEYS.get(sport, []))

    # For soccer: check existing daily cache files to find which leagues
    # contain our pending teams — avoids fetching all 12 soccer leagues.
    if "soccer" in sports_needed:
        pending_soccer = pending[pending["sport"] == "soccer"]
        pending_teams = set()
        for mid in pending_soccer["match_id"].dropna():
            parts = str(mid).split(" vs ")
            pending_teams.update(p.lower().strip() for p in parts)

        matched_keys: List[str] = []
        all_soccer_keys = _SPORT_KEYS["soccer"]

        for sk in all_soccer_keys:
            # Check today's existing cache for this key
            today_str = now.strftime("%Y-%m-%d")
            candidates = sorted(_CACHE_DIR.glob(f"{today_str}*_{sk}.json"))
            if not candidates:
                # No cache yet — include it (we don't know if it has our games)
                matched_keys.append(sk)
                continue
            try:
                games = json.loads(candidates[-1].read_text())
                for g in games:
                    ht = (g.get("home_team") or "").lower()
                    at = (g.get("away_team") or "").lower()
                    if any(t in ht or ht in t or t in at or at in t
                           for t in pending_teams if len(t) > 3):
                        matched_keys.append(sk)
                        break
            except Exception:
                matched_keys.append(sk)  # include on error

        if matched_keys:
            logger.info("Soccer: narrowed to %d / %d leagues: %s",
                        len(matched_keys), len(all_soccer_keys), matched_keys)
            result_keys.extend(matched_keys)
        else:
            # Couldn't narrow — fetch all soccer keys
            result_keys.extend(all_soccer_keys)

    return list(dict.fromkeys(result_keys))  # dedupe, preserve order


def _cache_path(sport_key: str, timestamp: str) -> Path:
    """Return path for a timestamped closing-odds snapshot."""
    return _CACHE_DIR / f"{timestamp}_{sport_key}.json"


def _fetch_sport_odds(sport_key: str, dry_run: bool = False) -> Optional[List[dict]]:
    """Fetch current odds for one sport key. Returns list of game dicts."""
    if dry_run:
        logger.info("[dry-run] Would fetch: %s", sport_key)
        return None

    if not _ODDS_API_KEY:
        logger.error("ODDS_API_KEY not set — cannot fetch odds")
        return None

    url = f"{_ODDS_API_BASE}/sports/{sport_key}/odds/"
    params = {
        "apiKey":      _ODDS_API_KEY,
        "regions":     _REGIONS,
        "markets":     _MARKETS,
        "oddsFormat":  _ODDS_FORMAT,
        "dateFormat":  "iso",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        remaining = int(resp.headers.get("x-requests-remaining", -1))
        used      = int(resp.headers.get("x-requests-used", -1))

        if resp.status_code == 422:
            logger.info("Sport %s: no active events (422)", sport_key)
            return []

        resp.raise_for_status()
        games = resp.json()

        logger.info(
            "Fetched %s: %d games  |  quota remaining: %d  (used: %d)",
            sport_key, len(games), remaining, used,
        )
        return games

    except requests.exceptions.HTTPError as e:
        logger.warning("HTTP error for %s: %s", sport_key, e)
        return None
    except Exception as exc:
        logger.error("Failed to fetch %s: %s", sport_key, exc)
        return None


def _save_snapshot(sport_key: str, games: List[dict], timestamp: str) -> Path:
    """Save odds snapshot to disk. Returns the file path written."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(sport_key, timestamp)
    path.write_text(json.dumps(games, indent=2))
    logger.info("Saved %d games → %s", len(games), path.name)
    return path


def _check_quota() -> Optional[int]:
    """Quick quota check — uses 1 request."""
    if not _ODDS_API_KEY:
        return None
    try:
        resp = requests.get(
            f"{_ODDS_API_BASE}/sports/",
            params={"apiKey": _ODDS_API_KEY},
            timeout=10,
        )
        return int(resp.headers.get("x-requests-remaining", -1))
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch closing odds for sports with pending bets"
    )
    parser.add_argument("--sport", help="Only fetch for this sport (e.g. mlb)")
    parser.add_argument("--dry-run", action="store_true", help="Don't call API")
    parser.add_argument("--all-sports", action="store_true",
                        help="Fetch all sports, not just ones with pending bets")
    args = parser.parse_args()

    now_utc   = datetime.now(tz=timezone.utc)
    timestamp = now_utc.strftime("%Y-%m-%d_%H%M")

    logger.info("=== Closing Odds Fetch — %s UTC ===", now_utc.strftime("%Y-%m-%d %H:%M"))

    # Determine which sport keys to fetch
    if args.sport:
        # Single sport specified — use all keys for that sport
        sport_keys_to_fetch = _SPORT_KEYS.get(args.sport, [])
        if not sport_keys_to_fetch:
            logger.error("Unknown sport '%s'", args.sport)
            return
    elif args.all_sports:
        sport_keys_to_fetch = [sk for keys in _SPORT_KEYS.values() for sk in keys]
    else:
        sport_keys_to_fetch = _pending_sport_keys()

    if not sport_keys_to_fetch:
        logger.info("No pending bets found — nothing to fetch. Run with --all-sports to force.")
        return

    # Check quota before fetching
    if not args.dry_run:
        quota = _check_quota()
        if quota is not None:
            if quota < 10:
                logger.error("Quota too low (%d remaining) — aborting to preserve budget", quota)
                return
            logger.info("Quota check: %d requests remaining  |  will use ~%d",
                        quota, len(sport_keys_to_fetch))

    # Fetch and save
    total_fetched = 0
    total_saved   = 0

    for sk in sport_keys_to_fetch:
        games = _fetch_sport_odds(sk, dry_run=args.dry_run)
        if games is None:
            continue
        if not games:
            continue   # 422 / no events — skip saving empty file
        total_fetched += len(games)
        if not args.dry_run:
            _save_snapshot(sk, games, timestamp)
            total_saved += 1

    logger.info(
        "=== Done: %d sport-keys fetched, %d snapshot files saved ===",
        total_saved, total_saved,
    )

    # Print summary of what's pending
    if not args.dry_run:
        try:
            import pandas as pd
            df = pd.read_parquet(_TRACKER_DIR / "predictions.parquet")
            df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True, errors="coerce")
            now = datetime.now(tz=timezone.utc)
            pending = df[(df["status"] == "pending") &
                         (df["commence_time"] < now + timedelta(hours=24))]
            if not pending.empty:
                print("\nPending bets for next 24h:")
                for _, r in pending.iterrows():
                    ct = r["commence_time"].strftime("%H:%M UTC") if pd.notna(r["commence_time"]) else "?"
                    print(f"  [{r['sport']:>10}]  {r['team_or_player']:<30} @ {r['bet_odds']:.2f}  "
                          f"({ct} · {r['match_id']})")
        except Exception:
            pass


if __name__ == "__main__":
    main()
