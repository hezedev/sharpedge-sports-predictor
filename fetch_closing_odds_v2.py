"""
fetch_closing_odds_v2.py
========================

REFACTORED closing odds fetcher using hybrid multi-source strategy.

Features:
- Primary: Betfair (unlimited free tier)
- Fallback: The Odds API (500 req/month)
- Optional: API-Football enrichment for soccer (100 req/day)
- Intelligent quota tracking across all sources
- Aggressive caching to minimize API calls

Usage:
    python fetch_closing_odds_v2.py                    # hybrid auto-selection
    python fetch_closing_odds_v2.py --source betfair  # force primary
    python fetch_closing_odds_v2.py --source odds_api # force fallback
    python fetch_closing_odds_v2.py --enrich-soccer   # add granular stats
    python fetch_closing_odds_v2.py --dry-run         # preview without API calls
    python fetch_closing_odds_v2.py --quota-report    # show usage across sources

Workflow (cron):
    08:00  python daily_scan.py --record-bets --notify        # morning picks
    17:00  python fetch_closing_odds_v2.py --enrich-soccer    # closing lines
    02:00  python settle.py                                    # settlement + CLV

Cost comparison (typical 15 sport-keys, daily run):
    Old (Odds API only): 15 req/day × 30 days = 450 req/month (92% of quota)
    New (Betfair primary): 0 req/month from Odds API (only fallback if needed)
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fetch_closing_v2")
sys.path.insert(0, ".")

from src.data.hybrid_odds_fetcher import HybridOddsFetcher
from src.data.api_football_enricher import APIFootballEnricher

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

_CACHE_DIR = Path("data/cache/odds")
_TRACKER_DIR = Path("data/tracker")
_QUOTA_TRACKER = Path("data/quota_tracker.json")

# Map our internal sport names → canonical sport keys
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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pending_sport_keys() -> List[str]:
    """
    Return the specific sport keys needed to cover all pending bets
    in the next 48h. Narrows down to relevant sports to minimize API calls.
    """
    pred_path = _TRACKER_DIR / "predictions.parquet"
    if not pred_path.exists():
        return []

    try:
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

    # For soccer: check existing daily cache to find which leagues have our games
    if "soccer" in sports_needed:
        pending_soccer = pending[pending["sport"] == "soccer"]
        pending_teams = set()
        for mid in pending_soccer["match_id"].dropna():
            parts = str(mid).split(" vs ")
            pending_teams.update(p.lower().strip() for p in parts)

        matched_keys: List[str] = []
        all_soccer_keys = _SPORT_KEYS["soccer"]

        for sk in all_soccer_keys:
            today_str = now.strftime("%Y-%m-%d")
            candidates = sorted(_CACHE_DIR.glob(f"{today_str}*_{sk}.json"))
            if not candidates:
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
                matched_keys.append(sk)

        if matched_keys:
            logger.info("Soccer: narrowed to %d / %d leagues: %s",
                        len(matched_keys), len(all_soccer_keys), matched_keys)
            result_keys.extend(matched_keys)
        else:
            result_keys.extend(all_soccer_keys)

    return list(dict.fromkeys(result_keys))


def _cache_path(sport_key: str, timestamp: str) -> Path:
    """Return path for a timestamped closing-odds snapshot."""
    return _CACHE_DIR / f"{timestamp}_{sport_key}.json"


def _save_snapshot(sport_key: str, games: List[dict], timestamp: str) -> Path:
    """Save odds snapshot to disk."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(sport_key, timestamp)
    path.write_text(json.dumps(games, indent=2))
    logger.info("Saved %d games → %s", len(games), path.name)
    return path


def _print_pending_summary() -> None:
    """Print summary of pending bets for next 24h."""
    try:
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


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch closing odds using hybrid multi-source strategy"
    )
    parser.add_argument("--source", choices=["betfair", "odds_api", "hybrid"],
                        default="hybrid",
                        help="Data source (default: hybrid with auto-fallback)")
    parser.add_argument("--sport", help="Only fetch for this sport (e.g. mlb)")
    parser.add_argument("--enrich-soccer", action="store_true",
                        help="Enrich soccer data with API-Football stats")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't call APIs; preview only")
    parser.add_argument("--all-sports", action="store_true",
                        help="Fetch all sports, not just ones with pending bets")
    parser.add_argument("--quota-report", action="store_true",
                        help="Show quota usage and exit")
    parser.add_argument("--health-check", action="store_true",
                        help="Check API health and exit")
    args = parser.parse_args()

    now_utc = datetime.now(tz=timezone.utc)
    timestamp = now_utc.strftime("%Y-%m-%d_%H%M")

    logger.info("=== Closing Odds Fetch (v2 - Hybrid) — %s UTC ===",
                now_utc.strftime("%Y-%m-%d %H:%M"))

    # Initialize hybrid fetcher
    hybrid = HybridOddsFetcher(
        prefer_source="betfair" if args.source != "odds_api" else "odds_api",
        quota_tracker_path=_QUOTA_TRACKER,
    )

    # Health check mode
    if args.health_check:
        health = hybrid.health_check()
        print(json.dumps(health, indent=2, default=str))
        return

    # Quota report mode
    if args.quota_report:
        report = hybrid.get_quota_report()
        print(json.dumps(report, indent=2, default=str))
        return

    # Determine which sport keys to fetch
    if args.sport:
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

    logger.info("Will fetch %d sport-keys: %s", len(sport_keys_to_fetch), sport_keys_to_fetch)

    # Fetch odds using hybrid strategy
    all_games = []
    source_breakdown = {}

    for sport_key in sport_keys_to_fetch:
        # Determine sport from key
        sport = next((s for s, keys in _SPORT_KEYS.items() if sport_key in keys), "unknown")

        # Create fetcher for this sport
        fetcher = HybridOddsFetcher(
            sport=sport,
            prefer_source="betfair" if args.source != "odds_api" else "odds_api",
        )

        # Fetch odds
        odds_df, meta = fetcher.fetch_odds(sport_key=sport_key, dry_run=args.dry_run)

        source = meta.get("source", "unknown")
        if source not in source_breakdown:
            source_breakdown[source] = 0
        source_breakdown[source] += 1

        if odds_df.empty:
            logger.warning("No odds for %s (source: %s)", sport_key, source)
            continue

        # Optionally enrich soccer data
        if sport == "soccer" and args.enrich_soccer and not args.dry_run:
            try:
                enricher = APIFootballEnricher()
                odds_df = enricher.enrich_soccer_odds(odds_df)
                logger.info("Enriched %s with API-Football stats", sport_key)
            except Exception as exc:
                logger.warning("Could not enrich %s: %s", sport_key, exc)

        # Convert to JSON-compatible format and save
        games = odds_df.to_dict("records") if not odds_df.empty else []
        if games and not args.dry_run:
            _save_snapshot(sport_key, games, timestamp)
            all_games.extend(games)

    # Summary
    logger.info("=== Fetch Complete ===")
    logger.info("Source breakdown: %s", source_breakdown)
    logger.info("Total games fetched: %d", len(all_games))

    # Quota report
    report = hybrid.get_quota_report()
    logger.info("Quota usage (this month): %s", report.get("this_month", {}))

    # Print pending bets summary
    _print_pending_summary()


if __name__ == "__main__":
    main()
