"""
settle.py
=========
Daily settlement script — checks actual results for pending predictions,
records P&L, and computes closing-line value (CLV).

Usage:
    python settle.py                     # settle all overdue pending bets
    python settle.py --date 2026-04-10   # settle bets for a specific date
    python settle.py --dashboard         # just print the P&L dashboard
    python settle.py --pending           # list all pending predictions
    python settle.py --sport tennis      # filter by sport

Settlement sources:
  - Soccer/Basketball: The Odds API (same key used in daily_scan.py)
  - Tennis: Jeff Sackmann ATP CSV (cross-reference match results)
  - Manual:  python settle.py --manual <pred_id> <result> [--closing-odds X]

CLV Interpretation:
  Positive CLV (bet_odds > closing_odds) means you got better than closing
  price — the gold standard of long-run profitability independent of results.
  A system with positive avg CLV over 100+ bets is likely genuinely +EV.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("settle")
sys.path.insert(0, ".")

from src.utils.results_tracker import (
    get_pending,
    get_pending_parlays,
    settle_prediction,
    settle_parlay,
    print_dashboard,
    compute_summary,
    daily_pnl,
    sport_breakdown,
)
from src.utils.sport_registry import soccer_scanable_keys


# ──────────────────────────────────────────────────────────────────────────────
# Result fetchers (sport-specific)
# ──────────────────────────────────────────────────────────────────────────────

_ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
_ODDS_API_BASE = "https://api.the-odds-api.com/v4"

_SPORT_KEYS = {
    "soccer": soccer_scanable_keys(),
    "basketball": ["basketball_nba"],
    "tennis": [
        # Grand Slams
        "tennis_atp_french_open", "tennis_atp_us_open",
        "tennis_atp_wimbledon", "tennis_atp_australian_open",
        # Masters 1000 / 500 / 250 ATP
        "tennis_atp_madrid_open", "tennis_atp_rome_open",
        "tennis_atp_miami_open", "tennis_atp_indian_wells_masters",
        "tennis_atp_canadian_open", "tennis_atp_cincinnati_open",
        "tennis_atp_shanghai_rolex_masters", "tennis_atp_paris_masters",
        "tennis_atp_vienna_open", "tennis_atp_stockholm_open",
        "tennis_atp_barcelona_open", "tennis_atp_monte_carlo_masters",
        # WTA
        "tennis_wta_french_open", "tennis_wta_us_open",
        "tennis_wta_wimbledon", "tennis_wta_australian_open",
        "tennis_wta_madrid_open", "tennis_wta_rome_open",
        "tennis_wta_miami_open", "tennis_wta_indian_wells_masters",
    ],
    "mlb":  ["baseball_mlb"],
    "nhl":  ["icehockey_nhl"],
}


def _fetch_scores(sport_key: str, date_str: str) -> dict:
    """
    Fetch completed game scores from The Odds API for a given sport+date.
    Returns {match_id_or_event_id: {"home": X, "away": Y, "completed": True}}
    """
    if not _ODDS_API_KEY:
        logger.warning("ODDS_API_KEY not set — cannot auto-fetch results")
        return {}
    try:
        url = f"{_ODDS_API_BASE}/sports/{sport_key}/scores/"
        params = {"apiKey": _ODDS_API_KEY, "daysFrom": 3, "dateFormat": "iso"}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        events = r.json()
        results = {}
        for ev in events:
            if not ev.get("completed"):
                continue
            scores = ev.get("scores") or []
            score_map = {s["name"]: s["score"] for s in scores}
            results[ev["id"]] = {
                "home_team":  ev.get("home_team"),
                "away_team":  ev.get("away_team"),
                "commence_time": ev.get("commence_time"),
                "scores":     score_map,
                "completed":  True,
            }
        return results
    except Exception as exc:
        logger.error("Failed to fetch scores for %s: %s", sport_key, exc)
        return {}


_SOFASCORE_SPORT_MAP = {
    "soccer":     "football",
    "basketball": "basketball",
    "tennis":     "tennis",
    "mlb":        "baseball",
    "nhl":        "ice-hockey",
}

_sofascore_headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.sofascore.com/",
}


def _fetch_sofascore(sport: str, date_str: str) -> list[dict]:
    """
    Fetch completed match results from SofaScore (no API key needed).
    Returns list of dicts: {home_team, away_team, home_score, away_score, status}
    """
    ss_sport = _SOFASCORE_SPORT_MAP.get(sport)
    if not ss_sport:
        return []
    url = f"https://api.sofascore.com/api/v1/sport/{ss_sport}/scheduled-events/{date_str}"
    try:
        r = requests.get(url, headers=_sofascore_headers, timeout=10)
        r.raise_for_status()
        events = r.json().get("events") or []
        results = []
        for ev in events:
            status = (ev.get("status") or {}).get("type", "")
            if status not in ("finished", "ended", "afterextratime", "afterpenalties"):
                continue
            home = (ev.get("homeTeam") or {}).get("name", "")
            away = (ev.get("awayTeam") or {}).get("name", "")
            hs = (ev.get("homeScore") or {}).get("current")
            as_ = (ev.get("awayScore") or {}).get("current")
            if home and away and hs is not None and as_ is not None:
                results.append({
                    "home_team":  home,
                    "away_team":  away,
                    "home_score": int(hs),
                    "away_score": int(as_),
                    "status":     status,
                })
        return results
    except Exception as exc:
        logger.debug("SofaScore fetch failed for %s %s: %s", sport, date_str, exc)
        return []


def _sofascore_winner(sport: str, team_or_player: str, home: str, away: str,
                      home_score: int, away_score: int) -> Optional[bool]:
    """Resolve a win/loss from SofaScore data using the same fuzzy matching."""
    if sport == "tennis":
        winner = home if home_score > away_score else away
    else:
        if home_score > away_score:
            winner = home
        elif away_score > home_score:
            winner = away
        else:
            winner = "draw"
    from src.utils.results_tracker import _is_win  # type: ignore
    return _is_win(team_or_player, winner)


def _fuzzy_match_sofascore(ss_events: list[dict], home_hint: str, away_hint: str) -> Optional[dict]:
    """Find the best SofaScore event matching the given home/away team names."""
    def norm(s: str) -> str:
        return s.lower().strip()

    nh, na = norm(home_hint), norm(away_hint)
    best, best_score = None, 0
    for ev in ss_events:
        eh, ea = norm(ev["home_team"]), norm(ev["away_team"])
        score = 0
        if nh and (nh in eh or eh in nh): score += 2
        if na and (na in ea or ea in na): score += 2
        if nh and len(nh) >= 4 and nh[:4] in eh: score += 1
        if na and len(na) >= 4 and na[:4] in ea: score += 1
        if score > best_score:
            best_score = score
            best = ev
    return best if best_score >= 2 else None


_ODDS_DISK_CACHE_DIR = Path("data/cache/odds")


def _closing_odds_from_disk_cache(
    event_id: str,
    sport_key: str,
    team_or_player: str,
    commence_time: Optional[str] = None,
) -> Optional[float]:
    """
    Look up closing odds from the daily disk-cached odds snapshots.

    Strategy:
      1. Find all cache files for this sport key (from both daily_scan.py and
         fetch_closing_odds.py — naming: YYYY-MM-DD[_HHMM]_sportkey.json).
      2. Select the snapshot taken CLOSEST TO (but before) game start time.
         This is the true closing line. If no time is available, use the latest.
      3. Fuzzy-match the game by event_id or home/away team name.
      4. Extract the best available h2h price for the predicted team/outcome.

    Costs 0 API quota — uses files already written to disk.
    """
    if not _ODDS_DISK_CACHE_DIR.exists():
        return None

    # Find all cache files for this sport key
    sport_slug = sport_key.replace("-", "_")
    cache_files = sorted(_ODDS_DISK_CACHE_DIR.glob(f"*_{sport_slug}.json"))
    if not cache_files:
        # Broader search for partial sport key match
        for sk_part in sport_key.split("_")[:2]:
            cache_files = sorted(_ODDS_DISK_CACHE_DIR.glob(f"*{sk_part}*.json"))
            if cache_files:
                break

    if not cache_files:
        return None

    # Parse commence_time to find the best snapshot
    commence_dt = None
    if commence_time:
        try:
            from datetime import datetime as _dt, timezone as _tz
            ct_str = str(commence_time).replace("Z", "+00:00")
            commence_dt = _dt.fromisoformat(ct_str)
            if commence_dt.tzinfo is None:
                commence_dt = commence_dt.replace(tzinfo=_tz.utc)
        except Exception:
            pass

    # Select best snapshot file: most recent one before game start
    # File naming: YYYY-MM-DD_sportkey.json OR YYYY-MM-DD_HHMM_sportkey.json
    def _file_timestamp(p: Path) -> Optional[datetime]:
        name = p.stem  # e.g. "2026-04-19_1700_baseball_mlb" or "2026-04-19_baseball_mlb"
        parts = name.split("_")
        try:
            # Try YYYY-MM-DD_HHMM prefix (from fetch_closing_odds.py)
            if len(parts) >= 2 and len(parts[1]) == 4 and parts[1].isdigit():
                return datetime.strptime(f"{parts[0]} {parts[1]}", "%Y-%m-%d %H%M").replace(
                    tzinfo=timezone.utc
                )
            # Fall back to YYYY-MM-DD only (from daily_scan.py)
            return datetime.strptime(parts[0], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except Exception:
            return None

    best_file = cache_files[-1]  # default: most recent
    if commence_dt is not None:
        # Pick snapshot taken closest to (but not after) game start
        candidates = []
        for f in cache_files:
            ts = _file_timestamp(f)
            if ts is not None and ts <= commence_dt:
                candidates.append((ts, f))
        if candidates:
            # Most recent snapshot before game start = closing line
            best_file = max(candidates, key=lambda x: x[0])[1]

    try:
        games = json.loads(best_file.read_text())
    except Exception:
        return None

    if not games:
        return None

    # Clean team name for matching (strip spread/total suffixes like "-1.5", "Over 220")
    team_raw   = team_or_player.lower()
    team_clean = team_raw.split(" -")[0].split(" +")[0].strip()
    # Remove "over"/"under" prefix for totals
    if team_clean.startswith("over ") or team_clean.startswith("under "):
        team_clean = " ".join(team_clean.split()[1:])

    for game in games:
        gid = game.get("id", "")
        ht  = (game.get("home_team") or "").lower()
        at  = (game.get("away_team") or "").lower()

        # Match by event id or team name
        id_match   = (gid == event_id)
        team_match = (team_clean in ht or ht in team_clean or
                      team_clean in at or at in team_clean or
                      team_raw   in ht or team_raw in at)
        if not (id_match or team_match):
            continue

        # Extract the best h2h price for the predicted team/side
        bookmakers = game.get("bookmakers") or []
        best_price = None
        for bk in bookmakers:
            for mkt in bk.get("markets", []):
                mkt_key = mkt.get("key", "")
                # h2h for moneylines; spreads for AH/puck line; totals for over/under
                if mkt_key not in ("h2h", "h2h_lay", "spreads", "totals"):
                    continue
                for outcome in mkt.get("outcomes", []):
                    oname = (outcome.get("name") or "").lower()
                    price = outcome.get("price")
                    if not price or price <= 1.0:
                        continue
                    if (team_clean in oname or oname in team_clean or
                            team_raw in oname):
                        if best_price is None or price > best_price:
                            best_price = price
        if best_price:
            return best_price

    return None


def _fetch_closing_odds(
    sport_key: str,
    event_id: str,
    team_or_player: str = "",
    commence_time: Optional[str] = None,
) -> Optional[float]:
    """
    Look up the closing moneyline/1X2 market for an event.

    Priority:
    1. Disk cache (free, zero quota) — uses last daily snapshot before game
    2. Odds API odds-history endpoint (costs quota, last resort)

    Returns best available closing decimal odds, or None.
    """
    # Try disk cache first (free)
    cached = _closing_odds_from_disk_cache(event_id, sport_key, team_or_player, commence_time)
    if cached:
        logger.debug("CLV: closing odds from disk cache for %s: %.2f", event_id, cached)
        return cached

    # Fall back to API (costs quota — disabled by default to protect budget)
    # Uncomment to enable live closing-odds fetch:
    # if not _ODDS_API_KEY:
    #     return None
    # try:
    #     url = f"{_ODDS_API_BASE}/sports/{sport_key}/odds-history/"
    #     params = {
    #         "apiKey": _ODDS_API_KEY,
    #         "eventId": event_id,
    #         "markets": "h2h",
    #         "oddsFormat": "decimal",
    #         "dateFormat": "iso",
    #     }
    #     r = requests.get(url, params=params, timeout=10)
    #     r.raise_for_status()
    #     data = r.json()
    #     bookmakers = (data.get("data") or {}).get("bookmakers") or []
    #     prices = []
    #     for bk in bookmakers:
    #         for mkt in bk.get("markets", []):
    #             if mkt.get("key") == "h2h":
    #                 for outcome in mkt.get("outcomes", []):
    #                     prices.append(outcome.get("price", 0))
    #     return max(prices) if prices else None
    # except Exception:
    #     return None

    return None


def _scores_to_winner(match_data: dict) -> Optional[tuple]:
    """
    Parse score data and return (home_score, away_score, home_team, away_team).
    Returns None if scores are missing or unparseable.
    """
    scores = match_data.get("scores") or {}
    home   = match_data.get("home_team", "")
    away   = match_data.get("away_team", "")
    try:
        h_score = int(scores.get(home, 0))
        a_score = int(scores.get(away, 0))
    except (ValueError, TypeError):
        return None
    return h_score, a_score, home, away


def _infer_result_soccer(match_data: dict, team_or_player: str) -> Optional[str]:
    """Return winning team name, 'draw', or None if scores unavailable."""
    r = _scores_to_winner(match_data)
    if r is None:
        return None
    h_score, a_score, home, away = r
    if h_score > a_score:
        return home
    elif a_score > h_score:
        return away
    return "draw"


def _infer_result_basketball(match_data: dict, team_or_player: str) -> Optional[str]:
    """Return winning team name (no draws)."""
    r = _scores_to_winner(match_data)
    if r is None:
        return None
    h_score, a_score, home, away = r
    return home if h_score > a_score else away


def _infer_result_tennis(match_data: dict, team_or_player: str) -> Optional[str]:
    """Return winning player name (API uses home=player1, away=player2)."""
    r = _scores_to_winner(match_data)
    if r is None:
        return None
    h_sets, a_sets, home, away = r
    return home if h_sets > a_sets else away


def _infer_result_mlb(match_data: dict, team_or_player: str) -> Optional[str]:
    """Return winning team name (no draws)."""
    r = _scores_to_winner(match_data)
    if r is None:
        return None
    h_score, a_score, home, away = r
    return home if h_score > a_score else away


def _infer_result_nhl(match_data: dict, team_or_player: str) -> Optional[str]:
    """Return winning team name (OT/SO counts; no draws)."""
    r = _scores_to_winner(match_data)
    if r is None:
        return None
    h_score, a_score, home, away = r
    return home if h_score > a_score else away


def _infer_result(sport: str, match_data: dict, team_or_player: str) -> Optional[str]:
    if sport == "soccer":
        return _infer_result_soccer(match_data, team_or_player)
    elif sport == "basketball":
        return _infer_result_basketball(match_data, team_or_player)
    elif sport == "tennis":
        return _infer_result_tennis(match_data, team_or_player)
    elif sport == "mlb":
        return _infer_result_mlb(match_data, team_or_player)
    elif sport == "nhl":
        return _infer_result_nhl(match_data, team_or_player)
    return None


def _resolve_outcome(team_or_player: str, actual_result: str, match_data: dict) -> bool:
    """
    Determine if a prediction won, given the team/player name and the actual
    result (which is now the winner's name or 'draw').

    Works for both old-style result labels ('home_win'/'away_win') and new-style
    actual team names (e.g. 'Arsenal', 'LA Lakers').
    """
    from src.utils.results_tracker import _is_win
    return _is_win(team_or_player, actual_result)


def _event_date(value: object) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return str(pd.Timestamp(text).date())
    except Exception:
        return text[:10] if len(text) >= 10 else None


def _select_matching_event(
    pred_match_id: str,
    pred_team: str,
    pred_commence: object,
    all_scores: dict,
) -> Optional[dict]:
    match_data = all_scores.get(pred_match_id)
    if match_data:
        return dict(match_data)

    mid_parts = str(pred_match_id).split(" vs ", 1)
    mid_home = mid_parts[0].lower().strip() if len(mid_parts) == 2 else ""
    mid_away = mid_parts[1].lower().strip() if len(mid_parts) == 2 else ""
    team_raw = str(pred_team).lower()
    team_clean = team_raw.split(" -")[0].split(" +")[0].strip()
    pred_date = _event_date(pred_commence)

    candidates: list[dict] = []
    for ev_id, ev_data in all_scores.items():
        ht = (ev_data.get("home_team") or "").lower()
        at = (ev_data.get("away_team") or "").lower()
        home_match = (mid_home and (mid_home in ht or ht in mid_home))
        away_match = (mid_away and (mid_away in at or at in mid_away))
        team_match = (team_clean in ht or ht in team_clean or team_clean in at or at in team_clean)
        if (home_match and away_match) or (mid_home and home_match) or team_match:
            candidate = dict(ev_data)
            candidate["_event_id"] = ev_id
            candidates.append(candidate)

    if not candidates:
        return None
    if pred_date:
        dated = [candidate for candidate in candidates if _event_date(candidate.get("commence_time")) == pred_date]
        if len(dated) == 1:
            return dated[0]
        if len(dated) > 1:
            return None
        return None
    if len(candidates) > 1:
        return None
    return candidates[0]


# ──────────────────────────────────────────────────────────────────────────────
# Main settlement logic
# ──────────────────────────────────────────────────────────────────────────────

def settle_overdue(date_filter: Optional[str] = None, sport_filter: Optional[str] = None):
    """
    Auto-settle all pending predictions whose commence_time has passed.
    """
    pending = get_pending()
    if pending.empty:
        logger.info("No pending predictions to settle.")
        return

    now = datetime.now(tz=timezone.utc)
    pending["commence_time"] = pd.to_datetime(pending["commence_time"], utc=True)
    overdue = pending[pending["commence_time"] < now].copy()

    if date_filter:
        target_date = pd.Timestamp(date_filter).date()
        overdue = overdue[
            overdue["commence_time"].dt.date == target_date
        ]

    if sport_filter:
        overdue = overdue[overdue["sport"] == sport_filter]

    if overdue.empty:
        logger.info("No overdue predictions to settle (filters: date=%s sport=%s).",
                    date_filter, sport_filter)
        return

    logger.info("Attempting to settle %d overdue predictions …", len(overdue))

    # Group by sport for batch fetching
    settled_count = 0
    skipped_count = 0

    for sport in overdue["sport"].unique():
        sport_preds = overdue[overdue["sport"] == sport]
        sport_keys  = _SPORT_KEYS.get(sport, [])

        # Build result lookup from API
        all_scores: dict = {}
        for sk in sport_keys:
            all_scores.update(_fetch_scores(sk, date_filter or ""))

        for _, pred in sport_preds.iterrows():
            match_id = pred["match_id"]
            match_data = _select_matching_event(
                pred_match_id=match_id,
                pred_team=pred["team_or_player"],
                pred_commence=pred.get("commence_time"),
                all_scores=all_scores,
            )

            if not match_data:
                # ── SofaScore fallback ───────────────────────────────────────
                date_str = str(pred["commence_time"])[:10]
                mid_parts = str(match_id).split(" vs ", 1)
                ss_home = mid_parts[0].strip() if len(mid_parts) == 2 else ""
                ss_away = mid_parts[1].strip() if len(mid_parts) == 2 else ""
                ss_events = _fetch_sofascore(sport, date_str)
                ss_match = _fuzzy_match_sofascore(ss_events, ss_home, ss_away)
                if ss_match:
                    logger.info("SofaScore fallback found: %s vs %s (%s-%s)",
                                ss_match["home_team"], ss_match["away_team"],
                                ss_match["home_score"], ss_match["away_score"])
                    won = _sofascore_winner(
                        sport, pred["team_or_player"],
                        ss_match["home_team"], ss_match["away_team"],
                        ss_match["home_score"], ss_match["away_score"],
                    )
                    if won is not None:
                        score_str = f"{ss_match['home_team']} {ss_match['home_score']}-{ss_match['away_score']} {ss_match['away_team']}"
                        settle_prediction(pred["pred_id"], score_str, won=won)
                        settled_count += 1
                    else:
                        skipped_count += 1
                    continue

                logger.warning(
                    "Could not find result for %s %s (match_id=%s) — "
                    "run: python settle.py --manual %s <result>",
                    sport, pred["team_or_player"], match_id, pred["pred_id"][:8],
                )
                skipped_count += 1
                continue

            actual_result = _infer_result(sport, match_data, pred["team_or_player"])
            if not actual_result:
                logger.warning("Could not infer result from score data for %s", pred["pred_id"][:8])
                skipped_count += 1
                continue

            # Resolve won/loss using match context (team names vs result labels)
            won = _resolve_outcome(pred["team_or_player"], actual_result, match_data)

            # Fetch closing odds for CLV (disk cache first, then API)
            closing_odds = None
            ev_id = match_data.get("_event_id", match_id)
            team_name = pred.get("team_or_player", "")
            commence = str(pred.get("commence_time", ""))
            for sk in sport_keys:
                closing_odds = _fetch_closing_odds(sk, ev_id, team_name, commence)
                if closing_odds:
                    break

            result = settle_prediction(
                pred["pred_id"],
                actual_result,
                closing_odds=closing_odds,
                won=won,
            )
            if result:
                settled_count += 1

    logger.info("Settlement complete: %d settled, %d skipped (manual needed).",
                settled_count, skipped_count)
    if skipped_count > 0:
        logger.info(
            "For manual settlement: python settle.py --manual <pred_id_prefix> <result>"
        )


def settle_parlays():
    """
    Auto-settle pending parlays whose legs all have known results.

    A parlay wins only when ALL legs win.  If any leg loses, the parlay loses.
    If any leg result is still unknown (match not finished / not in API), the
    parlay stays pending until the next run.
    """
    import json as _json

    pending = get_pending_parlays()
    if pending.empty:
        logger.info("No pending parlays to settle.")
        return

    logger.info("Checking %d pending parlay(s) …", len(pending))
    settled_count = 0
    waiting_count = 0

    # Cache score lookups per sport to avoid duplicate API calls
    _score_cache: dict = {}

    for _, row in pending.iterrows():
        parlay_id = row["parlay_id"]

        try:
            legs = _json.loads(row.get("legs_json") or "[]")
        except Exception:
            logger.warning("Parlay %s: could not parse legs_json", parlay_id[:8])
            waiting_count += 1
            continue

        if not legs:
            waiting_count += 1
            continue

        all_available = True
        any_lost      = False
        n_legs        = len(legs)
        n_won         = 0

        for leg in legs:
            sport    = leg.get("sport", "")
            match_id = leg.get("match_id", "")
            team     = leg.get("team", "")

            # Fetch scores for this sport (cached)
            if sport not in _score_cache:
                sport_all: dict = {}
                for sk in _SPORT_KEYS.get(sport, []):
                    sport_all.update(_fetch_scores(sk, ""))
                _score_cache[sport] = sport_all

            all_scores = _score_cache[sport]

            # Match by event id or home vs away name
            match_data = _select_matching_event(
                pred_match_id=match_id,
                pred_team=team,
                pred_commence=leg.get("commence_time"),
                all_scores=all_scores,
            )

            if not match_data:
                # SofaScore fallback for this leg
                leg_date = str(leg.get("commence_time", ""))[:10] or datetime.now(timezone.utc).strftime("%Y-%m-%d")
                mid_parts = str(match_id).split(" vs ", 1)
                ss_home = mid_parts[0].strip() if len(mid_parts) == 2 else ""
                ss_away = mid_parts[1].strip() if len(mid_parts) == 2 else ""
                ss_key = f"ss_{sport}_{leg_date}"
                if ss_key not in _score_cache:
                    _score_cache[ss_key] = _fetch_sofascore(sport, leg_date)
                ss_events = _score_cache[ss_key]
                ss_match = _fuzzy_match_sofascore(ss_events, ss_home, ss_away)
                if ss_match:
                    won = _sofascore_winner(
                        sport, team,
                        ss_match["home_team"], ss_match["away_team"],
                        ss_match["home_score"], ss_match["away_score"],
                    )
                    if won is not None:
                        if won:
                            n_won += 1
                        else:
                            any_lost = True
                        continue
                logger.debug(
                    "Parlay %s: leg '%s' (%s) not found in scores — match not completed yet",
                    parlay_id[:8], team, sport,
                )
                all_available = False
                break

            actual_result = _infer_result(sport, match_data, team)
            if not actual_result:
                logger.debug("Parlay %s: no score parseable for leg '%s'", parlay_id[:8], team)
                all_available = False
                break

            leg_won = _resolve_outcome(team, actual_result, match_data)
            if leg_won:
                n_won += 1
            else:
                any_lost = True

        if not all_available:
            waiting_count += 1
            continue

        # All legs have results
        parlay_won = (n_won == n_legs)
        result = settle_parlay(parlay_id, won=parlay_won)
        if result:
            settled_count += 1
            outcome = f"WON ({n_won}/{n_legs} legs)" if parlay_won else f"LOST ({n_won}/{n_legs} legs won)"
            logger.info("Parlay %s settled: %s", parlay_id[:8], outcome)

    logger.info(
        "Parlay settlement complete: %d settled, %d awaiting results.",
        settled_count, waiting_count,
    )


def settle_manual(pred_id_prefix: str, actual_result: str, closing_odds: Optional[float]):
    """Manually settle a prediction by ID prefix."""
    pending = get_pending()
    matches = pending[pending["pred_id"].str.startswith(pred_id_prefix)]

    if matches.empty:
        # Also check all (maybe already settled by mistake)
        from src.utils.results_tracker import _load, _PRED_FILE, _PRED_SCHEMA
        all_preds = _load(_PRED_FILE, _PRED_SCHEMA)
        matches = all_preds[all_preds["pred_id"].str.startswith(pred_id_prefix)]
        if not matches.empty:
            logger.error("Prediction %s… already settled as '%s'",
                         pred_id_prefix, matches.iloc[0]["status"])
        else:
            logger.error("No prediction found with ID prefix: %s", pred_id_prefix)
        return

    if len(matches) > 1:
        logger.error(
            "Ambiguous prefix '%s' matches %d predictions. Use longer prefix.",
            pred_id_prefix, len(matches),
        )
        for _, row in matches.iterrows():
            print(f"  {row['pred_id']}  {row['sport']}  {row['team_or_player']}")
        return

    pred_id = matches.iloc[0]["pred_id"]
    result  = settle_prediction(pred_id, actual_result, closing_odds=closing_odds)
    if result:
        won_str = "✓ WON" if result.get("won") else "✗ LOST"
        print(f"\n  {won_str}  P&L: {result['profit_units']:+.4f} units"
              + (f"  CLV: {result['clv']*100:+.2f}%" if result.get("clv") else ""))


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Settle predictions and track P&L + CLV",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--date", metavar="YYYY-MM-DD",
        help="Only settle predictions from this date",
    )
    parser.add_argument(
        "--sport", choices=["soccer", "basketball", "tennis", "mlb", "nhl"],
        help="Filter by sport",
    )
    parser.add_argument(
        "--manual", nargs=2, metavar=("PRED_ID_PREFIX", "RESULT"),
        help="Manually settle a prediction:\n"
             "  RESULT = home_win | away_win | draw | player1_win | player2_win",
    )
    parser.add_argument(
        "--closing-odds", type=float, metavar="DECIMAL",
        help="Closing odds for CLV calculation (use with --manual)",
    )
    parser.add_argument(
        "--dashboard", action="store_true",
        help="Print P&L dashboard and exit",
    )
    parser.add_argument(
        "--pending", action="store_true",
        help="List all pending predictions and exit",
    )
    parser.add_argument(
        "--export", metavar="FILE.csv",
        help="Export all settled bets to CSV",
    )
    parser.add_argument(
        "--parlays", action="store_true",
        help="List pending parlays and attempt to settle them, then exit",
    )
    args = parser.parse_args()

    if args.parlays:
        import json as _json
        pending_p = get_pending_parlays()
        if pending_p.empty:
            print("No pending parlays.")
        else:
            print(f"\n{len(pending_p)} pending parlay(s):\n")
            for _, row in pending_p.iterrows():
                try:
                    legs = _json.loads(row.get("legs_json") or "[]")
                    leg_str = "  +  ".join(f"{l['team']}({l['odds']})" for l in legs)
                except Exception:
                    leg_str = "?"
                ct = pd.Timestamp(row["recorded_at"]).strftime("%Y-%m-%d")
                print(
                    f"  {row['parlay_id'][:8]}…  [{row['tier']}/{row['bracket']}]  "
                    f"odds={row['combined_odds']:.2f}  EV={row['ev']:.3f}  "
                    f"Kelly={row['kelly_stake_pct']:.1f}%  {ct}\n"
                    f"    {leg_str}"
                )
        settle_parlays()
        print_dashboard()
        return

    if args.dashboard:
        print_dashboard()
        return

    if args.pending:
        pending = get_pending()
        if pending.empty:
            print("No pending predictions.")
        else:
            print(f"\n{len(pending)} pending prediction(s):\n")
            for _, row in pending.iterrows():
                ct = pd.Timestamp(row["commence_time"]).strftime("%Y-%m-%d %H:%M")
                print(
                    f"  {row['pred_id'][:8]}…  {row['sport']:12s}  "
                    f"{row['team_or_player']:25s}  @ {row['bet_odds']:.2f}  "
                    f"edge {row['edge']*100:+.1f}%  {ct}"
                )
        return

    if args.export:
        from src.utils.results_tracker import get_settled
        settled = get_settled()
        if settled.empty:
            print("No settled bets to export.")
        else:
            settled.to_csv(args.export, index=False)
            print(f"Exported {len(settled)} bets → {args.export}")
        return

    if args.manual:
        pred_prefix, result = args.manual
        settle_manual(pred_prefix, result, closing_odds=args.closing_odds)
        print_dashboard()
        return

    # Auto-settle overdue predictions (primary mode)
    settle_overdue(date_filter=args.date, sport_filter=args.sport)
    # Then settle any parlays whose legs are now all resolved
    settle_parlays()
    print_dashboard()


if __name__ == "__main__":
    main()
