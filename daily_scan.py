#!/usr/bin/env python3
"""
Daily Sports Value Bet Scanner + Parlay Builder
================================================
Fetches the latest match data, runs ML ensemble predictions, compares to
live bookmaker odds, and outputs:
  1. Single-bet value report (edge ≥ 3%)
  2. Parlay report — two tiers:
       • Value parlays    (all legs have edge ≥ 3%)
       • Speculative parlays (model-favoured legs, ML prob ≥ 50%)
     Target brackets: 5x (4.0–6.5), 10x (8.0–13.0), 20x (15.0–26.0)

Usage:
    python daily_scan.py                        # Soccer + Basketball, full report
    python daily_scan.py --sport soccer         # Soccer only
    python daily_scan.py --sport basketball     # Basketball only
    python daily_scan.py --dry-run              # Skip odds API (feature stats only)
    python daily_scan.py --bankroll 5000        # Set bankroll for stake amounts (£)

Cron example (every day at 07:00):
    0 7 * * * cd /path/to/sports_predictor && python daily_scan.py >> logs/daily.log 2>&1
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import warnings
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from config import settings

# Suppress sklearn version mismatch warnings (models trained on a different
# sklearn version still work correctly; noise-free logs are more useful).
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ── project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
load_dotenv(ROOT / ".env", override=True)

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "logs" / "daily_scan.log"),
    ],
)
logger = logging.getLogger("daily_scan")

# ── local imports ─────────────────────────────────────────────────────────────
from src.data.soccer_fetcher import SoccerFetcher
from src.committee import (
    committee_enabled,
    committee_required_for_parlays,
    max_conservative_parlay_legs,
    run_committee_pipeline,
    show_committee_details,
)
from src.data.balldontlie_fetcher import BallDontLieFetcher as BasketballFetcher
from src.data.tennis_fetcher import TennisFetcher
from src.data.mlb_fetcher import MLBFetcher
from src.data.nhl_fetcher import NHLFetcher
from src.features.soccer_features import SoccerFeatureEngineer
from src.features.basketball_features import BasketballFeatureEngineer
from src.features.tennis_features import TennisFeatureEngineer
from src.features.mlb_features import MLBFeatureEngineer
from src.features.nhl_features import NHLFeatureEngineer
from src.models.trainer import ModelTrainer, _SoftVotingWrapper
from src.models.artifacts import calibrator_path_for_tag, get_current_model_tag
from src.models.calibration import EnsembleCalibrator
from src.models.soccer_score_model import SoccerScoreModel
from src.models.mlb_side_model import MLBSideModel
from src.models.basketball_side_model import BasketballSideModel
from src.models.nhl_side_model import NHLSideModel
from src.markets.policy import annotate_bet, filter_and_rank_bets, get_market_policy, summarize_focused_prediction_policy, summarize_market_policy
from src.markets.decision_layer import classify_candidate_decision
from src.markets.availability import build_availability_context
from src.markets.adjustments import build_context_adjustments
from src.markets.environment import build_environment_context
from src.markets.freshness import audit_candidate_freshness
from src.markets.suitability import evaluate_market_suitability
from src.markets.true_probability import (
    PredictionFactor,
    build_pricing_decision,
    estimate_true_probability,
)
from src.notifications.telegram import TelegramNotifier
from src.risk.bankroll import BankrollManager
from src.risk.kelly import KellyCriterion
from src.risk.parlay_builder import ParlayBuilder, ParlayLeg
from src.features.feature_store import (
    FeatureStore,
    TeamResolver,
    build_entity_alias_map,
    resolve_canonical_name,
)
from src.models.totals_trainer import TotalsTrainer
from src.utils.odds_quota import (
    api_key_fingerprint,
    get_odds_budget_status,
    get_primary_odds_api_key,
    load_odds_api_usage,
    parse_odds_api_keys_from_env,
    save_odds_api_usage,
)
from src.utils.sport_registry import (
    SOCCER_ODDS_TO_COMPETITION,
    enrich_with_capability,
    get_capability_profile,
    soccer_pretty_label,
    soccer_scanable_keys,
)


def _policy_version_hash() -> str:
    policy_path = ROOT / "src" / "markets" / "policy.py"
    if not policy_path.exists():
        return "missing"
    return hashlib.sha256(policy_path.read_bytes()).hexdigest()[:12]


def _build_version_snapshot(*, args: argparse.Namespace) -> dict:
    models: Dict[str, dict] = {}
    for logical_sport, (artifact_sport, fallback_tag) in _VERSION_DEFAULT_TAGS.items():
        tag = get_current_model_tag(artifact_sport, fallback=fallback_tag) or fallback_tag
        cal_path = calibrator_path_for_tag(artifact_sport, tag)
        models[logical_sport] = {
            "artifact_sport": artifact_sport,
            "tag": tag,
            "calibrator_present": cal_path.exists(),
        }

    return {
        "scan_date": TODAY,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "policy_hash": _policy_version_hash(),
        "models": models,
        "scan_config": {
            "sport": args.sport,
            "market": args.market,
            "dry_run": bool(args.dry_run),
            "offline_odds": bool(args.offline_odds),
            "lean_context": bool(args.lean_context),
            "context_referee": bool(args.context_referee),
            "retrain": bool(args.retrain),
            "bankroll": float(args.bankroll),
            "min_edge": float(args.min_edge),
            "min_legs": int(args.min_legs),
            "max_legs": int(args.max_legs),
        },
    }

ODDS_KEY = get_primary_odds_api_key()
KELLY = KellyCriterion()
TODAY = datetime.now().strftime("%Y-%m-%d")
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / "logs").mkdir(parents=True, exist_ok=True)

_VERSION_DEFAULT_TAGS: Dict[str, Tuple[str, str]] = {
    "soccer": ("soccer", "pl_2024_25"),
    "basketball": ("basketball", "nba_2024_25"),
    "tennis": ("tennis", "atp_2022_25"),
    "tennis_wta": ("tennis", "wta_2022_24"),
    "mlb": ("mlb", "mlb_2024_25"),
    "nhl": ("nhl", "nhl_2024_25"),
}

# ─────────────────────────────────────────────────────────────────────────────
# Odds API budget management
# ─────────────────────────────────────────────────────────────────────────────

_ODDS_CACHE_TTL_MINUTES = 60        # reuse in-memory cached odds within same hour
_ODDS_BUDGET_WARN  = 100            # warn when remaining drops below this
_ODDS_BUDGET_STOP  = 0              # rotate/stop only when the active key is actually exhausted
_ODDS_KEY_LOW_REMAINING_THRESHOLD = 50
_ODDS_KEY_METADATA_STALE_HOURS = 24.0
_ODDS_AUTH_401_QUARANTINE_HOURS = 24 * 7
_ODDS_AUTH_403_QUARANTINE_HOURS = 24

# Minimum rows in training data for a team/player before we trust the snapshot.
# Teams with fewer appearances produce population-mean features (≈33% win prob)
# which look like easy edges — suppress these to avoid garbage picks.
_MIN_SNAPSHOT_ROWS = 15
_SOCCER_MIN_SNAPSHOT_ROWS = 6
_HISTORY_BLEND_FULL_ROWS = {
    "soccer": 24,
    "basketball": 22,
    "mlb": 24,
    "nhl": 22,
    "tennis": 10,
    "tennis_wta": 10,
}

# Parlays are disabled until single-bet win rate is validated.
# Parlays compound model errors and have produced -91% ROI in live tracking.
# Re-enable once we have 200+ settled single bets with positive ROI.
PARLAYS_ENABLED = False

_odds_remaining: int = 9999         # updated after each live request
_odds_cache: dict = {}              # {sport_key: (fetched_at, data)}  ← in-memory cache
_OFFLINE_ODDS_ONLY = os.environ.get("ODDS_OFFLINE_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}
_FORCE_FRESH_ODDS = False
_LEAN_CONTEXT_ONLY = os.environ.get("SCAN_LEAN_CONTEXT", "").strip().lower() in {"1", "true", "yes", "on"}
_FOCUSED_SCAN_ONLY = os.environ.get("SCAN_FOCUSED_LANES", "").strip().lower() in {"1", "true", "yes", "on"}
_scan_runtime_notes: List[dict] = []

# ── API Throttle ──────────────────────────────────────────────────────────────
# Enforces a minimum gap between live Odds API requests to simulate real-world
# rate limits and prevent accidental quota burns from tight loops.
_last_api_call_at: Optional[datetime] = None
_API_MIN_GAP_SECONDS = 45        # minimum seconds between live requests
_API_MAX_DAILY       = 0         # disabled: rely on live keys/provider limits instead
_api_calls_today: int = 0
_api_calls_date: str = ""        # YYYY-MM-DD of current day's count
_odds_api_auth_failed: bool = False
_odds_api_failed_fingerprints: set[str] = set()
_ODDS_KEY_POOL_FILE = ROOT / "data" / "odds_key_pool.json"


def _current_api_min_gap_seconds(
    *,
    remaining: Optional[int] = None,
    daily_allowance: Optional[int] = None,
    used_today: Optional[int] = None,
) -> int:
    """
    Derive a live Odds API gap from quota health.

    The old fixed 45s gap made full-board scans take 30+ minutes even when the
    monthly quota was effectively untouched. We now keep the conservative
    spacing only when quota is genuinely tight, while healthy quota gets a much
    shorter gap.
    """
    if remaining is None or remaining == 9999:
        return 1

    if remaining > 400:
        return 1
    if remaining > 250:
        return 2
    if remaining > 150:
        return 4
    if remaining > _ODDS_BUDGET_WARN:
        return 8
    return 15

def _api_throttle() -> bool:
    """
    Enforce the 45-second minimum gap between live API calls and the 2,000/day cap.
    Blocks (sleeps) if called too soon after the previous request.
    Returns False if the daily cap is reached (should not fetch at all).
    """
    global _last_api_call_at, _api_calls_today, _api_calls_date
    import time as _time

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if _api_calls_date != today_str:
        _api_calls_date = today_str
        _api_calls_today = 0

    if _API_MAX_DAILY > 0 and _api_calls_today >= _API_MAX_DAILY:
        logger.error(
            "API_THROTTLE: daily cap of %d requests reached — halting fetches",
            _API_MAX_DAILY,
        )
        return False

    budget = get_odds_budget_status(ODDS_KEY)
    min_gap = _current_api_min_gap_seconds(
        remaining=budget.get("remaining"),
        daily_allowance=budget.get("daily_allowance"),
        used_today=budget.get("used_today"),
    )

    if _last_api_call_at is not None:
        elapsed = (datetime.now(timezone.utc) - _last_api_call_at).total_seconds()
        wait = min_gap - elapsed
        if wait > 0:
            logger.info(
                "API_THROTTLE: waiting %.1fs before next request (min gap %ds)",
                wait, min_gap,
            )
            _time.sleep(wait)

    return True

# ── Line Velocity Tracker ─────────────────────────────────────────────────────
# Stores the first observed price for each (sport_key, game_id, team) tuple
# within a session. If the price shifts > _LINE_VELOCITY_THRESHOLD in a short
# window, the game is flagged Abstain_Status to protect against unpriced news.
_line_snapshot: dict = {}   # {(sport_key, game_id, team): (timestamp, price)}
_LINE_VELOCITY_THRESHOLD = 0.05   # 5% price movement triggers abstain
_SOCCER_PRODUCTION_MIN_EDGE = 0.03
_LINE_VELOCITY_WINDOW_MINUTES = 30  # only compare prices within this window

# Manual alias overrides for soccer clubs where automated normalization alone is
# either insufficient or too aggressive. Identity mappings are intentional for
# newly added leagues: they prevent fuzzy matching from jumping to the wrong
# club when we genuinely have no trustworthy history yet.
_SOCCER_ALIAS_OVERRIDES: dict[str, str] = {
    # ── Germany ────────────────────────────────────────────────────────────────
    "Bayern Munich": "FC Bayern München",
    "Bayern": "FC Bayern München",
    "Bayer Leverkusen": "Bayer 04 Leverkusen",
    "Leverkusen": "Bayer 04 Leverkusen",
    "Borussia Dortmund": "Borussia Dortmund",
    "Dortmund": "Borussia Dortmund",
    "Borussia Monchengladbach": "Borussia Mönchengladbach",
    "Mainz": "1. FSV Mainz 05",
    "Mainz 05": "1. FSV Mainz 05",
    "FSV Mainz 05": "1. FSV Mainz 05",
    "Heidenheim": "1. FC Heidenheim 1846",
    "1. FC Heidenheim": "1. FC Heidenheim 1846",
    "Union Berlin": "1. FC Union Berlin",
    "1. FC Union Berlin": "1. FC Union Berlin",
    "Cologne": "1. FC Köln",
    "Koln": "1. FC Köln",
    "1. FC Koln": "1. FC Köln",
    "Hoffenheim": "TSG 1899 Hoffenheim",
    "TSG Hoffenheim": "TSG 1899 Hoffenheim",
    "Werder Bremen": "SV Werder Bremen",
    "Hamburg": "Hamburger SV",
    "Augsburg": "FC Augsburg",
    "Bochum": "VfL Bochum 1848",
    "VfL Bochum": "VfL Bochum 1848",
    "Wolfsburg": "VfL Wolfsburg",
    "Stuttgart": "VfB Stuttgart",
    "Freiburg": "SC Freiburg",
    "St. Pauli": "FC St. Pauli 1910",
    "St Pauli": "FC St. Pauli 1910",
    "RB Leipzig": "RB Leipzig",
    "Eintracht Frankfurt": "Eintracht Frankfurt",
    # ── Spain ──────────────────────────────────────────────────────────────────
    "Real Madrid": "Real Madrid CF",
    "Barcelona": "FC Barcelona",
    "Atletico Madrid": "Club Atlético de Madrid",
    "Atletico de Madrid": "Club Atlético de Madrid",
    "Atlético Madrid": "Club Atlético de Madrid",
    "Real Betis": "Real Betis Balompié",
    "Celta Vigo": "RC Celta de Vigo",
    "Alaves": "Deportivo Alavés",
    "Deportivo Alaves": "Deportivo Alavés",
    "Osasuna": "CA Osasuna",
    "Getafe": "Getafe CF",
    "Mallorca": "RCD Mallorca",
    "Espanyol": "RCD Espanyol de Barcelona",
    "Rayo Vallecano": "Rayo Vallecano de Madrid",
    "Las Palmas": "UD Las Palmas",
    "Leganes": "CD Leganés",
    "Leganés": "CD Leganés",
    "Valladolid": "Real Valladolid CF",
    "Real Valladolid": "Real Valladolid CF",
    "Levante": "Levante UD",
    "Girona": "Girona FC",
    "Sevilla": "Sevilla FC",
    "Villarreal": "Villarreal CF",
    "Valencia": "Valencia CF",
    "Real Sociedad": "Real Sociedad de Fútbol",
    "Athletic Club": "Athletic Club",
    "Athletic Bilbao": "Athletic Club",
    # ── Italy ──────────────────────────────────────────────────────────────────
    "Inter Milan": "FC Internazionale Milano",
    "Inter": "FC Internazionale Milano",
    "AC Milan": "AC Milan",
    "Roma": "AS Roma",
    "Fiorentina": "ACF Fiorentina",
    "Napoli": "SSC Napoli",
    "Lazio": "SS Lazio",
    "Bologna": "Bologna FC 1909",
    "Monza": "AC Monza",
    "Pisa": "AC Pisa 1909",
    "Parma": "Parma Calcio 1913",
    "Cremonese": "US Cremonese",
    "Lecce": "US Lecce",
    "Sassuolo": "US Sassuolo Calcio",
    "Udinese": "Udinese Calcio",
    "Torino": "Torino FC",
    "Genoa": "Genoa CFC",
    "Venezia": "Venezia FC",
    "Empoli": "Empoli FC",
    "Cagliari": "Cagliari Calcio",
    "Hellas Verona": "Hellas Verona FC",
    "Juventus": "Juventus FC",
    "Atalanta": "Atalanta BC",
    # ── France ─────────────────────────────────────────────────────────────────
    "Paris Saint Germain": "Paris Saint-Germain FC",
    "Paris SG": "Paris Saint-Germain FC",
    "PSG": "Paris Saint-Germain FC",
    "Lyon": "Olympique Lyonnais",
    "Marseille": "Olympique de Marseille",
    "Monaco": "AS Monaco FC",
    "Nice": "OGC Nice",
    "Lens": "Racing Club de Lens",
    "RC Lens": "Racing Club de Lens",
    "Rennes": "Stade Rennais FC 1901",
    "Brest": "Stade Brestois 29",
    "Reims": "Stade de Reims",
    "Toulouse": "Toulouse FC",
    "Lorient": "FC Lorient",
    "Metz": "FC Metz",
    "Montpellier": "Montpellier HSC",
    "Angers": "Angers SCO",
    "Le Havre": "Le Havre AC",
    "Auxerre": "AJ Auxerre",
    "Nantes": "FC Nantes",
    "Strasbourg": "RC Strasbourg Alsace",
    "Lille": "Lille OSC",
    "Saint-Etienne": "AS Saint-Étienne",
    "Saint Etienne": "AS Saint-Étienne",
    "AS Saint-Etienne": "AS Saint-Étienne",
    # ── England ────────────────────────────────────────────────────────────────
    "Liverpool": "Liverpool FC",
    "Arsenal": "Arsenal FC",
    "Manchester City": "Manchester City FC",
    "Manchester United": "Manchester United FC",
    "Chelsea": "Chelsea FC",
    "Everton": "Everton FC",
    "Tottenham Hotspur": "Tottenham Hotspur FC",
    "Tottenham": "Tottenham Hotspur FC",
    "Spurs": "Tottenham Hotspur FC",
    "Aston Villa": "Aston Villa FC",
    "Nottingham Forest": "Nottingham Forest FC",
    "Brighton and Hove Albion": "Brighton & Hove Albion FC",
    "Brighton & Hove Albion": "Brighton & Hove Albion FC",
    "Brighton": "Brighton & Hove Albion FC",
    "West Ham United": "West Ham United FC",
    "West Ham": "West Ham United FC",
    "Wolverhampton Wanderers": "Wolverhampton Wanderers FC",
    "Wolves": "Wolverhampton Wanderers FC",
    "Crystal Palace": "Crystal Palace FC",
    "Newcastle United": "Newcastle United FC",
    "Newcastle": "Newcastle United FC",
    "Brentford": "Brentford FC",
    "Fulham": "Fulham FC",
    "Bournemouth": "AFC Bournemouth",
    "AFC Bournemouth": "AFC Bournemouth",
    "Burnley": "Burnley FC",
    "Leeds United": "Leeds United FC",
    "Leicester City": "Leicester City FC",
    "Ipswich Town": "Ipswich Town FC",
    "Sunderland": "Sunderland AFC",
    "Southampton": "Southampton FC",
    "Celtic": "Celtic FC",
    # ── Netherlands ────────────────────────────────────────────────────────────
    "Ajax": "AFC Ajax",
    "PSV Eindhoven": "PSV",
    "PSV": "PSV",
    "Feyenoord": "Feyenoord Rotterdam",
    # ── Denmark / Scandinavia ─────────────────────────────────────────────────
    "Copenhagen": "FC København",
    "FC Copenhagen": "FC København",
    # ── Portugal ───────────────────────────────────────────────────────────────
    "Benfica": "Sport Lisboa e Benfica",
    "SL Benfica": "Sport Lisboa e Benfica",
    "Sporting CP": "Sporting Clube de Portugal",
    "Sporting Lisbon": "Sporting Clube de Portugal",
    # ── J League expansion ────────────────────────────────────────────────────
    "FC Machida Zelvia": "Machida Zelvia",
    "Hiroshima Sanfrecce FC": "Sanfrecce Hiroshima",
    "Kashima Antlers": "Kashima",
    "Kyoto Purple Sanga": "Kyoto Sanga",
    "Urawa Red Diamonds": "Urawa",
    "Yokohama F Marinos": "Yokohama F. Marinos",
    # ── Saudi Pro League expansion ────────────────────────────────────────────
    "Al-Ahli": "Al-Ahli Jeddah",
    "Al-Hazem": "Al-Hazm",
    "Al-Hilal": "Al-Hilal Saudi FC",
    "Al-Ittihad": "Al-Ittihad FC",
    "Al-Khaleej": "Al Khaleej Saihat",
    "Al-Kholood": "Al Kholood",
    "Al-Najma": "Al-Najma",
    "Al-Nassr": "Al-Nassr",
    "Al-Okhdood": "Al Okhdood",
    "Al-Qadsiah": "Al-Qadisiyah FC",
    "Al-Riyadh": "Al Riyadh",
    "Al-Shabab": "Al Shabab",
    "Al-Taawoun": "Al Taawon",
    "Damac": "Damac",
    "Neom": "Neom",
    # ── Copa Libertadores expansion ───────────────────────────────────────────
    "Barcelona SC": "Barcelona SC",
    "Club Always Ready": "Always Ready",
    "Club Universitario de Deportes": "Universitario",
    "Cerro Porteño": "Cerro Porteno",
    "Estudiantes La Plata": "Estudiantes L.P.",
    "Flamengo-RJ": "CR Flamengo",
    "Junior FC": "Junior",
    "LDU Quito": "LDU de Quito",
    "Nacional de Montevideo": "Club Nacional",
    "Palmeiras-SP": "SE Palmeiras",
    "UCV FC": "UCV FC",
}

def _check_line_velocity(sport_key: str, game_id: str, team: str,
                          current_price: float) -> bool:
    """
    Track price movement for a (sport_key, game_id, team) tuple.

    Returns True (safe to bet) if movement is within threshold.
    Returns False (Abstain_Status) if price has moved > 5% since first seen.

    Side-effect: stores the first observed price in _line_snapshot.
    """
    if not game_id or current_price <= 0:
        return True

    key = (sport_key, game_id, team)
    now = datetime.now(timezone.utc)

    if key not in _line_snapshot:
        _line_snapshot[key] = (now, current_price)
        return True

    first_ts, first_price = _line_snapshot[key]
    age_minutes = (now - first_ts).total_seconds() / 60

    # Outside the velocity window — reset snapshot
    if age_minutes > _LINE_VELOCITY_WINDOW_MINUTES:
        _line_snapshot[key] = (now, current_price)
        return True

    if first_price <= 0:
        return True

    # Implied probability shift (price movement in prob space is more meaningful
    # than raw decimal shift — a move from 1.05→1.10 is more significant than 5.0→5.25)
    old_impl = 1.0 / first_price
    new_impl  = 1.0 / current_price
    shift = abs(new_impl - old_impl)

    if shift > _LINE_VELOCITY_THRESHOLD:
        logger.warning(
            "Abstain_Status [%s %s]: implied prob shifted %.1f%% in %.0f min "
            "(%.2f → %.2f) — likely unpriced news, skipping",
            team, sport_key, shift * 100, age_minutes, first_price, current_price,
        )
        return False

    return True

# Disk cache: data/cache/odds/<sport_key>.json
# One file per sport, refreshed when it is older than _ODDS_CACHE_MAX_AGE_HOURS.
# This means re-running the scan multiple times per day, or across midnight,
# costs ZERO additional API requests as long as the file is fresh enough.
# The file covers 3 days of upcoming games so there is no need to re-fetch daily.
_ODDS_DISK_CACHE_DIR  = ROOT / "data" / "cache" / "odds"
_ODDS_CACHE_MAX_AGE_HOURS = 20   # refresh at most once every 20h (saves ~15 req/day)
_ODDS_EMPTY_CACHE_REUSE_HOURS = 6
_ACTIVE_SPORTS_MAX_AGE_HOURS = 24

def _disk_cache_path(sport_key: str) -> "Path":
    """Return the stable (date-independent) disk-cache path for sport_key."""
    return _ODDS_DISK_CACHE_DIR / f"{sport_key}.json"


def _active_sports_cache_path() -> "Path":
    """Return the stable disk-cache path for the active sports list."""
    return _ODDS_DISK_CACHE_DIR / "__active_sports__.json"


def _active_sports_snapshot_path() -> "Path":
    """Return the dated snapshot path for the active sports list."""
    return _ODDS_DISK_CACHE_DIR / f"{TODAY}___active_sports__.json"


def _disk_cache_age_hours(sport_key: str) -> float:
    """Return how many hours old the cache file is (inf if missing)."""
    path = _disk_cache_path(sport_key)
    if not path.exists():
        return float("inf")
    return (datetime.now().timestamp() - path.stat().st_mtime) / 3600


def _load_disk_cache(sport_key: str) -> Optional[List[dict]]:
    """
    Load odds for sport_key from the stable disk cache if it is fresh enough
    (younger than _ODDS_CACHE_MAX_AGE_HOURS).

    Falls back to the most recent timestamped snapshot (written by
    fetch_closing_odds.py) when the stable file is missing.

    Returns None only when no usable cache exists at all.
    """
    import json

    def _try_load(path: "Path", label: str) -> Optional[List[dict]]:
        try:
            data = json.loads(path.read_text())
            logger.info("Odds [%s]: loaded from %s (%s) — 0 API requests used",
                        sport_key, label, path.name)
            return data
        except Exception as exc:
            logger.warning("Cache read failed for %s (%s): %s", sport_key, path.name, exc)
            return None

    # 1. Stable file — use if fresh enough
    path = _disk_cache_path(sport_key)
    age  = _disk_cache_age_hours(sport_key)
    if path.exists() and age < _ODDS_CACHE_MAX_AGE_HOURS:
        data = _try_load(path, f"disk cache ({age:.1f}h old)")
        if data is not None:
            return data

    # 2. Timestamped snapshots (fetch_closing_odds.py) — pick the freshest
    snapshots = sorted(
        list(_ODDS_DISK_CACHE_DIR.glob(f"*_{sport_key}.json")) +
        list(_ODDS_DISK_CACHE_DIR.glob(f"*_*_{sport_key}.json")),
        key=lambda p: p.stat().st_mtime,
    )
    if snapshots:
        best      = snapshots[-1]
        snap_age  = (datetime.now().timestamp() - best.stat().st_mtime) / 3600
        if snap_age < _ODDS_CACHE_MAX_AGE_HOURS:
            data = _try_load(best, f"snapshot ({snap_age:.1f}h old)")
            if data is not None:
                return data

    # 3. Stale fallback — anything on disk rather than burning API quota.
    #    This happens when quota is exhausted or the API is unreachable.
    all_files = sorted(
        list(_ODDS_DISK_CACHE_DIR.glob(f"*{sport_key}*.json")),
        key=lambda p: p.stat().st_mtime,
    )
    if all_files:
        best     = all_files[-1]
        stale_h  = (datetime.now().timestamp() - best.stat().st_mtime) / 3600
        logger.warning(
            "Odds [%s]: no fresh cache — using stale file %.1fh old (%s). "
            "API fetch will be attempted if quota allows.",
            sport_key, stale_h, best.name,
        )
        data = _try_load(best, f"stale fallback ({stale_h:.0f}h old)")
        if data is not None:
            return data

    return None


def _disk_cache_loaded_at(path: "Path") -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _load_disk_cache_bundle(sport_key: str) -> tuple[Optional[List[dict]], dict[str, Any]]:
    """
    Return the best on-disk cache payload together with metadata about the file
    that was selected.
    """
    import json

    def _read(path: "Path") -> Optional[List[dict]]:
        try:
            return json.loads(path.read_text())
        except Exception as exc:
            logger.warning("Cache read failed for %s (%s): %s", sport_key, path.name, exc)
            return None

    candidates: list[tuple["Path", str, float]] = []
    stable = _disk_cache_path(sport_key)
    stable_age = _disk_cache_age_hours(sport_key)
    if stable.exists() and stable_age < _ODDS_CACHE_MAX_AGE_HOURS:
        candidates.append((stable, "stable", stable_age))

    snapshots = sorted(
        list(_ODDS_DISK_CACHE_DIR.glob(f"*_{sport_key}.json")) +
        list(_ODDS_DISK_CACHE_DIR.glob(f"*_*_{sport_key}.json")),
        key=lambda p: p.stat().st_mtime,
    )
    if snapshots:
        freshest = snapshots[-1]
        snap_age = (datetime.now().timestamp() - freshest.stat().st_mtime) / 3600
        if snap_age < _ODDS_CACHE_MAX_AGE_HOURS:
            candidates.append((freshest, "snapshot", snap_age))

    stale_files = sorted(
        list(_ODDS_DISK_CACHE_DIR.glob(f"*{sport_key}*.json")),
        key=lambda p: p.stat().st_mtime,
    )
    if stale_files:
        stale = stale_files[-1]
        stale_age = (datetime.now().timestamp() - stale.stat().st_mtime) / 3600
        candidates.append((stale, "stale_fallback", stale_age))

    seen: set[Path] = set()
    for path, cache_kind, age_hours in candidates:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        data = _read(path)
        if data is None:
            continue
        logger.info(
            "Odds [%s]: loaded from %s (%s, %s) — 0 API requests used",
            sport_key,
            "stale fallback" if cache_kind == "stale_fallback" else "disk cache",
            path.name,
            f"{age_hours:.1f}h old",
        )
        return data, {
            "path": str(path),
            "cache_kind": cache_kind,
            "cache_loaded_at": _disk_cache_loaded_at(path),
            "cache_age_hours": age_hours,
        }

    return None, {
        "path": "",
        "cache_kind": "",
        "cache_loaded_at": None,
        "cache_age_hours": None,
    }


def _load_recent_empty_disk_cache(sport_key: str) -> Optional[List[dict]]:
    """
    Return a fresh empty cache payload for leagues that recently had no games.

    This avoids re-querying dozens of empty leagues during repeated daytime
    scans while still allowing them to refresh later in the day.
    """
    age_h = _disk_cache_age_hours(sport_key)
    if age_h >= _ODDS_EMPTY_CACHE_REUSE_HOURS:
        return None
    data = _load_disk_cache(sport_key)
    if data == []:
        logger.info(
            "Odds [%s]: reusing fresh empty cache (%.1fh old) — skipping live refetch for an empty slate",
            sport_key,
            age_h,
        )
        return data
    return None


def _save_disk_cache(sport_key: str, games: List[dict]) -> None:
    """
    Persist odds for sport_key to the stable disk cache.
    The file is overwritten only when a fresh API fetch was performed.
    Re-runs of the scan read from this file for up to _ODDS_CACHE_MAX_AGE_HOURS
    without touching the API at all.
    """
    import json
    _ODDS_DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _disk_cache_path(sport_key)
    try:
        path.write_text(json.dumps(games, indent=2))
        logger.info("Odds [%s]: saved %d events to disk cache (%s)", sport_key, len(games), path.name)
    except Exception as exc:
        logger.warning("Disk cache write failed for %s: %s", sport_key, exc)


def _load_active_sports_cache(allow_stale: bool = True) -> set:
    """Load active sports from disk, preferring fresh caches but allowing stale fallback."""
    import json

    candidates = []
    stable = _active_sports_cache_path()
    dated = _active_sports_snapshot_path()
    if stable.exists():
        candidates.append(stable)
    if dated.exists() and dated != stable:
        candidates.append(dated)
    candidates.extend(
        sorted(
            _ODDS_DISK_CACHE_DIR.glob("*__active_sports__.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    )

    seen = set()
    for path in candidates:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        age_h = (datetime.now().timestamp() - path.stat().st_mtime) / 3600
        if not allow_stale and age_h >= _ACTIVE_SPORTS_MAX_AGE_HOURS:
            continue
        try:
            active_sports = set(json.loads(path.read_text()))
        except Exception as exc:
            logger.warning("Active sports disk cache read failed for %s: %s", path.name, exc)
            continue

        freshness = f"{age_h:.1f}h old"
        label = "stale disk cache" if age_h >= _ACTIVE_SPORTS_MAX_AGE_HOURS else "disk cache"
        logger.info(
            "Active sports list: loaded from %s (%s, %d sports) — 0 API requests",
            path.name,
            freshness,
            len(active_sports),
        )
        return active_sports

    return set()


_USAGE_FILE = ROOT / "data" / "api_usage.json"

def _load_api_usage() -> dict:
    """Load persisted API usage counters."""
    return load_odds_api_usage(today=TODAY)

def _api_key_fingerprint() -> str:
    """Return a short fingerprint of the current API key for change detection."""
    return api_key_fingerprint(ODDS_KEY)


def _save_api_usage(remaining: int) -> None:
    """Persist remaining count. Automatically resets if the API key has changed."""
    save_odds_api_usage(api_key=ODDS_KEY, remaining=remaining)
    if ODDS_KEY:
        _update_odds_key_pool_usage(ODDS_KEY, remaining=remaining)


def _load_odds_key_pool() -> List[str]:
    return list(parse_odds_api_keys_from_env().get("keys") or [])


def _load_odds_key_pool_usage() -> dict[str, Any]:
    if _ODDS_KEY_POOL_FILE.exists():
        try:
            payload = json.loads(_ODDS_KEY_POOL_FILE.read_text())
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    return {}


def _save_odds_key_pool_usage(payload: dict[str, Any]) -> None:
    _ODDS_KEY_POOL_FILE.parent.mkdir(parents=True, exist_ok=True)
    _ODDS_KEY_POOL_FILE.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _update_odds_key_pool_usage(api_key: str, *, remaining: Optional[int] = None) -> None:
    fp = api_key_fingerprint(api_key)
    usage = _load_odds_key_pool_usage()
    row = usage.get(fp, {}) if isinstance(usage.get(fp), dict) else {}
    row["fingerprint"] = fp
    row["updated_at"] = datetime.now(timezone.utc).isoformat()
    if remaining is not None:
        row["remaining"] = int(remaining)
    usage[fp] = row
    _save_odds_key_pool_usage(usage)


def _odds_key_quarantine_active(row: dict[str, Any]) -> bool:
    text = str((row or {}).get("auth_quarantined_until") or "").strip()
    if not text:
        return False
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return False
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) < dt.astimezone(timezone.utc)


def _quarantine_odds_api_key_fingerprint(fp: str, *, status_code: int) -> None:
    if not fp:
        return
    usage = _load_odds_key_pool_usage()
    row = usage.get(fp, {}) if isinstance(usage.get(fp), dict) else {}
    now = datetime.now(timezone.utc)
    hours = _ODDS_AUTH_401_QUARANTINE_HOURS if int(status_code) == 401 else _ODDS_AUTH_403_QUARANTINE_HOURS
    row["fingerprint"] = fp
    row["updated_at"] = now.isoformat()
    row["last_auth_failed_at"] = now.isoformat()
    row["last_auth_status_code"] = int(status_code)
    row["auth_fail_count"] = int(row.get("auth_fail_count") or 0) + 1
    row["auth_quarantined_until"] = (now + timedelta(hours=hours)).isoformat()
    row["auth_quarantine_reason"] = f"http_{int(status_code)}"
    usage[fp] = row
    _save_odds_key_pool_usage(usage)


def _pool_metadata_age_hours(updated_at: str) -> Optional[float]:
    text = str(updated_at or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() / 3600)


def _odds_key_pool_inventory() -> dict[str, Any]:
    runtime_parse = parse_odds_api_keys_from_env()
    runtime_keys = list(runtime_parse.get("keys") or [])
    usage = _load_odds_key_pool_usage()
    tracked_rows = {
        str(fp): row
        for fp, row in usage.items()
        if fp != "_meta" and isinstance(row, dict)
    }
    runtime_fingerprints = [api_key_fingerprint(key) for key in runtime_keys]
    runtime_by_fp = {api_key_fingerprint(key): key for key in runtime_keys}
    all_fingerprints = sorted(set(tracked_rows) | set(runtime_fingerprints))
    rows: list[dict[str, Any]] = []
    for fp in all_fingerprints:
        raw = tracked_rows.get(fp, {})
        runtime_key = runtime_by_fp.get(fp)
        remaining = raw.get("remaining")
        known_remaining = int(remaining) if isinstance(remaining, int) else None
        updated_at = str(raw.get("updated_at") or "")
        metadata_age_hours = _pool_metadata_age_hours(updated_at)
        metadata_stale = metadata_age_hours is not None and metadata_age_hours > _ODDS_KEY_METADATA_STALE_HOURS
        rows.append({
            "key": runtime_key or "",
            "fingerprint": fp,
            "tracked": fp in tracked_rows,
            "runtime_available": runtime_key is not None,
            "usable": runtime_key is not None and not _odds_key_quarantine_active(raw),
            "remaining": known_remaining,
            "choice_remaining": known_remaining if known_remaining is not None else 500,
            "updated_at": updated_at,
            "metadata_age_hours": metadata_age_hours,
            "metadata_stale": metadata_stale,
            "env_index": runtime_fingerprints.index(fp) if fp in runtime_fingerprints else 9999,
            "is_low": known_remaining is not None and known_remaining < _ODDS_KEY_LOW_REMAINING_THRESHOLD,
            "auth_quarantined_until": str(raw.get("auth_quarantined_until") or ""),
            "auth_quarantine_reason": str(raw.get("auth_quarantine_reason") or ""),
            "auth_quarantined": _odds_key_quarantine_active(raw),
        })
    return {
        "canonical_pool_path": str(_ODDS_KEY_POOL_FILE.resolve()),
        "tracked_fingerprints": sorted(tracked_rows.keys()),
        "runtime_loaded_fingerprints": runtime_fingerprints,
        "runtime_parse_excluded": list(runtime_parse.get("excluded") or []),
        "rows": rows,
    }


def _log_odds_key_selector_diagnostics(diagnostics: dict[str, Any]) -> None:
    logger.info("Odds API key manager: canonical pool path=%s", diagnostics.get("canonical_pool_path", ""))
    logger.info(
        "Odds API key manager: tracked fingerprints=%s | runtime-loaded fingerprints=%s | usable fingerprints=%s",
        diagnostics.get("tracked_fingerprints", []),
        diagnostics.get("runtime_loaded_fingerprints", []),
        diagnostics.get("usable_fingerprints", []),
    )
    for item in diagnostics.get("runtime_parse_excluded", []):
        logger.info(
            "Odds API key manager: excluded runtime env key …%s (%s from %s)",
            item.get("fingerprint", ""),
            item.get("reason", ""),
            item.get("source", ""),
        )
    for fp in diagnostics.get("tracked_unavailable_fingerprints", []):
        logger.info(
            "Odds API key manager: tracked key exists but raw key not available at runtime (…%s)",
            fp,
        )
    for item in diagnostics.get("excluded", []):
        logger.info(
            "Odds API key manager: excluded key …%s (%s)",
            item.get("fingerprint", ""),
            item.get("reason", ""),
        )


def _odds_key_pool_rows() -> list[dict[str, Any]]:
    return list(_odds_key_pool_inventory().get("rows", []))


def _record_odds_key_selection(
    selected_row: dict[str, Any],
    *,
    reason: str,
    selector: str,
    min_remaining: int = 0,
    diagnostics: Optional[dict[str, Any]] = None,
) -> None:
    usage = _load_odds_key_pool_usage()
    usage["_meta"] = {
        "last_selected_fingerprint": str(selected_row.get("fingerprint") or ""),
        "last_selected_at": datetime.now(timezone.utc).isoformat(),
        "last_selected_reason": reason,
        "last_selected_selector": selector,
        "last_selected_remaining": selected_row.get("remaining"),
        "last_selected_min_remaining": int(min_remaining),
        "low_remaining_threshold": _ODDS_KEY_LOW_REMAINING_THRESHOLD,
        "metadata_stale_threshold_hours": _ODDS_KEY_METADATA_STALE_HOURS,
    }
    if diagnostics:
        usage["_meta"]["canonical_pool_path"] = str(diagnostics.get("canonical_pool_path") or "")
        usage["_meta"]["tracked_fingerprints"] = list(diagnostics.get("tracked_fingerprints") or [])
        usage["_meta"]["runtime_loaded_fingerprints"] = list(diagnostics.get("runtime_loaded_fingerprints") or [])
        usage["_meta"]["usable_fingerprints"] = list(diagnostics.get("usable_fingerprints") or [])
        usage["_meta"]["excluded_fingerprints"] = list(diagnostics.get("excluded_fingerprints") or [])
        usage["_meta"]["excluded_details"] = list(diagnostics.get("excluded") or [])
        usage["_meta"]["runtime_parse_excluded"] = list(diagnostics.get("runtime_parse_excluded") or [])
        usage["_meta"]["selected_below_low_threshold"] = bool(selected_row.get("is_low"))
        usage["_meta"]["healthier_usable_key_existed"] = bool(diagnostics.get("healthier_usable_key_existed"))
    _save_odds_key_pool_usage(usage)


def _choose_odds_api_key(
    *,
    exclude_fingerprint: str | None = None,
    min_remaining: int = 0,
) -> tuple[dict[str, Any] | None, str, dict[str, Any]]:
    inventory = _odds_key_pool_inventory()
    rows = list(inventory.get("rows", []))
    excluded: list[dict[str, Any]] = []
    tracked_unavailable = [row["fingerprint"] for row in rows if row["tracked"] and not row["runtime_available"]]
    usable_rows = [row for row in rows if row["usable"]]
    for row in rows:
        if bool(row.get("auth_quarantined")) and bool(row.get("runtime_available")):
            excluded.append({
                "fingerprint": row["fingerprint"],
                "reason": row.get("auth_quarantine_reason") or "auth_quarantined",
            })
    if exclude_fingerprint:
        for row in usable_rows:
            if row["fingerprint"] == exclude_fingerprint:
                excluded.append({"fingerprint": row["fingerprint"], "reason": "excluded_current_key"})
        usable_rows = [row for row in usable_rows if row["fingerprint"] != exclude_fingerprint]
    if _odds_api_failed_fingerprints:
        for row in usable_rows:
            if row["fingerprint"] in _odds_api_failed_fingerprints:
                excluded.append({"fingerprint": row["fingerprint"], "reason": "auth_failed_this_run"})
        usable_rows = [row for row in usable_rows if row["fingerprint"] not in _odds_api_failed_fingerprints]
    for fp in tracked_unavailable:
        excluded.append({"fingerprint": fp, "reason": "raw_key_missing"})

    if not usable_rows:
        diagnostics = {
            **inventory,
            "usable_fingerprints": [],
            "excluded_fingerprints": [item["fingerprint"] for item in excluded],
            "tracked_unavailable_fingerprints": tracked_unavailable,
            "selected_fingerprint": "",
            "selected_remaining": None,
            "selected_reason": "no runtime-available pooled keys",
            "selected_below_low_threshold": False,
            "healthier_usable_key_existed": False,
            "excluded": excluded,
        }
        return None, "no runtime-available pooled keys", diagnostics

    eligible: list[dict[str, Any]] = []
    for row in usable_rows:
        if int(row["choice_remaining"]) < int(min_remaining):
            excluded.append({"fingerprint": row["fingerprint"], "reason": f"below_min_remaining:{min_remaining}"})
            continue
        eligible.append(row)

    if not eligible:
        diagnostics = {
            **inventory,
            "usable_fingerprints": [row["fingerprint"] for row in usable_rows],
            "excluded_fingerprints": [item["fingerprint"] for item in excluded],
            "tracked_unavailable_fingerprints": tracked_unavailable,
            "selected_fingerprint": "",
            "selected_remaining": None,
            "selected_reason": f"no runtime key met minimum remaining threshold ({min_remaining})",
            "selected_below_low_threshold": False,
            "healthier_usable_key_existed": False,
            "excluded": excluded,
        }
        return None, f"no runtime key met minimum remaining threshold ({min_remaining})", diagnostics

    healthy = [row for row in eligible if not row["is_low"]]
    healthier_usable_key_existed = bool(healthy) and any(row["is_low"] for row in eligible)
    selection_pool = healthy or eligible
    for row in eligible:
        if row["is_low"] and row not in selection_pool:
            excluded.append({"fingerprint": row["fingerprint"], "reason": "low_quota_healthier_key_exists"})

    best = max(selection_pool, key=lambda row: (int(row["choice_remaining"]), -int(row["env_index"])))
    higher_tracked_excluded = [
        item for item in excluded
        if any(row["fingerprint"] == item["fingerprint"] and int(row["choice_remaining"]) > int(best["choice_remaining"]) for row in rows)
    ]

    if best["remaining"] is None:
        reason = "selected highest remaining runtime-available key with untracked quota metadata"
    else:
        reason = "selected highest remaining runtime-available key"
    if best["metadata_stale"]:
        reason = f"{reason}; metadata is stale ({best['metadata_age_hours']:.1f}h old)"
    if higher_tracked_excluded:
        reason = f"{reason}; higher tracked keys were excluded with explicit reasons"
    if min_remaining > 0:
        reason = f"{reason} (minimum remaining required: {min_remaining})"

    diagnostics = {
        **inventory,
        "usable_fingerprints": [row["fingerprint"] for row in eligible],
        "excluded_fingerprints": [item["fingerprint"] for item in excluded],
        "tracked_unavailable_fingerprints": tracked_unavailable,
        "selected_fingerprint": best["fingerprint"],
        "selected_remaining": best["remaining"],
        "selected_reason": reason,
        "selected_below_low_threshold": bool(best["is_low"]),
        "healthier_usable_key_existed": healthier_usable_key_existed and bool(best["is_low"]),
        "excluded": excluded,
    }
    return best, reason, diagnostics


def _select_odds_api_key() -> str:
    best, reason, diagnostics = _choose_odds_api_key()
    _log_odds_key_selector_diagnostics(diagnostics)
    if best is None:
        return ""
    _record_odds_key_selection(best, reason=reason, selector="scan_start", diagnostics=diagnostics)
    logger.info(
        "Odds API key manager: selected key …%s from pool of %d tracked / %d runtime (tracked remaining=%s, reason=%s, below_low_threshold=%s, healthier_usable_key_existed=%s)",
        best["fingerprint"],
        len(diagnostics.get("tracked_fingerprints", [])),
        len(diagnostics.get("runtime_loaded_fingerprints", [])),
        best["remaining"] if best["remaining"] is not None else "unknown",
        reason,
        bool(best["is_low"]),
        bool(diagnostics.get("healthier_usable_key_existed")),
    )
    return str(best["key"])


def _pool_mode_enabled() -> bool:
    return len(_load_odds_key_pool()) > 1


def _tracked_pool_remaining(api_key: str) -> Optional[int]:
    fp = api_key_fingerprint(api_key)
    row = _load_odds_key_pool_usage().get(fp, {})
    if isinstance(row, dict):
        remaining = row.get("remaining")
        if isinstance(remaining, int):
            return remaining
    return None


def _rotate_odds_api_key(*, min_remaining: int = _ODDS_BUDGET_STOP + 1) -> bool:
    global ODDS_KEY
    if len(_load_odds_key_pool()) <= 1:
        return False

    current_fp = api_key_fingerprint(ODDS_KEY)
    best, reason, diagnostics = _choose_odds_api_key(exclude_fingerprint=current_fp, min_remaining=min_remaining)
    _log_odds_key_selector_diagnostics(diagnostics)
    if best is None:
        return False

    ODDS_KEY = str(best["key"])
    os.environ["ODDS_API_KEY"] = str(best["key"])
    _record_odds_key_selection(best, reason=reason, selector="rotation", min_remaining=min_remaining, diagnostics=diagnostics)
    logger.info(
        "Odds API key manager: rotated to key …%s (tracked remaining=%s, reason=%s, below_low_threshold=%s, healthier_usable_key_existed=%s)",
        best["fingerprint"],
        best["remaining"] if best["remaining"] is not None else "unknown",
        reason,
        bool(best["is_low"]),
        bool(diagnostics.get("healthier_usable_key_existed")),
    )
    return True


def _maybe_rotate_low_active_odds_key() -> bool:
    """Proactively rotate away from a low active key when a healthier key exists."""
    global ODDS_KEY
    if not _pool_mode_enabled():
        return False
    if not isinstance(_odds_remaining, int):
        return False
    if _odds_remaining > _ODDS_KEY_LOW_REMAINING_THRESHOLD:
        return False

    current_fp = api_key_fingerprint(ODDS_KEY)
    best, reason, diagnostics = _choose_odds_api_key(
        exclude_fingerprint=current_fp,
        min_remaining=max(_ODDS_KEY_LOW_REMAINING_THRESHOLD + 1, _odds_remaining + 1),
    )
    if best is None:
        return False

    logger.info(
        "Odds API key manager: active key is low (%s remaining) — proactively rotating to healthier runtime key.",
        _odds_remaining,
    )
    _log_odds_key_selector_diagnostics(diagnostics)
    ODDS_KEY = str(best["key"])
    os.environ["ODDS_API_KEY"] = str(best["key"])
    _record_odds_key_selection(
        best,
        reason=f"proactive_low_quota_rotation: {reason}",
        selector="low_quota_rotation",
        min_remaining=max(_ODDS_KEY_LOW_REMAINING_THRESHOLD + 1, _odds_remaining + 1),
        diagnostics=diagnostics,
    )
    logger.info(
        "Odds API key manager: rotated to key …%s (tracked remaining=%s, reason=%s)",
        best["fingerprint"],
        best["remaining"] if best["remaining"] is not None else "unknown",
        reason,
    )
    return True


def _check_budget(priority: str = "high") -> bool:
    """Return True if it's safe to make another Odds API request.
    Automatically resets the counter when a new API key is detected."""
    global _odds_remaining, _odds_api_auth_failed
    if _odds_api_auth_failed:
        logger.error("Odds API auth previously failed in this run — skipping further live odds fetches.")
        return False
    usage = _load_api_usage()
    fp = _api_key_fingerprint()

    # Key changed since last run → treat quota as full until we hear otherwise
    if usage.get("key_fingerprint") != fp:
        logger.info("New API key detected (…%s) — resetting quota guard.", fp)
        _odds_remaining = 9999
        # Persist the fingerprint immediately so a bad key doesn't trigger
        # a fake "new key" reset on every subsequent request.
        save_odds_api_usage(api_key=ODDS_KEY, remaining=usage.get("odds_remaining"))
        return True

    budget = get_odds_budget_status(ODDS_KEY)
    _odds_remaining = budget.get("remaining", usage.get("odds_remaining", 9999))
    if _odds_remaining <= _ODDS_BUDGET_STOP:
        if _pool_mode_enabled() and _rotate_odds_api_key(min_remaining=1):
            return True
        logger.error(
            "Odds API key exhausted (%d remaining ≤ %d). "
            "Stopping live odds fetches until a usable key is available.",
            _odds_remaining, _ODDS_BUDGET_STOP,
        )
        return False
    if _odds_remaining <= _ODDS_BUDGET_WARN:
        logger.warning(
            "Odds API running low: %d requests remaining (warn threshold: %d).",
            _odds_remaining, _ODDS_BUDGET_WARN,
        )
        if _maybe_rotate_low_active_odds_key():
            return True
    return True


def _append_scan_note(note: dict[str, Any]) -> None:
    global _scan_runtime_notes
    entry = dict(note or {})
    if not entry:
        return
    key = json.dumps(entry, sort_keys=True, default=str)
    seen = {
        json.dumps(existing, sort_keys=True, default=str)
        for existing in (_scan_runtime_notes or [])
    }
    if key not in seen:
        _scan_runtime_notes.append(entry)


def _handle_odds_api_auth_failure(*, sport_key: str, status_code: int) -> bool:
    global ODDS_KEY, _odds_api_auth_failed, _odds_remaining
    failed_fp = api_key_fingerprint(ODDS_KEY)
    if failed_fp:
        _odds_api_failed_fingerprints.add(failed_fp)
        _quarantine_odds_api_key_fingerprint(failed_fp, status_code=int(status_code))

    logger.error(
        "Odds API authorization failed (%s) for %s on key …%s.",
        status_code,
        sport_key,
        failed_fp,
    )

    if _pool_mode_enabled() and _rotate_odds_api_key(min_remaining=1):
        _odds_api_auth_failed = False
        _odds_remaining = 9999
        _append_scan_note({
            "type": "odds_api_auth_recovered",
            "sport": sport_key,
            "status_code": int(status_code),
            "failed_fingerprint": failed_fp,
            "active_fingerprint": api_key_fingerprint(ODDS_KEY),
            "reason": "Odds API authorization failed on one runtime key, but the scan rotated to another key and kept live odds enabled.",
        })
        logger.warning(
            "Odds API auth failure recovered by rotating away from key …%s to key …%s.",
            failed_fp,
            api_key_fingerprint(ODDS_KEY),
        )
        return True

    _odds_api_auth_failed = True
    _append_scan_note({
        "type": "odds_api_degraded_mode",
        "sport": sport_key,
        "status_code": int(status_code),
        "failed_fingerprint": failed_fp,
        "reason": "Odds API authorization failed and no healthy fallback runtime key was available. The rest of this scan will skip further live odds fetches.",
    })
    logger.error(
        "Odds API authorization failed (%s) for %s — disabling further live odds fetches this run.",
        status_code,
        sport_key,
    )
    return False


def _disable_live_odds_for_run(reason: str, *, note_type: str = "odds_api_degraded_mode") -> None:
    global _odds_api_auth_failed
    _odds_api_auth_failed = True
    _append_scan_note({
        "type": note_type,
        "reason": reason,
    })
    logger.error("%s", reason)


_TIMEZONE = "Europe/Vienna"   # CEST (UTC+2) — used for today/tomorrow labels and cutoff


def _kick_off_label(commence_time: str) -> str:
    """Return 'Today HH:MM' or 'Tomorrow HH:MM' (local time) from a UTC ISO string."""
    try:
        from datetime import timedelta, timezone as tz
        import zoneinfo
        utc_dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
        # Use Vienna/Austria local time (CEST = UTC+2)
        try:
            local = zoneinfo.ZoneInfo(_TIMEZONE)
            local_dt = utc_dt.astimezone(local)
        except Exception:
            local_dt = utc_dt.astimezone()  # fallback to system tz
        today_local = datetime.now(local_dt.tzinfo).date()
        if local_dt.date() == today_local:
            return f"Today {local_dt.strftime('%H:%M')}"
        elif local_dt.date() == today_local + timedelta(days=1):
            return f"Tomorrow {local_dt.strftime('%H:%M')}"
        else:
            return local_dt.strftime("%a %d %b %H:%M")
    except Exception:
        return ""


def _pretty_soccer_league_name(sport_key: str) -> str:
    return soccer_pretty_label(sport_key)


def _recent_cache_event_state(sport_key: str, max_age_hours: float = 12.0) -> str:
    """
    Return a lightweight view of recent disk-cache usefulness for a league.

    Values:
      - ``nonempty``: fresh cache exists and contains at least one event
      - ``empty``: fresh cache exists but contains zero events
      - ``stale_or_missing``: no fresh cache signal
    """
    import json

    age_h = _disk_cache_age_hours(sport_key)
    if age_h >= max_age_hours:
        return "stale_or_missing"

    path = _disk_cache_path(sport_key)
    if not path.exists():
        return "stale_or_missing"
    try:
        data = json.loads(path.read_text())
    except Exception:
        return "stale_or_missing"
    if isinstance(data, list) and data:
        return "nonempty"
    if data == []:
        return "empty"
    return "stale_or_missing"


def _select_soccer_live_scope(active_soccer_keys: List[str]) -> tuple[List[str], List[str]]:
    """
    Prefer production leagues and review-only leagues with evidence of activity.

    This keeps the scan fast on healthy days without completely losing review
    coverage. When quota is abundant we still include the full review-only set.
    """
    if not active_soccer_keys:
        return [], []

    if os.getenv("SCAN_FULL_SOCCER_SCOPE", "").strip().lower() in {"1", "true", "yes", "on"}:
        return list(active_soccer_keys), []

    budget = get_odds_budget_status(ODDS_KEY, monthly_limit=500, reserve=_ODDS_BUDGET_STOP)
    remaining = budget.get("remaining")
    daily_allowance = budget.get("daily_allowance")
    used_today = budget.get("used_today", 0)
    healthy_quota = (
        remaining in (None, 9999)
        or (isinstance(remaining, int) and remaining > 250)
    ) and (
        daily_allowance is None
        or used_today < max(1, int(daily_allowance * 0.5))
    )

    production_keys: List[str] = []
    review_recent_nonempty: List[str] = []
    review_deferred: List[str] = []
    review_recent_empty: List[str] = []
    review_other: List[str] = []

    for sport_key in active_soccer_keys:
        profile = get_capability_profile(sport="soccer", sport_key=sport_key)
        if not profile.review_only:
            production_keys.append(sport_key)
            continue
        cache_state = _recent_cache_event_state(sport_key)
        if cache_state == "nonempty":
            review_recent_nonempty.append(sport_key)
        elif cache_state == "empty":
            review_recent_empty.append(sport_key)
        else:
            review_other.append(sport_key)

    if healthy_quota:
        # Even on healthy days, there is little value in re-looping over
        # review-only leagues that were confirmed empty very recently.
        selected = production_keys + review_recent_nonempty + review_other
        review_deferred = review_recent_empty
    else:
        selected = production_keys + review_recent_nonempty
        review_deferred = review_recent_empty + review_other

    return selected, review_deferred


def _bucket_no_candidate_reason(game: dict[str, Any]) -> str:
    sport = str(game.get("sport", "") or "").strip().lower()
    if sport == "soccer":
        sport_key = str(game.get("league_key") or game.get("_league_key") or "").strip()
        if sport_key:
            profile = get_capability_profile(sport="soccer", sport_key=sport_key)
            if not bool(profile.model_backed):
                return "league_not_model_covered"
    if not bool(game.get("model_available", False)):
        return "insufficient_history_or_model"
    if bool(game.get("abstain", False)):
        return "line_movement_or_abstain"
    outcomes = game.get("outcomes") if isinstance(game.get("outcomes"), list) else []
    if any(bool((outcome or {}).get("has_value", False)) for outcome in outcomes if isinstance(outcome, dict)):
        return "candidate_not_retained"
    return "no_edge"


def _future_windowed_games(games: List[dict]) -> List[dict]:
    """
    Recompute scan windows against the current clock and drop already-started games.

    Disk odds are cached for quota safety, so stale window labels cannot be trusted
    later in the day.
    """
    from datetime import timedelta

    now = datetime.now(timezone.utc)
    today_cutoff = now + timedelta(hours=18)
    overnight_cutoff = now + timedelta(hours=25)
    tomorrow_cutoff = now + timedelta(hours=42)
    day_after_cutoff = now + timedelta(hours=66)

    filtered: List[dict] = []
    for game in games:
        try:
            commence = datetime.fromisoformat(str(game["commence_time"]).replace("Z", "+00:00"))
        except Exception:
            continue

        if not now < commence <= day_after_cutoff:
            continue

        game = dict(game)
        if commence <= today_cutoff:
            game["_window"] = "today"
        elif commence <= overnight_cutoff:
            game["_window"] = "overnight"
        elif commence <= tomorrow_cutoff:
            game["_window"] = "tomorrow"
        else:
            game["_window"] = "day_after"
        filtered.append(game)

    return filtered


_ODDS_STALE_THRESHOLD_HOURS = 24.0


def _parse_utc_dt(value: object) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _isoformat_utc(value: object) -> Optional[str]:
    parsed = _parse_utc_dt(value)
    return parsed.isoformat() if parsed is not None else None


def _game_last_update_dt(
    game: dict,
    *,
    bookmaker_name: str = "",
    market_key: str = "",
) -> Optional[datetime]:
    updates: list[datetime] = []
    target_bookmaker = str(bookmaker_name or "").strip().lower()
    target_market = str(market_key or "").strip().lower()

    for bk in game.get("bookmakers", []):
        bk_title = str(bk.get("title", "") or "").strip().lower()
        if target_bookmaker and bk_title != target_bookmaker:
            continue
        bk_update = _parse_utc_dt(bk.get("last_update"))
        if bk_update is not None:
            updates.append(bk_update)
        for market in bk.get("markets", []):
            market_name = str(market.get("key", "") or "").strip().lower()
            if target_market and market_name != target_market:
                continue
            market_update = _parse_utc_dt(market.get("last_update"))
            if market_update is not None:
                updates.append(market_update)

    if updates:
        return max(updates)
    return None


def _odds_age_hours_from_dt(last_update: object, *, now: Optional[datetime] = None) -> Optional[float]:
    parsed = _parse_utc_dt(last_update)
    if parsed is None:
        return None
    current = now or datetime.now(timezone.utc)
    return max(0.0, (current - parsed).total_seconds() / 3600.0)


def _odds_source_detail(
    *,
    source_status: str,
    force_fresh: bool,
    fallback_used: bool,
    cache_used: bool,
    bookmaker_last_update: object,
    computed_age_hours: Optional[float],
) -> str:
    status = str(source_status or "").strip().lower()
    has_bookmaker_timestamp = _parse_utc_dt(bookmaker_last_update) is not None

    if status == "live_api":
        if has_bookmaker_timestamp and computed_age_hours is not None and computed_age_hours <= _ODDS_STALE_THRESHOLD_HOURS:
            return "fresh_odds_used"
        if has_bookmaker_timestamp:
            return "fetched_now_but_bookmaker_price_timestamp_old"
        return "fetched_now_but_bookmaker_timestamp_missing"
    if status == "stale_fallback":
        if force_fresh:
            return "fallback_odds_used_because_force_fresh_fetch_could_not_complete"
        return "loaded_from_stale_cache_fallback"
    if fallback_used:
        return "fallback_odds_used_because_fresh_odds_missing"
    if cache_used:
        if has_bookmaker_timestamp:
            return "cached_odds_used_with_bookmaker_timestamp"
        return "cached_odds_used_but_bookmaker_timestamp_missing"
    if has_bookmaker_timestamp and computed_age_hours is not None and computed_age_hours <= _ODDS_STALE_THRESHOLD_HOURS:
        return "fresh_odds_used"
    return "freshness_unverified"


def _tag_odds_payload(
    games: List[dict],
    *,
    source_status: str,
    fetched_at: Optional[datetime] = None,
    snapshot_age_hours: Optional[float] = None,
    cache_loaded_at: Optional[datetime] = None,
    cache_age_hours: Optional[float] = None,
    force_fresh: Optional[bool] = None,
    cache_used: bool = False,
    fallback_used: bool = False,
    source_reason: str = "",
) -> List[dict]:
    tagged: List[dict] = []
    fetch_dt = (fetched_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    fetched_iso = fetch_dt.isoformat()
    cache_loaded_iso = cache_loaded_at.astimezone(timezone.utc).isoformat() if cache_loaded_at is not None else None
    for game in games:
        row = dict(game)
        bookmaker_last_update = _game_last_update_dt(row)
        computed_age_hours = _odds_age_hours_from_dt(bookmaker_last_update, now=fetch_dt)
        age_basis = "bookmaker_last_update"
        if computed_age_hours is None and snapshot_age_hours is not None:
            computed_age_hours = float(snapshot_age_hours)
            age_basis = "cache_loaded_at" if cache_used else "fetch_time"
        row["_odds_source_status"] = source_status
        row["_odds_fetched_at"] = fetched_iso
        row["_odds_bookmaker_last_update"] = bookmaker_last_update.isoformat() if bookmaker_last_update is not None else None
        row["_odds_cache_loaded_at"] = cache_loaded_iso
        row["_odds_cache_age_hours"] = round(float(cache_age_hours), 3) if cache_age_hours is not None else None
        row["_odds_age_basis"] = age_basis
        row["_odds_force_fresh_requested"] = bool(_FORCE_FRESH_ODDS if force_fresh is None else force_fresh)
        row["_odds_cache_used"] = bool(cache_used)
        row["_odds_fallback_used"] = bool(fallback_used)
        row["_odds_source_reason"] = source_reason
        row["_odds_source_detail"] = _odds_source_detail(
            source_status=source_status,
            force_fresh=bool(_FORCE_FRESH_ODDS if force_fresh is None else force_fresh),
            fallback_used=bool(fallback_used),
            cache_used=bool(cache_used),
            bookmaker_last_update=bookmaker_last_update,
            computed_age_hours=computed_age_hours,
        )
        if computed_age_hours is not None:
            row["_odds_snapshot_age_hours"] = round(float(computed_age_hours), 3)
        tagged.append(row)
    return tagged


def _feature_freshness_context(sport: str) -> dict:
    meta = _feature_cache_meta.get(str(sport or ""), {})
    return {
        "standings_fetched_at": meta.get("fetched_at"),
        "standings_snapshot_age_hours": meta.get("age_hours"),
        "standings_source_status": meta.get("source_status", "unknown"),
    }


_ALL_MARKETS = "h2h,totals,spreads"   # fetch all in one request to save quota


def _fetch_odds_raw(sport_key: str) -> List[dict]:
    """
    Internal: make ONE Odds API request for ALL markets (h2h, totals, spreads),
    apply the time-window filter, store in cache under sport_key, and return
    the full filtered list.  Each game dict contains bookmaker entries for all
    available markets.

    This is called at most once per sport per session (60-min TTL).
    """
    global _odds_remaining, _odds_api_auth_failed
    from datetime import timedelta

    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    # 3-day lookahead so one API call covers today + tomorrow + day-after.
    # The result is cached for up to _ODDS_CACHE_MAX_AGE_HOURS (20h) so
    # multiple scan runs cost 0 extra requests.
    _now = datetime.now(timezone.utc)
    _cutoff = _now + timedelta(hours=72)

    params = {
        "regions":           "eu,uk",
        "markets":           _ALL_MARKETS,
        "oddsFormat":        "decimal",
        "commenceTimeFrom":  _now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commenceTimeTo":    _cutoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    def _live_request() -> requests.Response | None:
        global _last_api_call_at, _api_calls_today
        if not _api_throttle():
            logger.warning("API_THROTTLE: daily cap hit — returning empty list")
            return None
        _last_api_call_at = datetime.now(timezone.utc)
        _api_calls_today += 1
        return requests.get(
            url,
            params={**params, "apiKey": ODDS_KEY},
            timeout=15,
        )

    r = _live_request()
    if r is None:
        return []
    try:
        r.raise_for_status()
    except requests.HTTPError:
        if r.status_code in (401, 403) and _handle_odds_api_auth_failure(sport_key=sport_key, status_code=int(r.status_code)):
            r = _live_request()
            if r is None:
                return []
            r.raise_for_status()
        else:
            raise

    remaining_str = r.headers.get("x-requests-remaining", "?")
    try:
        _odds_remaining = int(remaining_str)
        _save_api_usage(_odds_remaining)
    except (ValueError, TypeError):
        pass

    used_str = r.headers.get("x-requests-used", "?")
    all_games = r.json()
    logger.info(
        "Odds API [%s] ALL-MARKETS: %d events | used=%s remaining=%s (saved ~2 requests vs per-market)",
        sport_key, len(all_games), used_str, remaining_str,
    )

    if _odds_remaining <= _ODDS_BUDGET_WARN:
        logger.warning(
            "⚠️  Only %d Odds API requests left this month!", _odds_remaining
        )

    filtered = _tag_odds_payload(
        _future_windowed_games(all_games),
        source_status="live_api",
        fetched_at=datetime.now(timezone.utc),
        snapshot_age_hours=0.0,
    )
    today_games = [g for g in filtered if g.get("_window") == "today"]
    overnight_games = [g for g in filtered if g.get("_window") == "overnight"]
    tomorrow_games = [g for g in filtered if g.get("_window") == "tomorrow"]
    day_after_games = [g for g in filtered if g.get("_window") == "day_after"]
    n_skipped = len(all_games) - len(filtered)
    logger.info(
        "Odds [%s]: today=%d  overnight=%d  tomorrow=%d  day_after=%d  skipped=%d",
        sport_key, len(today_games), len(overnight_games),
        len(tomorrow_games), len(day_after_games), n_skipped,
    )

    # Save to disk — so subsequent scan runs today skip the API entirely
    _save_disk_cache(sport_key, filtered)

    # Record opening odds for any events not yet seen (uses all_games before window filter)
    try:
        from src.data import opening_odds_store as _oos
        _oos_store = _oos.load()
        _oos.purge_old(_oos_store)
        added = _oos.record_games(_oos_store, all_games)
        if added:
            _oos.save(_oos_store)
            logger.info("Opening odds store: recorded %d new events for %s", added, sport_key)
    except Exception as _oos_exc:
        logger.debug("Opening odds store update skipped (%s): %s", sport_key, _oos_exc)

    # Also warm the in-memory cache
    _odds_cache[sport_key] = (datetime.now(), filtered)
    return filtered


def fetch_odds(sport_key: str, markets: str = "h2h", priority: str = "high") -> List[dict]:
    """
    Fetch upcoming odds from The Odds API with three cache layers:

      1. In-memory cache (60-min TTL) — zero cost within the same process.
      2. Disk cache (daily) — today's data saved to data/cache/odds/YYYY-MM-DD_<sport>.json.
         Any scan run later the same day reads from disk, costing 0 API requests.
      3. Live Odds API fetch — only when both caches miss. Fetches h2h+totals+spreads
         in ONE request per sport, then writes to disk + memory so future calls are free.

    Budget guard halts live fetches when remaining requests ≤ _ODDS_BUDGET_STOP.
    """
    global _odds_remaining

    if not ODDS_KEY:
        logger.warning("ODDS_API_KEY not set; skipping odds fetch.")
        return []

    def _filter_market(games: List[dict]) -> List[dict]:
        """Filter game list to those offering the requested market."""
        if any("commence_time" in g for g in games):
            games = _future_windowed_games(games)
        requested = markets.split(",")[0].strip()
        if requested == "h2h":
            return games
        return [
            g for g in games
            if any(
                mkt["key"] == requested
                for bk in g.get("bookmakers", [])
                for mkt in bk.get("markets", [])
            )
        ]

    # ── Layer 1: in-memory cache ──────────────────────────────────────
    # Even in force-fresh mode, reuse a payload that was already fetched live
    # earlier in this same scan process. "Force fresh" should bypass stale
    # disk/history, not re-burn the API 2-3 times for h2h/totals/spreads on
    # the exact same sport within the same run.
    if sport_key in _odds_cache:
        fetched_at, cached_data = _odds_cache[sport_key]
        age_minutes = (datetime.now() - fetched_at).total_seconds() / 60
        if age_minutes < _ODDS_CACHE_TTL_MINUTES:
            logger.info(
                "Odds [%s/%s]: in-memory cache hit (%.0f min old) — 0 API requests%s",
                sport_key, markets, age_minutes,
                " [fresh payload reused within current scan]" if _FORCE_FRESH_ODDS else "",
            )
            tagged_data = _tag_odds_payload(
                cached_data,
                source_status="fresh_run_cache" if _FORCE_FRESH_ODDS else "in_memory_cache",
                fetched_at=fetched_at,
                snapshot_age_hours=age_minutes / 60.0,
                source_reason="force_fresh_reused_current_run_fetch" if _FORCE_FRESH_ODDS else "",
            )
            return _filter_market(tagged_data)

    # ── Layer 2: disk cache (age-based, not date-based) ──────────────
    # If the file is younger than _ODDS_CACHE_MAX_AGE_HOURS we use it
    # regardless of whether it was written today or yesterday.
    # Only fetch from the API when the file is missing or too old.
    if not _FORCE_FRESH_ODDS:
        age_h = _disk_cache_age_hours(sport_key)
        if age_h < _ODDS_CACHE_MAX_AGE_HOURS:
            disk_data, disk_meta = _load_disk_cache_bundle(sport_key)
            if disk_data is not None:
                _odds_cache[sport_key] = (datetime.now(), disk_data)
                tagged_data = _tag_odds_payload(
                    disk_data,
                    source_status="disk_cache",
                    cache_loaded_at=disk_meta.get("cache_loaded_at"),
                    cache_age_hours=disk_meta.get("cache_age_hours"),
                    cache_used=True,
                    snapshot_age_hours=age_h,
                    source_reason="loaded_from_recent_disk_cache",
                )
                return _filter_market(tagged_data)

        recent_empty = _load_recent_empty_disk_cache(sport_key)
        if recent_empty is not None:
            _odds_cache[sport_key] = (datetime.now(), recent_empty)
            tagged_data = _tag_odds_payload(
                recent_empty,
                source_status="disk_empty_cache",
                snapshot_age_hours=_disk_cache_age_hours(sport_key),
            )
            return _filter_market(tagged_data)
    else:
        logger.info("Odds [%s/%s]: force-fresh mode bypassing disk odds cache", sport_key, markets)

    if _OFFLINE_ODDS_ONLY:
        disk_data, disk_meta = _load_disk_cache_bundle(sport_key)
        if disk_data is not None:
            _odds_cache[sport_key] = (datetime.now(), disk_data)
            logger.info(
                "Odds [%s/%s]: offline mode enabled — using saved disk cache only",
                sport_key,
                markets,
            )
            tagged_data = _tag_odds_payload(
                disk_data,
                source_status="offline_disk_cache",
                cache_loaded_at=disk_meta.get("cache_loaded_at"),
                cache_age_hours=disk_meta.get("cache_age_hours"),
                cache_used=True,
                snapshot_age_hours=_disk_cache_age_hours(sport_key),
                source_reason="offline_mode_saved_disk_cache",
            )
            return _filter_market(tagged_data)
        logger.warning(
            "Odds [%s/%s]: offline mode enabled and no saved cache exists — skipping live API fetch",
            sport_key,
            markets,
        )
        return []

    # ── Layer 3: live API fetch (only when cache is stale/missing) ────
    if not _check_budget(priority=priority):
        # Budget gone — use whatever stale data exists rather than returning []
        disk_data, disk_meta = _load_disk_cache_bundle(sport_key)
        if disk_data is not None:
            _odds_cache[sport_key] = (datetime.now(), disk_data)
            tagged_data = _tag_odds_payload(
                disk_data,
                source_status="stale_fallback",
                cache_loaded_at=disk_meta.get("cache_loaded_at"),
                cache_age_hours=disk_meta.get("cache_age_hours"),
                cache_used=True,
                fallback_used=True,
                snapshot_age_hours=_disk_cache_age_hours(sport_key),
                source_reason=(
                    "force_fresh_requested_but_live_fetch_blocked_by_budget_or_auth"
                    if _FORCE_FRESH_ODDS
                    else "live_fetch_blocked_by_budget_or_auth"
                ),
            )
            return _filter_market(tagged_data)
        return []

    return _fetch_odds_raw(sport_key)


def _prefetch_active_sports() -> set:
    """
    Fetch (or return cached) the full active sports list from The Odds API.
    Costs exactly 1 request, shared across all pipelines via _odds_cache.
    Call this once at scan startup so soccer/tennis can filter without
    paying extra requests.
    """
    global _odds_remaining
    import json
    _sports_cache_key = "__active_sports__"

    # ── Layer 1: in-memory ────────────────────────────────────────────
    if not _FORCE_FRESH_ODDS and _sports_cache_key in _odds_cache:
        fetched_at, active_sports = _odds_cache[_sports_cache_key]
        age_min = (datetime.now() - fetched_at).total_seconds() / 60
        if age_min < _ODDS_CACHE_TTL_MINUTES:
            logger.info("Active sports list: in-memory cache (%d sports)", len(active_sports))
            return active_sports

    # ── Layer 2: disk cache ───────────────────────────────────────────
    if not _FORCE_FRESH_ODDS:
        active_sports = _load_active_sports_cache(allow_stale=False)
        if active_sports:
            _odds_cache[_sports_cache_key] = (datetime.now(), active_sports)
            return active_sports

    if _OFFLINE_ODDS_ONLY:
        logger.warning("Active sports list: offline mode enabled and no saved cache exists.")
        return set()

    # ── Layer 3: live fetch ───────────────────────────────────────────
    if not ODDS_KEY or not _check_budget():
        return set()

    if not _api_throttle():
        logger.warning("API_THROTTLE: daily cap hit — returning empty set")
        return set()
    global _last_api_call_at, _api_calls_today
    _last_api_call_at = datetime.now(timezone.utc)
    _api_calls_today += 1

    try:
        r = requests.get(
            "https://api.the-odds-api.com/v4/sports/",
            params={"apiKey": ODDS_KEY},
            timeout=10,
        )
        remaining_str = r.headers.get("x-requests-remaining", "?")
        try:
            _odds_remaining = int(remaining_str)
            _save_api_usage(_odds_remaining)
        except (ValueError, TypeError):
            pass
        active_sports = {s["key"] for s in r.json() if s.get("active")}
        # Save to disk and memory
        _ODDS_DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_path = _active_sports_snapshot_path()
        stable_path = _active_sports_cache_path()
        snapshot_path.write_text(json.dumps(sorted(active_sports)))
        stable_path.write_text(json.dumps(sorted(active_sports)))
        _odds_cache[_sports_cache_key] = (datetime.now(), active_sports)
        logger.info(
            "Active sports list fetched: %d sports (remaining=%s) — saved to disk",
            len(active_sports), remaining_str,
        )
        return active_sports
    except Exception as exc:
        logger.warning("Could not fetch active sports list: %s", exc)
        stale_fallback = _load_active_sports_cache(allow_stale=True)
        if stale_fallback:
            logger.warning(
                "Active sports list: falling back to stale cache because live refresh failed (%d sports)",
                len(stale_fallback),
            )
            _odds_cache[_sports_cache_key] = (datetime.now(), stale_fallback)
            return stale_fallback
        return set()


_STALE_ODDS_RATIO = 1.6   # flag if best price is 60%+ above the median across books


def _candidate_odds_diagnostics(
    game: dict,
    *,
    market_key: str,
    selection_name: str,
    bookmaker_name: str,
) -> dict[str, Any]:
    created_at = datetime.now(timezone.utc)
    selected_last_update = _game_last_update_dt(
        game,
        bookmaker_name=bookmaker_name if not str(bookmaker_name or "").startswith("synthetic") else "",
        market_key=market_key,
    )
    game_last_update = _parse_utc_dt(game.get("_odds_bookmaker_last_update"))
    chosen_last_update = selected_last_update or game_last_update
    computed_age_hours = _odds_age_hours_from_dt(chosen_last_update, now=created_at)
    if computed_age_hours is None:
        try:
            raw_age = game.get("_odds_snapshot_age_hours")
            computed_age_hours = float(raw_age) if raw_age not in (None, "") else None
        except Exception:
            computed_age_hours = None

    source_status = str(game.get("_odds_source_status", "") or "")
    force_fresh = bool(game.get("_odds_force_fresh_requested", _FORCE_FRESH_ODDS))
    cache_used = bool(game.get("_odds_cache_used"))
    fallback_used = bool(game.get("_odds_fallback_used"))
    source_detail = _odds_source_detail(
        source_status=source_status,
        force_fresh=force_fresh,
        fallback_used=fallback_used,
        cache_used=cache_used,
        bookmaker_last_update=chosen_last_update,
        computed_age_hours=computed_age_hours,
    )
    return {
        "odds_source_status": source_status,
        "odds_source_detail": source_detail,
        "odds_source_reason": str(game.get("_odds_source_reason", "") or ""),
        "odds_fetched_at": game.get("_odds_fetched_at"),
        "bookmaker_last_update": chosen_last_update.isoformat() if chosen_last_update is not None else None,
        "cache_loaded_at": game.get("_odds_cache_loaded_at"),
        "cache_age_hours": game.get("_odds_cache_age_hours"),
        "candidate_created_at": created_at.isoformat(),
        "stale_threshold_hours": _ODDS_STALE_THRESHOLD_HOURS,
        "computed_odds_age_hours": round(float(computed_age_hours), 3) if computed_age_hours is not None else None,
        "odds_snapshot_age_hours": round(float(computed_age_hours), 3) if computed_age_hours is not None else game.get("_odds_snapshot_age_hours"),
        "odds_age_basis": game.get("_odds_age_basis", ""),
        "force_fresh_odds_active": force_fresh,
        "odds_cache_used": cache_used,
        "odds_fallback_used": fallback_used,
        "selected_market_key": market_key,
        "selected_outcome": selection_name,
        "selected_bookmaker": bookmaker_name,
    }


def _with_candidate_odds_diagnostics(
    payload: dict[str, Any],
    *,
    game: dict,
    market_key: str,
    selection_name: str,
    bookmaker_name: str,
) -> dict[str, Any]:
    diagnostics = _candidate_odds_diagnostics(
        game,
        market_key=market_key,
        selection_name=selection_name,
        bookmaker_name=bookmaker_name,
    )
    return {
        **payload,
        **diagnostics,
        "odds_diagnostics": diagnostics,
    }

def best_odds(game: dict, team_name: str) -> Tuple[Optional[float], str, bool]:
    """
    Return the best (highest) decimal odds for a team across all bookmakers,
    the bookmaker name offering that price, and a stale-line flag.

    stale=True means one book is pricing the team 60%+ higher than the median —
    likely a stale/erroneous line that shouldn't be used for Kelly sizing.

    Returns (price, bookmaker_name, is_stale_line).
    """
    prices: List[float] = []
    best_price: float = 0.0
    best_bk: str = ""
    for bk in game.get("bookmakers", []):
        bk_name = bk.get("title", "unknown")
        for mkt in bk.get("markets", []):
            if mkt["key"] == "h2h":
                for o in mkt["outcomes"]:
                    if o["name"] == team_name:
                        p = float(o["price"])
                        prices.append(p)
                        if p > best_price:
                            best_price = p
                            best_bk    = bk_name

    if best_price == 0:
        return (None, "", False)

    # Stale-line check: is the best price suspiciously far above the median?
    import statistics
    is_stale = False
    if len(prices) >= 3:
        med = statistics.median(prices)
        if med > 0 and best_price / med >= _STALE_ODDS_RATIO:
            logger.warning(
                "STALE LINE [%s]: best=%.2f vs median=%.2f (%.1fx) @ %s — flagging",
                team_name, best_price, med, best_price / med, best_bk,
            )
            is_stale = True

    return (float(best_price), best_bk, is_stale)


def median_odds(game: dict, team_name: str) -> Optional[float]:
    """Legacy helper — returns median price across all bookmakers."""
    import statistics
    prices = []
    for bk in game.get("bookmakers", []):
        for mkt in bk.get("markets", []):
            if mkt["key"] == "h2h":
                for o in mkt["outcomes"]:
                    if o["name"] == team_name:
                        prices.append(float(o["price"]))
    return statistics.median(prices) if prices else None


# ── Over/Under Poisson model ─────────────────────────────────────────────────
# Uses team's recent average goals scored/conceded to estimate λ for each side,
# then computes P(total goals > line) via Poisson CDF.  No separate ML model
# needed — purely analytical, calibrated against the same features we already have.

def _poisson_over_prob(lam: float, line: float) -> float:
    """P(X > line) where X ~ Poisson(lam). Line is typically 2.5."""
    import math
    floor_line = int(line)
    p_under_or_eq = sum(
        (lam ** k) * math.exp(-lam) / math.factorial(k)
        for k in range(floor_line + 1)
    )
    return max(0.0, min(1.0, 1.0 - p_under_or_eq))


def poisson_totals_bet(
    home_team: str,
    away_team: str,
    features_df,
    over_odds: float,
    under_odds: float,
    line: float = 2.5,
    window: str = "today",
    commence: str = "",
    league: str = "Soccer",
    sport: str = "",
) -> Optional[List[dict]]:
    """
    Compute over/under value bet using Poisson goal model.
    Returns a list of bet dicts (over and/or under) or None if no edge found.
    """
    import math

    # ── Estimate expected goals from PAST matches ONLY (no lookahead) ────────
    # We use pre-computed rolling engineered features (home_goals_scored_5,
    # home_goals_conceded_5, etc.) which are already shifted so they never
    # include the current match result. Falling back to league-average defaults
    # is safer than using raw result columns (home_goals/away_goals) which would
    # introduce lookahead bias by averaging actual goals from all historical rows.

    def _team_avg_goals(team: str, scored: bool, as_home: bool, window: int = 10) -> Optional[float]:
        """
        Pull the most recent pre-computed rolling-average goal feature for a team.
        Tries engineered column names first; falls back to computing from raw only
        if the raw column is explicitly named with a '_scored'/'_conceded' suffix
        (never from bare 'home_goals'/'away_goals' which contain current-match data).
        """
        side = "home_team" if as_home else "away_team"
        rows = features_df[features_df[side] == team]
        if rows.empty:
            return None

        # Priority 1: pre-rolled feature column (safe — shifted in feature engineer)
        direction = "scored" if scored else "conceded"
        for w in (window, 5, 10, 20):
            col = f"{'home' if as_home else 'away'}_goals_{direction}_{w}"
            if col in rows.columns:
                val = rows[col].dropna().iloc[-1] if not rows[col].dropna().empty else None
                if val is not None:
                    return float(val)

        # Priority 2: named rolling columns from soccer feature engineer
        alt_col = f"{'home' if as_home else 'away'}_{'attack' if scored else 'defence'}_strength"
        if alt_col in rows.columns:
            val = rows[alt_col].dropna().iloc[-1]
            if not pd.isna(val):
                return float(val)

        # Priority 3: compute from raw goal cols BUT only use matches BEFORE
        # a strict cutoff = the last date in the dataframe (i.e. historical only).
        # NEVER use 'home_goals'/'away_goals' directly — this would include the
        # current-match result. We must have a 'date' column to apply the cutoff.
        if "date" in features_df.columns:
            cutoff = features_df["date"].max()
            past = rows[features_df["date"] < cutoff]
            goal_col = ("home_goals" if as_home else "away_goals") if scored else \
                       ("away_goals" if as_home else "home_goals")
            if goal_col in past.columns and not past.empty:
                return float(past.tail(window)[goal_col].mean())

        return None

    home_scored   = _team_avg_goals(home_team, scored=True,  as_home=True)
    home_conceded = _team_avg_goals(home_team, scored=False, as_home=True)
    away_scored   = _team_avg_goals(away_team, scored=True,  as_home=False)
    away_conceded = _team_avg_goals(away_team, scored=False, as_home=False)

    if any(v is None for v in [home_scored, home_conceded, away_scored, away_conceded]):
        return None

    # League averages — use all historical rows (no current-match bias here since
    # we're computing a league-wide mean across many past matches)
    league_rows = features_df.iloc[:-1] if len(features_df) > 1 else features_df  # exclude last row
    league_avg_home = float(league_rows["home_goals"].mean()) if "home_goals" in league_rows.columns else 1.35
    league_avg_away = float(league_rows["away_goals"].mean()) if "away_goals" in league_rows.columns else 1.10

    lam_home = max(0.1, (home_scored / league_avg_home) * (away_conceded / league_avg_away) * league_avg_home)
    lam_away = max(0.1, (away_scored / league_avg_away) * (home_conceded / league_avg_home) * league_avg_away)
    lam_total = lam_home + lam_away

    p_over  = _poisson_over_prob(lam_total, line)
    p_under = 1.0 - p_over

    results = []
    for side, model_prob, odds, label in [
        ("over",  p_over,  over_odds,  f"Over {line}"),
        ("under", p_under, under_odds, f"Under {line}"),
    ]:
        if odds is None or odds <= 1.0:
            continue
        fair_prob = 1.0 / odds  # rough; we'll use vig-free below
        fp_over, fp_under = vig_free_prob(over_odds, under_odds)
        fair = fp_over if side == "over" else fp_under

        blended = _blend(model_prob, fair)
        edge = (blended * odds) - 1.0
        if edge < 0.03:
            continue
        _ev_cap = _SANITY_EV_CAP.get(sport, _SANITY_EV_CAP_DEFAULT)
        if edge > _ev_cap:
            continue
        market_impl = 1.0 / odds
        if blended / market_impl >= _SANITY_RATIO and (blended - market_impl) >= _SANITY_ABS_GAP:
            continue

        stake = KELLY.calculate(blended, odds)
        results.append({
            "market":        "totals",
            "team":          label,
            "ml_prob":       round(blended, 4),
            "fair_prob":     round(fair, 4),
            "odds":          round(odds, 3),
            "edge":          round(edge, 4),
            "kelly_stake_pct": round(stake * 100, 2),
            "flagged":       edge > _review_edge_threshold(sport, "totals"),
            "lam_home":      round(lam_home, 2),
            "lam_away":      round(lam_away, 2),
        })

    return results if results else None


# ── ML-based totals & spreads betting ────────────────────────────────────────
# Uses trained TotalsTrainer models (NHL, MLB totals + spreads; soccer spreads).
# For NBA totals: analytical normal-distribution model using ppg features.

def _nba_over_prob(home_ppg: float, away_ppg: float,
                   home_opp_ppg: float, away_opp_ppg: float,
                   line: float) -> float:
    """
    Estimate P(total > line) for NBA using predicted total and empirical std dev.
    Predicted total = average of both team's offensive/defensive averages.
    Total std dev ≈ 19 pts (from historical variance in game totals).
    Uses normal CDF approximation.
    """
    from scipy.stats import norm
    pred = (home_ppg + away_opp_ppg + away_ppg + home_opp_ppg) / 2.0
    sigma = 19.0  # empirical std dev of NBA game totals
    p_over = float(1.0 - norm.cdf(line, loc=pred, scale=sigma))
    return max(0.01, min(0.99, p_over))


def ml_totals_bet(
    sport: str,
    home_team: str,
    away_team: str,
    snapshot: pd.Series,
    feature_cols: list,
    totals_trainer,           # TotalsTrainer | None
    over_odds: float,
    under_odds: float,
    line: float,
    window: str = "today",
    commence: str = "",
    league: str = "",
    # NBA-specific kwargs
    home_ppg: float = 0.0,
    away_ppg: float = 0.0,
    home_opp_ppg: float = 0.0,
    away_opp_ppg: float = 0.0,
) -> Optional[List[dict]]:
    """
    Compute Over/Under value bets using ML model (NHL/MLB) or analytical model (NBA).
    Returns list of bet dicts or None.
    """
    if over_odds is None or under_odds is None or over_odds <= 1.0 or under_odds <= 1.0:
        return None

    fp_over, fp_under = vig_free_prob(over_odds, under_odds)

    # ── Compute model probabilities ────────────────────────────────────────────
    if sport == "basketball":
        if home_ppg <= 0 or away_ppg <= 0:
            return None
        p_over  = _nba_over_prob(home_ppg, away_ppg, home_opp_ppg, away_opp_ppg, line)
        p_under = 1.0 - p_over

    elif totals_trainer is not None:
        x = pd.DataFrame([snapshot], columns=feature_cols)
        p_over_arr = totals_trainer.predict_proba_over(x)
        p_over  = float(p_over_arr[0])
        p_under = 1.0 - p_over

    else:
        return None

    # ── Blend toward market and apply circuit breakers ─────────────────────────
    alpha = _SPORT_ALPHA.get(sport, 0.55)
    # Totals markets are generally harder to beat than h2h — tighter alpha
    totals_alpha = max(0.35, alpha - 0.10)

    p_over_w  = _winsorize_prob(p_over,  fp_over,  sport)
    p_under_w = _winsorize_prob(p_under, fp_under, sport)
    p_over_b  = _blend(p_over_w,  fp_over,  alpha=totals_alpha)
    p_under_b = _blend(p_under_w, fp_under, alpha=totals_alpha)

    results = []
    for side, model_prob, odds, fair_prob, label in [
        ("over",  p_over_b,  over_odds,  fp_over,  f"Over {line}"),
        ("under", p_under_b, under_odds, fp_under, f"Under {line}"),
    ]:
        edge = (model_prob * odds) - 1.0
        if edge < 0.03:
            continue
        _ev_cap = _SANITY_EV_CAP.get(sport, _SANITY_EV_CAP_DEFAULT)
        if edge > _ev_cap:
            logger.warning(
                "DATA_ANOMALY [%s %s]: EV=+%.0f%% exceeds %s cap of %.0f%% — rejected",
                label, f"{home_team} vs {away_team}", edge * 100, sport or "default", _ev_cap * 100,
            )
            continue
        ratio = model_prob / fair_prob if fair_prob > 0 else 999
        if ratio >= _SANITY_RATIO and (model_prob - fair_prob) >= _SANITY_ABS_GAP:
            continue

        stake = KELLY.calculate(model_prob, odds)
        results.append({
            "market":          "totals",
            "team":            label,
            "ml_prob":         round(model_prob, 4),
            "fair_prob":       round(fair_prob, 4),
            "odds":            round(odds, 3),
            "edge":            round(edge, 4),
            "kelly_stake_pct": round(stake * 100, 2),
            "flagged":         edge > _review_edge_threshold(sport, "totals"),
            "commence":        commence,
            "kick_off":        _kick_off_label(commence),
            "window":          window,
            "sport":           sport,
            "home":            home_team,
            "away":            away_team,
            "league":          league,
            "bookmaker":       "best",
        })

    return results if results else None


def ml_spreads_bet(
    sport: str,
    home_team: str,
    away_team: str,
    snapshot: pd.Series,
    feature_cols: list,
    spreads_trainer,          # TotalsTrainer | None
    home_spread_odds: float,  # odds for home covering the spread
    away_spread_odds: float,  # odds for away covering the spread
    line: float,              # e.g. -1.5 (home) / +1.5 (away) for NHL puck line
    window: str = "today",
    commence: str = "",
    league: str = "",
    home_name: str = "",
    away_name: str = "",
) -> Optional[List[dict]]:
    """
    Compute spread-cover value bets using trained ML model.
    home_cover = home_score - away_score > line
    """
    if spreads_trainer is None:
        return None
    if home_spread_odds is None or away_spread_odds is None:
        return None
    if home_spread_odds <= 1.0 or away_spread_odds <= 1.0:
        return None

    fp_home_cover, fp_away_cover = vig_free_prob(home_spread_odds, away_spread_odds)

    x = pd.DataFrame([snapshot], columns=feature_cols)
    p_home_cover_arr = spreads_trainer.predict_proba_over(x)
    p_home_cover = float(p_home_cover_arr[0])
    p_away_cover = 1.0 - p_home_cover

    alpha = _SPORT_ALPHA.get(sport, 0.55)
    spreads_alpha = max(0.35, alpha - 0.10)

    p_hc_w = _winsorize_prob(p_home_cover, fp_home_cover, sport)
    p_ac_w = _winsorize_prob(p_away_cover, fp_away_cover, sport)
    p_hc_b = _blend(p_hc_w, fp_home_cover, alpha=spreads_alpha)
    p_ac_b = _blend(p_ac_w, fp_away_cover, alpha=spreads_alpha)

    hn = home_name or home_team
    an = away_name or away_team
    results = []
    for side, model_prob, odds, fair_prob, label in [
        ("home_cover", p_hc_b, home_spread_odds, fp_home_cover, f"{hn} {line:+.1f}"),
        ("away_cover", p_ac_b, away_spread_odds, fp_away_cover, f"{an} {-line:+.1f}"),
    ]:
        edge = (model_prob * odds) - 1.0
        if edge < 0.03:
            continue
        _ev_cap = _SANITY_EV_CAP.get(sport, _SANITY_EV_CAP_DEFAULT)
        if edge > _ev_cap:
            logger.warning(
                "DATA_ANOMALY [%s %s]: EV=+%.0f%% exceeds %s cap of %.0f%% — rejected",
                label, f"{home_team} vs {away_team}", edge * 100, sport or "default", _ev_cap * 100,
            )
            continue
        ratio = model_prob / fair_prob if fair_prob > 0 else 999
        if ratio >= _SANITY_RATIO and (model_prob - fair_prob) >= _SANITY_ABS_GAP:
            continue

        stake = KELLY.calculate(model_prob, odds)
        results.append({
            "market":          "spreads",
            "team":            label,
            "ml_prob":         round(model_prob, 4),
            "fair_prob":       round(fair_prob, 4),
            "odds":            round(odds, 3),
            "edge":            round(edge, 4),
            "kelly_stake_pct": round(stake * 100, 2),
            "flagged":         edge > _review_edge_threshold(sport, "spreads"),
            "commence":        commence,
            "kick_off":        _kick_off_label(commence),
            "window":          window,
            "sport":           sport,
            "home":            home_team,
            "away":            away_team,
            "league":          league,
            "bookmaker":       "best",
        })

    return results if results else None


def vig_free_prob(odds_a: float, odds_b: float, odds_c: Optional[float] = None) -> Tuple:
    """Remove bookmaker margin and return fair implied probabilities."""
    raw = [1 / odds_a, 1 / odds_b]
    if odds_c is not None:
        raw.append(1 / odds_c)
    total = sum(raw)
    return tuple(p / total for p in raw)


# ── Market-anchored probability blending ─────────────────────────────────────
# Tree-based ensembles overestimate underdog win probability because they learn
# historical upsets without weighting how much information is already baked into
# liquid market odds. Blending in log-odds space shrinks model probs toward
# market consensus, preventing absurd edges like "+141% on Denver at 4.96".
#
# BLEND_ALPHA: model weight (0 = pure market, 1 = pure model).
#
# WHY 0.70 (not 0.50):
# The models are now calibrated via isotonic/Platt regression BEFORE blending.
# Using alpha=0.50 after calibration double-corrects — the calibrator already
# pulled probabilities toward true frequencies; then we'd pull them halfway back
# to the market again, undoing ~half the calibration work.
#
# At alpha=0.70:
#   - 70% calibrated model (the statistically corrected signal)
#   - 30% market consensus (a sanity anchor against gross model errors)
# This preserves calibration while still preventing lone-outlier bets where
# the model diverges wildly from every bookmaker.
#
# Raise toward 0.85 once you have 200+ settled bets showing positive CLV.
# Lower back toward 0.50 if calibration log-loss degrades on new data.
BLEND_ALPHA = 0.70

# ── Sport-specific blend weights ─────────────────────────────────────────────
# Lower alpha = rely more on market consensus, less on model.
#
# Tuning history:
#   v1: NHL=0.45, NBA=0.40 (both heavily market-led; models had only box-score features)
#   v2: NHL=0.55, NBA=0.50 (after adding xG/Corsi for NHL, SRS for NBA)
#       MLB raised from 0.55→0.60 (pitcher ERA/WHIP added; largest single-feature gain)
#       Soccer raised from 0.70→0.72 (Dixon-Coles xG added)
#
# Raise further toward 0.75-0.80 once you have 200+ settled bets with positive CLV.
# Lower back to 0.50 if calibration log-loss degrades on live data.
#
#   soccer    0.72 — DC-xG model adds real Poisson signal; 3-outcome structure is value-rich
#   tennis    0.65 — good data but small inference set; unchanged
#   baseball  0.60 — pitcher ERA/WHIP/K9 gives substantial non-public edge
#   hockey    0.55 — Corsi/xG/Fenwick features add real signal over box-score only
#   basketball 0.50 — SRS + multi-window rolling improves on PPG-ratio proxies
#
_SPORT_ALPHA = {
    "soccer":     0.72,
    "tennis":     0.55,
    "tennis_wta": 0.43,
    "mlb":        0.60,
    "nhl":        0.55,
    "basketball": 0.50,
}

# If a sport has no saved calibrator, we should trust the market more.
# These penalties are intentionally modest: they don't zero out model signal,
# they just stop us from treating raw probabilities like calibrated ones.
_UNCALIBRATED_ALPHA_PENALTY = {
    "soccer": 0.07,
    "tennis": 0.07,
    "tennis_wta": 0.10,
    "mlb": 0.06,
    "nhl": 0.08,
    "basketball": 0.10,
}
_alpha_log_cache: set[tuple[str, bool, float]] = set()

# CLV-based alpha auto-tuning settings
_CLV_MIN_BETS    = 50     # require at least this many settled bets per sport before auto-tuning
_CLV_ALPHA_STEP  = 0.03   # how much to adjust alpha per tuning pass
_CLV_ALPHA_MIN   = 0.35   # floor — never go fully market-led
_CLV_ALPHA_MAX   = 0.85   # ceiling — never ignore market entirely

_MARKET_HEALTH_MIN_BETS = 8
_MARKET_HEALTH_MIN_CLV_COVERAGE_PCT = 60.0
_MARKET_HEALTH_BY_MARKET: dict[tuple[str, str], dict[str, Any]] = {}


def _autotune_alpha_from_clv() -> None:
    """
    Read settled bets from tracker and auto-adjust _SPORT_ALPHA based on CLV.

    Logic:
      - Positive avg CLV (bet_odds > closing_odds on avg) → model is beating closing line
        → trust model more → raise alpha by _CLV_ALPHA_STEP
      - Negative avg CLV → model lags the closing line → lower alpha by _CLV_ALPHA_STEP
      - Only applies if >= _CLV_MIN_BETS settled bets with CLV data exist for that sport.
      - Changes are clamped to [_CLV_ALPHA_MIN, _CLV_ALPHA_MAX].

    This runs once at scan startup so each day's picks use updated weights.
    """
    settled_path = Path("data/tracker/settled.parquet")
    if not settled_path.exists():
        return

    try:
        df = pd.read_parquet(settled_path)
    except Exception as exc:
        logger.warning("CLV auto-tune: failed to load settled.parquet — %s", exc)
        return

    if "clv" not in df.columns or "sport" not in df.columns:
        return

    clv_df = df.dropna(subset=["clv"])
    if clv_df.empty:
        return

    adjustments = []
    for sport, group in clv_df.groupby("sport"):
        if sport not in _SPORT_ALPHA:
            continue
        if len(group) < _CLV_MIN_BETS:
            logger.debug(
                "CLV auto-tune [%s]: only %d bets with CLV (need %d) — skipping",
                sport, len(group), _CLV_MIN_BETS
            )
            continue

        avg_clv = group["clv"].mean()
        old_alpha = _SPORT_ALPHA[sport]

        if avg_clv > 0.005:       # meaningfully positive CLV → raise alpha
            new_alpha = min(old_alpha + _CLV_ALPHA_STEP, _CLV_ALPHA_MAX)
        elif avg_clv < -0.005:    # meaningfully negative CLV → lower alpha
            new_alpha = max(old_alpha - _CLV_ALPHA_STEP, _CLV_ALPHA_MIN)
        else:
            new_alpha = old_alpha  # neutral — leave alone

        if new_alpha != old_alpha:
            _SPORT_ALPHA[sport] = round(new_alpha, 3)
            adjustments.append(
                f"{sport}: {old_alpha:.2f} → {new_alpha:.2f}  (avg CLV={avg_clv:+.1%}, n={len(group)})"
            )

    if adjustments:
        logger.info("CLV auto-tune applied:\n  " + "\n  ".join(adjustments))
    else:
        logger.debug("CLV auto-tune: no adjustments needed")


def _load_market_health_snapshot() -> None:
    """Build a lightweight live governor snapshot from settled tracker results."""
    global _MARKET_HEALTH_BY_MARKET

    settled_path = Path("data/tracker/settled.parquet")
    if not settled_path.exists():
        _MARKET_HEALTH_BY_MARKET = {}
        return

    try:
        df = pd.read_parquet(settled_path)
    except Exception as exc:
        logger.warning("Market health snapshot: failed to load settled.parquet — %s", exc)
        _MARKET_HEALTH_BY_MARKET = {}
        return

    if df.empty or "sport" not in df.columns:
        _MARKET_HEALTH_BY_MARKET = {}
        return

    work = df.copy()
    if "market" not in work.columns:
        work["market"] = "moneyline"
    else:
        work["market"] = work["market"].fillna("moneyline").replace("", "moneyline")

    snapshot: dict[tuple[str, str], dict[str, Any]] = {}
    weak_summaries: list[str] = []
    grouped = work.groupby(["sport", "market"], dropna=False)
    for (sport, market), grp in grouped:
        bets = int(len(grp))
        stake = float(pd.to_numeric(grp.get("stake_units"), errors="coerce").fillna(0).sum()) if "stake_units" in grp.columns else 0.0
        pnl = float(pd.to_numeric(grp.get("profit_units"), errors="coerce").fillna(0).sum()) if "profit_units" in grp.columns else 0.0
        roi_pct = (pnl / stake * 100.0) if stake > 0 else 0.0
        clv_series = pd.to_numeric(grp.get("clv"), errors="coerce") if "clv" in grp.columns else pd.Series(dtype="float64")
        clv_clean = clv_series.dropna()
        clv_covered = int(clv_clean.shape[0])
        clv_coverage_pct = (clv_covered / bets * 100.0) if bets else 0.0
        avg_clv_pct = float(clv_clean.mean() * 100.0) if not clv_clean.empty else None

        if clv_clean.empty:
            clv_signal = "missing"
        elif avg_clv_pct > 0 and roi_pct > 0:
            clv_signal = "confirmed"
        elif avg_clv_pct > 0 and roi_pct <= 0:
            clv_signal = "variance"
        elif avg_clv_pct <= 0 and roi_pct > 0:
            clv_signal = "lucky"
        else:
            clv_signal = "weak"

        action = "watch"
        if bets >= _MARKET_HEALTH_MIN_BETS and clv_coverage_pct >= _MARKET_HEALTH_MIN_CLV_COVERAGE_PCT:
            if avg_clv_pct is not None and avg_clv_pct <= -10.0 and roi_pct <= -5.0:
                action = "pause"
            elif avg_clv_pct is not None and avg_clv_pct <= -5.0 and roi_pct <= -3.0:
                action = "review"
            elif clv_signal in {"variance", "lucky"} or (avg_clv_pct is not None and avg_clv_pct <= -2.0):
                action = "tighten"
            elif avg_clv_pct is not None and avg_clv_pct >= 1.5 and roi_pct >= 3.0:
                action = "confirmed"

        key = (str(sport or ""), str(market or "moneyline"))
        snapshot[key] = {
            "sport": key[0],
            "market": key[1],
            "bets": bets,
            "roi_pct": round(roi_pct, 2),
            "avg_clv_pct": None if avg_clv_pct is None else round(avg_clv_pct, 2),
            "clv_covered": clv_covered,
            "clv_coverage_pct": round(clv_coverage_pct, 1),
            "clv_signal": clv_signal,
            "action": action,
        }
        if action in {"pause", "review"}:
            weak_summaries.append(
                f"{key[0]}:{key[1]}={action} (ROI {roi_pct:+.1f}%, CLV {avg_clv_pct if avg_clv_pct is not None else 'n/a'}%, n={bets})"
            )

    _MARKET_HEALTH_BY_MARKET = snapshot
    if weak_summaries:
        logger.info("Market health governor loaded: %s", "; ".join(weak_summaries[:8]))


def _market_health_row(sport: str, market: str) -> Optional[dict[str, Any]]:
    return _MARKET_HEALTH_BY_MARKET.get((str(sport or ""), str(market or "moneyline")))


def _market_health_alpha_penalty(sport: str, market: str) -> float:
    row = _market_health_row(sport, market)
    if not row:
        return 0.0
    if int(row.get("bets", 0) or 0) < _MARKET_HEALTH_MIN_BETS:
        return 0.0
    if float(row.get("clv_coverage_pct", 0.0) or 0.0) < _MARKET_HEALTH_MIN_CLV_COVERAGE_PCT:
        return 0.0

    action = str(row.get("action", "watch"))
    signal = str(row.get("clv_signal", "missing"))
    if action == "pause":
        return 0.18
    if action == "review":
        return 0.12
    if action == "tighten":
        return 0.06 if signal != "lucky" else 0.08
    if signal == "variance":
        return 0.04
    return 0.0


def _disagreement_alpha_penalty(disagreement_pp: Optional[float]) -> float:
    if disagreement_pp is None:
        return 0.0
    gap = max(0.0, float(disagreement_pp) - 8.0)
    return round(min(0.10, gap * 0.005), 3)


def _market_health_adjusted_alpha(
    sport: str,
    market: str,
    base_alpha: float,
    *,
    disagreement_pp: Optional[float] = None,
) -> float:
    alpha = float(base_alpha)
    alpha = max(_CLV_ALPHA_MIN, alpha - _market_health_alpha_penalty(sport, market))
    alpha = max(_CLV_ALPHA_MIN, alpha - _disagreement_alpha_penalty(disagreement_pp))
    return round(float(np.clip(alpha, _CLV_ALPHA_MIN, _CLV_ALPHA_MAX)), 3)


def _live_blend_alpha(
    sport: str,
    market: str,
    base_alpha: float,
    *,
    home_rows: Optional[int] = None,
    away_rows: Optional[int] = None,
    min_rows: int = _MIN_SNAPSHOT_ROWS,
    disagreement_pp: Optional[float] = None,
) -> float:
    if home_rows is not None and away_rows is not None:
        alpha = _history_adjusted_alpha(
            sport,
            base_alpha,
            home_rows=int(home_rows or 0),
            away_rows=int(away_rows or 0),
            min_rows=min_rows,
        )
    else:
        alpha = round(float(np.clip(base_alpha, _CLV_ALPHA_MIN, _CLV_ALPHA_MAX)), 3)
    return _market_health_adjusted_alpha(
        sport,
        market,
        alpha,
        disagreement_pp=disagreement_pp,
    )


def _effective_alpha(sport: str, calibrated: bool) -> float:
    """Return live blend weight, lowering alpha when no calibrator is active."""
    base_alpha = _SPORT_ALPHA.get(sport, BLEND_ALPHA)
    alpha = base_alpha
    if not calibrated:
        alpha = max(_CLV_ALPHA_MIN, base_alpha - _UNCALIBRATED_ALPHA_PENALTY.get(sport, 0.08))

    key = (sport, calibrated, round(alpha, 3))
    if key not in _alpha_log_cache:
        status = "calibrated" if calibrated else "uncalibrated"
        logger.info("Blend alpha [%s]: %.2f (%s)", sport, alpha, status)
        _alpha_log_cache.add(key)
    return alpha


def _history_reliability(rows_available: int, *, min_rows: int, full_rows: int) -> float:
    """Score how trustworthy a snapshot is based on usable history volume."""
    rows = max(0, int(rows_available or 0))
    if rows <= min_rows:
        return 0.0
    if rows >= full_rows:
        return 1.0
    span = max(1, full_rows - min_rows)
    return float(np.clip((rows - min_rows) / span, 0.0, 1.0))


def _history_adjusted_alpha(
    sport: str,
    base_alpha: float,
    *,
    home_rows: int,
    away_rows: int,
    min_rows: int = _MIN_SNAPSHOT_ROWS,
) -> float:
    """
    Pull blend alpha toward the market when either side has shallow history.

    This keeps newer / newly promoted / renamed teams scorable without letting
    sparse snapshots create wildly overconfident probabilities.
    """
    full_rows = _HISTORY_BLEND_FULL_ROWS.get(sport, max(min_rows + 8, 24))
    reliability = min(
        _history_reliability(home_rows, min_rows=min_rows, full_rows=full_rows),
        _history_reliability(away_rows, min_rows=min_rows, full_rows=full_rows),
    )
    floor = max(_CLV_ALPHA_MIN, base_alpha - 0.18)
    adjusted = floor + ((base_alpha - floor) * reliability)
    return round(float(np.clip(adjusted, _CLV_ALPHA_MIN, _CLV_ALPHA_MAX)), 3)


def _blend(model_prob: float, market_implied: float, alpha: float = BLEND_ALPHA) -> float:
    """Blend model probability toward market-implied in log-odds space."""
    mp = max(min(model_prob,  0.9999), 0.0001)
    mkt = max(min(market_implied, 0.9999), 0.0001)
    lo_model  = np.log(mp  / (1 - mp))
    lo_market = np.log(mkt / (1 - mkt))
    blended   = alpha * lo_model + (1 - alpha) * lo_market
    return float(1.0 / (1.0 + np.exp(-blended)))


def _blend3(p_home: float, p_draw: float, p_away: float,
            fp_home: float, fp_draw: float, fp_away: float,
            alpha: float = BLEND_ALPHA):
    """Blend and renormalise three-outcome probabilities (soccer)."""
    bh = _blend(p_home, fp_home, alpha)
    bd = _blend(p_draw, fp_draw, alpha)
    ba = _blend(p_away, fp_away, alpha)
    total = bh + bd + ba
    return bh / total, bd / total, ba / total


_SOCCER_SCORE_MODEL = SoccerScoreModel()


def _normalize_three_probs(p_home: float, p_draw: float, p_away: float) -> tuple[float, float, float]:
    return _SOCCER_SCORE_MODEL.normalize_three_probs(p_home, p_draw, p_away).as_tuple()


def _soccer_structural_probs(snapshot: Optional[pd.Series]) -> Optional[tuple[float, float, float]]:
    view = _SOCCER_SCORE_MODEL.structural_probs_from_snapshot(snapshot)
    return None if view is None else view.as_tuple()


def _combine_soccer_probability_views(
    model_probs: tuple[float, float, float],
    structural_probs: Optional[tuple[float, float, float]],
) -> tuple[float, float, float]:
    return _SOCCER_SCORE_MODEL.combine_with_classifier(model_probs, structural_probs).as_tuple()


# ── Per-sport dynamic EV circuit breakers ────────────────────────────────────
# Each sport has its own cap based on how liquid/reliable its odds market is.
# Tighter caps on liquid markets (NBA/EPL) where a 10%+ edge almost always
# signals a model error. Looser on tennis/MLB where variance is higher.
# Anything above the cap is tagged DATA_ANOMALY and rejected — never placed.
_SANITY_EV_CAP: dict = {
    "soccer":     0.10,   # EPL/top leagues: tight market, >10% EV = data error
    "basketball": 0.10,   # NBA: extremely liquid, >10% EV = model overfit
    "nhl":        0.14,   # NHL: slightly looser now that moneyline/spread validation has improved
    "mlb":        0.13,   # MLB: some variance in lines, mild looser
    "tennis":     0.15,   # Tennis: wider spreads, small events — most permissive
    "tennis_wta": 0.12,   # WTA stays stricter until a held-out calibrator proves reliable
}
_SANITY_EV_CAP_DEFAULT = 0.12   # fallback for any unlisted sport
_SANITY_EV_FLAG  = 0.07          # flag with ⚠️ if edge > 7% (tightened from 12%)
_SANITY_ABS_GAP  = 0.18          # primary guard: drop if ml_prob − market_implied > 18pp
_SANITY_RATIO    = 2.5           # secondary guard: drop if ml_prob > 2.5x market implied
# Uses AND logic for ratio+gap — BOTH must trigger to drop (prevents false positives on
# legitimate longshot value where ratio is high but absolute gap is small).
# The EV_CAP is independent — fires alone if EV > 30% regardless of ratio/gap.

# ── Pre-blend probability winsorization ──────────────────────────────────────
# Before blending, clip model probability to within _WINSOR_MAX_GAP of the market.
# This prevents extreme model divergences (e.g. 0.85 model vs 0.45 market) from
# dominating the blend even at low alpha. The gap cap is per-sport:
#   - Lower-trust sports (basketball/NHL): tighter clamp, max 20pp over market
#   - Higher-trust sports (soccer/tennis): looser clamp, max 30pp over market
#
# This does NOT prevent genuine value bets — real edges rarely exceed 15% at liquid
# markets, so a model prob 30pp above market almost always signals a data error.
_WINSOR_GAP: dict = {
    "basketball": 0.15,   # tight: 15pp max over market — cal data sparse, overconfidence chronic
    "nhl":        0.15,   # tight: same rationale as basketball
    "mlb":        0.20,   # moderate: decent training set, binary target
    "tennis":     0.08,   # tighter: ATP still throws clay-moneyline outliers even with calibration
    "tennis_wta": 0.055,  # strict: WTA remains uncalibrated and needs a harder market anchor
    "soccer":     0.14,   # tighter: 3-way probabilities still need a stronger market anchor
}

_REVIEW_EDGE_BY_MARKET: dict[tuple[str, str], float] = {
    ("soccer", "double_chance"): 0.10,
    ("soccer", "draw_no_bet"): 0.11,
    ("soccer", "spreads"): 0.09,
    ("soccer", "totals"): 0.085,
    ("mlb", "moneyline"): 0.13,
    ("mlb", "spreads"): 0.125,
    ("nhl", "moneyline"): 0.14,
    ("tennis", "moneyline"): 0.15,
    ("tennis", "totals"): 0.13,
    ("tennis", "set_betting"): 0.14,
    ("tennis_wta", "moneyline"): 0.12,
    ("tennis_wta", "totals"): 0.11,
    ("tennis_wta", "set_betting"): 0.12,
    ("basketball", "moneyline"): 0.085,
    ("basketball", "totals"): 0.10,
    ("basketball", "team_total"): 0.10,
    ("nhl", "spreads"): 0.10,
}


def _review_edge_threshold(
    sport: str,
    market: str,
    *,
    synthetic_line: bool = False,
    bookmaker: str = "",
) -> float:
    sport_key = (sport or "").lower()
    market_key = (market or "").lower()
    threshold = _REVIEW_EDGE_BY_MARKET.get((sport_key, market_key), _SANITY_EV_FLAG)
    if synthetic_line or str(bookmaker or "").startswith("synthetic"):
        threshold = min(threshold, _SANITY_EV_CAP.get(sport_key, _SANITY_EV_CAP_DEFAULT) * 0.95)
    return threshold

def _winsorize_prob(model_prob: float, market_prob: float, sport: str) -> float:
    """
    Clamp model_prob to within _WINSOR_GAP[sport] of market_prob (in either direction).
    This prevents extreme divergences from dominating the blend.
    """
    gap = _WINSOR_GAP.get(sport, 0.25)
    return float(np.clip(model_prob, market_prob - gap, market_prob + gap))


def _soccer_market_clamp_gap(
    *,
    home_rows: int,
    away_rows: int,
    home_synthetic: bool,
    away_synthetic: bool,
) -> float:
    """
    Return a soccer-specific market clamp gap before EV calculation.

    Three-way soccer prices remain the noisiest lane in the current scan. When
    either side has synthetic or shallow history we intentionally force the
    model closer to the market before downstream EV checks.
    """
    min_rows = min(int(home_rows or 0), int(away_rows or 0))
    if home_synthetic and away_synthetic:
        return 0.04
    if home_synthetic or away_synthetic:
        return 0.045
    if min_rows < 10:
        return 0.07
    if min_rows < 18:
        return 0.09
    return 0.12


def _soccer_market_clamp_three_way(
    model_probs: tuple[float, float, float],
    market_probs: tuple[float, float, float],
    *,
    home_rows: int,
    away_rows: int,
    home_synthetic: bool = False,
    away_synthetic: bool = False,
) -> tuple[tuple[float, float, float], dict[str, Any]]:
    gap = _soccer_market_clamp_gap(
        home_rows=home_rows,
        away_rows=away_rows,
        home_synthetic=home_synthetic,
        away_synthetic=away_synthetic,
    )
    clamped = tuple(
        float(np.clip(model_prob, market_prob - gap, market_prob + gap))
        for model_prob, market_prob in zip(model_probs, market_probs)
    )
    normalized = _normalize_three_probs(*clamped)
    return normalized, {
        "gap": round(float(gap), 4),
        "home_synthetic": bool(home_synthetic),
        "away_synthetic": bool(away_synthetic),
    }


def _normalize_two_probs(p_home: float, p_away: float) -> tuple[float, float]:
    home = float(np.clip(p_home, 1e-6, 1.0 - 1e-6))
    away = float(np.clip(p_away, 1e-6, 1.0 - 1e-6))
    total = home + away
    if total <= 0:
        return 0.5, 0.5
    return home / total, away / total


def _apply_mlb_context_probability_adjustment(
    p_home: float,
    p_away: float,
    context: Optional[dict],
) -> tuple[float, float, dict]:
    """
    Let MLB-critical context affect the probability before market anchoring.

    This stays intentionally modest: missing/changed starters and incomplete
    lineups compress confidence toward 50/50, while bullpen load and weather
    can add small directional nudges when the enrichment payload has them.
    """
    context = context or {}
    base_home, base_away = _normalize_two_probs(p_home, p_away)
    home = base_home
    reliability = 1.0
    reasons: list[str] = []

    pitcher_changed = bool(
        context.get("home_pitcher_changed")
        or context.get("away_pitcher_changed")
        or context.get("pitcher_change_detected")
    )
    if pitcher_changed:
        reliability *= 0.75
        reasons.append("pitcher_change")

    home_starter_known = context.get("home_starter_confirmed") is not None or bool(str(context.get("home_starter_name") or "").strip())
    away_starter_known = context.get("away_starter_confirmed") is not None or bool(str(context.get("away_starter_name") or "").strip())
    if not (home_starter_known and away_starter_known):
        reliability *= 0.88 if not home_starter_known and not away_starter_known else 0.93
        reasons.append("starter_uncertainty")

    home_lineup_known = bool(context.get("home_lineup_confirmed")) or int(context.get("home_likely_starters_count", 0) or 0) >= 7
    away_lineup_known = bool(context.get("away_lineup_confirmed")) or int(context.get("away_likely_starters_count", 0) or 0) >= 7
    if not (home_lineup_known and away_lineup_known):
        reliability *= 0.91 if not home_lineup_known and not away_lineup_known else 0.95
        reasons.append("lineup_uncertainty")

    if int(context.get("weather_risk", 0) or 0):
        reliability *= 0.96
        reasons.append("weather_risk")

    home = 0.5 + ((home - 0.5) * reliability)

    home_games_l3d = _as_float(context.get("home_games_L3D"), 0.0)
    away_games_l3d = _as_float(context.get("away_games_L3D"), 0.0)
    bullpen_delta = float(np.clip((away_games_l3d - home_games_l3d) * 0.006, -0.018, 0.018))
    if abs(bullpen_delta) >= 0.003:
        home += bullpen_delta
        reasons.append("bullpen_load")

    adjusted_home, adjusted_away = _normalize_two_probs(home, 1.0 - home)
    debug = {
        "applied": bool(reasons),
        "reasons": reasons,
        "reliability": round(float(reliability), 3),
        "home_delta_pp": round((adjusted_home - base_home) * 100.0, 2),
        "away_delta_pp": round((adjusted_away - base_away) * 100.0, 2),
    }
    return adjusted_home, adjusted_away, debug


def _apply_basketball_context_probability_adjustment(
    p_home: float,
    p_away: float,
    context: Optional[dict],
) -> tuple[float, float, dict]:
    """Apply modest NBA availability/rest context before the market anchor."""
    context = context or {}
    base_home, base_away = _normalize_two_probs(p_home, p_away)
    home = base_home
    reliability = 1.0
    reasons: list[str] = []

    home_projected = int(context.get("home_projected_starters_count", 0) or context.get("home_likely_starters_count", 0) or 0)
    away_projected = int(context.get("away_projected_starters_count", 0) or context.get("away_likely_starters_count", 0) or 0)
    home_lineup_known = bool(context.get("home_lineup_confirmed")) or home_projected >= 4
    away_lineup_known = bool(context.get("away_lineup_confirmed")) or away_projected >= 4
    if not (home_lineup_known and away_lineup_known):
        reliability *= 0.90 if not home_lineup_known and not away_lineup_known else 0.95
        reasons.append("lineup_uncertainty")

    home_priority = int(context.get("home_priority_absences_count", 0) or 0)
    away_priority = int(context.get("away_priority_absences_count", 0) or 0)
    home_questionable = int(context.get("home_questionable_count", 0) or 0)
    away_questionable = int(context.get("away_questionable_count", 0) or 0)
    if home_priority or away_priority or home_questionable or away_questionable:
        reliability *= 0.92
        reasons.append("star_status_uncertainty")
        absence_delta = float(np.clip(((away_priority - home_priority) * 0.018) + ((away_questionable - home_questionable) * 0.006), -0.04, 0.04))
        home += absence_delta
        if abs(absence_delta) >= 0.004:
            reasons.append("availability_delta")

    rest_advantage = _as_float(context.get("rest_advantage"), 0.0)
    if abs(rest_advantage) >= 1.0:
        home += float(np.clip(rest_advantage * 0.008, -0.024, 0.024))
        reasons.append("rest_advantage")

    travel_penalty = 0.0
    if context.get("away_cross_country"):
        travel_penalty += 0.008
    if context.get("away_crossed_2tz"):
        travel_penalty += 0.006
    travel_bucket = _as_float(context.get("away_travel_bucket"), 0.0)
    if travel_bucket >= 2:
        travel_penalty += min(0.012, travel_bucket * 0.003)
    if travel_penalty:
        home += float(np.clip(travel_penalty, 0.0, 0.026))
        reasons.append("away_travel_load")

    fixture_congestion = bool(context.get("fixture_congestion_risk") or context.get("away_b2b") or context.get("home_b2b"))
    if fixture_congestion:
        reliability *= 0.96
        reasons.append("fixture_congestion")

    home = 0.5 + ((home - 0.5) * reliability)
    adjusted_home, adjusted_away = _normalize_two_probs(home, 1.0 - home)
    debug = {
        "applied": bool(reasons),
        "reasons": reasons,
        "reliability": round(float(reliability), 3),
        "home_delta_pp": round((adjusted_home - base_home) * 100.0, 2),
        "away_delta_pp": round((adjusted_away - base_away) * 100.0, 2),
    }
    return adjusted_home, adjusted_away, debug


def _apply_nhl_context_probability_adjustment(
    p_home: float,
    p_away: float,
    context: Optional[dict],
) -> tuple[float, float, dict]:
    """Apply modest NHL goalie/rest/special-teams context before market anchoring."""
    context = context or {}
    base_home, base_away = _normalize_two_probs(p_home, p_away)
    home = base_home
    reliability = 1.0
    reasons: list[str] = []

    home_goalie_known = bool(context.get("home_goalie_confirmed")) or bool(str(context.get("home_goalie_name") or "").strip())
    away_goalie_known = bool(context.get("away_goalie_confirmed")) or bool(str(context.get("away_goalie_name") or "").strip())
    if not (home_goalie_known and away_goalie_known):
        reliability *= 0.86 if not home_goalie_known and not away_goalie_known else 0.93
        reasons.append("goalie_uncertainty")

    goalie_status = str(context.get("goalie_status") or "").lower()
    if "uncertain" in goalie_status or "unconfirmed" in goalie_status:
        reliability *= 0.94
        reasons.append("goalie_status_uncertain")

    rest_advantage = _as_float(context.get("rest_advantage"), 0.0)
    if abs(rest_advantage) >= 1.0:
        home += float(np.clip(rest_advantage * 0.007, -0.021, 0.021))
        reasons.append("rest_advantage")

    travel_penalty = 0.0
    if context.get("away_cross_country"):
        travel_penalty += 0.007
    if context.get("away_crossed_2tz"):
        travel_penalty += 0.005
    travel_bucket = _as_float(context.get("away_travel_bucket"), 0.0)
    if travel_bucket >= 2:
        travel_penalty += min(0.010, travel_bucket * 0.0025)
    if travel_penalty:
        home += float(np.clip(travel_penalty, 0.0, 0.022))
        reasons.append("away_travel_load")

    if context.get("fixture_congestion_risk") or context.get("home_b2b") or context.get("away_b2b"):
        reliability *= 0.96
        reasons.append("fixture_congestion")

    pp_delta = _as_float(context.get("home_pp_pct_10"), 0.0) - _as_float(context.get("away_pp_pct_10"), 0.0)
    pk_delta = _as_float(context.get("home_pk_pct_10"), 0.0) - _as_float(context.get("away_pk_pct_10"), 0.0)
    special_teams_delta = float(np.clip((pp_delta + (0.6 * pk_delta)) * 0.025, -0.018, 0.018))
    if abs(special_teams_delta) >= 0.004:
        home += special_teams_delta
        reasons.append("special_teams_edge")

    home = 0.5 + ((home - 0.5) * reliability)
    adjusted_home, adjusted_away = _normalize_two_probs(home, 1.0 - home)
    debug = {
        "applied": bool(reasons),
        "reasons": reasons,
        "reliability": round(float(reliability), 3),
        "home_delta_pp": round((adjusted_home - base_home) * 100.0, 2),
        "away_delta_pp": round((adjusted_away - base_away) * 100.0, 2),
    }
    return adjusted_home, adjusted_away, debug


def _apply_soccer_context_probability_adjustment(
    p_home: float,
    p_draw: float,
    p_away: float,
    context: Optional[dict],
) -> tuple[float, float, float, dict]:
    """Apply modest soccer lineup/injury/rotation context before market anchoring."""
    context = context or {}
    base_home, base_draw, base_away = _normalize_three_probs(p_home, p_draw, p_away)
    home, draw, away = base_home, base_draw, base_away
    reliability = 1.0
    reasons: list[str] = []

    home_lineup_known = bool(context.get("home_lineup_confirmed")) or int(context.get("home_likely_starters_count", 0) or 0) >= 8
    away_lineup_known = bool(context.get("away_lineup_confirmed")) or int(context.get("away_likely_starters_count", 0) or 0) >= 8
    if not (home_lineup_known and away_lineup_known):
        reliability *= 0.90 if not home_lineup_known and not away_lineup_known else 0.95
        reasons.append("lineup_uncertainty")

    home_priority = int(context.get("home_priority_absences_count", 0) or context.get("home_spine_absences_count", 0) or 0)
    away_priority = int(context.get("away_priority_absences_count", 0) or context.get("away_spine_absences_count", 0) or 0)
    home_questionable = int(context.get("home_questionable_count", 0) or 0)
    away_questionable = int(context.get("away_questionable_count", 0) or 0)
    if home_priority or away_priority or home_questionable or away_questionable:
        reliability *= 0.93
        reasons.append("availability_context")
        home += float(np.clip(((away_priority - home_priority) * 0.012) + ((away_questionable - home_questionable) * 0.004), -0.035, 0.035))

    rotation_risk = bool(
        context.get("cup_rotation_risk")
        or context.get("european_rotation_risk")
        or context.get("fixture_congestion_risk")
        or context.get("final_day_volatility")
    )
    if rotation_risk:
        reliability *= 0.94
        reasons.append("rotation_or_congestion")

    if context.get("nothing_to_play_for"):
        reliability *= 0.96
        reasons.append("motivation_uncertainty")
    if context.get("playoff_motivation") or context.get("rivalry_fixture"):
        reasons.append("motivation_context")

    side_mass = max(1e-6, home + away)
    side_home_share = home / side_mass
    side_edge = (side_home_share - 0.5) * reliability
    side_mass = 1.0 - draw
    draw_boost = min(0.04, (1.0 - reliability) * 0.16)
    draw = float(np.clip(draw + draw_boost, 0.05, 0.55))
    side_mass = 1.0 - draw
    home = side_mass * (0.5 + side_edge)
    away = side_mass - home
    adjusted_home, adjusted_draw, adjusted_away = _normalize_three_probs(home, draw, away)
    debug = {
        "applied": bool(reasons),
        "reasons": reasons,
        "reliability": round(float(reliability), 3),
        "home_delta_pp": round((adjusted_home - base_home) * 100.0, 2),
        "draw_delta_pp": round((adjusted_draw - base_draw) * 100.0, 2),
        "away_delta_pp": round((adjusted_away - base_away) * 100.0, 2),
    }
    return adjusted_home, adjusted_draw, adjusted_away, debug


def _apply_tennis_context_probability_adjustment(
    p_player1: float,
    p_player2: float,
    snapshot: Optional[pd.Series],
    context: Optional[dict],
) -> tuple[float, float, dict]:
    """Apply modest tennis injury/fatigue/surface context before market anchoring."""
    context = context or {}
    base_p1, base_p2 = _normalize_two_probs(p_player1, p_player2)
    p1 = base_p1
    reliability = 1.0
    reasons: list[str] = []

    injury_or_retirement_risk = bool(
        context.get("injury_concern")
        or context.get("retirement_concern")
        or context.get("injury_status_uncertain")
        or context.get("retirement_risk")
    )
    if injury_or_retirement_risk:
        reliability *= 0.88
        reasons.append("injury_or_retirement_risk")

    fatigue_delta = 0.0
    if snapshot is not None and "load_diff" in snapshot.index:
        fatigue_delta = _as_float(snapshot.get("load_diff"), 0.0)
    fatigue_delta += _as_float(context.get("player1_match_load"), 0.0) - _as_float(context.get("player2_match_load"), 0.0)
    if abs(fatigue_delta) >= 0.5 or context.get("fatigue_risk") or context.get("recent_long_match"):
        p1 += float(np.clip(-fatigue_delta * 0.006, -0.024, 0.024))
        reliability *= 0.96
        reasons.append("fatigue_load")

    surface_edge = _as_float(snapshot.get("surface_win_diff"), 0.0) if snapshot is not None and "surface_win_diff" in snapshot.index else 0.0
    if abs(surface_edge) >= 0.05:
        p1 += float(np.clip(surface_edge * 0.035, -0.025, 0.025))
        reasons.append("surface_context")

    serve_edge = _as_float(snapshot.get("serve_balance_diff"), 0.0) if snapshot is not None and "serve_balance_diff" in snapshot.index else 0.0
    if abs(serve_edge) >= 0.02:
        p1 += float(np.clip(serve_edge * 0.05, -0.02, 0.02))
        reasons.append("serve_return_context")

    if context.get("travel_required") or context.get("timezone_shift") or context.get("travel_relevant"):
        reliability *= 0.97
        reasons.append("travel_context")

    p1 = 0.5 + ((p1 - 0.5) * reliability)
    adjusted_p1, adjusted_p2 = _normalize_two_probs(p1, 1.0 - p1)
    debug = {
        "applied": bool(reasons),
        "reasons": reasons,
        "reliability": round(float(reliability), 3),
        "player1_delta_pp": round((adjusted_p1 - base_p1) * 100.0, 2),
        "player2_delta_pp": round((adjusted_p2 - base_p2) * 100.0, 2),
    }
    return adjusted_p1, adjusted_p2, debug


_PROBABILITY_DEBUG_KEYS = (
    "soccer_probability_debug",
    "mlb_probability_debug",
    "basketball_probability_debug",
    "nhl_probability_debug",
    "tennis_probability_debug",
)


def _prediction_diagnostics_for_tracker(bet: dict) -> dict:
    """Extract passive probability diagnostics for post-result analysis."""
    probability_debug = {}
    for key in _PROBABILITY_DEBUG_KEYS:
        payload = bet.get(key)
        if isinstance(payload, dict) and payload:
            probability_debug = payload
            break

    context_debug = probability_debug.get("context_probability_adjustment") if isinstance(probability_debug, dict) else {}
    if not isinstance(context_debug, dict):
        context_debug = {}

    reasons = [
        str(item)
        for item in (context_debug.get("reasons") or [])
        if str(item).strip()
    ]

    return {
        "probability_debug": probability_debug,
        "probability_context_applied": bool(context_debug.get("applied", False)),
        "probability_context_reasons": json.dumps(reasons),
        "structural_available": bool(probability_debug.get("structural_available", False)) if isinstance(probability_debug, dict) else False,
        "structural_weight": probability_debug.get("structural_weight", 0.0) if isinstance(probability_debug, dict) else 0.0,
    }


def _apply_tennis_rank_gap_damping(
    model_prob: float,
    market_prob: float,
    snapshot: Optional[pd.Series],
    sport: str,
) -> float:
    """
    Compress WTA probabilities back toward the market when the matchup is
    dominated by large rank / ranking-points gaps.

    WTA currently lacks a held-out calibrator we trust, so large ranking
    mismatches can still produce overconfident underdog edges even after the
    normal winsor + blend path. ATP keeps its current calibrated lane.
    """
    if (sport or "").lower() != "tennis_wta" or snapshot is None:
        return float(model_prob)

    rank_gap = abs(_as_float(snapshot.get("rank_log_ratio"), 0.0))
    rank_pts_gap = abs(_as_float(snapshot.get("rank_pts_log_ratio"), 0.0))
    seed_gap = abs(_as_float(snapshot.get("seed_advantage"), 0.0))

    severity = max(
        rank_gap / 2.0,
        rank_pts_gap / 2.4,
        seed_gap * 0.4,
    )
    severity = float(np.clip(severity, 0.0, 1.0))
    if severity <= 0.0:
        return float(model_prob)

    compression = 1.0 - (0.22 * severity)
    return float(market_prob + ((model_prob - market_prob) * compression))

def _sanity_check(team: str, ml_prob: float, team_odds: float, sport: str = "") -> bool:
    """
    Drop bets where the model probability is implausibly far above the market.
    Uses AND logic for ratio+gap: BOTH must exceed thresholds to drop.
    This prevents false positives on longshots where ratio is high but the
    absolute gap (and thus real risk) is still small.
    """
    market_impl = 1.0 / team_odds
    ratio = ml_prob / market_impl if market_impl > 0 else 999
    gap   = ml_prob - market_impl
    if ratio >= _SANITY_RATIO and gap >= _SANITY_ABS_GAP:
        logger.warning(
            "SANITY DROP [%s]: ML=%.0f%% vs market=%.0f%% (odds=%.2f, ratio=%.1fx, gap=%.0f%%) — dropping",
            team, ml_prob * 100, market_impl * 100, team_odds, ratio, gap * 100,
        )
        return False
    return True


def _edge_outlier_decision(
    *,
    sport: str,
    edge: float,
    true_prob: float,
    market_prob: float,
) -> tuple[str, str]:
    cap = _SANITY_EV_CAP.get(sport, _SANITY_EV_CAP_DEFAULT)
    if edge <= cap:
        return "publish", ""

    gap = max(0.0, true_prob - market_prob)
    ratio = (true_prob / market_prob) if market_prob > 0 else 999.0
    extreme_cap = cap * 2.0
    moderate_cap = cap * 1.3
    if edge >= extreme_cap or gap >= (_SANITY_ABS_GAP * 1.2) or ratio >= (_SANITY_RATIO * 1.15):
        return "reject", f"edge profile looked too extreme for {sport or 'default'} market integrity"
    if edge <= moderate_cap and gap <= (_SANITY_ABS_GAP * 0.7):
        return "review", f"edge landed above the {sport or 'default'} EV cap but stayed inside the softer market-gap tolerance"
    return "reject", f"edge profile looked too extreme for {sport or 'default'} market integrity"


def _as_float(value, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _probability_factors_for_snapshot(
    sport: str,
    snapshot: Optional[pd.Series],
    *,
    home_team: str = "",
    away_team: str = "",
    selection: str = "",
    market: str = "moneyline",
) -> list[PredictionFactor]:
    if snapshot is None:
        return []

    sport = (sport or "").lower()
    selection = (selection or "").lower()
    home_like = selection in {"home", "home_or_draw", "over", "player1", home_team.lower()}
    away_like = selection in {"away", "away_or_draw", "under", "player2", away_team.lower()}
    direction = -1.0 if away_like else 1.0
    if selection == "draw":
        direction = 0.0

    factor_specs: list[tuple[str, str, str, float, float]] = []
    if sport == "soccer":
        factor_specs = [
            ("elo_diff", "team_strength", "ELO strength gap", 1 / 250.0, 0.0),
            ("dc_xg_diff", "matchup", "Dixon-Coles xG gap", 1 / 1.6, 0.0),
            ("xg_diff", "matchup", "Expected-goals proxy gap", 1 / 1.8, 0.0),
            ("form_diff", "recent_form", "Recent form gap", 1 / 0.5, 0.0),
        ]
    elif sport == "basketball":
        factor_specs = [
            ("net_rtg_diff", "team_strength", "Net-rating gap", 1 / 12.0, 0.0),
            ("srs_diff", "team_strength", "SRS gap", 1 / 10.0, 0.0),
            ("form_diff", "recent_form", "Recent form gap", 1 / 0.35, 0.0),
            ("rest_diff", "schedule", "Rest differential", 1 / 3.0, 0.0),
        ]
    elif sport == "mlb":
        factor_specs = [
            ("home_run_diff_10", "team_strength", f"{home_team or 'Home'} run-differential form", 1 / 3.5, 0.0),
            ("away_run_diff_10", "team_strength", f"{away_team or 'Away'} run-differential form", -1 / 3.5, 0.0),
            ("sp_era_diff", "lineup", "Starting-pitcher ERA gap", -1 / 1.5, 0.0),
            ("sp_whip_diff", "lineup", "Starting-pitcher WHIP gap", -1 / 0.35, 0.0),
        ]
    elif sport == "nhl":
        factor_specs = [
            ("home_xg_diff_10", "team_strength", f"{home_team or 'Home'} xG differential", 1 / 1.2, 0.0),
            ("away_xg_diff_10", "team_strength", f"{away_team or 'Away'} xG differential", -1 / 1.2, 0.0),
            ("home_goal_diff_10", "recent_form", f"{home_team or 'Home'} goal differential", 1 / 1.6, 0.0),
            ("away_goal_diff_10", "recent_form", f"{away_team or 'Away'} goal differential", -1 / 1.6, 0.0),
        ]
    elif sport == "tennis":
        factor_specs = [
            ("surface_win_diff", "matchup", "Surface win-rate edge", 1 / 0.35, 0.0),
            ("form_diff", "recent_form", "Recent form edge", 1 / 0.4, 0.0),
            ("serve_diff", "matchup", "Serve quality edge", 1 / 0.12, 0.0),
            ("h2h_p1_win_rate", "matchup", "Head-to-head edge", 1 / 0.5, -0.5),
        ]
    elif sport == "tennis_wta":
        factor_specs = [
            ("surface_win_diff", "matchup", "Surface win-rate edge", 1 / 0.4, 0.0),
            ("form_diff", "recent_form", "Recent form edge", 1 / 0.45, 0.0),
            ("serve_diff", "matchup", "Serve quality edge", 1 / 0.14, 0.0),
            ("h2h_p1_win_rate", "matchup", "Head-to-head edge", 1 / 0.55, -0.5),
        ]

    factors: list[PredictionFactor] = []
    for column, category, summary, scale, center in factor_specs:
        if column not in snapshot.index:
            continue
        raw_value = _as_float(snapshot.get(column))
        normalized = (raw_value - center) * scale
        if direction != 0.0:
            normalized *= direction
        factors.append(
            PredictionFactor(
                name=column,
                category=category,
                value=round(normalized, 3),
                summary=f"{summary}: {raw_value:+.3f}",
            )
        )
    return factors


def _context_adjustments(
    sport: str,
    selection: str,
    snapshot: Optional[pd.Series],
    context: Optional[dict],
) -> list[PredictionFactor]:
    return build_context_adjustments(sport, selection, snapshot, context)


def _availability_summary(selection: str, context: Optional[dict]) -> tuple[str, str]:
    context = context or {}
    selection = (selection or "").lower()
    if not context:
        return "", ""

    home_name = str(context.get("home_team_name", "") or "").lower()
    away_name = str(context.get("away_team_name", "") or "").lower()
    home_like = selection in {"home", "home_or_draw", "over", "player1"}
    away_like = selection in {"away", "away_or_draw", "under", "player2"}
    if not home_like and not away_like:
        if home_name and home_name in selection:
            home_like = True
        elif away_name and away_name in selection:
            away_like = True

    home_inj = int(context.get("home_injuries_count", 0) or 0)
    away_inj = int(context.get("away_injuries_count", 0) or 0)
    home_questionable = int(context.get("home_questionable_count", 0) or 0)
    away_questionable = int(context.get("away_questionable_count", 0) or 0)
    home_susp = int(context.get("home_suspensions_count", 0) or 0)
    away_susp = int(context.get("away_suspensions_count", 0) or 0)
    home_confirmed = context.get("home_starter_confirmed")
    away_confirmed = context.get("away_starter_confirmed")
    home_starter_name = str(context.get("home_starter_name", "") or "").strip()
    away_starter_name = str(context.get("away_starter_name", "") or "").strip()
    home_goalie_confirmed = context.get("home_goalie_confirmed")
    away_goalie_confirmed = context.get("away_goalie_confirmed")
    home_goalie_name = str(context.get("home_goalie_name", "") or "").strip()
    away_goalie_name = str(context.get("away_goalie_name", "") or "").strip()
    home_lineup_confirmed = int(context.get("home_lineup_confirmed", 0) or 0)
    away_lineup_confirmed = int(context.get("away_lineup_confirmed", 0) or 0)
    home_starters = int(context.get("home_likely_starters_count", 0) or 0)
    away_starters = int(context.get("away_likely_starters_count", 0) or 0)
    source = str(context.get("availability_source", "") or "")

    if home_like and (home_inj or home_susp or home_questionable):
        parts = [f"{home_inj} inj"]
        if home_susp:
            parts.append(f"{home_susp} susp")
        if home_questionable:
            parts.append(f"{home_questionable} q")
        return f"Home absences: {', '.join(parts)}", source
    if away_like and (away_inj or away_susp or away_questionable):
        parts = [f"{away_inj} inj"]
        if away_susp:
            parts.append(f"{away_susp} susp")
        if away_questionable:
            parts.append(f"{away_questionable} q")
        return f"Away absences: {', '.join(parts)}", source
    if home_like and home_confirmed is not None:
        starter_label = "Home starter confirmed" if int(home_confirmed) else "Home starter not fully confirmed"
        if home_starter_name:
            starter_label = f"{starter_label}: {home_starter_name}"
        return (
            starter_label,
            source,
        )
    if away_like and away_confirmed is not None:
        starter_label = "Away starter confirmed" if int(away_confirmed) else "Away starter not fully confirmed"
        if away_starter_name:
            starter_label = f"{starter_label}: {away_starter_name}"
        return (
            starter_label,
            source,
        )
    if home_like and home_goalie_confirmed is not None:
        goalie_label = "Home goalie confirmed" if int(home_goalie_confirmed) else "Home goalie not confirmed"
        if home_goalie_name:
            goalie_label = f"{goalie_label}: {home_goalie_name}"
        return (
            goalie_label,
            source,
        )
    if away_like and away_goalie_confirmed is not None:
        goalie_label = "Away goalie confirmed" if int(away_goalie_confirmed) else "Away goalie not confirmed"
        if away_goalie_name:
            goalie_label = f"{goalie_label}: {away_goalie_name}"
        return (
            goalie_label,
            source,
        )
    if home_like and home_goalie_name:
        return f"Home probable goalie: {home_goalie_name}", source
    if away_like and away_goalie_name:
        return f"Away probable goalie: {away_goalie_name}", source
    if home_like and home_starters:
        label = f"Home XI posted: {home_starters} starters" if home_lineup_confirmed else f"Home XI partial: {home_starters} listed"
        return label, source
    if away_like and away_starters:
        label = f"Away XI posted: {away_starters} starters" if away_lineup_confirmed else f"Away XI partial: {away_starters} listed"
        return label, source
    if (home_inj or away_inj or home_susp or away_susp or home_questionable or away_questionable):
        return (
            f"Availability context: H {home_inj}/{home_susp}/{home_questionable} · "
            f"A {away_inj}/{away_susp}/{away_questionable}",
            source,
        )
    return "", source


def _json_safe_context(value):
    """Convert nested scraper context into a JSON-safe structure."""
    if isinstance(value, dict):
        return {str(k): _json_safe_context(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_context(item) for item in value if item is not None]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)


def _refresh_tennis_matchup_snapshot(
    snapshot: pd.Series,
    *,
    p1_profile: Optional[dict] = None,
    p2_profile: Optional[dict] = None,
    h2h: Optional[dict] = None,
) -> pd.Series:
    """
    Recompute matchup-difference features from the two current player profiles.

    Tennis snapshots are built by averaging each player's recent rows. That is
    fine for player-specific fields like ``p1_form`` or ``roll_p1_ace_rate``,
    but matchup-difference columns such as ``form_diff`` or ``surface_win_diff``
    become stale if we simply inherit them from one player's averaged history.
    """
    refreshed = snapshot.copy()

    diff_specs = [
        ("surface_win_diff", "p1_surface_win", "p2_surface_win"),
        ("form_diff", "p1_form", "p2_form"),
        ("form_quality_diff", "p1_form_quality", "p2_form_quality"),
        ("serve_diff", "roll_p1_ace_rate", "roll_p2_ace_rate"),
        ("bp_save_diff", "roll_p1_bp_save", "roll_p2_bp_save"),
        ("return_pressure_diff", "roll_p1_return_pressure", "roll_p2_return_pressure"),
        ("break_conv_diff", "roll_p1_break_conv", "roll_p2_break_conv"),
    ]
    for diff_col, p1_col, p2_col in diff_specs:
        if diff_col in refreshed.index and p1_col in refreshed.index and p2_col in refreshed.index:
            refreshed[diff_col] = _as_float(refreshed.get(p1_col)) - _as_float(refreshed.get(p2_col))
    if "serve_balance_diff" in refreshed.index:
        refreshed["serve_balance_diff"] = (
            _as_float(refreshed.get("serve_diff"), 0.0)
            + (0.6 * _as_float(refreshed.get("bp_save_diff"), 0.0))
            - (0.35 * _as_float(refreshed.get("return_pressure_diff"), 0.0))
        )
    if "load_diff" in refreshed.index:
        load_pairs = [
            ("p1_recent_load", "p2_recent_load"),
            ("p1_load", "p2_load"),
        ]
        for p1_col, p2_col in load_pairs:
            if p1_col in refreshed.index and p2_col in refreshed.index:
                refreshed["load_diff"] = _as_float(refreshed.get(p1_col)) - _as_float(refreshed.get(p2_col))
                break

    p1_profile = p1_profile or {}
    p2_profile = p2_profile or {}
    p1_rank = p1_profile.get("rank")
    p2_rank = p2_profile.get("rank")
    if "rank_diff" in refreshed.index and p1_rank and p2_rank:
        p1_rank = max(float(p1_rank), 1.0)
        p2_rank = max(float(p2_rank), 1.0)
        refreshed["rank_diff"] = p2_rank - p1_rank
        if "rank_log_ratio" in refreshed.index:
            refreshed["rank_log_ratio"] = float(np.log(p2_rank / p1_rank))

    p1_rank_pts = p1_profile.get("rank_pts")
    p2_rank_pts = p2_profile.get("rank_pts")
    if "rank_pts_log_ratio" in refreshed.index and p1_rank_pts and p2_rank_pts:
        p1_rank_pts = max(float(p1_rank_pts), 1.0)
        p2_rank_pts = max(float(p2_rank_pts), 1.0)
        refreshed["rank_pts_log_ratio"] = float(np.log(p1_rank_pts / p2_rank_pts))

    if "age_diff" in refreshed.index and p1_profile.get("age") is not None and p2_profile.get("age") is not None:
        refreshed["age_diff"] = float(p1_profile["age"]) - float(p2_profile["age"])
    if "height_diff" in refreshed.index and p1_profile.get("height") is not None and p2_profile.get("height") is not None:
        refreshed["height_diff"] = float(p1_profile["height"]) - float(p2_profile["height"])
    if "seed_advantage" in refreshed.index and p1_profile.get("seeded") is not None and p2_profile.get("seeded") is not None:
        refreshed["seed_advantage"] = float(bool(p1_profile["seeded"])) - float(bool(p2_profile["seeded"]))

    h2h = h2h or {}
    if "h2h_p1_wins" in refreshed.index and h2h.get("p1_wins") is not None:
        refreshed["h2h_p1_wins"] = float(h2h["p1_wins"])
    if "h2h_total" in refreshed.index and h2h.get("total") is not None:
        refreshed["h2h_total"] = float(h2h["total"])
    if "h2h_p1_win_rate" in refreshed.index and h2h.get("total"):
        refreshed["h2h_p1_win_rate"] = float(h2h["p1_wins"]) / float(h2h["total"])

    return refreshed


def _scraped_context_sources(context: Optional[dict]) -> list[str]:
    context = context or {}
    sources: list[str] = []
    for key in ("availability_source", "lineup_source"):
        raw = str(context.get(key, "") or "").strip()
        if raw and raw not in sources:
            sources.append(raw)
    return sources


def _scraped_context_highlights(
    selection: str,
    context: Optional[dict],
    adjustments: Optional[list[PredictionFactor]] = None,
    availability_summary: str = "",
    limit: int = 6,
) -> list[str]:
    context = context or {}
    highlights: list[str] = []

    def _push(text: str) -> None:
        text = str(text or "").strip()
        if text and text not in highlights and len(highlights) < limit:
            highlights.append(text)

    if availability_summary:
        _push(availability_summary)

    home_name = str(context.get("home_team_name", "") or "")
    away_name = str(context.get("away_team_name", "") or "")
    selection_lower = str(selection or "").lower()
    home_like = selection_lower in {"home", "home_or_draw", "over", "player1"} or (home_name and home_name.lower() in selection_lower)
    away_like = selection_lower in {"away", "away_or_draw", "under", "player2"} or (away_name and away_name.lower() in selection_lower)

    def _side(side: str) -> str:
        return "Home" if side == "home" else "Away"

    focus_side = "home" if home_like and not away_like else "away" if away_like and not home_like else ""
    sides = [focus_side] if focus_side else ["home", "away"]

    for side in sides:
        injuries = int(context.get(f"{side}_injuries_count", 0) or 0)
        suspensions = int(context.get(f"{side}_suspensions_count", 0) or 0)
        questionable = int(context.get(f"{side}_questionable_count", 0) or 0)
        priority_absences = int(context.get(f"{side}_priority_absences_count", 0) or 0)
        spine_absences = int(context.get(f"{side}_spine_absences_count", 0) or 0)
        if injuries or suspensions or questionable:
            detail = f"{_side(side)} availability: {injuries} inj"
            if suspensions:
                detail += f", {suspensions} susp"
            if questionable:
                detail += f", {questionable} q"
            if priority_absences:
                detail += f", {priority_absences} priority"
            if spine_absences:
                detail += f", {spine_absences} spine"
            _push(detail)

        starter_confirmed = context.get(f"{side}_starter_confirmed")
        starter_name = str(context.get(f"{side}_starter_name", "") or "").strip()
        if starter_confirmed is not None:
            status = "confirmed" if bool(starter_confirmed) else "unconfirmed"
            _push(f"{_side(side)} starter {status}" + (f": {starter_name}" if starter_name else ""))

        goalie_confirmed = context.get(f"{side}_goalie_confirmed")
        goalie_name = str(context.get(f"{side}_goalie_name", "") or "").strip()
        if goalie_confirmed is not None or goalie_name:
            status = "confirmed" if bool(goalie_confirmed) else "unconfirmed"
            _push(f"{_side(side)} goalie {status}" + (f": {goalie_name}" if goalie_name else ""))

        lineup_confirmed = context.get(f"{side}_lineup_confirmed")
        likely_starters = int(context.get(f"{side}_likely_starters_count", 0) or 0)
        if lineup_confirmed is not None or likely_starters:
            status = "posted" if bool(lineup_confirmed) else "not posted"
            suffix = f" ({likely_starters} likely starters)" if likely_starters else ""
            _push(f"{_side(side)} lineup {status}{suffix}")

    weather_risk = int(context.get("weather_risk", 0) or 0)
    if weather_risk:
        wind_mph = float(context.get("wind_mph", 0) or 0)
        precip_mm = float(context.get("precip_mm", 0) or 0)
        _push(f"Weather risk flagged: wind {wind_mph:.0f} mph, precip {precip_mm:.1f} mm")

    rest_advantage = int(context.get("rest_advantage", 0) or 0)
    if rest_advantage:
        _push(f"Rest advantage context: {rest_advantage:+d}")
    if context.get("is_playoff"):
        _push("Playoff or high-leverage spot detected")
    if context.get("final_day_volatility"):
        _push("Final-day / end-of-season volatility flagged")
    if context.get("fixture_congestion_risk"):
        _push("Compressed fixture turnaround flagged")
    if context.get("cup_rotation_risk") or context.get("european_rotation_risk"):
        _push("Rotation risk elevated by cup / continental context")

    for factor in adjustments or []:
        summary = str(getattr(factor, "summary", "") or "").strip()
        if summary:
            _push(summary)

    return highlights


def build_value_bet(
    team: str,
    ml_prob: float,
    team_odds: float,
    fair_prob: float,
    min_edge: float = 0.05,
    stale_line: bool = False,
    median_price: Optional[float] = None,
    sport: str = "",
    market: str = "moneyline",
    raw_model_prob: Optional[float] = None,
    prediction_factors: Optional[list[PredictionFactor]] = None,
    context_adjustments: Optional[list[PredictionFactor]] = None,
    availability_context: Optional[dict] = None,
) -> Optional[dict]:
    """
    edge = true expected value = (ml_prob × odds) − 1
    e.g. 55% chance at 2.10 odds → EV = (0.55 × 2.10) − 1 = +15.5%
    min_edge=0.05 means we require at least +5% true EV to qualify.
    """
    estimate = estimate_true_probability(
        sport=sport or "generic",
        market=market,
        selection=team,
        base_prob=raw_model_prob if raw_model_prob is not None else ml_prob,
        factors=prediction_factors or [],
        adjustments=context_adjustments or [],
        confidence=0.64,
    )
    pricing = build_pricing_decision(
        true_prob=estimate.adjusted_prob,
        offered_odds=team_odds,
        fair_prob=fair_prob,
        min_edge=min_edge,
        lower_bound_prob=getattr(estimate, "confidence_low", estimate.adjusted_prob),
    )

    # True expected-value edge
    edge = pricing.edge
    if edge < min_edge:
        return None
    if not getattr(pricing, "lower_bound_passed", True):
        return None
    # Circuit breaker: edge exceeds sport-specific cap → DATA_ANOMALY, never place
    outlier_status, outlier_reason = _edge_outlier_decision(
        sport=sport,
        edge=edge,
        true_prob=estimate.adjusted_prob,
        market_prob=pricing.market_prob,
    )
    _ev_cap = _SANITY_EV_CAP.get(sport, _SANITY_EV_CAP_DEFAULT)
    if outlier_status == "reject":
        logger.warning(
            "DATA_ANOMALY [%s]: EV=+%.0f%% exceeds %s cap of %.0f%% — rejected",
            team, edge * 100, sport or "default", _ev_cap * 100,
        )
        return None
    if not _sanity_check(team, estimate.adjusted_prob, team_odds, sport=sport):
        return None

    # Kelly sizing — always use the ACTUAL odds we'd take, never a proxy.
    # If the line is stale (one outlier book far above median), skip the bet
    # entirely rather than sizing for a median price we can't actually get.
    # Betting at median when best-price was stale would mean we're accepting
    # worse odds than our Kelly was sized for — guaranteed -EV adjustment.
    if stale_line:
        logger.info("STALE SKIP [%s]: best odds %.2f flagged as stale (median=%.2f) — skipping",
                    team, team_odds, median_price or 0)
        return None

    stake = KELLY.calculate(estimate.adjusted_prob, team_odds)
    min_acceptable_odds = getattr(pricing, "minimum_acceptable_odds", None)
    if not min_acceptable_odds:
        min_acceptable_odds = round((1.0 + min_edge) / estimate.adjusted_prob, 3) if estimate.adjusted_prob > 0 else None
    confidence_low = round(getattr(estimate, "confidence_low", estimate.adjusted_prob), 4)
    confidence_high = round(getattr(estimate, "confidence_high", estimate.adjusted_prob), 4)
    lower_bound_edge = round(getattr(pricing, "lower_bound_edge", edge), 4)
    lower_bound_passed = bool(getattr(pricing, "lower_bound_passed", True))

    flagged = edge > _review_edge_threshold(sport, market)
    availability_note, availability_source = _availability_summary(team, availability_context)
    scraped_context = _json_safe_context(availability_context or {})
    scraped_context_highlights = _scraped_context_highlights(
        team,
        availability_context,
        estimate.adjustments,
        availability_summary=availability_note,
    )
    return {
        "team": team,
        "market": market,
        "ml_prob": round(estimate.adjusted_prob, 4),
        "model_prob_raw": round(estimate.base_prob, 4),
        "fair_prob": round(fair_prob, 4),
        "odds": round(team_odds, 3),
        "edge": round(edge, 4),          # true EV: (true_prob × odds) − 1
        "market_implied_prob": pricing.market_prob,
        "vig_free_implied_prob": getattr(pricing, "vig_free_implied_prob", round(fair_prob, 4)),
        "fair_odds": pricing.fair_odds,
        "minimum_acceptable_odds": min_acceptable_odds,
        "confidence_range_low": confidence_low,
        "confidence_range_high": confidence_high,
        "confidence_range": [confidence_low, confidence_high],
        "lower_bound_prob": round(getattr(pricing, "lower_bound_prob", confidence_low), 4),
        "lower_bound_edge": lower_bound_edge,
        "lower_bound_passed": lower_bound_passed,
        "odds_recheck_status": "pending",
        "odds_recheck_odds": round(team_odds, 3),
        "odds_recheck_delta": round(team_odds - min_acceptable_odds, 3) if min_acceptable_odds else None,
        "kelly_stake_pct": round(stake * 100, 2),
        "flagged": flagged,
        "edge_outlier_review": outlier_status == "review",
        "edge_outlier_reason": outlier_reason if outlier_status == "review" else "",
        "stale_line": stale_line,
        "true_probability": estimate.to_dict(),
        "prediction_factors": [factor.to_dict() for factor in estimate.factors],
        "context_adjustments": [factor.to_dict() for factor in estimate.adjustments],
        "scraped_context": scraped_context,
        "scraped_context_highlights": scraped_context_highlights,
        "scraped_context_sources": _scraped_context_sources(availability_context),
        "availability_summary": availability_note,
        "availability_source": availability_source,
    }


def build_refund_value_bet(
    team: str,
    win_prob: float,
    refund_prob: float,
    team_odds: float,
    fair_prob: float,
    min_edge: float = 0.05,
    sport: str = "",
    market: str = "draw_no_bet",
    synthetic_line: bool = False,
    raw_model_prob: Optional[float] = None,
    prediction_factors: Optional[list[PredictionFactor]] = None,
    context_adjustments: Optional[list[PredictionFactor]] = None,
    availability_context: Optional[dict] = None,
) -> Optional[dict]:
    """
    Build a value bet for refund markets such as draw-no-bet.

    EV = (win_prob * odds) + refund_prob - 1
    """
    if team_odds <= 1.0:
        return None

    estimate = estimate_true_probability(
        sport=sport or "generic",
        market=market,
        selection=team,
        base_prob=raw_model_prob if raw_model_prob is not None else win_prob,
        factors=prediction_factors or [],
        adjustments=context_adjustments or [],
        confidence=0.62,
    )
    resolved_prob = max(1e-9, 1.0 - refund_prob)
    conditional_prob = estimate.adjusted_prob / resolved_prob
    edge = (estimate.adjusted_prob * team_odds) + refund_prob - 1.0
    if edge < min_edge:
        return None
    confidence_low = round(getattr(estimate, "confidence_low", estimate.adjusted_prob), 4)
    confidence_high = round(getattr(estimate, "confidence_high", estimate.adjusted_prob), 4)
    lower_bound_prob = confidence_low
    lower_bound_edge = (lower_bound_prob * team_odds) + refund_prob - 1.0
    lower_bound_passed = lower_bound_edge > 0
    if not lower_bound_passed:
        return None

    market_prob = round(1.0 / team_odds, 4)
    outlier_status, outlier_reason = _edge_outlier_decision(
        sport=sport,
        edge=edge,
        true_prob=estimate.adjusted_prob,
        market_prob=market_prob,
    )
    _ev_cap = _SANITY_EV_CAP.get(sport, _SANITY_EV_CAP_DEFAULT)
    if outlier_status == "reject":
        logger.warning(
            "DATA_ANOMALY [%s]: refund-market EV=+%.0f%% exceeds %s cap of %.0f%% — rejected",
            team, edge * 100, sport or "default", _ev_cap * 100,
        )
        return None

    if not _sanity_check(team, conditional_prob, team_odds, sport=sport):
        return None

    # Temper Kelly by the probability that the market actually resolves.
    conditional_stake = KELLY.calculate(conditional_prob, team_odds)
    stake = conditional_stake * resolved_prob
    min_acceptable_odds = round(max(1.01, (1.0 + min_edge - refund_prob) / estimate.adjusted_prob), 3) if estimate.adjusted_prob > 0 else None

    availability_note, availability_source = _availability_summary(team, availability_context)
    scraped_context = _json_safe_context(availability_context or {})
    scraped_context_highlights = _scraped_context_highlights(
        team,
        availability_context,
        estimate.adjustments,
        availability_summary=availability_note,
    )
    return {
        "team": team,
        "market": market,
        "ml_prob": round(conditional_prob, 4),
        "model_prob_raw": round(estimate.base_prob, 4),
        "fair_prob": round(fair_prob, 4),
        "odds": round(team_odds, 3),
        "edge": round(edge, 4),
        "market_implied_prob": market_prob,
        "vig_free_implied_prob": round(fair_prob, 4),
        "fair_odds": round((1.0 / estimate.adjusted_prob), 3) if estimate.adjusted_prob > 0 else None,
        "minimum_acceptable_odds": min_acceptable_odds,
        "confidence_range_low": confidence_low,
        "confidence_range_high": confidence_high,
        "confidence_range": [confidence_low, confidence_high],
        "lower_bound_prob": lower_bound_prob,
        "lower_bound_edge": round(lower_bound_edge, 4),
        "lower_bound_passed": lower_bound_passed,
        "odds_recheck_status": "pending",
        "odds_recheck_odds": round(team_odds, 3),
        "odds_recheck_delta": round(team_odds - min_acceptable_odds, 3) if min_acceptable_odds else None,
        "kelly_stake_pct": round(stake * 100, 2),
        "flagged": edge > _review_edge_threshold(
            sport,
            market,
            synthetic_line=synthetic_line,
            bookmaker="synthetic_1x2" if synthetic_line else "",
        ),
        "edge_outlier_review": outlier_status == "review",
        "edge_outlier_reason": outlier_reason if outlier_status == "review" else "",
        "push_prob": round(refund_prob, 4),
        "synthetic_line": synthetic_line,
        "true_probability": estimate.to_dict(),
        "prediction_factors": [factor.to_dict() for factor in estimate.factors],
        "context_adjustments": [factor.to_dict() for factor in estimate.adjustments],
        "scraped_context": scraped_context,
        "scraped_context_highlights": scraped_context_highlights,
        "scraped_context_sources": _scraped_context_sources(availability_context),
        "availability_summary": availability_note,
        "availability_source": availability_source,
    }


def _bet_game_key(bet: dict) -> tuple[str, str, str, str]:
    return (
        str(bet.get("sport", "")),
        str(bet.get("window", "")),
        str(bet.get("home", "")),
        str(bet.get("away", "")),
    )


def _bet_exposure_family(bet: dict) -> str:
    market = str(bet.get("market", "moneyline"))
    if market in {"moneyline", "spreads", "double_chance", "draw_no_bet"}:
        return "side"
    if market in {"totals", "team_total", "btts"}:
        return "total"
    return "other"


def _selection_side(bet: dict) -> str:
    team = str(bet.get("team", "")).lower()
    home = str(bet.get("home", "")).lower()
    away = str(bet.get("away", "")).lower()
    if team == "draw":
        return "draw"
    if home and home in team and away and away not in team:
        return "home"
    if away and away in team and home and home not in team:
        return "away"
    return "other"


def _selection_handicap(bet: dict) -> Optional[float]:
    team = str(bet.get("team", "")).strip()
    if not team:
        return None
    token = team.split()[-1]
    if not token or token[0] not in {"+", "-"}:
        return None
    try:
        return float(token)
    except ValueError:
        return None


def _passes_market_integrity(bet: dict, moneyline_lookup: dict[tuple[tuple[str, str, str, str], str], float]) -> tuple[bool, str]:
    market = str(bet.get("market", "moneyline"))
    if market != "spreads":
        return True, ""

    handicap = _selection_handicap(bet)
    side = _selection_side(bet)
    if handicap is None or side not in {"home", "away"}:
        return True, ""

    ml_odds = moneyline_lookup.get((_bet_game_key(bet), side))
    if not ml_odds or ml_odds <= 1.0:
        return True, ""

    spread_odds = float(bet.get("odds", 0) or 0)
    if spread_odds <= 1.0:
        return False, "spread price is invalid"

    if handicap < 0 and spread_odds <= ml_odds:
        return False, "minus-handicap price is shorter than the same side moneyline"
    if handicap > 0 and spread_odds >= ml_odds:
        return False, "plus-handicap price is longer than the same side moneyline"
    return True, ""


def _production_sort_key(bet: dict) -> tuple[float, float, float]:
    return (
        float(bet.get("market_priority_score", 0)),
        float(bet.get("edge", 0.0)),
        float(bet.get("ml_prob", 0.0)),
    )


def _apply_live_lane_governor(bet: dict) -> dict:
    sport = str(bet.get("sport", "") or "")
    market = str(bet.get("market", "moneyline") or "moneyline")
    lane = _market_health_row(sport, market)
    if not lane:
        return bet

    action = str(lane.get("action", "watch") or "watch")
    if action in {"watch", "confirmed"}:
        return {
            **bet,
            "lane_governor_action": action,
            "lane_governor_signal": str(lane.get("clv_signal", "missing") or "missing"),
            "lane_governor_bets": int(lane.get("bets", 0) or 0),
            "lane_governor_roi_pct": lane.get("roi_pct"),
            "lane_governor_avg_clv_pct": lane.get("avg_clv_pct"),
        }

    reason = (
        f"lane performance governor: {sport} {market} has "
        f"ROI {float(lane.get('roi_pct', 0.0) or 0.0):+.1f}% and "
        f"avg CLV {lane.get('avg_clv_pct') if lane.get('avg_clv_pct') is not None else 'n/a'}% "
        f"across {int(lane.get('bets', 0) or 0)} settled bets"
    )
    governed = {
        **bet,
        "lane_governor_action": action,
        "lane_governor_reason": reason,
        "lane_governor_signal": str(lane.get("clv_signal", "missing") or "missing"),
        "lane_governor_bets": int(lane.get("bets", 0) or 0),
        "lane_governor_roi_pct": lane.get("roi_pct"),
        "lane_governor_avg_clv_pct": lane.get("avg_clv_pct"),
    }

    if action == "pause":
        if str(governed.get("market_status", "") or "") == "preferred":
            governed["lane_governor_review"] = True
        else:
            governed["lane_governor_block"] = True
        return governed

    if action == "review":
        governed["lane_governor_review"] = True
        return governed

    if action == "tighten":
        current = float(governed.get("stake_multiplier", 1.0) or 1.0)
        governed["stake_multiplier"] = round(max(0.2, current * 0.5), 2)
        governed["lane_governor_tightened"] = True
        return governed

    return governed


def _passes_minimum_odds_recheck(bet: dict) -> tuple[bool, str]:
    current_odds = float(bet.get("odds", 0) or 0)
    min_odds = float(bet.get("minimum_acceptable_odds", 0) or 0)
    if current_odds <= 1.0 or min_odds <= 1.0:
        bet["odds_recheck_status"] = "unavailable"
        bet["odds_recheck_odds"] = round(current_odds, 3) if current_odds > 0 else None
        bet["odds_recheck_delta"] = None
        return True, ""

    delta = round(current_odds - min_odds, 3)
    bet["odds_recheck_odds"] = round(current_odds, 3)
    bet["odds_recheck_delta"] = delta
    if current_odds < min_odds:
        bet["odds_recheck_status"] = "failed"
        return False, "live odds dropped below the minimum acceptable price threshold"

    bet["odds_recheck_status"] = "passed"
    return True, ""


def _supported_context_referee_market(bet: dict) -> bool:
    return str(bet.get("market", "") or "") in {"moneyline", "spreads", "totals", "double_chance", "draw_no_bet"}


def _context_referee_packet(bet: dict) -> dict:
    return {
        "task": "Review this already-approved value bet and decide whether contextual news or uncertainty should approve, review, or veto it.",
        "candidate": {
            "sport": bet.get("sport"),
            "market": bet.get("market"),
            "selection": bet.get("team"),
            "home_team": bet.get("home"),
            "away_team": bet.get("away"),
            "odds": bet.get("odds"),
            "minimum_acceptable_odds": bet.get("minimum_acceptable_odds"),
            "odds_recheck_status": bet.get("odds_recheck_status"),
            "edge": bet.get("edge"),
            "bookmaker": bet.get("bookmaker"),
            "kick_off": bet.get("kick_off"),
            "league": bet.get("league"),
            "policy": bet.get("market_policy_label"),
            "policy_reason": bet.get("market_policy_reason"),
            "availability_summary": bet.get("availability_summary"),
            "review_required": bet.get("review_required"),
            "review_reason": bet.get("review_reason"),
            "context_adjustments": bet.get("context_adjustments") or [],
            "prediction_factors": bet.get("prediction_factors") or [],
            "fair_prob": bet.get("fair_prob"),
            "market_implied_prob": bet.get("market_implied_prob"),
            "fair_odds": bet.get("fair_odds"),
            "true_probability": bet.get("true_probability") or {},
        },
        "scraper_context": {
            "sources": bet.get("scraped_context_sources") or [],
            "highlights": bet.get("scraped_context_highlights") or [],
            "availability": bet.get("scraped_context") or {},
        },
        "analyst_report": {
            "warnings": list(bet.get("scraped_context_highlights") or []),
            "unknowns": [],
            "signals": [],
        },
    }


def _run_context_referee(bet: dict) -> Optional[dict]:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return None

    model = os.getenv("OPENROUTER_REASONING_MODEL", "openrouter/free").strip() or "openrouter/free"
    system_prompt = (
        "You are a sports betting context referee layered on top of a quantitative betting system. "
        "You must only analyze the supplied candidate value bet. Never suggest a different game, team, or market. "
        "Ground your reasoning in the supplied structured report and context only. "
        "Use APPROVE when the context is supportive or neutral, REVIEW when the context adds material uncertainty, "
        "and VETO only when there is clearly critical negative context such as major injuries, lineups collapsing, "
        "manager/rotation disruption, or serious fatigue/news risk. "
        "Output valid JSON with keys: decision, recommendation, reasoning, why_for, why_against, biggest_risk, "
        "stake_guidance, critical_factors, only_context_based."
    )
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5000",
            "X-Title": "SharpEdge Sports Predictor",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(_context_referee_packet(bet), ensure_ascii=True)},
            ],
            "temperature": 0.2,
            "max_tokens": 400,
        },
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    message = (((payload.get("choices") or [{}])[0]).get("message") or {}).get("content")
    message_text = str(message or "").strip()
    try:
        parsed = json.loads(message_text)
    except Exception:
        parsed = {
            "decision": "REVIEW",
            "recommendation": message_text or "No reasoning returned.",
            "reasoning": message_text or "No reasoning returned.",
            "why_for": "",
            "why_against": "",
            "biggest_risk": "",
            "stake_guidance": "",
            "critical_factors": [],
            "only_context_based": True,
        }
    parsed.setdefault("decision", "REVIEW")
    parsed.setdefault("reasoning", "")
    parsed.setdefault("critical_factors", [])
    return {
        "provider": "openrouter",
        "model": payload.get("model", model),
        "content": parsed,
        "raw": message_text,
    }


def apply_context_referee(
    published: List[dict],
    review: List[dict],
    suppressed: List[dict],
    *,
    enabled: bool = False,
    max_candidates: int = 8,
) -> tuple[List[dict], List[dict], List[dict]]:
    if not enabled:
        return published, review, suppressed

    remaining = max(0, int(max_candidates))
    if remaining == 0:
        return published, review, suppressed

    new_published: List[dict] = []
    new_review = list(review)
    new_suppressed = list(suppressed)
    ordered = sorted(published, key=_production_sort_key, reverse=True)

    for bet in ordered:
        if remaining <= 0 or not _supported_context_referee_market(bet):
            new_published.append(bet)
            continue
        remaining -= 1
        try:
            result = _run_context_referee(bet)
        except Exception as exc:
            logger.warning("Context referee failed for %s %s: %s", bet.get("sport"), bet.get("team"), exc)
            bet["context_referee_decision"] = "ERROR"
            bet["context_referee_reason"] = str(exc)
            new_published.append(bet)
            continue

        if not result:
            new_published.append(bet)
            continue

        decision = str((result.get("content") or {}).get("decision", "REVIEW")).upper()
        reasoning = str((result.get("content") or {}).get("reasoning", "")).strip()
        bet["context_referee"] = result
        bet["context_referee_decision"] = decision
        bet["context_referee_reason"] = reasoning
        if decision == "VETO":
            new_suppressed.append({**bet, "suppressed": True, "suppression_reason": "context referee vetoed the bet due to critical negative context"})
        elif decision == "REVIEW":
            new_review.append({
                **bet,
                "review_required": True,
                "review_reason": reasoning or "context referee flagged material uncertainty",
                "recommendation_type": "manual_review",
            })
        else:
            new_published.append(bet)

    for bet in new_review:
        if "context_referee_decision" not in bet:
            bet["context_referee_decision"] = bet.get("context_referee_decision", "")
            bet["context_referee_reason"] = bet.get("context_referee_reason", "")
    return new_published, new_review, new_suppressed


def _has_availability_risk(bet: dict) -> bool:
    adjustments = bet.get("context_adjustments") or []
    for item in adjustments:
        if isinstance(item, dict):
            name = str(item.get("name", "")).lower()
            category = str(item.get("category", "")).lower()
        else:
            name = str(getattr(item, "name", "")).lower()
            category = str(getattr(item, "category", "")).lower()
        if name in {"starter_uncertainty", "lineup_uncertainty", "goalie_uncertainty", "availability_uncertainty"}:
            return True
        if category == "lineup" and "uncert" in name:
            return True
    return False


def _positive_context_count(bet: dict) -> int:
    adjustments = bet.get("context_adjustments") or []
    count = 0
    for item in adjustments:
        if isinstance(item, dict):
            value = float(item.get("value", 0.0) or 0.0)
            category = str(item.get("category", "")).lower()
        else:
            value = float(getattr(item, "value", 0.0) or 0.0)
            category = str(getattr(item, "category", "")).lower()
        if category in {"matchup", "coaching", "environment", "motivation", "schedule", "lineup"} and value > 0.003:
            count += 1
    return count


def _review_reason_bucket(reason: str) -> str:
    text = str(reason or "").strip().lower()
    if not text:
        return "other"
    if "prediction quality" in text or "probability quality" in text or "model probability" in text:
        return "prediction_quality"
    if "review-only" in text or "review only" in text or "league remains review-only" in text:
        return "review_only_league"
    if "stale" in text or "price looked stale" in text:
        return "stale_price"
    if "availability" in text or "starter uncertainty" in text or "lineup" in text:
        return "availability_risk"
    if "line velocity" in text or "news-driven movement" in text:
        return "line_movement"
    if "edge exceeded" in text or "edge profile" in text or "integrity check" in text:
        return "edge_review"
    if "experimental market" in text or "replay validation" in text:
        return "experimental_market"
    if "context referee" in text or "material uncertainty" in text:
        return "context_referee"
    return "other"


def _review_reason_breakdown(review_bets: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for bet in review_bets or []:
        bucket = _review_reason_bucket(str(bet.get("review_reason", "")))
        counts[bucket] = counts.get(bucket, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _review_reason_label(bucket: str) -> str:
    labels = {
        "prediction_quality": "prediction quality",
        "review_only_league": "review-only league",
        "stale_price": "stale price",
        "availability_risk": "availability risk",
        "line_movement": "line movement",
        "edge_review": "edge review",
        "experimental_market": "experimental market",
        "context_referee": "context referee",
        "other": "other",
    }
    return labels.get(bucket, bucket.replace("_", " "))


def _extract_probability_debug(bet: dict) -> dict[str, Any]:
    for key in _PROBABILITY_DEBUG_KEYS:
        payload = bet.get(key)
        if isinstance(payload, dict) and payload:
            return payload
    return {}


def _prediction_quality_assessment(bet: dict) -> tuple[bool, str]:
    """
    Reject candidates whose probability thesis is weak before manual review.

    Manual review should resolve missing evidence, not rescue unstable or
    implausible model probabilities. This gate only acts on explicit model
    quality signals so older/minimal test fixtures without diagnostics keep
    their previous behavior.
    """
    try:
        odds = float(bet.get("odds", 0) or 0)
        ml_prob = float(bet.get("ml_prob", 0) or 0)
        edge = float(bet.get("edge", 0) or 0)
    except (TypeError, ValueError):
        return False, "prediction quality failed: model probability, odds, or edge is not numeric"

    if odds <= 1.0 or not (0.01 <= ml_prob <= 0.99):
        return False, "prediction quality failed: model probability or odds is outside a usable range"
    if edge <= 0:
        return False, "prediction quality failed: model edge is not positive"

    if bet.get("lower_bound_passed") is False:
        return False, "prediction quality failed: confidence-range lower bound does not support the pick"

    lower_bound_edge = bet.get("lower_bound_edge")
    if lower_bound_edge not in (None, ""):
        try:
            if float(lower_bound_edge) <= 0:
                return False, "prediction quality failed: lower-bound edge is not positive"
        except (TypeError, ValueError):
            return False, "prediction quality failed: lower-bound edge is not numeric"

    raw_prob = bet.get("model_prob_raw")
    if raw_prob not in (None, ""):
        try:
            raw_gap = abs(float(ml_prob) - float(raw_prob))
            if raw_gap >= 0.08:
                return False, "prediction quality failed: context-adjusted probability moved too far from the raw model"
        except (TypeError, ValueError):
            return False, "prediction quality failed: raw model probability is not numeric"

    debug = _extract_probability_debug(bet)
    disagreement = debug.get("disagreement_pp") if debug else None
    if disagreement not in (None, ""):
        try:
            if float(disagreement) >= 22.0:
                return False, "prediction quality failed: classifier and structural model disagree too much"
        except (TypeError, ValueError):
            return False, "prediction quality failed: model-disagreement diagnostic is not numeric"

    history = debug.get("history_rows") if isinstance(debug.get("history_rows"), dict) else {}
    if history:
        try:
            min_history = min(
                int(value or 0)
                for value in history.values()
                if value not in (None, "")
            )
        except ValueError:
            min_history = 0
        if min_history and min_history < 5 and edge >= 0.06:
            return False, "prediction quality failed: high-edge candidate has too little historical support"

    synthetic_history = debug.get("synthetic_history") if isinstance(debug.get("synthetic_history"), dict) else {}
    if synthetic_history and all(bool(value) for value in synthetic_history.values()) and edge >= 0.05:
        return False, "prediction quality failed: high-edge candidate is built only from synthetic history"

    return True, ""


def _stale_price_diagnostics(review_bets: list[dict]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bet in review_bets or []:
        if _review_reason_bucket(str(bet.get("review_reason", ""))) != "stale_price":
            continue
        rows.append({
            "game": f"{bet.get('home', '')} vs {bet.get('away', '')}",
            "market": str(bet.get("market", "") or ""),
            "selected_outcome": str(bet.get("selected_outcome") or bet.get("team", "") or ""),
            "odds": bet.get("odds"),
            "bookmaker": str(bet.get("selected_bookmaker") or bet.get("bookmaker", "") or ""),
            "odds_source_status": str(bet.get("odds_source_status", "") or ""),
            "odds_source_detail": str(bet.get("odds_source_detail", "") or ""),
            "odds_fetched_at": bet.get("odds_fetched_at"),
            "bookmaker_last_update": bet.get("bookmaker_last_update"),
            "cache_loaded_at": bet.get("cache_loaded_at"),
            "candidate_created_at": bet.get("candidate_created_at"),
            "stale_threshold_hours": bet.get("stale_threshold_hours", _ODDS_STALE_THRESHOLD_HOURS),
            "computed_odds_age_hours": bet.get("computed_odds_age_hours", bet.get("odds_snapshot_age_hours")),
            "force_fresh_odds_active": bool(bet.get("force_fresh_odds_active")),
            "cache_used": bool(bet.get("odds_cache_used")),
            "fallback_used": bool(bet.get("odds_fallback_used")),
            "reason_moved_to_review": str(bet.get("review_reason", "") or ""),
            "committee_final_decision": str(bet.get("committee_final_decision", "") or ""),
            "arbiter_veto_flags": list(bet.get("committee_veto_flags") or []),
        })
    return rows


def _candidate_game_key(item: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(item.get("sport", "") or "").strip().lower(),
        str(item.get("home", "") or "").strip(),
        str(item.get("away", "") or "").strip(),
        str(item.get("commence", "") or item.get("commence_time", "") or "").strip(),
    )


def _bucket_suppression_reason(reason: str) -> str:
    text = str(reason or "").strip().lower()
    if not text:
        return "other"
    if "same-game" in text or "correlation" in text:
        return "correlation_guardrail"
    if "market" in text and ("fit" in text or "suitable" in text):
        return "market_fit"
    if "league or sport is not configured" in text:
        return "unsupported_lane"
    if "stale" in text:
        return "stale_data"
    if "zero" in text and "stake" in text:
        return "stake_zero"
    if "context referee vetoed" in text:
        return "negative_context"
    if "integrity" in text or "outlier" in text:
        return "integrity_guardrail"
    return "other"


def _sport_pipeline_diagnostics(
    *,
    soccer_games: list[dict],
    other_games: list[dict],
    published_bets: list[dict],
    review_bets: list[dict],
    suppressed_bets: list[dict],
) -> dict[str, Any]:
    sports: dict[str, dict[str, Any]] = {}

    def ensure_sport(sport: str) -> dict[str, Any]:
        key = str(sport or "").strip().lower()
        return sports.setdefault(key, {
            "scanned_games": 0,
            "model_available_games": 0,
            "abstained_games": 0,
            "candidate_games": 0,
            "published_games": 0,
            "review_games": 0,
            "suppressed_games": 0,
            "no_candidate_games": 0,
            "no_candidate_reason_breakdown": {},
            "review_reason_breakdown": {},
            "suppression_reason_breakdown": {},
        })

    scanned_keys: dict[str, set[tuple[str, str, str, str]]] = {}
    model_available_keys: dict[str, set[tuple[str, str, str, str]]] = {}
    abstained_keys: dict[str, set[tuple[str, str, str, str]]] = {}

    def add_game(game: dict[str, Any]) -> None:
        sport = str(game.get("sport", "") or "").strip().lower()
        if not sport:
            return
        stats = ensure_sport(sport)
        key = _candidate_game_key(game)
        scanned_keys.setdefault(sport, set()).add(key)
        if bool(game.get("model_available", False)):
            model_available_keys.setdefault(sport, set()).add(key)
        if bool(game.get("abstain", False)):
            abstained_keys.setdefault(sport, set()).add(key)
        stats["scanned_games"] = len(scanned_keys[sport])
        stats["model_available_games"] = len(model_available_keys.get(sport, set()))
        stats["abstained_games"] = len(abstained_keys.get(sport, set()))

    for game in soccer_games or []:
        add_game(game)
    for game in other_games or []:
        add_game(game)

    published_keys: dict[str, set[tuple[str, str, str, str]]] = {}
    review_keys: dict[str, set[tuple[str, str, str, str]]] = {}
    suppressed_keys: dict[str, set[tuple[str, str, str, str]]] = {}
    candidate_keys: dict[str, set[tuple[str, str, str, str]]] = {}

    for bet in published_bets or []:
        sport = str(bet.get("sport", "") or "").strip().lower()
        if not sport:
            continue
        ensure_sport(sport)
        key = _candidate_game_key(bet)
        published_keys.setdefault(sport, set()).add(key)
        candidate_keys.setdefault(sport, set()).add(key)

    for bet in review_bets or []:
        sport = str(bet.get("sport", "") or "").strip().lower()
        if not sport:
            continue
        stats = ensure_sport(sport)
        key = _candidate_game_key(bet)
        review_keys.setdefault(sport, set()).add(key)
        candidate_keys.setdefault(sport, set()).add(key)
        bucket = _review_reason_bucket(str(bet.get("review_reason", "") or bet.get("decision_reason", "")))
        stats["review_reason_breakdown"][bucket] = int(stats["review_reason_breakdown"].get(bucket, 0)) + 1

    for bet in suppressed_bets or []:
        sport = str(bet.get("sport", "") or "").strip().lower()
        if not sport:
            continue
        stats = ensure_sport(sport)
        key = _candidate_game_key(bet)
        suppressed_keys.setdefault(sport, set()).add(key)
        candidate_keys.setdefault(sport, set()).add(key)
        bucket = _bucket_suppression_reason(str(bet.get("suppression_reason", "") or bet.get("decision_reason", "")))
        stats["suppression_reason_breakdown"][bucket] = int(stats["suppression_reason_breakdown"].get(bucket, 0)) + 1

    for sport, stats in sports.items():
        missing_candidate_keys = scanned_keys.get(sport, set()) - candidate_keys.get(sport, set())
        if missing_candidate_keys:
            for game in (soccer_games or []):
                if str(game.get("sport", "") or "").strip().lower() != sport:
                    continue
                game_key = _candidate_game_key(game)
                if game_key not in missing_candidate_keys:
                    continue
                bucket = _bucket_no_candidate_reason(game)
                stats["no_candidate_reason_breakdown"][bucket] = int(stats["no_candidate_reason_breakdown"].get(bucket, 0)) + 1
            for game in (other_games or []):
                if str(game.get("sport", "") or "").strip().lower() != sport:
                    continue
                game_key = _candidate_game_key(game)
                if game_key not in missing_candidate_keys:
                    continue
                bucket = _bucket_no_candidate_reason(game)
                stats["no_candidate_reason_breakdown"][bucket] = int(stats["no_candidate_reason_breakdown"].get(bucket, 0)) + 1
        stats["candidate_games"] = len(candidate_keys.get(sport, set()))
        stats["published_games"] = len(published_keys.get(sport, set()))
        stats["review_games"] = len(review_keys.get(sport, set()))
        stats["suppressed_games"] = len(suppressed_keys.get(sport, set()))
        stats["no_candidate_games"] = max(0, int(stats["scanned_games"]) - int(stats["candidate_games"]))

    totals = {
        "scanned_games": sum(int(item["scanned_games"]) for item in sports.values()),
        "model_available_games": sum(int(item["model_available_games"]) for item in sports.values()),
        "abstained_games": sum(int(item["abstained_games"]) for item in sports.values()),
        "candidate_games": sum(int(item["candidate_games"]) for item in sports.values()),
        "published_games": sum(int(item["published_games"]) for item in sports.values()),
        "review_games": sum(int(item["review_games"]) for item in sports.values()),
        "suppressed_games": sum(int(item["suppressed_games"]) for item in sports.values()),
        "no_candidate_games": sum(int(item["no_candidate_games"]) for item in sports.values()),
    }

    return {"by_sport": sports, "totals": totals}


def _wta_audit_reason(*, best_edge: float, min_edge: float, review_only: bool) -> str:
    if best_edge >= min_edge and review_only:
        return "review-only launch policy"
    gap_pp = max(0.0, (min_edge - best_edge) * 100.0)
    if gap_pp > 0:
        return f"below min edge by {gap_pp:.1f}pp"
    return "no publishable edge"


def _record_wta_audit_match(
    *,
    home: str,
    away: str,
    league: str,
    commence: str,
    home_odds: float,
    away_odds: float,
    home_prob: float,
    away_prob: float,
    review_only: bool,
    min_edge: float,
) -> None:
    global _wta_audit_rows

    home_edge = (home_prob * home_odds) - 1.0 if home_odds > 1.0 else -1.0
    away_edge = (away_prob * away_odds) - 1.0 if away_odds > 1.0 else -1.0
    if home_edge >= away_edge:
        best_team = home
        best_prob = home_prob
        best_odds = home_odds
        best_edge = home_edge
    else:
        best_team = away
        best_prob = away_prob
        best_odds = away_odds
        best_edge = away_edge

    _wta_audit_rows.append(
        {
            "sport": "tennis_wta",
            "league": league,
            "home": home,
            "away": away,
            "team": best_team,
            "commence": commence,
            "odds": round(float(best_odds), 3),
            "ml_prob": round(float(best_prob), 4),
            "edge": round(float(best_edge), 4),
            "edge_pct": round(float(best_edge) * 100.0, 2),
            "review_only": review_only,
            "audit_reason": _wta_audit_reason(best_edge=float(best_edge), min_edge=float(min_edge), review_only=bool(review_only)),
        }
    )


def apply_publish_guardrails(bets: List[dict], bankroll: float) -> tuple[List[dict], List[dict], List[dict]]:
    """
    Apply the final publication layer before any bet reaches the live board.

    This is where we:
      - suppress non-production market tiers,
      - block stale or abstained bets,
      - reject contradictory cross-market prices,
      - prevent correlated same-game side stacks,
      - expose audit fields for the webapp/report.
    """
    annotated = []
    for bet in bets:
        enriched = enrich_with_capability(annotate_bet(bet))
        annotated.append(_apply_live_lane_governor(enriched))
    suppressed: List[dict] = []
    review: List[dict] = []
    published: List[dict] = []

    moneyline_lookup: dict[tuple[tuple[str, str, str, str], str], float] = {}
    for bet in annotated:
        if bet.get("market") != "moneyline":
            continue
        side = _selection_side(bet)
        if side in {"home", "away"}:
            moneyline_lookup[(_bet_game_key(bet), side)] = float(bet.get("odds", 0) or 0)
    peers_by_game: dict[tuple[str, str, str, str], list[dict]] = {}
    for bet in annotated:
        peers_by_game.setdefault(_bet_game_key(bet), []).append(bet)

    kept_families: set[tuple[tuple[str, str, str, str], str]] = set()
    kept_games: set[tuple[str, str, str, str]] = set()

    for bet in sorted(annotated, key=_production_sort_key, reverse=True):
        implied_prob = 0.0
        try:
            odds = float(bet.get("odds", 0) or 0)
            if odds > 1.0:
                implied_prob = 1.0 / odds
        except Exception:
            odds = 0.0
        bet["market_implied_prob"] = round(implied_prob, 4) if implied_prob else 0.0
        bet["fair_odds"] = round(1.0 / bet["ml_prob"], 3) if bet.get("ml_prob", 0) not in {0, None} else None
        bet["edge_pct"] = round(float(bet.get("edge", 0.0)) * 100, 2)
        bet["kelly_fraction_used"] = round(KELLY.fraction, 4)
        bet["recommendation_type"] = "value_bet"
        freshness_audit = audit_candidate_freshness(bet)
        bet.update(freshness_audit.to_dict())
        bet["freshness_check"] = freshness_audit.to_dict()
        suitability = evaluate_market_suitability(bet, peers_by_game.get(_bet_game_key(bet), []))
        bet["market_suitable"] = suitability.suitable
        bet["recommended_market"] = suitability.recommended_market
        bet["market_suitability_reason"] = suitability.reason
        bet["market_suitability_score"] = round(float(suitability.score), 3)
        prediction_quality_ok, prediction_quality_reason = _prediction_quality_assessment(bet)
        bet["prediction_quality_ok"] = prediction_quality_ok
        bet["prediction_quality_reason"] = prediction_quality_reason

        suppression_reason = ""
        review_reason = ""
        if not prediction_quality_ok:
            suppression_reason = prediction_quality_reason
        elif not bet.get("prediction_focus_allowed", True):
            suppression_reason = bet.get("prediction_lane_reason") or "sport/market is outside the focused prediction lanes"
        elif freshness_audit.suppression_reason:
            suppression_reason = freshness_audit.suppression_reason
        elif freshness_audit.review_reason:
            review_reason = freshness_audit.review_reason
        elif not bet.get("scanable", True):
            suppression_reason = "league or sport is not configured for the production launch board"
        elif not bet.get("publishable", True):
            review_reason = bet.get("launch_note") or "league remains review-only until model/training support is refreshed"
        elif not bet.get("production_allowed"):
            review_reason = "experimental market is still in replay validation, so it is held in review instead of being published"
        elif bet.get("lane_governor_block"):
            suppression_reason = bet.get("lane_governor_reason") or "lane performance governor blocked auto-publication for this market"
        elif bet.get("lane_governor_review"):
            review_reason = bet.get("lane_governor_reason") or "lane performance governor requires manual review for this market"
        else:
            ok, integrity_reason = _passes_market_integrity(bet, moneyline_lookup)
            if not ok:
                suppression_reason = integrity_reason
            else:
                odds_ok, odds_reason = _passes_minimum_odds_recheck(bet)
                if not odds_ok:
                    suppression_reason = odds_reason
            if not suppression_reason and not review_reason and not suitability.suitable:
                suppression_reason = suitability.reason
            if not suppression_reason and bet.get("stale_line"):
                review_reason = "best price looked stale relative to the wider market snapshot"
            elif not suppression_reason and bet.get("abstain"):
                review_reason = "line velocity filter flagged likely news-driven movement"
            elif not suppression_reason and bet.get("edge_outlier_review"):
                review_reason = bet.get("edge_outlier_reason") or "edge profile needs a manual integrity check"
            elif not suppression_reason and bet.get("flagged"):
                context_count = _positive_context_count(bet)
                edge_value = float(bet.get("edge", 0.0) or 0.0)
                review_threshold = _review_edge_threshold(
                    str(bet.get("sport", "") or ""),
                    str(bet.get("market", "") or ""),
                    synthetic_line=bool(bet.get("synthetic_line")),
                    bookmaker=str(bet.get("bookmaker", "") or ""),
                )
                soft_cap = _SANITY_EV_CAP.get(str(bet.get("sport", "")).lower(), _SANITY_EV_CAP_DEFAULT) * 0.95
                market_status = str(bet.get("market_status", "") or "")
                market_key = str(bet.get("market", "") or "")
                synthetic_like = bool(bet.get("synthetic_line")) or str(bet.get("bookmaker", "")).startswith("synthetic")
                promotable_preferred = (
                    market_status == "preferred"
                    and market_key in {"double_chance", "totals", "set_betting", "spreads"}
                    and edge_value <= max(review_threshold, soft_cap)
                    and context_count >= 1
                    and not synthetic_like
                )
                if (
                    (context_count >= 2 or promotable_preferred)
                    and edge_value <= max(review_threshold, soft_cap)
                    and not _has_availability_risk(bet)
                ):
                    bet["flagged_promoted"] = True
                    review_reason = ""
                else:
                    review_reason = "edge exceeded the review threshold and now requires manual confirmation"
            elif not suppression_reason and _has_availability_risk(bet):
                review_reason = "availability or starter uncertainty still needs human review"

        game_key = _bet_game_key(bet)
        family = _bet_exposure_family(bet)
        if not suppression_reason and (game_key, family) in kept_families:
            suppression_reason = f"same-game {family} exposure already represented by a higher-ranked play"
        elif not suppression_reason and family == "side" and game_key in kept_games:
            suppression_reason = "same-game side correlation guardrail kept only the strongest thesis"

        if suppression_reason:
            suppressed.append({
                **bet,
                "suppressed": True,
                "suppression_reason": suppression_reason,
                "publication_outcome": "suppressed",
                "publication_reason": suppression_reason,
            })
            continue
        if review_reason:
            stake_multiplier = float(bet.get("stake_multiplier", 1.0))
            raw_kelly_pct = float(bet.get("kelly_stake_pct", 0.0) or 0.0)
            adjusted_kelly_pct = round(raw_kelly_pct * stake_multiplier, 2)
            review.append({
                **bet,
                "review_required": True,
                "review_reason": review_reason,
                "recommendation_type": "manual_review",
                "kelly_base_pct": raw_kelly_pct,
                "kelly_stake_pct": adjusted_kelly_pct,
                "stake_abs": round(bankroll * adjusted_kelly_pct / 100, 2),
                "publication_outcome": "review",
                "publication_reason": review_reason,
            })
            continue

        stake_multiplier = float(bet.get("stake_multiplier", 1.0))
        raw_kelly_pct = float(bet.get("kelly_stake_pct", 0.0) or 0.0)
        adjusted_kelly_pct = round(raw_kelly_pct * stake_multiplier, 2)
        bet["kelly_base_pct"] = raw_kelly_pct
        bet["kelly_stake_pct"] = adjusted_kelly_pct
        bet["stake_abs"] = round(bankroll * adjusted_kelly_pct / 100, 2)
        bet["publish_ready"] = True
        bet["publication_outcome"] = "published"
        bet["publication_reason"] = "All publication guardrails passed."
        published.append(bet)
        kept_families.add((game_key, family))
        if family == "side":
            kept_games.add(game_key)

    return published, review, suppressed


def _apply_decision_labels(
    published: List[dict],
    review: List[dict],
    suppressed: List[dict],
) -> tuple[List[dict], List[dict], List[dict]]:
    labelled_published: List[dict] = []
    labelled_review: List[dict] = []
    labelled_suppressed: List[dict] = []

    for bet in published:
        decision_status, decision_reason = classify_candidate_decision(
            publish_ready=True,
            review_reason=str(bet.get("review_reason", "") or ""),
            suppression_reason=str(bet.get("suppression_reason", "") or ""),
        )
        labelled_published.append({
            **bet,
            "decision_status": decision_status,
            "decision_reason": decision_reason,
        })

    for bet in review:
        decision_status, decision_reason = classify_candidate_decision(
            publish_ready=False,
            review_reason=str(bet.get("review_reason", "") or ""),
            suppression_reason="",
        )
        labelled_review.append({
            **bet,
            "decision_status": decision_status,
            "decision_reason": decision_reason,
        })

    for bet in suppressed:
        decision_status, decision_reason = classify_candidate_decision(
            publish_ready=False,
            review_reason="",
            suppression_reason=str(bet.get("suppression_reason", "") or ""),
        )
        labelled_suppressed.append({
            **bet,
            "decision_status": decision_status,
            "decision_reason": decision_reason,
        })

    return labelled_published, labelled_review, labelled_suppressed


def to_parlay_leg(
    sport: str,
    home_team: str,
    away_team: str,
    team: str,
    odds: float,
    ml_prob: float,
    fair_prob: float,
    commence: str,
    window: str = "today",
    market: str = "",
) -> Optional[ParlayLeg]:
    """
    Convert a prediction into a ParlayLeg.
    Returns None if the sanity check flags an anomalous model/market gap.
    window: 'today' = before midnight Vienna, 'overnight' = 00:00–07:00.
    """
    if not _sanity_check(team, ml_prob, odds):
        return None
    edge = ml_prob - fair_prob
    return ParlayLeg(
        sport=sport,
        match_id=f"{home_team} vs {away_team}",
        team=team,
        odds=odds,
        ml_prob=ml_prob,
        fair_prob=fair_prob,
        edge=edge,
        commence=commence,
        window=window,
        market=market,
        home_team=home_team,
        away_team=away_team,
    )


def _committee_publish_note(bet: dict) -> str:
    final_decision = str(bet.get("committee_final_decision", "") or "")
    if final_decision == "BET_SUBSTITUTE":
        return "Published via approved substitute."
    if final_decision == "BET":
        return "Committee approved for publication."
    return ""


def _committee_to_parlay_leg(bet: dict) -> Optional[ParlayLeg]:
    final_decision = str(bet.get("committee_final_decision", "") or "")
    if final_decision not in {"BET", "BET_SUBSTITUTE"}:
        return None
    if final_decision == "BET_SUBSTITUTE" and not bool(bet.get("published_from_substitute")):
        return None
    try:
        odds = float(bet.get("odds") or 0.0)
        ml_prob = float(bet.get("ml_prob") or 0.0)
        fair_prob = float(bet.get("fair_prob") or 0.0)
    except Exception:
        return None
    return to_parlay_leg(
        sport=str(bet.get("sport", "") or ""),
        home_team=str(bet.get("home", "") or ""),
        away_team=str(bet.get("away", "") or ""),
        team=str(bet.get("team", "") or ""),
        odds=odds,
        ml_prob=ml_prob,
        fair_prob=fair_prob,
        commence=str(bet.get("commence") or bet.get("commence_time") or ""),
        window=str(bet.get("window", "today") or "today"),
        market=str(bet.get("market", "") or ""),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Feature cache helpers
# ─────────────────────────────────────────────────────────────────────────────

_FEATURE_CACHE_TTL_HOURS = 12   # rebuild features if cache is older than this
_FEATURE_CACHE_VERSIONS = {
    "soccer": 3,
    "basketball": 2,
    "tennis": 4,
    "tennis_wta": 4,
    "mlb": 2,
    "nhl": 2,
}


def _feature_cache_meta_path(cache: Path) -> Path:
    return cache.with_suffix(cache.suffix + ".meta.json")


def _feature_cache_scope_signature(sport: str) -> dict:
    sport_cfg = settings.get("sports", {}).get(sport, {})
    apis_cfg = settings.get("apis", {})
    signature = {
        "sport": sport,
        "seasons_to_fetch": sport_cfg.get("seasons_to_fetch"),
        "feature_version": _FEATURE_CACHE_VERSIONS.get(sport, 1),
    }
    if sport == "soccer":
        signature["competitions"] = list(apis_cfg.get("football_data", {}).get("competitions", []))
        signature["api_sports_competitions"] = [
            {
                "key": item.get("key"),
                "name": item.get("name"),
                "country": item.get("country"),
                "season_mode": item.get("season_mode"),
                "league_id": item.get("league_id"),
                "season_years": item.get("season_years"),
            }
            for item in sport_cfg.get("api_sports_competitions", [])
        ]
    if sport == "tennis":
        signature["tours"] = list(sport_cfg.get("tours", []))
    return signature


def _feature_cache_matches_scope(cache: Path, sport: str) -> bool:
    meta_path = _feature_cache_meta_path(cache)
    if not meta_path.exists():
        return False
    try:
        saved = json.loads(meta_path.read_text())
    except Exception:
        return False
    return saved == _feature_cache_scope_signature(sport)


def _write_feature_cache_scope(cache: Path, sport: str) -> None:
    meta_path = _feature_cache_meta_path(cache)
    try:
        meta_path.write_text(json.dumps(_feature_cache_scope_signature(sport), indent=2))
    except Exception as exc:
        logger.warning("[%s] Could not write feature cache metadata: %s", sport, exc)


def _resolve_soccer_team_name(
    odds_name: str,
    training_teams: set[str],
    norm_to_canonical: dict[str, str],
    soccer_resolver: TeamResolver,
    soccer_aliases: Optional[dict[str, str]] = None,
) -> str:
    """Resolve an odds-feed soccer club name to the best training-data name."""
    soccer_aliases = soccer_aliases or _SOCCER_ALIAS_OVERRIDES

    # 1. Manual alias table wins, including intentional identity mappings.
    if odds_name in soccer_aliases:
        return soccer_aliases[odds_name]

    # 2. Exact match in training data should beat any global fuzzy resolver.
    if odds_name in training_teams:
        return odds_name

    # 3. Local normalized alias map built from the actual training clubs.
    norm = TeamResolver._normalize(odds_name)
    if norm in norm_to_canonical:
        return norm_to_canonical[norm]

    # 4. Local token-overlap fuzzy match against the actual training clubs.
    resolved = resolve_canonical_name(odds_name, training_teams, alias_map=norm_to_canonical)
    if resolved != odds_name:
        return resolved

    # 5. Shared resolver fallback, but only if it resolves to a real training club.
    resolved = soccer_resolver.resolve(odds_name)
    if resolved != odds_name and resolved in training_teams:
        return resolved

    # Fallback: return original (will fail the history guard cleanly).
    return odds_name


def _sanitize_features_for_parquet(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Make feature caches parquet-safe.

    Some feature frames retain source columns with mixed object values
    (for example numeric seeds plus nulls plus stray strings). PyArrow rejects
    these mixed object columns when writing parquet. We preserve numeric-like
    columns as numeric and coerce the remaining object columns to pandas'
    string dtype so cache writes are stable.
    """
    if features_df.empty:
        return features_df

    sanitized = features_df.copy()
    object_cols = list(sanitized.select_dtypes(include=["object", "string"]).columns)
    for col in object_cols:
        series = sanitized[col]
        non_null = series.dropna()
        if non_null.empty:
            continue
        numeric = pd.to_numeric(series, errors="coerce")
        if int(numeric.notna().sum()) == int(non_null.shape[0]):
            sanitized[col] = numeric
        else:
            sanitized[col] = series.astype("string")
    return sanitized


def _load_features_cached(
    sport: str,
    cache_path: str,
    fetcher_factory,
    engineer_factory,
    label_map_attr: str = "label_map",
    ttl_hours: float = _FEATURE_CACHE_TTL_HOURS,
    allow_live_refresh: bool = True,
):
    """
    Load feature-engineered DataFrame from parquet cache if fresh,
    otherwise re-fetch raw data and re-engineer, then update cache.

    Returns (features_df, engineer_instance).
    engineer_instance.label_map is populated after this call.
    """
    global _feature_cache_meta
    from datetime import timedelta
    cache = Path(cache_path)

    is_fresh = (
        cache.exists()
        and (datetime.now() - datetime.fromtimestamp(cache.stat().st_mtime))
        < timedelta(hours=ttl_hours)
    )
    scope_matches = _feature_cache_matches_scope(cache, sport)

    def _load_cached_frame(source_status: str, log_label: str):
        logger.info("[%s] %s (%s)", sport, log_label, cache_path)
        df = pd.read_parquet(cache)
        age_hours = (datetime.now() - datetime.fromtimestamp(cache.stat().st_mtime)).total_seconds() / 3600.0
        _feature_cache_meta[sport] = {
            "fetched_at": datetime.fromtimestamp(cache.stat().st_mtime, tz=timezone.utc).isoformat(),
            "age_hours": round(age_hours, 3),
            "source_status": source_status,
        }
        engineer = engineer_factory()
        # Reconstruct label_map from the target column if present
        # Only call encode_target if the engineer supports it (soccer does, tennis doesn't)
        if "target" in df.columns and "result" in df.columns:
            if hasattr(engineer, "encode_target"):
                _, engineer.label_map = engineer.encode_target(df.copy(), target_col="result")
            elif not hasattr(engineer, "label_map") or not engineer.label_map:
                # For binary sports (tennis) infer label_map from numeric target values
                # Tennis: 0 = player2_win, 1 = player1_win
                unique_targets = sorted(df["target"].dropna().unique().astype(int))
                unique_results = df["result"].dropna().unique()
                # Build map: numeric → result string
                result_map = {}
                for val in unique_targets:
                    matching = [r for r in unique_results if str(val) in str(r)]
                    if matching:
                        result_map[val] = matching[0]
                engineer.label_map = result_map if result_map else {0: "player2_win", 1: "player1_win"}
        return df, engineer

    if is_fresh and scope_matches:
        return _load_cached_frame("feature_cache", "Loading features from cache")

    if cache.exists() and scope_matches and not allow_live_refresh:
        return _load_cached_frame("stale_feature_cache", "Loading stale feature cache without live refresh")

    if not allow_live_refresh and not cache.exists():
        fetcher = fetcher_factory()
        raw_snapshot = getattr(fetcher, "_raw_dir", None)
        raw_path = Path(raw_snapshot) / f"{sport}_all_seasons.parquet" if raw_snapshot else None
        if raw_path and raw_path.exists():
            logger.info("[%s] Feature cache missing — rebuilding from local raw snapshot (%s)", sport, raw_path)
            raw_df = pd.read_parquet(raw_path)
            engineer = engineer_factory()
            features_df = engineer.engineer_features(raw_df)
            cache.parent.mkdir(parents=True, exist_ok=True)
            parquet_df = _sanitize_features_for_parquet(features_df)
            parquet_df.to_parquet(cache)
            _write_feature_cache_scope(cache, sport)
            _feature_cache_meta[sport] = {
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "age_hours": 0.0,
                "source_status": "rebuilt_from_local_raw",
            }
            return features_df, engineer

        logger.warning("[%s] Feature cache missing and no local raw snapshot was found. Run refresh_feature_cache.py first.", sport)
        _feature_cache_meta[sport] = {
            "fetched_at": "",
            "age_hours": None,
            "source_status": "missing_feature_cache",
        }
        engineer = engineer_factory()
        return pd.DataFrame(), engineer

    if cache.exists() and is_fresh and not scope_matches:
        logger.info("[%s] Feature cache scope changed — rebuilding %s", sport, cache_path)
    else:
        logger.info("[%s] Cache stale or missing — fetching fresh data", sport)

    fetcher = fetcher_factory()
    raw_df = fetcher.fetch_all_seasons()
    engineer = engineer_factory()
    features_df = engineer.engineer_features(raw_df)
    cache.parent.mkdir(parents=True, exist_ok=True)
    parquet_df = _sanitize_features_for_parquet(features_df)
    parquet_df.to_parquet(cache)
    _write_feature_cache_scope(cache, sport)
    logger.info("[%s] Feature cache updated → %s (%d rows)", sport, cache_path, len(features_df))
    _feature_cache_meta[sport] = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "age_hours": 0.0,
        "source_status": "fresh_fetch",
    }
    return features_df, engineer


# Module-level store for soccer game detail data (populated by run_soccer,
# read by write_report to include in the summary JSON for the webapp).
_soccer_full_games: List[dict] = []

# Module-level store for all non-soccer games (basic info only, no ML outcomes).
# Populated by run_basketball / run_mlb / run_nhl / run_tennis.
# Each entry: {sport, home, away, league, commence, kick_off, window, home_odds, away_odds}
_other_sport_games: List[dict] = []
_feature_cache_meta: Dict[str, dict] = {}

# Small audit trail for WTA near-miss and review-only candidate inspection.
_wta_audit_rows: List[dict] = []

_MLB_SIDE_MODEL = MLBSideModel()
_BASKETBALL_SIDE_MODEL = BasketballSideModel()
_NHL_SIDE_MODEL = NHLSideModel()


def _extract_market_keys_from_odds_game(game: dict | None) -> list[str]:
    if not isinstance(game, dict):
        return []
    market_keys: list[str] = []
    for bookmaker in game.get("bookmakers", []) or []:
        if not isinstance(bookmaker, dict):
            continue
        for market in bookmaker.get("markets", []) or []:
            if not isinstance(market, dict):
                continue
            key = str(market.get("key", "") or "").strip().lower()
            if key and key not in market_keys:
                market_keys.append(key)
    return market_keys


def _merge_market_keys(existing: object, *extra_groups: object) -> list[str]:
    merged: list[str] = []

    def _add(values: object) -> None:
        if isinstance(values, (list, tuple)):
            for item in values:
                key = str(item or "").strip().lower()
                if key and key not in merged:
                    merged.append(key)

    _add(existing)
    for group in extra_groups:
        _add(group)
    return merged


def _detect_contextual_flags(game: dict, sport: str) -> dict:
    """
    Derive binary contextual features from the game metadata.

    is_playoff:
        Inferred from league/tournament name keywords. Odds-API game titles
        for playoff rounds typically contain 'playoff', 'postseason', 'final',
        'semifinal', 'quarterfinal', 'round 1', 'series'.

    rest_advantage:
        +1  if home team has a rest advantage (home played ≥2 days more recently)
        -1  if away team has a rest advantage
         0  if roughly equal (within 1 day)

        We estimate rest from the _window label and commence_time spacing —
        a game labeled 'day_after' for a team that played 'today' implies
        back-to-back. Falls back to 0 when data is insufficient.

    Returns dict with additive contextual flags used by the probability
    adjustment layer. These are intentionally conservative heuristics rather
    than hard truths.
    """
    flags = {
        "is_playoff": 0,
        "playoff_motivation": 0,
        "relegation_context": 0,
        "title_context": 0,
        "final_day_volatility": 0,
        "fixture_congestion_risk": 0,
        "cup_rotation_risk": 0,
        "european_rotation_risk": 0,
        "rest_advantage": 0,
    }

    # Playoff detection from description/league field
    desc = (game.get("description", "") + " " +
            game.get("sport_title", "") + " " +
            game.get("_league", "") + " " +
            game.get("league", "")).lower()
    playoff_keywords = ("playoff", "postseason", "final", "semifinal",
                        "quarterfinal", "quarter-final", "semi-final",
                        "round 1", "round 2", "round 3", "series",
                        "conference", "cup final", "champions league",
                        "knockout", "elimination")
    if any(kw in desc for kw in playoff_keywords):
        flags["is_playoff"] = 1
        flags["playoff_motivation"] = 1

    if any(kw in desc for kw in ("relegation", "survival", "staying up", "drop zone")):
        flags["relegation_context"] = 1

    if any(kw in desc for kw in ("title race", "title-decider", "title decider", "championship round", "league title")):
        flags["title_context"] = 1

    if any(kw in desc for kw in ("final day", "last round", "final round", "matchday 38", "matchday 46", "round 38")):
        flags["final_day_volatility"] = 1

    if any(kw in desc for kw in ("fa cup", "cup", "coppa", "pokal", "copa", "knockout")):
        flags["cup_rotation_risk"] = 1

    if any(kw in desc for kw in ("champions league", "europa league", "conference league", "libertadores", "sudamericana")):
        flags["european_rotation_risk"] = 1

    # Rest advantage: use _window label as a proxy
    # If a game is on back-to-back nights the window flips sooner.
    # We don't have per-team schedule data at fetch time, so we use
    # the rest_diff feature from the feature store when available.
    home_rest = game.get("home_rest_days")
    away_rest = game.get("away_rest_days")
    if home_rest is not None and away_rest is not None:
        diff = float(home_rest) - float(away_rest)
        if diff >= 1.5:
            flags["rest_advantage"] = 1   # home better rested
        elif diff <= -1.5:
            flags["rest_advantage"] = -1  # away better rested
        if min(float(home_rest), float(away_rest)) <= 3.0:
            flags["fixture_congestion_risk"] = 1

    return flags


# ─────────────────────────────────────────────────────────────────────────────
# Soccer pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_soccer(dry_run: bool = False) -> Tuple[List[dict], List[ParlayLeg]]:
    global _soccer_full_games, _scan_runtime_notes
    _soccer_full_games = []   # reset on each run
    logger.info("=== SOCCER PIPELINE ===")
    features_df, engineer = _load_features_cached(
        "soccer",
        "data/cache/soccer_features.parquet",
        SoccerFetcher,
        SoccerFeatureEngineer,
        allow_live_refresh=False,
    )
    if features_df.empty:
        logger.warning("No soccer data available.")
        return [], []
    logger.info("Soccer data: %d matches (cached)", len(features_df))

    y_all = features_df["target"].dropna()

    # Load or train model
    _soccer_tag = get_current_model_tag("soccer", fallback="pl_2024_25")
    trainer = ModelTrainer(sport="soccer")
    trainer.load_models(tag=_soccer_tag)
    if not trainer.trained_models:
        logger.info("No saved soccer models found; training now …")
        _always_drop = ["result", "home_goals", "away_goals", "home_ht", "away_ht",
                        "home_score", "away_score", "match_id", "date", "season",
                        "home_team", "away_team", "competition", "target"]
        X_tmp = features_df.drop(columns=[c for c in _always_drop if c in features_df.columns],
                                  errors="ignore").select_dtypes(include=[np.number]).fillna(0)
        X_tmp = X_tmp.loc[y_all.index]
        split = int(len(X_tmp) * 0.8)
        trainer.train(X_tmp.iloc[:split], y_all.iloc[:split],
                      X_tmp.iloc[split:], y_all.iloc[split:])
        trainer.save_models(tag=_soccer_tag)

    # Align feature matrix to exactly the columns the models were trained on
    sample_model = next(iter(trainer.trained_models.values()))
    feature_cols = list(sample_model.feature_names_in_)
    missing = [c for c in feature_cols if c not in features_df.columns]
    if missing:
        logger.warning("Adding zero-filled missing soccer features: %s", missing)
        for c in missing:
            features_df[c] = 0.0
    X_all = features_df[feature_cols].fillna(0)
    X_all = X_all.loc[y_all.index]

    n_classes = y_all.nunique()
    trainer.ensemble_model = _SoftVotingWrapper(
        estimators=list(trainer.trained_models.items()),
        weights=None,
        classes=np.array(sorted(y_all.unique())),
    )
    _soccer_cal = EnsembleCalibrator.load(calibrator_path_for_tag("soccer", _soccer_tag))
    _soccer_alpha = _effective_alpha("soccer", calibrated=_soccer_cal is not None)
    _soccer_spreads_model = TotalsTrainer.load(Path("data/models/soccer/spreads_soccer.joblib"))

    label_map = engineer.label_map  # e.g. {0: 'away_win', 1: 'draw', 2: 'home_win'}
    inv_map = {v: k for k, v in label_map.items()}  # 'home_win' -> 2

    _soccer_aliases = _SOCCER_ALIAS_OVERRIDES

    # Build a normalised → canonical name lookup once, reuse for all games
    _all_training_teams: set = set(features_df["home_team"].dropna().astype(str).unique()) | set(features_df["away_team"].dropna().astype(str).unique())

    _norm_to_canonical = build_entity_alias_map(_all_training_teams, extra_aliases=_soccer_aliases)
    _soccer_resolver = TeamResolver("soccer")

    def _swap_perspective(rows: pd.DataFrame) -> pd.DataFrame:
        rename_map = {}
        for col in rows.columns:
            if col.startswith("home_"):
                rename_map[col] = f"away_{col[5:]}"
            elif col.startswith("away_"):
                rename_map[col] = f"home_{col[5:]}"
            else:
                rename_map[col] = col

        swapped = rows.rename(columns=rename_map).reindex(columns=X_all.columns)
        for col in swapped.columns:
            if col.endswith("_diff"):
                swapped[col] = -swapped[col]
        return swapped

    def _resolve_team_name(odds_name: str) -> str:
        """Map odds API team name → training data canonical name via alias table + fuzzy normalisation."""
        return _resolve_soccer_team_name(
            odds_name,
            _all_training_teams,
            _norm_to_canonical,
            _soccer_resolver,
            _soccer_aliases,
        )

    _soccer_snapshot_baseline_cache: dict[tuple[str, bool], pd.Series] = {}

    def _competition_baseline_snapshot(competition_code: str, as_home: bool) -> pd.Series | None:
        cache_key = (str(competition_code or "__global__"), bool(as_home))
        cached = _soccer_snapshot_baseline_cache.get(cache_key)
        if cached is not None:
            return cached.copy()

        if competition_code and "competition" in features_df.columns:
            competition_mask = (features_df["competition"].astype(str) == str(competition_code)).reindex(X_all.index, fill_value=False)
            scoped = X_all.loc[competition_mask]
        else:
            scoped = pd.DataFrame(columns=X_all.columns)

        if scoped.empty:
            scoped = X_all
        if scoped.empty:
            return None

        baseline = scoped.tail(800).median(numeric_only=True).reindex(X_all.columns).fillna(scoped.median(numeric_only=True)).fillna(0.0)
        baseline.attrs["history_rows"] = 0
        baseline.attrs["synthetic_history"] = True
        baseline.attrs["resolved_team"] = ""
        baseline.attrs["baseline_scope"] = competition_code or "__global__"
        _soccer_snapshot_baseline_cache[cache_key] = baseline
        return baseline.copy()

    def get_team_snapshot(team: str, as_home: bool, *, competition_code: str = "") -> pd.Series | None:
        """
        Return feature snapshot for a team, or None if history is too thin.

        Key guards applied here:
        1. Name resolution: odds API names (e.g. "Liverpool") are fuzzy-matched
           to training data canonical names (e.g. "Liverpool FC").
        2. Minimum history: require at least 5 matches. Without this, a team
           that only appears once in our data gets the population median features,
           which assigns ~33% win probability regardless of actual quality.
        3. Cross-league ELO: our ELO is trained within-league. A Portuguese team
           (Sporting) playing at Arsenal has an ELO computed vs Primeira Liga sides,
           not Premier League sides. We cannot adjust for this here — we note it
           via a confidence flag so the sanity thresholds catch implausible edges.
        """
        canonical = _resolve_team_name(team)
        home_rows = X_all.loc[features_df["home_team"] == canonical]
        away_rows = X_all.loc[features_df["away_team"] == canonical]

        normalized_rows = []
        if as_home:
            if not home_rows.empty:
                normalized_rows.append(home_rows)
            if not away_rows.empty:
                normalized_rows.append(_swap_perspective(away_rows))
        else:
            if not away_rows.empty:
                normalized_rows.append(away_rows)
            if not home_rows.empty:
                normalized_rows.append(_swap_perspective(home_rows))

        rows = pd.concat(normalized_rows, ignore_index=False) if normalized_rows else pd.DataFrame(columns=X_all.columns)

        row_count = len(rows)
        baseline = _competition_baseline_snapshot(competition_code, as_home=as_home)
        if row_count <= 0 and baseline is not None:
            synthetic = baseline.copy()
            synthetic.attrs["history_rows"] = 0
            synthetic.attrs["resolved_team"] = canonical or team
            synthetic.attrs["synthetic_history"] = True
            logger.info(
                "BASELINE snapshot [%s]: no direct training rows found — using %s soccer baseline",
                team,
                competition_code or "global",
            )
            return synthetic
        if row_count < _SOCCER_MIN_SNAPSHOT_ROWS and baseline is not None:
            team_mean = rows.tail(10).mean()
            baseline_weight = max(0.35, 1.0 - (row_count / max(_SOCCER_MIN_SNAPSHOT_ROWS, 1)))
            snap = ((1.0 - baseline_weight) * team_mean) + (baseline_weight * baseline)
            snap.attrs["history_rows"] = row_count
            snap.attrs["resolved_team"] = canonical
            snap.attrs["synthetic_history"] = True
            snap.attrs["baseline_scope"] = competition_code or "__global__"
            logger.info(
                "BASELINE snapshot [%s]: only %d rows in training data — blending with %s soccer baseline",
                team,
                row_count,
                competition_code or "global",
            )
            return snap
        if row_count < _SOCCER_MIN_SNAPSHOT_ROWS:
            logger.info(
                "SKIP snapshot [%s]: only %d rows in training data — insufficient history",
                team, row_count
            )
            return None
        snap = rows.tail(10).mean()
        snap.attrs["history_rows"] = row_count
        snap.attrs["resolved_team"] = canonical
        snap.attrs["synthetic_history"] = False
        return snap

    if dry_run:
        logger.info("[dry-run] Skipping odds fetch for soccer.")
        return [], []

    # Filter to only leagues the Odds API has active right now — uses the
    # pre-fetched active sports cache (0 extra requests if already warmed up)
    _sports_cache_key = "__active_sports__"
    active_sports: set = set()
    if _sports_cache_key in _odds_cache:
        _cached_at, active_sports = _odds_cache[_sports_cache_key]
    excluded_soccer = {
        "soccer_fifa_world_cup_winner",
    }
    if active_sports:
        active_soccer_keys = sorted(
            k for k in active_sports
            if k in soccer_scanable_keys() and not k.endswith("_winner") and k not in excluded_soccer
        )
        skipped_leagues = sorted(
            k for k in active_sports
            if k.startswith("soccer_") and (k.endswith("_winner") or k in excluded_soccer)
        )
    else:
        active_soccer_keys = soccer_scanable_keys()
        skipped_leagues = []
    deferred_review_keys: List[str] = []
    if active_sports:
        active_soccer_keys, deferred_review_keys = _select_soccer_live_scope(active_soccer_keys)
    if (
        active_sports
        and not deferred_review_keys
        and os.getenv("SCAN_FULL_SOCCER_SCOPE", "").strip().lower() in {"1", "true", "yes", "on"}
    ):
        _scan_runtime_notes.append(
            {
                "type": "full_soccer_scope",
                "sport": "soccer",
                "reason": "Full soccer scope override kept review-only leagues in the live scan.",
            }
        )
    if skipped_leagues:
        logger.info(
            "Soccer: skipping %d non-match markets (%s) — %d active leagues/cups to fetch",
            len(skipped_leagues),
            ", ".join(_pretty_soccer_league_name(k) for k in skipped_leagues),
            len(active_soccer_keys),
        )
    if deferred_review_keys:
        _scan_runtime_notes.append(
            {
                "type": "deferred_leagues",
                "sport": "soccer",
                "count": len(deferred_review_keys),
                "reason": "low-value review-only leagues were deferred to conserve live quota and improve scan speed",
                "leagues": [_pretty_soccer_league_name(k) for k in deferred_review_keys],
            }
        )
        logger.info(
            "Soccer: deferring %d review-only leagues without recent activity because live quota is being conserved (%s)",
            len(deferred_review_keys),
            ", ".join(_pretty_soccer_league_name(k) for k in deferred_review_keys[:8])
            + (" …" if len(deferred_review_keys) > 8 else ""),
        )
    logger.info("Soccer: scanning %d active leagues/cups from the Odds API feed", len(active_soccer_keys))

    all_games: List[dict] = []
    for sport_key in active_soccer_keys:
        league_name = _pretty_soccer_league_name(sport_key)
        try:
            lg_games = fetch_odds(sport_key)
            today_cnt    = sum(1 for g in lg_games if g.get("_window") == "today")
            overnight_cnt = sum(1 for g in lg_games if g.get("_window") == "overnight")
            tomorrow_cnt  = sum(1 for g in lg_games if g.get("_window") == "tomorrow")
            day_after_cnt = sum(1 for g in lg_games if g.get("_window") == "day_after")
            logger.info("%s: %d upcoming (today=%d overnight=%d tomorrow=%d day_after=%d)",
                        league_name, len(lg_games), today_cnt, overnight_cnt,
                        tomorrow_cnt, day_after_cnt)
            totals_map = {}
            spreads_map = {}
            if lg_games:
                # Fetch optional market payloads once per league and reuse them
                # across every game in this league during the current run.
                try:
                    totals_games = fetch_odds(sport_key, markets="totals", priority="optional")
                    totals_map = {g["id"]: g for g in totals_games if "id" in g}
                except Exception:
                    totals_map = {}
                try:
                    spreads_games = fetch_odds(sport_key, markets="spreads", priority="optional")
                    spreads_map = {g["id"]: g for g in spreads_games if "id" in g}
                except Exception:
                    spreads_map = {}
            for g in lg_games:
                g["_league"] = league_name
                g["_league_key"] = sport_key
                g["_totals"] = totals_map.get(g.get("id"), {})
                g["_spreads"] = spreads_map.get(g.get("id"), {})
            all_games.extend(lg_games)
        except Exception as exc:
            logger.warning("Could not fetch odds for %s: %s", league_name, exc)

    logger.info("Total soccer games with odds: %d", len(all_games))

    value_bets: List[dict] = []
    parlay_legs: List[ParlayLeg] = []

    for game in all_games:
        home_team = game["home_team"]
        away_team = game["away_team"]
        commence  = game["commence_time"]
        window    = game.get("_window", "today")
        competition_code = str(SOCCER_ODDS_TO_COMPETITION.get(str(game.get("_league_key", "") or ""), "") or "")

        home_odds, home_bk, home_stale = best_odds(game, home_team)
        away_odds, away_bk, away_stale = best_odds(game, away_team)

        # Best draw odds
        draw_best: float = 0.0
        draw_bk: str = ""
        for bk in game.get("bookmakers", []):
            bk_name = bk.get("title", "unknown")
            for mkt in bk.get("markets", []):
                if mkt["key"] == "h2h" and len(mkt["outcomes"]) == 3:
                    for o in mkt["outcomes"]:
                        if o["name"].lower() == "draw" and o["price"] > draw_best:
                            draw_best = o["price"]
                            draw_bk   = bk_name
        draw_odds = float(draw_best) if draw_best > 0 else None

        if home_odds is None or away_odds is None:
            continue

        # ── Market-implied (vig-free) probabilities ───────────────────────────
        if draw_odds:
            fp_home, fp_draw, fp_away = vig_free_prob(home_odds, draw_odds, away_odds)
        else:
            fp_home, fp_away = vig_free_prob(home_odds, away_odds)
            fp_draw = 0.0

        home_med = median_odds(game, home_team)
        away_med = median_odds(game, away_team)
        ctx = _detect_contextual_flags(game, "soccer")
        game_info = {"sport": "soccer", "home": home_team, "away": away_team,
                     "commence": commence, "kick_off": _kick_off_label(commence),
                     "window": window, "league": game.get("_league", "Soccer"),
                     "league_key": game.get("_league_key", ""),
                     "event_id": game.get("id", ""),
                     "odds_snapshot_age_hours": game.get("_odds_snapshot_age_hours"),
                     "odds_source_status": game.get("_odds_source_status", ""),
                     "odds_fetched_at": game.get("_odds_fetched_at"),
                     "bookmaker_last_update": game.get("_odds_bookmaker_last_update"),
                     "cache_loaded_at": game.get("_odds_cache_loaded_at"),
                     "cache_age_hours": game.get("_odds_cache_age_hours"),
                     "odds_age_basis": game.get("_odds_age_basis", ""),
                     "odds_source_detail": game.get("_odds_source_detail", ""),
                     "odds_source_reason": game.get("_odds_source_reason", ""),
                     "force_fresh_odds_active": bool(game.get("_odds_force_fresh_requested", _FORCE_FRESH_ODDS)),
                     "odds_cache_used": bool(game.get("_odds_cache_used")),
                     "odds_fallback_used": bool(game.get("_odds_fallback_used")),
                     **_feature_freshness_context("soccer"),
                     **ctx}

        # ── Best double-chance odds from bookmakers ───────────────────────────
        hd_best, ad_best, hd_bk, ad_bk = 0.0, 0.0, "", ""
        for bk in game.get("bookmakers", []):
            bk_name_dc = bk.get("title", "")
            for mkt in bk.get("markets", []):
                if mkt.get("key") == "h2h_lay":
                    for o in mkt.get("outcomes", []):
                        n = o.get("name", "").lower()
                        p = o.get("price", 0)
                        if "home" in n and "draw" in n and p > hd_best:
                            hd_best = p; hd_bk = bk_name_dc
                        elif "away" in n and "draw" in n and p > ad_best:
                            ad_best = p; ad_bk = bk_name_dc
        # Fallback: estimate double-chance odds from 1x2 vig-free probs
        fp_hd = min(fp_home + fp_draw, 0.99) if fp_draw else None
        fp_ad = min(fp_away + fp_draw, 0.99) if fp_draw else None
        if hd_best < 1.1 and fp_hd:
            hd_best = round(1.0 / fp_hd * 0.95, 2)
        if ad_best < 1.1 and fp_ad:
            ad_best = round(1.0 / fp_ad * 0.95, 2)

        # Draw-no-bet odds are not available directly from the current odds feed,
        # so derive a conservative estimate from the 1X2 book when needed.
        hd_resolved = max(1e-9, fp_home + fp_away)
        home_dnb_fair = fp_home / hd_resolved
        away_dnb_fair = fp_away / hd_resolved
        home_dnb_odds = round((1.0 / home_dnb_fair) * 0.97, 2)
        away_dnb_odds = round((1.0 / away_dnb_fair) * 0.97, 2)

        def _outcome(label: str, team_key: str, prob: float, fair_p,
                     odds: float, bk: str, model_available: bool = True) -> dict:
            edge = round(prob * odds - 1, 4) if odds and model_available else None
            return {
                "label":           label,
                "team":            team_key,
                "ml_prob":         round(prob, 4) if model_available else None,
                "fair_prob":       round(fair_p, 4) if fair_p else None,
                "odds":            round(odds, 3) if odds else None,
                "bookmaker":       bk,
                "edge":            edge,
                "has_value":       bool(edge and edge >= 0.03),
                "model_available": model_available,
            }

        # ── Build basic game record with market odds only (no model yet) ──────
        # This ensures ALL games with odds appear in the webapp, even if we
        # don't have enough historical data to run the ML model for that game.
        full_game = enrich_with_capability({
            **game_info,
            "model_available": False,
            "available_market_keys": _merge_market_keys(
                _extract_market_keys_from_odds_game(game),
                _extract_market_keys_from_odds_game(game.get("_totals")),
                _extract_market_keys_from_odds_game(game.get("_spreads")),
            ),
            "outcomes": [
                _outcome("Home Win",     home_team,              fp_home, fp_home, home_odds,    home_bk, False),
                _outcome("Draw",         "Draw",                 fp_draw, fp_draw, draw_odds or 0, draw_bk, False),
                _outcome("Away Win",     away_team,              fp_away, fp_away, away_odds,    away_bk, False),
                _outcome("Home or Draw", f"{home_team} or Draw", fp_hd or 0, fp_hd, hd_best or 0, hd_bk, False),
                _outcome("Away or Draw", f"{away_team} or Draw", fp_ad or 0, fp_ad, ad_best or 0, ad_bk, False),
            ],
        }, sport="soccer", league=game.get("_league", "Soccer"), sport_key=game.get("_league_key", ""))
        _soccer_full_games.append(full_game)  # stored module-level for write_report

        # ── ML model probabilities (only when both teams have enough history) ─
        h_snap = get_team_snapshot(home_team, as_home=True, competition_code=competition_code)
        a_snap = get_team_snapshot(away_team, as_home=False, competition_code=competition_code)
        if h_snap is None or a_snap is None:
            logger.info("SKIP %s vs %s: insufficient team history (odds-only record stored)",
                        home_team, away_team)
            continue
        home_history_rows = int(h_snap.attrs.get("history_rows", 0) or 0)
        away_history_rows = int(a_snap.attrs.get("history_rows", 0) or 0)
        home_synthetic_history = bool(h_snap.attrs.get("synthetic_history"))
        away_synthetic_history = bool(a_snap.attrs.get("synthetic_history"))
        combined = h_snap.copy()
        combined.attrs["home_history_rows"] = home_history_rows
        combined.attrs["away_history_rows"] = away_history_rows
        combined.attrs["home_synthetic_history"] = home_synthetic_history
        combined.attrs["away_synthetic_history"] = away_synthetic_history
        for col in [c for c in feature_cols if c.startswith("away_")]:
            if col in a_snap.index:
                combined[col] = a_snap[col]
        ctx = {
            **ctx,
            **build_availability_context("soccer", game, combined),
            **build_environment_context("soccer", home_team, away_team, commence),
        }
        game_info.update(ctx)
        full_game.update(ctx)
        full_game["history_rows"] = {"home": home_history_rows, "away": away_history_rows}
        full_game["synthetic_history"] = {"home": home_synthetic_history, "away": away_synthetic_history}

        x = pd.DataFrame([combined], columns=feature_cols)
        raw_p = trainer.ensemble_model.predict_proba(x)
        proba = _soccer_cal.transform(raw_p)[0] if _soccer_cal else raw_p[0]

        home_idx = inv_map.get("home_win", 2)
        draw_idx = inv_map.get("draw", 1)
        away_idx = inv_map.get("away_win", 0)
        p_home = float(proba[home_idx]) if home_idx < len(proba) else 0.33
        p_draw = float(proba[draw_idx]) if draw_idx < len(proba) else 0.33
        p_away = float(proba[away_idx]) if away_idx < len(proba) else 0.33
        classifier_probs = _normalize_three_probs(p_home, p_draw, p_away)
        structural_probs = _soccer_structural_probs(combined)
        soccer_blend = _SOCCER_SCORE_MODEL.combine_with_classifier_diagnostics(
            classifier_probs,
            structural_probs,
        )
        p_home, p_draw, p_away = soccer_blend.combined.as_tuple()
        p_home, p_draw, p_away, soccer_context_probability_adjustment = _apply_soccer_context_probability_adjustment(
            p_home,
            p_draw,
            p_away,
            ctx,
        )
        pre_market_blend_probs = _normalize_three_probs(p_home, p_draw, p_away)
        clamped_probs, soccer_market_clamp = _soccer_market_clamp_three_way(
            pre_market_blend_probs,
            (float(fp_home), float(fp_draw or (1 - fp_home - fp_away)), float(fp_away)),
            home_rows=home_history_rows,
            away_rows=away_history_rows,
            home_synthetic=home_synthetic_history,
            away_synthetic=away_synthetic_history,
        )
        p_home, p_draw, p_away = clamped_probs

        # Market-anchor: blend calibrated model probs toward market consensus
        _soccer_live_alpha = _live_blend_alpha(
            "soccer",
            "moneyline",
            _soccer_alpha,
            home_rows=home_history_rows,
            away_rows=away_history_rows,
            min_rows=_SOCCER_MIN_SNAPSHOT_ROWS,
            disagreement_pp=float(soccer_blend.disagreement) * 100.0,
        )
        if home_synthetic_history and away_synthetic_history:
            _soccer_live_alpha = min(_soccer_live_alpha, 0.36)
        elif home_synthetic_history or away_synthetic_history:
            _soccer_live_alpha = min(_soccer_live_alpha, 0.40)
        p_home, p_draw, p_away = _blend3(p_home, p_draw, p_away,
                                         fp_home, fp_draw or (1 - fp_home - fp_away),
                                         fp_away, alpha=_soccer_live_alpha)
        final_probs = _normalize_three_probs(p_home, p_draw, p_away)
        soccer_probability_debug = {
            "classifier_probs": {
                "home": round(soccer_blend.classifier.home, 4),
                "draw": round(soccer_blend.classifier.draw, 4),
                "away": round(soccer_blend.classifier.away, 4),
            },
            "structural_probs": {
                "home": round(soccer_blend.structural.home, 4),
                "draw": round(soccer_blend.structural.draw, 4),
                "away": round(soccer_blend.structural.away, 4),
            } if soccer_blend.structural else None,
            "market_probs": {
                "home": round(float(fp_home), 4),
                "draw": round(float(fp_draw or (1 - fp_home - fp_away)), 4),
                "away": round(float(fp_away), 4),
            },
            "pre_market_blend_probs": {
                "home": round(pre_market_blend_probs[0], 4),
                "draw": round(pre_market_blend_probs[1], 4),
                "away": round(pre_market_blend_probs[2], 4),
            },
            "post_clamp_pre_blend_probs": {
                "home": round(clamped_probs[0], 4),
                "draw": round(clamped_probs[1], 4),
                "away": round(clamped_probs[2], 4),
            },
            "final_probs": {
                "home": round(final_probs[0], 4),
                "draw": round(final_probs[1], 4),
                "away": round(final_probs[2], 4),
            },
            "blend_alpha": round(float(_soccer_live_alpha), 3),
            "history_rows": {
                "home": home_history_rows,
                "away": away_history_rows,
            },
            "synthetic_history": {
                "home": home_synthetic_history,
                "away": away_synthetic_history,
            },
            "market_clamp": soccer_market_clamp,
            "structural_available": bool(soccer_blend.structural),
            "regime": soccer_blend.regime,
            "disagreement_pp": round(float(soccer_blend.disagreement) * 100, 1),
            "classifier_weight": round(float(soccer_blend.model_weight), 3),
            "structural_weight": round(float(soccer_blend.structural_weight), 3),
            "context_probability_adjustment": soccer_context_probability_adjustment,
        }
        game_info["soccer_probability_debug"] = soccer_probability_debug

        # Double-chance ML probabilities
        p_hd = min(p_home + p_draw, 0.99)
        p_ad = min(p_away + p_draw, 0.99)
        raw_home_prob = float(proba[home_idx]) if home_idx < len(proba) else 0.33
        raw_draw_prob = float(proba[draw_idx]) if draw_idx < len(proba) else 0.33
        raw_away_prob = float(proba[away_idx]) if away_idx < len(proba) else 0.33
        raw_hd_prob = min(raw_home_prob + raw_draw_prob, 0.99)
        raw_ad_prob = min(raw_away_prob + raw_draw_prob, 0.99)
        home_factors = _probability_factors_for_snapshot("soccer", combined, home_team=home_team, away_team=away_team, selection="home")
        away_factors = _probability_factors_for_snapshot("soccer", combined, home_team=home_team, away_team=away_team, selection="away")
        draw_factors = _probability_factors_for_snapshot("soccer", combined, home_team=home_team, away_team=away_team, selection="draw")
        home_adjustments = _context_adjustments("soccer", "home", combined, ctx)
        away_adjustments = _context_adjustments("soccer", "away", combined, ctx)
        draw_adjustments = _context_adjustments("soccer", "draw", combined, ctx)
        hd_adjustments = _context_adjustments("soccer", "home_or_draw", combined, ctx)
        ad_adjustments = _context_adjustments("soccer", "away_or_draw", combined, ctx)

        # ── Update game record with ML probabilities in place ─────────────────
        full_game["model_available"] = True
        full_game["soccer_probability_debug"] = soccer_probability_debug
        full_game["outcomes"] = [
            _outcome("Home Win",     home_team,              p_home, fp_home, home_odds,    home_bk, True),
            _outcome("Draw",         "Draw",                 p_draw, fp_draw, draw_odds or 0, draw_bk, True),
            _outcome("Away Win",     away_team,              p_away, fp_away, away_odds,    away_bk, True),
            _outcome("Home or Draw", f"{home_team} or Draw", p_hd,   fp_hd,   hd_best or 0, hd_bk,   True),
            _outcome("Away or Draw", f"{away_team} or Draw", p_ad,   fp_ad,   ad_best or 0, ad_bk,   True),
        ]

        # Single bets — today only
        dc_home_bet = None
        dc_away_bet = None
        if hd_best > 1.0 and fp_hd:
            _bet = build_value_bet(
                f"{home_team} or Draw",
                p_hd,
                hd_best,
                fp_hd,
                min_edge=_SOCCER_PRODUCTION_MIN_EDGE,
                sport="soccer",
                market="double_chance",
                raw_model_prob=p_hd,
                prediction_factors=home_factors + draw_factors,
                context_adjustments=hd_adjustments,
                availability_context=ctx,
            )
            if _bet:
                dc_home_bet = {**_bet, "market": "double_chance"}
        if ad_best > 1.0 and fp_ad:
            _bet = build_value_bet(
                f"{away_team} or Draw",
                p_ad,
                ad_best,
                fp_ad,
                min_edge=_SOCCER_PRODUCTION_MIN_EDGE,
                sport="soccer",
                market="double_chance",
                raw_model_prob=p_ad,
                prediction_factors=away_factors + draw_factors,
                context_adjustments=ad_adjustments,
                availability_context=ctx,
            )
            if _bet:
                dc_away_bet = {**_bet, "market": "double_chance"}

        for bet, bk_name in [
            (build_value_bet(home_team, p_home, home_odds, fp_home,
                             min_edge=_SOCCER_PRODUCTION_MIN_EDGE,
                             stale_line=home_stale, median_price=home_med, sport="soccer",
                             raw_model_prob=p_home, prediction_factors=home_factors,
                             context_adjustments=home_adjustments, availability_context=ctx), home_bk),
            (build_value_bet("Draw", p_draw, draw_odds or 3.5, fp_draw,
                             min_edge=_SOCCER_PRODUCTION_MIN_EDGE,
                             sport="soccer",
                             raw_model_prob=p_draw, prediction_factors=draw_factors,
                             context_adjustments=draw_adjustments, availability_context=ctx) if draw_odds else None, draw_bk),
            (build_value_bet(away_team, p_away, away_odds, fp_away,
                             min_edge=_SOCCER_PRODUCTION_MIN_EDGE,
                             stale_line=away_stale, median_price=away_med, sport="soccer",
                             raw_model_prob=p_away, prediction_factors=away_factors,
                             context_adjustments=away_adjustments, availability_context=ctx), away_bk),
            (dc_home_bet, hd_bk or "synthetic"),
            (dc_away_bet, ad_bk or "synthetic"),
            (build_refund_value_bet(
                f"{home_team} DNB",
                p_home,
                p_draw,
                home_dnb_odds,
                home_dnb_fair,
                min_edge=_SOCCER_PRODUCTION_MIN_EDGE,
                sport="soccer",
                synthetic_line=True,
                raw_model_prob=p_home,
                prediction_factors=home_factors,
                context_adjustments=home_adjustments,
                availability_context=ctx,
            ), "synthetic_1x2"),
            (build_refund_value_bet(
                f"{away_team} DNB",
                p_away,
                p_draw,
                away_dnb_odds,
                away_dnb_fair,
                min_edge=_SOCCER_PRODUCTION_MIN_EDGE,
                sport="soccer",
                synthetic_line=True,
                raw_model_prob=p_away,
                prediction_factors=away_factors,
                context_adjustments=away_adjustments,
                availability_context=ctx,
            ), "synthetic_1x2"),
        ]:
            if bet and window in ("today", "tomorrow", "day_after"):
                market_key = bet.get("market", "moneyline")
                enriched_bet = _with_candidate_odds_diagnostics(
                    {**game_info, **bet, "market": market_key, "bookmaker": bk_name},
                    game=game,
                    market_key="h2h" if market_key in {"moneyline", "double_chance", "draw_no_bet"} else market_key,
                    selection_name=str(bet.get("team", "") or ""),
                    bookmaker_name=str(bk_name or ""),
                )
                value_bets.append(enrich_with_capability(
                    enriched_bet,
                    sport="soccer",
                    league=game.get("_league", "Soccer"),
                    sport_key=game.get("_league_key", ""),
                ))

        # Parlay legs — both windows
        for team, ml_prob, team_odds, fair_prob in [
            (home_team, p_home, home_odds, fp_home),
            (away_team, p_away, away_odds, fp_away),
            (f"{home_team} or Draw", p_hd, hd_best, fp_hd),
            (f"{away_team} or Draw", p_ad, ad_best, fp_ad),
        ]:
            if team_odds and team_odds > 1.05:
                _leg = to_parlay_leg("soccer", home_team, away_team, team,
                                     team_odds, ml_prob, fair_prob, commence, window=window)
                if _leg:
                    parlay_legs.append(_leg)
        if draw_odds and draw_odds > 1.05:
            _leg = to_parlay_leg("soccer", home_team, away_team, "Draw",
                                 draw_odds, p_draw, fp_draw, commence, window=window)
            if _leg:
                parlay_legs.append(_leg)

        # ── Poisson over/under totals ──────────────────────────────────────────
        if window in ("today", "tomorrow", "day_after"):
            totals_data = game.get("_totals", {})
            _best_over: dict[float, float] = {}   # line -> best odds
            _best_under: dict[float, float] = {}
            for bk in totals_data.get("bookmakers", []):
                for mkt in bk.get("markets", []):
                    if mkt.get("key") != "totals":
                        continue
                    for o in mkt.get("outcomes", []):
                        name  = o.get("name", "")
                        price = float(o.get("price", 0))
                        if price <= 1.0:
                            continue
                        # Parse "Over 2.5" / "Under 2.5"
                        parts = name.split()
                        if len(parts) != 2:
                            continue
                        direction = parts[0].lower()
                        try:
                            ln = float(parts[1])
                        except ValueError:
                            continue
                        if direction == "over":
                            _best_over[ln]  = max(_best_over.get(ln, 0), price)
                        elif direction == "under":
                            _best_under[ln] = max(_best_under.get(ln, 0), price)

            # Try lines in priority order: 2.5, 2, 3
            for _line in (2.5, 2.0, 3.0):
                _ov = _best_over.get(_line)
                _un = _best_under.get(_line)
                if _ov and _un:
                    totals_results = poisson_totals_bet(
                        home_team, away_team, features_df,
                        over_odds=_ov, under_odds=_un, line=_line,
                        window=window, commence=commence,
                        league=game.get("_league", "Soccer"), sport="soccer",
                    )
                    if totals_results:
                        for tb in totals_results:
                            value_bets.append({
                                **game_info,
                                **tb,
                                "team_or_player": tb["team"],
                                "bet_odds": tb["odds"],
                                "bookmaker": "best",
                            })
                    break  # only process the first available line

        # ── Soccer Asian Handicap spreads (0.5 line) ───────────────────────────
        # Model predicts home win probability; AH 0.5 = home must win (no draw refund).
        # Use the soccer spreads model (trained on home_win vs draw+away_win).
        if window in ("today", "tomorrow", "day_after") and _soccer_spreads_model:
            _spreads_game = game.get("_spreads", {}) or {}
            _sh: float = 0.0
            _sa: float = 0.0
            _sh_line = 0.5
            for bk in _spreads_game.get("bookmakers", []):
                for mkt in bk.get("markets", []):
                    if mkt.get("key") != "spreads":
                        continue
                    for o in mkt.get("outcomes", []):
                        nm = o.get("name", "")
                        pt = o.get("point")
                        pr = float(o.get("price", 0))
                        if pr <= 1.0 or pt is None:
                            continue
                        if nm == home_team and float(pt) < 0:
                            _sh = max(_sh, pr)
                            _sh_line = abs(float(pt))
                        elif nm == away_team and float(pt) > 0:
                            _sa = max(_sa, pr)

            if _sh > 1.0 and _sa > 1.0:
                # Build X for the soccer spreads model
                _sx = pd.DataFrame([combined], columns=feature_cols)
                try:
                    _spreads_results = ml_spreads_bet(
                        "soccer", home_team, away_team, combined,
                        feature_cols, _soccer_spreads_model,
                        home_spread_odds=_sh, away_spread_odds=_sa,
                        line=_sh_line,
                        window=window, commence=commence,
                        league=game.get("_league", "Soccer"),
                        home_name=home_team, away_name=away_team,
                    )
                except Exception as _spreads_exc:
                    logger.debug("Soccer spreads skipped (%s vs %s): %s", home_team, away_team, _spreads_exc)
                    _spreads_results = None
                if _spreads_results:
                    for sb in _spreads_results:
                        value_bets.append(sb)

    return value_bets, parlay_legs


# ─────────────────────────────────────────────────────────────────────────────
# Basketball pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_basketball(dry_run: bool = False) -> Tuple[List[dict], List[ParlayLeg]]:
    global _other_sport_games
    logger.info("=== BASKETBALL PIPELINE ===")
    features_df, engineer = _load_features_cached(
        "basketball",
        "data/cache/basketball_features.parquet",
        BasketballFetcher,
        BasketballFeatureEngineer,
    )
    if features_df.empty:
        logger.warning("No basketball data available.")
        return [], []
    logger.info("NBA data: %d games (cached)", len(features_df))

    # Drop raw result-leaking columns but KEEP quarter features —
    # they will be filled from FeatureStore at inference time.
    _raw_leak_cols = [c for c in [
        "home_score", "away_score", "home_q1", "home_q2", "home_q3", "home_q4", "home_ot",
        "away_q1", "away_q2", "away_q3", "away_q4", "away_ot", "point_diff",
        "home_first_half", "home_second_half", "away_first_half", "away_second_half",
        "total_points", "result", "match_id", "date", "home_team", "away_team",
        "league_id", "league_name", "season", "status", "home_team_id", "away_team_id",
        "neg_point_diff",
    ] if c in features_df.columns]

    X_tmp = features_df.drop(columns=_raw_leak_cols + ["target"], errors="ignore").fillna(0)
    y_all = features_df["target"].dropna()
    X_tmp = X_tmp.loc[y_all.index]

    # Initialise FeatureStore — computes inference-time estimates for features
    # that are derived from raw quarter data (not available before a game is played).
    _feat_store = FeatureStore(features_df, sport="basketball", window=10)
    # NBA entity resolver: maps API full names → training names
    _nba_resolver = TeamResolver("nba")

    _bball_tag = get_current_model_tag("basketball", fallback="nba_2024_25")
    trainer = ModelTrainer(sport="basketball")
    trainer.load_models(tag=_bball_tag)
    if not trainer.trained_models:
        logger.info("No saved NBA models; training now...")
        split = int(len(X_tmp) * 0.8)
        trainer.train(X_tmp.iloc[:split], y_all.iloc[:split],
                      X_tmp.iloc[split:], y_all.iloc[split:])
        trainer.save_models(tag=_bball_tag)

    sample_model = next(iter(trainer.trained_models.values()))
    feature_cols = list(sample_model.feature_names_in_)
    missing = [c for c in feature_cols if c not in features_df.columns]
    if missing:
        logger.warning("Adding zero-filled missing basketball features: %s", missing)
        for c in missing:
            features_df[c] = 0.0
    X_all = features_df[feature_cols].fillna(0)
    X_all = X_all.loc[y_all.index]

    trainer.ensemble_model = _SoftVotingWrapper(
        estimators=list(trainer.trained_models.items()),
        weights=None,
        classes=np.array(sorted(y_all.unique())),
    )
    _bball_cal = EnsembleCalibrator.load(calibrator_path_for_tag("basketball", _bball_tag))
    _bball_alpha = _effective_alpha("basketball", calibrated=_bball_cal is not None)

    _SNAP_GAMES = 10

    def _swap_perspective(rows: pd.DataFrame) -> pd.DataFrame:
        rename_map = {}
        for col in rows.columns:
            if col.startswith("home_"):
                rename_map[col] = f"away_{col[5:]}"
            elif col.startswith("away_"):
                rename_map[col] = f"home_{col[5:]}"
            else:
                rename_map[col] = col

        swapped = rows.rename(columns=rename_map).reindex(columns=X_all.columns)
        for col in swapped.columns:
            if col.endswith("_diff"):
                swapped[col] = -swapped[col]
        return swapped

    def get_team_snapshot(team: str, as_home: bool) -> Optional[pd.Series]:
        # Resolve API name → training name via FeatureStore entity resolver
        resolved = _nba_resolver.resolve(team)
        home_rows = X_all.loc[features_df["home_team"] == resolved]
        away_rows = X_all.loc[features_df["away_team"] == resolved]

        normalized_rows = []
        if as_home:
            if not home_rows.empty:
                normalized_rows.append(home_rows)
            if not away_rows.empty:
                normalized_rows.append(_swap_perspective(away_rows))
        else:
            if not away_rows.empty:
                normalized_rows.append(away_rows)
            if not home_rows.empty:
                normalized_rows.append(_swap_perspective(home_rows))

        rows = pd.concat(normalized_rows, ignore_index=False) if normalized_rows else pd.DataFrame(columns=X_all.columns)
        row_count = len(rows)
        if row_count < _MIN_SNAPSHOT_ROWS:
            logger.warning("[basketball] Only %d rows for %s (resolved=%s) — skipping (min %d required)",
                           row_count, team, resolved, _MIN_SNAPSHOT_ROWS)
            return None
        snap = rows.tail(_SNAP_GAMES).mean()
        snap.attrs["history_rows"] = row_count
        snap.attrs["resolved_team"] = resolved
        return snap

    if dry_run:
        logger.info("[dry-run] Skipping odds fetch for basketball.")
        return [], []

    games = fetch_odds("basketball_nba")
    logger.info("Upcoming NBA games with odds: %d", len(games))

    value_bets: List[dict] = []
    parlay_legs: List[ParlayLeg] = []

    for game in games:
        home_team = game["home_team"]
        away_team = game["away_team"]
        commence  = game["commence_time"]
        window    = game.get("_window", "today")

        home_odds, home_bk, home_stale = best_odds(game, home_team)
        away_odds, away_bk, away_stale = best_odds(game, away_team)

        # Line velocity: abstain if either price has moved sharply (unpriced news)
        game_id = game.get("id", "")
        abstain_game = False
        if not _check_line_velocity("basketball_nba", game_id, home_team, home_odds or 0):
            abstain_game = True
        if not _check_line_velocity("basketball_nba", game_id, away_team, away_odds or 0):
            abstain_game = True

        # Always record the game for the All Games view (even if model can't run)
        game_record = enrich_with_capability({
            "sport": "basketball", "home": home_team, "away": away_team,
            "league": "NBA", "commence": commence,
            "kick_off": _kick_off_label(commence), "window": window,
            "home_odds": home_odds, "away_odds": away_odds,
            "league_key": "basketball_nba",
            "available_market_keys": _extract_market_keys_from_odds_game(game),
        }, sport="basketball", league="NBA", sport_key="basketball_nba")
        if abstain_game:
            game_record["abstain"] = True
        _other_sport_games.append(game_record)

        if home_odds is None or away_odds is None:
            continue
        if abstain_game:
            continue

        h_snap = get_team_snapshot(home_team, as_home=True)
        a_snap = get_team_snapshot(away_team, as_home=False)
        if h_snap is None or a_snap is None:
            continue
        home_history_rows = int(h_snap.attrs.get("history_rows", 0) or 0)
        away_history_rows = int(a_snap.attrs.get("history_rows", 0) or 0)

        combined = h_snap.copy()
        combined.attrs["home_history_rows"] = home_history_rows
        combined.attrs["away_history_rows"] = away_history_rows
        for col in [c for c in feature_cols if c.startswith("away_")]:
            if col in a_snap.index:
                combined[col] = a_snap[col]

        # Feature Store: fill in quarter-derived features from team history
        # instead of leaving them as 0 (which biases predictions toward average).
        extras = _feat_store.get_basketball_extras(home_team, away_team)
        for feat, val in extras.items():
            if feat in feature_cols:
                combined[feat] = val

        # Add rest days to game dict for contextual flags
        game["home_rest_days"] = float(extras.get("home_rest_days", 7))
        game["away_rest_days"] = float(extras.get("away_rest_days", 7))
        ctx = _detect_contextual_flags(game, "basketball")
        ctx = {
            **ctx,
            **build_availability_context("basketball", game, combined),
            **build_environment_context("basketball", home_team, away_team, commence),
        }

        x = pd.DataFrame([combined], columns=feature_cols)
        raw_p = trainer.ensemble_model.predict_proba(x)
        proba = _bball_cal.transform(raw_p)[0] if _bball_cal else raw_p[0]
        p_home = float(proba[1])
        p_away = float(proba[0])
        basketball_blend = _BASKETBALL_SIDE_MODEL.combine_with_classifier_diagnostics(
            (p_home, p_away),
            _BASKETBALL_SIDE_MODEL.structural_probs_from_snapshot(combined),
        )
        p_home, p_away = basketball_blend.combined.as_tuple()
        p_home, p_away, basketball_context_probability_adjustment = _apply_basketball_context_probability_adjustment(p_home, p_away, ctx)
        raw_home_prob = p_home
        raw_away_prob = p_away

        fp_home, fp_away = vig_free_prob(home_odds, away_odds)
        pre_market_blend_home = p_home
        pre_market_blend_away = p_away

        # Market-anchor: blend calibrated model probs toward market consensus.
        # Basketball uses a lower alpha (0.40) because calibration sets are small.
        # Winsorize first: clamp model prob to within 20pp of market to prevent
        # extreme divergences dominating even at low blend alpha.
        _nba_alpha = _live_blend_alpha(
            "basketball",
            "moneyline",
            _bball_alpha,
            home_rows=home_history_rows,
            away_rows=away_history_rows,
            disagreement_pp=float(basketball_blend.disagreement) * 100.0,
        )
        p_home = _winsorize_prob(p_home, fp_home, "basketball")
        p_away = _winsorize_prob(p_away, fp_away, "basketball")
        p_home = _blend(p_home, fp_home, alpha=_nba_alpha)
        p_away = _blend(p_away, fp_away, alpha=_nba_alpha)
        basketball_probability_debug = {
            "classifier_probs": {
                "home": round(basketball_blend.classifier.home, 4),
                "away": round(basketball_blend.classifier.away, 4),
            },
            "structural_probs": {
                "home": round(basketball_blend.structural.home, 4),
                "away": round(basketball_blend.structural.away, 4),
            } if basketball_blend.structural else None,
            "market_probs": {
                "home": round(float(fp_home), 4),
                "away": round(float(fp_away), 4),
            },
            "pre_market_blend_probs": {
                "home": round(float(pre_market_blend_home), 4),
                "away": round(float(pre_market_blend_away), 4),
            },
            "final_probs": {
                "home": round(float(p_home), 4),
                "away": round(float(p_away), 4),
            },
            "blend_alpha": round(float(_nba_alpha), 3),
            "history_rows": {
                "home": home_history_rows,
                "away": away_history_rows,
            },
            "structural_available": bool(basketball_blend.structural),
            "regime": basketball_blend.regime,
            "disagreement_pp": round(float(basketball_blend.disagreement) * 100, 1),
            "classifier_weight": round(float(basketball_blend.model_weight), 3),
            "structural_weight": round(float(basketball_blend.structural_weight), 3),
            "context_probability_adjustment": basketball_context_probability_adjustment,
            "home_lineup_confirmed": ctx.get("home_lineup_confirmed"),
            "away_lineup_confirmed": ctx.get("away_lineup_confirmed"),
            "home_projected_starters_count": int(ctx.get("home_projected_starters_count", 0) or 0),
            "away_projected_starters_count": int(ctx.get("away_projected_starters_count", 0) or 0),
            "rest_advantage": int(ctx.get("rest_advantage", 0) or 0),
        }
        game_record["model_available"] = True
        game_record["model_pick"] = home_team if p_home >= p_away else away_team
        game_record["basketball_probability_debug"] = basketball_probability_debug

        home_med = median_odds(game, home_team)
        away_med = median_odds(game, away_team)
        home_factors = _probability_factors_for_snapshot("basketball", combined, home_team=home_team, away_team=away_team, selection="home")
        away_factors = _probability_factors_for_snapshot("basketball", combined, home_team=home_team, away_team=away_team, selection="away")
        home_adjustments = _context_adjustments("basketball", "home", combined, ctx)
        away_adjustments = _context_adjustments("basketball", "away", combined, ctx)
        game_info = {"sport": "basketball", "home": home_team, "away": away_team,
                     "commence": commence, "kick_off": _kick_off_label(commence),
                     "window": window, "league": "NBA", "league_key": "basketball_nba",
                     "event_id": game.get("id", ""),
                     "odds_snapshot_age_hours": game.get("_odds_snapshot_age_hours"),
                     "odds_source_status": game.get("_odds_source_status", ""),
                     "odds_fetched_at": game.get("_odds_fetched_at"),
                     "bookmaker_last_update": game.get("_odds_bookmaker_last_update"),
                     "cache_loaded_at": game.get("_odds_cache_loaded_at"),
                     "cache_age_hours": game.get("_odds_cache_age_hours"),
                     "odds_age_basis": game.get("_odds_age_basis", ""),
                     "odds_source_detail": game.get("_odds_source_detail", ""),
                     "odds_source_reason": game.get("_odds_source_reason", ""),
                     "force_fresh_odds_active": bool(game.get("_odds_force_fresh_requested", _FORCE_FRESH_ODDS)),
                     "odds_cache_used": bool(game.get("_odds_cache_used")),
                     "odds_fallback_used": bool(game.get("_odds_fallback_used")),
                     **_feature_freshness_context("basketball"),
                     **ctx}
        game_info["basketball_probability_debug"] = basketball_probability_debug

        # Single bets — today only
        for bet, bk_name in [
            (build_value_bet(home_team, p_home, home_odds, fp_home,
                             stale_line=home_stale, median_price=home_med, sport="basketball",
                             raw_model_prob=raw_home_prob, prediction_factors=home_factors,
                             context_adjustments=home_adjustments, availability_context=ctx), home_bk),
            (build_value_bet(away_team, p_away, away_odds, fp_away,
                             stale_line=away_stale, median_price=away_med, sport="basketball",
                             raw_model_prob=raw_away_prob, prediction_factors=away_factors,
                             context_adjustments=away_adjustments, availability_context=ctx), away_bk),
        ]:
            if bet and window in ("today", "tomorrow", "day_after"):
                enriched_bet = _with_candidate_odds_diagnostics(
                    {**game_info, **bet, "bookmaker": bk_name, "league_key": "basketball_nba"},
                    game=game,
                    market_key="h2h",
                    selection_name=str(bet.get("team", "") or ""),
                    bookmaker_name=str(bk_name or ""),
                )
                value_bets.append(enrich_with_capability(
                    enriched_bet,
                    sport="basketball",
                    league="NBA",
                    sport_key="basketball_nba",
                ))

        # Parlay legs — both windows
        for team, ml_prob, team_odds, fair_prob in [
            (home_team, p_home, home_odds, fp_home),
            (away_team, p_away, away_odds, fp_away),
        ]:
            if team_odds and team_odds > 1.05:
                _leg = to_parlay_leg("basketball", home_team, away_team, team,
                                     team_odds, ml_prob, fair_prob, commence, window=window)
                if _leg:
                    parlay_legs.append(_leg)

        # ── NBA totals (Over/Under points) — analytical model ─────────────────
        if window in ("today", "tomorrow", "day_after"):
            try:
                _nba_totals_raw = fetch_odds("basketball_nba", markets="totals", priority="optional")
                _nba_totals_map = {g["id"]: g for g in _nba_totals_raw if "id" in g}
            except Exception:
                _nba_totals_map = {}

            totals_game = _nba_totals_map.get(game.get("id"), {})
            game_record["available_market_keys"] = _merge_market_keys(
                game_record.get("available_market_keys"),
                _extract_market_keys_from_odds_game(totals_game),
            )
            _best_over: dict = {}
            _best_under: dict = {}
            for bk in totals_game.get("bookmakers", []):
                for mkt in bk.get("markets", []):
                    if mkt.get("key") != "totals":
                        continue
                    for o in mkt.get("outcomes", []):
                        nm = o.get("name", "")
                        pr = float(o.get("price", 0))
                        pt = o.get("point")
                        if pr <= 1.0 or pt is None:
                            continue
                        ln = float(pt)
                        if nm.lower() == "over":
                            _best_over[ln]  = max(_best_over.get(ln, 0), pr)
                        elif nm.lower() == "under":
                            _best_under[ln] = max(_best_under.get(ln, 0), pr)

            # Use the most common NBA line from market
            _h_ppg     = float(combined.get("home_ppg", 0))
            _a_ppg     = float(combined.get("away_ppg", 0))
            _h_opp_ppg = float(combined.get("home_opp_ppg", 0))
            _a_opp_ppg = float(combined.get("away_opp_ppg", 0))

            for _line in sorted(_best_over.keys(), key=lambda x: abs(x - 225)):
                _ov = _best_over.get(_line)
                _un = _best_under.get(_line)
                if _ov and _un and _h_ppg > 0:
                    totals_results = ml_totals_bet(
                        "basketball", home_team, away_team, combined,
                        feature_cols, None,  # analytical, no ML model needed
                        over_odds=_ov, under_odds=_un, line=_line,
                        window=window, commence=commence, league="NBA",
                        home_ppg=_h_ppg, away_ppg=_a_ppg,
                        home_opp_ppg=_h_opp_ppg, away_opp_ppg=_a_opp_ppg,
                    )
                    if totals_results:
                        for tb in totals_results:
                            value_bets.append(tb)
                    break

        # ── NBA spreads ────────────────────────────────────────────────────────
        # NBA spread model not available (no actual game scores in training data)
        # Spreads are implicitly captured by the h2h model margin features.

    return value_bets, parlay_legs


# ─────────────────────────────────────────────────────────────────────────────
# Tennis pipeline
# ─────────────────────────────────────────────────────────────────────────────

# ATP sport keys available on The Odds API
_TENNIS_ODDS_KEYS = [
    "tennis_atp_monte_carlo_masters",
    "tennis_atp_french_open",
    "tennis_atp_wimbledon",
    "tennis_atp_us_open",
    "tennis_atp_australian_open",
    "tennis_atp_madrid_open",
    "tennis_atp_rome",
    "tennis_atp_canadian_open",
    "tennis_atp_cincinnati",
    "tennis_atp_paris_masters",
    "tennis_wta_french_open",
    "tennis_wta_wimbledon",
]

_TENNIS_DROP_COLS = [
    "match_id", "date", "tourney_name", "tourney_level_name", "surface",
    "round", "player1_name", "player1_id", "player2_name", "player2_id",
    "player1_hand", "player2_hand", "player1_seed", "player2_seed", "score",
    "result",
    "p1_ace", "p1_df", "p1_svpt", "p1_1stIn", "p1_1stWon", "p1_2ndWon",
    "p1_bpSaved", "p1_bpFaced",
    "p2_ace", "p2_df", "p2_svpt", "p2_1stIn", "p2_1stWon", "p2_2ndWon",
    "p2_bpSaved", "p2_bpFaced",
    "p1_1st_pct", "p1_ace_rate", "p1_bp_save",
    "p2_1st_pct", "p2_ace_rate", "p2_bp_save",
]


def _build_tennis_context(
    fdf: pd.DataFrame,
    tag: str,
    alpha_penalty_calibrated: bool,
    *,
    sport_key: str = "tennis",
) -> Optional[dict]:
    """
    Build a self-contained inference context for one tour (ATP or WTA).
    Returns a dict with everything needed to score a match, or None on failure.
    """
    if fdf.empty:
        return None

    y_all = fdf["target"] if "target" in fdf.columns else pd.Series(dtype=int)

    trainer = ModelTrainer(sport="tennis")
    trainer.load_models(tag=tag)
    if not trainer.trained_models:
        logger.warning("No saved tennis models for tag %s", tag)
        return None

    sample_model = next(iter(trainer.trained_models.values()))
    feature_cols = list(sample_model.feature_names_in_)
    missing = [c for c in feature_cols if c not in fdf.columns]
    if missing:
        for c in missing:
            fdf[c] = 0.0
    X_all = fdf[feature_cols].fillna(0)
    if not y_all.empty:
        X_all = X_all.loc[y_all.index]

    trainer.ensemble_model = _SoftVotingWrapper(
        estimators=list(trainer.trained_models.items()),
        weights=None,
        classes=np.array([0, 1]),
    )
    cal = EnsembleCalibrator.load(calibrator_path_for_tag("tennis", tag))
    alpha = _effective_alpha(sport_key, calibrated=cal is not None)

    all_players: set = (
        set(fdf["player1_name"].dropna().astype(str).unique())
        | set(fdf["player2_name"].dropna().astype(str).unique())
    )
    aliases = build_entity_alias_map(all_players)

    def _swap(rows: pd.DataFrame) -> pd.DataFrame:
        rename_map = {}
        for col in rows.columns:
            if col.startswith("player1_"):
                rename_map[col] = f"player2_{col[8:]}"
            elif col.startswith("player2_"):
                rename_map[col] = f"player1_{col[8:]}"
            elif col.startswith("p1_"):
                rename_map[col] = f"p2_{col[3:]}"
            elif col.startswith("p2_"):
                rename_map[col] = f"p1_{col[3:]}"
            else:
                rename_map[col] = col
        return rows.rename(columns=rename_map).reindex(columns=X_all.columns)

    def get_snapshot(player_name: str, as_p1: bool, surface: str = "Hard") -> Optional[pd.Series]:
        canonical = resolve_canonical_name(player_name, all_players, alias_map=aliases)
        p1_rows = X_all.loc[fdf["player1_name"] == canonical]
        p2_rows = X_all.loc[fdf["player2_name"] == canonical]

        normalized = []
        if as_p1:
            if not p1_rows.empty:
                normalized.append(p1_rows)
            if not p2_rows.empty:
                normalized.append(_swap(p2_rows))
        else:
            if not p2_rows.empty:
                normalized.append(p2_rows)
            if not p1_rows.empty:
                normalized.append(_swap(p1_rows))
        rows = pd.concat(normalized, ignore_index=False) if normalized else pd.DataFrame(columns=X_all.columns)

        if len(rows) < 3:
            logger.info("SKIP snapshot [%s]: only %d rows — insufficient history", player_name, len(rows))
            return None

        snap = rows.tail(10).mean()
        for surf in ("hard", "clay", "grass"):
            col = f"surface_{surf}"
            if col in snap.index:
                snap[col] = 1.0 if surf == surface.lower() else 0.0
        surf_col = "p1_surface_win" if as_p1 else "p2_surface_win"
        if surf_col in snap.index and "surface" in fdf.columns:
            surf_rows = rows.loc[fdf.loc[rows.index, "surface"] == surface]
            if len(surf_rows) >= 2:
                snap[surf_col] = float(surf_rows[surf_col].dropna().tail(10).mean())
        return snap

    def get_player_profile(player_name: str) -> dict:
        canonical = resolve_canonical_name(player_name, all_players, alias_map=aliases)
        p1_rows = fdf.loc[fdf["player1_name"] == canonical]
        p2_rows = fdf.loc[fdf["player2_name"] == canonical]

        def _latest(series_list: list[pd.Series], default=None):
            for series in series_list:
                clean = series.dropna()
                if not clean.empty:
                    return clean.iloc[-1]
            return default

        seeded_raw = _latest([p1_rows.get("player1_seed", pd.Series(dtype=object)), p2_rows.get("player2_seed", pd.Series(dtype=object))], default=None)
        return {
            "rank": _latest([p1_rows.get("player1_rank", pd.Series(dtype=float)), p2_rows.get("player2_rank", pd.Series(dtype=float))], default=None),
            "rank_pts": _latest([p1_rows.get("player1_rank_pts", pd.Series(dtype=float)), p2_rows.get("player2_rank_pts", pd.Series(dtype=float))], default=None),
            "age": _latest([p1_rows.get("player1_age", pd.Series(dtype=float)), p2_rows.get("player2_age", pd.Series(dtype=float))], default=None),
            "height": _latest([p1_rows.get("player1_ht", pd.Series(dtype=float)), p2_rows.get("player2_ht", pd.Series(dtype=float))], default=None),
            "seeded": None if seeded_raw is None else not pd.isna(seeded_raw),
        }

    def get_head_to_head(player1_name: str, player2_name: str, max_matches: int = 10) -> dict:
        p1 = resolve_canonical_name(player1_name, all_players, alias_map=aliases)
        p2 = resolve_canonical_name(player2_name, all_players, alias_map=aliases)
        mask = (
            ((fdf["player1_name"] == p1) & (fdf["player2_name"] == p2))
            | ((fdf["player1_name"] == p2) & (fdf["player2_name"] == p1))
        )
        rows = fdf.loc[mask].tail(max_matches)
        if rows.empty or "result" not in rows.columns:
            return {"p1_wins": 0.0, "total": 0.0}

        p1_wins = 0.0
        total = 0.0
        for _, row in rows.iterrows():
            total += 1.0
            if row["player1_name"] == p1 and row["result"] == "player1_win":
                p1_wins += 1.0
            elif row["player2_name"] == p1 and row["result"] == "player2_win":
                p1_wins += 1.0
        return {"p1_wins": p1_wins, "total": total}

    return {
        "fdf": fdf, "X_all": X_all, "feature_cols": feature_cols,
        "trainer": trainer, "cal": cal, "alpha": alpha,
        "get_snapshot": get_snapshot,
        "get_player_profile": get_player_profile,
        "get_head_to_head": get_head_to_head,
    }


def run_tennis(dry_run: bool = False) -> Tuple[List[dict], List[ParlayLeg]]:
    """Fetch ATP/WTA data, predict match outcomes, scan live tennis odds."""
    global _other_sport_games, _wta_audit_rows
    logger.info("=== TENNIS PIPELINE ===")
    _wta_audit_rows = []

    # ── ATP context ───────────────────────────────────────────────────────────
    fdf_atp, _ = _load_features_cached(
        "tennis",
        "data/cache/tennis_features.parquet",
        lambda: TennisFetcher(seasons=[2022, 2023, 2024, 2025, 2026], tours=["atp"]),
        TennisFeatureEngineer,
    )
    _tennis_tag = get_current_model_tag("tennis", fallback="atp_2022_25")
    atp_ctx = _build_tennis_context(
        fdf_atp,
        _tennis_tag,
        alpha_penalty_calibrated=True,
        sport_key="tennis",
    )
    if atp_ctx is None:
        logger.warning("No tennis ATP data or models — skipping.")
        return [], []
    logger.info("ATP data: %d matches (cached)", len(fdf_atp))

    # ── WTA context (optional) ────────────────────────────────────────────────
    wta_ctx = None
    tennis_model_files = list(Path("data/models/tennis").glob("*.joblib"))
    if _FOCUSED_SCAN_ONLY:
        logger.info("Focused-lanes scan: skipping WTA model context because only ATP tennis moneyline is focused")
    elif any("wta" in p.name.lower() for p in tennis_model_files):
        fdf_wta, _ = _load_features_cached(
            "tennis_wta",
            "data/cache/tennis_wta_features.parquet",
            lambda: TennisFetcher(seasons=[2022, 2023, 2024, 2025, 2026], tours=["wta"]),
            TennisFeatureEngineer,
        )
        _wta_tag = get_current_model_tag("tennis", fallback="wta_2022_24")
        # WTA tag is stored separately; try wta-specific tag first
        from src.models.artifacts import model_dir_for_sport as _tmd
        _wta_tag_path = _tmd("tennis") / "current_tag_wta.txt"
        if _wta_tag_path.exists():
            _wta_tag = _wta_tag_path.read_text().strip()
        else:
            # Infer from saved model files
            wta_model_files = [p for p in tennis_model_files if "wta" in p.name.lower() and p.name.startswith("xgboost")]
            if wta_model_files:
                import re as _re
                m = _re.search(r"xgboost_(.+)\.joblib", wta_model_files[0].name)
                _wta_tag = m.group(1) if m else "wta_2022_24"
        wta_ctx = _build_tennis_context(
            fdf_wta,
            _wta_tag,
            alpha_penalty_calibrated=True,
            sport_key="tennis_wta",
        )
        if wta_ctx:
            logger.info("WTA data: %d matches (cached), tag=%s", len(fdf_wta), _wta_tag)
        else:
            logger.warning("WTA models not usable — WTA events will be skipped")

    if dry_run:
        logger.info("[dry-run] Skipping tennis odds fetch.")
        return [], []

    # Discover active ATP and WTA markets
    active_keys = []
    try:
        active_sports = _prefetch_active_sports()
        if active_sports:
            active_keys = sorted([
                k for k in active_sports
                if k.startswith("tennis_atp_") or k.startswith("tennis_wta_")
            ])
        # Drop WTA keys if no WTA context is available
        if wta_ctx is None:
            skipped = [k for k in active_keys if k.startswith("tennis_wta_")]
            if skipped:
                logger.info("WTA scan skipped — no WTA model available (%d market(s))", len(skipped))
            active_keys = [k for k in active_keys if not k.startswith("tennis_wta_")]
        if _FOCUSED_SCAN_ONLY:
            skipped = [k for k in active_keys if k.startswith("tennis_wta_")]
            if skipped:
                logger.info("Focused-lanes scan: skipping WTA markets (%d) because only ATP tennis moneyline is focused", len(skipped))
            active_keys = [k for k in active_keys if not k.startswith("tennis_wta_")]
        logger.info("Active tennis markets: %s", active_keys)
    except Exception as exc:
        logger.warning("Could not fetch active sports: %s", exc)

    value_bets: List[dict] = []
    parlay_legs: List[ParlayLeg] = []

    for sport_key in active_keys:
        is_wta = sport_key.startswith("tennis_wta_")
        ctx_tour = wta_ctx if is_wta else atp_ctx
        if ctx_tour is None:
            continue

        feature_cols   = ctx_tour["feature_cols"]
        trainer        = ctx_tour["trainer"]
        _cal           = ctx_tour["cal"]
        _alpha         = ctx_tour["alpha"]
        get_snapshot   = ctx_tour["get_snapshot"]
        get_player_profile = ctx_tour["get_player_profile"]
        get_head_to_head = ctx_tour["get_head_to_head"]
        _fdf           = ctx_tour["fdf"]

        try:
            games = fetch_odds(sport_key)
        except Exception as exc:
            logger.warning("Could not fetch %s: %s", sport_key, exc)
            continue

        tournament_name = (
            sport_key.replace("tennis_atp_", "ATP ")
                     .replace("tennis_wta_", "WTA ")
                     .replace("_", " ").title()
        )
        logger.info("%s: %d upcoming matches", tournament_name, len(games))

        for game in games:
            p1_name = game["home_team"]
            p2_name = game["away_team"]
            commence = game["commence_time"]
            window = game.get("_window", "today")

            p1_odds, p1_bk, p1_stale = best_odds(game, p1_name)
            p2_odds, p2_bk, p2_stale = best_odds(game, p2_name)

        _other_sport_games.append(enrich_with_capability({
            "sport": "tennis_wta" if is_wta else "tennis",
            "home": p1_name, "away": p2_name,
            "league": tournament_name, "commence": commence,
            "kick_off": _kick_off_label(commence), "window": window,
            "home_odds": p1_odds, "away_odds": p2_odds,
            "league_key": sport_key,
            "available_market_keys": _extract_market_keys_from_odds_game(game),
        }, sport="tennis_wta" if is_wta else "tennis", league=tournament_name, sport_key=sport_key))

        if p1_odds is None or p2_odds is None:
            continue

        _key_lower = sport_key.lower()
        if any(t in _key_lower for t in ("clay", "roland_garros", "barcelona",
                                          "monte_carlo", "rome", "madrid",
                                          "hamburg", "gstaad", "bastad",
                                          "umag", "kitzbuhel", "bucharest")):
            match_surface = "Clay"
        elif any(t in _key_lower for t in ("wimbledon", "queens", "halle",
                                            "grass", "eastbourne", "hertogenbosch")):
            match_surface = "Grass"
        else:
            match_surface = "Hard"

            p1_snap = get_snapshot(p1_name, as_p1=True, surface=match_surface)
            p2_snap = get_snapshot(p2_name, as_p1=False, surface=match_surface)
            if p1_snap is None or p2_snap is None:
                logger.info("SKIP %s vs %s: insufficient player history", p1_name, p2_name)
                continue

            combined = p1_snap.copy()
            for col in [c for c in feature_cols if c.startswith("p2_") or c.startswith("player2_")]:
                if col in p2_snap.index:
                    combined[col] = p2_snap[col]
            combined = _refresh_tennis_matchup_snapshot(
                combined,
                p1_profile=get_player_profile(p1_name),
                p2_profile=get_player_profile(p2_name),
                h2h=get_head_to_head(p1_name, p2_name),
            )

            _tennis_sport_label = "tennis_wta" if is_wta else "tennis"

            ctx = _detect_contextual_flags(game, "tennis")
            ctx = {
                **ctx,
                **build_availability_context(_tennis_sport_label, game, combined),
                **build_environment_context(_tennis_sport_label, p1_name, p2_name, commence),
            }

            x = pd.DataFrame([combined], columns=feature_cols)
            raw_p = trainer.ensemble_model.predict_proba(x)
            proba = _cal.transform(raw_p)[0] if _cal else raw_p[0]
            p_p1 = float(proba[1])
            p_p2 = float(proba[0])
            p_p1, p_p2, tennis_context_probability_adjustment = _apply_tennis_context_probability_adjustment(
                p_p1,
                p_p2,
                combined,
                ctx,
            )
            raw_p1_prob = p_p1
            raw_p2_prob = p_p2

            fp_p1, fp_p2 = vig_free_prob(p1_odds, p2_odds)
            pre_market_blend_p1 = p_p1
            pre_market_blend_p2 = p_p2
            p_p1 = _winsorize_prob(p_p1, fp_p1, _tennis_sport_label)
            p_p2 = _winsorize_prob(p_p2, fp_p2, _tennis_sport_label)
            _tennis_live_alpha = _market_health_adjusted_alpha(
                _tennis_sport_label,
                "moneyline",
                _alpha,
            )
            p_p1 = _blend(p_p1, fp_p1, alpha=_tennis_live_alpha)
            p_p2 = _blend(p_p2, fp_p2, alpha=_tennis_live_alpha)
            p_p1 = _apply_tennis_rank_gap_damping(p_p1, fp_p1, combined, _tennis_sport_label)
            p_p2 = _apply_tennis_rank_gap_damping(p_p2, fp_p2, combined, _tennis_sport_label)
            total_prob = p_p1 + p_p2
            if total_prob > 0:
                p_p1 /= total_prob
                p_p2 /= total_prob
            tennis_probability_debug = {
                "classifier_probs": {
                    "player1": round(float(proba[1]), 4),
                    "player2": round(float(proba[0]), 4),
                },
                "market_probs": {
                    "player1": round(float(fp_p1), 4),
                    "player2": round(float(fp_p2), 4),
                },
                "pre_market_blend_probs": {
                    "player1": round(float(pre_market_blend_p1), 4),
                    "player2": round(float(pre_market_blend_p2), 4),
                },
                "final_probs": {
                    "player1": round(float(p_p1), 4),
                    "player2": round(float(p_p2), 4),
                },
                "blend_alpha": round(float(_tennis_live_alpha), 3),
                "context_probability_adjustment": tennis_context_probability_adjustment,
            }

            if is_wta:
                _record_wta_audit_match(
                    home=p1_name,
                    away=p2_name,
                    league=tournament_name,
                    commence=commence,
                    home_odds=float(p1_odds),
                    away_odds=float(p2_odds),
                    home_prob=float(p_p1),
                    away_prob=float(p_p2),
                    review_only=True,
                    min_edge=float(KELLY.min_edge),
                )

            p1_factors = _probability_factors_for_snapshot(_tennis_sport_label, combined, home_team=p1_name, away_team=p2_name, selection="player1")
            p2_factors = _probability_factors_for_snapshot(_tennis_sport_label, combined, home_team=p1_name, away_team=p2_name, selection="player2")
            p1_adjustments = _context_adjustments(_tennis_sport_label, "player1", combined, ctx)
            p2_adjustments = _context_adjustments(_tennis_sport_label, "player2", combined, ctx)

            game_info = {
                "sport": _tennis_sport_label,
                "tournament": tournament_name,
                "home": p1_name, "away": p2_name,
                "commence": commence,
                "kick_off": _kick_off_label(commence),
                "window": window,
                "league": tournament_name,
                "event_id": game.get("id", ""),
                "odds_snapshot_age_hours": game.get("_odds_snapshot_age_hours"),
                "odds_source_status": game.get("_odds_source_status", ""),
                "odds_fetched_at": game.get("_odds_fetched_at"),
                "bookmaker_last_update": game.get("_odds_bookmaker_last_update"),
                "cache_loaded_at": game.get("_odds_cache_loaded_at"),
                "cache_age_hours": game.get("_odds_cache_age_hours"),
                "odds_age_basis": game.get("_odds_age_basis", ""),
                "odds_source_detail": game.get("_odds_source_detail", ""),
                "odds_source_reason": game.get("_odds_source_reason", ""),
                "force_fresh_odds_active": bool(game.get("_odds_force_fresh_requested", _FORCE_FRESH_ODDS)),
                "odds_cache_used": bool(game.get("_odds_cache_used")),
                "odds_fallback_used": bool(game.get("_odds_fallback_used")),
                "tennis_probability_debug": tennis_probability_debug,
                **_feature_freshness_context(_tennis_sport_label),
                **ctx,
            }

            for team, ml_prob, team_odds, fair_prob, bk_name in [
                (p1_name, p_p1, p1_odds, fp_p1, p1_bk),
                (p2_name, p_p2, p2_odds, fp_p2, p2_bk),
            ]:
                bet = build_value_bet(
                    team, ml_prob, team_odds, fair_prob,
                    sport=_tennis_sport_label,
                    # Tennis now routes its tightened, market-anchored probability
                    # into the true-probability layer so anomaly checks reflect the
                    # live publish path rather than the pre-blend raw ensemble.
                    raw_model_prob=ml_prob,
                    prediction_factors=p1_factors if team == p1_name else p2_factors,
                    context_adjustments=p1_adjustments if team == p1_name else p2_adjustments,
                    availability_context=ctx,
                )
                if bet:
                    enriched_bet = _with_candidate_odds_diagnostics(
                        {**game_info, **bet, "bookmaker": bk_name, "league_key": sport_key},
                        game=game,
                        market_key="h2h",
                        selection_name=str(team or ""),
                        bookmaker_name=str(bk_name or ""),
                    )
                    value_bets.append(enrich_with_capability(
                        enriched_bet,
                        sport="tennis_wta" if is_wta else "tennis",
                        league=tournament_name,
                        sport_key=sport_key,
                    ))

            for team, ml_prob, team_odds, fair_prob in [
                (p1_name, p_p1, p1_odds, fp_p1),
                (p2_name, p_p2, p2_odds, fp_p2),
            ]:
                if team_odds and team_odds > 1.05:
                    _leg = to_parlay_leg(
                        "tennis", p1_name, p2_name, team,
                        team_odds, ml_prob, fair_prob, commence,
                    )
                    if _leg:
                        parlay_legs.append(_leg)

    return value_bets, parlay_legs


# ─────────────────────────────────────────────────────────────────────────────
# MLB pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_mlb(dry_run: bool = False) -> Tuple[List[dict], List[ParlayLeg]]:
    global _other_sport_games
    logger.info("=== MLB PIPELINE ===")
    features_df, engineer = _load_features_cached(
        "mlb",
        "data/cache/mlb_features.parquet",
        MLBFetcher,
        MLBFeatureEngineer,
    )
    if features_df.empty:
        logger.warning("No MLB data available.")
        return [], []
    logger.info("MLB data: %d games (cached)", len(features_df))

    y_all = features_df["target"].dropna()

    _mlb_tag = get_current_model_tag("mlb", fallback="mlb_2024_25")
    trainer = ModelTrainer(sport="mlb")
    trainer.load_models(tag=_mlb_tag)
    if not trainer.trained_models:
        logger.info("No saved MLB models found; training now …")
        _always_drop = ["result", "home_score", "away_score", "home_hits", "away_hits",
                        "home_errors", "away_errors", "home_innings", "away_innings",
                        "game_pk", "date", "season", "home_team", "away_team", "target"]
        X_tmp = features_df.drop(columns=[c for c in _always_drop if c in features_df.columns],
                                  errors="ignore").select_dtypes(include=[np.number]).fillna(0)
        X_tmp = X_tmp.loc[y_all.index]
        split = int(len(X_tmp) * 0.8)
        trainer.train(X_tmp.iloc[:split], y_all.iloc[:split],
                      X_tmp.iloc[split:], y_all.iloc[split:])
        trainer.save_models(tag=_mlb_tag)

    sample_model = next(iter(trainer.trained_models.values()))
    feature_cols = list(sample_model.feature_names_in_)
    missing = [c for c in feature_cols if c not in features_df.columns]
    if missing:
        logger.warning("Adding zero-filled missing MLB features: %s", missing)
        for c in missing:
            features_df[c] = 0.0
    X_all = features_df[feature_cols].fillna(0)
    X_all = X_all.loc[y_all.index]

    trainer.ensemble_model = _SoftVotingWrapper(
        estimators=list(trainer.trained_models.items()),
        weights=None,
        classes=np.array(sorted(y_all.unique())),
    )

    _mlb_cal = None
    _cal_path = calibrator_path_for_tag("mlb", _mlb_tag)
    if _cal_path.exists():
        _mlb_cal = EnsembleCalibrator.load(_cal_path)
    _mlb_alpha = _effective_alpha("mlb", calibrated=_mlb_cal is not None)

    _mlb_totals_model  = TotalsTrainer.load(Path("data/models/mlb/totals_mlb.joblib"))
    _mlb_spreads_model = TotalsTrainer.load(Path("data/models/mlb/spreads_mlb.joblib"))

    label_map = engineer.label_map   # {0: 'away_win', 1: 'draw', 2: 'home_win'}
    inv_map = {v: k for k, v in label_map.items()}

    # Entity resolver for MLB — handles any future API naming changes via JSON
    _mlb_resolver = TeamResolver("mlb")

    def get_team_snapshot_mlb(team: str, as_home: bool) -> Optional[pd.Series]:
        col = "home_team" if as_home else "away_team"
        norm = _mlb_resolver.resolve(team)
        rows = X_all.loc[features_df[col] == norm]
        row_count = len(rows)
        if row_count < _MIN_SNAPSHOT_ROWS:
            logger.warning("[mlb] Only %d rows for %s (norm=%s) — skipping", row_count, team, norm)
            return None
        snap = rows.tail(10).mean()
        snap.attrs["history_rows"] = row_count
        snap.attrs["resolved_team"] = norm
        return snap

    if dry_run:
        logger.info("[dry-run] Skipping odds fetch for MLB.")
        return [], []

    try:
        mlb_games = fetch_odds("baseball_mlb")
    except Exception as exc:
        logger.warning("Could not fetch MLB odds: %s", exc)
        return [], []

    today_cnt     = sum(1 for g in mlb_games if g.get("_window") == "today")
    overnight_cnt = sum(1 for g in mlb_games if g.get("_window") == "overnight")
    tomorrow_cnt  = sum(1 for g in mlb_games if g.get("_window") == "tomorrow")
    day_after_cnt = sum(1 for g in mlb_games if g.get("_window") == "day_after")
    logger.info("MLB: %d upcoming (today=%d overnight=%d tomorrow=%d day_after=%d)",
                len(mlb_games), today_cnt, overnight_cnt, tomorrow_cnt, day_after_cnt)

    # Fetch totals (run line / over-under)
    try:
        mlb_totals_games = fetch_odds("baseball_mlb", markets="totals", priority="optional")
        mlb_totals_map = {g["id"]: g for g in mlb_totals_games if "id" in g}
    except Exception:
        mlb_totals_map = {}

    value_bets: List[dict] = []
    parlay_legs: List[ParlayLeg] = []

    for game in mlb_games:
        home_team = game["home_team"]
        away_team = game["away_team"]
        commence  = game["commence_time"]
        window    = game.get("_window", "today")

        home_odds, home_bk, home_stale = best_odds(game, home_team)
        away_odds, away_bk, away_stale = best_odds(game, away_team)

        if home_odds is None or away_odds is None:
            continue

        h_snap = get_team_snapshot_mlb(home_team, as_home=True)
        a_snap = get_team_snapshot_mlb(away_team, as_home=False)
        if h_snap is None or a_snap is None:
            logger.info("SKIP %s vs %s: insufficient MLB team history", home_team, away_team)
            continue
        home_history_rows = int(h_snap.attrs.get("history_rows", 0) or 0)
        away_history_rows = int(a_snap.attrs.get("history_rows", 0) or 0)
        combined = h_snap.copy()
        combined.attrs["home_history_rows"] = home_history_rows
        combined.attrs["away_history_rows"] = away_history_rows
        for col in [c for c in feature_cols if c.startswith("away_")]:
            if col in a_snap.index:
                combined[col] = a_snap[col]

        # Add rest days to game dict for contextual flags (use defaults if not available)
        game["home_rest_days"] = float(h_snap.get("home_rest_days", 7))
        game["away_rest_days"] = float(a_snap.get("away_rest_days", 7))
        ctx = _detect_contextual_flags(game, "mlb")
        ctx = {
            **ctx,
            **build_availability_context("mlb", game, combined),
            **build_environment_context("mlb", home_team, away_team, commence),
        }

        x = pd.DataFrame([combined], columns=feature_cols)
        raw_p = trainer.ensemble_model.predict_proba(x)
        proba = _mlb_cal.transform(raw_p)[0] if _mlb_cal else raw_p[0]

        home_idx = inv_map.get("home_win", 2)
        away_idx = inv_map.get("away_win", 0)
        p_home = float(proba[home_idx]) if home_idx < len(proba) else 0.50
        p_away = float(proba[away_idx]) if away_idx < len(proba) else 0.50
        mlb_blend = _MLB_SIDE_MODEL.combine_with_classifier_diagnostics(
            (p_home, p_away),
            _MLB_SIDE_MODEL.structural_probs_from_snapshot(combined),
        )
        p_home, p_away = mlb_blend.combined.as_tuple()
        p_home, p_away, mlb_context_probability_adjustment = _apply_mlb_context_probability_adjustment(p_home, p_away, ctx)
        raw_home_prob = p_home
        raw_away_prob = p_away

        # No draw in MLB; vig-free market is 2-outcome
        fp_home, fp_away = vig_free_prob(home_odds, away_odds)

        pre_market_blend_home = p_home
        pre_market_blend_away = p_away

        # Market-anchor blend — MLB uses 0.55 (moderate trust in model)
        # Winsorize to within 25pp of market before blending
        _mlb_live_alpha = _live_blend_alpha(
            "mlb",
            "moneyline",
            _mlb_alpha,
            home_rows=home_history_rows,
            away_rows=away_history_rows,
            disagreement_pp=float(mlb_blend.disagreement) * 100.0,
        )
        p_home = _winsorize_prob(p_home, fp_home, "mlb")
        p_home = _blend(p_home, fp_home, alpha=_mlb_live_alpha)
        p_away = 1.0 - p_home
        mlb_probability_debug = {
            "classifier_probs": {
                "home": round(mlb_blend.classifier.home, 4),
                "away": round(mlb_blend.classifier.away, 4),
            },
            "structural_probs": {
                "home": round(mlb_blend.structural.home, 4),
                "away": round(mlb_blend.structural.away, 4),
            } if mlb_blend.structural else None,
            "market_probs": {
                "home": round(float(fp_home), 4),
                "away": round(float(fp_away), 4),
            },
            "pre_market_blend_probs": {
                "home": round(float(pre_market_blend_home), 4),
                "away": round(float(pre_market_blend_away), 4),
            },
            "final_probs": {
                "home": round(float(p_home), 4),
                "away": round(float(p_away), 4),
            },
            "blend_alpha": round(float(_mlb_live_alpha), 3),
            "history_rows": {
                "home": home_history_rows,
                "away": away_history_rows,
            },
            "structural_available": bool(mlb_blend.structural),
            "regime": mlb_blend.regime,
            "disagreement_pp": round(float(mlb_blend.disagreement) * 100, 1),
            "classifier_weight": round(float(mlb_blend.model_weight), 3),
            "structural_weight": round(float(mlb_blend.structural_weight), 3),
            "context_probability_adjustment": mlb_context_probability_adjustment,
            "home_starter_confirmed": ctx.get("home_starter_confirmed"),
            "away_starter_confirmed": ctx.get("away_starter_confirmed"),
            "home_starter_name": str(ctx.get("home_starter_name", "") or "").strip(),
            "away_starter_name": str(ctx.get("away_starter_name", "") or "").strip(),
            "home_lineup_confirmed": ctx.get("home_lineup_confirmed"),
            "away_lineup_confirmed": ctx.get("away_lineup_confirmed"),
            "home_likely_starters_count": int(ctx.get("home_likely_starters_count", 0) or 0),
            "away_likely_starters_count": int(ctx.get("away_likely_starters_count", 0) or 0),
            "weather_risk": int(ctx.get("weather_risk", 0) or 0),
            "wind_mph": ctx.get("wind_mph"),
        }

        _other_sport_games.append(enrich_with_capability({
            "sport": "mlb", "home": home_team, "away": away_team,
            "league": "MLB", "commence": commence,
            "kick_off": _kick_off_label(commence), "window": window,
            "home_odds": home_odds, "away_odds": away_odds,
            "league_key": "baseball_mlb",
            "available_market_keys": _merge_market_keys(
                _extract_market_keys_from_odds_game(game),
                _extract_market_keys_from_odds_game(mlb_totals_map.get(game.get("id"), {})),
            ),
            "model_available": True,
            "model_pick": home_team if p_home >= p_away else away_team,
            "mlb_probability_debug": mlb_probability_debug,
        }, sport="mlb", league="MLB", sport_key="baseball_mlb"))

        home_med = median_odds(game, home_team)
        away_med = median_odds(game, away_team)
        home_factors = _probability_factors_for_snapshot("mlb", combined, home_team=home_team, away_team=away_team, selection="home")
        away_factors = _probability_factors_for_snapshot("mlb", combined, home_team=home_team, away_team=away_team, selection="away")
        home_adjustments = _context_adjustments("mlb", "home", combined, ctx)
        away_adjustments = _context_adjustments("mlb", "away", combined, ctx)
        game_info = {"sport": "mlb", "home": home_team, "away": away_team,
                     "commence": commence, "kick_off": _kick_off_label(commence),
                     "window": window, "league": "MLB", "league_key": "baseball_mlb",
                     "event_id": game.get("id", ""),
                     "odds_snapshot_age_hours": game.get("_odds_snapshot_age_hours"),
                     "odds_source_status": game.get("_odds_source_status", ""),
                     "odds_fetched_at": game.get("_odds_fetched_at"),
                     "bookmaker_last_update": game.get("_odds_bookmaker_last_update"),
                     "cache_loaded_at": game.get("_odds_cache_loaded_at"),
                     "cache_age_hours": game.get("_odds_cache_age_hours"),
                     "odds_age_basis": game.get("_odds_age_basis", ""),
                     "odds_source_detail": game.get("_odds_source_detail", ""),
                     "odds_source_reason": game.get("_odds_source_reason", ""),
                     "force_fresh_odds_active": bool(game.get("_odds_force_fresh_requested", _FORCE_FRESH_ODDS)),
                     "odds_cache_used": bool(game.get("_odds_cache_used")),
                     "odds_fallback_used": bool(game.get("_odds_fallback_used")),
                     **_feature_freshness_context("mlb"),
                     **ctx}
        game_info["mlb_probability_debug"] = mlb_probability_debug

        # Single bets
        for bet, bk_name in [
            (build_value_bet(home_team, p_home, home_odds, fp_home,
                             stale_line=home_stale, median_price=home_med, sport="mlb",
                             raw_model_prob=raw_home_prob, prediction_factors=home_factors,
                             context_adjustments=home_adjustments, availability_context=ctx), home_bk),
            (build_value_bet(away_team, p_away, away_odds, fp_away,
                             stale_line=away_stale, median_price=away_med, sport="mlb",
                             raw_model_prob=raw_away_prob, prediction_factors=away_factors,
                             context_adjustments=away_adjustments, availability_context=ctx), away_bk),
        ]:
            if bet and window in ("today", "tomorrow", "day_after"):
                enriched_bet = _with_candidate_odds_diagnostics(
                    {**game_info, **bet, "bookmaker": bk_name},
                    game=game,
                    market_key="h2h",
                    selection_name=str(bet.get("team", "") or ""),
                    bookmaker_name=str(bk_name or ""),
                )
                value_bets.append(enrich_with_capability(
                    enriched_bet,
                    sport="mlb",
                    league="MLB",
                    sport_key="baseball_mlb",
                ))

        # Parlay legs
        for team, ml_prob, team_odds, fair_prob in [
            (home_team, p_home, home_odds, fp_home),
            (away_team, p_away, away_odds, fp_away),
        ]:
            if team_odds and team_odds > 1.05:
                _leg = to_parlay_leg("mlb", home_team, away_team, team,
                                     team_odds, ml_prob, fair_prob, commence, window=window)
                if _leg:
                    parlay_legs.append(_leg)

        # Totals (over/under runs — typical line 8.5 or 9.5)
        if window in ("today", "tomorrow", "day_after"):
            totals_data = mlb_totals_map.get(game.get("id"), {})
            _best_over: dict = {}
            _best_under: dict = {}
            for bk in totals_data.get("bookmakers", []):
                for mkt in bk.get("markets", []):
                    if mkt.get("key") != "totals":
                        continue
                    for o in mkt.get("outcomes", []):
                        name  = o.get("name", "")
                        price = float(o.get("price", 0))
                        if price <= 1.0:
                            continue
                        parts = name.split()
                        if len(parts) != 2:
                            continue
                        direction = parts[0].lower()
                        try:
                            ln = float(parts[1])
                        except ValueError:
                            continue
                        if direction == "over":
                            _best_over[ln]  = max(_best_over.get(ln, 0), price)
                        elif direction == "under":
                            _best_under[ln] = max(_best_under.get(ln, 0), price)

            # MLB totals: ML model — prefer 8.5, then 9.0, 9.5, 7.5
            for _line in (8.5, 9.0, 9.5, 7.5):
                _ov = _best_over.get(_line)
                _un = _best_under.get(_line)
                if _ov and _un:
                    totals_results = ml_totals_bet(
                        "mlb", home_team, away_team, combined,
                        feature_cols, _mlb_totals_model,
                        over_odds=_ov, under_odds=_un, line=_line,
                        window=window, commence=commence, league="MLB",
                    )
                    if totals_results:
                        for tb in totals_results:
                            value_bets.append(tb)
                    break

        # ── MLB spreads (run line) ─────────────────────────────────────────────
        if window in ("today", "tomorrow", "day_after") and _mlb_spreads_model:
            try:
                spreads_data_raw = fetch_odds("baseball_mlb", markets="spreads", priority="optional")
                _mlb_spreads_map = {g["id"]: g for g in spreads_data_raw if "id" in g}
            except Exception:
                _mlb_spreads_map = {}

            spreads_game = _mlb_spreads_map.get(game.get("id"), {})
            _other_sport_games[-1]["available_market_keys"] = _merge_market_keys(
                _other_sport_games[-1].get("available_market_keys"),
                _extract_market_keys_from_odds_game(spreads_game),
            )
            _best_hc: float = 0.0
            _best_ac: float = 0.0
            _spread_line = 1.5
            for bk in spreads_game.get("bookmakers", []):
                for mkt in bk.get("markets", []):
                    if mkt.get("key") != "spreads":
                        continue
                    for o in mkt.get("outcomes", []):
                        pt = o.get("point", 0)
                        pr = float(o.get("price", 0))
                        nm = o.get("name", "")
                        if pr <= 1.0:
                            continue
                        if nm == home_team and pt < 0:
                            _best_hc = max(_best_hc, pr)
                            _spread_line = abs(pt)
                        elif nm == away_team and pt > 0:
                            _best_ac = max(_best_ac, pr)

            if _best_hc > 1.0 and _best_ac > 1.0:
                spreads_results = ml_spreads_bet(
                    "mlb", home_team, away_team, combined,
                    feature_cols, _mlb_spreads_model,
                    home_spread_odds=_best_hc, away_spread_odds=_best_ac,
                    line=_spread_line,
                    window=window, commence=commence, league="MLB",
                    home_name=home_team, away_name=away_team,
                )
                if spreads_results:
                    for sb in spreads_results:
                        value_bets.append(sb)

    return value_bets, parlay_legs


# ─────────────────────────────────────────────────────────────────────────────
# NHL pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_nhl(dry_run: bool = False) -> Tuple[List[dict], List[ParlayLeg]]:
    global _other_sport_games
    logger.info("=== NHL PIPELINE ===")
    features_df, engineer = _load_features_cached(
        "nhl",
        "data/cache/nhl_features.parquet",
        NHLFetcher,
        NHLFeatureEngineer,
    )
    if features_df.empty:
        logger.warning("No NHL data available.")
        return [], []
    logger.info("NHL data: %d games (cached)", len(features_df))

    y_all = features_df["target"].dropna()

    _nhl_tag = get_current_model_tag("nhl", fallback="nhl_2024_25")
    trainer = ModelTrainer(sport="nhl")
    trainer.load_models(tag=_nhl_tag)
    if not trainer.trained_models:
        logger.info("No saved NHL models found; training now …")
        _always_drop = ["result", "home_score", "away_score", "home_goals",
                        "away_goals", "game_id", "date", "season",
                        "home_team", "away_team", "target"]
        X_tmp = features_df.drop(columns=[c for c in _always_drop if c in features_df.columns],
                                  errors="ignore").select_dtypes(include=[np.number]).fillna(0)
        X_tmp = X_tmp.loc[y_all.index]
        split = int(len(X_tmp) * 0.8)
        trainer.train(X_tmp.iloc[:split], y_all.iloc[:split],
                      X_tmp.iloc[split:], y_all.iloc[split:])
        trainer.save_models(tag=_nhl_tag)

    sample_model = next(iter(trainer.trained_models.values()))
    feature_cols = list(sample_model.feature_names_in_)
    missing = [c for c in feature_cols if c not in features_df.columns]
    if missing:
        logger.warning("Adding zero-filled missing NHL features: %s", missing)
        for c in missing:
            features_df[c] = 0.0
    X_all = features_df[feature_cols].fillna(0)
    X_all = X_all.loc[y_all.index]

    trainer.ensemble_model = _SoftVotingWrapper(
        estimators=list(trainer.trained_models.items()),
        weights=None,
        classes=np.array(sorted(y_all.unique())),
    )

    _nhl_cal = None
    _cal_path = calibrator_path_for_tag("nhl", _nhl_tag)
    if _cal_path.exists():
        _nhl_cal = EnsembleCalibrator.load(_cal_path)
    _nhl_alpha = _effective_alpha("nhl", calibrated=_nhl_cal is not None)

    _nhl_totals_model  = TotalsTrainer.load(Path("data/models/nhl/totals_nhl.joblib"))
    _nhl_spreads_model = TotalsTrainer.load(Path("data/models/nhl/spreads_nhl.joblib"))

    label_map = engineer.label_map
    inv_map = {v: k for k, v in label_map.items()}

    # Entity resolution via FeatureStore/TeamResolver — reads from data/team_ids.json.
    # To add a new alias, edit the JSON; no code changes needed.
    _nhl_resolver = TeamResolver("nhl")

    def get_team_snapshot_nhl(team: str, as_home: bool) -> Optional[pd.Series]:
        col = "home_team" if as_home else "away_team"
        # Resolve via JSON registry: "Carolina Hurricanes" → "Hurricanes"
        norm = _nhl_resolver.resolve(team)
        rows = X_all.loc[features_df[col] == norm]
        row_count = len(rows)
        if row_count < _MIN_SNAPSHOT_ROWS:
            logger.warning("[nhl] Only %d rows for %s (norm=%s) — skipping", row_count, team, norm)
            return None
        snap = rows.tail(10).mean()
        snap.attrs["history_rows"] = row_count
        snap.attrs["resolved_team"] = norm
        return snap

    if dry_run:
        logger.info("[dry-run] Skipping odds fetch for NHL.")
        return [], []

    try:
        nhl_games = fetch_odds("icehockey_nhl")
    except Exception as exc:
        logger.warning("Could not fetch NHL odds: %s", exc)
        return [], []

    today_cnt     = sum(1 for g in nhl_games if g.get("_window") == "today")
    overnight_cnt = sum(1 for g in nhl_games if g.get("_window") == "overnight")
    tomorrow_cnt  = sum(1 for g in nhl_games if g.get("_window") == "tomorrow")
    day_after_cnt = sum(1 for g in nhl_games if g.get("_window") == "day_after")
    logger.info("NHL: %d upcoming (today=%d overnight=%d tomorrow=%d day_after=%d)",
                len(nhl_games), today_cnt, overnight_cnt, tomorrow_cnt, day_after_cnt)

    # Totals
    try:
        nhl_totals_games = fetch_odds("icehockey_nhl", markets="totals", priority="optional")
        nhl_totals_map = {g["id"]: g for g in nhl_totals_games if "id" in g}
    except Exception:
        nhl_totals_map = {}

    value_bets: List[dict] = []
    parlay_legs: List[ParlayLeg] = []

    for game in nhl_games:
        home_team = game["home_team"]
        away_team = game["away_team"]
        commence  = game["commence_time"]
        window    = game.get("_window", "today")

        home_odds, home_bk, home_stale = best_odds(game, home_team)
        away_odds, away_bk, away_stale = best_odds(game, away_team)

        # Always record the game for the All Games view
        _other_sport_games.append(enrich_with_capability({
            "sport": "nhl", "home": home_team, "away": away_team,
            "league": "NHL", "commence": commence,
            "kick_off": _kick_off_label(commence), "window": window,
            "home_odds": home_odds, "away_odds": away_odds,
            "league_key": "icehockey_nhl",
            "available_market_keys": _merge_market_keys(
                _extract_market_keys_from_odds_game(game),
                _extract_market_keys_from_odds_game(nhl_totals_map.get(game.get("id"), {})),
            ),
        }, sport="nhl", league="NHL", sport_key="icehockey_nhl"))

        if home_odds is None or away_odds is None:
            continue

        h_snap = get_team_snapshot_nhl(home_team, as_home=True)
        a_snap = get_team_snapshot_nhl(away_team, as_home=False)
        if h_snap is None or a_snap is None:
            logger.info("SKIP %s vs %s: insufficient NHL team history", home_team, away_team)
            continue
        home_history_rows = int(h_snap.attrs.get("history_rows", 0) or 0)
        away_history_rows = int(a_snap.attrs.get("history_rows", 0) or 0)
        combined = h_snap.copy()
        combined.attrs["home_history_rows"] = home_history_rows
        combined.attrs["away_history_rows"] = away_history_rows
        for col in [c for c in feature_cols if c.startswith("away_")]:
            if col in a_snap.index:
                combined[col] = a_snap[col]

        # Add rest days to game dict for contextual flags (use defaults if not available)
        game["home_rest_days"] = float(h_snap.get("home_rest_days", 7))
        game["away_rest_days"] = float(a_snap.get("away_rest_days", 7))
        ctx = _detect_contextual_flags(game, "nhl")
        ctx = {
            **ctx,
            **build_availability_context("nhl", game, combined),
            **build_environment_context("nhl", home_team, away_team, commence),
        }

        x = pd.DataFrame([combined], columns=feature_cols)
        raw_p = trainer.ensemble_model.predict_proba(x)
        proba = _nhl_cal.transform(raw_p)[0] if _nhl_cal else raw_p[0]

        home_idx = inv_map.get("home_win", 2)
        away_idx = inv_map.get("away_win", 0)
        p_home = float(proba[home_idx]) if home_idx < len(proba) else 0.50
        p_away = float(proba[away_idx]) if away_idx < len(proba) else 0.50
        nhl_blend = _NHL_SIDE_MODEL.combine_with_classifier_diagnostics(
            (p_home, p_away),
            _NHL_SIDE_MODEL.structural_probs_from_snapshot(combined),
        )
        p_home, p_away = nhl_blend.combined.as_tuple()
        p_home, p_away, nhl_context_probability_adjustment = _apply_nhl_context_probability_adjustment(p_home, p_away, ctx)
        raw_home_prob = p_home
        raw_away_prob = p_away

        fp_home, fp_away = vig_free_prob(home_odds, away_odds)
        pre_market_blend_home = p_home
        pre_market_blend_away = p_away

        # NHL uses 0.45 — overconfidence observed, small cal set → market-led
        # Winsorize first: clamp model prob to within 20pp of market
        _nhl_live_alpha = _live_blend_alpha(
            "nhl",
            "moneyline",
            _nhl_alpha,
            home_rows=home_history_rows,
            away_rows=away_history_rows,
            disagreement_pp=float(nhl_blend.disagreement) * 100.0,
        )
        p_home = _winsorize_prob(p_home, fp_home, "nhl")
        p_home = _blend(p_home, fp_home, alpha=_nhl_live_alpha)
        p_away = 1.0 - p_home
        nhl_probability_debug = {
            "classifier_probs": {
                "home": round(nhl_blend.classifier.home, 4),
                "away": round(nhl_blend.classifier.away, 4),
            },
            "structural_probs": {
                "home": round(nhl_blend.structural.home, 4),
                "away": round(nhl_blend.structural.away, 4),
            } if nhl_blend.structural else None,
            "market_probs": {
                "home": round(float(fp_home), 4),
                "away": round(float(fp_away), 4),
            },
            "pre_market_blend_probs": {
                "home": round(float(pre_market_blend_home), 4),
                "away": round(float(pre_market_blend_away), 4),
            },
            "final_probs": {
                "home": round(float(p_home), 4),
                "away": round(float(p_away), 4),
            },
            "blend_alpha": round(float(_nhl_live_alpha), 3),
            "history_rows": {
                "home": home_history_rows,
                "away": away_history_rows,
            },
            "structural_available": bool(nhl_blend.structural),
            "regime": nhl_blend.regime,
            "disagreement_pp": round(float(nhl_blend.disagreement) * 100, 1),
            "classifier_weight": round(float(nhl_blend.model_weight), 3),
            "structural_weight": round(float(nhl_blend.structural_weight), 3),
            "context_probability_adjustment": nhl_context_probability_adjustment,
            "home_goalie_confirmed": ctx.get("home_goalie_confirmed"),
            "away_goalie_confirmed": ctx.get("away_goalie_confirmed"),
            "goalie_status": ctx.get("goalie_status"),
            "rest_advantage": int(ctx.get("rest_advantage", 0) or 0),
        }
        _other_sport_games[-1]["model_available"] = True
        _other_sport_games[-1]["model_pick"] = home_team if p_home >= p_away else away_team
        _other_sport_games[-1]["nhl_probability_debug"] = nhl_probability_debug

        home_med = median_odds(game, home_team)
        away_med = median_odds(game, away_team)
        home_factors = _probability_factors_for_snapshot("nhl", combined, home_team=home_team, away_team=away_team, selection="home")
        away_factors = _probability_factors_for_snapshot("nhl", combined, home_team=home_team, away_team=away_team, selection="away")
        home_adjustments = _context_adjustments("nhl", "home", combined, ctx)
        away_adjustments = _context_adjustments("nhl", "away", combined, ctx)
        game_info = {"sport": "nhl", "home": home_team, "away": away_team,
                     "commence": commence, "kick_off": _kick_off_label(commence),
                     "window": window, "league": "NHL", "league_key": "icehockey_nhl",
                     "event_id": game.get("id", ""),
                     "odds_snapshot_age_hours": game.get("_odds_snapshot_age_hours"),
                     "odds_source_status": game.get("_odds_source_status", ""),
                     "odds_fetched_at": game.get("_odds_fetched_at"),
                     "bookmaker_last_update": game.get("_odds_bookmaker_last_update"),
                     "cache_loaded_at": game.get("_odds_cache_loaded_at"),
                     "cache_age_hours": game.get("_odds_cache_age_hours"),
                     "odds_age_basis": game.get("_odds_age_basis", ""),
                     "odds_source_detail": game.get("_odds_source_detail", ""),
                     "odds_source_reason": game.get("_odds_source_reason", ""),
                     "force_fresh_odds_active": bool(game.get("_odds_force_fresh_requested", _FORCE_FRESH_ODDS)),
                     "odds_cache_used": bool(game.get("_odds_cache_used")),
                     "odds_fallback_used": bool(game.get("_odds_fallback_used")),
                     **_feature_freshness_context("nhl"),
                     **ctx}
        game_info["nhl_probability_debug"] = nhl_probability_debug

        for bet, bk_name in [
            (build_value_bet(home_team, p_home, home_odds, fp_home,
                             stale_line=home_stale, median_price=home_med, sport="nhl",
                             raw_model_prob=raw_home_prob, prediction_factors=home_factors,
                             context_adjustments=home_adjustments, availability_context=ctx), home_bk),
            (build_value_bet(away_team, p_away, away_odds, fp_away,
                             stale_line=away_stale, median_price=away_med, sport="nhl",
                             raw_model_prob=raw_away_prob, prediction_factors=away_factors,
                             context_adjustments=away_adjustments, availability_context=ctx), away_bk),
        ]:
            if bet and window in ("today", "tomorrow", "day_after"):
                enriched_bet = _with_candidate_odds_diagnostics(
                    {**game_info, **bet, "bookmaker": bk_name},
                    game=game,
                    market_key="h2h",
                    selection_name=str(bet.get("team", "") or ""),
                    bookmaker_name=str(bk_name or ""),
                )
                value_bets.append(enrich_with_capability(
                    enriched_bet,
                    sport="nhl",
                    league="NHL",
                    sport_key="icehockey_nhl",
                ))

        for team, ml_prob, team_odds, fair_prob in [
            (home_team, p_home, home_odds, fp_home),
            (away_team, p_away, away_odds, fp_away),
        ]:
            if team_odds and team_odds > 1.05:
                _leg = to_parlay_leg("nhl", home_team, away_team, team,
                                     team_odds, ml_prob, fair_prob, commence, window=window)
                if _leg:
                    parlay_legs.append(_leg)

        # Totals (NHL: typical line 5.5 or 6.0)
        if window in ("today", "tomorrow", "day_after"):
            totals_data = nhl_totals_map.get(game.get("id"), {})
            _best_over: dict = {}
            _best_under: dict = {}
            for bk in totals_data.get("bookmakers", []):
                for mkt in bk.get("markets", []):
                    if mkt.get("key") != "totals":
                        continue
                    for o in mkt.get("outcomes", []):
                        name  = o.get("name", "")
                        price = float(o.get("price", 0))
                        if price <= 1.0:
                            continue
                        parts = name.split()
                        if len(parts) != 2:
                            continue
                        direction = parts[0].lower()
                        try:
                            ln = float(parts[1])
                        except ValueError:
                            continue
                        if direction == "over":
                            _best_over[ln]  = max(_best_over.get(ln, 0), price)
                        elif direction == "under":
                            _best_under[ln] = max(_best_under.get(ln, 0), price)

            for _line in (5.5, 6.0, 6.5, 5.0):
                _ov = _best_over.get(_line)
                _un = _best_under.get(_line)
                if _ov and _un:
                    totals_results = ml_totals_bet(
                        "nhl", home_team, away_team, combined,
                        feature_cols, _nhl_totals_model,
                        over_odds=_ov, under_odds=_un, line=_line,
                        window=window, commence=commence, league="NHL",
                    )
                    if totals_results:
                        for tb in totals_results:
                            value_bets.append(tb)
                    break

        # ── NHL spreads (puck line) ────────────────────────────────────────────
        if window in ("today", "tomorrow", "day_after") and _nhl_spreads_model:
            try:
                nhl_spreads_raw = fetch_odds("icehockey_nhl", markets="spreads", priority="optional")
                _nhl_spreads_map = {g["id"]: g for g in nhl_spreads_raw if "id" in g}
            except Exception:
                _nhl_spreads_map = {}

            spreads_game = _nhl_spreads_map.get(game.get("id"), {})
            _best_hc: float = 0.0
            _best_ac: float = 0.0
            _spread_line = 1.5
            for bk in spreads_game.get("bookmakers", []):
                for mkt in bk.get("markets", []):
                    if mkt.get("key") != "spreads":
                        continue
                    for o in mkt.get("outcomes", []):
                        pt  = o.get("point", 0)
                        pr  = float(o.get("price", 0))
                        nm  = o.get("name", "")
                        if pr <= 1.0:
                            continue
                        if nm == home_team and pt is not None and float(pt) < 0:
                            _best_hc = max(_best_hc, pr)
                            _spread_line = abs(float(pt))
                        elif nm == away_team and pt is not None and float(pt) > 0:
                            _best_ac = max(_best_ac, pr)

            if _best_hc > 1.0 and _best_ac > 1.0:
                try:
                    spreads_results = ml_spreads_bet(
                        "nhl", home_team, away_team, combined,
                        feature_cols, _nhl_spreads_model,
                        home_spread_odds=_best_hc, away_spread_odds=_best_ac,
                        line=_spread_line,
                        window=window, commence=commence, league="NHL",
                        home_name=home_team, away_name=away_team,
                    )
                except Exception as _spreads_exc:
                    logger.debug("NHL spreads skipped (%s vs %s): %s", home_team, away_team, _spreads_exc)
                    spreads_results = None
                if spreads_results:
                    for sb in spreads_results:
                        value_bets.append(sb)

    return value_bets, parlay_legs


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

def write_report(
    all_bets: List[dict],
    review_bets: Optional[List[dict]] = None,
    suppressed_bets: Optional[List[dict]] = None,
    bankroll_blocked_bets: Optional[List[dict]] = None,
    wta_audit_rows: Optional[List[dict]] = None,
    parlay_results: Optional[dict] = None,
    requests_remaining: Optional[str] = None,
    bankroll: float = 1000.0,
    market_policy_summary: Optional[dict] = None,
    focused_prediction_summary: Optional[dict] = None,
    version_snapshot: Optional[dict] = None,
    pre_committee_review_breakdown: Optional[dict[str, int]] = None,
) -> Path:
    """Write a combined single-bet + parlay markdown report and JSON summary."""
    report_path = REPORTS_DIR / f"value_bets_{TODAY}.md"
    json_path = REPORTS_DIR / f"summary_{TODAY}.json"

    all_bets.sort(
        key=lambda b: (b.get("market_priority_score", 0), b["edge"]),
        reverse=True,
    )

    lines = [
        f"# Sports Betting Report — {TODAY}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"Odds API requests remaining: {requests_remaining or 'N/A'}",
        f"Bankroll: £{bankroll:,.2f}",
        "",
    ]

    # ── Section 1: Single-bet value scan ─────────────────────────────────
    lines += [
        "## Single-Bet Value Scanner",
        "",
        f"**Published value bets: {len(all_bets)}** (edge ≥ 3%, production filters passed)",
        f"**Blocked by bankroll/exposure: {len(bankroll_blocked_bets or [])}**",
        "",
    ]

    if market_policy_summary:
        preferred = market_policy_summary.get("preferred", [])[:6]
        experimental = market_policy_summary.get("experimental", [])[:4]
        disabled = market_policy_summary.get("disabled", [])[:4]
        lines += ["### Market Policy", ""]
        if preferred:
            lines.append("Preferred now: " + ", ".join(f"`{item['sport']}:{item['market']}`" for item in preferred))
        if experimental:
            lines.append("Experimental now: " + ", ".join(f"`{item['sport']}:{item['market']}`" for item in experimental))
        if disabled:
            lines.append("Suppressed now: " + ", ".join(f"`{item['sport']}:{item['market']}`" for item in disabled))
        lines.append("")

    if focused_prediction_summary:
        focused_rows = (
            focused_prediction_summary.get("primary", [])
            + focused_prediction_summary.get("secondary", [])
            + focused_prediction_summary.get("controlled", [])
        )
        if focused_rows:
            lines += ["### Focused Prediction Lanes", ""]
            lines.append(
                "Active focus: "
                + ", ".join(
                    f"`{item['sport']}:{item['market']}` ({item.get('label', item.get('status', 'focus'))})"
                    for item in focused_rows
                )
            )
            lines.append("")

    if not all_bets:
        lines.append("_No single-bet value found today._\n")
    else:
        sports = sorted({b["sport"] for b in all_bets})
        for sport in sports:
            sport_bets = [b for b in all_bets if b["sport"] == sport]
            lines += [f"### {sport.title()} ({len(sport_bets)} bets)", ""]
            for bet in sport_bets:
                home, away = bet["home"], bet["away"]
                commence = bet.get("commence", "")
                try:
                    dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
                    time_str = dt.strftime("%a %d %b %Y %H:%M UTC")
                except Exception:
                    time_str = commence
                stake_amt = bankroll * bet["kelly_stake_pct"] / 100
                bk = bet.get("bookmaker", "")
                bk_str = f"  |  Book: **{bk}**" if bk else ""
                league = bet.get("league", "")
                league_str = f"  |  {league}" if league else ""
                market = bet.get("market", "moneyline")
                market_tag = f"  |  Market: `{market}`" if market != "moneyline" else ""
                policy_tag = f"  |  Status: **{bet.get('market_policy_label', 'Experimental')}**"
                decision_tag = f"  |  Decision: **{bet.get('decision_status', 'BET')}**"
                committee_tag = (
                    f"  |  Committee: **{bet.get('committee_final_decision', '')}**"
                    if bet.get("committee_final_decision")
                    else ""
                )
                window_tag = f"  |  Window: {bet.get('window','today')}"
                lines += [
                    f"**{away} @ {home}**{league_str}  ",
                    f"Commence: {time_str}{window_tag}  ",
                    f"Bet on: **{bet['team']}** @ {bet['odds']}{bk_str}{market_tag}{policy_tag}{decision_tag}{committee_tag}  ",
                    f"Edge: +{bet['edge']*100:.1f}%  |  "
                    f"Model: {bet['ml_prob']*100:.1f}% vs Market: {bet.get('market_implied_prob', 0)*100:.1f}% vs Fair: {bet['fair_prob']*100:.1f}%  ",
                    f"Why this tier: {bet.get('market_policy_reason', 'No policy note available.')}  ",
                    f"Decision reason: {bet.get('decision_reason', 'All publication guardrails passed.')}  ",
                    f"Fractional Kelly ({bet.get('kelly_fraction_used', KELLY.fraction):.2f}x): {bet['kelly_stake_pct']:.1f}% = £{stake_amt:.2f}",
                    "",
                ]
                if bet.get("committee"):
                    committee = bet["committee"]
                    research = committee.get("research_mind", {})
                    model = committee.get("model_mind", {})
                    arbiter = committee.get("arbiter", {})
                    lines += [
                        f"Research Mind: {research.get('verdict', 'n/a')}  |  Confidence: {research.get('confidence', 'n/a')}  |  Freshness: {research.get('data_freshness', 'n/a')}  ",
                        f"Evidence status: {research.get('evidence_status', 'n/a')}  |  Concrete info score: {research.get('concrete_info_score', 'n/a')}  |  Source count: {research.get('source_count', 'n/a')} ({research.get('source_quality_summary', 'n/a')})  ",
                        f"Fixture verified: {research.get('fixture_verified', 'n/a')}  |  Odds age: {research.get('odds_age_minutes', 'n/a')} min  |  Odds freshness: {research.get('odds_freshness_status', 'n/a')}  ",
                        f"Lineup: {research.get('lineup_status', 'n/a')}  |  Injury: {research.get('injury_status', 'n/a')}  |  Motivation: {research.get('motivation_status', 'n/a')}  |  Rotation: {research.get('rotation_status', 'n/a')}  ",
                        f"Evidence: {', '.join(research.get('main_evidence', [])[:2]) or 'n/a'}  ",
                        f"Risks: {', '.join(research.get('main_risks', [])[:2]) or 'n/a'}  ",
                        f"Missing evidence: {', '.join(research.get('missing_evidence', [])[:3]) or 'n/a'}  ",
                        f"Conflicting evidence: {', '.join(research.get('conflicting_evidence', [])[:2]) or 'n/a'}  ",
                        f"Model Mind: {model.get('verdict', 'n/a')}  |  Prob: {model.get('model_probability', 'n/a')}  |  Vig-free: {model.get('vig_free_probability', 'n/a')}  |  Fair odds: {model.get('fair_odds', 'n/a')}  ",
                        f"Min odds: {model.get('minimum_acceptable_odds', 'n/a')}  |  Current odds: {model.get('current_odds', 'n/a')}  |  Edge: {model.get('edge', 'n/a')}  |  Risk tier: {arbiter.get('effective_risk_tier', model.get('risk_tier', 'n/a'))}  ",
                        f"Arbiter: {arbiter.get('agreement_status', 'n/a')}  |  Final: {arbiter.get('final_decision', 'n/a')}  |  Veto flags: {', '.join(arbiter.get('veto_flags', [])) or 'none'}  ",
                        f"Arbiter reason: {arbiter.get('reason', 'n/a')}  ",
                        f"Better substitute: {arbiter.get('better_substitute', 'n/a') or 'n/a'}  |  Parlay suitability: {arbiter.get('parlay_suitability', 'n/a') or 'n/a'}  ",
                        f"Final explanation: {arbiter.get('final_explanation', 'n/a')}  ",
                        "",
                    ]
                    if show_committee_details():
                        lines += ["```text", bet.get("committee_details_text", ""), "```", ""]

    if review_bets:
        lines += ["### Manual Review Queue", ""]
        review_breakdown = _review_reason_breakdown(review_bets)
        if review_breakdown:
            lines.append(
                "Breakdown: " + ", ".join(
                    f"{_review_reason_label(bucket)} ({count})"
                    for bucket, count in review_breakdown.items()
                )
            )
            lines.append("")
        for bet in review_bets[:12]:
            lines += [
                f"- `{bet.get('sport','').upper()}:{bet.get('market','moneyline')}` {bet.get('team','Unknown')} "
                f"({bet.get('home','')} vs {bet.get('away','')}) — "
                f"{bet.get('decision_status', 'HOLD')}: {bet.get('decision_reason') or bet.get('review_reason', 'held for review')}"
            ]
            if bet.get("committee"):
                research = (bet.get("committee") or {}).get("research_mind", {})
                lines += [
                    (
                        "  "
                        f"evidence_status={research.get('evidence_status', 'n/a')} "
                        f"concrete_info_score={research.get('concrete_info_score', 'n/a')} "
                        f"source_count={research.get('source_count', 'n/a')} "
                        f"source_quality={research.get('source_quality_summary', 'n/a')} "
                        f"missing={', '.join(research.get('missing_evidence', [])[:2]) or 'n/a'} "
                        f"odds_age_minutes={research.get('odds_age_minutes', 'n/a')} "
                        f"lineup={research.get('lineup_status', 'n/a')} "
                        f"injury={research.get('injury_status', 'n/a')} "
                        f"motivation={research.get('motivation_status', 'n/a')} "
                        f"rotation={research.get('rotation_status', 'n/a')}"
                    )
                ]
            if _review_reason_bucket(str(bet.get("review_reason", ""))) == "stale_price":
                lines += [
                    (
                        f"  odds_diag: source={bet.get('odds_source_status', 'unknown')} "
                        f"detail={bet.get('odds_source_detail', 'n/a')} "
                        f"age={bet.get('computed_odds_age_hours', bet.get('odds_snapshot_age_hours', 'n/a'))}h "
                        f"bookmaker_update={bet.get('bookmaker_last_update', 'n/a') or 'n/a'} "
                        f"cache_loaded={bet.get('cache_loaded_at', 'n/a') or 'n/a'} "
                        f"force_fresh={bool(bet.get('force_fresh_odds_active'))} "
                        f"fallback={bool(bet.get('odds_fallback_used'))}"
                    )
                ]
        if len(review_bets) > 12:
            lines.append(f"- _...and {len(review_bets) - 12} more review candidates._")
        lines.append("")

    if suppressed_bets:
        lines += ["### Suppressed Before Publish", ""]
        for bet in suppressed_bets[:12]:
            lines += [
                f"- `{bet.get('sport','').upper()}:{bet.get('market','moneyline')}` {bet.get('team','Unknown')} "
                f"({bet.get('home','')} vs {bet.get('away','')}) — "
                f"{bet.get('decision_status', 'NO BET')}: {bet.get('decision_reason') or bet.get('suppression_reason', 'suppressed by guardrail')}"
            ]
        if len(suppressed_bets) > 12:
            lines.append(f"- _...and {len(suppressed_bets) - 12} more suppressed candidates._")
        lines.append("")

    if bankroll_blocked_bets:
        lines += ["### Blocked By Bankroll", ""]
        for bet in bankroll_blocked_bets[:12]:
            lines += [
                f"- `{bet.get('sport','').upper()}:{bet.get('market','moneyline')}` {bet.get('team','Unknown')} "
                f"({bet.get('home','')} vs {bet.get('away','')}) — "
                f"{bet.get('decision_status', 'NO BET')}: {bet.get('decision_reason') or bet.get('suppression_reason', 'blocked by bankroll or exposure rules')}"
            ]
        if len(bankroll_blocked_bets) > 12:
            lines.append(f"- _...and {len(bankroll_blocked_bets) - 12} more bankroll-blocked candidates._")
        lines.append("")

    if wta_audit_rows:
        lines += ["### WTA Audit", ""]
        lines.append("Top current WTA near-misses and review-only candidates from the calibrated WTA lane.")
        lines.append("")
        for row in sorted(wta_audit_rows, key=lambda item: item.get("edge", -999), reverse=True)[:5]:
            lines.append(
                f"- `{row.get('team','Unknown')}` ({row.get('home','')} vs {row.get('away','')}) "
                f"@ {row.get('odds')} — edge {row.get('edge_pct', 0):+.1f}% "
                f"({row.get('audit_reason', 'audit only')})"
            )
        lines.append("")

    # ── Section 2: Parlay builder ─────────────────────────────────────────
    lines += ["---", "", "## Parlay Builder", ""]
    lines.append("_Target brackets: 5x (4.0–6.5), 10x (8.0–13.0), 20x (15.0–26.0)_\n")

    if parlay_results:
        lines.append(ParlayBuilder.format_report(parlay_results, bankroll=bankroll))
    else:
        lines.append("_Parlay data not available._\n")

    parlay_counts: Dict[str, Dict[str, int]] = {}
    if parlay_results:
        for tier, brackets in parlay_results.items():
            parlay_counts[tier] = {b: len(v) for b, v in brackets.items()}

    # Soccer game detail records (populated by run_soccer into _soccer_full_games)
    soccer_games = list(_soccer_full_games)
    # All non-soccer games (populated by run_basketball/tennis/mlb/nhl)
    other_games  = list(_other_sport_games)
    wta_audit = list(wta_audit_rows or [])
    real_bets    = all_bets  # already stripped of _full_game entries at main()
    sport_pipeline_diagnostics = _sport_pipeline_diagnostics(
        soccer_games=soccer_games,
        other_games=other_games,
        published_bets=real_bets,
        review_bets=review_bets or [],
        suppressed_bets=suppressed_bets or [],
    )

    summary = {
        "date": TODAY,
        "timestamp": datetime.now().isoformat(),
        "bankroll": bankroll,
        "scan_notes": list(_scan_runtime_notes),
        "single_bets": {
            "total": len(real_bets),
            "review_total": len(review_bets or []),
            "pre_committee_review_breakdown": dict(pre_committee_review_breakdown or {}),
            "review_breakdown": _review_reason_breakdown(review_bets or []),
            "stale_price_diagnostics": _stale_price_diagnostics(review_bets or []),
            "suppressed_total": len(suppressed_bets or []),
            "bankroll_blocked_total": len(bankroll_blocked_bets or []),
            "decision_breakdown": {
                "BET": len([b for b in real_bets if b.get("decision_status") == "BET"]),
                "BET SUBSTITUTE": len([b for b in real_bets if b.get("decision_status") == "BET SUBSTITUTE"]),
                "HOLD": len([b for b in (review_bets or []) if b.get("decision_status") == "HOLD"]),
                "WAIT FOR LINEUPS": len([b for b in (review_bets or []) if b.get("decision_status") == "WAIT FOR LINEUPS"]),
                "NO BET": len([b for b in (suppressed_bets or []) if b.get("decision_status") == "NO BET"]),
                "AVOID": len([b for b in (suppressed_bets or []) if b.get("decision_status") == "AVOID"]),
            },
            "committee_decision_breakdown": {
                "BET": len([b for b in real_bets if b.get("committee_final_decision") == "BET"]),
                "BET_SUBSTITUTE": len([b for b in real_bets if b.get("committee_final_decision") == "BET_SUBSTITUTE"]),
                "HOLD": len([b for b in (review_bets or []) if b.get("committee_final_decision") == "HOLD"]),
                "WAIT_FOR_LINEUPS": len([b for b in (review_bets or []) if b.get("committee_final_decision") == "WAIT_FOR_LINEUPS"]),
                "NO_BET": len([b for b in (suppressed_bets or []) if b.get("committee_final_decision") == "NO_BET"]),
                "AVOID": len([b for b in (suppressed_bets or []) if b.get("committee_final_decision") == "AVOID"]),
            },
            "by_sport": {
                sport: len([b for b in real_bets if b["sport"] == sport])
                for sport in {b["sport"] for b in real_bets}
            },
            "bets": real_bets,
            "review_bets": review_bets or [],
            "suppressed_bets": suppressed_bets or [],
            "bankroll_blocked_bets": bankroll_blocked_bets or [],
        },
        "soccer_games": soccer_games,   # full outcome breakdown for webapp
        "other_games":  other_games,    # basic info for all non-soccer games
        "wta_audit": wta_audit,
        "sport_pipeline_diagnostics": sport_pipeline_diagnostics,
        "parlays": parlay_counts,
        "requests_remaining": requests_remaining,
        "market_policy": market_policy_summary or {},
        "focused_prediction_lanes": focused_prediction_summary or {},
        "version_snapshot": version_snapshot or {},
    }

    def _summary_has_meaningful_content(payload: dict[str, Any]) -> bool:
        if not isinstance(payload, dict):
            return False
        single_bets = payload.get("single_bets") or {}
        if int(single_bets.get("total", 0) or 0) > 0:
            return True
        if int(single_bets.get("review_total", 0) or 0) > 0:
            return True
        if int(single_bets.get("suppressed_total", 0) or 0) > 0:
            return True
        if int(single_bets.get("bankroll_blocked_total", 0) or 0) > 0:
            return True
        if payload.get("soccer_games") or payload.get("other_games"):
            return True
        diagnostics = payload.get("sport_pipeline_diagnostics") or {}
        totals = diagnostics.get("totals") or {}
        for key in (
            "scanned_games",
            "model_available_games",
            "candidate_games",
            "published_games",
            "review_games",
            "suppressed_games",
            "no_candidate_games",
        ):
            try:
                if int(totals.get(key, 0) or 0) > 0:
                    return True
            except Exception:
                continue
        by_sport = diagnostics.get("by_sport") or {}
        if isinstance(by_sport, dict) and by_sport:
            return True
        return False

    new_summary_has_content = _summary_has_meaningful_content(summary)
    existing_summary = None
    existing_summary_has_content = False
    if json_path.exists():
        try:
            existing_summary = json.loads(json_path.read_text())
            existing_summary_has_content = _summary_has_meaningful_content(existing_summary)
        except Exception:
            existing_summary = None

    preserve_existing_summary = existing_summary_has_content and not new_summary_has_content
    preserve_existing_report = preserve_existing_summary and report_path.exists()

    if preserve_existing_report:
        logger.warning(
            "Preserving existing report %s because the current run produced an empty/degraded board.",
            report_path,
        )
    else:
        report_path.write_text("\n".join(lines))
        logger.info("Report written: %s", report_path)

    if preserve_existing_summary:
        logger.warning(
            "Preserving existing summary %s because the current run produced an empty/degraded board.",
            json_path,
        )
    else:
        json_path.write_text(json.dumps(summary, indent=2))
        logger.info("Summary written: %s", json_path)

    return report_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _retrain_sport(sport: str, tag: str) -> bool:
    """Retrain a single sport's models on the current feature cache. Returns True on success."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "retrain_and_calibrate.py", "--sports", sport],
        capture_output=False,
    )
    return result.returncode == 0


def main():
    global _odds_api_failed_fingerprints
    _odds_api_failed_fingerprints = set()
    parser = argparse.ArgumentParser(description="Daily sports value bet scanner + parlay builder")
    parser.add_argument("--sport", choices=["soccer", "basketball", "tennis", "mlb", "nhl", "all"], default="all")
    parser.add_argument("--market", choices=["moneyline", "spreads", "totals", "all"], default="all",
                        help="Restrict output to this market type (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Skip live odds API calls")
    parser.add_argument(
        "--offline-odds",
        action="store_true",
        help="Use saved odds/active-sports cache only; never call The Odds API",
    )
    parser.add_argument(
        "--force-fresh-odds",
        action="store_true",
        help="Bypass odds and active-sports caches for this run and force fresh Odds API pulls.",
    )
    parser.add_argument(
        "--lean-context",
        action="store_true",
        help="Skip non-essential enrichers like live weather and optional availability lookups to stay lean.",
    )
    parser.add_argument(
        "--context-referee",
        action="store_true",
        help="Run the OpenRouter context referee on a capped set of publish/review candidates.",
    )
    parser.add_argument(
        "--context-referee-max",
        type=int,
        default=8,
        help="Maximum number of candidates to send through the context referee (default 8).",
    )
    parser.add_argument(
        "--full-soccer-scope",
        action="store_true",
        help="Disable soccer speed-mode deferrals and scan the full active soccer scope.",
    )
    parser.add_argument(
        "--focused-lanes",
        action="store_true",
        help="When --sport all is used, scan only current focused prediction sports/lanes.",
    )
    parser.add_argument("--bankroll", type=float, default=1000.0, help="Bankroll in £ for stake sizing")
    parser.add_argument("--min-edge", type=float, default=0.03, help="Min edge for value parlays (default 0.03)")
    parser.add_argument("--min-legs", type=int, default=3, help="Min parlay legs (default 3)")
    parser.add_argument("--max-legs", type=int, default=6, help="Max parlay legs (default 6)")
    parser.add_argument(
        "--retrain", action="store_true",
        help="Re-train models on fresh data before scanning (rolling window update)",
    )
    parser.add_argument(
        "--record-bets", action="store_true",
        help="Persist value bets to data/tracker/predictions.parquet for later settlement",
    )
    parser.add_argument(
        "--notify", action="store_true",
        help="Send Telegram alert with top bets (requires TELEGRAM_TOKEN + TELEGRAM_CHAT_ID in .env)",
    )
    parser.add_argument(
        "--test-notify", action="store_true",
        help="Send a Telegram test message and exit",
    )
    args = parser.parse_args()
    global _OFFLINE_ODDS_ONLY, _LEAN_CONTEXT_ONLY, _FORCE_FRESH_ODDS, _FOCUSED_SCAN_ONLY, ODDS_KEY
    _OFFLINE_ODDS_ONLY = _OFFLINE_ODDS_ONLY or args.offline_odds
    _FORCE_FRESH_ODDS = bool(args.force_fresh_odds)
    _LEAN_CONTEXT_ONLY = _LEAN_CONTEXT_ONLY or args.lean_context
    _FOCUSED_SCAN_ONLY = _FOCUSED_SCAN_ONLY or bool(args.focused_lanes)
    if _LEAN_CONTEXT_ONLY:
        os.environ["SCAN_LEAN_CONTEXT"] = "1"
    if _FOCUSED_SCAN_ONLY:
        os.environ["SCAN_FOCUSED_LANES"] = "1"
    if args.full_soccer_scope:
        os.environ["SCAN_FULL_SOCCER_SCOPE"] = "1"
    if _FORCE_FRESH_ODDS and _OFFLINE_ODDS_ONLY:
        logger.warning("Force-fresh odds requested, but offline odds mode is enabled. Offline mode wins for this run.")
        _FORCE_FRESH_ODDS = False

    selected_odds_key = _select_odds_api_key()
    if selected_odds_key:
        ODDS_KEY = selected_odds_key
        os.environ["ODDS_API_KEY"] = selected_odds_key
    elif _FORCE_FRESH_ODDS and _pool_mode_enabled():
        _disable_live_odds_for_run(
            "No usable runtime Odds API key was available at scan start. Force-fresh live odds is disabled for this run; cached fallback odds may still be used.",
            note_type="odds_api_no_usable_key",
        )

    if args.test_notify:
        notifier = TelegramNotifier()
        ok = notifier.test()
        print("✓ Test message sent" if ok else "✗ Send failed — check TELEGRAM_TOKEN and TELEGRAM_CHAT_ID in .env")
        return

    # ── CLV-based alpha auto-tuning ──────────────────────────────────────
    # Adjust _SPORT_ALPHA based on historical CLV from settled bets.
    # Requires >= _CLV_MIN_BETS settled bets per sport to trigger.
    _autotune_alpha_from_clv()
    _load_market_health_snapshot()

    # ── BankrollManager ──────────────────────────────────────────────────
    bm = BankrollManager(initial_bankroll=args.bankroll)
    # Wire drawdown state into Kelly so sizing shrinks as losses mount
    bm_stats = bm.get_stats()
    KELLY.set_drawdown_state(bm_stats.get("drawdown_pct", 0.0))
    # Check circuit-breakers at startup using a nominal stake
    can_bet, reason = bm.can_place_bet(1.0)
    if not can_bet:
        logger.warning("BankrollManager: betting paused — %s", reason)

    logger.info(
        "=== Daily Scan — %s | sport=%s | bankroll=£%.0f | dry_run=%s | retrain=%s | offline_odds=%s | force_fresh_odds=%s | focused_lanes=%s ===",
        TODAY, args.sport, args.bankroll, args.dry_run, args.retrain, _OFFLINE_ODDS_ONLY, _FORCE_FRESH_ODDS, _FOCUSED_SCAN_ONLY,
    )
    version_snapshot = _build_version_snapshot(args=args)
    version_snapshot_json = json.dumps(version_snapshot, separators=(",", ":"), sort_keys=True)

    # ── Optional retrain ─────────────────────────────────────────────────
    if args.retrain:
        _all_retrain = ["soccer", "basketball", "tennis", "mlb", "nhl"]
        sports_to_retrain = (
            _all_retrain if args.sport == "all"
            else [args.sport] if args.sport in _all_retrain else []
        )
        for sp in sports_to_retrain:
            logger.info("Retraining %s models …", sp)
            if not _retrain_sport(sp, ""):
                logger.error("Retrain failed for %s — using existing models", sp)

    global _other_sport_games
    _other_sport_games = []   # reset on each full run
    global _wta_audit_rows
    _wta_audit_rows = []
    global _scan_runtime_notes
    _scan_runtime_notes = []

    all_bets: List[dict] = []
    all_legs: List[ParlayLeg] = []

    # Pre-fetch active sports list ONCE (1 request) — shared by soccer + tennis
    # pipelines so they can skip inactive leagues without extra API calls.
    if not args.dry_run:
        _prefetch_active_sports()

    if args.sport in ("soccer", "all"):
        try:
            bets, legs = run_soccer(dry_run=args.dry_run)
            all_bets += bets
            all_legs += legs
        except Exception as exc:
            logger.error("Soccer pipeline failed: %s", exc, exc_info=True)

    if args.sport in ("basketball", "all") and not (_FOCUSED_SCAN_ONLY and args.sport == "all"):
        try:
            bets, legs = run_basketball(dry_run=args.dry_run)
            all_bets += bets
            all_legs += legs
        except Exception as exc:
            logger.error("Basketball pipeline failed: %s", exc, exc_info=True)
    elif _FOCUSED_SCAN_ONLY and args.sport == "all":
        logger.info("Focused-lanes scan: skipping basketball pipeline (not currently a focused prediction sport)")

    if args.sport in ("tennis", "all"):
        try:
            bets, legs = run_tennis(dry_run=args.dry_run)
            all_bets += bets
            all_legs += legs
        except Exception as exc:
            logger.error("Tennis pipeline failed: %s", exc, exc_info=True)

    if args.sport in ("mlb", "all"):
        try:
            bets, legs = run_mlb(dry_run=args.dry_run)
            all_bets += bets
            all_legs += legs
        except Exception as exc:
            logger.error("MLB pipeline failed: %s", exc, exc_info=True)

    if args.sport in ("nhl", "all") and not (_FOCUSED_SCAN_ONLY and args.sport == "all"):
        try:
            bets, legs = run_nhl(dry_run=args.dry_run)
            all_bets += bets
            all_legs += legs
        except Exception as exc:
            logger.error("NHL pipeline failed: %s", exc, exc_info=True)
    elif _FOCUSED_SCAN_ONLY and args.sport == "all":
        logger.info("Focused-lanes scan: skipping NHL pipeline (not currently a focused prediction sport)")

    # ── Line movement enrichment ──────────────────────────────────────────
    # Attach opening-vs-current odds movement to every candidate so the
    # UI can show sharp/fade signals and the referee can factor them in.
    try:
        from src.data import opening_odds_store as _oos
        _oos_store = _oos.load()
        _lm_added = 0
        for _bet in all_bets:
            _eid = _bet.get("event_id", "")
            _team = _bet.get("team", "")
            _odds = float(_bet.get("odds") or 0)
            if not _eid or not _team or _odds <= 1.0:
                continue
            _mv = _oos.get_movement(_oos_store, _eid, _team, _odds)
            if _mv:
                _bet["line_movement"] = _mv
                _lm_added += 1
        if _lm_added:
            logger.info("Line movement: enriched %d/%d bets", _lm_added, len(all_bets))
    except Exception as _lm_exc:
        logger.debug("Line movement enrichment skipped: %s", _lm_exc)

    # ── Market filter ──────────────────────────────────────────────────────
    if args.market != "all":
        before = len(all_bets)
        all_bets = [b for b in all_bets if b.get("market", "moneyline") == args.market]
        logger.info("Market filter [%s]: %d → %d bets", args.market, before, len(all_bets))

    market_policy_summary = summarize_market_policy()
    focused_prediction_summary = summarize_focused_prediction_policy()
    all_bets, review_bets, suppressed_bets = apply_publish_guardrails(all_bets, bankroll=args.bankroll)
    pre_committee_review_breakdown = _review_reason_breakdown(review_bets)
    if review_bets:
        logger.info("Publish guardrails moved %d candidate bet(s) into the manual review queue", len(review_bets))
        if pre_committee_review_breakdown:
            logger.info(
                "Pre-committee manual review breakdown: %s",
                ", ".join(
                    f"{_review_reason_label(bucket)}={count}"
                    for bucket, count in pre_committee_review_breakdown.items()
                ),
            )
    if suppressed_bets:
        logger.info("Publish guardrails suppressed %d candidate bet(s)", len(suppressed_bets))
    all_bets, review_bets, suppressed_bets = apply_context_referee(
        all_bets,
        review_bets,
        suppressed_bets,
        enabled=bool(args.context_referee),
        max_candidates=args.context_referee_max,
    )

    committee_entries: List[dict] = []
    if committee_enabled():
        all_bets, review_bets, suppressed_bets, committee_entries = run_committee_pipeline(
            published=all_bets,
            review=review_bets,
            suppressed=suppressed_bets,
        )
        logger.info(
            "Committee gate: %d published, %d review, %d suppressed after Research/Model/Arbiter pass",
            len(all_bets),
            len(review_bets),
            len(suppressed_bets),
        )
        final_review_breakdown = _review_reason_breakdown(review_bets)
        if final_review_breakdown:
            logger.info(
                "Final review queue breakdown: %s",
                ", ".join(
                    f"{_review_reason_label(bucket)}={count}"
                    for bucket, count in final_review_breakdown.items()
                ),
            )

    # ── BankrollManager: final circuit-breaker pass on publish-ready bets ───
    filtered_bets: List[dict] = []
    bankroll_blocked_bets: List[dict] = []
    for bet in all_bets:
        stake = float(bet.get("stake_abs", 0.0) or 0.0)
        if stake <= 0:
            logger.debug("Skipping zero-stake bet: %s %s", bet["sport"], bet["team"])
            suppressed_bets.append({**bet, "suppressed": True, "suppression_reason": "stake reduced to zero by production staking rules"})
            continue
        allowed, reason = bm.can_place_bet(stake)
        if allowed:
            filtered_bets.append({**bet, "committee_publish_note": _committee_publish_note(bet)})
        else:
            logger.warning("BankrollManager blocked %s %s: %s", bet["sport"], bet["team"], reason)
            blocked_bet = {
                **bet,
                "suppressed": True,
                "bankroll_blocked": True,
                "suppression_reason": reason,
                "publication_outcome": "blocked_by_bankroll",
                "publication_reason": reason,
                "decision_status": "NO BET",
                "decision_reason": reason,
            }
            bankroll_blocked_bets.append(blocked_bet)
            suppressed_bets.append(blocked_bet)
    skipped = len(all_bets) - len(filtered_bets)
    if skipped:
        logger.info("BankrollManager: %d bet(s) blocked (%d remaining)", skipped, len(filtered_bets))

    if committee_enabled():
        all_bets = filter_and_rank_bets(filtered_bets)
        review_bets = sorted(
            review_bets,
            key=lambda b: (b.get("market_priority_score", 0), b.get("edge", 0)),
            reverse=True,
        )
        suppressed_bets = sorted(
            suppressed_bets,
            key=lambda b: (b.get("market_priority_score", 0), b.get("edge", 0)),
            reverse=True,
        )
    else:
        filtered_bets, review_bets, suppressed_bets = _apply_decision_labels(
            filtered_bets,
            review_bets,
            suppressed_bets,
        )
        all_bets = filter_and_rank_bets(filtered_bets)

    # ── Build parlays ────────────────────────────────────────────────────
    parlay_results = None
    if not PARLAYS_ENABLED:
        logger.info("Parlays disabled (PARLAYS_ENABLED=False). Re-enable once 200+ single bets settled with positive ROI.")
    elif not args.dry_run:
        parlay_input_legs = all_legs
        if committee_enabled() and committee_required_for_parlays():
            parlay_input_legs = [leg for leg in (_committee_to_parlay_leg(bet) for bet in all_bets) if leg is not None]
        if not parlay_input_legs:
            logger.info("Parlays skipped: no committee-approved legs were eligible for a slip.")
        else:
            builder = ParlayBuilder(
                min_edge=args.min_edge,
                min_prob=0.50,
                min_legs=args.min_legs,
                max_legs=min(int(args.max_legs), max_conservative_parlay_legs()),
                top_n=3,
                fraction=0.25,
                max_kelly_pct=3.0,
            )
            parlay_results = builder.build(parlay_input_legs)
            total_parlays = sum(
                len(v)
                for tier_data in parlay_results.values()
                for v in tier_data.values()
            )
            logger.info("Parlays built: %d total across all brackets/tiers", total_parlays)

            # Record top parlay per bracket/tier to the tracker
            if not args.dry_run:
                from src.utils.results_tracker import record_parlay as _record_parlay
                _recorded_parlays = 0
                for _tier, _brackets in parlay_results.items():
                    for _bracket, _parlays in _brackets.items():
                        if _parlays:
                            try:
                                _record_parlay(_parlays[0], bankroll=args.bankroll, version_snapshot=version_snapshot_json)
                                _recorded_parlays += 1
                            except Exception as _pe:
                                logger.debug("Failed to record parlay %s/%s: %s", _tier, _bracket, _pe)
                if _recorded_parlays:
                    logger.info("Recorded %d parlay(s) to tracker", _recorded_parlays)

    # ── Record bets to tracker (always — deduplication handled inside) ──
    if all_bets:
        from src.utils.results_tracker import record_predictions_bulk
        from datetime import datetime as _dt, timezone as _tz
        predictions = []
        for bet in all_bets:
            try:
                ct_str = bet.get("commence", "")
                ct = _dt.fromisoformat(ct_str.replace("Z", "+00:00")) if ct_str else _dt.now(tz=_tz.utc)
            except Exception:
                ct = _dt.now(tz=_tz.utc)
            probability_diagnostics = _prediction_diagnostics_for_tracker(bet)
            predictions.append({
                "sport":          bet["sport"],
                "match_id":       f"{bet.get('home','')} vs {bet.get('away','')}",
                "team_or_player": bet["team"],
                "commence_time":  ct,
                "market":         bet.get("market", "moneyline"),
                "market_status":  bet.get("market_status", "experimental"),
                "tier":           bet.get("market_policy_label", "Experimental"),
                "ml_prob":        bet["ml_prob"],
                "fair_prob":      bet["fair_prob"],
                "bet_odds":       bet["odds"],
                "bookmaker":      bet.get("bookmaker", "unknown"),
                "edge":           bet["edge"],
                "kelly_stake_pct":bet["kelly_stake_pct"] / 100,
                "stake_units":    bet.get("stake_abs", 0) / args.bankroll,
                "decision_status": bet.get("decision_status", ""),
                "decision_reason": bet.get("decision_reason", ""),
                "lower_bound_passed": bet.get("lower_bound_passed", False),
                "minimum_acceptable_odds": bet.get("minimum_acceptable_odds"),
                "freshness_check": bet.get("freshness_check", ""),
                "odds_freshness": bet.get("odds_freshness", ""),
                "lineup_freshness": bet.get("lineup_freshness", ""),
                "injury_news_freshness": bet.get("injury_news_freshness", ""),
                "standings_freshness": bet.get("standings_freshness", ""),
                "fixture_verified": bet.get("fixture_verified", True),
                "market_suitable": bet.get("market_suitable", True),
                "recommended_market": bet.get("recommended_market", ""),
                "context_factor_names": json.dumps([
                    str(item.get("name", ""))
                    for item in (bet.get("context_adjustments") or [])
                    if str(item.get("name", "")).strip()
                ]),
                **probability_diagnostics,
                "version_snapshot": version_snapshot_json,
            })
        ids = record_predictions_bulk(predictions)
        logger.info("Recorded %d predictions to tracker", len(ids))

    # ── API quota — use tracked value, no extra request ─────────────────
    if _OFFLINE_ODDS_ONLY:
        requests_remaining = "cached/offline"
    else:
        requests_remaining = str(_odds_remaining) if _odds_remaining < 9999 else None
    if requests_remaining is None:
        # Fall back to persisted value from previous run
        usage = _load_api_usage()
        rem = usage.get("odds_remaining")
        if rem:
            requests_remaining = str(rem)

    report_path = write_report(
        all_bets,
        suppressed_bets=suppressed_bets,
        review_bets=review_bets,
        bankroll_blocked_bets=bankroll_blocked_bets,
        wta_audit_rows=_wta_audit_rows,
        parlay_results=parlay_results,
        requests_remaining=requests_remaining,
        bankroll=args.bankroll,
        market_policy_summary=market_policy_summary,
        focused_prediction_summary=focused_prediction_summary,
        version_snapshot=version_snapshot,
        pre_committee_review_breakdown=pre_committee_review_breakdown,
    )

    # ── Console summary ──────────────────────────────────────────────────
    bm_stats = bm.get_stats()
    print(f"\n{'='*65}")
    print(f"Daily Scan Complete — {TODAY} | Bankroll: £{args.bankroll:,.0f}")
    if bm_stats.get("drawdown_pct", 0) > 0:
        print(f"Drawdown: {bm_stats['drawdown_pct']*100:.1f}%  "
              f"(limit: {bm._drawdown_limit*100:.0f}%)")
    print(f"{'='*65}")

    # Console: top 2 per sport then fill to 5, same logic as Telegram
    print(f"\nSINGLE BETS ({len(all_bets)} value bets found after all filters)")
    _ev_cap_summary = "/".join(f"{s}:{int(v*100)}%" for s, v in _SANITY_EV_CAP.items())
    print(f"  EV caps: {_ev_cap_summary}  |  "
          f"Sanity: ratio≤{_SANITY_RATIO}x AND gap≤{_SANITY_ABS_GAP*100:.0f}pp  |  "
          f"Blend α={BLEND_ALPHA}")
    if review_bets:
        _console_review_breakdown = _review_reason_breakdown(review_bets)
        if _console_review_breakdown:
            print(
                "  Review queue: " + ", ".join(
                    f"{_review_reason_label(bucket)}={count}"
                    for bucket, count in _console_review_breakdown.items()
                )
            )
        _stale_console = _stale_price_diagnostics(review_bets)[:3]
        for row in _stale_console:
            print(
                "    stale_diag: "
                f"{row.get('game')} | {row.get('selected_outcome')} | "
                f"source={row.get('odds_source_status')} detail={row.get('odds_source_detail')} "
                f"age={row.get('computed_odds_age_hours')}h "
                f"bookmaker_update={row.get('bookmaker_last_update') or 'n/a'} "
                f"cache_loaded={row.get('cache_loaded_at') or 'n/a'} "
                f"force_fresh={row.get('force_fresh_odds_active')} fallback={row.get('fallback_used')}"
            )
    if bankroll_blocked_bets:
        print(f"  Bankroll blocked: {len(bankroll_blocked_bets)} otherwise-valid bet(s)")
    if _wta_audit_rows:
        _top_wta = sorted(_wta_audit_rows, key=lambda row: row.get("edge", -999), reverse=True)[:2]
        print(
            "  WTA audit: " + "; ".join(
                f"{row.get('team')} {row.get('edge_pct', 0):+.1f}% ({row.get('audit_reason')})"
                for row in _top_wta
            )
        )
    _by_sport: dict = {}
    for _b in sorted(all_bets, key=lambda b: b["edge"], reverse=True):
        _by_sport.setdefault(_b.get("sport", "other"), []).append(_b)
    _console_top: list = []
    _seen: set = set()
    for _sp_bets in _by_sport.values():
        for _b in _sp_bets[:2]:
            if id(_b) not in _seen:
                _console_top.append(_b)
                _seen.add(id(_b))
    for _b in sorted(all_bets, key=lambda b: b["edge"], reverse=True):
        if len(_console_top) >= 5:
            break
        if id(_b) not in _seen:
            _console_top.append(_b)
            _seen.add(id(_b))
    for bet in sorted(_console_top, key=lambda b: (b.get("market_priority_score", 0), b["edge"]), reverse=True):
        stake = args.bankroll * bet["kelly_stake_pct"] / 100
        bk = bet.get("bookmaker", "")
        bk_tag = f" [{bk}]" if bk else ""
        league = bet.get("league", "")
        league_tag = f" {league}" if league else ""
        market = bet.get("market", "moneyline")
        market_tag = f" ({market})" if market != "moneyline" else ""
        status_tag = bet.get("market_policy_label", "Experimental")
        sport_label = f"{bet['sport'].upper()}{league_tag}"
        print(f"  [{sport_label}] {bet['team']}{market_tag} @ {bet['odds']}{bk_tag}  "
              f"edge={bet['edge']*100:.1f}%  tier={status_tag}  Kelly=£{stake:.0f}")
        if bet.get("committee_final_decision"):
            print(
                "    "
                f"Committee={bet.get('committee_final_decision')}  "
                f"Research={bet.get('research_mind_verdict', 'n/a')}  "
                f"Model={bet.get('model_mind_verdict', 'n/a')}  "
                f"Arbiter={bet.get('committee_agreement_status', 'n/a')}"
            )
            print(f"    Reason={bet.get('committee_reason') or bet.get('decision_reason', 'n/a')}")

    if parlay_results:
        print(f"\nPARLAYS:")
        _seen_legs: set = set()
        for bracket in ("5x", "10x", "20x"):
            for tier in ("value", "speculative"):
                parlays = parlay_results.get(tier, {}).get(bracket, [])
                if parlays:
                    best = parlays[0]
                    leg_key = frozenset((l.team, l.odds) for l in best.legs)
                    if leg_key in _seen_legs:
                        continue
                    _seen_legs.add(leg_key)
                    legs_str = " + ".join(f"{l.team}({l.odds})" for l in best.legs)
                    stake = args.bankroll * best.kelly_stake_pct / 100
                    tier_label = "LONGSHOT" if tier == "speculative" else tier.upper()
                    print(f"  [{bracket} {tier_label}] {legs_str}")
                    print(f"    → odds={best.combined_odds:.2f}  EV={best.ev:.3f}x  "
                          f"win={best.combined_prob*100:.1f}%  Kelly=£{stake:.0f}")

    print(f"\nReport: {report_path}")
    if requests_remaining:
        if requests_remaining.isdigit():
            rem = int(requests_remaining)
            if rem < 9999:   # only show bar when we have a real reading
                bar_full   = 25
                bar_filled = min(bar_full, int(bar_full * rem / 500))
                bar = "█" * bar_filled + "░" * (bar_full - bar_filled)
                warn = "  ⚠️  LOW" if rem <= _ODDS_BUDGET_WARN else ""
                print(f"Odds API  [{bar}] {rem}/500 remaining{warn}")
            else:
                print(f"Odds API  [cached data — no requests made this run]")
        else:
            print(f"Odds API  [{requests_remaining}]")
    print("=" * 65)

    # ── Show tracker dashboard always ────────────────────────────────────
    from src.utils.results_tracker import print_dashboard
    print_dashboard()

    # ── Telegram notification ─────────────────────────────────────────────
    if args.notify:
        notifier = TelegramNotifier()
        ok = notifier.send_daily_report(
            all_bets,
            parlay_results=parlay_results,
            bankroll=args.bankroll,
        )
        if ok:
            logger.info("Telegram notification sent successfully.")
        else:
            logger.warning("Telegram notification failed — check your .env credentials.")


if __name__ == "__main__":
    main()
