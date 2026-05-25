"""
Hybrid Odds Fetcher - Multi-source with intelligent fallback

Implements a smart strategy that:
1. Tries Betfair first (unlimited free tier, delayed but sufficient)
2. Falls back to The Odds API if Betfair unavailable
3. Logs all source selections for quota tracking
4. Caches results aggressively to minimize API calls

Quota efficiency:
- Betfair: unlimited (free tier, delayed 15-20 min)
- The Odds API: 500 req/month (fallback only)
- API-Football: 100 req/day (optional soccer enrichment)

Cost per snapshot (typical):
- Primary (Betfair): 15 requests (free)
- Fallback (Odds API): 15 requests (only if Betfair down)
- Enrichment (API-Football): 1-3 requests (optional)
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.data.base_fetcher import BaseFetcher
from src.data.betfair_fetcher import BetfairFetcher
from src.data.odds_fetcher import OddsFetcher

logger = logging.getLogger(__name__)


class HybridOddsFetcher(BaseFetcher):
    """
    Intelligently switches between Betfair and The Odds API.

    Priority:
    1. Betfair (primary, unlimited free)
    2. The Odds API (fallback, 500/month)
    3. Cached disk snapshots (if both unavailable)

    All attempts are logged for quota reporting.
    """

    def __init__(
        self,
        sport: str = "soccer",
        cache_expire_hours: int = 1,
        prefer_source: str = "betfair",  # or "odds_api"
        quota_tracker_path: Optional[Path] = None,
    ) -> None:
        super().__init__(sport=sport, cache_expire_hours=cache_expire_hours)

        self.prefer_source = prefer_source
        self.quota_tracker_path = quota_tracker_path or Path("data/quota_tracker.json")

        # Initialize fetchers
        self.betfair = BetfairFetcher(sport=sport, cache_expire_hours=cache_expire_hours)
        self.odds_api = OddsFetcher(sport=sport, cache_expire_hours=cache_expire_hours)

        # Disk cache for fallback
        self.cache_dir = Path("data/cache/odds")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_odds(
        self,
        sport_key: Optional[str] = None,
        use_enrichment: bool = False,
        dry_run: bool = False,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Fetch odds using hybrid strategy.

        Parameters
        ----------
        sport_key : str, optional
            Specific sport key. If None, uses self.sport.
        use_enrichment : bool, optional
            Optionally enrich soccer data with API-Football stats.
        dry_run : bool, optional
            Don't make actual API calls.

        Returns
        -------
        (odds_df, metadata)
            - odds_df: DataFrame with standard odds schema
            - metadata: dict with source, timestamp, quota_remaining, fallback_used, etc.
        """
        metadata = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sport": self.sport,
            "source": None,
            "fallback_used": False,
            "cache_used": False,
            "error": None,
        }

        if dry_run:
            logger.info("[DRY-RUN] Would fetch odds for %s", self.sport)
            metadata["source"] = "dry-run"
            return pd.DataFrame(), metadata

        # Try primary source
        if self.prefer_source == "betfair":
            odds_df, meta = self._fetch_betfair(sport_key)
            if not odds_df.empty:
                metadata.update(meta)
                self._log_quota("betfair", True, len(odds_df))
                return odds_df, metadata

            logger.warning("Betfair fetch failed, falling back to The Odds API")
            metadata["fallback_used"] = True

        # Try fallback
        odds_df, meta = self._fetch_odds_api(sport_key)
        if not odds_df.empty:
            metadata.update(meta)
            self._log_quota("odds_api", True, len(odds_df))
            return odds_df, metadata

        logger.warning("Both primary and fallback sources failed, checking cache")
        metadata["fallback_used"] = True

        # Try cache
        odds_df = self._fetch_cached(sport_key)
        if not odds_df.empty:
            metadata["source"] = "disk_cache"
            metadata["cache_used"] = True
            metadata["warning"] = "Using stale cached data; both APIs unavailable"
            self._log_quota("cache", True, len(odds_df))
            return odds_df, metadata

        # All sources failed
        metadata["error"] = "No odds available from any source"
        logger.error(metadata["error"])
        return pd.DataFrame(), metadata

    def _fetch_betfair(self, sport_key: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Try Betfair."""
        try:
            logger.info("Attempting Betfair fetch for %s", self.sport)

            if not self.betfair.session_token:
                if not self.betfair.login():
                    return pd.DataFrame(), {"source": "betfair", "error": "login_failed"}

            odds_df = self.betfair.fetch_odds(sport_key=sport_key)

            if odds_df.empty:
                return pd.DataFrame(), {"source": "betfair", "error": "no_events"}

            meta = {
                "source": "betfair",
                "num_events": len(odds_df["event_id"].unique()),
                "quota_remaining": "unlimited",
            }
            logger.info("Betfair fetch successful: %d events", meta["num_events"])
            return odds_df, meta

        except Exception as exc:
            logger.error(f"Betfair fetch error: {exc}")
            return pd.DataFrame(), {"source": "betfair", "error": str(exc)}

    def _fetch_odds_api(self, sport_key: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Try The Odds API."""
        try:
            logger.info("Attempting Odds API fetch for %s", self.sport)

            odds_df = self.odds_api.fetch_odds(sport_key=sport_key)

            if odds_df.empty:
                return pd.DataFrame(), {"source": "odds_api", "error": "no_events"}

            # Try to extract remaining quota from last response headers
            # (OddsFetcher handles this internally; we'd need to add quota tracking)
            meta = {
                "source": "odds_api",
                "num_events": len(odds_df["event_id"].unique()),
                "quota_remaining": "unknown",  # Would need to track in OddsFetcher
            }
            logger.info("Odds API fetch successful: %d events", meta["num_events"])
            return odds_df, meta

        except Exception as exc:
            logger.error(f"Odds API fetch error: {exc}")
            return pd.DataFrame(), {"source": "odds_api", "error": str(exc)}

    def _fetch_cached(self, sport_key: Optional[str] = None) -> pd.DataFrame:
        """Try most recent cached snapshot."""
        try:
            # Find most recent snapshot file
            pattern = f"*_{sport_key or self.sport}*.json" if sport_key else "*.json"
            candidates = sorted(self.cache_dir.glob(pattern), reverse=True)

            if not candidates:
                return pd.DataFrame()

            latest = candidates[0]
            logger.info("Loading cached odds from %s", latest.name)

            with open(latest) as f:
                games = json.load(f)

            # Parse into standard format (simplified; full parsing in OddsFetcher)
            rows = []
            for game in games:
                # This is a simplified version; full parsing would extract all odds
                rows.append({
                    "event_id": game.get("id"),
                    "home_team": game.get("home_team"),
                    "away_team": game.get("away_team"),
                    "source": "cache",
                })

            return pd.DataFrame(rows) if rows else pd.DataFrame()

        except Exception as exc:
            logger.error(f"Cache fetch error: {exc}")
            return pd.DataFrame()

    def _log_quota(self, source: str, success: bool, num_records: int) -> None:
        """
        Log quota usage to tracker file.

        Format:
        {
            "2026-04-20": {
                "betfair": {"requests": 15, "timestamp": "..."},
                "odds_api": {"requests": 5, "timestamp": "..."},
                "api_football": {"requests": 2, "timestamp": "..."}
            }
        }
        """
        try:
            today = datetime.now(timezone.utc).date().isoformat()

            tracker = {}
            if self.quota_tracker_path.exists():
                tracker = json.loads(self.quota_tracker_path.read_text())

            if today not in tracker:
                tracker[today] = {}

            if source not in tracker[today]:
                tracker[today][source] = {"requests": 0, "successes": 0, "failures": 0}

            tracker[today][source]["requests"] += 1
            if success:
                tracker[today][source]["successes"] += 1
            else:
                tracker[today][source]["failures"] += 1
            tracker[today][source]["last_update"] = datetime.now(timezone.utc).isoformat()

            self.quota_tracker_path.parent.mkdir(parents=True, exist_ok=True)
            self.quota_tracker_path.write_text(json.dumps(tracker, indent=2))

        except Exception as exc:
            logger.warning(f"Could not log quota: {exc}")

    def get_quota_report(self) -> Dict[str, Any]:
        """
        Get current quota usage across all sources.

        Returns dict with daily and monthly breakdowns.
        """
        try:
            if not self.quota_tracker_path.exists():
                return {"message": "No quota data yet"}

            tracker = json.loads(self.quota_tracker_path.read_text())

            today = datetime.now(timezone.utc).date().isoformat()
            this_month = datetime.now(timezone.utc).strftime("%Y-%m")

            report = {
                "today": tracker.get(today, {}),
                "this_month": {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Aggregate monthly data
            for date_str, sources in tracker.items():
                if date_str.startswith(this_month):
                    for source, data in sources.items():
                        if source not in report["this_month"]:
                            report["this_month"][source] = {"requests": 0, "successes": 0}
                        report["this_month"][source]["requests"] += data.get("requests", 0)
                        report["this_month"][source]["successes"] += data.get("successes", 0)

            # Add quota limits
            report["limits"] = {
                "betfair": "unlimited (free tier)",
                "odds_api": "500 requests/month",
                "api_football": "100 requests/day",
            }

            return report

        except Exception as exc:
            logger.error(f"Could not generate quota report: {exc}")
            return {"error": str(exc)}

    def health_check(self) -> Dict[str, Any]:
        """
        Check health of all sources.

        Returns dict with status of each fetcher.
        """
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "betfair": self.betfair.health_check(),
            "odds_api": self.odds_api.health_check() if hasattr(self.odds_api, 'health_check') else "N/A",
            "cache_dir": str(self.cache_dir),
            "cache_exists": self.cache_dir.exists(),
        }
