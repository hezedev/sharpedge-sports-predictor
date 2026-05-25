"""
Quota API Bridge - Connect hybrid fetcher quota tracking to webapp

Provides unified quota API that the webapp can call to display:
- Betfair status (unlimited, uptime)
- The Odds API quota (0-500 per month)
- API-Football daily quota (0-100 per day)

This bridges data/quota_tracker.json (hybrid fetcher) with the webapp's
dashboard display.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class QuotaAPIBridge:
    """
    Bridge between hybrid quota tracker and webapp display.

    Reads quota_tracker.json and provides unified API for the webapp
    to show quota status in the dashboard.
    """

    def __init__(self, quota_tracker_path: Path = Path("data/quota_tracker.json")):
        self.tracker_path = quota_tracker_path
        self.odds_api_limit = 500  # per month
        self.api_football_limit = 100  # per day

    def get_quota_status(self) -> Dict[str, Any]:
        """
        Get current quota status across all sources.

        Returns dict like:
        {
            "timestamp": "2026-04-20T17:15:00+00:00",
            "betfair": {
                "status": "healthy",
                "requests_today": 15,
                "requests_month": 300,
                "limit": "unlimited",
                "utilization_pct": 0
            },
            "odds_api": {
                "status": "active",
                "requests_month": 15,
                "limit": 500,
                "remaining": 485,
                "utilization_pct": 3
            },
            "api_football": {
                "status": "active",
                "requests_today": 5,
                "limit": 100,
                "remaining": 95,
                "utilization_pct": 5
            }
        }
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        this_month = datetime.now(timezone.utc).strftime("%Y-%m")

        status = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "betfair": {
                "status": "unknown",
                "requests_today": 0,
                "requests_month": 0,
                "limit": "unlimited",
                "utilization_pct": 0,
            },
            "odds_api": {
                "status": "unknown",
                "requests_today": 0,
                "requests_month": 0,
                "limit": self.odds_api_limit,
                "remaining": self.odds_api_limit,
                "utilization_pct": 0,
            },
            "api_football": {
                "status": "unknown",
                "requests_today": 0,
                "limit": self.api_football_limit,
                "remaining": self.api_football_limit,
                "utilization_pct": 0,
            },
        }

        if not self.tracker_path.exists():
            logger.warning("Quota tracker not found at %s", self.tracker_path)
            return status

        try:
            tracker = json.loads(self.tracker_path.read_text())

            # Aggregate today's data
            today_data = tracker.get(today, {})
            for source, data in today_data.items():
                requests = data.get("requests", 0)
                if source == "betfair":
                    status["betfair"]["requests_today"] = requests
                    status["betfair"]["status"] = "active" if requests > 0 else "idle"
                elif source == "odds_api":
                    status["odds_api"]["requests_today"] = requests
                    status["odds_api"]["status"] = "active" if requests > 0 else "idle"
                elif source == "api_football":
                    status["api_football"]["requests_today"] = requests
                    status["api_football"]["status"] = "active" if requests > 0 else "idle"

            # Aggregate monthly data
            month_totals = {}
            for date_str, sources in tracker.items():
                if date_str.startswith(this_month):
                    for source, data in sources.items():
                        if source not in month_totals:
                            month_totals[source] = 0
                        month_totals[source] += data.get("requests", 0)

            # Apply monthly totals
            for source in ["betfair", "odds_api", "api_football"]:
                requests = month_totals.get(source, 0)
                if source == "betfair":
                    status["betfair"]["requests_month"] = requests
                elif source == "odds_api":
                    status["odds_api"]["requests_month"] = requests
                    remaining = self.odds_api_limit - requests
                    status["odds_api"]["remaining"] = max(0, remaining)
                    utilization = requests / self.odds_api_limit * 100 if self.odds_api_limit > 0 else 0
                    status["odds_api"]["utilization_pct"] = round(utilization, 1)

            # API-Football daily (not accumulated, just today)
            api_football_today = today_data.get("api_football", {}).get("requests", 0)
            remaining_today = max(0, self.api_football_limit - api_football_today)
            status["api_football"]["remaining"] = remaining_today
            utilization_today = api_football_today / self.api_football_limit * 100
            status["api_football"]["utilization_pct"] = round(utilization_today, 1)

            return status

        except Exception as exc:
            logger.error("Error reading quota tracker: %s", exc)
            return status

    def get_simple_quota_for_webapp(self) -> Dict[str, Any]:
        """
        Get quota data in webapp-compatible format.

        Returns dict like:
        {
            "odds_remaining": 485,
            "odds_used_total": 15,
            "odds_start": 500,
            "betfair": "healthy",
            "api_football": "5/100"
        }
        """
        full_status = self.get_quota_status()

        return {
            # Keep Odds API format for backward compatibility
            "odds_remaining": full_status["odds_api"]["remaining"],
            "odds_used_total": full_status["odds_api"]["requests_month"],
            "odds_start": self.odds_api_limit,
            # Add hybrid sources
            "betfair": full_status["betfair"]["status"],
            "betfair_requests": full_status["betfair"]["requests_month"],
            "api_football": f"{full_status['api_football']['requests_today']}/{self.api_football_limit}",
            "api_football_remaining": full_status["api_football"]["remaining"],
        }

    def get_warning_level(self) -> str:
        """
        Determine warning level for UI color coding.

        Returns: "green" (ok), "yellow" (caution), "red" (critical)
        """
        status = self.get_quota_status()

        odds_utilization = status["odds_api"]["utilization_pct"]
        api_football_utilization = status["api_football"]["utilization_pct"]

        # Red: critical
        if odds_utilization > 80 or api_football_utilization > 90:
            return "red"

        # Yellow: caution
        if odds_utilization > 60 or api_football_utilization > 70:
            return "yellow"

        # Green: ok
        return "green"

    def save_legacy_api_usage(self) -> None:
        """
        Sync quota data back to api_usage.json for backward compatibility.

        This allows old code reading api_usage.json to still work.
        """
        try:
            simple_quota = self.get_simple_quota_for_webapp()

            api_usage = {
                "key_fingerprint": "hybrid_bridge",  # Flag for hybrid system
                "odds_remaining": simple_quota["odds_remaining"],
                "odds_used_total": simple_quota["odds_used_total"],
                "odds_requests_used_total": simple_quota["odds_used_total"],
                "odds_remaining_start": simple_quota["odds_start"],
                "betfair": simple_quota["betfair"],
                "api_football": simple_quota["api_football"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            usage_path = Path("data") / "api_usage.json"
            usage_path.parent.mkdir(parents=True, exist_ok=True)
            usage_path.write_text(json.dumps(api_usage, indent=2))

            logger.info("Updated api_usage.json with hybrid quota data")

        except Exception as exc:
            logger.error("Could not save legacy api_usage.json: %s", exc)
