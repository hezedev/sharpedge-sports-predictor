"""
Betfair Exchange API Odds Fetcher

Provides free access to delayed (15-20 min) streaming odds data for all sports.

Advantages:
- 100% free with Delayed App Key (no monthly quota limit like The Odds API)
- Covers all sports (soccer, tennis, basketball, MLB, NHL, etc.)
- Provides real-time streaming capability (delayed on free tier)
- Deep liquidity data (matched odds, unmatched orders)

Limitations:
- Delayed data on free tier (~15-20 min for closing lines, acceptable for daily snapshots)
- Requires registration (free)
- More complex API than The Odds API

Usage:
    fetcher = BetfairFetcher()
    fetcher.login()
    odds_df = fetcher.fetch_odds(sport="soccer")
    # Returns same schema as OddsFetcher for drop-in compatibility
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from src.data.base_fetcher import BaseFetcher
from src.utils.helpers import decimal_to_implied_probability, RateLimiter

logger = logging.getLogger(__name__)


class BetfairFetcher(BaseFetcher):
    """
    Betfair Exchange API wrapper for free delayed odds.

    Uses the Betfair Streaming API (ESA - Exchange Streaming API) to fetch
    matched odds and unmatched orders. On the free tier, data is delayed
    15-20 minutes but sufficient for daily closing-line snapshots.
    """

    def __init__(
        self,
        sport: str = "soccer",
        cache_expire_hours: int = 1,
        use_free_tier: bool = True,
    ) -> None:
        super().__init__(sport=sport, cache_expire_hours=cache_expire_hours)

        self.use_free_tier = use_free_tier
        self.app_key = os.environ.get("BETFAIR_APP_KEY", "")
        self.username = os.environ.get("BETFAIR_USERNAME", "")
        self.password = os.environ.get("BETFAIR_PASSWORD", "")
        self.session_token: Optional[str] = None
        self.session_id: Optional[str] = None

        # Betfair API endpoints
        self.login_url = "https://api.betfair.com/exchange/betting/json-rpc/v1"
        self.streaming_url = "https://stream-api.betfair.com/apd/8.9/betting"
        self.market_data_url = "https://api.betfair.com/exchange/betting/json-rpc/v1"

        # Rate limiter: free tier is unlimited on streaming but be respectful
        self._rate_limiter = RateLimiter(max_calls=1000, period_seconds=3600)

        # Sport → Betfair market types mapping
        self.sport_to_market_types = {
            "soccer": ["MATCH_ODDS", "GOALS_MATCH"],
            "basketball": ["MATCH_ODDS"],
            "tennis": ["MATCH_ODDS"],
            "mlb": ["MATCH_ODDS"],
            "nhl": ["MATCH_ODDS"],
        }

        if not (self.app_key and self.username and self.password):
            logger.warning(
                "Betfair credentials not set in environment. "
                "Set BETFAIR_APP_KEY, BETFAIR_USERNAME, BETFAIR_PASSWORD."
            )

    def login(self) -> bool:
        """
        Authenticate with Betfair API using interactive login.

        Returns True if successful, False otherwise.
        For production, consider storing session token with expiry.
        """
        if not (self.app_key and self.username and self.password):
            logger.error("Betfair credentials not configured")
            return False

        try:
            payload = {
                "jsonrpc": "2.0",
                "method": "AuthenticationService/login",
                "params": {
                    "appKey": self.app_key,
                    "username": self.username,
                    "password": self.password,
                },
                "id": 1,
            }

            resp = requests.post(
                self.login_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            resp.raise_for_status()

            result = resp.json()
            if "result" in result:
                self.session_token = result["result"].get("sessionToken")
                self.session_id = result["result"].get("sessionId")
                logger.info(f"Betfair login successful. Session: {self.session_id}")
                return True
            else:
                error = result.get("error", "Unknown error")
                logger.error(f"Betfair login failed: {error}")
                return False

        except Exception as exc:
            logger.error(f"Betfair login error: {exc}")
            return False

    def logout(self) -> None:
        """End Betfair session."""
        if not self.session_token:
            return

        try:
            payload = {
                "jsonrpc": "2.0",
                "method": "AuthenticationService/logout",
                "params": {"appKey": self.app_key},
                "id": 1,
            }
            requests.post(
                self.login_url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "X-Authentication": self.session_token,
                },
                timeout=10,
            )
            logger.info("Betfair logout successful")
        except Exception as exc:
            logger.error(f"Betfair logout error: {exc}")

    def fetch_matches(
        self,
        season: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch matched events for the sport."""
        return self.fetch_odds()

    def fetch_standings(
        self,
        season: Optional[str] = None,
        league_id: Optional[int] = None,
    ) -> pd.DataFrame:
        """Betfair does not provide standings."""
        return pd.DataFrame()

    def _parse_matches(self, raw_data: Any) -> pd.DataFrame:
        """Not used; odds parsing is in _parse_odds."""
        return pd.DataFrame()

    def fetch_odds(
        self,
        sport_key: Optional[str] = None,
        markets: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Fetch current matched odds for a sport from Betfair Exchange.

        Uses the Betfair Streaming API to fetch live matched odds.
        On free tier, data is delayed 15-20 minutes.

        Returns same schema as OddsFetcher for compatibility:
        - event_id, sport, commence_time, home_team, away_team,
          bookmaker (Betfair), market, outcome, price, implied_prob
        """
        if not self.session_token:
            if not self.login():
                logger.error("Cannot fetch odds without valid session")
                return pd.DataFrame()

        try:
            # Fetch market catalogue (events + odds)
            events = self._fetch_market_catalogue()
            if not events:
                logger.info(f"No Betfair events found for {self.sport}")
                return pd.DataFrame()

            # Parse into standard odds format
            df = self._parse_betfair_odds(events)
            logger.info(f"Fetched {len(events)} events with odds from Betfair")

            return df

        except Exception as exc:
            logger.error(f"Error fetching Betfair odds: {exc}")
            return pd.DataFrame()

    def _fetch_market_catalogue(self) -> List[Dict[str, Any]]:
        """
        Fetch Betfair market catalogue (events + runners + odds).

        Returns list of market dicts with matched odds.
        """
        if not self.session_token:
            return []

        market_types = self.sport_to_market_types.get(self.sport, ["MATCH_ODDS"])

        try:
            payload = {
                "jsonrpc": "2.0",
                "method": "SportsDataService/listMarketCatalogue",
                "params": {
                    "appKey": self.app_key,
                    "marketFilter": {
                        "marketTypes": market_types,
                        "inPlayOnly": False,
                    },
                    "marketProjection": [
                        "MARKET_DESCRIPTION",
                        "RUNNER_DESCRIPTION",
                        "RUNNER_METADATA",
                        "EVENT",
                        "COMPETITION",
                    ],
                    "maxResults": 200,
                },
                "id": 1,
            }

            resp = requests.post(
                self.market_data_url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "X-Authentication": self.session_token,
                },
                timeout=15,
            )
            resp.raise_for_status()

            result = resp.json()
            markets = result.get("result", [])

            # For each market, fetch its matched odds
            enriched = []
            for market in markets:
                market_id = market.get("marketId")
                if market_id:
                    # Fetch odds for this market
                    odds = self._fetch_matched_odds(market_id)
                    market["matchedOdds"] = odds
                    enriched.append(market)

            return enriched

        except Exception as exc:
            logger.error(f"Error fetching market catalogue: {exc}")
            return []

    def _fetch_matched_odds(self, market_id: str) -> List[Dict[str, Any]]:
        """Fetch matched odds for a specific market."""
        if not self.session_token:
            return []

        try:
            payload = {
                "jsonrpc": "2.0",
                "method": "SportsDataService/listMarketBook",
                "params": {
                    "appKey": self.app_key,
                    "marketIds": [market_id],
                    "priceProjection": {
                        "priceData": ["EX_BEST_OFFERS"],
                    },
                },
                "id": 1,
            }

            resp = requests.post(
                self.market_data_url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "X-Authentication": self.session_token,
                },
                timeout=15,
            )
            resp.raise_for_status()

            result = resp.json()
            books = result.get("result", [])

            odds = []
            for book in books:
                for runner in book.get("runners", []):
                    ex = runner.get("ex", {})
                    # Back odds (backer's perspective)
                    for back in ex.get("availableToBack", [])[:1]:  # take best back
                        odds.append({
                            "runner_id": runner.get("selectionId"),
                            "runner_name": runner.get("runnerName"),
                            "back_price": back.get("price"),
                            "back_size": back.get("size"),
                        })

            return odds

        except Exception as exc:
            logger.error(f"Error fetching matched odds for {market_id}: {exc}")
            return []

    def _parse_betfair_odds(self, markets: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Parse Betfair markets into standard odds DataFrame.

        Maps to: event_id, sport, commence_time, home_team, away_team,
                 bookmaker, market, outcome, price, implied_prob
        """
        rows = []

        for market in markets:
            try:
                market_desc = market.get("description", {})
                event = market_desc.get("event", {})
                event_id = event.get("id", "")
                event_name = event.get("name", "")

                # Extract home/away from event name (e.g., "Team A vs Team B")
                teams = event_name.split(" vs ")
                home_team = teams[0].strip() if len(teams) > 0 else ""
                away_team = teams[1].strip() if len(teams) > 1 else ""

                commence = market_desc.get("marketTime", datetime.now(timezone.utc).isoformat())

                runners = market.get("runners", [])

                # Parse each runner (outcome)
                for runner in runners:
                    runner_name = runner.get("runnerName", "")

                    # Use matched odds if available, else from market data
                    matched_odds = market.get("matchedOdds", [])
                    runner_odds = next((o for o in matched_odds
                                       if o.get("runner_name") == runner_name), None)

                    if runner_odds:
                        price = runner_odds.get("back_price", 0.0)
                    else:
                        # Fallback: extract from any available odds data
                        price = 1.0  # Default neutral price

                    if price > 1.0:  # Valid decimal odds
                        rows.append({
                            "event_id": event_id,
                            "sport_key": f"betfair_{self.sport}",
                            "commence_time": pd.to_datetime(commence),
                            "home_team": home_team,
                            "away_team": away_team,
                            "bookmaker": "Betfair Exchange",
                            "market": "h2h",  # Simplify to h2h for compatibility
                            "outcome": runner_name,
                            "price": price,
                            "point": None,
                            "implied_prob": decimal_to_implied_probability(price),
                        })

            except Exception as exc:
                logger.warning(f"Error parsing Betfair market: {exc}")
                continue

        return pd.DataFrame(rows)

    def health_check(self) -> Dict[str, Any]:
        """
        Check Betfair API status and credentials.

        Returns dict with status info.
        """
        status = {
            "service": "Betfair",
            "authenticated": bool(self.session_token),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if not self.session_token:
            if self.login():
                status["authenticated"] = True
                status["message"] = "Login successful"
            else:
                status["message"] = "Login failed"

        return status
