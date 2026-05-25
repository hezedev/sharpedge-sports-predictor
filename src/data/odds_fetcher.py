"""
Odds data fetcher using The Odds API.

Fetches pre-match odds from multiple bookmakers for soccer,
basketball, and tennis. Provides best-price selection and
implied probability calculations.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from config import settings
from src.data.base_fetcher import BaseFetcher
from src.utils.odds_quota import save_odds_api_usage
from src.utils.sport_registry import SOCCER_SPORT_KEYS
from src.utils.helpers import (
    RateLimiter,
    decimal_to_implied_probability,
    remove_vig,
    parse_date,
)

logger = logging.getLogger(__name__)

SPORT_KEYS = {
    "soccer": [
        *SOCCER_SPORT_KEYS,
    ],
    "basketball": [
        "basketball_nba",
    ],
    "tennis": [
        "tennis_atp_french_open",
        "tennis_atp_aus_open",
        "tennis_atp_us_open",
        "tennis_atp_wimbledon",
    ],
}


class OddsFetcher(BaseFetcher):
    """
    Fetcher for odds data from The Odds API.

    Supports moneyline (h2h), spreads, and totals markets
    across soccer, basketball, and tennis.
    """

    def __init__(
        self,
        sport: str = "soccer",
        cache_expire_hours: int = 1,
    ) -> None:
        super().__init__(sport=sport, cache_expire_hours=cache_expire_hours)

        odds_cfg = settings.get("apis", {}).get("odds_api", {})
        self._base_url = odds_cfg.get("base_url", "https://api.the-odds-api.com/v4")
        self._api_key = os.environ.get("ODDS_API_KEY", "")
        self._regions = odds_cfg.get("regions", ["eu"])
        self._markets = odds_cfg.get("markets", ["h2h"])
        self._rate_limiter = RateLimiter(
            max_calls=odds_cfg.get("rate_limit_per_month", 500),
            period_seconds=30 * 86400,
        )

        if not self._api_key:
            logger.warning("ODDS_API_KEY not set. Odds fetching will fail.")
        self.last_quota: Dict[str, Any] = {}

    def _get_json_with_quota(
        self,
        *,
        url: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any] | List[Dict[str, Any]]:
        """Perform a GET request and capture Odds API quota headers."""
        self._rate_limiter.wait_if_needed()
        try:
            response = self._cache.get(
                url=url,
                headers={},
                params=params,
                timeout=30,
            )
        except Exception as cache_exc:
            logger.debug("Cache.get failed (%s); using direct request.", cache_exc)
            response = requests.get(url, headers={}, params=params, timeout=30)

        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", "60"))
            raise requests.exceptions.HTTPError(
                f"429 Rate Limited. Retry after {retry_after}s",
                response=response,
            )

        response.raise_for_status()
        self._capture_quota_headers(response)
        return response.json()

    def _capture_quota_headers(self, response: requests.Response) -> None:
        """Persist The Odds API header quota state when available."""
        remaining = response.headers.get("x-requests-remaining")
        used_total = response.headers.get("x-requests-used")
        try:
            rem_int = int(remaining) if remaining is not None else None
        except (TypeError, ValueError):
            rem_int = None
        try:
            used_int = int(used_total) if used_total is not None else None
        except (TypeError, ValueError):
            used_int = None

        self.last_quota = {
            "remaining": rem_int,
            "used_total": used_int,
        }
        if rem_int is not None or used_int is not None:
            save_odds_api_usage(
                api_key=self._api_key,
                remaining=rem_int,
                used_total=used_int,
            )

    # ------------------------------------------------------------------
    # Core fetch
    # ------------------------------------------------------------------

    def fetch_matches(
        self,
        season: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> pd.DataFrame:
        """Not used for odds; use fetch_odds() instead."""
        return self.fetch_odds()

    def fetch_standings(
        self,
        season: Optional[str] = None,
        league_id: Optional[int] = None,
    ) -> pd.DataFrame:
        """Odds API does not provide standings."""
        return pd.DataFrame()

    def _parse_matches(self, raw_data: Any) -> pd.DataFrame:
        """Not used directly; odds parsing is in _parse_odds."""
        return pd.DataFrame()

    def fetch_odds(
        self,
        sport_key: Optional[str] = None,
        markets: Optional[List[str]] = None,
        regions: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Fetch current odds for a sport.

        Parameters
        ----------
        sport_key : str, optional
            Specific Odds API sport key. If None, fetches all
            configured sport keys for self.sport.
        markets : list[str], optional
            Markets to fetch. Defaults to configured markets.
        regions : list[str], optional
            Bookmaker regions. Defaults to configured regions.

        Returns
        -------
        pd.DataFrame
            Odds data with columns: event_id, sport, commence_time,
            home_team, away_team, bookmaker, market, outcome, price,
            implied_prob, point (for spreads/totals).
        """
        sport_keys = [sport_key] if sport_key else SPORT_KEYS.get(self.sport, [])
        use_markets = markets or self._markets
        use_regions = regions or self._regions

        all_odds: List[pd.DataFrame] = []

        for sk in sport_keys:
            logger.info("Fetching odds: sport_key=%s, markets=%s", sk, use_markets)

            params: Dict[str, Any] = {
                "apiKey": self._api_key,
                "regions": ",".join(use_regions),
                "markets": ",".join(use_markets),
                "oddsFormat": "decimal",
            }

            try:
                url = f"{self._base_url}/sports/{sk}/odds"
                raw = self._get_json_with_quota(url=url, params=params)

                # raw is a list of events
                if isinstance(raw, list):
                    events = raw
                else:
                    events = raw.get("data", raw.get("response", []))

                if not events:
                    logger.info("No odds events for %s", sk)
                    continue

                df = self._parse_odds(events, sport_key=sk)
                all_odds.append(df)
                logger.info("Fetched odds for %d events from %s", len(events), sk)

                # Log remaining requests from response headers
                # (The Odds API returns this in response headers)

            except Exception as exc:
                logger.error("Error fetching odds for %s: %s", sk, exc)
                continue

        if not all_odds:
            return pd.DataFrame()

        return pd.concat(all_odds, ignore_index=True)

    def fetch_historical_odds(
        self,
        sport_key: str,
        event_id: str,
        markets: Optional[List[str]] = None,
        date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical odds for a specific event.

        Parameters
        ----------
        sport_key : str
            The Odds API sport key.
        event_id : str
            Event/match ID.
        markets : list[str], optional
            Markets to query.
        date : str, optional
            ISO date string for the odds snapshot.

        Returns
        -------
        pd.DataFrame
            Historical odds data.
        """
        use_markets = markets or self._markets

        params: Dict[str, Any] = {
            "apiKey": self._api_key,
            "regions": ",".join(self._regions),
            "markets": ",".join(use_markets),
            "oddsFormat": "decimal",
        }
        if date:
            params["date"] = date

        url = f"{self._base_url}/historical/sports/{sport_key}/events/{event_id}/odds"
        raw = self._get_json_with_quota(url=url, params=params)

        data = raw.get("data", [])
        if not data:
            return pd.DataFrame()

        return self._parse_odds(data if isinstance(data, list) else [data], sport_key=sport_key)

    # ------------------------------------------------------------------
    # Best price selection
    # ------------------------------------------------------------------

    def get_best_odds(self, odds_df: pd.DataFrame) -> pd.DataFrame:
        """
        Select the best (highest) odds for each outcome across bookmakers.

        Parameters
        ----------
        odds_df : pd.DataFrame
            Full odds DataFrame from fetch_odds().

        Returns
        -------
        pd.DataFrame
            Best price per event per outcome, with bookmaker attribution.
        """
        if odds_df.empty:
            return odds_df

        idx = odds_df.groupby(["event_id", "market", "outcome"])["price"].idxmax()
        best = odds_df.loc[idx].copy()
        best = best.rename(columns={"bookmaker": "best_bookmaker"})
        logger.info("Extracted best odds for %d outcomes", len(best))
        return best.reset_index(drop=True)

    def get_consensus_odds(self, odds_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate market consensus (average) odds per outcome.

        Parameters
        ----------
        odds_df : pd.DataFrame
            Full odds DataFrame.

        Returns
        -------
        pd.DataFrame
            Consensus odds with average price, implied prob, and
            vig-removed fair probability.
        """
        if odds_df.empty:
            return odds_df

        consensus = (
            odds_df.groupby(["event_id", "home_team", "away_team", "market", "outcome"])
            .agg(
                avg_price=("price", "mean"),
                min_price=("price", "min"),
                max_price=("price", "max"),
                num_bookmakers=("bookmaker", "nunique"),
            )
            .reset_index()
        )

        consensus["avg_implied_prob"] = consensus["avg_price"].apply(
            decimal_to_implied_probability
        )

        # Remove vig per event/market
        fair_probs = []
        for (eid, market), group in consensus.groupby(["event_id", "market"]):
            raw_probs = group["avg_implied_prob"].tolist()
            fair = remove_vig(raw_probs)
            fair_probs.extend(fair)

        consensus["fair_prob"] = fair_probs
        return consensus

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_odds(
        self,
        events: List[Dict[str, Any]],
        sport_key: str = "",
    ) -> pd.DataFrame:
        """
        Parse raw Odds API events into a flat DataFrame.

        Parameters
        ----------
        events : list[dict]
            List of event dictionaries from the API.
        sport_key : str
            Sport key for tagging.

        Returns
        -------
        pd.DataFrame
            Flat odds table.
        """
        rows = []
        for event in events:
            event_id = event.get("id")
            commence = parse_date(event.get("commence_time"))
            home = event.get("home_team", "")
            away = event.get("away_team", "")

            for bookmaker in event.get("bookmakers", []):
                bk_name = bookmaker.get("title", "")

                for market in bookmaker.get("markets", []):
                    market_key = market.get("key", "")

                    for outcome in market.get("outcomes", []):
                        price = outcome.get("price", 0.0)
                        point = outcome.get("point")  # For spreads / totals

                        rows.append({
                            "event_id": event_id,
                            "sport_key": sport_key,
                            "commence_time": commence,
                            "home_team": home,
                            "away_team": away,
                            "bookmaker": bk_name,
                            "market": market_key,
                            "outcome": outcome.get("name", ""),
                            "price": price,
                            "point": point,
                            "implied_prob": decimal_to_implied_probability(price),
                        })

        df = pd.DataFrame(rows)
        if "commence_time" in df.columns and not df.empty:
            df["commence_time"] = pd.to_datetime(df["commence_time"])

        return df
