"""
Abstract base class for all sport data fetchers.

Provides shared infrastructure: caching, rate limiting, HTTP
request handling, and data persistence to parquet.
"""

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from config import settings
from src.utils.cache import DiskCache
from src.utils.helpers import RateLimiter, retry_with_backoff

logger = logging.getLogger(__name__)


class BaseFetcher(ABC):
    """
    Abstract base for all data fetchers.

    Subclasses must implement:
        - fetch_matches() -> pd.DataFrame
        - fetch_standings() -> pd.DataFrame
        - _parse_matches(raw) -> pd.DataFrame

    Parameters
    ----------
    sport : str
        Sport identifier ('soccer', 'basketball', 'tennis').
    cache_expire_hours : int
        How long to cache API responses.
    """

    def __init__(
        self,
        sport: str,
        cache_expire_hours: int = 24,
    ) -> None:
        self.sport = sport
        self._settings = settings
        self._sport_cfg = settings.get("sports", {}).get(sport, {})
        self._paths = settings.get("paths", {})

        # Setup cache
        self._cache = DiskCache(
            cache_name=f"{sport}_cache",
            expire_hours=cache_expire_hours,
        )

        # Raw / processed data directories (resolve to absolute paths)
        _project_root = Path(__file__).resolve().parent.parent.parent

        def _abs(rel: str) -> Path:
            p = Path(rel)
            return p if p.is_absolute() else _project_root / p

        self._raw_dir = _abs(self._paths.get("raw_data", "data/raw")) / sport
        self._processed_dir = _abs(self._paths.get("processed_data", "data/processed")) / sport
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        self._processed_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Initialized %s fetcher (cache=%dh)", sport, cache_expire_hours)

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    @retry_with_backoff(max_attempts=5, min_wait=1.0, max_wait=60.0)
    def _get(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ) -> Dict[str, Any]:
        """
        Perform a cached, rate-limited GET request.

        Parameters
        ----------
        url : str
            Full API endpoint URL.
        headers : dict, optional
            Request headers (typically with API key).
        params : dict, optional
            Query parameters.
        rate_limiter : RateLimiter, optional
            Rate limiter instance to respect API limits.

        Returns
        -------
        dict
            Parsed JSON response.

        Raises
        ------
        requests.exceptions.HTTPError
            If the response status code indicates an error.
        """
        import requests as _requests  # plain requests as fallback

        if rate_limiter:
            rate_limiter.wait_if_needed()

        try:
            response = self._cache.get(
                url=url,
                headers=headers,
                params=params,
                timeout=30,
            )
        except Exception as cache_exc:
            # Cache backend unavailable — fall back to a plain requests.get
            logger.debug("Cache.get failed (%s); using direct request.", cache_exc)
            response = _requests.get(
                url,
                headers=headers or {},
                params=params or {},
                timeout=30,
            )

        # Handle rate limit responses explicitly
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", "60"))
            logger.warning(
                "Rate limited (429) on %s. Retry-After: %ds",
                url, retry_after,
            )
            raise requests.exceptions.HTTPError(
                f"429 Rate Limited. Retry after {retry_after}s",
                response=response,
            )

        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_raw(self, data: pd.DataFrame, filename: str) -> Path:
        """Save raw DataFrame to parquet in the raw directory."""
        path = self._raw_dir / f"{filename}.parquet"
        data.to_parquet(path, index=False, engine="pyarrow")
        logger.info("Saved raw data: %s (%d rows)", path, len(data))
        return path

    def save_processed(self, data: pd.DataFrame, filename: str) -> Path:
        """Save processed DataFrame to parquet in the processed directory."""
        path = self._processed_dir / f"{filename}.parquet"
        data.to_parquet(path, index=False, engine="pyarrow")
        logger.info("Saved processed data: %s (%d rows)", path, len(data))
        return path

    def load_processed(self, filename: str) -> Optional[pd.DataFrame]:
        """Load a processed parquet file if it exists."""
        path = self._processed_dir / f"{filename}.parquet"
        if path.exists():
            df = pd.read_parquet(path, engine="pyarrow")
            logger.info("Loaded processed data: %s (%d rows)", path, len(df))
            return df
        logger.debug("No processed file found: %s", path)
        return None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fetch_matches(
        self,
        season: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch match results for a given season or date range.

        Must be implemented by each sport-specific fetcher.
        """
        ...

    @abstractmethod
    def fetch_standings(
        self,
        season: Optional[str] = None,
        league_id: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch league standings / rankings."""
        ...

    @abstractmethod
    def _parse_matches(self, raw_data: Any) -> pd.DataFrame:
        """Parse raw API response into a standardized DataFrame."""
        ...

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def fetch_all_seasons(self) -> pd.DataFrame:
        """
        Fetch matches for all configured seasons and concatenate.

        Returns
        -------
        pd.DataFrame
            Combined match data across seasons.
        """
        seasons_to_fetch = self._sport_cfg.get("seasons_to_fetch", 3)
        now = pd.Timestamp.now()
        # Football seasons start in July/August and are identified by their
        # START year (e.g. 2025/26 season = "2025").  If we're in Jan–Jul
        # the current season started the PREVIOUS calendar year.
        if self.sport == "soccer":
            current_season_year = now.year if now.month >= 8 else now.year - 1
        else:
            current_season_year = now.year
        all_frames: List[pd.DataFrame] = []

        for offset in range(seasons_to_fetch):
            season_year = current_season_year - offset
            season_str = str(season_year)
            logger.info("Fetching %s season %s", self.sport, season_str)

            try:
                df = self.fetch_matches(season=season_str)
                if df is not None and not df.empty:
                    all_frames.append(df)
            except Exception as exc:
                logger.error(
                    "Failed to fetch %s season %s: %s",
                    self.sport, season_str, exc,
                )
                continue

        if not all_frames:
            logger.warning("No data fetched for %s", self.sport)
            return pd.DataFrame()

        combined = pd.concat(all_frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=["match_id"]).sort_values(
            "date", ascending=True,
        ).reset_index(drop=True)

        self.save_raw(combined, f"{self.sport}_all_seasons")
        logger.info(
            "Combined %s data: %d matches across %d seasons",
            self.sport, len(combined), seasons_to_fetch,
        )
        return combined
