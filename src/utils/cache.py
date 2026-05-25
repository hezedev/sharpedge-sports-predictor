"""
Disk-based HTTP response cache.

Wraps requests-cache to transparently cache API responses,
respecting rate limits by serving cached data when available.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import requests_cache

from config import settings

logger = logging.getLogger(__name__)


class DiskCache:
    """
    Disk-backed cache for API responses using requests-cache.

    Uses SQLite backend where possible, falling back to in-memory
    cache if the filesystem path is not writable.

    Parameters
    ----------
    cache_name : str
        Name prefix for the cache database file.
    expire_hours : int
        Hours before cached responses expire. Default 24.
    """

    def __init__(
        self,
        cache_name: str = "sports_predictor_cache",
        expire_hours: int = 24,
    ) -> None:
        # Resolve an absolute cache directory
        cfg_path = settings.get("paths", {}).get("cache", "data/cache")
        cache_dir = Path(cfg_path)
        if not cache_dir.is_absolute():
            # Make it absolute relative to the project root (parent of src/)
            project_root = Path(__file__).resolve().parent.parent.parent
            cache_dir = project_root / cfg_path
        cache_dir.mkdir(parents=True, exist_ok=True)

        self._cache_path = cache_dir / cache_name
        self._expire_seconds = expire_hours * 3600

        # Try SQLite backend; fall back to in-memory on any I/O error
        try:
            self._session = requests_cache.CachedSession(
                cache_name=str(self._cache_path),
                backend="sqlite",
                expire_after=self._expire_seconds,
                allowable_codes=[200],
                allowable_methods=["GET"],
                stale_if_error=True,
            )
            # Probe the backend with a harmless operation
            _ = self._session.cache.responses  # triggers SQLite connection
            logger.info(
                "DiskCache initialized: path=%s, expire=%dh",
                self._cache_path,
                expire_hours,
            )
        except Exception as exc:
            logger.warning(
                "SQLite cache unavailable (%s); falling back to in-memory cache.", exc
            )
            self._session = requests_cache.CachedSession(
                cache_name=cache_name,
                backend="memory",
                expire_after=self._expire_seconds,
                allowable_codes=[200],
                allowable_methods=["GET"],
                stale_if_error=True,
            )
            logger.info("DiskCache initialized: backend=memory, expire=%dh", expire_hours)

    @property
    def session(self) -> requests_cache.CachedSession:
        """Return the cached requests session."""
        return self._session

    def get(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: int = 30,
    ) -> requests_cache.models.CachedResponse:
        """
        Perform a GET request through the cache layer.

        Parameters
        ----------
        url : str
            The URL to request.
        headers : dict, optional
            HTTP headers to include.
        params : dict, optional
            Query parameters.
        timeout : int
            Request timeout in seconds.

        Returns
        -------
        requests.Response
            The (possibly cached) response object.
        """
        response = self._session.get(
            url,
            headers=headers or {},
            params=params or {},
            timeout=timeout,
        )
        from_cache = getattr(response, "from_cache", False)
        logger.debug(
            "Cache %s: %s [%d]",
            "HIT" if from_cache else "MISS",
            url,
            response.status_code,
        )
        return response

    def clear(self) -> None:
        """Clear all cached responses."""
        self._session.cache.clear()
        logger.info("Cache cleared: %s", self._cache_path)

    def cache_stats(self) -> Dict[str, int]:
        """Return basic cache statistics."""
        urls = list(self._session.cache.urls())
        return {
            "cached_urls": len(urls),
            "backend": "sqlite",
            "expire_seconds": self._expire_seconds,
        }

    @staticmethod
    def make_cache_key(url: str, params: Optional[Dict] = None) -> str:
        """Generate a deterministic cache key from URL + params."""
        raw = url + json.dumps(params or {}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()
