"""
Common helper functions: retry decorators, date parsing, math utilities.
"""

import functools
import logging
import time
from datetime import datetime, date
from typing import Any, Callable, Optional, TypeVar, Union

import pandas as pd
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
import requests

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# ------------------------------------------------------------------
# Retry Decorator
# ------------------------------------------------------------------

def retry_with_backoff(
    max_attempts: int = 5,
    min_wait: float = 1.0,
    max_wait: float = 60.0,
) -> Callable[[F], F]:
    """
    Decorator factory for retrying functions with exponential backoff.

    Retries on requests.exceptions.RequestException (network errors,
    timeouts, 429/5xx via raise_for_status).

    Parameters
    ----------
    max_attempts : int
        Maximum number of retry attempts.
    min_wait : float
        Minimum wait time in seconds between retries.
    max_wait : float
        Maximum wait time in seconds between retries.

    Returns
    -------
    Callable
        Decorated function with retry behavior.
    """
    def decorator(func: F) -> F:
        @retry(
            retry=retry_if_exception_type(
                (requests.exceptions.RequestException, ConnectionError, TimeoutError)
            ),
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=min_wait, max=max_wait),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)
        return wrapper  # type: ignore[return-value]
    return decorator


# ------------------------------------------------------------------
# Date Utilities
# ------------------------------------------------------------------

def parse_date(
    value: Union[str, datetime, date, pd.Timestamp, None],
    fmt: Optional[str] = None,
) -> Optional[datetime]:
    """
    Parse a date from various input types into a datetime object.

    Parameters
    ----------
    value : str | datetime | date | pd.Timestamp | None
        The date value to parse.
    fmt : str, optional
        Explicit strptime format string. If None, common formats
        are attempted automatically.

    Returns
    -------
    datetime or None
        Parsed datetime, or None if parsing fails.
    """
    if value is None:
        return None

    if isinstance(value, datetime):
        return value

    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)

    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()

    if isinstance(value, str):
        if fmt:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                logger.warning("Failed to parse date '%s' with format '%s'", value, fmt)
                return None

        # Try common formats
        formats = [
            "%Y-%m-%d",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S%z",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%Y%m%d",
        ]
        for f in formats:
            try:
                return datetime.strptime(value, f)
            except ValueError:
                continue

        # Fallback to pandas parser
        try:
            return pd.to_datetime(value).to_pydatetime()
        except Exception:
            logger.warning("Could not parse date: '%s'", value)
            return None

    logger.warning("Unsupported date type: %s", type(value))
    return None


# ------------------------------------------------------------------
# Math Utilities
# ------------------------------------------------------------------

def safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0,
) -> float:
    """
    Safe division that returns a default value instead of raising
    ZeroDivisionError.

    Parameters
    ----------
    numerator : float
        The numerator.
    denominator : float
        The denominator.
    default : float
        Value to return if denominator is zero.

    Returns
    -------
    float
        Result of division or default.
    """
    if denominator == 0:
        return default
    return numerator / denominator


def decimal_to_implied_probability(decimal_odds: float) -> float:
    """
    Convert European decimal odds to implied probability.

    Parameters
    ----------
    decimal_odds : float
        Decimal odds (e.g. 2.50).

    Returns
    -------
    float
        Implied probability (0 to 1).
    """
    if decimal_odds <= 0:
        return 0.0
    return 1.0 / decimal_odds


def implied_probability_to_decimal(prob: float) -> float:
    """
    Convert an implied probability to decimal odds.

    Parameters
    ----------
    prob : float
        Probability between 0 and 1.

    Returns
    -------
    float
        Decimal odds.
    """
    if prob <= 0:
        return float("inf")
    return 1.0 / prob


def remove_vig(
    probabilities: list[float],
) -> list[float]:
    """
    Remove the bookmaker's overround (vig) from a set of
    implied probabilities to produce fair probabilities.

    Uses multiplicative normalization (proportional method).

    Parameters
    ----------
    probabilities : list[float]
        Raw implied probabilities (sum > 1.0).

    Returns
    -------
    list[float]
        Fair probabilities (sum ≈ 1.0).
    """
    total = sum(probabilities)
    if total <= 0:
        return probabilities
    return [p / total for p in probabilities]


def calculate_overround(probabilities: list[float]) -> float:
    """
    Calculate the bookmaker overround (margin) from implied probabilities.

    Parameters
    ----------
    probabilities : list[float]
        Implied probabilities from odds.

    Returns
    -------
    float
        Overround percentage (e.g. 0.05 = 5% margin).
    """
    return sum(probabilities) - 1.0


class RateLimiter:
    """
    Simple token-bucket rate limiter for API calls.

    Parameters
    ----------
    max_calls : int
        Maximum calls allowed in the time window.
    period_seconds : float
        Length of the time window in seconds.
    """

    def __init__(self, max_calls: int, period_seconds: float) -> None:
        self.max_calls = max_calls
        self.period = period_seconds
        self._calls: list[float] = []

    def wait_if_needed(self) -> None:
        """Block until a call can be made within the rate limit."""
        now = time.time()

        # Remove expired timestamps
        self._calls = [t for t in self._calls if now - t < self.period]

        if len(self._calls) >= self.max_calls:
            oldest = self._calls[0]
            sleep_time = self.period - (now - oldest) + 0.1
            if sleep_time > 0:
                logger.debug("Rate limit reached. Sleeping %.1fs", sleep_time)
                time.sleep(sleep_time)

        self._calls.append(time.time())

    def allow_request(self) -> bool:
        """Return whether a call can be made now, recording it when allowed."""
        now = time.time()
        self._calls = [t for t in self._calls if now - t < self.period]
        if len(self._calls) >= self.max_calls:
            return False
        self._calls.append(now)
        return True

    def __enter__(self) -> "RateLimiter":
        self.wait_if_needed()
        return self

    def __exit__(self, *args: Any) -> None:
        pass
