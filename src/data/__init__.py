"""Data fetching modules for all supported sports and odds providers."""

from src.data.base_fetcher import BaseFetcher
from src.data.soccer_fetcher import SoccerFetcher
from src.data.basketball_fetcher import BasketballFetcher
from src.data.tennis_fetcher import TennisFetcher
from src.data.odds_fetcher import OddsFetcher

__all__ = [
    "BaseFetcher",
    "SoccerFetcher",
    "BasketballFetcher",
    "TennisFetcher",
    "OddsFetcher",
]
