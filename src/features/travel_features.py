"""
Travel distance and timezone shift feature engineering.

Adds fatigue-relevant travel context:
    - home_tz_offset, away_tz_offset  — UTC offset of each team's home city
    - home_travel_tz_shift            — timezone hours shifted for the home team
      (0 if playing at home, |home_tz - last_city_tz| if travelling to home)
    - away_travel_tz_shift            — timezone hours the away team crossed
    - away_is_traveling               — 1 if away team is not at home
    - home_travel_burden              — rough travel distance bucket (0=home, 1=regional, 2=cross-country, 3=cross-continent)

The key research finding: teams flying across 2+ time zones perform worse
the next day, especially in the first half. This signal is orthogonal to
back-to-back flags (b2b covers schedule density; TZ covers biological cost).

Usage
-----
Call add_travel_features(df, sport) where df has:
    home_team, away_team, date  columns

Returns df with travel columns added.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Team → (UTC offset, latitude, longitude)
# ──────────────────────────────────────────────────────────────────────────────
# UTC offsets are approximate standard time (not DST adjusted — adds noise
# we don't want; the direction of travel is what matters).
# Lat/lon used for distance calculation.

_NBA_TEAMS: Dict[str, Tuple[float, float, float]] = {
    # (utc_offset, lat, lon)
    "Atlanta Hawks":           (-5, 33.7, -84.4),
    "Boston Celtics":          (-5, 42.4, -71.1),
    "Brooklyn Nets":           (-5, 40.7, -74.0),
    "Charlotte Hornets":       (-5, 35.2, -80.8),
    "Chicago Bulls":           (-6, 41.9, -87.6),
    "Cleveland Cavaliers":     (-5, 41.5, -81.7),
    "Dallas Mavericks":        (-6, 32.8, -96.8),
    "Denver Nuggets":          (-7, 39.7, -104.9),
    "Detroit Pistons":         (-5, 42.3, -83.0),
    "Golden State Warriors":   (-8, 37.8, -122.4),
    "Houston Rockets":         (-6, 29.8, -95.4),
    "Indiana Pacers":          (-5, 39.8, -86.2),
    "LA Clippers":             (-8, 34.0, -118.3),
    "Los Angeles Clippers":    (-8, 34.0, -118.3),
    "Los Angeles Lakers":      (-8, 34.0, -118.3),
    "Memphis Grizzlies":       (-6, 35.1, -90.0),
    "Miami Heat":              (-5, 25.8, -80.2),
    "Milwaukee Bucks":         (-6, 43.0, -87.9),
    "Minnesota Timberwolves":  (-6, 44.9, -93.3),
    "New Orleans Pelicans":    (-6, 30.0, -90.1),
    "New York Knicks":         (-5, 40.8, -73.9),
    "Oklahoma City Thunder":   (-6, 35.5, -97.5),
    "Orlando Magic":           (-5, 28.5, -81.4),
    "Philadelphia 76ers":      (-5, 39.9, -75.2),
    "Phoenix Suns":            (-7, 33.4, -112.1),
    "Portland Trail Blazers":  (-8, 45.5, -122.7),
    "Sacramento Kings":        (-8, 38.6, -121.5),
    "San Antonio Spurs":       (-6, 29.4, -98.4),
    "Toronto Raptors":         (-5, 43.6, -79.4),
    "Utah Jazz":               (-7, 40.8, -111.9),
    "Washington Wizards":      (-5, 38.9, -77.0),
    # Rebrands / alt names
    "New Jersey Nets":         (-5, 40.7, -74.0),
    "Seattle SuperSonics":     (-8, 47.6, -122.3),
}

_NHL_TEAMS: Dict[str, Tuple[float, float, float]] = {
    "Anaheim Ducks":           (-8, 33.8, -117.9),
    "Arizona Coyotes":         (-7, 33.5, -112.2),
    "Boston Bruins":           (-5, 42.4, -71.1),
    "Buffalo Sabres":          (-5, 42.9, -78.9),
    "Calgary Flames":          (-7, 51.0, -114.1),
    "Carolina Hurricanes":     (-5, 35.8, -78.7),
    "Chicago Blackhawks":      (-6, 41.9, -87.6),
    "Colorado Avalanche":      (-7, 39.7, -104.9),
    "Columbus Blue Jackets":   (-5, 40.0, -83.0),
    "Dallas Stars":            (-6, 32.8, -96.8),
    "Detroit Red Wings":       (-5, 42.3, -83.1),
    "Edmonton Oilers":         (-7, 53.5, -113.5),
    "Florida Panthers":        (-5, 26.2, -80.3),
    "Los Angeles Kings":       (-8, 34.0, -118.3),
    "Minnesota Wild":          (-6, 44.9, -93.1),
    "Montréal Canadiens":      (-5, 45.5, -73.6),
    "Montreal Canadiens":      (-5, 45.5, -73.6),
    "Nashville Predators":     (-6, 36.2, -86.8),
    "New Jersey Devils":       (-5, 40.7, -74.2),
    "New York Islanders":      (-5, 40.7, -73.6),
    "New York Rangers":        (-5, 40.8, -73.9),
    "Ottawa Senators":         (-5, 45.3, -75.9),
    "Philadelphia Flyers":     (-5, 39.9, -75.2),
    "Pittsburgh Penguins":     (-5, 40.4, -80.0),
    "San Jose Sharks":         (-8, 37.3, -121.9),
    "Seattle Kraken":          (-8, 47.6, -122.3),
    "St. Louis Blues":         (-6, 38.6, -90.2),
    "Tampa Bay Lightning":     (-5, 27.9, -82.5),
    "Toronto Maple Leafs":     (-5, 43.6, -79.4),
    "Utah Hockey Club":        (-7, 40.8, -111.9),
    "Utah Mammoth":            (-7, 40.8, -111.9),
    "Vancouver Canucks":       (-8, 49.3, -123.1),
    "Vegas Golden Knights":    (-8, 36.1, -115.2),
    "Washington Capitals":     (-5, 38.9, -77.0),
    "Winnipeg Jets":           (-6, 49.9, -97.1),
    # Defunct / relocated
    "Phoenix Coyotes":         (-7, 33.5, -112.2),
    "Thrashers":               (-5, 33.7, -84.4),
}

_MLB_TEAMS: Dict[str, Tuple[float, float, float]] = {
    "Arizona Diamondbacks":    (-7, 33.4, -112.1),
    "Atlanta Braves":          (-5, 33.9, -84.5),
    "Baltimore Orioles":       (-5, 39.3, -76.6),
    "Boston Red Sox":          (-5, 42.3, -71.1),
    "Chicago Cubs":            (-6, 41.9, -87.7),
    "Chicago White Sox":       (-6, 41.8, -87.6),
    "Cincinnati Reds":         (-5, 39.1, -84.5),
    "Cleveland Guardians":     (-5, 41.5, -81.7),
    "Colorado Rockies":        (-7, 39.8, -104.9),
    "Detroit Tigers":          (-5, 42.3, -83.0),
    "Houston Astros":          (-6, 29.8, -95.4),
    "Kansas City Royals":      (-6, 39.1, -94.5),
    "Los Angeles Angels":      (-8, 33.8, -117.9),
    "Los Angeles Dodgers":     (-8, 34.1, -118.2),
    "Miami Marlins":           (-5, 25.8, -80.2),
    "Milwaukee Brewers":       (-6, 43.0, -87.9),
    "Minnesota Twins":         (-6, 44.9, -93.3),
    "New York Mets":           (-5, 40.8, -73.8),
    "New York Yankees":        (-5, 40.8, -73.9),
    "Oakland Athletics":       (-8, 37.8, -122.2),
    "Athletics":               (-8, 37.8, -122.2),
    "Philadelphia Phillies":   (-5, 39.9, -75.2),
    "Pittsburgh Pirates":      (-5, 40.4, -80.0),
    "San Diego Padres":        (-8, 32.7, -117.2),
    "San Francisco Giants":    (-8, 37.8, -122.4),
    "Seattle Mariners":        (-8, 47.6, -122.3),
    "St. Louis Cardinals":     (-6, 38.6, -90.2),
    "Tampa Bay Rays":          (-5, 27.8, -82.6),
    "Texas Rangers":           (-6, 32.7, -97.1),
    "Toronto Blue Jays":       (-5, 43.6, -79.4),
    "Washington Nationals":    (-5, 38.9, -77.0),
    # Relocated
    "Cleveland Indians":       (-5, 41.5, -81.7),
    "Florida Marlins":         (-5, 25.8, -80.2),
    "Montreal Expos":          (-5, 45.5, -73.6),
}

_SOCCER_TIMEZONES: Dict[str, float] = {
    # European leagues: most teams in UTC+1 (CET) standard
    # UK/Ireland: UTC+0
    # Default to UTC+1 for any unlisted European club
}

# Combined lookup: sport → team_map
_TEAM_MAPS = {
    "basketball": _NBA_TEAMS,
    "nhl":        _NHL_TEAMS,
    "mlb":        _MLB_TEAMS,
    "soccer":     {},   # Soccer uses generic European tz
    "tennis":     {},
}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points in km."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def _travel_bucket(km: float) -> int:
    """Convert distance km to travel burden bucket (0–3)."""
    if km < 200:
        return 0   # local / no meaningful travel
    elif km < 800:
        return 1   # regional (1–2 hours flight)
    elif km < 2500:
        return 2   # cross-country (3–5 hours)
    else:
        return 3   # cross-continent (5+ hours)


def _lookup(team: str, sport: str) -> Optional[Tuple[float, float, float]]:
    """Return (tz_offset, lat, lon) or None if team not found."""
    tmap = _TEAM_MAPS.get(sport, {})
    if team in tmap:
        return tmap[team]
    # Fuzzy match: check if any key is a substring of the team name
    tl = team.lower()
    for key, val in tmap.items():
        if key.lower() in tl or tl in key.lower():
            return val
    return None


def add_travel_features(df: pd.DataFrame, sport: str) -> pd.DataFrame:
    """
    Add travel distance and timezone shift features to a game DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: home_team, away_team, date
    sport : str
        One of: basketball, nhl, mlb, soccer, tennis

    Returns
    -------
    pd.DataFrame with added columns:
        home_tz, away_tz           — UTC offset of each team's home city
        away_travel_tz_shift       — |tz difference| the away team crossed
        away_travel_km             — straight-line km from away HQ to home arena
        away_travel_bucket         — 0=local, 1=regional, 2=cross-country, 3=cross-cont
        home_tz, away_tz           — UTC offsets (useful for time-of-day modelling)
    """
    if df.empty:
        return df

    home_tz_list, away_tz_list = [], []
    away_tz_shift_list, away_km_list, away_bucket_list = [], [], []

    for _, row in df.iterrows():
        ht = row.get("home_team", "")
        at = row.get("away_team", "")

        h_info = _lookup(ht, sport)
        a_info = _lookup(at, sport)

        if h_info and a_info:
            h_tz, h_lat, h_lon = h_info
            a_tz, a_lat, a_lon = a_info

            tz_shift = abs(h_tz - a_tz)            # tz hours crossed by away team
            km = _haversine_km(a_lat, a_lon, h_lat, h_lon)  # away team travel distance
            bucket = _travel_bucket(km)
        else:
            h_tz = np.nan
            a_tz = np.nan
            tz_shift = np.nan
            km = np.nan
            bucket = np.nan

        home_tz_list.append(h_tz)
        away_tz_list.append(a_tz)
        away_tz_shift_list.append(tz_shift)
        away_km_list.append(km)
        away_bucket_list.append(bucket)

    df = df.copy()
    df["home_tz"]             = home_tz_list
    df["away_tz"]             = away_tz_list
    df["away_travel_tz_shift"] = away_tz_shift_list
    df["away_travel_km"]       = away_km_list
    df["away_travel_bucket"]   = away_bucket_list

    # Tz advantage: positive = home team has circadian advantage (they traveled west)
    df["tz_advantage"] = df["home_tz"] - df["away_tz"]

    # Away is traveling vs. playing at home (1 always, but signal is in *how far*)
    # We flag cross-country travel (bucket >= 2) separately as that's where the effect shows
    df["away_cross_country"] = (df["away_travel_bucket"] >= 2).astype(float)
    df["away_crossed_2tz"]   = (df["away_travel_tz_shift"] >= 2).astype(float)

    # Coverage stats
    covered = df["home_tz"].notna().mean() * 100
    logger.info(
        "Travel features added for %s: %.0f%% team coverage",
        sport, covered,
    )
    return df
