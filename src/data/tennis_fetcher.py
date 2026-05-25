"""
Tennis data fetcher.

Primary source: Jeff Sackmann ATP/WTA Dataset (GitHub).
Fallback source: tennis-data.co.uk (Excel files, updated through current year).

Sackmann has full serve statistics but stops publishing the current year until
it is complete. tennis-data.co.uk is updated weekly with results + rankings
but lacks per-match serve stats. Rolling serve-stat features carry forward from
the last Sackmann-sourced game when fallback data is used.
"""

import io
import logging
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests

from config import settings

logger = logging.getLogger(__name__)

_SACKMANN_BASES = {
    "atp": "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master",
    "wta": "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master",
}

# tennis-data.co.uk: one xlsx per year per tour
# ATP:  http://www.tennis-data.co.uk/{year}/{year}.xlsx
# WTA:  http://www.tennis-data.co.uk/{year}w/{year}.xlsx
_TDCUK_URL_TEMPLATES = {
    "atp": "http://www.tennis-data.co.uk/{year}/{year}.xlsx",
    "wta": "http://www.tennis-data.co.uk/{year}w/{year}.xlsx",
}

_TDCUK_ROUND_MAP = {
    "1st Round":    "R64",
    "2nd Round":    "R32",
    "3rd Round":    "R16",
    "4th Round":    "R16",
    "Quarterfinals": "QF",
    "Semifinals":   "SF",
    "The Final":    "F",
    "Round Robin":  "RR",
}

# Map Series (ATP) / Tier (WTA) → our internal tourney_level_name
_TDCUK_ATP_LEVEL_MAP = {
    "Grand Slam":  "Grand Slam",
    "Masters Cup": "ATP Finals",
    "Masters 1000": "Masters",
    "ATP500":      "ATP250/500",
    "ATP250":      "ATP250/500",
}
_TDCUK_WTA_LEVEL_MAP = {
    "Grand Slam":         "Grand Slam",
    "Tour Championships": "ATP Finals",
    "WTA1000":            "Masters",
    "WTA500":             "ATP250/500",
    "WTA250":             "ATP250/500",
}

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache"

_ROUND_ORDER = {
    "R128": 1, "R64": 2, "R32": 3, "R16": 4,
    "QF": 5, "SF": 6, "BR": 6, "F": 7, "RR": 3,
}


class TennisFetcher:
    """
    Fetches ATP/WTA match data.

    Tries JeffSackmann GitHub first (full serve stats). If that returns 404
    (year not yet published), falls back to tennis-data.co.uk (results +
    rankings, no serve stats — rolling serve features carry forward).

    Parameters
    ----------
    seasons : list[int], optional
        Calendar years to fetch. Defaults to last 4 years including current.
    tours : list[str], optional
        'atp', 'wta', or both. Defaults to ['atp'].
    cache_hours : int
        How long to keep downloaded files on disk. Current-year files are
        re-downloaded more aggressively (live season in progress).
    """

    def __init__(
        self,
        seasons: Optional[List[int]] = None,
        tours: Optional[List[str]] = None,
        cache_hours: int = 24,
    ) -> None:
        cfg = settings.get("sports", {}).get("tennis", {})
        n_seasons = cfg.get("seasons_to_fetch", 3)

        import datetime
        current_year = datetime.datetime.now().year
        self._current_year = current_year
        self._seasons = seasons or list(range(current_year - n_seasons, current_year + 1))
        self._tours = [t.lower() for t in (tours or cfg.get("tours", ["atp"])) if t.lower() in _SACKMANN_BASES]
        if not self._tours:
            self._tours = ["atp"]
        self._cache_hours = cache_hours
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_matches(self, season: Optional[int] = None) -> pd.DataFrame:
        years = [season] if season else self._seasons
        frames = []
        for yr in years:
            for tour in self._tours:
                df = self._fetch_year(yr, tour=tour)
                if df is not None and not df.empty:
                    frames.append(df)
        if not frames:
            return pd.DataFrame()
        combined = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset=["match_id"])
            .sort_values("date")
            .reset_index(drop=True)
        )
        logger.info("Tennis data: %d matches (%s)", len(combined), years)
        return combined

    def fetch_all_seasons(self) -> pd.DataFrame:
        return self.fetch_matches()

    # ------------------------------------------------------------------
    # Internal: fetch one year
    # ------------------------------------------------------------------

    def _fetch_year(self, year: int, *, tour: str = "atp") -> Optional[pd.DataFrame]:
        cache_path = _CACHE_DIR / f"tennis_{tour}_{year}.csv"

        # Current / in-progress years expire faster to pick up new results
        ttl = 6 if year >= self._current_year else self._cache_hours
        if cache_path.exists():
            age_h = (
                pd.Timestamp.now() - pd.Timestamp(cache_path.stat().st_mtime, unit="s")
            ).total_seconds() / 3600
            if age_h < ttl:
                logger.info("Tennis %s %d: loading from cache", tour.upper(), year)
                return self._parse_cached(pd.read_csv(cache_path), year, tour=tour)

        # 1. Try JeffSackmann (full serve stats)
        df = self._fetch_sackmann(year, tour=tour)
        if df is not None:
            df.to_csv(cache_path, index=False)
            return self._parse(df, year, tour=tour)

        # 2. Fallback: tennis-data.co.uk (results + rankings, no serve stats)
        logger.info("Tennis %s %d: Sackmann unavailable — trying tennis-data.co.uk", tour.upper(), year)
        df_tdcuk = self._fetch_tdcuk(year, tour=tour)
        if df_tdcuk is not None:
            # Save as CSV so the cache works next time
            df_tdcuk.to_csv(cache_path, index=False)
            logger.info("Tennis %s %d: fetched %d matches from tennis-data.co.uk", tour.upper(), year, len(df_tdcuk))
            return df_tdcuk

        logger.warning("Tennis %s %d: no data from any source", tour.upper(), year)
        return None

    def _fetch_sackmann(self, year: int, *, tour: str = "atp") -> Optional[pd.DataFrame]:
        base = _SACKMANN_BASES[tour]
        url = f"{base}/{tour}_matches_{year}.csv"
        logger.info("Tennis %s %d: trying Sackmann %s", tour.upper(), year, url)
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return pd.read_csv(io.StringIO(r.text))
        except Exception as exc:
            logger.warning("Sackmann %s %d failed: %s", tour.upper(), year, exc)
            return None

    def _fetch_tdcuk(self, year: int, *, tour: str = "atp") -> Optional[pd.DataFrame]:
        url = _TDCUK_URL_TEMPLATES[tour].format(year=year)
        try:
            r = requests.get(url, timeout=30)
            if r.status_code in (404, 300):
                # 300 = directory listing, file doesn't exist
                if r.status_code == 300:
                    # Follow redirect manually
                    import urllib.parse
                    loc = r.headers.get("Location", "")
                    if loc and loc.endswith(".xlsx"):
                        r = requests.get(loc if loc.startswith("http") else urllib.parse.urljoin(url, loc), timeout=30)
                        r.raise_for_status()
                    else:
                        return None
                else:
                    return None
            r.raise_for_status()
            raw = pd.read_excel(io.BytesIO(r.content))
            return self._parse_tdcuk(raw, year, tour=tour)
        except Exception as exc:
            logger.warning("tennis-data.co.uk %s %d failed: %s", tour.upper(), year, exc)
            return None

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse(raw: pd.DataFrame, year: int, *, tour: str = "atp") -> pd.DataFrame:
        """Parse JeffSackmann CSV into project schema."""
        df = raw.copy()
        df["date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
        df["match_id"] = df["tourney_id"].astype(str) + "_" + df["match_num"].astype(str)
        df["surface"] = df["surface"].fillna("Hard")
        df["round_num"] = df["round"].map(_ROUND_ORDER).fillna(3)
        level_map = {"G": "Grand Slam", "M": "Masters", "A": "ATP250/500",
                     "D": "Davis Cup", "F": "ATP Finals", "C": "Challenger"}
        df["tourney_level_name"] = df["tourney_level"].map(level_map).fillna("ATP250/500")
        df = df.rename(columns={
            "w_ace": "p1_ace", "w_df": "p1_df",
            "w_svpt": "p1_svpt", "w_1stIn": "p1_1stIn",
            "w_1stWon": "p1_1stWon", "w_2ndWon": "p1_2ndWon",
            "w_bpSaved": "p1_bpSaved", "w_bpFaced": "p1_bpFaced",
            "l_ace": "p2_ace", "l_df": "p2_df",
            "l_svpt": "p2_svpt", "l_1stIn": "p2_1stIn",
            "l_1stWon": "p2_1stWon", "l_2ndWon": "p2_2ndWon",
            "l_bpSaved": "p2_bpSaved", "l_bpFaced": "p2_bpFaced",
        })
        out = df[[
            "match_id", "date", "tourney_name", "tourney_level_name",
            "surface", "round", "round_num", "best_of",
            "winner_name", "winner_id", "winner_seed", "winner_rank",
            "winner_rank_points", "winner_hand", "winner_age", "winner_ht",
            "loser_name", "loser_id", "loser_seed", "loser_rank",
            "loser_rank_points", "loser_hand", "loser_age", "loser_ht",
            "p1_ace", "p1_df", "p1_svpt", "p1_1stIn", "p1_1stWon",
            "p1_2ndWon", "p1_bpSaved", "p1_bpFaced",
            "p2_ace", "p2_df", "p2_svpt", "p2_1stIn", "p2_1stWon",
            "p2_2ndWon", "p2_bpSaved", "p2_bpFaced",
            "score", "minutes",
        ]].copy()
        out = out.rename(columns={
            "winner_name": "player1_name", "winner_id": "player1_id",
            "winner_seed": "player1_seed", "winner_rank": "player1_rank",
            "winner_rank_points": "player1_rank_pts",
            "winner_hand": "player1_hand", "winner_age": "player1_age",
            "winner_ht": "player1_ht",
            "loser_name": "player2_name", "loser_id": "player2_id",
            "loser_seed": "player2_seed", "loser_rank": "player2_rank",
            "loser_rank_points": "player2_rank_pts",
            "loser_hand": "player2_hand", "loser_age": "player2_age",
            "loser_ht": "player2_ht",
        })
        out["tour"] = tour
        out["result"] = "player1_win"
        return out

    @staticmethod
    def _parse_cached(raw: pd.DataFrame, year: int, *, tour: str = "atp") -> pd.DataFrame:
        """Re-parse a previously cached CSV.

        Old caches may contain raw Sackmann data (winner_name column) rather
        than the project schema (player1_name). Detect and re-parse if so.
        """
        if "winner_name" in raw.columns:
            # Raw Sackmann format saved before the schema rename — re-parse.
            return TennisFetcher._parse(raw, year, tour=tour)
        df = raw.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        # Ensure round_num exists (old caches may lack it)
        if "round_num" not in df.columns and "round" in df.columns:
            df["round_num"] = df["round"].map(_ROUND_ORDER).fillna(3)
        return df

    @staticmethod
    def _parse_tdcuk(raw: pd.DataFrame, year: int, *, tour: str = "atp") -> pd.DataFrame:
        """
        Parse a tennis-data.co.uk xlsx into project schema.

        Serve stats (ace, df, svpt …) are absent — columns are set to NaN.
        Rolling serve features in downstream engineering carry forward from
        the last Sackmann-sourced game in the player's history.
        """
        df = raw.copy()

        # Drop walkovers — no actual match was played
        if "Comment" in df.columns:
            df = df[df["Comment"] != "Walkover"].copy()

        df["date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["date", "Winner", "Loser"])

        # Tournament level name
        level_col = "Series" if tour == "atp" else "Tier"
        level_map = _TDCUK_ATP_LEVEL_MAP if tour == "atp" else _TDCUK_WTA_LEVEL_MAP
        df["tourney_level_name"] = df[level_col].map(level_map).fillna("ATP250/500")

        # Round
        df["round"] = df["Round"].map(_TDCUK_ROUND_MAP).fillna("R64")
        df["round_num"] = df["round"].map(_ROUND_ORDER).fillna(2)

        # Synthetic match ID (tournament + date + players, slugified)
        def _slug(s):
            return re.sub(r"[^a-z0-9]", "_", str(s).lower())
        df["match_id"] = (
            "tdcuk_"
            + df["Tournament"].apply(_slug) + "_"
            + df["date"].dt.strftime("%Y%m%d") + "_"
            + df["Winner"].apply(_slug) + "_"
            + df["Loser"].apply(_slug)
        )

        out = pd.DataFrame({
            "match_id":          df["match_id"].values,
            "date":              df["date"].values,
            "tourney_name":      df["Tournament"].values,
            "tourney_level_name": df["tourney_level_name"].values,
            "surface":           df["Surface"].fillna("Hard").values,
            "round":             df["round"].values,
            "round_num":         df["round_num"].values,
            "best_of":           df["Best of"].values if "Best of" in df.columns else 3,
            "player1_name":      df["Winner"].values,
            "player1_id":        None,
            "player1_seed":      None,
            "player1_rank":      pd.to_numeric(df["WRank"], errors="coerce").values,
            "player1_rank_pts":  pd.to_numeric(df["WPts"], errors="coerce").values,
            "player1_hand":      None,
            "player1_age":       None,
            "player1_ht":        None,
            "player2_name":      df["Loser"].values,
            "player2_id":        None,
            "player2_seed":      None,
            "player2_rank":      pd.to_numeric(df["LRank"], errors="coerce").values,
            "player2_rank_pts":  pd.to_numeric(df["LPts"], errors="coerce").values,
            "player2_hand":      None,
            "player2_age":       None,
            "player2_ht":        None,
            # Serve stats absent — NaN; rolling features carry forward from prior Sackmann data
            "p1_ace": None, "p1_df": None, "p1_svpt": None, "p1_1stIn": None,
            "p1_1stWon": None, "p1_2ndWon": None, "p1_bpSaved": None, "p1_bpFaced": None,
            "p2_ace": None, "p2_df": None, "p2_svpt": None, "p2_1stIn": None,
            "p2_1stWon": None, "p2_2ndWon": None, "p2_bpSaved": None, "p2_bpFaced": None,
            "score":   None,
            "minutes": None,
            "tour":    tour,
            "result":  "player1_win",
        })
        return out
