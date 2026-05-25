"""
MLB-specific feature engineering.

Computes run-scoring form, real starting-pitcher stats (ERA, WHIP, K/9, BB/9),
home/away splits, rest days, streak metrics, and run differential trends.
Pitcher stats are per-season figures fetched from MLB Stats API.
"""

import logging
import math
from typing import Dict

import numpy as np
import pandas as pd

from src.features.base_engineer import BaseFeatureEngineer
from src.features.travel_features import add_travel_features
from src.utils.helpers import safe_divide

logger = logging.getLogger(__name__)

_ROLL_WINDOWS = [5, 10, 20]   # short / medium / long form
_MIN_GAMES    = 5              # minimum games per team before we trust features


class MLBFeatureEngineer(BaseFeatureEngineer):
    """
    Feature engineer for MLB game prediction.

    Features computed
    -----------------
    Offensive:
        home_rpg_N, away_rpg_N       — runs per game over last N
        home_hits_pg_N, away_hits_pg_N
        home_errors_pg_N, away_errors_pg_N
    Defensive / pitching proxy:
        home_ra_pg_N, away_ra_pg_N   — runs allowed per game
    Differentials:
        home_run_diff_N              — runs scored minus allowed
        away_run_diff_N
    Starting pitcher (season stats to date):
        home_sp_era, away_sp_era     — starter ERA
        home_sp_whip, away_sp_whip   — starter WHIP
        home_sp_k9, away_sp_k9       — K/9 innings
        home_sp_bb9, away_sp_bb9     — BB/9 innings
        home_sp_h9, away_sp_h9       — H/9 innings
        home_sp_gs, away_sp_gs       — games started (experience proxy)
        home_sp_ip, away_sp_ip       — innings pitched (workload)
        sp_era_diff                  — home_sp_era - away_sp_era (negative = home advantage)
        sp_whip_diff                 — home_sp_whip - away_sp_whip
        sp_k9_diff                   — home_sp_k9 - away_sp_k9
    Form:
        home_win_pct_N, away_win_pct_N
        home_streak, away_streak     — signed current streak (+3 = 3 wins)
    Home/away splits:
        home_home_wpct_N             — home team's win% when playing at home
        away_away_wpct_N             — away team's win% when playing away
    Rest:
        home_rest_days, away_rest_days
        home_b2b, away_b2b           — back-to-back flag (rest == 1)
    Season position:
        home_season_games_played, away_season_games_played
    Target:
        target — 0=away_win, 1=home_win  (binary; MLB has no draws)
    """

    _LABEL_MAP = {"away_win": 0, "draw": 0, "home_win": 1}  # binary; draws are extremely rare

    def __init__(self) -> None:
        super().__init__(sport="mlb")
        # Plain instance attribute so it can be reassigned by _load_features_cached
        self.label_map: Dict[int, str] = {v: k for k, v in self._LABEL_MAP.items()}

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            logger.warning("Empty DataFrame in MLBFeatureEngineer")
            return df

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Encode target
        df["target"] = df["result"].map(self._LABEL_MAP)

        df = self._add_rolling_features(df)
        df = self._add_elo_features(df)
        df = self._add_rest_features(df)
        df = self._add_schedule_density(df)
        df = self._add_streak(df)
        df = self._add_home_away_splits(df)
        df = self._add_pitcher_features(df)
        df = add_travel_features(df, sport="mlb")

        logger.info("MLB features: %d games, %d columns", len(df), df.shape[1])
        return df

    # ------------------------------------------------------------------
    # Rolling team stats (computed for each game row from prior games)
    # ------------------------------------------------------------------

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """For each game, compute rolling stats using only past games."""
        teams = pd.concat([
            df[["date", "home_team", "home_score", "away_score", "result",
                "home_hits", "home_errors"]].rename(columns={
                    "home_team": "team", "home_score": "scored",
                    "away_score": "conceded", "home_hits": "hits",
                    "home_errors": "errors",
                }).assign(is_home=1),
            df[["date", "away_team", "away_score", "home_score", "result",
                "away_hits", "away_errors"]].rename(columns={
                    "away_team": "team", "away_score": "scored",
                    "home_score": "conceded", "away_hits": "hits",
                    "away_errors": "errors",
                }).assign(is_home=0),
        ]).sort_values("date")

        teams["won"] = teams.apply(
            lambda r: 1 if (r["is_home"] == 1 and r["result"] == "home_win")
                        or (r["is_home"] == 0 and r["result"] == "away_win")
                      else 0, axis=1
        )

        # Build per-team rolling cache
        cache: Dict[str, pd.DataFrame] = {}
        for team, grp in teams.groupby("team"):
            grp = grp.sort_values("date").reset_index(drop=True)
            cache[team] = grp

        def _rolling_stats(team: str, before_date, window: int) -> dict:
            if team not in cache:
                return {}
            t = cache[team]
            past = t[t["date"] < before_date].tail(window)
            if len(past) < _MIN_GAMES:
                return {}
            return {
                "rpg":    past["scored"].mean(),
                "rapg":   past["conceded"].mean(),
                "hits_pg": past["hits"].mean(),
                "err_pg":  past["errors"].mean(),
                "win_pct": past["won"].mean(),
                "run_diff": (past["scored"] - past["conceded"]).mean(),
            }

        for w in _ROLL_WINDOWS:
            home_stats = df.apply(
                lambda r: _rolling_stats(r["home_team"], r["date"], w), axis=1)
            away_stats = df.apply(
                lambda r: _rolling_stats(r["away_team"], r["date"], w), axis=1)

            for stat in ("rpg", "rapg", "hits_pg", "err_pg", "win_pct", "run_diff"):
                df[f"home_{stat}_{w}"] = home_stats.apply(lambda d: d.get(stat, np.nan))
                df[f"away_{stat}_{w}"] = away_stats.apply(lambda d: d.get(stat, np.nan))

        return df

    # ------------------------------------------------------------------
    # Elo ratings
    # ------------------------------------------------------------------

    def _add_elo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chronological Elo for each MLB team.

        Run differential drives the MOV multiplier (capped at 2.5).
        Home field advantage is ~40 Elo points (MLB empirical value).
        """
        elo_initial: float = 1500.0
        elo_k: float = 15.0
        home_adv: float = 40.0

        elo_ratings: Dict[str, float] = {}
        home_elos, away_elos = [], []

        for _, row in df.iterrows():
            home, away = row["home_team"], row["away_team"]
            h_elo = elo_ratings.get(home, elo_initial)
            a_elo = elo_ratings.get(away, elo_initial)

            home_elos.append(h_elo)
            away_elos.append(a_elo)

            exp_home = 1.0 / (1.0 + 10.0 ** ((a_elo - h_elo - home_adv) / 400.0))
            margin = float(row.get("home_score", 0) or 0) - float(row.get("away_score", 0) or 0)
            actual_home = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)

            # MOV multiplier: run differential scaled, capped at 2.5
            k_mult = min(2.5, 1.0 + math.log(1.0 + abs(margin) * 0.5) * 0.4) if margin != 0 else 1.0
            k = elo_k * k_mult

            elo_ratings[home] = h_elo + k * (actual_home - exp_home)
            elo_ratings[away] = a_elo + k * ((1.0 - actual_home) - (1.0 - exp_home))

        df["home_elo"] = home_elos
        df["away_elo"] = away_elos
        df["elo_diff"] = df["home_elo"] - df["away_elo"]
        df["elo_win_prob"] = 1.0 / (
            1.0 + 10.0 ** ((df["away_elo"] - df["home_elo"] - home_adv) / 400.0)
        )
        logger.info("MLB Elo features computed for %d games", len(df))
        return df

    # ------------------------------------------------------------------
    # Schedule density
    # ------------------------------------------------------------------

    def _add_schedule_density(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Games played in the last N days for each team (home + away combined).

        MLB plays 162 games — schedule density matters for bullpen fatigue
        and starting-pitcher sequencing.
        """
        windows = [3, 5, 7, 10]

        team_dates: Dict[str, np.ndarray] = {}
        for team in set(df["home_team"]).union(df["away_team"]):
            h = df.loc[df["home_team"] == team, "date"].values
            a = df.loc[df["away_team"] == team, "date"].values
            team_dates[team] = np.sort(np.concatenate([h, a]))

        for side in ("home", "away"):
            tcol = f"{side}_team"
            for w in windows:
                col = f"{side}_games_L{w}D"
                delta = np.timedelta64(w, "D")
                counts = []
                for _, row in df.iterrows():
                    team = row[tcol]
                    d = row["date"]
                    arr = team_dates.get(team, np.array([], dtype="datetime64[ns]"))
                    if len(arr) == 0:
                        counts.append(0)
                        continue
                    lo = np.searchsorted(arr, d - delta, side="left")
                    hi = np.searchsorted(arr, d, side="left")
                    counts.append(int(hi - lo))
                df[col] = counts

        df["home_3in4"] = (df["home_games_L3D"] >= 2).astype(int)
        df["away_3in4"] = (df["away_games_L3D"] >= 2).astype(int)
        df["home_density_7"] = df["home_games_L7D"] / 7.0
        df["away_density_7"] = df["away_games_L7D"] / 7.0
        df["density_diff"] = df["home_density_7"] - df["away_density_7"]
        return df

    # ------------------------------------------------------------------
    # Rest / fatigue
    # ------------------------------------------------------------------

    def _add_rest_features(self, df: pd.DataFrame) -> pd.DataFrame:
        last_game: Dict[str, pd.Timestamp] = {}

        def _rest(team: str, current_date) -> int:
            if team not in last_game:
                return 7  # unknown → assume well-rested
            delta = (current_date - last_game[team]).days
            return min(delta, 10)

        home_rest, away_rest = [], []
        for _, row in df.iterrows():
            home_rest.append(_rest(row["home_team"], row["date"]))
            away_rest.append(_rest(row["away_team"], row["date"]))
            last_game[row["home_team"]] = row["date"]
            last_game[row["away_team"]] = row["date"]

        df["home_rest_days"] = home_rest
        df["away_rest_days"] = away_rest
        df["home_b2b"] = (df["home_rest_days"] == 1).astype(int)
        df["away_b2b"] = (df["away_rest_days"] == 1).astype(int)
        return df

    # ------------------------------------------------------------------
    # Streak
    # ------------------------------------------------------------------

    def _add_streak(self, df: pd.DataFrame) -> pd.DataFrame:
        streaks: Dict[str, int] = {}

        def _streak(team: str) -> int:
            return streaks.get(team, 0)

        def _update(team: str, won: bool):
            s = streaks.get(team, 0)
            if won:
                streaks[team] = max(s + 1, 1)
            else:
                streaks[team] = min(s - 1, -1)

        home_s, away_s = [], []
        for _, row in df.iterrows():
            home_s.append(_streak(row["home_team"]))
            away_s.append(_streak(row["away_team"]))
            _update(row["home_team"], row["result"] == "home_win")
            _update(row["away_team"], row["result"] == "away_win")

        df["home_streak"] = home_s
        df["away_streak"] = away_s
        return df

    # ------------------------------------------------------------------
    # Home/away context splits
    # ------------------------------------------------------------------

    def _add_home_away_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Win% for home team when playing at home,
        win% for away team when playing away.
        Uses rolling 20-game window of same-context games.
        """
        # home context: games where team was home
        home_records: Dict[str, list] = {}
        away_records: Dict[str, list] = {}

        h_home_wpct, a_away_wpct, sgp_home, sgp_away = [], [], [], []

        for _, row in df.iterrows():
            ht, at = row["home_team"], row["away_team"]
            dt = row["date"]

            # home team's home record
            hr = home_records.get(ht, [])
            h_home_wpct.append(np.mean(hr[-20:]) if len(hr) >= 3 else 0.5)

            # away team's away record
            ar = away_records.get(at, [])
            a_away_wpct.append(np.mean(ar[-20:]) if len(ar) >= 3 else 0.5)

            # season games played proxy
            sgp_home.append(len(hr))
            sgp_away.append(len(ar))

            # Update after recording current row's result
            home_records.setdefault(ht, []).append(1 if row["result"] == "home_win" else 0)
            away_records.setdefault(at, []).append(1 if row["result"] == "away_win" else 0)

        df["home_home_wpct_20"] = h_home_wpct
        df["away_away_wpct_20"] = a_away_wpct
        df["home_season_games_played"] = sgp_home
        df["away_season_games_played"] = sgp_away
        return df

    # ------------------------------------------------------------------
    # Starting pitcher features
    # ------------------------------------------------------------------

    # League-average defaults (used when pitcher data is missing)
    _SP_DEFAULTS = {
        "home_sp_era": 4.50, "away_sp_era": 4.50,
        "home_sp_whip": 1.30, "away_sp_whip": 1.30,
        "home_sp_k9":  8.0,  "away_sp_k9":  8.0,
        "home_sp_bb9": 3.0,  "away_sp_bb9": 3.0,
        "home_sp_h9":  9.0,  "away_sp_h9":  9.0,
        "home_sp_gs":  0,    "away_sp_gs":  0,
        "home_sp_ip":  0.0,  "away_sp_ip":  0.0,
    }

    def _add_pitcher_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add starting pitcher season stats as features.
        Columns home_sp_era, away_sp_era, etc. come directly from the
        fetcher (populated by MLBFetcher._parse_game).  If the raw data
        doesn't have them (e.g. older cached parquet), fill with league
        average defaults so training still works.
        """
        sp_cols = list(self._SP_DEFAULTS.keys())

        for col, default in self._SP_DEFAULTS.items():
            if col not in df.columns:
                df[col] = default
            else:
                # Fill any missing values with league average
                df[col] = df[col].fillna(default)

        # Winsorize ERA and WHIP to sane ranges (cap outliers)
        for side in ("home", "away"):
            df[f"{side}_sp_era"]  = df[f"{side}_sp_era"].clip(0.0, 9.0)
            df[f"{side}_sp_whip"] = df[f"{side}_sp_whip"].clip(0.5, 2.5)
            df[f"{side}_sp_k9"]   = df[f"{side}_sp_k9"].clip(2.0, 16.0)
            df[f"{side}_sp_bb9"]  = df[f"{side}_sp_bb9"].clip(0.5, 8.0)
            df[f"{side}_sp_h9"]   = df[f"{side}_sp_h9"].clip(4.0, 14.0)

        # Differentials — negative means home pitcher is better
        df["sp_era_diff"]  = df["home_sp_era"]  - df["away_sp_era"]
        df["sp_whip_diff"] = df["home_sp_whip"] - df["away_sp_whip"]
        df["sp_k9_diff"]   = df["home_sp_k9"]   - df["away_sp_k9"]
        df["sp_bb9_diff"]  = df["home_sp_bb9"]  - df["away_sp_bb9"]

        # Experience flag: fewer than 3 GS = essentially unknown (rookie/opener)
        df["home_sp_unknown"] = (df["home_sp_gs"] < 3).astype(int)
        df["away_sp_unknown"] = (df["away_sp_gs"] < 3).astype(int)

        logger.info(
            "MLB pitcher features added: %.1f%% of games have home starter data, "
            "%.1f%% have away starter data",
            (df["home_sp_gs"] > 0).mean() * 100,
            (df["away_sp_gs"] > 0).mean() * 100,
        )
        return df
