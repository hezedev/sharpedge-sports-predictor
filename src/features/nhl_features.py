"""
NHL-specific feature engineering.

Computes goals-per-game form, goals-allowed proxy (defence),
home/away splits, rest days, overtime propensity, and streak metrics.
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

_ROLL_WINDOWS = [5, 10, 20]
_MIN_GAMES = 5


class NHLFeatureEngineer(BaseFeatureEngineer):
    """
    Feature engineer for NHL game prediction.

    Features computed
    -----------------
    Scoring form:
        home_gpg_N, away_gpg_N       — goals per game over last N
        home_gapg_N, away_gapg_N     — goals allowed per game
        home_goal_diff_N, away_goal_diff_N
    Win/loss:
        home_win_pct_N, away_win_pct_N
        home_ot_pct_N, away_ot_pct_N  — fraction of games going to OT
    Shot-based (from play-by-play):
        home_sog_pg_N, away_sog_pg_N  — shots on goal per game (rolling N)
        home_cf_pct_N, away_cf_pct_N  — Corsi For% (ES shot attempts)
        home_ff_pct_N, away_ff_pct_N  — Fenwick For% (ES unblocked shots)
        home_xgf_pg_N, away_xgf_pg_N  — expected goals for per game
        home_xga_pg_N, away_xga_pg_N  — expected goals against per game
        home_xg_diff_N                — xGF - xGA per game
        home_pp_pct_N, away_pp_pct_N  — power play % (pp_goals/pp_opp)
        home_pk_pct_N, away_pk_pct_N  — penalty kill % (1 - opp pp%)
    Streak:
        home_streak, away_streak
    Context splits:
        home_home_wpct_20, away_away_wpct_20
    Rest:
        home_rest_days, away_rest_days
        home_b2b, away_b2b
    Season:
        home_season_games_played, away_season_games_played
    Target:
        target — 0=away_win, 1=home_win  (binary; OT/SO result still counted)
    """

    # NHL has no draws — games always end in a winner (via OT/SO if needed).
    # Use binary 0/1 so XGBoost and calibrators work correctly.
    _LABEL_MAP = {"away_win": 0, "draw": 0, "home_win": 1}

    def __init__(self) -> None:
        super().__init__(sport="nhl")
        # Plain instance attribute so it can be reassigned by _load_features_cached
        self.label_map: Dict[int, str] = {v: k for k, v in self._LABEL_MAP.items()}

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            logger.warning("Empty DataFrame in NHLFeatureEngineer")
            return df

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        df["target"] = df["result"].map(self._LABEL_MAP)

        df = self._add_rolling_features(df)
        df = self._add_shot_features(df)
        df = self._add_elo_features(df)
        df = self._add_rest_features(df)
        df = self._add_schedule_density(df)
        df = self._add_streak(df)
        df = self._add_home_away_splits(df)
        df = add_travel_features(df, sport="nhl")

        logger.info("NHL features: %d games, %d columns", len(df), df.shape[1])
        return df

    # ------------------------------------------------------------------
    # Rolling team stats
    # ------------------------------------------------------------------

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        teams = pd.concat([
            df[["date", "home_team", "home_score", "away_score", "result", "went_to_ot"]].rename(
                columns={"home_team": "team", "home_score": "scored", "away_score": "conceded"}
            ).assign(is_home=1),
            df[["date", "away_team", "away_score", "home_score", "result", "went_to_ot"]].rename(
                columns={"away_team": "team", "away_score": "scored", "home_score": "conceded"}
            ).assign(is_home=0),
        ]).sort_values("date")

        teams["won"] = teams.apply(
            lambda r: 1 if (r["is_home"] == 1 and r["result"] == "home_win")
                        or (r["is_home"] == 0 and r["result"] == "away_win")
                      else 0, axis=1
        )

        cache: Dict[str, pd.DataFrame] = {}
        for team, grp in teams.groupby("team"):
            cache[team] = grp.sort_values("date").reset_index(drop=True)

        def _rolling_stats(team: str, before_date, window: int) -> dict:
            if team not in cache:
                return {}
            t = cache[team]
            past = t[t["date"] < before_date].tail(window)
            if len(past) < _MIN_GAMES:
                return {}
            return {
                "gpg":       past["scored"].mean(),
                "gapg":      past["conceded"].mean(),
                "win_pct":   past["won"].mean(),
                "goal_diff": (past["scored"] - past["conceded"]).mean(),
                "ot_pct":    past["went_to_ot"].mean(),
            }

        for w in _ROLL_WINDOWS:
            home_stats = df.apply(lambda r: _rolling_stats(r["home_team"], r["date"], w), axis=1)
            away_stats = df.apply(lambda r: _rolling_stats(r["away_team"], r["date"], w), axis=1)

            for stat in ("gpg", "gapg", "win_pct", "goal_diff", "ot_pct"):
                df[f"home_{stat}_{w}"] = home_stats.apply(lambda d: d.get(stat, np.nan))
                df[f"away_{stat}_{w}"] = away_stats.apply(lambda d: d.get(stat, np.nan))

        return df

    # ------------------------------------------------------------------
    # Shot-based features (Corsi, Fenwick, xG, PP%)
    # ------------------------------------------------------------------

    def _add_shot_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rolling shot-quality and special-teams metrics.
        Source columns come from NHLFetcher play-by-play enrichment:
            home_corsi, away_corsi, home_fenwick, away_fenwick,
            home_xg, away_xg, home_shots, away_shots,
            home_pp_goals, away_pp_goals, home_pp_opp, away_pp_opp
        If any are missing (old cache), fill with 0 so pipeline still runs.
        """
        shot_cols = [
            "home_corsi", "away_corsi", "home_fenwick", "away_fenwick",
            "home_xg", "away_xg", "home_shots", "away_shots",
            "home_pp_goals", "away_pp_goals", "home_pp_opp", "away_pp_opp",
        ]
        for col in shot_cols:
            if col not in df.columns:
                df[col] = 0

        # Build per-team records with shot data
        home_df = df[["date", "home_team", "home_corsi", "away_corsi",
                       "home_fenwick", "away_fenwick", "home_xg", "away_xg",
                       "home_shots", "away_shots",
                       "home_pp_goals", "home_pp_opp",
                       "away_pp_goals", "away_pp_opp"]].copy()
        home_df = home_df.rename(columns={
            "home_team": "team",
            "home_corsi": "cf", "away_corsi": "ca",
            "home_fenwick": "ff", "away_fenwick": "fa",
            "home_xg": "xgf", "away_xg": "xga",
            "home_shots": "sog_f", "away_shots": "sog_a",
            "home_pp_goals": "ppg", "home_pp_opp": "ppo",
            "away_pp_goals": "pkg_opp", "away_pp_opp": "pko_opp",
        })

        away_df = df[["date", "away_team", "away_corsi", "home_corsi",
                       "away_fenwick", "home_fenwick", "away_xg", "home_xg",
                       "away_shots", "home_shots",
                       "away_pp_goals", "away_pp_opp",
                       "home_pp_goals", "home_pp_opp"]].copy()
        away_df = away_df.rename(columns={
            "away_team": "team",
            "away_corsi": "cf", "home_corsi": "ca",
            "away_fenwick": "ff", "home_fenwick": "fa",
            "away_xg": "xgf", "home_xg": "xga",
            "away_shots": "sog_f", "home_shots": "sog_a",
            "away_pp_goals": "ppg", "away_pp_opp": "ppo",
            "home_pp_goals": "pkg_opp", "home_pp_opp": "pko_opp",
        })

        teams_shot = pd.concat([home_df, away_df]).sort_values("date")
        cache_shot: Dict[str, pd.DataFrame] = {}
        for team, grp in teams_shot.groupby("team"):
            cache_shot[team] = grp.sort_values("date").reset_index(drop=True)

        def _shot_stats(team: str, before_date, window: int) -> dict:
            if team not in cache_shot:
                return {}
            t = cache_shot[team]
            past = t[t["date"] < before_date].tail(window)
            if len(past) < _MIN_GAMES:
                return {}
            cf_total = past["cf"].sum() + past["ca"].sum()
            ff_total = past["ff"].sum() + past["fa"].sum()
            ppo = past["ppo"].sum()
            pko = past["pko_opp"].sum()
            return {
                "sog_pg":   past["sog_f"].mean(),
                "cf_pct":   past["cf"].sum() / cf_total if cf_total > 0 else 0.5,
                "ff_pct":   past["ff"].sum() / ff_total if ff_total > 0 else 0.5,
                "xgf_pg":   past["xgf"].mean(),
                "xga_pg":   past["xga"].mean(),
                "xg_diff":  (past["xgf"] - past["xga"]).mean(),
                "pp_pct":   past["ppg"].sum() / ppo if ppo > 0 else 0.20,
                "pk_pct":   1.0 - (past["pkg_opp"].sum() / pko if pko > 0 else 0.20),
            }

        for w in _ROLL_WINDOWS:
            home_s = df.apply(lambda r: _shot_stats(r["home_team"], r["date"], w), axis=1)
            away_s = df.apply(lambda r: _shot_stats(r["away_team"], r["date"], w), axis=1)
            for stat in ("sog_pg", "cf_pct", "ff_pct", "xgf_pg", "xga_pg", "xg_diff", "pp_pct", "pk_pct"):
                df[f"home_{stat}_{w}"] = home_s.apply(lambda d: d.get(stat, np.nan))
                df[f"away_{stat}_{w}"] = away_s.apply(lambda d: d.get(stat, np.nan))

        has_data = (df["home_corsi"] > 0).mean() * 100
        logger.info("NHL shot features added (%.0f%% of games have PBP data)", has_data)
        return df

    # ------------------------------------------------------------------
    # Elo ratings
    # ------------------------------------------------------------------

    def _add_elo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chronological Elo for each NHL team.

        K-factor is boosted for multi-goal wins (decisive victories matter).
        Home ice advantage is ~65 Elo points (NHL empirical value).
        OT/SO wins count as 0.6 (partial credit — the market was closer).
        """
        elo_initial: float = 1500.0
        elo_k: float = 16.0
        home_adv: float = 65.0

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
            went_ot = bool(row.get("went_to_ot", False))

            if margin > 0:
                actual_home = 0.6 if went_ot else 1.0
            elif margin < 0:
                actual_home = 0.4 if went_ot else 0.0
            else:
                actual_home = 0.5

            # MOV multiplier: capped at 2 for NHL (low-scoring, every goal counts)
            k_mult = min(2.0, 1.0 + math.log(1.0 + abs(margin)) * 0.3) if margin != 0 else 1.0
            k = elo_k * k_mult

            elo_ratings[home] = h_elo + k * (actual_home - exp_home)
            elo_ratings[away] = a_elo + k * ((1.0 - actual_home) - (1.0 - exp_home))

        df["home_elo"] = home_elos
        df["away_elo"] = away_elos
        df["elo_diff"] = df["home_elo"] - df["away_elo"]
        df["elo_win_prob"] = 1.0 / (
            1.0 + 10.0 ** ((df["away_elo"] - df["home_elo"] - home_adv) / 400.0)
        )
        logger.info("NHL Elo features computed for %d games", len(df))
        return df

    # ------------------------------------------------------------------
    # Schedule density
    # ------------------------------------------------------------------

    def _add_schedule_density(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Games played in the last N days for each team (home + away combined).

        NHL has the densest schedule in North American sports (82 games, Oct-Apr).
        3-in-4-nights and weekly load are meaningful fatigue signals.
        """
        windows = [3, 5, 7, 10]

        team_dates: Dict[str, np.ndarray] = {}
        for team in set(df["home_team"]).union(df["away_team"]):
            h = df.loc[df["home_team"] == team, "date"].values
            a = df.loc[df["away_team"] == team, "date"].values
            team_dates[team] = np.sort(np.concatenate([h, a]))

        new_cols: Dict[str, list] = {}
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
                new_cols[col] = counts

        extras = pd.DataFrame(new_cols, index=df.index)
        extras["home_3in4"] = (extras["home_games_L3D"] >= 2).astype(int)
        extras["away_3in4"] = (extras["away_games_L3D"] >= 2).astype(int)
        extras["home_density_7"] = extras["home_games_L7D"] / 7.0
        extras["away_density_7"] = extras["away_games_L7D"] / 7.0
        extras["density_diff"] = extras["home_density_7"] - extras["away_density_7"]
        return pd.concat([df, extras], axis=1)

    # ------------------------------------------------------------------
    # Rest / fatigue
    # ------------------------------------------------------------------

    def _add_rest_features(self, df: pd.DataFrame) -> pd.DataFrame:
        last_game: Dict[str, pd.Timestamp] = {}

        home_rest, away_rest = [], []
        for _, row in df.iterrows():
            def _rest(team: str) -> int:
                if team not in last_game:
                    return 7
                return min((row["date"] - last_game[team]).days, 10)

            home_rest.append(_rest(row["home_team"]))
            away_rest.append(_rest(row["away_team"]))
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

        home_s, away_s = [], []
        for _, row in df.iterrows():
            home_s.append(streaks.get(row["home_team"], 0))
            away_s.append(streaks.get(row["away_team"], 0))

            for team, won in [
                (row["home_team"], row["result"] == "home_win"),
                (row["away_team"], row["result"] == "away_win"),
            ]:
                s = streaks.get(team, 0)
                streaks[team] = max(s + 1, 1) if won else min(s - 1, -1)

        return pd.concat([df, pd.DataFrame({"home_streak": home_s, "away_streak": away_s}, index=df.index)], axis=1)

    # ------------------------------------------------------------------
    # Home/away context splits
    # ------------------------------------------------------------------

    def _add_home_away_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        home_records: Dict[str, list] = {}
        away_records: Dict[str, list] = {}

        h_home_wpct, a_away_wpct, sgp_home, sgp_away = [], [], [], []

        for _, row in df.iterrows():
            ht, at = row["home_team"], row["away_team"]

            hr = home_records.get(ht, [])
            h_home_wpct.append(np.mean(hr[-20:]) if len(hr) >= 3 else 0.5)
            ar = away_records.get(at, [])
            a_away_wpct.append(np.mean(ar[-20:]) if len(ar) >= 3 else 0.5)
            sgp_home.append(len(hr))
            sgp_away.append(len(ar))

            home_records.setdefault(ht, []).append(1 if row["result"] == "home_win" else 0)
            away_records.setdefault(at, []).append(1 if row["result"] == "away_win" else 0)

        return pd.concat([df, pd.DataFrame({
            "home_home_wpct_20": h_home_wpct,
            "away_away_wpct_20": a_away_wpct,
            "home_season_games_played": sgp_home,
            "away_season_games_played": sgp_away,
        }, index=df.index)], axis=1)
