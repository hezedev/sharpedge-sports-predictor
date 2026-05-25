"""
Tennis Feature Engineering
==========================
Builds predictive features for ATP match outcomes from the Sackmann dataset.

Key design decisions:
- Sackmann data always lists the winner as player1. To avoid trivial leakage
  we RANDOMLY SWAP player1/player2 for ~50% of rows before computing features,
  so the model sees both winning and losing perspectives.
- All rolling statistics use shift(1) — no lookahead.
- Surface-specific features capture the clay/grass/hard split.
- Ranking log-ratio is the single strongest baseline feature in tennis.

Features produced:
    rank_log_ratio          log(p2_rank / p1_rank) — positive = p1 is higher ranked
    rank_pts_log_ratio      same for ranking points
    p1/p2_surface_win_rate  rolling win rate on this specific surface
    p1/p2_overall_win_rate  rolling overall win rate
    p1/p2_recent_form       win rate over last 10 matches
    p1/p2_form_quality      recent form weighted by opponent quality
    p1/p2_serve_rating      rolling 1st-serve % + ace rate
    p1/p2_break_save_rate   rolling bp save rate (clutch indicator)
    p1/p2_return_pressure   rolling break-pressure created on return
    p1/p2_break_conv        rolling break-point conversion on return
    p1/p2_match_load        matches played in last 30 days (fatigue)
    h2h_p1_win_rate         head-to-head win rate for p1 vs p2
    h2h_matches             number of prior h2h meetings
    round_num               ordinal round (1=R128, 7=Final)
    best_of                 3 or 5 sets
    surface_*               one-hot: Hard / Clay / Grass
    age_diff                p1_age - p2_age
    height_diff             p1_ht - p2_ht (cm)
    seed_advantage          p1 seeded and p2 not (0/1/-1)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_FORM_WINDOW = 10
_SURFACE_WINDOW = 20
_LOAD_DAYS = 30


class TennisFeatureEngineer:
    """Feature engineer for ATP tennis match prediction."""

    def __init__(self) -> None:
        self.label_map: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full feature engineering pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Output of TennisFetcher.fetch_all_seasons().
            player1 is always winner in raw data.

        Returns
        -------
        pd.DataFrame
            Feature-enriched, balanced DataFrame with target column.
        """
        if df.empty:
            return df

        df = df.sort_values("date").reset_index(drop=True)
        logger.info("Engineering tennis features for %d matches", len(df))

        # Step 1: Balance the dataset by randomly swapping perspectives
        df = self._balance_perspectives(df)

        # Step 2: Ranking features (instant, no rolling needed)
        df = self._ranking_features(df)

        # Step 3: Surface win rates (rolling, per-player, per-surface)
        df = self._surface_win_rates(df)

        # Step 4: Overall form (last N matches)
        df = self._form_features(df)

        # Step 5: Serve quality and return pressure
        df = self._serve_features(df)

        # Step 6: H2H record
        df = self._h2h_features(df)

        # Step 7: Physical / tournament context
        df = self._context_features(df)

        # Step 8: Match load / fatigue
        df = self._load_features(df)

        # Step 9: Encode target
        df["target"] = (df["result"] == "player1_win").astype(int)
        self.label_map = {"player1_win": 1, "player2_win": 0}

        logger.info("Tennis features complete: %d rows × %d cols", *df.shape)
        return df

    # ------------------------------------------------------------------
    # Balance
    # ------------------------------------------------------------------

    def _balance_perspectives(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Randomly flip 50% of rows so player1 is sometimes the loser.

        In Sackmann data player1 = winner always, which would cause
        trivial target leakage through features computed on player1's stats.
        After flipping, the model learns from both perspectives.
        """
        rng = np.random.default_rng(42)
        flip_mask = rng.random(len(df)) < 0.5

        p1_cols = [c for c in df.columns if c.startswith("player1_") or c.startswith("p1_")]
        p2_cols = [c for c in df.columns if c.startswith("player2_") or c.startswith("p2_")]

        flipped = df.copy()
        for c1, c2 in zip(p1_cols, p2_cols):
            flipped.loc[flip_mask, c1] = df.loc[flip_mask, c2].values
            flipped.loc[flip_mask, c2] = df.loc[flip_mask, c1].values

        # Flip result for swapped rows
        flipped.loc[flip_mask, "result"] = flipped.loc[flip_mask, "result"].map(
            {"player1_win": "player2_win", "player2_win": "player1_win"}
        )

        return flipped.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def _ranking_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Log-ratio of rankings and ranking points."""
        r1 = pd.to_numeric(df["player1_rank"], errors="coerce").fillna(300).clip(1)
        r2 = pd.to_numeric(df["player2_rank"], errors="coerce").fillna(300).clip(1)
        df["rank_log_ratio"] = np.log(r2 / r1)           # positive = p1 higher ranked
        df["rank_diff"] = r2 - r1

        pts1 = pd.to_numeric(df["player1_rank_pts"], errors="coerce").fillna(1).clip(1)
        pts2 = pd.to_numeric(df["player2_rank_pts"], errors="coerce").fillna(1).clip(1)
        df["rank_pts_log_ratio"] = np.log(pts1 / pts2)

        # Seed advantage
        s1 = df["player1_seed"].notna().astype(float)
        s2 = df["player2_seed"].notna().astype(float)
        df["seed_advantage"] = s1 - s2

        return df

    # ------------------------------------------------------------------
    # Surface win rates
    # ------------------------------------------------------------------

    def _surface_win_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling win rate per player per surface."""
        df["p1_surface_win"] = np.nan
        df["p2_surface_win"] = np.nan

        for surface in ["Hard", "Clay", "Grass"]:
            surf_mask = df["surface"] == surface

            for player_col, win_value, out_col in [
                ("player1_name", "player1_win", "p1_surface_win"),
                ("player2_name", "player2_win", "p2_surface_win"),
            ]:
                rates = self._rolling_win_rate(
                    df[surf_mask], player_col, "result", win_value, _SURFACE_WINDOW
                )
                df.loc[surf_mask, out_col] = rates.values

        df["p1_surface_win"] = df["p1_surface_win"].fillna(0.5)
        df["p2_surface_win"] = df["p2_surface_win"].fillna(0.5)
        df["surface_win_diff"] = df["p1_surface_win"] - df["p2_surface_win"]

        return df

    # ------------------------------------------------------------------
    # Form
    # ------------------------------------------------------------------

    def _form_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling overall win rate plus opponent-quality-adjusted form."""
        df["p1_form"] = self._rolling_win_rate(
            df, "player1_name", "result", "player1_win", _FORM_WINDOW
        )
        df["p2_form"] = self._rolling_win_rate(
            df, "player2_name", "result", "player2_win", _FORM_WINDOW
        )
        df["p1_form"] = df["p1_form"].fillna(0.5)
        df["p2_form"] = df["p2_form"].fillna(0.5)
        df["form_diff"] = df["p1_form"] - df["p2_form"]

        max_rank = 400.0
        p1_opp_rank = pd.to_numeric(df["player2_rank"], errors="coerce").fillna(max_rank).clip(1, max_rank)
        p2_opp_rank = pd.to_numeric(df["player1_rank"], errors="coerce").fillna(max_rank).clip(1, max_rank)
        p1_opp_quality = 1.0 - (np.log1p(p1_opp_rank) / np.log1p(max_rank))
        p2_opp_quality = 1.0 - (np.log1p(p2_opp_rank) / np.log1p(max_rank))
        p1_result_sign = np.where(df["result"] == "player1_win", 1.0, -1.0)
        p2_result_sign = np.where(df["result"] == "player2_win", 1.0, -1.0)
        df["_p1_quality_result"] = p1_result_sign * p1_opp_quality
        df["_p2_quality_result"] = p2_result_sign * p2_opp_quality
        df["p1_form_quality"] = self._rolling_stat(df, "player1_name", "_p1_quality_result", _FORM_WINDOW).fillna(0.0)
        df["p2_form_quality"] = self._rolling_stat(df, "player2_name", "_p2_quality_result", _FORM_WINDOW).fillna(0.0)
        df["form_quality_diff"] = df["p1_form_quality"] - df["p2_form_quality"]
        df = df.drop(columns=["_p1_quality_result", "_p2_quality_result"])

        return df

    # ------------------------------------------------------------------
    # Serve quality
    # ------------------------------------------------------------------

    def _serve_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling serve stats plus return-pressure and break-conversion proxies."""
        for prefix in ("p1", "p2"):
            svpt = pd.to_numeric(df[f"{prefix}_svpt"], errors="coerce").fillna(0)
            in1 = pd.to_numeric(df[f"{prefix}_1stIn"], errors="coerce").fillna(0)
            ace = pd.to_numeric(df[f"{prefix}_ace"], errors="coerce").fillna(0)
            bp_saved = pd.to_numeric(df[f"{prefix}_bpSaved"], errors="coerce").fillna(0)
            bp_faced = pd.to_numeric(df[f"{prefix}_bpFaced"], errors="coerce").fillna(1).clip(1)
            player_col = "player1_name" if prefix == "p1" else "player2_name"

            safe_svpt = svpt.replace(0, np.nan)
            df[f"{prefix}_1st_pct"] = in1 / safe_svpt
            df[f"{prefix}_ace_rate"] = ace / safe_svpt
            df[f"{prefix}_bp_save"] = bp_saved / bp_faced
            opp_prefix = "p2" if prefix == "p1" else "p1"
            opp_svpt = pd.to_numeric(df[f"{opp_prefix}_svpt"], errors="coerce").fillna(0).replace(0, np.nan)
            opp_bp_faced = pd.to_numeric(df[f"{opp_prefix}_bpFaced"], errors="coerce").fillna(0)
            opp_bp_saved = pd.to_numeric(df[f"{opp_prefix}_bpSaved"], errors="coerce").fillna(0)
            df[f"{prefix}_return_pressure"] = (opp_bp_faced / opp_svpt).fillna(0.0)
            safe_opp_bp_faced = opp_bp_faced.replace(0, np.nan)
            df[f"{prefix}_break_conv"] = ((opp_bp_faced - opp_bp_saved) / safe_opp_bp_faced).fillna(0.0)

            for stat_col in [
                f"{prefix}_1st_pct",
                f"{prefix}_ace_rate",
                f"{prefix}_bp_save",
                f"{prefix}_return_pressure",
                f"{prefix}_break_conv",
            ]:
                rolled = self._rolling_stat(df, player_col, stat_col, _FORM_WINDOW)
                df[f"roll_{stat_col}"] = rolled.fillna(df[stat_col].median())

        df["serve_diff"] = df["roll_p1_ace_rate"].fillna(0) - df["roll_p2_ace_rate"].fillna(0)
        df["bp_save_diff"] = df["roll_p1_bp_save"].fillna(0) - df["roll_p2_bp_save"].fillna(0)
        df["return_pressure_diff"] = df["roll_p1_return_pressure"].fillna(0) - df["roll_p2_return_pressure"].fillna(0)
        df["break_conv_diff"] = df["roll_p1_break_conv"].fillna(0) - df["roll_p2_break_conv"].fillna(0)
        df["serve_balance_diff"] = (
            df["serve_diff"].fillna(0)
            + (0.6 * df["bp_save_diff"].fillna(0))
            - (0.35 * df["return_pressure_diff"].fillna(0))
        )

        return df

    # ------------------------------------------------------------------
    # Head-to-head
    # ------------------------------------------------------------------

    def _h2h_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prior head-to-head win rate for player1 vs player2."""
        h2h_wins = pd.Series(0.0, index=df.index)
        h2h_total = pd.Series(0.0, index=df.index)

        history: Dict[Tuple[str, str], List[int]] = {}

        for idx, row in df.iterrows():
            p1, p2 = row["player1_name"], row["player2_name"]
            key_fw = (p1, p2)
            key_rv = (p2, p1)

            fw = history.get(key_fw, [])
            rv = history.get(key_rv, [])

            prior_wins = sum(fw) + sum(1 - r for r in rv)
            prior_total = len(fw) + len(rv)

            h2h_wins.iloc[idx] = prior_wins
            h2h_total.iloc[idx] = prior_total

            won = 1 if row["result"] == "player1_win" else 0
            history.setdefault(key_fw, []).append(won)

        df["h2h_p1_wins"] = h2h_wins
        df["h2h_total"] = h2h_total
        df["h2h_p1_win_rate"] = np.where(
            h2h_total > 0, h2h_wins / h2h_total, 0.5
        )
        return df

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------

    def _context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tournament context, physical attributes, surface one-hot."""
        for surf in ["Hard", "Clay", "Grass"]:
            df[f"surface_{surf.lower()}"] = (df["surface"] == surf).astype(float)

        df["round_num"] = df["round_num"].fillna(3)
        df["best_of"] = df["best_of"].fillna(3)
        p1_age = pd.to_numeric(df["player1_age"], errors="coerce").fillna(26)
        p2_age = pd.to_numeric(df["player2_age"], errors="coerce").fillna(26)
        p1_ht = pd.to_numeric(df["player1_ht"], errors="coerce").fillna(185)
        p2_ht = pd.to_numeric(df["player2_ht"], errors="coerce").fillna(185)
        df["age_diff"] = p1_age - p2_age
        df["height_diff"] = p1_ht - p2_ht

        level_ord = {"Grand Slam": 4, "ATP Finals": 4, "Masters": 3,
                     "ATP250/500": 2, "Challenger": 1, "Davis Cup": 2}
        df["tourney_level_num"] = df["tourney_level_name"].map(level_ord).fillna(2)

        return df

    # ------------------------------------------------------------------
    # Match load
    # ------------------------------------------------------------------

    def _load_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Matches played in last 30 days (fatigue proxy)."""
        p1_load = pd.Series(0.0, index=df.index)
        p2_load = pd.Series(0.0, index=df.index)

        p1_history: Dict[str, list] = {}
        p2_history: Dict[str, list] = {}

        for idx, row in df.iterrows():
            cutoff = row["date"] - pd.Timedelta(days=_LOAD_DAYS)
            d = row["date"]

            for player, hist_dict, load_series in [
                (row["player1_name"], p1_history, p1_load),
                (row["player2_name"], p2_history, p2_load),
            ]:
                past = hist_dict.get(player, [])
                recent = sum(1 for t in past if t >= cutoff)
                load_series.iloc[idx] = recent
                hist_dict.setdefault(player, []).append(d)

        df["p1_load"] = p1_load
        df["p2_load"] = p2_load
        df["load_diff"] = p1_load - p2_load

        return df

    # ------------------------------------------------------------------
    # Rolling helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rolling_win_rate(
        df: pd.DataFrame,
        player_col: str,
        result_col: str,
        win_value: str,
        window: int,
    ) -> pd.Series:
        rates = pd.Series(np.nan, index=df.index)
        for player in df[player_col].unique():
            mask = df[player_col] == player
            wins = (df.loc[mask, result_col] == win_value).astype(float)
            rolled = wins.rolling(window, min_periods=3).mean().shift(1)
            rates.loc[mask] = rolled.values
        return rates

    @staticmethod
    def _rolling_stat(
        df: pd.DataFrame,
        player_col: str,
        stat_col: str,
        window: int,
    ) -> pd.Series:
        result = pd.Series(np.nan, index=df.index)
        for player in df[player_col].unique():
            mask = df[player_col] == player
            rolled = df.loc[mask, stat_col].rolling(window, min_periods=3).mean().shift(1)
            result.loc[mask] = rolled.values
        return result
