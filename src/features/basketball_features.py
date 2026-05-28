"""
Basketball-specific feature engineering.

Computes pace-adjusted offensive/defensive ratings, form metrics,
scoring distribution, fatigue indicators, and home/away splits.

Advanced metrics added (Task #10):
    - SRS (Simple Rating System): avg point diff + SOS adjustment (3 iterations)
    - Multi-window rolling stats: win%, margin, PPG at 5/10/20 games
    - Home/away context splits: win% only in home/away games
    - Opponent-adjusted ORtg/DRtg using SRS-derived SOS
    - Scoring consistency features (std of margins)
"""

import logging
import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from config import settings
from src.features.base_engineer import BaseFeatureEngineer
from src.features.travel_features import add_travel_features
from src.utils.helpers import safe_divide

logger = logging.getLogger(__name__)

_ROLL_WINDOWS = [5, 10, 20]
_MIN_GAMES = 5


class BasketballFeatureEngineer(BaseFeatureEngineer):
    """
    Feature engineer for basketball game prediction.

    Features computed:
        - SRS-based strength ratings (3-iteration opponent-adjusted margin)
        - Rolling win%, margin, PPG at 5/10/20 game windows
        - Opponent-adjusted ORtg/DRtg proxies
        - Home/away context split win% (last 20 home/away games)
        - Pace proxy (total points per game)
        - Quarter-by-quarter momentum (Q4 strength)
        - Home/away win streaks
        - Rest days & back-to-back detection
        - Fatigue index (games in last 7 days)
        - Overtime propensity
    """

    def __init__(self) -> None:
        super().__init__(sport="basketball")
        self._pace_window = self._sport_cfg.get("pace_window", 10)

    @staticmethod
    def _add_columns(df: pd.DataFrame, columns: Dict[str, object]) -> pd.DataFrame:
        """Attach derived columns in one pass to avoid pandas fragmentation."""
        if not columns:
            return df
        existing = [col for col in columns if col in df.columns]
        base = df.drop(columns=existing) if existing else df
        return pd.concat([base, pd.DataFrame(columns, index=df.index)], axis=1)

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply full basketball feature engineering pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Raw basketball game data with columns: date, home_team,
            away_team, home_score, away_score, result, home_q1..q4, etc.

        Returns
        -------
        pd.DataFrame
            Feature-enriched DataFrame ready for training.
        """
        if df.empty:
            logger.warning("Empty DataFrame passed to basketball feature engineer")
            return df

        df = df.sort_values("date").reset_index(drop=True)
        logger.info("Engineering basketball features for %d games", len(df))

        # 1. Scoring features (PPG, opponent PPG)
        df = self._compute_scoring_features(df)

        # 2. Pace proxy
        df = self._compute_pace_features(df)

        # 3. Offensive / Defensive ratings (opponent-adjusted)
        df = self._compute_ratings(df)

        # 3b. SRS + Elo + multi-window rolling stats (advanced)
        df = self._compute_srs_features(df)
        df = self._compute_elo_features(df)
        df = self._compute_multiwindow_rolling(df)
        df = self._compute_home_away_splits(df)

        # 3c. Basketball-specific model hooks: possession-adjusted strength,
        # Four Factors, pace/totals, player availability, and matchup context.
        df = self._compute_possession_adjusted_features(df)
        df = self._compute_four_factors(df)
        df = self._compute_matchup_features(df)
        df = self._compute_availability_features(df)

        # 4. Form & streaks
        df = self._compute_form_features(df)

        # 5. Quarter momentum
        df = self._compute_quarter_features(df)

        # 6. Rest & fatigue + schedule density + travel features
        df = self._compute_rest_features(df)
        df = self._compute_schedule_density(df)
        df = add_travel_features(df, sport="basketball")
        df = self._compute_load_management_features(df)

        # 7. Point differential trends
        df = self._compute_margin_features(df)

        # 8. Encode target (binary: home_win / away_win)
        df, self.label_map = self.encode_target(df, target_col="result")

        # 9. Drop derived current-game leakage columns. We intentionally keep
        #    raw final scores in the feature cache so market backtests
        #    (totals/spreads/team totals) can derive labels later. Training
        #    configs still remove the raw score fields before fitting.
        leakage_cols = [
            "point_diff", "total_points",            # raw current-game values
            "home_q1", "home_q2", "home_q3", "home_q4", "home_ot",
            "away_q1", "away_q2", "away_q3", "away_q4", "away_ot",
            "home_first_half", "home_second_half",
            "away_first_half", "away_second_half",
            "home_half_ratio", "away_half_ratio", "went_to_ot",
        ]
        df = df.drop(columns=[c for c in leakage_cols if c in df.columns], errors="ignore")

        logger.info("Basketball features complete: %d rows, %d columns", *df.shape)
        return df

    # ------------------------------------------------------------------
    # Possession-adjusted strength / pace projection hooks
    # ------------------------------------------------------------------

    def _compute_possession_adjusted_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add pregame possession-adjusted team strength features.

        True NBA possession data is not always present in the cached API feed.
        When it is missing, this keeps explicit proxy columns and a warning flag
        instead of pretending raw points per game are equivalent to efficiency.
        """
        possession_available = {"home_possessions", "away_possessions"}.issubset(df.columns)
        df["possession_data_available"] = int(possession_available)

        if possession_available:
            df["home_off_rating_per_100"] = df.apply(
                lambda r: safe_divide(r.get("home_score", 0), r.get("home_possessions", 0), 1.13) * 100,
                axis=1,
            )
            df["away_off_rating_per_100"] = df.apply(
                lambda r: safe_divide(r.get("away_score", 0), r.get("away_possessions", 0), 1.13) * 100,
                axis=1,
            )
            df["home_def_rating_per_100"] = df.apply(
                lambda r: safe_divide(r.get("away_score", 0), r.get("away_possessions", 0), 1.13) * 100,
                axis=1,
            )
            df["away_def_rating_per_100"] = df.apply(
                lambda r: safe_divide(r.get("home_score", 0), r.get("home_possessions", 0), 1.13) * 100,
                axis=1,
            )
            df["team_pace"] = (df["home_possessions"].fillna(100) + df["away_possessions"].fillna(100)) / 2
        else:
            df["home_off_rating_per_100"] = df.get("home_ortg", pd.Series(100.0, index=df.index)).fillna(100.0)
            df["away_off_rating_per_100"] = df.get("away_ortg", pd.Series(100.0, index=df.index)).fillna(100.0)
            df["home_def_rating_per_100"] = df.get("home_drtg", pd.Series(100.0, index=df.index)).fillna(100.0)
            df["away_def_rating_per_100"] = df.get("away_drtg", pd.Series(100.0, index=df.index)).fillna(100.0)
            df["team_pace"] = df.get("expected_pace", pd.Series(220.0, index=df.index)).fillna(220.0) / 2.2

        df["home_net_rating_per_100"] = df["home_off_rating_per_100"] - df["home_def_rating_per_100"]
        df["away_net_rating_per_100"] = df["away_off_rating_per_100"] - df["away_def_rating_per_100"]
        df["net_rating_per_100_diff"] = df["home_net_rating_per_100"] - df["away_net_rating_per_100"]
        df["opponent_adjusted_net_rating_diff"] = (
            df.get("srs_diff", pd.Series(0.0, index=df.index)).fillna(0.0) * 0.55
            + df["net_rating_per_100_diff"].fillna(0.0) * 0.45
        )
        df["home_net_rating_split"] = (
            df.get("home_home_wpct_20", pd.Series(0.5, index=df.index)).fillna(0.5) - 0.5
        ) * 12.0
        df["away_net_rating_split"] = (
            df.get("away_away_wpct_20", pd.Series(0.5, index=df.index)).fillna(0.5) - 0.5
        ) * 12.0
        df["recent_net_rating_exp_decay"] = (
            df.get("margin_diff_5", pd.Series(0.0, index=df.index)).fillna(0.0) * 0.50
            + df.get("margin_diff_10", pd.Series(0.0, index=df.index)).fillna(0.0) * 0.30
            + df.get("margin_diff_20", pd.Series(0.0, index=df.index)).fillna(0.0) * 0.20
        )
        df["expected_matchup_pace"] = (
            df.get("home_pace", pd.Series(220.0, index=df.index)).fillna(220.0)
            + df.get("away_pace", pd.Series(220.0, index=df.index)).fillna(220.0)
        ) / 4.4
        df["slow_team_pace_control_adjustment"] = -np.maximum(
            0.0,
            98.0 - np.minimum(
                df.get("home_pace", pd.Series(220.0, index=df.index)).fillna(220.0) / 2.2,
                df.get("away_pace", pd.Series(220.0, index=df.index)).fillna(220.0) / 2.2,
            ),
        )
        df["possessions_projection"] = (
            df["expected_matchup_pace"].fillna(100.0) + df["slow_team_pace_control_adjustment"].fillna(0.0)
        ).clip(lower=88.0, upper=108.0)
        avg_eff = (
            df["home_off_rating_per_100"].fillna(113.0)
            + df["away_off_rating_per_100"].fillna(113.0)
            + df["home_def_rating_per_100"].fillna(113.0)
            + df["away_def_rating_per_100"].fillna(113.0)
        ) / 4.0
        df["projected_total_from_pace_efficiency"] = (df["possessions_projection"] * avg_eff * 2.0) / 100.0
        return df

    # ------------------------------------------------------------------
    # Four Factors / matchup hooks
    # ------------------------------------------------------------------

    def _compute_four_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        defaults = pd.Series(np.nan, index=df.index)

        def _col(name: str) -> pd.Series:
            return df[name] if name in df.columns else defaults

        for side, opp in (("home", "away"), ("away", "home")):
            fga = _col(f"{side}_fga")
            fg3m = _col(f"{side}_fg3m")
            fgm = _col(f"{side}_fgm")
            tov = _col(f"{side}_tov")
            fta = _col(f"{side}_fta")
            orb = _col(f"{side}_orb")
            opp_drb = _col(f"{opp}_drb")

            df[f"{side}_efg_pct"] = ((fgm + (0.5 * fg3m)) / fga.replace(0, np.nan)).fillna(0.52)
            df[f"{side}_tov_rate"] = (tov / (fga + (0.44 * fta) + tov).replace(0, np.nan)).fillna(0.13)
            df[f"{side}_oreb_rate"] = (orb / (orb + opp_drb).replace(0, np.nan)).fillna(0.25)
            df[f"{side}_free_throw_rate"] = (fta / fga.replace(0, np.nan)).fillna(0.24)

        df["efg_pct_diff"] = df["home_efg_pct"] - df["away_efg_pct"]
        df["tov_rate_diff"] = df["away_tov_rate"] - df["home_tov_rate"]
        df["oreb_rate_diff"] = df["home_oreb_rate"] - df["away_oreb_rate"]
        df["free_throw_rate_diff"] = df["home_free_throw_rate"] - df["away_free_throw_rate"]
        df["four_factors_diff"] = (
            (df["efg_pct_diff"] * 10.0)
            + (df["tov_rate_diff"] * 5.0)
            + (df["oreb_rate_diff"] * 3.0)
            + (df["free_throw_rate_diff"] * 2.0)
        )
        df["defensive_four_factors_available"] = int(
            any(c.endswith("_drb") or c.endswith("_stl") or c.endswith("_blk") for c in df.columns)
        )
        return df

    def _compute_matchup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        defaults = pd.Series(0.0, index=df.index)
        missing_cols: Dict[str, object] = {}
        for col in [
            "home_3pa_rate", "away_3pa_rate", "home_opp_3pa_allowed", "away_opp_3pa_allowed",
            "home_rim_frequency", "away_rim_frequency", "home_rim_protection", "away_rim_protection",
            "home_transition_frequency", "away_transition_frequency", "home_halfcourt_efficiency",
            "away_halfcourt_efficiency", "home_foul_rate", "away_foul_rate", "home_steal_rate",
            "away_steal_rate",
        ]:
            if col not in df.columns:
                missing_cols[col] = defaults
        df = self._add_columns(df, missing_cols)

        three_point_volume = df["home_3pa_rate"] + df["away_3pa_rate"]
        rim_pressure_edge = (
            df["home_rim_frequency"] - df["away_rim_protection"]
        ) - (
            df["away_rim_frequency"] - df["home_rim_protection"]
        )
        turnover_flag = (
            (df.get("home_tov_rate", pd.Series(0.13, index=df.index)) > 0.15) & (df["away_steal_rate"] > 0.08)
        ).astype(int) - (
            (df.get("away_tov_rate", pd.Series(0.13, index=df.index)) > 0.15) & (df["home_steal_rate"] > 0.08)
        ).astype(int)
        garbage_available = int(
            {"home_non_garbage_ortg", "away_non_garbage_ortg"}.issubset(df.columns)
        )
        return self._add_columns(
            df,
            {
                "three_point_volume_matchup": three_point_volume,
                "high_variance_total_3pa_flag": (three_point_volume >= 0.82).astype(int),
                "rim_pressure_edge": rim_pressure_edge,
                "transition_edge": df["home_transition_frequency"] - df["away_transition_frequency"],
                "halfcourt_efficiency_edge": df["home_halfcourt_efficiency"] - df["away_halfcourt_efficiency"],
                "foul_pressure_edge": df["away_foul_rate"] - df["home_foul_rate"],
                "turnover_prone_vs_steal_defense_flag": turnover_flag,
                "garbage_time_filtered_available": garbage_available,
                "garbage_time_warning": int(garbage_available == 0),
            },
        )

    # ------------------------------------------------------------------
    # Player availability and load-management hooks
    # ------------------------------------------------------------------

    def _compute_availability_features(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_cols = {}
        for col, default in {
            "home_star_player_missing": 0,
            "away_star_player_missing": 0,
            "home_expected_minutes_lost": 0.0,
            "away_expected_minutes_lost": 0.0,
            "home_replacement_quality_gap": 0.0,
            "away_replacement_quality_gap": 0.0,
            "home_top8_rotation_continuity": 1.0,
            "away_top8_rotation_continuity": 1.0,
            "home_questionable_count": 0,
            "away_questionable_count": 0,
            "home_doubtful_count": 0,
            "away_doubtful_count": 0,
        }.items():
            if col not in df.columns:
                missing_cols[col] = default
        df = self._add_columns(df, missing_cols)

        availability_delta = (
            ((df["away_expected_minutes_lost"] - df["home_expected_minutes_lost"]) / 48.0) * 1.6
            + ((df["away_star_player_missing"] - df["home_star_player_missing"]) * 2.2)
            + ((df["home_top8_rotation_continuity"] - df["away_top8_rotation_continuity"]) * 2.0)
            - (df["home_replacement_quality_gap"] - df["away_replacement_quality_gap"])
        )
        availability_net_rating_delta = availability_delta.clip(lower=-8.0, upper=8.0)
        injury_adjusted_net_rating_diff = (
            df.get("opponent_adjusted_net_rating_diff", pd.Series(0.0, index=df.index)).fillna(0.0)
            + availability_net_rating_delta.fillna(0.0)
        )
        late_injury_uncertainty_flag = (
            (df["home_questionable_count"] + df["away_questionable_count"] + df["home_doubtful_count"] + df["away_doubtful_count"]) > 0
        ).astype(int)
        return self._add_columns(
            df,
            {
                "availability_net_rating_delta": availability_net_rating_delta,
                "injury_adjusted_net_rating_diff": injury_adjusted_net_rating_diff,
                "late_injury_uncertainty_flag": late_injury_uncertainty_flag,
            },
        )

    def _compute_load_management_features(self, df: pd.DataFrame) -> pd.DataFrame:
        home_second = df.get("home_b2b", pd.Series(0.0, index=df.index)).fillna(0.0).astype(float)
        away_second = df.get("away_b2b", pd.Series(0.0, index=df.index)).fillna(0.0).astype(float)
        home_3in4 = df.get("home_3in4", pd.Series(0, index=df.index)).fillna(0).astype(int)
        away_3in4 = df.get("away_3in4", pd.Series(0, index=df.index)).fillna(0).astype(int)
        home_l5 = df.get("home_games_L5D", pd.Series(0, index=df.index)).fillna(0).astype(int)
        away_l5 = df.get("away_games_L5D", pd.Series(0, index=df.index)).fillna(0).astype(int)
        home_l7 = df.get("home_games_L7D", pd.Series(0, index=df.index)).fillna(0).astype(int)
        away_l7 = df.get("away_games_L7D", pd.Series(0, index=df.index)).fillna(0).astype(int)
        load_cols = {
            "home_second_night_b2b": home_second,
            "away_second_night_b2b": away_second,
            "home_third_game_in_four": home_3in4,
            "away_third_game_in_four": away_3in4,
            "home_games_last_5": home_l5,
            "away_games_last_5": away_l5,
            "home_games_last_7": home_l7,
            "away_games_last_7": away_l7,
            "rest_load_edge": (
                away_second - home_second
                + (away_3in4 - home_3in4) * 0.6
                + (away_l7 - home_l7) * 0.25
            ),
            "altitude_flag": df["home_team"].astype(str).str.contains("Denver", case=False, na=False).astype(int),
            "extended_road_trip_flag": (
                df.get("away_travel_bucket", pd.Series(0.0, index=df.index)).fillna(0.0) >= 3
            ).astype(int),
        }
        if "prior_game_ot_flag" not in df.columns:
            load_cols["prior_game_ot_flag"] = 0
        return self._add_columns(
            df,
            load_cols,
        )

    # ------------------------------------------------------------------
    # Scoring Features
    # ------------------------------------------------------------------

    def _compute_scoring_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling points scored and allowed for home and away teams."""
        w = self._form_window

        # Home team scoring
        df["home_ppg"] = self.compute_rolling_stats(
            df, "home_team", "home_score", w, "home_ppg",
            other_team_col="away_team", other_value_col="away_score"
        )
        df["home_opp_ppg"] = self.compute_rolling_stats(
            df, "home_team", "away_score", w, "home_opp_ppg",
            other_team_col="away_team", other_value_col="home_score"
        )

        # Away team scoring
        df["away_ppg"] = self.compute_rolling_stats(
            df, "away_team", "away_score", w, "away_ppg",
            other_team_col="home_team", other_value_col="home_score"
        )
        df["away_opp_ppg"] = self.compute_rolling_stats(
            df, "away_team", "home_score", w, "away_opp_ppg",
            other_team_col="home_team", other_value_col="away_score"
        )

        # Scoring differentials
        df["home_scoring_margin"] = df["home_ppg"].fillna(0) - df["home_opp_ppg"].fillna(0)
        df["away_scoring_margin"] = df["away_ppg"].fillna(0) - df["away_opp_ppg"].fillna(0)
        df["scoring_margin_diff"] = df["home_scoring_margin"] - df["away_scoring_margin"]

        return df

    # ------------------------------------------------------------------
    # Pace Features
    # ------------------------------------------------------------------

    def _compute_pace_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute pace proxy as total points per game.

        True pace requires possession data; we use total points as
        a reasonable proxy indicating game tempo.
        """
        w = self._pace_window

        df["total_points"] = df["home_score"].fillna(0) + df["away_score"].fillna(0)

        # Home team's game pace
        df["home_pace"] = self.compute_rolling_stats(
            df, "home_team", "total_points", w, "home_pace",
            other_team_col="away_team", other_value_col="total_points"
        )

        # Away team's game pace
        df["away_pace"] = self.compute_rolling_stats(
            df, "away_team", "total_points", w, "away_pace",
            other_team_col="home_team", other_value_col="total_points"
        )

        # Expected pace for this matchup
        df["expected_pace"] = (df["home_pace"].fillna(200) + df["away_pace"].fillna(200)) / 2

        # Over/under indicator (games trending high/low scoring)
        league_avg_pace = df["total_points"].expanding(min_periods=20).mean().shift(1)
        df["home_pace_vs_avg"] = df["home_pace"].fillna(0) - league_avg_pace.fillna(200)
        df["away_pace_vs_avg"] = df["away_pace"].fillna(0) - league_avg_pace.fillna(200)

        return df

    # ------------------------------------------------------------------
    # Offensive / Defensive Ratings
    # ------------------------------------------------------------------

    def _compute_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute offensive and defensive rating proxies.

        ORtg proxy = PPG / league_avg * 100
        DRtg proxy = Opp_PPG / league_avg * 100
        Net rating = ORtg - DRtg

        These are kept for backwards compat; SRS-adjusted versions added in
        _compute_srs_features().
        """
        league_avg_ppg = df["total_points"].expanding(min_periods=20).mean().shift(1) / 2

        # Home ratings
        df["home_ortg"] = df.apply(
            lambda r: safe_divide(r.get("home_ppg", 0), league_avg_ppg.get(r.name, 100), 1.0) * 100,
            axis=1,
        )
        df["home_drtg"] = df.apply(
            lambda r: safe_divide(r.get("home_opp_ppg", 0), league_avg_ppg.get(r.name, 100), 1.0) * 100,
            axis=1,
        )
        df["home_net_rtg"] = df["home_ortg"] - df["home_drtg"]

        # Away ratings
        df["away_ortg"] = df.apply(
            lambda r: safe_divide(r.get("away_ppg", 0), league_avg_ppg.get(r.name, 100), 1.0) * 100,
            axis=1,
        )
        df["away_drtg"] = df.apply(
            lambda r: safe_divide(r.get("away_opp_ppg", 0), league_avg_ppg.get(r.name, 100), 1.0) * 100,
            axis=1,
        )
        df["away_net_rtg"] = df["away_ortg"] - df["away_drtg"]

        # Net rating differential
        df["net_rtg_diff"] = df["home_net_rtg"] - df["away_net_rtg"]

        return df

    # ------------------------------------------------------------------
    # SRS (Simple Rating System) — opponent-adjusted point differential
    # ------------------------------------------------------------------

    def _compute_srs_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute SRS for each team as of each game date using only past games.

        SRS algorithm (iterative, 3 passes):
            1. Start: each team's SRS = avg point differential
            2. Iterate: SRS[t] = avg(margin[t,i] + SRS[opp_i]) / num_games
            3. Repeat 3 times to propagate through the schedule

        SRS is attached as home_srs and away_srs (season-to-date up to game date).
        Also computes:
            home_sos / away_sos  — strength of schedule (avg opponent SRS)
            srs_diff             — home_srs - away_srs
        """
        teams = list(set(df["home_team"].tolist() + df["away_team"].tolist()))

        # Build chronological game list for iteration
        game_records = []
        for _, row in df.iterrows():
            game_records.append({
                "date": row["date"],
                "home": row["home_team"],
                "away": row["away_team"],
                "home_margin": row.get("home_score", 0) - row.get("away_score", 0),
            })

        # Compute rolling SRS as of each unique date
        sorted_dates = sorted(df["date"].unique())

        # Build a lookup: for each game index → (home_srs, away_srs, home_sos, away_sos)
        srs_lookup: Dict[int, Tuple[float, float, float, float]] = {}

        for game_idx, row in df.iterrows():
            cutoff = row["date"]

            # Gather all past games (before this game's date)
            past = [g for g in game_records if g["date"] < cutoff]
            if len(past) < _MIN_GAMES * 2:
                srs_lookup[game_idx] = (0.0, 0.0, 0.0, 0.0)
                continue

            # Initialize SRS as avg margin per team
            team_margins: Dict[str, list] = {t: [] for t in teams}
            for g in past:
                team_margins[g["home"]].append(g["home_margin"])
                team_margins[g["away"]].append(-g["home_margin"])

            srs: Dict[str, float] = {}
            for t in teams:
                margins = team_margins.get(t, [])
                srs[t] = np.mean(margins) if margins else 0.0

            # 3 SRS iterations
            for _ in range(3):
                new_srs: Dict[str, float] = {}
                for t in teams:
                    margins_and_opp = []
                    for g in past:
                        if g["home"] == t:
                            margins_and_opp.append(g["home_margin"] + srs.get(g["away"], 0.0))
                        elif g["away"] == t:
                            margins_and_opp.append(-g["home_margin"] + srs.get(g["home"], 0.0))
                    new_srs[t] = np.mean(margins_and_opp) if margins_and_opp else 0.0
                srs = new_srs

            # Strength of schedule: avg SRS of opponents faced
            def _sos(team: str) -> float:
                opp_srs = []
                for g in past:
                    if g["home"] == team:
                        opp_srs.append(srs.get(g["away"], 0.0))
                    elif g["away"] == team:
                        opp_srs.append(srs.get(g["home"], 0.0))
                return np.mean(opp_srs) if opp_srs else 0.0

            ht, at = row["home_team"], row["away_team"]
            srs_lookup[game_idx] = (
                srs.get(ht, 0.0),
                srs.get(at, 0.0),
                _sos(ht),
                _sos(at),
            )

        df["home_srs"] = [srs_lookup[i][0] for i in df.index]
        df["away_srs"] = [srs_lookup[i][1] for i in df.index]
        df["home_sos"] = [srs_lookup[i][2] for i in df.index]
        df["away_sos"] = [srs_lookup[i][3] for i in df.index]
        df["srs_diff"] = df["home_srs"] - df["away_srs"]

        logger.info("SRS features computed for %d games", len(df))
        return df

    # ------------------------------------------------------------------
    # Multi-window rolling stats (5/10/20 games)
    # ------------------------------------------------------------------

    def _compute_multiwindow_rolling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rolling win%, avg margin, and PPG at 5/10/20 game windows.
        Uses strict look-ahead protection (only past games).
        """
        # Build per-team game history
        home_rows = df[["date", "home_team", "home_score", "away_score", "result"]].copy()
        home_rows = home_rows.rename(columns={"home_team": "team", "home_score": "scored", "away_score": "conceded"})
        home_rows["won"] = (home_rows["result"] == "home_win").astype(int)

        away_rows = df[["date", "away_team", "away_score", "home_score", "result"]].copy()
        away_rows = away_rows.rename(columns={"away_team": "team", "away_score": "scored", "home_score": "conceded"})
        away_rows["won"] = (away_rows["result"] == "away_win").astype(int)

        all_rows = pd.concat([home_rows, away_rows]).sort_values("date")
        cache: Dict[str, pd.DataFrame] = {}
        for team, grp in all_rows.groupby("team"):
            cache[team] = grp.sort_values("date").reset_index(drop=True)

        def _roll(team: str, before_date, window: int) -> dict:
            if team not in cache:
                return {}
            t = cache[team]
            past = t[t["date"] < before_date].tail(window)
            if len(past) < _MIN_GAMES:
                return {}
            return {
                "win_pct":    past["won"].mean(),
                "avg_margin": (past["scored"] - past["conceded"]).mean(),
                "ppg":        past["scored"].mean(),
                "rapg":       past["conceded"].mean(),
            }

        for w in _ROLL_WINDOWS:
            home_stats = df.apply(lambda r: _roll(r["home_team"], r["date"], w), axis=1)
            away_stats = df.apply(lambda r: _roll(r["away_team"], r["date"], w), axis=1)
            for stat in ("win_pct", "avg_margin", "ppg", "rapg"):
                df[f"home_{stat}_{w}"] = home_stats.apply(lambda d: d.get(stat, np.nan))
                df[f"away_{stat}_{w}"] = away_stats.apply(lambda d: d.get(stat, np.nan))
                if stat == "win_pct":
                    df[f"win_pct_diff_{w}"] = df[f"home_win_pct_{w}"] - df[f"away_win_pct_{w}"]
                if stat == "avg_margin":
                    df[f"margin_diff_{w}"] = df[f"home_avg_margin_{w}"] - df[f"away_avg_margin_{w}"]

        logger.info("Multi-window rolling stats added (%s windows)", _ROLL_WINDOWS)
        return df

    # ------------------------------------------------------------------
    # Home/away context splits
    # ------------------------------------------------------------------

    def _compute_home_away_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Win% for home team when playing at home (last 20 home games),
        win% for away team when playing away (last 20 away games).
        Season games played proxy.
        """
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

        return self._add_columns(
            df,
            {
                "home_home_wpct_20": h_home_wpct,
                "away_away_wpct_20": a_away_wpct,
                "home_season_games_played": sgp_home,
                "away_season_games_played": sgp_away,
            },
        )

    # ------------------------------------------------------------------
    # Form & Streaks
    # ------------------------------------------------------------------

    def _compute_form_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Win form and current streak length."""
        w = self._form_window

        home_win_form = self.compute_form(
            df, "home_team", "result", "home_win", w,
            other_team_col="away_team", other_target_result="away_win"
        )
        away_win_form = self.compute_form(
            df, "away_team", "result", "away_win", w,
            other_team_col="home_team", other_target_result="home_win"
        )
        return self._add_columns(
            df,
            {
                "home_win_form": home_win_form,
                "away_win_form": away_win_form,
                "form_diff": home_win_form.fillna(0.5) - away_win_form.fillna(0.5),
                "home_streak": self._compute_streak(df, "home_team", "result", "home_win"),
                "away_streak": self._compute_streak(df, "away_team", "result", "away_win"),
            },
        )

    def _compute_streak(
        self,
        df: pd.DataFrame,
        team_col: str,
        result_col: str,
        win_value: str,
    ) -> pd.Series:
        """
        Compute current winning/losing streak length for each team.

        Positive = win streak, negative = loss streak.
        """
        streaks = pd.Series(0, index=df.index, dtype=float)
        team_streaks: Dict[str, int] = {}

        for idx, row in df.iterrows():
            team = row[team_col]
            current = team_streaks.get(team, 0)
            streaks.iloc[idx] = current  # record pre-game streak

            # Update streak after game
            if row[result_col] == win_value:
                team_streaks[team] = max(current, 0) + 1
            else:
                team_streaks[team] = min(current, 0) - 1

        return streaks

    # ------------------------------------------------------------------
    # Quarter Features
    # ------------------------------------------------------------------

    def _compute_quarter_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute quarter-by-quarter scoring patterns.

        Focuses on Q4 performance (clutch indicator) and first-half
        vs second-half scoring balance.
        """
        # Check if quarter data is available
        q_cols = ["home_q1", "home_q2", "home_q3", "home_q4",
                  "away_q1", "away_q2", "away_q3", "away_q4"]
        available = [c for c in q_cols if c in df.columns]

        if len(available) < 8:
            logger.debug("Quarter data not fully available; skipping quarter features")
            return df

        # First half / second half scoring
        home_first_half = df["home_q1"].fillna(0) + df["home_q2"].fillna(0)
        home_second_half = df["home_q3"].fillna(0) + df["home_q4"].fillna(0)
        away_first_half = df["away_q1"].fillna(0) + df["away_q2"].fillna(0)
        away_second_half = df["away_q3"].fillna(0) + df["away_q4"].fillna(0)

        # Q4 strength (clutch performance)
        w = self._form_window
        home_q4_avg = self.compute_rolling_stats(
            df, "home_team", "home_q4", w, "home_q4",
            other_team_col="away_team", other_value_col="away_q4"
        )
        away_q4_avg = self.compute_rolling_stats(
            df, "away_team", "away_q4", w, "away_q4",
            other_team_col="home_team", other_value_col="home_q4"
        )

        # Second-half surge: do they improve in 2nd half?
        home_half_ratio = (home_second_half / home_first_half.replace(0, np.nan)).fillna(1.0)
        away_half_ratio = (away_second_half / away_first_half.replace(0, np.nan)).fillna(1.0)

        # Overtime flag (indicator of close-game propensity)
        quarter_cols: Dict[str, object] = {
            "home_first_half": home_first_half,
            "home_second_half": home_second_half,
            "away_first_half": away_first_half,
            "away_second_half": away_second_half,
            "home_q4_avg": home_q4_avg,
            "away_q4_avg": away_q4_avg,
            "home_half_ratio": home_half_ratio,
            "away_half_ratio": away_half_ratio,
        }
        if "home_ot" in df.columns:
            quarter_cols["went_to_ot"] = (df["home_ot"].fillna(0) > 0).astype(float)

        return self._add_columns(df, quarter_cols)

    # ------------------------------------------------------------------
    # Elo ratings (opponent-adjusted, with MOV scaling)
    # ------------------------------------------------------------------

    def _compute_elo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rolling Elo rating for each team, computed chronologically.

        K-factor scales with log(1 + margin) to reward dominant wins.
        Home court advantage is ~100 Elo points (NBA empirical value).
        """
        elo_initial: float = 1500.0
        elo_k: float = 20.0
        home_adv: float = 100.0

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
            if margin > 0:
                actual_home = 1.0
            elif margin < 0:
                actual_home = 0.0
            else:
                actual_home = 0.5

            # MOV multiplier: log scale capped at 3 to prevent blowouts dominating
            k_mult = min(3.0, math.log(1.0 + abs(margin)) * 0.35 + 0.7) if margin != 0 else 1.0
            k = elo_k * k_mult

            elo_ratings[home] = h_elo + k * (actual_home - exp_home)
            elo_ratings[away] = a_elo + k * ((1.0 - actual_home) - (1.0 - exp_home))

        df["home_elo"] = home_elos
        df["away_elo"] = away_elos
        df["elo_diff"] = df["home_elo"] - df["away_elo"]
        df["elo_win_prob"] = 1.0 / (
            1.0 + 10.0 ** ((df["away_elo"] - df["home_elo"] - home_adv) / 400.0)
        )
        logger.info("Basketball Elo features computed for %d games", len(df))
        return df

    # ------------------------------------------------------------------
    # Schedule density (3-in-4-nights, weekly load)
    # ------------------------------------------------------------------

    def _compute_schedule_density(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Games played in the last N days for each team (home + away combined).

        Windows: 3, 5, 7, 10 days.
        Also produces a 3-in-4-nights flag and a weekly density score.
        Uses searchsorted for O(n log n) per team.
        """
        windows = [3, 5, 7, 10]
        schedule_cols: Dict[str, object] = {}

        # Build sorted date arrays per team across all appearances
        team_dates: Dict[str, np.ndarray] = {}
        for team in set(df["home_team"]).union(df["away_team"]):
            h_dates = df.loc[df["home_team"] == team, "date"].values
            a_dates = df.loc[df["away_team"] == team, "date"].values
            team_dates[team] = np.sort(np.concatenate([h_dates, a_dates]))

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
                    hi = np.searchsorted(arr, d, side="left")  # excludes current game
                    counts.append(int(hi - lo))
                schedule_cols[col] = counts

        # 3-in-4-nights: ≥2 games in last 3 days (3 games in 4 nights = 2 prior)
        home_l3 = pd.Series(schedule_cols["home_games_L3D"], index=df.index)
        away_l3 = pd.Series(schedule_cols["away_games_L3D"], index=df.index)
        home_l7 = pd.Series(schedule_cols["home_games_L7D"], index=df.index)
        away_l7 = pd.Series(schedule_cols["away_games_L7D"], index=df.index)
        schedule_cols["home_3in4"] = (home_l3 >= 2).astype(int)
        schedule_cols["away_3in4"] = (away_l3 >= 2).astype(int)

        # Weekly density: games per day over last 7 days
        home_density = home_l7 / 7.0
        away_density = away_l7 / 7.0
        schedule_cols["home_density_7"] = home_density
        schedule_cols["away_density_7"] = away_density
        schedule_cols["density_diff"] = home_density - away_density

        return self._add_columns(df, schedule_cols)

    # ------------------------------------------------------------------
    # Rest & Fatigue
    # ------------------------------------------------------------------

    def _compute_rest_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Days rest and back-to-back detection."""
        home_rest = self.compute_days_rest(
            df, "home_team", other_team_col="away_team"
        )
        away_rest = self.compute_days_rest(
            df, "away_team", other_team_col="home_team"
        )

        return self._add_columns(
            df,
            {
                "home_rest_days": home_rest,
                "away_rest_days": away_rest,
                "rest_diff": home_rest.fillna(2) - away_rest.fillna(2),
                "home_b2b": (home_rest.fillna(3) <= 1.5).astype(float),
                "away_b2b": (away_rest.fillna(3) <= 1.5).astype(float),
                "home_fatigue": self._compute_fatigue_index(df, "home_team"),
                "away_fatigue": self._compute_fatigue_index(df, "away_team"),
            },
        )

    def _compute_fatigue_index(
        self,
        df: pd.DataFrame,
        team_col: str,
        window_days: int = 7,
    ) -> pd.Series:
        """
        Compute a fatigue index: number of games played in the last
        N days for each team.
        """
        fatigue = pd.Series(0.0, index=df.index)

        for team in df[team_col].unique():
            mask = df[team_col] == team
            team_dates = df.loc[mask, "date"].values

            counts = []
            for i, d in enumerate(team_dates):
                if i == 0:
                    counts.append(0)
                    continue
                cutoff = d - np.timedelta64(window_days, "D")
                recent = sum(1 for t in team_dates[:i] if t >= cutoff)
                counts.append(recent)

            fatigue.loc[mask] = counts

        return fatigue

    # ------------------------------------------------------------------
    # Point Differential / Margin
    # ------------------------------------------------------------------

    def _compute_margin_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling average margin of victory/defeat."""
        w = self._form_window

        df["point_diff"] = df["home_score"].fillna(0) - df["away_score"].fillna(0)
        df["neg_point_diff"] = -df["point_diff"]

        df["home_avg_margin"] = self.compute_rolling_stats(
            df, "home_team", "point_diff", w, "home_margin",
            other_team_col="away_team", other_value_col="neg_point_diff"
        )
        df["away_avg_margin"] = self.compute_rolling_stats(
            df, "away_team", "neg_point_diff", w, "away_margin",
            other_team_col="home_team", other_value_col="point_diff"
        )

        df["margin_diff"] = df["home_avg_margin"].fillna(0) - df["away_avg_margin"].fillna(0)

        # Consistency (std of margin)
        df["home_margin_std"] = self.compute_rolling_stats(
            df, "home_team", "point_diff", w, "home_margin_std", agg_func="std",
            other_team_col="away_team", other_value_col="neg_point_diff"
        )
        df["away_margin_std"] = self.compute_rolling_stats(
            df, "away_team", "neg_point_diff", w, "away_margin_std", agg_func="std",
            other_team_col="home_team", other_value_col="point_diff"
        )

        # Clean up temp column
        df = df.drop(columns=["neg_point_diff"], errors="ignore")

        return df
