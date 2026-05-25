"""
Abstract base class for sport-specific feature engineering.

Provides shared rolling window, encoding, and target generation logic.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)


class BaseFeatureEngineer(ABC):
    """
    Abstract base for all feature engineering classes.

    Subclasses must implement:
        - engineer_features(df) -> pd.DataFrame
        - _compute_sport_specific_features(df) -> pd.DataFrame

    Parameters
    ----------
    sport : str
        Sport identifier ('soccer', 'basketball', 'tennis').
    """

    def __init__(self, sport: str) -> None:
        self.sport = sport
        self._sport_cfg = settings.get("sports", {}).get(sport, {})
        self._feature_cfg = settings.get("features", {})
        self._form_window = self._sport_cfg.get("form_window", 5)
        self._drop_columns = self._feature_cfg.get("drop_columns", [])

        logger.info("Initialized %s feature engineer (form_window=%d)", sport, self._form_window)

    # ------------------------------------------------------------------
    # Shared feature computations
    # ------------------------------------------------------------------

    def compute_rolling_stats(
        self,
        df: pd.DataFrame,
        team_col: str,
        value_col: str,
        window: int,
        stat_name: str,
        agg_func: str = "mean",
        other_team_col: Optional[str] = None,
        other_value_col: Optional[str] = None,
    ) -> pd.Series:
        """
        Compute a rolling aggregate for a team over their last N matches.

        IMPORTANT: Uses shift(1) to prevent lookahead bias — the current
        row's value is excluded from the calculation.

        Parameters
        ----------
        df : pd.DataFrame
            Match data sorted by date.
        team_col : str
            Column identifying the team (e.g. 'home_team').
        value_col : str
            Column to aggregate (e.g. 'home_goals').
        window : int
            Number of past matches to include.
        stat_name : str
            Name prefix for the output column.
        agg_func : str
            Aggregation function ('mean', 'sum', 'std').

        Returns
        -------
        pd.Series
            Rolling aggregate values aligned to the DataFrame index.
        """
        result = pd.Series(np.nan, index=df.index, name=f"{stat_name}_roll{window}")

        if not other_team_col or not other_value_col:
            for team in df[team_col].unique():
                mask = df[team_col] == team
                team_vals = df.loc[mask, value_col].shift(1)  # shift to avoid lookahead
                result.loc[mask] = self._rolling_aggregate(team_vals, window, agg_func).values
            return result

        history = self._build_team_history(
            df=df,
            primary_team_col=team_col,
            primary_value=df[value_col],
            secondary_team_col=other_team_col,
            secondary_value=df[other_value_col],
        )
        history["rolled"] = np.nan

        for _, idxs in history.groupby("team", sort=False).groups.items():
            team_vals = history.loc[idxs, "value"].shift(1)
            history.loc[idxs, "rolled"] = self._rolling_aggregate(team_vals, window, agg_func).values

        primary_rows = history[history["source"] == "primary"].sort_values("row_idx")
        result.loc[primary_rows["row_idx"]] = primary_rows["rolled"].values
        return result

    def compute_form(
        self,
        df: pd.DataFrame,
        team_col: str,
        result_col: str,
        target_result: str,
        window: int,
        other_team_col: Optional[str] = None,
        other_target_result: Optional[str] = None,
    ) -> pd.Series:
        """
        Compute form as a win ratio over the last N matches.

        Parameters
        ----------
        df : pd.DataFrame
            Sorted match data.
        team_col : str
            Team column.
        result_col : str
            Result column (e.g. 'result').
        target_result : str
            What counts as a "win" (e.g. 'home_win').
        window : int
            Lookback window.

        Returns
        -------
        pd.Series
            Win ratio [0, 1] over last N games.
        """
        form = pd.Series(np.nan, index=df.index)

        if not other_team_col or other_target_result is None:
            is_win = (df[result_col] == target_result).astype(float)
            for team in df[team_col].unique():
                mask = df[team_col] == team
                team_wins = is_win.loc[mask].shift(1)
                rolled = team_wins.rolling(window=window, min_periods=1).mean()
                form.loc[mask] = rolled.values
            return form

        history = self._build_team_history(
            df=df,
            primary_team_col=team_col,
            primary_value=(df[result_col] == target_result).astype(float),
            secondary_team_col=other_team_col,
            secondary_value=(df[result_col] == other_target_result).astype(float),
        )
        history["rolled"] = np.nan

        for _, idxs in history.groupby("team", sort=False).groups.items():
            team_wins = history.loc[idxs, "value"].shift(1)
            history.loc[idxs, "rolled"] = team_wins.rolling(window=window, min_periods=1).mean().values

        primary_rows = history[history["source"] == "primary"].sort_values("row_idx")
        form.loc[primary_rows["row_idx"]] = primary_rows["rolled"].values

        return form

    def compute_days_rest(
        self,
        df: pd.DataFrame,
        team_col: str,
        date_col: str = "date",
        other_team_col: Optional[str] = None,
    ) -> pd.Series:
        """
        Compute days of rest since the team's last match.

        Parameters
        ----------
        df : pd.DataFrame
            Sorted match data.
        team_col : str
            Team column.
        date_col : str
            Date column.

        Returns
        -------
        pd.Series
            Days since last match (NaN for first match).
        """
        rest = pd.Series(np.nan, index=df.index)

        if not other_team_col:
            for team in df[team_col].unique():
                mask = df[team_col] == team
                dates = df.loc[mask, date_col].sort_values()
                diff = dates.diff().dt.total_seconds() / 86400
                rest.loc[mask] = diff.values
            return rest

        history = self._build_team_history(
            df=df,
            primary_team_col=team_col,
            primary_value=df[date_col],
            secondary_team_col=other_team_col,
            secondary_value=df[date_col],
            date_col=date_col,
        )
        history["rest_days"] = np.nan

        for _, idxs in history.groupby("team", sort=False).groups.items():
            dates = pd.to_datetime(history.loc[idxs, "event_date"])
            history.loc[idxs, "rest_days"] = dates.diff().dt.total_seconds().div(86400).values

        primary_rows = history[history["source"] == "primary"].sort_values("row_idx")
        rest.loc[primary_rows["row_idx"]] = primary_rows["rest_days"].values
        return rest

    @staticmethod
    def _rolling_aggregate(values: pd.Series, window: int, agg_func: str) -> pd.Series:
        if agg_func == "mean":
            return values.rolling(window=window, min_periods=1).mean()
        if agg_func == "sum":
            return values.rolling(window=window, min_periods=1).sum()
        if agg_func == "std":
            return values.rolling(window=window, min_periods=1).std()
        return values.rolling(window=window, min_periods=1).mean()

    @staticmethod
    def _build_team_history(
        df: pd.DataFrame,
        primary_team_col: str,
        primary_value: pd.Series,
        secondary_team_col: str,
        secondary_value: pd.Series,
        date_col: str = "date",
    ) -> pd.DataFrame:
        primary = pd.DataFrame(
            {
                "row_idx": df.index,
                "event_date": pd.to_datetime(df[date_col]),
                "team": df[primary_team_col],
                "value": primary_value,
                "source": "primary",
                "source_order": 0,
            }
        )
        secondary = pd.DataFrame(
            {
                "row_idx": df.index,
                "event_date": pd.to_datetime(df[date_col]),
                "team": df[secondary_team_col],
                "value": secondary_value,
                "source": "secondary",
                "source_order": 1,
            }
        )
        history = pd.concat([primary, secondary], ignore_index=True)
        history = history.sort_values(
            ["event_date", "row_idx", "source_order"],
            kind="mergesort",
        ).reset_index(drop=True)
        return history

    def encode_target(
        self,
        df: pd.DataFrame,
        target_col: str = "result",
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Encode the target variable as integers.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with target column.
        target_col : str
            Name of the target column.

        Returns
        -------
        tuple[pd.DataFrame, dict]
            DataFrame with encoded target, and label mapping.
        """
        labels = sorted(df[target_col].dropna().unique())
        label_map = {label: idx for idx, label in enumerate(labels)}

        df = df.copy()
        df["target"] = df[target_col].map(label_map)

        logger.info("Target encoding: %s", label_map)
        return df, label_map

    def prepare_for_training(
        self,
        df: pd.DataFrame,
        target_col: str = "target",
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare a feature DataFrame for model training by dropping
        non-feature columns and separating X/y.

        Parameters
        ----------
        df : pd.DataFrame
            Feature-engineered DataFrame.
        target_col : str
            Target column name.

        Returns
        -------
        tuple[pd.DataFrame, pd.Series]
            (X features, y target).
        """
        df_clean = df.dropna(subset=[target_col]).copy()

        # Drop configured non-feature columns
        cols_to_drop = [c for c in self._drop_columns if c in df_clean.columns]
        cols_to_drop.append(target_col)
        if "result" in df_clean.columns and "result" not in cols_to_drop:
            cols_to_drop.append("result")

        # Also drop any remaining string columns
        for col in df_clean.columns:
            if df_clean[col].dtype == "object" and col not in cols_to_drop:
                cols_to_drop.append(col)

        X = df_clean.drop(columns=cols_to_drop, errors="ignore")
        y = df_clean[target_col]

        # Fill any remaining NaN with 0 (after rolling features)
        X = X.fillna(0)

        logger.info(
            "Training data prepared: X shape=%s, y shape=%s, features=%s",
            X.shape, y.shape, list(X.columns),
        )
        return X, y

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering to raw match data.

        Must be implemented by each sport-specific subclass.
        """
        ...
