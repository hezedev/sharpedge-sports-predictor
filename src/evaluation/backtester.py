"""
Walk-forward backtesting engine.

Simulates a realistic betting pipeline by:
1. Training on historical data up to time T
2. Predicting on a forward window T..T+W
3. Simulating bets using the value detector and bankroll manager
4. Sliding forward and repeating

This avoids lookahead bias and gives a realistic estimate of
out-of-sample performance.
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from config import settings
from src.features.base_engineer import BaseFeatureEngineer
from src.models.trainer import ModelTrainer
from src.models.calibration import ProbabilityCalibrator
from src.risk.kelly import KellyCriterion
from src.risk.bankroll import BankrollManager, Bet
from src.risk.value_detector import ValueDetector
from src.evaluation.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


class Backtester:
    """
    Walk-forward backtesting engine for sports prediction models.

    Slides a training window forward through historical data,
    retraining the model periodically, and simulating bets on
    out-of-sample data.

    Parameters
    ----------
    sport : str
        Sport identifier.
    feature_engineer : BaseFeatureEngineer
        Feature engineering instance for this sport.
    initial_bankroll : float
        Starting bankroll for the simulation.
    """

    def __init__(
        self,
        sport: str,
        feature_engineer: BaseFeatureEngineer,
        initial_bankroll: float = 1000.0,
    ) -> None:
        self.sport = sport
        self.feature_engineer = feature_engineer

        bt_cfg = settings.get("backtest", {})
        self._window_days = bt_cfg.get("walk_forward_window", 90)
        self._min_train_samples = bt_cfg.get("min_training_samples", 200)
        self._slippage_pct = bt_cfg.get("slippage_pct", 0.01)
        self._commission_pct = bt_cfg.get("commission_pct", 0.0)

        self._bankroll = BankrollManager(initial_bankroll=initial_bankroll)
        self._kelly = KellyCriterion()
        self._value_detector = ValueDetector(kelly=self._kelly)
        self._metrics = MetricsCalculator()

        # Results storage
        self.predictions_log: List[Dict[str, Any]] = []
        self.bankroll_history: List[float] = [initial_bankroll]
        self.period_results: List[Dict[str, Any]] = []

        logger.info(
            "Backtester: sport=%s, window=%dd, min_train=%d, slippage=%.1f%%",
            sport, self._window_days, self._min_train_samples,
            self._slippage_pct * 100,
        )

    # ------------------------------------------------------------------
    # Main Backtest Loop
    # ------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        odds_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full walk-forward backtest.

        Parameters
        ----------
        df : pd.DataFrame
            Feature-engineered match data with 'date' and 'target' columns.
            Must be sorted by date.
        odds_df : pd.DataFrame, optional
            Historical odds data. If None, uses implied odds from
            a uniform market assumption.

        Returns
        -------
        dict
            Comprehensive backtest results.
        """
        if "date" not in df.columns or "target" not in df.columns:
            raise ValueError("DataFrame must have 'date' and 'target' columns")

        df = df.sort_values("date").reset_index(drop=True)
        dates = df["date"]
        min_date = dates.min()
        max_date = dates.max()

        logger.info(
            "Starting backtest: %s to %s (%d matches)",
            min_date.strftime("%Y-%m-%d"),
            max_date.strftime("%Y-%m-%d"),
            len(df),
        )

        # Determine the drop columns for feature preparation
        drop_cols = settings.get("features", {}).get("drop_columns", [])
        meta_cols = [c for c in drop_cols if c in df.columns]
        meta_cols.extend(["target", "result"])
        meta_cols = list(set(c for c in meta_cols if c in df.columns))

        # Walk forward
        window = timedelta(days=self._window_days)
        current_train_end = min_date + timedelta(
            days=max(self._window_days * 2, 180)
        )

        period_num = 0
        all_y_true = []
        all_y_proba = []
        all_y_pred = []

        while current_train_end < max_date:
            test_end = current_train_end + window

            # Split data
            train_mask = dates < current_train_end
            test_mask = (dates >= current_train_end) & (dates < test_end)

            train_df = df[train_mask]
            test_df = df[test_mask]

            if len(train_df) < self._min_train_samples:
                logger.debug(
                    "Skipping period %d: insufficient training data (%d < %d)",
                    period_num, len(train_df), self._min_train_samples,
                )
                current_train_end += window
                continue

            if test_df.empty:
                current_train_end += window
                continue

            logger.info(
                "Period %d: train=%d matches (to %s), test=%d matches (to %s)",
                period_num, len(train_df),
                current_train_end.strftime("%Y-%m-%d"),
                len(test_df),
                test_end.strftime("%Y-%m-%d"),
            )

            # Prepare features
            feature_cols = [c for c in df.columns if c not in meta_cols
                           and df[c].dtype in ["float64", "float32", "int64", "int32"]]
            if "target" in feature_cols:
                feature_cols.remove("target")

            X_train = train_df[feature_cols].fillna(0)
            y_train = train_df["target"]
            X_test = test_df[feature_cols].fillna(0)
            y_test = test_df["target"]

            # Train model for this period
            try:
                period_result = self._run_period(
                    X_train, y_train, X_test, y_test,
                    test_df, period_num, odds_df,
                )
                self.period_results.append(period_result)

                # Collect for aggregate metrics
                all_y_true.extend(y_test.tolist())
                if "y_proba" in period_result:
                    all_y_proba.extend(period_result["y_proba"])
                if "y_pred" in period_result:
                    all_y_pred.extend(period_result["y_pred"])

            except Exception as exc:
                logger.error("Error in period %d: %s", period_num, exc)

            current_train_end += window
            period_num += 1

        # Compile final results
        results = self._compile_results(
            np.array(all_y_true) if all_y_true else np.array([]),
            np.array(all_y_proba) if all_y_proba else np.array([]),
            np.array(all_y_pred) if all_y_pred else np.array([]),
        )

        logger.info("Backtest complete: %d periods, final bankroll=%.2f",
                     period_num, self._bankroll.current_bankroll)
        return results

    # ------------------------------------------------------------------
    # Single Period
    # ------------------------------------------------------------------

    def _run_period(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        test_df: pd.DataFrame,
        period_num: int,
        odds_df: Optional[pd.DataFrame],
    ) -> Dict[str, Any]:
        """
        Execute a single walk-forward period.

        Returns per-period metrics and predictions.
        """
        # Train models
        trainer = ModelTrainer(sport=self.sport)
        trainer.train(X_train, y_train)

        # Get predictions
        y_proba_list = []
        for name, model in trainer.trained_models.items():
            try:
                proba = model.predict_proba(X_test)
                y_proba_list.append(proba)
            except Exception as exc:
                logger.error("Prediction failed for %s: %s", name, exc)

        if not y_proba_list:
            return {"period": period_num, "error": "No predictions generated"}

        # Average probabilities
        y_proba = np.mean(y_proba_list, axis=0)
        y_pred = np.argmax(y_proba, axis=1)

        # Metrics for this period
        acc = float(np.mean(y_pred == y_test.values))
        try:
            ll = float(log_loss(y_test, y_proba, labels=sorted(y_train.unique())))
        except Exception:
            ll = float("inf")

        logger.info("Period %d: accuracy=%.4f, log_loss=%.4f", period_num, acc, ll)

        # Simulate bets (always run; real odds_df preferred over synthetic)
        n_bets = 0
        period_pnl = 0.0

        n_bets, period_pnl = self._simulate_bets(
            test_df, y_proba, odds_df,
        )

        self.bankroll_history.append(self._bankroll.current_bankroll)

        return {
            "period": period_num,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "accuracy": acc,
            "log_loss": ll,
            "n_bets": n_bets,
            "period_pnl": period_pnl,
            "bankroll": self._bankroll.current_bankroll,
            "y_proba": y_proba.tolist(),
            "y_pred": y_pred.tolist(),
        }

    # ------------------------------------------------------------------
    # Bet Simulation
    # ------------------------------------------------------------------

    def _simulate_bets(
        self,
        test_df: pd.DataFrame,
        y_proba: np.ndarray,
        odds_df: Optional[pd.DataFrame],
    ) -> Tuple[int, float]:
        """
        Simulate betting on test period matches using value detection.

        Market odds are derived from historical class frequencies (base rates)
        plus a realistic vig — NOT from model probabilities. This prevents
        circular reasoning where the model's own uncertainty creates fake edge.

        If real odds_df is provided, those are used directly (preferred).

        Market model:
            base_rate[cls] = empirical frequency of each outcome in test set
            vig = 1.05  (5% bookmaker margin — typical for US sports ML)
            market_odds[cls] = (1 / base_rate[cls]) / vig

        Returns (n_bets_placed, total_pnl).
        """
        n_bets = 0
        total_pnl = 0.0
        n_classes = y_proba.shape[1]

        # Build real odds lookup from odds_df if provided
        real_odds_lookup: dict = {}
        if odds_df is not None and not odds_df.empty:
            for _, orow in odds_df.iterrows():
                mid = str(orow.get("match_id", ""))
                if mid:
                    real_odds_lookup[mid] = orow

        # Compute base rates from test set (historical frequencies per class)
        # This gives a market-neutral reference price independent of model output
        targets = test_df["target"].dropna()
        base_rates: dict = {}
        for cls in range(n_classes):
            freq = (targets == cls).mean()
            base_rates[cls] = max(freq, 0.05)  # floor at 5%

        # VIG: realistic bookmaker margin per sport
        _VIG = {
            "soccer":     1.07,   # 3-way market, higher vig
            "basketball": 1.045,
            "nhl":        1.05,
            "mlb":        1.05,
            "tennis":     1.06,
        }
        vig = _VIG.get(self.sport, 1.055)

        for i, (idx, row) in enumerate(test_df.iterrows()):
            proba = y_proba[i]
            result = row.get("target")
            match_date = row.get("date")
            match_id   = str(row.get("match_id", idx))

            for cls_idx in range(n_classes):
                model_p = proba[cls_idx]

                # Determine market odds
                if match_id in real_odds_lookup:
                    # Real historical odds from odds_df
                    orow = real_odds_lookup[match_id]
                    market_odds = float(orow.get(f"odds_class_{cls_idx}", 0) or 0)
                    if market_odds <= 1.0:
                        # Fall through to synthetic
                        market_odds = (1.0 / base_rates[cls_idx]) / vig
                else:
                    # Synthetic market price: base rate + vig (independent of model)
                    market_odds = (1.0 / base_rates[cls_idx]) / vig

                # Apply slippage (execution friction)
                market_odds *= (1 - self._slippage_pct)

                if market_odds <= 1.01:
                    continue

                edge = self._kelly.edge(model_p, market_odds)
                if edge < self._kelly.min_edge:
                    continue

                kelly_frac = self._kelly.calculate(model_p, market_odds)
                if kelly_frac <= 0:
                    continue

                stake = self._bankroll.calculate_stake(kelly_frac)
                if stake < 0.01:
                    continue

                bet = Bet(
                    bet_id=str(uuid.uuid4())[:8],
                    sport=self.sport,
                    match_id=match_id,
                    outcome=f"class_{cls_idx}",
                    stake=stake,
                    odds=market_odds,
                    model_prob=model_p,
                    edge=edge,
                    kelly_fraction=kelly_frac,
                    timestamp=match_date if isinstance(match_date, datetime) else datetime.now(),
                )

                if self._bankroll.place_bet(bet):
                    won = (result == cls_idx)
                    pnl = self._bankroll.settle_bet(
                        bet.bet_id, "won" if won else "lost"
                    )
                    if pnl is not None:
                        total_pnl += pnl
                    n_bets += 1

        return n_bets, total_pnl

    # ------------------------------------------------------------------
    # Results Compilation
    # ------------------------------------------------------------------

    def _compile_results(
        self,
        all_y_true: np.ndarray,
        all_y_proba: np.ndarray,
        all_y_pred: np.ndarray,
    ) -> Dict[str, Any]:
        """Compile all period results into a final summary."""
        results: Dict[str, Any] = {
            "n_periods": len(self.period_results),
            "bankroll_history": self.bankroll_history,
            "final_bankroll": self._bankroll.current_bankroll,
            "bankroll_stats": self._bankroll.get_stats(),
            "period_details": self.period_results,
        }

        if len(all_y_true) > 0 and len(all_y_pred) > 0:
            results["overall_accuracy"] = float(np.mean(all_y_pred == all_y_true))

        if len(all_y_true) > 0 and len(all_y_proba) > 0:
            try:
                proba_array = np.array(all_y_proba)
                if proba_array.ndim == 2:
                    results["overall_brier"] = MetricsCalculator.brier_score(
                        all_y_true, proba_array
                    )
                    results["overall_ece"] = MetricsCalculator.expected_calibration_error(
                        all_y_true, proba_array
                    )
            except Exception as exc:
                logger.error("Error computing aggregate metrics: %s", exc)

        # Max drawdown
        if self.bankroll_history:
            dd, peak_idx, trough_idx = MetricsCalculator.max_drawdown(
                self.bankroll_history
            )
            results["max_drawdown"] = dd
            results["max_drawdown_peak_idx"] = peak_idx
            results["max_drawdown_trough_idx"] = trough_idx

        return results

    def get_summary_df(self) -> pd.DataFrame:
        """Return period-level summary as a DataFrame."""
        if not self.period_results:
            return pd.DataFrame()

        records = []
        for pr in self.period_results:
            records.append({
                "period": pr.get("period"),
                "train_size": pr.get("train_size"),
                "test_size": pr.get("test_size"),
                "accuracy": pr.get("accuracy"),
                "log_loss": pr.get("log_loss"),
                "n_bets": pr.get("n_bets"),
                "period_pnl": pr.get("period_pnl"),
                "bankroll": pr.get("bankroll"),
            })
        return pd.DataFrame(records)
