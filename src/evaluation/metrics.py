"""
Performance metrics for model evaluation and betting performance.

Provides both ML-centric metrics (Brier score, log loss, calibration)
and betting-centric metrics (ROI, CLV, yield, Sharpe).
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    classification_report,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Comprehensive metrics for evaluating prediction models and
    betting strategies.
    """

    # ------------------------------------------------------------------
    # ML Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def brier_score(
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> float:
        """
        Multi-class Brier score (lower is better).

        BS = (1/N) * sum_i sum_c (p_ic - o_ic)^2

        Parameters
        ----------
        y_true : np.ndarray
            True labels (integer encoded).
        y_proba : np.ndarray
            Predicted probabilities, shape (n_samples, n_classes).

        Returns
        -------
        float
            Brier score.
        """
        n_classes = y_proba.shape[1]
        # One-hot encode y_true
        y_onehot = np.zeros_like(y_proba)
        for i, label in enumerate(y_true):
            if 0 <= label < n_classes:
                y_onehot[i, int(label)] = 1.0

        bs = np.mean(np.sum((y_proba - y_onehot) ** 2, axis=1))
        return float(bs)

    @staticmethod
    def brier_skill_score(
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> float:
        """
        Brier Skill Score: improvement over naive baseline (uniform probs).

        BSS = 1 - BS_model / BS_baseline

        Parameters
        ----------
        y_true : np.ndarray
            True labels.
        y_proba : np.ndarray
            Predicted probabilities.

        Returns
        -------
        float
            Brier Skill Score. 1.0 = perfect, 0.0 = no skill, <0 = worse than random.
        """
        n_classes = y_proba.shape[1]
        n_samples = len(y_true)

        # Model Brier score
        bs_model = MetricsCalculator.brier_score(y_true, y_proba)

        # Baseline: uniform probabilities
        baseline_proba = np.full((n_samples, n_classes), 1.0 / n_classes)
        bs_baseline = MetricsCalculator.brier_score(y_true, baseline_proba)

        if bs_baseline == 0:
            return 0.0

        return 1.0 - (bs_model / bs_baseline)

    @staticmethod
    def log_loss_score(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        labels: Optional[List[int]] = None,
    ) -> float:
        """Compute log loss (cross-entropy)."""
        return float(log_loss(y_true, y_proba, labels=labels))

    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Simple accuracy score."""
        return float(accuracy_score(y_true, y_pred))

    @staticmethod
    def classification_summary(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        label_names: Optional[List[str]] = None,
    ) -> str:
        """Full classification report as a string."""
        return classification_report(
            y_true, y_pred, target_names=label_names, zero_division=0,
        )

    @staticmethod
    def confusion_matrix_df(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        label_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Return confusion matrix as a labeled DataFrame."""
        cm = confusion_matrix(y_true, y_pred)
        if label_names:
            return pd.DataFrame(cm, index=label_names, columns=label_names)
        return pd.DataFrame(cm)

    # ------------------------------------------------------------------
    # Calibration Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def expected_calibration_error(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Expected Calibration Error (ECE).

        Measures the gap between predicted confidence and actual accuracy
        across probability bins.

        Parameters
        ----------
        y_true : np.ndarray
            True labels.
        y_proba : np.ndarray
            Predicted probabilities (can be multi-class; uses max prob).
        n_bins : int
            Number of bins.

        Returns
        -------
        float
            ECE score (lower is better).
        """
        if y_proba.ndim == 2:
            confidences = np.max(y_proba, axis=1)
            predictions = np.argmax(y_proba, axis=1)
        else:
            confidences = y_proba
            predictions = (y_proba > 0.5).astype(int)

        accuracies = (predictions == y_true).astype(float)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
            mask = (confidences > lower) & (confidences <= upper)
            n_in_bin = mask.sum()

            if n_in_bin == 0:
                continue

            avg_confidence = confidences[mask].mean()
            avg_accuracy = accuracies[mask].mean()

            ece += (n_in_bin / len(y_true)) * abs(avg_accuracy - avg_confidence)

        return float(ece)

    # ------------------------------------------------------------------
    # Betting Performance Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def roi(total_pnl: float, total_staked: float) -> float:
        """
        Return on Investment.

        ROI = total_pnl / total_staked
        """
        if total_staked <= 0:
            return 0.0
        return total_pnl / total_staked

    @staticmethod
    def yield_pct(total_pnl: float, n_bets: int, avg_stake: float) -> float:
        """
        Yield (profit per unit staked across all bets).
        """
        total_staked = n_bets * avg_stake
        if total_staked <= 0:
            return 0.0
        return total_pnl / total_staked

    @staticmethod
    def closing_line_value(
        opening_odds: List[float],
        closing_odds: List[float],
    ) -> float:
        """
        Closing Line Value (CLV): measures whether you consistently
        beat the closing line (a strong indicator of long-term profitability).

        CLV = mean(closing_implied - opening_implied)

        Parameters
        ----------
        opening_odds : list[float]
            Odds at which bets were placed.
        closing_odds : list[float]
            Odds at market close (just before the event).

        Returns
        -------
        float
            Average CLV (positive = beating the closing line).
        """
        if not opening_odds or len(opening_odds) != len(closing_odds):
            return 0.0

        clvs = []
        for opening, closing in zip(opening_odds, closing_odds):
            if opening <= 0 or closing <= 0:
                continue
            opening_implied = 1.0 / opening
            closing_implied = 1.0 / closing
            clvs.append(closing_implied - opening_implied)

        return float(np.mean(clvs)) if clvs else 0.0

    @staticmethod
    def sharpe_ratio(
        returns: List[float],
        risk_free_rate: float = 0.0,
    ) -> float:
        """
        Sharpe ratio of betting returns.

        Parameters
        ----------
        returns : list[float]
            Per-bet returns (pnl / stake).
        risk_free_rate : float
            Risk-free rate per period.

        Returns
        -------
        float
            Sharpe ratio (higher is better).
        """
        if not returns or len(returns) < 2:
            return 0.0

        arr = np.array(returns)
        excess = arr - risk_free_rate
        std = np.std(excess, ddof=1)

        if std == 0:
            return 0.0

        return float(np.mean(excess) / std)

    @staticmethod
    def max_drawdown(bankroll_history: List[float]) -> Tuple[float, int, int]:
        """
        Calculate maximum drawdown from a bankroll history.

        Parameters
        ----------
        bankroll_history : list[float]
            Sequential bankroll values.

        Returns
        -------
        tuple[float, int, int]
            (max_drawdown_pct, peak_index, trough_index)
        """
        if not bankroll_history:
            return 0.0, 0, 0

        arr = np.array(bankroll_history)
        peak = arr[0]
        peak_idx = 0
        max_dd = 0.0
        max_dd_peak = 0
        max_dd_trough = 0

        for i in range(1, len(arr)):
            if arr[i] > peak:
                peak = arr[i]
                peak_idx = i
            else:
                dd = (peak - arr[i]) / peak if peak > 0 else 0.0
                if dd > max_dd:
                    max_dd = dd
                    max_dd_peak = peak_idx
                    max_dd_trough = i

        return float(max_dd), max_dd_peak, max_dd_trough

    # ------------------------------------------------------------------
    # Comprehensive Report
    # ------------------------------------------------------------------

    @classmethod
    def full_report(
        cls,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        y_pred: np.ndarray,
        label_names: Optional[List[str]] = None,
        bet_history: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """
        Generate a comprehensive evaluation report.

        Parameters
        ----------
        y_true : np.ndarray
            True labels.
        y_proba : np.ndarray
            Predicted probabilities.
        y_pred : np.ndarray
            Predicted class labels.
        label_names : list[str], optional
            Human-readable label names.
        bet_history : pd.DataFrame, optional
            Bet history with 'stake', 'pnl', 'odds' columns.

        Returns
        -------
        dict
            Comprehensive metrics dictionary.
        """
        report = {
            "accuracy": cls.accuracy(y_true, y_pred),
            "log_loss": cls.log_loss_score(y_true, y_proba),
            "brier_score": cls.brier_score(y_true, y_proba),
            "brier_skill_score": cls.brier_skill_score(y_true, y_proba),
            "ece": cls.expected_calibration_error(y_true, y_proba),
            "n_samples": len(y_true),
        }

        if bet_history is not None and not bet_history.empty:
            settled = bet_history.dropna(subset=["pnl"])
            if not settled.empty:
                total_pnl = settled["pnl"].sum()
                total_staked = settled["stake"].sum()
                returns = (settled["pnl"] / settled["stake"]).tolist()

                report["total_pnl"] = total_pnl
                report["total_staked"] = total_staked
                report["roi"] = cls.roi(total_pnl, total_staked)
                report["n_bets"] = len(settled)
                report["win_rate"] = (settled["pnl"] > 0).mean()
                report["sharpe"] = cls.sharpe_ratio(returns)

        logger.info("Evaluation report: %s", report)
        return report
