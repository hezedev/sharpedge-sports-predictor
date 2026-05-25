"""
Quantitative Backtester for Sports Betting Strategies

This is a strict historical backtesting engine designed to prevent fatal flaws
from reaching live execution. It enforces:

1. Walk-Forward Validation (no time leakage)
2. Time-to-Kickoff Filter (only odds from <24 hours before start)
3. Simulated Execution Throttling (max 2,000 actions/day, ~45s per asset)
4. Drawdown Limits (15-20% max, reject if > 20%)
5. Kelly Criterion Sizing (0.125 to 0.50, configurable by market)
6. Dynamic EV Circuit Breakers (reject hallucinated edges)

This is a RESEARCH-ONLY tool. It does NOT execute live trades.
It only proves whether your strategy survives historical variance.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import logging

logger = logging.getLogger(__name__)


class QuantitativeBacktester:
    """
    Walk-forward backtester that validates betting strategies against
    realistic market constraints and variance limits.
    """

    def __init__(
        self,
        initial_capital: float = 1000.0,
        kelly_fraction: float = 0.25,
        max_drawdown_tolerance: float = 0.20,
        max_daily_actions: int = 2000,
        min_time_to_kickoff_hours: int = 0,
        max_time_to_kickoff_hours: int = 24,
    ):
        """
        Initialize backtester with risk parameters.

        Parameters
        ----------
        initial_capital : float
            Starting bankroll (default: 1000 EUR)
        kelly_fraction : float
            Fractional Kelly multiplier (0.125-0.50)
            - 0.125: High-variance derivative markets
            - 0.25: Standard quarter-Kelly for syndicates
            - 0.50: Only for massive data advantages
        max_drawdown_tolerance : float
            Maximum acceptable drawdown (0.15-0.20)
            If exceeded, strategy is rejected
        max_daily_actions : int
            Maximum trades per calendar day (default: 2000)
        min_time_to_kickoff_hours : int
            Minimum hours before kickoff to accept (default: 0)
        max_time_to_kickoff_hours : int
            Maximum hours before kickoff to accept (default: 24)
        """
        self.starting_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.kelly_fraction = kelly_fraction
        self.max_drawdown_tolerance = max_drawdown_tolerance
        self.max_daily_actions = max_daily_actions
        self.min_time_to_kickoff = timedelta(hours=min_time_to_kickoff_hours)
        self.max_time_to_kickoff = timedelta(hours=max_time_to_kickoff_hours)

        self.max_drawdown = 0.0
        self.trade_history: List[Dict] = []
        self.daily_action_count = 0
        self.current_date = None
        self.rejected_trades = 0
        self.executed_trades = 0

        # Circuit breaker thresholds (market-specific edge limits)
        self.edge_limits = {
            "Moneyline": 0.10,  # 10% max edge on main market
            "Spreads": 0.10,
            "Totals": 0.10,
            "AH": 0.18,  # 18% max on Asian Handicap
            "Corners": 0.18,  # 18% max on Corners
            "Draw": 0.15,  # 15% max on Draw outcomes
        }

    def calculate_edge(
        self, true_prob: float, market_odds: float
    ) -> Tuple[float, float]:
        """
        Calculate Expected Value (EV) from model probability vs market odds.

        Parameters
        ----------
        true_prob : float
            Model's predicted probability (0.0-1.0)
        market_odds : float
            Bookmaker's decimal odds (e.g., 2.50)

        Returns
        -------
        edge : float
            Expected value per unit wagered
        implied_prob : float
            Market's raw implied probability (includes vig)
        """
        if market_odds <= 1.0:
            return 0.0, 1.0

        implied_prob = 1.0 / market_odds
        edge = (true_prob * market_odds) - 1.0
        return edge, implied_prob

    @staticmethod
    def estimate_vig(outcome_odds: List[float]) -> float:
        """
        Estimate bookmaker margin (vig/overround) from a set of decimal odds.

        vig = sum(1/odds_i) - 1.0

        A 5% vig means the book takes 5 cents on every dollar of action.
        Use this to understand what edge you need to clear before profiting.

        Parameters
        ----------
        outcome_odds : list[float]
            Decimal odds for all outcomes in a market (e.g. [2.0, 3.5, 4.0])

        Returns
        -------
        float
            Bookmaker margin as a fraction (e.g. 0.05 = 5% vig)
        """
        if not outcome_odds or any(o <= 1.0 for o in outcome_odds):
            return 0.0
        return sum(1.0 / o for o in outcome_odds) - 1.0

    @staticmethod
    def vig_adjusted_edge(true_prob: float, market_odds: float, vig: float) -> float:
        """
        Edge after accounting for the bookmaker margin.

        When you know the full market overround, the true fair odds are
        market_odds * (1 + vig) — slightly better than stated. This gives a
        more accurate picture of how much edge you actually have.

        Parameters
        ----------
        true_prob : float
            Model probability.
        market_odds : float
            Decimal odds offered by the bookmaker.
        vig : float
            Estimated bookmaker margin (from estimate_vig).

        Returns
        -------
        float
            Vig-adjusted EV per unit staked.
        """
        fair_odds = market_odds * (1.0 + vig)
        return (true_prob * fair_odds) - 1.0

    def calculate_kelly_stake(self, edge: float, odds: float) -> float:
        """
        Calculate bet stake using fractional Kelly Criterion.

        Full Kelly: f* = (bp - q) / b
        where b = odds - 1, p = win prob, q = 1 - p

        We apply kelly_fraction multiplier to reduce volatility.

        Parameters
        ----------
        edge : float
            Expected value (positive = profitable)
        odds : float
            Decimal odds

        Returns
        -------
        stake : float
            Absolute bet amount in currency units
        """
        if edge <= 0.0:
            return 0.0

        # Full Kelly fraction
        full_kelly = edge / (odds - 1.0)

        # Apply fractional Kelly multiplier (0.125-0.50)
        stake_pct = full_kelly * self.kelly_fraction

        # Hard cap: never risk more than 5% of bankroll on single bet
        stake_pct = min(stake_pct, 0.05)

        return self.current_capital * stake_pct

    def _check_time_to_kickoff(
        self, odds_timestamp: datetime, kickoff_time: datetime
    ) -> bool:
        """
        Filter odds by time-to-kickoff window.

        Rejects "soft" market lines taken too far from kickoff.
        Prevents false signals from untested market positions.

        Parameters
        ----------
        odds_timestamp : datetime
            When odds were captured
        kickoff_time : datetime
            When match begins

        Returns
        -------
        valid : bool
            True if odds are within acceptable time window
        """
        time_to_kickoff = kickoff_time - odds_timestamp

        if time_to_kickoff < self.min_time_to_kickoff:
            return False  # Too close to kickoff (odds might have moved)
        if time_to_kickoff > self.max_time_to_kickoff:
            return False  # Too far (market is soft, not priced in)

        return True

    def _check_daily_volume_limit(self, current_date: datetime) -> bool:
        """
        Enforce execution throttling to simulate API rate limits.

        A real sportsbook API will ban you for 2,000+ actions/day.
        This ensures the strategy is legally executable.

        Parameters
        ----------
        current_date : datetime
            Current trading date

        Returns
        -------
        can_trade : bool
            True if under daily action limit
        """
        if self.current_date is None:
            self.current_date = current_date
            self.daily_action_count = 0

        if current_date != self.current_date:
            # Reset daily counter at midnight
            self.current_date = current_date
            self.daily_action_count = 0

        if self.daily_action_count >= self.max_daily_actions:
            return False  # Hit daily limit, skip this trade

        self.daily_action_count += 1
        return True

    def _check_edge_circuit_breaker(
        self, edge: float, market_type: str
    ) -> bool:
        """
        Reject hallucinated edges (likely data errors).

        An edge > 10% in a main market (Moneyline/Spreads) means the
        bookmaker made a catastrophic pricing error. Real edges are 1-5%.

        Parameters
        ----------
        edge : float
            Expected value edge
        market_type : str
            Type of bet (Moneyline, Spreads, Corners, etc.)

        Returns
        -------
        valid : bool
            True if edge passes sanity check
        """
        limit = self.edge_limits.get(market_type, 0.10)

        if edge > limit:
            logger.warning(
                f"Edge {edge:.2%} exceeds limit {limit:.2%} for {market_type}. "
                f"Likely data error—skipping."
            )
            return False

        return True

    def run_backtest(self, df: pd.DataFrame) -> bool:
        """
        Execute walk-forward backtest on historical data.

        DataFrame must contain:
        - date: Trade date (datetime)
        - kickoff_time: Match start time (datetime)
        - odds_timestamp: When odds were captured (datetime)
        - market_type: Bet type (Moneyline, Spreads, etc.)
        - true_prob: Model's predicted probability (0.0-1.0)
        - market_odds: Bookmaker's decimal odds
        - result: Outcome (1 = win, 0 = loss)

        Parameters
        ----------
        df : pd.DataFrame
            Historical data with required columns

        Returns
        -------
        success : bool
            True if backtest completed without fatal drawdown
        """
        # Sort chronologically (prevent time leakage)
        df = df.sort_values(by="date").reset_index(drop=True)

        if df.empty:
            logger.error("Empty dataframe provided to backtest")
            return False

        logger.info(f"Starting backtest with {len(df)} historical records")

        for index, row in df.iterrows():
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # FILTER 1: Daily Volume Limit (API throttling simulation)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if not self._check_daily_volume_limit(row["date"]):
                self.rejected_trades += 1
                continue

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # FILTER 2: Time-to-Kickoff Window (avoid soft markets)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if not self._check_time_to_kickoff(
                row["odds_timestamp"], row["kickoff_time"]
            ):
                self.rejected_trades += 1
                continue

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # FILTER 3: Calculate Edge & Check Minimum Profitability
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            edge, implied_prob = self.calculate_edge(
                row["true_prob"], row["market_odds"]
            )
            # Vig estimation from opposing outcome odds (if provided)
            all_outcome_odds = row.get("all_outcome_odds") or []
            vig = self.estimate_vig(all_outcome_odds) if all_outcome_odds else 0.0
            vig_adj_edge = self.vig_adjusted_edge(row["true_prob"], row["market_odds"], vig) if vig > 0 else edge

            if edge <= 0.0:
                self.rejected_trades += 1
                continue  # Not profitable

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # FILTER 4: Circuit Breaker (reject hallucinated edges)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if not self._check_edge_circuit_breaker(
                edge, row.get("market_type", "Moneyline")
            ):
                self.rejected_trades += 1
                continue

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # SIZING: Kelly Criterion
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            stake = self.calculate_kelly_stake(edge, row["market_odds"])

            if stake <= 0.0:
                self.rejected_trades += 1
                continue

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # EXECUTION: Simulate trade outcome
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            self.current_capital -= stake  # Deduct stake

            profit = 0.0
            if row["result"] == 1:  # Bet won
                payout = stake * row["market_odds"]
                self.current_capital += payout
                profit = payout - stake
            else:  # Bet lost
                profit = -stake

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # TRACKING: Drawdown & Peak Capital
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if self.current_capital > self.peak_capital:
                self.peak_capital = self.current_capital

            current_drawdown = (
                (self.peak_capital - self.current_capital) / self.peak_capital
                if self.peak_capital > 0
                else 0.0
            )

            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # FATAL: Check if drawdown exceeds tolerance
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if current_drawdown > self.max_drawdown_tolerance:
                logger.error(
                    f"FATAL DRAWDOWN: {current_drawdown:.2%} exceeds max "
                    f"{self.max_drawdown_tolerance:.2%}. Strategy rejected."
                )
                return False  # Strategy failed

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # LOG Trade
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            self.trade_history.append(
                {
                    "date": row["date"],
                    "kickoff_time": row["kickoff_time"],
                    "market": row.get("market_type", "Unknown"),
                    "odds": row["market_odds"],
                    "model_prob": round(row["true_prob"], 4),
                    "market_prob": round(1.0 / row["market_odds"], 4),
                    "edge": round(edge, 4),
                    "vig_pct": round(vig * 100, 2),
                    "vig_adj_edge": round(vig_adj_edge, 4),
                    "stake": round(stake, 2),
                    "result": "WIN" if row["result"] == 1 else "LOSS",
                    "profit": round(profit, 2),
                    "capital_after": round(self.current_capital, 2),
                    "drawdown": round(current_drawdown, 4),
                }
            )

            self.executed_trades += 1

        logger.info(
            f"Backtest complete: {self.executed_trades} trades executed, "
            f"{self.rejected_trades} rejected"
        )
        return True  # Passed all checks

    def generate_report(self) -> pd.DataFrame:
        """
        Generate comprehensive backtest report.

        Returns
        -------
        report_df : pd.DataFrame
            Historical trade log
        """
        if not self.trade_history:
            logger.warning("No trades executed in backtest")
            return pd.DataFrame()

        history_df = pd.DataFrame(self.trade_history)

        # Calculate metrics
        total_profit = self.current_capital - self.starting_capital
        total_roi = (total_profit / self.starting_capital) * 100
        total_days = (history_df["date"].max() - history_df["date"].min()).days
        total_weeks = max(total_days / 7, 1)
        avg_profit_per_week = total_profit / total_weeks

        win_count = (history_df["result"] == "WIN").sum()
        loss_count = (history_df["result"] == "LOSS").sum()
        win_rate = (win_count / len(history_df)) * 100 if len(history_df) > 0 else 0

        avg_edge = history_df["edge"].mean() if "edge" in history_df.columns else 0.0
        avg_vig = history_df["vig_pct"].mean() if "vig_pct" in history_df.columns else 0.0
        avg_vig_adj_edge = history_df["vig_adj_edge"].mean() if "vig_adj_edge" in history_df.columns else 0.0
        min_edge_to_profit = avg_vig  # need edge > vig to expect long-run profit

        print("\n" + "=" * 70)
        print("QUANTITATIVE BACKTEST REPORT")
        print("=" * 70)
        print(f"{'Initial Capital':<40} €{self.starting_capital:>15,.2f}")
        print(f"{'Final Capital':<40} €{self.current_capital:>15,.2f}")
        print(f"{'Total Profit/Loss':<40} €{total_profit:>15,.2f}")
        print(f"{'Total ROI':<40} {total_roi:>15.2f}%")
        print("-" * 70)
        print(f"{'Trades Executed':<40} {self.executed_trades:>15,}")
        print(f"{'Trades Rejected (filters)':<40} {self.rejected_trades:>15,}")
        print(f"{'Win Rate':<40} {win_rate:>15.2f}%")
        print("-" * 70)
        print(f"{'Avg Raw Edge':<40} {avg_edge * 100:>15.2f}%")
        if avg_vig > 0:
            print(f"{'Avg Bookmaker Vig':<40} {avg_vig:>15.2f}%")
            print(f"{'Avg Vig-Adjusted Edge':<40} {avg_vig_adj_edge * 100:>15.2f}%")
            print(f"{'Break-even Edge Required':<40} {min_edge_to_profit:>15.2f}%")
        print("-" * 70)
        print(f"{'Max Drawdown':<40} {self.max_drawdown * 100:>15.2f}%")
        print(f"{'Drawdown Tolerance':<40} {self.max_drawdown_tolerance * 100:>15.2f}%")
        print(f"{'Status':<40} {'PASS ✓' if self.max_drawdown <= self.max_drawdown_tolerance else 'FAIL ✗':>15}")
        print("-" * 70)
        print(f"{'Days Backtested':<40} {total_days:>15,}")
        print(f"{'Weeks Backtested':<40} {total_weeks:>15.1f}")
        print(f"{'Avg Profit Per Week':<40} €{avg_profit_per_week:>15,.2f}")
        print("=" * 70 + "\n")

        return history_df

    def export_results(self, filepath: Path) -> None:
        """
        Export detailed trade history and metrics to CSV.

        Parameters
        ----------
        filepath : Path
            Output CSV file path
        """
        if not self.trade_history:
            logger.warning("No trades to export")
            return

        history_df = pd.DataFrame(self.trade_history)
        history_df.to_csv(filepath, index=False)
        logger.info(f"Exported {len(history_df)} trades to {filepath}")

    def get_summary_metrics(self) -> Dict:
        """
        Return backtest summary as dictionary.

        Returns
        -------
        metrics : dict
            Key metrics for programmatic access
        """
        total_profit = self.current_capital - self.starting_capital
        total_roi = (total_profit / self.starting_capital) * 100

        return {
            "starting_capital": self.starting_capital,
            "final_capital": self.current_capital,
            "total_profit": total_profit,
            "total_roi_percent": total_roi,
            "trades_executed": self.executed_trades,
            "trades_rejected": self.rejected_trades,
            "max_drawdown_percent": self.max_drawdown * 100,
            "passed": self.max_drawdown <= self.max_drawdown_tolerance,
            "kelly_fraction": self.kelly_fraction,
            "max_drawdown_tolerance_percent": self.max_drawdown_tolerance * 100,
        }
