"""
Bankroll management with drawdown protection and daily loss limits.

Tracks all bets, enforces position limits, and provides
circuit-breakers to protect capital during losing streaks.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timezone
from typing import Dict, List, Optional

import pandas as pd

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class Bet:
    """Record of a single placed bet."""
    bet_id: str
    sport: str
    match_id: str
    outcome: str
    stake: float
    odds: float
    model_prob: float
    edge: float
    kelly_fraction: float
    timestamp: datetime
    result: Optional[str] = None       # 'won' | 'lost' | 'void' | None (pending)
    pnl: Optional[float] = None        # profit/loss after settlement


class BankrollManager:
    """
    Manages bankroll, enforces betting limits, and tracks performance.

    Provides:
        - Drawdown circuit-breaker (pause betting at X% drawdown)
        - Daily loss limit
        - Maximum concurrent bet count
        - Bet history and P&L tracking
        - Session-level and lifetime statistics

    Parameters
    ----------
    initial_bankroll : float, optional
        Starting bankroll. Defaults to config value.
    """

    def __init__(self, initial_bankroll: Optional[float] = None) -> None:
        br_cfg = settings.get("risk", {}).get("bankroll", {})

        self.initial_bankroll = initial_bankroll or br_cfg.get("initial", 1000.0)
        self.current_bankroll = self.initial_bankroll
        self.peak_bankroll = self.initial_bankroll

        self._drawdown_limit = br_cfg.get("drawdown_limit", 0.20)
        self._daily_loss_limit = br_cfg.get("daily_loss_limit", 0.05)
        self._max_concurrent = br_cfg.get("max_concurrent_bets", 10)

        self._bet_history: List[Bet] = []
        self._pending_bets: Dict[str, Bet] = {}
        self._is_paused = False
        self._pause_reason: Optional[str] = None

        logger.info(
            "BankrollManager: initial=%.2f, drawdown_limit=%.0f%%, "
            "daily_loss=%.0f%%, max_concurrent=%d",
            self.initial_bankroll,
            self._drawdown_limit * 100,
            self._daily_loss_limit * 100,
            self._max_concurrent,
        )

    # ------------------------------------------------------------------
    # Bet Placement
    # ------------------------------------------------------------------

    def can_place_bet(self, stake: float) -> tuple[bool, str]:
        """
        Check whether a new bet can be placed.

        Parameters
        ----------
        stake : float
            Absolute stake amount.

        Returns
        -------
        tuple[bool, str]
            (allowed, reason)
        """
        if self._is_paused:
            return False, f"Betting paused: {self._pause_reason}"

        if stake <= 0:
            return False, "Invalid stake amount"

        if stake > self.current_bankroll:
            return False, "Insufficient bankroll"

        if len(self._pending_bets) >= self._max_concurrent:
            return False, f"Max concurrent bets ({self._max_concurrent}) reached"

        # Check drawdown
        drawdown = self._current_drawdown()
        if drawdown >= self._drawdown_limit:
            self._pause("Drawdown limit reached (%.1f%%)" % (drawdown * 100))
            return False, self._pause_reason or "Drawdown limit"

        # Check daily loss
        daily_loss = self._daily_loss()
        if daily_loss >= self._daily_loss_limit * self.initial_bankroll:
            self._pause("Daily loss limit reached (%.2f)" % daily_loss)
            return False, self._pause_reason or "Daily loss limit"

        return True, "OK"

    def place_bet(self, bet: Bet) -> bool:
        """
        Place a bet and deduct stake from bankroll.

        Parameters
        ----------
        bet : Bet
            Bet record to place.

        Returns
        -------
        bool
            True if bet was placed successfully.
        """
        allowed, reason = self.can_place_bet(bet.stake)
        if not allowed:
            logger.warning("Bet rejected (%s): %s", bet.bet_id, reason)
            return False

        self.current_bankroll -= bet.stake
        self._pending_bets[bet.bet_id] = bet
        self._bet_history.append(bet)

        logger.info(
            "Bet placed: %s | %s %s @ %.2f | stake=%.2f | bankroll=%.2f",
            bet.bet_id, bet.sport, bet.outcome, bet.odds,
            bet.stake, self.current_bankroll,
        )
        return True

    # ------------------------------------------------------------------
    # Settlement
    # ------------------------------------------------------------------

    def settle_bet(self, bet_id: str, result: str) -> Optional[float]:
        """
        Settle a pending bet.

        Parameters
        ----------
        bet_id : str
            ID of the bet to settle.
        result : str
            'won', 'lost', or 'void'.

        Returns
        -------
        float or None
            Profit/loss amount, or None if bet not found.
        """
        if bet_id not in self._pending_bets:
            logger.warning("Bet not found for settlement: %s", bet_id)
            return None

        bet = self._pending_bets.pop(bet_id)
        bet.result = result

        if result == "won":
            payout = bet.stake * bet.odds
            profit = payout - bet.stake
            self.current_bankroll += payout
            bet.pnl = profit

        elif result == "lost":
            bet.pnl = -bet.stake

        elif result == "void":
            self.current_bankroll += bet.stake  # refund
            bet.pnl = 0.0

        else:
            logger.error("Unknown result '%s' for bet %s", result, bet_id)
            return None

        # Update peak
        if self.current_bankroll > self.peak_bankroll:
            self.peak_bankroll = self.current_bankroll

        # Update in history
        for i, b in enumerate(self._bet_history):
            if b.bet_id == bet_id:
                self._bet_history[i] = bet
                break

        logger.info(
            "Bet settled: %s | result=%s | pnl=%.2f | bankroll=%.2f",
            bet_id, result, bet.pnl, self.current_bankroll,
        )
        return bet.pnl

    # ------------------------------------------------------------------
    # Risk Checks
    # ------------------------------------------------------------------

    def _current_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        if self.peak_bankroll <= 0:
            return 0.0
        return (self.peak_bankroll - self.current_bankroll) / self.peak_bankroll

    def _daily_loss(self) -> float:
        """Calculate total losses for today (UTC date to avoid timezone drift)."""
        today_utc = datetime.now(tz=timezone.utc).date()
        daily_losses = sum(
            abs(b.pnl)
            for b in self._bet_history
            if b.pnl is not None
            and b.pnl < 0
            and (
                # Handle both tz-aware and tz-naive timestamps gracefully
                (b.timestamp.tzinfo is not None and b.timestamp.astimezone(timezone.utc).date() == today_utc)
                or (b.timestamp.tzinfo is None and b.timestamp.date() == today_utc)
            )
        )
        return daily_losses

    def _pause(self, reason: str) -> None:
        """Pause all betting with a reason."""
        self._is_paused = True
        self._pause_reason = reason
        logger.warning("BETTING PAUSED: %s", reason)

    def resume(self) -> None:
        """Resume betting (manual override)."""
        self._is_paused = False
        self._pause_reason = None
        logger.info("Betting resumed")

    # ------------------------------------------------------------------
    # Stake Calculation
    # ------------------------------------------------------------------

    def calculate_stake(self, kelly_fraction: float) -> float:
        """
        Convert a Kelly fraction to an absolute stake amount.

        Parameters
        ----------
        kelly_fraction : float
            Kelly-recommended fraction of bankroll.

        Returns
        -------
        float
            Absolute stake in currency units.
        """
        if kelly_fraction <= 0:
            return 0.0
        return round(kelly_fraction * self.current_bankroll, 2)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, float]:
        """
        Calculate comprehensive bankroll statistics.

        Returns
        -------
        dict
            Performance metrics.
        """
        settled = [b for b in self._bet_history if b.pnl is not None]

        if not settled:
            return {
                "initial_bankroll": self.initial_bankroll,
                "current_bankroll": self.current_bankroll,
                "total_bets": 0,
                "pending_bets": len(self._pending_bets),
            }

        wins = [b for b in settled if b.result == "won"]
        losses = [b for b in settled if b.result == "lost"]
        total_staked = sum(b.stake for b in settled)
        total_pnl = sum(b.pnl for b in settled)

        win_rate = len(wins) / len(settled) if settled else 0.0
        avg_win = (sum(b.pnl for b in wins) / len(wins)) if wins else 0.0
        avg_loss = (sum(abs(b.pnl) for b in losses) / len(losses)) if losses else 0.0

        return {
            "initial_bankroll": self.initial_bankroll,
            "current_bankroll": self.current_bankroll,
            "peak_bankroll": self.peak_bankroll,
            "current_drawdown": self._current_drawdown(),
            "total_bets": len(settled),
            "pending_bets": len(self._pending_bets),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": win_rate,
            "total_staked": total_staked,
            "total_pnl": total_pnl,
            "roi": total_pnl / total_staked if total_staked > 0 else 0.0,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": avg_win / avg_loss if avg_loss > 0 else float("inf"),
            "is_paused": self._is_paused,
            "pause_reason": self._pause_reason,
        }

    def get_history_df(self) -> pd.DataFrame:
        """Return bet history as a DataFrame."""
        if not self._bet_history:
            return pd.DataFrame()

        records = []
        for b in self._bet_history:
            records.append({
                "bet_id": b.bet_id,
                "sport": b.sport,
                "match_id": b.match_id,
                "outcome": b.outcome,
                "stake": b.stake,
                "odds": b.odds,
                "model_prob": b.model_prob,
                "edge": b.edge,
                "kelly_fraction": b.kelly_fraction,
                "timestamp": b.timestamp,
                "result": b.result,
                "pnl": b.pnl,
            })
        return pd.DataFrame(records)
