"""
Kelly Criterion bet sizing.

Implements full Kelly and fractional Kelly for optimal bankroll
growth while controlling variance.
"""

import logging
from typing import Optional

import numpy as np

from config import settings

logger = logging.getLogger(__name__)


class KellyCriterion:
    """
    Kelly Criterion calculator for optimal bet sizing.

    Supports:
        - Full Kelly (maximum growth rate, high variance)
        - Fractional Kelly (reduced variance at cost of growth)
        - Multi-outcome Kelly for 3-way markets (soccer 1X2)
        - Bet caps to prevent over-exposure

    Parameters
    ----------
    fraction : float
        Kelly fraction (0.25 = quarter-Kelly). Range (0, 1].
    max_bet_pct : float
        Maximum bet size as fraction of bankroll.
    min_edge : float
        Minimum edge (model_prob - implied_prob) required to bet.
    """

    def __init__(
        self,
        fraction: Optional[float] = None,
        max_bet_pct: Optional[float] = None,
        min_edge: Optional[float] = None,
    ) -> None:
        risk_cfg = settings.get("risk", {}).get("kelly", {})

        self.fraction = fraction or risk_cfg.get("fraction", 0.25)
        self.max_bet_pct = max_bet_pct or risk_cfg.get("max_bet_pct", 0.05)
        self.min_edge = min_edge or risk_cfg.get("min_edge", 0.03)

        # Drawdown-aware dynamic sizing
        self._current_drawdown_pct: float = 0.0   # set externally by BankrollManager
        self._max_drawdown_target: float = 0.15   # target: never exceed 15% drawdown
        self._min_fraction: float = 0.05          # floor: never go below 5% of configured fraction
        self._profit_target_roi: float = 0.08     # target 8% ROI — scale up when exceeding this

        logger.info(
            "KellyCriterion: fraction=%.2f, max_bet=%.1f%%, min_edge=%.1f%%",
            self.fraction, self.max_bet_pct * 100, self.min_edge * 100,
        )

    def set_drawdown_state(self, current_drawdown_pct: float) -> None:
        """
        Called by BankrollManager to inform Kelly of current drawdown level.
        Kelly fraction is scaled down as drawdown approaches the target limit.
        """
        self._current_drawdown_pct = max(0.0, current_drawdown_pct)

    def _dynamic_fraction(self, base_fraction: float) -> float:
        """
        Scale Kelly fraction based on current drawdown.

        Logic:
          - drawdown = 0%:          use base_fraction unchanged
          - drawdown = 50% of max:  reduce to 75% of base_fraction
          - drawdown = 75% of max:  reduce to 50% of base_fraction
          - drawdown = 90% of max:  reduce to 25% of base_fraction
          - drawdown >= max:        use minimum fraction (floor)

        This is a smooth decay — not a step function — so sizing shrinks
        gradually as the session deteriorates, protecting against full ruin.
        """
        if self._max_drawdown_target <= 0:
            return base_fraction

        ratio = self._current_drawdown_pct / self._max_drawdown_target
        ratio = min(ratio, 1.0)

        # Smooth decay: scale = (1 - ratio)^1.5  (convex — shrinks faster near the limit)
        scale = (1.0 - ratio) ** 1.5
        # Enforce floor
        min_f = base_fraction * self._min_fraction
        dynamic = max(base_fraction * scale, min_f)

        if ratio > 0.3:
            logger.debug(
                "DynamicKelly: drawdown=%.1f%% (%.0f%% of limit), "
                "scale=%.2f, fraction %.3f→%.3f",
                self._current_drawdown_pct * 100, ratio * 100,
                scale, base_fraction, dynamic,
            )
        return dynamic

    # ------------------------------------------------------------------
    # Two-outcome Kelly (binary: win/lose)
    # ------------------------------------------------------------------

    def calculate(
        self,
        model_prob: float,
        decimal_odds: float,
    ) -> float:
        """
        Calculate fractional Kelly stake for a two-outcome bet.

        The Kelly formula for a simple bet:
            f* = (p * b - q) / b
        where:
            p = probability of winning
            q = 1 - p
            b = decimal_odds - 1 (net odds)

        Parameters
        ----------
        model_prob : float
            Model's estimated probability of this outcome.
        decimal_odds : float
            Bookmaker decimal odds for this outcome.

        Returns
        -------
        float
            Recommended stake as fraction of bankroll [0, max_bet_pct].
            Returns 0.0 if edge is below minimum threshold.
        """
        if model_prob <= 0 or model_prob >= 1 or decimal_odds <= 1:
            return 0.0

        implied_prob = 1.0 / decimal_odds
        edge = model_prob - implied_prob

        # Check minimum edge
        if edge < self.min_edge:
            return 0.0

        # Kelly formula
        b = decimal_odds - 1  # net odds (profit per unit staked)
        q = 1 - model_prob

        full_kelly = (model_prob * b - q) / b

        if full_kelly <= 0:
            return 0.0

        # Apply fraction and cap, with drawdown-aware dynamic scaling
        effective_fraction = self._dynamic_fraction(self.fraction)
        fractional = full_kelly * effective_fraction

        # Scale the hard cap down for high-odds bets — long shots have more
        # variance so we protect bankroll more aggressively:
        #   odds < 3.0  → full cap (5%)
        #   odds 3–6    → 3% cap
        #   odds 6–10   → 2% cap
        #   odds > 10   → 1% cap
        if decimal_odds >= 10:
            odds_cap = 0.01
        elif decimal_odds >= 6:
            odds_cap = 0.02
        elif decimal_odds >= 3:
            odds_cap = 0.03
        else:
            odds_cap = self.max_bet_pct
        effective_cap = min(self.max_bet_pct, odds_cap)
        capped = min(fractional, effective_cap)

        logger.debug(
            "Kelly: prob=%.3f, odds=%.2f, edge=%.3f, full=%.4f, frac=%.4f, "
            "cap=%.2f%%, final=%.4f",
            model_prob, decimal_odds, edge, full_kelly, fractional,
            effective_cap * 100, capped,
        )

        return round(capped, 6)

    # ------------------------------------------------------------------
    # Three-outcome Kelly (soccer 1X2)
    # ------------------------------------------------------------------

    def calculate_multiway(
        self,
        probabilities: dict[str, float],
        odds: dict[str, float],
    ) -> dict[str, float]:
        """
        Calculate Kelly stakes for a multi-outcome market.

        For a 3-way market (home/draw/away), we treat each outcome
        independently and only bet on outcomes with sufficient edge.
        This is a simplification of the full multi-outcome Kelly.

        Parameters
        ----------
        probabilities : dict
            Model probabilities per outcome, e.g.
            {'home_win': 0.55, 'draw': 0.25, 'away_win': 0.20}
        odds : dict
            Decimal odds per outcome, e.g.
            {'home_win': 1.85, 'draw': 3.40, 'away_win': 4.50}

        Returns
        -------
        dict
            Recommended stake (fraction of bankroll) per outcome.
            Outcomes with no edge return 0.0.
        """
        stakes: dict[str, float] = {}

        for outcome in probabilities:
            prob = probabilities[outcome]
            odd = odds.get(outcome, 0.0)

            if odd <= 1 or prob <= 0:
                stakes[outcome] = 0.0
                continue

            stake = self.calculate(prob, odd)
            stakes[outcome] = stake

        # If multiple outcomes have positive Kelly, only bet on
        # the one with the highest expected value
        positive_stakes = {k: v for k, v in stakes.items() if v > 0}

        if len(positive_stakes) > 1:
            # Keep only the outcome with highest EV
            best_outcome = max(
                positive_stakes,
                key=lambda k: probabilities[k] * odds[k] - 1,
            )
            for outcome in stakes:
                if outcome != best_outcome:
                    stakes[outcome] = 0.0

            logger.debug(
                "Multi-way Kelly: selected %s (EV=%.3f)",
                best_outcome,
                probabilities[best_outcome] * odds[best_outcome] - 1,
            )

        return stakes

    # ------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------

    def expected_value(
        self,
        model_prob: float,
        decimal_odds: float,
    ) -> float:
        """
        Calculate expected value of a bet.

        EV = (prob * odds) - 1

        Parameters
        ----------
        model_prob : float
            Estimated probability.
        decimal_odds : float
            Bookmaker decimal odds.

        Returns
        -------
        float
            Expected value (positive = profitable in expectation).
        """
        return model_prob * decimal_odds - 1

    def edge(
        self,
        model_prob: float,
        decimal_odds: float,
    ) -> float:
        """
        Calculate the edge (model probability vs implied probability).

        Parameters
        ----------
        model_prob : float
            Model's estimated probability.
        decimal_odds : float
            Bookmaker decimal odds.

        Returns
        -------
        float
            Edge (positive means model thinks outcome is more likely
            than the bookmaker implies).
        """
        implied = 1.0 / decimal_odds if decimal_odds > 0 else 1.0
        return model_prob - implied

    def growth_rate(
        self,
        model_prob: float,
        decimal_odds: float,
        stake_fraction: float,
    ) -> float:
        """
        Calculate expected log-growth rate (Kelly's criterion objective).

        G = p * ln(1 + f*b) + q * ln(1 - f)

        Parameters
        ----------
        model_prob : float
            Win probability.
        decimal_odds : float
            Decimal odds.
        stake_fraction : float
            Stake as fraction of bankroll.

        Returns
        -------
        float
            Expected log-growth rate per bet.
        """
        if stake_fraction <= 0 or stake_fraction >= 1:
            return 0.0

        b = decimal_odds - 1
        q = 1 - model_prob

        growth = (
            model_prob * np.log(1 + stake_fraction * b)
            + q * np.log(1 - stake_fraction)
        )
        return float(growth)
