"""
Value bet detection module.

Compares model-predicted probabilities against bookmaker odds
to identify positive expected value (EV+) opportunities.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from config import settings
from src.risk.kelly import KellyCriterion
from src.utils.helpers import decimal_to_implied_probability, remove_vig

logger = logging.getLogger(__name__)


@dataclass
class ValueBet:
    """A detected value betting opportunity."""
    match_id: str
    sport: str
    home_team: str
    away_team: str
    outcome: str
    model_prob: float
    implied_prob: float
    fair_prob: float
    best_odds: float
    best_bookmaker: str
    edge: float
    expected_value: float
    kelly_stake: float
    confidence: float
    commence_time: Optional[str] = None


class ValueDetector:
    """
    Detect value bets by comparing model probabilities with market odds.

    A value bet exists when the model's estimated probability
    exceeds the bookmaker's implied probability (after removing vig)
    by more than the configured minimum edge.

    Parameters
    ----------
    kelly : KellyCriterion, optional
        Kelly calculator instance. Created with defaults if not provided.
    min_edge : float, optional
        Minimum edge to qualify as a value bet.
    min_confidence : float, optional
        Minimum model confidence to consider.
    """

    def __init__(
        self,
        kelly: Optional[KellyCriterion] = None,
        min_edge: Optional[float] = None,
        min_confidence: float = 0.0,
    ) -> None:
        risk_cfg = settings.get("risk", {}).get("kelly", {})

        self.kelly = kelly or KellyCriterion()
        self.min_edge = min_edge or risk_cfg.get("min_edge", 0.03)
        self.min_confidence = min_confidence

        logger.info(
            "ValueDetector: min_edge=%.1f%%, min_confidence=%.1f%%",
            self.min_edge * 100, self.min_confidence * 100,
        )

    def detect(
        self,
        predictions: pd.DataFrame,
        odds: pd.DataFrame,
        match_info: Optional[pd.DataFrame] = None,
    ) -> List[ValueBet]:
        """
        Scan predictions and odds for value betting opportunities.

        Parameters
        ----------
        predictions : pd.DataFrame
            Model probability predictions. Columns are outcome labels,
            values are probabilities. Must have an index or column
            that can be matched to odds data.
        odds : pd.DataFrame
            Best odds per outcome from OddsFetcher.get_best_odds().
            Expected columns: event_id, home_team, away_team, market,
            outcome, price, best_bookmaker, implied_prob.
        match_info : pd.DataFrame, optional
            Additional match context (teams, date, etc.).

        Returns
        -------
        list[ValueBet]
            Detected value bets sorted by expected value (descending).
        """
        value_bets: List[ValueBet] = []

        if predictions.empty or odds.empty:
            logger.info("No predictions or odds available for value detection")
            return value_bets

        # Group odds by event
        for event_id, event_odds in odds.groupby("event_id"):
            home_team = event_odds["home_team"].iloc[0]
            away_team = event_odds["away_team"].iloc[0]
            commence = event_odds.get("commence_time", pd.Series([None])).iloc[0]

            # Try to match this event to a prediction row
            pred_row = self._match_prediction(
                predictions, home_team, away_team, match_info
            )
            if pred_row is None:
                continue

            # Get the fair probabilities (vig-removed) for this event
            h2h_odds = event_odds[event_odds["market"] == "h2h"]
            if h2h_odds.empty:
                continue

            implied_probs = h2h_odds["implied_prob"].tolist()
            fair_probs = remove_vig(implied_probs)

            # Check each outcome for value
            for i, (_, odds_row) in enumerate(h2h_odds.iterrows()):
                outcome_name = odds_row["outcome"]
                best_price = odds_row["price"]
                best_bk = odds_row.get("best_bookmaker", "unknown")
                implied_p = odds_row["implied_prob"]
                fair_p = fair_probs[i] if i < len(fair_probs) else implied_p

                # Map outcome name to prediction column
                model_prob = self._get_model_prob(pred_row, outcome_name, home_team, away_team)
                if model_prob is None:
                    continue

                # Calculate edge and EV
                edge = model_prob - fair_p
                ev = model_prob * best_price - 1

                # Check if this qualifies as a value bet
                if edge < self.min_edge:
                    continue
                if model_prob < self.min_confidence:
                    continue

                # Calculate Kelly stake
                kelly_stake = self.kelly.calculate(model_prob, best_price)

                if kelly_stake <= 0:
                    continue

                vb = ValueBet(
                    match_id=str(event_id),
                    sport="",
                    home_team=home_team,
                    away_team=away_team,
                    outcome=outcome_name,
                    model_prob=model_prob,
                    implied_prob=implied_p,
                    fair_prob=fair_p,
                    best_odds=best_price,
                    best_bookmaker=best_bk,
                    edge=edge,
                    expected_value=ev,
                    kelly_stake=kelly_stake,
                    confidence=model_prob,
                    commence_time=str(commence) if commence else None,
                )
                value_bets.append(vb)

                logger.debug(
                    "Value bet: %s vs %s | %s @ %.2f | edge=%.1f%% | EV=%.1f%% | kelly=%.3f",
                    home_team, away_team, outcome_name, best_price,
                    edge * 100, ev * 100, kelly_stake,
                )

        # Sort by expected value (descending)
        value_bets.sort(key=lambda x: x.expected_value, reverse=True)

        logger.info("Detected %d value bets", len(value_bets))
        return value_bets

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _match_prediction(
        self,
        predictions: pd.DataFrame,
        home_team: str,
        away_team: str,
        match_info: Optional[pd.DataFrame] = None,
    ) -> Optional[pd.Series]:
        """
        Find the prediction row matching a given match.

        Tries to match by team names in match_info or predictions index.
        """
        if match_info is not None and not match_info.empty:
            # Try matching by team names in match_info
            mask = (
                (match_info["home_team"].str.lower() == home_team.lower())
                & (match_info["away_team"].str.lower() == away_team.lower())
            )
            matched = match_info[mask]
            if not matched.empty:
                idx = matched.index[0]
                if idx in predictions.index:
                    return predictions.loc[idx]

        # Fallback: try matching by column content in predictions
        if "home_team" in predictions.columns:
            mask = (
                (predictions["home_team"].str.lower() == home_team.lower())
                & (predictions["away_team"].str.lower() == away_team.lower())
            )
            matched = predictions[mask]
            if not matched.empty:
                return matched.iloc[0]

        return None

    @staticmethod
    def _get_model_prob(
        pred_row: pd.Series,
        outcome_name: str,
        home_team: str,
        away_team: str,
    ) -> Optional[float]:
        """
        Extract model probability for an outcome, handling name mapping.

        The Odds API uses team names as outcomes (e.g., 'Arsenal'),
        while our models use 'home_win', 'draw', 'away_win'.
        """
        # Direct column match
        if outcome_name in pred_row.index:
            return float(pred_row[outcome_name])

        # Map team names to our label scheme
        name_lower = outcome_name.lower()

        if name_lower == home_team.lower() or name_lower == "home":
            for col in ["home_win", "player1_win"]:
                if col in pred_row.index:
                    return float(pred_row[col])

        elif name_lower == away_team.lower() or name_lower == "away":
            for col in ["away_win", "player2_win"]:
                if col in pred_row.index:
                    return float(pred_row[col])

        elif name_lower in ("draw", "tie", "x"):
            if "draw" in pred_row.index:
                return float(pred_row["draw"])

        return None

    def to_dataframe(self, value_bets: List[ValueBet]) -> pd.DataFrame:
        """Convert list of ValueBet objects to a DataFrame."""
        if not value_bets:
            return pd.DataFrame()

        records = []
        for vb in value_bets:
            records.append({
                "match_id": vb.match_id,
                "sport": vb.sport,
                "home_team": vb.home_team,
                "away_team": vb.away_team,
                "outcome": vb.outcome,
                "model_prob": round(vb.model_prob, 4),
                "implied_prob": round(vb.implied_prob, 4),
                "fair_prob": round(vb.fair_prob, 4),
                "best_odds": vb.best_odds,
                "best_bookmaker": vb.best_bookmaker,
                "edge_pct": round(vb.edge * 100, 2),
                "ev_pct": round(vb.expected_value * 100, 2),
                "kelly_stake": round(vb.kelly_stake, 4),
                "confidence": round(vb.confidence, 4),
                "commence_time": vb.commence_time,
            })

        df = pd.DataFrame(records)
        return df.sort_values("ev_pct", ascending=False).reset_index(drop=True)
