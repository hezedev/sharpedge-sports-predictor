from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass
class SuitabilityDecision:
    suitable: bool
    recommended_market: str
    reason: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _selection_side(bet: dict[str, Any]) -> str:
    team = str(bet.get("team", "")).lower()
    home = str(bet.get("home", "")).lower()
    away = str(bet.get("away", "")).lower()
    if team == "draw":
        return "draw"
    if " or draw" in team:
        if home and home in team:
            return "home"
        if away and away in team:
            return "away"
    if home and home in team and away and away not in team:
        return "home"
    if away and away in team and home and home not in team:
        return "away"
    return "other"


def _selection_handicap(bet: dict[str, Any]) -> Optional[float]:
    team = str(bet.get("team", "")).strip()
    if not team:
        return None
    token = team.split()[-1]
    if not token or token[0] not in {"+", "-"}:
        return None
    try:
        return float(token)
    except ValueError:
        return None


def _market_line(bet: dict[str, Any]) -> Optional[float]:
    team = str(bet.get("team", "")).strip().lower()
    if not team:
        return None
    token = team.split()[-1]
    try:
        return float(token.replace("+", ""))
    except ValueError:
        return None


def _candidate_by_market(peers: list[dict[str, Any]], market: str, side: str = "") -> Optional[dict[str, Any]]:
    for peer in peers:
        if str(peer.get("market", "")) != market:
            continue
        if side and _selection_side(peer) != side:
            continue
        return peer
    return None


def _soccer_decision(bet: dict[str, Any], peers: list[dict[str, Any]]) -> SuitabilityDecision:
    market = str(bet.get("market", "moneyline"))
    side = _selection_side(bet)
    side_ml = _candidate_by_market(peers, "moneyline", side) if side in {"home", "away"} else None
    side_prob = float((side_ml or bet).get("ml_prob", 0.0) or 0.0)
    dc = _candidate_by_market(peers, "double_chance", side) if side in {"home", "away"} else None
    dnb = _candidate_by_market(peers, "draw_no_bet", side) if side in {"home", "away"} else None
    spread = _candidate_by_market(peers, "spreads", side) if side in {"home", "away"} else None
    totals = _candidate_by_market(peers, "totals")
    home_ml = _candidate_by_market(peers, "moneyline", "home")
    away_ml = _candidate_by_market(peers, "moneyline", "away")
    home_prob = float(home_ml.get("ml_prob", 0.0) or 0.0) if home_ml else 0.0
    away_prob = float(away_ml.get("ml_prob", 0.0) or 0.0) if away_ml else 0.0
    side_gap = abs(home_prob - away_prob) if home_prob and away_prob else 0.0
    dc_prob = float(dc.get("ml_prob", 0.0) or 0.0) if dc else 0.0
    draw_risk = max(0.0, dc_prob - side_prob) if dc else 0.0
    handicap = _selection_handicap(bet)

    if market == "moneyline":
        if side_gap <= 0.06 and totals:
            return SuitabilityDecision(
                suitable=False,
                recommended_market="totals",
                reason="market suitability check preferred totals because the match profile looks balanced and side conviction is too thin for a result bet",
                score=0.25,
            )
        if side_prob < 0.48 and spread and (handicap or 0.0) > 0:
            return SuitabilityDecision(
                suitable=False,
                recommended_market="spreads",
                reason="market suitability check preferred a positive handicap because the underdog looks competitive without a strong enough outright win case",
                score=0.3,
            )
        if 0.48 <= side_prob <= 0.60:
            if draw_risk >= 0.18 and dnb:
                return SuitabilityDecision(
                    suitable=False,
                    recommended_market="draw_no_bet",
                    reason="market suitability check preferred draw-no-bet because the side can win, but draw risk is too material for a raw moneyline",
                    score=0.35,
                )
            if dc:
                return SuitabilityDecision(
                    suitable=False,
                    recommended_market="double_chance",
                    reason="market suitability check preferred double chance because the side projects to avoid defeat more clearly than it projects to win outright",
                    score=0.32,
                )
            return SuitabilityDecision(
                suitable=False,
                recommended_market="no_bet",
                reason="market suitability check rejected the moneyline because the side is liked, but the win case is not strong enough and no safer market is available",
                score=0.2,
            )

    if market == "double_chance":
        if side_ml and side_prob >= 0.60 and draw_risk < 0.12:
            if dnb:
                return SuitabilityDecision(
                    suitable=False,
                    recommended_market="draw_no_bet",
                    reason="market suitability check preferred draw-no-bet because the side has a real win path and double chance is too conservative for the profile",
                    score=0.42,
                )
            if side_ml:
                return SuitabilityDecision(
                    suitable=False,
                    recommended_market="moneyline",
                    reason="market suitability check preferred moneyline because draw protection looks overpriced relative to the side's win probability",
                    score=0.4,
                )

    if market == "draw_no_bet":
        spread_handicap = _selection_handicap(spread) if spread else None
        if side_ml and side_prob < 0.43 and spread and (spread_handicap or 0.0) > 0:
            return SuitabilityDecision(
                suitable=False,
                recommended_market="spreads",
                reason="market suitability check preferred a positive handicap because draw-no-bet still demands too much outright win equity from the underdog",
                score=0.28,
            )
        if side_prob < 0.46 and dc:
            return SuitabilityDecision(
                suitable=False,
                recommended_market="double_chance",
                reason="market suitability check preferred double chance because the side looks more likely to avoid defeat than to produce the win rate draw-no-bet needs",
                score=0.3,
            )
        if side_prob >= 0.60 and draw_risk < 0.12 and side_ml:
            return SuitabilityDecision(
                suitable=False,
                recommended_market="moneyline",
                reason="market suitability check preferred moneyline because draw protection is not doing enough work in this match profile",
                score=0.38,
            )

    if market == "spreads":
        if handicap is not None and handicap < 0 and side_prob < 0.58 and side_ml:
            return SuitabilityDecision(
                suitable=False,
                recommended_market="moneyline",
                reason="market suitability check preferred moneyline because the minus handicap is more aggressive than the side's win profile supports",
                score=0.33,
            )
        if handicap is not None and handicap > 0 and side_prob >= 0.58 and side_ml:
            return SuitabilityDecision(
                suitable=False,
                recommended_market="moneyline",
                reason="market suitability check preferred moneyline because the side looks strong enough that the positive handicap is too conservative",
                score=0.34,
            )

    if market == "totals":
        line = _market_line(bet)
        if line is not None and line >= 3.5 and float(bet.get("edge", 0.0) or 0.0) < 0.05:
            return SuitabilityDecision(
                suitable=False,
                recommended_market="no_bet",
                reason="market suitability check rejected the totals angle because the line is high and the edge is not strong enough to justify the volatility",
                score=0.26,
            )

    return SuitabilityDecision(
        suitable=True,
        recommended_market=market,
        reason="market fits the current match profile",
        score=0.75,
    )


def _generic_side_decision(bet: dict[str, Any], peers: list[dict[str, Any]]) -> SuitabilityDecision:
    market = str(bet.get("market", "moneyline"))
    side = _selection_side(bet)
    side_ml = _candidate_by_market(peers, "moneyline", side) if side in {"home", "away"} else None
    side_prob = float((side_ml or bet).get("ml_prob", 0.0) or 0.0)
    spread = _candidate_by_market(peers, "spreads", side) if side in {"home", "away"} else None
    totals = _candidate_by_market(peers, "totals")
    handicap = _selection_handicap(bet)

    if market == "moneyline":
        if side_prob < 0.48 and spread and (_selection_handicap(spread) or 0.0) > 0:
            return SuitabilityDecision(
                suitable=False,
                recommended_market="spreads",
                reason="market suitability check preferred the positive handicap because the underdog looks live without a strong enough outright win case",
                score=0.3,
            )
        if totals and 0.49 <= side_prob <= 0.56 and float(bet.get("edge", 0.0) or 0.0) < 0.05:
            return SuitabilityDecision(
                suitable=False,
                recommended_market="totals",
                reason="market suitability check preferred totals because the side edge is modest and the matchup does not justify forcing a result market",
                score=0.28,
            )

    if market == "spreads":
        if handicap is not None and handicap < 0 and side_prob < 0.57 and side_ml:
            return SuitabilityDecision(
                suitable=False,
                recommended_market="moneyline",
                reason="market suitability check preferred moneyline because the minus handicap is too aggressive for the current side probability",
                score=0.32,
            )
        if handicap is not None and handicap > 0 and side_prob >= 0.58 and side_ml:
            return SuitabilityDecision(
                suitable=False,
                recommended_market="moneyline",
                reason="market suitability check preferred moneyline because the positive handicap is too conservative for a strong favorite profile",
                score=0.31,
            )

    return SuitabilityDecision(
        suitable=True,
        recommended_market=market,
        reason="market fits the current match profile",
        score=0.72,
    )


def evaluate_market_suitability(bet: dict[str, Any], peers: list[dict[str, Any]]) -> SuitabilityDecision:
    sport = str(bet.get("sport", "") or "").lower()
    if sport == "soccer":
        return _soccer_decision(bet, peers)
    return _generic_side_decision(bet, peers)
