from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.prediction.lane_config import get_prediction_lane, summarize_prediction_lanes


DEFAULT_POLICY = {
    "status": "experimental",
    "score": 50,
    "label": "Experimental",
    "production_allowed": False,
    "parlay_allowed": False,
    "stake_multiplier": 0.0,
    "reason": "This market has not been validated strongly enough to be promoted above the default pool yet.",
}

STATUS_META = {
    "preferred": {"label": "Preferred", "score_boost": 0},
    "experimental": {"label": "Experimental", "score_boost": 0},
    "disabled": {"label": "Disabled", "score_boost": 0},
}

_MARKET_POLICIES = {
    "soccer": {
        "moneyline": {
            "status": "experimental",
            "score": 60,
            "label": "Limited",
            "production_allowed": True,
            "parlay_allowed": False,
            "stake_multiplier": 0.25,
            "reason": "Soccer 1X2 is now live at reduced stake sizing while double-chance, draw-no-bet, and handicap-style markets remain the stronger soccer lanes.",
        },
        "totals": {
            "status": "preferred",
            "score": 88,
            "label": "Preferred",
            "production_allowed": True,
            "parlay_allowed": True,
            "stake_multiplier": 1.0,
            "reason": "Soccer totals became much stronger once more lines were tested, with over 0.5 and over 1.5 clearly outperforming over 2.5.",
        },
        "btts": {
            "status": "disabled",
            "score": 15,
            "label": "Disabled",
            "production_allowed": False,
            "parlay_allowed": False,
            "stake_multiplier": 0.0,
            "reason": "Soccer BTTS backtested weaker and less stable than totals, so it is suppressed for now.",
        },
        "draw_no_bet": {
            "status": "preferred",
            "score": 86,
            "label": "Preferred",
            "production_allowed": True,
            "parlay_allowed": True,
            "stake_multiplier": 1.0,
            "reason": "Soccer draw-no-bet markets replayed strongly and add a safer alternative to raw 1X2 sides.",
        },
        "double_chance": {
            "status": "preferred",
            "score": 90,
            "label": "Preferred",
            "production_allowed": True,
            "parlay_allowed": True,
            "stake_multiplier": 1.0,
            "reason": "Soccer double-chance markets were among the strongest and best-calibrated results in the expanded replay sweep.",
        },
        "spreads": {
            "status": "preferred",
            "score": 92,
            "label": "Preferred",
            "production_allowed": True,
            "parlay_allowed": True,
            "stake_multiplier": 1.0,
            "reason": "Soccer handicap-style markets were the standout family in replay and now belong in the top live tier.",
        },
        "team_total": {
            "status": "experimental",
            "score": 64,
            "label": "Experimental",
            "production_allowed": False,
            "parlay_allowed": False,
            "stake_multiplier": 0.0,
            "reason": "Soccer team totals are usable, but they lag the stronger handicap, double-chance, and lower totals-line markets.",
        },
    },
    "basketball": {
        "moneyline": {
            "status": "experimental",
            "score": 62,
            "label": "Limited",
            "production_allowed": True,
            "parlay_allowed": False,
            "stake_multiplier": 0.4,
            "reason": "Basketball moneyline is now live at reduced stake sizing while totals and team totals remain the stronger basketball lanes.",
        },
        "totals": {
            "status": "preferred",
            "score": 84,
            "label": "Preferred",
            "production_allowed": True,
            "parlay_allowed": True,
            "stake_multiplier": 1.0,
            "reason": "Basketball totals improved materially once the replay coverage widened and now rank near the top of the live candidates.",
        },
        "team_total": {
            "status": "preferred",
            "score": 86,
            "label": "Preferred",
            "production_allowed": True,
            "parlay_allowed": True,
            "stake_multiplier": 1.0,
            "reason": "Basketball team totals, especially home team totals, were the strongest basketball market family in replay.",
        },
        "spreads": {
            "status": "experimental",
            "score": 64,
            "label": "Experimental",
            "production_allowed": False,
            "parlay_allowed": False,
            "stake_multiplier": 0.0,
            "reason": "Basketball spreads were decent, but still trailed totals and team totals in the replay ranking.",
        },
    },
    "tennis": {
        "moneyline": {
            "status": "preferred",
            "score": 88,
            "label": "Preferred",
            "production_allowed": True,
            "parlay_allowed": True,
            "stake_multiplier": 1.0,
            "reason": "Tennis moneyline remains strong, even though totals and straight-sets markets look even better.",
        },
        "totals": {
            "status": "preferred",
            "score": 96,
            "label": "Preferred",
            "production_allowed": True,
            "parlay_allowed": True,
            "stake_multiplier": 1.0,
            "reason": "Tennis total games over 22.5 was one of the strongest markets in the entire replay set.",
        },
        "set_betting": {
            "status": "preferred",
            "score": 94,
            "label": "Preferred",
            "production_allowed": True,
            "parlay_allowed": True,
            "stake_multiplier": 1.0,
            "reason": "Tennis straight-sets style markets replayed very strongly and belong in the top tier.",
        },
    },
    "nhl": {
        "moneyline": {
            "status": "experimental",
            "score": 60,
            "label": "Limited",
            "production_allowed": True,
            "parlay_allowed": False,
            "stake_multiplier": 0.3,
            "reason": "NHL moneyline is now live at reduced stake sizing, even though puck line and home team totals still rate better.",
        },
        "totals": {
            "status": "disabled",
            "score": 25,
            "label": "Disabled",
            "production_allowed": False,
            "parlay_allowed": False,
            "stake_multiplier": 0.0,
            "reason": "NHL full totals were the weakest NHL market in replay and are being suppressed.",
        },
        "spreads": {
            "status": "preferred",
            "score": 90,
            "label": "Preferred",
            "production_allowed": True,
            "parlay_allowed": True,
            "stake_multiplier": 1.0,
            "reason": "NHL home -1.5 was the best-performing NHL market in replay.",
        },
        "team_total": {
            "status": "preferred",
            "score": 83,
            "label": "Preferred",
            "production_allowed": True,
            "parlay_allowed": True,
            "stake_multiplier": 1.0,
            "reason": "NHL home team totals were strong and well-calibrated in replay.",
        },
    },
    "mlb": {
        "moneyline": {
            "status": "experimental",
            "score": 56,
            "label": "Limited",
            "production_allowed": True,
            "parlay_allowed": False,
            "stake_multiplier": 0.4,
            "reason": "MLB moneyline is now live at reduced stake sizing while replay ranking continues to mature.",
        },
        "totals": {
            "status": "experimental",
            "score": 54,
            "label": "Experimental",
            "production_allowed": False,
            "parlay_allowed": False,
            "stake_multiplier": 0.0,
            "reason": "MLB totals need a clearer replay ranking before they move into the preferred set.",
        },
        "team_total": {
            "status": "experimental",
            "score": 55,
            "label": "Experimental",
            "production_allowed": False,
            "parlay_allowed": False,
            "stake_multiplier": 0.0,
            "reason": "MLB team totals are promising, but they still need a completed replay ranking.",
        },
        "spreads": {
            "status": "experimental",
            "score": 50,
            "label": "Limited",
            "production_allowed": True,
            "parlay_allowed": False,
            "stake_multiplier": 0.25,
            "reason": "MLB run-line style markets remain the riskier MLB lane, so they stay live only at a smaller stake while replay ranking is still being validated.",
        },
    },
}


def _normalize_market(market: str | None) -> str:
    text = (market or "moneyline").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"", "h2h"}:
        return "moneyline"
    if "btts" in text:
        return "btts"
    if "draw_no_bet" in text or "dnb" in text:
        return "draw_no_bet"
    if "double_chance" in text:
        return "double_chance"
    if "straight_sets" in text or "set_betting" in text or "sets" in text:
        return "set_betting"
    if "team_total" in text:
        return "team_total"
    if "goal" in text:
        return "totals"
    if "total" in text:
        return "totals"
    if "spread" in text or "cover" in text or "handicap" in text or "puck" in text or "run_line" in text:
        return "spreads"
    return text


def get_market_policy(sport: str | None, market: str | None) -> dict[str, Any]:
    sport_key = (sport or "").strip().lower()
    market_key = _normalize_market(market)
    sport_policy = _MARKET_POLICIES.get(sport_key, {})
    policy = deepcopy(sport_policy.get(market_key, DEFAULT_POLICY))
    policy["sport"] = sport_key
    policy["market_key"] = market_key
    return policy


def annotate_bet(bet: dict[str, Any]) -> dict[str, Any]:
    annotated = dict(bet)
    policy = get_market_policy(annotated.get("sport"), annotated.get("market"))
    lane = get_prediction_lane(annotated.get("sport"), annotated.get("market"))
    annotated["market"] = _normalize_market(annotated.get("market"))
    annotated["market_status"] = policy["status"]
    annotated["market_priority_score"] = policy["score"]
    annotated["market_policy_label"] = policy["label"]
    annotated["market_policy_reason"] = policy["reason"]
    annotated["prediction_lane_status"] = lane["status"]
    annotated["prediction_lane_label"] = lane["label"]
    annotated["prediction_lane_reason"] = lane["reason"]
    annotated["prediction_focus_allowed"] = bool(lane["focus_allowed"])
    annotated["prediction_quality_floor"] = int(lane["quality_floor"])
    annotated["prediction_lane_requires"] = list(lane.get("requires") or [])
    annotated["production_allowed"] = bool(policy.get("production_allowed", policy["status"] == "preferred")) and bool(lane["focus_allowed"])
    annotated["parlay_allowed"] = bool(policy.get("parlay_allowed", True)) and bool(lane["parlay_allowed"])
    annotated["stake_multiplier"] = float(policy.get("stake_multiplier", 1.0))
    return annotated


def filter_and_rank_bets(bets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotated = [annotate_bet(bet) for bet in bets]
    allowed = [bet for bet in annotated if bet.get("production_allowed")]
    return sorted(
        allowed,
        key=lambda bet: (
            bet.get("market_priority_score", 0),
            bet.get("edge", 0.0),
            bet.get("ml_prob", 0.0),
        ),
        reverse=True,
    )


def summarize_market_policy() -> dict[str, list[dict[str, Any]]]:
    summary: dict[str, list[dict[str, Any]]] = {"preferred": [], "experimental": [], "disabled": []}
    for sport, sport_markets in _MARKET_POLICIES.items():
        for market, policy in sport_markets.items():
            item = {
                "sport": sport,
                "market": market,
                "status": policy["status"],
                "score": policy["score"],
                "label": policy["label"],
                "reason": policy["reason"],
            }
            summary[policy["status"]].append(item)
    for key in summary:
        summary[key] = sorted(summary[key], key=lambda item: (item["score"], item["sport"], item["market"]), reverse=True)
    return summary


def summarize_focused_prediction_policy() -> dict[str, list[dict[str, Any]]]:
    return summarize_prediction_lanes()
