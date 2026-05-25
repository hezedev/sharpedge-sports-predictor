from __future__ import annotations

from copy import deepcopy
from typing import Any

from config import settings


DEFAULT_LANE = {
    "status": "disabled",
    "label": "Analysis Only",
    "focus_allowed": False,
    "parlay_allowed": False,
    "quality_floor": 80,
    "reason": "This sport/market is outside the current focused prediction lanes.",
    "requires": [],
    "trusted_leagues": [],
}


_DEFAULT_FOCUSED_LANES: dict[str, dict[str, dict[str, Any]]] = {
    "mlb": {
        "moneyline": {
            "status": "primary",
            "label": "Primary Focus",
            "focus_allowed": True,
            "parlay_allowed": False,
            "quality_floor": 82,
            "reason": "Primary lane: MLB moneyline has the cleanest current signal and strong structured pre-game inputs.",
            "requires": [
                "probable_pitchers",
                "pitcher_change_check",
                "bullpen_workload",
                "weather",
                "park_factor",
                "lineups_or_projected_lineups",
                "travel_rest",
            ],
        },
    },
    "tennis": {
        "moneyline": {
            "status": "secondary",
            "label": "Secondary Focus",
            "focus_allowed": True,
            "parlay_allowed": False,
            "quality_floor": 80,
            "reason": "Secondary lane: ATP tennis moneyline is a clean 1v1 market with manageable surface/form/fatigue inputs.",
            "requires": [
                "surface",
                "recent_form",
                "ranking_or_elo",
                "fatigue",
                "injury_retirement_check",
                "tournament_context",
            ],
            "trusted_tours": ["atp"],
        },
    },
    "soccer": {
        "moneyline": {
            "status": "controlled",
            "label": "Controlled Focus",
            "focus_allowed": True,
            "parlay_allowed": False,
            "quality_floor": 86,
            "reason": "Controlled soccer wins lane: result bets are allowed only with strong team-news, motivation, and rotation context.",
            "requires": [
                "fixture_verified",
                "fresh_odds",
                "team_news",
                "motivation",
                "rotation_or_congestion",
                "home_away_form",
            ],
        },
        "totals": {
            "status": "controlled",
            "label": "Controlled Focus",
            "focus_allowed": True,
            "parlay_allowed": False,
            "quality_floor": 84,
            "reason": "Controlled soccer goals lane: totals need fresh lines plus goal-profile and lineup/rotation context.",
            "requires": [
                "fixture_verified",
                "fresh_odds",
                "goal_profile",
                "team_news",
                "rotation_or_congestion",
                "home_away_form",
            ],
        },
        "team_total": {
            "status": "controlled",
            "label": "Controlled Focus",
            "focus_allowed": True,
            "parlay_allowed": False,
            "quality_floor": 86,
            "reason": "Controlled soccer goals lane: team totals need team-specific attack/defense and lineup context.",
            "requires": [
                "fixture_verified",
                "fresh_odds",
                "team_goal_profile",
                "team_news",
                "rotation_or_congestion",
                "home_away_form",
            ],
        },
        "double_chance": {
            "status": "controlled",
            "label": "Controlled Focus",
            "focus_allowed": True,
            "parlay_allowed": False,
            "quality_floor": 82,
            "reason": "Controlled soccer safety lane: double chance is allowed when draw risk and team context are explicitly handled.",
            "requires": [
                "fixture_verified",
                "fresh_odds",
                "team_news",
                "motivation",
                "rotation_or_congestion",
                "home_away_form",
            ],
        },
        "draw_no_bet": {
            "status": "controlled",
            "label": "Controlled Focus",
            "focus_allowed": True,
            "parlay_allowed": False,
            "quality_floor": 82,
            "reason": "Controlled soccer safety lane: DNB is allowed when win case is plausible but draw risk remains material.",
            "requires": [
                "fixture_verified",
                "fresh_odds",
                "team_news",
                "motivation",
                "rotation_or_congestion",
                "home_away_form",
            ],
        },
    },
}


def normalize_prediction_market(market: str | None) -> str:
    text = (market or "moneyline").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"", "h2h", "wins", "win"}:
        return "moneyline"
    if "draw_no_bet" in text or "dnb" in text:
        return "draw_no_bet"
    if "double_chance" in text:
        return "double_chance"
    if "team_total" in text:
        return "team_total"
    if "goal" in text or "total" in text:
        return "totals"
    if "spread" in text or "cover" in text or "handicap" in text or "puck" in text or "run_line" in text:
        return "spreads"
    return text


def _configured_lanes() -> dict[str, dict[str, dict[str, Any]]]:
    configured = (settings or {}).get("prediction_lanes")
    if isinstance(configured, dict) and configured:
        return configured
    return _DEFAULT_FOCUSED_LANES


def get_prediction_lane(sport: str | None, market: str | None) -> dict[str, Any]:
    sport_key = (sport or "").strip().lower()
    market_key = normalize_prediction_market(market)
    sport_lanes = _configured_lanes().get(sport_key, {})
    lane = deepcopy(sport_lanes.get(market_key, DEFAULT_LANE))
    lane["sport"] = sport_key
    lane["market_key"] = market_key
    lane["focus_allowed"] = bool(lane.get("focus_allowed", lane.get("status") in {"primary", "secondary", "controlled"}))
    lane["parlay_allowed"] = bool(lane.get("parlay_allowed", False))
    lane["quality_floor"] = int(lane.get("quality_floor", 80) or 80)
    return lane


def summarize_prediction_lanes() -> dict[str, list[dict[str, Any]]]:
    summary: dict[str, list[dict[str, Any]]] = {
        "primary": [],
        "secondary": [],
        "controlled": [],
        "disabled": [],
    }
    for sport, markets in _configured_lanes().items():
        for market, lane in markets.items():
            item = {
                "sport": sport,
                "market": market,
                "status": str(lane.get("status", "disabled")),
                "label": str(lane.get("label", "")),
                "quality_floor": int(lane.get("quality_floor", 80) or 80),
                "reason": str(lane.get("reason", "")),
                "requires": list(lane.get("requires") or []),
            }
            summary.setdefault(item["status"], []).append(item)
    for key in summary:
        summary[key] = sorted(summary[key], key=lambda item: (item["sport"], item["market"]))
    return summary
