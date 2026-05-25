from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from .arbiter_mind import ConsensusArbiterMind
from .contracts import CommitteeDecision
from .model_mind import QuantModelMind
from .output_formatter import format_committee_pick_output
from .parlay_builder import CommitteeParlayBuilder, CommitteeParlayPlan
from .research_mind import ContextResearchMind


def _iso_in(*, hours: float = 0.0, minutes: float = 0.0) -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=hours, minutes=minutes)).isoformat()


def _base_candidate(
    *,
    sport: str = "soccer",
    home: str = "Alpha FC",
    away: str = "Beta FC",
    team: str = "Alpha FC",
    market: str = "moneyline",
    odds: float = 1.90,
    minimum_acceptable_odds: float = 1.78,
    ml_prob: float = 0.58,
    market_implied_prob: float = 0.5263,
    vig_free_implied_prob: float = 0.51,
    fair_odds: float = 1.724,
    edge: float = 0.102,
    confidence_range_low: float = 0.54,
    confidence_range_high: float = 0.62,
    lower_bound_passed: bool = True,
    recommended_market: str | None = None,
    kickoff_hours: float = 8.0,
) -> dict[str, Any]:
    if recommended_market is None:
        recommended_market = market

    return {
        "sport": sport,
        "market": market,
        "team": team,
        "home": home,
        "away": away,
        "status": "scheduled",
        "commence_time": _iso_in(hours=kickoff_hours),
        "odds_snapshot_age_hours": 1.0,
        "standings_snapshot_age_hours": 2.0,
        "availability_fetched_at": _iso_in(minutes=-30),
        "ml_prob": ml_prob,
        "market_implied_prob": market_implied_prob,
        "vig_free_implied_prob": vig_free_implied_prob,
        "fair_odds": fair_odds,
        "minimum_acceptable_odds": minimum_acceptable_odds,
        "odds": odds,
        "edge": edge,
        "confidence_range_low": confidence_range_low,
        "confidence_range_high": confidence_range_high,
        "lower_bound_passed": lower_bound_passed,
        "recommended_market": recommended_market,
        "scraped_context": {
            "home_team_name": home,
            "away_team_name": away,
            "availability_source": "api_football",
            "availability_fetched_at": _iso_in(minutes=-30),
            "is_playoff": 1,
            "playoff_motivation": 1,
        },
        "scraped_context_sources": ["api_football", "espn", "odds_snapshot"],
        "scraped_context_highlights": [
            "Playoff spot pressure favors the selected side",
            "Availability context fetched from api_football",
            "ESPN preview confirms the fixture context and team-news framing",
        ],
    }


def _nhl_complete_candidate(**overrides: Any) -> dict[str, Any]:
    defaults = {
        "sport": "nhl",
        "home": "Boston Bruins",
        "away": "Montreal Canadiens",
        "team": "Boston Bruins",
        "market": "spreads",
    }
    candidate = _base_candidate(**{**defaults, **overrides})
    candidate["scraped_context"] = {
        **candidate.get("scraped_context", {}),
        "availability_source": "nhl_api",
        "lineup_source": "nhl_api",
        "home_goalie_confirmed": 1,
        "away_goalie_confirmed": 1,
        "home_goalie_name": "Jeremy Swayman",
        "away_goalie_name": "Sam Montembeault",
    }
    candidate["scraped_context_sources"] = ["nhl_api", "espn", "odds_snapshot"]
    candidate["scraped_context_highlights"] = [
        "Projected starting goalies confirmed by NHL team reports",
        "Rest/travel spot reviewed against the latest schedule feed",
        "ESPN matchup preview confirms the game context and injury framing",
    ]
    candidate["prediction_factors"] = [
        {"name": "home_xg_diff_10", "summary": "Recent expected-goals differential is favorable."},
        {"name": "away_goal_diff_10", "summary": "Opponent recent goal differential is softer."},
    ]
    candidate["context_adjustments"] = [
        {"name": "back_to_back", "summary": "Rest/back-to-back context checked."},
        {"name": "travel_fatigue", "summary": "Travel fatigue context checked."},
        {"name": "special_teams_edge", "summary": "Power-play and penalty-kill gap checked."},
        {"name": "xg_structure", "summary": "Shot-quality structure checked."},
        {"name": "system_stability", "summary": "Home/away split context checked."},
    ]
    return candidate


def _basketball_complete_candidate(**overrides: Any) -> dict[str, Any]:
    defaults = {
        "sport": "basketball",
        "home": "Boston Celtics",
        "away": "New York Knicks",
        "team": "Boston Celtics",
        "market": "moneyline",
    }
    candidate = _base_candidate(**{**defaults, **overrides})
    candidate["scraped_context"] = {
        **candidate.get("scraped_context", {}),
        "availability_source": "api_sports_basketball",
        "lineup_source": "api_sports_basketball",
        "home_priority_absences_count": 0,
        "away_priority_absences_count": 0,
        "home_questionable_count": 0,
        "away_questionable_count": 0,
    }
    candidate["scraped_context_sources"] = ["api_sports_basketball", "balldontlie", "espn", "odds_snapshot"]
    candidate["scraped_context_highlights"] = [
        "NBA injury report checked with projected lineup context",
        "Travel and rest spot cross-checked from the latest schedule feed",
        "Preview source confirms the expected rotation footprint",
    ]
    candidate["prediction_factors"] = [
        {"name": "home_ortg", "summary": "Home offensive rating profile is trending above baseline."},
        {"name": "away_drtg", "summary": "Away defensive rating profile is softer than league average."},
    ]
    candidate["context_adjustments"] = [
        {"name": "back_to_back", "summary": "Rest/back-to-back context checked."},
        {"name": "travel_fatigue", "summary": "Travel context checked."},
        {"name": "pace_control", "summary": "Projected pace environment checked."},
        {"name": "closing_execution", "summary": "Offensive/defensive execution context checked."},
        {"name": "injury_report_edge", "summary": "Star availability and inactive report checked."},
        {"name": "rotation_quality_edge", "summary": "Rotation and minutes stability checked."},
    ]
    return candidate


def _mlb_complete_candidate(**overrides: Any) -> dict[str, Any]:
    defaults = {
        "sport": "mlb",
        "home": "New York Yankees",
        "away": "Boston Red Sox",
        "team": "New York Yankees",
        "market": "moneyline",
    }
    candidate = _base_candidate(**{**defaults, **overrides})
    candidate["scraped_context"] = {
        **candidate.get("scraped_context", {}),
        "availability_source": "mlb_stats_api",
        "lineup_source": "mlb_stats_api",
        "home_starter_confirmed": 1,
        "away_starter_confirmed": 1,
        "home_starter_name": "Gerrit Cole",
        "away_starter_name": "Brayan Bello",
        "home_starter_hand": "R",
        "away_starter_hand": "R",
        "home_starter_era": 2.95,
        "away_starter_era": 4.12,
        "home_starter_whip": 1.03,
        "away_starter_whip": 1.28,
        "home_rest_days": 1,
        "away_rest_days": 0,
        "away_travel_km": 305.0,
        "bullpen_fatigue_risk": 0,
        "park_factor_proxy": 0.99,
        "roof_status": "open_air",
    }
    candidate["scraped_context_sources"] = ["mlb_stats_api", "espn", "odds_snapshot", "openweather"]
    candidate["scraped_context_highlights"] = [
        "Probable starters confirmed by MLB Stats API.",
        "Lineup and injury framing cross-checked through MLB and ESPN reporting.",
        "Weather and park context reviewed for this matchup.",
    ]
    candidate["prediction_factors"] = [
        {"name": "sp_era_diff", "summary": "Starter ERA gap supports the home side."},
        {"name": "sp_whip_diff", "summary": "Starter WHIP profile supports the home side."},
    ]
    candidate["context_adjustments"] = [
        {"name": "bullpen_workload", "summary": "Bullpen workload checked over the last three days."},
        {"name": "travel_fatigue", "summary": "Travel and rest context checked."},
        {"name": "rest_advantage", "summary": "Series turnaround and rest edge checked."},
    ]
    return candidate


def _run_pick(candidate: dict[str, Any], *, min_edge: float = 0.03) -> dict[str, Any]:
    candidate = dict(candidate)
    model_mind = QuantModelMind(min_edge=min_edge)
    research_mind = ContextResearchMind()
    arbiter_mind = ConsensusArbiterMind(min_edge=min_edge)

    arbiter_candidate = dict(candidate)
    substitute_candidate = candidate.get("substitute_candidate")
    if isinstance(substitute_candidate, dict):
        arbiter_candidate["substitute_research"] = research_mind.evaluate(substitute_candidate)
        arbiter_candidate["substitute_model"] = model_mind.evaluate(substitute_candidate)

    research = research_mind.evaluate(candidate)
    model = model_mind.evaluate(candidate)
    committee = arbiter_mind.decide(candidate=arbiter_candidate, research=research, model=model)
    formatted_output = format_committee_pick_output(
        candidate=candidate,
        research=research,
        model=model,
        arbiter=committee,
    )

    return {
        "candidate": candidate,
        "research": research,
        "model": model,
        "committee": committee,
        "formatted_output": formatted_output,
    }


def _parlay_entry(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate": result["candidate"],
        "research_decision": result["research"],
        "model_decision": result["model"],
        "committee_decision": result["committee"],
    }


def _parlay_summary(plan: CommitteeParlayPlan) -> str:
    weakest = plan.weakest_leg or {}
    weakest_label = weakest.get("match_id") or weakest.get("team") or "n/a"
    return "\n".join(
        [
            f"Final verdict: {plan.final_verdict}",
            f"Accepted legs: {plan.number_of_legs}",
            f"Combined probability: {plan.estimated_combined_probability:.6f}",
            f"Weakest leg: {weakest_label}",
            f"Duplicate games: {', '.join(plan.duplicate_game_warnings) or 'none'}",
            f"Contradictory picks: {', '.join(plan.contradictory_picks) or 'none'}",
            f"Correlation warnings: {', '.join(plan.correlation_warnings) or 'none'}",
            f"Notes: {' | '.join(plan.notes) or 'n/a'}",
        ]
    )


def build_committee_demo_examples() -> dict[str, Any]:
    accepted_bet = _run_pick(_base_candidate())
    no_bet_low_edge = _run_pick(_base_candidate(edge=0.02))

    wait_for_lineups = _run_pick(
        {
            **_base_candidate(kickoff_hours=0.75),
            "scraped_context": {
                **_base_candidate(kickoff_hours=0.75)["scraped_context"],
                "home_lineup_confirmed": 0,
                "away_lineup_confirmed": 0,
                "home_likely_starters_count": 0,
                "away_likely_starters_count": 0,
            },
        }
    )

    avoid_both_disagree = _run_pick(
        {
            **_base_candidate(edge=0.01),
            "scraped_context": {
                **_base_candidate(edge=0.01)["scraped_context"],
                "home_priority_absences_count": 2,
                "home_suspensions_count": 1,
                "nothing_to_play_for": 1,
            },
        }
    )

    substitute_bet_accepted = _run_pick(
        {
            **_base_candidate(recommended_market="double_chance"),
            "substitute_candidate": _base_candidate(
                team="Alpha FC or Draw",
                market="double_chance",
                recommended_market="double_chance",
                odds=1.82,
                minimum_acceptable_odds=1.75,
                ml_prob=0.64,
                market_implied_prob=0.5495,
                vig_free_implied_prob=0.56,
                fair_odds=1.563,
                edge=0.08,
                confidence_range_low=0.60,
                confidence_range_high=0.68,
            ),
        }
    )

    substitute_rejected_no_edge = _run_pick(
        {
            **_base_candidate(recommended_market="double_chance"),
            "substitute_candidate": _base_candidate(
                team="Alpha FC or Draw",
                market="double_chance",
                recommended_market="double_chance",
                odds=1.72,
                minimum_acceptable_odds=1.75,
                ml_prob=0.57,
                market_implied_prob=0.5814,
                vig_free_implied_prob=0.56,
                fair_odds=1.754,
                edge=0.01,
                confidence_range_low=0.54,
                confidence_range_high=0.60,
            ),
        }
    )

    stale_data_hold = _run_pick({**_base_candidate(), "stale_line": True})
    short_odds_no_bet = _run_pick(_base_candidate(odds=1.70))

    blind_opposite_side_block = _run_pick(
        {
            **_base_candidate(recommended_market="double_chance"),
            "substitute_candidate": _base_candidate(
                team="Beta FC",
                market="moneyline",
                recommended_market="moneyline",
                home="Alpha FC",
                away="Beta FC",
            ),
        }
    )

    builder = CommitteeParlayBuilder()

    accepted_leg_two = _run_pick(_mlb_complete_candidate())
    accepted_leg_three = _run_pick(
        _nhl_complete_candidate()
    )
    rejected_parlay_leg = builder.build(
        [
            _parlay_entry(accepted_bet),
            _parlay_entry(accepted_leg_two),
            _parlay_entry(wait_for_lineups),
        ],
        parlay_name="Rejected leg demo",
        parlay_type="conservative",
    )

    opposite_side_accepted = _run_pick(
        _base_candidate(
            home="Alpha FC",
            away="Beta FC",
            team="Beta FC",
            ml_prob=0.59,
            market_implied_prob=0.5200,
            vig_free_implied_prob=0.50,
            edge=0.09,
        )
    )
    rejected_parlay_duplicate_conflict = builder.build(
        [
            _parlay_entry(accepted_bet),
            _parlay_entry(opposite_side_accepted),
            _parlay_entry(accepted_leg_two),
        ],
        parlay_name="Duplicate/conflict demo",
        parlay_type="conservative",
    )

    six_leg_entries = [
        _parlay_entry(_run_pick(_base_candidate(home="A1", away="B1", team="A1", sport="soccer"))),
        _parlay_entry(_run_pick(_mlb_complete_candidate(home="A2", away="B2", team="A2"))),
        _parlay_entry(_run_pick(_nhl_complete_candidate(home="A3", away="B3", team="A3"))),
        _parlay_entry(_run_pick(_basketball_complete_candidate(home="A4", away="B4", team="A4"))),
        _parlay_entry(_run_pick(_base_candidate(home="A5", away="B5", team="A5", sport="soccer", market="double_chance"))),
        _parlay_entry(_run_pick(_mlb_complete_candidate(home="A6", away="B6", team="A6", market="spreads"))),
    ]
    conservative_over_cap = builder.build(
        six_leg_entries,
        parlay_name="Six-leg cap demo",
        parlay_type="conservative",
    )

    return {
        "accepted_bet": accepted_bet,
        "no_bet_low_edge": no_bet_low_edge,
        "wait_for_lineups": wait_for_lineups,
        "avoid_both_disagree": avoid_both_disagree,
        "substitute_bet_accepted": substitute_bet_accepted,
        "substitute_rejected_no_edge": substitute_rejected_no_edge,
        "stale_data_hold": stale_data_hold,
        "short_odds_no_bet": short_odds_no_bet,
        "blind_opposite_side_block": blind_opposite_side_block,
        "rejected_parlay_leg": {
            "plan": rejected_parlay_leg,
            "formatted_output": _parlay_summary(rejected_parlay_leg),
        },
        "rejected_parlay_duplicate_conflict": {
            "plan": rejected_parlay_duplicate_conflict,
            "formatted_output": _parlay_summary(rejected_parlay_duplicate_conflict),
        },
        "conservative_over_cap": {
            "plan": conservative_over_cap,
            "formatted_output": _parlay_summary(conservative_over_cap),
        },
    }


def render_committee_demo_examples() -> str:
    examples = build_committee_demo_examples()
    ordered_keys = [
        "accepted_bet",
        "no_bet_low_edge",
        "wait_for_lineups",
        "avoid_both_disagree",
        "substitute_bet_accepted",
        "substitute_rejected_no_edge",
        "stale_data_hold",
        "short_odds_no_bet",
        "blind_opposite_side_block",
        "rejected_parlay_leg",
        "rejected_parlay_duplicate_conflict",
        "conservative_over_cap",
    ]
    sections: list[str] = []
    for key in ordered_keys:
        payload = examples[key]
        sections.append(f"[{key}]")
        sections.append(str(payload["formatted_output"]).strip())
        sections.append("")
    return "\n".join(sections).strip()


__all__ = [
    "build_committee_demo_examples",
    "render_committee_demo_examples",
]
