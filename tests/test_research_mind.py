from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.committee import ContextResearchMind, ResearchVerdict, VetoFlag


def _base_candidate() -> dict:
    kickoff = datetime.now(timezone.utc) + timedelta(hours=8)
    bookmaker_update = datetime.now(timezone.utc) - timedelta(hours=1)
    return {
        "sport": "soccer",
        "market": "moneyline",
        "team": "Alpha FC",
        "home": "Alpha FC",
        "away": "Beta FC",
        "status": "scheduled",
        "commence_time": kickoff.isoformat(),
        "odds_snapshot_age_hours": 1.0,
        "odds_source_status": "live_api",
        "odds_fetched_at": datetime.now(timezone.utc).isoformat(),
        "bookmaker_last_update": bookmaker_update.isoformat(),
        "scraped_context": {
            "home_team_name": "Alpha FC",
            "away_team_name": "Beta FC",
            "availability_source": "api_football",
            "availability_fetched_at": datetime.now(timezone.utc).isoformat(),
        },
        "scraped_context_sources": ["api_football"],
        "scraped_context_highlights": ["Availability context fetched from api_football"],
    }


def test_research_mind_holds_when_odds_are_stale() -> None:
    candidate = _base_candidate()
    candidate["stale_line"] = True

    decision = ContextResearchMind().evaluate(candidate)

    assert decision.research_verdict == ResearchVerdict.HOLD
    assert VetoFlag.STALE_ODDS in decision.veto_flags
    assert decision.data_freshness == "stale"


def test_research_mind_holds_and_signals_wait_for_lineups_near_kickoff() -> None:
    candidate = _base_candidate()
    candidate["commence_time"] = (datetime.now(timezone.utc) + timedelta(minutes=45)).isoformat()
    candidate["scraped_context"] = {
        **candidate["scraped_context"],
        "home_lineup_confirmed": 0,
        "away_lineup_confirmed": 0,
        "home_likely_starters_count": 0,
        "away_likely_starters_count": 0,
    }

    decision = ContextResearchMind().evaluate(candidate)

    assert decision.research_verdict == ResearchVerdict.HOLD
    assert decision.wait_for_lineups_signal is True
    assert VetoFlag.MISSING_LINEUPS in decision.veto_flags


def test_research_mind_avoids_finished_match() -> None:
    candidate = _base_candidate()
    candidate["status"] = "finished"

    decision = ContextResearchMind().evaluate(candidate)

    assert decision.research_verdict == ResearchVerdict.AVOID
    assert VetoFlag.FINISHED_MATCH in decision.veto_flags


def test_research_mind_holds_for_unclear_fixture() -> None:
    candidate = _base_candidate()
    candidate["scraped_context"] = {
        **candidate["scraped_context"],
        "home_team_name": "Wrong Home",
        "away_team_name": "Beta FC",
    }

    decision = ContextResearchMind().evaluate(candidate)

    assert decision.research_verdict == ResearchVerdict.HOLD
    assert VetoFlag.UNCLEAR_FIXTURE in decision.veto_flags


def test_research_mind_avoids_severe_rotation_risk() -> None:
    candidate = _base_candidate()
    candidate["commence_time"] = (datetime.now(timezone.utc) + timedelta(minutes=75)).isoformat()
    candidate["scraped_context"] = {
        **candidate["scraped_context"],
        "cup_rotation_risk": 1,
        "final_day_volatility": 1,
        "home_lineup_confirmed": 0,
        "away_lineup_confirmed": 0,
        "home_likely_starters_count": 0,
        "away_likely_starters_count": 0,
    }

    decision = ContextResearchMind().evaluate(candidate)

    assert decision.research_verdict == ResearchVerdict.AVOID
    assert VetoFlag.HIGH_ROTATION_RISK in decision.veto_flags
    assert VetoFlag.END_SEASON_CHAOS in decision.veto_flags


def test_research_mind_disagrees_when_availability_risk_is_material() -> None:
    candidate = _base_candidate()
    candidate["scraped_context"] = {
        **candidate["scraped_context"],
        "home_priority_absences_count": 2,
        "home_suspensions_count": 1,
    }

    decision = ContextResearchMind().evaluate(candidate)

    assert decision.research_verdict == ResearchVerdict.DISAGREE
    assert any("Injuries or suspensions" in item for item in decision.main_risks)


def test_research_mind_agrees_when_context_is_fresh_and_supportive() -> None:
    candidate = _base_candidate()
    candidate["scraped_context"] = {
        **candidate["scraped_context"],
        "is_playoff": 1,
        "playoff_motivation": 1,
    }
    candidate["scraped_context_highlights"] = [
        "Playoff spot pressure favors the selected side",
        "Availability context fetched from api_football",
    ]

    decision = ContextResearchMind().evaluate(candidate)

    assert decision.research_verdict == ResearchVerdict.AGREE
    assert decision.confidence in {"Medium", "High"}
    assert "api_football" in decision.sources_checked
    assert decision.data_freshness == "acceptable_freshness"


def test_research_mind_model_derived_evidence_only_caps_confidence_at_medium() -> None:
    candidate = _base_candidate()
    candidate["scraped_context"] = {}
    candidate["scraped_context_sources"] = []
    candidate["scraped_context_highlights"] = ["Soccer matchup edge from expected-goals profile and chance-quality shape."]
    candidate["prediction_factors"] = [{"summary": "Attack-vs-defense clash adjustment from rolling scoring and concession profiles."}]
    candidate.pop("odds_snapshot_age_hours", None)
    candidate.pop("bookmaker_last_update", None)

    decision = ContextResearchMind().evaluate(candidate)

    assert decision.confidence in {"Medium", "Low"}
    assert decision.source_count == 0
    assert decision.evidence_status == "INSUFFICIENT"


def test_research_mind_no_source_count_caps_confidence() -> None:
    candidate = _base_candidate()
    candidate["scraped_context"] = {}
    candidate["scraped_context_sources"] = []
    candidate["scraped_context_highlights"] = []
    candidate.pop("odds_snapshot_age_hours", None)
    candidate.pop("bookmaker_last_update", None)

    decision = ContextResearchMind().evaluate(candidate)

    assert decision.source_count == 0
    assert decision.confidence in {"Medium", "Low"}


def test_research_mind_missing_injury_check_for_lineup_sensitive_market_blocks_high_confidence() -> None:
    candidate = _base_candidate()
    candidate["scraped_context"] = {
        "home_team_name": "Alpha FC",
        "away_team_name": "Beta FC",
    }
    candidate["scraped_context_sources"] = []
    candidate["scraped_context_highlights"] = []

    decision = ContextResearchMind().evaluate(candidate)

    assert decision.injury_status == "not_checked"
    assert len(decision.sport_specific_missing_evidence) >= 1
    assert decision.confidence != "High"


def test_research_mind_allows_high_only_with_complete_concrete_checks() -> None:
    candidate = _base_candidate()
    candidate["scraped_context"] = {
        **candidate["scraped_context"],
        "availability_source": "api_football",
    }
    candidate["scraped_context_sources"] = ["api_football", "espn"]
    candidate["scraped_context_highlights"] = [
        "Availability context fetched from api_football",
        "Team news cross-check available from ESPN preview coverage",
    ]

    decision = ContextResearchMind().evaluate(candidate)

    assert decision.fixture_verified is True
    assert decision.sport == "soccer"
    assert decision.odds_freshness_status in {"fresh", "acceptable"}
    assert decision.market_availability_status == "available"
    assert decision.source_count >= 2
    assert decision.source_quality_summary != "weak"
    assert decision.confidence == "High"


def test_research_mind_continental_fixture_without_rotation_check_blocks_high_confidence() -> None:
    candidate = _base_candidate()
    candidate["scraped_context"] = {
        "home_team_name": "Alpha FC",
        "away_team_name": "Beta FC",
        "european_rotation_risk": 1,
    }
    candidate["scraped_context_sources"] = ["odds_snapshot"]
    candidate["scraped_context_highlights"] = []

    decision = ContextResearchMind().evaluate(candidate)

    assert decision.rotation_status == "not_checked"
    assert decision.confidence != "High"


def test_research_mind_bookmaker_only_soccer_sources_stay_weak() -> None:
    candidate = _base_candidate()
    candidate["scraped_context"] = {
        "home_team_name": "Alpha FC",
        "away_team_name": "Beta FC",
    }
    candidate["scraped_context_sources"] = ["bookmaker"]
    candidate["scraped_context_highlights"] = ["Expected-goals profile likes the home side."]

    decision = ContextResearchMind().evaluate(candidate)

    assert decision.source_quality_summary == "weak"
    assert decision.confidence != "High"
    assert decision.evidence_status in {"INSUFFICIENT", "PARTIAL"}


def test_research_mind_soccer_availability_context_updates_live_statuses() -> None:
    candidate = _base_candidate()
    candidate["scraped_context"] = {
        **candidate["scraped_context"],
        "availability_source": "api_football",
        "lineup_source": "api_football",
        "home_likely_starters_count": 11,
        "away_likely_starters_count": 11,
        "rotation_checked": 1,
        "motivation_checked": 1,
    }
    candidate["scraped_context_sources"] = ["api_football", "football_data"]

    decision = ContextResearchMind().evaluate(candidate)

    assert decision.lineup_status == "monitor"
    assert decision.injury_status in {"checked", "checked_fresh"}
    assert decision.source_quality_summary in {"mixed", "strong"}


def test_research_mind_uses_soccer_enrichment_status_markers() -> None:
    candidate = _base_candidate()
    candidate["scraped_context"] = {
        **candidate["scraped_context"],
        "soccer_lineup_status": "projected",
        "soccer_probable_lineup_status": "projected",
        "soccer_injury_status": "checked",
        "soccer_motivation_status": "checked",
        "soccer_rotation_status": "checked",
        "team_news_checked": 1,
        "probable_lineups_checked": 1,
    }
    candidate["scraped_context_sources"] = ["sportsmole.co.uk", "soccer_feature_cache"]

    decision = ContextResearchMind().evaluate(candidate)

    assert decision.lineup_status == "monitor"
    assert decision.injury_status in {"checked", "checked_fresh"}
    assert not any("injury/team news was not checked" in item for item in decision.missing_evidence)


def test_research_mind_risks_use_no_major_risks_only_after_concrete_checks() -> None:
    candidate = _base_candidate()
    candidate["scraped_context"] = {
        **candidate["scraped_context"],
        "availability_source": "api_football",
    }
    candidate["scraped_context_sources"] = ["api_football", "espn"]
    candidate["scraped_context_highlights"] = ["Availability context fetched from api_football"]

    decision = ContextResearchMind().evaluate(candidate)

    assert "No major risks detected from available evidence" in decision.main_risks


def test_research_mind_flags_limited_concrete_research_evidence_when_checks_are_thin() -> None:
    candidate = _base_candidate()
    candidate["scraped_context"] = {}
    candidate["scraped_context_sources"] = []
    candidate["scraped_context_highlights"] = []
    candidate.pop("odds_snapshot_age_hours", None)
    candidate.pop("bookmaker_last_update", None)

    decision = ContextResearchMind().evaluate(candidate)

    assert any("limited concrete research evidence" in item for item in decision.main_risks)


def test_research_mind_deduplicates_sources_checked_while_preserving_order() -> None:
    candidate = _base_candidate()
    candidate["scraped_context_sources"] = ["api_football", "odds_snapshot", "api_football"]

    decision = ContextResearchMind().evaluate(candidate)

    assert decision.sources_checked == ("api_football", "odds_snapshot")


def test_research_mind_uses_verified_fresh_only_when_all_channels_are_fresh() -> None:
    candidate = _base_candidate()
    candidate["commence_time"] = (datetime.now(timezone.utc) + timedelta(minutes=75)).isoformat()
    candidate["scraped_context"] = {
        **candidate["scraped_context"],
        "home_lineup_confirmed": 1,
        "away_lineup_confirmed": 1,
        "availability_fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    candidate["standings_snapshot_age_hours"] = 2.0

    decision = ContextResearchMind().evaluate(candidate)

    assert decision.data_freshness == "verified_fresh"


def test_research_mind_basketball_waits_for_lineups_when_no_concrete_report_near_tipoff() -> None:
    candidate = _base_candidate()
    candidate["sport"] = "basketball"
    candidate["home"] = "Boston Celtics"
    candidate["away"] = "New York Knicks"
    candidate["team"] = "Boston Celtics"
    candidate["commence_time"] = (datetime.now(timezone.utc) + timedelta(minutes=45)).isoformat()
    candidate["scraped_context"] = {
        "home_team_name": "Boston Celtics",
        "away_team_name": "New York Knicks",
        "availability_source": "feature_snapshot",
    }
    candidate["scraped_context_sources"] = ["feature_snapshot"]
    candidate["scraped_context_highlights"] = []

    decision = ContextResearchMind().evaluate(candidate)

    assert decision.research_verdict == ResearchVerdict.HOLD
    assert decision.wait_for_lineups_signal is True
    assert decision.lineup_status == "missing_near_kickoff"
    assert VetoFlag.MISSING_LINEUPS in decision.veto_flags


def test_research_mind_basketball_near_tipoff_with_concrete_report_stays_below_high_without_extra_sources() -> None:
    candidate = _base_candidate()
    candidate["sport"] = "basketball"
    candidate["home"] = "Boston Celtics"
    candidate["away"] = "New York Knicks"
    candidate["team"] = "Boston Celtics"
    candidate["commence_time"] = (datetime.now(timezone.utc) + timedelta(minutes=50)).isoformat()
    candidate["scraped_context"] = {
        "home_team_name": "Boston Celtics",
        "away_team_name": "New York Knicks",
        "availability_source": "api_sports_basketball",
        "lineup_source": "api_sports_basketball",
        "availability_fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    candidate["scraped_context_sources"] = ["api_sports_basketball"]
    candidate["scraped_context_highlights"] = ["NBA injury report checked close to tip-off"]

    decision = ContextResearchMind().evaluate(candidate)

    assert decision.lineup_status == "monitor"
    assert decision.injury_status == "checked_fresh"
    assert decision.confidence != "High"


def test_research_mind_allows_high_for_basketball_only_with_full_nba_context() -> None:
    candidate = _base_candidate()
    candidate["sport"] = "basketball"
    candidate["home"] = "Boston Celtics"
    candidate["away"] = "New York Knicks"
    candidate["team"] = "Boston Celtics"
    candidate["odds"] = 1.94
    candidate["commence_time"] = (datetime.now(timezone.utc) + timedelta(hours=5)).isoformat()
    candidate["scraped_context"] = {
        "home_team_name": "Boston Celtics",
        "away_team_name": "New York Knicks",
        "availability_source": "api_sports_basketball",
        "lineup_source": "api_sports_basketball",
        "availability_fetched_at": datetime.now(timezone.utc).isoformat(),
        "home_priority_absences_count": 0,
        "away_priority_absences_count": 0,
        "home_questionable_count": 0,
        "away_questionable_count": 0,
    }
    candidate["scraped_context_sources"] = ["api_sports_basketball", "balldontlie", "espn"]
    candidate["scraped_context_highlights"] = [
        "NBA injury report checked with projected lineup context",
        "Travel and rest spot cross-checked from the latest schedule feed",
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

    decision = ContextResearchMind().evaluate(candidate)

    assert decision.confidence == "High"
    assert decision.evidence_status in {"COMPLETE", "ACCEPTABLE"}


def test_research_mind_holds_basketball_when_star_player_status_is_uncertain() -> None:
    candidate = _base_candidate()
    candidate["sport"] = "basketball"
    candidate["home"] = "Boston Celtics"
    candidate["away"] = "New York Knicks"
    candidate["team"] = "Boston Celtics"
    candidate["odds"] = 1.94
    candidate["commence_time"] = (datetime.now(timezone.utc) + timedelta(hours=3)).isoformat()
    candidate["scraped_context"] = {
        "home_team_name": "Boston Celtics",
        "away_team_name": "New York Knicks",
        "availability_source": "api_sports_basketball",
        "lineup_source": "api_sports_basketball",
        "availability_fetched_at": datetime.now(timezone.utc).isoformat(),
        "home_priority_absences_count": 1,
        "home_questionable_count": 1,
        "away_priority_absences_count": 0,
        "away_questionable_count": 0,
    }
    candidate["scraped_context_sources"] = ["api_sports_basketball", "espn"]
    candidate["context_adjustments"] = [
        {"name": "lineup_uncertainty", "summary": "A core player remains questionable close to tip-off."},
        {"name": "back_to_back", "summary": "Rest context checked."},
        {"name": "travel_fatigue", "summary": "Travel context checked."},
        {"name": "closing_execution", "summary": "Ratings context checked."},
    ]

    decision = ContextResearchMind().evaluate(candidate)

    assert decision.research_verdict == ResearchVerdict.HOLD
    assert any("star-player injury status is still uncertain" in item.lower() for item in decision.missing_evidence)


def test_research_mind_basketball_back_to_back_without_rest_check_blocks_high_confidence() -> None:
    candidate = _base_candidate()
    candidate["sport"] = "basketball"
    candidate["home"] = "Boston Celtics"
    candidate["away"] = "New York Knicks"
    candidate["team"] = "Boston Celtics"
    candidate["odds"] = 1.94
    candidate["commence_time"] = (datetime.now(timezone.utc) + timedelta(hours=4)).isoformat()
    candidate["scraped_context"] = {
        "home_team_name": "Boston Celtics",
        "away_team_name": "New York Knicks",
        "availability_source": "api_sports_basketball",
        "lineup_source": "api_sports_basketball",
        "availability_fetched_at": datetime.now(timezone.utc).isoformat(),
        "fixture_congestion_risk": 1,
        "home_priority_absences_count": 0,
        "away_priority_absences_count": 0,
        "home_questionable_count": 0,
        "away_questionable_count": 0,
    }
    candidate["scraped_context_sources"] = ["api_sports_basketball", "balldontlie", "espn"]
    candidate["prediction_factors"] = [
        {"name": "home_ortg", "summary": "Home offensive rating profile is trending above baseline."},
        {"name": "away_drtg", "summary": "Away defensive rating profile is softer than league average."},
    ]
    candidate["context_adjustments"] = [
        {"name": "travel_fatigue", "summary": "Travel context checked."},
        {"name": "pace_control", "summary": "Projected pace environment checked."},
        {"name": "closing_execution", "summary": "Ratings context checked."},
        {"name": "injury_report_edge", "summary": "Injury report checked."},
        {"name": "rotation_quality_edge", "summary": "Rotation checked."},
    ]

    decision = ContextResearchMind().evaluate(candidate)

    assert decision.confidence != "High"
    assert any("rest/back-to-back context was not checked" in item.lower() for item in decision.missing_evidence)


def test_research_mind_holds_basketball_team_total_without_player_status_context() -> None:
    candidate = _base_candidate()
    candidate["sport"] = "basketball"
    candidate["market"] = "team_total"
    candidate["home"] = "Boston Celtics"
    candidate["away"] = "New York Knicks"
    candidate["team"] = "Boston Celtics Team Total Over"
    candidate["odds"] = 1.91
    candidate["commence_time"] = (datetime.now(timezone.utc) + timedelta(hours=3)).isoformat()
    candidate["scraped_context"] = {
        "home_team_name": "Boston Celtics",
        "away_team_name": "New York Knicks",
        "lineup_source": "api_sports_basketball",
        "availability_fetched_at": datetime.now(timezone.utc).isoformat(),
        "home_priority_absences_count": 1,
        "away_priority_absences_count": 0,
        "home_questionable_count": 0,
        "away_questionable_count": 0,
    }
    candidate["scraped_context_sources"] = ["espn", "balldontlie"]
    candidate["context_adjustments"] = [
        {"name": "back_to_back", "summary": "Rest context checked."},
        {"name": "travel_fatigue", "summary": "Travel context checked."},
        {"name": "pace_control", "summary": "Projected pace environment checked."},
        {"name": "closing_execution", "summary": "Ratings context checked."},
    ]

    decision = ContextResearchMind().evaluate(candidate)

    assert decision.research_verdict == ResearchVerdict.HOLD


def test_research_mind_basketball_totals_without_pace_context_downgrades_confidence() -> None:
    candidate = _base_candidate()
    candidate["sport"] = "basketball"
    candidate["market"] = "totals"
    candidate["home"] = "Boston Celtics"
    candidate["away"] = "New York Knicks"
    candidate["team"] = "Over 225.5"
    candidate["odds"] = 1.91
    candidate["commence_time"] = (datetime.now(timezone.utc) + timedelta(hours=4)).isoformat()
    candidate["scraped_context"] = {
        "home_team_name": "Boston Celtics",
        "away_team_name": "New York Knicks",
        "availability_source": "api_sports_basketball",
        "lineup_source": "api_sports_basketball",
        "availability_fetched_at": datetime.now(timezone.utc).isoformat(),
        "home_priority_absences_count": 0,
        "away_priority_absences_count": 0,
        "home_questionable_count": 0,
        "away_questionable_count": 0,
    }
    candidate["scraped_context_sources"] = ["api_sports_basketball", "balldontlie", "espn"]
    candidate["prediction_factors"] = [
        {"name": "home_ortg", "summary": "Home offensive rating profile is trending above baseline."},
        {"name": "away_drtg", "summary": "Away defensive rating profile is softer than league average."},
    ]
    candidate["context_adjustments"] = [
        {"name": "back_to_back", "summary": "Rest context checked."},
        {"name": "travel_fatigue", "summary": "Travel context checked."},
        {"name": "closing_execution", "summary": "Ratings context checked."},
        {"name": "injury_report_edge", "summary": "Injury report checked."},
    ]

    decision = ContextResearchMind().evaluate(candidate)

    assert decision.confidence != "High"
    assert any("pace context was not checked" in item.lower() for item in decision.missing_evidence)


def test_research_mind_allows_high_for_mlb_only_with_confirmed_starters_and_strong_sources() -> None:
    candidate = _base_candidate()
    candidate["sport"] = "mlb"
    candidate["home"] = "Los Angeles Dodgers"
    candidate["away"] = "San Diego Padres"
    candidate["team"] = "Los Angeles Dodgers"
    candidate["commence_time"] = (datetime.now(timezone.utc) + timedelta(hours=4)).isoformat()
    candidate["scraped_context"] = {
        "home_team_name": "Los Angeles Dodgers",
        "away_team_name": "San Diego Padres",
        "availability_source": "mlb_stats_api",
        "lineup_source": "mlb_stats_api",
        "availability_fetched_at": datetime.now(timezone.utc).isoformat(),
        "home_starter_confirmed": 1,
        "away_starter_confirmed": 1,
        "home_starter_name": "Tyler Glasnow",
        "away_starter_name": "Yu Darvish",
        "home_starter_hand": "R",
        "away_starter_hand": "R",
        "home_starter_era": 2.91,
        "away_starter_era": 3.84,
        "home_starter_whip": 0.97,
        "away_starter_whip": 1.18,
        "home_rest_days": 1,
        "away_rest_days": 0,
        "away_travel_km": 195.0,
        "roof_status": "open_air",
    }
    candidate["scraped_context_sources"] = ["mlb_stats_api", "espn", "openweather"]
    candidate["scraped_context_highlights"] = [
        "Dodgers probable starter confirmed by MLB feed",
        "Beat-report preview cross-check available from ESPN",
    ]
    candidate["prediction_factors"] = [
        {"name": "sp_era_diff", "summary": "Starter ERA differential supports the Dodgers."},
        {"name": "sp_whip_diff", "summary": "Starter WHIP differential supports the Dodgers."},
    ]
    candidate["context_adjustments"] = [
        {"name": "bullpen_workload", "summary": "Bullpen workload checked."},
        {"name": "travel_fatigue", "summary": "Travel spot checked."},
    ]

    decision = ContextResearchMind().evaluate(candidate)

    assert decision.lineup_status == "confirmed"
    assert decision.confidence == "High"


def test_research_mind_allows_high_for_nhl_only_with_confirmed_goalies_and_strong_sources() -> None:
    candidate = _base_candidate()
    candidate["sport"] = "nhl"
    candidate["home"] = "Boston Bruins"
    candidate["away"] = "Toronto Maple Leafs"
    candidate["team"] = "Boston Bruins"
    candidate["odds"] = 1.95
    candidate["commence_time"] = (datetime.now(timezone.utc) + timedelta(hours=4)).isoformat()
    candidate["scraped_context"] = {
        "home_team_name": "Boston Bruins",
        "away_team_name": "Toronto Maple Leafs",
        "availability_source": "nhl_api",
        "lineup_source": "nhl_api",
        "availability_fetched_at": datetime.now(timezone.utc).isoformat(),
        "home_goalie_confirmed": 1,
        "away_goalie_confirmed": 1,
        "home_goalie_name": "Jeremy Swayman",
        "away_goalie_name": "Joseph Woll",
    }
    candidate["scraped_context_sources"] = ["nhl_api", "espn"]
    candidate["scraped_context_highlights"] = [
        "Projected starting goalies confirmed by NHL team reports",
        "ESPN matchup preview cross-check available",
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

    decision = ContextResearchMind().evaluate(candidate)

    assert decision.lineup_status == "confirmed"
    assert decision.confidence == "High"


def test_research_mind_holds_nhl_when_goalie_is_unconfirmed_near_puck_drop() -> None:
    candidate = _base_candidate()
    candidate["sport"] = "nhl"
    candidate["home"] = "Boston Bruins"
    candidate["away"] = "Toronto Maple Leafs"
    candidate["team"] = "Boston Bruins"
    candidate["odds"] = 1.95
    candidate["commence_time"] = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
    candidate["scraped_context"] = {
        "home_team_name": "Boston Bruins",
        "away_team_name": "Toronto Maple Leafs",
        "availability_source": "nhl_api",
        "lineup_source": "nhl_api",
        "availability_fetched_at": datetime.now(timezone.utc).isoformat(),
        "home_goalie_confirmed": 0,
        "away_goalie_confirmed": 1,
        "home_goalie_name": "Jeremy Swayman",
        "away_goalie_name": "Joseph Woll",
    }
    candidate["scraped_context_sources"] = ["nhl_api", "espn"]

    decision = ContextResearchMind().evaluate(candidate)

    assert decision.research_verdict == ResearchVerdict.HOLD
    assert decision.lineup_status == "missing_near_kickoff"
    assert VetoFlag.MISSING_LINEUPS in decision.veto_flags


def test_research_mind_nhl_back_to_back_without_rest_check_downgrades_confidence() -> None:
    candidate = _base_candidate()
    candidate["sport"] = "nhl"
    candidate["home"] = "Boston Bruins"
    candidate["away"] = "Toronto Maple Leafs"
    candidate["team"] = "Boston Bruins"
    candidate["odds"] = 1.95
    candidate["scraped_context"] = {
        "home_team_name": "Boston Bruins",
        "away_team_name": "Toronto Maple Leafs",
        "availability_source": "nhl_api",
        "lineup_source": "nhl_api",
        "availability_fetched_at": datetime.now(timezone.utc).isoformat(),
        "home_goalie_confirmed": 1,
        "away_goalie_confirmed": 1,
        "home_goalie_name": "Jeremy Swayman",
        "away_goalie_name": "Joseph Woll",
        "fixture_congestion_risk": 1,
    }
    candidate["scraped_context_sources"] = ["nhl_api", "espn"]
    candidate["prediction_factors"] = [
        {"name": "home_xg_diff_10", "summary": "Recent expected-goals differential is favorable."},
    ]
    candidate["context_adjustments"] = [
        {"name": "travel_fatigue", "summary": "Travel fatigue checked."},
        {"name": "special_teams_edge", "summary": "Special teams checked."},
        {"name": "xg_structure", "summary": "Shot-quality structure checked."},
        {"name": "system_stability", "summary": "Home/away split checked."},
    ]

    decision = ContextResearchMind().evaluate(candidate)

    assert decision.confidence != "High"
    assert any("rest/back-to-back" in item.lower() for item in decision.missing_evidence)


def test_research_mind_nhl_unclear_injuries_block_high_confidence() -> None:
    candidate = _base_candidate()
    candidate["sport"] = "nhl"
    candidate["home"] = "Boston Bruins"
    candidate["away"] = "Toronto Maple Leafs"
    candidate["team"] = "Boston Bruins"
    candidate["odds"] = 1.95
    candidate["scraped_context"] = {
        "home_team_name": "Boston Bruins",
        "away_team_name": "Toronto Maple Leafs",
        "lineup_source": "nhl_api",
        "availability_fetched_at": datetime.now(timezone.utc).isoformat(),
        "home_goalie_confirmed": 1,
        "away_goalie_confirmed": 1,
        "home_goalie_name": "Jeremy Swayman",
        "away_goalie_name": "Joseph Woll",
    }
    candidate["scraped_context_sources"] = ["nhl_api", "espn"]
    candidate["prediction_factors"] = [
        {"name": "home_xg_diff_10", "summary": "Recent expected-goals differential is favorable."},
        {"name": "away_goal_diff_10", "summary": "Opponent recent goal differential is softer."},
    ]
    candidate["context_adjustments"] = [
        {"name": "back_to_back", "summary": "Rest/back-to-back context checked."},
        {"name": "travel_fatigue", "summary": "Travel fatigue context checked."},
        {"name": "special_teams_edge", "summary": "Special teams checked."},
        {"name": "xg_structure", "summary": "Shot-quality structure checked."},
        {"name": "system_stability", "summary": "Home/away split context checked."},
    ]

    decision = ContextResearchMind().evaluate(candidate)

    assert decision.injury_status == "not_checked"
    assert decision.confidence != "High"


def test_research_mind_holds_nhl_totals_without_confirmed_goalie_context() -> None:
    candidate = _base_candidate()
    candidate["sport"] = "nhl"
    candidate["market"] = "totals"
    candidate["team"] = "Over 5.5"
    candidate["home"] = "Boston Bruins"
    candidate["away"] = "Toronto Maple Leafs"
    candidate["odds"] = 1.91
    candidate["scraped_context"] = {
        "home_team_name": "Boston Bruins",
        "away_team_name": "Toronto Maple Leafs",
        "availability_source": "nhl_api",
        "lineup_source": "nhl_api",
        "availability_fetched_at": datetime.now(timezone.utc).isoformat(),
        "home_goalie_confirmed": 0,
        "away_goalie_confirmed": 1,
        "home_goalie_name": "Jeremy Swayman",
        "away_goalie_name": "Joseph Woll",
    }
    candidate["scraped_context_sources"] = ["nhl_api", "espn"]

    decision = ContextResearchMind().evaluate(candidate)

    assert decision.research_verdict == ResearchVerdict.HOLD
    assert any("goalie context" in item.lower() for item in decision.missing_evidence)


def test_research_mind_tennis_stays_conservative_even_with_good_sources() -> None:
    candidate = _base_candidate()
    candidate["sport"] = "tennis"
    candidate["home"] = "Player One"
    candidate["away"] = "Player Two"
    candidate["team"] = "Player One"
    candidate["scraped_context"] = {
        "home_team_name": "Player One",
        "away_team_name": "Player Two",
        "availability_source": "espn",
        "availability_fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    candidate["scraped_context_sources"] = ["espn", "newsapi"]
    candidate["scraped_context_highlights"] = [
        "Tournament preview confirms the scheduled matchup",
        "Player-condition note was checked from preview coverage",
    ]

    decision = ContextResearchMind().evaluate(candidate)

    assert decision.confidence != "High"
    assert decision.evidence_status != "COMPLETE"


def test_research_mind_holds_tennis_when_injury_or_retirement_concern_is_unverified() -> None:
    candidate = _base_candidate()
    candidate["sport"] = "tennis"
    candidate["home"] = "Player One"
    candidate["away"] = "Player Two"
    candidate["team"] = "Player One"
    candidate["odds"] = 1.98
    candidate["scraped_context"] = {
        "home_team_name": "Player One",
        "away_team_name": "Player Two",
        "availability_source": "espn",
        "availability_fetched_at": datetime.now(timezone.utc).isoformat(),
        "injury_concern": 1,
        "round": "Quarterfinal",
        "tournament": "ATP Rome",
    }
    candidate["scraped_context_sources"] = ["espn", "newsapi"]
    candidate["prediction_factors"] = [
        {"name": "surface_win_diff", "summary": "Surface win-rate edge: +0.120"},
        {"name": "form_diff", "summary": "Recent form edge: +0.210"},
        {"name": "h2h_p1_win_rate", "summary": "Head-to-head edge: +0.600"},
        {"name": "rank_log_ratio", "summary": "Ranking edge: +0.320"},
    ]

    decision = ContextResearchMind().evaluate(candidate)

    assert decision.research_verdict == ResearchVerdict.HOLD
    assert any("injury/retirement concern could not be verified" in item.lower() for item in decision.missing_evidence)


def test_research_mind_tennis_surface_missing_blocks_high_confidence() -> None:
    candidate = _base_candidate()
    candidate["sport"] = "tennis"
    candidate["home"] = "Player One"
    candidate["away"] = "Player Two"
    candidate["team"] = "Player One"
    candidate["odds"] = 1.98
    candidate["scraped_context"] = {
        "home_team_name": "Player One",
        "away_team_name": "Player Two",
        "availability_source": "espn",
        "availability_fetched_at": datetime.now(timezone.utc).isoformat(),
        "fatigue_checked": 1,
        "round": "Quarterfinal",
        "tournament": "ATP Rome",
        "travel_checked": 1,
        "injury_concern_checked": 1,
    }
    candidate["scraped_context_sources"] = ["espn", "newsapi"]
    candidate["prediction_factors"] = [
        {"name": "form_diff", "summary": "Recent form edge: +0.210"},
        {"name": "serve_diff", "summary": "Serve quality edge: +0.180"},
        {"name": "h2h_p1_win_rate", "summary": "Head-to-head edge: +0.600"},
        {"name": "rank_log_ratio", "summary": "Ranking edge: +0.320"},
    ]

    decision = ContextResearchMind().evaluate(candidate)

    assert decision.confidence != "High"
    assert any("surface context was not checked" in item.lower() for item in decision.missing_evidence)


def test_research_mind_tennis_long_recent_match_without_fatigue_check_downgrades() -> None:
    candidate = _base_candidate()
    candidate["sport"] = "tennis"
    candidate["home"] = "Player One"
    candidate["away"] = "Player Two"
    candidate["team"] = "Player One"
    candidate["odds"] = 1.98
    candidate["scraped_context"] = {
        "home_team_name": "Player One",
        "away_team_name": "Player Two",
        "availability_source": "espn",
        "availability_fetched_at": datetime.now(timezone.utc).isoformat(),
        "recent_long_match": 1,
        "round": "Semifinal",
        "tournament": "ATP Madrid",
        "travel_checked": 1,
        "injury_concern_checked": 1,
        "surface": "Clay",
    }
    candidate["scraped_context_sources"] = ["espn", "newsapi"]
    candidate["prediction_factors"] = [
        {"name": "surface_win_diff", "summary": "Surface win-rate edge: +0.120"},
        {"name": "form_diff", "summary": "Recent form edge: +0.210"},
        {"name": "serve_diff", "summary": "Serve quality edge: +0.180"},
        {"name": "h2h_p1_win_rate", "summary": "Head-to-head edge: +0.600"},
        {"name": "rank_log_ratio", "summary": "Ranking edge: +0.320"},
    ]

    decision = ContextResearchMind().evaluate(candidate)

    assert decision.confidence != "High"
    assert any("fatigue from recent matches was not checked" in item.lower() for item in decision.missing_evidence)


def test_research_mind_tennis_totals_without_serve_return_context_downgrades() -> None:
    candidate = _base_candidate()
    candidate["sport"] = "tennis"
    candidate["market"] = "totals"
    candidate["home"] = "Player One"
    candidate["away"] = "Player Two"
    candidate["team"] = "Over 22.5"
    candidate["odds"] = 1.95
    candidate["scraped_context"] = {
        "home_team_name": "Player One",
        "away_team_name": "Player Two",
        "availability_source": "espn",
        "availability_fetched_at": datetime.now(timezone.utc).isoformat(),
        "fatigue_checked": 1,
        "round": "Quarterfinal",
        "tournament": "ATP Rome",
        "travel_checked": 1,
        "injury_concern_checked": 1,
        "surface": "Clay",
    }
    candidate["scraped_context_sources"] = ["espn", "newsapi"]
    candidate["prediction_factors"] = [
        {"name": "surface_win_diff", "summary": "Surface win-rate edge: +0.120"},
        {"name": "form_diff", "summary": "Recent form edge: +0.210"},
        {"name": "h2h_p1_win_rate", "summary": "Head-to-head edge: +0.600"},
        {"name": "rank_log_ratio", "summary": "Ranking edge: +0.320"},
    ]

    decision = ContextResearchMind().evaluate(candidate)

    assert decision.confidence != "High"
    assert any("serve/return matchup context was not checked" in item.lower() for item in decision.missing_evidence)


def test_research_mind_holds_tennis_when_match_is_live_but_candidate_uses_pre_match_odds() -> None:
    candidate = _base_candidate()
    candidate["sport"] = "tennis"
    candidate["status"] = "live"
    candidate["home"] = "Player One"
    candidate["away"] = "Player Two"
    candidate["team"] = "Player One"
    candidate["odds"] = 1.98
    candidate["scraped_context"] = {
        "home_team_name": "Player One",
        "away_team_name": "Player Two",
        "availability_source": "espn",
        "availability_fetched_at": datetime.now(timezone.utc).isoformat(),
        "fatigue_checked": 1,
        "round": "Quarterfinal",
        "tournament": "ATP Rome",
        "travel_checked": 1,
        "injury_concern_checked": 1,
        "surface": "Clay",
    }
    candidate["scraped_context_sources"] = ["espn", "newsapi"]
    candidate["prediction_factors"] = [
        {"name": "surface_win_diff", "summary": "Surface win-rate edge: +0.120"},
        {"name": "form_diff", "summary": "Recent form edge: +0.210"},
        {"name": "serve_diff", "summary": "Serve quality edge: +0.180"},
        {"name": "h2h_p1_win_rate", "summary": "Head-to-head edge: +0.600"},
        {"name": "rank_log_ratio", "summary": "Ranking edge: +0.320"},
    ]

    decision = ContextResearchMind().evaluate(candidate)

    assert decision.research_verdict == ResearchVerdict.HOLD
    assert any("pre-match odds context" in item.lower() for item in decision.missing_evidence)


def test_research_mind_marks_missing_sources_as_insufficiently_verified() -> None:
    candidate = _base_candidate()
    candidate["scraped_context"] = {}
    candidate["scraped_context_sources"] = []
    candidate.pop("odds_snapshot_age_hours", None)
    candidate.pop("bookmaker_last_update", None)

    decision = ContextResearchMind().evaluate(candidate)

    assert decision.data_freshness == "insufficiently_verified"


def test_research_mind_does_not_overclaim_verified_fresh_when_bookmaker_timestamp_is_missing() -> None:
    candidate = _base_candidate()
    candidate.pop("bookmaker_last_update", None)

    decision = ContextResearchMind().evaluate(candidate)

    assert decision.data_freshness != "verified_fresh"
