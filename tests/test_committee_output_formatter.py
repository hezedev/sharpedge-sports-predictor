from __future__ import annotations

from src.committee import (
    AgreementStatus,
    CommitteeDecision,
    FinalDecision,
    ModelMindDecision,
    ModelVerdict,
    ResearchMindDecision,
    ResearchVerdict,
    VetoFlag,
    build_committee_pick_output,
    format_committee_pick_output,
)


def _candidate() -> dict:
    return {
        "home": "Alpha FC",
        "away": "Beta FC",
        "team": "Alpha FC",
        "market": "moneyline",
    }


def _research() -> ResearchMindDecision:
    return ResearchMindDecision(
        research_verdict=ResearchVerdict.HOLD,
        sport="soccer",
        confidence="Medium",
        main_evidence=("Playoff context detected", "Availability context fetched from api_football"),
        main_risks=("lineups are not confirmed",),
        suggested_better_market="double_chance",
        data_freshness="missing lineups",
        sources_checked=("api_football", "odds_snapshot"),
        evidence_status="PARTIAL",
        concrete_info_score=48,
        source_count=2,
        source_quality_summary="mixed",
        fixture_verified=True,
        odds_age_minutes=18,
        odds_freshness_status="fresh",
        market_availability_status="available",
        lineup_status="missing_near_kickoff",
        injury_status="checked_fresh",
        motivation_status="checked",
        rotation_status="checked",
        missing_evidence=("lineups are missing near kickoff",),
        sport_specific_missing_evidence=("lineups are missing near kickoff",),
        evidence_notes=("sources checked: api_football, odds_snapshot",),
        wait_for_lineups_signal=True,
        veto_flags=(VetoFlag.MISSING_LINEUPS,),
        metadata={"fixture_verified": True, "match_status": "pre_match"},
    )


def _model() -> ModelMindDecision:
    return ModelMindDecision(
        model_verdict=ModelVerdict.BET,
        model_probability=0.58,
        market_implied_probability=0.5263,
        vig_free_market_probability=0.51,
        fair_odds=1.724,
        minimum_acceptable_odds=1.78,
        current_odds=1.90,
        estimated_edge=0.102,
        confidence_interval=(0.54, 0.62),
        risk_tier="medium",
        suggested_market="double_chance",
        parlay_suitability="small_parlay_only",
        reasons=("positive edge", "current odds are above the minimum acceptable odds"),
    )


def _arbiter() -> CommitteeDecision:
    return CommitteeDecision(
        final_decision=FinalDecision.WAIT_FOR_LINEUPS,
        agreement_status=AgreementStatus.INSUFFICIENT_DATA,
        research_verdict=ResearchVerdict.HOLD,
        model_verdict=ModelVerdict.BET,
        veto_flags=(VetoFlag.MISSING_LINEUPS,),
        reasons=("lineups are not confirmed", "arbiter veto flags: MISSING_LINEUPS"),
        better_substitute="Alpha FC or Draw (double_chance)",
        metadata={"suggested_market": "double_chance"},
    )


def test_build_committee_pick_output_returns_required_sections() -> None:
    payload = build_committee_pick_output(
        candidate=_candidate(),
        research=_research(),
        model=_model(),
        arbiter=_arbiter(),
    )

    assert payload["game"] == "Alpha FC vs Beta FC"
    assert payload["original_pick"] == "Alpha FC"
    assert payload["market"] == "moneyline"
    assert payload["research_mind"]["verdict"] == "HOLD"
    assert payload["research_mind"]["sport"] == "soccer"
    assert payload["research_mind"]["evidence_status"] == "PARTIAL"
    assert payload["research_mind"]["concrete_info_score"] == 48
    assert payload["research_mind"]["market_availability_status"] == "available"
    assert payload["research_mind"]["sport_specific_missing_evidence"] == ["lineups are missing near kickoff"]
    assert payload["research_mind"]["suggested_better_market"] == "double_chance"
    assert payload["model_mind"]["verdict"] == "BET"
    assert payload["model_mind"]["minimum_acceptable_odds"] == 1.78
    assert payload["arbiter"]["agreement_status"] == "INSUFFICIENT_DATA"
    assert payload["arbiter"]["final_decision"] == "WAIT_FOR_LINEUPS"
    assert payload["arbiter"]["parlay_suitability"] == "blocked"
    assert payload["arbiter"]["better_substitute"] == "Alpha FC or Draw (double_chance)"
    assert payload["arbiter"]["final_explanation"]
    assert payload["evidence_enrichment"]["triggered"] is False


def test_format_committee_pick_output_renders_expected_labels() -> None:
    text = format_committee_pick_output(
        candidate=_candidate(),
        research=_research(),
        model=_model(),
        arbiter=_arbiter(),
    )

    assert "Game: Alpha FC vs Beta FC" in text
    assert "Original pick: Alpha FC" in text
    assert "Research Mind:" in text
    assert "- Verdict: HOLD" in text
    assert "- Sport: soccer" in text
    assert "- Data freshness: missing lineups" in text
    assert "- Evidence status: PARTIAL" in text
    assert "- Concrete info score: 48" in text
    assert "- Source count: 2" in text
    assert "- Market availability status: available" in text
    assert "- Lineup status: missing_near_kickoff" in text
    assert "- Sport-specific missing evidence: lineups are missing near kickoff" in text
    assert "Model Mind:" in text
    assert "- Model probability: 0.5800" in text
    assert "- Confidence range: 0.5400–0.6200" in text
    assert "Arbiter:" in text
    assert "- Agreement status: INSUFFICIENT_DATA" in text
    assert "- Final decision: WAIT_FOR_LINEUPS" in text
    assert "- Parlay suitability: blocked" in text
    assert "- Final explanation:" in text
    assert "Evidence Enrichment:" in text
    assert "- Triggered: no" in text


def test_formatter_uses_arbiter_reason_and_fallbacks() -> None:
    empty_research = ResearchMindDecision(research_verdict=ResearchVerdict.AGREE)
    empty_model = ModelMindDecision(model_verdict=ModelVerdict.NO_BET)
    arbiter = CommitteeDecision(
        final_decision=FinalDecision.NO_BET,
        agreement_status=AgreementStatus.PARTIAL_AGREEMENT,
        research_verdict=ResearchVerdict.AGREE,
        model_verdict=ModelVerdict.NO_BET,
        reasons=("edge is below threshold",),
    )

    payload = build_committee_pick_output(
        candidate={"team": "Gamma", "market": "totals"},
        research=empty_research,
        model=empty_model,
        arbiter=arbiter,
    )

    assert payload["game"] == "n/a"
    assert payload["arbiter"]["reason"] == "edge is below threshold"
    assert payload["arbiter"]["better_substitute"] == ""


def test_formatter_renders_tennis_enrichment_fields() -> None:
    payload = build_committee_pick_output(
        candidate={"home": "Carlos Alcaraz", "away": "Jannik Sinner", "team": "Carlos Alcaraz", "market": "moneyline"},
        research=ResearchMindDecision(research_verdict=ResearchVerdict.AGREE, sport="tennis"),
        model=ModelMindDecision(model_verdict=ModelVerdict.BET),
        arbiter=CommitteeDecision(
            final_decision=FinalDecision.BET,
            agreement_status=AgreementStatus.PARTIAL_AGREEMENT,
            research_verdict=ResearchVerdict.AGREE,
            model_verdict=ModelVerdict.BET,
        ),
        enrichment_summary={
            "triggered": True,
            "surface_status": "verified",
            "ranking_elo_status": "checked",
            "injury_retirement_status": "no_concern_found",
            "fatigue_status": "checked",
            "tournament_context_status": "checked",
            "style_matchup_status": "checked",
        },
    )

    assert payload["evidence_enrichment"]["surface_status"] == "verified"
    assert payload["evidence_enrichment"]["ranking_elo_status"] == "checked"
    assert payload["evidence_enrichment"]["injury_retirement_status"] == "no_concern_found"

    text = format_committee_pick_output(
        candidate={"home": "Carlos Alcaraz", "away": "Jannik Sinner", "team": "Carlos Alcaraz", "market": "moneyline"},
        research=ResearchMindDecision(research_verdict=ResearchVerdict.AGREE, sport="tennis"),
        model=ModelMindDecision(model_verdict=ModelVerdict.BET),
        arbiter=CommitteeDecision(
            final_decision=FinalDecision.BET,
            agreement_status=AgreementStatus.PARTIAL_AGREEMENT,
            research_verdict=ResearchVerdict.AGREE,
            model_verdict=ModelVerdict.BET,
        ),
        enrichment_summary={
            "triggered": True,
            "surface_status": "verified",
            "ranking_elo_status": "checked",
            "injury_retirement_status": "no_concern_found",
            "fatigue_status": "checked",
            "tournament_context_status": "checked",
            "style_matchup_status": "checked",
        },
    )

    assert "- Surface status: verified" in text
    assert "- Ranking/Elo status: checked" in text
    assert "- Injury/retirement status: no_concern_found" in text


def test_formatter_renders_mlb_enrichment_fields() -> None:
    payload = build_committee_pick_output(
        candidate={"home": "New York Yankees", "away": "Boston Red Sox", "team": "New York Yankees", "market": "moneyline"},
        research=ResearchMindDecision(research_verdict=ResearchVerdict.AGREE, sport="mlb"),
        model=ModelMindDecision(model_verdict=ModelVerdict.BET),
        arbiter=CommitteeDecision(
            final_decision=FinalDecision.BET,
            agreement_status=AgreementStatus.PARTIAL_AGREEMENT,
            research_verdict=ResearchVerdict.AGREE,
            model_verdict=ModelVerdict.BET,
        ),
        enrichment_summary={
            "triggered": True,
            "fixture_status": "scheduled",
            "probable_pitcher_status": "confirmed",
            "pitcher_change_status": "stable",
            "home_pitcher": "Gerrit Cole",
            "away_pitcher": "Brayan Bello",
            "pitcher_handedness_status": "checked",
            "lineup_status": "projected",
            "injury_status": "checked",
            "bullpen_status": "checked_proxy",
            "weather_status": "checked",
            "park_factor_status": "checked_proxy",
            "travel_rest_status": "checked",
            "market_fit_status": "acceptable",
        },
    )

    assert payload["evidence_enrichment"]["fixture_status"] == "scheduled"
    assert payload["evidence_enrichment"]["probable_pitcher_status"] == "confirmed"
    assert payload["evidence_enrichment"]["home_pitcher"] == "Gerrit Cole"

    text = format_committee_pick_output(
        candidate={"home": "New York Yankees", "away": "Boston Red Sox", "team": "New York Yankees", "market": "moneyline"},
        research=ResearchMindDecision(research_verdict=ResearchVerdict.AGREE, sport="mlb"),
        model=ModelMindDecision(model_verdict=ModelVerdict.BET),
        arbiter=CommitteeDecision(
            final_decision=FinalDecision.BET,
            agreement_status=AgreementStatus.PARTIAL_AGREEMENT,
            research_verdict=ResearchVerdict.AGREE,
            model_verdict=ModelVerdict.BET,
        ),
        enrichment_summary={
            "triggered": True,
            "fixture_status": "scheduled",
            "probable_pitcher_status": "confirmed",
            "pitcher_change_status": "stable",
            "home_pitcher": "Gerrit Cole",
            "away_pitcher": "Brayan Bello",
            "pitcher_handedness_status": "checked",
            "lineup_status": "projected",
            "injury_status": "checked",
            "bullpen_status": "checked_proxy",
            "weather_status": "checked",
            "park_factor_status": "checked_proxy",
            "travel_rest_status": "checked",
            "market_fit_status": "acceptable",
        },
    )

    assert "- Fixture status: scheduled" in text
    assert "- Probable pitcher status: confirmed" in text
    assert "- Home pitcher: Gerrit Cole" in text
    assert "- Bullpen status: checked_proxy" in text


def test_formatter_renders_soccer_enrichment_fields() -> None:
    payload = build_committee_pick_output(
        candidate={"home": "Alpha FC", "away": "Beta FC", "team": "Alpha FC", "market": "moneyline"},
        research=ResearchMindDecision(research_verdict=ResearchVerdict.AGREE, sport="soccer"),
        model=ModelMindDecision(model_verdict=ModelVerdict.BET),
        arbiter=CommitteeDecision(
            final_decision=FinalDecision.BET,
            agreement_status=AgreementStatus.PARTIAL_AGREEMENT,
            research_verdict=ResearchVerdict.AGREE,
            model_verdict=ModelVerdict.BET,
        ),
        enrichment_summary={
            "triggered": True,
            "source_quality": "strong",
            "providers_attempted": ["api_football", "availability", "news_context"],
            "providers_succeeded": ["availability", "news_context"],
            "providers_failed": ["standings"],
            "provider_failure_reasons": {"standings": "standings_unavailable"},
            "api_football_status": "not_found",
            "availability_status": "ok",
            "news_context_status": "usable_sources_found",
            "feature_cache_status": "mapping_failed",
            "standings_status": "not_found",
            "fixture_status": "scheduled",
            "lineup_status": "projected",
            "probable_lineup_status": "projected",
            "injury_status": "checked",
            "suspension_status": "checked",
            "goalkeeper_status": "confirmed",
            "motivation_status": "not_required",
            "rotation_status": "not_required",
            "fixture_congestion_status": "not_flagged",
            "home_away_form_status": "missing",
            "xg_context_status": "missing",
            "market_fit_status": "acceptable",
        },
    )

    assert payload["evidence_enrichment"]["source_quality"] == "strong"
    assert payload["evidence_enrichment"]["providers_attempted"] == ["api_football", "availability", "news_context"]
    assert payload["evidence_enrichment"]["provider_failure_reasons"] == {"standings": "standings_unavailable"}
    assert payload["evidence_enrichment"]["fixture_status"] == "scheduled"
    assert payload["evidence_enrichment"]["probable_lineup_status"] == "projected"
    assert payload["evidence_enrichment"]["goalkeeper_status"] == "confirmed"

    text = format_committee_pick_output(
        candidate={"home": "Alpha FC", "away": "Beta FC", "team": "Alpha FC", "market": "moneyline"},
        research=ResearchMindDecision(research_verdict=ResearchVerdict.AGREE, sport="soccer"),
        model=ModelMindDecision(model_verdict=ModelVerdict.BET),
        arbiter=CommitteeDecision(
            final_decision=FinalDecision.BET,
            agreement_status=AgreementStatus.PARTIAL_AGREEMENT,
            research_verdict=ResearchVerdict.AGREE,
            model_verdict=ModelVerdict.BET,
        ),
        enrichment_summary={
            "triggered": True,
            "source_quality": "strong",
            "providers_attempted": ["api_football", "availability", "news_context"],
            "providers_succeeded": ["availability", "news_context"],
            "providers_failed": ["standings"],
            "provider_failure_reasons": {"standings": "standings_unavailable"},
            "api_football_status": "not_found",
            "availability_status": "ok",
            "news_context_status": "usable_sources_found",
            "feature_cache_status": "mapping_failed",
            "standings_status": "not_found",
            "fixture_status": "scheduled",
            "lineup_status": "projected",
            "probable_lineup_status": "projected",
            "injury_status": "checked",
            "suspension_status": "checked",
            "goalkeeper_status": "confirmed",
            "motivation_status": "not_required",
            "rotation_status": "not_required",
            "fixture_congestion_status": "not_flagged",
            "home_away_form_status": "missing",
            "xg_context_status": "missing",
            "market_fit_status": "acceptable",
        },
    )

    assert "- Source quality: strong" in text
    assert "- Providers attempted: api_football, availability, news_context" in text
    assert "- API-Football status: not_found" in text
    assert "- Probable lineup status: projected" in text
    assert "- Goalkeeper status: confirmed" in text
    assert "- Fixture congestion status: not_flagged" in text
