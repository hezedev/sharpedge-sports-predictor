from __future__ import annotations

from src.committee import (
    AgreementStatus,
    CommitteeDecision,
    ConsensusArbiterMind,
    FinalDecision,
    ModelMindDecision,
    ModelVerdict,
    ResearchMindDecision,
    ResearchVerdict,
    VetoFlag,
)


def _base_candidate() -> dict:
    return {
        "sport": "soccer",
        "market": "moneyline",
        "team": "Alpha FC",
        "home": "Alpha FC",
        "away": "Beta FC",
    }


def _agreeing_research() -> ResearchMindDecision:
    return ResearchMindDecision(
        research_verdict=ResearchVerdict.AGREE,
        sport="soccer",
        confidence="High",
        main_evidence=("Context is supportive",),
        data_freshness="verified_fresh",
        sources_checked=("api_football", "odds_snapshot"),
        evidence_status="COMPLETE",
        concrete_info_score=88,
        source_count=2,
        source_quality_summary="strong",
        fixture_verified=True,
        odds_age_minutes=15,
        odds_freshness_status="fresh",
        market_availability_status="available",
        lineup_status="confirmed",
        injury_status="checked_fresh",
        motivation_status="not_required",
        rotation_status="not_required",
        metadata={
            "fixture_verified": True,
            "match_status": "pre_match",
            "odds_freshness": "fresh",
            "lineup_freshness": "fresh",
            "injury_news_freshness": "fresh",
            "standings_freshness": "fresh",
        },
    )


def _betting_model() -> ModelMindDecision:
    return ModelMindDecision(
        model_verdict=ModelVerdict.BET,
        model_probability=0.58,
        market_implied_probability=0.5263,
        vig_free_market_probability=0.51,
        fair_odds=1.724,
        minimum_acceptable_odds=1.78,
        current_odds=1.9,
        estimated_edge=0.10,
        confidence_interval=(0.54, 0.62),
        risk_tier="medium",
        suggested_market="moneyline",
        parlay_suitability="small_parlay_only",
        reasons=("positive edge",),
        metadata={"min_edge_threshold": 0.03},
    )


def test_arbiter_rejects_short_odds_even_when_both_minds_support_pick() -> None:
    model = ModelMindDecision(
        model_verdict=ModelVerdict.BET,
        model_probability=0.58,
        market_implied_probability=0.5263,
        vig_free_market_probability=0.51,
        fair_odds=1.724,
        minimum_acceptable_odds=1.78,
        current_odds=1.7,
        estimated_edge=0.10,
        confidence_interval=(0.54, 0.62),
        risk_tier="medium",
        suggested_market="moneyline",
        parlay_suitability="small_parlay_only",
        reasons=("positive edge",),
        metadata={"min_edge_threshold": 0.03},
    )

    decision = ConsensusArbiterMind(min_edge=0.03).decide(
        candidate=_base_candidate(),
        research=_agreeing_research(),
        model=model,
    )

    assert decision.final_decision == FinalDecision.NO_BET
    assert VetoFlag.ODDS_TOO_SHORT in decision.veto_flags


def test_arbiter_rejects_when_model_edge_is_below_threshold() -> None:
    model = ModelMindDecision(
        model_verdict=ModelVerdict.BET,
        model_probability=0.58,
        market_implied_probability=0.5263,
        vig_free_market_probability=0.51,
        fair_odds=1.724,
        minimum_acceptable_odds=1.78,
        current_odds=1.9,
        estimated_edge=0.02,
        confidence_interval=(0.54, 0.62),
        risk_tier="medium",
        suggested_market="moneyline",
        parlay_suitability="small_parlay_only",
        reasons=("positive edge",),
        metadata={"min_edge_threshold": 0.03},
    )

    decision = ConsensusArbiterMind(min_edge=0.03).decide(
        candidate=_base_candidate(),
        research=_agreeing_research(),
        model=model,
    )

    assert decision.final_decision == FinalDecision.NO_BET
    assert VetoFlag.LOW_EDGE in decision.veto_flags


def test_arbiter_waits_for_lineups_when_research_detects_missing_lineups() -> None:
    research = ResearchMindDecision(
        research_verdict=ResearchVerdict.HOLD,
        sport="soccer",
        confidence="Medium",
        main_evidence=("Context is supportive",),
        main_risks=("lineups missing",),
        data_freshness="missing",
        sources_checked=("api_football", "odds_snapshot"),
        wait_for_lineups_signal=True,
        veto_flags=(VetoFlag.MISSING_LINEUPS,),
        metadata={
            "fixture_verified": True,
            "match_status": "pre_match",
            "odds_freshness": "fresh",
            "lineup_freshness": "missing",
            "injury_news_freshness": "fresh",
            "standings_freshness": "fresh",
        },
    )

    decision = ConsensusArbiterMind(min_edge=0.03).decide(
        candidate=_base_candidate(),
        research=research,
        model=_betting_model(),
    )

    assert decision.final_decision == FinalDecision.WAIT_FOR_LINEUPS


def test_arbiter_holds_or_avoids_on_high_rotation_risk() -> None:
    research = ResearchMindDecision(
        research_verdict=ResearchVerdict.HOLD,
        sport="soccer",
        confidence="Medium",
        main_risks=("Rotation risk elevated",),
        data_freshness="verified_fresh",
        sources_checked=("api_football",),
        evidence_status="PARTIAL",
        veto_flags=(VetoFlag.HIGH_ROTATION_RISK,),
        metadata={"fixture_verified": True, "match_status": "pre_match"},
    )

    decision = ConsensusArbiterMind(min_edge=0.03).decide(
        candidate=_base_candidate(),
        research=research,
        model=_betting_model(),
    )

    assert decision.final_decision == FinalDecision.HOLD


def test_arbiter_allows_bet_with_acceptable_evidence_but_floors_risk_tier_and_parlay_suitability() -> None:
    research = ResearchMindDecision(
        research_verdict=ResearchVerdict.AGREE,
        sport="soccer",
        confidence="Medium",
        main_evidence=("Fixture verified", "Odds snapshot checked"),
        main_risks=("No major risks detected from available evidence",),
        data_freshness="acceptable_freshness",
        sources_checked=("api_football", "odds_snapshot"),
        evidence_status="ACCEPTABLE",
        concrete_info_score=78,
        source_count=2,
        source_quality_summary="mixed",
        fixture_verified=True,
        odds_age_minutes=20,
        odds_freshness_status="acceptable",
        market_availability_status="available",
        lineup_status="unknown",
        injury_status="checked_fresh",
        motivation_status="not_required",
        rotation_status="not_required",
        metadata={"fixture_verified": True, "match_status": "pre_match"},
    )

    decision = ConsensusArbiterMind(min_edge=0.03).decide(
        candidate=_base_candidate(),
        research=research,
        model=_betting_model(),
    )

    assert decision.final_decision == FinalDecision.BET
    assert decision.metadata["effective_risk_tier"] == "medium"
    assert decision.metadata["effective_parlay_suitability"] == "small_parlay_only"


def test_arbiter_avoids_when_both_minds_disagree() -> None:
    research = ResearchMindDecision(
        research_verdict=ResearchVerdict.DISAGREE,
        sport="soccer",
        confidence="Medium",
        main_risks=("Context pushes back against the play",),
        data_freshness="verified_fresh",
        sources_checked=("api_football",),
        metadata={"fixture_verified": True, "match_status": "pre_match"},
    )
    model = ModelMindDecision(
        model_verdict=ModelVerdict.NO_BET,
        model_probability=0.58,
        market_implied_probability=0.5263,
        vig_free_market_probability=0.51,
        fair_odds=1.724,
        minimum_acceptable_odds=1.78,
        current_odds=1.9,
        estimated_edge=0.01,
        confidence_interval=(0.54, 0.62),
        risk_tier="avoid",
        suggested_market="moneyline",
        parlay_suitability="avoid",
        reasons=("edge is below the configured threshold",),
        metadata={"min_edge_threshold": 0.03},
    )

    decision = ConsensusArbiterMind(min_edge=0.03).decide(
        candidate=_base_candidate(),
        research=research,
        model=model,
    )

    assert decision.final_decision == FinalDecision.AVOID
    assert decision.agreement_status == AgreementStatus.DISAGREEMENT


def test_arbiter_returns_bet_substitute_when_alternative_passes_all_checks() -> None:
    candidate = {
        **_base_candidate(),
        "substitute_candidate": {
            **_base_candidate(),
            "market": "double_chance",
            "team": "Alpha FC or Draw",
        },
        "substitute_research": _agreeing_research(),
        "substitute_model": ModelMindDecision(
            model_verdict=ModelVerdict.BET,
            model_probability=0.58,
            market_implied_probability=0.5263,
            vig_free_market_probability=0.51,
            fair_odds=1.724,
            minimum_acceptable_odds=1.78,
            current_odds=1.9,
            estimated_edge=0.10,
            confidence_interval=(0.54, 0.62),
            risk_tier="medium",
            suggested_market="double_chance",
            parlay_suitability="small_parlay_only",
            reasons=("positive edge",),
            metadata={"min_edge_threshold": 0.03},
        ),
    }
    model = ModelMindDecision(
        model_verdict=ModelVerdict.BET,
        model_probability=0.58,
        market_implied_probability=0.5263,
        vig_free_market_probability=0.51,
        fair_odds=1.724,
        minimum_acceptable_odds=1.78,
        current_odds=1.9,
        estimated_edge=0.10,
        confidence_interval=(0.54, 0.62),
        risk_tier="medium",
        suggested_market="double_chance",
        parlay_suitability="small_parlay_only",
        reasons=("positive edge",),
        metadata={"min_edge_threshold": 0.03},
    )

    decision = ConsensusArbiterMind(min_edge=0.03).decide(
        candidate=candidate,
        research=_agreeing_research(),
        model=model,
    )

    assert decision.final_decision == FinalDecision.BET_SUBSTITUTE
    assert "double_chance" in decision.better_substitute


def test_arbiter_returns_no_bet_when_substitute_has_no_edge() -> None:
    candidate = {
        **_base_candidate(),
        "substitute_candidate": {
            **_base_candidate(),
            "market": "double_chance",
            "team": "Alpha FC or Draw",
        },
        "substitute_research": _agreeing_research(),
        "substitute_model": ModelMindDecision(
            model_verdict=ModelVerdict.NO_BET,
            model_probability=0.58,
            market_implied_probability=0.5263,
            vig_free_market_probability=0.51,
            fair_odds=1.724,
            minimum_acceptable_odds=1.78,
            current_odds=1.9,
            estimated_edge=0.01,
            confidence_interval=(0.54, 0.62),
            risk_tier="avoid",
            suggested_market="double_chance",
            parlay_suitability="avoid",
            reasons=("edge is below the configured threshold",),
            metadata={"min_edge_threshold": 0.03},
        ),
    }
    model = ModelMindDecision(
        model_verdict=ModelVerdict.BET,
        model_probability=0.58,
        market_implied_probability=0.5263,
        vig_free_market_probability=0.51,
        fair_odds=1.724,
        minimum_acceptable_odds=1.78,
        current_odds=1.9,
        estimated_edge=0.10,
        confidence_interval=(0.54, 0.62),
        risk_tier="medium",
        suggested_market="double_chance",
        parlay_suitability="small_parlay_only",
        reasons=("positive edge",),
        metadata={"min_edge_threshold": 0.03},
    )

    decision = ConsensusArbiterMind(min_edge=0.03).decide(
        candidate=candidate,
        research=_agreeing_research(),
        model=model,
    )

    assert decision.final_decision == FinalDecision.NO_BET


def test_arbiter_blocks_blind_opposite_side_conversion() -> None:
    candidate = {
        **_base_candidate(),
        "substitute_candidate": {
            **_base_candidate(),
            "team": "Beta FC",
            "market": "moneyline",
        },
        "substitute_research": _agreeing_research(),
        "substitute_model": _betting_model(),
    }
    model = ModelMindDecision(
        model_verdict=ModelVerdict.BET,
        model_probability=0.58,
        market_implied_probability=0.5263,
        vig_free_market_probability=0.51,
        fair_odds=1.724,
        minimum_acceptable_odds=1.78,
        current_odds=1.9,
        estimated_edge=0.10,
        confidence_interval=(0.54, 0.62),
        risk_tier="medium",
        suggested_market="double_chance",
        parlay_suitability="small_parlay_only",
        reasons=("positive edge",),
        metadata={"min_edge_threshold": 0.03},
    )

    decision = ConsensusArbiterMind(min_edge=0.03).decide(
        candidate=candidate,
        research=_agreeing_research(),
        model=model,
    )

    assert decision.final_decision == FinalDecision.NO_BET


def test_arbiter_returns_plain_bet_when_all_checks_pass() -> None:
    decision = ConsensusArbiterMind(min_edge=0.03).decide(
        candidate=_base_candidate(),
        research=_agreeing_research(),
        model=_betting_model(),
    )

    assert isinstance(decision, CommitteeDecision)
    assert decision.final_decision == FinalDecision.BET
    assert decision.agreement_status == AgreementStatus.FULL_AGREEMENT


def test_arbiter_blocks_partial_evidence_even_when_model_and_research_agree() -> None:
    research = ResearchMindDecision(
        **{
            **_agreeing_research().__dict__,
            "evidence_status": "PARTIAL",
        }
    )

    decision = ConsensusArbiterMind(min_edge=0.03).decide(
        candidate=_base_candidate(),
        research=research,
        model=_betting_model(),
    )

    assert decision.final_decision == FinalDecision.NO_BET


def test_arbiter_adds_insufficient_evidence_veto_and_refuses_bet() -> None:
    research = ResearchMindDecision(
        **{
            **_agreeing_research().__dict__,
            "evidence_status": "INSUFFICIENT",
            "confidence": "Low",
        }
    )

    decision = ConsensusArbiterMind(min_edge=0.03).decide(
        candidate=_base_candidate(),
        research=research,
        model=_betting_model(),
    )

    assert decision.final_decision == FinalDecision.HOLD
    assert VetoFlag.INSUFFICIENT_EVIDENCE in decision.veto_flags


def test_arbiter_blocks_unverified_market_availability() -> None:
    research = ResearchMindDecision(
        **{
            **_agreeing_research().__dict__,
            "market_availability_status": "missing",
        }
    )

    decision = ConsensusArbiterMind(min_edge=0.03).decide(
        candidate=_base_candidate(),
        research=research,
        model=_betting_model(),
    )

    assert decision.final_decision == FinalDecision.HOLD
    assert VetoFlag.UNVERIFIED_MARKET_AVAILABILITY in decision.veto_flags


def test_arbiter_raises_pitcher_evidence_veto_for_critical_mlb_gap() -> None:
    candidate = {
        **_base_candidate(),
        "sport": "mlb",
        "market": "moneyline",
        "team": "Yankees",
        "home": "Yankees",
        "away": "Red Sox",
    }
    research = ResearchMindDecision(
        **{
            **_agreeing_research().__dict__,
            "sport": "mlb",
            "metadata": {
                **_agreeing_research().metadata,
                "critical_missing_evidence": ["probable starters not fully confirmed"],
            },
        }
    )

    decision = ConsensusArbiterMind(min_edge=0.03).decide(
        candidate=candidate,
        research=research,
        model=_betting_model(),
    )

    assert decision.final_decision == FinalDecision.HOLD
    assert VetoFlag.MISSING_SPORT_CRITICAL_EVIDENCE in decision.veto_flags
    assert VetoFlag.MISSING_PITCHER_EVIDENCE in decision.veto_flags


def test_arbiter_raises_goalie_lineup_and_surface_vetoes_from_research_gaps() -> None:
    nhl_research = ResearchMindDecision(
        **{
            **_agreeing_research().__dict__,
            "sport": "nhl",
            "metadata": {
                **_agreeing_research().metadata,
                "critical_missing_evidence": ["starting goalie projection not checked"],
            },
        }
    )
    nhl_decision = ConsensusArbiterMind(min_edge=0.03).decide(
        candidate={**_base_candidate(), "sport": "nhl"},
        research=nhl_research,
        model=_betting_model(),
    )
    assert VetoFlag.MISSING_GOALIE_EVIDENCE in nhl_decision.veto_flags

    nba_research = ResearchMindDecision(
        **{
            **_agreeing_research().__dict__,
            "sport": "basketball",
            "metadata": {
                **_agreeing_research().metadata,
                "critical_missing_evidence": ["star-player injury status uncertain"],
            },
        }
    )
    nba_decision = ConsensusArbiterMind(min_edge=0.03).decide(
        candidate={**_base_candidate(), "sport": "basketball"},
        research=nba_research,
        model=_betting_model(),
    )
    assert VetoFlag.MISSING_STAR_INJURY_STATUS in nba_decision.veto_flags

    tennis_research = ResearchMindDecision(
        **{
            **_agreeing_research().__dict__,
            "sport": "tennis",
            "metadata": {
                **_agreeing_research().metadata,
                "critical_missing_evidence": ["surface context was not checked for the tennis matchup"],
            },
        }
    )
    tennis_decision = ConsensusArbiterMind(min_edge=0.03).decide(
        candidate={**_base_candidate(), "sport": "tennis"},
        research=tennis_research,
        model=_betting_model(),
    )
    assert VetoFlag.MISSING_SURFACE_CONTEXT in tennis_decision.veto_flags


def test_arbiter_adds_stale_and_conflicting_evidence_veto_flags() -> None:
    research = ResearchMindDecision(
        **{
            **_agreeing_research().__dict__,
            "evidence_status": "CONFLICTING",
            "odds_freshness_status": "stale",
            "conflicting_evidence": ("source disagreement on key availability",),
        }
    )

    decision = ConsensusArbiterMind(min_edge=0.03).decide(
        candidate=_base_candidate(),
        research=research,
        model=_betting_model(),
    )

    assert decision.final_decision == FinalDecision.HOLD
    assert VetoFlag.CONFLICTING_EVIDENCE in decision.veto_flags
    assert VetoFlag.STALE_ODDS_EVIDENCE in decision.veto_flags
