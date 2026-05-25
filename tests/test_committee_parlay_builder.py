from __future__ import annotations

from src.committee import (
    AgreementStatus,
    CommitteeDecision,
    CommitteeParlayBuilder,
    CommitteeParlayPlan,
    FinalDecision,
    ModelMindDecision,
    ModelVerdict,
    ResearchMindDecision,
    ResearchVerdict,
)


def _candidate(match_id: str, team: str, market: str = "moneyline", sport: str = "soccer") -> dict:
    home, away = match_id.split(" vs ")
    return {
        "sport": sport,
        "home": home,
        "away": away,
        "team": team,
        "market": market,
        "commence_time": "2026-05-06T18:00:00Z",
    }


def _committee_decision(final_decision: FinalDecision = FinalDecision.BET) -> CommitteeDecision:
    return CommitteeDecision(
        final_decision=final_decision,
        agreement_status=AgreementStatus.FULL_AGREEMENT,
        research_verdict=ResearchVerdict.AGREE,
        model_verdict=ModelVerdict.BET if final_decision == FinalDecision.BET else ModelVerdict.NO_BET,
    )


def _research_decision(data_freshness: str = "verified_fresh") -> ResearchMindDecision:
    return ResearchMindDecision(
        research_verdict=ResearchVerdict.AGREE,
        confidence="High",
        main_evidence=("Fresh context",),
        data_freshness=data_freshness,
        sources_checked=("api_football",),
        evidence_status="COMPLETE",
        concrete_info_score=86,
        source_count=2,
        source_quality_summary="strong",
        fixture_verified=True,
        odds_age_minutes=15,
        odds_freshness_status="fresh",
        lineup_status="confirmed",
        injury_status="checked_fresh",
        motivation_status="not_required",
        rotation_status="not_required",
        metadata={"fixture_verified": True, "match_status": "pre_match"},
    )


def _model_decision(
    *,
    odds: float = 1.9,
    prob: float = 0.58,
    fair_prob: float = 0.51,
    edge: float = 0.10,
    risk_tier: str = "low",
) -> ModelMindDecision:
    return ModelMindDecision(
        model_verdict=ModelVerdict.BET,
        model_probability=prob,
        market_implied_probability=round(1 / odds, 4),
        vig_free_market_probability=fair_prob,
        fair_odds=1.724,
        minimum_acceptable_odds=1.78,
        current_odds=odds,
        estimated_edge=edge,
        confidence_interval=(0.54, 0.62),
        risk_tier=risk_tier,
        suggested_market="moneyline",
        parlay_suitability="good_leg",
        reasons=("positive edge",),
    )


def _entry(
    match_id: str,
    team: str,
    *,
    market: str = "moneyline",
    risk_tier: str = "low",
    final_decision: FinalDecision = FinalDecision.BET,
    sport: str = "soccer",
) -> dict:
    return {
        "candidate": _candidate(match_id, team, market=market, sport=sport),
        "committee_decision": _committee_decision(final_decision=final_decision),
        "research_decision": _research_decision(),
        "model_decision": _model_decision(risk_tier=risk_tier),
    }


def test_committee_parlay_accepts_only_arbiter_bet_legs() -> None:
    builder = CommitteeParlayBuilder()
    entries = [
        _entry("A vs B", "A"),
        _entry("C vs D", "C"),
        _entry("E vs F", "E", final_decision=FinalDecision.NO_BET),
    ]

    plan = builder.build(entries, parlay_name="Conservative 3", parlay_type="conservative")

    assert plan.final_verdict == "DO_NOT_BUILD"
    assert plan.number_of_legs == 2
    assert any(item["final_decision"] == "NO_BET" for item in plan.rejected_legs)


def test_committee_parlay_builds_clean_conservative_slip() -> None:
    builder = CommitteeParlayBuilder()
    entries = [
        _entry("A vs B", "A", sport="soccer", market="moneyline"),
        _entry("C vs D", "C", sport="mlb", market="moneyline"),
        _entry("E vs F", "E", sport="nhl", market="spreads"),
    ]

    plan = builder.build(entries, parlay_name="Conservative 3", parlay_type="conservative")

    assert isinstance(plan, CommitteeParlayPlan)
    assert plan.final_verdict == "BUILD"
    assert plan.number_of_legs == 3
    assert plan.estimated_combined_probability > 0
    assert plan.weakest_leg is not None


def test_committee_parlay_above_five_legs_cannot_be_conservative() -> None:
    builder = CommitteeParlayBuilder()
    entries = [
        _entry("A vs B", "A", sport="soccer", market="moneyline"),
        _entry("C vs D", "C", sport="mlb", market="moneyline"),
        _entry("E vs F", "E", sport="nhl", market="spreads"),
        _entry("G vs H", "G", sport="basketball", market="totals"),
        _entry("I vs J", "I", sport="tennis", market="moneyline"),
        _entry("K vs L", "K", sport="soccer", market="totals"),
    ]

    plan = builder.build(entries, parlay_name="Conservative 6", parlay_type="conservative")

    assert plan.final_verdict == "HIGH_RISK_ONLY"
    assert any("above 5 legs" in note.lower() for note in plan.notes)


def test_committee_parlay_flags_duplicate_games_and_blocks_conflicts() -> None:
    builder = CommitteeParlayBuilder()
    entries = [
        _entry("A vs B", "A"),
        _entry("A vs B", "B"),
        _entry("C vs D", "C"),
    ]

    plan = builder.build(entries)

    assert plan.final_verdict == "DO_NOT_BUILD"
    assert "A vs B" in plan.duplicate_game_warnings
    assert "A vs B" in plan.contradictory_picks


def test_committee_parlay_warns_on_correlated_picks() -> None:
    builder = CommitteeParlayBuilder()
    entries = [
        _entry("A vs B", "A", sport="soccer"),
        _entry("C vs D", "C", sport="soccer"),
        _entry("E vs F", "E", sport="soccer"),
    ]

    plan = builder.build(entries)

    assert plan.final_verdict == "DO_NOT_BUILD"
    assert plan.correlation_warnings


def test_committee_parlay_rejects_medium_risk_legs_from_conservative_pool() -> None:
    builder = CommitteeParlayBuilder()
    entries = [
        _entry("A vs B", "A", risk_tier="medium", sport="soccer", market="moneyline"),
        _entry("C vs D", "C", sport="mlb", market="moneyline"),
        _entry("E vs F", "E", sport="nhl", market="spreads"),
    ]

    plan = builder.build(entries)

    assert plan.final_verdict == "DO_NOT_BUILD"
    assert any(item["risk_tier"] == "medium" for item in plan.rejected_legs)


def test_committee_parlay_serializes_json_safe_output() -> None:
    builder = CommitteeParlayBuilder()
    entries = [
        _entry("A vs B", "A", sport="soccer", market="moneyline"),
        _entry("C vs D", "C", sport="mlb", market="moneyline"),
        _entry("E vs F", "E", sport="nhl", market="spreads"),
    ]

    payload = builder.build(entries, parlay_name="Conservative 3").to_dict()

    assert payload["parlay_name"] == "Conservative 3"
    assert payload["accepted_legs"]
    assert isinstance(payload["notes"], list)


def test_committee_parlay_blocks_partial_evidence_from_conservative_pool() -> None:
    builder = CommitteeParlayBuilder()
    partial_entry = _entry("A vs B", "A", sport="soccer", market="moneyline")
    partial_entry["research_decision"] = ResearchMindDecision(
        research_verdict=ResearchVerdict.AGREE,
        confidence="Medium",
        main_evidence=("Fixture verified",),
        main_risks=("limited concrete research evidence",),
        data_freshness="acceptable_freshness",
        sources_checked=("api_football",),
        evidence_status="PARTIAL",
        concrete_info_score=44,
        source_count=1,
        source_quality_summary="mixed",
        fixture_verified=True,
        odds_age_minutes=22,
        odds_freshness_status="acceptable",
        lineup_status="unknown",
        injury_status="checked_fresh",
        motivation_status="not_required",
        rotation_status="not_required",
        metadata={"fixture_verified": True, "match_status": "pre_match"},
    )

    plan = builder.build([partial_entry, _entry("C vs D", "C"), _entry("E vs F", "E")])

    assert plan.final_verdict == "DO_NOT_BUILD"
    assert any(item["evidence_status"] == "PARTIAL" for item in plan.rejected_legs)


def test_committee_parlay_rejects_acceptable_evidence_low_risk_leg_from_conservative_pool() -> None:
    builder = CommitteeParlayBuilder()
    acceptable_entry = _entry("A vs B", "A", sport="soccer", market="moneyline")
    acceptable_entry["research_decision"] = ResearchMindDecision(
        research_verdict=ResearchVerdict.AGREE,
        confidence="Medium",
        main_evidence=("Fixture verified", "Odds snapshot checked"),
        main_risks=("No major risks detected from available evidence",),
        data_freshness="acceptable_freshness",
        sources_checked=("api_football", "odds_snapshot"),
        evidence_status="ACCEPTABLE",
        concrete_info_score=76,
        source_count=2,
        source_quality_summary="mixed",
        fixture_verified=True,
        odds_age_minutes=16,
        odds_freshness_status="acceptable",
        lineup_status="unknown",
        injury_status="checked_fresh",
        motivation_status="not_required",
        rotation_status="not_required",
        metadata={"fixture_verified": True, "match_status": "pre_match"},
    )

    plan = builder.build([acceptable_entry, _entry("C vs D", "C"), _entry("E vs F", "E")])

    assert plan.final_verdict == "DO_NOT_BUILD"
    assert any(item["risk_tier"] == "medium" for item in plan.rejected_legs)


def test_committee_parlay_keeps_tennis_legs_out_of_conservative_pool_when_evidence_is_only_acceptable() -> None:
    builder = CommitteeParlayBuilder()
    tennis_entry = _entry("A vs B", "A", sport="tennis", market="moneyline")
    tennis_entry["research_decision"] = ResearchMindDecision(
        research_verdict=ResearchVerdict.AGREE,
        confidence="Medium",
        main_evidence=("Fixture verified", "Preview source checked"),
        main_risks=("No major risks detected from available evidence",),
        data_freshness="acceptable_freshness",
        sources_checked=("espn", "newsapi"),
        evidence_status="ACCEPTABLE",
        concrete_info_score=82,
        source_count=2,
        source_quality_summary="strong",
        fixture_verified=True,
        odds_age_minutes=18,
        odds_freshness_status="acceptable",
        lineup_status="unknown",
        injury_status="checked_fresh",
        motivation_status="not_required",
        rotation_status="not_required",
        metadata={"fixture_verified": True, "match_status": "pre_match"},
    )

    plan = builder.build([tennis_entry, _entry("C vs D", "C"), _entry("E vs F", "E")])

    assert plan.final_verdict == "DO_NOT_BUILD"
    assert any(item["match_id"] == "A vs B" and item["risk_tier"] == "medium" for item in plan.rejected_legs)
