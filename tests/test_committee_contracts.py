from __future__ import annotations

from src.committee import (
    AgreementStatus,
    ArbiterMind,
    CommitteeDecision,
    FinalDecision,
    ModelMind,
    ModelVerdict,
    ResearchMind,
    ResearchMindDecision,
    ResearchVerdict,
    VetoFlag,
)


def test_committee_enum_values_are_stable() -> None:
    assert ResearchVerdict.AGREE.value == "AGREE"
    assert ResearchVerdict.DISAGREE.value == "DISAGREE"
    assert ResearchVerdict.HOLD.value == "HOLD"
    assert ResearchVerdict.AVOID.value == "AVOID"

    assert ModelVerdict.BET.value == "BET"
    assert ModelVerdict.NO_BET.value == "NO_BET"
    assert ModelVerdict.HOLD.value == "HOLD"
    assert ModelVerdict.AVOID.value == "AVOID"

    assert FinalDecision.BET.value == "BET"
    assert FinalDecision.NO_BET.value == "NO_BET"
    assert FinalDecision.HOLD.value == "HOLD"
    assert FinalDecision.WAIT_FOR_LINEUPS.value == "WAIT_FOR_LINEUPS"
    assert FinalDecision.AVOID.value == "AVOID"
    assert FinalDecision.BET_SUBSTITUTE.value == "BET_SUBSTITUTE"

    assert AgreementStatus.FULL_AGREEMENT.value == "FULL_AGREEMENT"
    assert AgreementStatus.PARTIAL_AGREEMENT.value == "PARTIAL_AGREEMENT"
    assert AgreementStatus.DISAGREEMENT.value == "DISAGREEMENT"
    assert AgreementStatus.CONFLICT.value == "CONFLICT"
    assert AgreementStatus.INSUFFICIENT_DATA.value == "INSUFFICIENT_DATA"

    assert VetoFlag.STALE_ODDS.value == "STALE_ODDS"
    assert VetoFlag.WEAK_PARLAY_LEG.value == "WEAK_PARLAY_LEG"


def test_committee_decision_serializes_to_json_safe_dict() -> None:
    decision = CommitteeDecision(
        final_decision=FinalDecision.HOLD,
        agreement_status=AgreementStatus.PARTIAL_AGREEMENT,
        research_verdict=ResearchVerdict.HOLD,
        model_verdict=ModelVerdict.BET,
        veto_flags=(VetoFlag.STALE_NEWS, VetoFlag.MISSING_LINEUPS),
        reasons=("news is stale", "lineups not confirmed"),
        better_substitute="double_chance",
        metadata={"sport": "soccer", "market": "moneyline"},
    )

    payload = decision.to_dict()

    assert payload["final_decision"] == "HOLD"
    assert payload["agreement_status"] == "PARTIAL_AGREEMENT"
    assert payload["research_verdict"] == "HOLD"
    assert payload["model_verdict"] == "BET"
    assert payload["veto_flags"] == ["STALE_NEWS", "MISSING_LINEUPS"]
    assert payload["reasons"] == ["news is stale", "lineups not confirmed"]
    assert payload["better_substitute"] == "double_chance"
    assert payload["metadata"]["sport"] == "soccer"


def test_research_mind_decision_serializes_to_json_safe_dict() -> None:
    decision = ResearchMindDecision(
        research_verdict=ResearchVerdict.HOLD,
        sport="soccer",
        confidence="Medium",
        main_evidence=("Playoff motivation detected",),
        main_risks=("lineups are not confirmed",),
        suggested_better_market="double_chance",
        data_freshness="missing lineups",
        sources_checked=("api_football", "odds_snapshot"),
        evidence_status="PARTIAL",
        concrete_info_score=42,
        source_count=2,
        source_quality_summary="mixed",
        fixture_verified=True,
        odds_age_minutes=12,
        odds_freshness_status="fresh",
        market_availability_status="available",
        lineup_status="missing_near_kickoff",
        injury_status="checked_fresh",
        motivation_status="checked",
        rotation_status="checked",
        missing_evidence=("lineups are missing near kickoff",),
        sport_specific_missing_evidence=("lineups are missing near kickoff",),
        conflicting_evidence=(),
        evidence_notes=("sources checked: api_football, odds_snapshot",),
        wait_for_lineups_signal=True,
        veto_flags=(VetoFlag.MISSING_LINEUPS,),
        metadata={"sport": "soccer"},
    )

    payload = decision.to_dict()

    assert payload["research_verdict"] == "HOLD"
    assert payload["sport"] == "soccer"
    assert payload["confidence"] == "Medium"
    assert payload["main_evidence"] == ["Playoff motivation detected"]
    assert payload["main_risks"] == ["lineups are not confirmed"]
    assert payload["sources_checked"] == ["api_football", "odds_snapshot"]
    assert payload["evidence_status"] == "PARTIAL"
    assert payload["concrete_info_score"] == 42
    assert payload["source_count"] == 2
    assert payload["source_quality_summary"] == "mixed"
    assert payload["fixture_verified"] is True
    assert payload["odds_age_minutes"] == 12
    assert payload["market_availability_status"] == "available"
    assert payload["lineup_status"] == "missing_near_kickoff"
    assert payload["missing_evidence"] == ["lineups are missing near kickoff"]
    assert payload["sport_specific_missing_evidence"] == ["lineups are missing near kickoff"]
    assert payload["evidence_notes"] == ["sources checked: api_football, odds_snapshot"]
    assert payload["wait_for_lineups_signal"] is True
    assert payload["veto_flags"] == ["MISSING_LINEUPS"]
    assert payload["metadata"]["sport"] == "soccer"


def test_committee_protocols_are_runtime_checkable() -> None:
    class _Research:
        def evaluate(self, candidate):
            return ResearchMindDecision(
                research_verdict=ResearchVerdict.AGREE,
                main_evidence=("context is supportive",),
                metadata={"candidate": candidate},
            )

    class _Model:
        def evaluate(self, candidate):
            return {"verdict": "BET", "candidate": candidate}

    class _Arbiter:
        def decide(self, *, candidate, research, model):
            return CommitteeDecision(
                final_decision=FinalDecision.BET,
                agreement_status=AgreementStatus.FULL_AGREEMENT,
                research_verdict=ResearchVerdict.AGREE,
                model_verdict=ModelVerdict.BET,
                metadata={"candidate": candidate, "research": research, "model": model},
            )

    assert isinstance(_Research(), ResearchMind)
    assert isinstance(_Model(), ModelMind)
    assert isinstance(_Arbiter(), ArbiterMind)
