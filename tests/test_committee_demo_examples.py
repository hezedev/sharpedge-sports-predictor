from __future__ import annotations

from src.committee import FinalDecision
from src.committee.demo_examples import build_committee_demo_examples, render_committee_demo_examples


def test_demo_examples_cover_requested_committee_cases() -> None:
    examples = build_committee_demo_examples()

    assert examples["accepted_bet"]["committee"].final_decision == FinalDecision.BET
    assert examples["no_bet_low_edge"]["committee"].final_decision == FinalDecision.NO_BET
    assert examples["wait_for_lineups"]["committee"].final_decision == FinalDecision.WAIT_FOR_LINEUPS
    assert examples["avoid_both_disagree"]["committee"].final_decision == FinalDecision.AVOID
    assert examples["substitute_bet_accepted"]["committee"].final_decision == FinalDecision.BET_SUBSTITUTE
    assert examples["substitute_rejected_no_edge"]["committee"].final_decision == FinalDecision.NO_BET

    rejected_leg_plan = examples["rejected_parlay_leg"]["plan"]
    assert rejected_leg_plan.final_verdict == "DO_NOT_BUILD"
    assert any(item["final_decision"] == "WAIT_FOR_LINEUPS" for item in rejected_leg_plan.rejected_legs)

    conflict_plan = examples["rejected_parlay_duplicate_conflict"]["plan"]
    assert conflict_plan.final_verdict == "DO_NOT_BUILD"
    assert conflict_plan.duplicate_game_warnings
    assert conflict_plan.contradictory_picks


def test_demo_examples_confirm_core_safety_rules() -> None:
    examples = build_committee_demo_examples()

    assert examples["no_bet_low_edge"]["committee"].final_decision == FinalDecision.NO_BET
    assert examples["substitute_rejected_no_edge"]["committee"].final_decision == FinalDecision.NO_BET
    assert examples["blind_opposite_side_block"]["committee"].final_decision == FinalDecision.NO_BET
    assert examples["blind_opposite_side_block"]["committee"].better_substitute == ""
    assert examples["stale_data_hold"]["committee"].final_decision == FinalDecision.HOLD
    assert examples["wait_for_lineups"]["committee"].final_decision == FinalDecision.WAIT_FOR_LINEUPS
    assert examples["short_odds_no_bet"]["committee"].final_decision == FinalDecision.NO_BET

    six_leg_plan = examples["conservative_over_cap"]["plan"]
    assert six_leg_plan.final_verdict == "HIGH_RISK_ONLY"
    assert any("cannot be labelled conservative" in note.lower() for note in six_leg_plan.notes)


def test_demo_examples_render_readable_outputs() -> None:
    text = render_committee_demo_examples()

    assert "[accepted_bet]" in text
    assert "Final decision: BET" in text
    assert "Final decision: NO_BET" in text
    assert "Final decision: WAIT_FOR_LINEUPS" in text
    assert "Final decision: AVOID" in text
    assert "Final decision: BET_SUBSTITUTE" in text
    assert "[rejected_parlay_duplicate_conflict]" in text
    assert "Final verdict: DO_NOT_BUILD" in text
    assert "[conservative_over_cap]" in text
    assert "Final verdict: HIGH_RISK_ONLY" in text
