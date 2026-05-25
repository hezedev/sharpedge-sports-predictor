from __future__ import annotations

from typing import Final


DECISION_BET: Final[str] = "BET"
DECISION_BET_SUBSTITUTE: Final[str] = "BET SUBSTITUTE"
DECISION_NO_BET: Final[str] = "NO BET"
DECISION_HOLD: Final[str] = "HOLD"
DECISION_WAIT_FOR_LINEUPS: Final[str] = "WAIT FOR LINEUPS"
DECISION_AVOID: Final[str] = "AVOID"

VALID_DECISIONS: Final[set[str]] = {
    DECISION_BET,
    DECISION_BET_SUBSTITUTE,
    DECISION_NO_BET,
    DECISION_HOLD,
    DECISION_WAIT_FOR_LINEUPS,
    DECISION_AVOID,
}


def classify_candidate_decision(
    *,
    publish_ready: bool = False,
    review_reason: str = "",
    suppression_reason: str = "",
) -> tuple[str, str]:
    """
    Convert production-pipeline outcomes into an explicit user-facing verdict.

    This layer intentionally sits *after* model pricing and *before* UI/report
    publication so we can standardize the final recommendation language without
    changing the existing betting math.
    """
    review_reason = str(review_reason or "").strip()
    suppression_reason = str(suppression_reason or "").strip()
    review_lower = review_reason.lower()
    suppression_lower = suppression_reason.lower()

    if publish_ready:
        return DECISION_BET, "All publication guardrails passed."

    lineup_markers = (
        "lineup",
        "starter uncertainty",
        "starter",
        "availability",
        "goalie",
        "confirmed",
        "rotation",
    )
    if review_reason and any(marker in review_lower for marker in lineup_markers):
        return DECISION_WAIT_FOR_LINEUPS, review_reason

    if review_reason:
        return DECISION_HOLD, review_reason

    avoid_markers = (
        "same-game",
        "correlation",
        "contradict",
        "already represented",
        "not configured",
        "not scanable",
        "not scannable",
        "not publishable",
        "league or sport is not configured",
        "integrity",
        "market suitability",
    )
    if suppression_reason and any(marker in suppression_lower for marker in avoid_markers):
        return DECISION_AVOID, suppression_reason

    if suppression_reason:
        return DECISION_NO_BET, suppression_reason

    return DECISION_HOLD, "Candidate requires manual review before publication."
