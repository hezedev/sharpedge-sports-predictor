from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class ResearchVerdict(StrEnum):
    AGREE = "AGREE"
    DISAGREE = "DISAGREE"
    HOLD = "HOLD"
    AVOID = "AVOID"


class ModelVerdict(StrEnum):
    BET = "BET"
    NO_BET = "NO_BET"
    HOLD = "HOLD"
    AVOID = "AVOID"


class FinalDecision(StrEnum):
    BET = "BET"
    NO_BET = "NO_BET"
    HOLD = "HOLD"
    WAIT_FOR_LINEUPS = "WAIT_FOR_LINEUPS"
    AVOID = "AVOID"
    BET_SUBSTITUTE = "BET_SUBSTITUTE"


class AgreementStatus(StrEnum):
    FULL_AGREEMENT = "FULL_AGREEMENT"
    PARTIAL_AGREEMENT = "PARTIAL_AGREEMENT"
    DISAGREEMENT = "DISAGREEMENT"
    CONFLICT = "CONFLICT"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class VetoFlag(StrEnum):
    STALE_ODDS = "STALE_ODDS"
    STALE_ODDS_EVIDENCE = "STALE_ODDS_EVIDENCE"
    STALE_NEWS = "STALE_NEWS"
    MISSING_LINEUPS = "MISSING_LINEUPS"
    MISSING_LINEUP_EVIDENCE = "MISSING_LINEUP_EVIDENCE"
    MISSING_STAR_INJURY_STATUS = "MISSING_STAR_INJURY_STATUS"
    MISSING_SURFACE_CONTEXT = "MISSING_SURFACE_CONTEXT"
    MISSING_PITCHER_EVIDENCE = "MISSING_PITCHER_EVIDENCE"
    MISSING_GOALIE_EVIDENCE = "MISSING_GOALIE_EVIDENCE"
    MISSING_SPORT_CRITICAL_EVIDENCE = "MISSING_SPORT_CRITICAL_EVIDENCE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    UNVERIFIED_MARKET_AVAILABILITY = "UNVERIFIED_MARKET_AVAILABILITY"
    MATCH_ALREADY_LIVE = "MATCH_ALREADY_LIVE"
    FINISHED_MATCH = "FINISHED_MATCH"
    LOW_EDGE = "LOW_EDGE"
    ODDS_TOO_SHORT = "ODDS_TOO_SHORT"
    CONFIDENCE_INTERVAL_FAIL = "CONFIDENCE_INTERVAL_FAIL"
    BAD_MARKET_FIT = "BAD_MARKET_FIT"
    HIGH_ROTATION_RISK = "HIGH_ROTATION_RISK"
    END_SEASON_CHAOS = "END_SEASON_CHAOS"
    UNCLEAR_FIXTURE = "UNCLEAR_FIXTURE"
    DUPLICATE_PICK = "DUPLICATE_PICK"
    CORRELATED_PARLAY_RISK = "CORRELATED_PARLAY_RISK"
    CONTRADICTORY_PICK = "CONTRADICTORY_PICK"
    WEAK_PARLAY_LEG = "WEAK_PARLAY_LEG"


@dataclass(frozen=True)
class CommitteeDecision:
    """
    Shared committee contract for a final reviewable decision payload.

    This is intentionally runtime-agnostic for Phase 1. It can be used later by
    the research, model, and arbiter layers without changing the existing scan
    flow yet.
    """

    final_decision: FinalDecision
    agreement_status: AgreementStatus
    research_verdict: ResearchVerdict
    model_verdict: ModelVerdict
    veto_flags: tuple[VetoFlag, ...] = field(default_factory=tuple)
    reasons: tuple[str, ...] = field(default_factory=tuple)
    better_substitute: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["final_decision"] = self.final_decision.value
        payload["agreement_status"] = self.agreement_status.value
        payload["research_verdict"] = self.research_verdict.value
        payload["model_verdict"] = self.model_verdict.value
        payload["veto_flags"] = [flag.value for flag in self.veto_flags]
        payload["reasons"] = list(self.reasons)
        return payload


@dataclass(frozen=True)
class ResearchMindDecision:
    """
    Shared contract for the contextual / research committee member.

    This wraps the existing freshness, availability, and contextual evidence
    into a stable, JSON-safe structure without changing the real scan flow yet.
    """

    research_verdict: ResearchVerdict
    sport: str = ""
    confidence: str = "Low"
    main_evidence: tuple[str, ...] = field(default_factory=tuple)
    main_risks: tuple[str, ...] = field(default_factory=tuple)
    suggested_better_market: str = ""
    data_freshness: str = ""
    sources_checked: tuple[str, ...] = field(default_factory=tuple)
    evidence_status: str = ""
    concrete_info_score: int = 0
    source_count: int = 0
    source_quality_summary: str = ""
    fixture_verified: bool = False
    odds_age_minutes: int | None = None
    odds_freshness_status: str = ""
    market_availability_status: str = ""
    lineup_status: str = ""
    injury_status: str = ""
    motivation_status: str = ""
    rotation_status: str = ""
    missing_evidence: tuple[str, ...] = field(default_factory=tuple)
    sport_specific_missing_evidence: tuple[str, ...] = field(default_factory=tuple)
    conflicting_evidence: tuple[str, ...] = field(default_factory=tuple)
    evidence_notes: tuple[str, ...] = field(default_factory=tuple)
    wait_for_lineups_signal: bool = False
    veto_flags: tuple[VetoFlag, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["research_verdict"] = self.research_verdict.value
        payload["main_evidence"] = list(self.main_evidence)
        payload["main_risks"] = list(self.main_risks)
        payload["sources_checked"] = list(self.sources_checked)
        payload["missing_evidence"] = list(self.missing_evidence)
        payload["sport_specific_missing_evidence"] = list(self.sport_specific_missing_evidence)
        payload["conflicting_evidence"] = list(self.conflicting_evidence)
        payload["evidence_notes"] = list(self.evidence_notes)
        payload["veto_flags"] = [flag.value for flag in self.veto_flags]
        return payload


@dataclass(frozen=True)
class ModelMindDecision:
    """
    Shared contract for the quantitative committee member.

    This wraps the existing mathematical pricing output into a stable,
    JSON-safe structure without changing the real scan flow yet.
    """

    model_verdict: ModelVerdict
    model_probability: float | None = None
    market_implied_probability: float | None = None
    vig_free_market_probability: float | None = None
    fair_odds: float | None = None
    minimum_acceptable_odds: float | None = None
    current_odds: float | None = None
    estimated_edge: float | None = None
    confidence_interval: tuple[float | None, float | None] = (None, None)
    risk_tier: str = ""
    suggested_market: str = ""
    parlay_suitability: str = ""
    reasons: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["model_verdict"] = self.model_verdict.value
        payload["confidence_interval"] = list(self.confidence_interval)
        payload["reasons"] = list(self.reasons)
        return payload


@runtime_checkable
class ResearchMind(Protocol):
    """
    Interface for the future research/contextual layer.
    """

    def evaluate(self, candidate: dict[str, Any]) -> ResearchMindDecision:
        ...


@runtime_checkable
class ModelMind(Protocol):
    """
    Interface for the future quantitative model layer.
    """

    def evaluate(self, candidate: dict[str, Any]) -> ModelMindDecision:
        ...


@runtime_checkable
class ArbiterMind(Protocol):
    """
    Interface for the future final decision layer that reconciles research and model outputs.
    """

    def decide(
        self,
        *,
        candidate: dict[str, Any],
        research: ResearchMindDecision | dict[str, Any],
        model: ModelMindDecision | dict[str, Any],
    ) -> CommitteeDecision:
        ...
