from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from config import settings
from src.risk.parlay_builder import ParlayBuilder, ParlayLeg

from .arbiter_mind import ConsensusArbiterMind
from .contracts import CommitteeDecision, FinalDecision, ModelMindDecision, ResearchMindDecision, VetoFlag


_PARLAY_CFG = (((settings or {}).get("betting") or {}).get("parlay") or {})


@dataclass(frozen=True)
class CommitteeParlayPlan:
    parlay_name: str = ""
    parlay_type: str = "conservative"
    requested_legs: int = 0
    number_of_legs: int = 0
    total_odds: float = 0.0
    estimated_combined_probability: float = 0.0
    risk_tier: str = ""
    weakest_leg: dict[str, Any] | None = None
    correlation_warnings: tuple[str, ...] = field(default_factory=tuple)
    duplicate_game_warnings: tuple[str, ...] = field(default_factory=tuple)
    contradictory_picks: tuple[str, ...] = field(default_factory=tuple)
    accepted_legs: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    rejected_legs: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    final_verdict: str = "DO_NOT_BUILD"
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["correlation_warnings"] = list(self.correlation_warnings)
        payload["duplicate_game_warnings"] = list(self.duplicate_game_warnings)
        payload["contradictory_picks"] = list(self.contradictory_picks)
        payload["accepted_legs"] = list(self.accepted_legs)
        payload["rejected_legs"] = list(self.rejected_legs)
        payload["notes"] = list(self.notes)
        return payload


class CommitteeParlayBuilder:
    """
    Committee-side parlay gate.

    This does not replace the live parlay builder yet. It requires every leg to
    already have passed through the Arbiter and then applies the stricter
    committee-specific acceptance and quality rules.
    """

    def __init__(self) -> None:
        self.conservative_min_legs = int(_PARLAY_CFG.get("conservative_min_legs", 3) or 3)
        self.conservative_max_legs = int(_PARLAY_CFG.get("conservative_max_legs", 5) or 5)
        self._legacy_builder = ParlayBuilder(min_legs=self.conservative_min_legs, max_legs=max(self.conservative_max_legs, 7))
        self._major_veto_flags = ConsensusArbiterMind._major_veto_flags()

    def build(
        self,
        entries: list[dict[str, Any]],
        *,
        parlay_name: str = "",
        parlay_type: str = "conservative",
    ) -> CommitteeParlayPlan:
        accepted_entries: list[dict[str, Any]] = []
        rejected_entries: list[dict[str, Any]] = []
        parlay_legs: list[ParlayLeg] = []

        for entry in entries:
            if self._eligible_for_conservative(entry):
                accepted_entries.append(self._accepted_payload(entry))
                parlay_legs.append(self._to_parlay_leg(entry))
            else:
                rejected_entries.append(self._rejected_payload(entry))

        total_odds = 1.0
        combined_probability = 1.0
        for leg in parlay_legs:
            total_odds *= leg.odds
            combined_probability *= leg.ml_prob
        if not parlay_legs:
            total_odds = 0.0
            combined_probability = 0.0

        assessment = self._legacy_builder.assess_legs(parlay_legs, tier="value") if parlay_legs else {
            "risk_tier": "do-not-build",
            "build_verdict": "DO NOT BUILD",
            "duplicate_games": [],
            "conflicting_picks": [],
            "correlated_pick_groups": [],
            "notes": [],
            "weakest_leg": None,
        }

        weakest = assessment.get("weakest_leg")
        weakest_leg = None
        if weakest is not None:
            weakest_leg = {
                "team": weakest.team,
                "match_id": weakest.match_id,
                "market": weakest.market,
                "odds": weakest.odds,
                "model_probability": weakest.ml_prob,
                "edge": weakest.edge,
            }

        final_verdict = self._final_verdict(
            accepted_count=len(parlay_legs),
            assessment=assessment,
        )

        risk_tier = str(assessment.get("risk_tier", "") or "")
        if final_verdict == "DO_NOT_BUILD":
            risk_tier = "do-not-build"
        elif final_verdict == "HIGH_RISK_ONLY" and risk_tier == "conservative":
            risk_tier = "high-risk"

        notes = list(assessment.get("notes", []))
        if len(parlay_legs) > self.conservative_max_legs:
            notes.append("Anything above 5 legs cannot be labelled conservative.")
        if final_verdict == "DO_NOT_BUILD" and not notes:
            notes.append("Committee parlay quality is too weak to build safely.")
        if final_verdict == "HIGH_RISK_ONLY":
            notes.append("This slip should not be labelled safe or conservative.")
        if final_verdict != "BUILD":
            notes.append("One failed leg kills the full slip.")

        return CommitteeParlayPlan(
            parlay_name=parlay_name,
            parlay_type=parlay_type,
            requested_legs=len(entries),
            number_of_legs=len(parlay_legs),
            total_odds=round(total_odds, 4),
            estimated_combined_probability=round(combined_probability, 6),
            risk_tier=risk_tier,
            weakest_leg=weakest_leg,
            correlation_warnings=tuple(str(item) for item in assessment.get("correlated_pick_groups", []) or []),
            duplicate_game_warnings=tuple(str(item) for item in assessment.get("duplicate_games", []) or []),
            contradictory_picks=tuple(str(item) for item in assessment.get("conflicting_picks", []) or []),
            accepted_legs=tuple(accepted_entries),
            rejected_legs=tuple(rejected_entries),
            final_verdict=final_verdict,
            notes=tuple(dict.fromkeys(note for note in notes if note)),
        )

    def _eligible_for_conservative(self, entry: dict[str, Any]) -> bool:
        committee: CommitteeDecision = entry["committee_decision"]
        model: ModelMindDecision = entry["model_decision"]
        research: ResearchMindDecision = entry["research_decision"]
        effective_risk_tier = self._effective_risk_tier(research=research, model=model)

        if committee.final_decision != FinalDecision.BET:
            return False
        if str(research.evidence_status or "").upper() in {"PARTIAL", "INSUFFICIENT", "CONFLICTING"}:
            return False
        if str(effective_risk_tier or "").lower() not in {"low", "medium-low", "medium_low"}:
            return False
        if any(flag in self._major_veto_flags for flag in committee.veto_flags):
            return False
        if (model.estimated_edge or 0.0) <= 0:
            return False
        if str(research.data_freshness or "").lower() not in {"verified_fresh", "acceptable_freshness"}:
            return False
        return True

    @staticmethod
    def _accepted_payload(entry: dict[str, Any]) -> dict[str, Any]:
        candidate = entry["candidate"]
        committee: CommitteeDecision = entry["committee_decision"]
        model: ModelMindDecision = entry["model_decision"]
        return {
            "match_id": CommitteeParlayBuilder._match_id(candidate),
            "team": str(candidate.get("team", "") or ""),
            "market": str(candidate.get("market", "") or ""),
            "odds": model.current_odds,
            "model_probability": model.model_probability,
            "edge": model.estimated_edge,
            "risk_tier": CommitteeParlayBuilder._effective_risk_tier(research=entry["research_decision"], model=model),
            "final_decision": committee.final_decision.value,
            "evidence_status": str(entry["research_decision"].evidence_status or ""),
        }

    @staticmethod
    def _rejected_payload(entry: dict[str, Any]) -> dict[str, Any]:
        candidate = entry["candidate"]
        committee: CommitteeDecision = entry["committee_decision"]
        model: ModelMindDecision = entry["model_decision"]
        research: ResearchMindDecision = entry["research_decision"]
        return {
            "match_id": CommitteeParlayBuilder._match_id(candidate),
            "team": str(candidate.get("team", "") or ""),
            "market": str(candidate.get("market", "") or ""),
            "final_decision": committee.final_decision.value,
            "risk_tier": CommitteeParlayBuilder._effective_risk_tier(research=research, model=model),
            "data_freshness": research.data_freshness,
            "evidence_status": str(research.evidence_status or ""),
            "veto_flags": [flag.value for flag in committee.veto_flags],
        }

    @staticmethod
    def _match_id(candidate: dict[str, Any]) -> str:
        home = str(candidate.get("home", "") or "").strip()
        away = str(candidate.get("away", "") or "").strip()
        if home and away:
            return f"{home} vs {away}"
        return str(candidate.get("match_id", "") or "").strip()

    def _to_parlay_leg(self, entry: dict[str, Any]) -> ParlayLeg:
        candidate = entry["candidate"]
        model: ModelMindDecision = entry["model_decision"]
        return ParlayLeg(
            sport=str(candidate.get("sport", "") or ""),
            match_id=self._match_id(candidate),
            team=str(candidate.get("team", "") or ""),
            odds=float(model.current_odds or 0.0),
            ml_prob=float(model.model_probability or 0.0),
            fair_prob=float(model.vig_free_market_probability or 0.0),
            edge=float(model.estimated_edge or 0.0),
            commence=str(candidate.get("commence_time", "") or candidate.get("commence", "") or ""),
            market=str(candidate.get("market", "") or ""),
            home_team=str(candidate.get("home", "") or ""),
            away_team=str(candidate.get("away", "") or ""),
        )

    def _final_verdict(self, *, accepted_count: int, assessment: dict[str, Any]) -> str:
        if accepted_count < self.conservative_min_legs:
            return "DO_NOT_BUILD"
        if assessment.get("duplicate_games") or assessment.get("conflicting_picks"):
            return "DO_NOT_BUILD"
        if assessment.get("correlated_pick_groups"):
            return "DO_NOT_BUILD"
        if accepted_count > self.conservative_max_legs:
            return "HIGH_RISK_ONLY"
        if str(assessment.get("build_verdict", "") or "").upper() == "DO NOT BUILD":
            return "DO_NOT_BUILD"
        return "BUILD"

    @staticmethod
    def _effective_risk_tier(*, research: ResearchMindDecision, model: ModelMindDecision) -> str:
        current = str(model.risk_tier or "").strip().lower()
        if str(research.evidence_status or "").upper() == "ACCEPTABLE" and current in {"", "low", "medium-low", "medium_low"}:
            return "medium"
        return str(model.risk_tier or "")
