from __future__ import annotations

from typing import Any

from config import settings

from .contracts import (
    AgreementStatus,
    CommitteeDecision,
    FinalDecision,
    ModelMindDecision,
    ModelVerdict,
    ResearchMindDecision,
    ResearchVerdict,
    VetoFlag,
)


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class ConsensusArbiterMind:
    """
    Final committee gatekeeper that reconciles Research Mind and Model Mind.

    This phase does not modify the real scan flow. It only adds a stable
    Arbiter layer that can be integrated later.
    """

    def __init__(self, *, min_edge: float | None = None) -> None:
        risk_cfg = settings.get("risk", {}).get("kelly", {})
        committee_cfg = settings.get("committee", {}) or {}
        self.min_edge = float(min_edge if min_edge is not None else risk_cfg.get("min_edge", 0.03) or 0.03)
        self.block_stale_data = bool(committee_cfg.get("block_stale_data", True))
        self.block_missing_lineups_near_kickoff = bool(committee_cfg.get("block_missing_lineups_near_kickoff", True))

    def decide(
        self,
        *,
        candidate: dict[str, Any],
        research: ResearchMindDecision | dict[str, Any],
        model: ModelMindDecision | dict[str, Any],
    ) -> CommitteeDecision:
        research_decision = self._ensure_research_decision(research)
        model_decision = self._ensure_model_decision(model)

        combined_flags = self._combine_veto_flags(candidate, research_decision, model_decision)
        reasons = self._initial_reasons(research_decision, model_decision, combined_flags)
        agreement_status = self._agreement_status(research_decision, model_decision, combined_flags)

        final_decision = self._base_final_decision(
            candidate=candidate,
            research=research_decision,
            model=model_decision,
            veto_flags=combined_flags,
        )
        better_substitute = ""

        if final_decision != FinalDecision.BET:
            substitute_result = self._evaluate_substitute(candidate)
            if substitute_result is not None:
                sub_candidate, sub_research, sub_model, sub_flags = substitute_result
                final_decision = FinalDecision.BET_SUBSTITUTE
                better_substitute = self._describe_substitute(sub_candidate, sub_model, sub_research)
                combined_flags = tuple(dict.fromkeys([*combined_flags, *sub_flags]))
                agreement_status = AgreementStatus.PARTIAL_AGREEMENT
                reasons.append("original pick was rejected, but a safer substitute passed the full committee checks")

        return CommitteeDecision(
            final_decision=final_decision,
            agreement_status=agreement_status,
            research_verdict=research_decision.research_verdict,
            model_verdict=model_decision.model_verdict,
            veto_flags=combined_flags,
            reasons=tuple(dict.fromkeys(reason for reason in reasons if reason)),
            better_substitute=better_substitute,
            metadata={
                "candidate_market": str(candidate.get("market", "") or ""),
                "candidate_selection": str(candidate.get("team", "") or ""),
                "research_confidence": research_decision.confidence,
                "model_risk_tier": model_decision.risk_tier,
                "effective_risk_tier": self._effective_risk_tier(research_decision, model_decision),
                "effective_parlay_suitability": self._effective_parlay_suitability(research_decision, model_decision),
                "suggested_market": self._resolve_suggested_market(candidate, research_decision, model_decision),
            },
        )

    def _base_final_decision(
        self,
        *,
        candidate: dict[str, Any],
        research: ResearchMindDecision,
        model: ModelMindDecision,
        veto_flags: tuple[VetoFlag, ...],
    ) -> FinalDecision:
        flags = set(veto_flags)

        if (
            research.wait_for_lineups_signal
            or VetoFlag.MISSING_LINEUPS in flags
            or VetoFlag.MISSING_LINEUP_EVIDENCE in flags
        ):
            return FinalDecision.WAIT_FOR_LINEUPS

        if VetoFlag.FINISHED_MATCH in flags or VetoFlag.MATCH_ALREADY_LIVE in flags:
            return FinalDecision.AVOID

        if research.data_freshness == "insufficiently_verified":
            return FinalDecision.HOLD
        if self.block_missing_lineups_near_kickoff and research.data_freshness == "missing":
            return FinalDecision.WAIT_FOR_LINEUPS if research.wait_for_lineups_signal else FinalDecision.HOLD

        if VetoFlag.HIGH_ROTATION_RISK in flags or VetoFlag.END_SEASON_CHAOS in flags:
            return FinalDecision.AVOID if research.research_verdict == ResearchVerdict.AVOID else FinalDecision.HOLD

        if research.research_verdict == ResearchVerdict.DISAGREE and model.model_verdict in {ModelVerdict.NO_BET, ModelVerdict.AVOID, ModelVerdict.HOLD}:
            return FinalDecision.AVOID

        if model.model_verdict == ModelVerdict.AVOID:
            return FinalDecision.AVOID

        if model.model_verdict == ModelVerdict.NO_BET or VetoFlag.LOW_EDGE in flags or VetoFlag.ODDS_TOO_SHORT in flags:
            return FinalDecision.NO_BET

        if research.research_verdict == ResearchVerdict.AVOID:
            return FinalDecision.AVOID

        if research.research_verdict == ResearchVerdict.HOLD:
            return FinalDecision.HOLD

        if model.model_verdict == ModelVerdict.HOLD or VetoFlag.CONFIDENCE_INTERVAL_FAIL in flags:
            return FinalDecision.HOLD

        if VetoFlag.BAD_MARKET_FIT in flags:
            return FinalDecision.NO_BET

        if research.research_verdict == ResearchVerdict.DISAGREE:
            return FinalDecision.AVOID

        if (
            VetoFlag.UNCLEAR_FIXTURE in flags
            or VetoFlag.STALE_ODDS in flags
            or VetoFlag.STALE_ODDS_EVIDENCE in flags
            or VetoFlag.STALE_NEWS in flags
        ):
            return FinalDecision.HOLD

        if flags.intersection(
            {
                VetoFlag.INSUFFICIENT_EVIDENCE,
                VetoFlag.CONFLICTING_EVIDENCE,
                VetoFlag.MISSING_SPORT_CRITICAL_EVIDENCE,
                VetoFlag.MISSING_PITCHER_EVIDENCE,
                VetoFlag.MISSING_GOALIE_EVIDENCE,
                VetoFlag.MISSING_STAR_INJURY_STATUS,
                VetoFlag.MISSING_SURFACE_CONTEXT,
                VetoFlag.UNVERIFIED_MARKET_AVAILABILITY,
            }
        ):
            return FinalDecision.HOLD

        if not self._passes_bet_checks(candidate, research, model, flags):
            return FinalDecision.NO_BET

        return FinalDecision.BET

    def _passes_bet_checks(
        self,
        candidate: dict[str, Any],
        research: ResearchMindDecision,
        model: ModelMindDecision,
        flags: set[VetoFlag],
    ) -> bool:
        if research.research_verdict != ResearchVerdict.AGREE:
            return False
        if model.model_verdict != ModelVerdict.BET:
            return False
        freshness = str(research.data_freshness or "")
        if freshness == "insufficiently_verified":
            return False
        if self.block_stale_data and freshness == "stale":
            return False
        if self.block_missing_lineups_near_kickoff and freshness == "missing":
            return False
        if freshness not in {"verified_fresh", "acceptable_freshness", "stale", "missing"}:
            return False
        if str(research.evidence_status or "").upper() not in {"COMPLETE", "ACCEPTABLE"}:
            return False
        if str(research.market_availability_status or "").lower() != "available":
            return False
        if not bool(research.metadata.get("fixture_verified", False)):
            return False
        if str(research.metadata.get("match_status", "") or "") != "pre_match":
            return False
        if VetoFlag.BAD_MARKET_FIT in flags:
            return False
        if self._critical_research_missing(research):
            return False

        lower_bound = None
        if model.confidence_interval and len(model.confidence_interval) == 2:
            lower_bound = _coerce_float(model.confidence_interval[0])

        if model.estimated_edge is None or model.estimated_edge < self.min_edge:
            return False
        if model.current_odds is None or model.minimum_acceptable_odds is None or model.current_odds < model.minimum_acceptable_odds:
            return False
        if lower_bound is None or model.vig_free_market_probability is None or lower_bound <= model.vig_free_market_probability:
            return False
        if model.model_probability is None or model.vig_free_market_probability is None or model.model_probability <= model.vig_free_market_probability:
            return False
        if self._resolve_suggested_market(candidate, research, model):
            return False
        if flags.intersection(self._major_veto_flags()):
            return False
        return True

    @staticmethod
    def _major_veto_flags() -> set[VetoFlag]:
        return {
            VetoFlag.STALE_ODDS,
            VetoFlag.STALE_ODDS_EVIDENCE,
            VetoFlag.STALE_NEWS,
            VetoFlag.MISSING_LINEUPS,
            VetoFlag.MISSING_LINEUP_EVIDENCE,
            VetoFlag.MISSING_STAR_INJURY_STATUS,
            VetoFlag.MISSING_SURFACE_CONTEXT,
            VetoFlag.MISSING_PITCHER_EVIDENCE,
            VetoFlag.MISSING_GOALIE_EVIDENCE,
            VetoFlag.MISSING_SPORT_CRITICAL_EVIDENCE,
            VetoFlag.INSUFFICIENT_EVIDENCE,
            VetoFlag.CONFLICTING_EVIDENCE,
            VetoFlag.UNVERIFIED_MARKET_AVAILABILITY,
            VetoFlag.MATCH_ALREADY_LIVE,
            VetoFlag.FINISHED_MATCH,
            VetoFlag.LOW_EDGE,
            VetoFlag.ODDS_TOO_SHORT,
            VetoFlag.CONFIDENCE_INTERVAL_FAIL,
            VetoFlag.BAD_MARKET_FIT,
            VetoFlag.HIGH_ROTATION_RISK,
            VetoFlag.END_SEASON_CHAOS,
            VetoFlag.UNCLEAR_FIXTURE,
            VetoFlag.CONTRADICTORY_PICK,
        }

    def _combine_veto_flags(
        self,
        candidate: dict[str, Any],
        research: ResearchMindDecision,
        model: ModelMindDecision,
    ) -> tuple[VetoFlag, ...]:
        flags: list[VetoFlag] = list(research.veto_flags)
        flags.extend(self._research_evidence_veto_flags(research))

        if model.estimated_edge is not None and model.estimated_edge < self.min_edge:
            flags.append(VetoFlag.LOW_EDGE)
        if model.current_odds is not None and model.minimum_acceptable_odds is not None and model.current_odds < model.minimum_acceptable_odds:
            flags.append(VetoFlag.ODDS_TOO_SHORT)
        if not self._lower_bound_supported(model):
            flags.append(VetoFlag.CONFIDENCE_INTERVAL_FAIL)
        if self._resolve_suggested_market(candidate, research, model):
            flags.append(VetoFlag.BAD_MARKET_FIT)

        deduped: list[VetoFlag] = []
        for flag in flags:
            if flag not in deduped:
                deduped.append(flag)
        return tuple(deduped)

    def _research_evidence_veto_flags(self, research: ResearchMindDecision) -> list[VetoFlag]:
        flags: list[VetoFlag] = []
        evidence_status = str(research.evidence_status or "").upper()
        market_availability_status = str(research.market_availability_status or "").lower()
        odds_freshness_status = str(research.odds_freshness_status or "").lower()
        critical_missing = [str(item or "") for item in self._critical_missing_items(research)]

        if evidence_status == "INSUFFICIENT":
            flags.append(VetoFlag.INSUFFICIENT_EVIDENCE)
        if evidence_status == "CONFLICTING":
            flags.append(VetoFlag.CONFLICTING_EVIDENCE)
        if market_availability_status != "available":
            flags.append(VetoFlag.UNVERIFIED_MARKET_AVAILABILITY)
        if odds_freshness_status and odds_freshness_status not in {"fresh", "acceptable"}:
            flags.append(VetoFlag.STALE_ODDS_EVIDENCE)
        if critical_missing:
            flags.append(VetoFlag.MISSING_SPORT_CRITICAL_EVIDENCE)

        lowered = [item.lower() for item in critical_missing]
        if any("lineup" in item for item in lowered):
            flags.append(VetoFlag.MISSING_LINEUP_EVIDENCE)
        if any("pitcher" in item or "starter" in item for item in lowered):
            flags.append(VetoFlag.MISSING_PITCHER_EVIDENCE)
        if any("goalie" in item for item in lowered):
            flags.append(VetoFlag.MISSING_GOALIE_EVIDENCE)
        if any("star-player" in item or "star player" in item for item in lowered):
            flags.append(VetoFlag.MISSING_STAR_INJURY_STATUS)
        if any("surface" in item for item in lowered):
            flags.append(VetoFlag.MISSING_SURFACE_CONTEXT)

        return flags

    @staticmethod
    def _critical_missing_items(research: ResearchMindDecision) -> list[str]:
        metadata = research.metadata or {}
        explicit = metadata.get("critical_missing_evidence") or ()
        if explicit:
            return [str(item or "").strip() for item in explicit if str(item or "").strip()]
        return [str(item or "").strip() for item in research.sport_specific_missing_evidence if str(item or "").strip()]

    @staticmethod
    def _lower_bound_supported(model: ModelMindDecision) -> bool:
        if not model.confidence_interval or len(model.confidence_interval) != 2:
            return False
        lower_bound = _coerce_float(model.confidence_interval[0])
        if lower_bound is None or model.vig_free_market_probability is None:
            return False
        return lower_bound > model.vig_free_market_probability

    def _critical_research_missing(self, research: ResearchMindDecision) -> bool:
        return bool(self._critical_missing_items(research))

    @staticmethod
    def _initial_reasons(
        research: ResearchMindDecision,
        model: ModelMindDecision,
        flags: tuple[VetoFlag, ...],
    ) -> list[str]:
        reasons: list[str] = []
        reasons.extend(str(item) for item in research.main_risks if item)
        reasons.extend(str(item) for item in model.reasons if item)
        if flags:
            reasons.append("arbiter veto flags: " + ", ".join(flag.value for flag in flags))
        return reasons

    @staticmethod
    def _agreement_status(
        research: ResearchMindDecision,
        model: ModelMindDecision,
        flags: tuple[VetoFlag, ...],
    ) -> AgreementStatus:
        if research.research_verdict == ResearchVerdict.AGREE and model.model_verdict == ModelVerdict.BET and not flags:
            return AgreementStatus.FULL_AGREEMENT
        if research.research_verdict in {ResearchVerdict.HOLD, ResearchVerdict.AVOID} or model.model_verdict == ModelVerdict.HOLD:
            return AgreementStatus.INSUFFICIENT_DATA
        if research.research_verdict == ResearchVerdict.DISAGREE and model.model_verdict == ModelVerdict.BET:
            return AgreementStatus.CONFLICT
        if research.research_verdict == ResearchVerdict.DISAGREE and model.model_verdict in {ModelVerdict.NO_BET, ModelVerdict.AVOID, ModelVerdict.HOLD}:
            return AgreementStatus.DISAGREEMENT
        return AgreementStatus.PARTIAL_AGREEMENT

    def _evaluate_substitute(
        self,
        candidate: dict[str, Any],
    ) -> tuple[dict[str, Any], ResearchMindDecision, ModelMindDecision, tuple[VetoFlag, ...]] | None:
        substitute_candidate = candidate.get("substitute_candidate")
        substitute_research = candidate.get("substitute_research")
        substitute_model = candidate.get("substitute_model")

        if not isinstance(substitute_candidate, dict) or substitute_research is None or substitute_model is None:
            return None

        if self._is_blind_opposite_side_conversion(candidate, substitute_candidate):
            return None

        research_decision = self._ensure_research_decision(substitute_research)
        model_decision = self._ensure_model_decision(substitute_model)
        substitute_flags = self._combine_veto_flags(substitute_candidate, research_decision, model_decision)

        if self._passes_bet_checks(substitute_candidate, research_decision, model_decision, set(substitute_flags)):
            return substitute_candidate, research_decision, model_decision, substitute_flags
        return None

    @staticmethod
    def _is_blind_opposite_side_conversion(
        original_candidate: dict[str, Any],
        substitute_candidate: dict[str, Any],
    ) -> bool:
        original_market = str(original_candidate.get("market", "") or "")
        substitute_market = str(substitute_candidate.get("market", "") or "")
        if original_market != substitute_market:
            return False

        same_home = str(original_candidate.get("home", "") or "") == str(substitute_candidate.get("home", "") or "")
        same_away = str(original_candidate.get("away", "") or "") == str(substitute_candidate.get("away", "") or "")
        if not (same_home and same_away):
            return False

        original_team = str(original_candidate.get("team", "") or "")
        substitute_team = str(substitute_candidate.get("team", "") or "")
        if not original_team or not substitute_team:
            return False

        return original_team != substitute_team

    def _resolve_suggested_market(
        self,
        candidate: dict[str, Any],
        research: ResearchMindDecision,
        model: ModelMindDecision,
    ) -> str:
        current_market = str(candidate.get("market", "") or "")
        if model.suggested_market and model.suggested_market != current_market:
            return model.suggested_market
        if research.suggested_better_market and research.suggested_better_market != current_market:
            return research.suggested_better_market
        return ""

    def _ensure_research_decision(self, payload: ResearchMindDecision | dict[str, Any]) -> ResearchMindDecision:
        if isinstance(payload, ResearchMindDecision):
            return payload
        return ResearchMindDecision(
            research_verdict=ResearchVerdict(str(payload.get("research_verdict", "HOLD"))),
            sport=str(payload.get("sport", "") or ""),
            confidence=str(payload.get("confidence", "Low") or "Low"),
            main_evidence=tuple(payload.get("main_evidence", ()) or ()),
            main_risks=tuple(payload.get("main_risks", ()) or ()),
            suggested_better_market=str(payload.get("suggested_better_market", "") or ""),
            data_freshness=str(payload.get("data_freshness", "") or ""),
            sources_checked=tuple(payload.get("sources_checked", ()) or ()),
            evidence_status=str(payload.get("evidence_status", "") or ""),
            concrete_info_score=int(payload.get("concrete_info_score", 0) or 0),
            source_count=int(payload.get("source_count", 0) or 0),
            source_quality_summary=str(payload.get("source_quality_summary", "") or ""),
            fixture_verified=bool(payload.get("fixture_verified", False)),
            odds_age_minutes=int(_coerce_float(payload.get("odds_age_minutes")) or 0) or None,
            odds_freshness_status=str(payload.get("odds_freshness_status", "") or ""),
            market_availability_status=str(payload.get("market_availability_status", "") or ""),
            lineup_status=str(payload.get("lineup_status", "") or ""),
            injury_status=str(payload.get("injury_status", "") or ""),
            motivation_status=str(payload.get("motivation_status", "") or ""),
            rotation_status=str(payload.get("rotation_status", "") or ""),
            missing_evidence=tuple(payload.get("missing_evidence", ()) or ()),
            sport_specific_missing_evidence=tuple(payload.get("sport_specific_missing_evidence", payload.get("missing_evidence", ())) or ()),
            conflicting_evidence=tuple(payload.get("conflicting_evidence", ()) or ()),
            evidence_notes=tuple(payload.get("evidence_notes", ()) or ()),
            wait_for_lineups_signal=bool(payload.get("wait_for_lineups_signal", False)),
            veto_flags=tuple(VetoFlag(str(flag)) for flag in (payload.get("veto_flags", ()) or ())),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    @staticmethod
    def _effective_risk_tier(research: ResearchMindDecision, model: ModelMindDecision) -> str:
        current = str(model.risk_tier or "").strip().lower()
        if str(research.evidence_status or "").upper() == "ACCEPTABLE" and current in {"", "low", "medium-low", "medium_low"}:
            return "medium"
        return str(model.risk_tier or "")

    @staticmethod
    def _effective_parlay_suitability(research: ResearchMindDecision, model: ModelMindDecision) -> str:
        if str(research.evidence_status or "").upper() == "ACCEPTABLE":
            return "small_parlay_only"
        return str(model.parlay_suitability or "")

    def _ensure_model_decision(self, payload: ModelMindDecision | dict[str, Any]) -> ModelMindDecision:
        if isinstance(payload, ModelMindDecision):
            return payload
        interval = tuple(payload.get("confidence_interval", ()) or ())
        if len(interval) != 2:
            interval = (None, None)
        return ModelMindDecision(
            model_verdict=ModelVerdict(str(payload.get("model_verdict", "HOLD"))),
            model_probability=_coerce_float(payload.get("model_probability")),
            market_implied_probability=_coerce_float(payload.get("market_implied_probability")),
            vig_free_market_probability=_coerce_float(payload.get("vig_free_market_probability")),
            fair_odds=_coerce_float(payload.get("fair_odds")),
            minimum_acceptable_odds=_coerce_float(payload.get("minimum_acceptable_odds")),
            current_odds=_coerce_float(payload.get("current_odds")),
            estimated_edge=_coerce_float(payload.get("estimated_edge")),
            confidence_interval=interval,
            risk_tier=str(payload.get("risk_tier", "") or ""),
            suggested_market=str(payload.get("suggested_market", "") or ""),
            parlay_suitability=str(payload.get("parlay_suitability", "") or ""),
            reasons=tuple(payload.get("reasons", ()) or ()),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    @staticmethod
    def _describe_substitute(
        substitute_candidate: dict[str, Any],
        model: ModelMindDecision,
        research: ResearchMindDecision,
    ) -> str:
        team = str(substitute_candidate.get("team", "") or "").strip()
        market = str(substitute_candidate.get("market", "") or "").strip()
        suggested_market = model.suggested_market or research.suggested_better_market
        if team and market:
            return f"{team} ({market})"
        if suggested_market:
            return suggested_market
        return team or market
