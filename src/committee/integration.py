from __future__ import annotations

from typing import Any

from config import settings

from .arbiter_mind import ConsensusArbiterMind
from .contracts import CommitteeDecision, FinalDecision, ModelMindDecision, ResearchMindDecision
from .evidence_enrichment import EvidenceEnrichmentPass, EvidenceEnrichmentResult
from .model_mind import QuantModelMind
from .output_formatter import build_committee_pick_output, format_committee_pick_output
from .research_mind import ContextResearchMind


def committee_settings() -> dict[str, Any]:
    return dict((settings or {}).get("committee") or {})


def committee_enabled() -> bool:
    return bool(committee_settings().get("enable_committee_decision_layer", True))


def show_committee_details() -> bool:
    return bool(committee_settings().get("show_committee_details", False))


def allow_bet_substitutes() -> bool:
    return bool(committee_settings().get("allow_bet_substitutes", False))


def committee_required_for_parlays() -> bool:
    return bool(committee_settings().get("committee_required_for_parlays", True))


def evidence_enrichment_enabled() -> bool:
    return bool(committee_settings().get("enable_evidence_enrichment_pass", True))


def max_conservative_parlay_legs() -> int:
    return int(committee_settings().get("max_conservative_parlay_legs", 5) or 5)


def legacy_decision_status(final_decision: FinalDecision | str) -> str:
    value = str(getattr(final_decision, "value", final_decision) or "").strip().upper()
    mapping = {
        "BET": "BET",
        "NO_BET": "NO BET",
        "HOLD": "HOLD",
        "WAIT_FOR_LINEUPS": "WAIT FOR LINEUPS",
        "AVOID": "AVOID",
        "BET_SUBSTITUTE": "BET SUBSTITUTE",
    }
    return mapping.get(value, value.replace("_", " "))


def _dedupe_reason_lines(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _committee_reason_summary(
    *,
    candidate: dict[str, Any],
    research: ResearchMindDecision,
    committee: CommitteeDecision,
    payload: dict[str, Any],
    enrichment: EvidenceEnrichmentResult | None,
) -> str:
    reasons = [str(item).strip() for item in committee.reasons if str(item).strip()]
    sport = str(candidate.get("sport", "") or "").strip().lower()
    enrichment_payload = enrichment.to_dict() if enrichment else {}
    blocker_lines: list[str] = []

    if str(research.evidence_status or "").upper() == "INSUFFICIENT":
        blocker_lines.append("Evidence is still insufficient for publication.")
    elif str(research.evidence_status or "").upper() == "PARTIAL":
        blocker_lines.append("Evidence improved, but sport-critical context is still incomplete.")

    if str(research.source_quality_summary or "").lower() == "weak":
        blocker_lines.append("Source quality is still weak.")
    if int(research.source_count or 0) < 2:
        blocker_lines.append("Too few independent evidence sources are confirmed.")

    if sport == "soccer":
        if str(research.injury_status or "").lower() == "not_checked":
            blocker_lines.append("Injury and team-news context is still not properly checked.")
        if str(research.lineup_status or "").lower() in {"unknown", "missing_near_kickoff"}:
            blocker_lines.append("Lineup or probable XI context is still unclear.")
        if str(research.rotation_status or "").lower() == "not_checked":
            blocker_lines.append("Rotation risk has not been cleared yet.")
        if str(research.motivation_status or "").lower() == "not_checked":
            blocker_lines.append("Motivation or end-of-season context is still missing.")
        if str((enrichment_payload or {}).get("fixture_congestion_status", "") or "").lower() == "not_checked":
            blocker_lines.append("Fixture congestion or cup context is still missing.")

    blocker_lines.extend(str(item).strip() for item in (enrichment_payload or {}).get("remaining_missing_evidence", []) if str(item).strip())
    blocker_lines.extend(str(item).strip() for item in research.sport_specific_missing_evidence if str(item).strip())
    blocker_lines = _dedupe_reason_lines(blocker_lines)

    if reasons:
        summary = "; ".join(reasons)
        if blocker_lines and summary.lower() not in {line.lower() for line in blocker_lines}:
            summary = f"{summary}; {blocker_lines[0]}"
        return summary

    if blocker_lines:
        return "; ".join(blocker_lines[:2])

    return str(payload.get("arbiter", {}).get("final_explanation", "") or "Committee review completed.").strip()


def run_committee_pipeline(
    *,
    published: list[dict[str, Any]],
    review: list[dict[str, Any]],
    suppressed: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    research_mind = ContextResearchMind()
    model_mind = QuantModelMind()
    arbiter_mind = ConsensusArbiterMind()
    enrichment_pass = EvidenceEnrichmentPass()

    committee_entries: list[dict[str, Any]] = []
    final_published: list[dict[str, Any]] = []
    final_review: list[dict[str, Any]] = []
    final_suppressed: list[dict[str, Any]] = []

    all_candidates = (
        [("published", bet) for bet in published]
        + [("review", bet) for bet in review]
        + [("suppressed", bet) for bet in suppressed]
    )

    for source_bucket, original_candidate in all_candidates:
        candidate = dict(original_candidate)
        research = research_mind.evaluate(candidate)
        model = model_mind.evaluate(candidate)
        enrichment = enrichment_pass.run(candidate=candidate, research=research, model=model) if evidence_enrichment_enabled() else EvidenceEnrichmentResult(
            triggered=False,
            trigger_reasons=tuple(),
            missing_evidence_searched=tuple(),
            sources_checked=tuple(),
            sources_found=tuple(),
            evidence_before=str(research.evidence_status or ""),
            evidence_after=str(research.evidence_status or ""),
            concrete_score_before=int(research.concrete_info_score or 0),
            concrete_score_after=int(research.concrete_info_score or 0),
            remaining_missing_evidence=tuple(research.missing_evidence),
            final_arbiter_decision="",
            updated_candidate=dict(candidate),
        )
        candidate_for_committee = enrichment.updated_candidate if enrichment.triggered else candidate
        research_after = research_mind.evaluate(candidate_for_committee) if enrichment.triggered else research
        committee = arbiter_mind.decide(candidate=candidate_for_committee, research=research_after, model=model)
        enrichment = enrichment_pass.finalize(
            result=enrichment,
            research_after=research_after,
            final_arbiter_decision=committee.final_decision.value,
        ) if enrichment.triggered else enrichment
        enriched = enrich_candidate_with_committee(
            candidate=candidate_for_committee,
            research=research_after,
            model=model,
            committee=committee,
            source_bucket=source_bucket,
            enrichment=enrichment,
        )
        committee_entries.append(
            {
                "source_bucket": source_bucket,
                "candidate": enriched,
                "research_decision": research_after,
                "model_decision": model,
                "committee_decision": committee,
                "enrichment_result": enrichment.to_dict(),
            }
        )
        _bucket_committee_candidate(
            source_bucket=source_bucket,
            candidate=enriched,
            committee=committee,
            published=final_published,
            review=final_review,
            suppressed=final_suppressed,
        )

    return final_published, final_review, final_suppressed, committee_entries


def enrich_candidate_with_committee(
    *,
    candidate: dict[str, Any],
    research: ResearchMindDecision,
    model: ModelMindDecision,
    committee: CommitteeDecision,
    source_bucket: str,
    enrichment: EvidenceEnrichmentResult | None = None,
) -> dict[str, Any]:
    published_view = _committee_published_view(candidate, committee)
    enrichment_payload = enrichment.to_dict() if enrichment else None
    payload = build_committee_pick_output(
        candidate=candidate,
        research=research,
        model=model,
        arbiter=committee,
        enrichment_summary=enrichment_payload,
    )
    text = format_committee_pick_output(
        candidate=candidate,
        research=research,
        model=model,
        arbiter=committee,
        enrichment_summary=enrichment_payload,
    )
    final_decision = committee.final_decision.value
    effective_legacy_decision = committee.final_decision
    if committee.final_decision == FinalDecision.BET_SUBSTITUTE and not allow_bet_substitutes():
        effective_legacy_decision = FinalDecision.NO_BET
    legacy_status = legacy_decision_status(effective_legacy_decision)
    reason = _committee_reason_summary(
        candidate=candidate,
        research=research,
        committee=committee,
        payload=payload,
        enrichment=enrichment,
    )
    effective_parlay_suitability = str(committee.metadata.get("effective_parlay_suitability") or model.parlay_suitability or "")
    effective_stake_abs = float(candidate.get("stake_abs", 0.0) or 0.0)
    effective_kelly_pct = float(candidate.get("kelly_stake_pct", 0.0) or 0.0)
    if committee.final_decision not in {FinalDecision.BET, FinalDecision.BET_SUBSTITUTE}:
        effective_parlay_suitability = "blocked"
        effective_stake_abs = 0.0
        effective_kelly_pct = 0.0

    enriched = {
        **published_view,
        "committee_enabled": True,
        "committee_source_bucket": source_bucket,
        "committee": payload,
        "committee_details_text": text,
        "committee_final_decision": final_decision,
        "committee_agreement_status": committee.agreement_status.value,
        "committee_veto_flags": [flag.value for flag in committee.veto_flags],
        "committee_reason": reason if reason else "Committee review completed.",
        "committee_better_substitute": committee.better_substitute or "",
        "committee_parlay_suitability": effective_parlay_suitability,
        "committee_effective_risk_tier": str(committee.metadata.get("effective_risk_tier") or model.risk_tier or ""),
        "committee_effective_stake_abs": round(effective_stake_abs, 2),
        "committee_effective_kelly_pct": round(effective_kelly_pct, 4),
        "committee_show_details": show_committee_details(),
        "committee_enrichment": enrichment_payload if enrichment_payload else {
            "triggered": False,
            "trigger_reason": [],
            "missing_evidence_searched": [],
            "sources_checked": [],
            "sources_found": [],
            "source_quality": "",
            "providers_attempted": [],
            "providers_succeeded": [],
            "providers_failed": [],
            "provider_failure_reasons": {},
            "api_football_status": "",
            "availability_status": "",
            "news_context_status": "",
            "feature_cache_status": "",
            "standings_status": "",
            "fixture_status": "",
            "probable_lineup_status": "",
            "probable_pitcher_status": "",
            "pitcher_change_status": "",
            "home_pitcher": "",
            "away_pitcher": "",
            "pitcher_handedness_status": "",
            "lineup_status": "",
            "injury_status": "",
            "suspension_status": "",
            "goalkeeper_status": "",
            "motivation_status": "",
            "rotation_status": "",
            "fixture_congestion_status": "",
            "home_away_form_status": "",
            "xg_context_status": "",
            "bullpen_status": "",
            "weather_status": "",
            "park_factor_status": "",
            "travel_rest_status": "",
            "market_fit_status": "",
            "surface_status": "",
            "ranking_elo_status": "",
            "injury_retirement_status": "",
            "fatigue_status": "",
            "tournament_context_status": "",
            "style_matchup_status": "",
            "evidence_before": research.evidence_status,
            "evidence_after": research.evidence_status,
            "concrete_score_before": research.concrete_info_score,
            "concrete_score_after": research.concrete_info_score,
            "remaining_missing_evidence": list(research.missing_evidence),
            "final_arbiter_decision": committee.final_decision.value,
        },
        "committee_enrichment_source_quality": ((enrichment_payload or {}).get("source_quality", "") if enrichment_payload else ""),
        "research_mind_verdict": research.research_verdict.value,
        "research_mind_sport": research.sport,
        "research_mind_confidence": research.confidence,
        "research_mind_main_evidence": list(research.main_evidence),
        "research_mind_main_risks": list(research.main_risks),
        "research_mind_data_freshness": research.data_freshness,
        "research_mind_evidence_status": research.evidence_status,
        "research_mind_concrete_info_score": research.concrete_info_score,
        "research_mind_source_count": research.source_count,
        "research_mind_source_quality_summary": research.source_quality_summary,
        "research_mind_fixture_verified": research.fixture_verified,
        "research_mind_odds_age_minutes": research.odds_age_minutes,
        "research_mind_odds_freshness_status": research.odds_freshness_status,
        "research_mind_market_availability_status": research.market_availability_status,
        "research_mind_lineup_status": research.lineup_status,
        "research_mind_injury_status": research.injury_status,
        "research_mind_motivation_status": research.motivation_status,
        "research_mind_rotation_status": research.rotation_status,
        "research_mind_missing_evidence": list(research.missing_evidence),
        "research_mind_sport_specific_missing_evidence": list(research.sport_specific_missing_evidence),
        "research_mind_conflicting_evidence": list(research.conflicting_evidence),
        "research_mind_evidence_notes": list(research.evidence_notes),
        "research_mind_suggested_better_market": research.suggested_better_market or "",
        "research_mind_sources_checked": list(research.sources_checked),
        "model_mind_verdict": model.model_verdict.value,
        "model_mind_probability": model.model_probability,
        "model_mind_market_implied_probability": model.market_implied_probability,
        "model_mind_vig_free_probability": model.vig_free_market_probability,
        "model_mind_fair_odds": model.fair_odds,
        "model_mind_minimum_acceptable_odds": model.minimum_acceptable_odds,
        "model_mind_current_odds": model.current_odds,
        "model_mind_edge": model.estimated_edge,
        "model_mind_confidence_range": list(model.confidence_interval),
        "model_mind_risk_tier": str(committee.metadata.get("effective_risk_tier") or model.risk_tier or ""),
        "model_mind_base_risk_tier": model.risk_tier,
        "model_mind_suggested_market": model.suggested_market or "",
        "decision_status": legacy_status,
        "decision_reason": reason if reason else "Committee review completed.",
    }
    if enrichment_payload:
        for key, value in enrichment_payload.items():
            enriched[f"committee_enrichment_{key}"] = value
    if committee.final_decision == FinalDecision.BET:
        enriched["publish_ready"] = True
    return enriched


def _committee_published_view(candidate: dict[str, Any], committee: CommitteeDecision) -> dict[str, Any]:
    if committee.final_decision != FinalDecision.BET_SUBSTITUTE or not allow_bet_substitutes():
        return dict(candidate)

    substitute_candidate = candidate.get("substitute_candidate")
    if not isinstance(substitute_candidate, dict):
        return dict(candidate)

    return {
        **candidate,
        **substitute_candidate,
        "original_team": candidate.get("team", ""),
        "original_market": candidate.get("market", ""),
        "original_odds": candidate.get("odds"),
        "original_edge": candidate.get("edge"),
        "original_ml_prob": candidate.get("ml_prob"),
        "published_from_substitute": True,
    }


def _bucket_committee_candidate(
    *,
    source_bucket: str,
    candidate: dict[str, Any],
    committee: CommitteeDecision,
    published: list[dict[str, Any]],
    review: list[dict[str, Any]],
    suppressed: list[dict[str, Any]],
) -> None:
    if source_bucket == "suppressed":
        reason = str(candidate.get("suppression_reason") or candidate.get("committee_reason") or "suppressed before committee review").strip()
        suppressed.append({
            **candidate,
            "publish_ready": False,
            "suppressed": True,
            "suppression_reason": reason,
            "decision_status": "NO BET",
            "decision_reason": reason,
        })
        return

    final_decision = committee.final_decision
    if source_bucket == "published" and final_decision == FinalDecision.BET:
        published.append({**candidate, "publish_ready": True})
        return
    if source_bucket == "published" and final_decision == FinalDecision.BET_SUBSTITUTE and allow_bet_substitutes():
        published.append({**candidate, "publish_ready": True})
        return
    if final_decision in {FinalDecision.HOLD, FinalDecision.WAIT_FOR_LINEUPS}:
        review.append({**candidate, "review_required": True, "review_reason": candidate.get("committee_reason", "")})
        return
    suppressed.append({**candidate, "suppressed": True, "suppression_reason": candidate.get("committee_reason", "")})
