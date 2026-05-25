from __future__ import annotations

from typing import Any

from .contracts import CommitteeDecision, ModelMindDecision, ResearchMindDecision


def _string_list(values: tuple[str, ...] | list[str] | None) -> list[str]:
    if not values:
        return []
    return [str(value) for value in values if str(value).strip()]


def _format_float(value: float | None, decimals: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{decimals}f}"


def _format_probability(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _format_confidence_range(interval: tuple[float | None, float | None] | None) -> str:
    if not interval or len(interval) != 2:
        return "n/a"
    low, high = interval
    if low is None or high is None:
        return "n/a"
    return f"{low:.4f}–{high:.4f}"


def _game_label(candidate: dict[str, Any]) -> str:
    home = str(candidate.get("home", "") or "").strip()
    away = str(candidate.get("away", "") or "").strip()
    if home and away:
        return f"{home} vs {away}"
    return str(candidate.get("match_id", "") or "").strip() or "n/a"


def _final_explanation(
    *,
    candidate: dict[str, Any],
    research: ResearchMindDecision,
    model: ModelMindDecision,
    arbiter: CommitteeDecision,
) -> str:
    parts: list[str] = []
    if arbiter.final_decision.value == "BET":
        parts.append("Research has no major objection and the model still clears the price and confidence gates.")
    elif arbiter.final_decision.value == "BET_SUBSTITUTE":
        substitute = arbiter.better_substitute or "a safer substitute"
        parts.append(f"The original pick failed committee checks, but {substitute} passed instead.")
    elif arbiter.final_decision.value == "WAIT_FOR_LINEUPS":
        parts.append("The play is blocked until lineups/starter confirmations arrive.")
    elif arbiter.final_decision.value == "HOLD":
        parts.append("The committee found material uncertainty, so the pick should stay on hold.")
    elif arbiter.final_decision.value == "AVOID":
        parts.append("The committee found critical contextual or integrity problems and vetoed the play.")
    else:
        parts.append("The committee rejected the price/value case for the original pick.")

    suggested_market = str(
        arbiter.metadata.get("suggested_market")
        or model.suggested_market
        or research.suggested_better_market
        or ""
    ).strip()
    current_market = str(candidate.get("market", "") or "").strip()
    if suggested_market and suggested_market != current_market:
        parts.append(f"A better market fit appears to be {suggested_market}.")

    if arbiter.reasons:
        parts.append(str(arbiter.reasons[0]))

    return " ".join(part for part in parts if part).strip()


def build_committee_pick_output(
    *,
    candidate: dict[str, Any],
    research: ResearchMindDecision,
    model: ModelMindDecision,
    arbiter: CommitteeDecision,
    enrichment_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a structured committee output payload for one pick.

    This is a formatting layer only and does not alter existing scan/webapp
    flows. Callers can serialize this payload directly or render it with
    `format_committee_pick_output()`.
    """

    reason = "; ".join(_string_list(arbiter.reasons)) or "n/a"
    final_explanation = _final_explanation(
        candidate=candidate,
        research=research,
        model=model,
        arbiter=arbiter,
    )

    return {
        "game": _game_label(candidate),
        "original_pick": str(candidate.get("team", "") or "n/a"),
        "market": str(candidate.get("market", "") or "n/a"),
        "odds": model.current_odds,
        "research_mind": {
            "sport": research.sport or str(candidate.get("sport", "") or "n/a"),
            "verdict": research.research_verdict.value,
            "confidence": research.confidence,
            "main_evidence": _string_list(research.main_evidence),
            "main_risks": _string_list(research.main_risks),
            "data_freshness": research.data_freshness or "n/a",
            "evidence_status": research.evidence_status or "n/a",
            "concrete_info_score": research.concrete_info_score,
            "source_count": research.source_count,
            "source_quality_summary": research.source_quality_summary or "n/a",
            "fixture_verified": research.fixture_verified,
            "odds_age_minutes": research.odds_age_minutes,
            "odds_freshness_status": research.odds_freshness_status or "n/a",
            "market_availability_status": research.market_availability_status or "n/a",
            "lineup_status": research.lineup_status or "n/a",
            "injury_status": research.injury_status or "n/a",
            "motivation_status": research.motivation_status or "n/a",
            "rotation_status": research.rotation_status or "n/a",
            "missing_evidence": _string_list(research.missing_evidence),
            "sport_specific_missing_evidence": _string_list(research.sport_specific_missing_evidence or research.missing_evidence),
            "conflicting_evidence": _string_list(research.conflicting_evidence),
            "evidence_notes": _string_list(research.evidence_notes),
            "suggested_better_market": research.suggested_better_market or "",
            "sources_checked": _string_list(research.sources_checked),
        },
        "model_mind": {
            "verdict": model.model_verdict.value,
            "model_probability": model.model_probability,
            "market_implied_probability": model.market_implied_probability,
            "vig_free_probability": model.vig_free_market_probability,
            "fair_odds": model.fair_odds,
            "minimum_acceptable_odds": model.minimum_acceptable_odds,
            "current_odds": model.current_odds,
            "edge": model.estimated_edge,
            "confidence_range": list(model.confidence_interval),
            "risk_tier": model.risk_tier,
            "suggested_market": model.suggested_market or "",
        },
        "arbiter": {
            "agreement_status": arbiter.agreement_status.value,
            "veto_flags": [flag.value for flag in arbiter.veto_flags],
            "final_decision": arbiter.final_decision.value,
            "reason": reason,
            "better_substitute": arbiter.better_substitute or "",
            "parlay_suitability": (
                "blocked"
                if arbiter.final_decision.value not in {"BET", "BET_SUBSTITUTE"}
                else str(arbiter.metadata.get("effective_parlay_suitability") or model.parlay_suitability or "")
            ),
            "effective_risk_tier": str(arbiter.metadata.get("effective_risk_tier") or model.risk_tier or ""),
            "final_explanation": final_explanation,
        },
        "evidence_enrichment": {
            "triggered": bool((enrichment_summary or {}).get("triggered", False)),
            "trigger_reason": list((enrichment_summary or {}).get("trigger_reason", []) or []),
            "missing_evidence_searched": list((enrichment_summary or {}).get("missing_evidence_searched", []) or []),
            "sources_checked": list((enrichment_summary or {}).get("sources_checked", []) or []),
            "sources_found": list((enrichment_summary or {}).get("sources_found", []) or []),
            "source_quality": str((enrichment_summary or {}).get("source_quality", "") or ""),
            "providers_attempted": list((enrichment_summary or {}).get("providers_attempted", []) or []),
            "providers_succeeded": list((enrichment_summary or {}).get("providers_succeeded", []) or []),
            "providers_failed": list((enrichment_summary or {}).get("providers_failed", []) or []),
            "provider_failure_reasons": dict((enrichment_summary or {}).get("provider_failure_reasons", {}) or {}),
            "api_football_status": str((enrichment_summary or {}).get("api_football_status", "") or ""),
            "availability_status": str((enrichment_summary or {}).get("availability_status", "") or ""),
            "news_context_status": str((enrichment_summary or {}).get("news_context_status", "") or ""),
            "feature_cache_status": str((enrichment_summary or {}).get("feature_cache_status", "") or ""),
            "standings_status": str((enrichment_summary or {}).get("standings_status", "") or ""),
            "fixture_status": str((enrichment_summary or {}).get("fixture_status", "") or ""),
            "probable_lineup_status": str((enrichment_summary or {}).get("probable_lineup_status", "") or ""),
            "probable_pitcher_status": str((enrichment_summary or {}).get("probable_pitcher_status", "") or ""),
            "pitcher_change_status": str((enrichment_summary or {}).get("pitcher_change_status", "") or ""),
            "home_pitcher": str((enrichment_summary or {}).get("home_pitcher", "") or ""),
            "away_pitcher": str((enrichment_summary or {}).get("away_pitcher", "") or ""),
            "pitcher_handedness_status": str((enrichment_summary or {}).get("pitcher_handedness_status", "") or ""),
            "lineup_status": str((enrichment_summary or {}).get("lineup_status", "") or ""),
            "injury_status": str((enrichment_summary or {}).get("injury_status", "") or ""),
            "suspension_status": str((enrichment_summary or {}).get("suspension_status", "") or ""),
            "goalkeeper_status": str((enrichment_summary or {}).get("goalkeeper_status", "") or ""),
            "motivation_status": str((enrichment_summary or {}).get("motivation_status", "") or ""),
            "rotation_status": str((enrichment_summary or {}).get("rotation_status", "") or ""),
            "fixture_congestion_status": str((enrichment_summary or {}).get("fixture_congestion_status", "") or ""),
            "home_away_form_status": str((enrichment_summary or {}).get("home_away_form_status", "") or ""),
            "xg_context_status": str((enrichment_summary or {}).get("xg_context_status", "") or ""),
            "bullpen_status": str((enrichment_summary or {}).get("bullpen_status", "") or ""),
            "weather_status": str((enrichment_summary or {}).get("weather_status", "") or ""),
            "park_factor_status": str((enrichment_summary or {}).get("park_factor_status", "") or ""),
            "travel_rest_status": str((enrichment_summary or {}).get("travel_rest_status", "") or ""),
            "market_fit_status": str((enrichment_summary or {}).get("market_fit_status", "") or ""),
            "surface_status": str((enrichment_summary or {}).get("surface_status", "") or ""),
            "ranking_elo_status": str((enrichment_summary or {}).get("ranking_elo_status", "") or ""),
            "injury_retirement_status": str((enrichment_summary or {}).get("injury_retirement_status", "") or ""),
            "fatigue_status": str((enrichment_summary or {}).get("fatigue_status", "") or ""),
            "tournament_context_status": str((enrichment_summary or {}).get("tournament_context_status", "") or ""),
            "style_matchup_status": str((enrichment_summary or {}).get("style_matchup_status", "") or ""),
            "evidence_before": str((enrichment_summary or {}).get("evidence_before", "") or ""),
            "evidence_after": str((enrichment_summary or {}).get("evidence_after", research.evidence_status) or ""),
            "concrete_score_before": int((enrichment_summary or {}).get("concrete_score_before", research.concrete_info_score) or 0),
            "concrete_score_after": int((enrichment_summary or {}).get("concrete_score_after", research.concrete_info_score) or 0),
            "remaining_missing_evidence": list((enrichment_summary or {}).get("remaining_missing_evidence", list(research.missing_evidence)) or []),
            "final_arbiter_decision": str((enrichment_summary or {}).get("final_arbiter_decision", arbiter.final_decision.value) or arbiter.final_decision.value),
        },
    }


def format_committee_pick_output(
    *,
    candidate: dict[str, Any],
    research: ResearchMindDecision,
    model: ModelMindDecision,
    arbiter: CommitteeDecision,
    enrichment_summary: dict[str, Any] | None = None,
) -> str:
    payload = build_committee_pick_output(
        candidate=candidate,
        research=research,
        model=model,
        arbiter=arbiter,
        enrichment_summary=enrichment_summary,
    )

    lines = [
        f"Game: {payload['game']}",
        f"Original pick: {payload['original_pick']}",
        f"Market: {payload['market']}",
        f"Odds: {_format_float(payload['odds'])}",
        "",
        "Research Mind:",
        f"- Verdict: {payload['research_mind']['verdict']}",
        f"- Sport: {payload['research_mind']['sport']}",
        f"- Confidence: {payload['research_mind']['confidence']}",
        f"- Main evidence: {', '.join(payload['research_mind']['main_evidence']) or 'n/a'}",
        f"- Main risks: {', '.join(payload['research_mind']['main_risks']) or 'n/a'}",
        f"- Data freshness: {payload['research_mind']['data_freshness'] or 'n/a'}",
        f"- Evidence status: {payload['research_mind']['evidence_status']}",
        f"- Concrete info score: {payload['research_mind']['concrete_info_score']}",
        f"- Source count: {payload['research_mind']['source_count']}",
        f"- Source quality: {payload['research_mind']['source_quality_summary']}",
        f"- Fixture verified: {payload['research_mind']['fixture_verified']}",
        f"- Odds age minutes: {payload['research_mind']['odds_age_minutes'] if payload['research_mind']['odds_age_minutes'] is not None else 'n/a'}",
        f"- Odds freshness status: {payload['research_mind']['odds_freshness_status']}",
        f"- Market availability status: {payload['research_mind']['market_availability_status']}",
        f"- Lineup status: {payload['research_mind']['lineup_status']}",
        f"- Injury status: {payload['research_mind']['injury_status']}",
        f"- Motivation status: {payload['research_mind']['motivation_status']}",
        f"- Rotation status: {payload['research_mind']['rotation_status']}",
        f"- Missing evidence: {', '.join(payload['research_mind']['missing_evidence']) or 'n/a'}",
        f"- Sport-specific missing evidence: {', '.join(payload['research_mind']['sport_specific_missing_evidence']) or 'n/a'}",
        f"- Conflicting evidence: {', '.join(payload['research_mind']['conflicting_evidence']) or 'n/a'}",
        f"- Evidence notes: {', '.join(payload['research_mind']['evidence_notes']) or 'n/a'}",
        f"- Suggested better market: {payload['research_mind']['suggested_better_market'] or 'n/a'}",
        f"- Sources checked: {', '.join(payload['research_mind']['sources_checked']) or 'n/a'}",
        "",
        "Model Mind:",
        f"- Verdict: {payload['model_mind']['verdict']}",
        f"- Model probability: {_format_probability(payload['model_mind']['model_probability'])}",
        f"- Market implied probability: {_format_probability(payload['model_mind']['market_implied_probability'])}",
        f"- Vig-free probability: {_format_probability(payload['model_mind']['vig_free_probability'])}",
        f"- Fair odds: {_format_float(payload['model_mind']['fair_odds'])}",
        f"- Minimum acceptable odds: {_format_float(payload['model_mind']['minimum_acceptable_odds'])}",
        f"- Current odds: {_format_float(payload['model_mind']['current_odds'])}",
        f"- Edge: {_format_probability(payload['model_mind']['edge'])}",
        f"- Confidence range: {_format_confidence_range(model.confidence_interval)}",
        f"- Risk tier: {payload['model_mind']['risk_tier'] or 'n/a'}",
        f"- Suggested market: {payload['model_mind']['suggested_market'] or 'n/a'}",
        "",
        "Arbiter:",
        f"- Agreement status: {payload['arbiter']['agreement_status']}",
        f"- Veto flags: {', '.join(payload['arbiter']['veto_flags']) or 'none'}",
        f"- Final decision: {payload['arbiter']['final_decision']}",
        f"- Reason: {payload['arbiter']['reason']}",
        f"- Better substitute: {payload['arbiter']['better_substitute'] or 'n/a'}",
        f"- Parlay suitability: {payload['arbiter']['parlay_suitability'] or 'n/a'}",
        f"- Effective risk tier: {payload['arbiter']['effective_risk_tier'] or 'n/a'}",
        f"- Final explanation: {payload['arbiter']['final_explanation'] or 'n/a'}",
        "",
        "Evidence Enrichment:",
        f"- Triggered: {'yes' if payload['evidence_enrichment']['triggered'] else 'no'}",
        f"- Trigger reason: {', '.join(payload['evidence_enrichment']['trigger_reason']) or 'n/a'}",
        f"- Missing evidence searched: {', '.join(payload['evidence_enrichment']['missing_evidence_searched']) or 'n/a'}",
        f"- Sources checked: {', '.join(payload['evidence_enrichment']['sources_checked']) or 'n/a'}",
        f"- Sources found: {', '.join(payload['evidence_enrichment']['sources_found']) or 'n/a'}",
        f"- Source quality: {payload['evidence_enrichment']['source_quality'] or 'n/a'}",
        f"- Providers attempted: {', '.join(payload['evidence_enrichment']['providers_attempted']) or 'n/a'}",
        f"- Providers succeeded: {', '.join(payload['evidence_enrichment']['providers_succeeded']) or 'n/a'}",
        f"- Providers failed: {', '.join(payload['evidence_enrichment']['providers_failed']) or 'n/a'}",
        f"- Provider failure reasons: {payload['evidence_enrichment']['provider_failure_reasons'] or 'n/a'}",
        f"- API-Football status: {payload['evidence_enrichment']['api_football_status'] or 'n/a'}",
        f"- Availability status: {payload['evidence_enrichment']['availability_status'] or 'n/a'}",
        f"- News context status: {payload['evidence_enrichment']['news_context_status'] or 'n/a'}",
        f"- Feature cache status: {payload['evidence_enrichment']['feature_cache_status'] or 'n/a'}",
        f"- Standings status: {payload['evidence_enrichment']['standings_status'] or 'n/a'}",
        f"- Fixture status: {payload['evidence_enrichment']['fixture_status'] or 'n/a'}",
        f"- Probable lineup status: {payload['evidence_enrichment']['probable_lineup_status'] or 'n/a'}",
        f"- Probable pitcher status: {payload['evidence_enrichment']['probable_pitcher_status'] or 'n/a'}",
        f"- Pitcher change status: {payload['evidence_enrichment']['pitcher_change_status'] or 'n/a'}",
        f"- Home pitcher: {payload['evidence_enrichment']['home_pitcher'] or 'n/a'}",
        f"- Away pitcher: {payload['evidence_enrichment']['away_pitcher'] or 'n/a'}",
        f"- Pitcher handedness status: {payload['evidence_enrichment']['pitcher_handedness_status'] or 'n/a'}",
        f"- Lineup status: {payload['evidence_enrichment']['lineup_status'] or 'n/a'}",
        f"- Injury status: {payload['evidence_enrichment']['injury_status'] or 'n/a'}",
        f"- Suspension status: {payload['evidence_enrichment']['suspension_status'] or 'n/a'}",
        f"- Goalkeeper status: {payload['evidence_enrichment']['goalkeeper_status'] or 'n/a'}",
        f"- Motivation status: {payload['evidence_enrichment']['motivation_status'] or 'n/a'}",
        f"- Rotation status: {payload['evidence_enrichment']['rotation_status'] or 'n/a'}",
        f"- Fixture congestion status: {payload['evidence_enrichment']['fixture_congestion_status'] or 'n/a'}",
        f"- Home/away form status: {payload['evidence_enrichment']['home_away_form_status'] or 'n/a'}",
        f"- xG context status: {payload['evidence_enrichment']['xg_context_status'] or 'n/a'}",
        f"- Bullpen status: {payload['evidence_enrichment']['bullpen_status'] or 'n/a'}",
        f"- Weather status: {payload['evidence_enrichment']['weather_status'] or 'n/a'}",
        f"- Park factor status: {payload['evidence_enrichment']['park_factor_status'] or 'n/a'}",
        f"- Travel/rest status: {payload['evidence_enrichment']['travel_rest_status'] or 'n/a'}",
        f"- Market fit status: {payload['evidence_enrichment']['market_fit_status'] or 'n/a'}",
        f"- Surface status: {payload['evidence_enrichment']['surface_status'] or 'n/a'}",
        f"- Ranking/Elo status: {payload['evidence_enrichment']['ranking_elo_status'] or 'n/a'}",
        f"- Injury/retirement status: {payload['evidence_enrichment']['injury_retirement_status'] or 'n/a'}",
        f"- Fatigue status: {payload['evidence_enrichment']['fatigue_status'] or 'n/a'}",
        f"- Tournament context status: {payload['evidence_enrichment']['tournament_context_status'] or 'n/a'}",
        f"- Style matchup status: {payload['evidence_enrichment']['style_matchup_status'] or 'n/a'}",
        f"- Evidence before: {payload['evidence_enrichment']['evidence_before'] or 'n/a'}",
        f"- Evidence after: {payload['evidence_enrichment']['evidence_after'] or 'n/a'}",
        f"- Concrete score before/after: {payload['evidence_enrichment']['concrete_score_before']} → {payload['evidence_enrichment']['concrete_score_after']}",
        f"- Remaining missing evidence: {', '.join(payload['evidence_enrichment']['remaining_missing_evidence']) or 'n/a'}",
        f"- Final Arbiter decision: {payload['evidence_enrichment']['final_arbiter_decision'] or 'n/a'}",
    ]
    return "\n".join(lines)
