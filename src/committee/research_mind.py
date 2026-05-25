from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from config import settings
from src.markets.freshness import audit_candidate_freshness

from .contracts import ResearchMindDecision, ResearchVerdict, VetoFlag

_CONCRETE_SOURCE_NAMES = {
    "api_football",
    "api_sports",
    "api_sports_basketball",
    "football_data",
    "mlb_api",
    "mlb_stats_api",
    "nhl_api",
    "balldontlie",
    "espn",
    "rotowire",
    "mysportsfeeds",
    "sportmonks",
    "newsapi",
    "sofascore",
    "openweather",
    "onefootball.com",
    "sportsmole.co.uk",
    "thestatszone.com",
    "covers.com",
    "tennis_feature_cache",
    "mlb_feature_cache",
    "soccer_feature_cache",
    "team_official",
    "league_official",
}
_WEAK_SOURCE_NAMES = {"feature_snapshot", "odds_snapshot", "bookmaker", "bookmaker_or_odds_feed"}
_MODEL_DERIVED_HINTS = (
    "expected-goals",
    "expected goals",
    "chance-quality",
    "chance quality",
    "rolling scoring",
    "concession profile",
    "xg",
    "model",
    "statistical",
    "profile",
)
_LINEUP_SENSITIVE_MARKETS = {
    "moneyline",
    "spreads",
    "totals",
    "double_chance",
    "draw_no_bet",
    "h2h",
}
_CORE_EVIDENCE_SPORTS = {"soccer", "basketball", "mlb", "nhl"}
_CONSERVATIVE_RESEARCH_SPORTS = {"tennis", "tennis_wta"}


def _as_int(value: Any) -> int:
    try:
        if value is None or value == "":
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _dedupe(items: list[str]) -> list[str]:
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


class ContextResearchMind:
    """
    Thin adapter around the existing contextual and freshness layers.

    This does not alter the live scan flow. It converts the current research
    signals into a stable committee payload for future orchestration.
    """

    def __init__(self) -> None:
        committee_cfg = (settings or {}).get("committee") or {}
        self.block_stale_data = bool(committee_cfg.get("block_stale_data", True))
        self.block_missing_lineups_near_kickoff = bool(committee_cfg.get("block_missing_lineups_near_kickoff", True))
        self.min_sources_for_high = int(committee_cfg.get("research_min_sources_for_high", 2) or 2)
        self.min_concrete_score_for_high = int(committee_cfg.get("research_min_concrete_score_for_high", 75) or 75)

    def evaluate(self, candidate: dict[str, Any]) -> ResearchMindDecision:
        sport = str(candidate.get("sport", "") or "").strip().lower()
        context = candidate.get("scraped_context") or {}
        audit = audit_candidate_freshness(candidate)
        lead_hours = self._lead_hours(candidate)
        explicit_status = str(
            candidate.get("match_status")
            or candidate.get("game_status")
            or candidate.get("status")
            or ""
        ).strip().lower()
        sources_checked = tuple(self._collect_sources(candidate, context))
        source_count = len(sources_checked)
        odds_age_minutes = self._odds_age_minutes(candidate)
        odds_freshness_status = self._odds_freshness_status(candidate, audit)
        lineup_status = self._lineup_status(audit, context)
        injury_status = self._injury_status(candidate, context, audit)
        special_fixture = self._special_fixture_context(context)
        motivation_status = self._motivation_status(sport, context, special_fixture)
        rotation_status = self._rotation_status(sport, context, audit, special_fixture)
        source_quality_summary = self._source_quality_summary(sources_checked)
        model_derived_only = self._model_derived_only(candidate, sources_checked)
        market_availability_status = self._market_availability_status(candidate)
        sport_specific_high_ready = self._sport_specific_high_ready(
            sport=sport,
            candidate=candidate,
            context=context,
            lineup_status=lineup_status,
            injury_status=injury_status,
            motivation_status=motivation_status,
        )

        evidence = self._collect_evidence(
            candidate=candidate,
            context=context,
            audit=audit,
            odds_freshness_status=odds_freshness_status,
            lineup_status=lineup_status,
            injury_status=injury_status,
            motivation_status=motivation_status,
            rotation_status=rotation_status,
        )
        missing_evidence, critical_missing = self._missing_evidence(
            candidate=candidate,
            context=context,
            audit=audit,
            source_count=source_count,
            source_quality_summary=source_quality_summary,
            odds_freshness_status=odds_freshness_status,
            market_availability_status=market_availability_status,
            lineup_status=lineup_status,
            injury_status=injury_status,
            motivation_status=motivation_status,
            rotation_status=rotation_status,
            special_fixture=special_fixture,
            lead_hours=lead_hours,
            sport=sport,
        )
        conflicting_evidence = self._conflicting_evidence(candidate, context, explicit_status)
        concrete_info_score = self._concrete_info_score(
            audit=audit,
            source_count=source_count,
            source_quality_summary=source_quality_summary,
            odds_freshness_status=odds_freshness_status,
            lineup_status=lineup_status,
            injury_status=injury_status,
            motivation_status=motivation_status,
            rotation_status=rotation_status,
            model_derived_only=model_derived_only,
            conflicting_evidence=conflicting_evidence,
            critical_missing=critical_missing,
        )
        evidence_status = self._evidence_status(
            sport=sport,
            audit=audit,
            source_count=source_count,
            source_quality_summary=source_quality_summary,
            model_derived_only=model_derived_only,
            missing_evidence=missing_evidence,
            critical_missing=critical_missing,
            conflicting_evidence=conflicting_evidence,
            concrete_info_score=concrete_info_score,
        )
        evidence_notes = self._evidence_notes(
            audit=audit,
            sources_checked=sources_checked,
            source_quality_summary=source_quality_summary,
            odds_age_minutes=odds_age_minutes,
            odds_freshness_status=odds_freshness_status,
            lineup_status=lineup_status,
            injury_status=injury_status,
            motivation_status=motivation_status,
            rotation_status=rotation_status,
            model_derived_only=model_derived_only,
        )
        risks = self._collect_risks(
            candidate=candidate,
            context=context,
            audit=audit,
            explicit_status=explicit_status,
            missing_evidence=missing_evidence,
            conflicting_evidence=conflicting_evidence,
            evidence_status=evidence_status,
            source_quality_summary=source_quality_summary,
            market_availability_status=market_availability_status,
            lineup_status=lineup_status,
            injury_status=injury_status,
        )
        suggested_better_market = self._suggested_market(candidate)
        wait_for_lineups_signal = self.block_missing_lineups_near_kickoff and audit.lineup_freshness == "missing"
        veto_flags = self._build_veto_flags(audit, context, explicit_status)
        data_freshness = self._data_freshness(audit, sources_checked)

        verdict = ResearchVerdict.AGREE
        if explicit_status in {"postponed", "cancelled", "canceled", "suspended", "abandoned"}:
            verdict = ResearchVerdict.AVOID
        elif audit.match_status == "finished":
            verdict = ResearchVerdict.AVOID
        elif sport in _CONSERVATIVE_RESEARCH_SPORTS and audit.match_status == "live":
            verdict = ResearchVerdict.HOLD
        elif audit.match_status == "live":
            verdict = ResearchVerdict.AVOID
        elif str(candidate.get("context_referee_decision", "") or "").upper() == "VETO":
            verdict = ResearchVerdict.AVOID
        elif str(candidate.get("context_referee_decision", "") or "").upper() == "REVIEW":
            verdict = ResearchVerdict.HOLD
        elif not audit.fixture_verified:
            verdict = ResearchVerdict.HOLD
        elif self.block_stale_data and (
            audit.odds_freshness == "stale"
            or audit.injury_news_freshness == "stale"
            or audit.standings_freshness == "stale"
        ):
            verdict = ResearchVerdict.HOLD
        elif self._severe_rotation_risk(context, audit):
            verdict = ResearchVerdict.AVOID
        elif wait_for_lineups_signal:
            verdict = ResearchVerdict.HOLD
        elif self._tennis_market_hold(
            sport=sport,
            candidate=candidate,
            context=context,
            injury_status=injury_status,
            explicit_status=explicit_status,
        ):
            verdict = ResearchVerdict.HOLD
        elif self._basketball_market_hold(
            sport=sport,
            candidate=candidate,
            context=context,
            lineup_status=lineup_status,
            injury_status=injury_status,
        ):
            verdict = ResearchVerdict.HOLD
        elif self._nhl_goalie_market_hold(
            sport=sport,
            candidate=candidate,
            lineup_status=lineup_status,
        ):
            verdict = ResearchVerdict.HOLD
        elif self._moderate_rotation_risk(context):
            verdict = ResearchVerdict.HOLD
        elif self._research_disagrees(context):
            verdict = ResearchVerdict.DISAGREE

        confidence = self._confidence_for(
            verdict=verdict,
            evidence_status=evidence_status,
            concrete_info_score=concrete_info_score,
            source_count=source_count,
            source_quality_summary=source_quality_summary,
            fixture_verified=audit.fixture_verified,
            odds_freshness_status=odds_freshness_status,
            critical_missing=critical_missing,
            conflicting_evidence=conflicting_evidence,
            model_derived_only=model_derived_only,
            wait_for_lineups_signal=wait_for_lineups_signal,
            evidence=evidence,
            sport_specific_high_ready=sport_specific_high_ready,
        )

        return ResearchMindDecision(
            research_verdict=verdict,
            sport=sport,
            confidence=confidence,
            main_evidence=tuple(evidence[:5]),
            main_risks=tuple(risks[:5]),
            suggested_better_market=suggested_better_market,
            data_freshness=data_freshness,
            sources_checked=sources_checked,
            evidence_status=evidence_status,
            concrete_info_score=concrete_info_score,
            source_count=source_count,
            source_quality_summary=source_quality_summary,
            fixture_verified=audit.fixture_verified,
            odds_age_minutes=odds_age_minutes,
            odds_freshness_status=odds_freshness_status,
            market_availability_status=market_availability_status,
            lineup_status=lineup_status,
            injury_status=injury_status,
            motivation_status=motivation_status,
            rotation_status=rotation_status,
            missing_evidence=tuple(missing_evidence[:6]),
            sport_specific_missing_evidence=tuple(missing_evidence[:6]),
            conflicting_evidence=tuple(conflicting_evidence[:4]),
            evidence_notes=tuple(evidence_notes[:8]),
            wait_for_lineups_signal=wait_for_lineups_signal,
            veto_flags=veto_flags,
            metadata={
                "match_status": audit.match_status,
                "fixture_verified": audit.fixture_verified,
                "odds_freshness": audit.odds_freshness,
                "lineup_freshness": audit.lineup_freshness,
                "injury_news_freshness": audit.injury_news_freshness,
                "standings_freshness": audit.standings_freshness,
                "odds_freshness_status": odds_freshness_status,
                "market_availability_status": market_availability_status,
                "lineup_status": lineup_status,
                "injury_status": injury_status,
                "motivation_status": motivation_status,
                "rotation_status": rotation_status,
                "source_count": source_count,
                "source_quality_summary": source_quality_summary,
                "concrete_info_score": concrete_info_score,
                "critical_missing_evidence": list(critical_missing),
                "model_derived_only": model_derived_only,
                "sport_specific_high_ready": sport_specific_high_ready,
            },
        )

    def _collect_evidence(
        self,
        *,
        candidate: dict[str, Any],
        context: dict[str, Any],
        audit,
        odds_freshness_status: str,
        lineup_status: str,
        injury_status: str,
        motivation_status: str,
        rotation_status: str,
    ) -> list[str]:
        evidence: list[str] = []
        for item in candidate.get("scraped_context_highlights") or []:
            text = str(item or "").strip()
            if text:
                evidence.append(text)
        if audit.fixture_verified:
            evidence.append("Fixture verified against the contextual source payload")
        if odds_freshness_status in {"fresh", "acceptable"}:
            evidence.append("Odds snapshot is recent enough for a pre-match research check")
        if lineup_status == "confirmed":
            evidence.append("Lineup / starter confirmation is available for the current decision window")
        if injury_status == "checked_fresh":
            evidence.append("Injury and team-news context was checked from a concrete source")
        if motivation_status == "checked":
            evidence.append("Motivation context was checked for the fixture state")
        if rotation_status == "checked":
            evidence.append("Rotation risk was checked for the fixture context")
        if context.get("is_playoff") or context.get("playoff_motivation"):
            evidence.append("Playoff or high-leverage motivation context detected")
        if str(candidate.get("sport", "") or "").strip().lower() == "nhl":
            profile = self._nhl_research_profile(
                candidate=candidate,
                context=context,
                lineup_status=lineup_status,
            )
            if profile["special_teams_checked"]:
                evidence.append("Special-teams context was checked from power-play and penalty-kill form")
            if profile["shots_xg_checked"]:
                evidence.append("Shots/xG structure was checked from recent chance-quality form")
            if profile["splits_checked"]:
                evidence.append("Home/away split context was checked for venue comfort and system stability")
            if profile["travel_checked"]:
                evidence.append("Travel fatigue context was checked for the NHL schedule spot")
            if profile["rest_checked"]:
                evidence.append("Rest/back-to-back context was checked for the NHL schedule spot")
        if str(candidate.get("sport", "") or "").strip().lower() == "basketball":
            profile = self._basketball_research_profile(
                candidate=candidate,
                context=context,
                lineup_status=lineup_status,
                injury_status=injury_status,
            )
            if profile["rest_checked"]:
                evidence.append("Rest/back-to-back context was checked for the NBA schedule spot")
            if profile["travel_checked"]:
                evidence.append("Travel context was checked for the NBA schedule spot")
            if profile["pace_checked"]:
                evidence.append("Pace context was checked for the NBA totals environment")
            if profile["ratings_checked"]:
                evidence.append("Offensive/defensive rating context was checked for the matchup")
            if profile["usage_checked"]:
                evidence.append("Usage and on-ball redistribution context was checked for key-player absences")
        if str(candidate.get("sport", "") or "").strip().lower() == "mlb":
            profile = self._mlb_research_profile(
                candidate=candidate,
                context=context,
                lineup_status=lineup_status,
            )
            if profile["starter_confirmation_checked"]:
                evidence.append("Probable starter confirmation was checked for the MLB game")
            if profile["pitcher_quality_checked"]:
                evidence.append("Starting-pitcher quality context was checked from MLB features or trusted enrichment data")
            if profile["lineup_context_checked"]:
                evidence.append("Lineup context was checked for the MLB fixture")
            if profile["bullpen_checked"]:
                evidence.append("Bullpen workload context was checked for the MLB market")
            if profile["weather_checked"]:
                evidence.append("Weather context was checked for the MLB venue")
            if profile["park_checked"]:
                evidence.append("Park or venue split context was checked for the MLB matchup")
            if profile["travel_checked"]:
                evidence.append("Travel/rest context was checked for the MLB series spot")
        if str(candidate.get("sport", "") or "").strip().lower() in _CONSERVATIVE_RESEARCH_SPORTS:
            profile = self._tennis_research_profile(candidate=candidate, context=context, injury_status=injury_status)
            if profile["surface_checked"]:
                evidence.append("Surface context was checked for the tennis matchup")
            if profile["recent_form_checked"]:
                evidence.append("Recent form context was checked for the tennis matchup")
            if profile["tournament_context_checked"]:
                evidence.append("Tournament round/context was checked for the tennis matchup")
            if profile["travel_checked"]:
                evidence.append("Travel/time-zone context was checked for the tennis matchup")
            if profile["h2h_style_checked"]:
                evidence.append("Head-to-head style context was checked for the tennis matchup")
            if profile["ranking_checked"]:
                evidence.append("Ranking/Elo context was checked for the tennis matchup")
            if profile["serve_return_checked"]:
                evidence.append("Serve/return matchup context was checked for the tennis market")
            if profile["fatigue_checked"]:
                evidence.append("Fatigue context was checked from recent tennis match load")
        return _dedupe(evidence)

    def _collect_risks(
        self,
        *,
        candidate: dict[str, Any],
        context: dict[str, Any],
        audit,
        explicit_status: str,
        missing_evidence: list[str],
        conflicting_evidence: list[str],
        evidence_status: str,
        source_quality_summary: str,
        market_availability_status: str,
        lineup_status: str,
        injury_status: str,
    ) -> list[str]:
        risks: list[str] = []
        for text in (audit.suppression_reason, audit.review_reason, audit.fixture_verification_reason):
            item = str(text or "").strip()
            if item:
                risks.append(item)
        if explicit_status in {"postponed", "cancelled", "canceled", "suspended", "abandoned"}:
            risks.append("Match is postponed/cancelled/suspended, so pre-match research is no longer actionable")
        injuries = (
            _as_int(context.get("home_injuries_count"))
            + _as_int(context.get("away_injuries_count"))
            + _as_int(context.get("home_suspensions_count"))
            + _as_int(context.get("away_suspensions_count"))
        )
        if injuries:
            risks.append("Injuries or suspensions are present in the latest availability context")
        if context.get("cup_rotation_risk") or context.get("european_rotation_risk"):
            risks.append("Rotation risk is elevated by cup or continental scheduling context")
        if context.get("final_day_volatility"):
            risks.append("End-of-season volatility can distort motivation and lineup reliability")
        if context.get("nothing_to_play_for"):
            risks.append("Selected side may have less to play for than the opponent")
        referee_decision = str(candidate.get("context_referee_decision", "") or "").strip().upper()
        referee_reason = str(candidate.get("context_referee_reason", "") or "").strip()
        if referee_decision == "VETO":
            risks.append("Legacy context referee found critical negative context")
        elif referee_decision == "REVIEW":
            risks.append(referee_reason or "Legacy context referee flagged material uncertainty")
        risks.extend(conflicting_evidence)
        risks.extend(missing_evidence)
        if market_availability_status != "available":
            risks.append("market availability is incomplete")
        if source_quality_summary == "weak":
            risks.append("limited concrete research evidence")
        if str(candidate.get("sport", "") or "").strip().lower() == "nhl":
            profile = self._nhl_research_profile(
                candidate=candidate,
                context=context,
                lineup_status=lineup_status,
            )
            if profile["totals_like_market"] and lineup_status != "confirmed":
                risks.append("NHL totals/team-total research is incomplete without confirmed goalie context")
            if profile["back_to_back"] and not profile["rest_checked"]:
                risks.append("NHL back-to-back risk is elevated because rest context was not fully checked")
        if str(candidate.get("sport", "") or "").strip().lower() == "basketball":
            profile = self._basketball_research_profile(
                candidate=candidate,
                context=context,
                lineup_status=lineup_status,
                injury_status=injury_status,
            )
            if profile["star_player_uncertain"]:
                risks.append("Star-player availability is still uncertain in the latest NBA injury report")
            if profile["back_to_back"] and not profile["rest_checked"]:
                risks.append("NBA back-to-back risk is elevated because rest context was not fully checked")
            if profile["totals_like_market"] and not profile["pace_checked"]:
                risks.append("NBA totals/team-total research is incomplete without pace context")
            if profile["usage_context_required"] and not profile["usage_checked"]:
                risks.append("Key-player absences need usage redistribution context before this NBA market is trustworthy")
        if str(candidate.get("sport", "") or "").strip().lower() == "mlb":
            profile = self._mlb_research_profile(
                candidate=candidate,
                context=context,
                lineup_status=lineup_status,
            )
            if profile["pitcher_change_detected"]:
                risks.append("Probable starter changed during the latest MLB evidence refresh")
            if profile["totals_or_spread_market"] and not profile["bullpen_checked"]:
                risks.append("MLB totals/run-line research is incomplete without bullpen workload context")
            if profile["weather_material"] and not profile["weather_checked"]:
                risks.append("MLB weather-sensitive market is missing current weather context")
            if profile["underdog_minus_one_half"] and not profile["market_fit_ok"]:
                risks.append("Aggressive underdog -1.5 MLB run-line lacks enough support from pitchers, bullpen, and lineup context")
        if str(candidate.get("sport", "") or "").strip().lower() in _CONSERVATIVE_RESEARCH_SPORTS:
            profile = self._tennis_research_profile(
                candidate=candidate,
                context=context,
                injury_status=injury_status,
            )
            if profile["injury_concern_present"] and not profile["injury_verified"]:
                risks.append("Tennis injury/retirement concern is unresolved in the available evidence")
            if profile["fatigue_flag"] and not profile["fatigue_checked"]:
                risks.append("Tennis fatigue risk is elevated because recent match load was not checked")
            if profile["totals_or_spread_market"] and not profile["serve_return_checked"]:
                risks.append("Tennis totals/spread research is incomplete without serve/return matchup context")
            if profile["live_pre_match_conflict"]:
                risks.append("Match appears live while the candidate still relies on pre-match odds context")
        if not risks and evidence_status in {"COMPLETE", "ACCEPTABLE"}:
            risks.append("No major risks detected from available evidence")
        elif not risks:
            risks.append("limited concrete research evidence")
        return _dedupe(risks)

    def _collect_sources(self, candidate: dict[str, Any], context: dict[str, Any]) -> list[str]:
        sources: list[str] = []
        for raw in candidate.get("scraped_context_sources") or []:
            source = str(raw or "").strip()
            if source:
                sources.append(source)
        for key in ("availability_source", "lineup_source"):
            source = str(context.get(key, "") or candidate.get(key, "") or "").strip()
            if source:
                sources.append(source)
        if candidate.get("odds_snapshot_age_hours") is not None or candidate.get("stale_line"):
            sources.append("odds_snapshot")
        return _dedupe(sources)

    @staticmethod
    def _suggested_market(candidate: dict[str, Any]) -> str:
        current_market = str(candidate.get("market", "") or "")
        suggested = str(candidate.get("recommended_market", "") or "")
        if suggested and suggested != current_market:
            return suggested
        return ""

    @staticmethod
    def _data_freshness(audit, sources_checked: tuple[str, ...]) -> str:
        states = [
            audit.odds_freshness,
            audit.lineup_freshness,
            audit.injury_news_freshness,
            audit.standings_freshness,
        ]
        if any(state == "stale" for state in states):
            return "stale"
        if any(state == "missing" for state in states):
            return "missing"
        if not sources_checked or all(state == "unknown" for state in states):
            return "insufficiently_verified"
        if all(state == "fresh" for state in states):
            return "verified_fresh"
        if any(state == "fresh" for state in states) or any(state == "monitor" for state in states):
            return "acceptable_freshness"
        return "insufficiently_verified"

    @staticmethod
    def _severe_rotation_risk(context: dict[str, Any], audit) -> bool:
        if not (context.get("cup_rotation_risk") or context.get("european_rotation_risk")):
            return False
        if context.get("final_day_volatility"):
            return True
        return audit.lineup_freshness == "missing"

    @staticmethod
    def _moderate_rotation_risk(context: dict[str, Any]) -> bool:
        return bool(context.get("cup_rotation_risk") or context.get("european_rotation_risk") or context.get("fixture_congestion_risk"))

    @staticmethod
    def _research_disagrees(context: dict[str, Any]) -> bool:
        serious_absences = (
            _as_int(context.get("home_priority_absences_count"))
            + _as_int(context.get("away_priority_absences_count"))
            + _as_int(context.get("home_spine_absences_count"))
            + _as_int(context.get("away_spine_absences_count"))
        )
        if serious_absences >= 2:
            return True
        if context.get("nothing_to_play_for"):
            return True
        return False

    def _build_veto_flags(self, audit, context: dict[str, Any], explicit_status: str) -> tuple[VetoFlag, ...]:
        flags: list[VetoFlag] = []
        if self.block_stale_data and audit.odds_freshness == "stale":
            flags.append(VetoFlag.STALE_ODDS)
        if self.block_stale_data and audit.injury_news_freshness == "stale":
            flags.append(VetoFlag.STALE_NEWS)
        if self.block_missing_lineups_near_kickoff and audit.lineup_freshness == "missing":
            flags.append(VetoFlag.MISSING_LINEUPS)
        if audit.match_status == "live":
            flags.append(VetoFlag.MATCH_ALREADY_LIVE)
        if audit.match_status == "finished" or explicit_status in {"postponed", "cancelled", "canceled", "suspended", "abandoned"}:
            flags.append(VetoFlag.FINISHED_MATCH)
        if not audit.fixture_verified:
            flags.append(VetoFlag.UNCLEAR_FIXTURE)
        rotation_checked = bool(context.get("rotation_checked"))
        motivation_checked = bool(context.get("motivation_checked") or context.get("playoff_motivation") or context.get("nothing_to_play_for") or context.get("rivalry_fixture"))
        if (context.get("cup_rotation_risk") or context.get("european_rotation_risk")) and not rotation_checked:
            flags.append(VetoFlag.HIGH_ROTATION_RISK)
        if context.get("final_day_volatility") and not motivation_checked:
            flags.append(VetoFlag.END_SEASON_CHAOS)
        return tuple(dict.fromkeys(flags))

    def _confidence_for(
        self,
        *,
        verdict: ResearchVerdict,
        evidence_status: str,
        concrete_info_score: int,
        source_count: int,
        source_quality_summary: str,
        fixture_verified: bool,
        odds_freshness_status: str,
        critical_missing: list[str],
        conflicting_evidence: list[str],
        model_derived_only: bool,
        wait_for_lineups_signal: bool,
        evidence: list[str],
        sport_specific_high_ready: bool,
    ) -> str:
        strong_source_profile = source_quality_summary == "strong"
        high_allowed = (
            verdict == ResearchVerdict.AGREE
            and evidence_status in {"COMPLETE", "ACCEPTABLE"}
            and concrete_info_score >= self.min_concrete_score_for_high
            and fixture_verified
            and odds_freshness_status in {"fresh", "acceptable"}
            and source_count >= self.min_sources_for_high
            and strong_source_profile
            and not critical_missing
            and not conflicting_evidence
            and not model_derived_only
            and not wait_for_lineups_signal
            and sport_specific_high_ready
        )
        if high_allowed:
            return "High"
        if evidence_status in {"COMPLETE", "ACCEPTABLE", "PARTIAL"} or evidence or source_count:
            return "Medium"
        return "Low"

    @staticmethod
    def _lead_hours(candidate: dict[str, Any]) -> float | None:
        commence = _parse_dt(candidate.get("commence") or candidate.get("commence_time"))
        if commence is None:
            return None
        return (commence - datetime.now(timezone.utc)).total_seconds() / 3600.0

    @staticmethod
    def _odds_age_minutes(candidate: dict[str, Any]) -> int | None:
        age_hours = _as_float(candidate.get("computed_odds_age_hours"))
        if age_hours is None:
            age_hours = _as_float(candidate.get("odds_snapshot_age_hours"))
        if age_hours is not None:
            return max(0, int(round(age_hours * 60)))
        fetched_at = _parse_dt(candidate.get("odds_fetched_at"))
        bookmaker_last_update = _parse_dt(candidate.get("bookmaker_last_update"))
        if fetched_at is not None and bookmaker_last_update is not None:
            delta = fetched_at - bookmaker_last_update
            return max(0, int(round(delta.total_seconds() / 60.0)))
        return None

    @staticmethod
    def _odds_freshness_status(candidate: dict[str, Any], audit) -> str:
        if audit.odds_freshness == "fresh":
            return "fresh"
        if audit.odds_freshness == "stale":
            return "stale"
        age_hours = _as_float(candidate.get("computed_odds_age_hours"))
        if age_hours is None:
            age_hours = _as_float(candidate.get("odds_snapshot_age_hours"))
        if age_hours is not None and age_hours <= 24:
            return "acceptable"
        return "unknown"

    @staticmethod
    def _market_availability_status(candidate: dict[str, Any]) -> str:
        market = str(candidate.get("market", "") or "").strip().lower()
        odds = candidate.get("odds")
        if not market:
            return "unknown"
        if odds in (None, "", 0):
            pricing_markers = (
                candidate.get("odds_snapshot_age_hours"),
                candidate.get("odds_fetched_at"),
                candidate.get("bookmaker_last_update"),
                candidate.get("market_implied_prob"),
                candidate.get("minimum_acceptable_odds"),
            )
            if any(marker not in (None, "", 0) for marker in pricing_markers):
                return "available"
            return "missing"
        return "available"

    @staticmethod
    def _lineup_status(audit, context: dict[str, Any]) -> str:
        soccer_status = str(
            context.get("soccer_lineup_status")
            or context.get("soccer_probable_lineup_status")
            or ""
        ).strip().lower()
        if audit.lineup_freshness == "fresh":
            return "confirmed"
        if audit.lineup_freshness == "monitor":
            return "monitor"
        if audit.lineup_freshness == "missing":
            return "missing_near_kickoff"
        if any(bool(context.get(key)) for key in ("home_lineup_confirmed", "away_lineup_confirmed")):
            return "confirmed"
        if any(_as_int(context.get(key)) > 0 for key in ("home_likely_starters_count", "away_likely_starters_count", "home_lineup_spine_count", "away_lineup_spine_count")):
            return "monitor"
        if soccer_status in {"projected", "checked_proxy"} or context.get("probable_lineups_checked") or context.get("lineup_checked"):
            return "monitor"
        if soccer_status == "confirmed":
            return "confirmed"
        return "unknown"

    @staticmethod
    def _injury_status(candidate: dict[str, Any], context: dict[str, Any], audit) -> str:
        soccer_status = str(context.get("soccer_injury_status") or "").strip().lower()
        if audit.injury_news_freshness == "fresh":
            return "checked_fresh"
        if audit.injury_news_freshness == "stale":
            return "stale"
        if soccer_status == "checked_fresh":
            return "checked_fresh"
        if soccer_status in {"checked_proxy", "checked"} or context.get("team_news_checked"):
            return "checked"
        if str(context.get("availability_lookup_status") or "").strip() == "provider_failed":
            return "provider_failed"
        if str(context.get("availability_lookup_status") or "").strip() == "not_found":
            return "not_found"
        market = str(candidate.get("market", "") or "").lower()
        source = str(context.get("availability_source") or candidate.get("availability_source") or "").strip()
        if source:
            return "checked"
        if market in _LINEUP_SENSITIVE_MARKETS:
            return "not_checked"
        return "unknown"

    @staticmethod
    def _special_fixture_context(context: dict[str, Any]) -> bool:
        return bool(
            context.get("is_playoff")
            or context.get("playoff_motivation")
            or context.get("cup_rotation_risk")
            or context.get("european_rotation_risk")
            or context.get("final_day_volatility")
            or context.get("fixture_congestion_risk")
            or context.get("nothing_to_play_for")
            or context.get("rivalry_fixture")
        )

    @staticmethod
    def _motivation_status(sport: str, context: dict[str, Any], special_fixture: bool) -> str:
        soccer_status = str(context.get("soccer_motivation_status") or "").strip().lower()
        if soccer_status == "checked":
            return "checked"
        if soccer_status == "checked_proxy":
            return "checked_proxy"
        if context.get("motivation_checked"):
            return "checked"
        if context.get("is_playoff") or context.get("playoff_motivation") or context.get("final_day_volatility") or context.get("nothing_to_play_for") or context.get("rivalry_fixture"):
            return "checked"
        if sport in {"basketball", "mlb", "nhl"} and context.get("fixture_congestion_risk"):
            return "checked"
        if special_fixture:
            return "not_checked"
        return "not_required"

    @staticmethod
    def _rotation_status(sport: str, context: dict[str, Any], audit, special_fixture: bool) -> str:
        soccer_status = str(context.get("soccer_rotation_status") or "").strip().lower()
        if context.get("cup_rotation_risk") or context.get("european_rotation_risk") or context.get("fixture_congestion_risk"):
            if soccer_status == "checked":
                return "checked"
            if soccer_status == "checked_proxy":
                return "checked_proxy"
            if context.get("rotation_checked"):
                return "checked"
            if audit.lineup_freshness == "fresh" or audit.injury_news_freshness == "fresh":
                return "checked"
            return "not_checked"
        if sport in {"basketball", "mlb", "nhl"} and context.get("is_playoff"):
            return "checked" if audit.injury_news_freshness == "fresh" or audit.lineup_freshness in {"fresh", "monitor"} else "not_checked"
        if special_fixture and context.get("final_day_volatility"):
            return "checked" if audit.lineup_freshness in {"fresh", "monitor"} else "not_checked"
        return "not_required"

    @staticmethod
    def _source_quality_summary(sources_checked: tuple[str, ...]) -> str:
        concrete_sources = [src for src in sources_checked if str(src).lower() in _CONCRETE_SOURCE_NAMES]
        if len(concrete_sources) >= 2:
            return "strong"
        if len(concrete_sources) >= 1:
            return "mixed"
        if any(str(src).lower() in _WEAK_SOURCE_NAMES for src in sources_checked):
            return "weak"
        return "weak"

    @staticmethod
    def _model_derived_only(candidate: dict[str, Any], sources_checked: tuple[str, ...]) -> bool:
        if any(str(src).lower() in _CONCRETE_SOURCE_NAMES for src in sources_checked):
            return False
        evidence_text = " ".join(
            [
                *(str(item or "") for item in (candidate.get("scraped_context_highlights") or [])),
                *(str(item.get("summary") or item.get("name") or "") for item in (candidate.get("prediction_factors") or []) if isinstance(item, dict)),
                *(str(item.get("summary") or item.get("name") or "") for item in (candidate.get("context_adjustments") or []) if isinstance(item, dict)),
            ]
        ).lower()
        return any(hint in evidence_text for hint in _MODEL_DERIVED_HINTS) or not sources_checked

    def _missing_evidence(
        self,
        *,
        candidate: dict[str, Any],
        context: dict[str, Any],
        audit,
        source_count: int,
        source_quality_summary: str,
        odds_freshness_status: str,
        market_availability_status: str,
        lineup_status: str,
        injury_status: str,
        motivation_status: str,
        rotation_status: str,
        special_fixture: bool,
        lead_hours: float | None,
        sport: str,
    ) -> tuple[list[str], list[str]]:
        missing: list[str] = []
        critical: list[str] = []
        if not audit.fixture_verified:
            missing.append("fixture verification failed")
            critical.append("fixture verification failed")
        if source_count < self.min_sources_for_high:
            missing.append(f"source count below minimum ({source_count} < {self.min_sources_for_high})")
            critical.append("source count below minimum")
        if source_quality_summary == "weak":
            missing.append("source quality is weak")
            critical.append("source quality is weak")
        if odds_freshness_status not in {"fresh", "acceptable"}:
            missing.append("odds freshness is not verified")
            critical.append("odds freshness is not verified")
        if market_availability_status != "available":
            missing.append("market availability is incomplete")
            critical.append("market availability incomplete")
        if lineup_status == "missing_near_kickoff":
            missing.append("lineups are missing near kickoff")
            critical.append("lineups are missing near kickoff")
        market = str(candidate.get("market", "") or "").lower()
        if market in _LINEUP_SENSITIVE_MARKETS and injury_status not in {"checked_fresh", "unknown"}:
            missing.append("injury/team news was not checked for a lineup-sensitive market")
            critical.append("injury/team news not checked")
        if sport == "basketball" and lead_hours is not None and lead_hours <= 2 and injury_status != "checked_fresh":
            missing.append("final injury/inactive report was not checked close to tip-off")
            critical.append("final injury/inactive report not checked")
        if sport == "mlb":
            profile = self._mlb_research_profile(
                candidate=candidate,
                context=context,
                lineup_status=lineup_status,
            )
            if not profile["market_available"]:
                missing.append("market availability is incomplete for this MLB candidate")
                critical.append("market availability incomplete")
            if lead_hours is not None and lead_hours <= 6 and not profile["starter_confirmation_checked"]:
                missing.append("probable starters are not fully confirmed for the current decision window")
                critical.append("probable starters not fully confirmed")
            if profile["pitcher_change_detected"]:
                missing.append("probable starter changed during the latest MLB evidence refresh")
                critical.append("probable starter changed")
            if not profile["pitcher_quality_checked"]:
                missing.append("starting-pitcher quality was not checked for the MLB matchup")
            if not profile["lineup_context_checked"]:
                missing.append("lineup confirmation or projection was not checked for the MLB fixture")
            if profile["totals_or_spread_market"] and not profile["bullpen_checked"]:
                missing.append("bullpen workload was not checked for the MLB totals/run-line market")
            if profile["weather_material"] and not profile["weather_checked"]:
                missing.append("weather context was not checked for the MLB weather-sensitive market")
            if profile["totals_or_spread_market"] and not profile["park_checked"]:
                missing.append("park-factor context was not checked for the MLB totals/run-line market")
            if not profile["travel_checked"]:
                missing.append("travel/rest context was not checked for the MLB series spot")
            if profile["underdog_minus_one_half"] and not profile["market_fit_ok"]:
                missing.append("aggressive underdog -1.5 MLB run-line lacks strong support")
                critical.append("aggressive underdog run-line lacks support")
        if sport == "basketball":
            profile = self._basketball_research_profile(
                candidate=candidate,
                context=context,
                lineup_status=lineup_status,
                injury_status=injury_status,
            )
            if not profile["market_available"]:
                missing.append("market availability is incomplete for this NBA candidate")
                critical.append("market availability incomplete")
            if not profile["lineup_projection_checked"]:
                missing.append("starting or projected lineup context was not checked for this NBA fixture")
            if profile["star_player_uncertain"]:
                missing.append("star-player injury status is still uncertain")
                critical.append("star-player injury status uncertain")
            if profile["back_to_back"] and not profile["rest_checked"]:
                missing.append("rest/back-to-back context was not checked for the NBA schedule spot")
            if not profile["travel_checked"]:
                missing.append("travel context was not checked for the NBA fixture")
            if not profile["ratings_checked"]:
                missing.append("offensive/defensive rating context was not checked for the NBA fixture")
            if profile["totals_like_market"] and not profile["pace_checked"]:
                missing.append("pace context was not checked for the NBA totals environment")
            if profile["usage_context_required"] and not profile["usage_checked"]:
                missing.append("usage redistribution was not checked for key-player absences")
                if profile["player_status_sensitive_market"]:
                    critical.append("usage redistribution not checked for player-status-sensitive market")
        if sport in _CONSERVATIVE_RESEARCH_SPORTS:
            profile = self._tennis_research_profile(
                candidate=candidate,
                context=context,
                injury_status=injury_status,
            )
            if not profile["market_available"]:
                missing.append("market availability is incomplete for this tennis candidate")
                critical.append("market availability incomplete")
            if not profile["surface_checked"]:
                missing.append("surface context was not checked for the tennis matchup")
            if not profile["recent_form_checked"]:
                missing.append("recent form was not checked for the tennis matchup")
            if profile["injury_concern_present"] and not profile["injury_verified"]:
                missing.append("injury/retirement concern could not be verified for the tennis matchup")
                critical.append("injury/retirement concern unverified")
            if profile["fatigue_flag"] and not profile["fatigue_checked"]:
                missing.append("fatigue from recent matches was not checked for the tennis matchup")
            if not profile["tournament_context_checked"]:
                missing.append("tournament round/context was not checked for the tennis matchup")
            if profile["travel_relevant"] and not profile["travel_checked"]:
                missing.append("travel/time-zone context was not checked for the tennis matchup")
            if not profile["h2h_style_checked"]:
                missing.append("head-to-head style context was not checked for the tennis matchup")
            if not profile["ranking_checked"]:
                missing.append("ranking/Elo context was not checked for the tennis matchup")
            if profile["totals_or_spread_market"] and not profile["serve_return_checked"]:
                missing.append("serve/return matchup context was not checked for the tennis market")
            if profile["live_pre_match_conflict"]:
                missing.append("match appears live while only pre-match odds context is attached")
                critical.append("live match with pre-match odds context")
        if sport == "nhl" and lead_hours is not None and lead_hours <= 6 and lineup_status != "confirmed":
            missing.append("probable goalies are not fully confirmed for the current decision window")
            critical.append("probable goalies not fully confirmed")
        if sport == "nhl":
            profile = self._nhl_research_profile(
                candidate=candidate,
                context=context,
                lineup_status=lineup_status,
            )
            if not profile["market_available"]:
                missing.append("market availability is incomplete for this NHL candidate")
                critical.append("market availability incomplete")
            if not profile["goalie_projection_checked"]:
                missing.append("starting goalie projection or confirmation was not checked")
                critical.append("starting goalie projection not checked")
            if profile["totals_like_market"] and not profile["goalie_context_complete"]:
                missing.append("goalie context is incomplete for an NHL totals/team-total market")
                critical.append("goalie context incomplete for totals market")
            if profile["back_to_back"] and not profile["rest_checked"]:
                missing.append("rest/back-to-back context was not checked for the NHL schedule spot")
            if not profile["travel_checked"]:
                missing.append("travel context was not checked for the NHL fixture")
            if not profile["form_checked"]:
                missing.append("team defensive/offensive form was not checked for the NHL fixture")
            if not profile["special_teams_checked"]:
                missing.append("special-teams context was not checked for the NHL fixture")
            if not profile["shots_xg_checked"]:
                missing.append("shots/xG context was not checked for the NHL fixture")
            if not profile["splits_checked"]:
                missing.append("home/away split context was not checked for the NHL fixture")
        motivation_required = bool(
            context.get("is_playoff")
            or context.get("playoff_motivation")
            or context.get("final_day_volatility")
            or context.get("nothing_to_play_for")
            or context.get("rivalry_fixture")
        )
        if sport in {"basketball", "mlb", "nhl"} and context.get("fixture_congestion_risk"):
            motivation_required = True
        rotation_required = bool(
            context.get("cup_rotation_risk")
            or context.get("european_rotation_risk")
            or context.get("fixture_congestion_risk")
            or context.get("final_day_volatility")
        )
        if motivation_required and motivation_status != "checked":
            missing.append("motivation context was not checked for a high-context fixture")
            critical.append("motivation context not checked")
        if rotation_required and rotation_status != "checked":
            missing.append("rotation context was not checked for a high-context fixture")
            critical.append("rotation context not checked")
        if lead_hours is not None and lead_hours <= 2 and lineup_status != "confirmed":
            missing.append("near-kickoff lineup confirmation is incomplete")
        return _dedupe(missing), _dedupe(critical)

    @staticmethod
    def _conflicting_evidence(candidate: dict[str, Any], context: dict[str, Any], explicit_status: str) -> list[str]:
        conflicts: list[str] = []
        if explicit_status in {"postponed", "cancelled", "canceled", "suspended", "abandoned"}:
            conflicts.append("fixture status conflicts with a pre-match betting recommendation")
        referee_decision = str(candidate.get("context_referee_decision", "") or "").strip().upper()
        referee_reason = str(candidate.get("context_referee_reason", "") or "").strip()
        if referee_decision in {"VETO", "REVIEW"}:
            conflicts.append(referee_reason or "legacy context referee raised conflicting evidence")
        if context.get("playoff_motivation") and context.get("nothing_to_play_for"):
            conflicts.append("motivation signals are internally conflicting")
        return _dedupe(conflicts)

    @staticmethod
    def _concrete_info_score(
        *,
        audit,
        source_count: int,
        source_quality_summary: str,
        odds_freshness_status: str,
        lineup_status: str,
        injury_status: str,
        motivation_status: str,
        rotation_status: str,
        model_derived_only: bool,
        conflicting_evidence: list[str],
        critical_missing: list[str],
    ) -> int:
        score = 0
        if audit.fixture_verified:
            score += 15
        if odds_freshness_status == "fresh":
            score += 15
        elif odds_freshness_status == "acceptable":
            score += 10
        score += min(source_count, 3) * 8
        if source_quality_summary == "strong":
            score += 16
        elif source_quality_summary == "mixed":
            score += 8
        if lineup_status == "confirmed":
            score += 12
        elif lineup_status == "monitor":
            score += 6
        if injury_status == "checked_fresh":
            score += 12
        if motivation_status in {"checked", "not_required"}:
            score += 8
        if rotation_status in {"checked", "not_required"}:
            score += 8
        if model_derived_only:
            score -= 20
        score -= len(conflicting_evidence) * 12
        score -= len(critical_missing) * 10
        return max(0, min(100, score))

    @staticmethod
    def _evidence_status(
        *,
        sport: str,
        audit,
        source_count: int,
        source_quality_summary: str,
        model_derived_only: bool,
        missing_evidence: list[str],
        critical_missing: list[str],
        conflicting_evidence: list[str],
        concrete_info_score: int,
    ) -> str:
        if conflicting_evidence:
            return "CONFLICTING"
        if not audit.fixture_verified or source_count == 0 or source_quality_summary == "weak":
            return "INSUFFICIENT"
        if critical_missing or concrete_info_score < 45:
            return "PARTIAL"
        if not missing_evidence and concrete_info_score >= 85 and source_quality_summary == "strong":
            return "ACCEPTABLE" if sport in _CONSERVATIVE_RESEARCH_SPORTS else "COMPLETE"
        return "ACCEPTABLE"

    def _sport_specific_high_ready(
        self,
        *,
        sport: str,
        candidate: dict[str, Any],
        context: dict[str, Any],
        lineup_status: str,
        injury_status: str,
        motivation_status: str,
    ) -> bool:
        if sport in _CONSERVATIVE_RESEARCH_SPORTS:
            return False
        if sport == "basketball":
            profile = self._basketball_research_profile(
                candidate=candidate,
                context=context,
                lineup_status=lineup_status,
                injury_status=injury_status,
            )
            return (
                injury_status == "checked_fresh"
                and profile["market_available"]
                and profile["lineup_projection_checked"]
                and not profile["star_player_uncertain"]
                and profile["rest_checked"]
                and profile["travel_checked"]
                and profile["ratings_checked"]
                and (profile["pace_checked"] if profile["totals_like_market"] else True)
                and (profile["usage_checked"] if profile["usage_context_required"] else True)
                and (motivation_status == "checked" if profile["playoff_context_required"] else motivation_status in {"checked", "not_required"})
            )
        if sport == "mlb":
            profile = self._mlb_research_profile(
                candidate=candidate,
                context=context,
                lineup_status=lineup_status,
            )
            return (
                lineup_status == "confirmed"
                and profile["market_available"]
                and profile["starter_confirmation_checked"]
                and profile["pitcher_quality_checked"]
                and (profile["bullpen_checked"] if profile["totals_or_spread_market"] else True)
                and (profile["weather_checked"] if profile["weather_material"] else True)
                and profile["travel_checked"]
                and profile["market_fit_ok"]
            )
        if sport == "nhl":
            profile = self._nhl_research_profile(candidate=candidate, context=context, lineup_status=lineup_status)
            return (
                lineup_status == "confirmed"
                and injury_status == "checked_fresh"
                and profile["market_available"]
                and profile["goalie_projection_checked"]
                and profile["rest_checked"]
                and profile["travel_checked"]
                and profile["form_checked"]
                and profile["special_teams_checked"]
                and profile["shots_xg_checked"]
                and profile["splits_checked"]
                and (motivation_status == "checked" if profile["playoff_context_required"] else motivation_status in {"checked", "not_required"})
            )
        return True

    def _mlb_research_profile(
        self,
        *,
        candidate: dict[str, Any],
        context: dict[str, Any],
        lineup_status: str,
    ) -> dict[str, bool]:
        factor_names = self._signal_names(candidate, "prediction_factors")
        adjustment_names = self._signal_names(candidate, "context_adjustments")
        market = str(candidate.get("market", "") or "").strip().lower()
        market_available = bool(market and self._market_availability_status(candidate) == "available")
        starter_confirmation_checked = bool(
            lineup_status in {"confirmed", "monitor"}
            or (
                context.get("home_starter_confirmed") is not None
                and context.get("away_starter_confirmed") is not None
            )
        )
        pitcher_names_checked = bool(
            str(context.get("home_starter_name") or "").strip()
            and str(context.get("away_starter_name") or "").strip()
        )
        pitcher_change_detected = bool(
            context.get("home_pitcher_changed")
            or context.get("away_pitcher_changed")
            or context.get("pitcher_change_detected")
        )
        handedness_checked = bool(
            str(context.get("home_starter_hand") or "").strip()
            and str(context.get("away_starter_hand") or "").strip()
        )
        pitcher_quality_checked = bool(
            {"sp_era_diff", "sp_whip_diff", "sp_k9_diff"} & factor_names
            or {"starter_quality", "pitcher_command", "starter_form_synergy"} & adjustment_names
            or any(
                key in context
                for key in (
                    "sp_era_diff",
                    "sp_whip_diff",
                    "sp_k9_diff",
                    "home_starter_era",
                    "away_starter_era",
                    "home_starter_whip",
                    "away_starter_whip",
                    "home_starter_fip",
                    "away_starter_fip",
                )
            )
        )
        lineup_context_checked = bool(
            lineup_status in {"confirmed", "monitor"}
            or str(context.get("lineup_source") or "").strip()
            or _as_int(context.get("home_likely_starters_count")) > 0
            or _as_int(context.get("away_likely_starters_count")) > 0
        )
        bullpen_checked = bool(
            context.get("bullpen_workload_checked")
            or any(key in context for key in ("home_games_L3D", "away_games_L3D", "bullpen_fatigue_risk", "bullpen_quality_proxy"))
            or "bullpen_workload" in adjustment_names
        )
        weather_checked = bool(
            str(context.get("outdoor_weather_source") or "").strip()
            or str(context.get("roof_status") or "").strip()
        )
        park_checked = bool(
            str(context.get("park_factor_source") or "").strip()
            or "park_factor_proxy" in context
        )
        travel_checked = bool(
            "travel_fatigue" in adjustment_names
            or "rest_advantage" in adjustment_names
            or any(key in context for key in ("away_travel_km", "away_travel_tz_shift", "home_rest_days", "away_rest_days"))
        )
        totals_or_spread_market = market in {"totals", "team_total", "spreads"}
        weather_material = market in {"totals", "spreads"}
        line = _as_float(candidate.get("line"))
        if line is None:
            line = _as_float(candidate.get("market_line"))
        if line is None:
            import re as _re
            match = _re.search(r"([+-]\d+(?:\.\d+)?)", str(candidate.get("team") or ""))
            if match:
                line = _as_float(match.group(1))
        underdog_minus_one_half = bool(market == "spreads" and line is not None and line < 0 and _as_float(candidate.get("odds")) is not None and float(candidate.get("odds")) >= 2.0)
        strong_spread_support = bool(
            pitcher_quality_checked
            and bullpen_checked
            and lineup_context_checked
            and not pitcher_change_detected
        )
        market_fit_ok = not underdog_minus_one_half or strong_spread_support
        return {
            "market_available": market_available,
            "starter_confirmation_checked": starter_confirmation_checked,
            "pitcher_names_checked": pitcher_names_checked,
            "pitcher_change_detected": pitcher_change_detected,
            "handedness_checked": handedness_checked,
            "pitcher_quality_checked": pitcher_quality_checked,
            "lineup_context_checked": lineup_context_checked,
            "bullpen_checked": bullpen_checked,
            "weather_checked": weather_checked,
            "park_checked": park_checked,
            "travel_checked": travel_checked,
            "totals_or_spread_market": totals_or_spread_market,
            "weather_material": weather_material,
            "underdog_minus_one_half": underdog_minus_one_half,
            "market_fit_ok": market_fit_ok,
        }

    def _tennis_research_profile(
        self,
        *,
        candidate: dict[str, Any],
        context: dict[str, Any],
        injury_status: str,
    ) -> dict[str, bool]:
        factor_names = self._signal_names(candidate, "prediction_factors")
        adjustment_names = self._signal_names(candidate, "context_adjustments")
        market = str(candidate.get("market", "") or "").strip().lower()
        highlights = " ".join(str(item or "") for item in (candidate.get("scraped_context_highlights") or [])).lower()
        market_available = bool(market and candidate.get("odds") not in (None, "", 0))
        surface_checked = bool(
            "surface_win_diff" in factor_names
            or str(candidate.get("surface") or context.get("surface") or "").strip()
        )
        recent_form_checked = bool("form_diff" in factor_names)
        injury_concern_present = bool(
            context.get("injury_concern")
            or context.get("retirement_concern")
            or context.get("injury_status_uncertain")
            or context.get("retirement_risk")
            or "injury concern" in highlights
            or "retirement concern" in highlights
        )
        injury_verified = bool(
            context.get("injury_concern_checked")
            or context.get("retirement_concern_checked")
            or (injury_status == "checked_fresh" and not injury_concern_present)
        )
        fatigue_flag = bool(
            context.get("recent_long_match")
            or context.get("fatigue_risk")
            or any(key in context for key in ("p1_match_load", "p2_match_load", "player1_match_load", "player2_match_load"))
        )
        fatigue_checked = bool(
            context.get("fatigue_checked")
            or "travel_fatigue" in adjustment_names
            or any(key in context for key in ("p1_match_load", "p2_match_load", "player1_match_load", "player2_match_load"))
        )
        tournament_context_checked = bool(
            str(context.get("round") or candidate.get("round") or "").strip()
            or str(context.get("tournament") or candidate.get("tournament") or candidate.get("league") or "").strip()
            or context.get("tournament_context_checked")
        )
        travel_relevant = bool(context.get("travel_relevant") or context.get("travel_required") or context.get("timezone_shift"))
        travel_checked = bool("travel_fatigue" in adjustment_names or context.get("travel_checked") or not travel_relevant)
        h2h_style_checked = bool("h2h_p1_win_rate" in factor_names or "h2h_tactical_history" in adjustment_names)
        ranking_checked = bool(
            {"rank_log_ratio", "rank_pts_log_ratio", "elo_diff"} & factor_names
            or any(key in context for key in ("player1_rank", "player2_rank", "player1_elo", "player2_elo"))
        )
        serve_return_checked = bool(
            {"serve_diff", "return_pressure_diff", "serve_balance_diff"} & factor_names
            or context.get("serve_return_checked")
        )
        totals_or_spread_market = market in {"totals", "spreads"}
        live_pre_match_conflict = bool(candidate.get("odds")) and str(candidate.get("status") or "").strip().lower() == "live"
        return {
            "market_available": market_available,
            "surface_checked": surface_checked,
            "recent_form_checked": recent_form_checked,
            "injury_concern_present": injury_concern_present,
            "injury_verified": injury_verified,
            "fatigue_flag": fatigue_flag,
            "fatigue_checked": fatigue_checked,
            "tournament_context_checked": tournament_context_checked,
            "travel_relevant": travel_relevant,
            "travel_checked": travel_checked,
            "h2h_style_checked": h2h_style_checked,
            "ranking_checked": ranking_checked,
            "serve_return_checked": serve_return_checked,
            "totals_or_spread_market": totals_or_spread_market,
            "live_pre_match_conflict": live_pre_match_conflict,
        }

    def _basketball_research_profile(
        self,
        *,
        candidate: dict[str, Any],
        context: dict[str, Any],
        lineup_status: str,
        injury_status: str,
    ) -> dict[str, bool]:
        adjustment_names = self._signal_names(candidate, "context_adjustments")
        factor_names = self._signal_names(candidate, "prediction_factors")
        market = str(candidate.get("market", "") or "").strip().lower()
        market_available = bool(market and candidate.get("odds") not in (None, "", 0))
        lineup_projection_checked = bool(
            lineup_status in {"confirmed", "monitor"}
            or str(context.get("lineup_source") or "").strip()
        )
        back_to_back = bool("back_to_back" in adjustment_names or context.get("fixture_congestion_risk"))
        rest_checked = bool("back_to_back" in adjustment_names or "rest_advantage" in adjustment_names or "rest_advantage" in context)
        travel_checked = bool("travel_fatigue" in adjustment_names)
        pace_checked = bool("pace_control" in adjustment_names)
        ratings_checked = bool(
            {"closing_execution", "venue_comfort"} & adjustment_names
            or any(
                token in name
                for name in factor_names
                for token in ("ortg", "drtg", "scoring_margin", "opp_ppg", "ppg", "net_rating", "def_rating", "off_rating")
            )
        )
        home_priority = _as_int(context.get("home_priority_absences_count"))
        away_priority = _as_int(context.get("away_priority_absences_count"))
        home_questionable = _as_int(context.get("home_questionable_count"))
        away_questionable = _as_int(context.get("away_questionable_count"))
        star_player_uncertain = bool("lineup_uncertainty" in adjustment_names or home_questionable > 0 or away_questionable > 0)
        usage_context_required = bool(home_priority > 0 or away_priority > 0 or home_questionable > 0 or away_questionable > 0)
        usage_checked = bool(
            {"injury_report_edge", "rotation_quality_edge"} & adjustment_names
            or any(token in name for name in factor_names for token in ("usage", "on_off", "assist_share"))
            or not usage_context_required
        )
        playoff_context_required = bool(context.get("is_playoff") or context.get("playoff_motivation") or context.get("fixture_congestion_risk"))
        totals_like_market = market in {"totals", "team_total", "player_total"}
        player_status_sensitive_market = market in {"team_total", "player_total"}
        player_status_context_complete = bool(
            injury_status == "checked_fresh"
            and not star_player_uncertain
            and lineup_projection_checked
            and (usage_checked if usage_context_required else True)
        )
        return {
            "market_available": market_available,
            "lineup_projection_checked": lineup_projection_checked,
            "back_to_back": back_to_back,
            "rest_checked": rest_checked,
            "travel_checked": travel_checked,
            "pace_checked": pace_checked,
            "ratings_checked": ratings_checked,
            "star_player_uncertain": star_player_uncertain,
            "usage_context_required": usage_context_required,
            "usage_checked": usage_checked,
            "playoff_context_required": playoff_context_required,
            "totals_like_market": totals_like_market,
            "player_status_sensitive_market": player_status_sensitive_market,
            "player_status_context_complete": player_status_context_complete,
        }

    @staticmethod
    def _signal_names(candidate: dict[str, Any], key: str) -> set[str]:
        names: set[str] = set()
        for item in candidate.get(key) or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip().lower()
            if name:
                names.add(name)
        return names

    def _nhl_research_profile(
        self,
        *,
        candidate: dict[str, Any],
        context: dict[str, Any],
        lineup_status: str,
    ) -> dict[str, bool]:
        adjustment_names = self._signal_names(candidate, "context_adjustments")
        factor_names = self._signal_names(candidate, "prediction_factors")
        market = str(candidate.get("market", "") or "").strip().lower()
        market_available = bool(market and candidate.get("odds") not in (None, "", 0))
        goalie_projection_checked = bool(
            lineup_status in {"confirmed", "monitor"}
            or str(context.get("home_goalie_name") or "").strip()
            or str(context.get("away_goalie_name") or "").strip()
            or {"goalie_uncertainty", "goalie_stability", "goalie_quality", "probable_goalie_named"} & adjustment_names
        )
        back_to_back = bool("back_to_back" in adjustment_names or context.get("fixture_congestion_risk"))
        rest_checked = bool("back_to_back" in adjustment_names or "rest_advantage" in adjustment_names or "rest_advantage" in context)
        travel_checked = bool("travel_fatigue" in adjustment_names)
        form_checked = bool(
            {"home_goal_diff_10", "away_goal_diff_10", "home_xg_diff_10", "away_xg_diff_10"} & factor_names
            or {"xg_structure"} & adjustment_names
        )
        special_teams_checked = bool("special_teams_edge" in adjustment_names)
        shots_xg_checked = bool("xg_structure" in adjustment_names or {"home_xg_diff_10", "away_xg_diff_10"} & factor_names)
        splits_checked = bool("system_stability" in adjustment_names)
        playoff_context_required = bool(context.get("is_playoff") or context.get("playoff_motivation"))
        totals_like_market = market in {"totals", "team_total"}
        goalie_context_complete = lineup_status == "confirmed" if totals_like_market else goalie_projection_checked
        return {
            "market_available": market_available,
            "goalie_projection_checked": goalie_projection_checked,
            "goalie_context_complete": goalie_context_complete,
            "back_to_back": back_to_back,
            "rest_checked": rest_checked,
            "travel_checked": travel_checked,
            "form_checked": form_checked,
            "special_teams_checked": special_teams_checked,
            "shots_xg_checked": shots_xg_checked,
            "splits_checked": splits_checked,
            "playoff_context_required": playoff_context_required,
            "totals_like_market": totals_like_market,
        }

    def _nhl_goalie_market_hold(
        self,
        *,
        sport: str,
        candidate: dict[str, Any],
        lineup_status: str,
    ) -> bool:
        if sport != "nhl":
            return False
        market = str(candidate.get("market", "") or "").strip().lower()
        if market not in {"totals", "team_total"}:
            return False
        return lineup_status != "confirmed"

    def _basketball_market_hold(
        self,
        *,
        sport: str,
        candidate: dict[str, Any],
        context: dict[str, Any],
        lineup_status: str,
        injury_status: str,
    ) -> bool:
        if sport != "basketball":
            return False
        profile = self._basketball_research_profile(
            candidate=candidate,
            context=context,
            lineup_status=lineup_status,
            injury_status=injury_status,
        )
        if profile["star_player_uncertain"]:
            return True
        if not profile["player_status_sensitive_market"]:
            return False
        return not profile["player_status_context_complete"]

    def _tennis_market_hold(
        self,
        *,
        sport: str,
        candidate: dict[str, Any],
        context: dict[str, Any],
        injury_status: str,
        explicit_status: str,
    ) -> bool:
        if sport not in _CONSERVATIVE_RESEARCH_SPORTS:
            return False
        profile = self._tennis_research_profile(
            candidate=candidate,
            context=context,
            injury_status=injury_status,
        )
        if profile["injury_concern_present"] and not profile["injury_verified"]:
            return True
        if explicit_status == "live" and profile["live_pre_match_conflict"]:
            return True
        return False

    @staticmethod
    def _evidence_notes(
        *,
        audit,
        sources_checked: tuple[str, ...],
        source_quality_summary: str,
        odds_age_minutes: int | None,
        odds_freshness_status: str,
        lineup_status: str,
        injury_status: str,
        motivation_status: str,
        rotation_status: str,
        model_derived_only: bool,
    ) -> list[str]:
        notes = [
            f"source quality assessed as {source_quality_summary}",
            f"fixture verified={audit.fixture_verified}",
            f"odds freshness={odds_freshness_status}",
            f"lineup status={lineup_status}",
            f"injury status={injury_status}",
            f"motivation status={motivation_status}",
            f"rotation status={rotation_status}",
        ]
        if odds_age_minutes is not None:
            notes.append(f"odds age is approximately {odds_age_minutes} minutes")
        if sources_checked:
            notes.append("sources checked: " + ", ".join(sources_checked))
        if model_derived_only:
            notes.append("evidence leans on model/context features more than concrete external research")
        return _dedupe(notes)
