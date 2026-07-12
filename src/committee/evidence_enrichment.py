from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from math import log
from pathlib import Path
from copy import deepcopy
from typing import Any
import re
from urllib.parse import urlparse

import pandas as pd

from config import settings
from src.analysis.news_context import collect_matchup_news_context
from src.data.api_football_enricher import APIFootballEnricher
from src.data.soccer_fetcher import SoccerFetcher
from src.features.feature_store import TeamResolver, build_entity_alias_map, resolve_canonical_name
from src.markets.availability import build_availability_context
from src.markets.environment import build_environment_context
from src.utils.sport_registry import SOCCER_ODDS_TO_COMPETITION, resolve_soccer_key

from .contracts import ModelMindDecision, ResearchMindDecision

_DEFAULT_SOURCE_PRIORITY = (
    "official_league_or_team",
    "trusted_sports_data_api",
    "bookmaker_or_odds_feed",
    "reputable_sports_media",
    "established_stats_or_preview_site",
    "aggregator",
    "community_unverified",
)

_SOURCE_PRIORITY_RANK = {
    "official": 1,
    "team_official": 1,
    "league_official": 1,
    "official_league_or_team": 1,
    "api_football": 2,
    "api_sports": 2,
    "api_sports_basketball": 2,
    "football_data": 2,
    "mlb_api": 2,
    "mlb_stats_api": 2,
    "nhl_api": 2,
    "balldontlie": 2,
    "trusted_sports_data_api": 2,
    "odds_snapshot": 3,
    "bookmaker": 3,
    "bookmaker_or_odds_feed": 3,
    "espn": 4,
    "rotowire": 4,
    "reputable_sports_media": 4,
    "sportmonks": 5,
    "sofascore": 5,
    "established_stats_or_preview_site": 5,
    "flashscore": 6,
    "aggregator": 6,
    "reddit": 7,
    "community": 7,
    "community_unverified": 7,
}
_TENNIS_CACHE_SOURCE = "tennis_feature_cache"
_MLB_CACHE_SOURCE = "mlb_feature_cache"
_SOCCER_CACHE_SOURCE = "soccer_feature_cache"
_DEFAULT_TENNIS_DETAILS = {
    "surface_status": "",
    "ranking_elo_status": "",
    "injury_retirement_status": "",
    "fatigue_status": "",
    "tournament_context_status": "",
    "style_matchup_status": "",
}
_DEFAULT_MLB_DETAILS = {
    "fixture_status": "",
    "probable_pitcher_status": "",
    "pitcher_change_status": "",
    "home_pitcher": "",
    "away_pitcher": "",
    "pitcher_handedness_status": "",
    "lineup_status": "",
    "injury_status": "",
    "bullpen_status": "",
    "weather_status": "",
    "park_factor_status": "",
    "travel_rest_status": "",
    "market_fit_status": "",
}
_DEFAULT_SOCCER_DETAILS = {
    "fixture_status": "",
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
    "lineup_status": "",
    "probable_lineup_status": "",
    "injury_status": "",
    "suspension_status": "",
    "goalkeeper_status": "",
    "motivation_status": "",
    "rotation_status": "",
    "fixture_congestion_status": "",
    "home_away_form_status": "",
    "xg_context_status": "",
    "market_fit_status": "",
}
_SOCCER_FORM_WINDOWS = ("WWWWW", "WWWWD", "WWDWW", "WDWWW")
_SURFACE_HINTS = {
    "Clay": (
        "clay",
        "roland_garros",
        "barcelona",
        "monte_carlo",
        "rome",
        "madrid",
        "hamburg",
        "gstaad",
        "bastad",
        "umag",
        "kitzbuhel",
        "bucharest",
    ),
    "Grass": (
        "wimbledon",
        "queens",
        "halle",
        "grass",
        "eastbourne",
        "hertogenbosch",
    ),
}
_NEGATIVE_INJURY_PATTERNS = (
    "withdrew",
    "withdrawal",
    "retired",
    "retirement concern",
    "medical timeout",
    "fitness doubt",
    "not fully fit",
    "injury scare",
)
_SOCCER_NEWS_SOURCE_LABELS = {
    "espn.com": "espn",
    "90min.com": "90min.com",
    "football365.com": "football365.com",
    "fotmob.com": "fotmob.com",
    "onefootball.com": "onefootball.com",
    "rotowire.com": "rotowire",
    "sportsmole.co.uk": "sportsmole.co.uk",
    "thestatszone.com": "thestatszone.com",
    "transfermarkt.com": "transfermarkt",
    "whoscored.com": "whoscored",
    "covers.com": "covers.com",
    "flashscore.com": "flashscore",
    "reddit.com": "reddit",
}
_SOCCER_RELIABLE_NEWS_SOURCES = {
    "90min.com",
    "espn",
    "football365.com",
    "fotmob.com",
    "onefootball.com",
    "rotowire",
    "sportsmole.co.uk",
    "thestatszone.com",
    "transfermarkt",
    "whoscored",
    "covers.com",
    "flashscore",
}
_SOCCER_OFFICIAL_LEAGUE_DOMAINS = {
    "uefa.com",
    "fifa.com",
    "bundesliga.com",
    "premierleague.com",
    "laliga.com",
    "ligue1.com",
    "seriea.com",
    "dfb.de",
}
_SOCCER_MEDIA_DOMAINS = {
    "espn.com",
    "90min.com",
    "football365.com",
    "fotmob.com",
    "onefootball.com",
    "rotowire.com",
    "sportsmole.co.uk",
    "thestatszone.com",
    "transfermarkt.com",
    "whoscored.com",
    "covers.com",
    "flashscore.com",
    "reddit.com",
}
_KEYLIKE_PATTERN = re.compile(r"^[A-Za-z0-9_\-]{20,}$")


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in values:
        text = str(raw or "").strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _merge_lists(base: Any, extra: Any) -> list[Any]:
    existing = list(base or [])
    for item in list(extra or []):
        if item not in existing:
            existing.append(item)
    return existing


def _deep_merge_dict(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(dict(merged[key]), value)
        elif isinstance(value, list) and isinstance(merged.get(key), list):
            merged[key] = _merge_lists(merged[key], value)
        else:
            merged[key] = value
    return merged


def _looks_like_secret(value: str) -> bool:
    text = str(value or "").strip()
    lowered = text.lower()
    if not text:
        return False
    if lowered in _SOURCE_PRIORITY_RANK or lowered in _DEFAULT_SOURCE_PRIORITY or lowered in {_TENNIS_CACHE_SOURCE, _MLB_CACHE_SOURCE, _SOCCER_CACHE_SOURCE}:
        return False
    if any(marker in lowered for marker in ("api_key", "apikey", "token", "bearer", "secret", "sk-")):
        return True
    compact = text.replace("-", "").replace("_", "")
    return bool(_KEYLIKE_PATTERN.match(text)) and compact.isalnum()


def _sanitize_source_values(values: list[str]) -> list[str]:
    safe: list[str] = []
    for raw in values:
        text = str(raw or "").strip()
        if not text or _looks_like_secret(text):
            continue
        safe.append(text)
    return _dedupe(safe)


def _as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _latest_non_null(rows: pd.DataFrame, columns: tuple[str, ...]) -> float | None:
    if rows.empty:
        return None
    for column in columns:
        if column not in rows.columns:
            continue
        series = rows[column].dropna()
        if not series.empty:
            return _as_float(series.iloc[-1])
    return None


def _head_to_head_wins(rows: pd.DataFrame, player1: str) -> float:
    wins = 0.0
    if rows.empty or "result" not in rows.columns:
        return wins
    for _, row in rows.iterrows():
        if row.get("player1_name") == player1 and row.get("result") == "player1_win":
            wins += 1.0
        elif row.get("player2_name") == player1 and row.get("result") == "player2_win":
            wins += 1.0
    return wins


def _coerce_candidate_line(candidate: dict[str, Any]) -> float | None:
    for key in ("line", "market_line", "spread_line", "point", "points"):
        value = _as_float(candidate.get(key))
        if value is not None:
            return value
    team = str(candidate.get("team") or "")
    match = re.search(r"([+-]\d+(?:\.\d+)?)", team)
    if match:
        return _as_float(match.group(1))
    return None


def _candidate_weather_material(candidate: dict[str, Any]) -> bool:
    market = str(candidate.get("market", "") or "").strip().lower()
    return market in {"totals", "team_total", "spreads"}


@dataclass(frozen=True)
class EvidenceEnrichmentResult:
    triggered: bool
    trigger_reasons: tuple[str, ...]
    missing_evidence_searched: tuple[str, ...]
    sources_checked: tuple[str, ...]
    sources_found: tuple[str, ...]
    evidence_before: str
    evidence_after: str
    concrete_score_before: int
    concrete_score_after: int
    remaining_missing_evidence: tuple[str, ...]
    final_arbiter_decision: str
    updated_candidate: dict[str, Any]
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "triggered": self.triggered,
            "trigger_reason": list(self.trigger_reasons),
            "missing_evidence_searched": list(self.missing_evidence_searched),
            "sources_checked": list(self.sources_checked),
            "sources_found": list(self.sources_found),
            "evidence_before": self.evidence_before,
            "evidence_after": self.evidence_after,
            "concrete_score_before": self.concrete_score_before,
            "concrete_score_after": self.concrete_score_after,
            "remaining_missing_evidence": list(self.remaining_missing_evidence),
            "final_arbiter_decision": self.final_arbiter_decision,
        }
        payload.update({key: value for key, value in self.details.items()})
        return payload


class EvidenceEnrichmentPass:
    def __init__(self) -> None:
        committee_cfg = (settings or {}).get("committee") or {}
        self.min_sources_for_high = int(committee_cfg.get("research_min_sources_for_high", 2) or 2)
        self._soccer_fetcher_instance: SoccerFetcher | None = None

    def run(
        self,
        *,
        candidate: dict[str, Any],
        research: ResearchMindDecision,
        model: ModelMindDecision,
    ) -> EvidenceEnrichmentResult:
        trigger_reasons = self._trigger_reasons(candidate=candidate, research=research, model=model)
        tasks = self._task_list(candidate=candidate, research=research)
        sport = str(candidate.get("sport", "") or "").strip().lower()
        if not trigger_reasons:
            return EvidenceEnrichmentResult(
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
                details=self._details_for_sport(sport),
            )

        fetched_payload: dict[str, Any] = {}
        details = self._details_for_sport(sport)
        if sport in {"tennis", "tennis_wta"}:
            fetched_payload, details = self._fetch_tennis_enrichment_payload(candidate)
        elif sport == "soccer":
            fetched_payload, details = self._fetch_soccer_enrichment_payload(candidate)
        elif sport == "mlb":
            fetched_payload, details = self._fetch_mlb_enrichment_payload(candidate)
        explicit_payload = dict(candidate.get("evidence_enrichment_payload") or {})
        payload = self._merge_payloads(fetched_payload, explicit_payload)
        sources_checked = tuple(self._sources_checked(candidate, tasks, payload=payload))
        sources_found = tuple(self._sources_found(candidate, payload))
        updated_candidate = self._apply_payload(candidate, payload)

        return EvidenceEnrichmentResult(
            triggered=True,
            trigger_reasons=tuple(trigger_reasons),
            missing_evidence_searched=tuple(tasks),
            sources_checked=sources_checked,
            sources_found=sources_found,
            evidence_before=str(research.evidence_status or ""),
            evidence_after="",
            concrete_score_before=int(research.concrete_info_score or 0),
            concrete_score_after=int(research.concrete_info_score or 0),
            remaining_missing_evidence=tuple(research.missing_evidence),
            final_arbiter_decision="",
            updated_candidate=updated_candidate,
            details=details,
        )

    def finalize(
        self,
        *,
        result: EvidenceEnrichmentResult,
        research_after: ResearchMindDecision,
        final_arbiter_decision: str,
    ) -> EvidenceEnrichmentResult:
        if not result.triggered:
            return result
        return EvidenceEnrichmentResult(
            triggered=True,
            trigger_reasons=result.trigger_reasons,
            missing_evidence_searched=result.missing_evidence_searched,
            sources_checked=result.sources_checked,
            sources_found=result.sources_found,
            evidence_before=result.evidence_before,
            evidence_after=str(research_after.evidence_status or ""),
            concrete_score_before=result.concrete_score_before,
            concrete_score_after=int(research_after.concrete_info_score or 0),
            remaining_missing_evidence=tuple(research_after.missing_evidence),
            final_arbiter_decision=str(final_arbiter_decision or ""),
            updated_candidate=result.updated_candidate,
            details={**result.details, "source_quality": str(research_after.source_quality_summary or result.details.get("source_quality", ""))},
        )

    def _trigger_reasons(
        self,
        *,
        candidate: dict[str, Any],
        research: ResearchMindDecision,
        model: ModelMindDecision,
    ) -> list[str]:
        reasons: list[str] = []
        sport = str(candidate.get("sport", "") or "").strip().lower()
        evidence_status = str(research.evidence_status or "").upper()
        if evidence_status in {"INSUFFICIENT", "PARTIAL"}:
            reasons.append(f"evidence status is {evidence_status.lower()}")
        if int(research.concrete_info_score or 0) < 50:
            reasons.append("concrete info score is below 50")
        if int(research.source_count or 0) < self.min_sources_for_high:
            reasons.append("source count is below the configured minimum")
        if str(research.source_quality_summary or "").lower() == "weak":
            reasons.append("source quality is weak")
        if str(research.injury_status or "").lower() == "not_checked":
            reasons.append("injury or team-news context is not checked")
        if str(research.lineup_status or "").lower() in {"unknown", "missing_near_kickoff"}:
            reasons.append("lineup or probable lineup context is missing near start time")
        if research.sport_specific_missing_evidence:
            reasons.append("sport-critical evidence is missing")
        if sport == "soccer" and (
            str(research.rotation_status or "").lower() == "not_checked"
            or str(research.motivation_status or "").lower() == "not_checked"
        ):
            reasons.append("soccer rotation or motivation context is missing")
        if sport == "mlb" and any("pitcher" in str(item).lower() or "starter" in str(item).lower() for item in research.missing_evidence):
            reasons.append("probable pitcher evidence is missing")
        if sport in {"tennis", "tennis_wta"} and any(
            word in str(item).lower()
            for item in research.missing_evidence
            for word in ("surface", "ranking", "elo", "injury")
        ):
            reasons.append("tennis surface, ranking, or injury context is missing")
        if sport == "nhl" and any("goalie" in str(item).lower() for item in research.missing_evidence):
            reasons.append("goalie evidence is missing")
        if sport == "basketball" and any("star-player" in str(item).lower() or "star player" in str(item).lower() for item in research.missing_evidence):
            reasons.append("star-player injury status is missing")
        return _dedupe(reasons)

    def _task_list(self, *, candidate: dict[str, Any], research: ResearchMindDecision) -> list[str]:
        sport = str(candidate.get("sport", "") or "").strip().lower()
        base_missing = [str(item) for item in (research.sport_specific_missing_evidence or research.missing_evidence)]
        if sport == "soccer":
            tasks = [
                "lineups/probable lineups",
                "injuries/suspensions",
                "team news",
                "rotation risk",
                "motivation/context",
                "fixture congestion",
                "official/team/league sources",
                "reputable preview sources",
            ]
        elif sport == "mlb":
            tasks = [
                "probable pitchers",
                "pitcher changes",
                "lineup confirmation",
                "injuries",
                "bullpen workload",
                "weather/park factors",
                "handedness matchup",
            ]
        elif sport == "basketball":
            tasks = [
                "injury report",
                "star player availability",
                "projected lineup",
                "rest/back-to-back",
                "usage/pace impact",
                "playoff/rest motivation",
            ]
        elif sport in {"tennis", "tennis_wta"}:
            tasks = [
                "surface",
                "injury/retirement concerns",
                "recent match fatigue",
                "ranking/Elo context",
                "serve/return profile",
                "tournament context",
                "H2H style matchup",
            ]
        elif sport == "nhl":
            tasks = [
                "starting goalie",
                "injuries",
                "back-to-back/rest",
                "travel",
                "special teams",
                "goalie impact for totals/team totals",
            ]
        else:
            tasks = base_missing
        return _dedupe(base_missing + tasks)

    def _sources_checked(self, candidate: dict[str, Any], tasks: list[str], *, payload: dict[str, Any] | None = None) -> list[str]:
        explicit = candidate.get("evidence_enrichment_sources_checked")
        if explicit:
            return _sanitize_source_values([str(item) for item in explicit])
        checked = list(_DEFAULT_SOURCE_PRIORITY)
        if payload:
            checked.extend(payload.get("sources_found") or [])
        return _sanitize_source_values(checked)

    def _sources_found(self, candidate: dict[str, Any], payload: dict[str, Any]) -> list[str]:
        explicit = list(payload.get("sources_found") or candidate.get("evidence_enrichment_sources_found") or [])
        if explicit:
            return self._sort_sources(explicit)
        found: list[str] = []
        for source in list(payload.get("scraped_context_sources") or []):
            found.append(str(source))
        context = payload.get("scraped_context") or {}
        for key in ("availability_source", "lineup_source"):
            value = str(context.get(key) or payload.get(key) or "").strip()
            if value:
                found.append(value)
        return self._sort_sources(found)

    def _sort_sources(self, values: list[str]) -> list[str]:
        deduped = _sanitize_source_values([str(item) for item in values])
        return sorted(deduped, key=lambda item: (_SOURCE_PRIORITY_RANK.get(str(item).strip().lower(), 99), str(item).lower()))

    @staticmethod
    def _merge_payloads(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        if not base:
            return dict(override)
        if not override:
            return dict(base)
        return _deep_merge_dict(dict(base), dict(override))

    @staticmethod
    def _details_for_sport(sport: str) -> dict[str, Any]:
        if sport in {"tennis", "tennis_wta"}:
            return dict(_DEFAULT_TENNIS_DETAILS)
        if sport == "soccer":
            return dict(_DEFAULT_SOCCER_DETAILS)
        if sport == "mlb":
            return dict(_DEFAULT_MLB_DETAILS)
        return {}

    def _fetch_tennis_enrichment_payload(self, candidate: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        details = dict(_DEFAULT_TENNIS_DETAILS)
        try:
            sport = str(candidate.get("sport", "") or "").strip().lower()
            df = self._load_tennis_cache(sport)
            if df.empty:
                return {}, details

            player1 = str(candidate.get("home") or "").strip()
            player2 = str(candidate.get("away") or "").strip()
            if not player1 or not player2:
                return {}, details

            all_players = (
                set(df.get("player1_name", pd.Series(dtype=object)).dropna().astype(str).unique())
                | set(df.get("player2_name", pd.Series(dtype=object)).dropna().astype(str).unique())
            )
            alias_map = build_entity_alias_map(all_players)
            p1 = resolve_canonical_name(player1, all_players, alias_map=alias_map)
            p2 = resolve_canonical_name(player2, all_players, alias_map=alias_map)
            p1_rows = self._player_history(df, p1)
            p2_rows = self._player_history(df, p2)
            if p1_rows.empty or p2_rows.empty:
                return {}, details

            payload: dict[str, Any] = {
                "sources_found": [_TENNIS_CACHE_SOURCE, "bookmaker"],
                "scraped_context_sources": [_TENNIS_CACHE_SOURCE, "bookmaker"],
                "scraped_context": {
                    "player1_name": p1,
                    "player2_name": p2,
                    "availability_source": _TENNIS_CACHE_SOURCE,
                    "availability_fetched_at": datetime.now(timezone.utc).isoformat(),
                },
            }
            prediction_factors: list[dict[str, Any]] = []
            context_adjustments: list[dict[str, Any]] = []
            highlights: list[str] = []

            surface = self._surface_for_candidate(candidate, p1_rows, p2_rows)
            if surface:
                payload["surface"] = surface
                payload["scraped_context"]["surface"] = surface
                details["surface_status"] = "verified"
                highlights.append(f"Surface verified from tennis cache and tournament metadata: {surface}.")
                p1_surface = _as_float(p1_rows.get("p1_surface_win", pd.Series(dtype=float)).dropna().tail(10).mean())
                p2_surface = _as_float(p2_rows.get("p1_surface_win", pd.Series(dtype=float)).dropna().tail(10).mean())
                if p1_surface is not None and p2_surface is not None:
                    prediction_factors.append({
                        "name": "surface_win_diff",
                        "category": "matchup",
                        "summary": f"{surface} win-rate context checked from the tennis cache.",
                        "value": round(p1_surface - p2_surface, 4),
                    })
            else:
                details["surface_status"] = "missing"

            tournament = str(candidate.get("tournament") or candidate.get("league") or "").strip()
            round_name = str(candidate.get("round") or "").strip()
            latest_row = self._latest_matchup_row(df, p1, p2)
            if not tournament and latest_row is not None:
                tournament = str(latest_row.get("tourney_name") or "").strip()
            if not round_name and latest_row is not None:
                round_name = str(latest_row.get("round") or "").strip()
            if tournament or round_name:
                payload["tournament"] = tournament or candidate.get("tournament") or candidate.get("league") or ""
                if round_name:
                    payload["round"] = round_name
                    payload["scraped_context"]["round"] = round_name
                payload["scraped_context"]["tournament"] = tournament or payload.get("tournament", "")
                payload["scraped_context"]["tournament_context_checked"] = True
                details["tournament_context_status"] = "checked"
                highlights.append(
                    f"Tournament context checked: {(tournament or 'tournament unknown')} {(f'· {round_name}' if round_name else '')}".strip()
                )
            else:
                details["tournament_context_status"] = "missing"

            p1_rank = _latest_non_null(p1_rows, ("player1_rank",))
            p2_rank = _latest_non_null(p2_rows, ("player1_rank",))
            p1_rank_pts = _latest_non_null(p1_rows, ("player1_rank_pts",))
            p2_rank_pts = _latest_non_null(p2_rows, ("player1_rank_pts",))
            if p1_rank is not None and p2_rank is not None:
                payload["scraped_context"].update({
                    "player1_rank": p1_rank,
                    "player2_rank": p2_rank,
                })
                details["ranking_elo_status"] = "checked"
                rank_value = log(max(float(p2_rank), 1.0) / max(float(p1_rank), 1.0))
                prediction_factors.append({
                    "name": "rank_log_ratio",
                    "category": "ranking",
                    "summary": "Ranking context checked from the tennis feature cache.",
                    "value": round(rank_value, 4),
                })
                if p1_rank_pts is not None and p2_rank_pts is not None:
                    payload["scraped_context"].update({
                        "player1_rank_pts": p1_rank_pts,
                        "player2_rank_pts": p2_rank_pts,
                    })
                    pts_value = log(max(float(p1_rank_pts), 1.0) / max(float(p2_rank_pts), 1.0))
                    prediction_factors.append({
                        "name": "rank_pts_log_ratio",
                        "category": "ranking",
                        "summary": "Ranking-points context checked from the tennis feature cache.",
                        "value": round(pts_value, 4),
                    })
            else:
                details["ranking_elo_status"] = "missing"

            p1_form = _as_float(p1_rows.get("p1_form", pd.Series(dtype=float)).dropna().tail(10).mean())
            p2_form = _as_float(p2_rows.get("p1_form", pd.Series(dtype=float)).dropna().tail(10).mean())
            if p1_form is not None and p2_form is not None:
                prediction_factors.append({
                    "name": "form_diff",
                    "category": "recent_form",
                    "summary": "Recent form checked from the tennis history cache.",
                    "value": round(p1_form - p2_form, 4),
                })
                highlights.append("Recent form pulled from the tennis history cache.")

            p1_load = _latest_non_null(p1_rows, ("p1_load",))
            p2_load = _latest_non_null(p2_rows, ("p1_load",))
            recent_long_match = any(
                value is not None and float(value) >= 1.5
                for value in (p1_load, p2_load)
            )
            if p1_load is not None or p2_load is not None:
                payload["scraped_context"].update({
                    "fatigue_checked": True,
                    "p1_match_load": p1_load,
                    "p2_match_load": p2_load,
                })
                if recent_long_match:
                    payload["scraped_context"]["recent_long_match"] = True
                details["fatigue_status"] = "checked"
                context_adjustments.append({
                    "name": "travel_fatigue",
                    "category": "schedule",
                    "summary": "Recent match-load context checked from the tennis history cache.",
                    "value": round(float((p1_load or 0.0) - (p2_load or 0.0)), 4),
                })
            else:
                details["fatigue_status"] = "missing"

            h2h_rows = self._head_to_head_rows(df, p1, p2)
            if not h2h_rows.empty:
                p1_wins = _head_to_head_wins(h2h_rows, p1)
                total = float(len(h2h_rows))
                if total > 0:
                    prediction_factors.append({
                        "name": "h2h_p1_win_rate",
                        "category": "matchup",
                        "summary": "Head-to-head style context checked from the tennis history cache.",
                        "value": round(p1_wins / total, 4),
                    })
                    context_adjustments.append({
                        "name": "h2h_tactical_history",
                        "category": "matchup",
                        "summary": f"Head-to-head history checked across {int(total)} prior meeting(s).",
                    })
                    details["style_matchup_status"] = "checked"
            else:
                details["style_matchup_status"] = "missing"

            p1_serve = _as_float(p1_rows.get("roll_p1_ace_rate", pd.Series(dtype=float)).dropna().tail(10).mean())
            p2_serve = _as_float(p2_rows.get("roll_p1_ace_rate", pd.Series(dtype=float)).dropna().tail(10).mean())
            p1_return = _as_float(p1_rows.get("roll_p1_return_pressure", pd.Series(dtype=float)).dropna().tail(10).mean())
            p2_return = _as_float(p2_rows.get("roll_p1_return_pressure", pd.Series(dtype=float)).dropna().tail(10).mean())
            if p1_serve is not None and p2_serve is not None:
                payload["scraped_context"]["serve_return_checked"] = True
                prediction_factors.append({
                    "name": "serve_diff",
                    "category": "matchup",
                    "summary": "Serve profile checked from the tennis history cache.",
                    "value": round(p1_serve - p2_serve, 4),
                })
            if p1_return is not None and p2_return is not None:
                payload["scraped_context"]["serve_return_checked"] = True
                prediction_factors.append({
                    "name": "return_pressure_diff",
                    "category": "matchup",
                    "summary": "Return-pressure profile checked from the tennis history cache.",
                    "value": round(p1_return - p2_return, 4),
                })
            if p1_serve is not None and p2_serve is not None and p1_return is not None and p2_return is not None:
                prediction_factors.append({
                    "name": "serve_balance_diff",
                    "category": "matchup",
                    "summary": "Serve/return balance checked from the tennis history cache.",
                    "value": round((p1_serve - p2_serve) - (0.35 * (p1_return - p2_return)), 4),
                })

            injury_text = " ".join(
                [
                    *(str(item or "") for item in (candidate.get("scraped_context_highlights") or [])),
                    *(str(item or "") for item in (candidate.get("evidence_enrichment_notes") or [])),
                ]
            ).lower()
            if any(pattern in injury_text for pattern in _NEGATIVE_INJURY_PATTERNS):
                payload["scraped_context"].update({
                    "injury_concern": True,
                    "retirement_concern": True,
                    "injury_concern_checked": True,
                    "retirement_concern_checked": True,
                })
                payload["context_referee_decision"] = "VETO"
                payload["context_referee_reason"] = "Tennis enrichment found negative injury or retirement evidence."
                details["injury_retirement_status"] = "negative_signal"
                highlights.append("Negative injury/retirement signal preserved from available tennis evidence.")
            elif payload["scraped_context"].get("injury_concern") or payload["scraped_context"].get("retirement_concern") or candidate.get("scraped_context", {}).get("injury_concern") or candidate.get("scraped_context", {}).get("retirement_concern"):
                details["injury_retirement_status"] = "unresolved"
            else:
                details["injury_retirement_status"] = "no_concern_found"

            if highlights:
                payload["scraped_context_highlights"] = highlights
            if prediction_factors:
                payload["prediction_factors"] = prediction_factors
            if context_adjustments:
                payload["context_adjustments"] = context_adjustments
            if not payload["scraped_context_sources"]:
                payload["scraped_context_sources"] = [_TENNIS_CACHE_SOURCE]
            return payload, details
        except Exception:
            return {}, details

    def _fetch_soccer_enrichment_payload(self, candidate: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        details = dict(_DEFAULT_SOCCER_DETAILS)
        try:
            provider_failures: dict[str, str] = {}
            providers_attempted: list[str] = []
            providers_succeeded: list[str] = []
            providers_failed: list[str] = []

            def _mark_attempt(name: str) -> None:
                if name not in providers_attempted:
                    providers_attempted.append(name)

            def _mark_success(name: str, status: str = "ok") -> None:
                if name not in providers_succeeded:
                    providers_succeeded.append(name)
                details[f"{name}_status"] = status

            def _mark_failure(name: str, reason: str, status: str) -> None:
                if name not in providers_failed:
                    providers_failed.append(name)
                provider_failures[name] = str(reason or status)
                details[f"{name}_status"] = status

            home_team = str(candidate.get("home") or "").strip()
            away_team = str(candidate.get("away") or "").strip()
            commence = candidate.get("commence") or candidate.get("commence_time")
            fallback_status = str(
                candidate.get("match_status")
                or candidate.get("game_status")
                or candidate.get("status")
                or "scheduled"
            ).strip().lower()
            if not home_team or not away_team:
                return {}, details

            payload: dict[str, Any] = {
                "sources_found": ["bookmaker"],
                "scraped_context_sources": ["bookmaker"],
                "scraped_context": {
                    "home_team_name": home_team,
                    "away_team_name": away_team,
                },
            }

            _mark_attempt("api_football")
            api_enricher = self._soccer_api_enricher()
            fixture = self._fetch_soccer_fixture(home_team, away_team, commence)
            fixture_status = fallback_status or "scheduled"
            if fixture:
                fixture_status = self._soccer_fixture_status(fixture, fallback=fixture_status)
                teams = fixture.get("teams") or {}
                fixture_meta = fixture.get("fixture") or fixture
                payload["scraped_context"]["home_team_name"] = str((teams.get("home") or {}).get("name") or home_team)
                payload["scraped_context"]["away_team_name"] = str((teams.get("away") or {}).get("name") or away_team)
                fixture_date = str(fixture_meta.get("date") or "").strip()
                if fixture_date:
                    payload["commence_time"] = fixture_date
                league = fixture.get("league") or {}
                competition = str(league.get("name") or "").strip()
                round_name = str(league.get("round") or "").strip()
                if competition:
                    payload["league"] = competition
                    payload["scraped_context"]["competition_name"] = competition
                if round_name:
                    payload["scraped_context"]["competition_round"] = round_name
                payload["sources_found"].append("api_football")
                payload["scraped_context_sources"].append("api_football")
                _mark_success("api_football", "fixture_found")
            else:
                if getattr(api_enricher, "_is_temporarily_disabled", None) and api_enricher._is_temporarily_disabled():
                    _mark_failure("api_football", getattr(api_enricher, "_disabled_reason", "") or "provider_paused", "provider_paused")
                else:
                    _mark_failure("api_football", "fixture_not_found_or_provider_unavailable", "not_found")
            details["fixture_status"] = fixture_status
            if fixture_status in {"live", "finished", "postponed", "cancelled", "suspended"}:
                payload["status"] = fixture_status

            game = {
                "home_team": home_team,
                "away_team": away_team,
                "home": home_team,
                "away": away_team,
                "commence_time": payload.get("commence_time", commence),
                "status": fixture_status,
            }
            _mark_attempt("availability")
            try:
                availability_context = build_availability_context("soccer", game, None)
            except Exception as exc:
                availability_context = {}
                _mark_failure("availability", str(exc), "provider_failed")
            else:
                if availability_context and (
                    availability_context.get("availability_source")
                    or availability_context.get("lineup_source")
                    or availability_context.get("home_injuries_count") not in (None, "")
                    or availability_context.get("home_suspensions_count") not in (None, "")
                    or availability_context.get("home_likely_starters_count") not in (None, "")
                ):
                    _mark_success("availability", "ok")
                else:
                    _mark_failure("availability", "no_availability_signal", "not_found")
            payload["scraped_context"] = _deep_merge_dict(payload["scraped_context"], availability_context)
            now_iso = datetime.now(timezone.utc).isoformat()
            competition_code = self._soccer_competition_code(candidate, payload)
            _mark_attempt("feature_cache")
            snapshot = self._soccer_snapshot(home_team, away_team, competition=competition_code or None)
            if snapshot is not None and not snapshot.empty:
                _mark_success("feature_cache", "ok")
            else:
                _mark_failure("feature_cache", "mapping_failed_or_no_cache_snapshot", "mapping_failed")
            _mark_attempt("api_football")
            match_enrichment = self._soccer_match_enrichment(home_team, away_team, payload.get("commence_time", commence))
            if match_enrichment:
                _mark_success("api_football", "match_enrichment_ok")
            elif details.get("api_football_status") in {"", "not_found"}:
                if getattr(api_enricher, "_is_temporarily_disabled", None) and api_enricher._is_temporarily_disabled():
                    _mark_failure("api_football", getattr(api_enricher, "_disabled_reason", "") or "provider_paused", "provider_paused")
                else:
                    _mark_failure("api_football", provider_failures.get("api_football", "match_enrichment_unavailable"), details.get("api_football_status") or "not_found")
            _mark_attempt("standings")
            standings_context = self._soccer_standings_context(candidate, payload, home_team=home_team, away_team=away_team)
            if match_enrichment:
                payload["scraped_context"] = _deep_merge_dict(payload["scraped_context"], match_enrichment)
                payload["sources_found"].append("api_football")
                payload["scraped_context_sources"].append("api_football")
            if standings_context:
                payload["scraped_context"] = _deep_merge_dict(payload["scraped_context"], standings_context)
                payload["sources_found"].append("football_data")
                payload["scraped_context_sources"].append("football_data")
                payload["standings_snapshot_age_hours"] = 2.0
                _mark_success("standings", "ok")
            elif snapshot is not None and not snapshot.empty and (
                _as_float(snapshot.get("home_season_pts_rate")) is not None
                or _as_float(snapshot.get("away_season_pts_rate")) is not None
            ):
                home_pts_rate = _as_float(snapshot.get("home_season_pts_rate"))
                away_pts_rate = _as_float(snapshot.get("away_season_pts_rate"))
                payload["scraped_context"] = _deep_merge_dict(
                    payload["scraped_context"],
                    {
                        "standings_source": _SOCCER_CACHE_SOURCE,
                        "standings_checked": True,
                        "standings_proxy": True,
                        "home_season_pts_rate": home_pts_rate,
                        "away_season_pts_rate": away_pts_rate,
                    },
                )
                payload["sources_found"].append(_SOCCER_CACHE_SOURCE)
                payload["scraped_context_sources"].append(_SOCCER_CACHE_SOURCE)
                payload["standings_snapshot_age_hours"] = 12.0
                _mark_success("standings", "proxy")
            else:
                _mark_failure("standings", "standings_unavailable", "not_found")

            _mark_attempt("news_context")
            try:
                news_context = collect_matchup_news_context(
                    sport="soccer",
                    home_team=home_team,
                    away_team=away_team,
                    bet=str(candidate.get("team") or ""),
                    limit=4,
                    timeout=4,
                )
            except Exception as exc:
                news_context = {"sources": [], "highlights": [], "items": [], "channels": {}, "warnings": [str(exc)]}
                _mark_failure("news_context", str(exc), "provider_failed")
            official_context = self._soccer_official_news_context(
                news_context,
                home_team=home_team,
                away_team=away_team,
                competition_name=str(payload.get("league") or payload["scraped_context"].get("competition_name") or ""),
            )
            raw_news_sources = [
                _SOCCER_NEWS_SOURCE_LABELS.get(str(source or "").strip().lower(), str(source or "").strip().lower())
                for source in (news_context.get("sources") or [])
                if str(source or "").strip()
            ]
            news_sources = [
                str(source)
                for source in ((official_context.get("classified_sources") or []) or raw_news_sources)
            ]
            news_highlights = [str(item or "").strip() for item in (news_context.get("highlights") or []) if str(item or "").strip()]
            official_highlights = [str(item or "").strip() for item in (official_context.get("official_highlights") or []) if str(item or "").strip()]
            news_text = " ".join(news_highlights).lower()
            official_text = " ".join(official_highlights).lower()
            reliable_news = [source for source in news_sources if source in _SOCCER_RELIABLE_NEWS_SOURCES]
            official_sources = [source for source in news_sources if source in {"team_official", "league_official"}]
            community_only = bool(news_sources) and not reliable_news
            if official_sources or reliable_news:
                _mark_success("news_context", "usable_sources_found")
            elif news_sources:
                _mark_failure("news_context", "only_weak_or_community_sources_found", "weak_only")
            elif details.get("news_context_status") not in {"provider_failed"}:
                _mark_failure("news_context", "no_usable_sources_found", "not_found")

            sources_found = list(payload.get("sources_found") or [])
            for source in (
                str(availability_context.get("availability_source") or "").strip(),
                str(availability_context.get("lineup_source") or "").strip(),
            ):
                if source:
                    sources_found.append(source)
            sources_found.extend(news_sources)
            payload["sources_found"] = sources_found
            payload["scraped_context_sources"] = sources_found

            context = payload["scraped_context"]
            highlights: list[str] = []
            home_confirmed = int(bool(context.get("home_lineup_confirmed")))
            away_confirmed = int(bool(context.get("away_lineup_confirmed")))
            likely_home = int(_as_float(context.get("home_likely_starters_count")) or 0)
            likely_away = int(_as_float(context.get("away_likely_starters_count")) or 0)
            home_goalkeeper = int(_as_float(context.get("home_lineup_goalkeeper_named")) or 0)
            away_goalkeeper = int(_as_float(context.get("away_lineup_goalkeeper_named")) or 0)
            if home_confirmed and away_confirmed:
                details["lineup_status"] = "confirmed"
                details["probable_lineup_status"] = "confirmed"
                highlights.append("Confirmed starting XIs were found during soccer enrichment.")
            elif likely_home or likely_away:
                details["lineup_status"] = "projected"
                details["probable_lineup_status"] = "projected"
            elif reliable_news and any(term in news_text for term in ("lineup", "projected xi", "predicted xi", "starting xi", "team news")):
                details["lineup_status"] = "checked_proxy"
                details["probable_lineup_status"] = "projected"
            else:
                fallback_lineup_status = "checked_proxy" if (official_sources or reliable_news) else ("provider_failed" if details.get("availability_status") == "provider_failed" else "missing")
                details["lineup_status"] = fallback_lineup_status
                details["probable_lineup_status"] = fallback_lineup_status

            context["soccer_lineup_status"] = details["lineup_status"]
            context["soccer_probable_lineup_status"] = details["probable_lineup_status"]
            if details["lineup_status"] in {"confirmed", "projected", "checked_proxy"}:
                context["lineup_checked"] = 1
                context["probable_lineups_checked"] = 1
                lineup_source = str(context.get("lineup_source") or "").strip()
                proxy_lineup_source = reliable_news[0] if reliable_news else (official_sources[0] if official_sources else "")
                if not lineup_source and proxy_lineup_source:
                    context["lineup_source"] = proxy_lineup_source

            if home_goalkeeper and away_goalkeeper:
                details["goalkeeper_status"] = "confirmed"
            elif home_goalkeeper or away_goalkeeper:
                details["goalkeeper_status"] = "partial"
            else:
                details["goalkeeper_status"] = "provider_failed" if details.get("availability_status") == "provider_failed" else "missing"

            availability_source = str(context.get("availability_source") or "").strip()
            home_injuries = int(_as_float(context.get("home_injuries_count")) or 0)
            away_injuries = int(_as_float(context.get("away_injuries_count")) or 0)
            home_susp = int(_as_float(context.get("home_suspensions_count")) or 0)
            away_susp = int(_as_float(context.get("away_suspensions_count")) or 0)
            injury_terms_present = any(
                term in news_text
                for term in (
                    "injury", "injured", "fit", "fitness", "absent", "absence", "ruled out",
                    "returns", "available", "doubt", "questionable", "late test", "team news",
                )
            )
            suspension_terms_present = any(term in news_text for term in ("suspension", "suspended", "ban"))
            if reliable_news and injury_terms_present and not availability_source:
                context["availability_source"] = reliable_news[0]
                context["availability_fetched_at"] = now_iso
                availability_source = reliable_news[0]
            if official_sources and any(term in official_text for term in ("injury", "injured", "fit", "fitness", "available", "absence", "squad")):
                details["injury_status"] = "checked_fresh"
                context["availability_source"] = official_sources[0]
                context["availability_fetched_at"] = now_iso
            elif reliable_news and injury_terms_present:
                details["injury_status"] = "checked"
                context["availability_source"] = reliable_news[0]
                context["availability_fetched_at"] = now_iso
            else:
                details["injury_status"] = "checked" if availability_source else ("provider_failed" if details.get("availability_status") == "provider_failed" else "not_found")
            context["soccer_injury_status"] = details["injury_status"]
            if details["injury_status"] in {"checked_fresh", "checked_proxy", "checked"}:
                context["team_news_checked"] = 1
                if not str(context.get("availability_source") or "").strip():
                    proxy_availability_source = reliable_news[0] if reliable_news else (official_sources[0] if official_sources else "")
                    if proxy_availability_source:
                        context["availability_source"] = proxy_availability_source
                context.setdefault("availability_fetched_at", now_iso)
            if official_sources and any(term in official_text for term in ("suspension", "suspended", "ban", "disciplinary")):
                details["suspension_status"] = "checked_fresh"
            elif reliable_news and suspension_terms_present:
                details["suspension_status"] = "checked"
            else:
                details["suspension_status"] = "checked" if (availability_source or (reliable_news and suspension_terms_present)) else ("provider_failed" if details.get("availability_status") == "provider_failed" else "not_found")
            context["soccer_suspension_status"] = details["suspension_status"]
            if home_injuries or away_injuries or home_susp or away_susp:
                highlights.append("Injury or suspension counts were found from soccer availability enrichment.")
            elif reliable_news and injury_terms_present:
                highlights.append("Reputable soccer news sources checked the latest injury/team-news context.")
            if official_sources:
                highlights.append("Official club or league sources were checked during soccer enrichment.")

            existing_context = candidate.get("scraped_context") or {}
            merged_context = _deep_merge_dict(dict(existing_context), dict(context))
            competition_blob = " ".join(
                part for part in (
                    str(payload.get("league") or ""),
                    str(merged_context.get("competition_name") or ""),
                    str(merged_context.get("competition_round") or ""),
                    news_text,
                ) if part
            ).lower()
            if any(term in competition_blob for term in ("champions league", "europa league", "conference league", "continental")):
                merged_context["european_rotation_risk"] = 1
            if any(term in competition_blob for term in ("cup", "fa cup", "coppa", "pokal", "copa del rey", "knockout")):
                merged_context["cup_rotation_risk"] = 1
            if any(term in f"{news_text} {official_text}" for term in ("three games", "congestion", "fatigue", "rested", "rotation likely", "rotated", "busy schedule")):
                merged_context["fixture_congestion_risk"] = 1
            if any(term in competition_blob for term in ("title race", "must win", "must-win", "relegation", "european qualification", "playoff", "qualification battle")):
                merged_context["playoff_motivation"] = 1
            if any(term in competition_blob for term in ("derby", "rivalry")):
                merged_context["rivalry_fixture"] = 1
            if any(term in competition_blob for term in ("dead rubber", "nothing to play for", "already safe")):
                merged_context["nothing_to_play_for"] = 1
            if any(term in competition_blob for term in ("final day", "last round", "season finale", "matchday 38", "matchday 34")):
                merged_context["final_day_volatility"] = 1

            fixture_congestion = bool(
                merged_context.get("fixture_congestion_risk")
                or merged_context.get("cup_rotation_risk")
                or merged_context.get("european_rotation_risk")
            )
            details["fixture_congestion_status"] = "flagged" if fixture_congestion else "not_flagged"
            if merged_context.get("is_playoff") or merged_context.get("playoff_motivation") or merged_context.get("final_day_volatility") or merged_context.get("nothing_to_play_for") or merged_context.get("rivalry_fixture"):
                details["motivation_status"] = "checked" if official_sources or reliable_news or merged_context.get("playoff_motivation") or details.get("standings_status") == "ok" else "checked_proxy"
                if official_sources or reliable_news:
                    merged_context["motivation_checked"] = 1
                if details.get("standings_status") in {"ok", "proxy"}:
                    merged_context["motivation_checked"] = 1
            else:
                details["motivation_status"] = "not_required"
            if merged_context.get("cup_rotation_risk") or merged_context.get("european_rotation_risk") or merged_context.get("fixture_congestion_risk"):
                proxy_rotation_sources = official_sources or reliable_news or availability_source or context.get("lineup_source") or details.get("feature_cache_status") == "ok"
                details["rotation_status"] = "checked" if proxy_rotation_sources else "missing"
                if proxy_rotation_sources:
                    merged_context["rotation_checked"] = 1
            else:
                details["rotation_status"] = "not_required"
            merged_context["soccer_motivation_status"] = details["motivation_status"]
            merged_context["soccer_rotation_status"] = details["rotation_status"]
            merged_context["availability_lookup_status"] = details.get("availability_status") or ""
            merged_context["news_context_lookup_status"] = details.get("news_context_status") or ""
            merged_context["feature_cache_lookup_status"] = details.get("feature_cache_status") or ""
            merged_context["standings_lookup_status"] = details.get("standings_status") or ""
            context.update({k: v for k, v in merged_context.items() if k not in context or merged_context[k]})

            factor_names = {str((item or {}).get("name") or "").strip().lower() for item in (candidate.get("prediction_factors") or []) if isinstance(item, dict)}
            prediction_factors = list(payload.get("prediction_factors") or [])
            context_adjustments = list(payload.get("context_adjustments") or [])
            if snapshot is not None and not snapshot.empty:
                payload["sources_found"].append(_SOCCER_CACHE_SOURCE)
                payload["scraped_context_sources"].append(_SOCCER_CACHE_SOURCE)
                form_diff = _as_float(snapshot.get("form_diff"))
                xg_diff = _as_float(snapshot.get("xg_diff"))
                dc_xg_diff = _as_float(snapshot.get("dc_xg_diff"))
                h2h_rate = _as_float(snapshot.get("h2h_home_win_rate"))
                if form_diff is not None:
                    prediction_factors.append({
                        "name": "home_away_form",
                        "category": "form",
                        "summary": "Recent home/away form context checked from the soccer feature cache.",
                        "value": round(form_diff, 4),
                    })
                if xg_diff is not None:
                    prediction_factors.append({
                        "name": "xg_edge",
                        "category": "matchup",
                        "summary": "Expected-goals proxy context checked from the soccer feature cache.",
                        "value": round(xg_diff, 4),
                    })
                if dc_xg_diff is not None:
                    context_adjustments.append({
                        "name": "xg_structure",
                        "category": "matchup",
                        "summary": "Dixon-Coles xG structure checked from the soccer feature cache.",
                        "value": round(dc_xg_diff, 4),
                    })
                if h2h_rate is not None:
                    context_adjustments.append({
                        "name": "h2h_tactical_history",
                        "category": "matchup",
                        "summary": "Head-to-head tactical history checked from the soccer feature cache.",
                        "value": round(h2h_rate - 0.5, 4),
                    })
                payload["scraped_context"]["home_rest_days"] = round(_as_float(snapshot.get("home_rest_days")) or 0.0, 2)
                payload["scraped_context"]["away_rest_days"] = round(_as_float(snapshot.get("away_rest_days")) or 0.0, 2)

            home_form_live = str(payload["scraped_context"].get("home_form") or "").strip().upper()
            if home_form_live:
                prediction_factors.append({
                    "name": "home_form",
                    "category": "form",
                    "summary": "Recent form checked from API-Football match enrichment.",
                    "value": round(self._soccer_form_strength(home_form_live), 4),
                })
            if payload["scraped_context"].get("home_xg") is not None and payload["scraped_context"].get("away_xg") is not None:
                try:
                    home_xg = float(payload["scraped_context"]["home_xg"])
                    away_xg = float(payload["scraped_context"]["away_xg"])
                    prediction_factors.append({
                        "name": "home_xg",
                        "category": "matchup",
                        "summary": "Live matchup xG snapshot checked from API-Football.",
                        "value": round(home_xg - away_xg, 4),
                    })
                except Exception:
                    pass
            corners_avg = _as_float(payload["scraped_context"].get("home_corners_avg"))
            if corners_avg is not None:
                context_adjustments.append({
                    "name": "set_piece_pressure",
                    "category": "matchup",
                    "summary": "Set-piece pressure proxy checked from corner-generation context.",
                    "value": round(corners_avg / 10.0, 4),
                })

            payload["prediction_factors"] = _merge_lists(payload.get("prediction_factors") or [], prediction_factors)
            payload["context_adjustments"] = _merge_lists(payload.get("context_adjustments") or [], context_adjustments)
            factor_names = {str((item or {}).get("name") or "").strip().lower() for item in (payload.get("prediction_factors") or []) if isinstance(item, dict)}
            adjustment_names = {str((item or {}).get("name") or "").strip().lower() for item in (payload.get("context_adjustments") or []) if isinstance(item, dict)}
            details["home_away_form_status"] = "checked" if ({"home_form", "away_form", "home_away_form", "venue_split_edge"} & factor_names) else ("mapping_failed" if details.get("feature_cache_status") == "mapping_failed" else "missing")
            details["xg_context_status"] = "checked" if ({"xg_edge", "xg_delta", "home_xg", "away_xg", "xg_structure"} & factor_names) or ("xg_structure" in adjustment_names) else ("unavailable_with_reason" if details.get("api_football_status") in {"not_found", "provider_paused"} or details.get("feature_cache_status") == "mapping_failed" else "missing")

            market = str(candidate.get("market", "") or "").strip().lower()
            if market in {"moneyline", "double_chance", "draw_no_bet"}:
                details["market_fit_status"] = "acceptable"
            elif market == "spreads":
                details["market_fit_status"] = "needs_stronger_support"
            elif market == "totals":
                details["market_fit_status"] = "xg_supported" if details["xg_context_status"] == "checked" else "needs_xg_context"
            else:
                details["market_fit_status"] = "unsupported"
            real_research_sources = set(official_sources) | set(reliable_news)
            if details.get("feature_cache_status") == "ok":
                real_research_sources.add(_SOCCER_CACHE_SOURCE)
            if details.get("standings_status") == "ok":
                real_research_sources.add("football_data")
            if details.get("api_football_status") in {"fixture_found", "match_enrichment_ok"} and not (set(payload.get("sources_found") or []) <= {"bookmaker", "odds_snapshot"}):
                real_research_sources.add("api_football")
            details["source_quality"] = "strong" if len(real_research_sources) >= 2 or official_sources else "mixed" if len(real_research_sources) == 1 else "weak"

            if official_highlights:
                highlights.extend(official_highlights[:2])
            elif reliable_news:
                highlights.extend(news_highlights[:2])
            elif community_only:
                highlights.append("Only community or weak soccer news sources were found during enrichment.")
            if (official_sources or reliable_news) and any(term in f"{news_text} {official_text}" for term in ("rotation", "rested", "rest", "minutes load", "manager rotation")):
                highlights.append("Reliable soccer sources checked the latest rotation context.")
            if (official_sources or reliable_news) and any(term in f"{news_text} {official_text}" for term in ("title race", "must win", "derby", "relegation", "nothing to play for")):
                highlights.append("Reliable soccer sources checked the latest motivation context.")

            if highlights:
                payload["scraped_context_highlights"] = _dedupe(highlights)

            proxy_coverage = bool(
                official_sources
                or reliable_news
                or availability_source
                or details.get("feature_cache_status") == "ok"
                or details.get("standings_status") == "ok"
            )
            if details.get("api_football_status") == "provider_paused" and proxy_coverage:
                details["api_football_status"] = "proxy_covered"
                if "api_football" in providers_failed:
                    providers_failed = [name for name in providers_failed if name != "api_football"]
                provider_failures.pop("api_football", None)
            if details.get("availability_status") == "provider_failed" and (
                availability_source
                or official_sources
                or reliable_news
                or likely_home
                or likely_away
                or home_injuries
                or away_injuries
                or home_susp
                or away_susp
            ):
                details["availability_status"] = "checked_proxy"
                if "availability" in providers_failed:
                    providers_failed = [name for name in providers_failed if name != "availability"]
                provider_failures.pop("availability", None)

            details["providers_attempted"] = providers_attempted
            details["providers_succeeded"] = providers_succeeded
            details["providers_failed"] = providers_failed
            details["provider_failure_reasons"] = provider_failures
            return payload, details
        except Exception:
            return {}, details

    @staticmethod
    def _fetch_soccer_fixture(home_team: str, away_team: str, commence: Any) -> dict[str, Any] | None:
        enricher = EvidenceEnrichmentPass._soccer_api_enricher()
        if not enricher.api_key:
            return None
        try:
            return enricher._find_fixture(home_team, away_team, commence)
        except Exception:
            return None

    @staticmethod
    @lru_cache(maxsize=1)
    def _soccer_api_enricher() -> APIFootballEnricher:
        return APIFootballEnricher()

    @staticmethod
    def _soccer_fixture_status(fixture: dict[str, Any], *, fallback: str) -> str:
        status = (fixture.get("fixture") or {}).get("status") or fixture.get("status") or {}
        short = str(status.get("short") or status.get("status") or "").strip().upper()
        if short in {"NS", "TBD"}:
            return "scheduled"
        if short in {"PST"}:
            return "postponed"
        if short in {"1H", "HT", "2H", "ET", "BT", "P", "INT", "LIVE"}:
            return "live"
        if short in {"FT", "AET", "PEN"}:
            return "finished"
        if short in {"CANC", "ABD"}:
            return "cancelled"
        if short in {"SUSP"}:
            return "suspended"
        return str(fallback or "scheduled").strip().lower()

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_soccer_cache() -> pd.DataFrame:
        path = Path(__file__).resolve().parent.parent.parent / "data" / "cache" / "soccer_features.parquet"
        if not path.exists():
            return pd.DataFrame()
        frame = pd.read_parquet(path)
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
        return frame

    def _soccer_snapshot(self, home_team: str, away_team: str, *, competition: str | None = None) -> pd.Series | None:
        df = self._load_soccer_cache()
        if df.empty:
            return None
        team_resolver = TeamResolver("soccer")

        def _resolve_pair(work: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
            all_teams = set(work.get("home_team", pd.Series(dtype=object)).dropna().astype(str).unique()) | set(work.get("away_team", pd.Series(dtype=object)).dropna().astype(str).unique())
            alias_map = build_entity_alias_map(all_teams)
            resolved_home = resolve_canonical_name(home_team, all_teams, alias_map=alias_map)
            resolved_away = resolve_canonical_name(away_team, all_teams, alias_map=alias_map)
            if resolved_home == home_team:
                resolved_home = team_resolver.resolve(home_team)
            if resolved_away == away_team:
                resolved_away = team_resolver.resolve(away_team)
            return work, resolved_home, resolved_away

        work = df
        if competition and "competition" in work.columns:
            comp_rows = work.loc[work["competition"].astype(str) == str(competition)]
            if not comp_rows.empty:
                work = comp_rows

        for candidate_frame in (work, df):
            frame, resolved_home, resolved_away = _resolve_pair(candidate_frame)
            rows = frame.loc[
                (frame.get("home_team", pd.Series(dtype=object)) == resolved_home)
                & (frame.get("away_team", pd.Series(dtype=object)) == resolved_away)
            ]
            if rows.empty:
                reverse = frame.loc[
                    (frame.get("home_team", pd.Series(dtype=object)) == resolved_away)
                    & (frame.get("away_team", pd.Series(dtype=object)) == resolved_home)
                ]
                if not reverse.empty:
                    rows = reverse
            if not rows.empty:
                return rows.sort_values("date").tail(8).mean(numeric_only=True)

        return None

    @staticmethod
    def _soccer_competition_code(candidate: dict[str, Any], payload: dict[str, Any]) -> str:
        sport_key = str(candidate.get("sport_key") or candidate.get("league_key") or "").strip()
        if sport_key:
            mapped = SOCCER_ODDS_TO_COMPETITION.get(sport_key)
            if mapped:
                return mapped
        league = str(payload.get("league") or (payload.get("scraped_context") or {}).get("competition_name") or candidate.get("league") or "").strip()
        resolved_key = resolve_soccer_key(sport_key=sport_key or None, league=league or None)
        if resolved_key:
            return str(SOCCER_ODDS_TO_COMPETITION.get(resolved_key) or "")
        return ""

    @staticmethod
    def _soccer_form_strength(form: str) -> float:
        text = str(form or "").strip().upper()
        if not text:
            return 0.0
        score = 0.0
        for char in text[:5]:
            if char == "W":
                score += 1.0
            elif char == "D":
                score += 0.35
            elif char == "L":
                score -= 0.6
        return score / max(len(text[:5]), 1)

    @staticmethod
    def _slug_tokens(value: str) -> set[str]:
        text = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower())
        return {
            token
            for token in text.split()
            if token and token not in {"fc", "cf", "sc", "ac", "club", "the", "team", "de", "sv", "afc"}
        }

    def _classify_soccer_news_item(
        self,
        item: dict[str, Any],
        *,
        home_team: str,
        away_team: str,
        competition_name: str,
    ) -> str:
        url = str(item.get("url") or "").strip()
        title = str(item.get("title") or "").strip().lower()
        snippet = str(item.get("snippet") or "").strip().lower()
        source = str(item.get("source") or "").strip().lower()
        host = urlparse(url).netloc.lower().replace("www.", "") if url else source
        blob = " ".join(part for part in (host, title, snippet) if part)

        if host in _SOCCER_OFFICIAL_LEAGUE_DOMAINS:
            return "league_official"
        if host in _SOCCER_MEDIA_DOMAINS:
            return _SOCCER_NEWS_SOURCE_LABELS.get(host, source or host or "web")
        if "official" in blob:
            competition_tokens = self._slug_tokens(competition_name)
            if competition_tokens and any(token in host or token in blob for token in competition_tokens):
                return "league_official"
            team_tokens = self._slug_tokens(home_team) | self._slug_tokens(away_team)
            if team_tokens and any(token in host or token in blob for token in team_tokens):
                return "team_official"
        team_tokens = self._slug_tokens(home_team) | self._slug_tokens(away_team)
        if team_tokens and any(token in host for token in team_tokens) and host not in _SOCCER_MEDIA_DOMAINS:
            return "team_official"
        return _SOCCER_NEWS_SOURCE_LABELS.get(host, source or host or "web")

    def _soccer_official_news_context(
        self,
        news_context: dict[str, Any],
        *,
        home_team: str,
        away_team: str,
        competition_name: str,
    ) -> dict[str, Any]:
        items = [item for item in (news_context.get("items") or []) if isinstance(item, dict)]
        official_items: list[dict[str, Any]] = []
        classified_sources: list[str] = []
        highlights: list[str] = []
        for item in items:
            label = self._classify_soccer_news_item(
                item,
                home_team=home_team,
                away_team=away_team,
                competition_name=competition_name,
            )
            text = " ".join(
                part for part in (str(item.get("title") or "").strip(), str(item.get("snippet") or "").strip()) if part
            ).strip()
            classified_sources.append(label)
            if label in {"team_official", "league_official"}:
                official_items.append(item)
                if text:
                    highlights.append(text[:280])
        return {
            "classified_sources": _dedupe(classified_sources),
            "official_items": official_items,
            "official_highlights": _dedupe(highlights),
        }

    def _soccer_match_enrichment(self, home_team: str, away_team: str, commence: Any) -> dict[str, Any]:
        enricher = self._soccer_api_enricher()
        if not enricher.api_key:
            return {}
        try:
            if getattr(enricher, "_is_temporarily_disabled", None) and enricher._is_temporarily_disabled():
                return {}
            return enricher._fetch_match_enrichment(home_team, away_team, commence, ["form", "xg", "corners", "h2h"])
        except Exception:
            return {}

    def _soccer_fetcher(self) -> SoccerFetcher:
        if self._soccer_fetcher_instance is None:
            self._soccer_fetcher_instance = SoccerFetcher()
        return self._soccer_fetcher_instance

    def _soccer_standings_context(self, candidate: dict[str, Any], payload: dict[str, Any], *, home_team: str, away_team: str) -> dict[str, Any]:
        competition = self._soccer_competition_code(candidate, payload)
        if not competition:
            return {}
        fetcher = self._soccer_fetcher()
        if not fetcher._api_key:
            return {}
        try:
            standings = fetcher.fetch_standings(competition=competition)
        except Exception:
            return {}
        if standings.empty:
            return {}
        teams = set(standings["team_name"].dropna().astype(str).unique())
        alias_map = build_entity_alias_map(teams)
        resolved_home = resolve_canonical_name(home_team, teams, alias_map=alias_map)
        resolved_away = resolve_canonical_name(away_team, teams, alias_map=alias_map)
        home_rows = standings.loc[standings["team_name"].astype(str) == resolved_home]
        away_rows = standings.loc[standings["team_name"].astype(str) == resolved_away]
        if home_rows.empty or away_rows.empty:
            return {}
        home_row = home_rows.iloc[0]
        away_row = away_rows.iloc[0]
        total_teams = int(len(standings))
        max_points = float(pd.to_numeric(standings["points"], errors="coerce").max() or 0.0)
        relegation_cut = max(total_teams - 3, 1)
        home_points = float(home_row.get("points") or 0.0)
        away_points = float(away_row.get("points") or 0.0)
        home_pos = int(home_row.get("position") or 0)
        away_pos = int(away_row.get("position") or 0)
        context: dict[str, Any] = {
            "standings_source": "football_data",
            "standings_checked": True,
            "home_position": home_pos,
            "away_position": away_pos,
            "home_points": home_points,
            "away_points": away_points,
            "title_context": int((home_pos <= 3 and max_points - home_points <= 6) or (away_pos <= 3 and max_points - away_points <= 6)),
            "relegation_context": int((home_pos >= relegation_cut and abs(home_points - away_points) <= 10) or (away_pos >= relegation_cut and abs(home_points - away_points) <= 10)),
            "motivation_checked": 1,
        }
        if context["title_context"] or context["relegation_context"]:
            context["playoff_motivation"] = 1
        if total_teams >= 18:
            safe_midtable = range(7, max(relegation_cut - 1, 8))
            if home_pos in safe_midtable and away_pos in safe_midtable and abs(home_points - max_points) > 12:
                context["nothing_to_play_for"] = 1
        return context

    def _fetch_mlb_enrichment_payload(self, candidate: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        details = dict(_DEFAULT_MLB_DETAILS)
        try:
            home_team = str(candidate.get("home") or "").strip()
            away_team = str(candidate.get("away") or "").strip()
            commence = candidate.get("commence") or candidate.get("commence_time")
            explicit_status = str(
                candidate.get("match_status")
                or candidate.get("game_status")
                or candidate.get("status")
                or "scheduled"
            ).strip().lower()
            if not home_team or not away_team:
                return {}, details

            payload: dict[str, Any] = {
                "sources_found": [_MLB_CACHE_SOURCE, "bookmaker"],
                "scraped_context_sources": [_MLB_CACHE_SOURCE, "bookmaker"],
                "scraped_context": {
                    "home_team_name": home_team,
                    "away_team_name": away_team,
                },
            }

            details["fixture_status"] = explicit_status or "scheduled"
            if explicit_status in {"final", "finished", "postponed", "cancelled", "canceled", "suspended"}:
                payload["status"] = explicit_status

            snapshot = self._mlb_snapshot(home_team, away_team)
            game = {
                "home_team": home_team,
                "away_team": away_team,
                "home": home_team,
                "away": away_team,
                "commence_time": commence,
                "status": explicit_status or "scheduled",
            }

            availability_context = build_availability_context("mlb", game, snapshot) if snapshot is not None else build_availability_context("mlb", game, None)
            environment_context = build_environment_context("mlb", home_team, away_team, commence)
            payload["scraped_context"] = _deep_merge_dict(payload["scraped_context"], availability_context)
            payload["scraped_context"] = _deep_merge_dict(payload["scraped_context"], environment_context)

            sources_found = [_MLB_CACHE_SOURCE, "bookmaker"]
            availability_source = str(availability_context.get("availability_source") or "").strip()
            lineup_source = str(availability_context.get("lineup_source") or "").strip()
            weather_source = str(environment_context.get("outdoor_weather_source") or "").strip()
            for source in (availability_source, lineup_source, weather_source):
                if source:
                    sources_found.append(source)
            payload["sources_found"] = sources_found
            payload["scraped_context_sources"] = sources_found

            prediction_factors: list[dict[str, Any]] = []
            context_adjustments: list[dict[str, Any]] = []
            highlights: list[str] = []

            context = payload["scraped_context"]
            home_pitcher = str(context.get("home_starter_name") or "").strip()
            away_pitcher = str(context.get("away_starter_name") or "").strip()
            details["home_pitcher"] = home_pitcher
            details["away_pitcher"] = away_pitcher
            home_confirmed = int(bool(context.get("home_starter_confirmed")))
            away_confirmed = int(bool(context.get("away_starter_confirmed")))
            if home_confirmed and away_confirmed and home_pitcher and away_pitcher:
                details["probable_pitcher_status"] = "confirmed"
                highlights.append(f"Probable starters confirmed: {home_pitcher} vs {away_pitcher}.")
            elif home_confirmed or away_confirmed:
                details["probable_pitcher_status"] = "partial"
            else:
                details["probable_pitcher_status"] = "missing"

            previous_context = candidate.get("scraped_context") or {}
            prior_home_pitcher = str(previous_context.get("home_starter_name") or "").strip()
            prior_away_pitcher = str(previous_context.get("away_starter_name") or "").strip()
            pitcher_changed = bool(
                (prior_home_pitcher and home_pitcher and prior_home_pitcher != home_pitcher)
                or (prior_away_pitcher and away_pitcher and prior_away_pitcher != away_pitcher)
                or context.get("home_pitcher_changed")
                or context.get("away_pitcher_changed")
            )
            details["pitcher_change_status"] = "changed" if pitcher_changed else ("stable" if home_pitcher or away_pitcher else "unknown")
            if pitcher_changed:
                payload["context_referee_decision"] = "REVIEW"
                payload["context_referee_reason"] = "Probable starter changed during MLB evidence enrichment."

            hand_known = bool(context.get("home_starter_hand")) and bool(context.get("away_starter_hand"))
            details["pitcher_handedness_status"] = "checked" if hand_known else "missing"

            if snapshot is not None and not snapshot.empty:
                self._apply_mlb_snapshot_context(
                    snapshot=snapshot,
                    payload=payload,
                    prediction_factors=prediction_factors,
                    context_adjustments=context_adjustments,
                    details=details,
                    candidate=candidate,
                )

            likely_home = int(_as_float(context.get("home_likely_starters_count")) or 0)
            likely_away = int(_as_float(context.get("away_likely_starters_count")) or 0)
            lineup_confirmed = bool(context.get("home_lineup_confirmed")) and bool(context.get("away_lineup_confirmed"))
            if lineup_confirmed:
                details["lineup_status"] = "confirmed"
                highlights.append("Both MLB lineups are confirmed from the latest enrichment payload.")
            elif likely_home or likely_away or lineup_source:
                details["lineup_status"] = "projected"
            else:
                details["lineup_status"] = "missing"

            injuries_known = any(
                (_as_float(context.get(key)) or 0.0) > 0.0
                for key in (
                    "home_injuries_count",
                    "away_injuries_count",
                    "home_priority_absences_count",
                    "away_priority_absences_count",
                    "home_questionable_count",
                    "away_questionable_count",
                )
            )
            details["injury_status"] = "checked" if availability_source else "missing"
            if injuries_known:
                highlights.append("Injury or absence counts were present in the latest MLB availability context.")

            weather_risk = int(_as_float(context.get("weather_risk")) or 0)
            details["weather_status"] = "checked" if weather_source else ("not_material" if str(candidate.get("market", "")).lower() == "moneyline" else "missing")
            if weather_source:
                details["park_factor_status"] = "checked_proxy" if any(key in context for key in ("temperature_f", "wind_mph", "precip_mm")) else "checked"
                if weather_risk:
                    highlights.append("Outdoor weather risk is material for this MLB matchup.")
            else:
                details["park_factor_status"] = "missing"

            home_line = _coerce_candidate_line(candidate)
            market = str(candidate.get("market", "") or "").strip().lower()
            market_fit_status = "acceptable"
            if market == "spreads":
                if home_line and home_line < 0 and float(candidate.get("odds") or 0) >= 2.0:
                    market_fit_status = "aggressive_underdog_run_line"
                    payload["context_referee_decision"] = payload.get("context_referee_decision") or "VETO"
                    payload["context_referee_reason"] = payload.get("context_referee_reason") or "Aggressive MLB underdog -1.5 run-line lacks enough support."
                else:
                    market_fit_status = "needs_stronger_support"
            elif market == "totals":
                market_fit_status = "weather_sensitive"
            details["market_fit_status"] = market_fit_status

            if prediction_factors:
                payload["prediction_factors"] = prediction_factors
            if context_adjustments:
                payload["context_adjustments"] = context_adjustments
            if highlights:
                payload["scraped_context_highlights"] = highlights
            return payload, details
        except Exception:
            return {}, details

    @staticmethod
    @lru_cache(maxsize=2)
    def _load_tennis_cache(sport: str) -> pd.DataFrame:
        sport_key = "tennis_wta" if sport == "tennis_wta" else "tennis"
        filename = "tennis_wta_features.parquet" if sport_key == "tennis_wta" else "tennis_features.parquet"
        path = Path(__file__).resolve().parent.parent.parent / "data" / "cache" / filename
        if not path.exists():
            return pd.DataFrame()
        frame = pd.read_parquet(path)
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
        return frame

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_mlb_cache() -> pd.DataFrame:
        path = Path(__file__).resolve().parent.parent.parent / "data" / "cache" / "mlb_features.parquet"
        if not path.exists():
            return pd.DataFrame()
        frame = pd.read_parquet(path)
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
        return frame

    def _mlb_snapshot(self, home_team: str, away_team: str) -> pd.Series | None:
        df = self._load_mlb_cache()
        if df.empty:
            return None
        all_teams = set(df.get("home_team", pd.Series(dtype=object)).dropna().astype(str).unique()) | set(df.get("away_team", pd.Series(dtype=object)).dropna().astype(str).unique())
        alias_map = build_entity_alias_map(all_teams)
        resolved_home = resolve_canonical_name(home_team, all_teams, alias_map=alias_map)
        resolved_away = resolve_canonical_name(away_team, all_teams, alias_map=alias_map)
        home_rows = df.loc[df.get("home_team", pd.Series(dtype=object)) == resolved_home]
        away_rows = df.loc[df.get("away_team", pd.Series(dtype=object)) == resolved_away]
        if home_rows.empty or away_rows.empty:
            return None
        home_snap = home_rows.sort_values("date").tail(10).mean(numeric_only=True)
        away_snap = away_rows.sort_values("date").tail(10).mean(numeric_only=True)
        combined = home_snap.copy()
        for col in away_snap.index:
            if str(col).startswith("away_"):
                combined[col] = away_snap[col]
        return combined

    def _apply_mlb_snapshot_context(
        self,
        *,
        snapshot: pd.Series,
        payload: dict[str, Any],
        prediction_factors: list[dict[str, Any]],
        context_adjustments: list[dict[str, Any]],
        details: dict[str, Any],
        candidate: dict[str, Any],
    ) -> None:
        context = payload["scraped_context"]
        home_rest_days = _as_float(snapshot.get("home_rest_days"))
        away_rest_days = _as_float(snapshot.get("away_rest_days"))
        away_travel_km = _as_float(snapshot.get("away_travel_km"))
        away_travel_tz_shift = _as_float(snapshot.get("away_travel_tz_shift"))
        home_density = _as_float(snapshot.get("home_games_L3D"))
        away_density = _as_float(snapshot.get("away_games_L3D"))
        if home_rest_days is not None:
            context["home_rest_days"] = round(home_rest_days, 2)
        if away_rest_days is not None:
            context["away_rest_days"] = round(away_rest_days, 2)
        if away_travel_km is not None:
            context["away_travel_km"] = round(away_travel_km, 1)
        if away_travel_tz_shift is not None:
            context["away_travel_tz_shift"] = round(away_travel_tz_shift, 2)
        if any(value is not None for value in (home_rest_days, away_rest_days, away_travel_km, away_travel_tz_shift)):
            details["travel_rest_status"] = "checked"
        else:
            details["travel_rest_status"] = "missing"

        bullpen_checked = False
        if home_density is not None or away_density is not None:
            context["bullpen_workload_checked"] = True
            context["home_games_L3D"] = int(home_density or 0)
            context["away_games_L3D"] = int(away_density or 0)
            context["bullpen_fatigue_risk"] = bool((home_density or 0) >= 2 or (away_density or 0) >= 2)
            context_adjustments.append(
                {
                    "name": "bullpen_workload",
                    "category": "schedule",
                    "summary": "Bullpen workload proxy checked from recent MLB schedule density.",
                    "value": round(float((home_density or 0) - (away_density or 0)), 3),
                }
            )
            bullpen_checked = True
        details["bullpen_status"] = "checked_proxy" if bullpen_checked else "missing"

        era_diff = _as_float(snapshot.get("sp_era_diff"))
        whip_diff = _as_float(snapshot.get("sp_whip_diff"))
        k9_diff = _as_float(snapshot.get("sp_k9_diff"))
        if era_diff is not None:
            context["sp_era_diff"] = round(era_diff, 3)
            prediction_factors.append(
                {
                    "name": "sp_era_diff",
                    "category": "lineup",
                    "summary": "Starting-pitcher ERA gap checked from the MLB feature cache.",
                    "value": round(era_diff, 4),
                }
            )
        if whip_diff is not None:
            context["sp_whip_diff"] = round(whip_diff, 3)
            prediction_factors.append(
                {
                    "name": "sp_whip_diff",
                    "category": "lineup",
                    "summary": "Starting-pitcher WHIP gap checked from the MLB feature cache.",
                    "value": round(whip_diff, 4),
                }
            )
        if k9_diff is not None:
            context["sp_k9_diff"] = round(k9_diff, 3)
            prediction_factors.append(
                {
                    "name": "sp_k9_diff",
                    "category": "matchup",
                    "summary": "Starting-pitcher strikeout profile checked from the MLB feature cache.",
                    "value": round(k9_diff, 4),
                }
            )
        if away_travel_km is not None or away_travel_tz_shift is not None:
            context_adjustments.append(
                {
                    "name": "travel_fatigue",
                    "category": "schedule",
                    "summary": "Travel context checked from the MLB feature cache.",
                    "value": round(float((away_travel_km or 0.0) / 1000.0), 3),
                }
            )
        if home_rest_days is not None and away_rest_days is not None:
            context_adjustments.append(
                {
                    "name": "rest_advantage",
                    "category": "schedule",
                    "summary": "Rest context checked from recent MLB schedule spacing.",
                    "value": round(home_rest_days - away_rest_days, 3),
                }
            )
        if "park_factor_proxy" not in context:
            home_wpct = _as_float(snapshot.get("home_home_wpct_20"))
            away_wpct = _as_float(snapshot.get("away_away_wpct_20"))
            if home_wpct is not None and away_wpct is not None:
                context["park_factor_proxy"] = round(home_wpct - away_wpct, 4)
                details["park_factor_status"] = "checked_proxy"
        if _candidate_weather_material(candidate):
            details["weather_status"] = details["weather_status"] or "missing"


    @staticmethod
    def _player_history(df: pd.DataFrame, player_name: str) -> pd.DataFrame:
        p1_rows = df.loc[df.get("player1_name", pd.Series(dtype=object)) == player_name]
        p2_rows = EvidenceEnrichmentPass._swap_player_frame(df.loc[df.get("player2_name", pd.Series(dtype=object)) == player_name])
        rows = pd.concat([p1_rows, p2_rows], ignore_index=False) if not p2_rows.empty else p1_rows.copy()
        if rows.empty:
            return rows
        return rows.sort_values("date")

    @staticmethod
    def _swap_player_frame(rows: pd.DataFrame) -> pd.DataFrame:
        if rows.empty:
            return rows.copy()
        rename_map: dict[str, str] = {}
        for col in rows.columns:
            if col.startswith("player1_"):
                rename_map[col] = f"player2_{col[8:]}"
            elif col.startswith("player2_"):
                rename_map[col] = f"player1_{col[8:]}"
            elif col.startswith("p1_"):
                rename_map[col] = f"p2_{col[3:]}"
            elif col.startswith("p2_"):
                rename_map[col] = f"p1_{col[3:]}"
            else:
                rename_map[col] = col
        return rows.rename(columns=rename_map)

    @staticmethod
    def _head_to_head_rows(df: pd.DataFrame, player1: str, player2: str) -> pd.DataFrame:
        mask = (
            ((df.get("player1_name", pd.Series(dtype=object)) == player1) & (df.get("player2_name", pd.Series(dtype=object)) == player2))
            | ((df.get("player1_name", pd.Series(dtype=object)) == player2) & (df.get("player2_name", pd.Series(dtype=object)) == player1))
        )
        rows = df.loc[mask]
        if rows.empty:
            return rows
        return rows.sort_values("date").tail(10)

    @staticmethod
    def _latest_matchup_row(df: pd.DataFrame, player1: str, player2: str) -> pd.Series | None:
        rows = EvidenceEnrichmentPass._head_to_head_rows(df, player1, player2)
        if rows.empty:
            return None
        return rows.iloc[-1]

    @staticmethod
    def _surface_for_candidate(candidate: dict[str, Any], p1_rows: pd.DataFrame, p2_rows: pd.DataFrame) -> str:
        text = " ".join(
            [
                str(candidate.get("league_key") or ""),
                str(candidate.get("league") or ""),
                str(candidate.get("tournament") or ""),
            ]
        ).lower()
        for surface, hints in _SURFACE_HINTS.items():
            if any(hint in text for hint in hints):
                return surface
        for rows in (p1_rows, p2_rows):
            if "surface" in rows.columns:
                latest = rows.get("surface", pd.Series(dtype=object)).dropna()
                if not latest.empty:
                    return str(latest.iloc[-1])
        return ""

    def _apply_payload(self, candidate: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        if not payload:
            return deepcopy(candidate)
        updated = deepcopy(candidate)
        list_keys = {
            "scraped_context_sources",
            "scraped_context_highlights",
            "context_adjustments",
            "prediction_factors",
        }
        dict_keys = {"scraped_context", "true_probability"}
        for key, value in payload.items():
            if key in {"sources_found"}:
                continue
            if key in dict_keys and isinstance(value, dict):
                updated[key] = _deep_merge_dict(dict(updated.get(key) or {}), value)
            elif key in list_keys and isinstance(value, list):
                updated[key] = _merge_lists(updated.get(key) or [], value)
            else:
                updated[key] = value
        return updated
