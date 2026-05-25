from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

_ODDS_STALE_THRESHOLD_HOURS = 24.0


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _normalize_name(value: str) -> str:
    return " ".join(str(value or "").lower().replace(".", " ").split())


@dataclass
class FreshnessAudit:
    match_status: str
    fixture_verified: bool
    fixture_verification_reason: str
    odds_freshness: str
    lineup_freshness: str
    injury_news_freshness: str
    standings_freshness: str
    review_reason: str
    suppression_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _match_status(candidate: dict[str, Any], now: datetime) -> str:
    explicit = str(
        candidate.get("match_status")
        or candidate.get("game_status")
        or candidate.get("status")
        or ""
    ).strip().lower()
    if explicit in {"live", "in_play", "in-play", "inplay"}:
        return "live"
    if explicit in {"finished", "final", "completed", "closed"}:
        return "finished"
    if explicit in {"pre_match", "prematch", "scheduled", "not_started"}:
        return "pre_match"

    commence = _parse_dt(candidate.get("commence") or candidate.get("commence_time"))
    if commence is None:
        return "unknown"
    if now >= commence and (now - commence).total_seconds() >= 4 * 3600:
        return "finished"
    if now >= commence:
        return "live"
    return "pre_match"


def _fixture_verification(candidate: dict[str, Any]) -> tuple[bool, str]:
    context = candidate.get("scraped_context") or {}
    home_expected = _normalize_name(candidate.get("home", ""))
    away_expected = _normalize_name(candidate.get("away", ""))
    home_context = _normalize_name(context.get("home_team_name", ""))
    away_context = _normalize_name(context.get("away_team_name", ""))
    if not home_context and not away_context:
        return True, ""
    if home_context and home_expected and home_context != home_expected:
        return False, "fixture verification mismatch: home team differs between the odds board and context sources"
    if away_context and away_expected and away_context != away_expected:
        return False, "fixture verification mismatch: away team differs between the odds board and context sources"
    return True, ""


def _odds_freshness(candidate: dict[str, Any]) -> tuple[str, str]:
    if candidate.get("stale_line"):
        return "stale", "odds snapshot is stale relative to the wider market"
    source_status = str(candidate.get("odds_source_status", "") or "").lower()
    source_detail = str(candidate.get("odds_source_detail", "") or "").lower()
    age_hours = _as_float(candidate.get("odds_snapshot_age_hours"))
    bookmaker_last_update = _parse_dt(candidate.get("bookmaker_last_update"))
    if "stale" in source_status or "stale" in source_detail or "fallback" in source_detail:
        return "stale", "odds snapshot came from a stale fallback cache and needs a refresh"
    if age_hours is not None and age_hours > _ODDS_STALE_THRESHOLD_HOURS:
        return "stale", "odds snapshot is too old to trust for a new recommendation"
    if bookmaker_last_update is not None and age_hours is not None:
        return "fresh", ""
    if source_status == "live_api" and bookmaker_last_update is None:
        return "unknown", ""
    if age_hours is not None and source_status in {"disk_cache", "in_memory_cache", "offline_disk_cache", "disk_empty_cache"}:
        return "unknown", ""
    return "unknown", ""


def _lineup_freshness(candidate: dict[str, Any], lead_hours: float | None) -> tuple[str, str]:
    sport = str(candidate.get("sport", "") or "").lower()
    context = candidate.get("scraped_context") or {}
    source = str(context.get("lineup_source") or context.get("availability_source") or candidate.get("lineup_source") or candidate.get("availability_source") or "").strip().lower()
    fetched_at = _parse_dt(context.get("availability_fetched_at") or candidate.get("availability_fetched_at"))
    home_lineup_confirmed = int(context.get("home_lineup_confirmed", 0) or 0)
    away_lineup_confirmed = int(context.get("away_lineup_confirmed", 0) or 0)
    home_starters = int(context.get("home_likely_starters_count", 0) or 0)
    away_starters = int(context.get("away_likely_starters_count", 0) or 0)
    home_starter_confirmed = context.get("home_starter_confirmed")
    away_starter_confirmed = context.get("away_starter_confirmed")
    home_goalie_confirmed = context.get("home_goalie_confirmed")
    away_goalie_confirmed = context.get("away_goalie_confirmed")

    if sport == "soccer" and lead_hours is not None and lead_hours <= 2:
        if (home_lineup_confirmed or away_lineup_confirmed or home_starters or away_starters):
            return "fresh", ""
        return "missing", "lineup freshness check failed: close to kickoff but starting XIs are not posted yet"

    if sport == "mlb" and lead_hours is not None and lead_hours <= 6:
        if home_starter_confirmed is not None and away_starter_confirmed is not None:
            if int(bool(home_starter_confirmed)) and int(bool(away_starter_confirmed)):
                return "fresh", ""
        return "missing", "lineup freshness check failed: probable starters are not fully confirmed yet"

    if sport == "nhl" and lead_hours is not None and lead_hours <= 6:
        if home_goalie_confirmed is not None and away_goalie_confirmed is not None:
            if int(bool(home_goalie_confirmed)) and int(bool(away_goalie_confirmed)):
                return "fresh", ""
        return "missing", "lineup freshness check failed: probable goalies are not fully confirmed yet"

    if sport == "basketball" and lead_hours is not None and lead_hours <= 2:
        if source in {"api_sports_basketball", "balldontlie", "espn", "rotowire", "mysportsfeeds"}:
            if fetched_at is not None:
                age_hours = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 3600.0
                if age_hours > 8:
                    return "missing", "lineup freshness check failed: close to tip-off but the final injury/inactive report is too old"
            return "monitor", "lineup freshness check: final injury/inactive report is present, but verify late scratch risk near tip-off"
        return "missing", "lineup freshness check failed: close to tip-off but no concrete injury/inactive report is attached"

    return "unknown", ""


def _injury_news_freshness(candidate: dict[str, Any], lead_hours: float | None) -> tuple[str, str]:
    context = candidate.get("scraped_context") or {}
    source = str(context.get("availability_source") or candidate.get("availability_source") or "").strip().lower()
    fetched_at = _parse_dt(context.get("availability_fetched_at") or candidate.get("availability_fetched_at"))

    if lead_hours is not None and lead_hours <= 3:
        if source in {"", "feature_snapshot"}:
            return "stale", "injury/news freshness check failed: only snapshot availability data is present near kickoff"
        if fetched_at is not None:
            age_hours = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 3600.0
            if age_hours > 8:
                return "stale", "injury/news freshness check failed: availability context is too old"
        return "fresh", ""

    if source:
        return "fresh", ""
    return "unknown", ""


def _standings_freshness(candidate: dict[str, Any]) -> tuple[str, str]:
    age_hours = _as_float(candidate.get("standings_snapshot_age_hours"))
    if age_hours is None:
        return "unknown", ""
    if age_hours > 36:
        return "stale", "standings/model snapshot is older than the allowed freshness window"
    return "fresh", ""


def audit_candidate_freshness(candidate: dict[str, Any], *, now: datetime | None = None) -> FreshnessAudit:
    now = now or datetime.now(timezone.utc)
    match_status = _match_status(candidate, now)
    fixture_verified, fixture_reason = _fixture_verification(candidate)

    commence = _parse_dt(candidate.get("commence") or candidate.get("commence_time"))
    lead_hours = ((commence - now).total_seconds() / 3600.0) if commence is not None else None

    odds_freshness, odds_reason = _odds_freshness(candidate)
    lineup_freshness, lineup_reason = _lineup_freshness(candidate, lead_hours)
    injury_news_freshness, injury_reason = _injury_news_freshness(candidate, lead_hours)
    standings_freshness, standings_reason = _standings_freshness(candidate)

    suppression_reason = ""
    review_reason = ""

    if match_status == "finished":
        suppression_reason = "match status check failed: fixture is already finished"
    elif match_status == "live":
        suppression_reason = "match status check failed: fixture is already live, so pre-match pricing is no longer valid"
    elif not fixture_verified:
        review_reason = fixture_reason
    elif odds_freshness == "stale":
        review_reason = odds_reason
    elif lineup_freshness == "missing":
        review_reason = lineup_reason
    elif injury_news_freshness == "stale":
        review_reason = injury_reason
    elif standings_freshness == "stale":
        review_reason = standings_reason
    elif lineup_freshness == "monitor":
        review_reason = lineup_reason

    return FreshnessAudit(
        match_status=match_status,
        fixture_verified=fixture_verified,
        fixture_verification_reason=fixture_reason,
        odds_freshness=odds_freshness,
        lineup_freshness=lineup_freshness,
        injury_news_freshness=injury_news_freshness,
        standings_freshness=standings_freshness,
        review_reason=review_reason,
        suppression_reason=suppression_reason,
    )
