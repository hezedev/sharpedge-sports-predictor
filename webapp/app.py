"""
webapp/app.py
=============
Flask web interface for Sports Predictor.
Run with:  python webapp/app.py
Then open: http://localhost:5000
"""

import json
import math
import os
import re
import subprocess
import sys
import threading
import calendar
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo
from collections import defaultdict
from typing import Any, Optional

import pandas as pd
import numpy as np
import requests
from flask import Flask, jsonify, render_template, request, stream_with_context, Response
from dotenv import load_dotenv, set_key

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))
load_dotenv(BASE / ".env", override=True)

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


@app.after_request
def _disable_local_web_cache(response: Response) -> Response:
    path = str(request.path or "")
    if path == "/" or path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

from src.analysis import ManualGameAnalyst
from src.analysis.news_context import collect_matchup_news_context
from src.analysis.schemas import AnalysisReport, AnalysisSignal, SourceNote
from src.markets.decision_layer import classify_candidate_decision
from src.markets.policy import annotate_bet, summarize_focused_prediction_policy, summarize_market_policy, get_market_policy
from src.data.source_registry import source_status_summary
from src.risk.parlay_builder import ParlayBuilder, ParlayLeg
from src.models.artifacts import calibrator_path_for_tag, get_current_model_tag
from src.utils.odds_quota import get_odds_budget_status, get_primary_odds_api_key, parse_odds_api_keys_from_env
from src.utils.results_tracker import (
    compute_summary,
    get_settled,
    daily_pnl as _daily_pnl,
    mistake_report as _mistake_report,
    parlay_breakdown as _parlay_breakdown,
)
from src.utils.sport_registry import enrich_with_capability, get_capability_profile

# ── Hybrid Quota Bridge ────────────────────────────────────────────────────────
try:
    from src.utils.quota_api_bridge import QuotaAPIBridge
    quota_bridge = QuotaAPIBridge()
except ImportError:
    quota_bridge = None

# ── Scan state (in-memory, single user) ───────────────────────────────────────
_scan_running = False
_scan_log     = []
_scan_proc    = None   # subprocess.Popen handle so we can kill it
_last_results_settle_report: dict[str, Any] = {}

_reasoning_progress_lock = threading.Lock()
_reasoning_progress = {
    "running": False,
    "mode": "",
    "candidate_id": "",
    "stage": "",
    "log": [],
    "updated_at": "",
}

_APP_TZ = ZoneInfo("Europe/Vienna")
_SPORT_LIVE_WINDOWS = {
    "soccer": timedelta(hours=2, minutes=45),
    "basketball": timedelta(hours=3),
    "mlb": timedelta(hours=4, minutes=30),
    "nhl": timedelta(hours=3),
    "tennis": timedelta(hours=6),
    "tennis_wta": timedelta(hours=6),
}


def _annotate_bets(bets: list[dict]) -> list[dict]:
    annotated: list[dict] = []
    for bet in bets:
        enriched = enrich_with_capability(annotate_bet(bet))
        enriched.update(_committee_blocker_fields(enriched))
        annotated.append(enriched)
    return annotated


def _sport_scan_counts_from_summary(summary: dict[str, Any]) -> dict[str, int]:
    diagnostics = ((summary or {}).get("sport_pipeline_diagnostics") or {}).get("by_sport") or {}
    if isinstance(diagnostics, dict) and diagnostics:
        counts = {}
        for sport, row in diagnostics.items():
            try:
                counts[str(sport)] = int((row or {}).get("scanned_games", 0) or 0)
            except Exception:
                continue
        if counts:
            return counts

    counts: dict[str, int] = {}
    for game in summary.get("soccer_games", []) or []:
        counts["soccer"] = counts.get("soccer", 0) + 1
    for game in summary.get("other_games", []) or []:
        sport = str((game or {}).get("sport", "") or "").strip().lower()
        if not sport:
            continue
        counts[sport] = counts.get(sport, 0) + 1

    # Fall back to bet-derived coverage when a summary does not include full game lists.
    if not counts:
        for bucket in ("bets", "review_bets", "suppressed_bets"):
            for bet in ((summary.get("single_bets", {}) or {}).get(bucket, []) or []):
                sport = str((bet or {}).get("sport", "") or "").strip().lower()
                if not sport:
                    continue
                counts[sport] = max(1, counts.get(sport, 0))
    return counts


def _sport_funnel_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    diagnostics = (summary or {}).get("sport_pipeline_diagnostics") or {}
    by_sport = diagnostics.get("by_sport") or {}
    if isinstance(by_sport, dict) and by_sport:
        return by_sport
    return {}


def _market_coverage_from_summary(summary: dict[str, Any]) -> dict[str, list[str]]:
    by_sport: dict[str, list[str]] = {}

    def _add(sport: object, keys: object) -> None:
        sport_key = str(sport or "").strip().lower()
        if not sport_key:
            return
        row = by_sport.setdefault(sport_key, [])
        if isinstance(keys, (list, tuple)):
            for item in keys:
                key = str(item or "").strip().lower()
                if key and key not in row:
                    row.append(key)

    for game in (summary.get("soccer_games") or []):
        if not isinstance(game, dict):
            continue
        keys = game.get("available_market_keys")
        if not keys:
            inferred: list[str] = []
            labels = {str((outcome or {}).get("label") or "").strip().lower() for outcome in (game.get("outcomes") or []) if isinstance(outcome, dict)}
            if {"home win", "draw", "away win"} & labels:
                inferred.append("h2h")
            if {"home or draw", "away or draw"} & labels:
                inferred.append("double_chance")
            keys = inferred
        _add(game.get("sport") or "soccer", keys)

    for game in (summary.get("other_games") or []):
        if not isinstance(game, dict):
            continue
        _add(game.get("sport"), game.get("available_market_keys"))

    for bucket in ("bets", "review_bets", "suppressed_bets"):
        for bet in ((summary.get("single_bets", {}) or {}).get(bucket, []) or []):
            if not isinstance(bet, dict):
                continue
            _add(bet.get("sport"), [bet.get("market")])

    return {sport: keys for sport, keys in by_sport.items() if keys}


def _as_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _committee_blocker_fields(item: dict[str, Any]) -> dict[str, Any]:
    committee = item.get("committee") or {}
    research = committee.get("research_mind") if isinstance(committee, dict) else {}
    enrichment = committee.get("evidence_enrichment") if isinstance(committee, dict) else {}
    if not isinstance(research, dict):
        research = {}
    if not isinstance(enrichment, dict):
        enrichment = {}
    if not research:
        research = {
            "evidence_status": item.get("research_mind_evidence_status", ""),
            "source_count": item.get("research_mind_source_count"),
            "source_quality_summary": item.get("research_mind_source_quality_summary", ""),
            "lineup_status": item.get("research_mind_lineup_status", ""),
            "injury_status": item.get("research_mind_injury_status", ""),
            "motivation_status": item.get("research_mind_motivation_status", ""),
            "rotation_status": item.get("research_mind_rotation_status", ""),
            "missing_evidence": item.get("research_mind_missing_evidence", []),
            "sport_specific_missing_evidence": item.get("research_mind_sport_specific_missing_evidence", []),
        }
    if not enrichment:
        enrichment = dict(item.get("committee_enrichment") or {})

    def _pick_text(*values: Any) -> str:
        for value in values:
            text = str(value or "").strip()
            if text:
                return text
        return ""

    def _pick_list(*values: Any) -> list[str]:
        for value in values:
            items = _as_text_list(value)
            if items:
                return items
        return []

    def _is_missing(value: str, *, extra: set[str] | None = None) -> bool:
        key = str(value or "").strip().lower()
        missing = {"", "unknown", "missing", "not_checked", "not_found", "provider_failed", "missing_near_kickoff"}
        if extra:
            missing |= {str(item).strip().lower() for item in extra}
        return key in missing

    sport = str(item.get("sport", "") or "").strip().lower()
    evidence_status = _pick_text(research.get("evidence_status"), item.get("research_mind_evidence_status")).upper()
    source_quality = _pick_text(research.get("source_quality_summary"), item.get("research_mind_source_quality_summary")).lower()
    source_count = research.get("source_count", item.get("research_mind_source_count"))
    try:
        source_count_int = int(source_count) if source_count not in (None, "") else 0
    except (TypeError, ValueError):
        source_count_int = 0

    sources_found = _pick_list(enrichment.get("sources_found"), item.get("committee_enrichment_sources_found"))
    remaining_missing = _pick_list(enrichment.get("remaining_missing_evidence"), research.get("missing_evidence"), item.get("research_mind_missing_evidence"))
    sport_specific_missing = _pick_list(research.get("sport_specific_missing_evidence"), item.get("research_mind_sport_specific_missing_evidence"))
    veto_flags = _pick_list((committee.get("arbiter") or {}).get("veto_flags") if isinstance(committee, dict) else None, item.get("committee_veto_flags"))

    lineup_status = _pick_text(research.get("lineup_status"), item.get("research_mind_lineup_status"), enrichment.get("lineup_status"), enrichment.get("probable_lineup_status"))
    injury_status = _pick_text(research.get("injury_status"), item.get("research_mind_injury_status"), enrichment.get("injury_status"))
    motivation_status = _pick_text(research.get("motivation_status"), item.get("research_mind_motivation_status"), enrichment.get("motivation_status"))
    rotation_status = _pick_text(research.get("rotation_status"), item.get("research_mind_rotation_status"), enrichment.get("rotation_status"))
    fixture_congestion_status = _pick_text(enrichment.get("fixture_congestion_status"))

    blockers: list[str] = []
    if evidence_status == "INSUFFICIENT":
        blockers.append("Evidence is still insufficient for publication.")
    elif evidence_status == "PARTIAL":
        blockers.append("Evidence improved, but sport-critical context is still incomplete.")

    if source_quality == "weak" or (sources_found and set(sources_found) <= {"bookmaker", "odds_snapshot", "bookmaker_or_odds_feed"}):
        blockers.append("Only bookmaker or weak-source evidence is currently available.")
    elif source_count_int and source_count_int < 2:
        blockers.append("Too few independent evidence sources have been confirmed.")

    if sport == "soccer":
        if _is_missing(injury_status):
            blockers.append("Injury and team-news context is still not properly checked.")
        if _is_missing(lineup_status):
            blockers.append("Lineup or probable XI context is still unclear.")
        if _is_missing(rotation_status, extra={"not_required"}):
            blockers.append("Rotation risk has not been cleared yet.")
        if _is_missing(motivation_status, extra={"not_required"}):
            blockers.append("Motivation or end-of-season context is still missing.")
        if _is_missing(fixture_congestion_status, extra={"not_required"}):
            blockers.append("Fixture congestion or cup/continental context is still missing.")

    for value in sport_specific_missing + remaining_missing:
        if value not in blockers and len(blockers) < 6:
            blockers.append(value)

    if "MISSING_SPORT_CRITICAL_EVIDENCE" in veto_flags:
        blockers.append("Arbiter still sees missing sport-critical evidence.")
    if "WAIT_FOR_LINEUPS" in str(item.get("committee_final_decision", "")).upper().replace(" ", "_"):
        blockers.append("This pick is waiting for lineups before it can be reconsidered.")

    unique_blockers: list[str] = []
    seen: set[str] = set()
    for blocker in blockers:
        text = str(blocker or "").strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        unique_blockers.append(text)

    summary = ""
    if unique_blockers:
        summary = unique_blockers[0]
        if len(unique_blockers) > 1:
            summary = f"{summary} Next blocker: {unique_blockers[1]}"

    return {
        "committee_blockers": unique_blockers,
        "committee_blocker_summary": summary,
    }


def _set_reasoning_progress(
    stage: str,
    *,
    running: bool | None = None,
    mode: str | None = None,
    candidate_id: str | None = None,
) -> None:
    with _reasoning_progress_lock:
        if running is not None:
            _reasoning_progress["running"] = bool(running)
        if mode is not None:
            _reasoning_progress["mode"] = mode
        if candidate_id is not None:
            _reasoning_progress["candidate_id"] = candidate_id
        _reasoning_progress["stage"] = stage
        _reasoning_progress["updated_at"] = datetime.now(timezone.utc).isoformat()
        if stage:
            log = list(_reasoning_progress.get("log") or [])
            if not log or log[-1] != stage:
                log.append(stage)
            _reasoning_progress["log"] = log[-12:]


def _reset_reasoning_progress() -> None:
    with _reasoning_progress_lock:
        _reasoning_progress.update({
            "running": False,
            "mode": "",
            "candidate_id": "",
            "stage": "",
            "log": [],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })


def _parse_event_dt(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _event_local_date_str(value: object) -> str:
    dt_utc = _parse_event_dt(value)
    if dt_utc is None:
        return ""
    return dt_utc.astimezone(_APP_TZ).date().isoformat()


def _live_window_for_sport(sport: object) -> timedelta:
    return _SPORT_LIVE_WINDOWS.get(str(sport or "").lower(), timedelta(hours=3))


def _timing_bucket_for_local_date(event_local: datetime, now_local: datetime) -> str:
    day_delta = (event_local.date() - now_local.date()).days
    if day_delta < 0:
        return "past"
    if day_delta == 0:
        return "today"
    if day_delta == 1:
        return "tomorrow"
    if day_delta == 2:
        return "day_after"
    return "upcoming"


def _scheduled_label_for_local_date(event_local: datetime, now_local: datetime) -> str:
    bucket = _timing_bucket_for_local_date(event_local, now_local)
    if bucket == "today":
        return f"Today {event_local.strftime('%H:%M')}"
    if bucket == "tomorrow":
        return f"Tomorrow {event_local.strftime('%H:%M')}"
    if bucket == "day_after":
        return f"Day After {event_local.strftime('%H:%M')}"
    return event_local.strftime("%a %d %b %H:%M")


def _derive_event_timing(commence: object, sport: object = "") -> dict:
    dt_utc = _parse_event_dt(commence)
    now_utc = datetime.now(timezone.utc)
    now_local = now_utc.astimezone(_APP_TZ)
    if dt_utc is None:
        return {
            "status": "unknown",
            "status_label": "Unknown",
            "window": "upcoming",
            "kick_off": "",
            "time_label": "",
            "commence_local": "",
        }
    event_local = dt_utc.astimezone(_APP_TZ)
    live_window = _live_window_for_sport(sport)
    if dt_utc <= now_utc < dt_utc + live_window:
        status = "live"
        status_label = "LIVE"
    elif now_utc >= dt_utc + live_window:
        status = "played"
        status_label = "PLAYED"
    else:
        status = "scheduled"
        status_label = "SCHEDULED"
    window = _timing_bucket_for_local_date(event_local, now_local)
    time_label = _scheduled_label_for_local_date(event_local, now_local)
    kick_off = status_label if status in {"live", "played"} else time_label
    return {
        "status": status,
        "status_label": status_label,
        "window": window,
        "kick_off": kick_off,
        "time_label": time_label,
        "commence_local": event_local.isoformat(),
    }


def _apply_timing_to_game(game: dict, sport: object = "") -> dict:
    enriched = dict(game)
    timing = _derive_event_timing(
        enriched.get("commence") or enriched.get("commence_time") or enriched.get("kick_off"),
        sport or enriched.get("sport"),
    )
    if timing["status"] == "unknown":
        enriched.setdefault("status", "scheduled")
        enriched.setdefault("status_label", "SCHEDULED")
        enriched["window"] = enriched.get("window") or timing["window"]
        enriched["kick_off"] = enriched.get("kick_off") or timing["kick_off"]
        enriched["time_label"] = enriched.get("time_label") or enriched.get("kick_off") or ""
        enriched["commence_local"] = enriched.get("commence_local") or ""
        return enriched
    enriched["window"] = timing["window"]
    enriched["status"] = timing["status"]
    enriched["status_label"] = timing["status_label"]
    enriched["kick_off"] = timing["kick_off"]
    enriched["time_label"] = timing["time_label"]
    enriched["commence_local"] = timing["commence_local"]
    return enriched


def _apply_timing_to_bet(bet: dict) -> dict:
    enriched = dict(bet)
    timing = _derive_event_timing(
        enriched.get("commence") or enriched.get("commence_time") or enriched.get("kick_off"),
        enriched.get("sport"),
    )
    if timing["status"] == "unknown":
        enriched.setdefault("status", "scheduled")
        enriched.setdefault("status_label", "SCHEDULED")
        enriched["window"] = enriched.get("window") or timing["window"]
        enriched["kick_off"] = enriched.get("kick_off") or timing["kick_off"]
        enriched["time_label"] = enriched.get("time_label") or enriched.get("kick_off") or ""
        enriched["commence_local"] = enriched.get("commence_local") or ""
        return enriched
    enriched["window"] = timing["window"]
    enriched["status"] = timing["status"]
    enriched["status_label"] = timing["status_label"]
    enriched["kick_off"] = timing["kick_off"]
    enriched["time_label"] = timing["time_label"]
    enriched["commence_local"] = timing["commence_local"]
    return enriched


def _window_matches(item: dict, requested_window: str) -> bool:
    requested = str(requested_window or "").strip().lower()
    if not requested:
        return True
    item_window = str(item.get("window") or "").strip().lower()
    item_status = str(item.get("status") or "").strip().lower()
    if requested == "today":
        return item_window == "today" or item_status in {"live", "played"} or item_window == "past"
    return item_window == requested


def _apply_capability_to_game(game: dict) -> dict:
    enriched = enrich_with_capability(game)
    if enriched.get("review_only") and not enriched.get("abstain"):
        enriched["review_required"] = True
        enriched.setdefault("review_reason", enriched.get("launch_note") or "League remains review-only for launch.")
    if not enriched.get("decision_status"):
        decision_status, decision_reason = classify_candidate_decision(
            publish_ready=bool(enriched.get("publish_ready")),
            review_reason=str(enriched.get("review_reason", "") or ""),
            suppression_reason=str(enriched.get("suppression_reason", "") or ""),
        )
        enriched["decision_status"] = decision_status
        enriched["decision_reason"] = decision_reason
    return enriched


def _resolve_decision_fields(item: dict) -> dict:
    enriched = dict(item or {})
    if not enriched.get("decision_status"):
        decision_status, decision_reason = classify_candidate_decision(
            publish_ready=bool(enriched.get("publish_ready")),
            review_reason=str(enriched.get("review_reason", "") or ""),
            suppression_reason=str(enriched.get("suppression_reason", "") or ""),
        )
        enriched["decision_status"] = decision_status
        enriched["decision_reason"] = decision_reason
    enriched.setdefault("decision_reason", "")
    return enriched


def _reasoning_display_label(item: dict) -> str:
    decision_status = str(item.get("decision_status") or "").strip().upper()
    sport = str(item.get("sport", "")).upper()
    team = item.get("team", "Pick")
    home = item.get("home", "")
    away = item.get("away", "")
    prefix = f"{decision_status} · " if decision_status else ""
    return f"{prefix}{sport} · {team} · {home} vs {away}"


def _map_referee_decision_to_system(decision: str, reasoning: str = "") -> str:
    decision = str(decision or "").strip().upper()
    reasoning_text = str(reasoning or "").strip().lower()
    lineup_markers = ("lineup", "starter", "rotation", "availability", "goalie", "confirmed")
    if decision == "APPROVE":
        return "BET"
    if decision == "VETO":
        return "AVOID"
    if decision in {"REVIEW", "DATA_THIN"} and any(marker in reasoning_text for marker in lineup_markers):
        return "WAIT FOR LINEUPS"
    if decision in {"REVIEW", "DATA_THIN"}:
        return "HOLD"
    return ""


def _split_launch_safe_bets(bets: list[dict], review_bets: list[dict]) -> tuple[list[dict], list[dict]]:
    published: list[dict] = []
    review_queue: list[dict] = []
    for bet in review_bets:
        enriched = enrich_with_capability(bet)
        if not enriched.get("decision_status"):
            decision_status, decision_reason = classify_candidate_decision(
                publish_ready=False,
                review_reason=str(enriched.get("review_reason", "") or ""),
                suppression_reason=str(enriched.get("suppression_reason", "") or ""),
            )
            enriched["decision_status"] = decision_status
            enriched["decision_reason"] = decision_reason
        review_queue.append(enriched)
    for bet in bets:
        enriched = enrich_with_capability(bet)
        if enriched.get("publishable", True):
            if not enriched.get("decision_status"):
                decision_status, decision_reason = classify_candidate_decision(
                    publish_ready=True,
                    review_reason=str(enriched.get("review_reason", "") or ""),
                    suppression_reason=str(enriched.get("suppression_reason", "") or ""),
                )
                enriched["decision_status"] = decision_status
                enriched["decision_reason"] = decision_reason
            published.append(enriched)
            continue
        review_item = {
            **enriched,
            "review_required": True,
            "review_reason": enriched.get("review_reason")
            or enriched.get("launch_note")
            or "League remains review-only for launch.",
        }
        decision_status, decision_reason = classify_candidate_decision(
            publish_ready=False,
            review_reason=str(review_item.get("review_reason", "") or ""),
            suppression_reason="",
        )
        review_item["decision_status"] = decision_status
        review_item["decision_reason"] = decision_reason
        review_queue.append(review_item)
    return published, review_queue


def _json_safe(value):
    """Convert pandas/numpy missing values into strict JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    try:
        missing = pd.isna(value)
        if missing is True or (type(missing).__name__ == "bool_" and bool(missing)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return value


def _normalize_tracker_text(value: object) -> str:
    return " ".join(str(value or "").lower().replace("@", " vs ").split())


def _normalize_match_key(value: object) -> str:
    text = _normalize_tracker_text(value)
    parts = [part.strip() for part in text.replace(" @ ", " vs ").split(" vs ") if part.strip()]
    if len(parts) >= 2:
        return " vs ".join(sorted(parts[:2]))
    return text


def _prediction_lookup() -> dict[tuple[str, str, str], str]:
    tracker_dir = BASE / "data" / "tracker"
    frames = []
    for filename in ("predictions.parquet", "settled.parquet"):
        path = tracker_dir / filename
        if not path.exists() or path.stat().st_size < 100:
            continue
        try:
            frames.append(pd.read_parquet(path))
        except Exception:
            continue
    if not frames:
        return {}
    pred = pd.concat(frames, ignore_index=True)
    lookup: dict[tuple[str, str, str], str] = {}
    for _, row in pred.iterrows():
        key = (
            _normalize_tracker_text(row.get("sport")),
            _normalize_match_key(row.get("match_id")),
            _normalize_tracker_text(row.get("team_or_player")),
        )
        if key[0] and key[1] and key[2] and key not in lookup:
            lookup[key] = str(row.get("pred_id") or "")
    return lookup


def _attach_prediction_ids(bets: list[dict]) -> list[dict]:
    lookup = _prediction_lookup()
    if not lookup:
        return bets
    enriched = []
    for bet in bets:
        if bet.get("pred_id"):
            enriched.append(bet)
            continue
        match = f"{bet.get('home', '')} vs {bet.get('away', '')}"
        key = (
            _normalize_tracker_text(bet.get("sport")),
            _normalize_match_key(match),
            _normalize_tracker_text(bet.get("team")),
        )
        pred_id = lookup.get(key)
        enriched.append({**bet, "pred_id": pred_id} if pred_id else bet)
    return enriched


def _backfill_selection_prediction_ids(selections: list[dict]) -> bool:
    lookup = _prediction_lookup()
    if not lookup:
        return False
    changed = False
    for selection in selections:
        if selection.get("pred_id"):
            continue
        key = (
            _normalize_tracker_text(selection.get("sport")),
            _normalize_match_key(selection.get("match")),
            _normalize_tracker_text(selection.get("team")),
        )
        pred_id = lookup.get(key)
        if pred_id:
            selection["pred_id"] = pred_id
            changed = True
    return changed


def _today_summary() -> dict:
    summary, _, _, _ = _load_summary_report()
    return summary


def _summary_report_path(date_str: str) -> Path:
    return BASE / "reports" / f"summary_{date_str}.json"


def _available_summary_dates() -> list[str]:
    reports_dir = BASE / "reports"
    if not reports_dir.exists():
        return []
    dates: list[str] = []
    for path in reports_dir.glob("summary_*.json"):
        stem = path.stem
        if not stem.startswith("summary_"):
            continue
        candidate = stem[len("summary_"):]
        try:
            datetime.strptime(candidate, "%Y-%m-%d")
        except ValueError:
            continue
        dates.append(candidate)
    return sorted(set(dates), reverse=True)


def _summary_has_meaningful_board_content(summary: dict[str, Any]) -> bool:
    if not isinstance(summary, dict):
        return False
    single_bets = summary.get("single_bets") or {}
    if int(single_bets.get("total", 0) or 0) > 0:
        return True
    if int(single_bets.get("review_total", 0) or 0) > 0:
        return True
    if int(single_bets.get("suppressed_total", 0) or 0) > 0:
        return True
    if summary.get("soccer_games") or summary.get("other_games"):
        return True
    diagnostics = summary.get("sport_pipeline_diagnostics") or {}
    totals = diagnostics.get("totals") or {}
    for key in (
        "scanned_games",
        "model_available_games",
        "candidate_games",
        "published_games",
        "review_games",
        "suppressed_games",
        "no_candidate_games",
    ):
        try:
            if int(totals.get(key, 0) or 0) > 0:
                return True
        except Exception:
            continue
    by_sport = diagnostics.get("by_sport") or {}
    return isinstance(by_sport, dict) and bool(by_sport)


def _load_summary_report(
    requested_date: str | None = None,
    *,
    fallback_latest: bool = True,
) -> tuple[dict, Path | None, str, str]:
    date_text = str(requested_date or "").strip()
    checked_dates: list[str] = []

    if date_text:
        checked_dates.append(date_text)
    else:
        checked_dates.append(datetime.now(timezone.utc).strftime("%Y-%m-%d"))

    if fallback_latest:
        for candidate in _available_summary_dates():
            if candidate not in checked_dates:
                checked_dates.append(candidate)

    latest_empty_candidate: tuple[dict, Path | None, str] | None = None

    for date_str in checked_dates:
        summary_path = _summary_report_path(date_str)
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text())
        except Exception as exc:
            return {}, summary_path, date_str, str(exc)
        if _summary_has_meaningful_board_content(summary):
            return summary, summary_path, date_str, ""
        if latest_empty_candidate is None:
            latest_empty_candidate = (summary, summary_path, date_str)

    if latest_empty_candidate and not date_text and fallback_latest:
        summary, summary_path, date_str = latest_empty_candidate
        return summary, summary_path, date_str, ""

    if date_text:
        return {}, None, date_text, f"No scan found for {date_text}."
    return {}, None, "", "No scan summaries available yet."


def _parse_version_snapshot(value: object) -> dict:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _version_summary(summary: dict, pred: pd.DataFrame, settled: pd.DataFrame) -> dict:
    latest = _parse_version_snapshot(summary.get("version_snapshot"))
    rows = []
    for label, df in (("pending", pred), ("settled", settled)):
        if df.empty or "version_snapshot" not in df.columns:
            continue
        snapshots = df["version_snapshot"].apply(_parse_version_snapshot)
        for snap in snapshots:
            if not snap:
                continue
            rows.append({
                "source": label,
                "policy_hash": str(snap.get("policy_hash") or "unknown"),
                "scan_date": str(snap.get("scan_date") or ""),
            })

    row_df = pd.DataFrame(rows)
    if row_df.empty:
        counts = []
    else:
        grouped = (
            row_df.groupby(["source", "policy_hash", "scan_date"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["count", "source", "scan_date"], ascending=[False, True, False])
        )
        counts = grouped.to_dict(orient="records")

    model_rows = []
    for sport, info in sorted((latest.get("models") or {}).items()):
        if not isinstance(info, dict):
            continue
        model_rows.append({
            "sport": sport,
            "tag": str(info.get("tag") or "—"),
            "calibrator_present": bool(info.get("calibrator_present", False)),
            "artifact_sport": str(info.get("artifact_sport") or sport),
        })

    return {
        "latest": latest,
        "row_counts": counts,
        "pending_with_snapshot": int(pred.get("version_snapshot", pd.Series(dtype="object")).astype(str).replace("", pd.NA).notna().sum()) if not pred.empty and "version_snapshot" in pred.columns else 0,
        "settled_with_snapshot": int(settled.get("version_snapshot", pd.Series(dtype="object")).astype(str).replace("", pd.NA).notna().sum()) if not settled.empty and "version_snapshot" in settled.columns else 0,
        "model_rows": model_rows,
    }


def _odds_key_pool_summary() -> dict[str, Any]:
    path = BASE / "data" / "odds_key_pool.json"
    runtime_parse = parse_odds_api_keys_from_env()
    runtime_loaded_fingerprints = [str(item) for item in (runtime_parse.get("fingerprints") or []) if str(item)]
    runtime_loaded_set = set(runtime_loaded_fingerprints)
    runtime_parse_excluded = [
        {
            "fingerprint": str(item.get("fingerprint") or ""),
            "reason": str(item.get("reason") or ""),
            "source": str(item.get("source") or ""),
        }
        for item in (runtime_parse.get("excluded") or [])
        if isinstance(item, dict)
    ]
    def _runtime_only_summary() -> dict[str, Any]:
        return {
            "enabled": bool(runtime_loaded_fingerprints),
            "keys": [
                {
                    "fingerprint": fp,
                    "remaining": None,
                    "updated_at": "",
                    "selected": False,
                    "low_quota": False,
                    "runtime_available": True,
                    "usable": True,
                    "tracked": False,
                    "metadata_age_hours": None,
                    "metadata_stale": False,
                    "status": "runtime_only",
                    "exclusion_reason": "",
                }
                for fp in runtime_loaded_fingerprints
            ],
            "count": len(runtime_loaded_fingerprints),
            "tracked_count": 0,
            "runtime_loaded_count": len(runtime_loaded_fingerprints),
            "usable_count": len(runtime_loaded_fingerprints),
            "tracked_but_unavailable_count": 0,
            "low_quota_count": 0,
            "stale_metadata_count": 0,
            "tracked_fingerprints": [],
            "runtime_loaded_fingerprints": runtime_loaded_fingerprints,
            "usable_fingerprints": runtime_loaded_fingerprints,
            "excluded_fingerprints": [],
            "excluded_details": [],
            "runtime_parse_excluded": runtime_parse_excluded,
            "total_remaining": None,
            "canonical_pool_path": str(path.resolve()),
            "last_selected_fingerprint": "",
            "last_selected_at": "",
            "last_selected_reason": "",
            "last_selected_selector": "",
            "selected_below_low_threshold": False,
            "healthier_usable_key_existed": False,
            "active_fingerprint": "",
            "active_remaining": None,
        }
    if not path.exists():
        return _runtime_only_summary()
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return _runtime_only_summary()
    if not isinstance(payload, dict):
        return _runtime_only_summary()

    meta = payload.get("_meta", {}) if isinstance(payload.get("_meta"), dict) else {}
    active_fp = str(meta.get("last_selected_fingerprint") or "")
    low_threshold = int(meta.get("low_remaining_threshold", 50) or 50)
    stale_threshold_hours = float(meta.get("metadata_stale_threshold_hours", 24.0) or 24.0)
    meta_runtime_loaded_fingerprints = [str(item) for item in (meta.get("runtime_loaded_fingerprints") or []) if str(item)]
    if not runtime_loaded_fingerprints:
        runtime_loaded_fingerprints = meta_runtime_loaded_fingerprints
        runtime_loaded_set = set(runtime_loaded_fingerprints)
    usable_fingerprints = [str(item) for item in (meta.get("usable_fingerprints") or []) if str(item)]
    tracked_fingerprints = [str(item) for item in (meta.get("tracked_fingerprints") or []) if str(item)]
    excluded_details = [
        {
            "fingerprint": str(item.get("fingerprint") or ""),
            "reason": str(item.get("reason") or ""),
        }
        for item in (meta.get("excluded_details") or [])
        if isinstance(item, dict)
    ]
    excluded_by_fp = {item["fingerprint"]: item["reason"] for item in excluded_details if item.get("fingerprint")}
    usable_set = set(usable_fingerprints)
    tracked_rows = {
        str(fp): row
        for fp, row in payload.items()
        if fp != "_meta" and isinstance(row, dict)
    }
    tracked_fp_set = set(tracked_fingerprints) or set(tracked_rows)
    all_fingerprints = sorted(tracked_fp_set | set(tracked_rows) | runtime_loaded_set)
    rows: list[dict[str, Any]] = []
    for fp in all_fingerprints:
        row = tracked_rows.get(fp, {}) if isinstance(tracked_rows.get(fp), dict) else {}
        remaining = row.get("remaining")
        fingerprint = str(row.get("fingerprint") or fp)
        remaining_value = int(remaining) if isinstance(remaining, int) else None
        selected = active_fp == fingerprint
        updated_at = str(row.get("updated_at") or "")
        metadata_age_hours = None
        metadata_stale = False
        if updated_at:
            try:
                dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                metadata_age_hours = max(0.0, (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() / 3600)
                metadata_stale = metadata_age_hours > stale_threshold_hours
            except ValueError:
                metadata_age_hours = None
        runtime_available = fingerprint in runtime_loaded_set
        tracked = fingerprint in tracked_fp_set or fingerprint in tracked_rows
        usable = (fingerprint in usable_set) if usable_set else runtime_available
        auth_quarantined_until = str(row.get("auth_quarantined_until") or "")
        auth_quarantine_reason = str(row.get("auth_quarantine_reason") or "")
        auth_quarantined = False
        if auth_quarantined_until:
            try:
                dt = datetime.fromisoformat(auth_quarantined_until.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                auth_quarantined = datetime.now(timezone.utc) < dt.astimezone(timezone.utc)
            except ValueError:
                auth_quarantined = False
        if auth_quarantined:
            usable = False
        if auth_quarantined:
            status = "auth_quarantined"
            exclusion_reason = excluded_by_fp.get(fingerprint) or auth_quarantine_reason or "auth_quarantined"
        elif tracked and not runtime_available:
            status = "raw_key_missing"
            exclusion_reason = excluded_by_fp.get(fingerprint) or "raw_key_missing"
        elif remaining_value is not None and remaining_value < low_threshold:
            status = "low"
            exclusion_reason = excluded_by_fp.get(fingerprint) or ""
        elif metadata_stale:
            status = "stale_metadata"
            exclusion_reason = excluded_by_fp.get(fingerprint) or ""
        elif runtime_available and tracked and usable:
            status = "usable"
            exclusion_reason = excluded_by_fp.get(fingerprint) or ""
        else:
            status = "runtime_only"
            exclusion_reason = excluded_by_fp.get(fingerprint) or ""
        rows.append({
            "fingerprint": fingerprint,
            "remaining": remaining_value,
            "updated_at": updated_at,
            "selected": selected,
            "low_quota": remaining_value is not None and remaining_value < low_threshold,
            "runtime_available": runtime_available,
            "usable": usable,
            "tracked": tracked,
            "metadata_age_hours": metadata_age_hours,
            "metadata_stale": metadata_stale,
            "status": status,
            "exclusion_reason": exclusion_reason,
            "auth_quarantined_until": auth_quarantined_until,
            "auth_quarantine_reason": auth_quarantine_reason,
        })

    rows.sort(
        key=lambda row: (
            row.get("remaining") is None,
            -(row.get("remaining") or -1),
            row["fingerprint"],
        )
    )
    remaining_values = [row["remaining"] for row in rows if isinstance(row.get("remaining"), int)]
    active_row = next((row for row in rows if row["selected"]), None)
    tracked_unavailable_count = sum(1 for row in rows if row["status"] == "raw_key_missing")
    low_quota_count = sum(1 for row in rows if row["status"] == "low")
    stale_metadata_count = sum(1 for row in rows if row["status"] == "stale_metadata")
    return {
        "enabled": bool(rows),
        "count": len(rows),
        "canonical_pool_path": str(path.resolve()),
        "tracked_count": len(tracked_fp_set) if tracked_fp_set else len(tracked_rows),
        "runtime_loaded_count": len(runtime_loaded_fingerprints),
        "usable_count": len(usable_fingerprints) if usable_fingerprints else len(runtime_loaded_fingerprints),
        "tracked_but_unavailable_count": tracked_unavailable_count,
        "low_quota_count": low_quota_count,
        "stale_metadata_count": stale_metadata_count,
        "total_remaining": int(sum(remaining_values)) if remaining_values else None,
        "last_selected_fingerprint": active_fp,
        "last_selected_at": str(meta.get("last_selected_at") or ""),
        "last_selected_reason": str(meta.get("last_selected_reason") or ""),
        "last_selected_selector": str(meta.get("last_selected_selector") or ""),
        "low_remaining_threshold": low_threshold,
        "metadata_stale_threshold_hours": stale_threshold_hours,
        "tracked_fingerprints": tracked_fingerprints,
        "runtime_loaded_fingerprints": runtime_loaded_fingerprints,
        "usable_fingerprints": usable_fingerprints,
        "excluded_fingerprints": [str(item) for item in (meta.get("excluded_fingerprints") or []) if str(item)],
        "excluded_details": excluded_details,
        "runtime_parse_excluded": runtime_parse_excluded,
        "selected_below_low_threshold": bool(meta.get("selected_below_low_threshold", False)),
        "healthier_usable_key_existed": bool(meta.get("healthier_usable_key_existed", False)),
        "active_fingerprint": active_row["fingerprint"] if active_row else active_fp,
        "active_remaining": active_row["remaining"] if active_row else None,
        "keys": rows,
    }


def _budget_snapshot_from_remaining(remaining: object, *, monthly_limit: int = 500, reserve: int = 30) -> dict[str, Any]:
    # Odds API usage is governed by the live runtime key pool now, not by an
    # internal monthly pacing model. Keep the shape stable for the UI, but do
    # not invent a daily allowance or protected reserve.
    if not isinstance(remaining, int):
        return {
            "remaining": remaining,
            "days_left_in_cycle": None,
            "daily_allowance": None,
            "remaining_after_reserve": None,
        }
    return {
        "remaining": remaining,
        "days_left_in_cycle": None,
        "daily_allowance": None,
        "remaining_after_reserve": remaining,
    }


def _resolve_odds_dashboard_snapshot(usage: dict[str, Any], pool_summary: dict[str, Any]) -> dict[str, Any]:
    usage_fp = str(usage.get("key_fingerprint") or "")
    usage_remaining = usage.get("odds_remaining", 500)
    usage_start = usage.get("odds_remaining_start", 500)
    usage_used_today = usage.get("odds_requests_used_today", 0)
    usage_used_total = usage.get("odds_requests_used_total", 0)

    active_fp = str(pool_summary.get("active_fingerprint") or "")
    active_remaining = pool_summary.get("active_remaining")
    active_known = isinstance(active_remaining, int)

    if active_fp and active_known:
        same_key = usage_fp == active_fp
        start = usage_start if same_key and isinstance(usage_start, int) else 500
        used_total = usage_used_total if same_key and isinstance(usage_used_total, int) else max(0, start - active_remaining)
        used_today = usage_used_today if same_key and isinstance(usage_used_today, int) else max(0, min(start, used_total))
        return {
            "remaining": int(active_remaining),
            "start": int(start),
            "used_today": int(used_today),
            "used_total": int(used_total),
            "display_key_fingerprint": active_fp,
            "usage_key_fingerprint": usage_fp,
            "display_source": "key_pool_selected",
            "usage_sync_status": "aligned" if same_key else "pool_selected_differs_from_webapp_usage_file",
            "selection_reason": str(pool_summary.get("last_selected_reason") or ""),
        }

    return {
        "remaining": usage_remaining,
        "start": usage_start if isinstance(usage_start, int) else 500,
        "used_today": usage_used_today if isinstance(usage_used_today, int) else 0,
        "used_total": usage_used_total if isinstance(usage_used_total, int) else 0,
        "display_key_fingerprint": usage_fp,
        "usage_key_fingerprint": usage_fp,
        "display_source": "legacy_usage_file",
        "usage_sync_status": "legacy_usage_only",
        "selection_reason": "",
    }


def _pending_settlement_reason(row: pd.Series | dict) -> dict[str, str]:
    sport = str((row.get("sport") if isinstance(row, dict) else row.get("sport")) or "")
    commence = row.get("commence_time") if isinstance(row, dict) else row.get("commence_time")
    timing = _derive_event_timing(commence, sport)
    if not str(commence or "").strip():
        return {"key": "missing_event_time", "label": "Missing event time"}
    if timing["status"] in {"scheduled", "live"}:
        return {"key": "awaiting_completion", "label": "Awaiting completion"}
    if timing["status"] == "played":
        if bool((row.get("is_parlay_leg") if isinstance(row, dict) else row.get("is_parlay_leg"))):
            return {"key": "overdue_parlay_leg", "label": "Overdue parlay leg"}
        return {"key": "overdue_result", "label": "Overdue result"}
    return {"key": "unknown_timing", "label": "Unknown timing"}


def _settlement_reliability(
    pred: pd.DataFrame,
    settled: pd.DataFrame,
    manual_parlays: Optional[list[dict]] = None,
) -> dict[str, Any]:
    pending = pred[pred["status"] == "pending"].copy() if not pred.empty and "status" in pred.columns else pd.DataFrame()
    pending["reason_key"] = pending.apply(lambda row: _pending_settlement_reason(row)["key"], axis=1) if not pending.empty else pd.Series(dtype="object")
    pending["reason_label"] = pending.apply(lambda row: _pending_settlement_reason(row)["label"], axis=1) if not pending.empty else pd.Series(dtype="object")
    pending["event_date"] = pending["commence_time"].apply(_event_date_value) if not pending.empty and "commence_time" in pending.columns else pd.Series(dtype="object")
    pending["age_days"] = (
        (datetime.now(timezone.utc) - pd.to_datetime(pending["commence_time"], utc=True, errors="coerce")).dt.days
        if not pending.empty and "commence_time" in pending.columns
        else pd.Series(dtype="float64")
    )

    sports = set()
    if not pending.empty and "sport" in pending.columns:
        sports.update(str(value) for value in pending["sport"].dropna().unique().tolist())
    if not settled.empty and "sport" in settled.columns:
        sports.update(str(value) for value in settled["sport"].dropna().unique().tolist() if str(value) != "parlay")

    rows: list[dict] = []
    for sport in sorted(s for s in sports if s and s != "parlay"):
        pending_s = pending[pending["sport"] == sport].copy() if not pending.empty else pd.DataFrame()
        settled_s = settled[settled["sport"] == sport].copy() if not settled.empty else pd.DataFrame()
        settled_count = int(len(settled_s))
        pending_count = int(len(pending_s))
        tracked_total = settled_count + pending_count
        coverage = round((settled_count / tracked_total) * 100, 1) if tracked_total else 0.0
        overdue_count = int((pending_s["reason_key"].isin(["overdue_result", "overdue_parlay_leg"])).sum()) if not pending_s.empty else 0
        awaiting_count = int((pending_s["reason_key"] == "awaiting_completion").sum()) if not pending_s.empty else 0
        missing_time_count = int((pending_s["reason_key"] == "missing_event_time").sum()) if not pending_s.empty else 0
        oldest_pending_days = None
        if not pending_s.empty and "commence_time" in pending_s.columns:
            commence_series = pd.to_datetime(pending_s["commence_time"], utc=True, errors="coerce").dropna()
            if not commence_series.empty:
                oldest_pending_days = int((datetime.now(timezone.utc) - commence_series.min().to_pydatetime()).days)
        rows.append({
            "sport": sport,
            "tracked_total": tracked_total,
            "settled_count": settled_count,
            "pending_count": pending_count,
            "settlement_coverage_pct": coverage,
            "overdue_count": overdue_count,
            "awaiting_count": awaiting_count,
            "missing_time_count": missing_time_count,
            "oldest_pending_days": oldest_pending_days,
        })

    reason_rows: list[dict] = []
    if not pending.empty:
        grouped = (
            pending.groupby(["reason_key", "reason_label"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        reason_rows = grouped.to_dict(orient="records")

    pending_samples: list[dict] = []
    if not pending.empty:
        sample_cols = [
            "sport",
            "match_id",
            "team_or_player",
            "reason_key",
            "reason_label",
            "event_date",
            "age_days",
            "market",
            "tier",
            "bookmaker",
        ]
        available_cols = [col for col in sample_cols if col in pending.columns]
        pending_view = pending[available_cols].copy()
        if "age_days" in pending_view.columns:
            pending_view["age_days"] = pending_view["age_days"].fillna(-1)
        pending_view = pending_view.sort_values(
            by=["age_days", "sport"],
            ascending=[False, True],
            na_position="last",
        ).head(12)
        pending_samples = pending_view.rename(columns={
            "match_id": "match",
            "team_or_player": "pick",
            "reason_key": "reason",
            "reason_label": "reason_text",
        }).to_dict(orient="records")

    pending_manual_parlays = [p for p in (manual_parlays or []) if p.get("status") == "pending"]
    pending_manual_legs = sum(
        1
        for parlay in pending_manual_parlays
        for leg in parlay.get("legs", [])
        if leg.get("result") not in ("won", "lost")
    )

    return {
        "summary": {
            "sports": len(rows),
            "tracked_total": int(sum(row["tracked_total"] for row in rows)),
            "settled_total": int(sum(row["settled_count"] for row in rows)),
            "pending_total": int(sum(row["pending_count"] for row in rows)),
            "overdue_total": int(sum(row["overdue_count"] for row in rows)),
            "coverage_pct": round((sum(row["settled_count"] for row in rows) / sum(row["tracked_total"] for row in rows)) * 100, 1) if sum(row["tracked_total"] for row in rows) else 0.0,
            "manual_parlays_pending": len(pending_manual_parlays),
            "manual_legs_pending": int(pending_manual_legs),
        },
        "rows": rows,
        "reason_rows": reason_rows,
        "pending_samples": pending_samples,
        "last_attempt": dict(_last_results_settle_report) if _last_results_settle_report else {},
    }


def _retrain_trigger_rows(
    performance_matrix: list[dict],
    version_summary: dict,
) -> dict[str, Any]:
    if not performance_matrix:
        return {"summary": {"sports": 0, "retrain": 0, "watch": 0, "hold": 0}, "rows": []}

    latest_scan_date = str((version_summary.get("latest") or {}).get("scan_date") or "")
    try:
        latest_dt = datetime.fromisoformat(latest_scan_date).date() if latest_scan_date else None
    except Exception:
        latest_dt = None

    by_sport: dict[str, list[dict]] = defaultdict(list)
    for row in performance_matrix:
        by_sport[str(row.get("sport") or "")].append(row)

    rows: list[dict] = []
    for sport, sport_rows in sorted(by_sport.items()):
        bets = sum(int(r.get("bets", 0) or 0) for r in sport_rows)
        tracked_total = sum(int(r.get("tracked_total", r.get("bets", 0)) or 0) for r in sport_rows)
        pending_count = sum(int(r.get("pending_count", 0) or 0) for r in sport_rows)
        clv_covered = sum(int(r.get("clv_covered", 0) or 0) for r in sport_rows)
        total_stake_proxy = sum(max(float(r.get("bets", 0) or 0), 1.0) for r in sport_rows)
        weighted_roi = sum(float(r.get("roi", 0) or 0) * max(float(r.get("bets", 0) or 0), 1.0) for r in sport_rows) / total_stake_proxy
        clv_values = [float(r.get("avg_clv")) for r in sport_rows if r.get("avg_clv") is not None]
        weighted_clv = sum(float(r.get("avg_clv", 0) or 0) * max(float(r.get("clv_covered", 0) or 0), 1.0) for r in sport_rows if r.get("avg_clv") is not None)
        clv_den = sum(max(float(r.get("clv_covered", 0) or 0), 1.0) for r in sport_rows if r.get("avg_clv") is not None)
        avg_clv = (weighted_clv / clv_den) if clv_den > 0 else None
        settlement_coverage = round((bets / tracked_total) * 100, 1) if tracked_total else 0.0
        clv_coverage = round((clv_covered / bets) * 100, 1) if bets else 0.0
        weak_lanes = [r for r in sport_rows if r.get("clv_signal") == "weak"]
        variance_lanes = [r for r in sport_rows if r.get("clv_signal") == "variance"]
        confirmed_lanes = [r for r in sport_rows if r.get("clv_signal") == "confirmed"]
        weak_share = (len(weak_lanes) / len(sport_rows)) if sport_rows else 0.0
        days_since_scan = (datetime.now(timezone.utc).date() - latest_dt).days if latest_dt else None

        action = "hold"
        confidence = "low"
        reason = "Healthy enough to keep monitoring without a retrain."

        if bets < 12 or settlement_coverage < 65.0:
            action = "watch"
            reason = "Sample or settlement coverage is still too thin for a trustworthy retrain call."
        elif clv_coverage < 45.0:
            action = "watch"
            reason = "CLV coverage is still too low; collect more close-line evidence before retraining."
        elif avg_clv is not None and avg_clv > 0.5 and weighted_roi <= -3.0:
            action = "hold"
            confidence = "medium"
            reason = "Results are soft, but positive CLV suggests variance or selection issues rather than model drift."
        elif avg_clv is not None and avg_clv <= -1.0 and weighted_roi <= -5.0 and weak_share >= 0.5 and bets >= 20:
            action = "retrain"
            confidence = "high" if days_since_scan is None or days_since_scan >= 3 else "medium"
            reason = "Negative CLV and weak live ROI across multiple lanes suggest the model is drifting and deserves a refresh."
        elif avg_clv is not None and avg_clv <= -0.5 and weighted_roi <= -3.0 and bets >= 16:
            action = "watch"
            confidence = "medium"
            reason = "Early drift signal detected; if the next settled batch stays weak, this sport should move to retrain."
        elif weak_lanes and not confirmed_lanes:
            action = "watch"
            reason = "This sport is leaning weak, but the evidence is not broad enough yet for an automatic retrain trigger."

        rows.append({
            "sport": sport,
            "bets": bets,
            "tracked_total": tracked_total,
            "pending_count": pending_count,
            "settlement_coverage_pct": settlement_coverage,
            "clv_coverage_pct": clv_coverage,
            "roi": round(weighted_roi, 2),
            "avg_clv": None if avg_clv is None else round(avg_clv, 2),
            "weak_lanes": len(weak_lanes),
            "variance_lanes": len(variance_lanes),
            "confirmed_lanes": len(confirmed_lanes),
            "lanes": len(sport_rows),
            "days_since_scan": days_since_scan,
            "latest_tag": next((row.get("tag") for row in version_summary.get("model_rows", []) if row.get("sport") == sport), "—"),
            "action": action,
            "confidence": confidence,
            "reason": reason,
        })

    rows.sort(key=lambda row: ({"retrain": 0, "watch": 1, "hold": 2}.get(row["action"], 3), row["roi"], row["avg_clv"] or 0))
    return {
        "summary": {
            "sports": len(rows),
            "retrain": sum(1 for row in rows if row["action"] == "retrain"),
            "watch": sum(1 for row in rows if row["action"] == "watch"),
            "hold": sum(1 for row in rows if row["action"] == "hold"),
        },
        "rows": rows,
    }


def _rebuild_candidates(
    performance_matrix: list[dict],
    retrain_triggers: dict[str, Any],
    governor_recommendations: list[dict],
    replay_support: dict[tuple[str, str], dict[str, Any]],
    version_summary: dict[str, Any],
    calibration_snapshot: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    if not performance_matrix:
        return {"summary": {"candidates": 0, "retrain": 0, "policy": 0, "watch": 0}, "rows": []}

    retrain_by_sport = {
        str(row.get("sport")): row
        for row in (retrain_triggers.get("rows") or [])
    }
    governor_by_lane = {
        f"{row.get('sport')}:{row.get('market')}:{row.get('tier')}": row
        for row in governor_recommendations
    }
    model_tag_by_sport = {
        str(row.get("sport")): str(row.get("tag") or "—")
        for row in (version_summary.get("model_rows") or [])
    }
    calibration_by_sport = {
        str(row.get("sport")): row
        for row in ((calibration_snapshot or {}).get("by_sport") or [])
    }

    def _policy_template(sport: str, market: str, *, mode: str) -> dict[str, Any]:
        current = get_market_policy(sport, market)
        current_status = str(current.get("status") or "experimental")
        current_score = int(current.get("score", 50) or 50)
        current_stake = float(current.get("stake_multiplier", 0.0) or 0.0)
        if mode == "pause":
            return {
                "status": "disabled",
                "score": max(10, min(current_score, 25)),
                "stake_multiplier": 0.0,
                "summary": f"Pause {sport} {market} by disabling live publication until evidence improves.",
            }
        if mode == "demote":
            return {
                "status": "experimental",
                "score": max(35, current_score - 15),
                "stake_multiplier": round(max(0.0, min(current_stake, 0.25)), 2),
                "summary": f"Demote {sport} {market} into the limited/experimental lane with smaller stake sizing.",
            }
        if mode == "tighten":
            return {
                "status": current_status,
                "score": max(35, current_score - 5),
                "stake_multiplier": round(max(0.0, current_stake * 0.75), 2),
                "summary": f"Tighten {sport} {market} without fully demoting it by reducing score and stake sizing.",
            }
        return {
            "status": current_status,
            "score": current_score,
            "stake_multiplier": current_stake,
            "summary": f"Keep {sport} {market} unchanged while collecting more evidence.",
        }

    rows: list[dict[str, Any]] = []
    for row in performance_matrix:
        sport = str(row.get("sport") or "")
        market = str(row.get("market") or "")
        tier = str(row.get("tier") or "")
        tier_status = str(row.get("tier_status") or "experimental")
        bets = int(row.get("bets", 0) or 0)
        roi = float(row.get("roi", 0) or 0)
        avg_clv = row.get("avg_clv")
        tracked_total = int(row.get("tracked_total", bets) or bets)
        settlement_coverage_pct = float(row.get("settlement_coverage_pct", 0) or 0)
        clv_coverage_pct = float(row.get("clv_coverage_pct", 0) or 0)
        clv_signal = str(row.get("clv_signal") or "missing")
        lane = f"{sport}:{market}:{tier}"
        replay_row = replay_support.get((sport, market), {})
        replay_level = str(replay_row.get("support_level") or "missing")
        replay_games = int(replay_row.get("games_scored", 0) or 0)
        governor = governor_by_lane.get(lane)
        sport_trigger = retrain_by_sport.get(sport)
        calibration_row = calibration_by_sport.get(sport, {})
        calibration_gap_pp = float(calibration_row.get("gap_pp", 0) or 0)
        calibration_brier = calibration_row.get("brier")
        calibration_log_loss = calibration_row.get("log_loss")
        calibration_avg_prob = calibration_row.get("avg_prob_pct")
        calibration_win_rate = calibration_row.get("win_rate_pct")
        calibration_bets = int(calibration_row.get("bets", 0) or 0)

        action = None
        trigger = None
        confidence = "low"
        rationale = ""
        next_step = ""
        draft_command = None
        draft_policy = None
        policy_template = None

        if sport_trigger and sport_trigger.get("action") == "retrain":
            action = "retrain"
            trigger = "sport_retrain_trigger"
            confidence = str(sport_trigger.get("confidence") or "medium")
            rationale = (
                f"{sport} has sport-level drift: ROI {sport_trigger.get('roi', 0):+.2f}%, "
                f"avg CLV {sport_trigger.get('avg_clv') if sport_trigger.get('avg_clv') is not None else '—'}%, "
                f"weak lanes {sport_trigger.get('weak_lanes', 0)}/{sport_trigger.get('lanes', 0)}."
            )
            draft_command = f".venv/bin/python retrain_and_calibrate.py --sport {sport}"
            next_step = "Refresh models and calibrator, then run a fresh scan and compare version-tagged results."
        elif governor and governor.get("action") in {"demote", "pause"}:
            gov_action = str(governor.get("action"))
            action = "policy_tighten" if gov_action == "demote" else "policy_pause"
            trigger = "lane_governor"
            confidence = str(governor.get("confidence") or "medium")
            rationale = str(governor.get("reason") or "")
            policy_template = _policy_template(sport, market, mode="demote" if gov_action == "demote" else "pause")
            if gov_action == "demote":
                draft_policy = policy_template["summary"]
                next_step = "Review the draft policy preview, tighten live exposure, and replay the lane before promoting again."
            else:
                draft_policy = policy_template["summary"]
                next_step = "Keep the lane off the live board, collect replay and CLV evidence, then reassess."
        elif (
            bets >= 12
            and settlement_coverage_pct >= 70
            and clv_coverage_pct >= 60
            and avg_clv is not None
            and avg_clv <= -1.0
            and roi <= -4.0
            and replay_level in {"weak", "mixed"}
        ):
            action = "threshold_tighten"
            trigger = "lane_performance"
            confidence = "medium" if replay_level == "mixed" else "high"
            rationale = (
                f"{lane} is weak live (ROI {roi:+.2f}%, avg CLV {avg_clv:+.2f}%) with replay support {replay_level} "
                f"over {replay_games} scored games."
            )
            policy_template = _policy_template(sport, market, mode="tighten")
            draft_policy = policy_template["summary"]
            next_step = "Test a stricter threshold or lower stake multiplier through replay, then re-scan."
        elif (
            tracked_total >= 8
            and settlement_coverage_pct < 70
        ) or (
            bets >= 8
            and clv_coverage_pct < 60
        ) or clv_signal in {"missing", "variance", "lucky"}:
            action = "watch"
            trigger = "evidence_thin"
            confidence = "low"
            rationale = (
                f"{lane} is not ready for a rebuild action yet: settled {bets}/{tracked_total}, "
                f"settlement {settlement_coverage_pct:.1f}%, CLV coverage {clv_coverage_pct:.1f}%, signal {clv_signal}."
            )
            next_step = "Collect more settled bets or CLV coverage before changing policy or retraining."

        severe_calibration_gap = calibration_bets >= 10 and calibration_gap_pp >= 30.0
        weak_probability_quality = (
            (calibration_brier is not None and float(calibration_brier) >= 0.22)
            or (calibration_log_loss is not None and float(calibration_log_loss) >= 0.75)
        )
        if (
            calibration_bets >= 8
            and calibration_gap_pp >= 20.0
            and (weak_probability_quality or severe_calibration_gap)
            and (not action or trigger == "evidence_thin")
        ):
            action = "retrain"
            trigger = "sport_miscalibration"
            confidence = "high" if calibration_gap_pp >= 30.0 else "medium"
            rationale = (
                f"{sport} is materially miscalibrated: avg predicted win rate {calibration_avg_prob:.1f}% "
                f"vs actual {calibration_win_rate:.1f}% (gap {calibration_gap_pp:.1f}pp), "
                f"Brier {float(calibration_brier or 0):.4f}, log loss {float(calibration_log_loss or 0):.4f}."
            )
            draft_command = f".venv/bin/python retrain_and_calibrate.py --sport {sport}"
            next_step = "Rebuild and recalibrate this sport before trusting new lane-level policy moves."

        if not action:
            continue

        if calibration_row and trigger != "sport_miscalibration":
            rationale = (
                f"{rationale} Calibration: avg predicted {calibration_avg_prob:.1f}% vs actual "
                f"{calibration_win_rate:.1f}% (gap {calibration_gap_pp:.1f}pp)."
            )

        rows.append({
            "sport": sport,
            "market": market,
            "tier": tier,
            "tier_status": tier_status,
            "lane": lane,
            "action": action,
            "trigger": trigger,
            "confidence": confidence,
            "bets": bets,
            "tracked_total": tracked_total,
            "settlement_coverage_pct": round(settlement_coverage_pct, 1),
            "clv_coverage_pct": round(clv_coverage_pct, 1),
            "roi": round(roi, 2),
            "avg_clv": None if avg_clv is None else round(float(avg_clv), 2),
            "clv_signal": clv_signal,
            "replay_support": replay_level,
            "replay_games": replay_games,
            "model_tag": model_tag_by_sport.get(sport, "—"),
            "calibration_gap_pp": round(calibration_gap_pp, 1) if calibration_row else None,
            "calibration_bets": calibration_bets if calibration_row else None,
            "calibration_brier": None if calibration_brier is None else round(float(calibration_brier), 4),
            "calibration_log_loss": None if calibration_log_loss is None else round(float(calibration_log_loss), 4),
            "rationale": rationale,
            "next_step": next_step,
            "draft_command": draft_command,
            "draft_policy": draft_policy,
            "policy_template": policy_template,
        })

    action_order = {"retrain": 0, "policy_tighten": 1, "policy_pause": 2, "threshold_tighten": 3, "watch": 4}
    confidence_order = {"high": 0, "medium": 1, "low": 2}
    rows.sort(key=lambda row: (
        action_order.get(str(row.get("action")), 9),
        confidence_order.get(str(row.get("confidence")), 9),
        str(row.get("sport")),
        str(row.get("market")),
    ))
    return {
        "summary": {
            "candidates": len(rows),
            "retrain": sum(1 for row in rows if row["action"] == "retrain"),
            "policy": sum(1 for row in rows if str(row["action"]).startswith("policy") or row["action"] == "threshold_tighten"),
            "watch": sum(1 for row in rows if row["action"] == "watch"),
        },
        "rows": rows,
    }


def _reasoning_candidate_id(bet: dict) -> str:
    parts = [
        str(bet.get("sport", "")),
        str(bet.get("market", "")),
        str(bet.get("home", "")),
        str(bet.get("away", "")),
        str(bet.get("team", "")),
    ]
    return "|".join(parts)


def _supported_reasoning_market(bet: dict) -> bool:
    return bet.get("market") in {"moneyline", "spreads", "totals", "double_chance", "draw_no_bet"}


def _today_reasoning_bets() -> list[dict]:
    summary = _today_summary()
    single_bets = summary.get("single_bets", {})
    bets = [
        _resolve_decision_fields(_apply_timing_to_bet(bet))
        for bet in _annotate_bets(
            (single_bets.get("bets", []) + single_bets.get("review_bets", []) + single_bets.get("suppressed_bets", []))
        )
    ]
    bets = [
        bet for bet in bets
        if bet.get("window") in ("today", "tomorrow") and _supported_reasoning_market(bet)
    ]
    return sorted(
        bets,
        key=lambda b: (b.get("market_priority_score", 0), b.get("edge", 0)),
        reverse=True,
    )


def _top_context_summaries(adjustments: list[dict], limit: int = 3) -> list[str]:
    preferred = {"matchup", "coaching", "environment", "lineup", "schedule", "motivation"}
    picked: list[str] = []
    for item in adjustments or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("category", "")) not in preferred:
            continue
        summary = str(item.get("summary") or item.get("name") or "").strip()
        if summary and summary not in picked:
            picked.append(summary)
        if len(picked) >= limit:
            break
    return picked


def _top_scraped_context_highlights(highlights: list, limit: int = 4) -> list[str]:
    picked: list[str] = []
    for item in highlights or []:
        text = str(item or "").strip()
        if text and text not in picked:
            picked.append(text)
        if len(picked) >= limit:
            break
    return picked


def _attach_fresh_news_context(report, sport: str, home_team: str, away_team: str, bet: str) -> dict:
    context = collect_matchup_news_context(
        sport=sport,
        home_team=home_team,
        away_team=away_team,
        bet=bet,
        timeout=4,
    )
    if not hasattr(report, "warnings") or getattr(report, "warnings") is None:
        report.warnings = []
    if not hasattr(report, "data_points") or getattr(report, "data_points") is None:
        report.data_points = {}
    report.data_points["fresh_news_context"] = context
    for highlight in context.get("highlights", [])[:3]:
        note = f"Fresh web context: {highlight}"
        if note not in report.warnings:
            report.warnings.append(note)
    for warning in context.get("warnings", [])[:2]:
        note = f"Fresh web context: {warning}"
        if note not in report.warnings:
            report.warnings.append(note)
    return context


def _as_float_or_none(value) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _signal_score_from_factor(payload: dict, fallback: float = 0.0) -> float:
    score = _as_float_or_none(payload.get("score"))
    if score is not None:
        return score
    value = _as_float_or_none(payload.get("value"))
    if value is None:
        return fallback
    # Normalize raw factor values into a bounded display score so they are legible.
    if abs(value) >= 1:
        return max(-2.5, min(2.5, value))
    return max(-2.5, min(2.5, value * 4.0))


def _dedupe_strings(values: list[str], limit: int | None = None) -> list[str]:
    picked: list[str] = []
    seen: set[str] = set()
    for raw in values:
        text = str(raw or "").strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        picked.append(text)
        if limit and len(picked) >= limit:
            break
    return picked


def _candidate_reasoning_report(candidate: dict, market: str, selection: str) -> AnalysisReport:
    """Build a fast, cached report for the guarded reasoning lane."""
    fair_prob = _as_float_or_none(candidate.get("fair_prob"))
    price = _as_float_or_none(candidate.get("odds"))
    edge = _as_float_or_none(candidate.get("edge"))
    true_probability = candidate.get("true_probability") or {}
    confidence = _as_float_or_none(true_probability.get("confidence"))
    if confidence is None:
        confidence = fair_prob if fair_prob is not None else 0.5
    confidence = max(0.05, min(0.95, confidence))

    report = AnalysisReport(
        sport=str(candidate.get("sport", "")).lower(),
        home_team=str(candidate.get("home", "")),
        away_team=str(candidate.get("away", "")),
        market=market,
        bet=str(candidate.get("team", "")),
        selection=selection or "",
        verdict="review" if candidate.get("review_required") else "pass",
        confidence=confidence,
        fair_prob=fair_prob,
        price_used=price,
        edge_pct=edge,
    )
    report.data_points.update({
        "market": {
            "best_prices": {str(candidate.get("team", "Pick")): price} if price else {},
            "fair_probabilities": {str(candidate.get("team", "Pick")): fair_prob} if fair_prob else {},
            "sources": [],
            "warnings": [],
            "unknowns": [],
        },
        "true_probability": true_probability,
        "prediction_factors": candidate.get("prediction_factors") or [],
        "context_adjustments": candidate.get("context_adjustments") or [],
        "scraped_context": candidate.get("scraped_context") or {},
    })

    for factor in candidate.get("prediction_factors") or []:
        if not isinstance(factor, dict):
            continue
        name = str(factor.get("name") or factor.get("category") or "model signal").strip()
        summary = str(factor.get("summary") or factor.get("description") or name).strip()
        score = _signal_score_from_factor(factor, fallback=0.0)
        signal_conf = _as_float_or_none(factor.get("confidence")) or confidence
        if summary:
            report.signals.append(AnalysisSignal(
                name=name,
                summary=summary,
                score=score,
                confidence=max(0.05, min(0.95, signal_conf)),
                data=factor,
            ))

    for item in candidate.get("context_adjustments") or []:
        if not isinstance(item, dict):
            continue
        summary = str(item.get("summary") or item.get("name") or "").strip()
        if summary:
            report.signals.append(AnalysisSignal(
                name=str(item.get("category") or "context"),
                summary=summary,
                score=_signal_score_from_factor(item, fallback=0.0),
                confidence=_as_float_or_none(item.get("confidence")) or 0.55,
                data=item,
            ))

    if candidate.get("review_required") and candidate.get("review_reason"):
        report.warnings.append(str(candidate.get("review_reason")))
    if candidate.get("availability_summary"):
        report.warnings.append(f"System availability: {candidate.get('availability_summary')}")
    if candidate.get("odds_recheck_status"):
        report.warnings.append(f"Odds recheck: {candidate.get('odds_recheck_status')}")
    if candidate.get("market_policy_reason"):
        report.warnings.append(str(candidate.get("market_policy_reason")))
    for highlight in _dedupe_strings(_top_scraped_context_highlights(candidate.get("scraped_context_highlights") or []), limit=4):
        report.warnings.append(f"System scraper: {highlight}")

    deduped_signals: list[AnalysisSignal] = []
    seen_signal_keys: set[tuple[str, str]] = set()
    for signal in report.signals:
        key = (str(signal.name or "").strip().lower(), str(signal.summary or "").strip().lower())
        if key in seen_signal_keys:
            continue
        seen_signal_keys.add(key)
        deduped_signals.append(signal)
    report.signals = deduped_signals

    report.sources.append(SourceNote(
        name="Daily scan summary",
        detail="Cached value-bet candidate, model factors, odds recheck, and publish policy from today's scan.",
    ))
    if candidate.get("scraped_context_sources"):
        report.sources.append(SourceNote(
            name="Scan scraper context",
            detail=", ".join(str(src) for src in candidate.get("scraped_context_sources")[:4]),
        ))
    return report


def _evidence_channel(status: str, label: str, detail: str, sources: list[str] | None = None) -> dict:
    return {
        "status": status,
        "label": label,
        "detail": detail,
        "sources": sources or [],
    }


def _build_evidence_profile(report: AnalysisReport, fresh_news_context: dict | None = None, candidate: dict | None = None) -> dict:
    """
    Score the reliability of the data channels feeding a deep analysis.

    This is the internal safety layer: odds-only edges should not become production-quality
    recommendations when matchup, availability, or schedule evidence is thin.
    """
    candidate = candidate or {}
    fresh_news_context = fresh_news_context or {}
    data_points = report.data_points or {}
    market = data_points.get("market") or {}
    warnings = [str(item).lower() for item in (report.warnings or []) + (report.unknowns or [])]
    signals = report.signals or []
    fresh_sources = [str(src) for src in fresh_news_context.get("sources", []) if src]
    fresh_highlights = [str(item) for item in fresh_news_context.get("highlights", []) if item]
    fresh_channels = fresh_news_context.get("channels") or {}
    scraped_sources = [str(src) for src in candidate.get("scraped_context_sources") or [] if src]
    scraped_highlights = [str(item) for item in candidate.get("scraped_context_highlights") or [] if item]
    availability_summary = str(candidate.get("availability_summary") or "").strip()

    channels: dict[str, dict] = {}
    channels["bet_identity"] = _evidence_channel(
        "strong" if report.sport and report.home_team and report.away_team and report.bet else "thin",
        "Bet Identity",
        "Sport, matchup, market, selection, and bet text are present." if report.home_team and report.away_team else "Missing core bet identity fields.",
        ["Daily scan candidate" if candidate else "Manual form"],
    )

    price_sources = []
    for source in market.get("sources") or []:
        if isinstance(source, dict):
            price_sources.append(str(source.get("name") or "Market source"))
        else:
            price_sources.append(str(getattr(source, "name", source)))
    if market.get("event", {}).get("cached"):
        price_sources.append("Odds disk cache")
    if report.price_used is not None and report.edge_pct is not None:
        price_status = "strong" if price_sources or candidate.get("odds_recheck_status") else "medium"
        price_detail = "Usable price and edge are available."
    else:
        price_status = "thin"
        price_detail = "No reliable price/edge pair is available."
    channels["price_validity"] = _evidence_channel(price_status, "Price Validity", price_detail, price_sources or ["The Odds API / cache"])

    form_signals = [
        signal for signal in signals
        if str(signal.name).lower() not in {"market edge"} and abs(float(signal.score or 0)) > 0.01
    ]
    thin_matchup_warning = any(
        phrase in warning
        for warning in warnings
        for phrase in ("not enough recent games", "standings context was unavailable", "rest-day comparison was unavailable")
    )
    if len(form_signals) >= 3 and not thin_matchup_warning:
        matchup_status = "strong"
        matchup_detail = f"{len(form_signals)} non-market matchup signals available."
    elif len(form_signals) >= 1:
        matchup_status = "medium"
        matchup_detail = f"{len(form_signals)} non-market matchup signal(s) available."
    else:
        matchup_status = "thin"
        matchup_detail = "Matchup/form evidence is missing or too weak."
    channels["matchup_context"] = _evidence_channel(matchup_status, "Matchup Context", matchup_detail, ["Sport historical data"])

    availability_text = " ".join([availability_summary, *scraped_highlights, *fresh_highlights]).lower()
    availability_terms = ("injury", "injured", "lineup", "starter", "starting", "goalie", "pitcher", "questionable", "out", "suspension")
    if availability_summary or any(term in availability_text for term in availability_terms):
        availability_status = "medium"
        availability_detail = "Availability/team-news evidence was found, but may still need confirmation."
    else:
        availability_status = "thin"
        availability_detail = "No confirmed injury, lineup, starter, goalie, or rotation channel was available."
    channels["availability"] = _evidence_channel(
        availability_status,
        "Availability",
        availability_detail,
        scraped_sources + fresh_sources,
    )

    if fresh_sources and fresh_highlights:
        news_status = "medium"
        news_detail = f"{len(fresh_highlights)} fresh web context highlight(s) from {len(fresh_sources)} source(s)."
    elif fresh_sources:
        news_status = "medium"
        news_detail = f"{len(fresh_sources)} preview/news source(s) found, but no high-confidence extracted facts yet."
    else:
        news_status = "thin"
        news_detail = "No fresh preview/news source produced usable context."
    channels["fresh_news"] = _evidence_channel(news_status, "Fresh News", news_detail, fresh_sources)

    for channel_key, channel_payload in fresh_channels.items():
        if not isinstance(channel_payload, dict):
            continue
        found_sources = [str(src) for src in channel_payload.get("sources") or [] if src]
        found_highlights = [str(item) for item in channel_payload.get("highlights") or [] if item]
        trust = str(channel_payload.get("trust") or "context")
        if found_highlights:
            status = "medium"
            detail = f"{len(found_highlights)} snippet highlight(s); trust level: {trust}."
        elif found_sources:
            status = "medium"
            detail = f"{len(found_sources)} source(s) found; trust level: {trust}."
        else:
            status = "thin"
            detail = f"No usable snippets found; trust level: {trust}."
        channels[f"web_{channel_key}"] = _evidence_channel(
            status,
            str(channel_payload.get("label") or channel_key.replace("_", " ").title()),
            detail,
            found_sources,
        )

    risk_flags = []
    if channels["matchup_context"]["status"] == "thin":
        risk_flags.append("Matchup/form data is thin.")
    if channels["availability"]["status"] == "thin":
        risk_flags.append("Availability channel is thin.")
    if channels["price_validity"]["status"] == "thin":
        risk_flags.append("Price validity is thin.")
    if report.edge_pct is not None and report.edge_pct > 0.08 and channels["matchup_context"]["status"] == "thin":
        risk_flags.append("Large edge appears market-led rather than supported by matchup evidence.")

    core_statuses = [channels[key]["status"] for key in ("price_validity", "matchup_context", "availability")]
    if "thin" in core_statuses:
        quality = "thin"
    elif core_statuses.count("strong") >= 2:
        quality = "strong"
    else:
        quality = "medium"

    decision = "DATA THIN" if quality == "thin" else "REVIEW" if quality == "medium" else "APPROVE"
    return {
        "quality": quality,
        "decision": decision,
        "channels": channels,
        "risk_flags": risk_flags,
        "provider_plan": {
            "price": ["The Odds API", "Pinnacle API if configured"],
            "availability": ["API-Football/API-Sports", "MLB Stats API", "NHL API", "Rotowire/MySportsFeeds if configured"],
            "matchup": ["football-data.org", "MLB Stats API", "NHL API", "Sports-Reference/Sportmonks if configured"],
            "news": ["Fresh preview search", "NewsAPI if configured"],
            "context_channels": ["ESPN", "Flashscore fallback", "Reddit community/unverified"],
            "weather": ["OpenWeatherMap if venue coordinates are available"],
        },
    }


def _apply_evidence_gate(report: AnalysisReport, evidence_profile: dict) -> None:
    if not hasattr(report, "data_points") or report.data_points is None:
        report.data_points = {}
    report.data_points["evidence_profile"] = evidence_profile
    quality = str(evidence_profile.get("quality") or "thin")
    decision = str(evidence_profile.get("decision") or "DATA THIN")
    note = f"Evidence hub: {decision} — data quality is {quality}."
    if note not in report.warnings:
        report.warnings.append(note)
    for flag in evidence_profile.get("risk_flags", [])[:3]:
        flag_note = f"Evidence risk: {flag}"
        if flag_note not in report.warnings:
            report.warnings.append(flag_note)
    if quality == "thin":
        report.verdict = "data_thin"
        report.confidence = min(float(report.confidence or 0.5), 0.52)
    elif quality == "medium" and str(report.verdict).lower() == "support":
        report.verdict = "review"
        report.confidence = min(float(report.confidence or 0.6), 0.65)


def _candidate_context_packet(candidate: dict, report_payload: dict) -> dict:
    evidence = (report_payload.get("data_points") or {}).get("evidence_profile") or {}
    report_signals = [
        signal for signal in (report_payload.get("signals") or [])
        if str(signal.get("name", "")).lower() != "market edge"
    ]
    return {
        "task": (
            "Act as a context-only referee. Evaluate whether real-world evidence supports, contradicts, "
            "or is too thin to trust the supplied system pick. Do not use model edge, fair probability, "
            "or pricing math to justify the decision."
        ),
        "system_pick": {
            "sport": candidate.get("sport"),
            "market": candidate.get("market"),
            "selection": candidate.get("team"),
            "home_team": candidate.get("home"),
            "away_team": candidate.get("away"),
            "kick_off": candidate.get("kick_off"),
            "league": candidate.get("league"),
            "availability_summary": candidate.get("availability_summary"),
            "review_required": candidate.get("review_required"),
            "review_reason": candidate.get("review_reason"),
        },
        "evidence_profile": evidence,
        "non_market_signals": report_signals,
        "watchouts": {
            "warnings": report_payload.get("warnings") or [],
            "unknowns": report_payload.get("unknowns") or [],
        },
        "fresh_news_context": (report_payload.get("data_points") or {}).get("fresh_news_context") or {},
        "scan_scraper_context": {
            "sources": candidate.get("scraped_context_sources") or [],
            "highlights": candidate.get("scraped_context_highlights") or [],
            "availability": candidate.get("scraped_context") or {},
        },
    }


def _openrouter_reasoning_layer(candidate: dict, report_payload: dict,
                                system_prompt: str | None = None) -> dict | None:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return None

    model = os.getenv("OPENROUTER_REASONING_MODEL", "openrouter/free").strip() or "openrouter/free"
    if system_prompt is None:
        system_prompt = _REASONING_SYSTEM_PROMPT
    user_prompt = json.dumps(_candidate_context_packet(candidate, report_payload), ensure_ascii=True)

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5000",
            "X-Title": "SharpEdge Sports Predictor",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 500,
        },
        timeout=12,
    )
    response.raise_for_status()
    payload = response.json()
    message = (((payload.get("choices") or [{}])[0]).get("message") or {}).get("content")
    message_text = str(message or "").strip()
    parsed: dict
    try:
        parsed = json.loads(message_text)
    except Exception:
        parsed = {
            "decision": "REVIEW",
            "agrees_with_system": False,
            "recommendation": message_text or "No reasoning returned.",
            "reasoning": message_text or "No reasoning returned.",
            "why_for": "",
            "why_against": "",
            "biggest_risk": "",
            "stake_guidance": "",
            "critical_factors": [],
            "only_context_based": True,
        }
    parsed.setdefault("decision", "REVIEW")
    parsed.setdefault("agrees_with_system", parsed.get("decision") == "APPROVE")
    parsed.setdefault("recommendation", "")
    parsed.setdefault("reasoning", "")
    parsed.setdefault("why_for", "")
    parsed.setdefault("why_against", "")
    parsed.setdefault("biggest_risk", "")
    parsed.setdefault("stake_guidance", "")
    parsed.setdefault("critical_factors", [])
    parsed.setdefault("only_context_based", True)
    return {
        "provider": "openrouter",
        "model": payload.get("model", model),
        "content": parsed,
        "raw": message_text,
    }


_REASONING_SYSTEM_PROMPT = (
    "You are an independent sports betting context referee. "
    "You are deliberately separated from the quantitative model and must not rely on model edge, fair probability, "
    "Kelly staking, or odds-derived math. You may only use matchup evidence, availability, schedule context, "
    "fresh news, warnings, and evidence-channel quality. "
    "Evaluate whether you agree with the supplied system pick from context alone. Never suggest another game, team, or market. "
    "Use APPROVE only when context clearly supports or is neutral for the pick, REVIEW when uncertainty is material, "
    "VETO when context clearly contradicts the pick, and DATA_THIN when evidence is not reliable enough. "
    "Output valid JSON with keys: decision, agrees_with_system, recommendation, reasoning, why_for, why_against, "
    "biggest_risk, stake_guidance, critical_factors, only_context_based."
)

_LONGSHOT_SYSTEM_PROMPT = (
    "You are a specialist in identifying credible upsets in sports betting for longshot parlays. "
    "Your job is NOT to find safe picks — it is to find underdog picks that have a specific, real reason to win TODAY. "
    "You may ONLY use context evidence: team motivation, recent form vs this opponent, key injuries to the favourite, "
    "tactical mismatches, venue advantage, schedule fatigue, head-to-head patterns, and sharp market signals. "
    "Do NOT use model probabilities, edge, or Kelly math. "
    "Ask yourself one question: Is there a concrete, specific reason this underdog can win this match? "
    "APPROVE if there is a clear contextual upset catalyst (e.g. favourite missing key player, underdog on revenge game, "
    "back-to-back fatigue for favourite, strong home record in this fixture). "
    "REVIEW if an upset is plausible but the evidence is circumstantial or thin. "
    "VETO if the favourite is clearly dominant and the underdog has no realistic path to victory. "
    "DATA_THIN if you cannot form a judgment from the available evidence. "
    "Output valid JSON with exactly these keys: decision, agrees_with_system, recommendation, reasoning, "
    "why_for, why_against, biggest_risk, upset_catalyst, stake_guidance, critical_factors, only_context_based. "
    "The 'upset_catalyst' field must name the single most important factor that could cause an upset, "
    "or an empty string if none exists."
)

_REASONING_DEFAULTS = {
    "decision": "REVIEW",
    "agrees_with_system": False,
    "recommendation": "",
    "reasoning": "",
    "why_for": "",
    "why_against": "",
    "biggest_risk": "",
    "upset_catalyst": "",
    "stake_guidance": "",
    "critical_factors": [],
    "only_context_based": True,
}


def _parse_reasoning_json(text: str) -> dict:
    """Parse LLM JSON output, falling back to a REVIEW verdict on failure."""
    import re
    text = text.strip()
    # Strip DeepSeek R1 <think>…</think> reasoning blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()
    # Extract first JSON object if extra text surrounds it
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = {**_REASONING_DEFAULTS, "recommendation": text, "reasoning": text}
    for k, v in _REASONING_DEFAULTS.items():
        parsed.setdefault(k, v)
    parsed.setdefault("agrees_with_system", parsed.get("decision") == "APPROVE")
    return parsed


def _claude_reasoning_layer(candidate: dict, report_payload: dict,
                            system_prompt: str | None = None) -> dict | None:
    """Run the context-only referee using Claude (Anthropic API)."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        import anthropic as _anthropic
    except ImportError:
        return None

    model = os.getenv("CLAUDE_REASONING_MODEL", "claude-haiku-4-5-20251001").strip()
    user_prompt = json.dumps(_candidate_context_packet(candidate, report_payload), ensure_ascii=True)
    effective_prompt = system_prompt if system_prompt is not None else _REASONING_SYSTEM_PROMPT

    client = _anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=700,
        system=effective_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=0.2,
    )
    raw = (response.content[0].text if response.content else "").strip()
    parsed = _parse_reasoning_json(raw)
    return {
        "provider": "anthropic",
        "model": model,
        "content": parsed,
        "raw": raw,
    }


def _reasoning_layer(candidate: dict, report_payload: dict,
                     system_prompt: str | None = None) -> tuple[dict | None, str | None]:
    """Try Claude first, fall back to OpenRouter. Returns (result, error_str).
    Pass system_prompt to override the default referee prompt (e.g. for longshot mode)."""
    if os.getenv("ANTHROPIC_API_KEY", "").strip():
        try:
            result = _claude_reasoning_layer(candidate, report_payload, system_prompt=system_prompt)
            if result is not None:
                return result, None
        except Exception:
            pass  # Fall through to OpenRouter

    if os.getenv("OPENROUTER_API_KEY", "").strip():
        try:
            result = _openrouter_reasoning_layer(candidate, report_payload, system_prompt=system_prompt)
            return result, None
        except Exception as exc:
            return None, str(exc)

    return None, "No LLM API key configured (set ANTHROPIC_API_KEY or OPENROUTER_API_KEY)"


# ══════════════════════════════════════════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    app_js = BASE / "webapp" / "static" / "app.js"
    style_css = BASE / "webapp" / "static" / "style.css"
    return render_template(
        "index.html",
        app_js_v=int(app_js.stat().st_mtime) if app_js.exists() else 0,
        style_css_v=int(style_css.stat().st_mtime) if style_css.exists() else 0,
    )


# ══════════════════════════════════════════════════════════════════════════════
# API — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/dashboard")
def api_dashboard():
    """Return today's summary + api usage for the header bar."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    usage_path   = BASE / "data" / "api_usage.json"

    summary, summary_path, summary_date, summary_error = _load_summary_report()

    # Try to sync hybrid quota to api_usage.json
    if quota_bridge:
        try:
            quota_bridge.save_legacy_api_usage()
        except Exception:
            pass  # Don't block if sync fails

    usage = {}
    if usage_path.exists():
        usage = json.loads(usage_path.read_text())

    bankroll = float(os.getenv("INITIAL_BANKROLL", 1000))
    odds_key_pool = _odds_key_pool_summary()
    odds_snapshot = _resolve_odds_dashboard_snapshot(usage, odds_key_pool)
    odds_remaining = odds_snapshot.get("remaining", 500)
    odds_start = odds_snapshot.get("start", 500)
    used_today = odds_snapshot.get("used_today", 0)
    used_total = odds_snapshot.get("used_total", 0)
    odds_budget = _budget_snapshot_from_remaining(odds_remaining, monthly_limit=0, reserve=0)
    daily_allowance = odds_budget.get("daily_allowance")
    if isinstance(odds_remaining, int) and odds_remaining <= 0:
        quota_mode = "critical"
    elif isinstance(odds_remaining, int) and odds_remaining <= 100:
        quota_mode = "caution"
    else:
        quota_mode = "healthy"

    tracker_summary = compute_summary()
    settled = get_settled()
    process_by_sport = {}
    if not settled.empty and "sport" in settled.columns:
        resolved = settled[settled["status"].isin(["won", "lost"])].copy()
        if not resolved.empty:
            for sport, group in resolved.groupby("sport"):
                clv_series = pd.to_numeric(group.get("clv"), errors="coerce").dropna()
                process_by_sport[str(sport)] = {
                    "bets": int(len(group)),
                    "avg_edge": round(float(pd.to_numeric(group.get("edge"), errors="coerce").mean()), 4) if "edge" in group else None,
                    "roi": round(float(pd.to_numeric(group.get("profit_units"), errors="coerce").sum() / pd.to_numeric(group.get("stake_units"), errors="coerce").sum()), 4)
                    if pd.to_numeric(group.get("stake_units"), errors="coerce").sum() > 0 else None,
                    "avg_clv": round(float(clv_series.mean()), 4) if not clv_series.empty else None,
                    "clv_positive_pct": round(float((clv_series > 0).mean()), 4) if not clv_series.empty else None,
                }

    # Daily P&L for mini chart (per-day, not cumulative)
    try:
        _pnl_df = _daily_pnl()
        daily_pnl_list = [
            {"date": str(row["date"]), "profit_units": round(float(row["profit_units"]), 4)}
            for _, row in _pnl_df.iterrows()
        ] if not _pnl_df.empty else []
    except Exception:
        daily_pnl_list = []

    # Model tags from data/models/*/current_tag.txt
    models_dir = BASE / "data" / "models"
    model_tags = {}
    for sport_dir in sorted(models_dir.iterdir()) if models_dir.exists() else []:
        if not sport_dir.is_dir():
            continue
        tag_file = sport_dir / "current_tag.txt"
        if tag_file.exists():
            model_tags[sport_dir.name] = tag_file.read_text().strip()

    # Total games scanned today should come from the saved game slates, not
    # from published bet counts. On quiet days the board can have 0 value bets
    # but still dozens of scanned games.
    by_sport = summary.get("single_bets", {}).get("by_sport", {})
    total_games = (
        summary.get("total_games")
        or (
            len(summary.get("soccer_games", []))
            + len(summary.get("other_games", []))
        )
        or sum(by_sport.values())
        or 0
    )

    # Build response with both legacy and hybrid quota data
    response_data = {
        "date":            summary_date or today,
        "requested_date":  today,
        "summary_date":    summary_date or "",
        "summary_available_dates": _available_summary_dates(),
        "summary_error":   summary_error,
        "bankroll":        bankroll,
        "total_bets":      summary.get("single_bets", {}).get("total", 0),
        "by_sport":        by_sport,
        "scan_notes":      summary.get("scan_notes", []),
        "odds_remaining":  odds_remaining,
        "odds_used_today": used_today,
        "odds_used_total": used_total,
        "odds_start":      odds_start,
        "scan_time":       summary.get("timestamp", ""),
        "odds_daily_allowance": daily_allowance,
        "odds_days_left_in_cycle": odds_budget.get("days_left_in_cycle"),
        "odds_remaining_after_reserve": odds_budget.get("remaining_after_reserve"),
        "odds_reserve": 0,
        "odds_unlimited_mode": True,
        "quota_mode": quota_mode,
        "odds_display_source": odds_snapshot.get("display_source"),
        "odds_display_key_fingerprint": odds_snapshot.get("display_key_fingerprint"),
        "odds_usage_key_fingerprint": odds_snapshot.get("usage_key_fingerprint"),
        "odds_usage_sync_status": odds_snapshot.get("usage_sync_status"),
        "odds_selection_reason": odds_snapshot.get("selection_reason"),
        "market_policy": summary.get("market_policy") or summarize_market_policy(),
        "focused_prediction_lanes": summary.get("focused_prediction_lanes") or summarize_focused_prediction_policy(),
        "process_summary": tracker_summary,
        "process_by_sport": process_by_sport,
        "daily_pnl":       daily_pnl_list,
        "model_tags":      model_tags,
        "total_games":     total_games,
        "odds_key_pool":   odds_key_pool,
    }

    # Add hybrid quota info if available
    if quota_bridge:
        try:
            quota_status = quota_bridge.get_quota_status()
            response_data["quota"] = {
                "betfair": quota_status.get("betfair", {}),
                "odds_api": quota_status.get("odds_api", {}),
                "api_football": quota_status.get("api_football", {}),
                "warning_level": quota_bridge.get_warning_level(),
            }
        except Exception:
            pass  # Silently ignore if quota bridge fails

    return jsonify(response_data)


# ══════════════════════════════════════════════════════════════════════════════
# API — PICKS
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/soccer/games")
def api_soccer_games():
    """Return all soccer games with full outcome breakdown (all 5 bet types)."""
    requested_date = request.args.get("date", "").strip() or None
    summary, _, summary_date, summary_error = _load_summary_report(requested_date, fallback_latest=not bool(requested_date))
    if not summary:
        return jsonify({"games": [], "error": summary_error or "No scan summary available.", "summary_date": summary_date})
    games = [_apply_capability_to_game(_apply_timing_to_game(game, "soccer")) for game in summary.get("soccer_games", [])]
    # Filter to window if requested
    window = request.args.get("window")
    if window:
        games = [g for g in games if g.get("window") == window]
    games = sorted(games, key=lambda g: g.get("commence", ""))
    return jsonify({"games": games, "total": len(games), "summary_date": summary_date, "requested_date": requested_date or ""})


@app.route("/api/games")
def api_games():
    """
    Return all scanned games as a flat list, deduped by match.
    Includes soccer (from soccer_games, which has all 36 games) plus
    all other sports derived from the value bets (every bet = a game).
    Supports ?sport=soccer&market=moneyline&window=today filters.
    """
    requested_date = request.args.get("date", "").strip() or None
    summary, _, summary_date, summary_error = _load_summary_report(requested_date, fallback_latest=not bool(requested_date))
    if not summary:
        return jsonify({"games": [], "error": summary_error or "No scan summary available.", "summary_date": summary_date})

    sport_filter  = request.args.get("sport")
    request.args.get("market")  # accepted but not yet used as a filter
    window_filter = request.args.get("window")
    date_filter = request.args.get("date", "").strip()

    games = []
    seen  = set()   # dedup key: (sport, home, away)

    # ── Soccer: full game list (all 36 games, not just value bets) ────────────
    if not sport_filter or sport_filter == "soccer":
        for g in summary.get("soccer_games", []):
            key = ("soccer", g.get("home", ""), g.get("away", ""))
            if key in seen:
                continue
            seen.add(key)
            # Find model's top single-outcome pick
            pick_label = None
            best_prob  = -1
            for o in g.get("outcomes", []):
                if o["label"] in ("Home Win", "Draw", "Away Win") and o.get("ml_prob") is not None and o["ml_prob"] > best_prob and o.get("odds", 0) >= 1.01:
                    best_prob  = o["ml_prob"]
                    pick_label = o["label"]
            # Best odds for each single outcome
            odds_map = {o["label"]: o.get("odds") for o in g.get("outcomes", [])}
            game_row = _apply_capability_to_game(_apply_timing_to_game({
                "sport":      "soccer",
                "home":       g.get("home"),
                "away":       g.get("away"),
                "league":     g.get("league", "Soccer"),
                "league_key": g.get("league_key", ""),
                "kick_off":   g.get("kick_off", ""),
                "window":     g.get("window", ""),
                "commence":   g.get("commence", ""),
                "model_pick": pick_label,
                "model_available": g.get("model_available", False),
                "home_odds":  odds_map.get("Home Win"),
                "draw_odds":  odds_map.get("Draw"),
                "away_odds":  odds_map.get("Away Win"),
                "has_value":  any(o.get("has_value") for o in g.get("outcomes", [])),
            }, "soccer"))
            if window_filter and not _window_matches(game_row, window_filter):
                continue
            if date_filter and _event_local_date_str(game_row.get("commence")) != date_filter:
                continue
            games.append(game_row)

    # ── Other sports: use other_games list (all games, not just value bets) ──
    for g in summary.get("other_games", []):
        sport = g.get("sport", "")
        if sport_filter and sport != sport_filter:
            continue

        home = g.get("home", "")
        away = g.get("away", "")
        key  = (sport, home, away)
        if key in seen:
            continue
        seen.add(key)
        game_row = _apply_capability_to_game(_apply_timing_to_game({
            "sport":    sport,
            "home":     home,
            "away":     away,
            "league":   g.get("league", sport.upper()),
            "league_key": g.get("league_key", ""),
            "kick_off": g.get("kick_off", ""),
            "window":   g.get("window", ""),
            "commence": g.get("commence", ""),
            "model_available": bool(g.get("model_available", False)),
            "model_pick": g.get("model_pick"),
            "home_odds":    g.get("home_odds"),
            "away_odds":    g.get("away_odds"),
            "has_value":    False,
            "is_playoff":   g.get("is_playoff", 0),
            "rest_advantage": g.get("rest_advantage", 0),
            "abstain":      g.get("abstain", False),
            "mlb_probability_debug": g.get("mlb_probability_debug"),
            "basketball_probability_debug": g.get("basketball_probability_debug"),
            "nhl_probability_debug": g.get("nhl_probability_debug"),
        }, sport))
        if window_filter and not _window_matches(game_row, window_filter):
            continue
        if date_filter and _event_local_date_str(game_row.get("commence")) != date_filter:
            continue
        games.append(game_row)

    # Also check value bets to mark has_value (and forward new signals) on matching games
    bets = _annotate_bets(summary.get("single_bets", {}).get("bets", []))
    for b in bets:
        sport = b.get("sport", "soccer")
        if sport == "soccer":
            continue  # already handled
        # Find the matching game and flag it as having value
        for g in games:
            if (g["sport"] == sport and g["home"] == b.get("home") and
                    g["away"] == b.get("away")):
                g["has_value"] = True
                # Elevate upgrade signals from bet to game row if present
                if b.get("is_playoff"):
                    g["is_playoff"] = b["is_playoff"]
                if b.get("abstain"):
                    g["abstain"] = True
                break

    # Sort by commence time
    games.sort(key=lambda g: g.get("commence", ""))

    return jsonify({
        "games": games,
        "total": len(games),
        "summary_date": summary_date,
        "requested_date": requested_date or "",
        "available_dates": _available_summary_dates(),
    })


@app.route("/api/picks")
def api_picks():
    """Return value bets from the selected or latest available summary report."""
    requested_date = request.args.get("date", "").strip() or None
    summary, _, summary_date, summary_error = _load_summary_report(requested_date, fallback_latest=not bool(requested_date))
    if not summary:
        return jsonify(_json_safe({
            "bets": [],
            "review_bets": [],
            "total": 0,
            "review_total": 0,
            "error": summary_error or "No scan summary available.",
            "summary_date": summary_date,
            "requested_date": requested_date or "",
            "available_dates": _available_summary_dates(),
        }))
    bets = [
        _apply_timing_to_bet(bet)
        for bet in _attach_prediction_ids(_annotate_bets(summary.get("single_bets", {}).get("bets", [])))
    ]
    review_bets = [
        _apply_timing_to_bet(bet)
        for bet in _attach_prediction_ids(_annotate_bets(summary.get("single_bets", {}).get("review_bets", [])))
    ]

    bets, review_bets = _split_launch_safe_bets(bets, review_bets)
    sport_scan_counts = _sport_scan_counts_from_summary(summary)
    sport_funnel = _sport_funnel_from_summary(summary)
    market_coverage = _market_coverage_from_summary(summary)
    scan_notes = list(summary.get("scan_notes", []) or [])
    if sport_scan_counts:
        scan_notes.append({
            "type": "sport_scan_counts",
            "counts": sport_scan_counts,
            "reason": "Current full-game scan coverage by supported sport lane.",
        })
    if market_coverage:
        scan_notes.append({
            "type": "market_coverage",
            "by_sport": market_coverage,
            "reason": "Markets actually seen in the latest scan, even when no picks published.",
        })
    if sport_funnel:
        scan_notes.append({
            "type": "sport_funnel",
            "by_sport": sport_funnel,
            "reason": "How each sport moved from scanned games into published, review, suppressed, or no-candidate buckets.",
        })

    # Filters from query params
    sports  = request.args.getlist("sport")    # e.g. ?sport=nhl&sport=mlb
    markets = request.args.getlist("market")   # e.g. ?market=totals&market=spreads
    windows = request.args.getlist("window")   # e.g. ?window=today
    date_filter = request.args.get("date", "").strip()

    if sports:
        bets = [b for b in bets if b.get("sport") in sports]
    if markets:
        def _market(b):
            return b.get("market", "moneyline")
        bets = [b for b in bets if _market(b) in markets]
    if windows:
        bets = [b for b in bets if any(_window_matches(b, window) for window in windows)]
        review_bets = [b for b in review_bets if any(_window_matches(b, window) for window in windows)]
    if date_filter:
        bets = [b for b in bets if _event_local_date_str(b.get("commence") or b.get("commence_time")) == date_filter]
        review_bets = [b for b in review_bets if _event_local_date_str(b.get("commence") or b.get("commence_time")) == date_filter]
    if sports:
        review_bets = [b for b in review_bets if b.get("sport") in sports]
    if markets:
        review_bets = [b for b in review_bets if b.get("market", "moneyline") in markets]

    # Sort by edge descending
    bets = sorted(
        bets,
        key=lambda b: (b.get("market_priority_score", 0), b.get("edge", 0)),
        reverse=True,
    )
    review_bets = sorted(
        review_bets,
        key=lambda b: (b.get("market_priority_score", 0), b.get("edge", 0)),
        reverse=True,
    )

    return jsonify(_json_safe({
        "bets": bets,
        "review_bets": review_bets,
        "total": len(bets),
        "review_total": len(review_bets),
        "scan_notes": scan_notes,
        "sport_scan_counts": sport_scan_counts,
        "market_coverage": market_coverage,
        "sport_funnel": sport_funnel,
        "market_policy": summary.get("market_policy") or summarize_market_policy(),
        "focused_prediction_lanes": summary.get("focused_prediction_lanes") or summarize_focused_prediction_policy(),
        "summary_date": summary_date,
        "requested_date": requested_date or "",
        "available_dates": _available_summary_dates(),
    }))


@app.route("/api/reasoning/candidates")
def api_reasoning_candidates():
    """Return today's eligible value bets for the separate reasoning scan."""
    bets = [bet for bet in _today_reasoning_bets() if bool(bet.get("reasoning_supported", True))]
    candidates = []
    for bet in bets:
        candidates.append({
            "id": _reasoning_candidate_id(bet),
            "sport": bet.get("sport"),
            "market": bet.get("market"),
            "selection": bet.get("team"),
            "home_team": bet.get("home"),
            "away_team": bet.get("away"),
            "odds": bet.get("odds"),
            "edge": bet.get("edge"),
            "bookmaker": bet.get("bookmaker"),
            "kick_off": bet.get("kick_off"),
            "league": bet.get("league"),
            "league_key": bet.get("league_key", ""),
            "launch_label": bet.get("launch_label", ""),
            "launch_note": bet.get("launch_note", ""),
            "minimum_acceptable_odds": bet.get("minimum_acceptable_odds"),
            "odds_recheck_status": bet.get("odds_recheck_status"),
            "odds_recheck_delta": bet.get("odds_recheck_delta"),
            "fair_prob": bet.get("fair_prob"),
            "market_implied_prob": bet.get("market_implied_prob"),
            "fair_odds": bet.get("fair_odds"),
            "availability_summary": bet.get("availability_summary", ""),
            "context_adjustments": bet.get("context_adjustments") or [],
            "prediction_factors": bet.get("prediction_factors") or [],
            "true_probability": bet.get("true_probability") or {},
            "scraped_context": bet.get("scraped_context") or {},
            "scraped_context_highlights": bet.get("scraped_context_highlights") or [],
            "scraped_context_sources": bet.get("scraped_context_sources") or [],
            "context_referee_decision": bet.get("context_referee_decision", ""),
            "context_referee_reason": bet.get("context_referee_reason", ""),
            "decision_status": bet.get("decision_status", ""),
            "decision_reason": bet.get("decision_reason", ""),
            "committee_final_decision": bet.get("committee_final_decision", ""),
            "committee_agreement_status": bet.get("committee_agreement_status", ""),
            "committee_veto_flags": bet.get("committee_veto_flags", []),
            "committee_reason": bet.get("committee_reason", ""),
            "committee_better_substitute": bet.get("committee_better_substitute", ""),
            "committee_parlay_suitability": bet.get("committee_parlay_suitability", ""),
            "committee": bet.get("committee") or {},
            "committee_details_text": bet.get("committee_details_text", ""),
            "suppression_reason": bet.get("suppression_reason", ""),
            "review_required": bool(bet.get("review_required")),
            "review_reason": bet.get("review_reason", ""),
            "display": _reasoning_display_label(bet),
        })
    return jsonify(_json_safe({"candidates": candidates, "total": len(candidates)}))


@app.route("/api/market-policy")
def api_market_policy():
    return jsonify(summarize_market_policy())


@app.route("/api/launch-config")
def api_launch_config():
    return jsonify({
        "positioning": {
            "default": "Multi-sport production platform",
            "supported_scope": {
                "soccer": "full current supported league set",
                "basketball": "full NBA slate",
                "mlb": "full MLB slate",
                "nhl": "full NHL slate",
                "tennis": "full ATP slate",
                "tennis_wta": "full WTA slate",
            },
        },
        "production_path": [
            ".venv/bin/python retrain_and_calibrate.py --sports soccer",
            ".venv/bin/python daily_scan.py --sport soccer",
            ".venv/bin/python webapp/app.py",
        ],
        "notes": [
            "Soccer naturally shows more games because it spans many supported leagues, while NBA, MLB, NHL, ATP, and WTA each map to a smaller number of competitions.",
            "Per-sport scan counts reflect what the latest report actually scanned, not just what was published as picks.",
        ],
    })


# ══════════════════════════════════════════════════════════════════════════════
# API — PARLAYS
# ══════════════════════════════════════════════════════════════════════════════

_MANUAL_PARLAYS_FILE  = BASE / "data" / "tracker" / "manual_parlays.json"
_MY_SELECTIONS_FILE   = BASE / "data" / "tracker" / "my_selections.json"


# ══════════════════════════════════════════════════════════════════════════════
# MY SELECTIONS — user-tracked bets for the Results tab
# ══════════════════════════════════════════════════════════════════════════════

def _load_my_selections() -> list:
    """Load the list of bets the user has explicitly chosen to track."""
    if _MY_SELECTIONS_FILE.exists():
        try:
            return json.loads(_MY_SELECTIONS_FILE.read_text())
        except Exception:
            pass
    return []


def _save_my_selections(selections: list) -> None:
    _MY_SELECTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _MY_SELECTIONS_FILE.write_text(json.dumps(selections, indent=2))


@app.route("/api/my-selections", methods=["GET"])
def api_get_my_selections():
    return jsonify({"selections": _load_my_selections()})


@app.route("/api/my-selections", methods=["POST"])
def api_add_my_selection():
    """
    Add a bet to My Selections.
    Body: {
      id: str,              # unique key (pred_id for system bets, custom for soccer)
      sport: str,
      team: str,            # the pick label / team name
      match: str,           # "Home vs Away"
      odds: float,
      edge: float,
      ml_prob: float,
      kick_off: str,
      date: str,            # YYYY-MM-DD when selection was made
      source: str,          # "pick" | "soccer_outcome"
      pred_id?: str,        # set when it maps to a tracked prediction
    }
    """
    data = request.get_json()
    if not data or not data.get("id"):
        return jsonify({"error": "Missing id"}), 400
    selections = _load_my_selections()
    # Prevent duplicates
    if any(s["id"] == data["id"] for s in selections):
        return jsonify({"ok": True, "duplicate": True})
    selections.append({
        "id":       data["id"],
        "sport":    data.get("sport", ""),
        "team":     data.get("team", ""),
        "match":    data.get("match", ""),
        "odds":     data.get("odds"),
        "edge":     data.get("edge"),
        "ml_prob":  data.get("ml_prob"),
        "kick_off": data.get("kick_off", ""),
        "commence": data.get("commence") or data.get("commence_time") or "",
        "date":     data.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d")),
        "source":   data.get("source", "pick"),
        "pred_id":  data.get("pred_id"),
        "result":   None,    # None = pending, "won" | "lost" once settled
        "profit":   None,
    })
    _save_my_selections(selections)
    return jsonify({"ok": True, "total": len(selections)})


@app.route("/api/my-selections/<sel_id>", methods=["DELETE"])
def api_remove_my_selection(sel_id):
    selections = _load_my_selections()
    selections = [s for s in selections if s["id"] != sel_id]
    _save_my_selections(selections)
    return jsonify({"ok": True, "total": len(selections)})


@app.route("/api/my-selections/<sel_id>/settle", methods=["POST"])
def api_settle_my_selection(sel_id):
    """Mark a user selection as won or lost. Body: { won: bool }"""
    data = request.get_json()
    won  = bool(data.get("won", False))
    selections = _load_my_selections()
    for s in selections:
        if s["id"] == sel_id:
            s["result"]  = "won" if won else "lost"
            odds         = float(s.get("odds") or 1)
            stake        = float(s.get("stake", 1))
            s["profit"]  = round((odds - 1) * stake if won else -stake, 4)
            break
    _save_my_selections(selections)
    return jsonify({"ok": True})


@app.route("/api/my-selections/results", methods=["GET"])
def api_my_selections_results():
    """
    Return stats and bet-by-bet breakdown for the user's own selections.
    Supports ?date=YYYY-MM-DD filter.
    """
    date_filter = request.args.get("date")
    selections  = _load_my_selections()
    if _backfill_selection_prediction_ids(selections):
        _save_my_selections(selections)

    if date_filter:
        selections = [s for s in selections if s.get("date", "").startswith(date_filter)]

    total   = len(selections)
    pending = [s for s in selections if s.get("result") is None]
    settled = [s for s in selections if s.get("result") is not None]
    won     = [s for s in settled if s["result"] == "won"]

    # Cross-reference with the tracker to auto-fill results from settled.parquet
    settled_path = BASE / "data" / "tracker" / "settled.parquet"
    if settled_path.exists():
        try:
            df = pd.read_parquet(settled_path)
            pred_id_set = {s["pred_id"] for s in selections if s.get("pred_id")}
            if pred_id_set and "pred_id" in df.columns:
                df_matched = df[df["pred_id"].isin(pred_id_set)]
                for _, row in df_matched.iterrows():
                    pid = row["pred_id"]
                    for s in selections:
                        if s.get("pred_id") == pid and s.get("result") is None:
                            s["result"] = "won" if row.get("won") else "lost"
                            odds  = float(s.get("odds") or 1)
                            s["profit"] = round((odds - 1) if row.get("won") else -1, 4)
        except Exception:
            pass

    pending = [_apply_capability_to_game(_apply_timing_to_bet(s)) for s in selections if s.get("result") is None]
    settled = [_apply_capability_to_game(_apply_timing_to_bet(s)) for s in selections if s.get("result") is not None]
    won     = [s for s in settled if s["result"] == "won"]
    total_profit = sum(s.get("profit") or 0 for s in settled)
    stake_total  = float(len(settled))
    roi          = (total_profit / stake_total * 100) if stake_total else 0

    # Recompute after auto-fill
    pending = [s for s in selections if s.get("result") is None]
    settled = [s for s in selections if s.get("result") is not None]
    won     = [s for s in settled if s["result"] == "won"]
    for s in pending:
        sport = str(s.get("sport", "")).lower()
        if sport in ("tennis", "tennis_wta"):
            s["settlement_note"] = "Tennis auto-settle is not available yet; use Won/Lost manually."
        elif s.get("pred_id"):
            s["settlement_note"] = "Linked to System Book; Check & Settle can update this once the score source returns a final."
        else:
            s["settlement_note"] = "Not linked to a System Book prediction; use Won/Lost manually."
    total_profit = sum(s.get("profit") or 0 for s in settled)
    stake_total  = float(len(settled))
    roi          = (total_profit / stake_total * 100) if stake_total else 0

    # By-sport breakdown
    by_sport = {}
    for s in settled:
        sport = s.get("sport", "other")
        if sport not in by_sport:
            by_sport[sport] = {"wins": 0, "total": 0, "pnl": 0}
        by_sport[sport]["total"] += 1
        by_sport[sport]["pnl"]   += s.get("profit") or 0
        if s["result"] == "won":
            by_sport[sport]["wins"] += 1
    for sport, v in by_sport.items():
        t = v["total"]
        by_sport[sport]["win_rate"] = round(v["wins"] / t * 100, 1) if t else 0
        by_sport[sport]["roi"]      = round(v["pnl"] / t * 100, 2) if t else 0
        by_sport[sport]["pnl"]      = round(v["pnl"], 4)

    return jsonify({
        "selections": selections,
        "overall": {
            "total":    total,
            "pending":  len(pending),
            "settled":  len(settled),
            "won":      len(won),
            "win_rate": round(len(won) / len(settled) * 100, 1) if settled else 0,
            "pnl":      round(total_profit, 4),
            "roi":      round(roi, 2),
        },
        "by_sport":  by_sport,
        "pending":   pending,
        "settled":   settled,
        "date_filter": date_filter,
    })


@app.route("/api/my-selections/settle-all", methods=["POST"])
def api_settle_my_selections_all():
    """Automatically settle tracked bets by matching them directly to score feeds."""
    import re as _re
    import requests as _req
    from datetime import timedelta as _td

    selections = _load_my_selections()
    pending = [s for s in selections if s.get("result") is None]
    if not pending:
        return jsonify({"settled": 0, "still_pending": 0, "total_profit": 0, "results": [], "message": "No tracked bets pending."})

    sport_keys = {
        "basketball": "basketball_nba",
        "nhl": "icehockey_nhl",
        "mlb": "baseball_mlb",
        "soccer": None,
        "tennis": None,
        "tennis_wta": None,
    }
    sports_needed = sorted({str(s.get("sport", "")).lower() for s in pending})
    scores_by_sport: dict[str, list[dict]] = {}
    errors: list[str] = []
    odds_key = get_primary_odds_api_key()

    for sport in sports_needed:
        api_sport = sport_keys.get(sport)
        if not api_sport:
            continue
        try:
            resp = _req.get(
                f"https://api.the-odds-api.com/v4/sports/{api_sport}/scores/",
                params={"apiKey": odds_key, "daysFrom": 3},
                timeout=10,
            )
            if resp.status_code != 200:
                errors.append(f"{sport}: HTTP {resp.status_code}")
                continue
            scores_by_sport[sport] = [g for g in resp.json() if g.get("completed") and g.get("scores")]
        except Exception as exc:
            errors.append(f"{sport}: {exc}")

    if "soccer" in sports_needed:
        fd_key = os.getenv("FOOTBALL_DATA_API_KEY", "")
        if not fd_key:
            errors.append("soccer: FOOTBALL_DATA_API_KEY not set")
        else:
            try:
                resp = _req.get(
                    "https://api.football-data.org/v4/matches",
                    headers={"X-Auth-Token": fd_key},
                    params={
                        "status": "FINISHED",
                        "dateFrom": (datetime.now(timezone.utc) - _td(days=4)).strftime("%Y-%m-%d"),
                        "dateTo": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    },
                    timeout=10,
                )
                if resp.status_code == 200:
                    games = []
                    for m in resp.json().get("matches", []):
                        ft = m.get("score", {}).get("fullTime", {})
                        if ft.get("home") is None or ft.get("away") is None:
                            continue
                        home = m.get("homeTeam", {}).get("name", "")
                        away = m.get("awayTeam", {}).get("name", "")
                        games.append({
                            "home_team": home,
                            "away_team": away,
                            "scores": [{"name": home, "score": str(ft["home"])}, {"name": away, "score": str(ft["away"])}],
                            "commence_time": m.get("utcDate", ""),
                            "completed": True,
                        })
                    scores_by_sport["soccer"] = games
                else:
                    errors.append(f"soccer: football-data.org HTTP {resp.status_code}")
            except Exception as exc:
                errors.append(f"soccer: {exc}")

    def _norm(name: str) -> str:
        return _re.sub(r"[^a-z0-9]", " ", str(name).lower()).strip()

    def _team_match(a: str, b: str) -> bool:
        na, nb = _norm(a), _norm(b)
        if na == nb:
            return True
        ta, tb = set(na.split()), set(nb.split())
        if not ta or not tb:
            return False
        shorter, longer = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
        return shorter.issubset(longer)

    def _game_date(game: dict) -> str | None:
        commence = str(game.get("commence_time") or game.get("utcDate") or "").strip()
        if not commence:
            return None
        try:
            return str(pd.Timestamp(commence).date())
        except Exception:
            return commence[:10] if len(commence) >= 10 else None

    def _selection_date(selection: dict) -> str | None:
        for field in ("commence", "commence_time"):
            value = str(selection.get(field) or "").strip()
            if value:
                try:
                    return str(pd.Timestamp(value).date())
                except Exception:
                    if len(value) >= 10:
                        return value[:10]
        value = str(selection.get("date") or "").strip()
        return value[:10] if len(value) >= 10 else None

    def _resolve(selection: dict, game: dict) -> bool | None:
        scores = {s["name"]: int(s["score"]) for s in game.get("scores", [])}
        if len(scores) < 2:
            return None
        pick = str(selection.get("team", ""))
        mtype = _mtype(pick)
        if mtype == "moneyline":
            if pick.lower().strip() == "draw":
                vals = list(scores.values())
                return vals[0] == vals[1]
            for team, score in scores.items():
                if _team_match(pick, team):
                    other = [v for k, v in scores.items() if k != team][0]
                    return score > other
        if mtype == "spreads":
            match = _re.search(r"([+-][\d.]+)\s*$", pick)
            if not match:
                return None
            spread = float(match.group(1))
            team_part = _re.sub(r"[+-][\d.]+\s*$", "", pick).strip()
            for team, score in scores.items():
                if _team_match(team_part, team):
                    other = [v for k, v in scores.items() if k != team][0]
                    adjusted = score + spread
                    return None if adjusted == other else adjusted > other
        if mtype == "totals":
            match = _re.search(r"(Over|Under)\s+([\d.]+)", pick, _re.I)
            if not match:
                return None
            total = sum(scores.values())
            line = float(match.group(2))
            if total == line:
                return None
            return (match.group(1).lower() == "over") == (total > line)
        return None

    settled_count = 0
    total_profit = 0.0
    results = []
    still_pending = 0
    unsupported = []
    for selection in pending:
        sport = str(selection.get("sport", "")).lower()
        if sport in ("tennis", "tennis_wta"):
            unsupported.append("tennis")
            still_pending += 1
            continue
        games = scores_by_sport.get(sport, [])
        if not games:
            still_pending += 1
            continue
        parts = [p.strip() for p in str(selection.get("match", "")).replace(" @ ", " vs ").split(" vs ")]
        if len(parts) < 2:
            still_pending += 1
            continue
        matching_games = []
        for game in games:
            home, away = game.get("home_team", ""), game.get("away_team", "")
            if ((_team_match(parts[0], home) or _team_match(parts[0], away)) and
                    (_team_match(parts[1], home) or _team_match(parts[1], away))):
                matching_games.append(game)
        if not matching_games:
            still_pending += 1
            continue

        expected_date = _selection_date(selection)
        if expected_date:
            dated_matches = [game for game in matching_games if _game_date(game) == expected_date]
            if not dated_matches:
                still_pending += 1
                results.append({
                    "pick": selection.get("team", ""),
                    "sport": sport,
                    "match": selection.get("match", ""),
                    "status": "date_mismatch",
                    "message": f"Found matching teams, but no completed score for event date {expected_date}.",
                })
                continue
            matching_games = dated_matches
        elif len(matching_games) > 1:
            still_pending += 1
            results.append({
                "pick": selection.get("team", ""),
                "sport": sport,
                "match": selection.get("match", ""),
                "status": "ambiguous_date",
                "message": "Multiple completed games matched these teams; add/keep the event timestamp before auto-settling.",
            })
            continue

        matched = sorted(matching_games, key=lambda game: str(game.get("commence_time") or ""))[-1]
        won = _resolve(selection, matched)
        if won is None:
            still_pending += 1
            continue
        odds = float(selection.get("odds") or 1)
        profit = round((odds - 1) if won else -1, 4)
        selection["result"] = "won" if won else "lost"
        selection["profit"] = profit
        selection["settled_at"] = datetime.now(timezone.utc).isoformat()
        settled_count += 1
        total_profit += profit
        results.append({"pick": selection.get("team", ""), "sport": sport, "won": won, "profit": profit, "match": selection.get("match", "")})

    if settled_count:
        _save_my_selections(selections)

    unsupported = sorted(set(unsupported))
    return jsonify({
        "settled": settled_count,
        "still_pending": still_pending,
        "total_profit": round(total_profit, 4),
        "results": results,
        "errors": errors,
        "unsupported": unsupported,
        "message": (
            f"Settled {settled_count} tracked bet{'s' if settled_count != 1 else ''}. "
            f"{still_pending} still pending."
            + (f" ({', '.join(unsupported)} auto-settle not available)" if unsupported else "")
        ),
    })


def _load_manual_parlays() -> list:
    if _MANUAL_PARLAYS_FILE.exists():
        try:
            return json.loads(_MANUAL_PARLAYS_FILE.read_text())
        except Exception:
            pass
    return []


def _save_manual_parlays(parlays: list) -> None:
    _MANUAL_PARLAYS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _MANUAL_PARLAYS_FILE.write_text(json.dumps(parlays, indent=2))


@app.route("/api/parlays")
def api_parlays():
    """Return system parlays for a given date (default: today)."""
    date = request.args.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    report_path  = BASE / "reports" / f"value_bets_{date}.md"
    summary_path = BASE / "reports" / f"summary_{date}.json"

    if not summary_path.exists():
        return jsonify({"parlays": [], "error": f"No scan found for {date}."})

    parlays = _parse_parlays_from_md(report_path) if report_path.exists() else []
    return jsonify({"parlays": parlays, "date": date})


@app.route("/api/parlays/manual", methods=["GET"])
def api_manual_parlays_get():
    """Return all saved manual parlays, optionally filtered by ?date=YYYY-MM-DD."""
    date_filter = request.args.get("date", None)
    all_parlays = _load_manual_parlays()
    if date_filter:
        all_parlays = [p for p in all_parlays if p.get("date", "").startswith(date_filter)]
    return jsonify({"parlays": all_parlays})


@app.route("/api/parlays/manual", methods=["POST"])
def api_manual_parlays_save():
    """Save a manual parlay. Body: { legs, name? }"""
    data = request.get_json()
    legs = data.get("legs", [])
    name = data.get("name", "").strip()
    if len(legs) < 2:
        return jsonify({"error": "Need at least 2 legs"}), 400

    combined_odds = 1.0
    win_prob = 1.0
    for leg in legs:
        combined_odds *= float(leg["odds"])
        win_prob *= float(leg["ml_prob"])

    ev = win_prob * combined_odds
    edge = ev - 1.0
    kelly = max(0, (win_prob * combined_odds - 1) / (combined_odds - 1)) * 0.5
    bankroll = float(os.getenv("INITIAL_BANKROLL", 1000))

    win_pct = win_prob * 100
    if win_pct >= 0.1:
        wp_str = f"{win_pct:.2f}%"
    elif win_pct >= 0.001:
        wp_str = f"{win_pct:.4f}%"
    else:
        one_in = int(round(1 / win_prob)) if win_prob > 0 else 0
        wp_str = f"1 in {one_in:,}"

    import uuid
    parlay = {
        "id":            str(uuid.uuid4())[:8],
        "name":          name or f"{len(legs)}-leg Parlay",
        "date":          datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "saved_at":      datetime.now(timezone.utc).isoformat(),
        "type":          "manual",
        "status":        "pending",   # pending | won | lost
        "n_legs":        len(legs),
        "combined_odds": round(combined_odds, 2),
        "win_prob":      wp_str,
        "ev":            round(ev, 3),
        "edge":          round(edge * 100, 2),
        "kelly_stake":   round(kelly * bankroll, 2),
        "kelly_pct":     round(kelly * 100, 2),
        "legs": [
            {
                "team":     leg.get("team", ""),
                "match":    leg.get("match", ""),
                "sport":    leg.get("sport", ""),
                "odds":     round(float(leg["odds"]), 2),
                "ml_prob":  round(float(leg["ml_prob"]), 4),
                "edge":     round(float(leg.get("edge", 0)), 4),
                "kick_off": leg.get("kick_off", ""),
            }
            for leg in legs
        ],
    }

    all_parlays = _load_manual_parlays()
    all_parlays.append(parlay)
    _save_manual_parlays(all_parlays)
    return jsonify({"saved": True, "parlay": parlay})


@app.route("/api/parlays/manual/<parlay_id>", methods=["PATCH"])
def api_manual_parlays_update(parlay_id):
    """Update a manual parlay's name or status. Body: { name?, status? }"""
    data = request.get_json()
    all_parlays = _load_manual_parlays()
    for p in all_parlays:
        if p.get("id") == parlay_id:
            if "name" in data and data["name"].strip():
                p["name"] = data["name"].strip()
            if "status" in data and data["status"] in ("pending", "won", "lost"):
                p["status"] = data["status"]
            break
    else:
        return jsonify({"error": "Not found"}), 404
    _save_manual_parlays(all_parlays)
    return jsonify({"updated": True})


@app.route("/api/parlays/manual/<parlay_id>", methods=["DELETE"])
def api_manual_parlays_delete(parlay_id):
    """Delete a saved manual parlay by id."""
    all_parlays = _load_manual_parlays()
    all_parlays = [p for p in all_parlays if p.get("id") != parlay_id]
    _save_manual_parlays(all_parlays)
    return jsonify({"deleted": True})


_live_scores_cache: dict = {"ts": 0.0, "payload": None}
_LIVE_SCORES_TTL = 25  # seconds — matches the 30 s poll interval

@app.route("/api/live-scores")
def api_live_scores():
    """Return live/finished scores for today's games (SofaScore, no API key).
    Only events that are in-progress, at half-time, or finished are returned —
    notstarted events are stripped so the browser payload stays small.
    Response is cached for 25 s to avoid hammering SofaScore on every poll."""
    import time as _time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    now = _time.monotonic()
    cached = _live_scores_cache
    if cached["payload"] is not None and now - cached["ts"] < _LIVE_SCORES_TTL:
        return jsonify(cached["payload"])

    today = datetime.now(_APP_TZ).strftime("%Y-%m-%d")
    _SS_UNIQUE = {
        "football":   "soccer",
        "basketball": "basketball",
        "tennis":     "tennis",
        "baseball":   "mlb",
        "ice-hockey": "nhl",
    }
    _SS_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.sofascore.com/",
        "Origin": "https://www.sofascore.com",
    }
    _ACTIVE = {"inprogress", "live", "halftime", "finished", "ended",
               "afterextratime", "afterpenalties", "postponed"}

    def _fetch_sport(ss_sport: str, our_sport: str) -> list[dict]:
        url = f"https://api.sofascore.com/api/v1/sport/{ss_sport}/scheduled-events/{today}"
        try:
            r = requests.get(url, headers=_SS_HEADERS, timeout=8)
            r.raise_for_status()
            out = []
            for ev in (r.json().get("events") or []):
                status_obj = ev.get("status") or {}
                st = status_obj.get("type", "notstarted")
                if st not in _ACTIVE:
                    continue
                home = (ev.get("homeTeam") or {}).get("name", "")
                away = (ev.get("awayTeam") or {}).get("name", "")
                if not home or not away:
                    continue
                time_obj = ev.get("time") or {}
                out.append({
                    "sport":       our_sport,
                    "home":        home,
                    "away":        away,
                    "status_type": st,
                    "home_score":  (ev.get("homeScore") or {}).get("current"),
                    "away_score":  (ev.get("awayScore") or {}).get("current"),
                    "minutes":     time_obj.get("played"),
                    "injury_time": time_obj.get("injuryTime2") or time_obj.get("injuryTime1"),
                })
            return out
        except Exception:
            return []

    events: list[dict] = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futs = {pool.submit(_fetch_sport, ss, our): ss for ss, our in _SS_UNIQUE.items()}
        for fut in as_completed(futs):
            events.extend(fut.result())

    payload = {"events": events, "date": today}
    _live_scores_cache["payload"] = payload
    _live_scores_cache["ts"] = now
    return jsonify(payload)


@app.route("/api/parlays/settle-auto", methods=["POST"])
def api_parlays_settle_auto():
    """Run the automatic parlay settlement engine (same as `python settle.py --parlays`)."""
    try:
        sys.path.insert(0, str(BASE))
        from settle import settle_parlays as _settle_parlays
        from src.utils.results_tracker import get_pending_parlays
        pending_before = len(get_pending_parlays())
        _settle_parlays()
        pending_after = len(get_pending_parlays())
        settled_count = pending_before - pending_after
        return jsonify({
            "ok": True,
            "settled": settled_count,
            "still_pending": pending_after,
        })
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/settle-all", methods=["POST"])
def api_settle_all():
    """Combined fast settlement: settles individual tracked bets AND per-leg manual/AI parlay legs.
    Uses SofaScore (free, cached) for today + yesterday, parallel across all sports.
    Returns bets_settled, parlays_settled, per-parlay leg breakdown."""
    import re as _re
    import time as _time
    import unicodedata as _ud
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from datetime import timedelta as _td

    _SS_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.sofascore.com/",
        "Origin": "https://www.sofascore.com",
    }
    _SS_MAP = {
        "football":   "soccer",
        "basketball": "basketball",
        "tennis":     "tennis",
        "baseball":   "mlb",
        "ice-hockey": "nhl",
    }
    _ODDS_SCORES_MAP = {
        "basketball": "basketball_nba",
        "nhl": "icehockey_nhl",
        "mlb": "baseball_mlb",
    }
    _DONE = {"finished", "ended", "afterextratime", "afterpenalties"}

    payload = request.get_json(silent=True) or {}
    backlog_mode = str(payload.get("mode", "")).lower() == "backlog"
    today_local = datetime.now(_APP_TZ).date()
    today_str = today_local.strftime("%Y-%m-%d")
    lookback_days = 60 if backlog_mode else 45
    lookback_floor = (today_local - _td(days=lookback_days)).strftime("%Y-%m-%d")
    selections  = _load_my_selections()
    pending_sel = [s for s in selections if s.get("result") is None]
    all_parlays = _load_manual_parlays()
    pred_path = BASE / "data" / "tracker" / "predictions.parquet"
    pending_pred_rows = pd.DataFrame()
    if pred_path.exists() and pred_path.stat().st_size > 100:
        try:
            pred_df = pd.read_parquet(pred_path)
            if "status" in pred_df.columns:
                pending_pred_rows = pred_df[pred_df["status"] == "pending"].copy()
        except Exception:
            pending_pred_rows = pd.DataFrame()

    def _ss_day(ss_sport: str, our_sport: str, date_str: str) -> list[dict]:
        url = f"https://api.sofascore.com/api/v1/sport/{ss_sport}/scheduled-events/{date_str}"
        try:
            r = requests.get(url, headers=_SS_HEADERS, timeout=8)
            r.raise_for_status()
            out = []
            for ev in (r.json().get("events") or []):
                st = (ev.get("status") or {}).get("type", "")
                if st not in _DONE:
                    continue
                home = (ev.get("homeTeam") or {}).get("name", "")
                away = (ev.get("awayTeam") or {}).get("name", "")
                hs   = (ev.get("homeScore") or {}).get("current")
                as_  = (ev.get("awayScore") or {}).get("current")
                if not home or not away or hs is None or as_ is None:
                    continue
                out.append({"sport": our_sport, "home": home, "away": away,
                            "home_score": int(hs), "away_score": int(as_), "status": st,
                            "event_date": date_str})
            return out
        except Exception:
            return []

    def _espn_day(league_sport: str, league: str, our_sport: str, date_str: str) -> list[dict]:
        date_compact = date_str.replace("-", "")
        url = f"https://site.api.espn.com/apis/site/v2/sports/{league_sport}/{league}/scoreboard"
        try:
            resp = requests.get(url, params={"dates": date_compact}, timeout=10)
            resp.raise_for_status()
            out: list[dict] = []
            for event in resp.json().get("events", []) or []:
                status_type = str((((event.get("status") or {}).get("type") or {}).get("name") or "")).lower()
                if status_type not in _DONE and status_type not in {"status_final", "final"}:
                    continue
                competitors = (((event.get("competitions") or [{}])[0]).get("competitors") or [])
                if len(competitors) < 2:
                    continue

                def _comp_name(comp: dict) -> str:
                    team = comp.get("team") or {}
                    athlete = comp.get("athlete") or {}
                    return str(team.get("displayName") or athlete.get("displayName") or comp.get("displayName") or "")

                def _comp_score(comp: dict) -> int | None:
                    score = comp.get("score")
                    if score not in (None, ""):
                        try:
                            return int(score)
                        except Exception:
                            pass
                    linescores = comp.get("linescores") or []
                    if linescores:
                        try:
                            return sum(int(item.get("value", 0)) for item in linescores)
                        except Exception:
                            return None
                    if comp.get("winner") is True:
                        return 1
                    if comp.get("winner") is False:
                        return 0
                    return None

                home_comp = next((c for c in competitors if str(c.get("homeAway", "")).lower() == "home"), None)
                away_comp = next((c for c in competitors if str(c.get("homeAway", "")).lower() == "away"), None)
                if home_comp is None or away_comp is None:
                    home_comp, away_comp = competitors[0], competitors[1]

                home = _comp_name(home_comp)
                away = _comp_name(away_comp)
                home_score = _comp_score(home_comp)
                away_score = _comp_score(away_comp)
                if not home or not away or home_score is None or away_score is None:
                    continue

                out.append({
                    "sport": our_sport,
                    "home": home,
                    "away": away,
                    "home_score": int(home_score),
                    "away_score": int(away_score),
                    "status": "finished",
                    "event_date": date_str,
                })
            return out
        except Exception:
            return []

    def _fallback_finished_events(sports_needed: set[str], date_from: str, date_to: str) -> tuple[list[dict], list[str], list[str]]:
        import requests as _req

        fallback_events: list[dict] = []
        fallback_errors: list[str] = []
        fallback_sources: list[str] = []
        odds_key = get_primary_odds_api_key()

        days_from = 3
        try:
            days_from = max(3, min(21, (pd.Timestamp(date_to).date() - pd.Timestamp(date_from).date()).days + 1))
        except Exception:
            pass

        for sport, odds_key_name in _ODDS_SCORES_MAP.items():
            if sport not in sports_needed:
                continue
            try:
                resp = _req.get(
                    f"https://api.the-odds-api.com/v4/sports/{odds_key_name}/scores/",
                    params={"apiKey": odds_key, "daysFrom": days_from},
                    timeout=10,
                )
                resp.raise_for_status()
                fallback_sources.append("odds_api")
                for game in resp.json() or []:
                    if not game.get("completed") or not game.get("scores"):
                        continue
                    scores = {
                        str(item.get("name", "")): item.get("score")
                        for item in (game.get("scores") or [])
                        if item.get("name") not in (None, "")
                    }
                    if len(scores) < 2:
                        continue
                    teams = list(scores.keys())
                    try:
                        home_score = int(scores[teams[0]])
                        away_score = int(scores[teams[1]])
                    except Exception:
                        continue
                    fallback_events.append({
                        "sport": sport,
                        "home": game.get("home_team", teams[0]),
                        "away": game.get("away_team", teams[1]),
                        "home_score": home_score,
                        "away_score": away_score,
                        "status": "finished",
                        "event_date": _event_date_value(game.get("commence_time")) or date_to,
                    })
            except Exception as exc:
                fallback_errors.append(f"{sport}: {exc}")

        if "soccer" in sports_needed:
            fd_key = os.getenv("FOOTBALL_DATA_API_KEY", "")
            if not fd_key:
                fallback_errors.append("soccer: FOOTBALL_DATA_API_KEY not set")
            else:
                try:
                    resp = _req.get(
                        "https://api.football-data.org/v4/matches",
                        headers={"X-Auth-Token": fd_key},
                        params={"status": "FINISHED", "dateFrom": date_from, "dateTo": date_to},
                        timeout=10,
                    )
                    resp.raise_for_status()
                    fallback_sources.append("football_data")
                    for match in resp.json().get("matches", []):
                        ft = match.get("score", {}).get("fullTime", {})
                        if ft.get("home") is None or ft.get("away") is None:
                            continue
                        fallback_events.append({
                            "sport": "soccer",
                            "home": (match.get("homeTeam") or {}).get("name", ""),
                            "away": (match.get("awayTeam") or {}).get("name", ""),
                            "home_score": int(ft["home"]),
                            "away_score": int(ft["away"]),
                            "status": "finished",
                            "event_date": _event_date_value(match.get("utcDate")) or date_to,
                        })
                except Exception as exc:
                    fallback_errors.append(f"soccer: {exc}")

        espn_map = {
            "basketball": ("basketball", "nba"),
            "nhl": ("hockey", "nhl"),
            "mlb": ("baseball", "mlb"),
            "tennis": ("tennis", "atp"),
            "tennis_wta": ("tennis", "wta"),
        }
        for sport, (league_sport, league) in espn_map.items():
            if sport not in sports_needed:
                continue
            added = 0
            for date_str in pd.date_range(date_from, date_to, freq="D").strftime("%Y-%m-%d"):
                day_events = _espn_day(league_sport, league, sport, date_str)
                fallback_events.extend(day_events)
                added += len(day_events)
            if added:
                fallback_sources.append("espn")

        return fallback_events, fallback_errors, sorted(set(fallback_sources))

    # Seed from live-scores cache (finished events already there)
    now = _time.monotonic()
    cached_payload = _live_scores_cache.get("payload")
    finished_events: list[dict] = []
    if cached_payload and now - _live_scores_cache.get("ts", 0) < 300:
        for ev in (cached_payload.get("events") or []):
            if ev.get("status_type") in _DONE and ev.get("home_score") is not None:
                finished_events.append({
                    "sport":      ev["sport"],
                    "home":       ev["home"],
                    "away":       ev["away"],
                    "home_score": int(ev["home_score"]),
                    "away_score": int(ev["away_score"]),
                    "event_date": _event_date_value(ev.get("commence_time")) or today_str,
                })

    target_dates = set()
    if not backlog_mode:
        target_dates.add(today_str)
    if not finished_events and not backlog_mode:
        target_dates.add((today_local - _td(days=1)).strftime("%Y-%m-%d"))
    for selection in pending_sel:
        event_date = _selection_event_date(selection)
        if event_date and lookback_floor <= event_date <= today_str:
            target_dates.add(event_date)
    for parlay in all_parlays:
        if parlay.get("status") in ("won", "lost"):
            continue
        for leg in parlay.get("legs", []):
            if leg.get("result") in ("won", "lost"):
                continue
            event_date = _selection_event_date(leg)
            if event_date and lookback_floor <= event_date <= today_str:
                target_dates.add(event_date)
    if not pending_pred_rows.empty and "commence_time" in pending_pred_rows.columns:
        for value in pending_pred_rows["commence_time"].tolist():
            event_date = _event_date_value(value)
            if event_date and lookback_floor <= event_date <= today_str:
                target_dates.add(event_date)
    if not target_dates:
        target_dates.add(today_str)

    sports_needed = {
        str(item.get("sport", "")).lower()
        for item in pending_sel
        if str(item.get("sport", "")).strip()
    }
    sports_needed.update(
        str(leg.get("sport", "")).lower()
        for parlay in all_parlays
        if parlay.get("status") not in ("won", "lost")
        for leg in parlay.get("legs", [])
        if leg.get("result") not in ("won", "lost") and str(leg.get("sport", "")).strip()
    )
    if not pending_pred_rows.empty and "sport" in pending_pred_rows.columns:
        sports_needed.update(
            str(value).lower()
            for value in pending_pred_rows["sport"].tolist()
            if str(value).strip()
        )
    source_errors: list[str] = []
    score_sources: list[str] = ["sofascore"]

    # Parallel fetch: exact pending-event dates instead of only today/yesterday.
    tasks = [(ss, our, date_str) for date_str in sorted(target_dates) for ss, our in _SS_MAP.items()]

    with ThreadPoolExecutor(max_workers=10) as pool:
        futs = [pool.submit(_ss_day, ss, our, d) for ss, our, d in tasks]
        for fut in as_completed(futs):
            finished_events.extend(fut.result())

    if backlog_mode or not finished_events:
        fallback_events, fallback_errors, fallback_sources = _fallback_finished_events(
            sports_needed=sports_needed,
            date_from=min(target_dates),
            date_to=max(target_dates),
        )
        finished_events.extend(fallback_events)
        source_errors.extend(fallback_errors)
        score_sources.extend(fallback_sources)

    # ── normalisation helpers ────────────────────────────────────────────────
    _TEAM_STOPWORDS = {"fc", "cf", "sc", "ac", "ca", "cd", "club", "de", "the", "team", "afc", "cfc"}
    _TEAM_ALIASES = {
        "utd": "united",
        "st": "saint",
        "st.": "saint",
        "nottm": "nottingham",
    }
    _TEAM_PHRASE_ALIASES = {
        "man utd": "manchester united",
        "man united": "manchester united",
        "man city": "manchester city",
        "psg": "paris saint germain",
        "paris sg": "paris saint germain",
        "spurs": "tottenham hotspur",
        "wolves": "wolverhampton wanderers",
        "qpr": "queens park rangers",
        "spal": "spal",
    }

    def _norm(s: str) -> str:
        text = _ud.normalize("NFKD", str(s or ""))
        text = "".join(ch for ch in text if not _ud.combining(ch)).lower()
        for source, target in _TEAM_PHRASE_ALIASES.items():
            text = text.replace(source, target)
        text = _re.sub(r"[^a-z0-9]", " ", text)
        tokens = [_TEAM_ALIASES.get(tok, tok) for tok in text.split()]
        tokens = [tok for tok in tokens if tok and tok not in _TEAM_STOPWORDS]
        return " ".join(tokens).strip()

    def _compact(s: str) -> str:
        return "".join(_norm(s).split())

    def _tmatch(a: str, b: str) -> bool:
        na, nb = _norm(a), _norm(b)
        if na == nb:
            return True
        if _compact(a) and _compact(a) == _compact(b):
            return True
        ta, tb = set(na.split()), set(nb.split())
        if not ta or not tb:
            return False
        shorter, longer = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
        if shorter.issubset(longer):
            return True
        overlap = len(ta & tb)
        coverage = overlap / max(len(shorter), 1)
        containment = 0.2 if na in nb or nb in na else 0.0
        return (coverage + containment) >= 0.95 and overlap >= 1

    unresolved_rows: list[dict[str, str]] = []

    def _push_unresolved(*, scope: str, sport: str, match: str, pick: str, reason: str, message: str) -> None:
        unresolved_rows.append({
            "scope": scope,
            "sport": sport,
            "match": match,
            "pick": pick,
            "reason": reason,
            "message": message,
        })

    # Pre-index finished events by sport for O(1) bucket lookup
    _idx: dict[str, list[dict]] = {}
    for ev in finished_events:
        sp = ev["sport"]
        _idx.setdefault(sp, []).append(ev)

    def _find_game_match(sport: str, home: str, away: str, event_date: str | None = None):
        sport_events = _idx.get(sport, [])
        if not sport_events:
            return {"status": "no_score_source", "game": None}

        candidates = []
        for ev in sport_events:
            same_order = _tmatch(ev["home"], home) and _tmatch(ev["away"], away)
            swapped_order = _tmatch(ev["home"], away) and _tmatch(ev["away"], home)
            if same_order or swapped_order:
                candidates.append(ev)
        if not candidates:
            return {"status": "team_mismatch", "game": None}

        if event_date:
            dated = [ev for ev in candidates if _game_event_date(ev) == event_date]
            if not dated:
                return {"status": "date_mismatch", "game": None}
            candidates = dated
        elif len(candidates) > 1:
            return {"status": "ambiguous_date", "game": None}

        return {
            "status": "matched",
            "game": sorted(candidates, key=lambda ev: str(_game_event_date(ev) or ""))[-1],
        }

    # ── 1. Settle individual tracked bets ───────────────────────────────────
    sel_settled = 0
    sel_profit  = 0.0

    for sel in pending_sel:
        sport = str(sel.get("sport", "")).lower()
        raw_match = str(sel.get("match", "")).replace(" @ ", " vs ")
        parts = [p.strip() for p in raw_match.split(" vs ", 1)]
        if len(parts) < 2:
            _push_unresolved(scope="tracked", sport=sport, match=raw_match, pick=str(sel.get("team", "")), reason="bad_match_format", message="Tracked match could not be parsed into home/away teams.")
            continue
        match_result = _find_game_match(sport, parts[0], parts[1], _selection_event_date(sel))
        game = match_result["game"]
        if game is None:
            reason = str(match_result["status"])
            message_map = {
                "no_score_source": "No finished score feed was available for this sport/date yet.",
                "team_mismatch": "Finished score events were found, but none matched the tracked teams.",
                "date_mismatch": "Matching teams were found, but not on the tracked event date.",
                "ambiguous_date": "Multiple finished games matched these teams and the tracked date could not disambiguate them.",
            }
            _push_unresolved(scope="tracked", sport=sport, match=raw_match, pick=str(sel.get("team", "")), reason=reason, message=message_map.get(reason, "No finished score event matched this tracked bet yet."))
            continue
        won = _resolve_pick_from_game(str(sel.get("team", "")), game, team_matcher=_tmatch)
        if won is None:
            _push_unresolved(scope="tracked", sport=sport, match=raw_match, pick=str(sel.get("team", "")), reason="market_resolution", message="A finished game was found, but the tracked market could not be resolved cleanly.")
            continue
        odds = float(sel.get("odds") or 1)
        sel["result"]      = "won" if won else "lost"
        sel["profit"]      = round((odds - 1) if won else -1, 4)
        sel["settled_at"]  = datetime.now(timezone.utc).isoformat()
        sel_settled += 1
        sel_profit  += sel["profit"]

    if sel_settled:
        _save_my_selections(selections)

    # ── 2. Settle manual + AI parlay legs ────────────────────────────────────
    parlays_changed = False
    parlay_results: list[dict] = []

    for parlay in all_parlays:
        if parlay.get("status") in ("won", "lost"):
            continue
        legs = parlay.get("legs", [])
        if not legs:
            continue

        newly_resolved = 0
        for leg in legs:
            if leg.get("result") in ("won", "lost"):
                continue
            sport = str(leg.get("sport", "")).lower()
            raw_match = str(leg.get("match", "")).replace(" @ ", " vs ")
            parts = [p.strip() for p in raw_match.split(" vs ", 1)]
            if len(parts) < 2:
                _push_unresolved(scope="parlay_leg", sport=sport, match=raw_match, pick=str(leg.get("team", "")), reason="bad_match_format", message="Parlay leg match could not be parsed into home/away teams.")
                continue
            match_result = _find_game_match(sport, parts[0], parts[1], _selection_event_date(leg))
            game = match_result["game"]
            if game is None:
                reason = str(match_result["status"])
                message_map = {
                    "no_score_source": "No finished score feed was available for this parlay leg yet.",
                    "team_mismatch": "Finished score events were found, but none matched this parlay leg.",
                    "date_mismatch": "Matching teams were found, but not on this parlay leg's tracked date.",
                    "ambiguous_date": "Multiple finished games matched this parlay leg and no tracked timestamp could disambiguate them.",
                }
                _push_unresolved(scope="parlay_leg", sport=sport, match=raw_match, pick=str(leg.get("team", "")), reason=reason, message=message_map.get(reason, "No finished score event matched this parlay leg yet."))
                continue
            won = _resolve_pick_from_game(str(leg.get("team", "")), game, team_matcher=_tmatch)
            if won is None:
                _push_unresolved(scope="parlay_leg", sport=sport, match=raw_match, pick=str(leg.get("team", "")), reason="market_resolution", message="A finished game was found, but the parlay leg outcome could not be resolved cleanly.")
                continue
            leg["result"]     = "won" if won else "lost"
            leg["settled_at"] = datetime.now(timezone.utc).isoformat()
            newly_resolved   += 1
            parlays_changed   = True

        # Recount after this pass
        n_won     = sum(1 for l in legs if l.get("result") == "won")
        n_lost    = sum(1 for l in legs if l.get("result") == "lost")
        n_pending = len(legs) - n_won - n_lost

        # Settle whole parlay: lost as soon as any leg is lost; won when all done+won
        if n_lost > 0 and parlay.get("status") == "pending":
            parlay["status"]      = "lost"
            parlay["settled_at"]  = datetime.now(timezone.utc).isoformat()
            parlays_changed = True
        elif n_pending == 0 and n_lost == 0 and n_won == len(legs):
            parlay["status"]      = "won"
            parlay["settled_at"]  = datetime.now(timezone.utc).isoformat()
            parlays_changed = True

        if newly_resolved > 0:
            parlay_results.append({
                "id":           parlay.get("id", ""),
                "name":         parlay.get("name", ""),
                "status":       parlay.get("status", "pending"),
                "legs_won":     n_won,
                "legs_lost":    n_lost,
                "legs_pending": n_pending,
                "legs_total":   len(legs),
            })

    if parlays_changed:
        _save_manual_parlays(all_parlays)

    # ── 3. Settle pending System Book rows using the same fetched score pool ─
    system_settled = 0
    system_profit = 0.0
    settled_path = BASE / "data" / "tracker" / "settled.parquet"
    system_pending_remaining = 0
    if not pending_pred_rows.empty:
        try:
            pred = pd.read_parquet(pred_path)
            pending_pred = pending_pred_rows
            if not pending_pred.empty:
                pred_ids_settled: list[str] = []
                settled_rows: list[dict] = []
                for _, row in pending_pred.iterrows():
                    sport = str(row.get("sport", "")).lower()
                    if sport == "parlay":
                        continue
                    match_id = str(row.get("match_id", "")).replace(" @ ", " vs ")
                    parts = [p.strip() for p in match_id.split(" vs ", 1)]
                    if len(parts) < 2:
                        _push_unresolved(scope="system_book", sport=sport, match=match_id, pick=str(row.get("team_or_player", "")), reason="bad_match_format", message="System Book match could not be parsed into home/away teams.")
                        continue
                    match_result = _find_game_match(sport, parts[0], parts[1], _event_date_value(row.get("commence_time")))
                    game = match_result["game"]
                    if game is None:
                        reason = str(match_result["status"])
                        message_map = {
                            "no_score_source": "No finished score feed was available for this System Book row yet.",
                            "team_mismatch": "Finished score events were found, but none matched the System Book teams.",
                            "date_mismatch": "Matching teams were found, but not on the System Book event date.",
                            "ambiguous_date": "Multiple finished games matched the System Book teams and the event date could not disambiguate them.",
                        }
                        _push_unresolved(scope="system_book", sport=sport, match=match_id, pick=str(row.get("team_or_player", "")), reason=reason, message=message_map.get(reason, "No finished score event matched the System Book row yet."))
                        continue
                    won = _resolve_pick_from_game(str(row.get("team_or_player", "")), game, team_matcher=_tmatch)
                    if won is None:
                        _push_unresolved(scope="system_book", sport=sport, match=match_id, pick=str(row.get("team_or_player", "")), reason="market_resolution", message="A finished game was found, but the System Book market could not be resolved cleanly.")
                        continue

                    stake = float(row.get("stake_units", 0) or 0)
                    odds = float(row.get("bet_odds", 1) or 1)
                    closing_odds = None
                    clv = None
                    try:
                        from settle import _fetch_closing_odds, _SPORT_KEYS as _SETTLE_SPORT_KEYS
                        ev_id = str(game.get("_event_id") or row.get("match_id") or "")
                        team_name = str(row.get("team_or_player", "") or "")
                        commence = str(row.get("commence_time", "") or "")
                        for sk in _SETTLE_SPORT_KEYS.get(sport, []):
                            closing_odds = _fetch_closing_odds(sk, ev_id, team_name, commence)
                            if closing_odds:
                                break
                        if closing_odds and odds:
                            clv = round((odds / float(closing_odds)) - 1.0, 6)
                    except Exception:
                        closing_odds = None
                        clv = None
                    profit = round((odds - 1) * stake if won else -stake, 4)
                    system_settled += 1
                    system_profit += profit
                    pred_ids_settled.append(str(row["pred_id"]))
                    settled_rows.append({
                        "pred_id": str(row["pred_id"]),
                        "settled_at": datetime.now(timezone.utc).isoformat(),
                        "sport": sport,
                        "match_id": str(row.get("match_id", "")),
                        "team_or_player": str(row.get("team_or_player", "")),
                        "commence_time": row.get("commence_time"),
                        "recorded_at": row.get("recorded_at"),
                        "market": str(row.get("market", row.get("market_type", "moneyline"))),
                        "market_status": str(row.get("market_status", "experimental")),
                        "tier": str(row.get("tier", "Experimental")),
                        "bet_odds": odds,
                        "bookmaker": str(row.get("bookmaker", "unknown")),
                        "edge": float(row.get("edge", 0) or 0),
                        "ml_prob": float(row.get("ml_prob", 0) or 0),
                        "fair_prob": float(row.get("fair_prob", 0) or 0),
                        "stake_units": stake,
                        "kelly_stake_pct": float(row.get("kelly_stake_pct", 0) or 0),
                        "is_parlay_leg": bool(row.get("is_parlay_leg", False)),
                        "actual_result": "won" if won else "lost",
                        "won": won,
                        "profit_units": profit,
                        "closing_odds": closing_odds,
                        "clv": clv,
                        "status": "won" if won else "lost",
                    })

                if settled_rows:
                    new_df = pd.DataFrame(settled_rows)
                    if settled_path.exists() and settled_path.stat().st_size > 100:
                        existing = pd.read_parquet(settled_path)
                        for col in ("commence_time", "recorded_at"):
                            if col in existing.columns and pd.api.types.is_datetime64_any_dtype(existing[col]):
                                new_df[col] = pd.to_datetime(new_df[col], utc=True)
                        pd.concat([existing, new_df], ignore_index=True).to_parquet(settled_path, index=False)
                    else:
                        new_df.to_parquet(settled_path, index=False)

                    pred = pred[~pred["pred_id"].astype(str).isin(pred_ids_settled)]
                    pred.to_parquet(pred_path, index=False)

                system_pending_remaining = int((pred["status"] == "pending").sum()) if "status" in pred.columns else 0
        except Exception:
            pass

    # ── 4. Also run system parlay settlement (parquet-based) ─────────────────
    sys_settled = 0
    try:
        sys.path.insert(0, str(BASE))
        from settle import settle_parlays as _settle_parlays
        from src.utils.results_tracker import get_pending_parlays
        pending_before = len(get_pending_parlays())
        _settle_parlays()
        sys_settled = pending_before - len(get_pending_parlays())
    except Exception:
        pass

    total_parlays = len([p for p in parlay_results if p["status"] in ("won", "lost")]) + sys_settled
    still_pending = len([s for s in selections if s.get("result") is None]) + system_pending_remaining
    unresolved_summary = []
    unresolved_by_reason = []
    unresolved_by_sport = []
    if unresolved_rows:
        unresolved_df = pd.DataFrame(unresolved_rows)
        unresolved_summary = (
            unresolved_df.groupby(["reason", "scope"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .to_dict(orient="records")
        )
        unresolved_by_reason = (
            unresolved_df.groupby(["reason"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .to_dict(orient="records")
        )
        unresolved_by_sport = (
            unresolved_df.groupby(["sport", "reason"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["count", "sport"], ascending=[False, True])
            .to_dict(orient="records")
        )

    global _last_results_settle_report
    response_payload = {
        "ok":               True,
        "mode":             "backlog" if backlog_mode else "standard",
        "bets_settled":     sel_settled + system_settled,
        "bets_profit":      round(sel_profit + system_profit, 4),
        "parlays_settled":  total_parlays,
        "parlay_results":   parlay_results,
        "still_pending":    still_pending,
        "score_pool":       len(finished_events),
        "target_dates_scanned": len(target_dates),
        "score_sources":    sorted(set(score_sources)),
        "errors":           source_errors,
        "unresolved_summary": unresolved_summary,
        "unresolved_by_reason": unresolved_by_reason,
        "unresolved_by_sport": unresolved_by_sport,
        "unresolved_samples": unresolved_rows[:8],
    }
    _last_results_settle_report = {
        **response_payload,
        "attempted_at": datetime.now(timezone.utc).isoformat(),
    }
    return jsonify(response_payload)


@app.route("/api/parlays/ai-build")
def api_parlays_ai_build():
    """SSE stream: evaluate ALL of today's active bets (preferred + experimental) via LLM,
    then build two parlays — a Value parlay (highest ml_prob APPROVE legs) and a
    Longshot parlay (highest odds APPROVE/REVIEW legs)."""
    _market_map = {
        "moneyline": "h2h", "spreads": "spreads", "totals": "totals",
        "double_chance": "double_chance", "draw_no_bet": "draw_no_bet",
    }

    def _sse(data: dict) -> str:
        return f"data: {json.dumps(data, default=str)}\n\n"

    def _build_legs(entries: list[dict]) -> list[dict]:
        legs = []
        for e in entries:
            c = e["candidate"]
            legs.append({
                "team":           c.get("team", ""),
                "match":          f"{c.get('home', '')} vs {c.get('away', '')}",
                "sport":          c.get("sport", ""),
                "market":         c.get("market", ""),
                "market_status":  c.get("market_status", ""),
                "odds":           float(c.get("odds") or 1.0),
                "ml_prob":        float(c.get("ml_prob") or 0.5),
                "edge":           float(c.get("edge") or 0.0),
                "kick_off":       c.get("kick_off", ""),
                "why_for":        e["why_for"],
                "reasoning":      e["reasoning"],
                "decision":       e["decision"],
                "upset_catalyst": e.get("upset_catalyst", ""),
            })
        return legs

    def _parlay_stats(legs: list[dict]) -> tuple[float, float, float, float, str]:
        combined_odds = 1.0
        win_prob = 1.0
        for leg in legs:
            combined_odds *= leg["odds"]
            win_prob *= leg["ml_prob"]
        ev = win_prob * combined_odds
        edge_val = ev - 1.0
        kelly = (
            max(0.0, (win_prob * combined_odds - 1.0) / (combined_odds - 1.0)) * 0.5
            if combined_odds > 1 else 0.0
        )
        win_pct = win_prob * 100
        if win_pct >= 0.1:
            wp_str = f"{win_pct:.2f}%"
        elif win_pct >= 0.001:
            wp_str = f"{win_pct:.4f}%"
        else:
            one_in = int(round(1.0 / win_prob)) if win_prob > 0 else 0
            wp_str = f"1 in {one_in:,}"
        return combined_odds, win_prob, edge_val, kelly, wp_str

    def _assess_parlay_legs(legs: list[dict], tier: str) -> dict[str, Any]:
        builder = ParlayBuilder()
        parlay_legs = [
            ParlayLeg(
                sport=str(leg.get("sport", "")),
                match_id=str(leg.get("match", "")),
                team=str(leg.get("team", "")),
                odds=float(leg.get("odds") or 1.0),
                ml_prob=float(leg.get("ml_prob") or 0.0),
                fair_prob=max(0.0, float(leg.get("ml_prob") or 0.0) - float(leg.get("edge") or 0.0)),
                edge=float(leg.get("edge") or 0.0),
                commence=str(leg.get("commence") or leg.get("kick_off") or ""),
                market=str(leg.get("market", "")),
            )
            for leg in legs
        ]
        assessment = builder.assess_legs(parlay_legs, tier=tier)
        weakest = assessment.get("weakest_leg")
        return {
            "risk_tier": assessment.get("risk_tier"),
            "build_verdict": assessment.get("build_verdict"),
            "duplicate_games": assessment.get("duplicate_games", []),
            "conflicting_picks": assessment.get("conflicting_picks", []),
            "correlated_pick_groups": assessment.get("correlated_pick_groups", []),
            "validation_notes": assessment.get("notes", []),
            "weakest_leg": {
                "team": weakest.team,
                "match_id": weakest.match_id,
                "sport": weakest.sport,
                "market": weakest.market,
                "odds": weakest.odds,
                "ml_prob": weakest.ml_prob,
                "edge": weakest.edge,
            } if weakest else None,
        }

    def _make_parlay(name: str, parlay_type: str, legs: list[dict]) -> dict:
        import uuid
        combined_odds, win_prob, edge_val, kelly, wp_str = _parlay_stats(legs)
        style_tier = "value" if parlay_type == "ai_value" else "speculative"
        assessment = _assess_parlay_legs(legs, tier=style_tier)
        bankroll = float(os.getenv("INITIAL_BANKROLL", 1000))
        return {
            "id":            str(uuid.uuid4())[:8],
            "name":          name,
            "date":          datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "saved_at":      datetime.now(timezone.utc).isoformat(),
            "type":          parlay_type,
            "status":        "pending",
            "n_legs":        len(legs),
            "combined_odds": round(combined_odds, 2),
            "win_prob":      wp_str,
            "ev":            round(win_prob * combined_odds, 3),
            "edge":          round(edge_val * 100, 2),
            "kelly_stake":   round(kelly * bankroll, 2),
            "kelly_pct":     round(kelly * 100, 2),
            "combined_probability": round(win_prob, 6),
            "risk_tier":     assessment["risk_tier"],
            "build_verdict": assessment["build_verdict"],
            "weakest_leg":   assessment["weakest_leg"],
            "duplicate_games": assessment["duplicate_games"],
            "conflicting_picks": assessment["conflicting_picks"],
            "correlated_pick_groups": assessment["correlated_pick_groups"],
            "validation_notes": assessment["validation_notes"],
            "legs": [
                {
                    "team":           leg["team"],
                    "match":          leg["match"],
                    "sport":          leg["sport"],
                    "market":         leg["market"],
                    "market_status":  leg.get("market_status", ""),
                    "odds":           round(leg["odds"], 2),
                    "ml_prob":        round(leg["ml_prob"], 4),
                    "edge":           round(leg["edge"], 4),
                    "kick_off":       leg.get("kick_off", ""),
                    "why_for":        leg.get("why_for", ""),
                    "reasoning":      leg.get("reasoning", ""),
                    "decision":       leg.get("decision", ""),
                    "upset_catalyst": leg.get("upset_catalyst", ""),
                }
                for leg in legs
            ],
        }

    def _dedup_pick(pool: list[dict], max_legs: int) -> list[dict]:
        """Pick up to max_legs entries with no two from the same game."""
        seen: set[str] = set()
        picked: list[dict] = []
        for e in pool:
            gk = f"{e['candidate'].get('home', '')}|{e['candidate'].get('away', '')}"
            if gk in seen:
                continue
            seen.add(gk)
            picked.append(e)
            if len(picked) >= max_legs:
                break
        return picked

    def _dedup_pick_diverse(pool: list[dict], max_legs: int, max_per_sport: int = 2) -> list[dict]:
        """Like _dedup_pick but also limits legs per sport for correlation control."""
        seen_games: set[str] = set()
        sport_counts: dict[str, int] = {}
        picked: list[dict] = []
        for e in pool:
            c = e["candidate"]
            gk = f"{c.get('home', '')}|{c.get('away', '')}"
            sport = str(c.get("sport", "")).lower()
            if gk in seen_games:
                continue
            if sport_counts.get(sport, 0) >= max_per_sport:
                continue
            seen_games.add(gk)
            sport_counts[sport] = sport_counts.get(sport, 0) + 1
            picked.append(e)
            if len(picked) >= max_legs:
                break
        return picked

    # Threshold above which a candidate is treated as a potential longshot leg
    _LONGSHOT_ODDS_THRESHOLD = 2.0
    # Minimum model probability to qualify for longshot parlay — cuts pure noise legs
    _LONGSHOT_MIN_ML_PROB    = 0.28

    # Window filter from query param: today | tomorrow | both (default: today)
    _window_param = request.args.get("window", "today").strip().lower()
    if _window_param not in ("today", "tomorrow", "both"):
        _window_param = "today"
    _window_label = {"today": "today's", "tomorrow": "tomorrow's", "both": "today's + tomorrow's"}[_window_param]

    def generate():
        # Fetch all active bets then filter by requested window
        all_candidates = _today_reasoning_bets()
        if _window_param == "today":
            candidates = [c for c in all_candidates if c.get("window") == "today"]
        elif _window_param == "tomorrow":
            candidates = [c for c in all_candidates if c.get("window") == "tomorrow"]
        else:
            candidates = all_candidates

        if not candidates:
            yield _sse({"stage": "error", "message": f"No active bets found for {_window_label} window — run a scan first."})
            return

        yield _sse({"stage": "start", "total": len(candidates)})

        evaluated = []
        for i, candidate in enumerate(candidates):
            label = (
                f"{str(candidate.get('sport', '')).upper()} · "
                f"{candidate.get('team', '?')} "
                f"({candidate.get('home', '?')} vs {candidate.get('away', '?')})"
            )
            tier = candidate.get("market_status", "experimental")
            odds = float(candidate.get("odds") or 1.0)

            # Use the longshot upset-focused prompt for higher-odds candidates
            is_longshot_candidate = odds >= _LONGSHOT_ODDS_THRESHOLD
            prompt_to_use = _LONGSHOT_SYSTEM_PROMPT if is_longshot_candidate else None

            yield _sse({"stage": "evaluating", "n": i + 1, "total": len(candidates),
                        "label": label, "tier": tier,
                        "mode": "longshot" if is_longshot_candidate else "value"})

            chosen_market = _market_map.get(candidate.get("market"), "h2h")
            try:
                report = _candidate_reasoning_report(candidate, market=chosen_market, selection="")
                payload = _sanitize_payload(json.loads(json.dumps(report.to_dict(), default=str)))
                payload["warnings"] = list(getattr(report, "warnings", []) or [])
                llm_result, llm_error = _reasoning_layer(candidate, payload, system_prompt=prompt_to_use)
            except Exception as exc:
                llm_result, llm_error = None, str(exc)

            decision = "REVIEW"
            why_for = why_against = reasoning_snippet = biggest_risk = upset_catalyst = ""
            if llm_result and isinstance(llm_result.get("content"), dict):
                content = llm_result["content"]
                decision         = str(content.get("decision", "REVIEW")).upper()
                why_for          = str(content.get("why_for", "")).strip()
                why_against      = str(content.get("why_against", "")).strip()
                reasoning_snippet = str(content.get("reasoning", "")).strip()
                biggest_risk     = str(content.get("biggest_risk", "")).strip()
                upset_catalyst   = str(content.get("upset_catalyst", "")).strip()

            evaluated.append({
                "candidate":      candidate,
                "decision":       decision,
                "why_for":        why_for,
                "why_against":    why_against,
                "reasoning":      reasoning_snippet,
                "biggest_risk":   biggest_risk,
                "upset_catalyst": upset_catalyst,
                "llm_error":      llm_error,
            })
            yield _sse({
                "stage":          "evaluated",
                "n":              i + 1,
                "total":          len(candidates),
                "label":          label,
                "tier":           tier,
                "decision":       decision,
                "why_for":        why_for[:140],
                "reasoning":      reasoning_snippet[:200],
                "upset_catalyst": upset_catalyst[:120],
            })

        # ── VALUE PARLAY ─────────────────────────────────────────────────────────
        # APPROVE only, sorted by ml_prob descending (safest legs first), 3–4 legs
        value_pool = [e for e in evaluated if e["decision"] == "APPROVE"]
        if len(value_pool) < 2:
            value_pool = [e for e in evaluated if e["decision"] in ("APPROVE", "REVIEW")]
        value_sorted = sorted(value_pool,
                              key=lambda x: float(x["candidate"].get("ml_prob") or 0),
                              reverse=True)
        value_selected = _dedup_pick(value_sorted, max_legs=4)

        # ── LONGSHOT PARLAY ──────────────────────────────────────────────────────
        # Requirements:
        #   • APPROVE or REVIEW decision
        #   • ml_prob >= 0.28  (filters noise — model must see at least some real chance)
        #   • odds  >= 1.5     (keeps it genuinely longshot territory)
        # Sorted by EV (ml_prob × odds) descending — finds upsets the model believes in,
        # not just the biggest raw odds numbers.
        # De-duped with max 2 legs per sport to reduce hidden correlation.
        longshot_pool = [
            e for e in evaluated
            if e["decision"] in ("APPROVE", "REVIEW")
            and float(e["candidate"].get("ml_prob") or 0) >= _LONGSHOT_MIN_ML_PROB
            and float(e["candidate"].get("odds") or 1.0) >= 1.5
        ]
        longshot_sorted = sorted(
            longshot_pool,
            key=lambda x: float(x["candidate"].get("ml_prob") or 0) * float(x["candidate"].get("odds") or 1.0),
            reverse=True,
        )
        longshot_selected = _dedup_pick_diverse(longshot_sorted, max_legs=6, max_per_sport=2)

        saved = []
        all_parlays = _load_manual_parlays()
        _win_suffix = {"today": "", "tomorrow": " (Tomorrow)", "both": " (Today+Tomorrow)"}[_window_param]

        if len(value_selected) >= 2:
            v_legs = _build_legs(value_selected)
            v_parlay = _make_parlay(f"AI Value {len(v_legs)}-Leg{_win_suffix}", "ai_value", v_legs)
            if v_parlay.get("build_verdict") != "DO NOT BUILD":
                all_parlays.append(v_parlay)
                saved.append(("value", v_parlay))
        else:
            v_parlay = None

        if len(longshot_selected) >= 2:
            l_legs = _build_legs(longshot_selected)
            l_parlay = _make_parlay(f"AI Longshot {len(l_legs)}-Leg{_win_suffix}", "ai_longshot", l_legs)
            if l_parlay.get("build_verdict") != "DO NOT BUILD":
                all_parlays.append(l_parlay)
                saved.append(("longshot", l_parlay))
        else:
            l_parlay = None

        if not saved:
            yield _sse({"stage": "error", "message": "Not enough approved bets to build a parlay. Try scanning with Context Referee enabled."})
            return

        _save_manual_parlays(all_parlays)
        yield _sse({
            "stage": "done",
            "value_parlay":    v_parlay,
            "longshot_parlay": l_parlay,
            "evaluated_count": len(evaluated),
            "approved_count":  sum(1 for e in evaluated if e["decision"] == "APPROVE"),
        })

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/parlay/calculate", methods=["POST"])
def api_parlay_calculate():
    """
    Calculate stats for a custom parlay built by the user.
    POST body: { "legs": [ {"ml_prob": 0.54, "odds": 2.12, "team": "...", "sport": "..."}, ... ] }
    """
    data = request.get_json()
    legs = data.get("legs", [])
    if len(legs) < 2:
        return jsonify({"error": "Need at least 2 legs"}), 400

    builder = ParlayBuilder()
    parlay_legs = [
        ParlayLeg(
            sport=str(leg.get("sport", "")),
            match_id=str(leg.get("match") or f"{leg.get('home', '')} vs {leg.get('away', '')}"),
            team=str(leg.get("team", "")),
            odds=float(leg.get("odds") or 1.0),
            ml_prob=float(leg.get("ml_prob") or 0.0),
            fair_prob=max(0.0, float(leg.get("fair_prob") or (float(leg.get("ml_prob") or 0.0) - float(leg.get("edge") or 0.0)))),
            edge=float(leg.get("edge") or 0.0),
            commence=str(leg.get("commence") or leg.get("kick_off") or ""),
            market=str(leg.get("market", "")),
            home_team=str(leg.get("home", "")),
            away_team=str(leg.get("away", "")),
        )
        for leg in legs
    ]
    assessment = builder.assess_legs(parlay_legs, tier=str(data.get("tier") or "custom").lower())

    combined_odds = 1.0
    win_prob      = 1.0
    for leg in legs:
        combined_odds *= float(leg["odds"])
        win_prob      *= float(leg["ml_prob"])

    ev   = win_prob * combined_odds
    edge = ev - 1.0
    # Half-Kelly for parlays
    kelly = max(0, (win_prob * combined_odds - 1) / (combined_odds - 1)) * 0.5

    bankroll = float(os.getenv("INITIAL_BANKROLL", 1000))

    # Format win_prob: use enough decimal places so it never shows as "0%"
    win_pct = win_prob * 100
    if win_pct >= 0.1:
        win_prob_str = f"{win_pct:.2f}"
    elif win_pct >= 0.001:
        win_prob_str = f"{win_pct:.4f}"
    elif win_pct >= 0.00001:
        win_prob_str = f"{win_pct:.6f}"
    else:
        # Express as "1 in N"
        one_in = int(round(1 / win_prob)) if win_prob > 0 else 999999999
        win_prob_str = f"1 in {one_in:,}"

    return jsonify({
        "legs":          len(legs),
        "combined_odds": round(combined_odds, 2),
        "combined_probability": round(win_prob, 6),
        "win_prob":      win_prob_str,
        "ev":            round(ev, 3),
        "edge":          round(edge * 100, 2),
        "kelly_pct":     round(kelly * 100, 2),
        "kelly_stake":   round(kelly * bankroll, 2),
        "risk_tier":     assessment["risk_tier"],
        "build_verdict": assessment["build_verdict"],
        "duplicate_games": assessment["duplicate_games"],
        "conflicting_picks": assessment["conflicting_picks"],
        "correlated_pick_groups": assessment["correlated_pick_groups"],
        "validation_notes": assessment["notes"],
        "weakest_leg": {
            "team": assessment["weakest_leg"].team,
            "match_id": assessment["weakest_leg"].match_id,
            "sport": assessment["weakest_leg"].sport,
            "market": assessment["weakest_leg"].market,
            "odds": assessment["weakest_leg"].odds,
            "ml_prob": assessment["weakest_leg"].ml_prob,
            "edge": assessment["weakest_leg"].edge,
        } if assessment.get("weakest_leg") else None,
    })


# ══════════════════════════════════════════════════════════════════════════════
# API — RESULTS / TRACKER
# ══════════════════════════════════════════════════════════════════════════════

def _mtype(name: str) -> str:
    name = str(name or "")
    lower = name.lower()
    if "draw no bet" in lower or lower.endswith(" dnb"):
        return "draw_no_bet"
    if " or draw" in lower:
        return "double_chance"
    if "Over" in name or "Under" in name:
        return "totals"
    if any(x in name for x in ["+1.5", "-1.5", "+0.5", "-0.5", "+2.5", "-2.5"]):
        return "spreads"
    return "moneyline"


def _event_date_value(value: object) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return str(pd.Timestamp(text).date())
    except Exception:
        return text[:10] if len(text) >= 10 else None


def _selection_event_date(selection: dict) -> str | None:
    for field in ("commence", "commence_time", "date"):
        date_value = _event_date_value(selection.get(field))
        if date_value:
            return date_value
    return None


def _game_event_date(game: dict) -> str | None:
    for field in ("event_date", "commence_time", "utcDate"):
        date_value = _event_date_value(game.get(field))
        if date_value:
            return date_value
    return None


def _extract_game_scoreline(game: dict) -> tuple[str, str, int, int] | None:
    home = str(game.get("home_team") or game.get("home") or "").strip()
    away = str(game.get("away_team") or game.get("away") or "").strip()
    if not home or not away:
        return None

    if game.get("home_score") is not None and game.get("away_score") is not None:
        try:
            return home, away, int(game.get("home_score")), int(game.get("away_score"))
        except Exception:
            return None

    scores = game.get("scores") or []
    if isinstance(scores, dict):
        try:
            return home, away, int(scores.get(home)), int(scores.get(away))
        except Exception:
            return None

    if isinstance(scores, list):
        score_map = {}
        for item in scores:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            score = item.get("score")
            if not name or score is None:
                continue
            try:
                score_map[name] = int(score)
            except Exception:
                continue
        try:
            return home, away, int(score_map.get(home)), int(score_map.get(away))
        except Exception:
            return None

    return None


def _resolve_pick_from_game(
    pick: str,
    game: dict,
    *,
    team_matcher,
):
    scoreline = _extract_game_scoreline(game)
    if scoreline is None:
        return None
    home, away, home_score, away_score = scoreline
    pick = str(pick or "").strip()
    pick_l = pick.lower()
    mtype = _mtype(pick)

    if mtype == "totals":
        match = re.search(r"(over|under)\s+([\d.]+)", pick_l, re.I)
        if not match:
            return None
        total = home_score + away_score
        line = float(match.group(2))
        if total == line:
            return None
        return (match.group(1).lower() == "over") == (total > line)

    if mtype == "spreads":
        match = re.search(r"([+-][\d.]+)\s*$", pick)
        if not match:
            return None
        spread = float(match.group(1))
        team_part = re.sub(r"[+-][\d.]+\s*$", "", pick).strip()
        if team_matcher(team_part, home):
            adjusted = home_score + spread
            return None if adjusted == away_score else adjusted > away_score
        if team_matcher(team_part, away):
            adjusted = away_score + spread
            return None if adjusted == home_score else adjusted > home_score
        return None

    if mtype == "draw_no_bet":
        team_part = re.sub(r"\s*(dnb|draw no bet)\s*$", "", pick, flags=re.I).strip()
        if home_score == away_score:
            return None
        if team_matcher(team_part, home):
            return home_score > away_score
        if team_matcher(team_part, away):
            return away_score > home_score
        return None

    if mtype == "double_chance":
        if home_score == away_score:
            return True
        team_part = re.sub(r"\s+or\s+draw\s*$", "", pick, flags=re.I).strip()
        if team_matcher(team_part, home):
            return home_score >= away_score
        if team_matcher(team_part, away):
            return away_score >= home_score
        return None

    if pick_l in ("draw", "tie"):
        return home_score == away_score
    if team_matcher(pick, home):
        return home_score > away_score
    if team_matcher(pick, away):
        return away_score > home_score
    return None


def _event_date_series(df: pd.DataFrame, primary: str = "commence_time", fallback: str = "recorded_at") -> pd.Series:
    if df.empty:
        return pd.Series(dtype="object")
    primary_dt = pd.to_datetime(df.get(primary), utc=True, errors="coerce") if primary in df.columns else pd.Series(pd.NaT, index=df.index)
    fallback_dt = pd.to_datetime(df.get(fallback), utc=True, errors="coerce") if fallback in df.columns else pd.Series(pd.NaT, index=df.index)
    resolved = primary_dt.fillna(fallback_dt)
    return resolved.dt.date.astype(str)


def _stats_from_settled(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"bets": 0, "wins": 0, "win_rate": 0, "pnl": 0, "roi": 0}
    total_bets = len(df)
    total_wins = int(df["won"].sum())
    total_pnl = float(df["profit_units"].sum())
    total_staked = float(df["stake_units"].sum())
    overall_roi = total_pnl / total_staked * 100 if total_staked > 0 else 0
    return {
        "bets": total_bets,
        "wins": total_wins,
        "win_rate": round(total_wins / total_bets * 100, 1) if total_bets else 0,
        "pnl": round(total_pnl, 4),
        "roi": round(overall_roi, 2),
    }


def _recent_settled_slice(df: pd.DataFrame, limit: int = 30) -> pd.DataFrame:
    if df.empty:
        return df
    order_col = "settled_at" if "settled_at" in df.columns else "recorded_at" if "recorded_at" in df.columns else None
    if not order_col:
        return df.tail(limit)
    ordered = df.copy()
    ordered["_recent_sort_ts"] = pd.to_datetime(ordered[order_col], errors="coerce", utc=True)
    ordered = ordered.sort_values("_recent_sort_ts", ascending=False, na_position="last")
    return ordered.head(limit).drop(columns=["_recent_sort_ts"], errors="ignore")


def _odds_bucket_label(odds: object) -> str:
    try:
        value = float(odds)
    except Exception:
        return "unknown"
    if value <= 1.67:
        return "≤1.67"
    if value <= 2.19:
        return "1.68–2.19"
    if value <= 3.49:
        return "2.20–3.49"
    return "3.50+"


def _odds_bucket_summary(df: pd.DataFrame) -> list[dict]:
    if df.empty or "bet_odds" not in df.columns:
        return []
    work = df.copy()
    work["odds_bucket"] = work["bet_odds"].apply(_odds_bucket_label)
    rows: list[dict] = []
    bucket_order = {"≤1.67": 0, "1.68–2.19": 1, "2.20–3.49": 2, "3.50+": 3, "unknown": 4}
    for bucket, grp in work.groupby("odds_bucket", dropna=False):
        stats = _stats_from_settled(grp)
        rows.append({
            "bucket": str(bucket),
            "bets": int(stats["bets"]),
            "win_rate": stats["win_rate"],
            "pnl": stats["pnl"],
            "roi": stats["roi"],
            "avg_odds": round(float(pd.to_numeric(grp["bet_odds"], errors="coerce").dropna().mean()), 2)
            if pd.to_numeric(grp["bet_odds"], errors="coerce").dropna().shape[0] else None,
        })
    return sorted(rows, key=lambda row: (bucket_order.get(row["bucket"], 99), row["bucket"]))


def _lane_highlights(rows: list[dict], limit: int = 5) -> dict:
    settled_rows = [row for row in rows if int(row.get("bets", 0) or 0) > 0]
    eligible = [row for row in settled_rows if int(row.get("bets", 0) or 0) >= 3] or settled_rows

    def _shape(row: dict) -> dict:
        return {
            "sport": str(row.get("sport", "")),
            "market": str(row.get("market", "")),
            "tier": str(row.get("tier", "")),
            "bets": int(row.get("bets", 0) or 0),
            "win_rate": float(row.get("win_rate", 0) or 0),
            "roi": float(row.get("roi", 0) or 0),
            "pnl": float(row.get("pnl", 0) or 0),
            "avg_edge": float(row.get("avg_edge", 0) or 0),
            "avg_clv": row.get("avg_clv"),
            "clv_signal": str(row.get("clv_signal", "")),
        }

    best = sorted(eligible, key=lambda row: (float(row.get("roi", 0) or 0), float(row.get("pnl", 0) or 0)), reverse=True)[:limit]
    worst = sorted(eligible, key=lambda row: (float(row.get("roi", 0) or 0), float(row.get("pnl", 0) or 0)))[:limit]
    return {
        "best": [_shape(row) for row in best],
        "worst": [_shape(row) for row in worst],
    }


def _calibration_snapshot(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty or "ml_prob" not in df.columns or "won" not in df.columns:
        return {"summary": {}, "by_sport": [], "buckets": []}

    work = df.copy()
    work["ml_prob_num"] = pd.to_numeric(work["ml_prob"], errors="coerce")
    work["won_num"] = work["won"].fillna(False).astype(bool).astype(int)
    work = work.dropna(subset=["ml_prob_num"])
    if work.empty:
        return {"summary": {}, "by_sport": [], "buckets": []}

    work["ml_prob_num"] = work["ml_prob_num"].clip(1e-6, 1 - 1e-6)

    def _shape(grp: pd.DataFrame, sport: Optional[str] = None) -> dict[str, Any]:
        probs = grp["ml_prob_num"].astype(float)
        won = grp["won_num"].astype(float)
        avg_prob = float(probs.mean())
        win_rate = float(won.mean())
        brier = float(((probs - won) ** 2).mean())
        log_loss = float((-(won * np.log(probs)) - ((1.0 - won) * np.log(1.0 - probs))).mean())
        gap_pp = abs(avg_prob - win_rate) * 100.0
        return {
            **({"sport": str(sport)} if sport is not None else {}),
            "bets": int(len(grp)),
            "avg_prob_pct": round(avg_prob * 100, 1),
            "win_rate_pct": round(win_rate * 100, 1),
            "gap_pp": round(gap_pp, 1),
            "brier": round(brier, 4),
            "log_loss": round(log_loss, 4),
        }

    summary = _shape(work)
    by_sport = [
        _shape(grp, sport=str(sport))
        for sport, grp in work.groupby("sport", dropna=False)
        if len(grp) >= 3
    ]
    by_sport.sort(key=lambda row: (row["gap_pp"], row["brier"], -row["bets"]))

    bucket_edges = [
        (0.00, 0.45, "≤45%"),
        (0.45, 0.55, "46–55%"),
        (0.55, 0.65, "56–65%"),
        (0.65, 0.75, "66–75%"),
        (0.75, 1.01, "76%+"),
    ]
    buckets: list[dict[str, Any]] = []
    for lower, upper, label in bucket_edges:
        if upper >= 1.0:
            grp = work[(work["ml_prob_num"] >= lower) & (work["ml_prob_num"] <= 1.0)]
        else:
            grp = work[(work["ml_prob_num"] >= lower) & (work["ml_prob_num"] < upper)]
        if grp.empty:
            continue
        shaped = _shape(grp)
        buckets.append({
            "bucket": label,
            "bets": shaped["bets"],
            "avg_prob_pct": shaped["avg_prob_pct"],
            "win_rate_pct": shaped["win_rate_pct"],
            "gap_pp": shaped["gap_pp"],
        })

    ece = 0.0
    total_bets = max(int(summary["bets"]), 1)
    for row in buckets:
        ece += (row["bets"] / total_bets) * (row["gap_pp"] / 100.0)
    summary["ece"] = round(ece, 4)

    return {
        "summary": summary,
        "by_sport": by_sport,
        "buckets": buckets,
    }


def _calibration_governor(snapshot: Optional[dict[str, Any]]) -> dict[str, Any]:
    by_sport = list(((snapshot or {}).get("by_sport") or []))
    buckets = list(((snapshot or {}).get("buckets") or []))
    if not by_sport:
        return {"summary": {}, "rows": []}

    rows: list[dict[str, Any]] = []
    critical = 0
    moderate = 0
    watch = 0
    for row in by_sport:
        sport = str(row.get("sport") or "")
        bets = int(row.get("bets", 0) or 0)
        gap_pp = float(row.get("gap_pp", 0) or 0)
        brier = float(row.get("brier", 0) or 0)
        log_loss = float(row.get("log_loss", 0) or 0)
        avg_prob_pct = float(row.get("avg_prob_pct", 0) or 0)
        win_rate_pct = float(row.get("win_rate_pct", 0) or 0)
        overconfident = avg_prob_pct > win_rate_pct

        severity = ""
        action = ""
        next_step = ""
        if bets >= 10 and (gap_pp >= 30.0 or (gap_pp >= 20.0 and (brier >= 0.25 or log_loss >= 0.80))):
            severity = "critical"
            action = "retrain_first"
            next_step = "Retrain and recalibrate this sport before trusting new live expansions."
            critical += 1
        elif bets >= 8 and (gap_pp >= 15.0 or brier >= 0.20 or log_loss >= 0.70):
            severity = "moderate"
            action = "recalibrate_reduce_trust"
            next_step = "Keep live exposure conservative and tighten calibration before changing policy."
            moderate += 1
        elif bets >= 5 and gap_pp >= 8.0:
            severity = "watch"
            action = "monitor"
            next_step = "Collect more settled bets and watch whether the gap widens or normalizes."
            watch += 1
        else:
            continue

        rows.append({
            "sport": sport,
            "bets": bets,
            "severity": severity,
            "action": action,
            "gap_pp": round(gap_pp, 1),
            "brier": round(brier, 4),
            "log_loss": round(log_loss, 4),
            "avg_prob_pct": round(avg_prob_pct, 1),
            "win_rate_pct": round(win_rate_pct, 1),
            "bias": "overconfident" if overconfident else "underconfident",
            "reason": (
                f"Avg predicted {avg_prob_pct:.1f}% vs actual {win_rate_pct:.1f}% "
                f"(gap {gap_pp:.1f}pp), Brier {brier:.4f}, log loss {log_loss:.4f}."
            ),
            "next_step": next_step,
        })

    rows.sort(key=lambda item: (
        {"critical": 0, "moderate": 1, "watch": 2}.get(str(item.get("severity")), 9),
        -float(item.get("gap_pp", 0) or 0),
        -int(item.get("bets", 0) or 0),
    ))

    worst_bucket = None
    if buckets:
        worst_bucket = sorted(
            buckets,
            key=lambda item: (-float(item.get("gap_pp", 0) or 0), -int(item.get("bets", 0) or 0)),
        )[0]

    summary = {
        "critical": critical,
        "moderate": moderate,
        "watch": watch,
        "sports_flagged": len(rows),
        "worst_bucket": worst_bucket.get("bucket") if worst_bucket else None,
        "worst_bucket_gap_pp": worst_bucket.get("gap_pp") if worst_bucket else None,
    }
    return {"summary": summary, "rows": rows}


def _active_calibration_status() -> dict[str, Any]:
    sports = ["soccer", "mlb", "basketball", "nhl", "tennis"]
    rows: list[dict[str, Any]] = []
    calibrated = 0
    for sport in sports:
        tag = get_current_model_tag(sport)
        has_calibrator = bool(tag and calibrator_path_for_tag(sport, tag).exists())
        if has_calibrator:
            calibrated += 1
        rows.append({
            "sport": sport,
            "active_tag": tag,
            "has_calibrator": has_calibrator,
            "status": "calibrated" if has_calibrator else "uncalibrated",
        })
    return {
        "summary": {
            "sports": len(rows),
            "calibrated": calibrated,
            "uncalibrated": len(rows) - calibrated,
        },
        "rows": rows,
    }


def _current_tier_for_row(sport: object, market: object) -> tuple[str, str]:
    policy = get_market_policy(str(sport or ""), str(market or "moneyline"))
    return (
        str(policy.get("label", "Experimental")),
        str(policy.get("status", "experimental")),
    )


def _performance_matrix(df: pd.DataFrame, pending_df: Optional[pd.DataFrame] = None) -> list[dict]:
    if df.empty and (pending_df is None or pending_df.empty):
        return []

    work = df.copy() if not df.empty else pd.DataFrame()
    if "market_type" not in work.columns:
        work["market_type"] = work["team_or_player"].apply(_mtype) if "team_or_player" in work.columns else "moneyline"
    if "market" not in work.columns:
        work["market"] = work["market_type"]
    else:
        work["market"] = work["market"].fillna(work["market_type"]).replace("", pd.NA).fillna(work["market_type"])

    tier_meta = work.apply(
        lambda row: _current_tier_for_row(row.get("sport"), row.get("market")),
        axis=1,
    )
    fallback_tier_labels = [meta[0] for meta in tier_meta]
    fallback_tier_status = [meta[1] for meta in tier_meta]
    if "tier" in work.columns:
        work["tier_label"] = work["tier"].replace("", pd.NA).fillna(pd.Series(fallback_tier_labels, index=work.index))
    else:
        work["tier_label"] = fallback_tier_labels
    if "market_status" in work.columns:
        work["tier_status"] = work["market_status"].replace("", pd.NA).fillna(pd.Series(fallback_tier_status, index=work.index))
    else:
        work["tier_status"] = fallback_tier_status

    if work.empty:
        return []

    pending_work = pd.DataFrame()
    if pending_df is not None and not pending_df.empty:
        pending_work = pending_df.copy()
        if "market_type" not in pending_work.columns:
            pending_work["market_type"] = pending_work["team_or_player"].apply(_mtype) if "team_or_player" in pending_work.columns else "moneyline"
        if "market" not in pending_work.columns:
            pending_work["market"] = pending_work["market_type"]
        else:
            pending_work["market"] = pending_work["market"].fillna(pending_work["market_type"]).replace("", pd.NA).fillna(pending_work["market_type"])
        pending_tier_meta = pending_work.apply(
            lambda row: _current_tier_for_row(row.get("sport"), row.get("market")),
            axis=1,
        )
        pending_fallback_tier_labels = [meta[0] for meta in pending_tier_meta]
        pending_fallback_tier_status = [meta[1] for meta in pending_tier_meta]
        if "tier" in pending_work.columns:
            pending_work["tier_label"] = pending_work["tier"].replace("", pd.NA).fillna(pd.Series(pending_fallback_tier_labels, index=pending_work.index))
        else:
            pending_work["tier_label"] = pending_fallback_tier_labels
        if "market_status" in pending_work.columns:
            pending_work["tier_status"] = pending_work["market_status"].replace("", pd.NA).fillna(pd.Series(pending_fallback_tier_status, index=pending_work.index))
        else:
            pending_work["tier_status"] = pending_fallback_tier_status

    pending_counts: dict[tuple[str, str, str, str], int] = {}
    if not pending_work.empty:
        grouped_pending = pending_work.groupby(["sport", "market", "tier_label", "tier_status"], dropna=False).size()
        pending_counts = {
            (str(s), str(m), str(tl), str(ts)): int(count)
            for (s, m, tl, ts), count in grouped_pending.items()
        }

    rows: list[dict] = []
    for (sport, market, tier_label, tier_status), grp in work.groupby(
        ["sport", "market", "tier_label", "tier_status"], dropna=False
    ):
        bets = len(grp)
        wins = int(grp["won"].fillna(False).astype(bool).sum()) if "won" in grp.columns else 0
        pnl = float(grp["profit_units"].fillna(0).sum()) if "profit_units" in grp.columns else 0.0
        stake = float(grp["stake_units"].fillna(0).sum()) if "stake_units" in grp.columns else 0.0
        roi = (pnl / stake * 100) if stake > 0 else 0.0
        avg_edge = float(grp["edge"].fillna(0).mean()) if "edge" in grp.columns else 0.0
        clv_series = pd.to_numeric(grp["clv"], errors="coerce") if "clv" in grp.columns else pd.Series(dtype="float64")
        clv_clean = clv_series.dropna()
        avg_clv = None if clv_clean.empty else round(float(clv_clean.mean() * 100), 2)
        clv_positive_pct = None if clv_clean.empty else round(float((clv_clean > 0).mean() * 100), 1)
        clv_covered = int(clv_clean.shape[0])
        pending_count = int(pending_counts.get((str(sport), str(market), str(tier_label), str(tier_status)), 0))
        tracked_total = bets + pending_count
        settlement_coverage_pct = round((bets / tracked_total) * 100, 1) if tracked_total else 0.0
        clv_coverage_pct = round((clv_covered / bets) * 100, 1) if bets else 0.0
        if clv_clean.empty:
            clv_signal = "missing"
        elif avg_clv > 0 and roi > 0:
            clv_signal = "confirmed"
        elif avg_clv > 0 and roi <= 0:
            clv_signal = "variance"
        elif avg_clv <= 0 and roi > 0:
            clv_signal = "lucky"
        else:
            clv_signal = "weak"
        rows.append(
            {
                "sport": str(sport),
                "market": str(market),
                "tier": str(tier_label),
                "tier_status": str(tier_status),
                "bets": bets,
                "wins": wins,
                "win_rate": round(wins / bets * 100, 1) if bets else 0,
                "pnl": round(pnl, 4),
                "roi": round(roi, 2),
                "avg_edge": round(avg_edge * 100, 2),
                "avg_clv": avg_clv,
                "clv_positive_pct": clv_positive_pct,
                "clv_covered": clv_covered,
                "clv_coverage_pct": clv_coverage_pct,
                "clv_signal": clv_signal,
                "pending_count": pending_count,
                "tracked_total": tracked_total,
                "settlement_coverage_pct": settlement_coverage_pct,
            }
        )

    for (sport, market, tier_label, tier_status), pending_count in pending_counts.items():
        if any(
            row["sport"] == sport and row["market"] == market and row["tier"] == tier_label and row["tier_status"] == tier_status
            for row in rows
        ):
            continue
        rows.append(
            {
                "sport": sport,
                "market": market,
                "tier": tier_label,
                "tier_status": tier_status,
                "bets": 0,
                "wins": 0,
                "win_rate": 0,
                "pnl": 0.0,
                "roi": 0.0,
                "avg_edge": 0.0,
                "avg_clv": None,
                "clv_positive_pct": None,
                "clv_covered": 0,
                "clv_coverage_pct": 0.0,
                "clv_signal": "missing",
                "pending_count": pending_count,
                "tracked_total": pending_count,
                "settlement_coverage_pct": 0.0,
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            -row["tracked_total"],
            -row["bets"],
            -row["roi"],
            row["sport"],
            row["market"],
            row["tier"],
        ),
    )


def _governor_recommendations(matrix_rows: list[dict]) -> list[dict]:
    recommendations: list[dict] = []
    for row in matrix_rows:
        bets = int(row.get("bets", 0) or 0)
        roi = float(row.get("roi", 0) or 0)
        avg_clv = row.get("avg_clv")
        clv_signal = str(row.get("clv_signal", "missing"))
        tier_status = str(row.get("tier_status", "experimental"))
        sport = str(row.get("sport", ""))
        market = str(row.get("market", ""))
        tier = str(row.get("tier", ""))
        lane = f"{sport}:{market}:{tier}"
        pending_count = int(row.get("pending_count", 0) or 0)
        tracked_total = int(row.get("tracked_total", bets) or bets)
        settlement_coverage_pct = float(row.get("settlement_coverage_pct", 0) or 0)
        clv_coverage_pct = float(row.get("clv_coverage_pct", 0) or 0)

        if tracked_total >= 8 and settlement_coverage_pct < 70:
            recommendations.append({
                "lane": lane,
                "sport": sport,
                "market": market,
                "tier": tier,
                "action": "watch",
                "confidence": "low",
                "reason": f"Only {settlement_coverage_pct:.1f}% of tracked bets are settled ({bets}/{tracked_total}); backlog of {pending_count} can still distort the lane read.",
            })
            continue

        if bets >= 8 and clv_coverage_pct < 60:
            recommendations.append({
                "lane": lane,
                "sport": sport,
                "market": market,
                "tier": tier,
                "action": "watch",
                "confidence": "low",
                "reason": f"Settlement is usable, but CLV coverage is only {clv_coverage_pct:.1f}% ({row.get('clv_covered', 0)}/{bets}); do not trust promotion or demotion yet.",
            })
            continue

        if bets < 8:
            recommendations.append({
                "lane": lane,
                "sport": sport,
                "market": market,
                "tier": tier,
                "action": "watch",
                "confidence": "low",
                "reason": f"Only {bets} settled bets so far — too little sample to promote or demote confidently.",
            })
            continue

        if tier_status != "preferred" and bets >= 12 and avg_clv is not None and avg_clv >= 1.5 and roi >= 3.0:
            recommendations.append({
                "lane": lane,
                "sport": sport,
                "market": market,
                "tier": tier,
                "action": "promote",
                "confidence": "high" if bets >= 20 else "medium",
                "reason": f"{bets} settled bets with {settlement_coverage_pct:.1f}% settlement coverage, ROI {roi:+.2f}%, and avg CLV {avg_clv:+.2f}% support promotion.",
            })
            continue

        if tier_status == "preferred" and bets >= 12 and avg_clv is not None and avg_clv <= -1.0 and roi <= -5.0:
            recommendations.append({
                "lane": lane,
                "sport": sport,
                "market": market,
                "tier": tier,
                "action": "demote",
                "confidence": "high" if bets >= 20 else "medium",
                "reason": f"{bets} settled bets with {settlement_coverage_pct:.1f}% settlement coverage, ROI {roi:+.2f}%, and avg CLV {avg_clv:+.2f}% suggest the lane is over-promoted.",
            })
            continue

        if tier_status != "preferred" and clv_signal == "variance" and bets >= 10:
            recommendations.append({
                "lane": lane,
                "sport": sport,
                "market": market,
                "tier": tier,
                "action": "watch",
                "confidence": "medium",
                "reason": f"Positive CLV but weak realized ROI across {bets} bets suggests variance, not a clear failure.",
            })
            continue

        if clv_signal == "lucky" and bets >= 10:
            recommendations.append({
                "lane": lane,
                "sport": sport,
                "market": market,
                "tier": tier,
                "action": "watch",
                "confidence": "medium",
                "reason": f"ROI is positive, but avg CLV is negative across {bets} bets — results may be running ahead of price quality.",
            })
            continue

        if tier_status != "preferred" and bets >= 10 and avg_clv is not None and avg_clv <= -1.0 and roi <= -3.0:
            recommendations.append({
                "lane": lane,
                "sport": sport,
                "market": market,
                "tier": tier,
                "action": "pause",
                "confidence": "medium",
                "reason": f"{bets} settled bets with negative ROI and CLV suggest the lane should stay constrained or be paused.",
            })
            continue

    action_order = {"promote": 0, "demote": 1, "pause": 2, "watch": 3}
    confidence_order = {"high": 0, "medium": 1, "low": 2}
    return sorted(
        recommendations,
        key=lambda row: (
            action_order.get(row["action"], 9),
            confidence_order.get(row["confidence"], 9),
            row["sport"],
            row["market"],
        ),
    )


def _load_replay_market_support() -> dict[tuple[str, str], dict[str, Any]]:
    results_dir = BASE / "reports" / "backtests" / "markets"
    if not results_dir.exists():
        return {}

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in results_dir.glob("*_summary.json"):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        if payload.get("status") != "ok":
            continue
        sport = str(payload.get("sport", "")).strip().lower()
        market_type = str(payload.get("market_type", "")).strip().lower()
        if not sport or not market_type:
            continue
        grouped[(sport, market_type)].append(payload)

    support: dict[tuple[str, str], dict[str, Any]] = {}
    per_sport_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (sport, market_type), items in grouped.items():
        row = {
            "sport": sport,
            "market": market_type,
            "spec_count": len(items),
            "games_scored": int(sum(int(i.get("games_scored", 0) or 0) for i in items)),
            "avg_accuracy": float(sum(float(i.get("overall_accuracy", 0) or 0) for i in items) / len(items)),
            "avg_log_loss": float(sum(float(i.get("overall_log_loss", 0) or 0) for i in items) / len(items)),
            "avg_ece": float(sum(float(i.get("overall_ece", 0) or 0) for i in items) / len(items)),
        }
        per_sport_rows[sport].append(row)

    for sport, rows in per_sport_rows.items():
        ranked = sorted(rows, key=lambda r: (r["avg_log_loss"], -r["avg_accuracy"], r["avg_ece"]))
        total = len(ranked)
        for idx, row in enumerate(ranked, start=1):
            if total <= 2:
                support_level = "strong" if idx == 1 else "weak"
            else:
                if idx <= max(1, math.ceil(total / 3)):
                    support_level = "strong"
                elif idx >= total - max(1, math.floor(total / 3)) + 1:
                    support_level = "weak"
                else:
                    support_level = "mixed"
            row["rank_within_sport"] = idx
            row["support_level"] = support_level
            support[(sport, row["market"])] = row

    return support


def _replay_support_rows(replay_support: dict[tuple[str, str], dict[str, Any]]) -> list[dict]:
    rows = []
    for (_, _), replay in replay_support.items():
        rows.append(
            {
                "sport": str(replay.get("sport", "")),
                "market": str(replay.get("market", "")),
                "support_level": str(replay.get("support_level", "missing")),
                "rank_within_sport": int(replay.get("rank_within_sport", 0) or 0),
                "spec_count": int(replay.get("spec_count", 0) or 0),
                "games_scored": int(replay.get("games_scored", 0) or 0),
                "avg_accuracy": round(float(replay.get("avg_accuracy", 0) or 0) * 100, 1),
                "avg_log_loss": round(float(replay.get("avg_log_loss", 0) or 0), 4),
                "avg_ece": round(float(replay.get("avg_ece", 0) or 0), 4),
            }
        )

    support_order = {"strong": 0, "mixed": 1, "weak": 2, "missing": 3}
    return sorted(
        rows,
        key=lambda row: (
            support_order.get(row["support_level"], 9),
            row["sport"],
            row["rank_within_sport"],
            row["market"],
        ),
    )


def _replay_policy_audit(replay_support: dict[tuple[str, str], dict[str, Any]]) -> list[dict]:
    status_rank = {"disabled": 0, "experimental": 1, "preferred": 2}
    recommended_status = {"weak": "disabled", "mixed": "experimental", "strong": "preferred"}
    recommended_label = {"disabled": "Disabled", "experimental": "Limited", "preferred": "Preferred"}
    rows: list[dict] = []

    for (_, _), replay in replay_support.items():
        sport = str(replay.get("sport", ""))
        market = str(replay.get("market", ""))
        support_level = str(replay.get("support_level", "missing"))
        current = get_market_policy(sport, market)
        target_status = recommended_status.get(support_level, "experimental")
        current_status = str(current.get("status", "experimental"))

        if status_rank.get(current_status, 1) == status_rank.get(target_status, 1):
            alignment = "aligned"
        elif status_rank.get(current_status, 1) < status_rank.get(target_status, 1):
            alignment = "underpromoted"
        else:
            alignment = "overpromoted"

        rows.append({
            "sport": sport,
            "market": market,
            "support_level": support_level,
            "current_status": current_status,
            "current_label": str(current.get("label", "Experimental")),
            "recommended_status": target_status,
            "recommended_label": recommended_label[target_status],
            "alignment": alignment,
            "rank_within_sport": int(replay.get("rank_within_sport", 0) or 0),
            "games_scored": int(replay.get("games_scored", 0) or 0),
            "avg_accuracy": round(float(replay.get("avg_accuracy", 0) or 0) * 100, 1),
            "avg_log_loss": round(float(replay.get("avg_log_loss", 0) or 0), 4),
            "avg_ece": round(float(replay.get("avg_ece", 0) or 0), 4),
        })

    alignment_order = {"overpromoted": 0, "underpromoted": 1, "aligned": 2}
    return sorted(
        rows,
        key=lambda row: (
            alignment_order.get(row["alignment"], 9),
            row["sport"],
            row["rank_within_sport"],
            row["market"],
        ),
    )


def _replay_portfolio_simulation(replay_support: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    """
    Simulate the current market policy against the replay corpus.

    This is portfolio-level rather than event-level: it answers which replayed
    lane families are currently publishable, constrained, or held out, and what
    their weighted historical quality looks like under the current stack.
    """
    bucket_rows: dict[str, list[dict[str, Any]]] = {
        "preferred_live": [],
        "limited_live": [],
        "held_out": [],
    }
    lane_rows: list[dict[str, Any]] = []

    for (_, _), replay in replay_support.items():
        sport = str(replay.get("sport", ""))
        market = str(replay.get("market", ""))
        current = get_market_policy(sport, market)
        games_scored = int(replay.get("games_scored", 0) or 0)

        if bool(current.get("production_allowed")) and str(current.get("status")) == "preferred":
            bucket = "preferred_live"
        elif bool(current.get("production_allowed")):
            bucket = "limited_live"
        else:
            bucket = "held_out"

        row = {
            "sport": sport,
            "market": market,
            "bucket": bucket,
            "status": str(current.get("status", "experimental")),
            "label": str(current.get("label", "Experimental")),
            "games_scored": games_scored,
            "avg_accuracy": float(replay.get("avg_accuracy", 0) or 0),
            "avg_log_loss": float(replay.get("avg_log_loss", 0) or 0),
            "avg_ece": float(replay.get("avg_ece", 0) or 0),
            "support_level": str(replay.get("support_level", "missing")),
        }
        bucket_rows[bucket].append(row)
        lane_rows.append(row)

    def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        games = sum(int(row["games_scored"]) for row in rows)
        if games <= 0:
            return {
                "lanes": len(rows),
                "games_scored": 0,
                "avg_accuracy": None,
                "avg_log_loss": None,
                "avg_ece": None,
            }
        return {
            "lanes": len(rows),
            "games_scored": games,
            "avg_accuracy": round(sum(row["avg_accuracy"] * row["games_scored"] for row in rows) / games * 100, 1),
            "avg_log_loss": round(sum(row["avg_log_loss"] * row["games_scored"] for row in rows) / games, 4),
            "avg_ece": round(sum(row["avg_ece"] * row["games_scored"] for row in rows) / games, 4),
        }

    published_rows = bucket_rows["preferred_live"] + bucket_rows["limited_live"]
    simulation = {
        "preferred_live": _summarize(bucket_rows["preferred_live"]),
        "limited_live": _summarize(bucket_rows["limited_live"]),
        "held_out": _summarize(bucket_rows["held_out"]),
        "published_total": _summarize(published_rows),
        "all_lanes": _summarize(lane_rows),
        "lane_rows": sorted(
            lane_rows,
            key=lambda row: (
                {"preferred_live": 0, "limited_live": 1, "held_out": 2}.get(row["bucket"], 9),
                row["sport"],
                row["market"],
            ),
        ),
    }
    return simulation


def _replay_policy_scenarios(replay_support: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    """
    Compare the current live policy with a replay-aligned policy.

    The replay-aligned scenario promotes strong replay lanes, keeps mixed lanes
    limited, and holds weak lanes out. This gives a grounded counterfactual for
    whether today's policy is too conservative or too permissive.
    """
    current_games = 0
    current_weighted_accuracy = 0.0
    current_weighted_log_loss = 0.0
    current_weighted_ece = 0.0
    current_lane_count = 0

    aligned_games = 0
    aligned_weighted_accuracy = 0.0
    aligned_weighted_log_loss = 0.0
    aligned_weighted_ece = 0.0
    aligned_lane_count = 0

    promoted_by_alignment = 0
    held_out_by_alignment = 0

    for (_, _), replay in replay_support.items():
        sport = str(replay.get("sport", ""))
        market = str(replay.get("market", ""))
        games_scored = int(replay.get("games_scored", 0) or 0)
        avg_accuracy = float(replay.get("avg_accuracy", 0) or 0)
        avg_log_loss = float(replay.get("avg_log_loss", 0) or 0)
        avg_ece = float(replay.get("avg_ece", 0) or 0)
        support_level = str(replay.get("support_level", "mixed"))

        current = get_market_policy(sport, market)
        current_live = bool(current.get("production_allowed"))
        aligned_live = support_level in {"strong", "mixed"}

        if current_live:
            current_games += games_scored
            current_weighted_accuracy += avg_accuracy * games_scored
            current_weighted_log_loss += avg_log_loss * games_scored
            current_weighted_ece += avg_ece * games_scored
            current_lane_count += 1

        if aligned_live:
            aligned_games += games_scored
            aligned_weighted_accuracy += avg_accuracy * games_scored
            aligned_weighted_log_loss += avg_log_loss * games_scored
            aligned_weighted_ece += avg_ece * games_scored
            aligned_lane_count += 1

        if aligned_live and not current_live:
            promoted_by_alignment += 1
        if current_live and not aligned_live:
            held_out_by_alignment += 1

    def _safe_summary(games: int, weighted_accuracy: float, weighted_log_loss: float, weighted_ece: float, lanes: int) -> dict[str, Any]:
        if games <= 0:
            return {
                "games_scored": 0,
                "lanes": lanes,
                "avg_accuracy": None,
                "avg_log_loss": None,
                "avg_ece": None,
            }
        return {
            "games_scored": games,
            "lanes": lanes,
            "avg_accuracy": round(weighted_accuracy / games * 100, 1),
            "avg_log_loss": round(weighted_log_loss / games, 4),
            "avg_ece": round(weighted_ece / games, 4),
        }

    current_summary = _safe_summary(
        current_games,
        current_weighted_accuracy,
        current_weighted_log_loss,
        current_weighted_ece,
        current_lane_count,
    )
    aligned_summary = _safe_summary(
        aligned_games,
        aligned_weighted_accuracy,
        aligned_weighted_log_loss,
        aligned_weighted_ece,
        aligned_lane_count,
    )
    return {
        "current_policy": current_summary,
        "replay_aligned_policy": aligned_summary,
        "delta_games": aligned_summary["games_scored"] - current_summary["games_scored"],
        "delta_lanes": aligned_summary["lanes"] - current_summary["lanes"],
        "promoted_by_alignment": promoted_by_alignment,
        "held_out_by_alignment": held_out_by_alignment,
    }


def _load_replay_event_rows(max_rows_per_lane: int = 2500) -> pd.DataFrame:
    results_dir = BASE / "reports" / "backtests" / "markets"
    if not results_dir.exists():
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for path in results_dir.glob("*_events.parquet"):
        try:
            df = pd.read_parquet(path)
        except Exception:
            continue
        if df.empty:
            continue
        keep_cols = [
            "date",
            "sport",
            "market",
            "market_type",
            "match_id",
            "home_team",
            "away_team",
            "player1_name",
            "player2_name",
            "y_true",
            "y_pred",
            "pred_confidence",
            "correct",
            "event_log_loss",
            "period",
            "train_end",
            "test_end",
        ]
        present = [c for c in keep_cols if c in df.columns]
        slim = df[present].copy()
        if "date" in slim.columns:
            slim["date"] = pd.to_datetime(slim["date"], errors="coerce")
        if max_rows_per_lane and len(slim) > max_rows_per_lane:
            slim = slim.sort_values("date").tail(max_rows_per_lane)
        frames.append(slim)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "date" in out.columns:
        out = out[out["date"].notna()].copy()
    return out


def _classify_replay_publish_decision(sport: str, market: str, pred_confidence: Any) -> tuple[str, str]:
    policy = get_market_policy(sport, market)
    confidence = float(pred_confidence) if pred_confidence is not None and not pd.isna(pred_confidence) else None

    if not bool(policy.get("production_allowed")):
        return "hold_out", "Market policy is not production-allowed for this lane."

    status = str(policy.get("status", "experimental"))
    if confidence is None:
        return "review", "Replay confidence is missing, so the event cannot be auto-published safely."

    if status == "preferred":
        if confidence >= 0.58:
            return "publish", "Preferred lane with replay confidence at or above 58%."
        return "review", "Preferred lane, but replay confidence stayed below the publish threshold."

    if confidence >= 0.66:
        return "publish", "Limited lane with replay confidence at or above 66%."
    return "review", "Limited lane, but replay confidence stayed below the publish threshold."


def _replay_slate_history(events: pd.DataFrame) -> dict[str, Any]:
    if events.empty:
        return {"rows": [], "summary": {"dates": 0, "events": 0, "published_events": 0, "held_out_events": 0}}

    df = events.copy()
    df["sport"] = df["sport"].astype(str).str.lower()
    df["market_type"] = df["market_type"].astype(str).str.lower()
    df["date_only"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df[df["date_only"].notna()].copy()
    if df.empty:
        return {"rows": [], "summary": {"dates": 0, "events": 0, "published_events": 0, "held_out_events": 0}}

    policy_buckets: list[str] = []
    policy_labels: list[str] = []
    for _, row in df.iterrows():
        policy = get_market_policy(str(row.get("sport", "")), str(row.get("market_type", "")))
        if bool(policy.get("production_allowed")) and str(policy.get("status")) == "preferred":
            bucket = "preferred_live"
        elif bool(policy.get("production_allowed")):
            bucket = "limited_live"
        else:
            bucket = "held_out"
        policy_buckets.append(bucket)
        policy_labels.append(str(policy.get("label", "Experimental")))
    df["policy_bucket"] = policy_buckets
    df["policy_label"] = policy_labels
    df["published_flag"] = df["policy_bucket"].isin({"preferred_live", "limited_live"}).astype(int)
    df["held_out_flag"] = (df["policy_bucket"] == "held_out").astype(int)
    df["preferred_flag"] = (df["policy_bucket"] == "preferred_live").astype(int)
    df["limited_flag"] = (df["policy_bucket"] == "limited_live").astype(int)
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce").fillna(0.0)
    df["event_log_loss"] = pd.to_numeric(df["event_log_loss"], errors="coerce").fillna(0.0)

    rows: list[dict[str, Any]] = []
    for date_only, grp in df.groupby("date_only", sort=False):
        published = grp[grp["published_flag"] == 1]
        held_out = grp[grp["held_out_flag"] == 1]
        rows.append({
            "date": str(date_only),
            "events": int(len(grp)),
            "published_events": int(len(published)),
            "held_out_events": int(len(held_out)),
            "preferred_events": int(grp["preferred_flag"].sum()),
            "limited_events": int(grp["limited_flag"].sum()),
            "sports": int(grp["sport"].nunique()),
            "markets": int(grp["market_type"].nunique()),
            "published_accuracy": round(float(published["correct"].mean()) * 100, 1) if len(published) else None,
            "published_log_loss": round(float(published["event_log_loss"].mean()), 4) if len(published) else None,
            "held_out_accuracy": round(float(held_out["correct"].mean()) * 100, 1) if len(held_out) else None,
            "top_lanes": sorted(
                [f"{sport}:{market}" for sport, market in grp.groupby(["sport", "market_type"]).size().index]
            )[:6],
        })

    rows = sorted(rows, key=lambda row: row["date"], reverse=True)
    return {
        "rows": rows,
        "summary": {
            "dates": len(rows),
            "events": int(len(df)),
            "published_events": int(df["published_flag"].sum()),
            "held_out_events": int(df["held_out_flag"].sum()),
        },
    }


def _replay_slate_event_rows(events: pd.DataFrame, max_dates: int = 8, max_events_per_date: int = 20) -> list[dict[str, Any]]:
    if events.empty:
        return []

    df = events.copy()
    df["sport"] = df["sport"].astype(str).str.lower()
    df["market_type"] = df["market_type"].astype(str).str.lower()
    df["date_only"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df[df["date_only"].notna()].copy()
    if df.empty:
        return []

    policy_buckets: list[str] = []
    policy_labels: list[str] = []
    for _, row in df.iterrows():
        policy = get_market_policy(str(row.get("sport", "")), str(row.get("market_type", "")))
        if bool(policy.get("production_allowed")) and str(policy.get("status")) == "preferred":
            bucket = "preferred_live"
        elif bool(policy.get("production_allowed")):
            bucket = "limited_live"
        else:
            bucket = "held_out"
        policy_buckets.append(bucket)
        policy_labels.append(str(policy.get("label", "Experimental")))
    df["policy_bucket"] = policy_buckets
    df["policy_label"] = policy_labels
    df["pred_confidence"] = pd.to_numeric(df.get("pred_confidence"), errors="coerce")
    df["event_log_loss"] = pd.to_numeric(df.get("event_log_loss"), errors="coerce")
    df["correct"] = pd.to_numeric(df.get("correct"), errors="coerce")

    selected_dates = sorted(df["date_only"].dropna().unique().tolist(), reverse=True)[:max_dates]
    out: list[dict[str, Any]] = []
    for date_only in selected_dates:
        grp = df[df["date_only"] == date_only].copy()
        grp = grp.sort_values(
            by=["policy_bucket", "pred_confidence", "sport", "market_type", "match_id"],
            ascending=[True, False, True, True, True],
        ).head(max_events_per_date)
        for _, row in grp.iterrows():
            publish_decision, publish_reason = _classify_replay_publish_decision(
                str(row.get("sport", "")),
                str(row.get("market_type", "")),
                row.get("pred_confidence"),
            )
            out.append({
                "date": str(date_only),
                "sport": str(row.get("sport", "")),
                "market": str(row.get("market_type", "")),
                "match_id": str(row.get("match_id", "")),
                "policy_bucket": str(row.get("policy_bucket", "")),
                "policy_label": str(row.get("policy_label", "")),
                "pred_confidence": round(float(row["pred_confidence"]), 4) if pd.notna(row["pred_confidence"]) else None,
                "correct": bool(int(row["correct"])) if pd.notna(row["correct"]) else None,
                "event_log_loss": round(float(row["event_log_loss"]), 4) if pd.notna(row["event_log_loss"]) else None,
                "publish_decision": publish_decision,
                "publish_reason": publish_reason,
            })
    return out


def _replay_publish_audit(replay_slate_events: list[dict[str, Any]]) -> dict[str, Any]:
    if not replay_slate_events:
        return {
            "summary": {"dates": 0, "publish": 0, "review": 0, "hold_out": 0},
            "rows": [],
        }

    df = pd.DataFrame(replay_slate_events)
    rows: list[dict[str, Any]] = []
    for date_only, grp in df.groupby("date", sort=False):
        publish = int((grp["publish_decision"] == "publish").sum())
        review = int((grp["publish_decision"] == "review").sum())
        hold_out = int((grp["publish_decision"] == "hold_out").sum())
        rows.append({
            "date": str(date_only),
            "events": int(len(grp)),
            "publish": publish,
            "review": review,
            "hold_out": hold_out,
            "publish_rate": round(publish / len(grp) * 100, 1) if len(grp) else None,
        })

    summary = {
        "dates": len(rows),
        "publish": int((df["publish_decision"] == "publish").sum()),
        "review": int((df["publish_decision"] == "review").sum()),
        "hold_out": int((df["publish_decision"] == "hold_out").sum()),
    }
    rows = sorted(rows, key=lambda row: row["date"], reverse=True)
    return {"summary": summary, "rows": rows}


def _parlay_style_label(tier: Any) -> str:
    text = str(tier or "").strip().lower()
    if text in {"speculative", "longshot"}:
        return "Longshot"
    if text == "value":
        return "Value"
    return text.title() or "Unknown"


def _parlay_style_from_type(parlay_type: Any) -> str:
    text = str(parlay_type or "").strip().lower()
    if text in {"ai_longshot", "speculative", "longshot"}:
        return "Longshot"
    if text in {"ai_value", "value"}:
        return "Value"
    return "Custom"


def _parlay_source_label(parlay_type: Any) -> str:
    text = str(parlay_type or "").strip().lower()
    if text in {"ai_value", "ai_longshot"}:
        return "AI"
    if text == "manual":
        return "Manual"
    return "System"


def _classify_parlay_bracket(odds: Any) -> str:
    try:
        value = float(odds or 0)
    except Exception:
        return "unknown"
    if 4.0 <= value < 6.5:
        return "5x"
    if 6.5 <= value < 14.0:
        return "10x"
    if 20.0 <= value < 40.0:
        return "20x"
    return "other"


def _parlay_sport_mix_label(legs_json: Any) -> str:
    try:
        payload = json.loads(legs_json or "[]")
    except Exception:
        return "Unknown"
    sports = sorted({str(leg.get("sport", "")).strip().lower() for leg in payload if leg.get("sport")})
    if not sports:
        return "Unknown"
    if len(sports) == 1:
        return sports[0]
    return "mixed"


def _parlay_performance_matrix(date_filter: Optional[str] = None, manual_parlays: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
    system_df = _parlay_breakdown()
    frames: list[pd.DataFrame] = []

    if not system_df.empty:
        sys_work = system_df.copy()
        sys_work["recorded_date"] = pd.to_datetime(sys_work["recorded_at"], errors="coerce").dt.strftime("%Y-%m-%d")
        sys_work["source"] = "System"
        sys_work["style"] = sys_work["tier"].apply(_parlay_style_label)
        sys_work["sport_mix"] = sys_work["legs_json"].apply(_parlay_sport_mix_label)
        frames.append(sys_work)

    manual_rows: list[dict[str, Any]] = []
    for parlay in manual_parlays or []:
        status = str(parlay.get("status", "")).lower()
        if status not in {"won", "lost"}:
            continue
        saved_at = parlay.get("saved_at") or parlay.get("settled_at") or ""
        recorded_date = str(parlay.get("date", ""))[:10]
        kelly_pct = float(parlay.get("kelly_pct") or 0.0)
        stake_units = round(kelly_pct / 100, 6)
        combined_odds = float(parlay.get("combined_odds") or 0.0)
        profit_units = round((combined_odds - 1) * stake_units if status == "won" else -stake_units, 4)
        legs = parlay.get("legs") or []
        manual_rows.append({
            "source": _parlay_source_label(parlay.get("type")),
            "style": _parlay_style_from_type(parlay.get("type")),
            "bracket": str(parlay.get("bracket") or _classify_parlay_bracket(combined_odds)),
            "n_legs": int(parlay.get("n_legs") or len(legs) or 0),
            "combined_odds": combined_odds,
            "edge": float(parlay.get("edge") or 0.0) / 100.0,
            "ev": float(parlay.get("ev") or 0.0),
            "stake_units": stake_units,
            "status": status,
            "won": status == "won",
            "profit_units": profit_units,
            "recorded_at": saved_at,
            "recorded_date": recorded_date,
            "legs_json": json.dumps(legs),
            "sport_mix": _parlay_sport_mix_label(json.dumps(legs)),
        })
    if manual_rows:
        frames.append(pd.DataFrame(manual_rows))

    if not frames:
        return {"summary_cards": [], "matrix_rows": [], "detail_rows": []}

    work = pd.concat(frames, ignore_index=True)
    if date_filter:
        work = work[work["recorded_date"] == date_filter].copy()
    if work.empty:
        return {"summary_cards": [], "matrix_rows": [], "detail_rows": []}

    summary_cards = []
    for (source, style), grp in work.groupby(["source", "style"], dropna=False):
        if grp.empty:
            continue
        bets = int(len(grp))
        wins = int(grp["won"].sum())
        stake = float(grp["stake_units"].sum())
        pnl = float(grp["profit_units"].sum())
        roi = pnl / stake if stake > 0 else 0.0
        summary_cards.append({
            "source": str(source),
            "style": style,
            "bets": bets,
            "win_rate": round(wins / bets * 100, 1) if bets else 0.0,
            "roi": round(roi * 100, 2),
            "pnl": round(pnl, 4),
            "avg_odds": round(float(grp["combined_odds"].mean()), 2),
        })

    matrix_rows: list[dict[str, Any]] = []
    grouped = work.groupby(["source", "style", "bracket", "n_legs", "sport_mix"], dropna=False)
    for (source, style, bracket, n_legs, sport_mix), grp in grouped:
        bets = int(len(grp))
        wins = int(grp["won"].sum())
        stake = float(grp["stake_units"].sum())
        pnl = float(grp["profit_units"].sum())
        roi = pnl / stake if stake > 0 else 0.0
        matrix_rows.append({
            "source": str(source),
            "style": str(style),
            "bracket": str(bracket or "unknown"),
            "n_legs": int(n_legs or 0),
            "sport_mix": str(sport_mix),
            "bets": bets,
            "win_rate": round(wins / bets * 100, 1) if bets else 0.0,
            "roi": round(roi * 100, 2),
            "pnl": round(pnl, 4),
            "avg_odds": round(float(grp["combined_odds"].mean()), 2),
            "avg_edge": round(float(grp["edge"].mean()) * 100, 2),
            "avg_ev": round(float(grp["ev"].mean()), 3),
        })

    source_order = {"System": 0, "AI": 1, "Manual": 2}
    style_order = {"Value": 0, "Longshot": 1, "Custom": 2}
    bracket_order = {"5x": 0, "10x": 1, "20x": 2, "other": 3}
    matrix_rows = sorted(
        matrix_rows,
        key=lambda row: (
            source_order.get(row["source"], 9),
            style_order.get(row["style"], 9),
            bracket_order.get(row["bracket"], 9),
            row["n_legs"],
            row["sport_mix"],
        ),
    )
    detail_rows = []
    detail_work = work.copy()
    detail_work["recorded_at_ts"] = pd.to_datetime(detail_work["recorded_at"], errors="coerce", utc=True)
    detail_work["recorded_at_str"] = detail_work["recorded_at_ts"].dt.strftime("%Y-%m-%d %H:%M")
    for _, row in detail_work.sort_values("recorded_at_ts", ascending=False).iterrows():
        detail_rows.append({
            "source": str(row.get("source", "")),
            "style": str(row.get("style", "")),
            "bracket": str(row.get("bracket", "")),
            "n_legs": int(row.get("n_legs", 0) or 0),
            "sport_mix": str(row.get("sport_mix", "")),
            "combined_odds": round(float(row.get("combined_odds", 0) or 0), 2),
            "edge": round(float(row.get("edge", 0) or 0) * 100, 2),
            "ev": round(float(row.get("ev", 0) or 0), 3),
            "stake_units": round(float(row.get("stake_units", 0) or 0), 4),
            "profit_units": round(float(row.get("profit_units", 0) or 0), 4),
            "won": bool(row.get("won", False)),
            "recorded_at": row.get("recorded_at_str") or "",
            "legs_json": str(row.get("legs_json", "[]")),
        })
    return {"summary_cards": summary_cards, "matrix_rows": matrix_rows, "detail_rows": detail_rows}


def _apply_replay_validation(
    recommendations: list[dict],
    replay_support: dict[tuple[str, str], dict[str, Any]],
) -> list[dict]:
    validated: list[dict] = []
    for rec in recommendations:
        sport = str(rec.get("sport", "")).lower()
        market = str(rec.get("market", "")).lower()
        replay = replay_support.get((sport, market))
        out = dict(rec)
        if not replay:
            out["replay_support"] = "missing"
            out["replay_note"] = "No replay summary is available for this sport/market lane yet."
            if out["action"] in {"promote", "demote", "pause"}:
                out["action"] = "watch"
                out["confidence"] = "low"
                out["reason"] = f"{out['reason']} Replay support is missing, so the lane should stay under watch."
            validated.append(out)
            continue

        out["replay_support"] = replay["support_level"]
        out["replay_note"] = (
            f"Replay rank {replay['rank_within_sport']} in {sport} with "
            f"log loss {replay['avg_log_loss']:.4f}, accuracy {replay['avg_accuracy']:.4f}, "
            f"ECE {replay['avg_ece']:.4f} across {replay['games_scored']} scored games."
        )

        if out["action"] == "promote" and replay["support_level"] == "weak":
            out["action"] = "watch"
            out["confidence"] = "low"
            out["reason"] = f"{out['reason']} Historical replay for {sport} {market} is weak, so promotion is not validated yet."
        elif out["action"] in {"demote", "pause"} and replay["support_level"] == "strong":
            out["action"] = "watch"
            out["confidence"] = "low"
            out["reason"] = f"{out['reason']} Historical replay for {sport} {market} is strong, so live weakness should be investigated before demotion."

        validated.append(out)

    action_order = {"promote": 0, "demote": 1, "pause": 2, "watch": 3}
    confidence_order = {"high": 0, "medium": 1, "low": 2}
    return sorted(
        validated,
        key=lambda row: (
            action_order.get(row["action"], 9),
            confidence_order.get(row["confidence"], 9),
            row["sport"],
            row["market"],
        ),
    )


def _governor_change_preview(recommendations: list[dict]) -> list[dict]:
    previews: list[dict] = []
    for rec in recommendations:
        action = str(rec.get("action", "watch"))
        if action == "watch":
            continue

        sport = str(rec.get("sport", ""))
        market = str(rec.get("market", ""))
        tier = str(rec.get("tier", ""))
        current = get_market_policy(sport, market)
        draft = {
            "status": current.get("status"),
            "label": current.get("label"),
            "production_allowed": current.get("production_allowed"),
            "parlay_allowed": current.get("parlay_allowed"),
            "stake_multiplier": current.get("stake_multiplier"),
        }

        if action == "promote":
            draft.update({
                "status": "preferred",
                "label": "Preferred",
                "production_allowed": True,
                "parlay_allowed": True,
                "stake_multiplier": 1.0,
            })
        elif action == "demote":
            draft.update({
                "status": "experimental",
                "label": "Limited",
                "production_allowed": True,
                "parlay_allowed": False,
                "stake_multiplier": min(float(current.get("stake_multiplier", 0.4) or 0.4), 0.4),
            })
        elif action == "pause":
            draft.update({
                "status": "disabled",
                "label": "Disabled",
                "production_allowed": False,
                "parlay_allowed": False,
                "stake_multiplier": 0.0,
            })
        else:
            continue

        changed_fields = [
            field for field in draft
            if draft[field] != current.get(field)
        ]
        if not changed_fields:
            continue

        previews.append({
            "sport": sport,
            "market": market,
            "tier": tier,
            "action": action,
            "confidence": rec.get("confidence", "low"),
            "reason": rec.get("reason", ""),
            "replay_support": rec.get("replay_support", "missing"),
            "file": str(BASE / "src" / "markets" / "policy.py"),
            "current": {
                "status": current.get("status"),
                "label": current.get("label"),
                "production_allowed": current.get("production_allowed"),
                "parlay_allowed": current.get("parlay_allowed"),
                "stake_multiplier": current.get("stake_multiplier"),
            },
            "draft": draft,
            "changed_fields": changed_fields,
            "summary": (
                f"{sport} {market} would move from {current.get('label')} "
                f"to {draft.get('label')} in the market policy draft."
            ),
        })

    action_order = {"promote": 0, "demote": 1, "pause": 2}
    confidence_order = {"high": 0, "medium": 1, "low": 2}
    return sorted(
        previews,
        key=lambda row: (
            action_order.get(row["action"], 9),
            confidence_order.get(str(row["confidence"]), 9),
            row["sport"],
            row["market"],
        ),
    )


@app.route("/api/results/dates")
def api_results_dates():
    """Return sorted list of event dates with tracker or saved parlay activity."""
    pred_path = BASE / "data" / "tracker" / "predictions.parquet"
    settled_path = BASE / "data" / "tracker" / "settled.parquet"
    dates: set[str] = set()
    if pred_path.exists():
        pred = pd.read_parquet(pred_path)
        if len(pred):
            dates.update(_event_date_series(pred).dropna().tolist())
    if settled_path.exists() and os.path.getsize(settled_path) >= 100:
        settled = pd.read_parquet(settled_path)
        if len(settled):
            dates.update(_event_date_series(settled).dropna().tolist())
    for p in _load_manual_parlays():
        if p.get("date"):
            dates.add(str(p["date"])[:10])
    dates = sorted(dates, reverse=True)
    return jsonify({"dates": dates})


@app.route("/api/results")
def api_results():
    """Return tracker data with optional ?date=YYYY-MM-DD filter."""
    pred_path    = BASE / "data" / "tracker" / "predictions.parquet"
    settled_path = BASE / "data" / "tracker" / "settled.parquet"
    date_filter  = request.args.get("date", None)   # e.g. "2026-04-18"

    # ── Load predictions (needed for pending bets + parlay grouping) ────────
    if not pred_path.exists():
        return jsonify({"error": "No predictions yet.", "settled": [], "pending": [],
                        "parlays": [], "pnl": [], "by_sport": {}, "by_market": {},
                        "segments": {"value_bets": _stats_from_settled(pd.DataFrame()), "parlays": _stats_from_settled(pd.DataFrame())}})

    pred = pd.read_parquet(pred_path)
    if "status" not in pred.columns:
        pred["status"] = "pending"
    if "is_parlay_leg" not in pred.columns:
        pred["is_parlay_leg"] = False
    pred["date"] = _event_date_series(pred)

    if date_filter:
        pred_day = pred[pred["date"] == date_filter]
    else:
        pred_day = pred

    # ── Pending bets for the selected day ──────────────────────────────────
    pending_rows = pred_day[pred_day["status"] == "pending"].copy()
    if "team_or_player" in pending_rows.columns:
        pending_rows["market_type"] = pending_rows["team_or_player"].apply(_mtype)
    else:
        pending_rows["market_type"] = "moneyline"
    pending_cols = ["pred_id", "date", "sport", "match_id", "team_or_player", "bet_odds",
                    "edge", "stake_units", "market_type", "market", "market_status", "tier",
                    "bookmaker", "status", "commence_time", "version_snapshot",
                    "is_parlay_leg", "parlay_id", "ml_prob"]
    pending_cols = [c for c in pending_cols if c in pending_rows.columns]
    pending_records = pending_rows[pending_cols].to_dict(orient="records")
    for r in pending_records:
        r["is_parlay_leg"] = bool(r.get("is_parlay_leg", False))
        timing = _derive_event_timing(r.get("commence_time"), r.get("sport"))
        r["status"] = timing["status"]
        r["status_label"] = timing["status_label"]
        r["time_label"] = timing["time_label"]
        r["kick_off"] = timing["kick_off"]
        r["version_snapshot"] = _parse_version_snapshot(r.get("version_snapshot"))
        r.update(enrich_with_capability(r))

    # ── Parlays for the selected day ───────────────────────────────────────
    # Group parlay legs from predictions by parlay_id
    parlays = []
    parlay_pred = pred_day[pred_day["is_parlay_leg"] == True].copy()
    if len(parlay_pred) > 0 and "parlay_id" in parlay_pred.columns:
        for pid, grp in parlay_pred.groupby("parlay_id"):
            if not pid:
                continue
            combined_odds = 1.0
            win_prob = 1.0
            legs = []
            for _, row in grp.iterrows():
                combined_odds *= float(row.get("bet_odds", 1))
                win_prob      *= float(row.get("ml_prob", 0.5))
                legs.append({
                    "team":       row.get("team_or_player", ""),
                    "match":      row.get("match_id", ""),
                    "sport":      row.get("sport", ""),
                    "odds":       round(float(row.get("bet_odds", 1)), 2),
                    "ml_prob":    round(float(row.get("ml_prob", 0.5)), 4),
                    "edge":       round(float(row.get("edge", 0)), 4),
                    "market":     _mtype(row.get("team_or_player", "")),
                    "status":     row.get("status", "pending"),
                })
            win_pct = win_prob * 100
            if win_pct >= 0.1:
                wp_str = f"{win_pct:.2f}%"
            elif win_pct >= 0.001:
                wp_str = f"{win_pct:.4f}%"
            else:
                one_in = int(round(1 / win_prob)) if win_prob > 0 else 0
                wp_str = f"1 in {one_in:,}"
            ev = win_prob * combined_odds
            parlays.append({
                "parlay_id":     pid,
                "date":          grp.iloc[0]["date"],
                "legs":          legs,
                "n_legs":        len(legs),
                "combined_odds": round(combined_odds, 2),
                "win_prob":      wp_str,
                "ev":            round(ev, 3),
            })

    # ── Settled bets ───────────────────────────────────────────────────────
    if not settled_path.exists() or os.path.getsize(settled_path) < 100:
        settled_all = pd.DataFrame()
    else:
        settled_all = pd.read_parquet(settled_path)
        if len(settled_all):
            settled_all["date"] = _event_date_series(settled_all)
            settled_all["market_type"] = settled_all["team_or_player"].apply(_mtype)

    if len(settled_all):
        is_parlay_leg = settled_all["is_parlay_leg"].fillna(False).astype(bool) if "is_parlay_leg" in settled_all.columns else pd.Series(False, index=settled_all.index)
        value_settled_all = settled_all[(settled_all["sport"] != "parlay") & (~is_parlay_leg)].copy()
        parlay_settled_all = settled_all[settled_all["sport"] == "parlay"].copy()
    else:
        value_settled_all = pd.DataFrame()
        parlay_settled_all = pd.DataFrame()

    if date_filter and len(value_settled_all):
        settled = value_settled_all[value_settled_all["date"] == date_filter]
    else:
        settled = value_settled_all

    if date_filter and len(parlay_settled_all):
        parlay_settled = parlay_settled_all[parlay_settled_all["date"] == date_filter]
    else:
        parlay_settled = parlay_settled_all

    # ── Stats (computed over filtered window) ─────────────────────────────
    if len(settled) == 0:
        overall = {"bets": 0, "wins": 0, "win_rate": 0, "pnl": 0, "roi": 0}
        pnl_daily = []
        by_sport  = {}
        by_market = {}
        records   = []
    else:
        pnl_daily_df = (
            value_settled_all.groupby("date")["profit_units"]   # cumulative always uses all-time singles
            .sum().cumsum().reset_index()
            .rename(columns={"profit_units": "cumulative_pnl"})
        )
        pnl_daily_df["cumulative_pnl"] = pnl_daily_df["cumulative_pnl"].round(4)
        pnl_daily = pnl_daily_df.to_dict(orient="records")

        by_sport = {}
        for sport, grp in settled.groupby("sport"):
            wins  = int(grp["won"].sum())
            total = len(grp)
            pnl   = float(grp["profit_units"].sum())
            stake = grp["stake_units"].sum()
            roi   = pnl / stake if stake > 0 else 0
            by_sport[sport] = {
                "wins": wins, "total": total,
                "win_rate": round(wins / total * 100, 1) if total else 0,
                "pnl": round(pnl, 4), "roi": round(roi * 100, 2),
            }

        by_market = {}
        for mtype, grp in settled.groupby("market_type"):
            wins  = int(grp["won"].sum())
            total = len(grp)
            pnl   = float(grp["profit_units"].sum())
            stake = grp["stake_units"].sum()
            roi   = pnl / stake if stake > 0 else 0
            by_market[mtype] = {
                "wins": wins, "total": total,
                "win_rate": round(wins / total * 100, 1) if total else 0,
                "pnl": round(pnl, 4), "roi": round(roi * 100, 2),
            }

        recent  = settled.sort_values("settled_at", ascending=False)
        cols    = ["date", "sport", "match_id", "team_or_player", "bet_odds",
                   "edge", "stake_units", "won", "profit_units", "market_type", "status",
                   "market", "market_status", "tier", "bookmaker", "version_snapshot",
                   "is_parlay_leg", "parlay_id", "clv", "commence_time", "settled_at"]
        cols    = [c for c in cols if c in recent.columns]
        records = recent[cols].to_dict(orient="records")
        for r in records:
            r["won"] = bool(r.get("won", False))
            r["is_parlay_leg"] = bool(r.get("is_parlay_leg", False))
            r["market"] = r.pop("market_type", "moneyline")
            # Normalise timestamps to ISO strings
            for tf in ("commence_time", "settled_at"):
                if tf in r and hasattr(r[tf], "isoformat"):
                    r[tf] = r[tf].isoformat()
                elif tf in r and r[tf] is not None:
                    r[tf] = str(r[tf])
            timing = _derive_event_timing(r.get("commence_time"), r.get("sport"))
            r["status"] = timing["status"]
            r["status_label"] = timing["status_label"]
            r["time_label"] = timing["time_label"]
            r["kick_off"] = timing["kick_off"]
            r["version_snapshot"] = _parse_version_snapshot(r.get("version_snapshot"))
            r.update(enrich_with_capability(r))
        overall = _stats_from_settled(settled)

    # ── System parlays from markdown report ───────────────────────────────
    report_date  = date_filter or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    report_path  = BASE / "reports" / f"value_bets_{report_date}.md"
    system_parlays = []
    if report_path.exists():
        for p in _parse_parlays_from_md(report_path):
            p["source"] = "system"
            system_parlays.append(p)

    # ── Manual parlays saved by user ───────────────────────────────────────
    manual_parlays = _load_manual_parlays()
    if date_filter:
        manual_parlays = [p for p in manual_parlays if p.get("date", "").startswith(date_filter)]
    for p in manual_parlays:
        p["source"] = "manual"

    # Match settled parlay rows back onto visible parlay cards when possible.
    parlay_results = []
    if len(parlay_settled):
        parlay_results = parlay_settled.sort_values("settled_at", ascending=False).to_dict(orient="records")
        parlay_lookup = {
            (str(r.get("team_or_player", "")), str(r.get("date", ""))): r
            for r in parlay_results
        }
        for bucket in (system_parlays, manual_parlays):
            for p in bucket:
                key = (str(p.get("name") or f"{p.get('n_legs', '?')}-leg Parlay"), str(p.get("date", ""))[:10])
                settled_row = parlay_lookup.get(key)
                if settled_row:
                    p["status"] = "won" if settled_row.get("won") else "lost"
                    p["profit_units"] = round(float(settled_row.get("profit_units", 0)), 4)

    segments = {
        "value_bets": _stats_from_settled(settled),
        "parlays": _stats_from_settled(parlay_settled),
    }
    settled_all_rows = pd.concat([settled, parlay_settled], ignore_index=True) if (len(settled) or len(parlay_settled)) else pd.DataFrame()
    overall = _stats_from_settled(settled_all_rows)
    recent_singles = _recent_settled_slice(settled, limit=30)
    evaluation_summary = {
        "all_bets": overall,
        "singles_only": _stats_from_settled(settled),
        "recent_singles": _stats_from_settled(recent_singles),
    }
    version_summary = _version_summary(_today_summary(), pred, settled_all)
    parlay_performance = _parlay_performance_matrix(date_filter, manual_parlays)
    settlement_reliability = _settlement_reliability(pred_day if isinstance(pred_day, pd.DataFrame) else pred, settled_all, manual_parlays)
    mistake_reports = _mistake_report(date_filter)

    performance_matrix = _performance_matrix(settled, pending_rows)
    odds_buckets = _odds_bucket_summary(settled)
    lane_highlights = _lane_highlights(performance_matrix)
    calibration_snapshot = _calibration_snapshot(settled)
    calibration_governor = _calibration_governor(calibration_snapshot)
    active_calibration_status = _active_calibration_status()
    retrain_triggers = _retrain_trigger_rows(performance_matrix, version_summary)
    replay_support = _load_replay_market_support()
    replay_events = _load_replay_event_rows()
    replay_support_matrix = _replay_support_rows(replay_support)
    replay_policy_audit = _replay_policy_audit(replay_support)
    replay_portfolio = _replay_portfolio_simulation(replay_support)
    replay_scenarios = _replay_policy_scenarios(replay_support)
    replay_slates = _replay_slate_history(replay_events)
    replay_slate_events = _replay_slate_event_rows(replay_events)
    replay_publish_audit = _replay_publish_audit(replay_slate_events)
    governor_recommendations = _apply_replay_validation(
        _governor_recommendations(performance_matrix),
        replay_support,
    )
    governor_change_preview = _governor_change_preview(governor_recommendations)
    rebuild_candidates = _rebuild_candidates(
        performance_matrix,
        retrain_triggers,
        governor_recommendations,
        replay_support,
        version_summary,
        calibration_snapshot,
    )
    return jsonify(_json_safe({
        "overall":        overall,
        "evaluation_summary": evaluation_summary,
        "segments":       segments,
        "version_summary": version_summary,
        "settlement_reliability": settlement_reliability,
        "mistake_reports": mistake_reports,
        "retrain_triggers": retrain_triggers,
        "parlay_performance": parlay_performance,
        "pnl":            pnl_daily,
        "by_sport":       by_sport,
        "by_market":      by_market,
        "performance_matrix": performance_matrix,
        "odds_buckets": odds_buckets,
        "lane_highlights": lane_highlights,
        "calibration_snapshot": calibration_snapshot,
        "calibration_governor": calibration_governor,
        "active_calibration_status": active_calibration_status,
        "replay_support_matrix": replay_support_matrix,
        "replay_policy_audit": replay_policy_audit,
        "replay_portfolio": replay_portfolio,
        "replay_scenarios": replay_scenarios,
        "replay_slates": replay_slates,
        "replay_slate_events": replay_slate_events,
        "replay_publish_audit": replay_publish_audit,
        "governor_recommendations": governor_recommendations,
        "governor_change_preview": governor_change_preview,
        "rebuild_candidates": rebuild_candidates,
        "settled":        records if len(settled) else [],
        "pending":        pending_records,
        "parlays":        parlays,          # parlay legs from predictions parquet
        "parlay_results": parlay_results,
        "system_parlays": system_parlays,   # from scan markdown report
        "manual_parlays": manual_parlays,   # user-saved from parlay builder
        "date_filter":    date_filter,
    }))


@app.route("/api/results/settle", methods=["POST"])
def api_settle_bet():
    """
    Settle a pending bet (move from predictions to settled).
    Body: { pred_id: str, won: bool, closing_odds?: float }
    """
    data         = request.get_json()
    pred_id      = data.get("pred_id", "").strip()
    won          = bool(data.get("won", False))
    closing_odds = data.get("closing_odds", None)

    if not pred_id:
        return jsonify({"error": "pred_id required"}), 400

    pred_path    = BASE / "data" / "tracker" / "predictions.parquet"
    settled_path = BASE / "data" / "tracker" / "settled.parquet"

    if not pred_path.exists():
        return jsonify({"error": "No predictions file"}), 404

    pred = pd.read_parquet(pred_path)
    row_mask = pred["pred_id"] == pred_id
    if not row_mask.any():
        return jsonify({"error": "Bet not found"}), 404

    row = pred[row_mask].iloc[0].to_dict()

    # Compute profit: (odds - 1) * stake if won, else -stake
    stake       = float(row.get("stake_units", 0))
    odds        = float(row.get("bet_odds", 1))
    profit      = round((odds - 1) * stake if won else -stake, 4)
    clv         = None
    if closing_odds:
        try:
            closing_odds = float(closing_odds)
            clv = round((odds / closing_odds) - 1.0, 6) if closing_odds > 1.0 else None
        except Exception:
            pass

    settled_row = {
        "pred_id":        pred_id,
        "settled_at":     datetime.now(timezone.utc).isoformat(),
        "sport":          row.get("sport", ""),
        "match_id":       row.get("match_id", ""),
        "team_or_player": row.get("team_or_player", ""),
        "commence_time":  row.get("commence_time", ""),
        "recorded_at":    row.get("recorded_at", ""),
        "market":         row.get("market", row.get("market_type", "moneyline")),
        "market_status":  row.get("market_status", "experimental"),
        "tier":           row.get("tier", "Experimental"),
        "bet_odds":       odds,
        "bookmaker":      row.get("bookmaker", "unknown"),
        "edge":           float(row.get("edge", 0)),
        "ml_prob":        float(row.get("ml_prob", 0)),
        "fair_prob":      float(row.get("fair_prob", 0)),
        "stake_units":    stake,
        "kelly_stake_pct": float(row.get("kelly_stake_pct", 0)),
        "is_parlay_leg":  bool(row.get("is_parlay_leg", False)),
        "version_snapshot": row.get("version_snapshot", ""),
        "actual_result":  "won" if won else "lost",
        "won":            won,
        "profit_units":   profit,
        "closing_odds":   closing_odds,
        "clv":            clv,
        "status":         "won" if won else "lost",
    }

    # Append to settled parquet
    new_row_df = pd.DataFrame([settled_row])
    if settled_path.exists() and settled_path.stat().st_size > 100:
        existing = pd.read_parquet(settled_path)
        settled_df = pd.concat([existing, new_row_df], ignore_index=True)
    else:
        settled_df = new_row_df
    settled_df.to_parquet(settled_path, index=False)

    # Remove from predictions parquet
    pred_updated = pred[~row_mask]
    pred_updated.to_parquet(pred_path, index=False)

    return jsonify({
        "settled": True,
        "pred_id": pred_id,
        "won":     won,
        "profit":  profit,
        "pick":    row.get("team_or_player", ""),
    })


@app.route("/api/results/settle-parlay", methods=["POST"])
def api_settle_parlay():
    """
    Settle a parlay as a unit.
    Body: {
      won: bool,
      source: "manual" | "system",
      parlay_id?: str,          # manual parlays: the id from manual_parlays.json
      parlay_data: {            # full parlay object echoed back from the frontend
        combined_odds, legs, kelly_stake, ev, win_prob, n_legs, name?, date?
      }
    }
    Writes one row to settled.parquet representing the whole parlay.
    For manual parlays, also updates status in manual_parlays.json.
    """
    import uuid as _uuid
    data        = request.get_json()
    won         = bool(data.get("won", False))
    source      = data.get("source", "manual")   # "manual" | "system"
    parlay_id   = data.get("parlay_id", "")
    pdata       = data.get("parlay_data", {})
    settled_path = BASE / "data" / "tracker" / "settled.parquet"

    # --- derive stake and profit ------------------------------------------
    # For manual parlays, kelly_stake is stored as a number (units).
    # For system parlays, kelly_stake is a string like "£25.20" — strip £.
    raw_kelly = pdata.get("kelly_stake", 0)
    if isinstance(raw_kelly, str):
        try:
            stake_units = float(raw_kelly.replace("£", "").strip())
        except ValueError:
            stake_units = 0.0
    else:
        stake_units = float(raw_kelly or 0)

    bankroll = float(os.getenv("INITIAL_BANKROLL", 1000))
    # kelly_stake for system parlays is an absolute £ figure — convert to units
    if stake_units > 10:   # it's a £ amount, not fraction of bankroll
        stake_units = round(stake_units / bankroll, 4)

    combined_odds = float(pdata.get("combined_odds", 1))
    profit = round((combined_odds - 1) * stake_units if won else -stake_units, 4)

    n_legs = pdata.get("n_legs") or len(pdata.get("legs", []))
    name   = pdata.get("name") or f"{n_legs}-leg Parlay"
    date   = pdata.get("date") or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    now_ts = pd.Timestamp(datetime.now(timezone.utc))
    date_ts = pd.Timestamp(date + "T12:00:00", tz="UTC")

    settled_row = {
        "pred_id":         str(_uuid.uuid4())[:8],
        "settled_at":      now_ts.isoformat(),
        "sport":           "parlay",
        "match_id":        name,
        "team_or_player":  name,
        "commence_time":   date_ts,
        "recorded_at":     date_ts,
        "bet_odds":        combined_odds,
        "edge":            float(pdata.get("edge", 0) or 0) / 100,  # stored as pct in pdata
        "ml_prob":         0.0,
        "fair_prob":       0.0,
        "stake_units":     stake_units,
        "kelly_stake_pct": round(stake_units * 100, 2),
        "is_parlay_leg":   False,
        "version_snapshot": pdata.get("version_snapshot", ""),
        "actual_result":   "won" if won else "lost",
        "won":             won,
        "profit_units":    profit,
        "closing_odds":    None,
        "clv":             None,
        "status":          "won" if won else "lost",
    }

    # Append to settled parquet — align dtypes to avoid schema conflicts
    new_row_df = pd.DataFrame([settled_row])
    if settled_path.exists() and settled_path.stat().st_size > 100:
        existing = pd.read_parquet(settled_path)
        # Cast timestamp columns to match existing schema
        for col in ("commence_time", "recorded_at"):
            if col in existing.columns and pd.api.types.is_datetime64_any_dtype(existing[col]):
                new_row_df[col] = pd.to_datetime(new_row_df[col], utc=True)
        settled_df = pd.concat([existing, new_row_df], ignore_index=True)
    else:
        settled_df = new_row_df
    settled_df.to_parquet(settled_path, index=False)

    # Update status in manual_parlays.json if applicable
    if source == "manual" and parlay_id:
        all_parlays = _load_manual_parlays()
        for p in all_parlays:
            if p.get("id") == parlay_id:
                p["status"] = "won" if won else "lost"
                p["settled_at"] = datetime.now(timezone.utc).isoformat()
                break
        _save_manual_parlays(all_parlays)

    return jsonify({
        "settled": True,
        "won":     won,
        "profit":  profit,
        "name":    name,
    })


@app.route("/api/results/clear", methods=["POST"])
def api_results_clear():
    """Archive and clear all predictions + settled bets."""
    tracker_dir = BASE / "data" / "tracker"
    archive_dir = tracker_dir / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    cleared = []
    for fname in ("predictions.parquet", "settled.parquet", "parlays.parquet"):
        src = tracker_dir / fname
        if src.exists() and src.stat().st_size > 100:
            dst = archive_dir / f"{src.stem}_{timestamp}{src.suffix}"
            src.rename(dst)
            cleared.append(fname)
    # Also clear my-selections
    my_sel = tracker_dir / "my_selections.json"
    if my_sel.exists():
        dst = archive_dir / f"my_selections_{timestamp}.json"
        my_sel.rename(dst)
        cleared.append("my_selections.json")
    return jsonify({"ok": True, "cleared": cleared, "archive_ts": timestamp})


@app.route("/api/results/settle-all", methods=["POST"])
def api_results_settle_all():
    """
    Automatically fetch real game results from The Odds API scores endpoint,
    compare against pending bets, and settle what's completed.
    Returns { settled, still_pending, total_profit, results, errors }
    """
    import uuid as _uuid, requests as _req, re as _re, unicodedata as _ud

    pred_path    = BASE / "data" / "tracker" / "predictions.parquet"
    settled_path = BASE / "data" / "tracker" / "settled.parquet"
    odds_key     = get_primary_odds_api_key()

    if not pred_path.exists():
        return jsonify({"error": "No predictions on record"}), 404

    pred = pd.read_parquet(pred_path)
    pending = pred[pred["status"] == "pending"].copy()
    if pending.empty:
        return jsonify({"settled": 0, "still_pending": 0, "total_profit": 0,
                        "results": [], "message": "No pending bets to settle."})

    existing_settled = (
        pd.read_parquet(settled_path)
        if settled_path.exists() and settled_path.stat().st_size > 100
        else pd.DataFrame()
    )

    # ── Sport key mapping ─────────────────────────────────────────────────────
    SPORT_KEYS = {
        "basketball": "basketball_nba",
        "nhl":        "icehockey_nhl",
        "mlb":        "baseball_mlb",
        "soccer":     None,   # handled via football-data.org below
        "tennis":     None,   # no scores API available
    }

    # ── Fetch scores for each sport that has pending bets ─────────────────────
    scores_by_sport = {}   # sport -> list of completed game dicts
    errors = []
    sports_needed = pending["sport"].unique().tolist()

    for sport in sports_needed:
        api_key = SPORT_KEYS.get(sport)
        if not api_key:
            continue   # tennis/soccer handled separately
        try:
            r = _req.get(
                f"https://api.the-odds-api.com/v4/sports/{api_key}/scores/",
                params={"apiKey": odds_key, "daysFrom": 3},
                timeout=10,
            )
            if r.status_code != 200:
                errors.append(f"{sport}: HTTP {r.status_code}")
                continue
            games = [g for g in r.json() if g.get("completed") and g.get("scores")]
            scores_by_sport[sport] = games
        except Exception as e:
            errors.append(f"{sport}: {e}")

    # ── Fetch soccer scores from football-data.org (single request for all leagues) ──
    if "soccer" in sports_needed:
        fd_key = os.getenv("FOOTBALL_DATA_API_KEY", "")
        if not fd_key:
            errors.append("soccer: FOOTBALL_DATA_API_KEY not set")
        else:
            from datetime import timedelta as _td
            date_from = (datetime.now(timezone.utc) - _td(days=4)).strftime("%Y-%m-%d")
            date_to   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            try:
                resp = _req.get(
                    "https://api.football-data.org/v4/matches",
                    headers={"X-Auth-Token": fd_key},
                    params={"status": "FINISHED", "dateFrom": date_from, "dateTo": date_to},
                    timeout=10,
                )
                if resp.status_code == 200:
                    soccer_games = []
                    for m in resp.json().get("matches", []):
                        ft = m.get("score", {}).get("fullTime", {})
                        home_score = ft.get("home")
                        away_score = ft.get("away")
                        if home_score is None or away_score is None:
                            continue
                        home_name = m["homeTeam"]["name"]
                        away_name = m["awayTeam"]["name"]
                        # Normalise into same shape as Odds API games
                        soccer_games.append({
                            "home_team": home_name,
                            "away_team": away_name,
                            "commence_time": m.get("utcDate", ""),
                            "completed": True,
                            "scores": [
                                {"name": home_name, "score": str(home_score)},
                                {"name": away_name, "score": str(away_score)},
                            ],
                        })
                    if soccer_games:
                        scores_by_sport["soccer"] = soccer_games
                elif resp.status_code == 429:
                    errors.append("soccer: football-data.org rate limit — try again in a minute")
                else:
                    errors.append(f"soccer: football-data.org HTTP {resp.status_code}")
            except Exception as e:
                errors.append(f"soccer: {e}")

    # ── Helper: normalise team name for fuzzy matching ────────────────────────
    _TEAM_STOPWORDS = {"fc", "cf", "sc", "ac", "ca", "cd", "club", "de", "the", "team"}

    def _norm(name: str) -> str:
        """Normalise to lowercase alphanum, strip accents, and remove noisy club tokens."""
        text = _ud.normalize("NFKD", str(name or ""))
        text = "".join(ch for ch in text if not _ud.combining(ch)).lower()
        text = _re.sub(r"[^a-z0-9]", " ", text)
        tokens = [tok for tok in text.split() if tok and tok not in _TEAM_STOPWORDS]
        return " ".join(tokens).strip()

    def _compact(name: str) -> str:
        return "".join(_norm(name).split())

    def _team_match(bet_team: str, api_team: str) -> bool:
        """
        Strict bidirectional match: both full normalised names must share
        a meaningful token. Avoids false positives like 'Wild' matching 'Wilder'.
        """
        nb = _norm(bet_team)
        ng = _norm(api_team)
        if nb == ng:
            return True
        if _compact(bet_team) and _compact(bet_team) == _compact(api_team):
            return True
        # Token overlap: every word in the shorter name must appear in the longer
        tokens_b = set(nb.split())
        tokens_g = set(ng.split())
        if not tokens_b or not tokens_g:
            return False
        shorter = tokens_b if len(tokens_b) <= len(tokens_g) else tokens_g
        longer  = tokens_b if len(tokens_b) >  len(tokens_g) else tokens_g
        # Require all tokens of the shorter name to appear in the longer
        return shorter.issubset(longer)

    unresolved_rows: list[dict[str, str]] = []

    def _mark_unresolved(row: dict | pd.Series, reason: str, message: str) -> None:
        unresolved_rows.append({
            "sport": str(row.get("sport", "")),
            "match": str(row.get("match_id", "")),
            "pick": str(row.get("team_or_player", "")),
            "reason": reason,
            "message": message,
        })

    def _game_date(game: dict) -> str | None:
        """Extract YYYY-MM-DD from game's commence_time if present."""
        ct = game.get("commence_time", "")
        if ct:
            return ct[:10]
        return None

    def _resolve(row: dict, game: dict) -> bool | None:
        """Return True=won, False=lost, None=can't determine."""
        scores = {s["name"]: int(s["score"]) for s in game.get("scores", [])}
        if len(scores) < 2:
            return None
        home, away = game["home_team"], game["away_team"]
        home_score = scores.get(home, 0)
        away_score = scores.get(away, 0)

        pick  = str(row.get("team_or_player", ""))
        mtype = _mtype(pick)

        if mtype == "moneyline":
            if pick.strip().lower() == "draw":
                return home_score == away_score
            # Find which team in the game matches the pick
            for team_name, score in scores.items():
                if _team_match(pick, team_name):
                    other = [s for n, s in scores.items() if n != team_name][0]
                    return score > other
            return None

        elif mtype == "totals":
            total = home_score + away_score
            m = _re.search(r"(Over|Under)\s+([\d.]+)", pick, _re.I)
            if not m:
                return None
            direction, line = m.group(1).lower(), float(m.group(2))
            if total == line:
                return None   # push
            return (direction == "over") == (total > line)

        elif mtype == "spreads":
            m = _re.search(r"([+-][\d.]+)\s*$", pick)
            if not m:
                return None
            spread = float(m.group(1))
            team_part = _re.sub(r"[+-][\d.]+\s*$", "", pick).strip()
            for team_name in scores:
                if _team_match(team_part, team_name):
                    my_score    = scores[team_name]
                    other_score = [s for n, s in scores.items() if n != team_name][0]
                    covered = (my_score + spread) > other_score
                    push    = (my_score + spread) == other_score
                    return None if push else covered
            return None

        return None

    # ── Walk pending bets and match to completed games ────────────────────────
    new_rows          = []
    pred_ids_settled  = []
    results           = []
    total_profit      = 0.0
    still_pending_cnt = 0

    for _, row in pending.iterrows():
        sport    = row.get("sport", "")
        games    = scores_by_sport.get(sport, [])
        match_id = str(row.get("match_id", ""))

        # Extract the bet's expected game date from commence_time
        bet_ct = row.get("commence_time")
        bet_date = None
        if bet_ct is not None:
            try:
                bet_date = str(pd.Timestamp(bet_ct).date())
            except Exception:
                pass

        if not games:
            still_pending_cnt += 1
            _mark_unresolved(row, "no_score_source", f"No finished score feed was available for {sport}.")
            continue

        # Parse "Home vs Away" into two team names
        parts = [p.strip() for p in match_id.replace(" @ ", " vs ").split(" vs ")]
        bet_home = parts[0] if parts else ""
        bet_away = parts[1] if len(parts) > 1 else ""

        matching_games = []
        for g in games:
            api_home, api_away = g["home_team"], g["away_team"]

            # Both teams must be present in the match (order may differ)
            home_ok = _team_match(bet_home, api_home) or _team_match(bet_home, api_away)
            away_ok = _team_match(bet_away, api_home) or _team_match(bet_away, api_away)
            if not (home_ok and away_ok):
                continue
            matching_games.append(g)

        if not matching_games:
            still_pending_cnt += 1
            _mark_unresolved(row, "team_mismatch", "No finished game matched the tracked teams in the score feeds.")
            continue

        if bet_date:
            dated_matches = [g for g in matching_games if _game_date(g) == bet_date]
            if not dated_matches:
                still_pending_cnt += 1
                _mark_unresolved(row, "date_mismatch", f"Matching teams were found, but not for tracked event date {bet_date}.")
                results.append({
                    "pick": str(row.get("team_or_player", "")),
                    "sport": sport,
                    "match": match_id,
                    "status": "date_mismatch",
                    "message": f"Matching teams were found, but not for event date {bet_date}. Keeping this bet pending.",
                })
                continue
            matching_games = dated_matches
        elif len(matching_games) > 1:
            still_pending_cnt += 1
            _mark_unresolved(row, "ambiguous_date", "Multiple finished games matched these teams and the tracker date could not disambiguate them.")
            results.append({
                "pick": str(row.get("team_or_player", "")),
                "sport": sport,
                "match": match_id,
                "status": "ambiguous_date",
                "message": "Multiple completed games matched these teams and no event timestamp was available. Keeping this bet pending.",
            })
            continue

        matched_game = sorted(matching_games, key=lambda g: str(g.get("commence_time") or ""))[-1]

        won = _resolve(row.to_dict(), matched_game)
        if won is None:
            still_pending_cnt += 1
            _mark_unresolved(row, "market_resolution", "A finished game was found, but the market outcome could not be resolved cleanly.")
            continue

        stake  = float(row.get("stake_units", 0))
        odds   = float(row.get("bet_odds", 1))
        profit = round((odds - 1) * stake if won else -stake, 4)
        total_profit += profit
        pred_ids_settled.append(str(row["pred_id"]))

        new_rows.append({
            "pred_id":        str(row["pred_id"]),
            "settled_at":     datetime.now(timezone.utc).isoformat(),
            "sport":          sport,
            "match_id":       match_id,
            "team_or_player": str(row.get("team_or_player", "")),
            "commence_time":  row.get("commence_time"),
            "recorded_at":    row.get("recorded_at"),
            "bet_odds":       odds,
            "edge":           float(row.get("edge", 0)),
            "ml_prob":        float(row.get("ml_prob", 0)),
            "fair_prob":      float(row.get("fair_prob", 0)),
            "stake_units":    stake,
            "kelly_stake_pct": float(row.get("kelly_stake_pct", 0)),
            "is_parlay_leg":  bool(row.get("is_parlay_leg", False)),
            "version_snapshot": row.get("version_snapshot", ""),
            "actual_result":  "won" if won else "lost",
            "won":            won,
            "profit_units":   profit,
            "closing_odds":   None,
            "clv":            None,
            "status":         "won" if won else "lost",
        })
        results.append({
            "pick":   str(row.get("team_or_player", "")),
            "sport":  sport,
            "won":    won,
            "profit": profit,
            "match":  match_id,
        })

    # ── Write to settled.parquet ──────────────────────────────────────────────
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        if not existing_settled.empty:
            for col in ("commence_time", "recorded_at"):
                if col in existing_settled.columns and pd.api.types.is_datetime64_any_dtype(existing_settled[col]):
                    new_df[col] = pd.to_datetime(new_df[col], utc=True)
            settled_df = pd.concat([existing_settled, new_df], ignore_index=True)
        else:
            settled_df = new_df
        settled_df.to_parquet(settled_path, index=False)

        # Remove from predictions
        pred_updated = pred[~pred["pred_id"].isin(pred_ids_settled)]
        pred_updated.to_parquet(pred_path, index=False)

    # ── Also settle manual parlays whose legs all have scores ─────────────────
    # (manual parlays are stored separately, not in predictions parquet)
    manual_parlays = _load_manual_parlays()
    parlay_settled_ids = []
    parlay_new_rows    = []

    for p in manual_parlays:
        if p.get("status") != "pending":
            continue
        legs = p.get("legs", [])
        if not legs:
            continue

        leg_results = []
        for leg in legs:
            leg_sport = leg.get("sport", "").lower()
            leg_team  = leg.get("team", "")
            leg_match = leg.get("match", "")
            leg_games = scores_by_sport.get(leg_sport, [])

            found = None
            for g in leg_games:
                home, away = g["home_team"], g["away_team"]
                if (home in leg_match or away in leg_match or
                        _team_match(leg_team, home) or _team_match(leg_team, away)):
                    fake_row = {
                        "team_or_player": leg_team,
                        "bet_odds":       leg.get("odds", 2),
                        "stake_units":    0,
                    }
                    outcome = _resolve(fake_row, g)
                    if outcome is not None:
                        found = outcome
                    break
            leg_results.append(found)

        # Only settle if ALL legs resolved
        if any(r is None for r in leg_results):
            continue

        parlay_won = all(leg_results)
        raw_kelly  = p.get("kelly_stake", 0)
        stake_units = float(raw_kelly or 0)
        bankroll    = float(os.getenv("INITIAL_BANKROLL", 1000))
        if stake_units > 10:
            stake_units = round(stake_units / bankroll, 4)
        combined_odds = float(p.get("combined_odds", 1))
        profit = round((combined_odds - 1) * stake_units if parlay_won else -stake_units, 4)
        total_profit += profit

        p["status"]     = "won" if parlay_won else "lost"
        p["settled_at"] = datetime.now(timezone.utc).isoformat()
        parlay_settled_ids.append(p["id"])
        date    = p.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        date_ts = pd.Timestamp(date + "T12:00:00", tz="UTC")
        parlay_new_rows.append({
            "pred_id":        str(_uuid.uuid4())[:8],
            "settled_at":     datetime.now(timezone.utc).isoformat(),
            "sport":          "parlay",
            "match_id":       p.get("name", "Parlay"),
            "team_or_player": p.get("name", "Parlay"),
            "commence_time":  date_ts,
            "recorded_at":    date_ts,
            "bet_odds":       combined_odds,
            "edge":           float(p.get("edge", 0)) / 100,
            "ml_prob":        0.0,
            "fair_prob":      0.0,
            "stake_units":    stake_units,
            "kelly_stake_pct": round(stake_units * 100, 2),
            "is_parlay_leg":  False,
            "actual_result":  "won" if parlay_won else "lost",
            "won":            parlay_won,
            "profit_units":   profit,
            "closing_odds":   None,
            "clv":            None,
            "status":         "won" if parlay_won else "lost",
        })
        results.append({
            "pick":   p.get("name", "Parlay"),
            "sport":  "parlay",
            "won":    parlay_won,
            "profit": profit,
            "match":  f"{len(legs)}-leg parlay",
        })

    if parlay_new_rows:
        _save_manual_parlays(manual_parlays)
        pnew = pd.DataFrame(parlay_new_rows)
        # reload latest settled (may have been updated above)
        if settled_path.exists() and settled_path.stat().st_size > 100:
            ex = pd.read_parquet(settled_path)
            for col in ("commence_time", "recorded_at"):
                if col in ex.columns and pd.api.types.is_datetime64_any_dtype(ex[col]):
                    pnew[col] = pd.to_datetime(pnew[col], utc=True)
            pd.concat([ex, pnew], ignore_index=True).to_parquet(settled_path, index=False)
        else:
            pnew.to_parquet(settled_path, index=False)

    total_settled = len(new_rows) + len(parlay_new_rows)
    # soccer is supported via football-data.org; tennis is the only truly unsupported sport
    unsupported   = [s for s in sports_needed if SPORT_KEYS.get(s) is None and s not in scores_by_sport]

    return jsonify({
        "settled":       total_settled,
        "still_pending": still_pending_cnt,
        "total_profit":  round(total_profit, 4),
        "results":       results,
        "errors":        errors,
        "unsupported":   unsupported,
        "message": (
            f"Settled {total_settled} bet{'s' if total_settled != 1 else ''}. "
            f"{still_pending_cnt} still in play or awaiting results."
            + (f" ({', '.join(unsupported)} results not available yet)" if unsupported else "")
        ),
    })


# ══════════════════════════════════════════════════════════════════════════════
# API — SCAN
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/scan/status")
def api_scan_status():
    global _scan_running, _scan_proc, _scan_log
    if _scan_running:
        proc = _scan_proc
        if proc is None:
            _scan_running = False
            if not _scan_log or not any("Finished" in line or "Stopped" in line or "ERROR" in line for line in _scan_log[-3:]):
                _scan_log.append("[scan] Scan state reset — worker process no longer exists.")
        else:
            code = proc.poll()
            if code is not None:
                _scan_running = False
                _scan_proc = None
                if not _scan_log or not any("Finished" in line or "Stopped" in line for line in _scan_log[-3:]):
                    if code == -15 or code == -9:
                        _scan_log.append("[scan] Stopped by user.")
                    else:
                        _scan_log.append(f"[scan] Finished (exit {code})")
    return jsonify({"running": _scan_running, "log": _scan_log[-100:]})


@app.route("/api/reasoning/status")
def api_reasoning_status():
    with _reasoning_progress_lock:
        payload = {
            "running": _reasoning_progress.get("running", False),
            "mode": _reasoning_progress.get("mode", ""),
            "candidate_id": _reasoning_progress.get("candidate_id", ""),
            "stage": _reasoning_progress.get("stage", ""),
            "log": list(_reasoning_progress.get("log") or [])[-12:],
            "updated_at": _reasoning_progress.get("updated_at", ""),
        }
    return jsonify(payload)


@app.route("/api/scan/start", methods=["POST"])
def api_scan_start():
    global _scan_running, _scan_log, _scan_proc

    if _scan_running:
        return jsonify({"error": "Scan already running"}), 409

    data    = request.get_json() or {}
    sport   = data.get("sport", "all")    # single sport string or "all"
    market  = data.get("market", "all")   # "moneyline" | "spreads" | "totals" | "all"
    retrain = data.get("retrain", False)
    offline_odds = data.get("offline_odds", False)
    force_fresh_odds = data.get("force_fresh_odds", False)
    lean_context = data.get("lean_context", False)
    context_referee = data.get("context_referee", False)
    full_soccer_scope = data.get("full_soccer_scope", False)
    focused_lanes = data.get("focused_lanes", sport == "all")

    def _run():
        global _scan_running, _scan_log, _scan_proc
        _scan_running = True
        _scan_log     = ["[scan] Starting…"]

        bankroll = data.get("bankroll", float(os.getenv("INITIAL_BANKROLL", 1000)))
        cmd = [sys.executable, str(BASE / "daily_scan.py"),
               "--record-bets",
               "--bankroll", str(bankroll),
               "--sport", sport or "all",
               "--market", market or "all"]
        if retrain:
            cmd += ["--retrain"]
        if offline_odds:
            cmd += ["--offline-odds"]
        if force_fresh_odds:
            cmd += ["--force-fresh-odds"]
        if lean_context:
            cmd += ["--lean-context"]
        if context_referee:
            cmd += ["--context-referee"]
        if full_soccer_scope:
            cmd += ["--full-soccer-scope"]
        if focused_lanes:
            cmd += ["--focused-lanes"]
        if data.get("notify", False):
            cmd += ["--notify"]

        try:
            proc = subprocess.Popen(
                cmd, cwd=str(BASE),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            _scan_proc = proc
            for line in proc.stdout:
                line = line.rstrip()
                _scan_log.append(line)
            proc.wait()
            if proc.returncode == -15 or proc.returncode == -9:
                _scan_log.append("[scan] Stopped by user.")
            else:
                _scan_log.append(f"[scan] Finished (exit {proc.returncode})")
        except Exception as e:
            _scan_log.append(f"[scan] ERROR: {e}")
        finally:
            _scan_running = False
            _scan_proc = None

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"started": True})


@app.route("/api/scan/stop", methods=["POST"])
def api_scan_stop():
    global _scan_proc, _scan_running, _scan_log
    if not _scan_running:
        return jsonify({"error": "No scan running"}), 409
    proc = _scan_proc
    if proc is not None:
        try:
            proc.terminate()   # SIGTERM — clean shutdown
            _scan_log.append("[scan] Stop requested — terminating…")
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"stopped": True})


@app.route("/api/scan/stream")
def api_scan_stream():
    """SSE endpoint — streams scan log lines to the browser in real time."""
    def _generate():
        import time
        idx = 0
        heartbeat_counter = 0
        while True:
            if idx < len(_scan_log):
                for line in _scan_log[idx:]:
                    yield f"data: {json.dumps(line)}\n\n"
                idx = len(_scan_log)
                heartbeat_counter = 0  # reset after real data
            if not _scan_running and idx >= len(_scan_log):
                yield "data: __DONE__\n\n"
                break
            time.sleep(0.4)
            heartbeat_counter += 1
            # Send SSE comment every ~5s to keep connection alive during long ops
            if heartbeat_counter >= 12:
                yield ": heartbeat\n\n"
                heartbeat_counter = 0

    return Response(stream_with_context(_generate()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


_CTRL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitize_payload(obj):
    """Recursively remove stray control characters from all string values."""
    if isinstance(obj, str):
        return _CTRL_CHAR_RE.sub(" ", obj)
    if isinstance(obj, dict):
        return {k: _sanitize_payload(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_payload(v) for v in obj]
    return obj


@app.route("/api/analyze-game", methods=["POST"])
def api_analyze_game():
    """Run deep manual analysis for a single matchup and bet."""
    data = request.get_json() or {}
    required = {
        "sport": (data.get("sport") or "").strip(),
        "home_team": (data.get("home_team") or "").strip(),
        "away_team": (data.get("away_team") or "").strip(),
        "bet": (data.get("bet") or "").strip(),
    }
    missing = [key for key, value in required.items() if not value]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    sport = required["sport"].lower()
    if sport not in {"soccer", "basketball", "mlb", "nhl"}:
        return jsonify({"error": f"Unsupported sport: {sport}"}), 400

    try:
        started = perf_counter()
        analyst = ManualGameAnalyst()
        report = analyst.analyze_game(
            sport=sport,
            home_team=required["home_team"],
            away_team=required["away_team"],
            bet=required["bet"],
            market=(data.get("market") or "h2h").strip() or "h2h",
            selection=(data.get("selection") or "").strip() or None,
            price=float(data["price"]) if data.get("price") not in (None, "") else None,
        )
        fresh_news_context = _attach_fresh_news_context(
            report,
            sport=sport,
            home_team=required["home_team"],
            away_team=required["away_team"],
            bet=required["bet"],
        )
        evidence_profile = _build_evidence_profile(report, fresh_news_context=fresh_news_context)
        _apply_evidence_gate(report, evidence_profile)
        elapsed_ms = int((perf_counter() - started) * 1000)
        report.data_points["analysis_mode"] = "full_live_manual_analysis"
        report.data_points["elapsed_ms"] = elapsed_ms
        payload = _sanitize_payload(json.loads(json.dumps(report.to_dict(), default=str)))
        manual_candidate = {
            "sport": sport,
            "market": (data.get("market") or "h2h").strip() or "h2h",
            "team": required["bet"],
            "home": required["home_team"],
            "away": required["away_team"],
            "league": data.get("league", ""),
            "availability_summary": "",
            "review_required": evidence_profile.get("quality") == "thin",
            "review_reason": "; ".join(evidence_profile.get("risk_flags", [])[:2]),
        }
        llm_reasoning, llm_error = _reasoning_layer(manual_candidate, payload)
        return jsonify({
            "ok": True,
            "report": payload,
            "fresh_news_context": fresh_news_context,
            "evidence_profile": evidence_profile,
            "analysis_mode": "full_live_manual_analysis",
            "elapsed_ms": elapsed_ms,
            "markdown": report.to_markdown(),
            "llm_reasoning": llm_reasoning,
            "llm_error": llm_error,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/reasoning/scan", methods=["POST"])
def api_reasoning_scan():
    """Run a guarded reasoning scan for one of today's model-approved value bets."""
    data = request.get_json() or {}
    candidate_id = (data.get("candidate_id") or "").strip()
    mode = str(data.get("mode") or "guarded").strip().lower()
    if not candidate_id:
        return jsonify({"error": "candidate_id required"}), 400

    candidates = _today_reasoning_bets()
    selected = next((bet for bet in candidates if _reasoning_candidate_id(bet) == candidate_id), None)
    if selected is None:
        return jsonify({"error": "Candidate not found in today's eligible value bets"}), 404

    market_map = {
        "moneyline": "h2h",
        "spreads": "spreads",
        "totals": "totals",
        "double_chance": "double_chance",
        "draw_no_bet": "draw_no_bet",
    }
    team_text = str(selected.get("team", "")).lower()
    selection = ""
    if selected.get("market") == "moneyline":
        if "draw" in team_text:
            selection = "draw"
        elif selected.get("team") == selected.get("home"):
            selection = "home"
        else:
            selection = "away"
    elif selected.get("market") == "totals":
        selection = "over" if "over" in team_text else "under" if "under" in team_text else ""
    elif selected.get("market") == "spreads":
        selection = "home" if str(selected.get("team", "")).startswith(str(selected.get("home", ""))) else "away"
    elif selected.get("market") == "double_chance":
        home_name = str(selected.get("home", ""))
        away_name = str(selected.get("away", ""))
        if home_name and home_name.lower() in team_text and "draw" in team_text:
            selection = "home_or_draw"
        elif away_name and away_name.lower() in team_text and "draw" in team_text:
            selection = "away_or_draw"
        else:
            selection = "home_or_away"
    elif selected.get("market") == "draw_no_bet":
        selection = "home" if str(selected.get("home", "")).lower() in team_text else "away"

    try:
        _reset_reasoning_progress()
        _set_reasoning_progress(
            "Preparing candidate and market inputs…",
            running=True,
            mode=mode,
            candidate_id=candidate_id,
        )
        started = perf_counter()
        chosen_market = market_map.get(selected.get("market"), "h2h")
        analysis_mode = "guarded_cached_review_plus_live_context"
        if mode == "full_live":
            analysis_mode = "full_live_candidate_verification"
            _set_reasoning_progress("Rebuilding the full live analyst report…")
            analyst = ManualGameAnalyst()
            report = analyst.analyze_game(
                sport=str(selected.get("sport", "")).lower(),
                home_team=str(selected.get("home", "")),
                away_team=str(selected.get("away", "")),
                bet=str(selected.get("team", "")),
                market=chosen_market,
                selection=selection or None,
                price=float(selected["odds"]) if selected.get("odds") is not None else None,
                fair_prob=float(selected["fair_prob"]) if selected.get("fair_prob") is not None else None,
            )
        else:
            _set_reasoning_progress("Reviewing the cached board candidate…")
            report = _candidate_reasoning_report(
                selected,
                market=chosen_market,
                selection=selection,
            )
        _set_reasoning_progress("Collecting fresh web context and source checks…")
        fresh_news_context = _attach_fresh_news_context(
            report,
            sport=str(selected.get("sport", "")).lower(),
            home_team=str(selected.get("home", "")),
            away_team=str(selected.get("away", "")),
            bet=str(selected.get("team", "")),
        )
        if not hasattr(report, "warnings") or getattr(report, "warnings") is None:
            report.warnings = []
        for summary in _top_context_summaries(selected.get("context_adjustments") or []):
            note = f"System context: {summary}"
            if note not in report.warnings:
                report.warnings.append(note)
        for highlight in _top_scraped_context_highlights(selected.get("scraped_context_highlights") or []):
            note = f"System scraper: {highlight}"
            if note not in report.warnings:
                report.warnings.append(note)
        if selected.get("availability_summary"):
            avail_note = f"System availability: {selected.get('availability_summary')}"
            if avail_note not in report.warnings:
                report.warnings.append(avail_note)
        _set_reasoning_progress("Scoring evidence quality and timing gates…")
        evidence_profile = _build_evidence_profile(report, fresh_news_context=fresh_news_context, candidate=selected)
        _apply_evidence_gate(report, evidence_profile)
        elapsed_ms = int((perf_counter() - started) * 1000)
        report.data_points["analysis_mode"] = analysis_mode
        report.data_points["elapsed_ms"] = elapsed_ms
        payload = _sanitize_payload(json.loads(json.dumps(report.to_dict(), default=str)))
        payload["warnings"] = list(getattr(report, "warnings", []) or [])
        _set_reasoning_progress("Applying the context-only referee…")
        llm_reasoning, llm_error = _reasoning_layer(selected, payload)
        referee_system_decision = ""
        if llm_reasoning and isinstance(llm_reasoning.get("content"), dict):
            referee_decision = str(llm_reasoning["content"].get("decision", "")).upper()
            referee_reason = str(llm_reasoning["content"].get("reasoning", "")).strip()
            referee_system_decision = _map_referee_decision_to_system(referee_decision, referee_reason)
            if referee_decision:
                note = f"Context referee: {referee_decision}" + (f" — {referee_reason}" if referee_reason else "")
                if note not in report.warnings:
                    report.warnings.append(note)
                payload = _sanitize_payload(json.loads(json.dumps(report.to_dict(), default=str)))
                payload["warnings"] = list(getattr(report, "warnings", []) or [])
        _set_reasoning_progress("Finalizing the reasoning report…")
        return jsonify({
            "ok": True,
            "candidate": {
                "id": candidate_id,
                "display": _reasoning_display_label(selected),
                "selection": selected.get("team"),
                "home_team": selected.get("home"),
                "away_team": selected.get("away"),
                "market": selected.get("market"),
                "league": selected.get("league"),
                "league_key": selected.get("league_key", ""),
                "launch_label": selected.get("launch_label", ""),
                "launch_note": selected.get("launch_note", ""),
                "odds": selected.get("odds"),
                "minimum_acceptable_odds": selected.get("minimum_acceptable_odds"),
                "odds_recheck_status": selected.get("odds_recheck_status"),
                "odds_recheck_delta": selected.get("odds_recheck_delta"),
                "edge": selected.get("edge"),
                "fair_prob": selected.get("fair_prob"),
                "market_implied_prob": selected.get("market_implied_prob"),
                "fair_odds": selected.get("fair_odds"),
                "availability_summary": selected.get("availability_summary", ""),
                "context_adjustments": selected.get("context_adjustments") or [],
                "prediction_factors": selected.get("prediction_factors") or [],
                "true_probability": selected.get("true_probability") or {},
                "scraped_context": selected.get("scraped_context") or {},
                "scraped_context_highlights": selected.get("scraped_context_highlights") or [],
                "scraped_context_sources": selected.get("scraped_context_sources") or [],
                "fresh_news_context": fresh_news_context,
                "decision_status": selected.get("decision_status", ""),
                "decision_reason": selected.get("decision_reason", ""),
                "suppression_reason": selected.get("suppression_reason", ""),
                "review_required": bool(selected.get("review_required")),
                "review_reason": selected.get("review_reason", ""),
                "market_policy_label": selected.get("market_policy_label", ""),
                "market_policy_reason": selected.get("market_policy_reason", ""),
            },
            "report": payload,
            "fresh_news_context": fresh_news_context,
            "evidence_profile": evidence_profile,
            "analysis_mode": analysis_mode,
            "elapsed_ms": elapsed_ms,
            "markdown": report.to_markdown(),
            "llm_reasoning": llm_reasoning,
            "referee_system_decision": referee_system_decision,
            "llm_error": llm_error,
        })
    except Exception as exc:
        _set_reasoning_progress(f"Reasoning scan failed: {exc}", running=False)
        return jsonify({"error": str(exc)}), 500
    finally:
        if _reasoning_progress.get("running"):
            _set_reasoning_progress("Reasoning scan finished.", running=False)


# ── Closing odds fetch ────────────────────────────────────────────────────────

_closing_running = False
_closing_log: list = []

@app.route("/api/closing-odds/status")
def api_closing_status():
    return jsonify({"running": _closing_running, "log": _closing_log[-50:]})


@app.route("/api/closing-odds/fetch", methods=["POST"])
def api_closing_fetch():
    global _closing_running, _closing_log

    if _closing_running:
        return jsonify({"error": "Already running"}), 409

    data = request.get_json() or {}
    sport = data.get("sport")   # optional — omit to fetch all pending sports

    def _run():
        global _closing_running, _closing_log
        _closing_running = True
        _closing_log = ["[closing-odds] Starting…"]

        cmd = [sys.executable, str(BASE / "fetch_closing_odds.py")]
        if sport:
            cmd += ["--sport", sport]

        try:
            proc = subprocess.Popen(
                cmd, cwd=str(BASE),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in proc.stdout:
                _closing_log.append(line.rstrip())
            proc.wait()
            _closing_log.append(f"[closing-odds] Finished (exit {proc.returncode})")
        except Exception as e:
            _closing_log.append(f"[closing-odds] ERROR: {e}")
        finally:
            _closing_running = False

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"started": True})


# ── Auto-settle ───────────────────────────────────────────────────────────────

_settle_running = False
_settle_log: list = []

@app.route("/api/settle/status")
def api_settle_status():
    return jsonify({"running": _settle_running, "log": _settle_log[-50:]})


@app.route("/api/settle/run", methods=["POST"])
def api_settle_run():
    global _settle_running, _settle_log

    if _settle_running:
        return jsonify({"error": "Already running"}), 409

    data   = request.get_json() or {}
    sport  = data.get("sport")
    date   = data.get("date")

    def _run():
        global _settle_running, _settle_log
        _settle_running = True
        _settle_log = ["[settle] Starting…"]

        cmd = [sys.executable, str(BASE / "settle.py")]
        if sport:
            cmd += ["--sport", sport]
        if date:
            cmd += ["--date", date]

        try:
            proc = subprocess.Popen(
                cmd, cwd=str(BASE),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in proc.stdout:
                _settle_log.append(line.rstrip())
            proc.wait()
            _settle_log.append(f"[settle] Finished (exit {proc.returncode})")
        except Exception as e:
            _settle_log.append(f"[settle] ERROR: {e}")
        finally:
            _settle_running = False

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"started": True})


# ══════════════════════════════════════════════════════════════════════════════
# API — API MANAGER
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/apis/status")
def api_apis_status():
    """Check health of all external APIs (reuses api_status.py logic)."""
    sys.path.insert(0, str(BASE))
    from api_status import (check_odds_api, check_football_data,
                             check_balldontlie, check_mlb_api,
                             check_nhl_api, check_telegram)
    results = []
    for fn in [check_odds_api, check_football_data, check_balldontlie,
               check_mlb_api, check_nhl_api, check_telegram]:
        try:
            results.append(fn())
        except Exception as e:
            results.append({"name": fn.__name__, "status": "error", "detail": str(e)})

    # LLM / Reasoning provider status
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if anthropic_key:
        claude_model = os.getenv("CLAUDE_REASONING_MODEL", "claude-haiku-4-5-20251001")
        results.append({
            "name": "Claude (Reasoning)",
            "status": "ok",
            "detail": f"Active · {claude_model}",
        })
    elif openrouter_key:
        or_model = os.getenv("OPENROUTER_REASONING_MODEL", "openrouter/free")
        results.append({
            "name": "OpenRouter (Reasoning)",
            "status": "warn",
            "detail": f"Active · {or_model} — set ANTHROPIC_API_KEY for better quality",
        })
    else:
        results.append({
            "name": "Reasoning LLM",
            "status": "error",
            "detail": "No key set — add ANTHROPIC_API_KEY or OPENROUTER_API_KEY to .env",
        })

    odds_key_pool = _odds_key_pool_summary()
    active_fp = str(odds_key_pool.get("active_fingerprint") or "")
    active_remaining = odds_key_pool.get("active_remaining")
    runtime_count = int(odds_key_pool.get("runtime_loaded_count") or 0)
    usable_count = int(odds_key_pool.get("usable_count") or 0)

    for item in results:
        if str(item.get("name") or "") != "The Odds API":
            continue
        if active_fp:
            item["detail"] = (
                f"Active runtime key …{active_fp} · {active_remaining} remaining"
                if isinstance(active_remaining, int)
                else f"Active runtime key …{active_fp} · remaining unknown"
            )
            if isinstance(active_remaining, int):
                item["remaining"] = active_remaining
                item["total"] = max(active_remaining, active_remaining)
                item["used"] = None
                if active_remaining <= 10:
                    item["status"] = "critical"
                elif active_remaining <= 100:
                    item["status"] = "warn"
                else:
                    item["status"] = "ok"
        elif runtime_count > 0:
            item["detail"] = f"{runtime_count} runtime keys loaded · {usable_count} usable"
            item["used"] = None
            item["remaining"] = None
            item["total"] = None
            item["status"] = "warn" if usable_count <= 0 else "ok"
        break

    return jsonify({"apis": results, "odds_key_pool": odds_key_pool})


@app.route("/api/apis/update", methods=["POST"])
def api_apis_update():
    """Update a key in .env."""
    data  = request.get_json()
    var   = data.get("var", "").strip()
    value = data.get("value", "").strip()

    ALLOWED = {"ODDS_API_KEY", "ODDS_API_KEYS", "FOOTBALL_DATA_API_KEY", "BALLDONTLIE_API_KEY",
               "API_SPORTS_KEY", "RAPIDAPI_KEY", "OPENWEATHER_API_KEY",
               "NEWS_API_KEY",
               "TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID", "INITIAL_BANKROLL",
               "ANTHROPIC_API_KEY", "CLAUDE_REASONING_MODEL",
               "OPENROUTER_API_KEY", "OPENROUTER_REASONING_MODEL"}
    if var not in ALLOWED:
        return jsonify({"error": f"Variable {var!r} not allowed"}), 400
    if not value:
        return jsonify({"error": "Empty value"}), 400

    env_path = BASE / ".env"
    set_key(str(env_path), var, value)
    load_dotenv(env_path, override=True)
    return jsonify({"ok": True, "var": var})


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _parse_parlays_from_md(path: Path):
    """Parse parlay blocks from the markdown report."""
    text = path.read_text()
    parlays = []
    blocks = text.split("**Parlay ")
    for block in blocks[1:]:
        try:
            lines = block.strip().splitlines()
            _ = lines[0]  # header line, used only to confirm block has content

            # Extract combined odds
            co_part = [l for l in lines if "combined odds:" in l]
            combined_odds = float(co_part[0].split("combined odds:")[1].split(")")[0].strip()) if co_part else 0

            # Win prob, EV, Kelly
            wp_line = next((l for l in lines if "Win probability:" in l), "")
            ev_line = next((l for l in lines if "Expected value:" in l), "")
            ks_line = next((l for l in lines if "Kelly stake:" in l), "")
            # bracket is determined later from position in text

            legs = []
            for l in lines:
                if l.strip().startswith("- ["):
                    # e.g. "  - [MLB] **Cincinnati Reds** vs Minnesota Twins @ 2.25 (ML: 47.5%  edge: +3.0%)"
                    sport_part = l.split("]")[0].replace("- [", "").strip()
                    rest = l.split("]")[1].strip() if "]" in l else l
                    team = rest.split("**")[1] if "**" in rest else rest
                    odds_part = rest.split("@")[1].split("(")[0].strip() if "@" in rest else "0"
                    ml_part   = rest.split("ML:")[1].split("%")[0].strip() if "ML:" in rest else "50"
                    edge_part = rest.split("edge:")[1].split("%")[0].strip().replace("+","") if "edge:" in rest else "0"
                    legs.append({
                        "sport":   sport_part,
                        "team":    team,
                        "odds":    float(odds_part) if odds_part else 0,
                        "ml_prob": float(ml_part) / 100 if ml_part else 0.5,
                        "edge":    float(edge_part) if edge_part else 0,
                    })

            if not legs:
                continue

            # Determine bracket from surrounding text position
            pos = text.find("**Parlay " + block[:20])
            bracket = "10x"
            for br in ["5x", "10x", "20x"]:
                if f"**{br} target" in text[:pos]:
                    bracket = br

            # Value vs longshot/speculative
            ptype = "speculative"
            if "🎯 Value Parlays" in text:
                spec_section_start = max(
                    text.find("⚡ Speculative Parlays"),
                    text.find("⚡ Longshot Parlays"),
                )
                if spec_section_start < 0 or pos < spec_section_start:
                    ptype = "value"

            parlays.append({
                "type":          ptype,
                "bracket":       bracket,
                "combined_odds": combined_odds,
                "win_prob":      float(wp_line.split(":")[1].replace("%","").strip()) if wp_line else 0,
                "ev":            float(ev_line.split(":")[1].split("x")[0].strip()) if ev_line else 0,
                "kelly_stake":   ks_line.split("=")[1].strip() if "=" in ks_line else "",
                "legs":          legs,
            })
        except Exception:
            continue

    return parlays


# ══════════════════════════════════════════════════════════════════════════════
# API — HYBRID QUOTA TRACKING
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/quota/status")
def api_quota_status():
    """Return full quota status across all sources (Betfair, Odds API, API-Football)."""
    if not quota_bridge:
        return jsonify({"error": "Quota bridge not available"}), 500
    try:
        return jsonify(quota_bridge.get_quota_status())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/quota/simple")
def api_quota_simple():
    """Return simplified quota format (backward compatible with api_usage.json)."""
    if not quota_bridge:
        return jsonify({"error": "Quota bridge not available"}), 500
    try:
        return jsonify(quota_bridge.get_simple_quota_for_webapp())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/quota/warning")
def api_quota_warning():
    """Return warning level for UI color coding (green/yellow/red)."""
    if not quota_bridge:
        return jsonify({"level": "unknown"}), 500
    try:
        return jsonify({"level": quota_bridge.get_warning_level()})
    except Exception:
        return jsonify({"level": "unknown"}), 500


@app.route("/api/sources/status")
def api_sources_status():
    """Return configured and missing sports data providers."""
    try:
        return jsonify(source_status_summary())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/quota/sync-legacy")
def api_quota_sync_legacy():
    """Sync hybrid quota data to legacy api_usage.json for backward compatibility."""
    if not quota_bridge:
        return jsonify({"error": "Quota bridge not available"}), 500
    try:
        quota_bridge.save_legacy_api_usage()
        return jsonify({"status": "synced"})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# Live betting routes
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/live")
def live_page():
    return render_template("live.html")


@app.route("/api/live/edges")
def api_live_edges():
    """Run live tennis scan and return edges + match states as JSON."""
    try:
        from src.live.scanner import scan
        force = request.args.get("force", "false").lower() == "true"
        result = scan(force=force)
        return jsonify(result)
    except Exception as exc:
        logger.error("Live scan error: %s", exc)
        return jsonify({"edges": [], "matches": [], "scanned": 0, "error": str(exc)}), 500


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    print("\n  🏆 Sports Predictor Web App")
    print(f"  Open: http://localhost:{port}\n")
    app.run(debug=True, port=port, threaded=True)
