"""Shared helpers for tracking and budgeting The Odds API quota."""

from __future__ import annotations

import calendar
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


USAGE_FILE = Path("data/api_usage.json")
_ODDS_API_KEY_PATTERN = re.compile(r"^[A-Za-z0-9._-]{8,}$")


def normalize_odds_api_key(value: object) -> str:
    """Normalize an Odds API key value without exposing the raw token."""
    text = str(value or "").strip()
    while len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    return text


def is_valid_odds_api_key(value: object) -> bool:
    """Return whether the normalized key looks usable for runtime loading."""
    key = normalize_odds_api_key(value)
    return bool(key) and _ODDS_API_KEY_PATTERN.fullmatch(key) is not None


def api_key_fingerprint(api_key: str) -> str:
    """Return a short fingerprint of an API key for change detection."""
    key = normalize_odds_api_key(api_key)
    return key[-8:] if len(key) >= 8 else key


def parse_odds_api_keys_from_env(env: Mapping[str, object] | None = None) -> Dict[str, Any]:
    """Parse runtime Odds API keys from env with validation and diagnostics."""
    env_map = env if env is not None else os.environ
    runtime_keys: list[str] = []
    runtime_fingerprints: list[str] = []
    excluded: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    def _append_candidate(
        raw_value: object,
        *,
        source: str,
        index: int | None = None,
        report_empty: bool = True,
    ) -> None:
        normalized = normalize_odds_api_key(raw_value)
        fingerprint = api_key_fingerprint(normalized)
        detail: dict[str, Any] = {"source": source, "reason": "", "fingerprint": fingerprint}
        if index is not None:
            detail["index"] = int(index)
        if not normalized:
            if not report_empty:
                return
            detail["reason"] = "empty"
            excluded.append(detail)
            return
        if not is_valid_odds_api_key(normalized):
            detail["reason"] = "invalid_format"
            excluded.append(detail)
            return
        if normalized in seen_keys:
            detail["reason"] = "duplicate"
            excluded.append(detail)
            return
        seen_keys.add(normalized)
        runtime_keys.append(normalized)
        runtime_fingerprints.append(api_key_fingerprint(normalized))

    if "ODDS_API_KEY" in env_map:
        _append_candidate(env_map.get("ODDS_API_KEY", ""), source="ODDS_API_KEY", report_empty=False)

    if "ODDS_API_KEYS" in env_map:
        raw_pool = normalize_odds_api_key(env_map.get("ODDS_API_KEYS", ""))
        for index, chunk in enumerate(raw_pool.replace("\n", ",").split(",")):
            _append_candidate(chunk, source="ODDS_API_KEYS", index=index)

    return {
        "primary_key": runtime_keys[0] if runtime_keys else "",
        "keys": runtime_keys,
        "fingerprints": runtime_fingerprints,
        "excluded": excluded,
    }


def get_primary_odds_api_key(env: Mapping[str, object] | None = None) -> str:
    """Return the first valid runtime Odds API key after normalization."""
    return str(parse_odds_api_keys_from_env(env).get("primary_key") or "")


def load_odds_api_usage(today: Optional[str] = None) -> Dict[str, Any]:
    """Load persisted quota state."""
    today = today or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if USAGE_FILE.exists():
        try:
            data = json.loads(USAGE_FILE.read_text())
            if data.get("date") != today:
                data["date"] = today
                data["odds_requests_used_today"] = 0
            return data
        except Exception:
            pass
    return {
        "date": today,
        "odds_requests_used_today": 0,
        "odds_requests_used_total": 0,
        "odds_remaining": 9999,
    }


def save_odds_api_usage(
    *,
    api_key: str,
    remaining: Optional[int] = None,
    used_total: Optional[int] = None,
) -> Dict[str, Any]:
    """Persist quota state from live API headers."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    usage = load_odds_api_usage(today=today)
    fp = api_key_fingerprint(api_key)

    if usage.get("key_fingerprint") != fp:
        usage = {
            "date": today,
            "key_fingerprint": fp,
            "odds_requests_used_today": 0,
        }

    prev_remaining = usage.get("odds_remaining")
    usage["key_fingerprint"] = fp
    usage["date"] = today

    if remaining is not None:
        usage["odds_remaining"] = remaining
        usage["odds_remaining_start"] = usage.get("odds_remaining_start", remaining)
        if (
            isinstance(prev_remaining, int)
            and prev_remaining != 9999
            and prev_remaining >= remaining
        ):
            usage["odds_requests_used_today"] = usage.get("odds_requests_used_today", 0) + (
                prev_remaining - remaining
            )

    if used_total is not None:
        usage["odds_requests_used_total"] = used_total
        if remaining is not None:
            usage["odds_remaining_start"] = remaining + used_total
    elif remaining is not None and isinstance(usage.get("odds_remaining_start"), int):
        usage["odds_requests_used_total"] = usage["odds_remaining_start"] - remaining

    USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    USAGE_FILE.write_text(json.dumps(usage, indent=2))
    return usage


def get_odds_budget_status(
    api_key: str,
    *,
    monthly_limit: int = 500,
    reserve: int = 30,
) -> Dict[str, Any]:
    """Return live runtime quota status for the active Odds API key.

    We no longer impose an internal monthly-budget or daily-allowance policy
    layer here. The scanner should use whatever valid runtime keys are loaded,
    rotate when a key is exhausted, and reflect the provider's real remaining
    count rather than an artificial 500/month pacing model.
    """
    now = datetime.now(timezone.utc)
    usage = load_odds_api_usage(today=now.strftime("%Y-%m-%d"))
    fp = api_key_fingerprint(api_key)

    if usage.get("key_fingerprint") not in (None, "", fp):
        usage["odds_remaining"] = 9999
        usage["odds_requests_used_today"] = 0

    remaining = usage.get("odds_remaining", 9999)
    used_today = usage.get("odds_requests_used_today", 0)

    if not isinstance(remaining, int) or remaining == 9999:
        return {
            "remaining": remaining,
            "used_today": used_today,
            "days_left_in_cycle": None,
            "daily_allowance": None,
            "remaining_after_reserve": None,
        }

    return {
        "remaining": remaining,
        "used_today": used_today,
        "days_left_in_cycle": None,
        "daily_allowance": None,
        "remaining_after_reserve": remaining,
    }
