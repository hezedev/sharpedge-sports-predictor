from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HEALTH_PATH = Path("data/cache/provider_health.json")
_DEFAULT_MIN_REMAINING = 10


def _min_remaining() -> int:
    try:
        return max(0, int(os.environ.get("SOURCE_MIN_REMAINING", _DEFAULT_MIN_REMAINING)))
    except Exception:
        return _DEFAULT_MIN_REMAINING


def _load() -> dict[str, Any]:
    try:
        return json.loads(_HEALTH_PATH.read_text())
    except Exception:
        return {}


def _save(payload: dict[str, Any]) -> None:
    _HEALTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    _HEALTH_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _remaining_from_headers(headers: Any) -> int | None:
    for name in (
        "x-ratelimit-requests-remaining",
        "x-ratelimit-remaining",
        "X-RateLimit-Remaining",
        "x-requests-remaining",
    ):
        value = headers.get(name) if hasattr(headers, "get") else None
        try:
            if value is not None and str(value).strip():
                return int(float(str(value).strip()))
        except Exception:
            continue
    return None


def record_provider_response(provider: str, response: Any) -> None:
    remaining = _remaining_from_headers(getattr(response, "headers", {}))
    status_code = getattr(response, "status_code", None)
    payload = _load()
    entry = dict(payload.get(provider) or {})
    if remaining is not None:
        entry["remaining"] = remaining
    if status_code is not None:
        entry["last_status"] = int(status_code)
    entry["updated_at"] = datetime.now(timezone.utc).isoformat()
    payload[provider] = entry
    _save(payload)


def provider_quota_low(provider: str, *, min_remaining: int | None = None) -> bool:
    entry = _load().get(provider) or {}
    remaining = entry.get("remaining")
    try:
        remaining_int = int(remaining)
    except Exception:
        return False
    threshold = _min_remaining() if min_remaining is None else int(min_remaining)
    return remaining_int <= threshold


def provider_health_snapshot() -> dict[str, Any]:
    payload = _load()
    return {
        "providers": payload,
        "min_remaining": _min_remaining(),
        "low_quota_providers": [
            provider for provider in sorted(payload) if provider_quota_low(provider)
        ],
    }
