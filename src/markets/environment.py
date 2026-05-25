from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import requests

from src.features.travel_features import _lookup


def _coerce_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _weather_risk(temperature_f: float, wind_mph: float, precip_mm: float) -> int:
    if wind_mph >= 14 or precip_mm >= 0.75:
        return 1
    if temperature_f and (temperature_f <= 42 or temperature_f >= 88):
        return 1
    return 0


def build_environment_context(
    sport: str,
    home_team: str,
    away_team: str,
    commence: Any,
) -> dict[str, Any]:
    sport = (sport or "").lower()
    result: dict[str, Any] = {"home_team_name": home_team, "away_team_name": away_team}
    result["environment_fetched_at"] = datetime.now(timezone.utc).isoformat()
    if os.environ.get("SCAN_LEAN_CONTEXT", "").strip().lower() in {"1", "true", "yes", "on"}:
        return result

    # Travel burden is already encoded in feature snapshots; this module only adds
    # optional live outdoor weather context when we have enough location data.
    if sport not in {"mlb", "soccer"}:
        return result

    api_key = os.environ.get("OPENWEATHER_API_KEY", "").strip()
    if not api_key:
        return result

    match_dt = _coerce_datetime(commence)
    if match_dt is None:
        return result
    lead_hours = (match_dt - datetime.now(timezone.utc)).total_seconds() / 3600.0
    if lead_hours < -6 or lead_hours > 120:
        return result

    home_lookup = _lookup(home_team, sport)
    if not home_lookup:
        return result
    _, lat, lon = home_lookup
    session = requests.Session()
    session.headers.update({"User-Agent": "sports-predictor/1.0"})

    payload: dict[str, Any] | None = None
    try:
        if lead_hours <= 6:
            resp = session.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={"lat": lat, "lon": lon, "appid": api_key, "units": "imperial"},
                timeout=8,
            )
            resp.raise_for_status()
            payload = resp.json()
        else:
            resp = session.get(
                "https://api.openweathermap.org/data/2.5/forecast",
                params={"lat": lat, "lon": lon, "appid": api_key, "units": "imperial"},
                timeout=8,
            )
            resp.raise_for_status()
            forecast = resp.json()
            entries = forecast.get("list") or []
            if entries:
                target_ts = match_dt.timestamp()
                payload = min(
                    entries,
                    key=lambda item: abs(float(item.get("dt", target_ts)) - target_ts),
                )
    except Exception:
        return result

    if not payload:
        return result

    main = payload.get("main") or {}
    wind = payload.get("wind") or {}
    rain = payload.get("rain") or {}
    snow = payload.get("snow") or {}
    temperature_f = float(main.get("temp", 0.0) or 0.0)
    wind_mph = float(wind.get("speed", 0.0) or 0.0)
    precip_mm = float(rain.get("3h", rain.get("1h", 0.0)) or 0.0) + float(
        snow.get("3h", snow.get("1h", 0.0)) or 0.0
    )

    result.update(
        {
            "outdoor_weather_source": "openweather",
            "temperature_f": round(temperature_f, 1),
            "wind_mph": round(wind_mph, 1),
            "precip_mm": round(precip_mm, 2),
            "weather_risk": _weather_risk(temperature_f, wind_mph, precip_mm),
        }
    )
    return result
