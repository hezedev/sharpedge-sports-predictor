from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import requests

from src.features.travel_features import _lookup

_MLB_PARK_FACTORS: dict[str, float] = {
    "Arizona Diamondbacks": 1.02,
    "Atlanta Braves": 1.01,
    "Baltimore Orioles": 0.98,
    "Boston Red Sox": 1.05,
    "Chicago Cubs": 1.03,
    "Chicago White Sox": 1.01,
    "Cincinnati Reds": 1.08,
    "Cleveland Guardians": 0.98,
    "Colorado Rockies": 1.15,
    "Detroit Tigers": 0.97,
    "Houston Astros": 1.00,
    "Kansas City Royals": 1.01,
    "Los Angeles Angels": 0.99,
    "Los Angeles Dodgers": 1.00,
    "Miami Marlins": 0.96,
    "Milwaukee Brewers": 1.00,
    "Minnesota Twins": 0.99,
    "New York Mets": 0.97,
    "New York Yankees": 1.03,
    "Oakland Athletics": 0.96,
    "Athletics": 0.96,
    "Philadelphia Phillies": 1.02,
    "Pittsburgh Pirates": 0.97,
    "San Diego Padres": 0.96,
    "San Francisco Giants": 0.94,
    "Seattle Mariners": 0.96,
    "St. Louis Cardinals": 0.99,
    "Tampa Bay Rays": 0.99,
    "Texas Rangers": 1.01,
    "Toronto Blue Jays": 1.02,
    "Washington Nationals": 1.00,
}


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


def _open_meteo_forecast(lat: float, lon: float, match_dt: datetime) -> dict[str, Any]:
    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,precipitation,wind_speed_10m",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "mm",
            "timezone": "UTC",
            "forecast_days": 7,
        },
        timeout=8,
    )
    resp.raise_for_status()
    payload = resp.json()
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return {}
    target = match_dt.replace(minute=0, second=0, microsecond=0)
    parsed_times: list[datetime] = []
    for raw in times:
        try:
            parsed_times.append(datetime.fromisoformat(str(raw)).replace(tzinfo=timezone.utc))
        except Exception:
            parsed_times.append(target)
    idx = min(range(len(parsed_times)), key=lambda i: abs((parsed_times[i] - target).total_seconds()))

    def _at(key: str, default: float = 0.0) -> float:
        values = hourly.get(key) or []
        try:
            return float(values[idx])
        except Exception:
            return default

    return {
        "source": "open_meteo",
        "temperature_f": _at("temperature_2m"),
        "wind_mph": _at("wind_speed_10m"),
        "precip_mm": _at("precipitation"),
    }


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

    if sport == "mlb":
        park_factor = _MLB_PARK_FACTORS.get(home_team)
        if park_factor is not None:
            result.update(
                {
                    "park_factor_source": "static_mlb_park_proxy",
                    "park_factor_proxy": round(float(park_factor), 3),
                    "park_run_environment": (
                        "hitter_friendly"
                        if park_factor >= 1.04
                        else "pitcher_friendly"
                        if park_factor <= 0.97
                        else "neutral"
                    ),
                }
            )

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
    source = "openweather"
    api_key = os.environ.get("OPENWEATHER_API_KEY", "").strip()
    try:
        if api_key:
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
        else:
            payload = _open_meteo_forecast(lat, lon, match_dt)
            source = "open_meteo"
    except Exception:
        return result

    if not payload:
        return result

    if source == "open_meteo":
        temperature_f = float(payload.get("temperature_f", 0.0) or 0.0)
        wind_mph = float(payload.get("wind_mph", 0.0) or 0.0)
        precip_mm = float(payload.get("precip_mm", 0.0) or 0.0)
    else:
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
            "outdoor_weather_source": source,
            "temperature_f": round(temperature_f, 1),
            "wind_mph": round(wind_mph, 1),
            "precip_mm": round(precip_mm, 2),
            "weather_risk": _weather_risk(temperature_f, wind_mph, precip_mm),
        }
    )
    return result
