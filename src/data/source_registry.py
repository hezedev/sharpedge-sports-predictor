from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class SourceProvider:
    key: str
    name: str
    category: str
    sports: tuple[str, ...]
    evidence: tuple[str, ...]
    env_vars: tuple[str, ...] = ()
    reliability: str = "fallback"
    cost_tier: str = "free"
    priority: int = 50
    notes: str = ""
    configured: bool = False
    missing_env_vars: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return asdict(self)


_PROVIDER_DEFINITIONS: tuple[SourceProvider, ...] = (
    SourceProvider(
        key="odds_api",
        name="The Odds API",
        category="odds",
        sports=("soccer", "basketball", "nhl", "mlb", "tennis"),
        evidence=("current_odds", "bookmaker_last_update", "spreads", "totals"),
        env_vars=("ODDS_API_KEY", "ODDS_API_KEYS"),
        reliability="structured",
        cost_tier="free_or_paid",
        priority=10,
        notes="Primary current odds feed. Keep force-fresh scans on this when quota allows.",
    ),
    SourceProvider(
        key="betfair",
        name="Betfair Exchange",
        category="odds",
        sports=("soccer", "basketball", "nhl", "tennis"),
        evidence=("exchange_liquidity", "market_depth", "price_validation"),
        env_vars=("BETFAIR_USERNAME", "BETFAIR_PASSWORD", "BETFAIR_APP_KEY"),
        reliability="structured",
        cost_tier="account_required",
        priority=18,
        notes="Useful as a liquidity/price sanity check when credentials are configured.",
    ),
    SourceProvider(
        key="api_sports_football",
        name="API-Sports Football",
        category="availability",
        sports=("soccer",),
        evidence=("fixtures", "injuries", "suspensions", "lineups", "team_form", "h2h"),
        env_vars=("API_SPORTS_KEY",),
        reliability="structured",
        cost_tier="free_or_paid",
        priority=12,
        notes="Prefer the direct API-Sports key over RapidAPI to reduce 403/provider mismatch risk.",
    ),
    SourceProvider(
        key="rapidapi_football",
        name="API-Football via RapidAPI",
        category="availability",
        sports=("soccer",),
        evidence=("fixtures", "injuries", "suspensions", "lineups", "team_form", "h2h"),
        env_vars=("RAPIDAPI_KEY",),
        reliability="structured_but_quota_sensitive",
        cost_tier="free_or_paid",
        priority=22,
        notes="Fallback soccer enrichment path. 403/429 responses should downgrade confidence.",
    ),
    SourceProvider(
        key="football_data",
        name="football-data.org",
        category="fixtures_results",
        sports=("soccer",),
        evidence=("fixtures", "results", "standings", "competition_context"),
        env_vars=("FOOTBALL_DATA_API_KEY",),
        reliability="structured",
        cost_tier="free_or_paid",
        priority=28,
        notes="Good fixture/result/standings companion, not a lineup/injury source.",
    ),
    SourceProvider(
        key="api_sports_basketball",
        name="API-Sports Basketball",
        category="availability",
        sports=("basketball",),
        evidence=("injuries", "lineup_context", "schedule", "team_form"),
        env_vars=("API_SPORTS_KEY",),
        reliability="structured",
        cost_tier="free_or_paid",
        priority=14,
        notes="Primary structured NBA availability source currently supported by the scanner.",
    ),
    SourceProvider(
        key="balldontlie",
        name="BallDontLie",
        category="historical_stats",
        sports=("basketball",),
        evidence=("games", "scores", "teams", "box_score_history"),
        env_vars=("BALLDONTLIE_API_KEY",),
        reliability="structured",
        cost_tier="free_or_paid",
        priority=32,
        notes="Good NBA historical source; not enough by itself for injury decisions.",
    ),
    SourceProvider(
        key="nhl_public_api",
        name="NHL Public API",
        category="availability",
        sports=("nhl",),
        evidence=("rosters", "player_stats", "schedule", "goalie_metrics_proxy"),
        env_vars=(),
        reliability="structured_keyless",
        cost_tier="free",
        priority=16,
        notes="No key required. Strong for rosters/stats; confirmed starter status still needs a goalie-specific source or news confirmation.",
    ),
    SourceProvider(
        key="dailyfaceoff_context",
        name="Daily Faceoff",
        category="availability_context",
        sports=("nhl",),
        evidence=("probable_goalies", "line_combinations"),
        env_vars=(),
        reliability="web_context",
        cost_tier="free",
        priority=34,
        notes="Useful goalie context, but treat as fallback unless a stable licensed feed is added.",
    ),
    SourceProvider(
        key="espn_context",
        name="ESPN",
        category="news_context",
        sports=("soccer", "basketball", "nhl", "mlb", "tennis"),
        evidence=("previews", "injury_context", "scores", "settlement_fallback"),
        env_vars=(),
        reliability="web_context",
        cost_tier="free",
        priority=40,
        notes="Good fallback context/settlement source, not the primary availability source.",
    ),
    SourceProvider(
        key="newsapi",
        name="NewsAPI",
        category="news_context",
        sports=("soccer", "basketball", "nhl", "mlb", "tennis"),
        evidence=("news_search", "preview_context", "injury_mentions"),
        env_vars=("NEWS_API_KEY",),
        reliability="structured_search",
        cost_tier="free_or_paid",
        priority=42,
        notes="Better than raw search for fresh news if configured; still not official lineup data.",
    ),
    SourceProvider(
        key="openweather",
        name="OpenWeatherMap",
        category="environment",
        sports=("soccer", "mlb", "tennis"),
        evidence=("weather", "wind", "temperature", "precipitation"),
        env_vars=("OPENWEATHER_API_KEY",),
        reliability="structured",
        cost_tier="free_or_paid",
        priority=36,
        notes="Useful for MLB weather/park risk and outdoor soccer/tennis context.",
    ),
)


def _has_env(env: Mapping[str, str], names: tuple[str, ...]) -> bool:
    if not names:
        return True
    return any(str(env.get(name) or "").strip() for name in names)


def get_source_registry(env: Mapping[str, str] | None = None) -> list[SourceProvider]:
    env = os.environ if env is None else env
    providers: list[SourceProvider] = []
    for provider in _PROVIDER_DEFINITIONS:
        configured = _has_env(env, provider.env_vars)
        missing = tuple(name for name in provider.env_vars if not str(env.get(name) or "").strip())
        providers.append(
            SourceProvider(
                **{
                    **asdict(provider),
                    "configured": configured,
                    "missing_env_vars": () if configured else missing,
                }
            )
        )
    return sorted(providers, key=lambda item: (item.priority, item.key))


def source_status_summary(env: Mapping[str, str] | None = None) -> dict:
    providers = get_source_registry(env)
    by_sport: dict[str, list[dict]] = {}
    missing_critical: dict[str, list[str]] = {}
    for provider in providers:
        payload = provider.as_dict()
        for sport in provider.sports:
            by_sport.setdefault(sport, []).append(payload)

    critical_requirements = {
        "soccer": ("api_sports_football", "rapidapi_football"),
        "basketball": ("api_sports_basketball",),
        "nhl": ("nhl_public_api",),
        "mlb": (),
        "tennis": (),
    }
    for sport, alternatives in critical_requirements.items():
        if not alternatives:
            continue
        if not any(next((p.configured for p in providers if p.key == key), False) for key in alternatives):
            missing_critical[sport] = list(alternatives)

    return {
        "providers": [provider.as_dict() for provider in providers],
        "by_sport": by_sport,
        "configured_count": sum(1 for provider in providers if provider.configured),
        "structured_configured_count": sum(
            1
            for provider in providers
            if provider.configured and provider.reliability.startswith("structured")
        ),
        "missing_critical": missing_critical,
        "recommendations": _recommendations(providers),
    }


def _recommendations(providers: list[SourceProvider]) -> list[str]:
    configured = {provider.key for provider in providers if provider.configured}
    notes: list[str] = []
    if "api_sports_football" not in configured and "rapidapi_football" not in configured:
        notes.append("Add API_SPORTS_KEY for direct soccer lineup/injury enrichment; use RAPIDAPI_KEY only as fallback.")
    if "api_sports_basketball" not in configured:
        notes.append("Add API_SPORTS_KEY to improve NBA injury/availability checks.")
    if "balldontlie" not in configured:
        notes.append("Add BALLDONTLIE_API_KEY for stronger NBA historical box-score coverage.")
    if "odds_api" not in configured:
        notes.append("Add ODDS_API_KEY or ODDS_API_KEYS before live market scans.")
    if "newsapi" not in configured:
        notes.append("Optional: add NEWS_API_KEY to reduce dependence on DuckDuckGo/Google timeouts for context search.")
    return notes
