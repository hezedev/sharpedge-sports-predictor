"""Shared sport/league capability registry for launch decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class CapabilityProfile:
    sport: str
    league_key: str
    league_label: str
    scanable: bool
    model_backed: bool
    publishable: bool
    review_only: bool
    reasoning_supported: bool
    competition_code: Optional[str] = None
    launch_label: str = "Review"
    launch_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SOCCER_COMPETITION_CAPABILITIES: dict[str, CapabilityProfile] = {
    "soccer_epl": CapabilityProfile(
        sport="soccer",
        league_key="soccer_epl",
        league_label="EPL",
        scanable=True,
        model_backed=True,
        publishable=True,
        review_only=False,
        reasoning_supported=True,
        competition_code="PL",
        launch_label="Production",
        launch_note="Top-flight soccer league with refreshed model support.",
    ),
    "soccer_efl_champ": CapabilityProfile(
        sport="soccer",
        league_key="soccer_efl_champ",
        league_label="Championship",
        scanable=True,
        model_backed=True,
        publishable=True,
        review_only=False,
        reasoning_supported=True,
        competition_code="ELC",
        launch_label="Production",
        launch_note="Expanded production soccer league with current training scope.",
    ),
    "soccer_germany_bundesliga": CapabilityProfile("soccer", "soccer_germany_bundesliga", "Bundesliga", True, True, True, False, True, "BL1", "Production", "Top-flight soccer league with refreshed model support."),
    "soccer_italy_serie_a": CapabilityProfile("soccer", "soccer_italy_serie_a", "Serie A", True, True, True, False, True, "SA", "Production", "Top-flight soccer league with refreshed model support."),
    "soccer_spain_la_liga": CapabilityProfile("soccer", "soccer_spain_la_liga", "La Liga", True, True, True, False, True, "PD", "Production", "Top-flight soccer league with refreshed model support."),
    "soccer_france_ligue_one": CapabilityProfile("soccer", "soccer_france_ligue_one", "Ligue 1", True, True, True, False, True, "FL1", "Production", "Top-flight soccer league with refreshed model support."),
    "soccer_uefa_champs_league": CapabilityProfile("soccer", "soccer_uefa_champs_league", "Champions League", True, True, True, False, True, "CL", "Production", "Elite competition with current model/training coverage."),
    "soccer_portugal_primeira_liga": CapabilityProfile("soccer", "soccer_portugal_primeira_liga", "Primeira Liga", True, True, True, False, True, "PPL", "Production", "Production-backed soccer league with mapped standings/training."),
    "soccer_netherlands_eredivisie": CapabilityProfile("soccer", "soccer_netherlands_eredivisie", "Eredivisie", True, True, True, False, True, "DED", "Production", "Production-backed soccer league with mapped standings/training."),
    "soccer_brazil_campeonato": CapabilityProfile("soccer", "soccer_brazil_campeonato", "Brazil Serie A", True, True, True, False, True, "BSA", "Production", "Expanded production soccer league with current training scope."),
    "soccer_austria_bundesliga": CapabilityProfile("soccer", "soccer_austria_bundesliga", "Austria Bundesliga", True, False, False, True, True, None, "Review", "Visible in scans, but held out until training/model support is refreshed."),
    "soccer_belgium_first_div": CapabilityProfile("soccer", "soccer_belgium_first_div", "Belgian Pro League", True, False, False, True, True, None, "Review", "Visible in scans, but held out until training/model support is refreshed."),
    "soccer_brazil_serie_b": CapabilityProfile("soccer", "soccer_brazil_serie_b", "Brazil Serie B", True, False, False, True, True, None, "Review", "Second-division soccer held in review until deeper validation is complete."),
    "soccer_conmebol_copa_libertadores": CapabilityProfile("soccer", "soccer_conmebol_copa_libertadores", "Copa Libertadores", True, True, True, False, True, None, "Production", "Continental competition promoted after the South America coverage expansion and retraining pass."),
    "soccer_conmebol_copa_sudamericana": CapabilityProfile("soccer", "soccer_conmebol_copa_sudamericana", "Copa Sudamericana", True, True, True, False, True, None, "Production", "Continental competition promoted after the South America coverage expansion and retraining pass."),
    "soccer_denmark_superliga": CapabilityProfile("soccer", "soccer_denmark_superliga", "Denmark Superliga", True, False, False, True, True, None, "Review", "Visible in scans, but held out until training/model support is refreshed."),
    "soccer_england_league1": CapabilityProfile("soccer", "soccer_england_league1", "League One", True, False, False, True, True, None, "Review", "Lower-division soccer remains review-only at launch."),
    "soccer_england_league2": CapabilityProfile("soccer", "soccer_england_league2", "League Two", True, False, False, True, True, None, "Review", "Lower-division soccer remains review-only at launch."),
    "soccer_fa_cup": CapabilityProfile("soccer", "soccer_fa_cup", "FA Cup", True, False, False, True, True, None, "Review", "Cup competition remains review-only at launch."),
    "soccer_fifa_world_cup": CapabilityProfile("soccer", "soccer_fifa_world_cup", "World Cup", True, False, False, True, True, None, "Review", "Tournament football remains review-only at launch."),
    "soccer_finland_veikkausliiga": CapabilityProfile("soccer", "soccer_finland_veikkausliiga", "Veikkausliiga", True, False, False, True, True, None, "Review", "Visible in scans, but held out until training/model support is refreshed."),
    "soccer_france_coupe_de_france": CapabilityProfile("soccer", "soccer_france_coupe_de_france", "Coupe de France", True, False, False, True, True, None, "Review", "Cup competition remains review-only at launch."),
    "soccer_france_ligue_two": CapabilityProfile("soccer", "soccer_france_ligue_two", "Ligue 2", True, False, False, True, True, None, "Review", "Second-division soccer remains review-only at launch."),
    "soccer_germany_bundesliga2": CapabilityProfile("soccer", "soccer_germany_bundesliga2", "Bundesliga 2", True, False, False, True, True, None, "Review", "Second-division soccer remains review-only at launch."),
    "soccer_germany_dfb_pokal": CapabilityProfile("soccer", "soccer_germany_dfb_pokal", "DFB Pokal", True, False, False, True, True, None, "Review", "Cup competition remains review-only at launch."),
    "soccer_germany_liga3": CapabilityProfile("soccer", "soccer_germany_liga3", "Liga 3", True, False, False, True, True, None, "Review", "Lower-division soccer remains review-only at launch."),
    "soccer_greece_super_league": CapabilityProfile("soccer", "soccer_greece_super_league", "Greece Super League", True, False, False, True, True, None, "Review", "Visible in scans, but held out until training/model support is refreshed."),
    "soccer_italy_coppa_italia": CapabilityProfile("soccer", "soccer_italy_coppa_italia", "Coppa Italia", True, False, False, True, True, None, "Review", "Cup competition remains review-only at launch."),
    "soccer_italy_serie_b": CapabilityProfile("soccer", "soccer_italy_serie_b", "Serie B", True, False, False, True, True, None, "Review", "Second-division soccer remains review-only at launch."),
    "soccer_japan_j_league": CapabilityProfile("soccer", "soccer_japan_j_league", "J League", True, True, True, False, True, None, "Production", "Expanded production soccer league with refreshed Japan top-flight and feeder-club history."),
    "soccer_korea_kleague1": CapabilityProfile("soccer", "soccer_korea_kleague1", "K League 1", True, False, False, True, True, None, "Review", "Visible in scans, but held out until training/model support is refreshed."),
    "soccer_league_of_ireland": CapabilityProfile("soccer", "soccer_league_of_ireland", "League of Ireland", True, False, False, True, True, None, "Review", "Visible in scans, but held out until training/model support is refreshed."),
    "soccer_mexico_ligamx": CapabilityProfile("soccer", "soccer_mexico_ligamx", "Liga MX", True, False, False, True, True, None, "Review", "Visible in scans, but held out until training/model support is refreshed."),
    "soccer_norway_eliteserien": CapabilityProfile("soccer", "soccer_norway_eliteserien", "Eliteserien", True, False, False, True, True, None, "Review", "Visible in scans, but held out until training/model support is refreshed."),
    "soccer_poland_ekstraklasa": CapabilityProfile("soccer", "soccer_poland_ekstraklasa", "Ekstraklasa", True, False, False, True, True, None, "Review", "Visible in scans, but held out until training/model support is refreshed."),
    "soccer_russia_premier_league": CapabilityProfile("soccer", "soccer_russia_premier_league", "Russia Premier League", True, False, False, True, True, None, "Review", "Visible in scans, but held out until training/model support is refreshed."),
    "soccer_saudi_arabia_pro_league": CapabilityProfile("soccer", "soccer_saudi_arabia_pro_league", "Saudi Pro League", True, True, True, False, True, None, "Production", "Expanded production soccer league with refreshed domestic and promoted-club coverage."),
    "soccer_spl": CapabilityProfile("soccer", "soccer_spl", "Scottish Prem", True, False, False, True, True, None, "Review", "Visible in scans, but held out until training/model support is refreshed."),
    "soccer_spain_segunda_division": CapabilityProfile("soccer", "soccer_spain_segunda_division", "Segunda Division", True, False, False, True, True, None, "Review", "Second-division soccer remains review-only at launch."),
    "soccer_sweden_allsvenskan": CapabilityProfile("soccer", "soccer_sweden_allsvenskan", "Allsvenskan", True, False, False, True, True, None, "Review", "Visible in scans, but held out until training/model support is refreshed."),
    "soccer_sweden_superettan": CapabilityProfile("soccer", "soccer_sweden_superettan", "Superettan", True, False, False, True, True, None, "Review", "Lower-division soccer remains review-only at launch."),
    "soccer_switzerland_superleague": CapabilityProfile("soccer", "soccer_switzerland_superleague", "Switzerland Super League", True, False, False, True, True, None, "Review", "Visible in scans, but held out until training/model support is refreshed."),
    "soccer_turkey_super_league": CapabilityProfile("soccer", "soccer_turkey_super_league", "Super Lig", True, False, False, True, True, None, "Review", "Visible in scans, but held out until training/model support is refreshed."),
    "soccer_uefa_europa_conference_league": CapabilityProfile("soccer", "soccer_uefa_europa_conference_league", "Europa Conference League", True, True, True, False, True, None, "Production", "Continental competition promoted after the broader soccer history expansion and current validation pass."),
    "soccer_uefa_europa_league": CapabilityProfile("soccer", "soccer_uefa_europa_league", "Europa League", True, True, True, False, True, None, "Production", "Continental competition promoted after the broader soccer history expansion and current validation pass."),
    "soccer_usa_mls": CapabilityProfile("soccer", "soccer_usa_mls", "MLS", True, False, False, True, True, None, "Review", "Visible in scans, but held out until training/model support is refreshed."),
}

SPORT_BASE_CAPABILITIES: dict[str, CapabilityProfile] = {
    "soccer": CapabilityProfile("soccer", "soccer", "Soccer", True, True, True, False, True, None, "Production", "Generic soccer path defaults to production until a specific review-only league is identified."),
    "basketball": CapabilityProfile("basketball", "basketball_nba", "NBA", True, True, True, False, True, None, "Production", "NBA is live with full current league coverage for the supported basketball lane."),
    "mlb": CapabilityProfile("mlb", "baseball_mlb", "MLB", True, True, True, False, True, None, "Production", "MLB is live with full current league coverage for the supported baseball lane."),
    "nhl": CapabilityProfile("nhl", "icehockey_nhl", "NHL", True, True, True, False, True, None, "Production", "NHL is live with full current league coverage for the supported hockey lane."),
    "tennis": CapabilityProfile("tennis", "tennis_atp", "ATP", True, True, True, False, True, None, "Production", "ATP is live with the current supported market set."),
    "tennis_wta": CapabilityProfile("tennis_wta", "tennis_wta", "WTA", True, True, True, False, True, None, "Production", "WTA now has dedicated model artifacts and a calibrator and is live in the current supported tennis lane."),
}

SOCCER_SPORT_KEYS: list[str] = list(SOCCER_COMPETITION_CAPABILITIES.keys())
SOCCER_ODDS_TO_COMPETITION: dict[str, str] = {
    key: profile.competition_code
    for key, profile in SOCCER_COMPETITION_CAPABILITIES.items()
    if profile.competition_code
}
SOCCER_PRETTY_LABELS: dict[str, str] = {
    key: profile.league_label
    for key, profile in SOCCER_COMPETITION_CAPABILITIES.items()
}
SOCCER_LEAGUE_NAME_TO_KEY: dict[str, str] = {
    profile.league_label.lower(): key
    for key, profile in SOCCER_COMPETITION_CAPABILITIES.items()
}
SOCCER_LEAGUE_SHORTHAND_ALIASES: dict[str, str] = {
    "world_cup": "soccer_fifa_world_cup",
    "worldcup": "soccer_fifa_world_cup",
    "fifa_world_cup": "soccer_fifa_world_cup",
    "fifa": "soccer_fifa_world_cup",
    "wc": "soccer_fifa_world_cup",
}


def soccer_scanable_keys() -> list[str]:
    return [key for key, profile in SOCCER_COMPETITION_CAPABILITIES.items() if profile.scanable]


def soccer_pretty_label(sport_key: str) -> str:
    return SOCCER_PRETTY_LABELS.get(sport_key, sport_key.replace("soccer_", "").replace("_", " ").title())


def resolve_soccer_key(*, sport_key: Optional[str] = None, league: Optional[str] = None) -> Optional[str]:
    if sport_key:
        return sport_key if sport_key in SOCCER_COMPETITION_CAPABILITIES else None
    if league:
        return SOCCER_LEAGUE_NAME_TO_KEY.get(str(league).strip().lower())
    return None


def get_capability_profile(
    *,
    sport: Optional[str] = None,
    league: Optional[str] = None,
    sport_key: Optional[str] = None,
) -> CapabilityProfile:
    normalized_sport = str(sport or "").strip().lower()
    resolved_key = resolve_soccer_key(sport_key=sport_key, league=league) if normalized_sport == "soccer" else None
    if resolved_key:
        return SOCCER_COMPETITION_CAPABILITIES[resolved_key]
    if normalized_sport == "soccer" and sport_key:
        return CapabilityProfile(
            sport="soccer",
            league_key=str(sport_key),
            league_label=soccer_pretty_label(str(sport_key)),
            scanable=True,
            model_backed=False,
            publishable=False,
            review_only=True,
            reasoning_supported=True,
            competition_code=None,
            launch_label="Review",
            launch_note="Active soccer market discovered from the odds feed, but not yet mapped to production model coverage.",
        )
    if normalized_sport in SPORT_BASE_CAPABILITIES:
        return SPORT_BASE_CAPABILITIES[normalized_sport]
    return CapabilityProfile(
        sport=normalized_sport or "unknown",
        league_key=sport_key or normalized_sport or "unknown",
        league_label=str(league or sport_key or normalized_sport or "Unknown"),
        scanable=False,
        model_backed=False,
        publishable=False,
        review_only=True,
        reasoning_supported=False,
        competition_code=None,
        launch_label="Review",
        launch_note="This sport/league is not configured as a launch-supported production lane.",
    )


def enrich_with_capability(
    row: dict[str, Any],
    *,
    sport: Optional[str] = None,
    league: Optional[str] = None,
    sport_key: Optional[str] = None,
) -> dict[str, Any]:
    profile = get_capability_profile(
        sport=sport or row.get("sport"),
        league=league or row.get("league"),
        sport_key=sport_key or row.get("league_key") or row.get("sport_key"),
    )
    enriched = dict(row)
    enriched.update(profile.to_dict())
    return enriched
