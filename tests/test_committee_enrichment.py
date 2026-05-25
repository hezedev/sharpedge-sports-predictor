from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.committee import (
    EvidenceEnrichmentPass,
    ModelMindDecision,
    ModelVerdict,
    ResearchMindDecision,
    ResearchVerdict,
)
from src.committee import integration as committee_integration
from src.committee import evidence_enrichment as enrichment_module
from src.data.api_football_enricher import APIFootballEnricher


def _base_research(*, sport: str = "soccer", market_status: str = "INSUFFICIENT") -> ResearchMindDecision:
    return ResearchMindDecision(
        research_verdict=ResearchVerdict.AGREE,
        sport=sport,
        confidence="Low",
        data_freshness="acceptable_freshness",
        sources_checked=("odds_snapshot",),
        evidence_status=market_status,
        concrete_info_score=0,
        source_count=1,
        source_quality_summary="weak",
        fixture_verified=True,
        odds_age_minutes=20,
        odds_freshness_status="acceptable",
        market_availability_status="available",
        lineup_status="unknown",
        injury_status="not_checked",
        motivation_status="not_checked" if sport == "soccer" else "unknown",
        rotation_status="not_checked" if sport == "soccer" else "unknown",
        missing_evidence=("limited concrete research evidence",),
        sport_specific_missing_evidence=("limited concrete research evidence",),
        metadata={"fixture_verified": True, "match_status": "pre_match", "critical_missing_evidence": ["limited concrete research evidence"]},
    )


def _base_model() -> ModelMindDecision:
    return ModelMindDecision(
        model_verdict=ModelVerdict.BET,
        model_probability=0.60,
        market_implied_probability=0.54,
        vig_free_market_probability=0.51,
        fair_odds=1.67,
        minimum_acceptable_odds=1.80,
        current_odds=1.92,
        estimated_edge=0.08,
        confidence_interval=(0.55, 0.64),
        risk_tier="medium",
        suggested_market="moneyline",
        parlay_suitability="good_leg",
        reasons=("model edge is valid",),
    )


def _candidate(*, sport: str = "soccer", market: str = "moneyline", hours_from_now: float = 5.0) -> dict:
    commence = (datetime.now(timezone.utc) + timedelta(hours=hours_from_now)).isoformat()
    home = {
        "soccer": "Alpha FC",
        "mlb": "New York Yankees",
        "basketball": "Boston Celtics",
        "tennis": "Carlos Alcaraz",
        "tennis_wta": "Iga Swiatek",
        "nhl": "Boston Bruins",
    }.get(sport, "Home Team")
    away = {
        "soccer": "Beta FC",
        "mlb": "Boston Red Sox",
        "basketball": "New York Knicks",
        "tennis": "Jannik Sinner",
        "tennis_wta": "Coco Gauff",
        "nhl": "New York Rangers",
    }.get(sport, "Away Team")
    team = home if sport not in {"tennis", "tennis_wta"} else home
    return {
        "sport": sport,
        "market": market,
        "team": team,
        "home": home,
        "away": away,
        "status": "scheduled",
        "commence": commence,
        "commence_time": commence,
        "edge": 0.08,
        "odds": 1.92,
        "ml_prob": 0.60,
        "fair_prob": 0.52,
        "market_implied_prob": 0.54,
        "vig_free_implied_prob": 0.51,
        "minimum_acceptable_odds": 1.80,
        "confidence_range_low": 0.55,
        "confidence_range_high": 0.64,
        "lower_bound_passed": True,
        "publish_ready": True,
        "production_allowed": True,
        "market_status": "preferred",
        "stake_abs": 25.0,
        "kelly_stake_pct": 2.5,
        "odds_snapshot_age_hours": 1.0,
        "odds_fetched_at": datetime.now(timezone.utc).isoformat(),
        "bookmaker_last_update": (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat(),
        "scraped_context": {
            "home_team_name": home,
            "away_team_name": away,
        },
        "scraped_context_sources": ["odds_snapshot"],
        "scraped_context_highlights": [],
        "context_adjustments": [],
        "prediction_factors": [],
    }


class _FakeModelMind:
    def evaluate(self, candidate: dict) -> ModelMindDecision:
        return _base_model()


class _LowEdgeModelMind:
    def evaluate(self, candidate: dict) -> ModelMindDecision:
        base = _base_model()
        return ModelMindDecision(
            model_verdict=ModelVerdict.NO_BET,
            model_probability=base.model_probability,
            market_implied_probability=base.market_implied_probability,
            vig_free_market_probability=base.vig_free_market_probability,
            fair_odds=base.fair_odds,
            minimum_acceptable_odds=base.minimum_acceptable_odds,
            current_odds=base.current_odds,
            estimated_edge=0.01,
            confidence_interval=base.confidence_interval,
            risk_tier=base.risk_tier,
            suggested_market=base.suggested_market,
            parlay_suitability=base.parlay_suitability,
            reasons=("edge is below threshold",),
        )


def _tennis_candidate(*, sport: str = "tennis", market: str = "moneyline") -> dict:
    candidate = _candidate(sport=sport, market=market, hours_from_now=8)
    candidate.update(
        {
            "league": "ATP Rome" if sport == "tennis" else "WTA Rome",
            "tournament": "ATP Rome" if sport == "tennis" else "WTA Rome",
            "league_key": "tennis_atp_rome" if sport == "tennis" else "tennis_wta_rome",
            "status": "scheduled",
            "home": "Carlos Alcaraz" if sport == "tennis" else "Iga Swiatek",
            "away": "Jannik Sinner" if sport == "tennis" else "Coco Gauff",
            "team": "Carlos Alcaraz" if sport == "tennis" else "Iga Swiatek",
            "scraped_context_sources": ["odds_snapshot"],
            "scraped_context": {},
            "scraped_context_highlights": [],
            "prediction_factors": [],
            "context_adjustments": [],
        }
    )
    return candidate


def _soccer_fixture_payload(*, short_status: str = "NS", league_name: str = "Bundesliga", round_name: str = "Regular Season") -> dict:
    return {
        "fixture": {
            "id": 12345,
            "date": (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat(),
            "status": {"short": short_status},
        },
        "teams": {
            "home": {"name": "Alpha FC"},
            "away": {"name": "Beta FC"},
        },
        "league": {
            "name": league_name,
            "round": round_name,
        },
    }


def _soccer_availability_payload(
    *,
    confirmed: bool = False,
    probable_count: int = 0,
    injuries: int = 0,
    suspensions: int = 0,
    goalkeepers_named: bool = False,
) -> dict:
    return {
        "availability_source": "api_football",
        "lineup_source": "api_football",
        "availability_fetched_at": datetime.now(timezone.utc).isoformat(),
        "home_lineup_confirmed": 1 if confirmed else 0,
        "away_lineup_confirmed": 1 if confirmed else 0,
        "home_likely_starters_count": probable_count,
        "away_likely_starters_count": probable_count,
        "home_lineup_goalkeeper_named": 1 if goalkeepers_named else 0,
        "away_lineup_goalkeeper_named": 1 if goalkeepers_named else 0,
        "home_injuries_count": injuries,
        "away_injuries_count": 0,
        "home_suspensions_count": suspensions,
        "away_suspensions_count": 0,
    }


def _soccer_news_context(*, sources: list[str], highlights: list[str], items: list[dict] | None = None) -> dict:
    return {
        "sources": sources,
        "highlights": highlights,
        "items": items or [],
        "channels": {},
        "warnings": [],
    }


def _soccer_cache_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-05-05", tz="UTC"),
                "competition": "DE-Bundesliga",
                "home_team": "Alpha FC",
                "away_team": "Beta FC",
                "form_diff": 0.28,
                "xg_diff": 0.34,
                "dc_xg_diff": 0.18,
                "h2h_home_win_rate": 0.62,
                "home_rest_days": 5.0,
                "away_rest_days": 3.0,
                "home_season_pts_rate": 2.15,
                "away_season_pts_rate": 1.78,
            },
            {
                "date": pd.Timestamp("2026-05-07", tz="UTC"),
                "competition": "DE-Bundesliga",
                "home_team": "Alpha FC",
                "away_team": "Beta FC",
                "form_diff": 0.31,
                "xg_diff": 0.29,
                "dc_xg_diff": 0.14,
                "h2h_home_win_rate": 0.58,
                "home_rest_days": 4.0,
                "away_rest_days": 2.0,
                "home_season_pts_rate": 2.05,
                "away_season_pts_rate": 1.72,
            },
        ]
    )


def _soccer_standings_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"team_name": "Alpha FC", "position": 2, "points": 73},
            {"team_name": "Beta FC", "position": 4, "points": 68},
            {"team_name": "Gamma FC", "position": 1, "points": 75},
            {"team_name": "Delta FC", "position": 18, "points": 24},
        ]
    )


def _soccer_match_enrichment_payload() -> dict:
    return {
        "home_form": "WWDWW",
        "away_form": "LDWLL",
        "home_xg": 1.84,
        "away_xg": 1.11,
        "home_corners_avg": 6.2,
        "away_corners_avg": 3.9,
    }


def _tennis_cache_frame() -> pd.DataFrame:
    rows = [
        {
            "date": pd.Timestamp("2026-04-20", tz="UTC"),
            "player1_name": "Carlos Alcaraz",
            "player2_name": "Lorenzo Musetti",
            "result": "player1_win",
            "surface": "Clay",
            "round": "QF",
            "tourney_name": "ATP Rome",
            "player1_rank": 3,
            "player1_rank_pts": 7800,
            "player2_rank": 12,
            "player2_rank_pts": 3400,
            "p1_surface_win": 0.78,
            "p1_form": 0.82,
            "roll_p1_ace_rate": 0.118,
            "roll_p1_return_pressure": 0.345,
            "p1_load": 1.25,
        },
        {
            "date": pd.Timestamp("2026-04-23", tz="UTC"),
            "player1_name": "Carlos Alcaraz",
            "player2_name": "Casper Ruud",
            "result": "player1_win",
            "surface": "Clay",
            "round": "SF",
            "tourney_name": "ATP Rome",
            "player1_rank": 3,
            "player1_rank_pts": 7800,
            "player2_rank": 8,
            "player2_rank_pts": 4100,
            "p1_surface_win": 0.79,
            "p1_form": 0.84,
            "roll_p1_ace_rate": 0.121,
            "roll_p1_return_pressure": 0.352,
            "p1_load": 1.15,
        },
        {
            "date": pd.Timestamp("2026-04-19", tz="UTC"),
            "player1_name": "Jannik Sinner",
            "player2_name": "Daniil Medvedev",
            "result": "player1_win",
            "surface": "Clay",
            "round": "QF",
            "tourney_name": "ATP Rome",
            "player1_rank": 2,
            "player1_rank_pts": 8200,
            "player2_rank": 6,
            "player2_rank_pts": 4700,
            "p1_surface_win": 0.72,
            "p1_form": 0.76,
            "roll_p1_ace_rate": 0.109,
            "roll_p1_return_pressure": 0.311,
            "p1_load": 1.42,
        },
        {
            "date": pd.Timestamp("2026-04-24", tz="UTC"),
            "player1_name": "Jannik Sinner",
            "player2_name": "Alexander Zverev",
            "result": "player1_loss",
            "surface": "Clay",
            "round": "SF",
            "tourney_name": "ATP Rome",
            "player1_rank": 2,
            "player1_rank_pts": 8200,
            "player2_rank": 5,
            "player2_rank_pts": 5200,
            "p1_surface_win": 0.73,
            "p1_form": 0.74,
            "roll_p1_ace_rate": 0.111,
            "roll_p1_return_pressure": 0.303,
            "p1_load": 1.58,
        },
        {
            "date": pd.Timestamp("2026-04-26", tz="UTC"),
            "player1_name": "Carlos Alcaraz",
            "player2_name": "Jannik Sinner",
            "result": "player1_win",
            "surface": "Clay",
            "round": "F",
            "tourney_name": "ATP Rome",
            "player1_rank": 3,
            "player1_rank_pts": 7800,
            "player2_rank": 2,
            "player2_rank_pts": 8200,
            "p1_surface_win": 0.79,
            "p1_form": 0.85,
            "roll_p1_ace_rate": 0.122,
            "roll_p1_return_pressure": 0.355,
            "p1_load": 1.20,
        },
    ]
    return pd.DataFrame(rows)


def _mlb_candidate(*, market: str = "moneyline", team: str = "New York Yankees", hours_from_now: float = 5.0) -> dict:
    candidate = _candidate(sport="mlb", market=market, hours_from_now=hours_from_now)
    candidate.update(
        {
            "league": "MLB",
            "league_key": "baseball_mlb",
            "status": "scheduled",
            "team": team,
            "scraped_context": {},
            "scraped_context_sources": ["odds_snapshot"],
            "scraped_context_highlights": [],
            "prediction_factors": [],
            "context_adjustments": [],
        }
    )
    return candidate


def _mlb_cache_frame() -> pd.DataFrame:
    rows = [
        {
            "date": pd.Timestamp("2026-05-01", tz="UTC"),
            "home_team": "New York Yankees",
            "away_team": "Boston Red Sox",
            "home_rest_days": 1.0,
            "away_rest_days": 0.0,
            "home_games_L3D": 1,
            "away_games_L3D": 2,
            "home_home_wpct_20": 0.64,
            "away_away_wpct_20": 0.46,
            "sp_era_diff": -0.8,
            "sp_whip_diff": -0.14,
            "sp_k9_diff": 1.7,
            "away_travel_km": 310.0,
            "away_travel_tz_shift": 0.0,
        },
        {
            "date": pd.Timestamp("2026-05-03", tz="UTC"),
            "home_team": "New York Yankees",
            "away_team": "Boston Red Sox",
            "home_rest_days": 1.0,
            "away_rest_days": 1.0,
            "home_games_L3D": 1,
            "away_games_L3D": 1,
            "home_home_wpct_20": 0.66,
            "away_away_wpct_20": 0.44,
            "sp_era_diff": -0.9,
            "sp_whip_diff": -0.16,
            "sp_k9_diff": 1.9,
            "away_travel_km": 295.0,
            "away_travel_tz_shift": 0.0,
        },
    ]
    return pd.DataFrame(rows)


def _mlb_official_availability_payload(*, changed: bool = False) -> dict:
    return {
        "home_starter_confirmed": 1,
        "away_starter_confirmed": 1,
        "home_starter_name": "Gerrit Cole" if not changed else "Carlos Rodon",
        "away_starter_name": "Brayan Bello",
        "availability_source": "mlb_stats_api",
        "lineup_source": "mlb_stats_api",
        "home_likely_starters_count": 9,
        "away_likely_starters_count": 9,
        "home_starter_hand": "R",
        "away_starter_hand": "R",
    }


def _mlb_weather_payload() -> dict:
    return {
        "outdoor_weather_source": "openweather",
        "temperature_f": 71.0,
        "wind_mph": 6.0,
        "precip_mm": 0.0,
        "weather_risk": 0,
    }


def test_insufficient_evidence_triggers_enrichment() -> None:
    result = EvidenceEnrichmentPass().run(candidate=_candidate(), research=_base_research(), model=_base_model())
    assert result.triggered is True
    assert any("evidence status" in reason for reason in result.trigger_reasons)


def test_concrete_score_zero_triggers_enrichment() -> None:
    result = EvidenceEnrichmentPass().run(candidate=_candidate(), research=_base_research(), model=_base_model())
    assert any("concrete info score is below 50" in reason for reason in result.trigger_reasons)


def test_weak_source_quality_triggers_enrichment() -> None:
    result = EvidenceEnrichmentPass().run(candidate=_candidate(), research=_base_research(), model=_base_model())
    assert any("source quality is weak" in reason for reason in result.trigger_reasons)


def test_missing_soccer_injuries_and_rotation_triggers_soccer_enrichment_tasks() -> None:
    research = _base_research(sport="soccer", market_status="PARTIAL")
    result = EvidenceEnrichmentPass().run(candidate=_candidate(sport="soccer"), research=research, model=_base_model())
    searched = set(result.missing_evidence_searched)
    assert "injuries/suspensions" in searched
    assert "rotation risk" in searched
    assert "motivation/context" in searched


def test_missing_mlb_pitcher_triggers_enrichment() -> None:
    research = ResearchMindDecision(
        **{
            **_base_research(sport="mlb", market_status="PARTIAL").__dict__,
            "missing_evidence": ("probable starters are not fully confirmed for the current decision window",),
            "sport_specific_missing_evidence": ("probable starters not fully confirmed",),
            "metadata": {"fixture_verified": True, "match_status": "pre_match", "critical_missing_evidence": ["probable starters not fully confirmed"]},
        }
    )
    result = EvidenceEnrichmentPass().run(candidate=_candidate(sport="mlb"), research=research, model=_base_model())
    assert any("probable pitcher evidence is missing" in reason for reason in result.trigger_reasons)
    assert "probable pitchers" in result.missing_evidence_searched


def test_missing_tennis_surface_ranking_and_injury_context_triggers_enrichment() -> None:
    research = ResearchMindDecision(
        **{
            **_base_research(sport="tennis", market_status="INSUFFICIENT").__dict__,
            "missing_evidence": (
                "surface context was not checked for the tennis matchup",
                "ranking/Elo context is missing",
                "player injury context is missing",
            ),
            "sport_specific_missing_evidence": (
                "surface context was not checked for the tennis matchup",
                "ranking/Elo context is missing",
                "player injury context is missing",
            ),
            "metadata": {"fixture_verified": True, "match_status": "pre_match", "critical_missing_evidence": ["surface context was not checked for the tennis matchup"]},
        }
    )
    result = EvidenceEnrichmentPass().run(candidate=_candidate(sport="tennis"), research=research, model=_base_model())
    searched = set(result.missing_evidence_searched)
    assert "surface" in searched
    assert "ranking/Elo context" in searched
    assert "injury/retirement concerns" in searched


def test_enrichment_improves_evidence_but_not_enough_returns_hold(monkeypatch) -> None:
    candidate = _candidate(sport="basketball", hours_from_now=3)
    candidate["evidence_enrichment_payload"] = {
        "scraped_context": {
            "availability_source": "api_sports_basketball",
            "lineup_source": "api_sports_basketball",
            "availability_fetched_at": datetime.now(timezone.utc).isoformat(),
            "home_priority_absences_count": 1,
            "home_questionable_count": 1,
        },
        "scraped_context_sources": ["api_sports_basketball", "espn"],
        "context_adjustments": [
            {"name": "back_to_back", "summary": "Rest context checked."},
            {"name": "travel_fatigue", "summary": "Travel context checked."},
            {"name": "lineup_uncertainty", "summary": "A core player remains questionable."},
        ],
    }
    monkeypatch.setattr(committee_integration, "QuantModelMind", lambda: _FakeModelMind())
    published, review, suppressed, entries = committee_integration.run_committee_pipeline(published=[candidate], review=[], suppressed=[])
    assert published == []
    assert len(review) == 1
    assert suppressed == []
    assert review[0]["committee_enrichment"]["triggered"] is True
    assert review[0]["committee_final_decision"] == "HOLD"


def test_enrichment_resolves_evidence_and_model_gates_pass_to_bet(monkeypatch) -> None:
    candidate = _candidate(sport="soccer")
    candidate["evidence_enrichment_payload"] = {
        "scraped_context": {
            "availability_source": "api_football",
            "lineup_source": "api_football",
            "availability_fetched_at": datetime.now(timezone.utc).isoformat(),
            "home_lineup_confirmed": 1,
            "away_lineup_confirmed": 1,
            "home_likely_starters_count": 11,
            "away_likely_starters_count": 11,
        },
        "scraped_context_sources": ["api_football", "espn"],
        "scraped_context_highlights": [
            "Projected lineups checked from official team feed",
            "Team news cross-check available from ESPN preview coverage",
        ],
    }
    monkeypatch.setattr(committee_integration, "QuantModelMind", lambda: _FakeModelMind())
    published, review, suppressed, _ = committee_integration.run_committee_pipeline(published=[candidate], review=[], suppressed=[])
    assert len(published) == 1
    assert review == []
    assert suppressed == []
    assert published[0]["committee_enrichment"]["evidence_before"] in {"INSUFFICIENT", "PARTIAL"}
    assert published[0]["committee_enrichment"]["evidence_after"] in {"ACCEPTABLE", "COMPLETE"}
    assert published[0]["committee_final_decision"] == "BET"


def test_enrichment_finds_negative_evidence_and_returns_avoid(monkeypatch) -> None:
    candidate = _candidate(sport="soccer")
    candidate["evidence_enrichment_payload"] = {
        "scraped_context": {
            "availability_source": "api_football",
            "availability_fetched_at": datetime.now(timezone.utc).isoformat(),
            "home_priority_absences_count": 2,
            "home_suspensions_count": 1,
        },
        "scraped_context_sources": ["api_football", "espn"],
    }
    monkeypatch.setattr(committee_integration, "QuantModelMind", lambda: _FakeModelMind())
    published, review, suppressed, _ = committee_integration.run_committee_pipeline(published=[candidate], review=[], suppressed=[])
    assert published == []
    assert review == []
    assert len(suppressed) == 1
    assert suppressed[0]["committee_final_decision"] == "AVOID"


def test_no_reliable_evidence_found_keeps_candidate_off_board(monkeypatch) -> None:
    candidate = _candidate(sport="soccer")
    monkeypatch.setattr(committee_integration, "QuantModelMind", lambda: _FakeModelMind())
    published, review, suppressed, _ = committee_integration.run_committee_pipeline(published=[candidate], review=[], suppressed=[])
    assert published == []
    assert len(review) + len(suppressed) == 1
    assert (review or suppressed)[0]["committee_enrichment"]["triggered"] is True


def test_insufficient_evidence_still_blocks_stake_and_parlay_after_enrichment(monkeypatch) -> None:
    candidate = _candidate(sport="soccer")
    monkeypatch.setattr(committee_integration, "QuantModelMind", lambda: _FakeModelMind())
    published, review, suppressed, _ = committee_integration.run_committee_pipeline(published=[candidate], review=[], suppressed=[])
    blocked = (review or suppressed)[0]
    assert blocked["committee_effective_stake_abs"] == 0.0
    assert blocked["committee_effective_kelly_pct"] == 0.0
    assert blocked["committee_parlay_suitability"] == "blocked"


def test_enrichment_output_shows_evidence_before_and_after(monkeypatch) -> None:
    candidate = _candidate(sport="soccer")
    candidate["evidence_enrichment_payload"] = {
        "scraped_context": {
            "availability_source": "api_football",
            "lineup_source": "api_football",
            "availability_fetched_at": datetime.now(timezone.utc).isoformat(),
            "home_lineup_confirmed": 1,
            "away_lineup_confirmed": 1,
            "home_likely_starters_count": 11,
            "away_likely_starters_count": 11,
        },
        "scraped_context_sources": ["api_football", "espn"],
        "sources_found": ["api_football", "espn"],
    }
    monkeypatch.setattr(committee_integration, "QuantModelMind", lambda: _FakeModelMind())
    published, _, _, _ = committee_integration.run_committee_pipeline(published=[candidate], review=[], suppressed=[])
    payload = published[0]["committee"]["evidence_enrichment"]
    assert payload["triggered"] is True
    assert payload["evidence_before"] in {"INSUFFICIENT", "PARTIAL"}
    assert payload["evidence_after"] in {"ACCEPTABLE", "COMPLETE"}
    assert payload["concrete_score_after"] >= payload["concrete_score_before"]


def test_soccer_fetch_enrichment_populates_fixture_and_lineup_details(monkeypatch) -> None:
    research = _base_research(sport="soccer", market_status="PARTIAL")
    monkeypatch.setattr(
        EvidenceEnrichmentPass,
        "_fetch_soccer_fixture",
        staticmethod(lambda home, away, commence: _soccer_fixture_payload()),
    )
    monkeypatch.setattr(
        enrichment_module,
        "build_availability_context",
        lambda sport, game, snapshot=None: _soccer_availability_payload(
            confirmed=False,
            probable_count=11,
            injuries=1,
            suspensions=1,
            goalkeepers_named=True,
        ),
    )

    result = EvidenceEnrichmentPass().run(candidate=_candidate(sport="soccer"), research=research, model=_base_model())

    assert result.triggered is True
    assert "api_football" in result.sources_found
    assert result.details["fixture_status"] == "scheduled"
    assert result.details["lineup_status"] == "projected"
    assert result.details["probable_lineup_status"] == "projected"
    assert result.details["injury_status"] == "checked"
    assert result.details["suspension_status"] == "checked"
    assert result.details["goalkeeper_status"] == "confirmed"


def test_soccer_missing_lineups_near_kickoff_keeps_wait_for_lineups(monkeypatch) -> None:
    candidate = _candidate(sport="soccer", hours_from_now=0.5)
    monkeypatch.setattr(
        EvidenceEnrichmentPass,
        "_fetch_soccer_fixture",
        staticmethod(lambda home, away, commence: _soccer_fixture_payload()),
    )
    monkeypatch.setattr(
        enrichment_module,
        "build_availability_context",
        lambda sport, game, snapshot=None: _soccer_availability_payload(
            confirmed=False,
            probable_count=0,
            injuries=0,
            suspensions=0,
            goalkeepers_named=False,
        ),
    )
    monkeypatch.setattr(committee_integration, "QuantModelMind", lambda: _FakeModelMind())

    published, review, suppressed, _ = committee_integration.run_committee_pipeline(published=[candidate], review=[], suppressed=[])

    assert published == []
    assert suppressed == []
    assert len(review) == 1
    assert review[0]["committee_final_decision"] == "WAIT_FOR_LINEUPS"
    assert review[0]["committee_enrichment"]["lineup_status"] == "missing"


def test_soccer_live_fixture_blocks_prematch_bet(monkeypatch) -> None:
    candidate = _candidate(sport="soccer")
    monkeypatch.setattr(
        EvidenceEnrichmentPass,
        "_fetch_soccer_fixture",
        staticmethod(lambda home, away, commence: _soccer_fixture_payload(short_status="1H")),
    )
    monkeypatch.setattr(
        enrichment_module,
        "build_availability_context",
        lambda sport, game, snapshot=None: _soccer_availability_payload(
            confirmed=True,
            probable_count=11,
            injuries=0,
            suspensions=0,
            goalkeepers_named=True,
        ),
    )
    monkeypatch.setattr(committee_integration, "QuantModelMind", lambda: _FakeModelMind())

    published, review, suppressed, _ = committee_integration.run_committee_pipeline(published=[candidate], review=[], suppressed=[])

    assert published == []
    assert review == []
    assert len(suppressed) == 1
    assert suppressed[0]["committee_final_decision"] == "AVOID"
    assert suppressed[0]["committee_enrichment"]["fixture_status"] == "live"


def test_reputable_soccer_news_source_can_clear_injury_status(monkeypatch) -> None:
    research = _base_research(sport="soccer", market_status="PARTIAL")
    monkeypatch.setattr(
        EvidenceEnrichmentPass,
        "_fetch_soccer_fixture",
        staticmethod(lambda home, away, commence: _soccer_fixture_payload()),
    )
    monkeypatch.setattr(
        enrichment_module,
        "build_availability_context",
        lambda sport, game, snapshot=None: {"home_team_name": "Alpha FC", "away_team_name": "Beta FC"},
    )
    monkeypatch.setattr(
        enrichment_module,
        "collect_matchup_news_context",
        lambda **kwargs: _soccer_news_context(
            sources=["espn.com", "onefootball.com"],
            highlights=["Team news update: no fresh injury concerns and expected starters available."],
        ),
    )

    result = EvidenceEnrichmentPass().run(candidate=_candidate(sport="soccer"), research=research, model=_base_model())

    assert result.details["injury_status"] == "checked"
    assert "espn" in result.sources_found


def test_official_soccer_source_can_clear_injury_and_rotation_status(monkeypatch) -> None:
    candidate = _candidate(sport="soccer")
    candidate["scraped_context"] = {
        **candidate["scraped_context"],
        "european_rotation_risk": 1,
    }
    monkeypatch.setattr(
        EvidenceEnrichmentPass,
        "_fetch_soccer_fixture",
        staticmethod(lambda home, away, commence: _soccer_fixture_payload(league_name="UEFA Champions League", round_name="Semi-finals")),
    )
    monkeypatch.setattr(
        enrichment_module,
        "build_availability_context",
        lambda sport, game, snapshot=None: {"home_team_name": "Alpha FC", "away_team_name": "Beta FC"},
    )
    monkeypatch.setattr(
        enrichment_module,
        "collect_matchup_news_context",
        lambda **kwargs: _soccer_news_context(
            sources=["alphafc.com"],
            highlights=["Official team news: no fresh injury concerns and manager confirms a strong XI without planned rotation."],
            items=[
                {
                    "title": "Official team news: no fresh injury concerns",
                    "snippet": "Manager confirms a strong XI without planned rotation for the European tie.",
                    "url": "https://www.alphafc.com/news/official-team-news",
                    "source": "alphafc.com",
                }
            ],
        ),
    )

    result = EvidenceEnrichmentPass().run(candidate=candidate, research=_base_research(sport="soccer", market_status="PARTIAL"), model=_base_model())

    assert result.details["injury_status"] == "checked_fresh"
    assert result.details["rotation_status"] == "checked"
    assert result.details["source_quality"] == "strong"
    assert "team_official" in result.sources_found


def test_soccer_suspension_evidence_is_reflected(monkeypatch) -> None:
    research = _base_research(sport="soccer", market_status="PARTIAL")
    monkeypatch.setattr(
        EvidenceEnrichmentPass,
        "_fetch_soccer_fixture",
        staticmethod(lambda home, away, commence: _soccer_fixture_payload()),
    )
    monkeypatch.setattr(
        enrichment_module,
        "build_availability_context",
        lambda sport, game, snapshot=None: _soccer_availability_payload(
            confirmed=False,
            probable_count=11,
            injuries=0,
            suspensions=1,
            goalkeepers_named=True,
        ),
    )
    monkeypatch.setattr(
        enrichment_module,
        "collect_matchup_news_context",
        lambda **kwargs: _soccer_news_context(sources=["espn.com"], highlights=["Suspension update confirmed for the home side."]),
    )

    result = EvidenceEnrichmentPass().run(candidate=_candidate(sport="soccer"), research=research, model=_base_model())

    assert result.details["suspension_status"] == "checked"


def test_soccer_continental_fixture_requires_rotation_context(monkeypatch) -> None:
    candidate = _candidate(sport="soccer")
    candidate["scraped_context"] = {
        **candidate["scraped_context"],
        "european_rotation_risk": 1,
    }
    monkeypatch.setattr(EvidenceEnrichmentPass, "_fetch_soccer_fixture", staticmethod(lambda home, away, commence: _soccer_fixture_payload(league_name="UEFA Champions League", round_name="Semi-finals")))
    monkeypatch.setattr(enrichment_module, "build_availability_context", lambda sport, game, snapshot=None: {"home_team_name": "Alpha FC", "away_team_name": "Beta FC"})
    monkeypatch.setattr(enrichment_module, "collect_matchup_news_context", lambda **kwargs: _soccer_news_context(sources=["reddit.com"], highlights=["Fans expect rotation tonight."]))
    monkeypatch.setattr(committee_integration, "QuantModelMind", lambda: _FakeModelMind())

    published, review, suppressed, _ = committee_integration.run_committee_pipeline(published=[candidate], review=[], suppressed=[])

    assert published == []
    assert len(review) == 1
    assert review[0]["committee_enrichment"]["rotation_status"] in {"missing", "checked_proxy"}
    assert review[0]["committee_final_decision"] in {"HOLD", "WAIT_FOR_LINEUPS"}


def test_soccer_reliable_rotation_check_can_resolve_rotation_hold(monkeypatch) -> None:
    candidate = _candidate(sport="soccer")
    candidate["scraped_context"] = {
        **candidate["scraped_context"],
        "european_rotation_risk": 1,
    }
    candidate["evidence_enrichment_payload"] = {
        "scraped_context": {
            "availability_source": "api_football",
            "lineup_source": "api_football",
            "availability_fetched_at": datetime.now(timezone.utc).isoformat(),
            "home_lineup_confirmed": 1,
            "away_lineup_confirmed": 1,
            "home_likely_starters_count": 11,
            "away_likely_starters_count": 11,
        },
    }
    monkeypatch.setattr(EvidenceEnrichmentPass, "_fetch_soccer_fixture", staticmethod(lambda home, away, commence: _soccer_fixture_payload(league_name="UEFA Champions League", round_name="Semi-finals")))
    monkeypatch.setattr(enrichment_module, "build_availability_context", lambda sport, game, snapshot=None: _soccer_availability_payload(confirmed=True, probable_count=11, injuries=0, suspensions=0, goalkeepers_named=True))
    monkeypatch.setattr(
        enrichment_module,
        "collect_matchup_news_context",
        lambda **kwargs: _soccer_news_context(
            sources=["espn.com", "onefootball.com"],
            highlights=["Manager confirms a strong lineup with no major rotation despite the European tie."],
        ),
    )
    monkeypatch.setattr(committee_integration, "QuantModelMind", lambda: _FakeModelMind())

    published, review, suppressed, _ = committee_integration.run_committee_pipeline(published=[candidate], review=[], suppressed=[])

    assert published == []
    assert len(review) == 1
    assert suppressed == []
    assert review[0]["committee_enrichment"]["rotation_status"] == "checked"
    assert "HIGH_ROTATION_RISK" not in review[0]["committee_veto_flags"]


def test_soccer_end_season_without_motivation_context_remains_hold(monkeypatch) -> None:
    candidate = _candidate(sport="soccer")
    candidate["scraped_context"] = {
        **candidate["scraped_context"],
        "final_day_volatility": 1,
    }
    monkeypatch.setattr(EvidenceEnrichmentPass, "_fetch_soccer_fixture", staticmethod(lambda home, away, commence: _soccer_fixture_payload(league_name="Bundesliga", round_name="Final Round")))
    monkeypatch.setattr(EvidenceEnrichmentPass, "_soccer_match_enrichment", lambda self, home, away, commence: {})
    monkeypatch.setattr(EvidenceEnrichmentPass, "_soccer_snapshot", lambda self, home, away, competition=None: None)
    monkeypatch.setattr(EvidenceEnrichmentPass, "_soccer_standings_context", lambda self, candidate, payload, home_team, away_team: {})
    monkeypatch.setattr(enrichment_module, "build_availability_context", lambda sport, game, snapshot=None: {"home_team_name": "Alpha FC", "away_team_name": "Beta FC"})
    monkeypatch.setattr(enrichment_module, "collect_matchup_news_context", lambda **kwargs: _soccer_news_context(sources=["reddit.com"], highlights=["Fans debating final day scenarios."]))
    monkeypatch.setattr(committee_integration, "QuantModelMind", lambda: _FakeModelMind())

    published, review, suppressed, _ = committee_integration.run_committee_pipeline(published=[candidate], review=[], suppressed=[])

    assert published == []
    blocked = (review or suppressed)[0]
    assert blocked["committee"]["research_mind"]["evidence_status"] in {"PARTIAL", "INSUFFICIENT"}


def test_soccer_reliable_motivation_context_improves_evidence(monkeypatch) -> None:
    candidate = _candidate(sport="soccer")
    candidate["scraped_context"] = {
        **candidate["scraped_context"],
        "final_day_volatility": 1,
    }
    candidate["evidence_enrichment_payload"] = {
        "scraped_context": {
            "availability_source": "api_football",
            "lineup_source": "api_football",
            "availability_fetched_at": datetime.now(timezone.utc).isoformat(),
            "home_lineup_confirmed": 1,
            "away_lineup_confirmed": 1,
            "home_likely_starters_count": 11,
            "away_likely_starters_count": 11,
        },
    }
    monkeypatch.setattr(EvidenceEnrichmentPass, "_fetch_soccer_fixture", staticmethod(lambda home, away, commence: _soccer_fixture_payload(league_name="Bundesliga", round_name="Final Round")))
    monkeypatch.setattr(EvidenceEnrichmentPass, "_soccer_match_enrichment", lambda self, home, away, commence: {})
    monkeypatch.setattr(EvidenceEnrichmentPass, "_soccer_snapshot", lambda self, home, away, competition=None: None)
    monkeypatch.setattr(EvidenceEnrichmentPass, "_soccer_standings_context", lambda self, candidate, payload, home_team, away_team: {})
    monkeypatch.setattr(enrichment_module, "build_availability_context", lambda sport, game, snapshot=None: _soccer_availability_payload(confirmed=True, probable_count=11, injuries=0, suspensions=0, goalkeepers_named=True))
    monkeypatch.setattr(
        enrichment_module,
        "collect_matchup_news_context",
        lambda **kwargs: _soccer_news_context(
            sources=["espn.com", "onefootball.com"],
            highlights=["Must-win title race scenario confirmed in preview coverage with full motivation context."],
        ),
    )
    monkeypatch.setattr(committee_integration, "QuantModelMind", lambda: _FakeModelMind())

    published, review, suppressed, _ = committee_integration.run_committee_pipeline(published=[candidate], review=[], suppressed=[])

    assert published == []
    blocked = (review or suppressed)[0]
    assert blocked["committee_enrichment"]["motivation_status"] == "checked"
    assert blocked["committee_enrichment"]["source_quality"] in {"mixed", "strong"}
    assert blocked["committee_enrichment"]["concrete_score_after"] > blocked["committee_enrichment"]["concrete_score_before"]


def test_official_league_source_can_clear_soccer_motivation_context(monkeypatch) -> None:
    candidate = _candidate(sport="soccer")
    candidate["scraped_context"] = {
        **candidate["scraped_context"],
        "final_day_volatility": 1,
    }
    monkeypatch.setattr(
        EvidenceEnrichmentPass,
        "_fetch_soccer_fixture",
        staticmethod(lambda home, away, commence: _soccer_fixture_payload(league_name="Bundesliga", round_name="Final Round")),
    )
    monkeypatch.setattr(EvidenceEnrichmentPass, "_soccer_match_enrichment", lambda self, home, away, commence: {})
    monkeypatch.setattr(EvidenceEnrichmentPass, "_soccer_snapshot", lambda self, home, away, competition=None: None)
    monkeypatch.setattr(EvidenceEnrichmentPass, "_soccer_standings_context", lambda self, candidate, payload, home_team, away_team: {})
    monkeypatch.setattr(
        enrichment_module,
        "build_availability_context",
        lambda sport, game, snapshot=None: _soccer_availability_payload(confirmed=True, probable_count=11, injuries=0, suspensions=0, goalkeepers_named=True),
    )
    monkeypatch.setattr(
        enrichment_module,
        "collect_matchup_news_context",
        lambda **kwargs: _soccer_news_context(
            sources=["bundesliga.com"],
            highlights=["Official round preview confirms title-race stakes and European qualification pressure."],
            items=[
                {
                    "title": "Official round preview",
                    "snippet": "Title-race stakes and European qualification pressure define the final-round matchup.",
                    "url": "https://www.bundesliga.com/en/bundesliga/news/official-round-preview",
                    "source": "bundesliga.com",
                }
            ],
        ),
    )
    monkeypatch.setattr(committee_integration, "QuantModelMind", lambda: _FakeModelMind())

    published, review, suppressed, _ = committee_integration.run_committee_pipeline(published=[candidate], review=[], suppressed=[])

    assert published == []
    blocked = (review or suppressed)[0]
    assert blocked["committee_enrichment"]["motivation_status"] == "checked"
    assert "league_official" in blocked["committee_enrichment"]["sources_found"]
    assert blocked["committee_enrichment"]["source_quality"] == "strong"


def test_soccer_form_xg_cache_and_live_match_enrichment_populate_soccer_statuses(monkeypatch) -> None:
    research = _base_research(sport="soccer", market_status="PARTIAL")
    monkeypatch.setattr(
        EvidenceEnrichmentPass,
        "_fetch_soccer_fixture",
        staticmethod(lambda home, away, commence: _soccer_fixture_payload(league_name="Bundesliga")),
    )
    monkeypatch.setattr(EvidenceEnrichmentPass, "_load_soccer_cache", staticmethod(lambda: _soccer_cache_frame()))
    monkeypatch.setattr(
        EvidenceEnrichmentPass,
        "_soccer_match_enrichment",
        lambda self, home, away, commence: _soccer_match_enrichment_payload(),
    )
    monkeypatch.setattr(enrichment_module, "build_availability_context", lambda sport, game, snapshot=None: _soccer_availability_payload(probable_count=11))
    monkeypatch.setattr(enrichment_module, "collect_matchup_news_context", lambda **kwargs: _soccer_news_context(sources=["espn.com"], highlights=["Preview confirms the matchup context remains stable."]))

    result = EvidenceEnrichmentPass().run(candidate=_candidate(sport="soccer", market="totals"), research=research, model=_base_model())

    assert result.details["home_away_form_status"] == "checked"
    assert result.details["xg_context_status"] == "checked"
    assert result.details["market_fit_status"] == "xg_supported"
    assert "soccer_feature_cache" in result.sources_found
    assert "api_football" in result.sources_found


def test_soccer_snapshot_falls_back_to_global_cache_and_team_resolver(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "home_team": ["Liverpool FC"],
            "away_team": ["Manchester United"],
            "competition": ["PL"],
            "date": [pd.Timestamp("2026-05-01T12:00:00Z")],
            "xg_diff": [0.22],
            "dc_xg_diff": [0.15],
        }
    )
    monkeypatch.setattr(EvidenceEnrichmentPass, "_load_soccer_cache", staticmethod(lambda: frame.copy()))

    class _StubResolver:
        def __init__(self, sport: str) -> None:
            self.sport = sport

        def resolve(self, name: str) -> str:
            return {
                "Liverpool": "Liverpool FC",
                "Man Utd": "Manchester United",
            }.get(name, name)

    monkeypatch.setattr(enrichment_module, "TeamResolver", _StubResolver)

    snapshot = EvidenceEnrichmentPass()._soccer_snapshot("Liverpool", "Man Utd", competition="BL1")

    assert snapshot is not None
    assert round(float(snapshot["xg_diff"]), 2) == 0.22


def test_soccer_standings_context_can_check_motivation_without_news(monkeypatch) -> None:
    research = _base_research(sport="soccer", market_status="PARTIAL")
    monkeypatch.setattr(
        EvidenceEnrichmentPass,
        "_fetch_soccer_fixture",
        staticmethod(lambda home, away, commence: _soccer_fixture_payload(league_name="Bundesliga")),
    )
    monkeypatch.setattr(
        EvidenceEnrichmentPass,
        "_soccer_standings_context",
        lambda self, candidate, payload, home_team, away_team: {
            "standings_source": "football_data",
            "standings_checked": True,
            "home_position": 2,
            "away_position": 4,
            "home_points": 73.0,
            "away_points": 68.0,
            "title_context": 1,
            "playoff_motivation": 1,
            "motivation_checked": 1,
        },
    )
    monkeypatch.setattr(
        EvidenceEnrichmentPass,
        "_soccer_match_enrichment",
        lambda self, home, away, commence: {},
    )
    monkeypatch.setattr(enrichment_module, "build_availability_context", lambda sport, game, snapshot=None: _soccer_availability_payload(probable_count=11))
    monkeypatch.setattr(enrichment_module, "collect_matchup_news_context", lambda **kwargs: _soccer_news_context(sources=[], highlights=[]))

    result = EvidenceEnrichmentPass().run(candidate=_candidate(sport="soccer"), research=research, model=_base_model())

    assert result.details["motivation_status"] == "checked"
    assert "football_data" in result.sources_found
    assert result.updated_candidate["scraped_context"]["motivation_checked"] == 1
    assert result.updated_candidate["scraped_context"]["playoff_motivation"] == 1


def test_soccer_feature_cache_can_cover_stale_standings_when_live_table_unavailable(monkeypatch) -> None:
    research = _base_research(sport="soccer", market_status="PARTIAL")
    candidate = _candidate(sport="soccer")
    candidate["standings_snapshot_age_hours"] = 240.0
    monkeypatch.setattr(
        EvidenceEnrichmentPass,
        "_fetch_soccer_fixture",
        staticmethod(lambda home, away, commence: _soccer_fixture_payload(league_name="Bundesliga")),
    )
    monkeypatch.setattr(EvidenceEnrichmentPass, "_load_soccer_cache", staticmethod(lambda: _soccer_cache_frame()))
    monkeypatch.setattr(EvidenceEnrichmentPass, "_soccer_standings_context", lambda self, candidate, payload, home_team, away_team: {})
    monkeypatch.setattr(EvidenceEnrichmentPass, "_soccer_match_enrichment", lambda self, home, away, commence: {})
    monkeypatch.setattr(enrichment_module, "build_availability_context", lambda sport, game, snapshot=None: _soccer_availability_payload(probable_count=11))
    monkeypatch.setattr(enrichment_module, "collect_matchup_news_context", lambda **kwargs: _soccer_news_context(sources=[], highlights=[]))

    result = EvidenceEnrichmentPass().run(candidate=candidate, research=research, model=_base_model())

    assert result.details["standings_status"] == "proxy"
    assert result.updated_candidate["standings_snapshot_age_hours"] == 12.0
    assert result.updated_candidate["scraped_context"]["standings_source"] == "soccer_feature_cache"


def test_soccer_pipeline_with_cache_and_standings_can_publish_when_gates_pass(monkeypatch) -> None:
    class _FakeSoccerFetcher:
        def __init__(self) -> None:
            self._api_key = "present"

        def fetch_standings(self, competition: str) -> pd.DataFrame:
            return _soccer_standings_frame()

    monkeypatch.setattr(
        EvidenceEnrichmentPass,
        "_fetch_soccer_fixture",
        staticmethod(lambda home, away, commence: _soccer_fixture_payload(league_name="Bundesliga")),
    )
    monkeypatch.setattr(EvidenceEnrichmentPass, "_load_soccer_cache", staticmethod(lambda: _soccer_cache_frame()))
    monkeypatch.setattr(
        EvidenceEnrichmentPass,
        "_soccer_match_enrichment",
        lambda self, home, away, commence: _soccer_match_enrichment_payload(),
    )
    monkeypatch.setattr(enrichment_module, "SoccerFetcher", _FakeSoccerFetcher)
    monkeypatch.setattr(
        enrichment_module,
        "build_availability_context",
        lambda sport, game, snapshot=None: _soccer_availability_payload(
            confirmed=True,
            probable_count=11,
            injuries=0,
            suspensions=0,
            goalkeepers_named=True,
        ),
    )
    monkeypatch.setattr(
        enrichment_module,
        "collect_matchup_news_context",
        lambda **kwargs: _soccer_news_context(
            sources=["espn.com", "onefootball.com"],
            highlights=["Official team news confirms no major absences before kickoff."],
        ),
    )
    monkeypatch.setattr(committee_integration, "QuantModelMind", lambda: _FakeModelMind())

    published, review, suppressed, _ = committee_integration.run_committee_pipeline(
        published=[_candidate(sport="soccer", market="moneyline")],
        review=[],
        suppressed=[],
    )

    assert len(published) == 1
    assert review == []
    assert suppressed == []
    assert published[0]["committee_enrichment"]["source_quality"] in {"mixed", "strong"}
    assert "api_football" in published[0]["committee_enrichment"]["sources_found"]
    assert published[0]["committee_enrichment"]["injury_status"] in {"checked", "checked_fresh"}
    assert published[0]["committee_enrichment"]["suspension_status"] in {"checked", "checked_fresh"}
    assert published[0]["committee_enrichment"]["home_away_form_status"] == "checked"
    assert published[0]["committee_enrichment"]["xg_context_status"] == "checked"
    assert published[0]["research_mind_source_quality_summary"] in {"mixed", "strong"}
    assert published[0]["research_mind_injury_status"] in {"checked", "checked_fresh"}
    assert published[0]["committee_final_decision"] == "BET"


def test_soccer_weak_community_only_source_does_not_approve(monkeypatch) -> None:
    candidate = _candidate(sport="soccer")
    monkeypatch.setattr(EvidenceEnrichmentPass, "_fetch_soccer_fixture", staticmethod(lambda home, away, commence: None))
    monkeypatch.setattr(EvidenceEnrichmentPass, "_soccer_match_enrichment", lambda self, home, away, commence: {})
    monkeypatch.setattr(EvidenceEnrichmentPass, "_soccer_snapshot", lambda self, home, away, competition=None: None)
    monkeypatch.setattr(EvidenceEnrichmentPass, "_soccer_standings_context", lambda self, candidate, payload, home_team, away_team: {})
    monkeypatch.setattr(enrichment_module, "build_availability_context", lambda sport, game, snapshot=None: {"home_team_name": "Alpha FC", "away_team_name": "Beta FC"})
    monkeypatch.setattr(enrichment_module, "collect_matchup_news_context", lambda **kwargs: _soccer_news_context(sources=["reddit.com"], highlights=["Fans think the lineup will be strong."]))
    monkeypatch.setattr(committee_integration, "QuantModelMind", lambda: _FakeModelMind())

    published, review, suppressed, _ = committee_integration.run_committee_pipeline(published=[candidate], review=[], suppressed=[])

    assert published == []
    assert len(review) + len(suppressed) == 1
    blocked = (review or suppressed)[0]
    assert blocked["committee_enrichment"]["source_quality"] == "weak"
    assert blocked["research_mind_source_quality_summary"] == "weak"


def test_soccer_negative_injury_or_rotation_evidence_remains_hold_or_avoid(monkeypatch) -> None:
    candidate = _candidate(sport="soccer")
    monkeypatch.setattr(EvidenceEnrichmentPass, "_fetch_soccer_fixture", staticmethod(lambda home, away, commence: _soccer_fixture_payload(league_name="UEFA Champions League", round_name="Semi-finals")))
    monkeypatch.setattr(enrichment_module, "build_availability_context", lambda sport, game, snapshot=None: _soccer_availability_payload(confirmed=False, probable_count=0, injuries=2, suspensions=1, goalkeepers_named=False))
    monkeypatch.setattr(
        enrichment_module,
        "collect_matchup_news_context",
        lambda **kwargs: _soccer_news_context(
            sources=["espn.com", "onefootball.com"],
            highlights=["Manager hints at rotation with several absences and a late fitness test for a key attacker."],
        ),
    )
    monkeypatch.setattr(committee_integration, "QuantModelMind", lambda: _FakeModelMind())

    published, review, suppressed, _ = committee_integration.run_committee_pipeline(published=[candidate], review=[], suppressed=[])

    assert published == []
    assert len(review) + len(suppressed) == 1
    assert (review or suppressed)[0]["committee_final_decision"] in {"HOLD", "WAIT_FOR_LINEUPS", "AVOID"}


def test_soccer_enrichment_provider_failure_does_not_crash(monkeypatch) -> None:
    monkeypatch.setattr(EvidenceEnrichmentPass, "_fetch_soccer_fixture", staticmethod(lambda home, away, commence: _soccer_fixture_payload()))
    monkeypatch.setattr(enrichment_module, "build_availability_context", lambda sport, game, snapshot=None: (_ for _ in ()).throw(RuntimeError("provider down")))
    monkeypatch.setattr(enrichment_module, "collect_matchup_news_context", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("news down")))
    result = EvidenceEnrichmentPass().run(candidate=_candidate(sport="soccer"), research=_base_research(sport="soccer"), model=_base_model())
    assert result.triggered is True
    assert result.details["availability_status"] == "provider_failed"
    assert result.details["news_context_status"] == "provider_failed"
    assert result.details["injury_status"] == "provider_failed"
    assert "availability" in result.details["providers_failed"]
    assert "news_context" in result.details["providers_failed"]


def test_soccer_enrichment_provider_paused_with_proxy_sources_marks_proxy_covered(monkeypatch) -> None:
    candidate = _candidate(sport="soccer")

    class _PausedEnricher:
        api_key = "present"
        _disabled_reason = "403 Forbidden from API-Football"

        @staticmethod
        def _is_temporarily_disabled() -> bool:
            return True

    monkeypatch.setattr(EvidenceEnrichmentPass, "_soccer_api_enricher", staticmethod(lambda: _PausedEnricher()))
    monkeypatch.setattr(EvidenceEnrichmentPass, "_fetch_soccer_fixture", staticmethod(lambda home, away, commence: None))
    monkeypatch.setattr(
        enrichment_module,
        "build_availability_context",
        lambda sport, game, snapshot=None: {
            "availability_source": "team_official",
            "home_likely_starters_count": 11,
            "away_likely_starters_count": 11,
        },
    )
    monkeypatch.setattr(
        enrichment_module,
        "collect_matchup_news_context",
        lambda **kwargs: _soccer_news_context(
            sources=["espn.com"],
            highlights=["Projected lineup and injury report both look stable."],
        ),
    )
    monkeypatch.setattr(EvidenceEnrichmentPass, "_soccer_match_enrichment", lambda self, home, away, commence: {})
    monkeypatch.setattr(EvidenceEnrichmentPass, "_soccer_snapshot", lambda self, home, away, competition=None: pd.Series({"form_diff": 0.15}))
    monkeypatch.setattr(EvidenceEnrichmentPass, "_soccer_standings_context", lambda self, candidate, payload, home_team, away_team: {"playoff_motivation": 0})

    result = EvidenceEnrichmentPass().run(candidate=candidate, research=_base_research(sport="soccer"), model=_base_model())

    assert result.details["api_football_status"] == "proxy_covered"
    assert result.details["availability_status"] in {"ok", "checked_proxy"}
    assert "api_football" not in result.details["providers_failed"]


def test_soccer_bookmaker_only_evidence_stays_weak_and_blocked(monkeypatch) -> None:
    candidate = _candidate(sport="soccer")
    monkeypatch.setattr(EvidenceEnrichmentPass, "_fetch_soccer_fixture", staticmethod(lambda home, away, commence: None))
    monkeypatch.setattr(enrichment_module, "build_availability_context", lambda sport, game, snapshot=None: {})
    monkeypatch.setattr(enrichment_module, "collect_matchup_news_context", lambda **kwargs: _soccer_news_context(sources=[], highlights=[]))
    monkeypatch.setattr(EvidenceEnrichmentPass, "_soccer_match_enrichment", lambda self, home, away, commence: {})
    monkeypatch.setattr(EvidenceEnrichmentPass, "_soccer_snapshot", lambda self, home, away, competition=None: None)
    monkeypatch.setattr(EvidenceEnrichmentPass, "_soccer_standings_context", lambda self, candidate, payload, home_team, away_team: {})
    monkeypatch.setattr(committee_integration, "QuantModelMind", lambda: _FakeModelMind())

    published, review, suppressed, _ = committee_integration.run_committee_pipeline(published=[candidate], review=[], suppressed=[])

    assert published == []
    blocked = (review or suppressed)[0]
    assert blocked["committee_enrichment"]["sources_found"] == ["bookmaker"]
    assert blocked["committee_enrichment"]["source_quality"] == "weak"
    assert blocked["research_mind_source_quality_summary"] == "weak"
    assert blocked["research_mind_injury_status"] in {"not_checked", "not_found", "provider_failed"}
    assert blocked["committee"]["research_mind"]["evidence_status"] in {"INSUFFICIENT", "PARTIAL"}
    assert blocked["committee_final_decision"] in {"HOLD", "WAIT_FOR_LINEUPS", "AVOID"}


def test_soccer_summary_json_keeps_enrichment_provider_debug_fields(monkeypatch, tmp_path) -> None:
    import daily_scan

    class _FakeSoccerFetcher:
        def __init__(self) -> None:
            self._api_key = "present"

        def fetch_standings(self, competition: str) -> pd.DataFrame:
            return _soccer_standings_frame()

    monkeypatch.setattr(
        EvidenceEnrichmentPass,
        "_fetch_soccer_fixture",
        staticmethod(lambda home, away, commence: _soccer_fixture_payload(league_name="Bundesliga")),
    )
    monkeypatch.setattr(EvidenceEnrichmentPass, "_load_soccer_cache", staticmethod(lambda: _soccer_cache_frame()))
    monkeypatch.setattr(
        EvidenceEnrichmentPass,
        "_soccer_match_enrichment",
        lambda self, home, away, commence: _soccer_match_enrichment_payload(),
    )
    monkeypatch.setattr(enrichment_module, "SoccerFetcher", _FakeSoccerFetcher)
    monkeypatch.setattr(
        enrichment_module,
        "build_availability_context",
        lambda sport, game, snapshot=None: _soccer_availability_payload(
            confirmed=False,
            probable_count=11,
            injuries=0,
            suspensions=0,
            goalkeepers_named=True,
        ),
    )
    monkeypatch.setattr(
        enrichment_module,
        "collect_matchup_news_context",
        lambda **kwargs: _soccer_news_context(
            sources=["espn.com", "onefootball.com"],
            highlights=["Preview confirms no major absences and stable matchup context."],
        ),
    )
    monkeypatch.setattr(committee_integration, "QuantModelMind", lambda: _FakeModelMind())

    published, review, suppressed, _ = committee_integration.run_committee_pipeline(
        published=[_candidate(sport="soccer", market="totals")],
        review=[],
        suppressed=[],
    )
    bet = (published or review or suppressed)[0]

    monkeypatch.setattr(daily_scan, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(daily_scan, "TODAY", "2026-05-07")
    monkeypatch.setattr(daily_scan, "_soccer_full_games", [])
    monkeypatch.setattr(daily_scan, "_other_sport_games", [])
    monkeypatch.setattr(daily_scan, "_scan_runtime_notes", [])

    daily_scan.write_report(
        all_bets=published,
        review_bets=review,
        suppressed_bets=suppressed,
        bankroll=1000.0,
    )

    summary = json.loads((tmp_path / "summary_2026-05-07.json").read_text())
    review_payloads = summary.get("single_bets", {}).get("review_bets", [])
    candidate_payloads = summary.get("single_bets", {}).get("bets", []) + review_payloads + summary.get("single_bets", {}).get("suppressed_bets", [])
    written = candidate_payloads[0]
    nested = ((written.get("committee") or {}).get("evidence_enrichment") or {})

    assert nested.get("providers_attempted")
    assert nested.get("api_football_status")
    assert nested.get("availability_status")
    assert nested.get("feature_cache_status")
    assert nested.get("standings_status")
    assert any(src != "bookmaker" for src in nested.get("sources_found", []))
    assert written.get("committee_enrichment_providers_attempted")
    assert written.get("committee_enrichment_api_football_status")


def test_api_football_pause_short_circuits_fixture_lookup(monkeypatch) -> None:
    monkeypatch.setenv("API_SPORTS_KEY", "test-key")
    enricher = APIFootballEnricher()
    enricher._temporarily_disable(hours=1, reason="429 Too Many Requests from API-Football")
    called = {"count": 0}

    def boom(path, params=None):
        called["count"] += 1
        raise AssertionError("should not hit API while paused")

    monkeypatch.setattr(enricher, "_get_json", boom)
    fixture = enricher._find_fixture("Alpha FC", "Beta FC")
    assert fixture is None
    assert called["count"] == 0


def test_tennis_missing_surface_triggers_enrichment() -> None:
    research = ResearchMindDecision(
        **{
            **_base_research(sport="tennis", market_status="INSUFFICIENT").__dict__,
            "missing_evidence": ("surface context was not checked for the tennis matchup",),
            "sport_specific_missing_evidence": ("surface context was not checked for the tennis matchup",),
            "metadata": {"fixture_verified": True, "match_status": "pre_match", "critical_missing_evidence": []},
        }
    )
    result = EvidenceEnrichmentPass().run(candidate=_tennis_candidate(), research=research, model=_base_model())
    assert result.triggered is True
    assert "surface" in result.missing_evidence_searched


def test_tennis_missing_ranking_context_triggers_enrichment() -> None:
    research = ResearchMindDecision(
        **{
            **_base_research(sport="tennis", market_status="PARTIAL").__dict__,
            "missing_evidence": ("ranking/Elo context was not checked for the tennis matchup",),
            "sport_specific_missing_evidence": ("ranking/Elo context was not checked for the tennis matchup",),
            "metadata": {"fixture_verified": True, "match_status": "pre_match", "critical_missing_evidence": []},
        }
    )
    result = EvidenceEnrichmentPass().run(candidate=_tennis_candidate(), research=research, model=_base_model())
    assert result.triggered is True
    assert "ranking/Elo context" in result.missing_evidence_searched


def test_tennis_enrichment_source_success_improves_evidence_score(monkeypatch) -> None:
    monkeypatch.setattr(EvidenceEnrichmentPass, "_load_tennis_cache", staticmethod(lambda sport: _tennis_cache_frame()))
    monkeypatch.setattr(committee_integration, "QuantModelMind", lambda: _FakeModelMind())
    published, _, _, _ = committee_integration.run_committee_pipeline(published=[_tennis_candidate()], review=[], suppressed=[])
    assert len(published) == 1
    enrichment = published[0]["committee_enrichment"]
    assert enrichment["concrete_score_after"] > enrichment["concrete_score_before"]
    assert enrichment["surface_status"] == "verified"
    assert enrichment["ranking_elo_status"] == "checked"


def test_tennis_weak_source_only_does_not_approve(monkeypatch) -> None:
    monkeypatch.setattr(EvidenceEnrichmentPass, "_load_tennis_cache", staticmethod(lambda sport: pd.DataFrame()))
    monkeypatch.setattr(committee_integration, "QuantModelMind", lambda: _FakeModelMind())
    published, review, suppressed, _ = committee_integration.run_committee_pipeline(published=[_tennis_candidate()], review=[], suppressed=[])
    assert published == []
    assert len(review) + len(suppressed) == 1


def test_tennis_unresolved_injury_concern_keeps_hold(monkeypatch) -> None:
    candidate = _tennis_candidate()
    candidate["scraped_context"] = {"injury_concern": True}
    monkeypatch.setattr(EvidenceEnrichmentPass, "_load_tennis_cache", staticmethod(lambda sport: _tennis_cache_frame()))
    monkeypatch.setattr(committee_integration, "QuantModelMind", lambda: _FakeModelMind())
    published, review, suppressed, _ = committee_integration.run_committee_pipeline(published=[candidate], review=[], suppressed=[])
    assert published == []
    assert len(review) == 1
    assert suppressed == []
    assert review[0]["committee_final_decision"] in {"HOLD", "WAIT_FOR_LINEUPS"}


def test_tennis_negative_injury_concern_returns_avoid(monkeypatch) -> None:
    candidate = _tennis_candidate()
    candidate["scraped_context_highlights"] = ["Player withdrew last week after a medical timeout and is not fully fit."]
    monkeypatch.setattr(EvidenceEnrichmentPass, "_load_tennis_cache", staticmethod(lambda sport: _tennis_cache_frame()))
    monkeypatch.setattr(committee_integration, "QuantModelMind", lambda: _FakeModelMind())
    published, review, suppressed, _ = committee_integration.run_committee_pipeline(published=[candidate], review=[], suppressed=[])
    assert published == []
    assert review == []
    assert len(suppressed) == 1
    assert suppressed[0]["committee_final_decision"] == "AVOID"


def test_tennis_acceptable_evidence_and_model_gates_can_become_bet(monkeypatch) -> None:
    monkeypatch.setattr(EvidenceEnrichmentPass, "_load_tennis_cache", staticmethod(lambda sport: _tennis_cache_frame()))
    monkeypatch.setattr(committee_integration, "QuantModelMind", lambda: _FakeModelMind())
    published, review, suppressed, _ = committee_integration.run_committee_pipeline(published=[_tennis_candidate()], review=[], suppressed=[])
    assert len(published) == 1
    assert review == []
    assert suppressed == []
    assert published[0]["research_mind_evidence_status"] == "ACCEPTABLE"
    assert published[0]["committee_final_decision"] == "BET"


def test_tennis_acceptable_evidence_but_model_edge_fail_becomes_no_bet(monkeypatch) -> None:
    monkeypatch.setattr(EvidenceEnrichmentPass, "_load_tennis_cache", staticmethod(lambda sport: _tennis_cache_frame()))
    monkeypatch.setattr(committee_integration, "QuantModelMind", lambda: _LowEdgeModelMind())
    published, review, suppressed, _ = committee_integration.run_committee_pipeline(published=[_tennis_candidate()], review=[], suppressed=[])
    assert published == []
    assert review == []
    assert len(suppressed) == 1
    assert suppressed[0]["committee_final_decision"] == "NO_BET"


def test_tennis_enrichment_failure_does_not_crash(monkeypatch) -> None:
    monkeypatch.setattr(EvidenceEnrichmentPass, "_load_tennis_cache", staticmethod(lambda sport: (_ for _ in ()).throw(RuntimeError("boom"))))
    monkeypatch.setattr(committee_integration, "QuantModelMind", lambda: _FakeModelMind())
    published, review, suppressed, _ = committee_integration.run_committee_pipeline(published=[_tennis_candidate()], review=[], suppressed=[])
    assert published == []
    assert len(review) + len(suppressed) == 1


def test_tennis_enrichment_output_includes_tennis_fields(monkeypatch) -> None:
    monkeypatch.setattr(EvidenceEnrichmentPass, "_load_tennis_cache", staticmethod(lambda sport: _tennis_cache_frame()))
    monkeypatch.setattr(committee_integration, "QuantModelMind", lambda: _FakeModelMind())
    published, _, _, _ = committee_integration.run_committee_pipeline(published=[_tennis_candidate()], review=[], suppressed=[])
    payload = published[0]["committee"]["evidence_enrichment"]
    assert payload["surface_status"] == "verified"
    assert payload["ranking_elo_status"] == "checked"
    assert payload["injury_retirement_status"] == "no_concern_found"
    assert payload["fatigue_status"] == "checked"
    assert payload["tournament_context_status"] == "checked"
    assert payload["style_matchup_status"] == "checked"


def test_tennis_enrichment_source_labels_never_log_raw_api_keys(monkeypatch) -> None:
    candidate = _tennis_candidate()
    candidate["evidence_enrichment_sources_checked"] = ["official_league_or_team", "sk-test-123456789012345678901234"]
    candidate["evidence_enrichment_payload"] = {
        "sources_found": ["bookmaker", "api_key=supersecret1234567890123"],
    }
    monkeypatch.setattr(EvidenceEnrichmentPass, "_load_tennis_cache", staticmethod(lambda sport: _tennis_cache_frame()))
    result = EvidenceEnrichmentPass().run(candidate=candidate, research=_base_research(sport="tennis"), model=_base_model())
    summary = result.to_dict()
    assert "sk-test-123456789012345678901234" not in summary["sources_checked"]
    assert "api_key=supersecret1234567890123" not in summary["sources_found"]


def test_missing_mlb_probable_pitcher_triggers_enrichment() -> None:
    research = ResearchMindDecision(
        **{
            **_base_research(sport="mlb", market_status="PARTIAL").__dict__,
            "missing_evidence": ("probable starters are not fully confirmed for the current decision window",),
            "sport_specific_missing_evidence": ("probable starters not fully confirmed",),
            "metadata": {"fixture_verified": True, "match_status": "pre_match", "critical_missing_evidence": ["probable starters not fully confirmed"]},
        }
    )
    result = EvidenceEnrichmentPass().run(candidate=_mlb_candidate(), research=research, model=_base_model())
    assert result.triggered is True
    assert "probable pitchers" in result.missing_evidence_searched


def test_mlb_pitcher_change_triggers_hold_and_re_evaluation(monkeypatch) -> None:
    monkeypatch.setattr(EvidenceEnrichmentPass, "_load_mlb_cache", staticmethod(lambda: _mlb_cache_frame()))
    monkeypatch.setattr(enrichment_module, "build_availability_context", lambda sport, game, snapshot=None: _mlb_official_availability_payload(changed=True))
    monkeypatch.setattr(enrichment_module, "build_environment_context", lambda sport, home, away, commence: _mlb_weather_payload())
    monkeypatch.setattr(committee_integration, "QuantModelMind", lambda: _FakeModelMind())
    candidate = _mlb_candidate()
    candidate["scraped_context"] = {"home_starter_name": "Gerrit Cole", "away_starter_name": "Brayan Bello"}
    published, review, suppressed, _ = committee_integration.run_committee_pipeline(published=[candidate], review=[], suppressed=[])
    assert published == []
    assert len(review) == 1
    assert suppressed == []
    assert review[0]["committee_final_decision"] == "HOLD"


def test_mlb_official_payload_improves_pitcher_evidence(monkeypatch) -> None:
    monkeypatch.setattr(EvidenceEnrichmentPass, "_load_mlb_cache", staticmethod(lambda: _mlb_cache_frame()))
    monkeypatch.setattr(enrichment_module, "build_availability_context", lambda sport, game, snapshot=None: _mlb_official_availability_payload())
    monkeypatch.setattr(enrichment_module, "build_environment_context", lambda sport, home, away, commence: {})
    result = EvidenceEnrichmentPass().run(candidate=_mlb_candidate(), research=_base_research(sport="mlb"), model=_base_model())
    payload = result.updated_candidate
    assert payload["scraped_context"]["home_starter_name"] == "Gerrit Cole"
    assert result.details["probable_pitcher_status"] == "confirmed"


def test_mlb_confirmed_starters_improve_evidence_score(monkeypatch) -> None:
    monkeypatch.setattr(EvidenceEnrichmentPass, "_load_mlb_cache", staticmethod(lambda: _mlb_cache_frame()))
    monkeypatch.setattr(enrichment_module, "build_availability_context", lambda sport, game, snapshot=None: _mlb_official_availability_payload())
    monkeypatch.setattr(enrichment_module, "build_environment_context", lambda sport, home, away, commence: _mlb_weather_payload())
    monkeypatch.setattr(committee_integration, "QuantModelMind", lambda: _FakeModelMind())
    published, _, _, _ = committee_integration.run_committee_pipeline(published=[_mlb_candidate()], review=[], suppressed=[])
    assert len(published) == 1
    enrichment = published[0]["committee_enrichment"]
    assert enrichment["concrete_score_after"] > enrichment["concrete_score_before"]
    assert enrichment["probable_pitcher_status"] == "confirmed"


def test_mlb_missing_bullpen_workload_downgrades_totals_candidate(monkeypatch) -> None:
    cache = _mlb_cache_frame().drop(columns=["home_games_L3D", "away_games_L3D"])
    monkeypatch.setattr(EvidenceEnrichmentPass, "_load_mlb_cache", staticmethod(lambda: cache))
    monkeypatch.setattr(enrichment_module, "build_availability_context", lambda sport, game, snapshot=None: _mlb_official_availability_payload())
    monkeypatch.setattr(enrichment_module, "build_environment_context", lambda sport, home, away, commence: _mlb_weather_payload())
    monkeypatch.setattr(committee_integration, "QuantModelMind", lambda: _FakeModelMind())
    published, review, suppressed, _ = committee_integration.run_committee_pipeline(published=[_mlb_candidate(market="totals")], review=[], suppressed=[])
    assert published == []
    assert len(review) + len(suppressed) == 1


def test_mlb_missing_weather_downgrades_totals_candidate_when_material(monkeypatch) -> None:
    monkeypatch.setattr(EvidenceEnrichmentPass, "_load_mlb_cache", staticmethod(lambda: _mlb_cache_frame()))
    monkeypatch.setattr(enrichment_module, "build_availability_context", lambda sport, game, snapshot=None: _mlb_official_availability_payload())
    monkeypatch.setattr(enrichment_module, "build_environment_context", lambda sport, home, away, commence: {})
    monkeypatch.setattr(committee_integration, "QuantModelMind", lambda: _FakeModelMind())
    published, review, suppressed, _ = committee_integration.run_committee_pipeline(published=[_mlb_candidate(market="totals")], review=[], suppressed=[])
    assert published == []
    assert len(review) + len(suppressed) == 1


def test_mlb_underdog_minus_one_point_five_without_support_becomes_avoid_or_review(monkeypatch) -> None:
    cache = _mlb_cache_frame().drop(columns=["home_games_L3D", "away_games_L3D", "sp_era_diff", "sp_whip_diff", "sp_k9_diff"])
    monkeypatch.setattr(EvidenceEnrichmentPass, "_load_mlb_cache", staticmethod(lambda: cache))
    monkeypatch.setattr(enrichment_module, "build_availability_context", lambda sport, game, snapshot=None: _mlb_official_availability_payload())
    monkeypatch.setattr(enrichment_module, "build_environment_context", lambda sport, home, away, commence: {})
    monkeypatch.setattr(committee_integration, "QuantModelMind", lambda: _FakeModelMind())
    candidate = _mlb_candidate(market="spreads", team="Boston Red Sox -1.5")
    candidate["odds"] = 2.25
    candidate["line"] = -1.5
    published, review, suppressed, _ = committee_integration.run_committee_pipeline(published=[candidate], review=[], suppressed=[])
    assert published == []
    assert len(review) + len(suppressed) == 1
    assert (review or suppressed)[0]["committee_final_decision"] in {"HOLD", "AVOID"}


def test_mlb_weak_source_only_does_not_approve(monkeypatch) -> None:
    monkeypatch.setattr(EvidenceEnrichmentPass, "_load_mlb_cache", staticmethod(lambda: pd.DataFrame()))
    monkeypatch.setattr(enrichment_module, "build_availability_context", lambda sport, game, snapshot=None: {})
    monkeypatch.setattr(enrichment_module, "build_environment_context", lambda sport, home, away, commence: {})
    monkeypatch.setattr(committee_integration, "QuantModelMind", lambda: _FakeModelMind())
    published, review, suppressed, _ = committee_integration.run_committee_pipeline(published=[_mlb_candidate()], review=[], suppressed=[])
    assert published == []
    assert len(review) + len(suppressed) == 1


def test_mlb_acceptable_evidence_and_model_gates_pass_can_become_bet(monkeypatch) -> None:
    monkeypatch.setattr(EvidenceEnrichmentPass, "_load_mlb_cache", staticmethod(lambda: _mlb_cache_frame()))
    monkeypatch.setattr(enrichment_module, "build_availability_context", lambda sport, game, snapshot=None: _mlb_official_availability_payload())
    monkeypatch.setattr(enrichment_module, "build_environment_context", lambda sport, home, away, commence: _mlb_weather_payload())
    monkeypatch.setattr(committee_integration, "QuantModelMind", lambda: _FakeModelMind())
    published, review, suppressed, _ = committee_integration.run_committee_pipeline(published=[_mlb_candidate()], review=[], suppressed=[])
    assert len(published) == 1
    assert review == []
    assert suppressed == []
    assert published[0]["committee_final_decision"] == "BET"


def test_mlb_acceptable_evidence_but_model_edge_fail_becomes_no_bet(monkeypatch) -> None:
    monkeypatch.setattr(EvidenceEnrichmentPass, "_load_mlb_cache", staticmethod(lambda: _mlb_cache_frame()))
    monkeypatch.setattr(enrichment_module, "build_availability_context", lambda sport, game, snapshot=None: _mlb_official_availability_payload())
    monkeypatch.setattr(enrichment_module, "build_environment_context", lambda sport, home, away, commence: _mlb_weather_payload())
    monkeypatch.setattr(committee_integration, "QuantModelMind", lambda: _LowEdgeModelMind())
    published, review, suppressed, _ = committee_integration.run_committee_pipeline(published=[_mlb_candidate()], review=[], suppressed=[])
    assert published == []
    assert review == []
    assert len(suppressed) == 1
    assert suppressed[0]["committee_final_decision"] == "NO_BET"


def test_mlb_enrichment_failure_does_not_crash(monkeypatch) -> None:
    monkeypatch.setattr(EvidenceEnrichmentPass, "_load_mlb_cache", staticmethod(lambda: (_ for _ in ()).throw(RuntimeError("boom"))))
    monkeypatch.setattr(committee_integration, "QuantModelMind", lambda: _FakeModelMind())
    published, review, suppressed, _ = committee_integration.run_committee_pipeline(published=[_mlb_candidate()], review=[], suppressed=[])
    assert published == []
    assert len(review) + len(suppressed) == 1


def test_mlb_finished_or_postponed_game_cannot_become_active_bet(monkeypatch) -> None:
    monkeypatch.setattr(EvidenceEnrichmentPass, "_load_mlb_cache", staticmethod(lambda: _mlb_cache_frame()))
    monkeypatch.setattr(enrichment_module, "build_availability_context", lambda sport, game, snapshot=None: _mlb_official_availability_payload())
    monkeypatch.setattr(enrichment_module, "build_environment_context", lambda sport, home, away, commence: _mlb_weather_payload())
    monkeypatch.setattr(committee_integration, "QuantModelMind", lambda: _FakeModelMind())
    candidate = _mlb_candidate()
    candidate["status"] = "final"
    published, review, suppressed, _ = committee_integration.run_committee_pipeline(published=[candidate], review=[], suppressed=[])
    assert published == []
    assert review == []
    assert len(suppressed) == 1


def test_mlb_enrichment_output_includes_mlb_fields(monkeypatch) -> None:
    monkeypatch.setattr(EvidenceEnrichmentPass, "_load_mlb_cache", staticmethod(lambda: _mlb_cache_frame()))
    monkeypatch.setattr(enrichment_module, "build_availability_context", lambda sport, game, snapshot=None: _mlb_official_availability_payload())
    monkeypatch.setattr(enrichment_module, "build_environment_context", lambda sport, home, away, commence: _mlb_weather_payload())
    monkeypatch.setattr(committee_integration, "QuantModelMind", lambda: _FakeModelMind())
    published, _, _, _ = committee_integration.run_committee_pipeline(published=[_mlb_candidate()], review=[], suppressed=[])
    payload = published[0]["committee"]["evidence_enrichment"]
    assert payload["fixture_status"] == "scheduled"
    assert payload["probable_pitcher_status"] == "confirmed"
    assert payload["home_pitcher"] == "Gerrit Cole"
    assert payload["away_pitcher"] == "Brayan Bello"
    assert payload["bullpen_status"] in {"checked_proxy", "checked"}
    assert payload["weather_status"] == "checked"
    assert payload["market_fit_status"]


def test_mlb_source_labels_never_log_raw_api_keys(monkeypatch) -> None:
    candidate = _mlb_candidate()
    candidate["evidence_enrichment_sources_checked"] = ["mlb_stats_api", "api_key=supersecret1234567890123"]
    candidate["evidence_enrichment_payload"] = {"sources_found": ["bookmaker", "sk-test-12345678901234567890"]}
    monkeypatch.setattr(EvidenceEnrichmentPass, "_load_mlb_cache", staticmethod(lambda: _mlb_cache_frame()))
    monkeypatch.setattr(enrichment_module, "build_availability_context", lambda sport, game, snapshot=None: _mlb_official_availability_payload())
    monkeypatch.setattr(enrichment_module, "build_environment_context", lambda sport, home, away, commence: {})
    result = EvidenceEnrichmentPass().run(candidate=candidate, research=_base_research(sport="mlb"), model=_base_model())
    summary = result.to_dict()
    assert "api_key=supersecret1234567890123" not in summary["sources_checked"]
    assert "sk-test-12345678901234567890" not in summary["sources_found"]
