from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.committee import ModelMindDecision, ModelVerdict, ResearchMindDecision, ResearchVerdict
from src.committee import integration as committee_integration


def _candidate() -> dict:
    commence = (datetime.now(timezone.utc) + timedelta(hours=5)).isoformat()
    return {
        "sport": "soccer",
        "market": "moneyline",
        "team": "Alpha FC",
        "home": "Alpha FC",
        "away": "Beta FC",
        "edge": 0.08,
        "odds": 1.92,
        "ml_prob": 0.60,
        "fair_prob": 0.52,
        "market_implied_prob": 0.54,
        "vig_free_implied_prob": 0.51,
        "minimum_acceptable_odds": 1.78,
        "confidence_range_low": 0.55,
        "confidence_range_high": 0.64,
        "lower_bound_passed": True,
        "publish_ready": True,
        "production_allowed": True,
        "market_status": "preferred",
        "recommended_market": "double_chance",
        "commence": commence,
        "commence_time": commence,
        "odds_snapshot_age_hours": 1.0,
        "standings_snapshot_age_hours": 3.0,
        "scraped_context": {
            "home_team_name": "Alpha FC",
            "away_team_name": "Beta FC",
            "availability_source": "api_football",
            "availability_fetched_at": datetime.now(timezone.utc).isoformat(),
            "home_lineup_confirmed": 1,
            "away_lineup_confirmed": 1,
            "home_likely_starters_count": 11,
            "away_likely_starters_count": 11,
        },
        "scraped_context_sources": ["api_football"],
        "scraped_context_highlights": ["Availability context fetched from api_football"],
        "substitute_candidate": {
            "sport": "soccer",
            "market": "double_chance",
            "team": "Alpha FC or Draw",
            "home": "Alpha FC",
            "away": "Beta FC",
            "edge": 0.06,
            "odds": 1.82,
            "ml_prob": 0.64,
            "fair_prob": 0.56,
            "market_implied_prob": 0.55,
            "vig_free_implied_prob": 0.54,
            "minimum_acceptable_odds": 1.74,
            "confidence_range_low": 0.58,
            "confidence_range_high": 0.69,
            "lower_bound_passed": True,
            "recommended_market": "double_chance",
            "commence": commence,
            "commence_time": commence,
            "odds_snapshot_age_hours": 1.0,
            "standings_snapshot_age_hours": 3.0,
            "scraped_context": {
                "home_team_name": "Alpha FC",
                "away_team_name": "Beta FC",
                "availability_source": "api_football",
                "availability_fetched_at": datetime.now(timezone.utc).isoformat(),
                "home_lineup_confirmed": 1,
                "away_lineup_confirmed": 1,
                "home_likely_starters_count": 11,
                "away_likely_starters_count": 11,
            },
            "scraped_context_sources": ["api_football"],
            "scraped_context_highlights": ["Availability context fetched from api_football"],
        },
        "substitute_research": ResearchMindDecision(
            research_verdict=ResearchVerdict.AGREE,
            sport="soccer",
            confidence="High",
            main_evidence=("Safer market retained the edge.",),
            data_freshness="verified_fresh",
            sources_checked=("api_football", "odds_snapshot"),
            evidence_status="COMPLETE",
            concrete_info_score=90,
            source_count=2,
            source_quality_summary="strong",
            fixture_verified=True,
            odds_age_minutes=15,
            odds_freshness_status="fresh",
            market_availability_status="available",
            lineup_status="confirmed",
            injury_status="checked_fresh",
            motivation_status="not_required",
            rotation_status="not_required",
            metadata={
                "fixture_verified": True,
                "match_status": "pre_match",
                "critical_missing_evidence": [],
            },
        ),
        "substitute_model": ModelMindDecision(
            model_verdict=ModelVerdict.BET,
            model_probability=0.64,
            market_implied_probability=0.55,
            vig_free_market_probability=0.54,
            fair_odds=1.56,
            minimum_acceptable_odds=1.74,
            current_odds=1.82,
            estimated_edge=0.06,
            confidence_interval=(0.58, 0.69),
            risk_tier="low",
            suggested_market="double_chance",
            parlay_suitability="good_leg",
            reasons=("safer substitute clears the edge bar",),
        ),
    }


def test_substitute_only_publishes_when_enabled(monkeypatch) -> None:
    candidate = _candidate()

    monkeypatch.setattr(committee_integration, "allow_bet_substitutes", lambda: True)
    published, review, suppressed, _ = committee_integration.run_committee_pipeline(
        published=[candidate],
        review=[],
        suppressed=[],
    )

    assert len(published) == 1
    assert review == []
    assert suppressed == []
    assert published[0]["committee_final_decision"] == "BET_SUBSTITUTE"
    assert published[0]["published_from_substitute"] is True
    assert published[0]["team"] == "Alpha FC or Draw"


def test_substitute_stays_off_board_when_disabled(monkeypatch) -> None:
    candidate = _candidate()

    monkeypatch.setattr(committee_integration, "allow_bet_substitutes", lambda: False)
    published, review, suppressed, _ = committee_integration.run_committee_pipeline(
        published=[candidate],
        review=[],
        suppressed=[],
    )

    assert published == []
    assert review == []
    assert len(suppressed) == 1
    assert suppressed[0]["committee_final_decision"] == "BET_SUBSTITUTE"
    assert suppressed[0]["decision_status"] == "NO BET"
