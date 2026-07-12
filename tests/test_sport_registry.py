from src.utils.sport_registry import get_capability_profile


def test_promoted_soccer_leagues_are_now_publishable() -> None:
    for league_key in (
        "soccer_japan_j_league",
        "soccer_saudi_arabia_pro_league",
        "soccer_conmebol_copa_libertadores",
        "soccer_conmebol_copa_sudamericana",
        "soccer_uefa_europa_conference_league",
        "soccer_uefa_europa_league",
    ):
        profile = get_capability_profile(sport="soccer", sport_key=league_key)
        assert profile.model_backed is True
        assert profile.publishable is True
        assert profile.review_only is False
        assert profile.launch_label == "Production"


def test_mls_remains_review_only_until_its_coverage_is_promoted() -> None:
    profile = get_capability_profile(sport="soccer", sport_key="soccer_usa_mls")
    assert profile.model_backed is False
    assert profile.publishable is False
    assert profile.review_only is True
    assert profile.launch_label == "Review"


def test_world_cup_is_scanable_but_review_only() -> None:
    profile = get_capability_profile(sport="soccer", sport_key="soccer_fifa_world_cup")
    assert profile.scanable is True
    assert profile.model_backed is False
    assert profile.publishable is False
    assert profile.review_only is True
    assert profile.reasoning_supported is True
    assert profile.launch_label == "Review"


def test_discovered_soccer_league_is_review_only_not_generic_production() -> None:
    profile = get_capability_profile(sport="soccer", sport_key="soccer_chile_campeonato")
    assert profile.scanable is True
    assert profile.model_backed is False
    assert profile.publishable is False
    assert profile.review_only is True
    assert profile.launch_label == "Review"


def test_wta_is_model_backed_and_now_publishable() -> None:
    profile = get_capability_profile(sport="tennis_wta")
    assert profile.model_backed is True
    assert profile.publishable is True
    assert profile.review_only is False
    assert profile.reasoning_supported is True
    assert profile.launch_label == "Production"
    assert "calibrator" in profile.launch_note.lower()
