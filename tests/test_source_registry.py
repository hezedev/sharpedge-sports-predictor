from __future__ import annotations

from src.data.source_registry import get_source_registry, source_status_summary


def test_source_registry_marks_keyless_nhl_api_configured() -> None:
    providers = {provider.key: provider for provider in get_source_registry(env={})}

    assert providers["nhl_public_api"].configured is True
    assert providers["nhl_public_api"].missing_env_vars == ()


def test_source_registry_prefers_direct_api_sports_for_soccer() -> None:
    summary = source_status_summary(env={"API_SPORTS_KEY": "direct-key"})
    soccer_sources = [item["key"] for item in summary["by_sport"]["soccer"]]

    assert "api_sports_football" in soccer_sources
    assert "soccer" not in summary["missing_critical"]
    assert any(item["key"] == "api_sports_football" and item["configured"] for item in summary["providers"])


def test_source_registry_reports_missing_critical_sources() -> None:
    summary = source_status_summary(env={})

    assert "soccer" in summary["missing_critical"]
    assert "basketball" in summary["missing_critical"]
    assert "nhl" not in summary["missing_critical"]
    assert any("API_SPORTS_KEY" in item for item in summary["recommendations"])
