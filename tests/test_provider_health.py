from __future__ import annotations

from src.data import provider_health


class _Response:
    status_code = 200
    headers = {"x-ratelimit-requests-remaining": "4"}


def test_provider_health_records_and_flags_low_quota(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(provider_health, "_HEALTH_PATH", tmp_path / "provider_health.json")
    monkeypatch.setenv("SOURCE_MIN_REMAINING", "5")

    provider_health.record_provider_response("api_sports_football", _Response())

    assert provider_health.provider_quota_low("api_sports_football") is True
    snapshot = provider_health.provider_health_snapshot()
    assert "api_sports_football" in snapshot["low_quota_providers"]


def test_provider_health_unknown_provider_is_not_low(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(provider_health, "_HEALTH_PATH", tmp_path / "provider_health.json")

    assert provider_health.provider_quota_low("api_sports_basketball") is False
