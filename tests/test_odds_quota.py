"""Tests for shared Odds API quota helpers."""

from src.utils import odds_quota


def test_parse_odds_api_keys_from_env_supports_odds_api_key_only() -> None:
    parsed = odds_quota.parse_odds_api_keys_from_env({"ODDS_API_KEY": "primary-key-11111111"})

    assert parsed["keys"] == ["primary-key-11111111"]
    assert parsed["fingerprints"] == ["11111111"]
    assert parsed["excluded"] == []


def test_parse_odds_api_keys_from_env_supports_odds_api_keys_only() -> None:
    parsed = odds_quota.parse_odds_api_keys_from_env({"ODDS_API_KEYS": "alpha-key-11111111,beta-key-22222222"})

    assert parsed["keys"] == ["alpha-key-11111111", "beta-key-22222222"]
    assert parsed["fingerprints"] == ["11111111", "22222222"]


def test_parse_odds_api_keys_from_env_supports_both_sources() -> None:
    parsed = odds_quota.parse_odds_api_keys_from_env({
        "ODDS_API_KEY": "primary-key-11111111",
        "ODDS_API_KEYS": "primary-key-11111111,backup-key-22222222",
    })

    assert parsed["keys"] == ["primary-key-11111111", "backup-key-22222222"]
    assert parsed["fingerprints"] == ["11111111", "22222222"]
    assert any(item["reason"] == "duplicate" and item["fingerprint"] == "11111111" for item in parsed["excluded"])


def test_parse_odds_api_keys_from_env_strips_quotes_and_spaces() -> None:
    parsed = odds_quota.parse_odds_api_keys_from_env({
        "ODDS_API_KEY": "  'primary-key-11111111'  ",
        "ODDS_API_KEYS": ' "backup-key-22222222, gamma-key-33333333" ',
    })

    assert parsed["keys"] == [
        "primary-key-11111111",
        "backup-key-22222222",
        "gamma-key-33333333",
    ]
    assert parsed["fingerprints"] == ["11111111", "22222222", "33333333"]


def test_parse_odds_api_keys_from_env_ignores_invalid_or_empty_values() -> None:
    parsed = odds_quota.parse_odds_api_keys_from_env({
        "ODDS_API_KEY": '""',
        "ODDS_API_KEYS": "good-key-11111111, , bad key , short, good-key-22222222",
    })

    assert parsed["keys"] == ["good-key-11111111", "good-key-22222222"]
    assert parsed["fingerprints"] == ["11111111", "22222222"]
    assert any(item["reason"] == "invalid_format" and item["fingerprint"] == "bad key"[-8:] for item in parsed["excluded"])
    assert any(item["reason"] == "invalid_format" and item["fingerprint"] == "short" for item in parsed["excluded"])
    assert any(item["reason"] == "empty" for item in parsed["excluded"])


def test_api_key_fingerprint_uses_normalized_last_eight() -> None:
    assert odds_quota.api_key_fingerprint("  'quoted-key-12345678'  ") == "12345678"


def test_save_odds_api_usage_tracks_daily_and_total(tmp_path, monkeypatch) -> None:
    usage_path = tmp_path / "api_usage.json"
    monkeypatch.setattr(odds_quota, "USAGE_FILE", usage_path)

    odds_quota.save_odds_api_usage(api_key="abcdef12345678", remaining=490, used_total=10)
    odds_quota.save_odds_api_usage(api_key="abcdef12345678", remaining=488, used_total=12)

    usage = odds_quota.load_odds_api_usage()
    assert usage["odds_remaining"] == 488
    assert usage["odds_requests_used_total"] == 12
    assert usage["odds_requests_used_today"] == 2


def test_get_odds_budget_status_computes_daily_allowance(tmp_path, monkeypatch) -> None:
    usage_path = tmp_path / "api_usage.json"
    monkeypatch.setattr(odds_quota, "USAGE_FILE", usage_path)
    odds_quota.save_odds_api_usage(api_key="abcdef12345678", remaining=130, used_total=370)

    status = odds_quota.get_odds_budget_status(
        "abcdef12345678",
        monthly_limit=500,
        reserve=30,
    )
    assert status["remaining"] == 130
    assert status["remaining_after_reserve"] == 100
    assert status["daily_allowance"] is not None
