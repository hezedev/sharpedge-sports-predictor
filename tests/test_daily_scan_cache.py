from __future__ import annotations

import json

import daily_scan


def test_fetch_odds_offline_uses_saved_disk_cache(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(daily_scan, "_ODDS_DISK_CACHE_DIR", tmp_path)
    monkeypatch.setattr(daily_scan, "_OFFLINE_ODDS_ONLY", True)
    monkeypatch.setattr(daily_scan, "ODDS_KEY", "test-key")
    daily_scan._odds_cache = {}

    cached_games = [{"id": "match-1", "bookmakers": []}]
    (tmp_path / "tennis_atp_madrid_open.json").write_text(json.dumps(cached_games))

    def _unexpected_live_fetch(*args, **kwargs):
        raise AssertionError("live API should not be called in offline mode")

    monkeypatch.setattr(daily_scan, "_fetch_odds_raw", _unexpected_live_fetch)

    games = daily_scan.fetch_odds("tennis_atp_madrid_open")

    assert len(games) == 1
    assert games[0]["id"] == cached_games[0]["id"]
    assert games[0]["bookmakers"] == cached_games[0]["bookmakers"]
    assert games[0]["_odds_source_status"] in {"offline_disk_cache", "disk_cache"}


def test_prefetch_active_sports_offline_uses_latest_saved_cache(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(daily_scan, "_ODDS_DISK_CACHE_DIR", tmp_path)
    monkeypatch.setattr(daily_scan, "_OFFLINE_ODDS_ONLY", True)
    monkeypatch.setattr(daily_scan, "ODDS_KEY", "test-key")
    daily_scan._odds_cache = {}

    cached_sports = ["soccer_epl", "tennis_atp_madrid_open"]
    (tmp_path / "2026-04-20___active_sports__.json").write_text(json.dumps(cached_sports))

    active_sports = daily_scan._prefetch_active_sports()

    assert active_sports == set(cached_sports)
