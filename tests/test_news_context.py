from __future__ import annotations

import requests

from src.analysis import news_context


class _FakeResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None


def test_collect_matchup_news_context_uses_google_news_rss_fallback(monkeypatch) -> None:
    monkeypatch.setattr(news_context, "_GOOGLE_NEWS_PAUSED_UNTIL", None)
    monkeypatch.setattr(news_context, "_GOOGLE_NEWS_PAUSE_REASON", "")
    monkeypatch.setattr(news_context, "_search_duckduckgo", lambda *args, **kwargs: ([], "search_provider_paused:403"))

    rss = b"""<?xml version="1.0" encoding="UTF-8"?>
    <rss><channel>
      <item>
        <title>Alpha FC vs Beta FC predicted lineups and team news</title>
        <link>https://www.sportsmole.co.uk/football/alpha-fc/preview</link>
        <description>Team news includes injuries, suspensions and predicted XI notes.</description>
        <source url="https://www.sportsmole.co.uk">sportsmole.co.uk</source>
      </item>
    </channel></rss>
    """
    monkeypatch.setattr(news_context.requests, "get", lambda *args, **kwargs: _FakeResponse(rss))

    payload = news_context.collect_matchup_news_context(
        "soccer",
        "Alpha FC",
        "Beta FC",
        bet="Alpha FC",
        include_channels=False,
    )

    assert payload["sources"] == ["sportsmole.co.uk"]
    assert any("predicted lineups" in item.lower() for item in payload["highlights"])


def test_google_news_rss_pauses_after_timeout(monkeypatch) -> None:
    monkeypatch.setattr(news_context, "_GOOGLE_NEWS_PAUSED_UNTIL", None)
    monkeypatch.setattr(news_context, "_GOOGLE_NEWS_PAUSE_REASON", "")

    calls = {"count": 0}

    def _timeout(*args, **kwargs):
        calls["count"] += 1
        raise requests.exceptions.Timeout("read timed out")

    monkeypatch.setattr(news_context.requests, "get", _timeout)

    items, error = news_context._search_google_news_rss("Alpha Beta team news", limit=2, timeout=1)
    paused_items, paused_error = news_context._search_google_news_rss("Alpha Beta lineups", limit=2, timeout=1)

    assert items == []
    assert "timed out" in error
    assert paused_items == []
    assert paused_error == "google_news_paused:timeout"
    assert calls["count"] == 1


def test_search_context_uses_gdelt_before_google_news(monkeypatch) -> None:
    monkeypatch.delenv("NEWS_API_KEY", raising=False)
    news_context._SEARCH_RESULT_CACHE.clear()
    monkeypatch.setattr(news_context, "_GDELT_PAUSED_UNTIL", None)
    monkeypatch.setattr(news_context, "_GDELT_PAUSE_REASON", "")

    ddg_calls = {"count": 0}
    google_calls = {"count": 0}

    def _ddg(*args, **kwargs):
        ddg_calls["count"] += 1
        return [], "search_provider_paused:403"

    def _google(*args, **kwargs):
        google_calls["count"] += 1
        return [], "should not be needed"

    class _GdeltResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "articles": [
                    {
                        "title": "Alpha FC team news and predicted lineup",
                        "url": "https://example.com/alpha-team-news",
                        "domain": "example.com",
                    }
                ]
            }

    monkeypatch.setattr(news_context.requests, "get", lambda *args, **kwargs: _GdeltResponse())
    monkeypatch.setattr(news_context, "_search_duckduckgo", _ddg)
    monkeypatch.setattr(news_context, "_search_google_news_rss", _google)

    items, error = news_context._search_context("Alpha FC Beta FC lineups", limit=2, timeout=1)

    assert error is None
    assert [item.source for item in items] == ["example.com"]
    assert ddg_calls["count"] == 0
    assert google_calls["count"] == 0


def test_search_context_uses_newsapi_before_scrape_fallbacks(monkeypatch) -> None:
    monkeypatch.setenv("NEWS_API_KEY", "news-key")
    news_context._SEARCH_RESULT_CACHE.clear()
    fallback_calls = {"gdelt": 0, "duck": 0, "google": 0}

    class _NewsAPIResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "articles": [
                    {
                        "title": "Alpha FC team news and injuries",
                        "description": "Projected lineup notes are available.",
                        "url": "https://news.example.com/alpha",
                        "source": {"name": "News Example"},
                    }
                ]
            }

    def _fake_get(*args, **kwargs):
        return _NewsAPIResponse()

    monkeypatch.setattr(news_context.requests, "get", _fake_get)
    monkeypatch.setattr(news_context, "_search_gdelt_doc", lambda *args, **kwargs: fallback_calls.__setitem__("gdelt", 1) or ([], "gdelt"))
    monkeypatch.setattr(news_context, "_search_duckduckgo", lambda *args, **kwargs: fallback_calls.__setitem__("duck", 1) or ([], "duck"))
    monkeypatch.setattr(news_context, "_search_google_news_rss", lambda *args, **kwargs: fallback_calls.__setitem__("google", 1) or ([], "google"))

    items, error = news_context._search_context("Alpha FC Beta FC lineup", limit=2, timeout=1)

    assert error is None
    assert [item.source for item in items] == ["news example"]
    assert fallback_calls == {"gdelt": 0, "duck": 0, "google": 0}


def test_search_context_reuses_successful_results_within_scan(monkeypatch) -> None:
    news_context._SEARCH_RESULT_CACHE.clear()
    calls = {"count": 0}

    def _newsapi(*args, **kwargs):
        calls["count"] += 1
        return [news_context.NewsItem("Team news", "Lineup confirmed", "https://example.com", "example.com")], None

    monkeypatch.setattr(news_context, "_search_newsapi", _newsapi)

    first, _ = news_context._search_context("Alpha FC Beta FC lineup", limit=2, timeout=1)
    second, _ = news_context._search_context("Alpha FC Beta FC lineup", limit=2, timeout=1)

    assert first == second
    assert calls["count"] == 1


def test_soccer_news_channels_include_lineup_and_availability_sources() -> None:
    assert "lineup_context" in news_context._CHANNELS
    assert "availability_context" in news_context._CHANNELS
    assert "rotowire.com" in news_context._CHANNELS["lineup_context"]["sites"]
    assert "transfermarkt.com" in news_context._CHANNELS["availability_context"]["sites"]
