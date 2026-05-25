"""Lightweight matchup news/context collector for manual analysis."""

from __future__ import annotations

import html
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import requests

logger = logging.getLogger(__name__)

_SEARCH_PROVIDER_PAUSED_UNTIL: datetime | None = None
_SEARCH_PROVIDER_PAUSE_REASON: str = ""
_GDELT_PAUSED_UNTIL: datetime | None = None
_GDELT_PAUSE_REASON: str = ""
_GOOGLE_NEWS_PAUSED_UNTIL: datetime | None = None
_GOOGLE_NEWS_PAUSE_REASON: str = ""


_SIGNAL_TERMS = (
    "injury", "injured", "out", "questionable", "doubtful", "probable",
    "lineup", "starter", "starting", "rotation", "rest", "fatigue",
    "weather", "suspension", "suspended", "preview", "team news",
)

_CHANNELS = {
    "official_team_context": {
        "label": "Official Team Context",
        "trust": "official",
        "query_terms": "\"official\" team news injuries lineup squad availability",
        "sites": [],
    },
    "official_league_context": {
        "label": "Official League Context",
        "trust": "official",
        "query_terms": "\"official\" match preview disciplinary suspensions round preview",
        "sites": [],
    },
    "preview_context": {
        "label": "Preview Context",
        "trust": "context",
        "query_terms": "preview prediction team news lineup injuries predicted XI",
        "sites": [
            "sportsmole.co.uk",
            "onefootball.com",
            "thestatszone.com",
            "covers.com",
            "rotowire.com",
            "football365.com",
            "90min.com",
        ],
    },
    "lineup_context": {
        "label": "Lineup Context",
        "trust": "context",
        "query_terms": "predicted lineups probable lineups starting XI team news",
        "sites": ["whoscored.com", "fotmob.com", "rotowire.com", "sportsmole.co.uk"],
    },
    "availability_context": {
        "label": "Availability Context",
        "trust": "context",
        "query_terms": "injuries suspensions unavailable doubtful team news",
        "sites": ["transfermarkt.com", "rotowire.com", "sportsmole.co.uk", "90min.com"],
    },
    "espn_context": {
        "label": "ESPN Context",
        "trust": "context",
        "query_terms": "ESPN preview injuries lineup schedule",
        "sites": ["espn.com"],
    },
    "flashscore_context": {
        "label": "Flashscore Context",
        "trust": "fallback_context",
        "query_terms": "Flashscore h2h form fixtures results",
        "sites": ["flashscore.com"],
    },
    "reddit_community": {
        "label": "Reddit Community Signal",
        "trust": "community_unverified",
        "query_terms": "reddit injury lineup team news fans preview",
        "sites": ["reddit.com"],
    },
}


@dataclass
class NewsItem:
    title: str
    snippet: str
    url: str
    source: str


def _clean_text(value: str) -> str:
    value = html.unescape(re.sub(r"<[^>]+>", " ", value or ""))
    return " ".join(value.split())


def _clean_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.path.startswith("/l/"):
        uddg = parse_qs(parsed.query).get("uddg", [""])[0]
        if uddg:
            return unquote(uddg)
    return value


def _source_name(url: str) -> str:
    host = urlparse(url).netloc.lower().replace("www.", "")
    return host or "web"


def _parse_duckduckgo_html(markup: str, limit: int) -> list[NewsItem]:
    items: list[NewsItem] = []
    blocks = re.split(r'<div[^>]+class="[^"]*result[^"]*"[^>]*>', markup)
    for block in blocks[1:]:
        link_match = re.search(
            r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            block,
            re.S,
        )
        if not link_match:
            continue
        url = _clean_url(html.unescape(link_match.group(1)))
        title = _clean_text(link_match.group(2))
        snippet_match = re.search(
            r'<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>|'
            r'<div[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</div>',
            block,
            re.S,
        )
        snippet = _clean_text((snippet_match.group(1) or snippet_match.group(2)) if snippet_match else "")
        if not title or not url:
            continue
        items.append(NewsItem(title=title, snippet=snippet, url=url, source=_source_name(url)))
        if len(items) >= limit:
            break
    return items


def _search_duckduckgo(query: str, limit: int, timeout: int) -> tuple[list[NewsItem], str | None]:
    global _SEARCH_PROVIDER_PAUSED_UNTIL, _SEARCH_PROVIDER_PAUSE_REASON
    now = datetime.now(timezone.utc)
    if _SEARCH_PROVIDER_PAUSED_UNTIL and now < _SEARCH_PROVIDER_PAUSED_UNTIL:
        return [], _SEARCH_PROVIDER_PAUSE_REASON or "search_provider_paused"
    url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 SportsPredictor/1.0"},
            timeout=timeout,
        )
        response.raise_for_status()
        return _parse_duckduckgo_html(response.text, limit=limit), None
    except Exception as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code in {403, 429}:
            _SEARCH_PROVIDER_PAUSED_UNTIL = now + timedelta(minutes=20)
            _SEARCH_PROVIDER_PAUSE_REASON = f"search_provider_paused:{status_code}"
        logger.info("Fresh matchup news context unavailable: %s", exc)
        return [], str(exc)


def _search_google_news_rss(query: str, limit: int, timeout: int) -> tuple[list[NewsItem], str | None]:
    global _GOOGLE_NEWS_PAUSED_UNTIL, _GOOGLE_NEWS_PAUSE_REASON
    now = datetime.now(timezone.utc)
    if _GOOGLE_NEWS_PAUSED_UNTIL and now < _GOOGLE_NEWS_PAUSED_UNTIL:
        return [], _GOOGLE_NEWS_PAUSE_REASON or "google_news_paused"
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 SportsPredictor/1.0"},
            timeout=timeout,
        )
        response.raise_for_status()
        root = ET.fromstring(response.content)
        items: list[NewsItem] = []
        for item in root.findall(".//item"):
            title = _clean_text(item.findtext("title") or "")
            link = _clean_url(item.findtext("link") or "")
            snippet = _clean_text(item.findtext("description") or "")
            source_node = item.find("source")
            source = _clean_text(source_node.text if source_node is not None and source_node.text else "")
            if not source:
                source = _source_name(link)
            if title and link:
                items.append(NewsItem(title=title, snippet=snippet, url=link, source=source.lower().replace("www.", "")))
            if len(items) >= limit:
                break
        return items, None
    except Exception as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        is_timeout = isinstance(exc, requests.exceptions.Timeout) or "timed out" in str(exc).lower()
        if status_code in {403, 429} or is_timeout:
            _GOOGLE_NEWS_PAUSED_UNTIL = now + timedelta(minutes=20)
            reason = f"google_news_paused:{status_code}" if status_code else "google_news_paused:timeout"
            _GOOGLE_NEWS_PAUSE_REASON = reason
        logger.info("Fresh Google News context unavailable: %s", exc)
        return [], str(exc)


def _search_gdelt_doc(query: str, limit: int, timeout: int) -> tuple[list[NewsItem], str | None]:
    global _GDELT_PAUSED_UNTIL, _GDELT_PAUSE_REASON
    now = datetime.now(timezone.utc)
    if _GDELT_PAUSED_UNTIL and now < _GDELT_PAUSED_UNTIL:
        return [], _GDELT_PAUSE_REASON or "gdelt_paused"
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    try:
        response = requests.get(
            url,
            params={
                "query": query,
                "mode": "artlist",
                "format": "json",
                "maxrecords": max(1, min(limit, 10)),
                "timespan": "7d",
                "sort": "hybridrel",
            },
            headers={"User-Agent": "Mozilla/5.0 SportsPredictor/1.0"},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        articles = payload.get("articles") or payload.get("items") or []
        items: list[NewsItem] = []
        for article in articles if isinstance(articles, list) else []:
            if not isinstance(article, dict):
                continue
            title = _clean_text(str(article.get("title") or ""))
            link = _clean_url(str(article.get("url") or article.get("link") or ""))
            source = _clean_text(str(article.get("domain") or article.get("source") or ""))
            if not source:
                source = _source_name(link)
            if title and link:
                items.append(NewsItem(title=title, snippet="", url=link, source=source.lower().replace("www.", "")))
            if len(items) >= limit:
                break
        return items, None
    except Exception as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        is_timeout = isinstance(exc, requests.exceptions.Timeout) or "timed out" in str(exc).lower()
        if status_code in {403, 429} or is_timeout:
            _GDELT_PAUSED_UNTIL = now + timedelta(minutes=20)
            reason = f"gdelt_paused:{status_code}" if status_code else "gdelt_paused:timeout"
            _GDELT_PAUSE_REASON = reason
        logger.info("Fresh GDELT context unavailable: %s", exc)
        return [], str(exc)


def _search_context(query: str, limit: int, timeout: int) -> tuple[list[NewsItem], str | None]:
    gdelt_items, gdelt_error = _search_gdelt_doc(query, limit=limit, timeout=timeout)
    if gdelt_items:
        return gdelt_items, None
    items, error = _search_duckduckgo(query, limit=limit, timeout=timeout)
    if items:
        return items, error
    fallback_items, fallback_error = _search_google_news_rss(query, limit=limit, timeout=timeout)
    if fallback_items:
        return fallback_items, None
    errors = []
    if error:
        errors.append(error)
    if gdelt_error:
        errors.append(f"gdelt:{gdelt_error}")
    if fallback_error:
        errors.append(f"google_news:{fallback_error}")
    return [], "; ".join(errors) if errors else None


def _highlights_from_items(items: list[NewsItem], limit: int) -> list[str]:
    highlights: list[str] = []
    for item in items:
        text = f"{item.title}. {item.snippet}".strip()
        lowered = text.lower()
        if any(term in lowered for term in _SIGNAL_TERMS):
            highlights.append(text[:280])
    if not highlights and items:
        highlights = [f"{item.title}. {item.snippet}".strip()[:280] for item in items[:2]]
    return highlights[:limit]


def _dedupe_items(items: list[NewsItem]) -> list[NewsItem]:
    seen: set[str] = set()
    deduped: list[NewsItem] = []
    for item in items:
        key = item.url or f"{item.source}:{item.title}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _channel_query(base_query: str, channel: dict[str, Any]) -> str:
    sites = channel.get("sites") or []
    site_clause = " OR ".join(f"site:{site}" for site in sites)
    return " ".join(part for part in [base_query, channel.get("query_terms", ""), site_clause] if part)


def collect_matchup_news_context(
    sport: str,
    home_team: str,
    away_team: str,
    bet: str = "",
    limit: int = 4,
    timeout: int = 8,
    include_channels: bool = True,
) -> dict[str, Any]:
    """
    Fetch a compact web-news context packet for one selected matchup.

    This is intentionally bounded: one search request, a few snippets, no full-page crawling.
    """
    query = " ".join(
        part for part in [
            home_team,
            away_team,
            sport,
            "preview injuries lineup team news",
            bet,
        ] if part
    )
    if not query.strip():
        return {"query": "", "items": [], "highlights": [], "sources": [], "warnings": ["No matchup query available."]}

    items, error = _search_context(query, limit=limit, timeout=timeout)
    warnings = [f"Fresh web context unavailable: {error}"] if error else []

    channels: dict[str, Any] = {}
    channel_items: list[NewsItem] = []
    if include_channels:
        per_channel_limit = max(2, min(3, limit))
        channel_timeout = max(1, min(timeout, 2))
        for channel_key, channel in _CHANNELS.items():
            channel_query = _channel_query(query, channel)
            found, channel_error = _search_context(
                channel_query,
                limit=per_channel_limit,
                timeout=channel_timeout,
            )
            channel_items.extend(found)
            channels[channel_key] = {
                "label": channel["label"],
                "trust": channel["trust"],
                "query": channel_query,
                "items": [asdict(item) for item in found],
                "highlights": _highlights_from_items(found, limit=per_channel_limit),
                "sources": sorted({item.source for item in found}),
                "warnings": [channel_error] if channel_error else ([] if found else ["No snippets found for this channel."]),
            }

    items = _dedupe_items(items + channel_items)
    highlights = _highlights_from_items(items, limit=limit)

    sources = []
    for item in items:
        if item.source not in sources:
            sources.append(item.source)

    return {
        "query": query,
        "items": [asdict(item) for item in items],
        "highlights": highlights[:limit],
        "sources": sources,
        "channels": channels,
        "warnings": warnings or ([] if items else ["No fresh matchup news snippets found."]),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
