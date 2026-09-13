import logging

import requests

from nba_live_agent import tavily_client
from nba_live_agent.tools import get_general_news


def test_get_general_news_no_key_reports_unavailable(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    res = tavily_client.get_general_news("Lakers", use_cache=False)
    assert res.status == "api_error"
    assert res.items == []
    assert "TAVILY_API_KEY" in res.message


def test_get_general_news_placeholder_key_reports_unavailable(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "your_tavily_api_key_here")
    res = tavily_client.get_general_news("Lakers", use_cache=False)
    assert res.status == "api_error"
    assert res.items == []


def test_get_general_news_empty_query():
    res = tavily_client.get_general_news("   ", use_cache=False)
    assert res.status == "no_results"
    assert res.items == []


def test_get_general_news_caching(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    query = "CelticsTestCache"

    res1 = tavily_client.get_general_news(query, use_cache=True)
    res2 = tavily_client.get_general_news(query, use_cache=True)

    assert res1 is res2


def test_get_general_news_tool_reports_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    res = get_general_news.invoke({"query": "Warriors"})
    assert res.status == "api_error"
    assert res.items == []


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_get_general_news_real_api_returns_items(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        return _FakeResponse(
            {
                "results": [
                    {"title": "Lakers sign player", "url": "https://example.com/1", "content": "..."},
                ]
            }
        )

    monkeypatch.setattr(tavily_client.requests, "post", fake_post)

    res = tavily_client.get_general_news("Lakers injury report", use_cache=False)

    assert res.status == "ok"
    assert len(res.items) == 1
    assert res.items[0].title == "Lakers sign player"
    assert captured["url"] == tavily_client.TAVILY_SEARCH_URL
    assert captured["json"]["api_key"] == "fake-key"
    assert "Lakers injury report" in captured["json"]["query"]


def test_get_general_news_no_results(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
    monkeypatch.setattr(tavily_client.requests, "post", lambda *a, **k: _FakeResponse({"results": []}))

    res = tavily_client.get_general_news("ObscureQueryNoResults", use_cache=False)

    assert res.status == "no_results"
    assert res.items == []


def test_get_general_news_logs_warning_on_api_failure(monkeypatch, caplog):
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key")

    def fake_post(*a, **k):
        raise requests.RequestException("network unreachable")

    monkeypatch.setattr(tavily_client.requests, "post", fake_post)

    with caplog.at_level(logging.WARNING, logger="nba_live_agent.tavily_client"):
        res = tavily_client.get_general_news("NetworkFailureQuery", use_cache=False)

    assert res.status == "api_error"
    assert any("Tavily search failed" in r.message for r in caplog.records)
