import logging

from nba_live_agent import x_client
from nba_live_agent.tools import get_x_expert_insights


def test_get_x_insights_mock(monkeypatch):
    monkeypatch.delenv("X_BEARER_TOKEN", raising=False)
    res = x_client.get_x_insights("Lakers", use_cache=False)
    assert res.status == "ok"
    assert res.is_mock is True
    assert len(res.posts) > 0
    assert any("stevejones20" in post.handle for post in res.posts)
    assert any("佐々木クリス" in post.author for post in res.posts)


def test_get_x_insights_empty_query():
    res = x_client.get_x_insights("   ", use_cache=False)
    assert res.status == "no_posts_found"
    assert len(res.posts) == 0


def test_get_x_insights_caching(monkeypatch):
    monkeypatch.delenv("X_BEARER_TOKEN", raising=False)
    query = "CelticsTestCache"

    # First call - fresh mock
    res1 = x_client.get_x_insights(query, use_cache=True)

    # Second call - should return cached instance
    res2 = x_client.get_x_insights(query, use_cache=True)

    assert res1 is res2


def test_get_x_expert_insights_tool(monkeypatch):
    monkeypatch.delenv("X_BEARER_TOKEN", raising=False)
    res = get_x_expert_insights.invoke({"query": "Warriors"})
    assert res.status == "ok"
    assert len(res.posts) > 0


def test_get_x_insights_logs_warning_on_real_api_failure(monkeypatch, caplog):
    monkeypatch.setenv("X_BEARER_TOKEN", "fake-token")
    monkeypatch.setattr(
        x_client.urllib.request,
        "urlopen",
        lambda *a, **k: (_ for _ in ()).throw(OSError("network unreachable")),
    )

    with caplog.at_level(logging.WARNING, logger="nba_live_agent.x_client"):
        res = x_client.get_x_insights("NetworkFailureQuery", use_cache=False)

    assert res.status == "ok"
    assert res.is_mock is True
    assert any(
        r.levelno == logging.WARNING and "falling back to mock data" in r.message
        for r in caplog.records
    )
