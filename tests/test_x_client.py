import logging
from types import SimpleNamespace

from nba_live_agent import x_client
from nba_live_agent.tools import get_x_expert_insights


def test_get_x_insights_no_token_reports_unavailable(monkeypatch):
    monkeypatch.delenv("X_BEARER_TOKEN", raising=False)
    res = x_client.get_x_insights("Lakers", use_cache=False)
    assert res.status == "api_error"
    assert res.is_mock is False
    assert res.posts == []
    assert "X_BEARER_TOKEN" in res.message


def test_get_x_insights_placeholder_token_reports_unavailable(monkeypatch):
    # .env_example ships with a placeholder value; copying it into .env
    # without editing it must behave the same as leaving the var unset.
    monkeypatch.setenv("X_BEARER_TOKEN", "your_x_bearer_token_here")
    res = x_client.get_x_insights("Lakers", use_cache=False)
    assert res.status == "api_error"
    assert res.is_mock is False
    assert res.posts == []


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


def test_get_x_expert_insights_tool_reports_unavailable_without_token(monkeypatch):
    monkeypatch.delenv("X_BEARER_TOKEN", raising=False)
    res = get_x_expert_insights.invoke({"query": "Warriors"})
    assert res.status == "api_error"
    assert res.posts == []


def test_get_x_insights_real_api_scopes_to_expert_accounts(monkeypatch):
    monkeypatch.setenv("X_BEARER_TOKEN", "fake-token")
    captured = {}

    class FakeClient:
        def __init__(self, bearer_token):
            captured["bearer_token"] = bearer_token

        def search_recent_tweets(self, query, max_results, tweet_fields):
            captured["query"] = query
            return SimpleNamespace(data=None)

    monkeypatch.setattr(x_client.tweepy, "Client", FakeClient)

    x_client.get_x_insights("Lakers", use_cache=False)

    assert captured["bearer_token"] == "fake-token"
    # Every curated handle must be present as a `from:` filter, not just a
    # generic verified-account search — this is what makes "commentary from
    # a curated list of NBA analysts" an accurate claim, not just "any
    # verified account that happens to mention the query".
    for account in x_client.EXPERT_ACCOUNTS:
        assert f"from:{account['handle']}" in captured["query"]


def test_get_x_insights_real_api_maps_tweet_fields(monkeypatch):
    monkeypatch.setenv("X_BEARER_TOKEN", "fake-token")

    fake_tweet = SimpleNamespace(id=123, text="Great adjustment tonight.", author_id=456, created_at="2026-01-16")

    class FakeClient:
        def __init__(self, bearer_token):
            pass

        def search_recent_tweets(self, query, max_results, tweet_fields):
            return SimpleNamespace(data=[fake_tweet])

    monkeypatch.setattr(x_client.tweepy, "Client", FakeClient)

    res = x_client.get_x_insights("Lakers", use_cache=False)

    assert res.status == "ok"
    assert len(res.posts) == 1
    post = res.posts[0]
    assert post.content == "Great adjustment tonight."
    assert post.handle == "456"
    assert post.url == "https://x.com/i/web/status/123"


def test_get_x_insights_logs_warning_on_real_api_failure(monkeypatch, caplog):
    monkeypatch.setenv("X_BEARER_TOKEN", "fake-token")

    class FailingClient:
        def __init__(self, bearer_token):
            pass

        def search_recent_tweets(self, query, max_results, tweet_fields):
            raise OSError("network unreachable")

    monkeypatch.setattr(x_client.tweepy, "Client", FailingClient)

    with caplog.at_level(logging.WARNING, logger="nba_live_agent.x_client"):
        res = x_client.get_x_insights("NetworkFailureQuery", use_cache=False)

    assert res.status == "ok"
    assert res.is_mock is True
    assert any(
        r.levelno == logging.WARNING and "falling back to mock data" in r.message
        for r in caplog.records
    )
