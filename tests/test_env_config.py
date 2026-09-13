import pytest

from nba_live_agent.env_config import get_configured_env, require_google_api_key


def test_get_configured_env_unset_returns_none(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert get_configured_env("GOOGLE_API_KEY") is None


def test_get_configured_env_placeholder_returns_none(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "your_google_api_key_here")
    assert get_configured_env("GOOGLE_API_KEY") is None


def test_get_configured_env_real_value_returned(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "a-real-key")
    assert get_configured_env("GOOGLE_API_KEY") == "a-real-key"


def test_require_google_api_key_exits_when_unset(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        require_google_api_key()


def test_require_google_api_key_exits_when_placeholder(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "your_google_api_key_here")
    with pytest.raises(SystemExit):
        require_google_api_key()


def test_require_google_api_key_passes_when_configured(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "a-real-key")
    require_google_api_key()
