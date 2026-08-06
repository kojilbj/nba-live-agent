import pytest
from nba_live_agent import nba_client


def test_with_retry_success():
    call_count = 0

    def flaky_func():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ValueError("Temporary failure")
        return "success"

    result = nba_client._with_retry(flaky_func, retries=3, initial_delay=0.01)
    assert result == "success"
    assert call_count == 2


def test_with_retry_failure():
    def always_fails():
        raise RuntimeError("Persistent error")

    with pytest.raises(RuntimeError, match="Persistent error"):
        nba_client._with_retry(always_fails, retries=2, initial_delay=0.01)


def test_custom_headers_configured():
    from nba_api.stats.library.http import STATS_HEADERS

    assert "User-Agent" in STATS_HEADERS
    assert "Mozilla/5.0" in STATS_HEADERS["User-Agent"]
    assert STATS_HEADERS.get("Referer") == "https://www.nba.com/"
