import logging

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


def test_with_retry_logs_warning_on_each_failed_attempt(caplog):
    call_count = 0

    def flaky_func():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ValueError("Temporary failure")
        return "success"

    with caplog.at_level(logging.WARNING, logger="nba_live_agent.nba_client"):
        nba_client._with_retry(flaky_func, retries=3, initial_delay=0.01)

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
    assert "Temporary failure" in caplog.records[0].message


def test_with_retry_logs_warning_before_raising_on_exhaustion(caplog):
    def always_fails():
        raise RuntimeError("Persistent error")

    with caplog.at_level(logging.WARNING, logger="nba_live_agent.nba_client"):
        with pytest.raises(RuntimeError, match="Persistent error"):
            nba_client._with_retry(always_fails, retries=2, initial_delay=0.01)

    assert len(caplog.records) == 2
    assert all("Persistent error" in r.message for r in caplog.records)


def test_custom_headers_configured():
    from nba_api.stats.library.http import STATS_HEADERS

    assert "User-Agent" in STATS_HEADERS
    assert "Mozilla/5.0" in STATS_HEADERS["User-Agent"]
    assert STATS_HEADERS.get("Referer") == "https://www.nba.com/"
