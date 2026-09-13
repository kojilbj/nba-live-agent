"""Tavily-backed general web search for NBA news the other tools don't
cover -- injury reports, roster/trade moves, coaching changes, and other
breaking news that isn't in the box-score APIs (nba_client) or the curated
expert-account search (x_client). Unlike x_client, a failed or unconfigured
request never falls back to fabricated results: reporting fake news as if
it were real is a much worse failure mode than reporting none.
"""

import logging
import time

import requests

from nba_live_agent.env_config import get_configured_env
from nba_live_agent.models import NewsItem, NewsResult

logger = logging.getLogger(__name__)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"

# Simple in-memory cache: key -> (timestamp, NewsResult)
_CACHE: dict[str, tuple[float, NewsResult]] = {}
CACHE_TTL_SECONDS = 120.0


def get_general_news(query: str, use_cache: bool = True) -> NewsResult:
    """Search the web for NBA news relevant to query via Tavily.
    Returns status="api_error" when TAVILY_API_KEY isn't configured -- this
    tool is unavailable without a real API key, so callers shouldn't treat
    the message field as a real search result. Caching is applied with a
    default 2-minute TTL.
    """
    clean_query = query.strip()
    if not clean_query:
        return NewsResult(status="no_results", query=query, items=[], message="Query was empty.")

    cache_key = clean_query.lower()
    now = time.time()

    if use_cache and cache_key in _CACHE:
        cached_time, cached_result = _CACHE[cache_key]
        if now - cached_time < CACHE_TTL_SECONDS:
            return cached_result

    api_key = get_configured_env("TAVILY_API_KEY")

    if not api_key:
        logger.warning("TAVILY_API_KEY is not set; get_general_news is unavailable")
        result = NewsResult(
            status="api_error",
            query=clean_query,
            items=[],
            message="TAVILY_API_KEY is not set, so this tool is unavailable. Configure a Tavily API key to use it.",
        )
        _CACHE[cache_key] = (now, result)
        return result

    try:
        response = requests.post(
            TAVILY_SEARCH_URL,
            json={
                "api_key": api_key,
                "query": f"NBA {clean_query}",
                "topic": "news",
                "search_depth": "basic",
                "max_results": 5,
            },
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as e:
        logger.warning("Tavily search failed for %r: %s", clean_query, e)
        result = NewsResult(
            status="api_error",
            query=clean_query,
            items=[],
            message=f"Tavily search failed: {e}",
        )
        _CACHE[cache_key] = (now, result)
        return result

    items = [
        NewsItem(title=r.get("title", ""), url=r.get("url", ""), content=r.get("content", ""))
        for r in data.get("results", [])
    ]
    result = (
        NewsResult(status="ok", query=clean_query, items=items)
        if items
        else NewsResult(status="no_results", query=clean_query, items=[], message="No results found.")
    )
    _CACHE[cache_key] = (now, result)
    return result
