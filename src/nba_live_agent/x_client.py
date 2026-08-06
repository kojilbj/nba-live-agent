import json
import logging
import os
import time
import urllib.request

from nba_live_agent.models import XInsightsResult, XPost

logger = logging.getLogger(__name__)

EXPERT_ACCOUNTS = [
    {"handle": "stevejones20", "name": "Steve Jones Jr."},
    {"handle": "NekiasNBA", "name": "Nekias Duncan"},
    {"handle": "ThinkingBBall", "name": "Thinking Basketball"},
    {"handle": "ChrisSasaki", "name": "佐々木クリス"},
    {"handle": "kirkgoldsberry", "name": "Kirk Goldsberry"},
    {"handle": "ZachLowe_NBA", "name": "Zach Lowe"},
    {"handle": "MoDakhil_NBA", "name": "Mo Dakhil"},
    {"handle": "bencfalk", "name": "Ben Falk (Cleaning The Glass)"},
    {"handle": "CaitlinCooperNBA", "name": "Caitlin Cooper"},
    {"handle": "StephNoh", "name": "Steph Noh"},
]

# Simple in-memory cache: key -> (timestamp, XInsightsResult)
_CACHE: dict[str, tuple[float, XInsightsResult]] = {}
CACHE_TTL_SECONDS = 120.0


def _generate_mock_posts(query: str) -> list[XPost]:
    posts: list[XPost] = []

    # Mock post 1: Steve Jones Jr. (Tactical Adjustment)
    posts.append(
        XPost(
            handle="stevejones20",
            author="Steve Jones Jr.",
            content=(
                f"Noticed an immediate adjustment for {query.title()}: they stopped dropping "
                "on high pick-and-rolls and switched aggressively on ball screens to take away open 3s."
            ),
            timestamp="10m ago",
            url="https://x.com/stevejones20/status/1000000000000000001",
        )
    )

    # Mock post 2: Chris Sasaki (Japanese Tactical Analysis)
    posts.append(
        XPost(
            handle="ChrisSasaki",
            author="佐々木クリス",
            content=(
                f"{query.title()}の今日のセットオフェンス、相手のヘッジディフェンスに対して"
                "ショートローラー経由でコーナースリーを徹底的に突く修正が見事ですね。"
                "スペーシングが非常に効いています。"
            ),
            timestamp="15m ago",
            url="https://x.com/ChrisSasaki/status/1000000000000000002",
        )
    )

    # Mock post 3: Thinking Basketball (Data & Efficiency)
    posts.append(
        XPost(
            handle="ThinkingBBall",
            author="Thinking Basketball",
            content=(
                f"When {query.title()} attacks in transition, their rim frequency jumps to 45%. "
                "In half-court sets against drop coverage, efficiency dips significantly without secondary playmaking."
            ),
            timestamp="30m ago",
            url="https://x.com/ThinkingBBall/status/1000000000000000003",
        )
    )

    # Mock post 4: Nekias Duncan (Matchup / Defense)
    posts.append(
        XPost(
            handle="NekiasNBA",
            author="Nekias Duncan",
            content=(
                f"Great defensive rotations on {query.title()} tonight. Watch how the weak-side helper tags "
                "the roller before recovering to the shooter. Pure textbook execution."
            ),
            timestamp="45m ago",
            url="https://x.com/NekiasNBA/status/1000000000000000004",
        )
    )

    return posts


def get_x_insights(query: str, use_cache: bool = True) -> XInsightsResult:
    """Fetch expert posts on X relevant to the query (team or player).
    Returns status="api_error" when X_BEARER_TOKEN isn't configured — this
    tool is unavailable without a real API key, so callers shouldn't be
    handed simulated posts as if they were genuine expert commentary.
    Mock data is only used as a fallback when a configured request fails.
    Caching is applied with a default 2-minute TTL.
    """
    clean_query = query.strip()
    if not clean_query:
        return XInsightsResult(
            status="no_posts_found",
            query=query,
            posts=[],
            is_mock=True,
            message="Query was empty.",
        )

    cache_key = clean_query.lower()
    now = time.time()

    if use_cache and cache_key in _CACHE:
        cached_time, cached_result = _CACHE[cache_key]
        if now - cached_time < CACHE_TTL_SECONDS:
            return cached_result

    bearer_token = os.getenv("X_BEARER_TOKEN")

    if not bearer_token:
        logger.warning("X_BEARER_TOKEN is not set; get_x_expert_insights is unavailable")
        result = XInsightsResult(
            status="api_error",
            query=clean_query,
            posts=[],
            is_mock=False,
            message="X_BEARER_TOKEN is not set, so this tool is unavailable. Configure an X API key to use it.",
        )
        _CACHE[cache_key] = (now, result)
        return result

    # Real X API v2 search endpoint call when token is available
    try:
        encoded_query = urllib.parse.quote(f"{clean_query} is:verified")
        endpoint = "https://api.twitter.com/2/tweets/search/recent"
        url = f"{endpoint}?query={encoded_query}&max_results=10&tweet.fields=created_at,author_id"
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {bearer_token}",
                "User-Agent": "nba-live-agent/1.0",
            },
        )

        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            raw_tweets = data.get("data", [])
            posts = [
                XPost(
                    handle=tweet.get("author_id", "unknown"),
                    author="X Expert",
                    content=tweet.get("text", ""),
                    timestamp=tweet.get("created_at"),
                    url=f"https://x.com/i/web/status/{tweet.get('id')}",
                )
                for tweet in raw_tweets
            ]
            result = XInsightsResult(
                status="ok" if posts else "no_posts_found",
                query=clean_query,
                posts=posts,
                is_mock=False,
            )
            _CACHE[cache_key] = (now, result)
            return result
    except Exception as e:
        logger.warning("X API request failed, falling back to mock data: %s", e)
        mock_posts = _generate_mock_posts(clean_query)
        result = XInsightsResult(
            status="ok",
            query=clean_query,
            posts=mock_posts,
            is_mock=True,
            message=f"X API request failed ({e}); fell back to simulated expert insights.",
        )
        _CACHE[cache_key] = (now, result)
        return result
