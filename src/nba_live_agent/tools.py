"""LangGraph-bindable tools. Each docstring is part of the prompt Gemini sees —
it's what the model reads to decide when and how to call the tool, not just
documentation for humans.
"""

from langchain_core.tools import tool

from nba_live_agent import nba_client
from nba_live_agent.models import (
    BoxScoreResult,
    GameResolution,
    PlayByPlayResult,
)


@tool
def resolve_game(query: str, date: str = "today") -> GameResolution:
    """Resolve a free-text game description to a specific NBA game_id.

    Call this once at the start of a session to figure out which game the
    user is watching or asking about, e.g. query="Lakers vs Celtics" or
    query="BOS". date defaults to "today" (today's live/scheduled slate);
    pass a specific "YYYY-MM-DD" date to resolve a past or future game
    instead — the system prompt tells you today's date, so convert relative
    terms like "yesterday" or "last night" to a concrete date yourself.

    Returns status="ok" with a game_id on a single unambiguous match.
    status="not_found" means no game matched the query; if games were
    scheduled that date anyway, available_games lists all of them — read
    this out to the user as numbered options instead of just saying "not
    found" and stopping. status="ambiguous" means the query matched more
    than one game on that date; candidates lists them the same way — ask
    the user to pick one rather than guessing. status="unsupported_date"
    means the date string couldn't be parsed.
    """
    raw = nba_client._resolve_game_raw(query, date)
    return GameResolution(
        status=raw["status"],
        game_id=raw.get("game_id"),
        home_team=raw.get("home_team"),
        away_team=raw.get("away_team"),
        message=raw.get("message"),
        candidates=raw.get("candidates", []),
        available_games=raw.get("available_games", []),
    )


@tool
def get_play_by_play(game_id: str, period: int | None = None) -> PlayByPlayResult:
    """Get chronological play-by-play events for a game.

    Pass period (1-4, or higher for overtime) to scope to one quarter, e.g.
    to answer "why isn't LeBron scoring this quarter?". Omit period for the
    full game so far. Use this to reconstruct shot attempts/outcomes,
    substitutions, and fouls within a time window.

    If the requested period hasn't been played yet, status will be
    "period_not_played" rather than an empty list of events.
    """
    raw = nba_client._get_play_by_play_raw(game_id, period)
    return PlayByPlayResult(
        status=raw["status"],
        events=raw.get("events", []),
        message=raw.get("message"),
    )


@tool
def get_boxscore(game_id: str) -> BoxScoreResult:
    """Get the current live boxscore snapshot for a game: every player's
    points, assists, rebounds, and shooting line so far.

    Use this when a question needs game-level context beyond one period,
    e.g. "he's 1-for-6 for the game, not just this period."
    """
    raw = nba_client._get_boxscore_raw(game_id)
    return BoxScoreResult(
        status=raw["status"],
        home_team=raw.get("home_team"),
        away_team=raw.get("away_team"),
        home_players=raw.get("home_players", []),
        away_players=raw.get("away_players", []),
        message=raw.get("message"),
    )
