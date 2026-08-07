"""LangGraph-bindable tools. Each docstring is part of the prompt Gemini sees —
it's what the model reads to decide when and how to call the tool, not just
documentation for humans.
"""

from langchain_core.tools import tool

from nba_live_agent import nba_client, x_client
from nba_live_agent.models import (
    BoxScoreResult,
    GameResolution,
    HustleStatsResult,
    MatchupsResult,
    PlayByPlayResult,
    XInsightsResult,
)


@tool
def resolve_game(query: str, date: str | None = None) -> GameResolution:
    """Resolve a free-text game description to a specific NBA game_id.

    Call this once at the start of a session to figure out which game the
    user is watching or asking about. Usually just pass query as the team
    name(s) only (e.g. "Lakers vs Celtics" or "BOS") — strip out anything
    else the user mentioned (players, specific plays, "the one where...");
    matching is a plain substring/word check against team names, so extra
    words don't help and can cause a false "not found". Leave date unset —
    this searches a 5-day window (today +/- 2 days) automatically, so you
    don't need to guess or compute an exact date from relative phrasing
    like "yesterday" yourself. Only pass a specific date="YYYY-MM-DD" when
    the user names an explicit date (e.g. "the game on January 15th").

    Returns status="ok" with a game_id on a single unambiguous match; when
    resolved via the date window, game_date reports which day it fell on.
    status="not_found" means no game matched the query; if games were
    scheduled in range anyway, available_games lists all of them (each
    tagged with its date when resolved via the window) — read this out to
    the user as numbered options instead of just saying "not found" and
    stopping. status="ambiguous" means the query matched more than one
    game; candidates lists them the same way — ask the user to pick one
    rather than guessing. status="unsupported_date" means an explicitly
    passed date string couldn't be parsed.
    """
    raw = nba_client._resolve_game_raw(query, date)
    return GameResolution(
        status=raw["status"],
        game_id=raw.get("game_id"),
        home_team=raw.get("home_team"),
        away_team=raw.get("away_team"),
        game_date=raw.get("game_date"),
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


@tool
def get_matchups(game_id: str, player_name: str) -> MatchupsResult:
    """Get defensive matchup data for a game: every opponent who guarded the
    named player, sorted by how much time they spent matched up against
    them (most first), with their stats in that specific matchup (points
    allowed, shooting allowed, etc.).

    Use this for defense-specific questions standard boxscore/play-by-play
    data can't answer, e.g. "who guarded Brunson the most?" or "how did he
    do against Fox specifically?" — do NOT guess or infer matchups from
    playing time/position; call this instead. matchups[0] is whoever spent
    the most time on the named player.

    status="player_not_found" means no player in this game matched
    player_name — double-check the spelling with the user rather than
    guessing who they meant.
    """
    raw = nba_client._get_matchups_raw(game_id, player_name)
    return MatchupsResult(
        status=raw["status"],
        player_name=raw.get("player_name"),
        matchups=raw.get("matchups", []),
        message=raw.get("message"),
    )


@tool
def get_hustle_stats(game_id: str) -> HustleStatsResult:
    """Get per-player hustle-stat totals for a game: screen assists,
    deflections, charges drawn, box outs, contested shots, and loose balls
    recovered, for every player on both teams.

    Use this for "who had the most screen assists/deflections/charges
    drawn/etc. in the game" questions. These are whole-game totals per
    player, NOT paired to a specific teammate or possession — this tool
    cannot answer "who screened for X specifically" or "how many times did X
    screen for Y"; say that's not available rather than guessing if asked.
    """
    raw = nba_client._get_hustle_stats_raw(game_id)
    return HustleStatsResult(
        status=raw["status"],
        home_team=raw.get("home_team"),
        away_team=raw.get("away_team"),
        home_players=raw.get("home_players", []),
        away_players=raw.get("away_players", []),
        message=raw.get("message"),
    )


@tool
def get_x_expert_insights(query: str) -> XInsightsResult:
    """Get qualitative tactical insights and expert observations from X (formerly Twitter).

    Use this tool when a question asks for tactical context, defense adjustments,
    or reasons behind performance that raw stats alone do not fully explain
    (e.g., "Why is LeBron struggling in the 3rd quarter?", "What tactical adjustments did Lakers make?").

    query can be a team name (e.g. "Lakers"), a player name (e.g. "Steph Curry"), or a combination.

    status="api_error" means this tool is unavailable right now (e.g. no X API
    key configured) — tell the user it's unavailable rather than treating the
    message field as real commentary.
    """
    return x_client.get_x_insights(query)
