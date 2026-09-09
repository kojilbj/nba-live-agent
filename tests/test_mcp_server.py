"""Tests for the MCP server tool wrappers.

Each MCP tool function is a thin wrapper around the corresponding
LangChain tool in ``tools.py`` that converts the returned Pydantic model
to a plain dict. We mock at the ``nba_client``/``x_client`` layer (same
as ``test_tools_errors.py``) and call the MCP tool functions directly —
no real MCP transport is started.
"""

import asyncio
from unittest.mock import patch

from nba_live_agent import mcp_server, tools as agent_tools, x_client
from nba_live_agent.models import XInsightsResult


def _fake_team(team_id, name, tricode):
    return {"teamId": team_id, "teamName": name, "teamTricode": tricode}


def test_resolve_game_returns_dict_with_game_id():
    with patch("nba_live_agent.nba_client.scoreboard.ScoreBoard") as mock_sb:
        mock_sb.return_value.games.get_dict.return_value = [
            {
                "gameId": "0022500001",
                "homeTeam": _fake_team(1610612738, "Celtics", "BOS"),
                "awayTeam": _fake_team(1610612747, "Lakers", "LAL"),
                "gameStatus": 2,
            }
        ]
        result = mcp_server.resolve_game(query="Lakers vs Celtics", date="today")

    assert isinstance(result, dict)
    assert result["status"] == "ok"
    assert result["game_id"] == "0022500001"


def test_resolve_game_default_date_is_none():
    with patch(
        "nba_live_agent.nba_client._games_for_window_raw",
        return_value={"status": "ok", "games": []},
    ):
        result = mcp_server.resolve_game(query="Lakers")

    assert result["status"] == "not_found"


def test_get_play_by_play_returns_dict_with_events():
    with (
        patch("nba_live_agent.nba_client._game_status", return_value=2),
        patch("nba_live_agent.nba_client.playbyplay.PlayByPlay") as mock_pbp,
    ):
        mock_pbp.return_value.actions.get_dict.return_value = [
            {
                "period": 1,
                "clock": "PT00M00.00S",
                "teamTricode": "BOS",
                "playerNameI": "J. Brown",
                "actionType": "2pt",
                "subType": "jumpshot",
                "description": "J. Brown makes 2-pt shot",
                "scoreHome": "2",
                "scoreAway": "0",
            }
        ]
        result = mcp_server.get_play_by_play(game_id="0022500001", period=1)

    assert result["status"] == "ok"
    assert result["events"][0]["player_name"] == "J. Brown"


def test_get_boxscore_returns_dict():
    with patch("nba_live_agent.nba_client.boxscore.BoxScore") as mock_box:
        mock_box.return_value.game.get_dict.return_value = {
            "gameStatus": 3,
            "homeTeam": {"teamName": "Celtics", "teamTricode": "BOS", "players": []},
            "awayTeam": {"teamName": "Lakers", "teamTricode": "LAL", "players": []},
        }
        result = mcp_server.get_boxscore(game_id="0022500001")

    assert result["status"] == "ok"
    assert result["home_team"] == "Celtics"


def test_get_matchups_returns_dict():
    with patch("nba_live_agent.nba_client.boxscorematchupsv3.BoxScoreMatchupsV3") as mock_matchups:
        ds = mock_matchups.return_value.player_stats
        ds.get_dict.return_value = {"headers": [], "data": []}
        result = mcp_server.get_matchups(game_id="0022500001", player_name="Brunson")

    assert result["status"] == "game_not_started"


def test_get_hustle_stats_returns_dict():
    with patch("nba_live_agent.nba_client.boxscorehustlev2.BoxScoreHustleV2") as mock_hustle:
        mock_hustle.return_value.team_stats.get_dict.return_value = {"headers": [], "data": []}
        mock_hustle.return_value.player_stats.get_dict.return_value = {"headers": [], "data": []}
        result = mcp_server.get_hustle_stats(game_id="0022500001")

    assert result["status"] == "game_not_started"


def test_get_x_expert_insights_returns_dict():
    fake_result = XInsightsResult(status="ok", query="Lakers", posts=[], is_mock=True)
    with patch.object(x_client, "get_x_insights", return_value=fake_result):
        result = mcp_server.get_x_expert_insights(query="Lakers")

    assert result["status"] == "ok"
    assert result["is_mock"] is True


def test_all_six_tools_are_registered_on_the_server():
    tools = asyncio.run(mcp_server.mcp.list_tools())
    names = {t.name for t in tools}
    assert names == {
        "resolve_game",
        "get_play_by_play",
        "get_boxscore",
        "get_matchups",
        "get_hustle_stats",
        "get_x_expert_insights",
    }


def test_tool_descriptions_match_the_langchain_tools_verbatim():
    """The MCP tool description is what the calling LLM reads to decide
    when/how to call the tool - it must stay byte-for-byte in sync with
    tools.py's docstring (the source of truth), not a hand-copied string
    that can silently drift when tools.py is edited.
    """
    mcp_tools = {t.name: t.description for t in asyncio.run(mcp_server.mcp.list_tools())}

    for name in mcp_tools:
        langchain_tool = getattr(agent_tools, name)
        assert mcp_tools[name] == langchain_tool.description
