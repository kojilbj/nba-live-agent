"""MCP server exposing the agent's NBA data tools directly to MCP clients
(e.g. Claude Code), without going through the Gemini agent loop.

Each tool below delegates to the corresponding LangChain tool in
``tools.py`` via ``.invoke()`` and reuses that tool's ``.description`` as
the MCP tool description verbatim, rather than hand-copying the docstring
here: this text is what the calling LLM reads to decide when/how to call
the tool, so it must stay in sync with tools.py (the single source of
truth) instead of silently drifting.
"""

from mcp.server.mcpserver import MCPServer

from nba_live_agent import tools as agent_tools

mcp = MCPServer("nba-live-agent")


@mcp.tool(description=agent_tools.resolve_game.description)
def resolve_game(query: str, date: str | None = None) -> dict:
    result = agent_tools.resolve_game.invoke({"query": query, "date": date})
    return result.model_dump()


@mcp.tool(description=agent_tools.get_play_by_play.description)
def get_play_by_play(game_id: str, period: int | None = None) -> dict:
    result = agent_tools.get_play_by_play.invoke({"game_id": game_id, "period": period})
    return result.model_dump()


@mcp.tool(description=agent_tools.get_boxscore.description)
def get_boxscore(game_id: str) -> dict:
    result = agent_tools.get_boxscore.invoke({"game_id": game_id})
    return result.model_dump()


@mcp.tool(description=agent_tools.get_matchups.description)
def get_matchups(game_id: str, player_name: str) -> dict:
    result = agent_tools.get_matchups.invoke({"game_id": game_id, "player_name": player_name})
    return result.model_dump()


@mcp.tool(description=agent_tools.get_hustle_stats.description)
def get_hustle_stats(game_id: str) -> dict:
    result = agent_tools.get_hustle_stats.invoke({"game_id": game_id})
    return result.model_dump()


@mcp.tool(description=agent_tools.get_x_expert_insights.description)
def get_x_expert_insights(query: str) -> dict:
    result = agent_tools.get_x_expert_insights.invoke({"query": query})
    return result.model_dump()


if __name__ == "__main__":
    mcp.run()
