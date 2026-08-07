"""Shared resolve/QA orchestration used by more than one entry point
(cli.py, api.py). Prompt templates, tool-list/prompt builders, and message
helpers live here rather than in cli.py so entry points depend on library
code instead of on each other.
"""

import json

from langchain_core.messages import AIMessage, ToolMessage

from nba_live_agent.tools import (
    get_boxscore,
    get_hustle_stats,
    get_matchups,
    get_play_by_play,
    get_x_expert_insights,
)

RESOLVE_SYSTEM_PROMPT_TEMPLATE = (
    "You are an NBA in-game analyst. Today's date is {today}. Your only job "
    "right now is to resolve which game the user is watching via "
    "resolve_game. resolve_game's date param defaults to today's "
    "live/scheduled slate, but also accepts a specific YYYY-MM-DD date for a "
    "past or future game — if the user names a game that isn't today's "
    "(e.g. 'yesterday', 'last night', 'the Lakers Celtics game from January "
    "15'), convert that to a concrete date yourself and pass it. If the "
    "result is ambiguous or not found, don't just say so and stop — "
    "resolve_game's candidates/available_games field lists the real games "
    "on that date; present those as a numbered list ('1. Lakers @ "
    "Celtics', '2. ...') and ask the user to pick one, so they can respond "
    "with a number instead of having to type a team name precisely. Only "
    "fall back to asking them to re-describe the game if that list is "
    "empty too."
)

QA_SYSTEM_PROMPT_TEMPLATE = (
    "You are an NBA in-game analyst. Today's date is {today}. You are "
    "already locked onto a specific game — game_id={game_id} ({away_team} "
    "@ {home_team}) — so do not call resolve_game; use {tool_list} directly "
    "with this game_id or team/player names to answer questions. Give a causal, specific "
    "answer grounded in data. When analyzing performance, team strategy, or player dynamics, "
    "proactively query multiple relevant data sources together (e.g. combining boxscore/play-by-play"
    "{x_commentary_example}, or get_matchups for defensive coverage) "
    "to form a multi-angle response. Defensive matchup questions ('who guarded X the most') — call get_matchups. "
    "Hustle-stat questions — call get_hustle_stats. If the requested period hasn't been played "
    "yet, or a named player doesn't appear in the tool results, say so plainly instead of guessing."
)


def run_turn_stream(graph, messages: list):
    """API-facing twin of cli._run_turn: drives the same
    graph.stream(..., stream_mode="updates") loop, but yields events
    instead of printing progress, so a caller (api.py) can forward them to
    a client as they happen. Yields {"kind": "tool_call", "name": ...} each
    time the agent requests a tool call, and finally
    {"kind": "done", "messages": [...], "tokens": N} once the turn
    completes. A mid-turn failure propagates as an exception through the
    generator — callers that want a single HTTP error status instead of a
    partial result (see run_turn_collect) can let it raise; callers
    streaming a response body (which can't change its status code after
    starting) need to catch it around iteration instead.
    """
    all_messages = list(messages)
    total_tokens = 0
    for update in graph.stream({"messages": messages}, stream_mode="updates"):
        for node_name, node_output in update.items():
            new_messages = node_output["messages"]
            all_messages.extend(new_messages)
            if node_name == "agent":
                last = new_messages[-1]
                usage = getattr(last, "usage_metadata", None)
                if usage:
                    total_tokens += usage.get("total_tokens", 0)
                for call in getattr(last, "tool_calls", []):
                    yield {"kind": "tool_call", "name": call["name"]}
    yield {"kind": "done", "messages": all_messages, "tokens": total_tokens}


def run_turn_collect(graph, messages: list) -> tuple[list, int]:
    """Non-streaming twin of run_turn_stream for callers that just want the
    final result (messages, tokens) and are fine with a mid-turn failure
    propagating as a plain exception.
    """
    for event in run_turn_stream(graph, messages):
        if event["kind"] == "done":
            return event["messages"], event["tokens"]


def last_ai_message(messages) -> AIMessage:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    raise RuntimeError("No AIMessage found in graph output.")


def extract_game_info(messages) -> dict | None:
    """Pull the resolved game_id/team names out of resolve_game's ToolMessage,
    if resolution succeeded this turn. This is the one piece of state that
    carries forward across questions — everything else in the turn's
    messages (tool calls, raw play-by-play/boxscore payloads) is discarded
    once the answer is printed, so per-question cost doesn't grow with
    session length.
    """
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and message.name == "resolve_game":
            try:
                data = json.loads(message.content)
            except (json.JSONDecodeError, AttributeError):
                return None
            return data if data.get("status") == "ok" else None
    return None


def build_qa_tools(x_commentary: bool) -> list:
    """The QA-phase tool list. get_x_expert_insights is opt-in (excluded by
    default) since each real call to X's API costs money (X API v2 is
    pay-per-usage — see README) once X_BEARER_TOKEN is configured; the
    --x-commentary flag (or its web-app checkbox equivalent) is the only
    thing that turns it on, independent of whether a token happens to be set.
    """
    tools = [get_play_by_play, get_boxscore, get_matchups, get_hustle_stats]
    if x_commentary:
        tools.append(get_x_expert_insights)
    return tools


def build_qa_prompt(*, today: str, game_id: str, away_team: str, home_team: str, x_commentary: bool) -> str:
    """Builds QA_SYSTEM_PROMPT_TEMPLATE's tool references to match whatever
    build_qa_tools actually bound for this run — per build_graph's own
    contract, the model shouldn't be told about a tool it doesn't have.
    """
    tool_names = ["get_play_by_play", "get_boxscore", "get_matchups", "get_hustle_stats"]
    if x_commentary:
        tool_names.append("get_x_expert_insights")
    tool_list = ", ".join(tool_names[:-1]) + f", and {tool_names[-1]}"
    x_commentary_example = " with get_x_expert_insights for qualitative context" if x_commentary else ""

    return QA_SYSTEM_PROMPT_TEMPLATE.format(
        today=today,
        game_id=game_id,
        away_team=away_team,
        home_team=home_team,
        tool_list=tool_list,
        x_commentary_example=x_commentary_example,
    )
