import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from nba_live_agent.agent import build_graph
from nba_live_agent.session import (
    RESOLVE_SYSTEM_PROMPT_TEMPLATE,
    build_qa_prompt,
    build_qa_tools,
    extract_game_info,
    last_ai_message,
    run_turn_collect,
    run_turn_stream,
)
from nba_live_agent.tools import get_boxscore, get_hustle_stats, get_matchups, get_play_by_play, get_x_expert_insights


def test_build_qa_tools_excludes_x_by_default():
    tools = build_qa_tools(x_commentary=False)

    assert get_x_expert_insights not in tools
    assert tools == [get_play_by_play, get_boxscore, get_matchups, get_hustle_stats]


def test_build_qa_tools_includes_x_when_enabled():
    tools = build_qa_tools(x_commentary=True)

    assert get_x_expert_insights in tools
    assert tools == [
        get_play_by_play,
        get_boxscore,
        get_matchups,
        get_hustle_stats,
        get_x_expert_insights,
    ]


def test_build_qa_prompt_excludes_x_mentions_by_default():
    prompt = build_qa_prompt(
        today="2026-01-16", game_id="G1", away_team="Celtics", home_team="Lakers", x_commentary=False
    )

    assert "get_x_expert_insights" not in prompt
    assert "get_play_by_play" in prompt
    assert "get_hustle_stats" in prompt


def test_build_qa_prompt_includes_x_when_enabled():
    prompt = build_qa_prompt(
        today="2026-01-16", game_id="G1", away_team="Celtics", home_team="Lakers", x_commentary=True
    )

    assert "get_x_expert_insights" in prompt
    # Mentioned twice: once in the tool list, once in the multi-source example.
    assert prompt.count("get_x_expert_insights") == 2


def test_resolve_prompt_forbids_speculating_about_system_health():
    prompt = RESOLVE_SYSTEM_PROMPT_TEMPLATE.format(today="2026-01-16")

    assert "speculate" in prompt
    assert "api_error" in prompt
    assert "other leagues" in prompt


def _resolve_tool_message(status: str, **extra) -> ToolMessage:
    payload = {"status": status, **extra}
    return ToolMessage(content=json.dumps(payload), name="resolve_game", tool_call_id="call_1")


def test_extract_game_info_ok_status_returns_dict():
    messages = [
        HumanMessage("Lakers"),
        _resolve_tool_message("ok", game_id="G1", home_team="Lakers", away_team="Celtics"),
    ]

    info = extract_game_info(messages)

    assert info == {"status": "ok", "game_id": "G1", "home_team": "Lakers", "away_team": "Celtics"}


def test_extract_game_info_ambiguous_status_returns_none():
    messages = [_resolve_tool_message("ambiguous", candidates=["Lakers @ Celtics", "Lakers @ Warriors"])]

    assert extract_game_info(messages) is None


def test_extract_game_info_not_found_status_returns_none():
    messages = [_resolve_tool_message("not_found", available_games=["Lakers @ Celtics"])]

    assert extract_game_info(messages) is None


def test_extract_game_info_non_json_content_returns_none():
    messages = [ToolMessage(content="not json", name="resolve_game", tool_call_id="call_1")]

    assert extract_game_info(messages) is None


def test_extract_game_info_no_resolve_game_message_returns_none():
    messages = [HumanMessage("hi"), AIMessage(content="hello")]

    assert extract_game_info(messages) is None


def test_last_ai_message_found():
    ai = AIMessage(content="answer")
    messages = [HumanMessage("hi"), ai]

    assert last_ai_message(messages) is ai


def test_last_ai_message_raises_when_absent():
    with pytest.raises(RuntimeError, match="No AIMessage found"):
        last_ai_message([HumanMessage("hi")])


class _SimpleModel:
    """A single-turn model — no tool calls, so the graph ends after one
    agent_node pass. Reports usage_metadata like a real chat model would."""

    def invoke(self, messages):
        usage = {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}
        return AIMessage(content="final answer", usage_metadata=usage)


class _FailingModel:
    def invoke(self, messages):
        raise RuntimeError("model boom")


def test_run_turn_collect_returns_messages_and_summed_tokens():
    graph = build_graph(_SimpleModel(), tools=[])

    messages, tokens = run_turn_collect(graph, [SystemMessage("sys"), HumanMessage("hi")])

    assert tokens == 3
    assert messages[-1].content == "final answer"
    # Input messages are preserved at the front, unlike _run_turn's failure
    # path which only preserves input on error — here it's just always true
    # since all_messages starts as list(messages).
    assert messages[0].content == "sys"


def test_run_turn_collect_propagates_exception_instead_of_swallowing():
    """Unlike cli._run_turn (which catches, logs, and returns a partial
    result so the interactive session can continue), run_turn_collect must
    let a mid-turn failure propagate so api.py can turn it into an HTTP
    error response instead of a 200 with no real answer.
    """
    graph = build_graph(_FailingModel(), tools=[])

    with pytest.raises(RuntimeError, match="model boom"):
        run_turn_collect(graph, [HumanMessage("hi")])


@tool
def fake_tool(x: int) -> str:
    """A tool that just echoes back its input, used to test run_turn_stream's
    tool_call events."""
    return f"got {x}"


class _ToolThenAnswerModel:
    """First invoke(): requests fake_tool. Second invoke() (after the tool
    result is appended to messages): a final answer with no more tool
    calls, ending the turn."""

    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(content="", tool_calls=[{"name": "fake_tool", "args": {"x": 1}, "id": "call_1"}])
        usage = {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}
        return AIMessage(content="final answer", usage_metadata=usage)


def test_run_turn_stream_yields_tool_call_then_done():
    graph = build_graph(_ToolThenAnswerModel(), tools=[fake_tool])

    events = list(run_turn_stream(graph, [HumanMessage("hi")]))

    assert events[0] == {"kind": "tool_call", "name": "fake_tool"}
    assert events[-1]["kind"] == "done"
    assert events[-1]["tokens"] == 5
    assert events[-1]["messages"][-1].content == "final answer"


def test_run_turn_stream_propagates_exception():
    graph = build_graph(_FailingModel(), tools=[])

    with pytest.raises(RuntimeError, match="model boom"):
        list(run_turn_stream(graph, [HumanMessage("hi")]))
