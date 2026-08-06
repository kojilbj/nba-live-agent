import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from nba_live_agent.agent import build_graph


@tool
def failing_tool(x: int) -> str:
    """A tool that always raises, used to test failure-path logging."""
    raise RuntimeError("boom")


class _ToolCallModel:
    """Stands in for a chat model with .bind_tools() already applied —
    always returns a single tool call to failing_tool.
    """

    def invoke(self, messages):
        return AIMessage(
            content="",
            tool_calls=[{"name": "failing_tool", "args": {"x": 1}, "id": "call_1"}],
        )


class _FailingModel:
    def invoke(self, messages):
        raise RuntimeError("model boom")


def test_tool_failure_is_logged_and_still_propagates(caplog):
    graph = build_graph(_ToolCallModel(), tools=[failing_tool])

    with caplog.at_level(logging.ERROR, logger="nba_live_agent.agent"):
        with pytest.raises(RuntimeError, match="boom"):
            graph.invoke({"messages": [HumanMessage("hi")]})

    assert any(
        r.levelno == logging.ERROR and "failing_tool" in r.message for r in caplog.records
    )


def test_model_invocation_failure_is_logged_and_still_propagates(caplog):
    graph = build_graph(_FailingModel(), tools=[])

    with caplog.at_level(logging.ERROR, logger="nba_live_agent.agent"):
        with pytest.raises(RuntimeError, match="model boom"):
            graph.invoke({"messages": [HumanMessage("hi")]})

    assert any(r.levelno == logging.ERROR for r in caplog.records)
