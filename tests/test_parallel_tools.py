import time
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, ToolCall
from nba_live_agent.agent import AgentState, build_graph


def test_parallel_tool_execution():
    def slow_tool_1(query: str):

        time.sleep(0.2)
        return f"result_1 for {query}"

    def slow_tool_2(query: str):
        time.sleep(0.2)
        return f"result_2 for {query}"

    mock_tool_1 = MagicMock()
    mock_tool_1.name = "tool_1"
    mock_tool_1.invoke.side_effect = slow_tool_1

    mock_tool_2 = MagicMock()
    mock_tool_2.name = "tool_2"
    mock_tool_2.invoke.side_effect = slow_tool_2

    tools = [mock_tool_1, mock_tool_2]

    mock_model = MagicMock()
    graph = build_graph(mock_model, tools)

    # Initial state with a tool call requesting both tools
    ai_msg = AIMessage(
        content="",
        tool_calls=[
            ToolCall(name="tool_1", args={"query": "Lakers"}, id="call_1"),
            ToolCall(name="tool_2", args={"query": "Lakers"}, id="call_2"),
        ],
    )
    state: AgentState = {"messages": [ai_msg]}

    start_time = time.time()
    result = graph.nodes["tools"].invoke(state)
    elapsed = time.time() - start_time

    # Sequential execution would take >= 0.4s. Parallel should take ~0.2s.
    assert elapsed < 0.35
    assert len(result["messages"]) == 2
