"""The hand-rolled ReAct loop: agent node decides tool-call vs. final answer,
tools node executes whichever tool(s) were requested, then control returns
to the agent node. This two-node loop *is* the "agentic loop."
"""

import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel


from nba_live_agent.tools import (
    get_boxscore,
    get_hustle_stats,
    get_matchups,
    get_play_by_play,
    get_x_expert_insights,
    resolve_game,
)

logger = logging.getLogger(__name__)

TOOLS = [
    resolve_game,
    get_play_by_play,
    get_boxscore,
    get_matchups,
    get_hustle_stats,
    get_x_expert_insights,
]


class AgentState(TypedDict):

    messages: Annotated[Sequence[BaseMessage], add_messages]


def build_graph(model_with_tools, tools):
    """Takes any chat model with .bind_tools(tools) already applied, so the
    loop wiring can be tested independently of which model/API is behind it.

    tools must be exactly what was bound to model_with_tools — the graph
    only executes what the model can see, so callers that want to restrict
    which tools are available in a given phase (e.g. resolve-only vs.
    QA-only) do so by binding a narrower list, not by prompting the model
    not to call the rest. A prompt instruction alone doesn't reliably stop
    the model from calling a tool it can technically still see.
    """
    tools_by_name = {t.name: t for t in tools}

    def agent_node(state: AgentState) -> dict:
        try:
            response = model_with_tools.invoke(state["messages"])
        except Exception:
            logger.exception("Model invocation failed")
            raise
        return {"messages": [response]}

    def _execute_tool_call(call):
        try:
            result = tools_by_name[call["name"]].invoke(call["args"])
        except Exception:
            logger.exception("Tool %s failed (args=%s)", call["name"], call["args"])
            raise
        content = result.model_dump_json() if isinstance(result, BaseModel) else str(result)
        return ToolMessage(content=content, name=call["name"], tool_call_id=call["id"])

    def tools_node(state: AgentState) -> dict:
        last_message = state["messages"][-1]
        tool_calls = last_message.tool_calls
        if not tool_calls:
            return {"messages": []}

        with ThreadPoolExecutor(max_workers=len(tool_calls)) as executor:
            outputs = list(executor.map(_execute_tool_call, tool_calls))
        return {"messages": outputs}

    def should_continue(state: AgentState) -> str:
        last_message = state["messages"][-1]

        return "tools" if last_message.tool_calls else "end"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def build_live_graph(tools: list[BaseTool] = TOOLS, model: str = "gemini-3.5-flash-lite"):
    """Wires the loop to the real Gemini model. Pinned to the Flash-Lite tier
    (current-gen "3.5" family, not "2.5" — Google has been cutting off new-key
    access to older generations, e.g. gemini-2.5-flash) rather than
    "gemini-flash-latest": the latest-alias tier (gemini-3.6-flash as of Aug
    2026) carries a much stricter free-tier quota (20 requests/day) that's
    easy to exhaust during dev/testing, and Flash-Lite is ~5x cheaper per
    token besides. See https://ai.google.dev/gemini-api/docs/models and
    https://ai.google.dev/gemini-api/docs/pricing.

    tools defaults to the full TOOLS list but callers building a
    phase-restricted graph (see build_graph's docstring) should pass a
    narrower list explicitly.
    """
    # No temperature kwarg: gemini-3.5-flash-lite uses fixed sampling
    # defaults and ignores it, so passing temperature=0 only produced a
    # UserWarning on every call with no effect.
    llm = ChatGoogleGenerativeAI(model=model)
    return build_graph(llm.bind_tools(tools), tools)
