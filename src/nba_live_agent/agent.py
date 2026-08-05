"""The hand-rolled ReAct loop: agent node decides tool-call vs. final answer,
tools node executes whichever tool(s) were requested, then control returns
to the agent node. This two-node loop *is* the "agentic loop."
"""

from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from nba_live_agent.tools import get_boxscore, get_play_by_play, resolve_game

TOOLS = [resolve_game, get_play_by_play, get_boxscore]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def build_graph(model_with_tools):
    """Takes any chat model with .bind_tools(TOOLS) already applied, so the
    loop wiring can be tested independently of which model/API is behind it.
    """

    def agent_node(state: AgentState) -> dict:
        response = model_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def tools_node(state: AgentState) -> dict:
        last_message = state["messages"][-1]
        outputs = []
        for call in last_message.tool_calls:
            result = TOOLS_BY_NAME[call["name"]].invoke(call["args"])
            content = result.model_dump_json() if isinstance(result, BaseModel) else str(result)
            outputs.append(
                ToolMessage(content=content, name=call["name"], tool_call_id=call["id"])
            )
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


def build_live_graph(model: str = "gemini-3.5-flash-lite"):
    """Wires the loop to the real Gemini model. Pinned to the Flash-Lite tier
    (current-gen "3.5" family, not "2.5" — Google has been cutting off new-key
    access to older generations, e.g. gemini-2.5-flash) rather than
    "gemini-flash-latest": the latest-alias tier (gemini-3.6-flash as of Aug
    2026) carries a much stricter free-tier quota (20 requests/day) that's
    easy to exhaust during dev/testing, and Flash-Lite is ~5x cheaper per
    token besides. See https://ai.google.dev/gemini-api/docs/models and
    https://ai.google.dev/gemini-api/docs/pricing.
    """
    # No temperature kwarg: gemini-3.5-flash-lite uses fixed sampling
    # defaults and ignores it, so passing temperature=0 only produced a
    # UserWarning on every call with no effect.
    llm = ChatGoogleGenerativeAI(model=model)
    return build_graph(llm.bind_tools(TOOLS))
