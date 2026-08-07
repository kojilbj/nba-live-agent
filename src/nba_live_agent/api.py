"""FastAPI backend for the web app (see streamlit_app.py for the frontend).

Fully stateless: every field a handler needs comes in the request body.
/resolve's caller round-trips the resolve-phase message history itself
(needed for the "ambiguous -> numbered list -> reply with a number" UX);
/ask needs no history at all, matching the QA phase's existing "fresh turn
per question" design (see session.py).
"""

import os
from contextlib import asynccontextmanager
from datetime import date
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel

from nba_live_agent.agent import build_live_graph
from nba_live_agent.logging_config import configure_logging
from nba_live_agent.session import (
    RESOLVE_SYSTEM_PROMPT_TEMPLATE,
    build_qa_prompt,
    build_qa_tools,
    extract_game_info,
    last_ai_message,
    run_turn_collect,
)
from nba_live_agent.tools import resolve_game


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    configure_logging(verbose=os.environ.get("NBA_AGENT_VERBOSE") == "1")
    # Two QA graph variants (with/without get_x_expert_insights) built once
    # at startup and reused across requests, mirroring how cli.run() builds
    # qa_graph once and reuses it across the CLI's while-loop.
    app.state.resolve_graph = build_live_graph(tools=[resolve_game])
    app.state.qa_graph = build_live_graph(tools=build_qa_tools(x_commentary=False))
    app.state.qa_graph_x = build_live_graph(tools=build_qa_tools(x_commentary=True))
    yield


app = FastAPI(title="NBA Live Agent API", lifespan=lifespan)


class WireMessage(BaseModel):
    role: Literal["system", "human", "ai", "tool"]
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict] | None = None


class ResolveRequest(BaseModel):
    messages: list[WireMessage] = []
    description: str


class ResolveResponse(BaseModel):
    messages: list[WireMessage]
    reply: str
    resolved: bool
    game_id: str | None = None
    home_team: str | None = None
    away_team: str | None = None
    tokens_used: int


class AskRequest(BaseModel):
    game_id: str
    away_team: str | None = None
    home_team: str | None = None
    question: str
    x_commentary: bool = False


class AskResponse(BaseModel):
    answer: str
    tokens_used: int


def _from_wire(wire_messages: list[WireMessage]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for wm in wire_messages:
        if wm.role == "system":
            messages.append(SystemMessage(content=wm.content))
        elif wm.role == "human":
            messages.append(HumanMessage(content=wm.content))
        elif wm.role == "ai":
            messages.append(AIMessage(content=wm.content, tool_calls=wm.tool_calls or []))
        elif wm.role == "tool":
            messages.append(ToolMessage(content=wm.content, name=wm.name, tool_call_id=wm.tool_call_id))
    return messages


def _to_wire(messages: list[BaseMessage]) -> list[WireMessage]:
    wire = []
    for m in messages:
        if isinstance(m, SystemMessage):
            wire.append(WireMessage(role="system", content=m.text))
        elif isinstance(m, HumanMessage):
            wire.append(WireMessage(role="human", content=m.text))
        elif isinstance(m, AIMessage):
            wire.append(WireMessage(role="ai", content=m.text, tool_calls=m.tool_calls or None))
        elif isinstance(m, ToolMessage):
            wire.append(WireMessage(role="tool", content=m.text, name=m.name, tool_call_id=m.tool_call_id))
    return wire


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/resolve", response_model=ResolveResponse)
def resolve(req: ResolveRequest, request: Request) -> ResolveResponse:
    today = date.today().isoformat()
    if req.messages:
        messages = _from_wire(req.messages)
    else:
        messages = [SystemMessage(RESOLVE_SYSTEM_PROMPT_TEMPLATE.format(today=today))]
    messages.append(HumanMessage(f"The game I'm watching is: {req.description}"))

    try:
        messages, tokens = run_turn_collect(request.app.state.resolve_graph, messages)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Something went wrong talking to the model: {e}") from e

    game_info = extract_game_info(messages) or {}
    return ResolveResponse(
        messages=_to_wire(messages),
        reply=last_ai_message(messages).text,
        resolved=bool(game_info),
        game_id=game_info.get("game_id"),
        home_team=game_info.get("home_team"),
        away_team=game_info.get("away_team"),
        tokens_used=tokens,
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, request: Request) -> AskResponse:
    today = date.today().isoformat()
    qa_prompt = build_qa_prompt(
        today=today,
        game_id=req.game_id,
        away_team=req.away_team,
        home_team=req.home_team,
        x_commentary=req.x_commentary,
    )
    graph = request.app.state.qa_graph_x if req.x_commentary else request.app.state.qa_graph
    try:
        messages, tokens = run_turn_collect(graph, [SystemMessage(qa_prompt), HumanMessage(req.question)])
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Something went wrong talking to the model: {e}") from e

    return AskResponse(answer=last_ai_message(messages).text, tokens_used=tokens)
