"""FastAPI backend for the web app (see streamlit_app.py for the frontend).

Fully stateless: every field a handler needs comes in the request body.
/resolve's caller round-trips the resolve-phase message history itself
(needed for the "ambiguous -> numbered list -> reply with a number" UX);
/ask needs no history at all, matching the QA phase's existing "fresh turn
per question" design (see session.py).
"""

import contextvars
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import date
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
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
    run_turn_stream,
)
from nba_live_agent.tools import resolve_game

logger = logging.getLogger(__name__)

# Per-request sink for anything logged under the "nba_live_agent" logger
# (nba_client's retry warnings, etc.) while a /resolve or /ask turn is in
# flight, so the frontend can show them alongside the answer instead of
# only being visible in the server's own console/log file. A contextvar
# rather than a plain module-level list: concurrent requests each set
# their own sink, so their log records can't leak into each other's
# streams. anyio's threadpool (which is what runs each sync generator
# step) preserves context across the thread hop, so this works even
# though the generator doesn't run on one fixed thread for its whole life.
_current_log_sink: contextvars.ContextVar[list | None] = contextvars.ContextVar("_current_log_sink", default=None)


class _ContextLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        sink = _current_log_sink.get()
        if sink is not None:
            sink.append(self.format(record))


_log_handler = _ContextLogHandler(level=logging.WARNING)
_log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    configure_logging(verbose=os.environ.get("NBA_AGENT_VERBOSE") == "1")
    # configure_logging() clears the "nba_live_agent" logger's handlers, so
    # this has to be attached after it runs, not at module import time.
    logging.getLogger("nba_live_agent").addHandler(_log_handler)
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


def _ndjson(payload: dict) -> str:
    return json.dumps(payload) + "\n"


@app.post("/resolve")
def resolve(req: ResolveRequest, request: Request) -> StreamingResponse:
    today = date.today().isoformat()
    if req.messages:
        messages = _from_wire(req.messages)
    else:
        messages = [SystemMessage(RESOLVE_SYSTEM_PROMPT_TEMPLATE.format(today=today))]
    messages.append(HumanMessage(f"The game I'm watching is: {req.description}"))

    def event_stream():
        # The whole body is wrapped in try/except rather than just the
        # run_turn_stream loop: a StreamingResponse can't change its HTTP
        # status code once headers are sent, so a failure — whenever it
        # happens — has to surface as an in-band {"type": "error"} event
        # instead of an HTTP error status.
        log_sink: list[str] = []
        turn_iter = run_turn_stream(request.app.state.resolve_graph, messages)
        try:
            while True:
                # Re-assert the sink immediately before every advance, not
                # just once up front: a contextvar set once at the top of a
                # sync generator streamed via StreamingResponse does NOT
                # reliably survive across its own later yields — each can
                # resume in a freshly copied context (verified empirically)
                # — so tools_node's contextvars.copy_context() would see
                # nothing set if we didn't redo this every time.
                _current_log_sink.set(log_sink)
                try:
                    event = next(turn_iter)
                except StopIteration:
                    break
                while log_sink:
                    yield _ndjson({"type": "log", "line": log_sink.pop(0)})
                if event["kind"] == "tool_call":
                    yield _ndjson({"type": "tool_call", "name": event["name"]})
                else:
                    result_messages = event["messages"]
                    game_info = extract_game_info(result_messages) or {}
                    final = ResolveResponse(
                        messages=_to_wire(result_messages),
                        reply=last_ai_message(result_messages).text,
                        resolved=bool(game_info),
                        game_id=game_info.get("game_id"),
                        home_team=game_info.get("home_team"),
                        away_team=game_info.get("away_team"),
                        tokens_used=event["tokens"],
                    )
                    yield _ndjson({"type": "final", **final.model_dump()})
        except Exception as e:
            while log_sink:
                yield _ndjson({"type": "log", "line": log_sink.pop(0)})
            logger.exception("Resolve turn failed")
            yield _ndjson({"type": "error", "detail": f"Something went wrong talking to the model: {e}"})
        finally:
            _current_log_sink.set(None)

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.post("/ask")
def ask(req: AskRequest, request: Request) -> StreamingResponse:
    today = date.today().isoformat()
    qa_prompt = build_qa_prompt(
        today=today,
        game_id=req.game_id,
        away_team=req.away_team,
        home_team=req.home_team,
        x_commentary=req.x_commentary,
    )
    graph = request.app.state.qa_graph_x if req.x_commentary else request.app.state.qa_graph
    turn_messages = [SystemMessage(qa_prompt), HumanMessage(req.question)]

    def event_stream():
        log_sink: list[str] = []
        turn_iter = run_turn_stream(graph, turn_messages)
        try:
            while True:
                # See /resolve's event_stream for why this is re-asserted
                # every iteration instead of once up front.
                _current_log_sink.set(log_sink)
                try:
                    event = next(turn_iter)
                except StopIteration:
                    break
                while log_sink:
                    yield _ndjson({"type": "log", "line": log_sink.pop(0)})
                if event["kind"] == "tool_call":
                    yield _ndjson({"type": "tool_call", "name": event["name"]})
                else:
                    final = AskResponse(answer=last_ai_message(event["messages"]).text, tokens_used=event["tokens"])
                    yield _ndjson({"type": "final", **final.model_dump()})
        except Exception as e:
            while log_sink:
                yield _ndjson({"type": "log", "line": log_sink.pop(0)})
            logger.exception("Ask turn failed")
            yield _ndjson({"type": "error", "detail": f"Something went wrong talking to the model: {e}"})
        finally:
            _current_log_sink.set(None)

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
