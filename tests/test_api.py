import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage

from nba_live_agent.api import app


class _FakeGraph:
    """Stands in for a compiled LangGraph — records what it was called with
    and replays a pre-scripted sequence of stream() updates, so tests never
    touch a real model or nba_api.
    """

    def __init__(self, updates):
        self._updates = updates
        self.received_inputs = []

    def stream(self, input, stream_mode="updates"):
        self.received_inputs.append(input)
        yield from self._updates


class _RaisingGraph:
    def stream(self, input, stream_mode="updates"):
        raise RuntimeError("boom")


def _ai_update(node="agent", **kwargs):
    return {node: {"messages": [AIMessage(**kwargs)]}}


def _tool_update(status: str, **extra):
    payload = {"status": status, **extra}
    tool_message = ToolMessage(content=json.dumps(payload), name="resolve_game", tool_call_id="call_1")
    return {"tools": {"messages": [tool_message]}}


def _events(resp) -> list[dict]:
    """/resolve and /ask return newline-delimited JSON; parse each line."""
    return [json.loads(line) for line in resp.text.splitlines() if line]


@pytest.fixture
def client(monkeypatch):
    # lifespan's build_live_graph() calls construct a real ChatGoogleGenerativeAI,
    # which validates a GOOGLE_API_KEY is *present* at construction time (but
    # never makes a network call there) — a dummy value is enough to get past
    # startup; every test below replaces app.state's graphs before use anyway.
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_resolve_first_call_not_resolved(client):
    graph = _FakeGraph(
        [
            _ai_update(content="", tool_calls=[{"name": "resolve_game", "args": {"query": "Lakers"}, "id": "call_1"}]),
            _tool_update("ambiguous", candidates=["Lakers @ Celtics", "Lakers @ Warriors"]),
            _ai_update(content="Did you mean 1. Lakers @ Celtics or 2. Lakers @ Warriors?"),
        ]
    )
    client.app.state.resolve_graph = graph

    resp = client.post("/resolve", json={"messages": [], "description": "Lakers"})

    assert resp.status_code == 200
    events = _events(resp)
    assert events[0] == {"type": "tool_call", "name": "resolve_game"}
    final = events[-1]
    assert final["type"] == "final"
    assert final["resolved"] is False
    assert final["game_id"] is None
    assert "Lakers @ Celtics" in final["reply"]
    # First call with no prior history seeds a fresh SystemMessage.
    assert graph.received_inputs[0]["messages"][0].type == "system"


def test_resolve_wire_round_trip_preserves_tool_calls(client):
    first_graph = _FakeGraph(
        [
            _ai_update(content="", tool_calls=[{"name": "resolve_game", "args": {"query": "Lakers"}, "id": "call_1"}]),
            _tool_update("ambiguous", candidates=["Lakers @ Celtics", "Lakers @ Warriors"]),
            _ai_update(content="Did you mean 1. Lakers @ Celtics or 2. Lakers @ Warriors?"),
        ]
    )
    client.app.state.resolve_graph = first_graph

    first_resp = client.post("/resolve", json={"messages": [], "description": "Lakers"})
    wire_messages = _events(first_resp)[-1]["messages"]

    second_graph = _FakeGraph([_ai_update(content="ok, got it")])
    client.app.state.resolve_graph = second_graph

    client.post("/resolve", json={"messages": wire_messages, "description": "1"})

    reconstructed = second_graph.received_inputs[0]["messages"]
    ai_with_call = next(m for m in reconstructed if getattr(m, "tool_calls", None))
    tool_result = next(m for m in reconstructed if isinstance(m, ToolMessage))
    assert ai_with_call.tool_calls[0]["id"] == "call_1"
    assert tool_result.tool_call_id == "call_1"
    assert tool_result.name == "resolve_game"


def test_resolve_resolved_path_populates_game_info(client):
    graph = _FakeGraph(
        [
            _ai_update(content="", tool_calls=[{"name": "resolve_game", "args": {"query": "Lakers"}, "id": "call_1"}]),
            _tool_update("ok", game_id="G1", home_team="Lakers", away_team="Celtics"),
            _ai_update(content="Got it — Celtics @ Lakers."),
        ]
    )
    client.app.state.resolve_graph = graph

    resp = client.post("/resolve", json={"messages": [], "description": "Lakers vs Celtics"})

    final = _events(resp)[-1]
    assert final["type"] == "final"
    assert final["resolved"] is True
    assert final["game_id"] == "G1"
    assert final["home_team"] == "Lakers"
    assert final["away_team"] == "Celtics"


def test_ask_routes_to_x_commentary_graph_when_enabled(client):
    plain_graph = _FakeGraph([_ai_update(content="answer without X")])
    x_graph = _FakeGraph([_ai_update(content="answer with X")])
    client.app.state.qa_graph = plain_graph
    client.app.state.qa_graph_x = x_graph

    resp = client.post(
        "/ask",
        json={
            "game_id": "G1",
            "away_team": "Celtics",
            "home_team": "Lakers",
            "question": "Why isn't he scoring?",
            "x_commentary": True,
        },
    )

    assert resp.status_code == 200
    final = _events(resp)[-1]
    assert final["type"] == "final"
    assert final["answer"] == "answer with X"
    assert len(x_graph.received_inputs) == 1
    assert len(plain_graph.received_inputs) == 0


def test_ask_routes_to_plain_graph_by_default(client):
    plain_graph = _FakeGraph([_ai_update(content="answer without X")])
    x_graph = _FakeGraph([_ai_update(content="answer with X")])
    client.app.state.qa_graph = plain_graph
    client.app.state.qa_graph_x = x_graph

    resp = client.post(
        "/ask",
        json={"game_id": "G1", "away_team": "Celtics", "home_team": "Lakers", "question": "Why isn't he scoring?"},
    )

    assert resp.status_code == 200
    final = _events(resp)[-1]
    assert final["type"] == "final"
    assert final["answer"] == "answer without X"
    assert len(plain_graph.received_inputs) == 1
    assert len(x_graph.received_inputs) == 0


def test_resolve_returns_error_event_on_graph_failure(client):
    """A StreamingResponse can't change its HTTP status after headers are
    sent, so a mid-turn failure surfaces as an in-band {"type": "error"}
    event instead of an HTTP error status — the status stays 200.
    """
    client.app.state.resolve_graph = _RaisingGraph()

    resp = client.post("/resolve", json={"messages": [], "description": "Lakers"})

    assert resp.status_code == 200
    final = _events(resp)[-1]
    assert final["type"] == "error"
    assert "detail" in final


def test_ask_returns_error_event_on_graph_failure(client):
    client.app.state.qa_graph = _RaisingGraph()

    resp = client.post(
        "/ask",
        json={"game_id": "G1", "away_team": "Celtics", "home_team": "Lakers", "question": "Why isn't he scoring?"},
    )

    assert resp.status_code == 200
    final = _events(resp)[-1]
    assert final["type"] == "error"
    assert "detail" in final
