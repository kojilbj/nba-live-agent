"""Streamlit frontend for the NBA live-game analyst. Talks to the FastAPI
backend (src/nba_live_agent/api.py) over HTTP only — never imports
nba_live_agent internals, so it needs no PYTHONPATH/sys.path setup, same as
how run.py is a thin launcher for cli.py.
"""

import json
import os

import requests
import streamlit as st

API_BASE_URL = os.environ.get("NBA_AGENT_API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 60

# Mirrors cli.py's TOOL_STATUS_MESSAGES — duplicated rather than imported
# since this file never reaches into nba_live_agent internals (see the
# module docstring above).
TOOL_STATUS_MESSAGES = {
    "resolve_game": "Looking up the game...",
    "get_play_by_play": "Pulling play-by-play...",
    "get_boxscore": "Checking the boxscore...",
    "get_matchups": "Checking matchup data...",
    "get_hustle_stats": "Checking hustle stats...",
    "get_x_expert_insights": "Searching X for expert commentary...",
}

st.set_page_config(page_title="NBA Live Agent", page_icon="🏀", layout="centered")


def _get_secret(name: str) -> str | None:
    """Checks Streamlit secrets first (works with HF Spaces / Streamlit
    Cloud secret managers), then falls back to a plain env var. Wrapped in
    try/except since st.secrets raises if no secrets.toml exists at all
    (the common case for local dev, where the gate should just be skipped).
    """
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name)


def _check_password() -> bool:
    """Shared-password gate for public deployments, so a stray visitor
    can't run up the real Gemini/X API keys behind this app. Not real
    auth — good enough for a single-user demo, not for protecting
    anything sensitive. Skipped entirely if APP_PASSWORD isn't set
    (e.g. local dev).
    """
    if st.session_state.get("_authed"):
        return True

    app_password = _get_secret("APP_PASSWORD")
    if not app_password:
        return True

    st.title("🏀 NBA Live Agent")
    entered = st.text_input("Password", type="password")
    if entered == app_password:
        st.session_state["_authed"] = True
        st.rerun()
    elif entered:
        st.error("Wrong password.")
    return False


if not _check_password():
    st.stop()


def _init_state() -> None:
    # Restore the resolved game from the URL first, if present, so a page
    # reload (which wipes session_state but not the URL) lands back in the
    # QA phase instead of bouncing to "which game are you watching?" —
    # chat history itself doesn't survive a reload, only which game you'd
    # locked onto. Guarded on "resolved" not already being set so this only
    # fires once per session, not on every rerun.
    if "resolved" not in st.session_state and st.query_params.get("game_id"):
        st.session_state.resolved = True
        st.session_state.game_id = st.query_params.get("game_id")
        st.session_state.away_team = st.query_params.get("away_team")
        st.session_state.home_team = st.query_params.get("home_team")

    defaults = {
        "resolved": False,
        "resolve_messages": [],  # wire-format history, round-tripped with /resolve verbatim
        "resolve_chat_log": [],  # display-only bubbles
        "qa_chat_log": [],  # display-only bubbles — never sent to the API
        "resolve_pending": None,  # description awaiting a /resolve response, if any
        "qa_pending": None,  # question awaiting an /ask response, if any
        "game_id": None,
        "away_team": None,
        "home_team": None,
        "total_tokens": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _stream_turn(url: str, payload: dict) -> dict:
    """POSTs to a streaming (NDJSON) endpoint inside a live st.status() box
    — spinner while running, label updated as each tool_call event arrives
    — and returns whichever event ends the stream: {"type": "final", ...}
    or {"type": "error", ...}.
    """
    with st.status("Thinking...", state="running") as status_box:
        try:
            with requests.post(url, json=payload, stream=True, timeout=REQUEST_TIMEOUT) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    event = json.loads(line)
                    if event["type"] == "tool_call":
                        label = TOOL_STATUS_MESSAGES.get(event["name"], f"Calling {event['name']}...")
                        status_box.update(label=label)
                    elif event["type"] == "final":
                        status_box.update(label="Done", state="complete")
                        return event
                    else:  # "error"
                        status_box.update(label="Error", state="error")
                        return event
        except requests.RequestException as e:
            status_box.update(label="Error", state="error")
            return {"type": "error", "detail": f"Couldn't reach the backend: {e}"}
    return {"type": "error", "detail": "The backend closed the connection unexpectedly."}


_init_state()

with st.sidebar:
    st.header("🏀 NBA Live Agent")
    st.checkbox("Enable X commentary (costs money per call)", value=False, key="x_commentary")
    st.caption(f"Tokens used this session: {st.session_state.total_tokens:,}")
    if st.button("New session"):
        st.session_state.clear()
        st.query_params.clear()
        st.rerun()

if not st.session_state.resolved:
    st.title("Which game are you watching?")
    st.caption("e.g. \"Lakers vs Celtics\", or a specific date like \"Lakers Celtics from Jan 15\"")

    for entry in st.session_state.resolve_chat_log:
        with st.chat_message(entry["role"]):
            st.write(entry["content"])

    description = st.chat_input("Describe the game...", disabled=st.session_state.resolve_pending is not None)
    if description and st.session_state.resolve_pending is None:
        st.session_state.resolve_pending = description
        with st.chat_message("user"):
            st.write(description)
        st.session_state.resolve_chat_log.append({"role": "user", "content": description})
        st.rerun()

    if st.session_state.resolve_pending is not None:
        with st.chat_message("assistant"):
            final = _stream_turn(
                f"{API_BASE_URL}/resolve",
                {"messages": st.session_state.resolve_messages, "description": st.session_state.resolve_pending},
            )
            if final["type"] == "error":
                st.write(final["detail"])
                st.session_state.resolve_chat_log.append({"role": "assistant", "content": final["detail"]})
            else:
                st.write(final["reply"])
                st.session_state.resolve_messages = final["messages"]
                st.session_state.resolve_chat_log.append({"role": "assistant", "content": final["reply"]})
                st.session_state.total_tokens += final["tokens_used"]
                if final["resolved"]:
                    st.session_state.resolved = True
                    st.session_state.game_id = final["game_id"]
                    st.session_state.away_team = final["away_team"]
                    st.session_state.home_team = final["home_team"]
                    st.query_params["game_id"] = final["game_id"] or ""
                    st.query_params["away_team"] = final["away_team"] or ""
                    st.query_params["home_team"] = final["home_team"] or ""
        st.session_state.resolve_pending = None
        st.rerun()

else:
    st.title(f"{st.session_state.away_team} @ {st.session_state.home_team}")
    st.caption("Ask questions about the game. Each question is answered fresh — no memory of prior questions.")

    for entry in st.session_state.qa_chat_log:
        with st.chat_message(entry["role"]):
            st.write(entry["content"])

    question = st.chat_input("Ask a question...", disabled=st.session_state.qa_pending is not None)
    if question and st.session_state.qa_pending is None:
        st.session_state.qa_pending = question
        with st.chat_message("user"):
            st.write(question)
        st.session_state.qa_chat_log.append({"role": "user", "content": question})
        st.rerun()

    if st.session_state.qa_pending is not None:
        with st.chat_message("assistant"):
            final = _stream_turn(
                f"{API_BASE_URL}/ask",
                {
                    "game_id": st.session_state.game_id,
                    "away_team": st.session_state.away_team,
                    "home_team": st.session_state.home_team,
                    "question": st.session_state.qa_pending,
                    "x_commentary": st.session_state.x_commentary,
                },
            )
            if final["type"] == "error":
                st.write(final["detail"])
                st.session_state.qa_chat_log.append({"role": "assistant", "content": final["detail"]})
            else:
                st.write(final["answer"])
                st.session_state.qa_chat_log.append({"role": "assistant", "content": final["answer"]})
                st.session_state.total_tokens += final["tokens_used"]
        st.session_state.qa_pending = None
        st.rerun()
