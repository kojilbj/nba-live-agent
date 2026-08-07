"""Streamlit frontend for the NBA live-game analyst. Talks to the FastAPI
backend (src/nba_live_agent/api.py) over HTTP only — never imports
nba_live_agent internals, so it needs no PYTHONPATH/sys.path setup, same as
how run.py is a thin launcher for cli.py.
"""

import os

import requests
import streamlit as st

API_BASE_URL = os.environ.get("NBA_AGENT_API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 60

st.set_page_config(page_title="NBA Live Agent", page_icon="🏀", layout="centered")
st.logo("🏀", size="medium")


def _init_state() -> None:
    defaults = {
        "resolved": False,
        "resolve_messages": [],  # wire-format history, round-tripped with /resolve verbatim
        "resolve_chat_log": [],  # display-only bubbles
        "qa_chat_log": [],  # display-only bubbles — never sent to the API
        "game_id": None,
        "away_team": None,
        "home_team": None,
        "total_tokens": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()

with st.sidebar:
    st.header("🏀 NBA Live Agent")
    st.checkbox("Enable X commentary (costs money per call)", value=False, key="x_commentary")
    st.caption(f"Tokens used this session: {st.session_state.total_tokens:,}")
    if st.button("New session"):
        st.session_state.clear()
        st.rerun()

if not st.session_state.resolved:
    st.title("Which game are you watching?")
    st.caption("e.g. \"Lakers vs Celtics\", or a specific date like \"Lakers Celtics from Jan 15\"")

    for entry in st.session_state.resolve_chat_log:
        with st.chat_message(entry["role"]):
            st.write(entry["content"])

    description = st.chat_input("Describe the game...")
    if description:
        st.session_state.resolve_chat_log.append({"role": "user", "content": description})
        try:
            resp = requests.post(
                f"{API_BASE_URL}/resolve",
                json={"messages": st.session_state.resolve_messages, "description": description},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            st.session_state.resolve_chat_log.append(
                {"role": "assistant", "content": f"Couldn't reach the backend: {e}"}
            )
        else:
            st.session_state.resolve_messages = data["messages"]
            st.session_state.resolve_chat_log.append({"role": "assistant", "content": data["reply"]})
            st.session_state.total_tokens += data["tokens_used"]
            if data["resolved"]:
                st.session_state.resolved = True
                st.session_state.game_id = data["game_id"]
                st.session_state.away_team = data["away_team"]
                st.session_state.home_team = data["home_team"]
        st.rerun()

else:
    st.title(f"{st.session_state.away_team} @ {st.session_state.home_team}")
    st.caption("Ask questions about the game. Each question is answered fresh — no memory of prior questions.")

    for entry in st.session_state.qa_chat_log:
        with st.chat_message(entry["role"]):
            st.write(entry["content"])

    question = st.chat_input("Ask a question...")
    if question:
        st.session_state.qa_chat_log.append({"role": "user", "content": question})
        try:
            resp = requests.post(
                f"{API_BASE_URL}/ask",
                json={
                    "game_id": st.session_state.game_id,
                    "away_team": st.session_state.away_team,
                    "home_team": st.session_state.home_team,
                    "question": question,
                    "x_commentary": st.session_state.x_commentary,
                },
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            st.session_state.qa_chat_log.append({"role": "assistant", "content": f"Couldn't reach the backend: {e}"})
        else:
            st.session_state.qa_chat_log.append({"role": "assistant", "content": data["answer"]})
            st.session_state.total_tokens += data["tokens_used"]
        st.rerun()
