# nba-live-agent

A CLI agent that reasons about a live NBA game using a hand-built LangGraph ReAct loop (Gemini + tool calling) over the `nba_api` live data feed. See [`docs/superpowers/specs/2026-08-05-nba-live-agent-design.md`](docs/superpowers/specs/2026-08-05-nba-live-agent-design.md) for the full design.

## Setup

```bash
python3.11 -m venv .venv   # 3.11/3.12 recommended over very new Python releases
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # fill in GOOGLE_API_KEY
```

## Run

```bash
python run.py
```

## Test

```bash
pytest
```

## Project layout

- `src/nba_live_agent/nba_client.py` — plain wrapper functions around `nba_api`
- `src/nba_live_agent/models.py` — Pydantic schemas (`GameResolution`, `PlayEvent`, etc.)
- `src/nba_live_agent/tools.py` — LangGraph-bindable `@tool` functions
- `src/nba_live_agent/agent.py` — the hand-rolled agent/tools LangGraph loop
- `src/nba_live_agent/cli.py` — interactive session loop
- `tests/test_tools_errors.py` — offline unit tests for error-handling paths

## Manual smoke test (do this on your own machine, during/after a live game)

Development happened in a sandboxed environment whose network egress is blocked by NBA.com's CDN (Akamai returns 403 for the live `data.nba.com` endpoints, and `stats.nba.com` — used by `resolve_game` for non-today dates — just times out; general internet access works fine, so this is a network policy on NBA's end, not a code bug). That means none of `scoreboard`/`playbyplay`/`boxscore`/`scoreboardv2` were ever exercised against real data — only against the exact response schemas pulled from `nba_api`'s own source, and offline logic (team-name matching, error branching) was tested with mocks. Before trusting this for real:

1. `python run.py`, describe a real in-progress or recent game (e.g. "Lakers vs Celtics"), confirm it resolves to the right teams. If there's no live game right now (e.g. off-season), describe a specific past game instead (e.g. "Lakers vs Celtics from January 15" or "yesterday's Warriors game") — `resolve_game` now takes a `YYYY-MM-DD` date, and the agent should convert relative phrasing to a concrete date on its own.
2. Ask the design doc's example queries in the same session:
   - "Why isn't LeBron scoring this quarter?"
   - "How has Steph Curry been shooting in the second half?"
   - "What's LeBron's shooting line for the game so far?"
   - "Has Player X been on the bench a lot this period?"
3. Confirm a second question doesn't re-trigger `resolve_game` (the session should hold the game_id).
4. Force each error path once: a nonsense team name, a period beyond what's been played, a made-up player name.
5. Watch specifically for "second half" style questions — `get_play_by_play`'s `period` param is a single int, so "second half" isn't a single value it accepts natively; see how the agent handles this (multiple tool calls vs. asking for clarification) and treat any awkwardness there as a known gap, not a bug to silently patch.

If any live field name turns out to differ from what's in `nba_client.py` (print the raw dict from a real response to check), that's the one place likely to need adjustment — everything else in the pipeline is decoupled from `nba_api`'s exact JSON shape.
