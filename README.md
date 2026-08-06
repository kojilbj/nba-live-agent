# nba-live-agent

A CLI agent that reasons about a live or historical NBA game using a hand-built LangGraph ReAct loop (Gemini + tool calling) over the `nba_api` data feeds, with optional expert commentary pulled from X. You name the game you're watching once at the start of a session; from then on you ask questions like "why isn't LeBron scoring this quarter?" and it pulls play-by-play/boxscore data and gives a causal answer, not a stat dump.

## Demo

🎥 Coming soon.

## Setup

```bash
python3.11 -m venv .venv   # 3.11/3.12 recommended over very new Python releases
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # fill in GOOGLE_API_KEY (required); X_BEARER_TOKEN is optional
```

Without `X_BEARER_TOKEN`, `get_x_expert_insights` returns simulated expert commentary instead of hitting the real X API.

## Run

```bash
python run.py                  # interactive session, X commentary off
python run.py --verbose        # also print DEBUG-level logs to the console
python run.py --x-commentary   # enable get_x_expert_insights (costs real money if X_BEARER_TOKEN is set — see below)
```

**`get_x_expert_insights` is opt-in, off by default**, independent of whether `X_BEARER_TOKEN` is configured. X API v2 is pay-per-usage, not a flat subscription: **$0.005 per post read**, so a single call at `max_results=10` can cost up to $0.05 (see [X's pricing docs](https://docs.x.com/x-api/getting-started/pricing)). Caching (2-minute TTL) reduces repeat charges within a session, but it can still add up over a long session. Pass `--x-commentary` explicitly when you want it.

## Test

```bash
pytest
flake8
```

## How it works

Two-node LangGraph loop:

```mermaid
flowchart LR
    In(["messages"]) --> Agent
    Agent["agent node<br/>calls Gemini"] -- "tool_calls" --> Tools["tools node<br/>runs tools in parallel"]
    Tools -- "ToolMessage(s)<br/>appended to messages" --> Agent
    Agent -- "no tool_calls" --> Out(["final answer"])
```

- **`agent` node** — Gemini (via `langchain-google-genai`) with tools bound. Given the running message history, decides whether to call a tool or produce a final answer.
- **`tools` node** — executes whichever tool(s) the `agent` node requested, in parallel (via a thread pool) when the model requests more than one.

At session start you name the game (e.g. "Lakers vs Celtics", or a specific date for a past game); `resolve_game` resolves this once and the resulting `game_id` is held as context for every subsequent question in the session — the model doesn't re-resolve the game per question.

### Tools

- `resolve_game(query, date=None)` — free text → `game_id` + both team names. Date defaults to unset, which searches a 5-day window (today ± 2 days) so the model doesn't have to guess an exact date from relative phrasing; pass `date="YYYY-MM-DD"` only when the user names one explicitly. Returns a structured "not found" / "ambiguous" result (candidates tagged with their date) rather than guessing.
- `get_play_by_play(game_id, period=None)` — chronological events, optionally scoped to one period.
- `get_boxscore(game_id)` — current live stats snapshot for every player.
- `get_matchups(game_id, player_name)` — per-defender breakdown of who guarded a given player.
- `get_hustle_stats(game_id)` — screen assists, deflections, charges drawn, box outs, contested shots, loose balls recovered.
- `get_x_expert_insights(query)` — qualitative tactical commentary from a curated list of NBA analysts on X, for context raw stats don't explain (falls back to simulated posts without an `X_BEARER_TOKEN`, or if the real API call fails). Real-time/recent games only — see [X commentary](#x-commentary) below. **Opt-in via `--x-commentary`** (off by default, regardless of whether a token is set) since real calls cost money — see [Run](#run).

`nba_client.py` retries each `nba_api` call with backoff and falls back from the live feed to the historical stats feed when the live feed doesn't have a game anymore. All of this is decoupled from LangGraph — it's plain functions returning status dicts.

### Data source

[`nba_api`](https://github.com/swar/nba_api) — free, open-source wrapper around NBA.com's data feeds. It's unofficial and technically against NBA.com's terms of use, which is common for projects like this but worth being upfront about.

### X commentary

`get_x_expert_insights` calls X API v2's `search/recent` endpoint, which only searches posts from the **last ~7 days**. It works well for a live or very recent game, but returns no results for older games — e.g. querying it about a Finals game from a couple months back returns nothing, even with a valid, working `X_BEARER_TOKEN`. Pulling commentary on older games would require X's separate (and significantly more expensive) full-archive search product, which this project doesn't use. In practice this means: real-time and recent games only, not a general historical archive.

## Logging

By default the CLI prints only its own status/answer output; console logging stays at `WARNING`. Pass `--verbose`/`-v` to also print `DEBUG`-level logs (tool calls, retries, fallbacks) to the console. A `nba_live_agent.log` file (gitignored) always captures full `DEBUG` detail regardless of verbosity, for after-the-fact troubleshooting. See `src/nba_live_agent/logging_config.py`.

**Known quirk:** `stats.nba.com` sits behind Akamai, which can silently "tarpit" requests — the TCP connection succeeds instantly but no HTTP response ever arrives — instead of returning a fast error. This surfaces as a `ReadTimeout` after retries are exhausted; it's an external rate-limiting/anti-scraping behavior, not a bug in this code.

## Project layout

- `src/nba_live_agent/nba_client.py` — plain wrapper functions around `nba_api`
- `src/nba_live_agent/x_client.py` — X (Twitter) expert-commentary client, with mock fallback
- `src/nba_live_agent/models.py` — Pydantic schemas (`GameResolution`, `PlayEvent`, etc.)
- `src/nba_live_agent/tools.py` — LangGraph-bindable `@tool` functions
- `src/nba_live_agent/agent.py` — the hand-rolled agent/tools LangGraph loop
- `src/nba_live_agent/cli.py` — interactive session loop
- `src/nba_live_agent/logging_config.py` — logging setup (`--verbose`, log file)
- `tests/` — unit tests: mocked `nba_api`/X calls, error-handling and fallback paths, log-record assertions

## Manual smoke test

Automated tests mock every external call, so they don't catch a live API field renaming or an actual network-behavior change. Before trusting this against a real game:

1. `python run.py`, describe a real in-progress or recent game by team name alone (e.g. "Lakers") — `resolve_game` should find it via the ±2-day window without needing a date. Confirm a team with more than one game in that window comes back as numbered candidates instead of guessing. For a game clearly outside the window, name a specific date instead (e.g. "Lakers vs Celtics from January 15").
2. Ask a few of: "Why isn't LeBron scoring this quarter?", "How has Steph Curry been shooting in the second half?", "What's LeBron's shooting line for the game so far?", "Has Player X been on the bench a lot this period?"
3. Confirm a second question doesn't re-trigger `resolve_game` (the session should hold the `game_id`).
4. Force each error path once: a nonsense team name, a period beyond what's been played, a made-up player name.
5. If a live field name turns out to differ from what's in `nba_client.py`, run with `--verbose` (or check `nba_live_agent.log`) to see the raw response in the traceback.
