# NBA Live Agent — Design

## Context

Built to close a specific, recurring gap: across internship applications, postings kept naming agent frameworks, tool calling, and LLM pipelines as a requirement or strong nice-to-have, while the only AI project on hand (Leaffliction) is computer vision/classical ML, not LLM/agent work. One posting (Ewake) states hands-on AI agent/LLM experience as a hard requirement and tests it directly in interview ("show us something you've built"). Rather than continue applying with the gap undemonstrated, this project closes it with a real, working agent.

## What it is

A CLI agent for reasoning about a live NBA game in progress. You tell it which game you're watching once at the start of a session; from then on you ask questions like "why isn't LeBron scoring this quarter?" and it pulls live play-by-play, reconstructs what actually happened, and gives a causal answer — not a stat dump.

The interesting agentic work is not the API call itself; it's turning a stream of raw play-by-play events into a causal explanation, and deciding when additional context (full-game shooting split, bench time) is needed to answer well.

## Scope (v1)

**In scope:**
- Single-player, in-game reasoning questions, scoped to one game per session
- Period-scoped primary answers, enriched with game-level context when the agent judges it useful
- CLI interface only

**Out of scope (v2+):**
- Team-level questions ("why did the Celtics go on that run")
- Trade evaluator / cross-game comparisons
- Web UI
- Persisting sessions across CLI runs

## Architecture

Python. LangGraph with a hand-built `StateGraph` (not the `create_react_agent` prebuilt) — two nodes:

- **`agent` node** — Gemini (via `langchain-google-genai`) with tools bound. Given the running message history, decides whether to call a tool or produce a final answer.
- **`tools` node** — executes whichever tool(s) the `agent` node requested (LangGraph's `ToolNode`).

Conditional edge from `agent`: if the last message contains tool calls, route to `tools`; otherwise route to `END`. Edge from `tools` always routes back to `agent`. This is the standard ReAct loop, built explicitly rather than via the prebuilt helper.

### Session model

At session start, the user names the game they're watching (e.g. "Lakers vs Celtics"). The agent resolves this once via `resolve_game` and holds the resulting `game_id` as session-level context for every subsequent question — the LLM doesn't re-resolve the game per question. This avoids re-litigating ambiguity on every turn and matches how a person actually watches a game: they already know what they're watching.

## Tools

```
resolve_game(query: str, date: str = today) -> GameResolution
```
Takes free text like "Lakers vs Celtics" or a team abbreviation, returns the matching `game_id` and both team names. If the query matches more than one game or none, returns a structured "not found / ambiguous" result rather than guessing.

```
get_play_by_play(game_id: str, period: int | None = None) -> list[PlayEvent]
```
Chronological events for the game, optionally filtered to one period. Used to reconstruct a player's shot attempts/outcomes, substitutions (time on/off court), and fouls within the requested window.

```
get_boxscore(game_id: str) -> BoxScore
```
Current live stats snapshot. Pulled on demand by the agent when game-level context would strengthen the answer (e.g. "he's 1-for-6 for the game, not just this period").

## Data source

`nba_api` (github.com/swar/nba_api) — free, open-source wrapper around NBA.com's live data feeds. Chosen over balldontlie.io, whose free tier excludes live box scores/play-by-play (paid tier is $9.99/mo).

**Caveat:** unofficial, technically against NBA.com's terms of use — common for side projects like this; worth being upfront about if asked in an interview.

## Error handling

| Condition | Behavior |
|---|---|
| Game not found for stated teams/date | Tell the user directly; ask them to double-check team names or confirm a game is happening today. No silent fallback to old/different data. |
| Ambiguous match (team abbreviation matches multiple franchises) | Ask a clarifying question rather than guessing. |
| Player not found on either roster in the resolved game | Tell the user; suggest checking the name/spelling. |
| Requested period hasn't happened yet | Say so plainly rather than fabricating an answer. |
| Game not yet started / no play-by-play available | Report that the game hasn't tipped off. |

## Tech stack

- Python (required by `nba_api`; also the primary/most mature SDK language for both `langchain-google-genai` and LangGraph)
- `langgraph`
- `langchain-google-genai` (Gemini)
- `nba_api`
- `python-dotenv` for API key management
- CLI: plain `input()`/`print()`, or `rich`/`typer` for nicer formatting — decide during implementation, not a design-level concern

## Example queries (v1 target set)

- "Why isn't LeBron scoring this quarter?"
- "How has Steph Curry been shooting in the second half?"
- "What's LeBron's shooting line for the game so far?"
- "Has Player X been on the bench a lot this period?"

## Testing

Manual/example-query smoke testing against live or recent games is the primary validation method for v1 — this is a portfolio/demo project, not a production system, so a full automated test suite isn't warranted. Worth a small set of unit tests around the tool functions' error-handling paths (game not found, ambiguous match) since those are easy to get wrong silently.
