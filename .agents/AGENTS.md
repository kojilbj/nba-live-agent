# NBA Live Agent Guidelines

## Overview
`nba-live-agent` is a CLI agent that reasons about live and historical NBA games using a LangGraph ReAct loop powered by Gemini and `nba_api`.

## Project Structure & Architecture
- `src/nba_live_agent/nba_client.py`: Plain wrapper functions around `nba_api` data feeds.
- `src/nba_live_agent/models.py`: Pydantic schemas (`GameResolution`, `PlayEvent`, etc.).
- `src/nba_live_agent/tools.py`: `@tool` functions bound to LangGraph.
- `src/nba_live_agent/agent.py`: Hand-rolled agent/tools LangGraph loop.
- `src/nba_live_agent/cli.py`: Interactive CLI session loop.
- `src/nba_live_agent/logging_config.py`: `configure_logging()` — the CLI entry point's one place for wiring up log handlers; other modules just call `logging.getLogger(__name__)`.

## Development Rules
- Keep `nba_client.py` decoupled from LangGraph dependencies.
- Ensure all tool outputs handle network/API errors gracefully.
- Maintain test coverage in `tests/test_tools_errors.py`.
