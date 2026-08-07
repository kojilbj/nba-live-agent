"""Interactive CLI: resolve the game once, then answer each question as its
own fresh turn (system prompt + question only) rather than accumulating
conversation history — the only state carried across questions is the
resolved game_id/team names baked into the per-question system prompt, so
per-question cost doesn't grow with session length. No persistence across
process runs either way.
"""

import argparse
import json
import logging
from datetime import date

try:
    # Importing readline (unused directly) hooks input() into GNU
    # Readline/libedit for line editing — arrow-key cursor movement,
    # backspace-word, history. Without it, input() falls back to a raw
    # line read where left/right arrows print escape codes instead of
    # moving the cursor. Not available on Windows, whose console handles
    # line editing natively anyway.
    import readline  # noqa: F401
except ImportError:
    pass

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from nba_live_agent.agent import build_live_graph
from nba_live_agent.logging_config import configure_logging
from nba_live_agent.tools import (
    get_boxscore,
    get_hustle_stats,
    get_matchups,
    get_play_by_play,
    get_x_expert_insights,
    resolve_game,
)

logger = logging.getLogger(__name__)


RESOLVE_SYSTEM_PROMPT_TEMPLATE = (
    "You are an NBA in-game analyst. Today's date is {today}. Your only job "
    "right now is to resolve which game the user is watching via "
    "resolve_game. resolve_game's date param defaults to today's "
    "live/scheduled slate, but also accepts a specific YYYY-MM-DD date for a "
    "past or future game — if the user names a game that isn't today's "
    "(e.g. 'yesterday', 'last night', 'the Lakers Celtics game from January "
    "15'), convert that to a concrete date yourself and pass it. If the "
    "result is ambiguous or not found, don't just say so and stop — "
    "resolve_game's candidates/available_games field lists the real games "
    "on that date; present those as a numbered list ('1. Lakers @ "
    "Celtics', '2. ...') and ask the user to pick one, so they can respond "
    "with a number instead of having to type a team name precisely. Only "
    "fall back to asking them to re-describe the game if that list is "
    "empty too."
)

QA_SYSTEM_PROMPT_TEMPLATE = (
    "You are an NBA in-game analyst. Today's date is {today}. You are "
    "already locked onto a specific game — game_id={game_id} ({away_team} "
    "@ {home_team}) — so do not call resolve_game; use {tool_list} directly "
    "with this game_id or team/player names to answer questions. Give a causal, specific "
    "answer grounded in data. When analyzing performance, team strategy, or player dynamics, "
    "proactively query multiple relevant data sources together (e.g. combining boxscore/play-by-play"
    "{x_commentary_example}, or get_matchups for defensive coverage) "
    "to form a multi-angle response. Defensive matchup questions ('who guarded X the most') — call get_matchups. "
    "Hustle-stat questions — call get_hustle_stats. If the requested period hasn't been played "
    "yet, or a named player doesn't appear in the tool results, say so plainly instead of guessing."
)


TOOL_STATUS_MESSAGES = {
    "resolve_game": "Looking up the game...",
    "get_play_by_play": "Pulling play-by-play...",
    "get_boxscore": "Checking the boxscore...",
    "get_matchups": "Checking matchup data...",
    "get_hustle_stats": "Checking hustle stats...",
    "get_x_expert_insights": "Searching X for expert commentary...",
}


def _run_turn(graph, messages: list, session_usage: dict) -> list:

    """Streams the graph step by step instead of one blocking invoke() so we
    can print what the agent is doing (which tool, or "thinking") while it
    works, rather than leaving the terminal silent for several seconds.

    Prints the final answer itself (or an error) rather than leaving that to
    the caller, since a mid-turn failure (API quota, network) means there's
    no new AIMessage for the caller to print. Returns messages as far as the
    turn got, unchanged from the input on failure, so the session can
    continue instead of crashing.

    Each agent-node LLM call reports its own token usage via
    AIMessage.usage_metadata; summing those across the turn (and across
    turns via session_usage) gives real measured token counts instead of an
    estimate — a growing conversation resends its whole history every turn,
    so cost isn't just proportional to what you typed this time.
    """
    print("Thinking...")
    all_messages = list(messages)
    turn_tokens = 0
    try:
        for update in graph.stream({"messages": messages}, stream_mode="updates"):
            for node_name, node_output in update.items():
                new_messages = node_output["messages"]
                all_messages.extend(new_messages)
                if node_name == "agent":
                    last = new_messages[-1]
                    usage = getattr(last, "usage_metadata", None)
                    if usage:
                        turn_tokens += usage.get("total_tokens", 0)
                    for call in getattr(last, "tool_calls", []):
                        status = TOOL_STATUS_MESSAGES.get(call["name"], f"Calling {call['name']}...")
                        print(f"  {status}")
    except Exception as e:
        logger.exception("Graph execution failed")
        print(f"Something went wrong talking to the model: {e}")
        return all_messages

    session_usage["total_tokens"] += turn_tokens
    print(_last_ai_message(all_messages).text)
    print(f"[tokens — this turn: {turn_tokens:,} | session total: {session_usage['total_tokens']:,}]")
    return all_messages


def _last_ai_message(messages) -> AIMessage:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    raise RuntimeError("No AIMessage found in graph output.")


def _extract_game_info(messages) -> dict | None:
    """Pull the resolved game_id/team names out of resolve_game's ToolMessage,
    if resolution succeeded this turn. This is the one piece of state that
    carries forward across questions — everything else in the turn's
    messages (tool calls, raw play-by-play/boxscore payloads) is discarded
    once the answer is printed, so per-question cost doesn't grow with
    session length.
    """
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and message.name == "resolve_game":
            try:
                data = json.loads(message.content)
            except (json.JSONDecodeError, AttributeError):
                return None
            return data if data.get("status") == "ok" else None
    return None


def _build_qa_tools(x_commentary: bool) -> list:
    """The QA-phase tool list. get_x_expert_insights is opt-in (excluded by
    default) since each real call to X's API costs money (X API v2 is
    pay-per-usage — see README) once X_BEARER_TOKEN is configured; the
    --x-commentary flag is the only thing that turns it on, independent of
    whether a token happens to be set.
    """
    tools = [get_play_by_play, get_boxscore, get_matchups, get_hustle_stats]
    if x_commentary:
        tools.append(get_x_expert_insights)
    return tools


def _build_qa_prompt(*, today: str, game_id: str, away_team: str, home_team: str, x_commentary: bool) -> str:
    """Builds QA_SYSTEM_PROMPT_TEMPLATE's tool references to match whatever
    _build_qa_tools actually bound for this run — per build_graph's own
    contract, the model shouldn't be told about a tool it doesn't have.
    """
    tool_names = ["get_play_by_play", "get_boxscore", "get_matchups", "get_hustle_stats"]
    if x_commentary:
        tool_names.append("get_x_expert_insights")
    tool_list = ", ".join(tool_names[:-1]) + f", and {tool_names[-1]}"
    x_commentary_example = " with get_x_expert_insights for qualitative context" if x_commentary else ""

    return QA_SYSTEM_PROMPT_TEMPLATE.format(
        today=today,
        game_id=game_id,
        away_team=away_team,
        home_team=home_team,
        tool_list=tool_list,
        x_commentary_example=x_commentary_example,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive NBA live-game analyst CLI.")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print DEBUG-level logs to the console."
    )
    parser.add_argument(
        "--x-commentary",
        action="store_true",
        help=(
            "Enable the get_x_expert_insights tool (qualitative commentary from X). Off by "
            "default: X API v2 is pay-per-usage (~$0.005/post read, so up to ~$0.05 per call at "
            "max_results=10) once X_BEARER_TOKEN is configured -- this flag is the only thing "
            "that turns real spending on, independent of whether a token happens to be set. "
            "See README for details."
        ),
    )
    args = parser.parse_args()
    configure_logging(verbose=args.verbose)

    load_dotenv()

    run(verbose=args.verbose, x_commentary=args.x_commentary)


def run(verbose: bool, x_commentary: bool) -> None:
    today = date.today().isoformat()
    session_usage = {"total_tokens": 0}

    if not x_commentary:
        print("(X commentary tool is off — pass --x-commentary to enable it. See README for API cost info.)")

    # Separate graphs per phase so the model literally cannot call
    # get_boxscore/get_play_by_play while resolving the game (or
    # resolve_game once locked onto one) — a prompt instruction alone
    # doesn't reliably stop it from reaching for a tool it can still see.
    resolve_graph = build_live_graph(tools=[resolve_game])
    qa_graph = build_live_graph(tools=_build_qa_tools(x_commentary))

    print("Enter the team name(s) for the game you're watching (e.g. 'Lakers')")
    resolve_prompt = RESOLVE_SYSTEM_PROMPT_TEMPLATE.format(today=today)
    messages = [SystemMessage(resolve_prompt)]

    game_info = None
    while True:
        game_description = input("> ").strip()
        if not game_description:
            continue
        messages.append(HumanMessage(f"The game I'm watching is: {game_description}"))
        messages = _run_turn(resolve_graph, messages, session_usage)

        game_info = _extract_game_info(messages)
        if game_info:
            break
        print("Let's try again — describe the game you're watching.")

    print("\nAsk questions about the game. Type 'quit' or 'exit' to end.\n")

    qa_prompt = _build_qa_prompt(
        today=today,
        game_id=game_info["game_id"],
        away_team=game_info.get("away_team"),
        home_team=game_info.get("home_team"),
        x_commentary=x_commentary,
    )

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if question.lower() in ("quit", "exit"):
            print("Goodbye.")
            break
        if not question:
            continue

        # Fresh message list per question — only game_id/team names (baked
        # into qa_prompt) carry over, not prior turns' tool outputs.
        turn_messages = [SystemMessage(qa_prompt), HumanMessage(question)]
        _run_turn(qa_graph, turn_messages, session_usage)


if __name__ == "__main__":
    main()
