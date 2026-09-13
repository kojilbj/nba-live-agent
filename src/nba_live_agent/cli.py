"""Interactive CLI: resolve the game once, then answer each question as its
own fresh turn (system prompt + question only) rather than accumulating
conversation history — the only state carried across questions is the
resolved game_id/team names baked into the per-question system prompt, so
per-question cost doesn't grow with session length. No persistence across
process runs either way.
"""

import argparse
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
from langchain_core.messages import HumanMessage, SystemMessage

from nba_live_agent.agent import build_live_graph
from nba_live_agent.env_config import require_google_api_key
from nba_live_agent.logging_config import configure_logging
from nba_live_agent.session import (
    RESOLVE_SYSTEM_PROMPT_TEMPLATE,
    build_qa_prompt,
    build_qa_tools,
    extract_game_info,
    last_ai_message,
)
from nba_live_agent.tools import resolve_game

logger = logging.getLogger(__name__)


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
    print(last_ai_message(all_messages).text)
    print(f"[tokens — this turn: {turn_tokens:,} | session total: {session_usage['total_tokens']:,}]")
    return all_messages


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
    require_google_api_key()

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
    qa_graph = build_live_graph(tools=build_qa_tools(x_commentary))

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

        game_info = extract_game_info(messages)
        if game_info:
            break
        print("Let's try again — describe the game you're watching.")

    print("\nAsk questions about the game. Type 'quit' or 'exit' to end.\n")

    qa_prompt = build_qa_prompt(
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
