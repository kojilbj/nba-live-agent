from nba_live_agent import cli
from nba_live_agent.tools import get_boxscore, get_hustle_stats, get_matchups, get_play_by_play, get_x_expert_insights


def test_build_qa_tools_excludes_x_by_default():
    tools = cli._build_qa_tools(x_commentary=False)

    assert get_x_expert_insights not in tools
    assert tools == [get_play_by_play, get_boxscore, get_matchups, get_hustle_stats]


def test_build_qa_tools_includes_x_when_enabled():
    tools = cli._build_qa_tools(x_commentary=True)

    assert get_x_expert_insights in tools
    assert tools == [
        get_play_by_play,
        get_boxscore,
        get_matchups,
        get_hustle_stats,
        get_x_expert_insights,
    ]


def test_build_qa_prompt_excludes_x_mentions_by_default():
    prompt = cli._build_qa_prompt(
        today="2026-01-16", game_id="G1", away_team="Celtics", home_team="Lakers", x_commentary=False
    )

    assert "get_x_expert_insights" not in prompt
    assert "get_play_by_play" in prompt
    assert "get_hustle_stats" in prompt


def test_build_qa_prompt_includes_x_when_enabled():
    prompt = cli._build_qa_prompt(
        today="2026-01-16", game_id="G1", away_team="Celtics", home_team="Lakers", x_commentary=True
    )

    assert "get_x_expert_insights" in prompt
    # Mentioned twice: once in the tool list, once in the multi-source example.
    assert prompt.count("get_x_expert_insights") == 2
