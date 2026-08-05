from unittest.mock import patch

from nba_live_agent import nba_client
from nba_live_agent.nba_client import (
    GAME_STATUS_FINAL,
    GAME_STATUS_LIVE,
    GAME_STATUS_NOT_STARTED,
    _get_boxscore_raw,
    _get_play_by_play_raw,
    _resolve_game_raw,
)


def _fake_team(team_id, name, tricode):
    return {"teamId": team_id, "teamName": name, "teamTricode": tricode}


def test_resolve_game_not_found():
    with patch.object(nba_client.scoreboard, "ScoreBoard") as mock_sb:
        mock_sb.return_value.games.get_dict.return_value = [
            {
                "gameId": "0022500001",
                "homeTeam": _fake_team(1610612738, "Celtics", "BOS"),
                "awayTeam": _fake_team(1610612747, "Lakers", "LAL"),
                "gameStatus": GAME_STATUS_LIVE,
            }
        ]
        result = _resolve_game_raw("Nonsense Team", "today")

    assert result["status"] == "not_found"
    assert result["available_games"] == ["Lakers @ Celtics"]


def test_resolve_game_not_found_no_games_lists_nothing():
    with patch.object(nba_client.scoreboard, "ScoreBoard") as mock_sb:
        mock_sb.return_value.games.get_dict.return_value = []
        result = _resolve_game_raw("Nonsense Team", "today")

    assert result["status"] == "not_found"
    assert "available_games" not in result


def test_resolve_game_ambiguous():
    with patch.object(nba_client.scoreboard, "ScoreBoard") as mock_sb:
        mock_sb.return_value.games.get_dict.return_value = [
            {
                "gameId": "0022500001",
                "homeTeam": _fake_team(1610612747, "Lakers", "LAL"),
                "awayTeam": _fake_team(1610612738, "Celtics", "BOS"),
                "gameStatus": GAME_STATUS_LIVE,
            },
            {
                "gameId": "0022500002",
                "homeTeam": _fake_team(1610612746, "Clippers", "LAC"),
                "awayTeam": _fake_team(1610612744, "Warriors", "GSW"),
                "gameStatus": GAME_STATUS_LIVE,
            },
        ]
        result = _resolve_game_raw("Los Angeles", "today")

    assert result["status"] == "ambiguous"
    assert len(result["candidates"]) == 2


def test_resolve_game_no_games_today():
    with patch.object(nba_client.scoreboard, "ScoreBoard") as mock_sb:
        mock_sb.return_value.games.get_dict.return_value = []
        result = _resolve_game_raw("Lakers", "today")

    assert result["status"] == "not_found"


def test_resolve_game_unsupported_date():
    result = _resolve_game_raw("Lakers", "January 1st")
    assert result["status"] == "unsupported_date"


def _fake_game_header(game_id, home_id, away_id, status):
    return {
        "GAME_ID": game_id,
        "HOME_TEAM_ID": home_id,
        "VISITOR_TEAM_ID": away_id,
        "GAME_STATUS_ID": status,
    }


def test_resolve_game_past_date_ok():
    with patch.object(nba_client.scoreboardv2, "ScoreboardV2") as mock_sb:
        mock_sb.return_value.get_normalized_dict.return_value = {
            "GameHeader": [
                _fake_game_header("0022500001", 1610612738, 1610612747, GAME_STATUS_FINAL)
            ]
        }
        result = _resolve_game_raw("Lakers vs Celtics", "2026-01-15")

    mock_sb.assert_called_once_with(game_date="2026-01-15")
    assert result["status"] == "ok"
    assert result["game_id"] == "0022500001"
    assert result["game_status"] == GAME_STATUS_FINAL


def test_resolve_game_past_date_not_found():
    with patch.object(nba_client.scoreboardv2, "ScoreboardV2") as mock_sb:
        mock_sb.return_value.get_normalized_dict.return_value = {"GameHeader": []}
        result = _resolve_game_raw("Lakers", "2026-01-15")

    assert result["status"] == "not_found"


def test_play_by_play_period_not_played():
    with patch.object(nba_client, "_game_status", return_value=GAME_STATUS_LIVE), patch.object(
        nba_client.playbyplay, "PlayByPlay"
    ) as mock_pbp:
        mock_pbp.return_value.actions.get_dict.return_value = [
            {
                "period": 1,
                "clock": "PT00M00.00S",
                "teamTricode": "BOS",
                "playerNameI": "J. Brown",
                "actionType": "2pt",
                "subType": "jumpshot",
                "description": "J. Brown makes 2-pt shot",
                "scoreHome": "2",
                "scoreAway": "0",
            }
        ]
        result = _get_play_by_play_raw("0022500001", period=3)

    assert result["status"] == "period_not_played"
    assert result["events"] == []


def test_play_by_play_game_not_started():
    with patch.object(nba_client, "_game_status", return_value=GAME_STATUS_NOT_STARTED):
        result = _get_play_by_play_raw("0022500001", period=1)

    assert result["status"] == "game_not_started"
    assert result["events"] == []


def test_boxscore_game_not_started():
    with patch.object(nba_client.boxscore, "BoxScore") as mock_box:
        mock_box.return_value.game.get_dict.return_value = {"gameStatus": GAME_STATUS_NOT_STARTED}
        result = _get_boxscore_raw("0022500001")

    assert result["status"] == "game_not_started"


def test_boxscore_ok():
    player = {
        "name": "Jaylen Brown",
        "statistics": {
            "minutesCalculated": "PT25M",
            "points": 21,
            "assists": 8,
            "reboundsTotal": 2,
            "fieldGoalsMade": 6,
            "fieldGoalsAttempted": 12,
            "threePointersMade": 2,
            "threePointersAttempted": 5,
        },
    }
    with patch.object(nba_client.boxscore, "BoxScore") as mock_box:
        mock_box.return_value.game.get_dict.return_value = {
            "gameStatus": GAME_STATUS_FINAL,
            "homeTeam": {"teamName": "Celtics", "teamTricode": "BOS", "players": [player]},
            "awayTeam": {"teamName": "Lakers", "teamTricode": "LAL", "players": []},
        }
        result = _get_boxscore_raw("0022500001")

    assert result["status"] == "ok"
    assert result["home_players"][0]["name"] == "Jaylen Brown"
    assert result["home_players"][0]["points"] == 21
