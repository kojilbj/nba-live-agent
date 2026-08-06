from unittest.mock import MagicMock, patch

from nba_live_agent import nba_client
from nba_live_agent.nba_client import (
    GAME_STATUS_FINAL,
    GAME_STATUS_LIVE,
    GAME_STATUS_NOT_STARTED,
    _get_boxscore_raw,
    _get_hustle_stats_raw,
    _get_matchups_raw,
    _get_play_by_play_raw,
    _matching_teams,
    _resolve_game_raw,
)


def _fake_team(team_id, name, tricode):
    return {"teamId": team_id, "teamName": name, "teamTricode": tricode}


def _fake_dataset(headers, rows):
    """A V3-stats-endpoint DataSet stub: .get_dict() -> {"headers", "data"}."""
    ds = MagicMock()
    ds.get_dict.return_value = {"headers": headers, "data": rows}
    return ds


def test_matching_teams_recognizes_informal_names():
    matches = {t["abbreviation"] for t in _matching_teams("NYC vs SAS")}
    assert matches == {"NYK", "SAS"}


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

    mock_sb.assert_called_once_with(game_date="2026-01-15", timeout=nba_client.DEFAULT_TIMEOUT)

    assert result["status"] == "ok"
    assert result["game_id"] == "0022500001"
    assert result["game_status"] == GAME_STATUS_FINAL


def test_resolve_game_past_date_not_found():
    with patch.object(nba_client.scoreboardv2, "ScoreboardV2") as mock_sb:
        mock_sb.return_value.get_normalized_dict.return_value = {"GameHeader": []}
        result = _resolve_game_raw("Lakers", "2026-01-15")

    assert result["status"] == "not_found"


def test_resolve_game_finds_conditional_game_with_tbd_home_team():
    """A conditional playoff slot (e.g. Finals "if necessary") can have
    HOME_TEAM_ID come back null straight from the API, with no home team
    at all in the data yet - not just something we failed to map. It should
    still be findable via the known (away) side, with home reported as TBD.
    """
    with patch.object(nba_client.scoreboardv2, "ScoreboardV2") as mock_sb:
        mock_sb.return_value.get_normalized_dict.return_value = {
            "GameHeader": [_fake_game_header("0042500405", None, 1610612752, GAME_STATUS_NOT_STARTED)]
        }
        result = _resolve_game_raw("Knicks", "2026-06-13")

    assert result["status"] == "ok"
    assert result["away_team"] == "Knicks"
    assert result["home_team"] == "TBD"


def test_resolve_game_drops_games_with_no_known_team_at_all():
    with patch.object(nba_client.scoreboardv2, "ScoreboardV2") as mock_sb:
        mock_sb.return_value.get_normalized_dict.return_value = {
            "GameHeader": [_fake_game_header("0042500406", None, None, GAME_STATUS_NOT_STARTED)]
        }
        result = _resolve_game_raw("Knicks", "2026-06-14")

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


def test_play_by_play_falls_back_to_stats_when_live_feed_fails():
    with (
        patch.object(nba_client, "_game_status", side_effect=ValueError("empty response")),
        patch.object(nba_client.playbyplayv3, "PlayByPlayV3") as mock_pbp,
    ):
        mock_pbp.return_value.play_by_play = _fake_dataset(
            headers=[
                "period", "clock", "teamTricode", "playerNameI", "actionType",
                "subType", "description", "scoreHome", "scoreAway",
            ],
            rows=[
                [1, "PT10M00.00S", "LAL", "R. Hachimura", "2pt", "jumpshot", "makes shot", "2", "0"],
            ],
        )
        result = _get_play_by_play_raw("0022500001", period=1)

    assert result["status"] == "ok"
    assert result["events"][0]["player_name"] == "R. Hachimura"


def test_play_by_play_stats_fallback_also_fails():
    with (
        patch.object(nba_client, "_game_status", side_effect=ValueError("empty response")),
        patch.object(
            nba_client.playbyplayv3, "PlayByPlayV3", side_effect=ValueError("also empty")
        ),
    ):
        result = _get_play_by_play_raw("0022500001")

    assert result["status"] == "api_error"


def test_boxscore_falls_back_to_stats_when_live_feed_fails():
    with (
        patch.object(nba_client.boxscore, "BoxScore", side_effect=ValueError("empty response")),
        patch.object(nba_client.boxscoretraditionalv3, "BoxScoreTraditionalV3") as mock_box,
    ):
        mock_box.return_value.team_stats = _fake_dataset(
            headers=["teamId", "teamName"],
            rows=[[100, "Lakers"], [200, "Thunder"]],
        )
        mock_box.return_value.player_stats = _fake_dataset(
            headers=[
                "teamId", "firstName", "familyName", "teamTricode", "minutes",
                "points", "assists", "reboundsTotal", "fieldGoalsMade",
                "fieldGoalsAttempted", "threePointersMade", "threePointersAttempted",
            ],
            rows=[
                [100, "Rui", "Hachimura", "LAL", "25:30", 18, 2, 5, 7, 12, 2, 4],
            ],
        )
        result = _get_boxscore_raw("0022500001")

    assert result["status"] == "ok"
    assert result["home_team"] == "Lakers"
    assert result["away_team"] == "Thunder"
    assert result["home_players"][0]["name"] == "Rui Hachimura"
    assert result["away_players"] == []


def test_boxscore_stats_fallback_also_fails():
    with (
        patch.object(nba_client.boxscore, "BoxScore", side_effect=ValueError("empty response")),
        patch.object(
            nba_client.boxscoretraditionalv3,
            "BoxScoreTraditionalV3",
            side_effect=ValueError("also empty"),
        ),
    ):
        result = _get_boxscore_raw("0022500001")

    assert result["status"] == "api_error"


def _fake_matchup_row(off_first, off_last, def_first, def_last, minutes_sort, **extra):
    row = {
        "firstNameOff": off_first,
        "familyNameOff": off_last,
        "nameIOff": f"{off_first[0]}. {off_last}",
        "firstNameDef": def_first,
        "familyNameDef": def_last,
        "teamTricode": extra.get("teamTricode", "SAS"),
        "matchupMinutes": extra.get("matchupMinutes", "5:00"),
        "matchupMinutesSort": minutes_sort,
        "partialPossessions": extra.get("partialPossessions", 4.0),
        "playerPoints": extra.get("playerPoints", 2),
        "matchupFieldGoalsMade": extra.get("matchupFieldGoalsMade", 1),
        "matchupFieldGoalsAttempted": extra.get("matchupFieldGoalsAttempted", 2),
    }
    return row


def test_matchups_sorted_by_time_guarded_most_first():
    rows = [
        _fake_matchup_row("Jalen", "Brunson", "De'Aaron", "Fox", 300, matchupMinutes="5:00"),
        _fake_matchup_row("Jalen", "Brunson", "Stephon", "Castle", 720, matchupMinutes="12:00"),
    ]
    with patch.object(nba_client.boxscorematchupsv3, "BoxScoreMatchupsV3") as mock_matchups:
        mock_matchups.return_value.player_stats = _fake_dataset(
            headers=list(rows[0].keys()), rows=[list(r.values()) for r in rows]
        )
        result = _get_matchups_raw("0022500001", "brunson")

    assert result["status"] == "ok"
    assert result["player_name"] == "Jalen Brunson"
    assert [m["defender_name"] for m in result["matchups"]] == ["Stephon Castle", "De'Aaron Fox"]


def test_matchups_player_not_found():
    rows = [_fake_matchup_row("Jalen", "Brunson", "De'Aaron", "Fox", 300)]
    with patch.object(nba_client.boxscorematchupsv3, "BoxScoreMatchupsV3") as mock_matchups:
        mock_matchups.return_value.player_stats = _fake_dataset(
            headers=list(rows[0].keys()), rows=[list(r.values()) for r in rows]
        )
        result = _get_matchups_raw("0022500001", "Nonexistent Player")

    assert result["status"] == "player_not_found"


def test_matchups_no_data_means_not_started():
    with patch.object(nba_client.boxscorematchupsv3, "BoxScoreMatchupsV3") as mock_matchups:
        mock_matchups.return_value.player_stats = _fake_dataset(headers=[], rows=[])
        result = _get_matchups_raw("0022500001", "Brunson")

    assert result["status"] == "game_not_started"


def test_matchups_api_error():
    with patch.object(
        nba_client.boxscorematchupsv3, "BoxScoreMatchupsV3", side_effect=ValueError("boom")
    ):
        result = _get_matchups_raw("0022500001", "Brunson")

    assert result["status"] == "api_error"


def test_hustle_stats_ok():
    with patch.object(nba_client.boxscorehustlev2, "BoxScoreHustleV2") as mock_hustle:
        mock_hustle.return_value.team_stats = _fake_dataset(
            headers=["teamId", "teamName"],
            rows=[[100, "Lakers"], [200, "Thunder"]],
        )
        mock_hustle.return_value.player_stats = _fake_dataset(
            headers=[
                "teamId", "firstName", "familyName", "teamTricode", "screenAssists",
                "deflections", "chargesDrawn", "boxOuts", "contestedShots",
                "looseBallsRecoveredTotal",
            ],
            rows=[
                [100, "Rui", "Hachimura", "LAL", 3, 2, 1, 4, 5, 2],
                [200, "Chet", "Holmgren", "OKC", 1, 4, 0, 6, 3, 1],
            ],
        )
        result = _get_hustle_stats_raw("0022500001")

    assert result["status"] == "ok"
    assert result["home_team"] == "Lakers"
    assert result["away_team"] == "Thunder"
    assert result["home_players"][0]["name"] == "Rui Hachimura"
    assert result["home_players"][0]["screen_assists"] == 3
    assert result["away_players"][0]["name"] == "Chet Holmgren"


def test_hustle_stats_no_data_means_not_started():
    with patch.object(nba_client.boxscorehustlev2, "BoxScoreHustleV2") as mock_hustle:
        mock_hustle.return_value.team_stats = _fake_dataset(headers=[], rows=[])
        mock_hustle.return_value.player_stats = _fake_dataset(headers=[], rows=[])
        result = _get_hustle_stats_raw("0022500001")

    assert result["status"] == "game_not_started"


def test_hustle_stats_api_error():
    with patch.object(
        nba_client.boxscorehustlev2, "BoxScoreHustleV2", side_effect=ValueError("boom")
    ):
        result = _get_hustle_stats_raw("0022500001")

    assert result["status"] == "api_error"
