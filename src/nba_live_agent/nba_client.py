"""Thin wrappers around nba_api. Plain functions, no LangGraph/LLM involved.

gameStatus convention (per nba_api's live scoreboard/boxscore payloads):
1 = not started, 2 = in progress, 3 = final.
"""

from datetime import datetime

from nba_api.live.nba.endpoints import boxscore, playbyplay, scoreboard
from nba_api.stats.endpoints import boxscoretraditionalv3, playbyplayv3, scoreboardv2
from nba_api.stats.static import teams

GAME_STATUS_NOT_STARTED = 1
GAME_STATUS_LIVE = 2
GAME_STATUS_FINAL = 3


def _normalize(text: str) -> str:
    return text.lower().strip()


def _format_matchup(game: dict) -> str:
    return f"{game['awayTeam']['teamName']} @ {game['homeTeam']['teamName']}"


def _rows_as_dicts(data_set) -> list[dict]:
    """V3 stats endpoints (unlike the legacy resultSets-based ones) return a
    plain headers/data table per dataset rather than something
    get_normalized_dict() understands, so zip them ourselves."""
    raw = data_set.get_dict()
    return [dict(zip(raw["headers"], row)) for row in raw["data"]]


def _matching_teams(query: str) -> list[dict]:
    q = _normalize(query)
    q_words = set(q.split())
    matches = []
    for team in teams.get_teams():
        identifiers = [
            _normalize(team["full_name"]),
            _normalize(team["nickname"]),
            _normalize(team["city"]),
            _normalize(team["abbreviation"]),
        ]
        if any(identifier in q for identifier in identifiers) or team["abbreviation"].lower() in q_words:
            matches.append(team)
    return matches


def _games_for_today_raw() -> dict:
    """Fetch today's slate from the live scoreboard feed.

    Returns {"status": "ok", "games": [...]} or an error dict shaped like
    _resolve_game_raw's return value.
    """
    try:
        sb = scoreboard.ScoreBoard()
        games = sb.games.get_dict()
    except Exception as e:
        return {
            "status": "api_error",
            "message": f"Couldn't reach NBA's live data feed: {e}",
        }
    return {"status": "ok", "games": games}


def _games_for_date_raw(date: str) -> dict:
    """Fetch a specific date's slate from the stats scoreboard feed (works for
    past and future dates; the live feed only ever has today's).

    Returns games in the same shape as the live feed (gameId, homeTeam/
    awayTeam with teamId + teamName, gameStatus) so callers don't need to
    care which feed a game came from.
    """
    try:
        # ScoreboardV2 is deprecated in favor of V3 over a LineScore bug for
        # 2025-10-22..2025-12-25 games; we only read GameHeader (team ids +
        # status), which isn't affected, and V2's normalized dict is far
        # simpler to work with than V3's headers/rows tables.
        sb = scoreboardv2.ScoreboardV2(game_date=date)
        game_headers = sb.get_normalized_dict()["GameHeader"]
    except Exception as e:
        return {
            "status": "api_error",
            "message": f"Couldn't reach NBA's stats data feed: {e}",
        }

    teams_by_id = {team["id"]: team for team in teams.get_teams()}
    games = []
    for gh in game_headers:
        home = teams_by_id.get(gh["HOME_TEAM_ID"])
        away = teams_by_id.get(gh["VISITOR_TEAM_ID"])
        if not home or not away:
            continue
        games.append(
            {
                "gameId": gh["GAME_ID"],
                "homeTeam": {"teamId": home["id"], "teamName": home["nickname"]},
                "awayTeam": {"teamId": away["id"], "teamName": away["nickname"]},
                "gameStatus": gh["GAME_STATUS_ID"],
            }
        )
    return {"status": "ok", "games": games}


def _resolve_game_raw(query: str, date: str = "today") -> dict:
    """Find a game matching a free-text query like "Lakers vs Celtics" on the
    given date (today's live slate, or a specific YYYY-MM-DD date — past or
    future, live or already final).

    Returns a dict with one of:
    - {"status": "ok", "game_id", "home_team", "away_team", "game_status"}
    - {"status": "not_found", "message", "available_games": [...]} — games
      were scheduled on that date, just none matched the query; lists them
      all so the caller can offer them as options instead of dead-ending
    - {"status": "ambiguous", "message", "candidates": [...]}
    - {"status": "unsupported_date", "message"}
    - {"status": "api_error", "message"}
    """
    if date == "today":
        fetch = _games_for_today_raw()
        date_label = "today"
    else:
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            return {
                "status": "unsupported_date",
                "message": f"'{date}' isn't a date I understand; use YYYY-MM-DD or 'today'.",
            }
        fetch = _games_for_date_raw(date)
        date_label = date

    if fetch["status"] != "ok":
        return fetch
    games = fetch["games"]

    if not games:
        return {
            "status": "not_found",
            "message": f"No NBA games are scheduled for {date_label}.",
        }

    query_teams = {t["id"] for t in _matching_teams(query)}
    matches = []
    for game in games:
        home_id = game["homeTeam"]["teamId"]
        away_id = game["awayTeam"]["teamId"]
        if query_teams & {home_id, away_id}:
            matches.append(game)

    if not matches:
        available = [_format_matchup(g) for g in games]
        return {
            "status": "not_found",
            "message": (
                f"No game on {date_label} matches '{query}'. Here's what's "
                f"actually on {date_label}: {', '.join(available)}."
            ),
            "available_games": available,
        }

    if len(matches) > 1:
        candidates = [_format_matchup(g) for g in matches]
        return {
            "status": "ambiguous",
            "message": f"'{query}' matches multiple games on {date_label}: {', '.join(candidates)}.",
            "candidates": candidates,
        }

    game = matches[0]
    return {
        "status": "ok",
        "game_id": game["gameId"],
        "home_team": game["homeTeam"]["teamName"],
        "away_team": game["awayTeam"]["teamName"],
        "game_status": game["gameStatus"],
    }


def _game_status(game_id: str) -> int:
    box = boxscore.BoxScore(game_id)
    return box.game.get_dict()["gameStatus"]


def _events_from_actions(actions: list[dict], period: int | None) -> dict:
    current_period = max((a["period"] for a in actions), default=0)
    if period is not None and period > current_period:
        return {
            "status": "period_not_played",
            "events": [],
            "message": f"Period {period} hasn't happened yet (current period: {current_period}).",
        }

    if period is not None:
        actions = [a for a in actions if a["period"] == period]

    events = [
        {
            "period": a["period"],
            "clock": a["clock"],
            "team_tricode": a.get("teamTricode"),
            "player_name": a.get("playerNameI"),
            "action_type": a["actionType"],
            "sub_type": a.get("subType"),
            "description": a["description"],
            "score_home": a["scoreHome"],
            "score_away": a["scoreAway"],
        }
        for a in actions
    ]
    return {"status": "ok", "events": events}


def _get_play_by_play_via_stats_raw(game_id: str, period: int | None) -> dict:
    """Fallback for games the live feed no longer has (see
    _get_play_by_play_raw). stats.nba.com keeps full-season history, so this
    works for any past game regardless of age — at the cost of not reflecting
    a game that's live right now, which the live feed would.
    """
    try:
        pbp = playbyplayv3.PlayByPlayV3(game_id=game_id)
        actions = _rows_as_dicts(pbp.play_by_play)
    except Exception as e:
        return {
            "status": "api_error",
            "events": [],
            "message": f"Couldn't reach NBA's stats data feed either: {e}",
        }

    if not actions:
        return {
            "status": "game_not_started",
            "events": [],
            "message": "The game hasn't tipped off yet.",
        }

    return _events_from_actions(actions, period)


def _get_play_by_play_raw(game_id: str, period: int | None = None) -> dict:
    """Chronological play events for a game, optionally filtered to one
    period. Tries the live data feed first (lowest latency for a game
    that's actually in progress); falls back to the stats feed's
    historical play-by-play when the live feed doesn't have this game
    anymore — confirmed via an empty response for games as little as a few
    months old, so this isn't a rare edge case for anything but very
    recent games.
    """
    try:
        status = _game_status(game_id)
        if status == GAME_STATUS_NOT_STARTED:
            return {
                "status": "game_not_started",
                "events": [],
                "message": "The game hasn't tipped off yet.",
            }

        pbp = playbyplay.PlayByPlay(game_id)
        actions = pbp.actions.get_dict()
    except Exception:
        return _get_play_by_play_via_stats_raw(game_id, period)

    return _events_from_actions(actions, period)


def _player_line(player: dict, team_tricode: str) -> dict:
    stats = player["statistics"]
    return {
        "name": player["name"],
        "team_tricode": team_tricode,
        "minutes": stats["minutesCalculated"],
        "points": stats["points"],
        "assists": stats["assists"],
        "rebounds": stats["reboundsTotal"],
        "field_goals_made": stats["fieldGoalsMade"],
        "field_goals_attempted": stats["fieldGoalsAttempted"],
        "three_pointers_made": stats["threePointersMade"],
        "three_pointers_attempted": stats["threePointersAttempted"],
    }


def _player_line_from_stats(row: dict) -> dict:
    return {
        "name": f"{row.get('firstName', '')} {row.get('familyName', '')}".strip(),
        "team_tricode": row.get("teamTricode"),
        "minutes": row.get("minutes"),
        "points": row.get("points"),
        "assists": row.get("assists"),
        "rebounds": row.get("reboundsTotal"),
        "field_goals_made": row.get("fieldGoalsMade"),
        "field_goals_attempted": row.get("fieldGoalsAttempted"),
        "three_pointers_made": row.get("threePointersMade"),
        "three_pointers_attempted": row.get("threePointersAttempted"),
    }


def _get_boxscore_via_stats_raw(game_id: str) -> dict:
    """Fallback for games the live feed no longer has (see _get_boxscore_raw).

    PlayerStats rows don't mark home/away directly; TeamStats always lists
    home before away (per nba_api's own parser), so we read team identity
    from there and partition players by teamId.
    """
    try:
        box = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
        team_rows = _rows_as_dicts(box.team_stats)
        player_rows = _rows_as_dicts(box.player_stats)
    except Exception as e:
        return {
            "status": "api_error",
            "message": f"Couldn't reach NBA's stats data feed either: {e}",
        }

    if len(team_rows) < 2:
        return {"status": "game_not_started", "message": "The game hasn't tipped off yet."}

    home_team, away_team = team_rows[0], team_rows[1]
    return {
        "status": "ok",
        "home_team": home_team["teamName"],
        "away_team": away_team["teamName"],
        "home_players": [
            _player_line_from_stats(p) for p in player_rows if p["teamId"] == home_team["teamId"]
        ],
        "away_players": [
            _player_line_from_stats(p) for p in player_rows if p["teamId"] == away_team["teamId"]
        ],
    }


def _get_boxscore_raw(game_id: str) -> dict:
    """Current live stats snapshot for a game. Tries the live data feed
    first; falls back to the stats feed's historical boxscore when the live
    feed doesn't have this game anymore (see _get_play_by_play_raw for why).
    """
    try:
        box = boxscore.BoxScore(game_id)
        game = box.game.get_dict()
    except Exception:
        return _get_boxscore_via_stats_raw(game_id)

    if game["gameStatus"] == GAME_STATUS_NOT_STARTED:
        return {"status": "game_not_started", "message": "The game hasn't tipped off yet."}

    home = game["homeTeam"]
    away = game["awayTeam"]
    return {
        "status": "ok",
        "home_team": home["teamName"],
        "away_team": away["teamName"],
        "home_players": [_player_line(p, home["teamTricode"]) for p in home["players"]],
        "away_players": [_player_line(p, away["teamTricode"]) for p in away["players"]],
    }
