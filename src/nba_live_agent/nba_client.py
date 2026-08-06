"""Thin wrappers around nba_api. Plain functions, no LangGraph/LLM involved.

gameStatus convention (per nba_api's live scoreboard/boxscore payloads):
1 = not started, 2 = in progress, 3 = final.
"""

import logging
import time
from datetime import date as date_cls
from datetime import datetime, timedelta

from nba_api.live.nba.endpoints import boxscore, playbyplay, scoreboard
from nba_api.stats.endpoints import (
    boxscorehustlev2,
    boxscorematchupsv3,
    boxscoretraditionalv3,
    playbyplayv3,
    scoreboardv2,
)
from nba_api.stats.static import teams

logger = logging.getLogger(__name__)

try:
    from nba_api.stats.library.http import STATS_HEADERS

    STATS_HEADERS.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.nba.com/",
        "Accept-Language": "en-US,en;q=0.9",
        "x-nba-stats-origin": "stats",
        "x-nba-stats-token": "true",
    })
except Exception as exc:
    logger.warning("Failed to set custom STATS_HEADERS: %s", exc)


GAME_STATUS_NOT_STARTED = 1
GAME_STATUS_LIVE = 2
GAME_STATUS_FINAL = 3

_API_CACHE: dict[str, tuple[float, dict]] = {}
CACHE_TTL = 30.0
DEFAULT_TIMEOUT = 5


def _with_retry(func, retries: int = 3, initial_delay: float = 0.5, backoff_factor: float = 1.5):
    """Retries a callable up to `retries` times on Exception."""
    last_exc = None
    delay = initial_delay
    for attempt in range(retries):
        try:
            return func()
        except Exception as exc:
            last_exc = exc
            logger.warning("Attempt %d/%d failed: %s", attempt + 1, retries, exc)
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= backoff_factor
    raise last_exc


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


# Common informal names that don't derive from nba_api's official
# full_name/city/nickname/abbreviation fields (e.g. "NYC" isn't a substring
# of "New York" and isn't the Knicks' abbreviation "NYK"), so the substring
# check below can't catch them on its own. Keyed by informal word -> the
# team's official abbreviation.
INFORMAL_TEAM_ALIASES = {
    "nyc": "NYK",  # Nets are officially "Brooklyn", not "New York", so this isn't ambiguous
    "philly": "PHI",
    "sf": "GSW",
    "nola": "NOP",
}


def _matching_teams(query: str) -> list[dict]:
    q = _normalize(query)
    q_words = set(q.split())
    aliased_abbreviations = {INFORMAL_TEAM_ALIASES[w] for w in q_words if w in INFORMAL_TEAM_ALIASES}
    matches = []
    for team in teams.get_teams():
        identifiers = [
            _normalize(team["full_name"]),
            _normalize(team["nickname"]),
            _normalize(team["city"]),
            _normalize(team["abbreviation"]),
        ]
        if (
            any(identifier in q for identifier in identifiers)
            or team["abbreviation"].lower() in q_words
            or team["abbreviation"] in aliased_abbreviations
        ):
            matches.append(team)
    return matches


def _games_for_today_raw() -> dict:
    """Fetch today's slate from the live scoreboard feed.

    Returns {"status": "ok", "games": [...]} or an error dict shaped like
    _resolve_game_raw's return value.
    """
    try:
        sb = _with_retry(lambda: scoreboard.ScoreBoard(timeout=DEFAULT_TIMEOUT))
        games = sb.games.get_dict()
    except Exception as e:
        logger.exception("Failed to fetch today's live scoreboard")
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
        sb = _with_retry(lambda: scoreboardv2.ScoreboardV2(game_date=date, timeout=DEFAULT_TIMEOUT))
        game_headers = sb.get_normalized_dict()["GameHeader"]
    except Exception as e:
        logger.exception("Failed to fetch scoreboard for date=%s", date)
        return {
            "status": "api_error",
            "message": f"Couldn't reach NBA's stats data feed: {e}",
        }

    teams_by_id = {team["id"]: team for team in teams.get_teams()}

    games = []
    for gh in game_headers:
        home = teams_by_id.get(gh["HOME_TEAM_ID"])
        away = teams_by_id.get(gh["VISITOR_TEAM_ID"])
        if not home and not away:
            # Neither side is a known team - nothing usable to match or show.
            continue
        # Conditional/contingent playoff slots (e.g. a Finals "if necessary"
        # game) can be scheduled with only one side determined - HOME_TEAM_ID
        # comes back null from the API itself, not just unmapped here. Keep
        # the game findable by whichever side *is* known instead of
        # dropping it, with a placeholder for the undetermined side.
        games.append(
            {
                "gameId": gh["GAME_ID"],
                "homeTeam": {"teamId": home["id"], "teamName": home["nickname"]}
                if home
                else {"teamId": None, "teamName": "TBD"},
                "awayTeam": {"teamId": away["id"], "teamName": away["nickname"]}
                if away
                else {"teamId": None, "teamName": "TBD"},
                "gameStatus": gh["GAME_STATUS_ID"],
            }
        )
    return {"status": "ok", "games": games}


def _games_for_window_raw(center: date_cls, radius_days: int = 2) -> dict:
    """Fetch games across [center - radius_days, center + radius_days]
    (radius_days=2 -> 5 days total), tagging each game with the date it was
    fetched for (as "gameDate") so callers can disambiguate/display it.

    The center day goes through the live feed (lower latency, live status);
    the surrounding days go through the stats feed, one request per day.
    If a single day's fetch fails, that day is skipped (logged as a
    warning) rather than failing the whole window - only returns
    status="api_error" if every day in the window failed.

    The stats feed already retries each request 3x with backoff on its own
    (see _with_retry); a broad outage there (e.g. the Akamai tarpit quirk -
    see README) means every remaining stats-feed day in the window would
    otherwise repeat that same multi-retry timeout for nothing. Once one
    stats-feed day fails, the rest are skipped without retrying - the live
    feed (today) is unaffected and still tried normally.
    """
    games: list[dict] = []
    failures = 0
    total = 2 * radius_days + 1
    stats_feed_down = False

    for offset in range(-radius_days, radius_days + 1):
        day = center + timedelta(days=offset)
        day_str = day.strftime("%Y-%m-%d")

        if offset == 0:
            fetch = _games_for_today_raw()
        elif stats_feed_down:
            failures += 1
            logger.warning(
                "Skipping %s in resolve_game window: stats feed already failed earlier in this "
                "window search, not retrying",
                day_str,
            )
            continue
        else:
            fetch = _games_for_date_raw(day_str)

        if fetch["status"] != "ok":
            failures += 1
            logger.warning("Skipping %s in resolve_game window: %s", day_str, fetch.get("message"))
            if offset != 0:
                stats_feed_down = True
            continue

        for game in fetch["games"]:
            game["gameDate"] = day_str
        games.extend(fetch["games"])

    if failures == total:
        return {
            "status": "api_error",
            "message": f"Couldn't reach NBA's data feeds for any day in the {total}-day window around {center}.",
        }

    return {"status": "ok", "games": games}


def _match_games(games: list[dict], query: str, date_label: str, include_date_in_label: bool = False) -> dict:
    """Shared "filter games by query, build ok/not_found/ambiguous" logic
    used by both the date-window and single-date resolution paths.
    """

    def label(game: dict) -> str:
        base = _format_matchup(game)
        return f"{base} ({game['gameDate']})" if include_date_in_label else base

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
        available = [label(g) for g in games]
        return {
            "status": "not_found",
            "message": (
                f"No game for {date_label} matches '{query}'. Here's what's "
                f"actually on for {date_label}: {', '.join(available)}."
            ),
            "available_games": available,
        }

    if len(matches) > 1:
        candidates = [label(g) for g in matches]
        return {
            "status": "ambiguous",
            "message": f"'{query}' matches multiple games for {date_label}: {', '.join(candidates)}.",
            "candidates": candidates,
        }

    game = matches[0]
    result = {
        "status": "ok",
        "game_id": game["gameId"],
        "home_team": game["homeTeam"]["teamName"],
        "away_team": game["awayTeam"]["teamName"],
        "game_status": game["gameStatus"],
    }
    if include_date_in_label:
        result["game_date"] = game["gameDate"]
    return result


def _resolve_game_raw(query: str, date: str | None = None) -> dict:
    """Find a game matching a free-text query like "Lakers vs Celtics".

    date=None (the default): searches a 5-day window (today +/- 2 days) so
    callers don't need to guess an exact date - team name alone is usually
    enough. date="today": today's live slate only. date="YYYY-MM-DD": a
    specific past or future date only, when the user names one explicitly.

    Returns a dict with one of:
    - {"status": "ok", "game_id", "home_team", "away_team", "game_status",
      "game_date"} — game_date is only present for the date=None window path
    - {"status": "not_found", "message", "available_games": [...]} — games
      were scheduled in range, just none matched the query; lists them
      all so the caller can offer them as options instead of dead-ending
    - {"status": "ambiguous", "message", "candidates": [...]}
    - {"status": "unsupported_date", "message"}
    - {"status": "api_error", "message"}
    """
    if date is None:
        fetch = _games_for_window_raw(datetime.now().date())
        if fetch["status"] != "ok":
            return fetch
        return _match_games(fetch["games"], query, "the surrounding few days", include_date_in_label=True)

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
    return _match_games(fetch["games"], query, date_label)


def _game_status(game_id: str) -> int:
    box = _with_retry(lambda: boxscore.BoxScore(game_id, timeout=DEFAULT_TIMEOUT))
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
        pbp = _with_retry(lambda: playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=DEFAULT_TIMEOUT))
        actions = _rows_as_dicts(pbp.play_by_play)
    except Exception as e:
        logger.exception("Failed to fetch stats play-by-play for game_id=%s", game_id)
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

        pbp = _with_retry(lambda: playbyplay.PlayByPlay(game_id, timeout=DEFAULT_TIMEOUT))
        actions = pbp.actions.get_dict()
    except Exception as exc:
        logger.warning(
            "Live play-by-play unavailable for game_id=%s, falling back to stats feed: %s", game_id, exc
        )
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
        box = _with_retry(
            lambda: boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id, timeout=DEFAULT_TIMEOUT)
        )
        team_rows = _rows_as_dicts(box.team_stats)
        player_rows = _rows_as_dicts(box.player_stats)
    except Exception as e:
        logger.exception("Failed to fetch stats boxscore for game_id=%s", game_id)
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
        box = _with_retry(lambda: boxscore.BoxScore(game_id, timeout=DEFAULT_TIMEOUT))
        game = box.game.get_dict()
    except Exception as exc:
        logger.warning(
            "Live boxscore unavailable for game_id=%s, falling back to stats feed: %s", game_id, exc
        )
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


def _matchup_line(row: dict) -> dict:
    return {
        "defender_name": f"{row.get('firstNameDef', '')} {row.get('familyNameDef', '')}".strip(),
        "defender_team_tricode": row.get("teamTricode"),
        "matchup_minutes": row.get("matchupMinutes"),
        "partial_possessions": row.get("partialPossessions"),
        "points_allowed": row.get("playerPoints"),
        "field_goals_made_allowed": row.get("matchupFieldGoalsMade"),
        "field_goals_attempted_allowed": row.get("matchupFieldGoalsAttempted"),
    }


def _hustle_line_from_row(row: dict) -> dict:
    return {
        "name": f"{row.get('firstName', '')} {row.get('familyName', '')}".strip(),
        "team_tricode": row.get("teamTricode"),
        "screen_assists": row.get("screenAssists"),
        "deflections": row.get("deflections"),
        "charges_drawn": row.get("chargesDrawn"),
        "box_outs": row.get("boxOuts"),
        "contested_shots": row.get("contestedShots"),
        "loose_balls_recovered": row.get("looseBallsRecoveredTotal"),
    }


def _get_hustle_stats_raw(game_id: str) -> dict:
    """Per-player hustle-stat totals for a game (screen assists, deflections,
    charges drawn, box outs, contested shots, loose balls recovered) — NBA's
    "Hustle Stats" tracking data. These are per-player game totals only, not
    paired to a specific teammate/possession, so this can answer "who had
    the most screen assists" but not "who screened for X specifically" (see
    issue #16). stats.nba.com only, no live-feed equivalent to try first.
    """
    try:
        hustle = _with_retry(
            lambda: boxscorehustlev2.BoxScoreHustleV2(game_id=game_id, timeout=DEFAULT_TIMEOUT)
        )
        team_rows = _rows_as_dicts(hustle.team_stats)
        player_rows = _rows_as_dicts(hustle.player_stats)
    except Exception as e:
        logger.exception("Failed to fetch hustle stats for game_id=%s", game_id)
        return {
            "status": "api_error",
            "message": f"Couldn't reach NBA's stats data feed: {e}",
        }

    if len(team_rows) < 2:
        return {
            "status": "game_not_started",
            "message": "The game hasn't tipped off yet, or hustle stats aren't available for it.",
        }

    home_team, away_team = team_rows[0], team_rows[1]
    return {
        "status": "ok",
        "home_team": home_team["teamName"],
        "away_team": away_team["teamName"],
        "home_players": [
            _hustle_line_from_row(p) for p in player_rows if p["teamId"] == home_team["teamId"]
        ],
        "away_players": [
            _hustle_line_from_row(p) for p in player_rows if p["teamId"] == away_team["teamId"]
        ],
    }


def _get_matchups_raw(game_id: str, player_name: str) -> dict:
    """Per-defender breakdown of who guarded a given (offensive) player and
    for how long, sorted by time spent guarding them (most first). This is
    NBA Advanced Stats' "Matchups" tracking data — it only exists on
    stats.nba.com, there's no live-feed equivalent to try first.
    """
    try:
        matchups = _with_retry(
            lambda: boxscorematchupsv3.BoxScoreMatchupsV3(game_id=game_id, timeout=DEFAULT_TIMEOUT)
        )
        rows = _rows_as_dicts(matchups.player_stats)
    except Exception as e:
        logger.exception("Failed to fetch matchups for game_id=%s", game_id)
        return {
            "status": "api_error",
            "matchups": [],
            "message": f"Couldn't reach NBA's stats data feed: {e}",
        }

    if not rows:
        return {
            "status": "game_not_started",
            "matchups": [],
            "message": "The game hasn't tipped off yet, or matchup data isn't available for it.",
        }

    q = _normalize(player_name)
    target_rows = [
        r
        for r in rows
        if q in _normalize(f"{r.get('firstNameOff', '')} {r.get('familyNameOff', '')}")
        or q in _normalize(r.get("nameIOff") or "")
    ]

    if not target_rows:
        return {
            "status": "player_not_found",
            "matchups": [],
            "message": f"No player matching '{player_name}' found in this game.",
        }

    target_rows.sort(key=lambda r: r.get("matchupMinutesSort") or 0, reverse=True)
    matched_player = f"{target_rows[0].get('firstNameOff', '')} {target_rows[0].get('familyNameOff', '')}".strip()

    return {
        "status": "ok",
        "player_name": matched_player,
        "matchups": [_matchup_line(r) for r in target_rows],
    }
