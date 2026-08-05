from typing import Literal

from pydantic import BaseModel


class GameResolution(BaseModel):
    status: Literal["ok", "not_found", "ambiguous", "unsupported_date", "api_error"]
    game_id: str | None = None
    home_team: str | None = None
    away_team: str | None = None
    message: str | None = None
    candidates: list[str] = []
    available_games: list[str] = []


class PlayEvent(BaseModel):
    period: int
    clock: str
    team_tricode: str | None
    player_name: str | None
    action_type: str
    sub_type: str | None
    description: str
    score_home: str
    score_away: str


class PlayByPlayResult(BaseModel):
    status: Literal["ok", "period_not_played", "game_not_started", "api_error"]
    events: list[PlayEvent] = []
    message: str | None = None


class PlayerLine(BaseModel):
    name: str
    team_tricode: str
    minutes: str
    points: int
    assists: int
    rebounds: int
    field_goals_made: int
    field_goals_attempted: int
    three_pointers_made: int
    three_pointers_attempted: int


class BoxScoreResult(BaseModel):
    status: Literal["ok", "game_not_started", "api_error"]
    home_team: str | None = None
    away_team: str | None = None
    home_players: list[PlayerLine] = []
    away_players: list[PlayerLine] = []
    message: str | None = None


class MatchupLine(BaseModel):
    defender_name: str
    defender_team_tricode: str | None
    matchup_minutes: str | None
    partial_possessions: float | None
    points_allowed: int | None
    field_goals_made_allowed: int | None
    field_goals_attempted_allowed: int | None


class MatchupsResult(BaseModel):
    status: Literal["ok", "game_not_started", "player_not_found", "api_error"]
    player_name: str | None = None
    matchups: list[MatchupLine] = []
    message: str | None = None


class HustleLine(BaseModel):
    name: str
    team_tricode: str | None
    screen_assists: int | None
    deflections: int | None
    charges_drawn: int | None
    box_outs: int | None
    contested_shots: int | None
    loose_balls_recovered: int | None


class HustleStatsResult(BaseModel):
    status: Literal["ok", "game_not_started", "api_error"]
    home_team: str | None = None
    away_team: str | None = None
    home_players: list[HustleLine] = []
    away_players: list[HustleLine] = []
    message: str | None = None
