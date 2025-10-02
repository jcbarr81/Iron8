from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Optional, Literal

Half = Literal["T", "B"]

class Count(BaseModel):
    balls: int = Field(ge=0, le=3)
    strikes: int = Field(ge=0, le=2)

class Bases(BaseModel):
    # Use non-underscore field names; keep aliases "1B","2B","3B" for JSON IO
    B1: Optional[str] = Field(None, alias="1B")
    B2: Optional[str] = Field(None, alias="2B")
    B3: Optional[str] = Field(None, alias="3B")

    # Allow parsing by aliases and (optionally) serializing by field name
    model_config = ConfigDict(populate_by_name=True)

class Score(BaseModel):
    away: int
    home: int

class Weather(BaseModel):
    temp_f: Optional[float] = None
    wind_mph: Optional[float] = None
    wind_dir_deg: Optional[float] = None

class Rules(BaseModel):
    dh: Optional[bool] = True
    shift_restrictions: Optional[bool] = True
    extras_runner_on_2nd: Optional[bool] = True
    challenges_enabled: Optional[bool] = False
    overturn_rate: Optional[float] = 0.2
    three_batter_min: Optional[bool] = True

class State(BaseModel):
    inning: int
    half: Half
    outs: int
    bases: Bases
    score: Score
    count: Count
    park_id: Optional[str] = None
    weather: Optional[Weather] = None
    rules: Optional[Rules] = None

class Ratings(BaseModel):
    # Batter or pitcher fields (some may be None depending on role)
    ch: Optional[int] = None   # batter contact (if present)
    ph: Optional[int] = None   # batter power (if present)
    gf: Optional[int] = None
    pl: Optional[int] = None
    sc: Optional[int] = None
    sp: Optional[int] = None   # speed (baserunning/steals)
    fa: Optional[int] = None
    arm: Optional[int] = None

    # pitcher mix and traits
    control: Optional[int] = None
    movement: Optional[int] = None
    hold_runner: Optional[int] = None
    endurance: Optional[int] = None
    fb: Optional[int] = None
    sl: Optional[int] = None
    si: Optional[int] = None
    cu: Optional[int] = None
    cb: Optional[int] = None
    scb: Optional[int] = None
    kn: Optional[int] = None

    # alternative to "arm" if your file uses "as"
    as_: Optional[int] = Field(None, alias="as")

class Batter(BaseModel):
    player_id: str
    bats: Literal["L", "R"]
    ratings: Optional[Ratings] = None

class Pitcher(BaseModel):
    player_id: str
    throws: Literal["L", "R"]
    pitch_count: Optional[int] = None
    tto_index: Optional[int] = None
    batters_faced_in_stint: Optional[int] = None
    entered_cold: Optional[bool] = None
    ratings: Optional[Ratings] = None

class SimRequest(BaseModel):
    game_id: str
    state: State
    batter: Batter
    pitcher: Pitcher
    defense: Optional[Dict[str, int]] = None  # e.g., {"SS":7,"2B":1,"CF":8}
    return_probabilities: bool = True
    return_contact: bool = False
    seed: Optional[int] = None

class SimResponse(BaseModel):
    # allow field names like "model_version" without warnings
    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    probs: Dict[str, float]
    sampled_event: str
    bip_detail: Optional[Dict[str, object]] = None
    advancement: Optional[Dict[str, object]] = None
    rng: Optional[Dict[str, int]] = None
    trace_id: Optional[str] = None
    win_prob_home: Optional[float] = None
    leverage_index: Optional[float] = None


# Micro-events: Steal/PO/WP/PB
class Runner(BaseModel):
    player_id: Optional[str] = None
    ratings: Optional[Ratings] = None


class StealRequest(BaseModel):
    state: State
    runner_base: Literal["1B", "2B"]
    runner: Runner
    pitcher: Pitcher
    defense: Optional[Dict[str, object]] = None
    seed: Optional[int] = None


class StealResponse(BaseModel):
    event: Literal["SB", "CS", "PO", "WP", "PB"]
    outs_recorded: int
    bases_ending: Dict[str, Optional[str]]
    rng: Optional[Dict[str, int]] = None


# Pitch-by-pitch
class PitchRequest(BaseModel):
    game_id: str
    state: State
    batter: Batter
    pitcher: Pitcher
    defense: Optional[Dict[str, object]] = None
    return_contact: bool = True
    seed: Optional[int] = None
    # Optional umpire profile and pitch location hint
    umpire: Optional[Dict[str, float]] = None  # {"overall_bias": -1..1, "edge_bias": -1..1}
    edge: Optional[bool] = False


class PitchResponse(BaseModel):
    pitch_type: str
    result: Literal["ball", "called_strike", "swing_miss", "foul", "in_play", "hbp"]
    next_count: Count
    probs: Optional[Dict[str, float]] = None
    pre_pitch: Optional[Literal["no_play", "PO"]] = None
    pre_info: Optional[Dict[str, object]] = None
    bip_detail: Optional[Dict[str, object]] = None
    advancement: Optional[Dict[str, object]] = None
    rng: Optional[Dict[str, int]] = None


# Strategy advisories
class PinchHitRequest(BaseModel):
    state: State
    current_batter: Batter
    pitcher: Pitcher
    candidates: list[Batter]


class PinchHitResponse(BaseModel):
    recommend: bool
    best_candidate_id: Optional[str] = None
    score_delta: float = 0.0


class BullpenRequest(BaseModel):
    state: State
    upcoming_batter: Batter
    pitcher_candidates: list[Pitcher]


class BullpenResponse(BaseModel):
    recommend: bool
    best_pitcher_id: Optional[str] = None
    rationale: Optional[str] = None


class BuntRequest(BaseModel):
    state: State
    batter: Batter


class BuntResponse(BaseModel):
    recommend: bool
    rationale: Optional[str] = None


class IBBRequest(BaseModel):
    state: State
    batter: Batter


class IBBResponse(BaseModel):
    recommend: bool
    rationale: Optional[str] = None


# Game summaries and logs (stateless computation from provided logs)
class GameEvent(BaseModel):
    inning: int
    half: Literal["T", "B"]
    team: Literal["home", "away"]
    rbi: int = 0
    outs_recorded: int = 0
    error: Optional[bool] = None


class BoxscoreRequest(BaseModel):
    game_id: str
    home_team: str
    away_team: str
    log: list[GameEvent]


class BoxscoreResponse(BaseModel):
    game_id: str
    runs: Dict[str, int]
    earned_runs: Dict[str, int]
    outs: int
    winner: Optional[str] = None


class GameLogRequest(BaseModel):
    game_id: str
    log: list[GameEvent]


class GameLogResponse(BaseModel):
    game_id: str
    events: list[GameEvent]


# Three-batter minimum rules check
class ThreeBatterMinRequest(BaseModel):
    three_batter_min: Optional[bool] = True
    batters_faced_in_stint: int
    inning_ended: bool = False
    injury_exception: bool = False


class ThreeBatterMinResponse(BaseModel):
    allowed: bool
    reason: Optional[str] = None
