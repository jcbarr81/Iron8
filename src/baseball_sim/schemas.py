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
    model_version: str
    probs: Dict[str, float]
    sampled_event: str
    bip_detail: Optional[Dict[str, float]] = None
    advancement: Optional[Dict[str, object]] = None
    rng: Optional[Dict[str, int]] = None
