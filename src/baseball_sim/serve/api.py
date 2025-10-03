from fastapi import FastAPI
from pathlib import Path
from ..schemas import (
    SimRequest,
    SimResponse,
    StealRequest,
    StealResponse,
    BoxscoreRequest,
    BoxscoreResponse,
    GameLogRequest,
    GameLogResponse,
    PitchRequest,
    PitchResponse,
    ThreeBatterMinRequest,
    ThreeBatterMinResponse,
    PinchHitRequest,
    PinchHitResponse,
    BullpenRequest,
    BullpenResponse,
    BuntRequest,
    BuntResponse,
    IBBRequest,
    IBBResponse,
)
from ..features.ratings_adapter import batter_features, pitcher_features
from ..features.context_features import context_features
from ..calibration.calibrators import IdentityCalibrator
from ..models.pa_model import PaOutcomeModel
from ..models.bip_model import BipModel
from ..models.advancement_model import FieldingConverter, BaserunningAdvancer
from ..models.micro_events import MicroEvents
from ..models.pitch_engine import PitchEngine
from ..utils.wp import win_prob_home as _wp_home, leverage_index as _lev_index
import uuid
from ..features.transforms import z_0_99
from ..sampler.rng import sample_event
from ..utils.players import load_players_csv, get_ratings_for
from ..utils.parks import load_parks_yaml, features_for_park, fence_at_angle
from ..utils.fatigue import apply_fatigue
from ..serve.onnx_runtime import OnnxRunner
from ..config import load_settings
import os
from ..utils.strategy import advise_pinch_hit, advise_bullpen, advise_bunt, advise_ibb

app = FastAPI()
_pa_model = PaOutcomeModel()
_calib = IdentityCalibrator()
_bip_model = BipModel()
_fielding = FieldingConverter()
_advancer = BaserunningAdvancer()
_micro = MicroEvents()
_pitch_engine = PitchEngine()
_onnx_runner = None

# Attempt to load trained model at server import; fall back silently if missing.
try:
    _ROOT = Path(__file__).resolve().parents[3]
    _MODEL_PATH = _ROOT / "artifacts" / "pa_model.joblib"
    if _MODEL_PATH.exists():
        _pa_model.load(str(_MODEL_PATH))
except Exception:
    # Keep baseline behavior if load fails
    pass

# Attempt to load calibrator if present; otherwise remain identity
try:
    _CAL_PATH = _ROOT / "artifacts" / "pa_calibrator.joblib"
    if _CAL_PATH.exists():
        from joblib import load as _joblib_load  # lazy import

        _cal_obj = _joblib_load(str(_CAL_PATH))
        # Must provide an apply(probs, slice_meta) -> probs mapping
        if hasattr(_cal_obj, "apply"):
            _calib = _cal_obj
except Exception:
    pass

# Attempt to load ONNX runner if configured
try:
    _settings = load_settings()
    if getattr(_settings, "use_onnx", False):
        _ONNX_PATH = _ROOT / "artifacts" / "pa_model.onnx"
        _FEATS_JSON = _ROOT / "artifacts" / "pa_feature_names.json"
        if _ONNX_PATH.exists() and _FEATS_JSON.exists():
            import json

            data = json.loads(_FEATS_JSON.read_text(encoding="utf-8"))
            feats = data.get("feature_names", [])
            classes = data.get("class_names", []) or getattr(_pa_model, "_classes", [])
            _onnx_runner = OnnxRunner(str(_ONNX_PATH), feature_names=feats, class_names=classes)
except Exception:
    _onnx_runner = None

# Load players cache (ratings) from CSV; safe if missing
_PLAYERS_CACHE = {}
try:
    _PLAYERS_PATH = _ROOT / "data" / "players.csv"
    _PLAYERS_CACHE = load_players_csv(str(_PLAYERS_PATH))
    print(f"[serve.api] Loaded {len(_PLAYERS_CACHE)} players from {_PLAYERS_PATH}")
except Exception:
    _PLAYERS_CACHE = {}

# Load parks config; safe if missing
_PARKS_CFG = {}
try:
    _PARKS_PATH = _ROOT / "config" / "parks.yaml"
    _PARKS_CFG = load_parks_yaml(str(_PARKS_PATH))
    if _PARKS_CFG:
        print(f"[serve.api] Loaded parks config from {_PARKS_PATH}")
except Exception:
    _PARKS_CFG = {}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/v1/sim/plate-appearance", response_model=SimResponse)
def plate_appearance(req: SimRequest):
    # Prepare batter/pitcher dicts; inject ratings from cache if missing
    b_dict = req.batter.model_dump()
    p_dict = req.pitcher.model_dump()

    if b_dict.get("ratings") is None:
        br = get_ratings_for(b_dict.get("player_id"), _PLAYERS_CACHE)
        if br:
            b_dict["ratings"] = br

    if p_dict.get("ratings") is None:
        pr = get_ratings_for(p_dict.get("player_id"), _PLAYERS_CACHE)
        if pr:
            # Normalize potential alias 'as' -> 'arm'
            if "arm" not in pr and "as" in pr:
                pr = {**pr, "arm": pr.get("as")}
                pr.pop("as", None)
            else:
                pr = {k: v for k, v in pr.items() if k != "as"}
            p_dict["ratings"] = pr

    # Build feature dict
    b_feats = batter_features(b_dict)
    p_feats = pitcher_features(p_dict)
    c_feats = context_features(req.state, req.pitcher, req.batter)
    park_feats = {}
    try:
        park_feats = features_for_park(req.state.park_id, _PARKS_CFG)
    except Exception:
        park_feats = {}
    feats = {**b_feats, **p_feats, **c_feats, **park_feats}
    # Apply pitcher fatigue (pitch_count, TTO, warm-up)
    feats = apply_fatigue(
        feats,
        pitch_count=req.pitcher.pitch_count,
        tto_index=req.pitcher.tto_index,
        entered_cold=getattr(req.pitcher, "entered_cold", None),
        batters_faced_in_stint=getattr(req.pitcher, "batters_faced_in_stint", None),
    )
    # Ensure speed feature present if ratings include 'sp'
    if "bat_speed_z" not in feats:
        try:
            sp = None
            if req.batter.ratings and getattr(req.batter.ratings, "sp", None) is not None:
                sp = getattr(req.batter.ratings, "sp")
            elif isinstance(b_dict.get("ratings"), dict):
                sp = b_dict["ratings"].get("sp")
            if sp is not None:
                feats["bat_speed_z"] = z_0_99(sp)
        except Exception:
            pass

    # Apply RISP gating: batter RISP bonus only counts when a runner is on 2B or 3B
    if "bat_risp_z" in feats:
        try:
            risp_gate = float(c_feats.get("ctx_is_risp", 0) or 0.0)
        except Exception:
            risp_gate = 0.0
        feats["bat_risp_z"] = float(feats["bat_risp_z"] or 0.0) * risp_gate

    # TODO(Codex): add more context features (park/rules/handedness effects)

    # Predict probabilities via ONNX if available, else sklearn model/baseline
    if _onnx_runner is not None:
        probs = _onnx_runner.predict_proba(feats)
    else:
        probs = _pa_model.predict_proba(feats)
    probs = _calib.apply(probs, slice_meta=None)

    event = sample_event(probs, seed=req.seed)
    # Compute simple win probability and leverage index
    wp_home = _wp_home(req.state.model_dump(by_alias=True))
    li = _lev_index(req.state.model_dump(by_alias=True))
    bip_detail = None
    advancement = None
    if req.return_contact:
        try:
            bip_detail = _bip_model.predict(feats, seed=req.seed)
            # Fielding conversion (outs/DP) with minimal base updates
            # Build a simple state dict for fielding
            bases = {
                "1B": req.state.bases.B1,
                "2B": req.state.bases.B2,
                "3B": req.state.bases.B3,
            }
            state_for_fielding = {
                "outs": req.state.outs,
                "bases": bases,
                "weather": (req.state.weather.model_dump() if req.state.weather else {}),
                "park_id": req.state.park_id,
            }
            field_res = _fielding.convert(
                bip_detail=bip_detail,
                features=feats,
                defense=req.defense or {},
                state=state_for_fielding,
                seed=req.seed,
                parks_cfg=_PARKS_CFG,
            )
            # Fallback HR check using fence profile if converter did not set it
            if not bool(field_res.get("hr")) and bip_detail and isinstance(bip_detail, dict):
                try:
                    import math
                    fx = fence_at_angle(req.state.park_id, _PARKS_CFG, float(bip_detail.get("spray_deg", 0.0) or 0.0))
                    fence_dist = fx.get("fence_dist_ft") if isinstance(fx, dict) else None
                    wall_h = fx.get("wall_height_ft") if isinstance(fx, dict) else None
                    if fence_dist is not None:
                        ev = float(bip_detail.get("ev_mph", 0.0) or 0.0)
                        la = float(bip_detail.get("la_deg", 0.0) or 0.0)
                        wind_mph = float(feats.get("ctx_wind_mph", 0.0) or 0.0)
                        wind_dir = float(feats.get("ctx_wind_dir_deg", 0.0) or 0.0)
                        alt_ft = float(feats.get("ctx_altitude_ft", 0.0) or 0.0)
                        tail = math.cos(math.radians(wind_dir))
                        la_term = 0.0
                        if la > 10.0:
                            if la <= 40.0:
                                la_term = 6.5 * (la - 10.0)
                            else:
                                la_term = max(0.0, 6.5 * 30.0 - 8.0 * (la - 40.0))
                        carry_ft = 40.0 + 2.5 * ev + la_term + 1.5 * wind_mph * tail + 0.003 * alt_ft
                        eff_fence = float(fence_dist) + (0.4 * float(wall_h) if wall_h is not None else 0.0)
                        if carry_ft >= eff_fence + 5.0:
                            field_res["hr"] = True
                            field_res["out"] = False
                            field_res["outs_recorded"] = 0
                            bip_detail["out_subtype"] = "HR"
                except Exception:
                    pass
            # Merge fielding info into bip_detail
            for k in ("out_subtype", "fielder", "dp", "sf", "error", "hr"):
                if k in field_res:
                    bip_detail[k] = field_res[k]
            advancement = {
                "outs_recorded": field_res.get("outs_recorded", 0),
                "bases_ending": field_res.get("bases_ending", bases),
                "rbi": 0,
            }
            # Home run handling: clear bases and count RBI
            is_hr = bool(field_res.get("hr"))
            if is_hr:
                runners = int(bases.get("1B") is not None) + int(bases.get("2B") is not None) + int(
                    bases.get("3B") is not None
                )
                advancement["rbi"] = runners + 1
                advancement["outs_recorded"] = 0
                advancement["bases_ending"] = {"1B": None, "2B": None, "3B": None}
                bip_detail["out_subtype"] = "HR"
            # Sac fly credit
            if bool(field_res.get("sf")):
                advancement["rbi"] = advancement.get("rbi", 0) + 1
            # If ball is not an out, attempt baserunning advancement (e.g., 3B -> home)
            if (not is_hr) and (not bool(field_res.get("out"))):
                adv = _advancer.advance(
                    bip_detail=bip_detail,
                    fielding_result=field_res,
                    features=feats,
                    defense=req.defense or {},
                    state=state_for_fielding,
                )
                advancement["bases_ending"] = adv.get("bases_ending", advancement["bases_ending"])
                advancement["rbi"] = int(adv.get("rbi", 0)) + int(advancement.get("rbi", 0))
        except Exception:
            bip_detail = bip_detail or None
            advancement = advancement or None
    resp = SimResponse(
        model_version=_pa_model.version,
        probs=probs,
        sampled_event=event,
        bip_detail=bip_detail,
        advancement=advancement,
        rng={"seed": req.seed} if req.seed is not None else None,
        trace_id=uuid.uuid4().hex,
        win_prob_home=wp_home,
        leverage_index=li,
    )
    return resp


@app.post("/v1/sim/steal", response_model=StealResponse)
def steal(req: StealRequest):
    state = req.state.model_dump(by_alias=True)
    pitcher = req.pitcher.model_dump()
    runner = req.runner.model_dump() if req.runner is not None else {}
    defense = req.defense or {}

    result = _micro.attempt_steal(state=state,
                                  runner_base=req.runner_base,
                                  runner=runner,
                                  pitcher=pitcher,
                                  defense=defense)
    return StealResponse(
        event=result["event"],
        outs_recorded=int(result.get("outs_recorded", 0)),
        bases_ending=result.get("bases_ending", state.get("bases", {})),
        rng={"seed": req.seed} if req.seed is not None else None,
    )


@app.post("/v1/game/boxscore", response_model=BoxscoreResponse)
def boxscore(req: BoxscoreRequest):
    runs = {"home": 0, "away": 0}
    er = {"home": 0, "away": 0}
    outs = 0
    for ev in req.log:
        t = ev.team
        runs[t] = runs.get(t, 0) + int(ev.rbi or 0)
        # naive earned runs: runs on plays without error
        if not bool(ev.error):
            er[t] = er.get(t, 0) + int(ev.rbi or 0)
        outs += int(ev.outs_recorded or 0)
    winner = None
    if runs["home"] != runs["away"]:
        winner = req.home_team if runs["home"] > runs["away"] else req.away_team
    return BoxscoreResponse(
        game_id=req.game_id,
        runs=runs,
        earned_runs=er,
        outs=outs,
        winner=winner,
    )


@app.post("/v1/game/log", response_model=GameLogResponse)
def game_log(req: GameLogRequest):
    return GameLogResponse(game_id=req.game_id, events=req.log)


@app.post("/v1/rules/check-three-batter-min", response_model=ThreeBatterMinResponse)
def check_three_batter_min(req: ThreeBatterMinRequest):
    if not bool(req.three_batter_min):
        return ThreeBatterMinResponse(allowed=True)
    if bool(req.injury_exception) or bool(req.inning_ended):
        return ThreeBatterMinResponse(allowed=True)
    if int(req.batters_faced_in_stint) >= 3:
        return ThreeBatterMinResponse(allowed=True)
    return ThreeBatterMinResponse(allowed=False, reason="Three-batter minimum not reached")


@app.post("/admin/reload-players")
def admin_reload_players():
    token_required = os.environ.get("ADMIN_TOKEN")
    client_token = os.environ.get("ADMIN_TOKEN_PASSTHRU")  # internal convenience
    # If ADMIN_TOKEN is set, enforce it via env passthru or header (in future)
    if token_required and client_token != token_required:
        # Simple check; in a fuller impl, read from headers: X-Admin-Token
        return {"ok": False, "error": "forbidden"}, 403
    global _PLAYERS_CACHE
    try:
        path = _ROOT / "data" / "players.csv"
        _PLAYERS_CACHE = load_players_csv(str(path))
        return {"ok": True, "count": len(_PLAYERS_CACHE)}
    except Exception as e:
        return {"ok": False, "error": str(e)}, 500


@app.post("/v1/strategy/pinch-hit", response_model=PinchHitResponse)
def strategy_pinch_hit(req: PinchHitRequest):
    state = req.state.model_dump(by_alias=True)
    cur = req.current_batter.model_dump()
    pit = req.pitcher.model_dump()
    cands = [b.model_dump() for b in req.candidates]
    res = advise_pinch_hit(state, cur, pit, cands)
    return PinchHitResponse(**res)


@app.post("/v1/strategy/bullpen", response_model=BullpenResponse)
def strategy_bullpen(req: BullpenRequest):
    state = req.state.model_dump(by_alias=True)
    bat = req.upcoming_batter.model_dump()
    cands = [p.model_dump() for p in req.pitcher_candidates]
    res = advise_bullpen(state, bat, cands)
    return BullpenResponse(**res)


@app.post("/v1/strategy/bunt", response_model=BuntResponse)
def strategy_bunt(req: BuntRequest):
    state = req.state.model_dump(by_alias=True)
    bat = req.batter.model_dump()
    res = advise_bunt(state, bat)
    return BuntResponse(**res)


@app.post("/v1/strategy/ibb", response_model=IBBResponse)
def strategy_ibb(req: IBBRequest):
    state = req.state.model_dump(by_alias=True)
    bat = req.batter.model_dump()
    res = advise_ibb(state, bat)
    return IBBResponse(**res)


@app.post("/v1/sim/pitch", response_model=PitchResponse)
def pitch(req: PitchRequest):
    # Build features as in PA path
    b_dict = req.batter.model_dump()
    p_dict = req.pitcher.model_dump()
    if b_dict.get("ratings") is None:
        br = get_ratings_for(b_dict.get("player_id"), _PLAYERS_CACHE)
        if br:
            b_dict["ratings"] = br
    if p_dict.get("ratings") is None:
        pr = get_ratings_for(p_dict.get("player_id"), _PLAYERS_CACHE)
        if pr:
            if "arm" not in pr and "as" in pr:
                pr = {**pr, "arm": pr.get("as")}
                pr.pop("as", None)
            else:
                pr = {k: v for k, v in pr.items() if k != "as"}
            p_dict["ratings"] = pr

    b_feats = batter_features(b_dict)
    p_feats = pitcher_features(p_dict)
    c_feats = context_features(req.state, req.pitcher, req.batter)
    park_feats = {}
    try:
        park_feats = features_for_park(req.state.park_id, _PARKS_CFG)
    except Exception:
        park_feats = {}
    feats = {**b_feats, **p_feats, **c_feats, **park_feats}
    feats = apply_fatigue(
        feats,
        pitch_count=req.pitcher.pitch_count,
        tto_index=req.pitcher.tto_index,
        entered_cold=getattr(req.pitcher, "entered_cold", None),
        batters_faced_in_stint=getattr(req.pitcher, "batters_faced_in_stint", None),
    )

    # Ensure speed feature present if ratings include 'sp'
    if "bat_speed_z" not in feats:
        try:
            sp = None
            if req.batter.ratings and getattr(req.batter.ratings, "sp", None) is not None:
                sp = getattr(req.batter.ratings, "sp")
            elif isinstance(b_dict.get("ratings"), dict):
                sp = b_dict["ratings"].get("sp")
            if sp is not None:
                feats["bat_speed_z"] = z_0_99(sp)
        except Exception:
            pass

    # Pitch simulation
    state_dict = req.state.model_dump(by_alias=True)
    sim = _pitch_engine.pitch(
        state=state_dict,
        batter=b_dict,
        pitcher=p_dict,
        features=feats,
        seed=req.seed,
        umpire=(req.umpire or {}),
        edge=bool(req.edge),
    )

    # If pre-pitch event (PO), return immediately with no pitch
    if sim.get("pre_pitch") and sim.get("pre_pitch") != "no_play":
        return PitchResponse(
            pitch_type="",
            result="ball",  # ignored in pre-pitch context
            next_count=req.state.count,
            probs=None,
            pre_pitch=sim.get("pre_pitch"),
            pre_info=sim.get("pre_info"),
            rng={"seed": req.seed} if req.seed is not None else None,
        )

    result = sim["result"]
    pitch_type = sim["pitch_type"]
    next_count = sim["next_count"]

    # Replay/challenge: on edge, allow overturn of called_strike to ball
    try:
        rules = req.state.rules
        if bool(req.edge) and result == "called_strike" and rules and bool(getattr(rules, "challenges_enabled", False)):
            rate = float(getattr(rules, "overturn_rate", 0.2) or 0.2)
            if rate >= 1.0 or (rate > 0 and np.random.default_rng(req.seed).random() < rate):
                result = "ball"
                next_count = {"balls": min(3, req.state.count.balls + 1), "strikes": req.state.count.strikes}
    except Exception:
        pass

    bip_detail = None
    advancement = None
    if result == "in_play" and req.return_contact:
        try:
            bip_detail = _bip_model.predict(feats, seed=req.seed)
            bases = {
                "1B": req.state.bases.B1,
                "2B": req.state.bases.B2,
                "3B": req.state.bases.B3,
            }
            state_for_fielding = {
                "outs": req.state.outs,
                "bases": bases,
                "weather": (req.state.weather.model_dump() if req.state.weather else {}),
                "park_id": req.state.park_id,
            }
            field_res = _fielding.convert(
                bip_detail=bip_detail,
                features=feats,
                defense=req.defense or {},
                state=state_for_fielding,
                seed=req.seed,
                parks_cfg=_PARKS_CFG,
            )
            # Fallback HR check if needed
            if not bool(field_res.get("hr")) and bip_detail and isinstance(bip_detail, dict):
                try:
                    import math
                    fx = fence_at_angle(req.state.park_id, _PARKS_CFG, float(bip_detail.get("spray_deg", 0.0) or 0.0))
                    fence_dist = fx.get("fence_dist_ft") if isinstance(fx, dict) else None
                    wall_h = fx.get("wall_height_ft") if isinstance(fx, dict) else None
                    if fence_dist is not None:
                        ev = float(bip_detail.get("ev_mph", 0.0) or 0.0)
                        la = float(bip_detail.get("la_deg", 0.0) or 0.0)
                        wind_mph = float(feats.get("ctx_wind_mph", 0.0) or 0.0)
                        wind_dir = float(feats.get("ctx_wind_dir_deg", 0.0) or 0.0)
                        alt_ft = float(feats.get("ctx_altitude_ft", 0.0) or 0.0)
                        tail = math.cos(math.radians(wind_dir))
                        la_term = 0.0
                        if la > 10.0:
                            if la <= 40.0:
                                la_term = 6.5 * (la - 10.0)
                            else:
                                la_term = max(0.0, 6.5 * 30.0 - 8.0 * (la - 40.0))
                        carry_ft = 40.0 + 2.5 * ev + la_term + 1.5 * wind_mph * tail + 0.003 * alt_ft
                        eff_fence = float(fence_dist) + (0.4 * float(wall_h) if wall_h is not None else 0.0)
                        if carry_ft >= eff_fence + 5.0:
                            field_res["hr"] = True
                            field_res["out"] = False
                            field_res["outs_recorded"] = 0
                            bip_detail["out_subtype"] = "HR"
                except Exception:
                    pass
            for k in ("out_subtype", "fielder", "dp", "sf", "error", "hr"):
                if k in field_res:
                    bip_detail[k] = field_res[k]
            advancement = {
                "outs_recorded": field_res.get("outs_recorded", 0),
                "bases_ending": field_res.get("bases_ending", bases),
                "rbi": 0,
            }
            is_hr = bool(field_res.get("hr"))
            if is_hr:
                runners = int(bases.get("1B") is not None) + int(bases.get("2B") is not None) + int(
                    bases.get("3B") is not None
                )
                advancement["rbi"] = runners + 1
                advancement["outs_recorded"] = 0
                advancement["bases_ending"] = {"1B": None, "2B": None, "3B": None}
                bip_detail["out_subtype"] = "HR"
            if bool(field_res.get("sf")):
                advancement["rbi"] = advancement.get("rbi", 0) + 1
            if (not is_hr) and (not bool(field_res.get("out"))):
                adv = _advancer.advance(
                    bip_detail=bip_detail,
                    fielding_result=field_res,
                    features=feats,
                    defense=req.defense or {},
                    state=state_for_fielding,
                )
                advancement["bases_ending"] = adv.get("bases_ending", advancement["bases_ending"])
                advancement["rbi"] = int(adv.get("rbi", 0)) + int(advancement.get("rbi", 0))
        except Exception:
            bip_detail = bip_detail or None
            advancement = advancement or None

    return PitchResponse(
        pitch_type=pitch_type,
        result=result,
        next_count=next_count,
        probs=sim.get("probs"),
        bip_detail=bip_detail,
        advancement=advancement,
        rng={"seed": req.seed} if req.seed is not None else None,
    )
