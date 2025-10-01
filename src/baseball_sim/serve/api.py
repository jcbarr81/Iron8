from fastapi import FastAPI
from pathlib import Path
from ..schemas import SimRequest, SimResponse
from ..features.ratings_adapter import batter_features, pitcher_features
from ..features.context_features import context_features
from ..calibration.calibrators import IdentityCalibrator
from ..models.pa_model import PaOutcomeModel
from ..sampler.rng import sample_event
from ..utils.players import load_players_csv, get_ratings_for

app = FastAPI()
_pa_model = PaOutcomeModel()
_calib = IdentityCalibrator()

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

# Load players cache (ratings) from CSV; safe if missing
_PLAYERS_CACHE = {}
try:
    _PLAYERS_PATH = _ROOT / "data" / "players.csv"
    _PLAYERS_CACHE = load_players_csv(str(_PLAYERS_PATH))
    print(f"[serve.api] Loaded {len(_PLAYERS_CACHE)} players from {_PLAYERS_PATH}")
except Exception:
    _PLAYERS_CACHE = {}

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
    c_feats = context_features(req.state, req.pitcher)
    feats = {**b_feats, **p_feats, **c_feats}

    # TODO(Codex): add context features (count/base/outs/park/rules)
    # TODO(Codex): RISP gating only if bases include runner on 2B or 3B

    probs = _pa_model.predict_proba(feats)
    probs = _calib.apply(probs, slice_meta=None)

    event = sample_event(probs, seed=req.seed)
    resp = SimResponse(
        model_version=_pa_model.version,
        probs=probs,
        sampled_event=event,
        rng={"seed": req.seed} if req.seed is not None else None
    )
    return resp
