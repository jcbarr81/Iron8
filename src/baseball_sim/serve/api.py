from fastapi import FastAPI
from ..schemas import SimRequest, SimResponse
from ..features.ratings_adapter import batter_features, pitcher_features
from ..calibration.calibrators import IdentityCalibrator
from ..models.pa_model import PaOutcomeModel
from ..sampler.rng import sample_event

app = FastAPI()
_pa_model = PaOutcomeModel()
_calib = IdentityCalibrator()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/v1/sim/plate-appearance", response_model=SimResponse)
def plate_appearance(req: SimRequest):
    # Build feature dict
    b_feats = batter_features(req.batter.model_dump())
    p_feats = pitcher_features(req.pitcher.model_dump())
    feats = {**b_feats, **p_feats}

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
