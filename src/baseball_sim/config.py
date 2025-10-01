from pydantic import BaseModel
import yaml

class Settings(BaseModel):
    model_artifact_dir: str
    pa_model_path: str
    calibrator_path: str
    use_onnx: bool
    onnx_path: str
    host: str
    port: int
    enable_bip: bool
    enable_advancement: bool
    shift_restrictions: bool
    extras_runner_on_2nd: bool

def load_settings(path: str = "config/settings.example.yaml") -> Settings:
    with open(path, "r") as f:
        y = yaml.safe_load(f)
    m = y.get("model", {})
    s = y.get("service", {})
    sim = y.get("simulation", {})
    rules = sim.get("rules", {})
    return Settings(
        model_artifact_dir=m.get("artifact_dir"),
        pa_model_path=m.get("pa_model_path"),
        calibrator_path=m.get("calibrator_path"),
        use_onnx=m.get("use_onnx", False),
        onnx_path=m.get("onnx_path"),
        host=s.get("host", "0.0.0.0"),
        port=s.get("port", 8000),
        enable_bip=sim.get("enable_bip", False),
        enable_advancement=sim.get("enable_advancement", False),
        shift_restrictions=rules.get("shift_restrictions", True),
        extras_runner_on_2nd=rules.get("extras_runner_on_2nd", True),
    )
