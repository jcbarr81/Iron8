# Placeholder for optional Ball-In-Play model
class BipModel:
    def predict(self, features: dict) -> dict:
        # TODO(Codex): Generate EV/LA/spray based on ratings shifts
        return {"ev_mph": 100.0, "la_deg": 25.0, "spray_deg": 10.0}
