from typing import Dict

class IdentityCalibrator:
    """Placeholder; replace with per-class isotonic/Platt."""
    def apply(self, probs: Dict[str, float], slice_meta: Dict[str, str] | None = None) -> Dict[str, float]:
        s = sum(probs.values()) or 1.0
        return {k: v / s for k, v in probs.items()}
