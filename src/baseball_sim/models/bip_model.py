from __future__ import annotations

from typing import Dict, Optional
import numpy as np


class BipModel:
    """Ball-In-Play generator producing EV/LA/spray and contact_type.

    Monotone responses:
    - `bat_power_z` increases EV
    - `pit_stuff_z` modestly reduces EV
    - `bat_gbfb_z` lowers LA (more grounders)
    - `bat_pull_z` shifts spray toward pull side; sign depends on handedness
      (LHB: positive, RHB: negative)
    """

    def __init__(self,
                 base_ev: float = 88.0,
                 base_la: float = 15.0,
                 ev_per_power: float = 6.0,
                 ev_per_stuff: float = -2.0,
                 la_per_gbfb: float = -6.0,
                 spray_per_pull: float = 12.0,
                 ev_noise: float = 1.5,
                 la_noise: float = 2.0,
                 spray_noise: float = 3.0):
        self.base_ev = base_ev
        self.base_la = base_la
        self.ev_per_power = ev_per_power
        self.ev_per_stuff = ev_per_stuff
        self.la_per_gbfb = la_per_gbfb
        self.spray_per_pull = spray_per_pull
        self.ev_noise = ev_noise
        self.la_noise = la_noise
        self.spray_noise = spray_noise

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        return float(max(lo, min(hi, x)))

    @staticmethod
    def _classify_contact_type(la_deg: float) -> str:
        if la_deg < 10.0:
            return "GB"
        if la_deg < 25.0:
            return "LD"
        if la_deg < 50.0:
            return "FB"
        return "PU"

    def predict(self, features: Dict[str, float], seed: Optional[int] = None) -> Dict[str, float | str]:
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

        power = float(features.get("bat_power_z", 0.0) or 0.0)
        stuff = float(features.get("pit_stuff_z", 0.0) or 0.0)
        gbfb = float(features.get("bat_gbfb_z", 0.0) or 0.0)
        pull = float(features.get("bat_pull_z", 0.0) or 0.0)
        is_lhb = int(bool(features.get("bat_isLHB", 0)))

        # EV: base + power gain - stuff penalty + weather + noise
        ev = self.base_ev + self.ev_per_power * power + self.ev_per_stuff * stuff
        # Temperature effect: ~+0.1 mph per 2°F above 70
        temp_f = float(features.get("ctx_temp_f", 70.0) or 70.0)
        ev += 0.05 * (temp_f - 70.0)
        ev += rng.normal(0.0, self.ev_noise)
        ev = self._clip(ev, 50.0, 115.0)

        # LA: base + gbfb shift + wind + noise (gbfb positive -> lower LA)
        la = self.base_la + self.la_per_gbfb * gbfb
        wind_mph = float(features.get("ctx_wind_mph", 0.0) or 0.0)
        wind_dir = float(features.get("ctx_wind_dir_deg", 0.0) or 0.0)
        # Tail/head wind: use cos(direction) as proxy (0 deg = straight out, 180 = straight in)
        import math
        tail = math.cos(math.radians(wind_dir))
        # Apply wind LA effect regardless; it mostly impacts air balls but small effect is fine
        la += 0.03 * wind_mph * tail
        la += rng.normal(0.0, self.la_noise)
        la = self._clip(la, -5.0, 70.0)

        # Spray: pull shift by handedness (LHB positive, RHB negative) + noise
        pull_sign = 1.0 if is_lhb == 1 else -1.0
        spray = 0.0 + self.spray_per_pull * pull * pull_sign
        spray += rng.normal(0.0, self.spray_noise)
        spray = self._clip(spray, -45.0, 45.0)

        contact_type = self._classify_contact_type(la)

        return {
            "ev_mph": round(float(ev), 2),
            "la_deg": round(float(la), 2),
            "spray_deg": round(float(spray), 2),
            "contact_type": contact_type,
        }
