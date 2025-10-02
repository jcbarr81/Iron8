from __future__ import annotations

from typing import Dict, Any, Optional
from ..features.transforms import z_0_99


class MicroEvents:
    """Deterministic steal/pickoff/WP/PB resolver based on ratings.

    Inputs:
    - runner_speed: from runner.ratings.sp (0-99)
    - pitcher_hold: from pitcher.ratings.hold_runner (0-99)
    - catcher_arm: from defense["arm_by_pos"]["C"] or defense["C"] (0-99)
    - catcher_field: from defense["oaa_by_pos"]["C"] or defense["C"] (0-99)
    """

    @staticmethod
    def _norm(v: Optional[int]) -> float:
        return float(z_0_99(v) or 0.0)  # -3..3

    @staticmethod
    def _norm01(v: Optional[int]) -> float:
        z = MicroEvents._norm(v)
        return max(0.0, min(1.0, (z + 3.0) / 6.0))

    def attempt_steal(self, state: Dict[str, Any], runner_base: str, runner: Dict[str, Any], pitcher: Dict[str, Any], defense: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        bases = state.get("bases", {}) if isinstance(state, dict) else {}

        # Ratings
        r_sp = None
        if isinstance(runner, dict):
            rr = runner.get("ratings") or {}
            r_sp = rr.get("sp")
        p_hold = None
        if isinstance(pitcher, dict):
            pr = pitcher.get("ratings") or {}
            p_hold = pr.get("hold_runner")

        cat_arm = None
        cat_fld = None
        if isinstance(defense, dict):
            arm_map = defense.get("arm_by_pos") if isinstance(defense.get("arm_by_pos"), dict) else None
            oaa_map = defense.get("oaa_by_pos") if isinstance(defense.get("oaa_by_pos"), dict) else None
            cat_arm = (arm_map or {}).get("C") if arm_map else defense.get("C")
            cat_fld = (oaa_map or {}).get("C") if oaa_map else defense.get("C")

        sp01 = self._norm01(r_sp)
        hold01 = self._norm01(p_hold)
        arm01 = self._norm01(cat_arm)
        fld01 = self._norm01(cat_fld)

        # Deterministic scores
        # Pickoff more likely with strong hold + arm
        po_score = 0.4 * (hold01 + arm01)
        # Passed ball more likely with poor catching
        pb_score = 0.7 * (1.0 - fld01)
        # Steal success more likely with speed, less with hold and arm
        sb_score = 0.5 + 0.7 * sp01 - 0.4 * hold01 - 0.5 * arm01

        # Resolve priority: Pickoff > PB/WP > SB/CS
        if po_score > 0.5:
            # Runner picked off at the base
            bases_end = dict(bases)
            if runner_base in bases_end:
                bases_end[runner_base] = None
            return {"event": "PO", "outs_recorded": 1, "bases_ending": bases_end}

        if pb_score > 0.5:
            # Passed ball: all runners advance one base if forced
            b = dict(bases)
            new_b = {"1B": None, "2B": None, "3B": None}
            # 3B -> home does not remain on bases
            if b.get("2B") is not None:
                new_b["3B"] = b["2B"]
            if b.get("1B") is not None:
                new_b["2B"] = b["1B"]
            return {"event": "PB", "outs_recorded": 0, "bases_ending": new_b}

        # Otherwise decide steal success
        if sb_score > 0.5:
            # Successful steal: runner advances one base
            b = dict(bases)
            runner_id = b.get(runner_base)
            targets = {"1B": "2B", "2B": "3B"}
            to_base = targets.get(runner_base)
            if to_base:
                b[runner_base] = None
                b[to_base] = runner_id
            return {"event": "SB", "outs_recorded": 0, "bases_ending": b}
        else:
            # Caught stealing: runner out
            b = dict(bases)
            if runner_base in b:
                b[runner_base] = None
            return {"event": "CS", "outs_recorded": 1, "bases_ending": b}

