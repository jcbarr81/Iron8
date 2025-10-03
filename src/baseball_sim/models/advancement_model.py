from __future__ import annotations

from typing import Dict, Optional, Any
import math
import numpy as np
from ..utils.parks import fence_at_angle


INF_POS = ("3B", "SS", "2B", "1B", "P", "C")
OF_POS = ("LF", "CF", "RF")


def _norm_rating(x: Optional[float | int]) -> float:
    if x is None:
        return 0.0
    try:
        return max(0.0, min(1.0, float(x) / 99.0))
    except Exception:
        return 0.0


class FieldingConverter:
    """Convert BIP to outs/hits and simple base outcomes.

    - Picks a fielder based on contact_type and spray.
    - Computes out probability adjusted by fielder `fa`.
    - Applies infield-fly rule for PU with runners and <2 outs.
    - Applies simple DP logic on GB with runner on 1B and <2 outs.
    - Returns out subtype and fielder; minimal bases_ending updates (DP removes runner on 1B).
    """

    def __init__(self):
        ...

    @staticmethod
    def _choose_fielder(contact_type: str, spray_deg: float) -> str:
        if contact_type == "GB":
            if spray_deg <= -10:
                return "3B"
            if spray_deg <= 0:
                return "SS"
            if spray_deg <= 10:
                return "2B"
            return "1B"
        if contact_type == "PU":
            if spray_deg <= -10:
                return "3B"
            if spray_deg <= 0:
                return "SS"
            if spray_deg <= 10:
                return "2B"
            return "1B"
        # FB/LD -> outfield split
        if spray_deg <= -15:
            return "LF"
        if spray_deg <= 15:
            return "CF"
        return "RF"

    @staticmethod
    def _base_out_prob(contact_type: str, ev: float, la: float) -> float:
        # Heuristic base catch probabilities by contact type and quality
        if contact_type == "GB":
            # Harder GB -> lower out prob
            return 0.70 - 0.003 * (ev - 85.0)
        if contact_type == "LD":
            # Line drives mostly hits
            return 0.25 - 0.003 * (ev - 90.0)
        if contact_type == "FB":
            # Typical flies are often caught
            return 0.85 - 0.004 * (ev - 90.0)
        # PU: infield popups are almost always outs
        return 0.97

    def convert(
        self,
        bip_detail: Dict[str, Any],
        features: Dict[str, float],
        defense: Optional[Dict[str, int]],
        state: Dict[str, Any],
        seed: Optional[int] = None,
        parks_cfg: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

        ev = float(bip_detail.get("ev_mph", 0.0) or 0.0)
        la = float(bip_detail.get("la_deg", 0.0) or 0.0)
        spray = float(bip_detail.get("spray_deg", 0.0) or 0.0)
        ctype = str(bip_detail.get("contact_type", "GB") or "GB")

        # Infield-fly rule: PU with runners on and <2 outs
        bases = state.get("bases", {}) if isinstance(state, dict) else {}
        outs = int(state.get("outs", 0) if isinstance(state, dict) else 0)
        risp = int((bases.get("2B") is not None) or (bases.get("3B") is not None))
        if ctype == "PU" and outs < 2 and (bases.get("1B") is not None or risp):
            fielder = self._choose_fielder("PU", spray)
            return {
                "out": True,
                "out_subtype": "PU_OUT",
                "fielder": fielder,
                "dp": False,
                "outs_recorded": 1,
                "bases_ending": dict(bases),  # runners hold
            }

        # HR/off-wall check using fence interpolation for air balls
        if ctype in {"FB", "LD"} and parks_cfg is not None:
            try:
                park_id = state.get("park_id") if isinstance(state, dict) else None
                fx = fence_at_angle(park_id, parks_cfg, spray)
                fence_dist = fx.get("fence_dist_ft") if isinstance(fx, dict) else None
                wall_h = fx.get("wall_height_ft") if isinstance(fx, dict) else None
                if fence_dist is not None:
                    # Simple carry heuristic from EV/LA with small weather/alt tweaks
                    wind_mph = float(features.get("ctx_wind_mph", 0.0) or 0.0)
                    wind_dir = float(features.get("ctx_wind_dir_deg", 0.0) or 0.0)
                    tail = math.cos(math.radians(wind_dir))
                    alt_ft = float(features.get("ctx_altitude_ft", 0.0) or 0.0)

                    # LA contribution: benefit up to ~40 deg, then taper
                    la_term = 0.0
                    if la > 10.0:
                        if la <= 40.0:
                            la_term = 6.5 * (la - 10.0)
                        else:
                            la_term = max(0.0, 6.5 * 30.0 - 8.0 * (la - 40.0))
                    carry_ft = 40.0 + 2.5 * ev + la_term + 1.5 * wind_mph * tail + 0.003 * alt_ft

                    eff_fence = float(fence_dist)
                    if wall_h is not None:
                        try:
                            eff_fence += 0.4 * float(wall_h)
                        except Exception:
                            pass

                    if carry_ft >= eff_fence + 5.0:
                        # Home run
                        return {
                            "out": False,
                            "out_subtype": "HR",
                            "hr": True,
                            "dp": False,
                            "outs_recorded": 0,
                            "bases_ending": dict(bases),  # final bases handled by caller
                            "error": False,
                        }
                    else:
                        # Mark off-wall for caller context if near fence
                        near = (carry_ft + 5.0) >= float(fence_dist)
                        if near:
                            # Continue to normal fielding but flag it
                            bip_detail["off_wall"] = True
            except Exception:
                # If anything fails, fall back to normal processing
                pass

        # Pick fielder by spray and contact type
        fielder = self._choose_fielder(ctype, spray)

        # Fielder ability from defense (oaa_by_pos) if available; else 0
        fa = 0.0
        if isinstance(defense, dict):
            oaa = defense.get("oaa_by_pos") or defense
            if isinstance(oaa, dict):
                fa = _norm_rating(oaa.get(fielder))

        # Adjust out probability by fielder skill
        p_out = self._base_out_prob(ctype, ev, la)
        # Defensive skill influence
        p_out += 0.15 * fa

        # Defensive positioning & strategy adjustments
        align = defense.get("alignment") if isinstance(defense, dict) else None
        infield_align = (align or {}).get("infield") if isinstance(align, dict) else None
        outfield_align = (align or {}).get("outfield") if isinstance(align, dict) else None

        # Infield alignment adjustments
        if ctype in {"GB", "LD", "PU"}:
            if infield_align == "in":
                if ctype == "GB":
                    p_out += 0.05  # reduce singles on grounders (playing in)
                if ctype == "LD":
                    p_out -= 0.05  # more line drives get through
            elif infield_align == "dp":
                if ctype == "GB":
                    p_out += 0.02  # slight increase favoring DP situations
            elif infield_align == "corners":
                if ctype in {"GB", "PU"} and fielder in {"3B", "1B"}:
                    p_out += 0.04

        # Outfield alignment adjustments
        if ctype in {"FB", "LD"}:
            if outfield_align == "deep":
                if la >= 25.0:
                    p_out += 0.05  # catch more deep balls
                else:
                    p_out -= 0.05  # allow more short flies to drop (singles)
            elif outfield_align == "shallow":
                if la < 20.0:
                    p_out += 0.05  # convert more short flies
                else:
                    p_out -= 0.05  # give up more deep hits

        # Clamp to [0,1]
        p_out = max(0.0, min(1.0, p_out))

        # Weather adjustments: headwind raises FB/LD out prob, tailwind reduces
        try:
            weather = state.get("weather", {}) if isinstance(state, dict) else {}
            wind_mph = float(weather.get("wind_mph", 0.0) or 0.0)
            wind_dir = float(weather.get("wind_dir_deg", 0.0) or 0.0)
            import math
            tail = math.cos(math.radians(wind_dir))
            if ctype in {"FB", "LD"}:
                p_out += 0.04 * wind_mph * (-tail) / 20.0  # headwind (+), tailwind (-)
        except Exception:
            pass

        # Error model: chance to turn an out into ROE depends on skill/contact
        if ctype in {"GB", "PU"}:
            p_err = max(0.0, 0.06 - 0.04 * fa + 0.001 * max(0.0, ev - 85.0))
        else:  # FB/LD
            p_err = max(0.0, 0.03 - 0.02 * fa + 0.001 * max(0.0, ev - 95.0))

        # Sample out vs hit; account for potential error
        roll_out = rng.random()
        is_out = (roll_out < p_out)
        error = False
        if is_out and (rng.random() < p_err):
            # Error overturns the out -> batter reaches, no out recorded
            is_out = False
            error = True

        # Double-play logic on GB with <2 outs and runner on 1B
        dp = False
        outs_recorded = 0
        bases_ending = dict(bases)

        if ctype == "GB" and outs < 2 and bases.get("1B") is not None and not error:
            # IF ability boosts DP chance (use IF position if applicable, else avg INF)
            if fielder in INF_POS:
                if_fa = fa
            else:
                # average the infield if provided
                if_ratings = [
                    _norm_rating((defense or {}).get("SS") if defense else None),
                    _norm_rating((defense or {}).get("2B") if defense else None),
                    _norm_rating((defense or {}).get("3B") if defense else None),
                    _norm_rating((defense or {}).get("1B") if defense else None),
                ]
                if_fa = float(np.mean(if_ratings)) if if_ratings else 0.0

            p_dp = 0.15 + 0.35 * if_fa  # up to ~0.5 DP chance at elite IF
            if is_out and (rng.random() < p_dp):
                dp = True
                outs_recorded = 2
                # Remove runner from 1B; batter out
                bases_ending = dict(bases)
                bases_ending["1B"] = None
                return {
                    "out": True,
                    "out_subtype": "GB_OUT",
                    "fielder": fielder,
                    "dp": True,
                    "outs_recorded": outs_recorded,
                    "bases_ending": bases_ending,
                    "error": False,
                }

        # Fielder's choice on GB: take lead runner, batter safe
        if ctype == "GB" and is_out and outs < 2 and bases.get("1B") is not None and not dp:
            bases_ending = dict(bases)
            bases_ending["1B"] = "BATT"  # placeholder id
            # Lead runner out at 2B
            bases_ending["2B"] = None
            return {
                "out": True,
                "out_subtype": "FC",
                "fielder": fielder,
                "dp": False,
                "outs_recorded": 1,
                "bases_ending": bases_ending,
                "error": False,
            }

        if is_out:
            outs_recorded = 1
            subtype = (
                "GB_OUT" if ctype == "GB" else "LD_OUT" if ctype == "LD" else "FB_OUT" if ctype == "FB" else "PU_OUT"
            )
            # Sac fly credit: flyout with <2 outs and runner on 3B -> runner scores
            sf = False
            if subtype in {"FB_OUT", "LD_OUT"} and outs < 2 and bases.get("3B") is not None:
                sf = True
                bases_ending = dict(bases)
                bases_ending["3B"] = None
            return {
                "out": True,
                "out_subtype": subtype,
                "fielder": fielder,
                "dp": False,
                "outs_recorded": outs_recorded,
                "bases_ending": bases_ending,  # hold by default (or SF handled above)
                "sf": sf,
                "error": False,
            }

        # Not an out: treat as hit or ROE, no advancement here (handled later)
        return {
            "out": False,
            "fielder": fielder,
            "dp": False,
            "outs_recorded": 0,
            "bases_ending": bases_ending,
            "error": error,
        }


class BaserunningAdvancer:
    """Advance runners on hits using simple, deterministic policies.

    Currently handles 3B -> Home decisions on OF hits using runner speed,
    outfielder arm (from defense), EV/LA depth proxies, and returns RBI count.
    """

    @staticmethod
    def _of_arm(defense: Optional[Dict[str, int]], fielder: str) -> float:
        if not isinstance(defense, dict):
            return 0.0
        # Prefer explicit arm_by_pos; fallback to generic positional value
        arm_map = defense.get("arm_by_pos") if isinstance(defense.get("arm_by_pos"), dict) else None
        val = None
        if arm_map:
            val = arm_map.get(fielder)
        if val is None:
            val = defense.get(fielder)
        return _norm_rating(val)

    def advance(self,
                bip_detail: Dict[str, Any],
                fielding_result: Dict[str, Any],
                features: Dict[str, float],
                defense: Optional[Dict[str, int]],
                state: Dict[str, Any]) -> Dict[str, Any]:
        # Default passthrough
        bases = dict(fielding_result.get("bases_ending", state.get("bases", {})))
        rbi = 0

        # Only consider when not an out
        if bool(fielding_result.get("out")):
            return {"bases_ending": bases, "rbi": rbi}

        ctype = str(bip_detail.get("contact_type", "GB") or "GB")
        fielder = str(fielding_result.get("fielder", ""))
        ev = float(bip_detail.get("ev_mph", 0.0) or 0.0)
        la = float(bip_detail.get("la_deg", 0.0) or 0.0)
        # Runner speed (use batter's for now as a proxy if present)
        sp = float(features.get("bat_speed_z", 0.0) or 0.0)

        # Send 3B -> Home on OF hit
        if ctype in {"FB", "LD"} and fielder in OF_POS and bases.get("3B") is not None:
            of_arm = self._of_arm(defense, fielder)
            # Deterministic score; threshold at 0.5
            depth = max(0.0, (la - 20.0) / 20.0)  # deeper flies help
            hard = max(0.0, (ev - 85.0) / 15.0)   # harder contact travels
            score = 0.5 + 0.5 * sp - 0.6 * of_arm + 0.2 * depth + 0.1 * hard
            if score > 0.5:
                # Runner scores
                bases["3B"] = None
                rbi += 1
            else:
                # Hold
                ...

        return {"bases_ending": bases, "rbi": rbi}
