from typing import Dict
from .transforms import z_0_99

def batter_features(b: Dict) -> Dict[str, float]:
    r = (b.get("ratings") or {})
    bats = b.get("bats", "R")

    feats = {
        "bat_contact_z": z_0_99(r.get("ch")),   # if provided
        "bat_power_z":   z_0_99(r.get("ph")),
        "bat_gbfb_z":    z_0_99(r.get("gf")),   # ↑gf => lower LA
        "bat_pull_z":    z_0_99(r.get("pl")),   # spray -> pull by handedness
        "bat_risp_z":    z_0_99(r.get("sc")),   # gate only when RISP
        "bat_field_z":   z_0_99(r.get("fa")),
        "bat_arm_z":     z_0_99(r.get("arm")),
        "bat_isLHB":     1.0 if bats == "L" else 0.0,
    }
    return {k:v for k,v in feats.items() if v is not None}

def pitcher_features(p: Dict) -> Dict[str, float]:
    r = (p.get("ratings") or {})
    # Handle both 'arm' and alias 'as' and pydantic's internal 'as_' key
    arm = r.get("arm", r.get("as", r.get("as_")))

    # Latent 'stuff' from velo + mix
    parts = [
        0.50 * (z_0_99(arm) or 0.0),
        0.20 * (z_0_99(r.get("fb")) or 0.0),
        0.15 * (z_0_99(r.get("sl")) or 0.0),
        0.10 * (z_0_99(r.get("si")) or 0.0),
        0.03 * max((z_0_99(r.get("cb")) or 0.0), (z_0_99(r.get("cu")) or 0.0)),
        0.02 * max((z_0_99(r.get("scb")) or 0.0), (z_0_99(r.get("kn")) or 0.0)),
    ]
    stuff = sum(parts)

    feats = {
        "pit_stamina_z": z_0_99(r.get("endurance")),
        "pit_control_z": z_0_99(r.get("control")),
        "pit_move_z":    z_0_99(r.get("movement")),
        "pit_hold_z":    z_0_99(r.get("hold_runner")),
        "pit_field_z":   z_0_99(r.get("fa")),
        "pit_arm_z":     z_0_99(arm),
        "pit_stuff_z":   stuff,
        # GB vs Air: sinker+curve push GB; FB+SL push air
        "pit_gb_bias_z": ((z_0_99(r.get("si")) or 0.0) + (z_0_99(r.get("cb")) or 0.0))
                         - ((z_0_99(r.get("fb")) or 0.0) + (z_0_99(r.get("sl")) or 0.0)),
    }
    return {k:v for k,v in feats.items() if v is not None}
