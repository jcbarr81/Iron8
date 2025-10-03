from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional


def load_parks_yaml(path: str | Path) -> Dict[str, Dict[str, Any]]:
    """Load parks.yaml into a dict mapping park_id -> attributes.

    Tolerant to missing file or parse errors; returns {} in those cases.
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        # Lazy import to avoid hard dependency during tests that don't need it
        import yaml  # type: ignore

        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def features_for_park(park_id: str | None, parks_cfg: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Return park context features for the given park_id.

    Currently surfaces:
    - "ctx_park_rf_short_porches" -> 0/1
    - "ctx_park_lf_short_porches" -> 0/1
    - "ctx_roof_dome" -> 0/1 when YAML roof == "dome"
    - "ctx_roof_retractable" -> 0/1 when YAML roof == "retractable"
    - "ctx_altitude_ft" -> float (if present in YAML)

    Missing park_id or unknown attributes produce an empty dict.
    """
    if not park_id:
        return {}
    if not parks_cfg:
        return {}
    park = parks_cfg.get(str(park_id), {})
    if not isinstance(park, dict) or not park:
        return {}

    def _b(key: str) -> int:
        try:
            return 1 if bool(park.get(key, False)) else 0
        except Exception:
            return 0

    feats: Dict[str, Any] = {
        "ctx_park_rf_short_porches": _b("rf_short_porches"),
        "ctx_park_lf_short_porches": _b("lf_short_porches"),
    }

    # Roof handling: expect one of {open, dome, retractable}
    try:
        roof = str(park.get("roof", "")).strip().lower()
        if roof:
            feats["ctx_roof_dome"] = 1 if roof == "dome" else 0
            feats["ctx_roof_retractable"] = 1 if roof == "retractable" else 0
    except Exception:
        pass

    # Altitude in feet if present
    try:
        if park.get("altitude_ft") is not None:
            feats["ctx_altitude_ft"] = float(park.get("altitude_ft"))
    except Exception:
        pass
    # Drop zeros if all false? Keep explicit features for clarity.
    return feats


def _anchors_from_dims(park: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    """Build anchor arrays (angles_deg, fence_dist_ft) from coarse dims.

    Angles are defined as:
    -45: LF line, -30: LF gap, 0: CF, +30: RF gap, +45: RF line
    Returns (angles, distances) with only present values, sorted by angle.
    """
    dims = park.get("dims") or {}
    if not isinstance(dims, dict) or not dims:
        return ([], [])
    anchors: List[Tuple[float, float]] = []
    try:
        mapping = [
            (-45.0, dims.get("lf_line_ft")),
            (-30.0, dims.get("lf_gap_ft")),
            (0.0, dims.get("cf_ft")),
            (30.0, dims.get("rf_gap_ft")),
            (45.0, dims.get("rf_line_ft")),
        ]
        for angle, val in mapping:
            if val is None:
                continue
            try:
                anchors.append((angle, float(val)))
            except Exception:
                pass
        anchors.sort(key=lambda x: x[0])
        if not anchors:
            return ([], [])
        angles = [a for a, _ in anchors]
        dists = [d for _, d in anchors]
        return (angles, dists)
    except Exception:
        return ([], [])


def _anchors_from_walls(park: Dict[str, Any]) -> Tuple[List[float], List[Optional[float]]]:
    """Build anchor arrays (angles_deg, wall_height_ft) from coarse wall heights.

    Uses same angle scheme as _anchors_from_dims.
    Returns (angles, heights) with only present values, sorted by angle.
    """
    walls = park.get("wall_heights_ft") or {}
    if not isinstance(walls, dict) or not walls:
        return ([], [])
    anchors: List[Tuple[float, Optional[float]]] = []
    try:
        mapping = [
            (-45.0, walls.get("lf")),
            (-30.0, walls.get("lf_gap")),
            (0.0, walls.get("cf")),
            (30.0, walls.get("rf_gap")),
            (45.0, walls.get("rf")),
        ]
        for angle, val in mapping:
            if val is None:
                continue
            try:
                anchors.append((angle, float(val)))
            except Exception:
                pass
        anchors.sort(key=lambda x: x[0])
        if not anchors:
            return ([], [])
        angles = [a for a, _ in anchors]
        heights = [h for _, h in anchors]
        return (angles, heights)
    except Exception:
        return ([], [])


def _interp_linear(x: float, xs: List[float], ys: List[float]) -> Optional[float]:
    """Piecewise-linear interpolation with clamping at ends.

    Returns None if xs/ys invalid.
    """
    if not xs or not ys or len(xs) != len(ys):
        return None
    # Clamp
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    # Find segment
    for i in range(1, len(xs)):
        if x <= xs[i]:
            x0, x1 = xs[i - 1], xs[i]
            y0, y1 = ys[i - 1], ys[i]
            # Guard divide by zero
            if x1 == x0:
                return y0
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return ys[-1]


def fence_profile_for_park(park_id: str | None, parks_cfg: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Return coarse fence profile arrays derived from dims and wall heights.

    Returns keys:
    - angles_deg: list of anchor angles
    - dist_ft: list of fence distances at anchors
    - height_ft: list of wall heights at anchors (may be empty)
    """
    if not park_id or not parks_cfg:
        return {}
    park = parks_cfg.get(str(park_id), {})
    if not isinstance(park, dict) or not park:
        return {}
    a_dist, dists = _anchors_from_dims(park)
    a_wall, walls = _anchors_from_walls(park)
    out: Dict[str, Any] = {}
    if a_dist:
        out["angles_deg"] = a_dist
        out["dist_ft"] = dists
    if a_wall:
        out["height_ft"] = walls
        # If height anchors are present but distance anchors missing, still return heights
        if not a_dist:
            out["angles_deg"] = a_wall
    return out


def fence_at_angle(park_id: str | None, parks_cfg: Dict[str, Dict[str, Any]], spray_deg: float) -> Dict[str, Optional[float]]:
    """Compute fence distance and wall height at a given spray angle.

    Uses piecewise-linear interpolation between coarse anchors derived from YAML dims and wall_heights_ft.
    Returns: {"fence_dist_ft": float|None, "wall_height_ft": float|None}
    """
    prof = fence_profile_for_park(park_id, parks_cfg)
    if not prof:
        return {"fence_dist_ft": None, "wall_height_ft": None}
    angles = prof.get("angles_deg") or []
    dists = prof.get("dist_ft") or []
    walls = prof.get("height_ft") or []
    res = {"fence_dist_ft": None, "wall_height_ft": None}
    if angles and dists:
        res["fence_dist_ft"] = _interp_linear(float(spray_deg), list(angles), list(dists))
    if walls:
        # If heights provided without explicit angles, assume they align with angles
        if not angles:
            return {"fence_dist_ft": None, "wall_height_ft": None}
        res["wall_height_ft"] = _interp_linear(float(spray_deg), list(angles), list(walls))
    return res
