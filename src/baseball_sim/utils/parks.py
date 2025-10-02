from __future__ import annotations

from pathlib import Path
from typing import Dict, Any


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


def features_for_park(park_id: str | None, parks_cfg: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    """Return simple directional features for the given park_id.

    Surfaces booleans as 0/1:
    - ctx_park_rf_short_porches
    - ctx_park_lf_short_porches
    Missing park_id or attributes produce an empty dict.
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

    feats = {
        "ctx_park_rf_short_porches": _b("rf_short_porches"),
        "ctx_park_lf_short_porches": _b("lf_short_porches"),
    }
    # Drop zeros if all false? Keep explicit features for clarity.
    return feats
