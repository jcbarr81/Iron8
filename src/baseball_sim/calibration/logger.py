from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


def maybe_log_pa(route: str, probs: Dict[str, float], sampled_event: str, slice_meta: Dict[str, str]) -> None:
    """Append a simple CSV line with PA prediction telemetry if enabled.

    Enable by setting env var PA_LOG_ENABLE=1. Optional PA_LOG_PATH overrides path.
    Default path: artifacts/pa_logs.csv (created if missing).
    """
    try:
        if os.environ.get("PA_LOG_ENABLE", "0") not in {"1", "true", "TRUE", "yes", "YES"}:
            return
        path = os.environ.get("PA_LOG_PATH")
        if not path:
            root = Path(__file__).resolve().parents[3]
            path = str(root / "artifacts" / "pa_logs.csv")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # flatten probs into deterministic order
        fields = [
            "ts",
            "route",
            "sampled",
            "slice_key",
            "matchup",
            "roof",
        ]
        # preserve probabilities as a compact JSON-ish string
        prob_str = ";".join(f"{k}:{float(v):.6f}" for k, v in sorted(probs.items()))
        row = {
            "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "route": route,
            "sampled": sampled_event,
            "slice_key": slice_meta.get("slice_key", ""),
            "matchup": slice_meta.get("matchup", ""),
            "roof": slice_meta.get("roof", ""),
        }
        header_written = p.exists()
        with p.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields + ["probs"])
            if not header_written:
                w.writeheader()
            row["probs"] = prob_str
            w.writerow(row)
    except Exception:
        # Logging must be best-effort and never break the service
        return

