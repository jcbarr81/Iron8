import numpy as np
from typing import Dict

def sample_event(probs: Dict[str,float], seed: int | None = None) -> str:
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    classes = list(probs.keys())
    p = np.array([probs[c] for c in classes], dtype=float)
    p = p / p.sum()
    idx = rng.choice(len(classes), p=p)
    return classes[idx]
