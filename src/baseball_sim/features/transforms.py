from typing import Optional

def z_0_99(x: Optional[int], center: float = 50.0, scale: float = 15.0,
           lo: float = -3.0, hi: float = 3.0) -> Optional[float]:
    if x is None:
        return None
    z = (float(x) - center) / scale
    return max(lo, min(hi, z))
