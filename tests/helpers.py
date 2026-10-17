from __future__ import annotations

import numpy as np


def unit(*values: float) -> list[float]:
    v = np.array(values, dtype=float)
    return (v / np.linalg.norm(v)).tolist()
