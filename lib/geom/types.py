import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Background:
    points_xyz: np.ndarray      # (N,3) array of grid points
    grid_w: int               # grid width
    grid_h: int               # grid height
    T_inv: np.ndarray           # (4,4) inverse transform matrix
    metadata: Dict[str, Any]    # additional info
