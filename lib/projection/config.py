
from enum import Enum, auto
from dataclasses import dataclass
import os

class Scaling(Enum):
    NULL = 0
    MIN_Z   = 1 
    MEAN_Z    = 2 
    RESHAPE_BG_Z  = 3 

class VisMode(Enum):
    Null = auto()
    MSingle = auto()
    MAccum = auto()

@dataclass
class Mapper3DConfig:
    hfov_deg: float
    output_dir: str
    visMode: VisMode = VisMode.MAccum
    shape: tuple = None
    color_mode: str = 'constant'  # 'image' | 'proximity' | 'constant' | 'none'
    mesh_u: int = 40
    mesh_v: int = 40
    downsample_dstW: int = 5280
    scaling: Scaling = Scaling.MIN_Z

    def __post_init__(self):
        self.output_dir = os.path.join(self.output_dir, "rawdepth")
