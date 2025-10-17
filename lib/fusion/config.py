from enum import Enum, auto
from dataclasses import dataclass
import os

class FlatFusionMode(Enum):
    Replace_25D = auto()
    Drop = auto()
    Unfold = auto()
    NDFDrop = auto()
    Depyramidize = auto()

@dataclass
class BGPatternFuserConfig:
    hfov_deg: float
    output_dir: str
    flat_mode: FlatFusionMode = FlatFusionMode.NDFDrop
    downsample_dstW: int = 1000

    def __post_init__(self):
        self.output_dir = os.path.join(self.output_dir, "fusion")