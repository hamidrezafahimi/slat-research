from enum import Enum, auto
from dataclasses import dataclass

class FlatFusionMode(Enum):
    Replace_25D = auto()
    Drop = auto()
    Unfold = auto()
    ndfDrop = auto()

@dataclass
class BGPatternFuserConfig:
    hfov_deg: float
    flat_mode: FlatFusionMode = FlatFusionMode.ndfDrop