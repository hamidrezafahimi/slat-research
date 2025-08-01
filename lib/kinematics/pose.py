from enum import Enum, auto
from dataclasses import dataclass
import math

class Rot(Enum):
    RAD = auto()
    DEG = auto()

def deg2rad(deg): return math.radians(deg)
def rad2deg(rad): return math.degrees(rad)

@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    z: float

    _roll_rad: float
    _pitch_rad: float
    _yaw_rad: float

    _roll_deg: float
    _pitch_deg: float
    _yaw_deg: float

    def __init__(self, *, x, y, z, roll, pitch, yaw, rot: Rot = Rot.RAD):
        object.__setattr__(self, 'x', x)
        object.__setattr__(self, 'y', y)
        object.__setattr__(self, 'z', z)

        if rot == Rot.RAD:
            object.__setattr__(self, '_roll_rad', roll)
            object.__setattr__(self, '_pitch_rad', pitch)
            object.__setattr__(self, '_yaw_rad', yaw)

            object.__setattr__(self, '_roll_deg', rad2deg(roll))
            object.__setattr__(self, '_pitch_deg', rad2deg(pitch))
            object.__setattr__(self, '_yaw_deg', rad2deg(yaw))
        elif rot == Rot.DEG:
            object.__setattr__(self, '_roll_deg', roll)
            object.__setattr__(self, '_pitch_deg', pitch)
            object.__setattr__(self, '_yaw_deg', yaw)

            object.__setattr__(self, '_roll_rad', deg2rad(roll))
            object.__setattr__(self, '_pitch_rad', deg2rad(pitch))
            object.__setattr__(self, '_yaw_rad', deg2rad(yaw))
        else:
            raise ValueError("rot must be Rot.RAD or Rot.DEG")

    def rpy_rad(self):
        return self._roll_rad, self._pitch_rad, self._yaw_rad

    def rpy_deg(self):
        return self._roll_deg, self._pitch_deg, self._yaw_deg
