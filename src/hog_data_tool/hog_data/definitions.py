from dataclasses import dataclass
from enum import StrEnum, auto

from matplotlib.pylab import Enum


class RegimeEnum(Enum):
    ONBOARDING = 0
    POWER = 5
    SRENGTH_POWER = 4
    STRENGTH = 3
    STRENGTH_ENDURANCE = 2
    ENDURANCE = 1


class SideEnum(StrEnum):
    LEFT = auto()
    RIGHT = auto()


class GripperEnum(StrEnum):
    CRUSHER = auto()
    MICRO = auto()
    PRIME = auto()


class SessionDataColumn(StrEnum):
    DATE_TIME = auto()
    WEIGHT = auto()
    MAX_HOLD = auto()
 

class WeightUnit(StrEnum):
    LBS = auto()
    KGS = auto()


@dataclass
class HogRegime:
    regime: RegimeEnum
    lower_bound_s: int
    upper_bound_s: int

    @property
    def midpoint_s(self) -> float:
        return (self.lower_bound_s + self.upper_bound_s) / 2


HOG_REGIEME_MAPPINGS = {
    RegimeEnum.POWER: HogRegime(RegimeEnum.POWER, 30, 60),
    RegimeEnum.SRENGTH_POWER: HogRegime(RegimeEnum.SRENGTH_POWER, 61, 90),
    RegimeEnum.STRENGTH: HogRegime(RegimeEnum.STRENGTH, 91, 120),
    RegimeEnum.STRENGTH_ENDURANCE: HogRegime(RegimeEnum.STRENGTH_ENDURANCE, 121, 180),
    RegimeEnum.ENDURANCE: HogRegime(RegimeEnum.ENDURANCE, 181, 300),
}
