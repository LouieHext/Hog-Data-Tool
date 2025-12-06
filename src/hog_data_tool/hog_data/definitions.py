from dataclasses import dataclass
from enum import StrEnum, auto
from matplotlib.pylab import Enum


class RegiemeEnum(Enum):
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
    REPS = auto()
    REST = auto()
    WEIGHT = auto()
    MAX_HOLD = auto()
    VOLUME = auto()
    power = auto()
    success_power = auto()
    anaerobic = auto()
    success_anaerobic = auto()
    success_aerobic = auto()


@dataclass
class HogRegieme:
    regieme: RegiemeEnum
    lower_bound_s: int
    upper_bound_s: int

    @property
    def midpoint_s(self) -> float:
        return (self.lower_bound_s + self.upper_bound_s) / 2


HOG_REGIEME_MAPPINGS = {
    RegiemeEnum.POWER: HogRegieme(RegiemeEnum.POWER, 30, 60),
    RegiemeEnum.SRENGTH_POWER: HogRegieme(RegiemeEnum.SRENGTH_POWER, 61, 90),
    RegiemeEnum.STRENGTH: HogRegieme(RegiemeEnum.STRENGTH, 91, 120),
    RegiemeEnum.STRENGTH_ENDURANCE: HogRegieme(RegiemeEnum.STRENGTH_ENDURANCE, 121, 180),
    RegiemeEnum.ENDURANCE: HogRegieme(RegiemeEnum.ENDURANCE, 181, 300),
}
