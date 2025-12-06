from enum import StrEnum, auto

from matplotlib.pylab import Enum


class SessionEnum(Enum):
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
