from __future__ import annotations
import csv
from datetime import datetime
from enum import Enum, StrEnum, auto
from functools import cached_property
from typing import Any
from pydantic import BaseModel, Field
import pydantic
import pandas as pd

class SessinoEnum(Enum):
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

class HogSessionConfig(BaseModel):
    power: float = Field(ge=0)
    sucess_power: float = Field(ge=0)
    anaerobic: float = Field(ge=0)
    success_anaerobic: float = Field(ge=0)
    aerobic: float = Field(ge=0)
    success_aerobic: float = Field(ge=0)


def extract_session_config(data: dict) -> dict:
    return dict(
        power=data.get("power", 0),
        success_power=data.get("success_power", 0),
        anaerobic=data.get("anaerobic", 0),
        success_anaerobic=data.get("success_anaerobic", 0),
        aerobic=data.get("aerobic", 0),
        success_aerobic=data.get("success_aerobic", 0),
    )

class HogDataRow(BaseModel):
    session_number: SessinoEnum
    date_time: datetime
    side: SideEnum
    gripper: GripperEnum
    reps: int = Field(gt=0)
    rest: float = Field(ge=0)
    weight: float = Field(ge=0)
    max_hold: int = Field(ge=0)
    volume: float = Field(ge=0)
    session_config: HogSessionConfig

    # pre validate to build session config
    @classmethod
    @pydantic.model_validator(mode="before")
    def parse_session_config(cls, data: dict[str, Any]) -> dict[str, Any]:
        session_config = HogSessionConfig(
            **extract_session_config(data)
        )
        data["session_config"] = session_config
        return data
    
class HogData(BaseModel):
    data: list[HogDataRow]

    @classmethod
    @pydantic.model_validator(mode="before")
    def sort_data_by_date(cls, data: dict[str, Any]) -> dict[str, Any]:
        data["data"] = sorted(
            data.get("data", []),
            key=lambda row: row["date_time"],
        )
        return data

    @cached_property
    def full_df(self) -> pd.DataFrame:
        return pd.DataFrame([row.model_dump() for row in self.data])
    
    @cached_property
    def latest_date(self) -> datetime:
        return max(row.date_time for row in self.data)
    
    @classmethod
    def from_csv(cls, path: str) -> HogData:
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = [HogDataRow.model_validate(row) for row in reader]
        return cls(data=rows)

def remove_onboarding_sessions(df: pd.DataFrame, onboarding_session_count: int = 30) -> pd.DataFrame:
    """Remove onboarding sessions from the DataFrame."""
    sorted_df = df.sort_values(by="date_time").reset_index(drop=True)
    return sorted_df.iloc[onboarding_session_count:].reset_index(drop=True)
    