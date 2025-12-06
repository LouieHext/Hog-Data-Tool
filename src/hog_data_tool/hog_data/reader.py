from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import pydantic
from pydantic import BaseModel, Field, field_validator

from hog_data_tool.hog_data.definitions import GripperEnum, RegiemeEnum, SideEnum


class HogDataRow(BaseModel):

    model_config = pydantic.ConfigDict(
        use_enum_values=True,
        strict=False,
        arbitrary_types_allowed=True,
        extra="ignore",
    )

    session_number: RegiemeEnum
    date_time: datetime
    side: SideEnum
    gripper: GripperEnum
    reps: int = Field(gt=0)
    rest: float = Field(ge=0)
    weight: float = Field(ge=0)
    max_hold: int = Field(ge=0)
    volume: float = Field(ge=0)
    power: float = Field(ge=0)
    success_power: float = Field(ge=0)
    anaerobic: float = Field(ge=0)
    success_anaerobic: float = Field(ge=0)
    success_aerobic: float = Field(ge=0)

    @field_validator("session_number", mode="before")
    def coerce_session_number(cls, v):
        if isinstance(v, str) and v.isdigit():
            return int(v)
        return v


def load_hog_data_from_csv(path: Path) -> list[HogDataRow]:
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = [HogDataRow.model_validate(row) for row in reader]
    return rows
