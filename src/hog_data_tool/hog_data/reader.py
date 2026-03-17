from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import pandas as pd
import pydantic
from pydantic import BaseModel, Field, field_validator

from hog_data_tool.hog_data.definitions import (
    GripperEnum,
    RegimeEnum,
    SessionDataColumn,
    SideEnum,
)


class HogDataRow(BaseModel):
    """
    Pydantic model representing a single row from the raw HOG CSV export.

    Fields correspond directly to CSV columns.

    This model is used prior to conversion into analytic DataFrames.
    """

    model_config = pydantic.ConfigDict(
        use_enum_values=True,
        strict=False,
        arbitrary_types_allowed=True,
        extra="ignore",
    )

    session_number: RegimeEnum
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
    def coerce_session_number(cls, v) -> int:
        """
        Ensure that the session_number is an integer.
        Accepts numeric strings and converts them to int.
        """
        if isinstance(v, str) and v.isdigit():
            return int(v)
        raise ValueError(f"session number {v} is not an integer")


def load_hog_data_from_csv(path: Path) -> list[HogDataRow]:
    """
    Load HOG data from a CSV file into a list of HogDataRow objects.

    Args:
        path: Path to the CSV file.

    Returns:
        A list of validated HogDataRow instances.
    """
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = [HogDataRow.model_validate(row) for row in reader]
    return rows


def load_generic_session_csv(path: Path) -> pd.DataFrame:
    """
    Load a generic session CSV (e.g. from alternative data sources) into a DataFrame.

    Expected columns: date_time, reps, rest, weight, max_hold, side, gripper.
    Side values should be "left" / "right" (case-insensitive).
    gripper is used to split data into multiple GripperData (any string values).
    date_time is parsed as datetime.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame with SessionDataColumn columns plus "side" and "gripper".

    Raises:
        ValueError: If required columns are missing or invalid.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    required = {col.value for col in SessionDataColumn} | {"side", "gripper"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Generic session CSV missing columns: {missing}")
    
    df = df[[col.value for col in SessionDataColumn] + ["side", "gripper"]].copy()
    df[SessionDataColumn.DATE_TIME] = pd.to_datetime(df[SessionDataColumn.DATE_TIME])

    side_map = {"left": SideEnum.LEFT, "right": SideEnum.RIGHT}
    df["side"] = df["side"].astype(str).str.strip().str.lower().map(side_map)
    if df["side"].isna().any():
        raise ValueError(
            "Generic session CSV 'side' column must contain only 'left' or 'right'"
        )

    df["gripper"] = df["gripper"].astype(str).str.strip()

    return df


def load_generic_session_data(path: Path) -> pd.DataFrame:
    """
    Load generic session data from a path (file or directory).

    If path is a file, loads that CSV. If path is a directory, loads all
    *.csv files and concatenates them. Expects the same schema as
    load_generic_session_csv (including "gripper" column).

    Args:
        path: Path to a CSV file or a directory of CSV files.

    Returns:
        Single DataFrame with SessionDataColumn + side + gripper.
    """
    if path.is_file():
        return load_generic_session_csv(path)

    if path.is_dir():
        paths = sorted(path.glob("*.csv"))
        if not paths:
            return pd.DataFrame()
        return pd.concat(
            [load_generic_session_csv(p) for p in paths],
            ignore_index=True,
        )
    raise ValueError(f"Path is neither file nor directory: {path}")
