from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import cached_property

import pandas as pd

from hog_data_tool.hog_data.enums import SessionDataColumn

type SessionDataFrame = pd.DataFrame


@dataclass()
class FullSessionData:
    df: SessionDataFrame

    def __post_init__(self):
        self._validate_df_obeys_schema()
        # sort df by date time
        self.df = self.df.sort_values(by=SessionDataColumn.DATE_TIME.value)

    @property
    def weight(self) -> pd.Series[int]:
        return self.df[SessionDataColumn.WEIGHT]

    @property
    def max_hold(self) -> pd.Series[int]:
        return self.df[SessionDataColumn.MAX_HOLD]

    @property
    def date(self) -> pd.Series[datetime]:
        return self.df[SessionDataColumn.DATE_TIME]

    @cached_property
    def latest_date(self) -> datetime:
        return self.df[SessionDataColumn.DATE_TIME].max()

    def _validate_df_obeys_schema(self) -> None:
        expected_columns = {col.value for col in SessionDataColumn}
        actual_columns = set(self.df.columns)
        missing_columns = expected_columns - actual_columns
        extra_columns = actual_columns - expected_columns
        if missing_columns:
            raise ValueError(f"FullSessionData is missing expected columns: {missing_columns}")
        if extra_columns:
            raise ValueError(f"FullSessionData has unexpected extra columns: {extra_columns}")
