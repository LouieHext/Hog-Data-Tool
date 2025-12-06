from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import cached_property

import pandas as pd

from hog_data_tool.env_config import get_weight_unit
from hog_data_tool.hog_data.definitions import SessionDataColumn, WeightUnit

type SessionDataFrame = pd.DataFrame


@dataclass()
class FullSessionData:
    df: SessionDataFrame
    label: str = ""
    def __post_init__(self):
        self._validate_df_obeys_schema()

        # sort df by date time
        self.df = self.df.sort_values(by=SessionDataColumn.DATE_TIME.value)

        # drop time zone if column has tz-aware datetimes
        if self.df[SessionDataColumn.DATE_TIME].dt.tz is not None:
            self.df[SessionDataColumn.DATE_TIME] = self.df[
                SessionDataColumn.DATE_TIME
            ].dt.tz_convert(None)

    @property
    def weight_unit(self) -> WeightUnit:
        return get_weight_unit()

    @property
    def number_of_sessions(self) -> int:
        return len(self.df)

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

    def select_sessions_from_range(
        self,
        start_session: int,
        end_session: int,
    ) -> FullSessionData:
        """Select sessions from start_session to end_session (inclusive)."""
        selected_df = self.df.iloc[start_session - 1 : end_session]
        return FullSessionData(df=selected_df, label=self.label)

    @property
    def session_age_days(self) -> pd.Series[int]:
        return (self.latest_date - self.date).dt.days  # pyright: ignore[reportOperatorIssue]

    @cached_property
    def sorted_session_dates(self) -> pd.Series[datetime]:
        return self.date.sort_values().reset_index(drop=True)

    @property
    def normalised_session_age(self) -> pd.Series[float]:
        max_age = self.session_age_days.max()
        return self.session_age_days / max_age

    @cached_property
    def rolling_session_gap_days(self) -> pd.Series[float]:
        days_since_last_session = self.sorted_session_dates.diff().dt.days.fillna(0)
        rolling_avg = days_since_last_session.rolling(window=7).mean()
        return rolling_avg

    @cached_property
    def rolling_sessions_per_week(self) -> pd.Series[int]:
        df = self.df.copy()
        df["week"] = df[SessionDataColumn.DATE_TIME].dt.to_period("W").apply(lambda r: r.start_time)
        sessions_per_week = df.groupby("week").size()
        rolling_avg = sessions_per_week.rolling(window=4).mean()
        return rolling_avg

    def _validate_df_obeys_schema(self) -> None:
        expected_columns = {col.value for col in SessionDataColumn}
        actual_columns = set(self.df.columns)
        missing_columns = expected_columns - actual_columns
        extra_columns = actual_columns - expected_columns
        if missing_columns:
            raise ValueError(f"FullSessionData is missing expected columns: {missing_columns}")
        if extra_columns:
            raise ValueError(f"FullSessionData has unexpected extra columns: {extra_columns}")
