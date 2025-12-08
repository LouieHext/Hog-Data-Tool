from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import pandas as pd

from hog_data_tool.env_config import get_weight_unit
from hog_data_tool.hog_data.definitions import SessionDataColumn, WeightUnit

type SessionDataFrame = pd.DataFrame


@dataclass()
class FullSessionData:
    df: SessionDataFrame
    label: str = ""

    def __post_init__(self) -> None:
        self.df = self.df.copy()

        self._validate_df_obeys_schema()
        self._handle_timezone_data()

        self.df = self.df.sort_values(by=SessionDataColumn.DATE_TIME.value)

        self._compute_additional_features()

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
    def date(self) -> pd.Series[pd.Timestamp]:
        return self.df[SessionDataColumn.DATE_TIME]

    @cached_property
    def latest_date(self) -> pd.Timestamp:
        return self.df[SessionDataColumn.DATE_TIME].max()

    @property
    def session_age_days(self) -> pd.Series[int]:
        return self.df["session_age_days"]

    @property
    def session_gap_in_days(self) -> pd.Series[int]:
        return self.df["session_gap_days"]

    @property
    def sessions_per_week(self) -> pd.Series[float]:
        return self.df["sessions_per_week"]

    @cached_property
    def normalised_session_age(self) -> pd.Series[float]:
        max_age = self.session_age_days.max()
        return self.session_age_days / max_age

    @cached_property
    def rolling_session_gap_days(self) -> pd.Series[float]:
        return self.session_gap_in_days.rolling(window=7).mean()

    @cached_property
    def rolling_sessions_per_week(self) -> pd.Series[int]:
        return self.sessions_per_week.rolling(window=4).mean()

    def select_sessions_from_range(
        self,
        start_session: int,
        end_session: int,
    ) -> FullSessionData:
        """Select sessions from start_session to end_session (inclusive)."""
        selected_df = self.df.iloc[start_session - 1 : end_session]
        return FullSessionData(df=selected_df, label=self.label)

    def _compute_additional_features(self) -> None:
        """Injects some useful additional features into the df"""

        # if additional features already computed, skip
        if "session_age_days" in self.df.columns:
            return

        self.df["session_age_days"] = (
            self.latest_date - self.date  # pyright: ignore[reportOperatorIssue]
        ).dt.days
        self.df["session_gap_days"] = self.date.diff().dt.days.fillna(0)
        self.df["week"] = self.date.dt.to_period("W").dt.start_time
        self.df["sessions_per_week"] = self.df.groupby("week").transform("size")

    def _validate_df_obeys_schema(self) -> None:
        expected_columns = {col.value for col in SessionDataColumn}
        actual_columns = set(self.df.columns)
        missing_columns = expected_columns - actual_columns
        if missing_columns:
            raise ValueError(f"FullSessionData is missing expected columns: {missing_columns}")

    def _handle_timezone_data(self) -> None:
        # drop time zone if column has tz-aware datetimes
        if self.date.dt.tz is not None:
            self.df[SessionDataColumn.DATE_TIME] = self.date.dt.tz_convert(None)
