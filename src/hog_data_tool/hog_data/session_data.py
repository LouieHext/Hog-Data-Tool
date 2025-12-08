from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import pandas as pd

from hog_data_tool.env_config import get_weight_unit
from hog_data_tool.hog_data.definitions import SessionDataColumn, WeightUnit

type SessionDataFrame = pd.DataFrame


@dataclass()
class FullSessionData:
    """
    Container for all session data for a single gripper/side.

    This class wraps a pandas DataFrame containing HOG session data
    and computes several useful derived features such as:
    - session age (days since the latest session)
    - gaps between sessions
    - sessions per week
    - rolling statistics for usage patterns

    Parameters
    ----------
    df : SessionDataFrame
        Raw session dataframe. Will be copied internally.
    label : str, optional
        Name used for legends/plots.
    """

    df: SessionDataFrame
    label: str = ""

    def __post_init__(self) -> None:
        """
        Validate schema, normalise datetime fields, sort the dataframe,
        and compute all additional derived features.
        """
        self.df = self.df.copy()

        self._validate_df_obeys_schema()
        self._handle_timezone_data()

        self.df = self.df.sort_values(by=SessionDataColumn.DATE_TIME.value)

        self._compute_additional_features()

    @property
    def weight_unit(self) -> WeightUnit:
        """Return the currently active global weight unit."""
        return get_weight_unit()

    @property
    def number_of_sessions(self) -> int:
        """Return the total number of sessions in this dataset."""
        return len(self.df)

    @property
    def weight(self) -> pd.Series[int]:
        """Return the session weight column."""
        return self.df[SessionDataColumn.WEIGHT]

    @property
    def max_hold(self) -> pd.Series[int]:
        """Return the max hold column for each session."""
        return self.df[SessionDataColumn.MAX_HOLD]

    @property
    def date(self) -> pd.Series[pd.Timestamp]:
        """Return the datetime column representing session timestamps."""
        return self.df[SessionDataColumn.DATE_TIME]

    @cached_property
    def latest_date(self) -> pd.Timestamp:
        """Return the timestamp of the most recent session."""
        return self.df[SessionDataColumn.DATE_TIME].max()

    @property
    def session_age_days(self) -> pd.Series[int]:
        """Return the number of days from each session to the latest one."""
        return self.df["session_age_days"]

    @property
    def session_gap_in_days(self) -> pd.Series[int]:
        """Return day differences between consecutive sessions."""
        return self.df["session_gap_days"]

    @cached_property
    def normalised_session_age(self) -> pd.Series[float]:
        """
        Return session age scaled to [0, 1], where 1 = oldest session
        and 0 = most recent.
        """
        max_age = self.session_age_days.max()
        return self.session_age_days / max_age

    @cached_property
    def rolling_session_gap_days(self) -> pd.Series[float]:
        """Return a 7-session rolling mean of day gaps between sessions."""
        return self.session_gap_in_days.rolling(window=7).mean()

    @cached_property
    def rolling_sessions_per_week(self) -> pd.Series[int]:
        """Return a 4-week rolling mean of weekly session counts."""
        sessions_per_week = self.df.groupby("week").size()
        return sessions_per_week.rolling(window=4).mean()

    def select_sessions_from_range(
        self,
        start_session: int,
        end_session: int,
    ) -> FullSessionData:
        """
        Return a new FullSessionData object containing only sessions within
        the given index range (1-indexed inclusive).

        Parameters
        ----------
        start_session : int
            First session index to include.
        end_session : int
            Last session index to include.

        Returns
        -------
        FullSessionData
            A new instance with the subset of sessions.
        """
        selected_df = self.df.iloc[start_session - 1 : end_session]
        return FullSessionData(df=selected_df, label=self.label)

    def _compute_additional_features(self) -> None:
        """
        Compute and inject additional derived columns:
        - session_age_days
        - session_gap_days
        - week (start-of-week timestamp)
        - sessions_per_week

        No-op if these features already exist.
        """
        if "session_age_days" in self.df.columns:
            return

        self.df["session_age_days"] = (
            self.latest_date - self.date  # pyright: ignore[reportOperatorIssue]
        ).dt.days

        self.df["session_gap_days"] = self.date.diff().dt.days.fillna(0)
        self.df["week"] = self.date.dt.to_period("W").dt.start_time

    def _validate_df_obeys_schema(self) -> None:
        """
        Ensure the dataframe contains all required columns defined in
        SessionDataColumn. Raise ValueError if any are missing.
        """
        expected = {col.value for col in SessionDataColumn}
        actual = set(self.df.columns)
        missing = expected - actual
        if missing:
            raise ValueError(f"FullSessionData is missing expected columns: {missing}")

    def _handle_timezone_data(self) -> None:
        """
        Convert timezone-aware timestamps to naive ones if necessary.
        This keeps all time calculations consistent.
        """
        if self.date.dt.tz is not None:
            self.df[SessionDataColumn.DATE_TIME] = self.date.dt.tz_convert(None)
