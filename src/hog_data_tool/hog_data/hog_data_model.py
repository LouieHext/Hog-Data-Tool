from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from hog_data_tool.env_config import get_weight_unit
from hog_data_tool.hog_data.definitions import (
    GripperEnum,
    SessionDataColumn,
    SideEnum,
    WeightUnit,
)
from hog_data_tool.hog_data.reader import load_hog_data_from_csv
from hog_data_tool.hog_data.session_data import FullSessionData
from hog_data_tool.visualisations.visualisation import (
    SessionPlotMethod,
    SharedSessionPlotMethod,
)

type SessionDataFrame = pd.DataFrame


@dataclass()
class StructuredHogData:
    """
    Container for all hog grip strength session data split by gripper type
    (Micro, Crusher, Prime) and side (left/right).

    Provides helpers for loading data, accessing grouped subsets, and creating plots.
    """

    micro_data: GripperData
    crusher_data: GripperData
    prime_data: GripperData
    other_data: dict[str, GripperData]

    @classmethod
    def from_hog_df(cls, df: pd.DataFrame) -> StructuredHogData:
        """
        Build a StructuredHogData object from a raw hog dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Raw dataframe containing all hog session data.

        Returns
        -------
        StructuredHogData
            Parsed and structured per-gripper data.
        """
        return cls(
            micro_data=GripperData.from_df(df, GripperEnum.MICRO),
            crusher_data=GripperData.from_df(df, GripperEnum.CRUSHER),
            prime_data=GripperData.from_df(df, GripperEnum.PRIME),
            other_data={},
        )

    @classmethod
    def from_csv(cls, path: Path) -> StructuredHogData:
        """
        Load hog session data from a CSV file and construct a StructuredHogData object.

        Parameters
        ----------
        path : Path
            Path to the CSV file.

        Returns
        -------
        StructuredHogData
            Structured dataset parsed from CSV.
        """
        rows = load_hog_data_from_csv(path)
        df = pd.DataFrame([row.model_dump() for row in rows])

        if get_weight_unit() == WeightUnit.KGS:
            df[SessionDataColumn.WEIGHT] /= 2.21

        return cls.from_hog_df(df)

    @classmethod
    def from_csvs(
        cls,
        hog_path: Path,
        other_path: Path | None = None,
    ) -> StructuredHogData:
        """
        Load structured data from the hog CSV and optionally from another path.

        The hog CSV is parsed as HOG data (Micro, Crusher, Prime). If
        `other_path` is set (file or directory), generic session CSVs are
        loaded and split by their "gripper" column; each unique gripper
        becomes one entry in other_data.

        Parameters
        ----------
        hog_path : Path
            Path to the main HOG export CSV.
        other_path : Path, optional
            Path to a single CSV or a directory of CSVs. Each CSV must have
            date_time, reps, rest, weight, max_hold, side, gripper. Gripper
            values are used as keys in other_data.

        Returns
        -------
        StructuredHogData
            Combined structured dataset.
        """
        from hog_data_tool.hog_data.reader import load_generic_session_data

        base = cls.from_csv(hog_path)
        other_data = dict(base.other_data)
        if other_path is not None:
            df = load_generic_session_data(other_path)
            if not df.empty:
                if get_weight_unit() == WeightUnit.KGS:
                    df[SessionDataColumn.WEIGHT] /= 2.21
                for gripper_val in df["gripper"].unique():
                    other_data[str(gripper_val)] = GripperData.from_df(
                        df, gripper_val, filter_by_gripper=True
                    )
        return cls(
            micro_data=base.micro_data,
            crusher_data=base.crusher_data,
            prime_data=base.prime_data,
            other_data=other_data,
        )

    @property
    def all_data(self) -> list[FullSessionData]:
        """Return a list of all left/right session datasets for all grippers."""
        return list(self.named_data_pairs.values())

    @property
    def all_gripper_data(self) -> list[GripperData]:
        """Return a list containing GripperData for Micro, Crusher, Prime and any other_data."""
        return [
            self.crusher_data,
            self.micro_data,
            self.prime_data,
            *self.other_data.values(),
        ]

    @property
    def right_gripper_data(self) -> list[FullSessionData]:
        """Return all right-hand session datasets for all grippers."""
        return [data.right_data for data in self.all_gripper_data]

    @property
    def left_gripper_data(self) -> list[FullSessionData]:
        """Return all left-hand session datasets for all grippers."""
        return [data.left_data for data in self.all_gripper_data]

    @property
    def named_data_pairs(self) -> dict[str, FullSessionData]:
        """
        Return a dictionary mapping human-readable names
        (e.g. 'micro_left') to FullSessionData instances.
        Includes hog grippers plus any other_data keys as {name}_left / {name}_right.
        """
        out: dict[str, FullSessionData] = {
            "micro_left": self.micro_data.left_data,
            "micro_right": self.micro_data.right_data,
            "crusher_left": self.crusher_data.left_data,
            "crusher_right": self.crusher_data.right_data,
            "prime_left": self.prime_data.left_data,
            "prime_right": self.prime_data.right_data,
        }
        for name, gd in self.other_data.items():
            out[f"{name}_left"] = gd.left_data
            out[f"{name}_right"] = gd.right_data
        return out

    def create_plot_for_all_grippers(
        self,
        plot_method: SessionPlotMethod,
        output_path: Path,
        min_sessions: int = 30,
        **kwargs,
    ) -> dict[str, tuple[Figure, Axes]]:
        """
        Create a separate plot for each gripper/side dataset.

        Parameters
        ----------
        plot_method : SessionPlotMethod
            Callable that receives a FullSessionData instance and produces a plot.
        output_path : Path
            Directory in which plots will be saved.
        min_sessions : int, default=10
            Skip grippers with fewer than this many sessions.

        Returns
        -------
        dict[str, tuple[Figure, Axes]]
            Mapping names (e.g. 'micro_left') to their created figure/axes.
        """
        results = {}
        for name, gripper_data in self.named_data_pairs.items():
            if gripper_data.number_of_sessions < min_sessions:
                print(
                    f"Skipping {name}: only {gripper_data.number_of_sessions} sessions (< {min_sessions})"
                )
                continue
            results[name] = plot_method(
                gripper_data,
                output_path / f"{name}_plot.png",
                **kwargs,
            )
        return results

    def create_shared_gripper_plot(
        self,
        plot_method: SharedSessionPlotMethod,
        output_path: Path,
        only_show_right: bool = True,
        **kwargs,
    ) -> tuple[Figure, Axes]:
        """
        Create a single plot shared across multiple session datasets.

        Parameters
        ----------
        plot_method : SharedSessionPlotMethod
            Function taking a list of FullSessionData objects and producing a shared plot.
        output_path : Path
            Where to save the resulting plot.
        only_show_right : bool, default=True
            If True, only include right-hand sessions.

        Returns
        -------
        (Figure, Axes)
            Matplotlib figure and axes.
        """
        all_data = self.all_data
        if only_show_right:
            all_data = self.right_gripper_data

        return plot_method(all_data, output_path, **kwargs)


@dataclass()
class GripperData:
    """
    Container for left and right session datasets for a single gripper type.
    `gripper` is GripperEnum for hog data (Micro, Crusher, Prime) or str for other sources.
    """

    gripper: GripperEnum | str
    left_data: FullSessionData
    right_data: FullSessionData

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        gripper: GripperEnum | str,
        *,
        filter_by_gripper: bool = True,
    ) -> GripperData:
        """
        Construct GripperData by splitting the dataframe on side (left/right).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with SessionDataColumn columns and "side". If
            filter_by_gripper is True, must also have a "gripper" column.
        gripper : GripperEnum | str
            Gripper or source identifier. When filter_by_gripper is True,
            only rows where df["gripper"] == gripper are used (hog-style).
        filter_by_gripper : bool, default True
            If True, filter to rows with this gripper before splitting by side.
            If False, use all rows (e.g. generic CSV with no gripper column).

        Returns
        -------
        GripperData
            Left and right datasets for this gripper/source.
        """
        if filter_by_gripper:
            df = df[df["gripper"] == gripper]
        cols = [col.value for col in SessionDataColumn]
        left_df = df[df["side"] == SideEnum.LEFT][cols].copy()
        right_df = df[df["side"] == SideEnum.RIGHT][cols].copy()
        label = str(gripper)
        return cls(
            gripper=gripper,
            left_data=FullSessionData(df=left_df, label=f"{label} Left"),
            right_data=FullSessionData(df=right_df, label=f"{label} Right"),
        )
