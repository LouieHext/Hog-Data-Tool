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

    micro_data: GripperData
    crusher_data: GripperData
    prime_data: GripperData

    @classmethod
    def from_hog_df(cls, df: pd.DataFrame) -> StructuredHogData:
        return cls(
            micro_data=GripperData.from_hog_df(df, GripperEnum.MICRO),
            crusher_data=GripperData.from_hog_df(df, GripperEnum.CRUSHER),
            prime_data=GripperData.from_hog_df(df, GripperEnum.PRIME),
        )

    @classmethod
    def from_csv(cls, path: Path) -> StructuredHogData:
        rows = load_hog_data_from_csv(path)
        df = pd.DataFrame([row.model_dump() for row in rows])

        if get_weight_unit() == WeightUnit.KGS:
            df[SessionDataColumn.WEIGHT] /= 2.21
        return cls.from_hog_df(df)

    @property
    def all_data(self) -> list[FullSessionData]:
        return list(self.named_data_pairs.values())

    @property
    def all_gripper_data(self) -> list[GripperData]:
        return [self.crusher_data, self.micro_data, self.prime_data]

    @property
    def right_gripper_data(self) -> list[FullSessionData]:
        return [data.right_data for data in self.all_gripper_data]

    @property
    def left_gripper_data(self) -> list[FullSessionData]:
        return [data.left_data for data in self.all_gripper_data]

    @property
    def named_data_pairs(self) -> dict[str, FullSessionData]:
        return {
            "micro_left": self.micro_data.left_data,
            "micro_right": self.micro_data.right_data,
            "crusher_left": self.crusher_data.left_data,
            "crusher_right": self.crusher_data.right_data,
            "prime_left": self.prime_data.left_data,
            "prime_right": self.prime_data.right_data,
        }

    def create_plot_for_all_grippers(
        self, plot_method: SessionPlotMethod, output_path: Path, **kwargs
    ) -> dict[str, tuple[Figure, Axes]]:
        results = {}
        for name, gripper_data in self.named_data_pairs.items():
            results[name] = plot_method(gripper_data, output_path / f"{name}_plot.png", **kwargs)
        return results

    def create_shared_gripper_plot(
        self,
        plot_method: SharedSessionPlotMethod,
        output_path: Path,
        only_show_right: bool = True,
        **kwargs,
    ) -> tuple[Figure, Axes]:

        all_data = self.all_data
        if only_show_right:
            all_data = self.right_gripper_data

        return plot_method(all_data, output_path, **kwargs)


@dataclass()
class GripperData:

    gripper: GripperEnum
    left_data: FullSessionData
    right_data: FullSessionData

    @classmethod
    def from_hog_df(cls, df: pd.DataFrame, gripper: GripperEnum) -> GripperData:
        left_df = df[(df["gripper"] == gripper) & (df["side"] == SideEnum.LEFT)]
        right_df = df[(df["gripper"] == gripper) & (df["side"] == SideEnum.RIGHT)]

        # filter on session data columns now that filtering is done
        left_df = left_df[[col.value for col in SessionDataColumn]]
        right_df = right_df[[col.value for col in SessionDataColumn]]

        return cls(
            gripper=gripper,
            left_data=FullSessionData(df=left_df, label=f"{gripper.value} Left"),
            right_data=FullSessionData(df=right_df, label=f"{gripper.value} Right"),
        )
