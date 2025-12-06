from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from hog_data_tool.hog_data.enums import GripperEnum, SessionDataColumn, SideEnum
from hog_data_tool.hog_data.reader import load_hog_data_from_csv
from hog_data_tool.hog_data.session_data import FullSessionData
from hog_data_tool.visualisations.visualisation import (
    SessionPlotMethod,
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
        return cls.from_hog_df(df)

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
        self, plot_method: SessionPlotMethod, output_path: Path
    ) -> None:
        for name, gripper_data in self.named_data_pairs.items():
            plot_method(
                gripper_data,
                output_path=output_path / f"{name}_plot.png",
            )


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
            left_data=FullSessionData(df=left_df),
            right_data=FullSessionData(df=right_df),
        )
