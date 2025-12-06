import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from hog_data_tool.hog_data.session_data import FullSessionData


def plot_power_curve(
    data: FullSessionData,
    output_path: Path | None = None,
) -> tuple[Figure, Axes]:

    session_age = (data.latest_date - data.date).dt.days  # pyright: ignore[reportOperatorIssue]
    alpha = 1 - (session_age / session_age.max()) * 0.9

    # scatter plot, opacacity relative date
    fig, ax = _create_figure()
    title = f"Power Curve (Weight vs Max Hold Time) ({data.latest_date.date()})"
    x_label = "Weight (lbs)"
    y_label = "Max Hold Time (s)"
    _style_axis(ax, title=title, xlabel=x_label, ylabel=y_label)
    _set_hog_time_axis(ax)

    ax.scatter(
        data.weight,
        data.max_hold,
        c="blue",
        alpha=alpha,
    )

    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path)
        plt.close(fig)

    return fig, ax


def _set_hog_time_axis(
    ax: Axes,
) -> None:
    """Set consistent time axis formatting for HOG plots."""
    ax.set_yticks([30, 60, 90, 120, 180, 240, 300])
    ax.set_ylim(0, 320)


def _style_axis(
    ax: Axes,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    rotation: int = 45,
    grid: bool = True,
) -> None:
    """Apply consistent styling to an axis."""
    if title:
        ax.set_title(title, fontsize=14, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, labelpad=8)

    for lbl in ax.get_xticklabels():
        lbl.set_rotation(rotation)

    ax.tick_params(axis="both", which="major", labelsize=10)

    if grid:
        ax.grid(axis="y", linestyle="--", alpha=0.7)


def _create_figure(
    figsize: tuple[int, int] = (12, 6),
) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    return fig, ax
