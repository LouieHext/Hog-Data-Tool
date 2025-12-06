from collections.abc import Callable
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from hog_data_tool.hog_data.session_data import FullSessionData
from hog_data_tool.visualisations.utils import create_figure, set_hog_time_axis, style_axis

type SessionPlotMethod = Callable[[FullSessionData, Path | None], tuple[Figure, Axes]]


def plot_power_curve(
    data: FullSessionData,
    output_path: Path | None = None,
) -> tuple[Figure, Axes]:
    alpha = (1 - data.normalised_session_age) * 0.9

    # scatter plot, opacacity relative date
    fig, ax = create_figure()
    title = f"Power Curve (Weight vs Max Hold Time) ({data.latest_date.date()})"
    x_label = "Weight (lbs)"
    y_label = "Max Hold Time (s)"
    style_axis(ax, title=title, xlabel=x_label, ylabel=y_label)
    set_hog_time_axis(ax)

    ax.scatter(
        data.weight,
        data.max_hold,
        alpha=alpha,
        c="blue",
        s=80, 
        edgecolors='black',
        linewidths=0.5
    )

    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path)
        plt.close(fig)

    return fig, ax


def plot_inverted_power_curve(
    data: FullSessionData,
    output_path: Path | None = None,
) -> tuple[Figure, Axes]:

    alpha = (1 - data.normalised_session_age) * 0.9

    # scatter plot, opacacity relative date
    fig, ax = create_figure()
    title = f"Inverted Power Curve (Weight vs Inverted Max Hold Time) ({data.latest_date.date()})"
    y_label = "Weight (lbs)"
    x_label = "Inverted Max Hold Time (s)"
    style_axis(ax, title=title, xlabel=x_label, ylabel=y_label)

    ax.scatter(
        data.max_hold,
        data.weight,
        c='royalblue',
        alpha=alpha,
        s=80,  # marker size
        edgecolors='black',
        linewidths=0.5
    )

    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path)
        plt.close(fig)

    return fig, ax