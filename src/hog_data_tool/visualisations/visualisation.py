from collections.abc import Callable
from pathlib import Path

from hog_data_tool.hog_data.enums import SessionDataColumn
from hog_data_tool.hog_data.session_data import FullSessionData
from hog_data_tool.visualisations.utils import (
    create_figure,
    save_figure,
    set_hog_time_axis,
    style_axis,
)
from matplotlib.axes import Axes
from matplotlib.figure import Figure

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
        data.weight, data.max_hold, alpha=alpha, c="blue", s=80, edgecolors="black", linewidths=0.5
    )

    save_figure(fig, output_path)

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
        c="royalblue",
        alpha=alpha,
        s=80,  # marker size
        edgecolors="black",
        linewidths=0.5,
    )

    save_figure(fig, output_path)

    return fig, ax


def plot_session_gap(
    data: FullSessionData,
    output_path: Path | None = None,
) -> tuple[Figure, Axes]:

    # line plot of session frequency over time
    fig, ax = create_figure()
    title = f"Session Gap Over Time ({data.latest_date.date()})"
    x_label = "Date"
    y_label = "Time since last session (days)"
    style_axis(ax, title=title, xlabel=x_label, ylabel=y_label)

    session_dates = data.date.sort_values().reset_index(drop=True)
    days_since_last_session = session_dates.diff().dt.days.fillna(0)
    rolling_avg = days_since_last_session.rolling(window=7).mean()

    ax.plot(
        session_dates,
        days_since_last_session,
        marker="o",
        linestyle="-",
        color="green",
        markersize=6,
        linewidth=2,
    )
    ax.plot(
        session_dates,
        rolling_avg,
        marker="o",
        linestyle="-",
        color="orange",
        markersize=6,
        linewidth=2,
    )

    save_figure(fig, output_path)

    return fig, ax


def plot_session_frequency(
    data: FullSessionData,
    output_path: Path | None = None,
) -> tuple[Figure, Axes]:

    fig, ax = create_figure()
    title = f"Sessions Per Week ({data.latest_date.date()})"
    y_label = "Number of Sessions"
    x_label = "Date"
    style_axis(ax, title=title, xlabel=x_label, ylabel=y_label)

    df = data.df.copy()
    df["week"] = df[SessionDataColumn.DATE_TIME].dt.to_period("W").apply(lambda r: r.start_time)
    sessions_per_week = df.groupby("week").size()
    rolling_avg = sessions_per_week.rolling(window=4).mean()

    ax.plot(
        sessions_per_week.index,
        sessions_per_week.values,
        marker="o",
        linestyle="-",
        color="purple",
        markersize=6,
        linewidth=2,
        label="Sessions per Week",
    )
    ax.plot(
        rolling_avg.index,
        rolling_avg.values,
        marker="o",
        linestyle="-",
        color="red",
        markersize=6,
        linewidth=2,
        label="4-Week Rolling Average",
    )

    save_figure(fig, output_path)

    return fig, ax
