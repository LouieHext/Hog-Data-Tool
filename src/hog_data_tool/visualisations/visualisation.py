from collections.abc import Callable
from pathlib import Path

from hog_data_tool.analysis.curve_fit import fit_power_curve_with_hyerbolic_decay
from hog_data_tool.analysis.progress import rolling_average_weight_in_regiemes
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
type SharedSessionPlotMethod = Callable[[list[FullSessionData], Path | None], tuple[Figure, Axes]]


def plot_power_curve(
    data: FullSessionData,
    output_path: Path | None = None,
    show_curve_fit: bool = True
) -> tuple[Figure, Axes]:
    alpha = (1 - data.normalised_session_age) * 0.9

    # scatter plot, opacacity relative date
    fig, ax = create_figure()
    title = f"Power Curve (Weight vs Max Hold Time) ({data.latest_date.date()})"
    x_label = f"Weight ({data.weight_unit})"
    y_label = "Max Hold Time (s)"
    style_axis(ax, title=title, xlabel=x_label, ylabel=y_label)
    set_hog_time_axis(ax)

    ax.scatter(
        data.weight, data.max_hold, alpha=alpha, c="blue", s=80, edgecolors="black", linewidths=0.5
    )

    if show_curve_fit:
        curve_fit = fit_power_curve_with_hyerbolic_decay(data.weight, data.max_hold)
        weights = data.weight.sort_values().to_numpy()
        hold_fit = curve_fit.predict(weights)

        ax.plot(
            weights,
            hold_fit,
            color="red",
            linewidth=2,
        )
    
    save_figure(fig, output_path)

    return fig, ax


def plot_rolling_average_weight_in_regiemes(
    data: FullSessionData,
    output_path: Path | None = None,
) -> tuple[Figure, Axes]:

    regieme_weights = rolling_average_weight_in_regiemes(data)
    if not regieme_weights:
        return

    fig, ax = create_figure()
    title = f"Rolling Average Weight in Regiemes"
    x_label = "Session date"
    y_label = f"Weight ({data.weight_unit})"
    style_axis(ax, title=title, xlabel=x_label, ylabel=y_label)

    for regieme, results in regieme_weights.items():
        weights = [r[0] for r in results]
        dates = [r[1] for r in results]
        ax.plot(
            dates,
            weights,
            marker="o",
            linestyle="-",
            markersize=6,
            linewidth=2,
            label=regieme.name,
        )

    # put legend outside plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
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
    y_label = f"Weight ({data.weight_unit})"
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
    data: FullSessionData | list[FullSessionData],
    output_path: Path | None = None,
) -> tuple[Figure, Axes]:

    if not (isinstance(data, list)):
        data = [data]

    # line plot of session frequency over time
    dates = [data.latest_date.date() for data in data]
    fig, ax = create_figure()
    title = f"Session Gap Over Time ({max(dates)})"
    x_label = "Date"
    y_label = "Time since last session (days)"
    style_axis(ax, title=title, xlabel=x_label, ylabel=y_label)

    upper_value = 0
    for session_data in data:
        rolling_gap_na_indexs = session_data.rolling_session_gap_days.dropna().index
        rolling_gap_days = session_data.rolling_session_gap_days.dropna()
        start_dates = session_data.sorted_session_dates.loc[rolling_gap_na_indexs]

        if rolling_gap_days.empty:
            continue

        upper_value = max(upper_value, rolling_gap_days.max())

        ax.plot(
            start_dates,
            rolling_gap_days,
            marker="o",
            linestyle="-",
            markersize=6,
            linewidth=2,
            label=session_data.label,
        )

    ax.set_ylim(bottom=0, top=1.5 * upper_value)
    ax.legend()

    save_figure(fig, output_path)

    return fig, ax


def plot_session_frequency(
    data: FullSessionData | list[FullSessionData],
    output_path: Path | None = None,
) -> tuple[Figure, Axes]:

    if not (isinstance(data, list)):
        data = [data]

    dates = [data.latest_date.date() for data in data]

    fig, ax = create_figure()
    title = f"Sessions Per Week ({max(dates)})"
    y_label = "Number of Sessions"
    x_label = "Date"
    style_axis(ax, title=title, xlabel=x_label, ylabel=y_label)

    upper_value = 0
    for session_data in data:
        rolling_sessions = session_data.rolling_sessions_per_week.dropna()
        if rolling_sessions.empty:
            continue
        upper_value = max(upper_value, rolling_sessions.max())
        ax.plot(
            rolling_sessions.index,
            rolling_sessions.values,
            marker="o",
            linestyle="-",
            markersize=6,
            linewidth=2,
            label=session_data.label,
        )

    ax.set_ylim(bottom=0, top=1.5 * upper_value)
    ax.legend()
    save_figure(fig, output_path)

    return fig, ax
