from collections.abc import Callable
from pathlib import Path

from hog_data_tool.analysis.curve_fit import fit_power_curve_with_hyperbolic_decay
from hog_data_tool.analysis.progress import (
    find_sparse_weight,
    rolling_average_weight_in_regimes,
)
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
type SharedSessionPlotMethod = Callable[
    [list[FullSessionData] | FullSessionData, Path | None], tuple[Figure, Axes]
]


def plot_power_curve(
    data: FullSessionData, output_path: Path | None = None, show_curve_fit: bool = True
) -> tuple[Figure, Axes]:
    """
    Plot a scatter of weight vs max hold time for a session, optionally overlaying
    a hyperbolic curve fit.

    Args:
        data: FullSessionData containing weights and max hold times.
        output_path: Optional path to save the generated figure.
        show_curve_fit: Whether to overlay the fitted hyperbolic curve.

    Returns:
        A tuple containing the matplotlib Figure and Axes objects.
    """
    alpha = (1 - data.normalised_session_age) * 0.9

    # scatter plot, opacacity relative date
    fig, ax = create_figure()
    title = f"Power Curve (Weight vs Max Hold Time) ({data.latest_date.date()})"
    x_label = f"Weight ({data.weight_unit})"
    y_label = "Max Hold Time (s)"
    style_axis(ax, title=title, x_label=x_label, y_label=y_label)
    set_hog_time_axis(ax)

    ax.scatter(
        data.weight,
        data.max_hold,
        alpha=alpha,  # pyright: ignore[reportArgumentType]
        c="blue",
        s=80,
        edgecolors="black",
        linewidths=0.5,  # pyright: ignore[reportArgumentType]
    )

    if show_curve_fit:
        curve_fit = fit_power_curve_with_hyperbolic_decay(data.weight, data.max_hold)
        weights = data.weight.sort_values().to_numpy()
        hold_fit = curve_fit.predict(weights)

        ax.plot(
            weights,
            hold_fit,
            color="red",
            linewidth=2,
        )

        # find least dense region and suggest a weight
        weight = find_sparse_weight(data=data)
        hold_time = curve_fit.predict(weight)
        # add this as single point and include in legend
        ax.plot(
            weight,
            hold_time,
            color="black",
            marker="*",
            markersize=10,
            label=f"Suggest weight {round(weight, 2)} {data.weight_unit.name}",
        )
        ax.legend()

    save_figure(fig, output_path)

    return fig, ax


def plot_rolling_average_weight_in_regimes(
    data: FullSessionData,
    output_path: Path | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot the rolling average predicted weight for each HOG regime over time.

    Args:
        data: FullSessionData containing session weights and hold times.
        output_path: Optional path to save the generated figure.

    Returns:
        A tuple containing the matplotlib Figure and Axes objects.
    """

    fig, ax = create_figure()

    regime_weights = rolling_average_weight_in_regimes(data)

    if not regime_weights:
        return fig, ax

    title = "Rolling Average Weight in Regimes"
    x_label = "Session date"
    y_label = f"Weight ({data.weight_unit})"
    style_axis(ax, title=title, x_label=x_label, y_label=y_label)

    for regime, results in regime_weights.items():
        weights = [r[0] for r in results]
        dates = [r[1] for r in results]
        ax.plot(
            dates,
            weights,
            marker="o",
            linestyle="-",
            markersize=6,
            linewidth=2,
            label=regime.name,
        )

    # put legend outside plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    save_figure(fig, output_path)

    return fig, ax


def plot_inverted_power_curve(
    data: FullSessionData, output_path: Path | None = None, show_curve_fit: bool = True
) -> tuple[Figure, Axes]:
    """
    Plot weight against max hold time in an inverted format, optionally overlaying
    the inverted hyperbolic curve fit.

    Args:
        data: FullSessionData containing weights and max hold times.
        output_path: Optional path to save the generated figure.
        show_curve_fit: Whether to overlay the inverted fitted curve.

    Returns:
        A tuple containing the matplotlib Figure and Axes objects.
    """

    alpha = (1 - data.normalised_session_age) * 0.9

    # scatter plot, opacacity relative date
    fig, ax = create_figure()
    title = f"Inverted Power Curve (Weight vs Inverted Max Hold Time) ({data.latest_date.date()})"
    y_label = f"Weight ({data.weight_unit})"
    x_label = "Inverted Max Hold Time (s)"
    style_axis(ax, title=title, x_label=x_label, y_label=y_label)

    ax.scatter(
        data.max_hold,
        data.weight,
        c="royalblue",
        alpha=alpha,  # pyright: ignore[reportArgumentType]
        s=80,
        edgecolors="black",
        linewidths=0.5,
    )

    if show_curve_fit:
        curve_fit = fit_power_curve_with_hyperbolic_decay(data.weight, data.max_hold)
        hold_times = data.max_hold.sort_values().to_numpy()
        weights = curve_fit.inverted_predict(hold_times)

        ax.plot(
            hold_times,
            weights,
            color="red",
            linewidth=2,
        )

    save_figure(fig, output_path)

    return fig, ax


def plot_session_gap(
    data: FullSessionData | list[FullSessionData],
    output_path: Path | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot the time gap between consecutive sessions over time.

    Args:
        data: A single FullSessionData or a list of FullSessionData objects.
        output_path: Optional path to save the generated figure.

    Returns:
        A tuple containing the matplotlib Figure and Axes objects.
    """

    if not (isinstance(data, list)):
        data = [data]

    # line plot of session frequency over time
    dates = [data.latest_date.date() for data in data]
    fig, ax = create_figure()
    title = f"Session Gap Over Time ({max(dates)})"
    x_label = "Date"
    y_label = "Time since last session (days)"
    style_axis(ax, title=title, x_label=x_label, y_label=y_label)

    upper_value = 0
    for session_data in data:
        rolling_gap_na_indexs = session_data.rolling_session_gap_days.dropna().index
        rolling_gap_days = session_data.rolling_session_gap_days.dropna()
        start_dates = session_data.date.loc[rolling_gap_na_indexs]

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
    """
    Plot the number of sessions per week (rolling average) over time.

    Args:
        data: A single FullSessionData or a list of FullSessionData objects.
        output_path: Optional path to save the generated figure.

    Returns:
        A tuple containing the matplotlib Figure and Axes objects.
    """

    if not (isinstance(data, list)):
        data = [data]

    dates = [data.latest_date.date() for data in data]

    fig, ax = create_figure()
    title = f"Sessions Per Week ({max(dates)})"
    y_label = "Number of Sessions"
    x_label = "Date"
    style_axis(ax, title=title, x_label=x_label, y_label=y_label)

    upper_value = 0
    for session_data in data:
        rolling_sessions = session_data.rolling_sessions_per_week.dropna()
        if rolling_sessions.empty:
            continue

        upper_value = max(upper_value, rolling_sessions.max())
        ax.plot(
            rolling_sessions.index.to_numpy(),
            rolling_sessions.to_numpy(),
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
