from collections.abc import Callable
from pathlib import Path

import numpy as np
from hog_data_tool.analysis.curve_fit import (
    HyperbolicCurveFit,
    PiecewiseCurveFit,
    fit_piecewise_power_curve,
    fit_power_curve_with_hyperbolic_decay,
)
from hog_data_tool.analysis.progress import (
    HOLD_TIME_BANDS,
    HOLD_TIME_COVERAGE_DISPLAY_Y_HI,
    HOLD_TIME_COVERAGE_DISPLAY_Y_LO,
    coverage_mean_intensity_vs_hold,
    find_sparse_weight,
    pick_recommendation_band_index,
    recent_coverage_scores_at_hold_times,
    rolling_average_weight_in_regimes,
)
from hog_data_tool.hog_data.session_data import FullSessionData
from hog_data_tool.visualisations.utils import (
    create_figure,
    save_figure,
    set_hog_time_axis,
    style_axis,
)
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

type SessionPlotMethod = Callable[[FullSessionData, Path | None], tuple[Figure, Axes]]
type SharedSessionPlotMethod = Callable[
    [list[FullSessionData] | FullSessionData, Path | None], tuple[Figure, Axes]
]


def _session_scatter_face_edge_rgba(
    data: FullSessionData,
    *,
    face_color: str = "tab:blue",
    alpha_scale: float = 0.9,
    linewidths: float = 1.2,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Face and edge RGBA arrays for weight–hold scatter: blue (or ``face_color``) fill and black
    ring, each channel using the same per-point alpha (recency).
    """
    alphas = ((1 - data.normalised_session_age) * alpha_scale).to_numpy(dtype=float)
    n = len(alphas)
    fr, fg, fb, _ = to_rgba(face_color)
    facecolors = np.column_stack([np.full(n, fr), np.full(n, fg), np.full(n, fb), alphas])
    edgecolors = np.column_stack([np.zeros(n), np.zeros(n), np.zeros(n), alphas])
    return facecolors, edgecolors, linewidths


def plot_power_curve(
    data: FullSessionData,
    output_path: Path | None = None,
    show_curve_fit: bool = True,
    max_sessions_for_fit: int = 30,
) -> tuple[Figure, Axes]:
    """
    Plot a scatter of weight vs max hold time for a session, optionally overlaying
    a hyperbolic curve fit.

    Args:
        data: FullSessionData containing weights and max hold times.
        output_path: Optional path to save the generated figure.
        show_curve_fit: Whether to overlay the fitted hyperbolic curve.
        max_sessions_for_fit: Maximum number of recent sessions to use for curve fitting.
            Defaults to 30 to focus on recent performance.

    Returns:
        A tuple containing the matplotlib Figure and Axes objects.
    """
    fc, ec, lw = _session_scatter_face_edge_rgba(data)

    fig, ax = create_figure()
    title = f"Power Curve (Weight vs Max Hold Time) ({data.label}) ({data.latest_date.date()})"
    x_label = f"Weight ({data.weight_unit})"
    y_label = "Max Hold Time (s)"
    style_axis(ax, title=title, x_label=x_label, y_label=y_label)
    set_hog_time_axis(ax)

    ax.scatter(
        data.weight,
        data.max_hold,
        s=80,
        facecolors=fc.tolist(),
        edgecolors=ec.tolist(),
        linewidths=lw,
    )

    if show_curve_fit:
        # Filter to recent sessions for curve fitting
        if data.number_of_sessions > max_sessions_for_fit:
            fit_data = data.select_sessions_from_range(
                start_session=data.number_of_sessions - max_sessions_for_fit + 1,
                end_session=data.number_of_sessions,
            )
        else:
            fit_data = data

        curve_fit = fit_power_curve_with_hyperbolic_decay(
            fit_data.weight,
            fit_data.max_hold,
            session_age=fit_data.normalised_session_age,
        )
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


def draw_hold_time_coverage_bands_on_axes(
    ax: Axes,
    data: FullSessionData,
    *,
    coverage_cmap: str = "Blues",
    band_alpha: float = 0.5,
    cmap_vmin: float = 0.22,
    cmap_vmax: float = 0.92,
    coverage_y_samples: int = 601,
    coverage_smooth_sigma: float = 2.5,
) -> np.ndarray:
    """
    Draw a smooth coverage wash over ``HOLD_TIME_COVERAGE_DISPLAY_Y_LO`` … ``_HI`` (0–300 s):
    mean per-band score at each hold time, Gaussian smoothing, thin horizontal slices by intensity.

    Returns per-band coverage scores (same order as ``HOLD_TIME_BANDS``) for recommendation.
    """
    scores = recent_coverage_scores_at_hold_times(data)
    y, intensity = coverage_mean_intensity_vs_hold(
        scores,
        y_min=HOLD_TIME_COVERAGE_DISPLAY_Y_LO,
        y_max=HOLD_TIME_COVERAGE_DISPLAY_Y_HI,
        n_samples=coverage_y_samples,
        smooth_sigma=coverage_smooth_sigma,
    )
    i_max = float(np.max(intensity))
    if i_max <= 0:
        f_norm = np.zeros_like(intensity)
    else:
        f_norm = intensity / i_max
    cmap = colormaps[coverage_cmap]
    for k in range(len(y) - 1):
        t = 0.5 * (float(f_norm[k]) + float(f_norm[k + 1]))
        t_color = cmap_vmin + (cmap_vmax - cmap_vmin) * t
        facecolor = to_rgba(cmap(t_color), alpha=band_alpha)
        ax.axhspan(float(y[k]), float(y[k + 1]), facecolor=facecolor, zorder=0, linewidth=0)
    return scores


def draw_next_session_hold_recommendation_on_axes(
    ax: Axes,
    data: FullSessionData,
    curve_fit: HyperbolicCurveFit | PiecewiseCurveFit,
    scores: np.ndarray,
) -> tuple[float, float | None]:
    """
    Mark the recommended next-session hold (lowest-scoring band among midpoints in
    ``[HOLD_TIME_RECOMMEND_LO, HOLD_TIME_RECOMMEND_HI]``) with a horizontal line and diamond.
    """
    best_idx = pick_recommendation_band_index(scores)
    lo, hi = float(HOLD_TIME_BANDS[best_idx, 0]), float(HOLD_TIME_BANDS[best_idx, 1])
    best_h = (lo + hi) / 2.0
    score_at_best = float(scores[best_idx])
    unit = data.weight_unit.name
    suggested_w: float | None = None
    try:
        suggested_w = float(curve_fit.weight_for_hold(best_h))
    except ValueError:
        pass

    ax.axhline(
        y=best_h,
        color="red",
        linestyle="--",
        linewidth=1.8,
        alpha=0.9,
        zorder=4,
        label=(
            f"Next hold ~{best_h:.0f}s @ ~{suggested_w:.1f} {unit} (coverage {score_at_best:.2f})"
            if suggested_w is not None
            else f"Next hold ~{best_h:.0f}s (coverage {score_at_best:.2f})"
        ),
    )
    if suggested_w is not None:
        ax.scatter(
            [suggested_w],
            [best_h],
            color="red",
            marker="D",
            s=100,
            zorder=6,
            edgecolors="black",
            linewidths=1.0,
        )
    return best_h, suggested_w


def plot_piecewise_power_curve(
    data: FullSessionData,
    output_path: Path | None = None,
    max_sessions_for_fit: int = 30,
    coverage_cmap: str = "Blues",
    coverage_band_alpha: float = 0.5,
) -> tuple[Figure, Axes]:
    """
    Plot a piecewise power curve with linear (power) and hyperbolic (endurance) segments.

    The model uses:
    - Linear fit for high weights (power regime, short hold times)
    - Hyperbolic decay for low weights (endurance regime, long hold times)

    The transition point between regimes is automatically determined by optimization,
    constrained to have hold time >= 60s at the transition.

    Behind the scatter, coverage is a smooth wash from 0–300 s (mean of overlapping band scores,
    then Gaussian smoothing). The suggested hold uses only bands whose midpoint lies in 40–240 s.

    Args:
        data: FullSessionData containing weights and max hold times.
        output_path: Optional path to save the generated figure.
        max_sessions_for_fit: Maximum number of recent sessions to use for curve fitting.
        coverage_cmap: Matplotlib colormap name for coverage bands (e.g. ``Blues``, ``Greens``).
        coverage_band_alpha: Fixed opacity for each band (0–1).

    Returns:
        A tuple containing the matplotlib Figure and Axes objects.
    """
    fc, ec, lw = _session_scatter_face_edge_rgba(data)

    fig, ax = create_figure()
    title = f"Piecewise Power Curve ({data.label}) ({data.latest_date.date()})"
    x_label = f"Weight ({data.weight_unit})"
    y_label = "Max Hold Time (s)"
    style_axis(ax, title=title, x_label=x_label, y_label=y_label)
    set_hog_time_axis(ax)

    coverage_scores = draw_hold_time_coverage_bands_on_axes(
        ax,
        data,
        coverage_cmap=coverage_cmap,
        band_alpha=coverage_band_alpha,
    )

    ax.scatter(
        data.weight,
        data.max_hold,
        s=80,
        facecolors=fc.tolist(),
        edgecolors=ec.tolist(),
        linewidths=lw,
        zorder=3,
    )

    if data.number_of_sessions > max_sessions_for_fit:
        fit_data = data.select_sessions_from_range(
            start_session=data.number_of_sessions - max_sessions_for_fit + 1,
            end_session=data.number_of_sessions,
        )
    else:
        fit_data = data

    active_fit: HyperbolicCurveFit | PiecewiseCurveFit

    if fit_data.number_of_sessions < max_sessions_for_fit:
        active_fit = fit_power_curve_with_hyperbolic_decay(
            fit_data.weight,
            fit_data.max_hold,
            session_age=fit_data.normalised_session_age,
        )
        weights = np.linspace(data.weight.min(), data.weight.max(), 100)
        hold_fit = active_fit.predict(weights)
        ax.plot(weights, hold_fit, color="red", linewidth=2, zorder=2)
    else:
        try:
            active_fit = fit_piecewise_power_curve(
                fit_data.weight,
                fit_data.max_hold,
                session_age=fit_data.normalised_session_age,
            )

            weights = np.linspace(data.weight.min(), data.weight.max(), 200)
            hold_fit = active_fit.predict(weights)
            ax.plot(weights, hold_fit, color="red", linewidth=2, zorder=2)

        except (RuntimeError, ValueError):
            active_fit = fit_power_curve_with_hyperbolic_decay(
                fit_data.weight,
                fit_data.max_hold,
                session_age=fit_data.normalised_session_age,
            )
            weights = np.linspace(data.weight.min(), data.weight.max(), 100)
            hold_fit = active_fit.predict(weights)
            ax.plot(weights, hold_fit, color="red", linewidth=2, zorder=2)

    draw_next_session_hold_recommendation_on_axes(ax, data, active_fit, coverage_scores)
    ax.legend(loc="upper right")

    ax.set_ylim(HOLD_TIME_COVERAGE_DISPLAY_Y_LO, HOLD_TIME_COVERAGE_DISPLAY_Y_HI)

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

    title = f"Rolling Average Weight in Regimes ({data.label})"
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
    data: FullSessionData,
    output_path: Path | None = None,
    show_curve_fit: bool = True,
    max_sessions_for_fit: int = 10,
) -> tuple[Figure, Axes]:
    """
    Plot weight against max hold time in an inverted format, optionally overlaying
    the inverted hyperbolic curve fit.

    Args:
        data: FullSessionData containing weights and max hold times.
        output_path: Optional path to save the generated figure.
        show_curve_fit: Whether to overlay the inverted fitted curve.
        max_sessions_for_fit: Maximum number of recent sessions to use for curve fitting.
            Defaults to 30 to focus on recent performance.

    Returns:
        A tuple containing the matplotlib Figure and Axes objects.
    """

    fc, ec, lw = _session_scatter_face_edge_rgba(data, face_color="royalblue")

    fig, ax = create_figure()
    title = f"Inverted Power Curve (Weight vs Inverted Max Hold Time) ({data.label}) ({data.latest_date.date()})"
    y_label = f"Weight ({data.weight_unit})"
    x_label = "Inverted Max Hold Time (s)"
    style_axis(ax, title=title, x_label=x_label, y_label=y_label)

    ax.scatter(
        data.max_hold,
        data.weight,
        s=80,
        facecolors=fc.tolist(),
        edgecolors=ec.tolist(),
        linewidths=lw,
    )

    if show_curve_fit:
        # Filter to recent sessions for curve fitting
        if data.number_of_sessions > max_sessions_for_fit:
            fit_data = data.select_sessions_from_range(
                start_session=data.number_of_sessions - max_sessions_for_fit + 1,
                end_session=data.number_of_sessions,
            )
        else:
            fit_data = data

        curve_fit = fit_power_curve_with_hyperbolic_decay(
            fit_data.weight,
            fit_data.max_hold,
            session_age=fit_data.normalised_session_age,
        )
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
    date = data[0].latest_date.date()
    title = f"Session Gap Over Time ({data[0].label}) ({date})"
    x_label = "Date"
    fig, ax = create_figure()
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

    date = data[0].latest_date.date()
    title = f"Sessions Per Week ({data[0].label}) ({date})"
    fig, ax = create_figure()
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
