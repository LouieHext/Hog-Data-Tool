from collections import defaultdict

import numpy as np
from scipy.ndimage import gaussian_filter1d

from hog_data_tool.analysis.curve_fit import (
    fit_piecewise_power_curve,
    fit_power_curve_with_hyperbolic_decay,
)
from hog_data_tool.hog_data.definitions import HOG_REGIEME_MAPPINGS, RegimeEnum
from hog_data_tool.hog_data.session_data import FullSessionData


def rolling_average_weight_in_regimes(
    data: FullSessionData,
    initial_session_count: int = 30,
    max_sessions_per_window: int = 30,
    use_piecewise: bool = True,
) -> defaultdict[RegimeEnum, list[tuple[int, int]]]:
    """
    Compute a rolling average of predicted weights for different HOG regimes
    across sessions using curve fitting.

    Args:
        data: FullSessionData containing session weights and hold times.
        initial_session_count: Session index to start the rolling calculation.
            Also determines fit method: piecewise if >= 15, hyperbolic otherwise.
        max_sessions_per_window: Maximum number of past sessions to include in the rolling window.
        use_piecewise: Whether to use piecewise fit (True) or hyperbolic (False).

    Returns:
        A defaultdict mapping each RegiemeEnum to a list of tuples:
        (predicted_weight, date_of_session), where predicted_weight is the
        weight corresponding to the regime's midpoint hold time.
    """

    # Decide fitting method based on initial session count
    use_piecewise = use_piecewise and initial_session_count >= 15

    session_number = initial_session_count
    results = defaultdict(list)

    while session_number < data.number_of_sessions:

        # select data from before session number only
        date_of_session = data.date.iloc[session_number - 1]
        session_data = data.select_sessions_from_range(
            start_session=max(1, session_number - max_sessions_per_window),
            end_session=session_number,
        )
        session_number += 1

        try:
            if use_piecewise:
                curve_fit = fit_piecewise_power_curve(
                    session_data.weight,
                    session_data.max_hold,
                    session_age=session_data.normalised_session_age,
                )
            else:
                curve_fit = fit_power_curve_with_hyperbolic_decay(
                    session_data.weight,
                    session_data.max_hold,
                    session_age=session_data.normalised_session_age,
                )
        except (RuntimeError, ValueError):
            # Fall back to hyperbolic on piecewise fitting error
            try:
                curve_fit = fit_power_curve_with_hyperbolic_decay(
                    session_data.weight,
                    session_data.max_hold,
                    session_age=session_data.normalised_session_age,
                )
            except (RuntimeError, ValueError):
                # Skip this session if fitting fails completely
                continue

        # create predicted midpoints for each regime
        for regime in HOG_REGIEME_MAPPINGS.values():
            midpoint = regime.midpoint_s
            try:
                predicted_weight = curve_fit.weight_for_hold(midpoint)
            except ValueError:
                predicted_weight = np.nan
            results[regime.regime].append((predicted_weight, date_of_session))

    return results


def find_sparse_weight(
    data: FullSessionData,
) -> float:
    """
    Find the midpoint of the largest gap between weights.
    """
    sorted_weight = np.sort(data.weight)
    gaps = np.diff(sorted_weight)
    idx = np.argmax(gaps)
    sparse_weight = (sorted_weight[idx] + sorted_weight[idx + 1]) / 2

    return sparse_weight


HOLD_TIME_BAND_T_LO = 0.0
HOLD_TIME_BAND_T_HI = 300.0

# Hold-time axis used for the smooth coverage wash (full plot extent).
HOLD_TIME_COVERAGE_DISPLAY_Y_LO = 0.0
HOLD_TIME_COVERAGE_DISPLAY_Y_HI = 300.0

# Only bands whose midpoint lies in this interval are candidates for next-session recommendation.
HOLD_TIME_RECOMMEND_LO = 40.0
HOLD_TIME_RECOMMEND_HI = 250.0


def sliding_hold_windows(
    step: float,
    window_width: float,
    t_min: float,
    t_max: float,
    *,
    clip_upper: float | None = None,
) -> np.ndarray:
    """
    Build overlapping hold-time windows ``[lo, hi]`` (seconds, inclusive on both ends).

    Starts at ``t_min``, then ``t_min + step``, ... while each start ``s < t_max``.
    Each band is ``[s, min(s + window_width, hi_cap)]`` where ``hi_cap`` is ``clip_upper``
    if given, otherwise ``t_max``.

    Overlap between consecutive bands is ``window_width - step`` when unclamped.
    """
    hi_cap = float(t_max if clip_upper is None else clip_upper)
    t_lo = float(t_min)
    t_end = float(t_max)
    if t_lo >= t_end or step <= 0 or window_width <= 0:
        return np.zeros((0, 2), dtype=float)
    bands: list[list[float]] = []
    s = t_lo
    while s < t_end:
        hi = min(s + window_width, hi_cap)
        bands.append([s, hi])
        s += step
    return np.array(bands, dtype=float) if bands else np.zeros((0, 2), dtype=float)


def build_hold_time_bands(
    t_lo: float = HOLD_TIME_BAND_T_LO,
    t_hi: float = HOLD_TIME_BAND_T_HI,
) -> np.ndarray:
    """
    Stack three sliding regimes: step/window (10/20) for starts below 90 s, (20/40) below 180 s,
    (30/60) up to ``t_hi``. Window tops are always clipped to ``t_hi``.
    """
    if t_lo >= t_hi:
        return np.zeros((0, 2), dtype=float)
    parts: list[np.ndarray] = []
    if t_lo < 90.0:
        parts.append(
            sliding_hold_windows(
                1.0,
                20.0,
                t_lo,
                min(90.0, t_hi),
                clip_upper=t_hi,
            )
        )
    mid_start = max(t_lo, 90.0)
    if mid_start < min(180.0, t_hi):
        parts.append(
            sliding_hold_windows(
                2.5,
                40.0,
                mid_start,
                min(180.0, t_hi),
                clip_upper=t_hi,
            )
        )
    high_start = max(t_lo, 180.0)
    if high_start < t_hi:
        parts.append(
            sliding_hold_windows(
                3.5,
                60.0,
                high_start,
                t_hi,
                clip_upper=t_hi,
            )
        )
    parts = [p for p in parts if len(p) > 0]
    if not parts:
        return np.zeros((0, 2), dtype=float)
    return np.vstack(parts)


HOLD_TIME_BANDS = build_hold_time_bands()


def recent_coverage_scores_at_hold_times(data: FullSessionData) -> np.ndarray:
    """
    One score per row in ``HOLD_TIME_BANDS`` (from :func:`build_hold_time_bands`). Sessions in
    overlapping ranges contribute to each band they fall in. Recency weight is
    (1 - normalised_session_age)**2.
    """
    holds = data.max_hold.to_numpy(dtype=float)
    age = data.normalised_session_age.to_numpy(dtype=float)
    recency_weight = np.clip(1.0 - age, 0.0, 1.0) ** 2
    n = len(HOLD_TIME_BANDS)
    scores = np.zeros(n, dtype=float)
    for i in range(n):
        lo, hi = float(HOLD_TIME_BANDS[i, 0]), float(HOLD_TIME_BANDS[i, 1])
        mask = (holds >= lo) & (holds <= hi)
        scores[i] = float(recency_weight[mask].sum())
    return scores


def recommendation_eligible_mask() -> np.ndarray:
    """True for bands whose midpoint is in ``[HOLD_TIME_RECOMMEND_LO, HOLD_TIME_RECOMMEND_HI]``."""
    bands = HOLD_TIME_BANDS
    if len(bands) == 0:
        return np.zeros(0, dtype=bool)
    mid = (bands[:, 0] + bands[:, 1]) / 2.0
    return (mid >= HOLD_TIME_RECOMMEND_LO) & (mid <= HOLD_TIME_RECOMMEND_HI)


def pick_recommendation_band_index(scores: np.ndarray) -> int:
    """Argmin over recommendation-eligible bands; if none eligible, argmin over all bands."""
    m = recommendation_eligible_mask()
    if np.any(m):
        idx = np.flatnonzero(m)
        return int(idx[np.argmin(scores[idx])])
    return int(np.argmin(scores))


def coverage_mean_intensity_vs_hold(
    band_scores: np.ndarray,
    *,
    y_min: float = HOLD_TIME_BAND_T_LO,
    y_max: float = HOLD_TIME_BAND_T_HI,
    n_samples: int = 601,
    smooth_sigma: float = 2.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Dense hold-time grid where each sample is the **mean** of per-band scores over all bands
    whose ``[lo, hi]`` contain that hold time. Corrects for edge vs middle: middle hold times sit
    in more overlapping windows, but averaging (not summing) keeps scale comparable.

    Applies 1-D Gaussian smoothing along hold time (``sigma`` in sample index units; 0 to skip).

    Returns
    -------
    y : np.ndarray
        Hold times (seconds), shape ``(n_samples,)``.
    intensity : np.ndarray
        Non-negative smoothed mean intensity, same shape.
    """
    y = np.linspace(float(y_min), float(y_max), int(n_samples))
    bands = HOLD_TIME_BANDS
    n_b = len(bands)
    if n_b == 0 or len(band_scores) != n_b:
        return y, np.zeros_like(y)
    f = np.zeros(len(y), dtype=float)
    for j, yy in enumerate(y):
        inside = (bands[:, 0] <= yy) & (bands[:, 1] >= yy)
        if np.any(inside):
            f[j] = float(np.mean(band_scores[inside]))
    if smooth_sigma > 0:
        f = gaussian_filter1d(f, sigma=float(smooth_sigma), mode="nearest")
    return y, np.maximum(f, 0.0)


def recommend_next_session_hold_time(data: FullSessionData) -> tuple[float, np.ndarray]:
    """
    Midpoint of the lowest-scoring band among those eligible for recommendation (midpoint in
    ``[HOLD_TIME_RECOMMEND_LO, HOLD_TIME_RECOMMEND_HI]``), and all band scores.
    """
    scores = recent_coverage_scores_at_hold_times(data)
    best_i = pick_recommendation_band_index(scores)
    lo, hi = float(HOLD_TIME_BANDS[best_i, 0]), float(HOLD_TIME_BANDS[best_i, 1])
    return (lo + hi) / 2.0, scores
