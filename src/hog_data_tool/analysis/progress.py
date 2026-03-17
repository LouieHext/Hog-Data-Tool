from collections import defaultdict

import numpy as np

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
