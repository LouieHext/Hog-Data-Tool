from collections import defaultdict

import numpy as np

from hog_data_tool.analysis.curve_fit import (
    fit_power_curve_with_hyperbolic_decay,
)
from hog_data_tool.hog_data.definitions import HOG_REGIEME_MAPPINGS, RegimeEnum
from hog_data_tool.hog_data.session_data import FullSessionData


def rolling_average_weight_in_regimes(
    data: FullSessionData,
    initial_session_count: int = 10,
    max_sessions_per_window: int = 20,
) -> defaultdict[RegimeEnum, list[tuple[int, int]]]:
    """
    Compute a rolling average of predicted weights for different HOG regimes
    across sessions using a hyperbolic decay curve fit.

    Args:
        data: FullSessionData containing session weights and hold times.
        initial_session_count: Session index to start the rolling calculation.
        max_sessions_per_window: Maximum number of past sessions to include in the rolling window.

    Returns:
        A defaultdict mapping each RegiemeEnum to a list of tuples:
        (predicted_weight, date_of_session), where predicted_weight is the
        weight corresponding to the regime's midpoint hold time.
    """

    session_number = initial_session_count
    results = defaultdict(list)
    curve_fit = None
    while session_number < data.number_of_sessions:

        # select data from before session number only
        date_of_session = data.date.iloc[session_number - 1]
        session_data = data.select_sessions_from_range(
            start_session=max(1, session_number - max_sessions_per_window),
            end_session=session_number,
        )
        session_number += 1

        # Fit hyperbolic decay curve to the rolling session data
        curve_fit = fit_power_curve_with_hyperbolic_decay(
            session_data.weight, session_data.max_hold, curve_fit
        )

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
