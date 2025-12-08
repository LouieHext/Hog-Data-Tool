from collections import defaultdict

import numpy as np

from hog_data_tool.analysis.curve_fit import fit_power_curve_with_hyerbolic_decay
from hog_data_tool.hog_data.definitions import HOG_REGIEME_MAPPINGS, RegiemeEnum
from hog_data_tool.hog_data.session_data import FullSessionData


def rolling_average_weight_in_regiemes(
    data: FullSessionData,
    initial_session_count: int = 10,
    max_sessinons_per_window: int = 20,
) -> defaultdict[RegiemeEnum, list[tuple[int, int]]]:

    session_number = initial_session_count
    results = defaultdict(list)
    curve_fit = None
    while session_number < data.number_of_sessions:

        # select data from before session number only
        date_of_session = data.date.iloc[session_number - 1]
        session_data = data.select_sessions_from_range(
            start_session=max(1, session_number - max_sessinons_per_window),
            end_session=session_number,
        )
        session_number += 1
        curve_fit = fit_power_curve_with_hyerbolic_decay(
            session_data.weight, session_data.max_hold, curve_fit
        )

        for regieme in HOG_REGIEME_MAPPINGS.values():
            midpoint = regieme.midpoint_s
            try:
                predicted_weight = curve_fit.weight_for_hold(midpoint)
            except ValueError:
                predicted_weight = np.nan
            results[regieme.regieme].append((predicted_weight, date_of_session))

    return results
