from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def hyperbolic_decay(w: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Hyperbolic power curve function.

    hold(w) = a / (b + w) + c

    Args:
        w: Weight(s) at which hold time is measured.
        a: Curve parameter controlling scaling.
        b: Curve parameter controlling horizontal offset.
        c: Curve parameter controlling asymptotic minimum hold.

    Returns:
        Hold time(s) corresponding to the input weight(s).
    """
    return a / (b + w) + c


@dataclass
class HyperbolicCurveFit:
    """
    Stores parameters of a hyperbolic decay curve and provides prediction utilities.

    Attributes:
        a: Curve scaling parameter.
        b: Curve horizontal offset.
        c: Curve asymptotic minimum hold.
    """

    a: float
    b: float
    c: float

    def predict(self, w: np.ndarray | float) -> np.ndarray:
        """
        Predict hold times for given weights using the curve parameters.

        Args:
            w: Weight(s) as float or array-like.

        Returns:
            Predicted hold times as np.ndarray.
        """
        w = np.asarray(w, dtype=float)
        return hyperbolic_decay(w, self.a, self.b, self.c)

    def inverted_predict(self, hold_time: np.ndarray) -> np.ndarray:
        """
        Invert the hyperbolic curve to predict weight(s) for a given hold time.

        Args:
            hold_time: Hold time(s) as np.ndarray.

        Returns:
            Corresponding weight(s) as np.ndarray.
        """
        return (self.a / (hold_time - self.c)) - self.b

    def weight_for_hold(self, hold_time: float) -> float:
        """
        Solve for the weight required to achieve a target hold time.

        Args:
            hold_time: Target hold time (scalar).

        Returns:
            Required weight to achieve the hold time.

        Raises:
            ValueError: If hold_time ≤ c, inversion is not possible.
        """
        H = hold_time
        if H <= self.c:
            raise ValueError(
                f"Cannot invert: hold_time ≤ c ({self.c}). "
                "Curve approaches hold_time=c as weight→∞."
            )
        return (self.a / (H - self.c)) - self.b


def fit_power_curve_with_hyperbolic_decay(
    weight: pd.Series, hold_times: pd.Series, previous_fit: HyperbolicCurveFit | None = None
) -> HyperbolicCurveFit:
    """
    Fit a hyperbolic decay curve to weight vs hold time data.

    Args:
        weight: Series of weights.
        hold_times: Series of corresponding hold times.
        previous_fit: Optional previous fit parameters to initialize the optimizer.

    Returns:
        A HyperbolicCurveFit object with fitted parameters a, b, c.
    """
    w = weight.to_numpy(dtype=float)
    y = hold_times.to_numpy(dtype=float)

    # Initial guesses
    if previous_fit is not None:
        a0, b0, c0 = previous_fit.a, previous_fit.b, previous_fit.c
    else:
        a0 = (y.max() - y.min()) * (w.mean() + 1)
        b0 = w.min() + 1
        c0 = y.min()

    (a, b, c), _ = curve_fit(
        hyperbolic_decay,
        w,
        y,
        p0=[a0, b0, c0],
        maxfev=20_000,
    )

    return HyperbolicCurveFit(a=a, b=b, c=c)
