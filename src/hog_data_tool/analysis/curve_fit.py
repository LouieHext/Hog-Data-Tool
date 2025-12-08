from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def hyerbolic_decay(w: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Hyperbolic power curve:
        hold(w) = a / (b + w) + c
    """
    return a / (b + w) + c


@dataclass()
class HyperbolicCurveFit:
    a: float
    b: float
    c: float

    def predict(self, w: np.ndarray | float) -> np.ndarray:
        w = np.asarray(w, dtype=float)
        return hyerbolic_decay(w, self.a, self.b, self.c)

    def inverted_predict(self, hold_time: np.ndarray) -> np.ndarray:
        return (self.a / (hold_time - self.c)) - self.b

    def weight_for_hold(self, hold_time: float) -> float:
        """
        Inverts the curve to solve for weight given a target hold time.
        """
        H = hold_time
        if H <= self.c:
            raise ValueError(
                f"Cannot invert: hold_time ≤ c ({self.c}). "
                "Curve approaches hold_time=c as weight→∞."
            )

        return (self.a / (H - self.c)) - self.b


def fit_power_curve_with_hyerbolic_decay(
    weight: pd.Series, hold_times: pd.Series, previous_fit: HyperbolicCurveFit | None = None
) -> HyperbolicCurveFit:

    w = weight.to_numpy(dtype=float)
    y = hold_times.to_numpy(dtype=float)

    # Initial guesses
    if previous_fit is not None:
        a0 = previous_fit.a
        b0 = previous_fit.b
        c0 = previous_fit.c
    else:
        a0 = (y.max() - y.min()) * (w.mean() + 1)
        b0 = w.min() + 1
        c0 = y.min()

    (a, b, c), _ = curve_fit(
        hyerbolic_decay,
        w,
        y,
        p0=[a0, b0, c0],
        maxfev=20_000,
    )

    return HyperbolicCurveFit(
        a=a,
        b=b,
        c=c,
    )
