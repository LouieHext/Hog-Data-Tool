from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def linear_model(w: np.ndarray, m: float, intercept: float) -> np.ndarray:
    """
    Linear model for power regime (high weights, short hold times).

    hold(w) = m * w + intercept

    Args:
        w: Weight(s) at which hold time is measured.
        m: Slope (negative - higher weight = lower hold time).
        intercept: Y-intercept.

    Returns:
        Hold time(s) corresponding to the input weight(s).
    """
    return m * w + intercept


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
    weight: pd.Series,
    hold_times: pd.Series,
    previous_fit: HyperbolicCurveFit | None = None,
    session_age: pd.Series | None = None,
    age_weight_strength: float = 0.3,
) -> HyperbolicCurveFit:
    """
    Fit a hyperbolic decay curve to weight vs hold time data.

    Args:
        weight: Series of weights.
        hold_times: Series of corresponding hold times.
        previous_fit: Optional previous fit parameters to initialize the optimizer.
        session_age: Optional normalized session age (0=newest, 1=oldest) for weighting.
            Newer sessions get more weight in the fit.
        age_weight_strength: Controls how much to downweight older sessions.
            0 = equal weight, higher = more emphasis on recent sessions.

    Returns:
        A HyperbolicCurveFit object with fitted parameters a, b, c.
    """
    w = weight.to_numpy(dtype=float)
    y = hold_times.to_numpy(dtype=float)

    # Compute sigma for weighting (lower sigma = higher weight)
    # sigma = 1 + age_weight_strength * normalized_age
    # So newest (age=0) gets sigma=1, oldest (age=1) gets sigma=1+strength
    if session_age is not None:
        sigma = 1.0 + age_weight_strength * session_age.to_numpy(dtype=float)
    else:
        sigma = None

    # Initial guesses
    if previous_fit is not None:
        a0, b0, c0 = previous_fit.a, previous_fit.b, previous_fit.c
    else:
        a0 = (y.max() - y.min()) * (w.mean() + 1)
        b0 = w.min() + 1
        c0 = max(0, y.min() - 10)  # c should be below minimum hold time

    # Bounds to help optimizer: a > 0, b > 0, c >= 0
    bounds = ([0, 0, 0], [np.inf, np.inf, y.min()])

    try:
        (a, b, c), _ = curve_fit(
            hyperbolic_decay,
            w,
            y,
            p0=[a0, b0, c0],
            sigma=sigma,
            absolute_sigma=True if sigma is not None else False,
            bounds=bounds,
            maxfev=50_000,
        )
    except RuntimeError:
        # Fall back to unweighted fitting if weighted fails
        (a, b, c), _ = curve_fit(
            hyperbolic_decay,
            w,
            y,
            p0=[a0, b0, c0],
            bounds=bounds,
            maxfev=50_000,
        )

    return HyperbolicCurveFit(a=a, b=b, c=c)


@dataclass
class PiecewiseCurveFit:
    """
    Piecewise power curve combining linear (power regime) and hyperbolic (endurance regime).

    For weights >= transition_weight: linear model (power regime)
    For weights < transition_weight: hyperbolic decay (endurance regime)

    The curves are continuous at the transition point.

    Attributes:
        transition_weight: Weight at which model switches from linear to hyperbolic.
        linear_m: Slope of linear portion.
        linear_intercept: Intercept of linear portion.
        hyper_a: Hyperbolic scaling parameter.
        hyper_b: Hyperbolic horizontal offset.
        hyper_c: Hyperbolic asymptotic minimum.
    """

    transition_weight: float
    linear_m: float
    linear_intercept: float
    hyper_a: float
    hyper_b: float
    hyper_c: float

    @property
    def transition_hold_time(self) -> float:
        """Hold time at the transition point."""
        return self.linear_m * self.transition_weight + self.linear_intercept

    def predict(self, w: np.ndarray | float) -> np.ndarray:
        """
        Predict hold times using the piecewise model.

        Args:
            w: Weight(s) as float or array-like.

        Returns:
            Predicted hold times as np.ndarray.
        """
        w = np.asarray(w, dtype=float)
        result = np.zeros_like(w)

        # Linear regime (high weights, power)
        linear_mask = w >= self.transition_weight
        result[linear_mask] = linear_model(
            w[linear_mask], self.linear_m, self.linear_intercept
        )

        # Hyperbolic regime (low weights, endurance)
        hyper_mask = w < self.transition_weight
        result[hyper_mask] = hyperbolic_decay(
            w[hyper_mask], self.hyper_a, self.hyper_b, self.hyper_c
        )

        return result

    def weight_for_hold(self, hold_time: float) -> float:
        """
        Solve for the weight required to achieve a target hold time.

        Args:
            hold_time: Target hold time (scalar).

        Returns:
            Required weight to achieve the hold time.
        """
        # Check if in linear or hyperbolic regime based on hold time
        if hold_time <= self.transition_hold_time:
            # Linear regime (short hold = high weight)
            return (hold_time - self.linear_intercept) / self.linear_m
        else:
            # Hyperbolic regime
            if hold_time <= self.hyper_c:
                raise ValueError(
                    f"Cannot invert: hold_time ≤ c ({self.hyper_c}). "
                    "Curve approaches hold_time=c as weight→∞."
                )
            return (self.hyper_a / (hold_time - self.hyper_c)) - self.hyper_b


def fit_piecewise_power_curve(
    weight: pd.Series,
    hold_times: pd.Series,
    min_points_per_regime: int = 5,
    min_transition_hold_time: float = 60.0,
    session_age: pd.Series | None = None,
    age_weight_strength: float = 0.5,
) -> PiecewiseCurveFit:
    """
    Fit a piecewise curve with linear (power) and hyperbolic (endurance) segments.

    The transition point is found by optimizing for minimum total error while
    enforcing continuity at the transition.

    Args:
        weight: Series of weights.
        hold_times: Series of corresponding hold times.
        min_points_per_regime: Minimum data points required in each regime.
        min_transition_hold_time: Minimum hold time (seconds) at transition point.
            Ensures the linear (power) regime only covers short hold times.
        session_age: Optional normalized session age (0=newest, 1=oldest) for weighting.
            Newer sessions get more weight in the fit.
        age_weight_strength: Controls how much to downweight older sessions.
            0 = equal weight, higher = more emphasis on recent sessions.

    Returns:
        A PiecewiseCurveFit object with fitted parameters.
    """
    w = weight.to_numpy(dtype=float)
    y = hold_times.to_numpy(dtype=float)

    # Sort by weight for easier splitting
    sort_idx = np.argsort(w)
    w_sorted = w[sort_idx]
    y_sorted = y[sort_idx]

    # Compute sigma for weighting if session_age provided
    if session_age is not None:
        age_sorted = session_age.to_numpy(dtype=float)[sort_idx]
        sigma_sorted = 1.0 + age_weight_strength * age_sorted
    else:
        sigma_sorted = None

    def compute_error_for_transition(transition_idx: int) -> float:
        """Compute total MSE for a given transition index."""
        if transition_idx < min_points_per_regime:
            return np.inf
        if len(w_sorted) - transition_idx < min_points_per_regime:
            return np.inf

        # Check minimum transition hold time constraint
        # The transition is between indices transition_idx-1 and transition_idx
        # Linear regime is high weights (indices >= transition_idx) with LOW hold times
        # So we check if the hold times in linear regime are below the minimum
        avg_linear_hold = y_sorted[transition_idx:].mean()
        if avg_linear_hold >= min_transition_hold_time:
            return np.inf  # Linear regime should have short hold times (< 60s)

        # Split data
        w_hyper = w_sorted[:transition_idx]  # lower weights (endurance)
        y_hyper = y_sorted[:transition_idx]
        w_linear = w_sorted[transition_idx:]  # higher weights (power)
        y_linear = y_sorted[transition_idx:]

        # Split sigma if provided
        if sigma_sorted is not None:
            sigma_hyper = sigma_sorted[:transition_idx]
            sigma_linear = sigma_sorted[transition_idx:]
        else:
            sigma_hyper = None
            sigma_linear = None

        try:
            # Fit linear to high weights (slope should be negative)
            linear_params, _ = curve_fit(
                linear_model,
                w_linear,
                y_linear,
                p0=[-0.5, y_linear.max()],
                sigma=sigma_linear,
                absolute_sigma=True if sigma_linear is not None else False,
                bounds=([-np.inf, 0], [0, np.inf]),  # m <= 0, intercept >= 0
                maxfev=10_000,
            )

            # Fit hyperbolic to low weights
            a0 = (y_hyper.max() - y_hyper.min()) * (w_hyper.mean() + 1)
            b0 = w_hyper.min() + 1
            c0 = max(0, y_hyper.min() - 10)
            hyper_bounds = ([0, 0, 0], [np.inf, np.inf, y_hyper.min()])
            hyper_params, _ = curve_fit(
                hyperbolic_decay,
                w_hyper,
                y_hyper,
                p0=[a0, b0, c0],
                sigma=sigma_hyper,
                absolute_sigma=True if sigma_hyper is not None else False,
                bounds=hyper_bounds,
                maxfev=10_000,
            )

            # Compute weighted errors if sigma provided
            linear_pred = linear_model(w_linear, *linear_params)
            hyper_pred = hyperbolic_decay(w_hyper, *hyper_params)

            if sigma_sorted is not None:
                # Weight errors by inverse of sigma (lower sigma = higher weight)
                linear_weights = 1.0 / sigma_linear
                hyper_weights = 1.0 / sigma_hyper
                linear_mse = np.average((y_linear - linear_pred) ** 2, weights=linear_weights)
                hyper_mse = np.average((y_hyper - hyper_pred) ** 2, weights=hyper_weights)
            else:
                linear_mse = np.mean((y_linear - linear_pred) ** 2)
                hyper_mse = np.mean((y_hyper - hyper_pred) ** 2)

            # Penalize discontinuity at transition
            trans_w = (w_sorted[transition_idx - 1] + w_sorted[transition_idx]) / 2
            linear_at_trans = linear_model(trans_w, *linear_params)
            hyper_at_trans = hyperbolic_decay(trans_w, *hyper_params)
            continuity_penalty = (linear_at_trans - hyper_at_trans) ** 2

            return linear_mse + hyper_mse + 0.5 * continuity_penalty

        except (RuntimeError, ValueError):
            return np.inf

    # Find optimal transition point
    best_idx = min_points_per_regime
    best_error = np.inf

    for idx in range(min_points_per_regime, len(w_sorted) - min_points_per_regime + 1):
        error = compute_error_for_transition(idx)
        if error < best_error:
            best_error = error
            best_idx = idx

    # If no valid transition found, raise error
    if best_error == np.inf:
        raise ValueError(
            f"Could not find valid transition point with min_hold_time >= {min_transition_hold_time}s. "
            "Try lowering min_transition_hold_time or ensuring data spans both regimes."
        )

    # Final fit with best transition
    w_hyper = w_sorted[:best_idx]
    y_hyper = y_sorted[:best_idx]
    w_linear = w_sorted[best_idx:]
    y_linear = y_sorted[best_idx:]

    if sigma_sorted is not None:
        sigma_hyper = sigma_sorted[:best_idx]
        sigma_linear = sigma_sorted[best_idx:]
    else:
        sigma_hyper = None
        sigma_linear = None

    # Fit linear
    linear_params, _ = curve_fit(
        linear_model,
        w_linear,
        y_linear,
        p0=[-0.5, y_linear.max()],
        sigma=sigma_linear,
        absolute_sigma=True if sigma_linear is not None else False,
        maxfev=10000,
    )

    # Fit hyperbolic
    a0 = (y_hyper.max() - y_hyper.min()) * (w_hyper.mean() + 1)
    b0 = w_hyper.min() + 1
    c0 = y_hyper.min()
    hyper_params, _ = curve_fit(
        hyperbolic_decay,
        w_hyper,
        y_hyper,
        p0=[a0, b0, c0],
        sigma=sigma_hyper,
        absolute_sigma=True if sigma_hyper is not None else False,
        maxfev=10000,
    )

    # Transition weight is midpoint between regimes
    transition_weight = (w_sorted[best_idx - 1] + w_sorted[best_idx]) / 2

    # Adjust for continuity: shift linear intercept to match hyperbolic at transition
    hyper_at_trans = hyperbolic_decay(transition_weight, *hyper_params)
    linear_at_trans = linear_model(transition_weight, *linear_params)
    adjusted_intercept = linear_params[1] + (hyper_at_trans - linear_at_trans)

    return PiecewiseCurveFit(
        transition_weight=transition_weight,
        linear_m=linear_params[0],
        linear_intercept=adjusted_intercept,
        hyper_a=hyper_params[0],
        hyper_b=hyper_params[1],
        hyper_c=hyper_params[2],
    )
