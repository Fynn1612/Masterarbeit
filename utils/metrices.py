import numpy as np
from scipy import stats

def evaluate_intervals(y_pred: np.ndarray,
                       y_std: np.ndarray,
                       y_true: np.ndarray,
                       coverage: float = 0.95):
    """
    calculate Coverage (PICP) and Mean Prediction Interval Width (MPIW)
    for a custom confidence level, based on the normal assumption.

    Args:
        y_pred : 1D array with mean predictions
        y_std  : 1D array with prediction standard deviations
        y_true : 1D array with true values
        coverage : desired confidence level (default: 0.95)

    Returns:
        dict with {'coverage': PICP, 'MPIW': MPIW, 'lower': y_lower, 'upper': y_upper}
    """
    assert y_pred.shape == y_std.shape == y_true.shape
    if np.any(y_std <= 0):
        raise ValueError("y_std must be strictly positive")

    #  quantile for the standard normal distribution
    alpha = 1 - coverage
    z = stats.norm.ppf(1 - alpha/2)  # z-value for upper bound, e.g. 1.96 for 95%

    # Calculate intervals
    y_lower = y_pred - z * y_std
    y_upper = y_pred + z * y_std

    # Coverage (PICP)
    in_interval = (y_true >= y_lower) & (y_true <= y_upper)
    picp = np.mean(in_interval)

    # MPIW
    mpiw = np.mean(y_upper - y_lower)

    return {
        "coverage": picp,
        "MPIW": mpiw,
        "lower": y_lower,
        "upper": y_upper
    }
