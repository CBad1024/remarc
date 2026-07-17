import numpy as np
import statsmodels.api as sm

def detrend_fitness(y, method="loess", frac=0.05, window=None):
    """
    Remove slow baseline trend from trajectory.
    """
    if method == "loess":
        # lowess returns a 2D array [x, y], we just want y
        trend = sm.nonparametric.lowess(y, np.arange(len(y)), frac=frac, return_sorted=False)
    elif method == "rolling":
        if window is None:
            window = max(1, len(y) // 20)
        # uniform filter / rolling mean
        trend = np.convolve(y, np.ones(window)/window, mode='same')
        # fix edges
        trend[:window//2] = trend[window//2]
        trend[-window//2:] = trend[-window//2-1]
    else:
        raise ValueError(f"Unknown detrend method: {method}")
    
    residual = y - trend
    return trend, residual
