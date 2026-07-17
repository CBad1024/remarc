import numpy as np
import pywt

def make_log_scales(period_min, period_max, num_scales=64):
    """
    Generate logarithmically spaced scales.
    For cmor1.5-1.0 wavelet, central_frequency is 1.0, so scale roughly equals period.
    """
    periods = np.logspace(np.log10(period_min), np.log10(period_max), num_scales)
    scales = periods
    return scales

def compute_cwt(residual, scales, wavelet="cmor1.5-1.0", dt=1.0):
    """
    Compute CWT and return coefficients, power, periods, and COI.
    """
    coefs, freqs = pywt.cwt(residual, scales, wavelet, sampling_period=dt)
    power = np.abs(coefs) ** 2
    periods = 1.0 / freqs
    
    # Compute Cone of Influence (COI)
    # For Morlet, e-folding time is roughly sqrt(2) * s.
    # We will return the max valid period for each time point t.
    N = len(residual)
    t = np.arange(N) * dt
    # distance to edge
    edge_dist = np.minimum(t, t[-1] - t)
    # e-folding time roughly equal to 1.41 * scale. So COI max period is approx 1.41 * edge_dist
    coi = 1.41 * edge_dist
    
    return coefs, power, periods, coi

def build_coi_mask(periods, coi):
    """
    Mask out regions outside the cone of influence.
    Returns boolean array of shape (num_scales, num_timepoints).
    True means valid (inside COI).
    """
    # periods[:, None] -> shape (num_scales, 1)
    # coi[None, :] -> shape (1, num_timepoints)
    mask = periods[:, None] <= coi[None, :]
    return mask

def apply_mask(power, mask):
    """
    Apply mask to power, returning a copy with NaNs outside COI.
    """
    masked = power.copy()
    masked[~mask] = np.nan
    return masked
