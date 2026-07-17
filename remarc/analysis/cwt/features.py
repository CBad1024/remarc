import numpy as np

def compute_bandpower(power_masked, periods, band):
    """
    Compute total power within a period band [min_period, max_period].
    """
    band_mask = (periods >= band[0]) & (periods <= band[1])
    # power_masked shape: (num_scales, num_timepoints)
    # return shape: (num_timepoints,)
    return np.nansum(power_masked[band_mask, :], axis=0)

def extract_peaks(power_masked, periods):
    """
    Find the dominant period at each time step.
    """
    dominant_period = np.full(power_masked.shape[1], np.nan)
    
    # We can just check valid columns
    valid_cols = ~np.all(np.isnan(power_masked), axis=0)
    if np.any(valid_cols):
        idx = np.nanargmax(power_masked[:, valid_cols], axis=0)
        dominant_period[valid_cols] = periods[idx]
        
    global_peak_period = np.nan
    mean_power = np.nanmean(power_masked, axis=1)
    if not np.all(np.isnan(mean_power)):
        global_peak_period = periods[np.nanargmax(mean_power)]
        
    return dominant_period, global_peak_period

def compute_spectral_entropy(power_masked):
    """
    Compute spectral entropy for the global mean power.
    """
    mean_power = np.nanmean(power_masked, axis=1)
    if np.all(np.isnan(mean_power)):
        return np.nan
    # normalize
    p_norm = mean_power / np.nansum(mean_power)
    p_norm = p_norm[p_norm > 0]
    return -np.sum(p_norm * np.log(p_norm))
