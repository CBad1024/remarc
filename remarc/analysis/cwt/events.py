import numpy as np
import pandas as pd

def extract_event_windows(df, window=20):
    """
    Extract fixed-size windows around drug switch events.
    
    Args:
        df: DataFrame for a SINGLE run containing 't', 'switch_flag', 
            'residual', 'short_power', 'long_power'
        window: number of steps before and after switch to extract
        
    Returns:
        List of dictionaries with aligned features
    """
    # Find all times where switch_flag == 1
    switch_times = df[df["switch_flag"] == 1]["t"].values
    
    # We will need fast lookup by t. Assuming t is contiguous and sorted 0..N-1
    t_vals = df["t"].values
    resid_vals = df["residual"].values
    sp_vals = df["short_power"].values
    lp_vals = df["long_power"].values
    
    events = []
    max_t = len(t_vals) - 1
    
    for t_s in switch_times:
        start_t = t_s - window
        end_t = t_s + window
        
        # skip if too close to edge
        if start_t < 0 or end_t > max_t:
            continue
            
        # extract
        # this assumes t exactly matches row index for this run
        resid_win = resid_vals[start_t:end_t+1]
        sp_win = sp_vals[start_t:end_t+1]
        lp_win = lp_vals[start_t:end_t+1]
        
        events.append({
            "switch_time": t_s,
            "residual_window": resid_win,
            "short_power_window": sp_win,
            "long_power_window": lp_win,
            "pre_switch_resid_mean": np.mean(resid_win[:window]),
            "post_switch_resid_mean": np.mean(resid_win[window+1:]),
            "pre_switch_sp_mean": np.mean(sp_win[:window]),
            "post_switch_sp_mean": np.mean(sp_win[window+1:])
        })
        
    return events
