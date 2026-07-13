import numpy as np
from scipy.stats import linregress

def detect_steady_state(states_array: np.ndarray, window_size: int = 20, slope_threshold: float = 1e-4) -> int:
    """
    Detects the steady state step using Rolling Linear Regression.
    
    Args:
        states_array: np.ndarray of shape (steps, num_lines) or (runs, steps, num_lines).
                      If (runs, steps, num_lines), it will compute the mean across runs first.
        window_size: The number of recent steps to use for linear regression.
        slope_threshold: The maximum absolute slope to be considered "flat".
        
    Returns:
        The step index where the steady state is reached.
        Returns -1 if steady state is not reached within the array.
    """
    if states_array.ndim == 3:
        states_array = np.mean(states_array, axis=0)
        
    steps, num_lines = states_array.shape
    if steps < window_size:
        return -1
        
    x = np.arange(window_size)
    
    for t in range(window_size, steps + 1):
        window_data = states_array[t - window_size : t]
        
        all_flat = True
        for line_idx in range(num_lines):
            y = window_data[:, line_idx]
            slope, _, _, _, _ = linregress(x, y)
            
            if abs(slope) > slope_threshold:
                all_flat = False
                break
                
        if all_flat:
            return t - 1  # The step at which we confirm steady state
            
    return -1
