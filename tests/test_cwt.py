import numpy as np
from remarc.analysis.cwt.preprocess import detrend_fitness
from remarc.analysis.cwt.wavelet import make_log_scales, compute_cwt
from remarc.analysis.cwt.features import compute_bandpower, extract_peaks

def test_cwt_synthetic_sinusoid():
    # 1. Pure sinusoid with known period
    dt = 1.0
    N = 1000
    t = np.arange(N) * dt
    true_period = 40.0
    y = np.sin(2 * np.pi * t / true_period)
    
    # Process
    trend, resid = detrend_fitness(y, method="loess", frac=0.1)
    # Loess shouldn't remove the 40-step cycle if frac=0.1 (window = 100)
    
    scales = make_log_scales(period_min=2, period_max=200, num_scales=64)
    coefs, power, periods, coi = compute_cwt(resid, scales, dt=dt)
    
    # Extract peaks
    dominant_period, global_peak_period = extract_peaks(power, periods)
    
    # The global peak period should be very close to 40.0
    assert np.isclose(global_peak_period, true_period, rtol=0.1), f"Expected ~40, got {global_peak_period}"
    
    # Mid-signal (away from edges), dominant period should also be ~40
    mid_idx = N // 2
    assert np.isclose(dominant_period[mid_idx], true_period, rtol=0.1)

def test_cwt_trend_removal():
    # 2. Slow trend + sinusoid
    dt = 1.0
    N = 1000
    t = np.arange(N) * dt
    true_period = 30.0
    y_osc = np.sin(2 * np.pi * t / true_period)
    y_trend = 5 * np.sin(2 * np.pi * t / 500.0) # Very slow trend
    y = y_osc + y_trend
    
    trend, resid = detrend_fitness(y, method="loess", frac=0.2)
    
    # Trend should capture the slow 500-period sinusoid
    assert np.allclose(trend, y_trend, atol=0.5)
    
    # Resid should be just the 30-period sinusoid
    assert np.allclose(resid, y_osc, atol=0.5)
