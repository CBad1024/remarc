import pandas as pd
import numpy as np
from pathlib import Path
from remarc.analysis.cwt.io import trajectories_to_dataframe
from remarc.analysis.cwt.preprocess import detrend_fitness
from remarc.analysis.cwt.wavelet import make_log_scales, compute_cwt, build_coi_mask, apply_mask
from remarc.analysis.cwt.features import compute_bandpower, extract_peaks, compute_spectral_entropy
from remarc.analysis.cwt.plots import plot_cwt_summary

def execute_cwt_pipeline(fitness_dict, actions_dict, output_dir, dt=1.0):
    """
    Execute the CWT pipeline on evaluation results.
    
    Args:
        fitness_dict: dict of {policy_name: fitness_matrix}
        actions_dict: dict of {policy_name: action_matrix}
        output_dir: Path object for saving plots and CSVs
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration
    period_min = 2.0
    period_max = 200.0
    num_scales = 64
    short_band = [2.0, 10.0]
    long_band = [50.0, 200.0]
    
    scales = make_log_scales(period_min, period_max, num_scales)
    
    # 1. Combine all data into DataFrame
    dfs = []
    run_offset = 0
    for policy_id in fitness_dict.keys():
        fit = fitness_dict[policy_id]
        act = actions_dict[policy_id]
        
        df = trajectories_to_dataframe(fit, act, policy_id, start_run_id=run_offset)
        dfs.append(df)
        run_offset += len(fit)
        
    full_df = pd.concat(dfs, ignore_index=True)
    
    # 1.5 Process Mean Trajectories
    mean_metrics = []
    for policy_id, fit in fitness_dict.items():
        mean_y = np.mean(fit, axis=0)
        t = np.arange(len(mean_y)) * dt
        
        trend, resid = detrend_fitness(mean_y, method="loess", frac=0.05)
        coefs, power, periods, coi = compute_cwt(resid, scales, dt=dt)
        mask = build_coi_mask(periods, coi)
        power_masked = apply_mask(power, mask)
        
        sp = compute_bandpower(power_masked, periods, short_band)
        lp = compute_bandpower(power_masked, periods, long_band)
        dom_period, global_peak = extract_peaks(power_masked, periods)
        entropy = compute_spectral_entropy(power_masked)
        
        sp_mean = np.nanmean(sp)
        lp_mean = np.nanmean(lp)
        mean_metrics.append({
            "policy_id": policy_id,
            "global_peak_period": global_peak,
            "short_power_mean": sp_mean,
            "long_power_mean": lp_mean,
            "short_long_ratio": sp_mean / (lp_mean + 1e-9),
            "spectral_entropy": entropy
        })
        
        plot_path = output_dir / f"cwt_{policy_id.replace(' ', '_')}_MEAN_TRAJECTORY.png"
        plot_cwt_summary(t, mean_y, trend, resid, power_masked, periods, coi, run_id="MEAN", save_path=plot_path)

    pd.DataFrame(mean_metrics).to_csv(output_dir / "cwt_mean_trajectory_metrics.csv", index=False)

    run_metrics = []
    all_events = []
    
    # 2. Process each run
    # To limit plotting overhead, we only plot the first 3 runs per policy
    plots_done = {p: 0 for p in fitness_dict.keys()}
    
    for run_id, run_df in full_df.groupby("run_id"):
        t = run_df["t"].to_numpy()
        y = run_df["fitness"].to_numpy()
        policy = run_df["policy_id"].iloc[0]
        
        # Preprocess
        trend, resid = detrend_fitness(y, method="loess", frac=0.05)
        
        # Wavelet
        coefs, power, periods, coi = compute_cwt(resid, scales, dt=dt)
        mask = build_coi_mask(periods, coi)
        power_masked = apply_mask(power, mask)
        
        # Features
        sp = compute_bandpower(power_masked, periods, short_band)
        lp = compute_bandpower(power_masked, periods, long_band)
        dom_period, global_peak = extract_peaks(power_masked, periods)
        entropy = compute_spectral_entropy(power_masked)
        
        # Aggregate metrics
        sp_mean = np.nanmean(sp)
        lp_mean = np.nanmean(lp)
        metrics = {
            "run_id": run_id,
            "policy_id": policy,
            "global_peak_period": global_peak,
            "short_power_mean": sp_mean,
            "long_power_mean": lp_mean,
            "short_long_ratio": sp_mean / (lp_mean + 1e-9),
            "spectral_entropy": entropy
        }
        
        # Phase 4: Event-Aligned Analysis
        run_df = run_df.copy()
        run_df["residual"] = resid
        run_df["short_power"] = sp
        run_df["long_power"] = lp
        
        from remarc.analysis.cwt.events import extract_event_windows
        run_events = extract_event_windows(run_df, window=20)
        for ev in run_events:
            ev["run_id"] = run_id
            ev["policy_id"] = policy
            all_events.append(ev)
            
        run_metrics.append(metrics)
        
        # Plot only first 3 per policy
        if plots_done[policy] < 3:
            plot_path = output_dir / f"cwt_{policy.replace(' ', '_')}_run{run_id}.png"
            plot_cwt_summary(t, y, trend, resid, power_masked, periods, coi, run_id, save_path=plot_path)
            plots_done[policy] += 1
            
    # 3. Summarize
    summary_df = pd.DataFrame(run_metrics)
    summary_df.to_csv(output_dir / "cwt_run_summaries.csv", index=False)
    
    policy_agg = summary_df.groupby("policy_id").agg({
        "global_peak_period": ["mean", "std"],
        "short_power_mean": ["mean", "std"],
        "long_power_mean": ["mean", "std"],
        "short_long_ratio": ["mean"],
        "spectral_entropy": ["mean"]
    })
    policy_agg.columns = ['_'.join(col).strip() for col in policy_agg.columns.values]
    policy_agg.to_csv(output_dir / "cwt_policy_aggregates.csv")
    
    # Plot PSTH
    from remarc.analysis.cwt.plots import plot_event_psth
    if all_events:
        plot_event_psth(all_events, window=20, save_path=output_dir / "cwt_drug_switch_psth.png")
    
    events_df = pd.DataFrame([{
        "run_id": ev["run_id"],
        "policy_id": ev["policy_id"],
        "switch_time": ev["switch_time"],
        "pre_switch_resid_mean": ev["pre_switch_resid_mean"],
        "post_switch_resid_mean": ev["post_switch_resid_mean"],
        "pre_switch_sp_mean": ev["pre_switch_sp_mean"],
        "post_switch_sp_mean": ev["post_switch_sp_mean"]
    } for ev in all_events])
    
    events_df.to_csv(output_dir / "cwt_drug_switch_events.csv", index=False)
    
    print(f"CWT Analysis complete. Outputs saved to {output_dir}")
