import matplotlib.pyplot as plt
import numpy as np

def plot_cwt_summary(t, y, trend, resid, power_masked, periods, coi, run_id, save_path=None):
    """
    Generate a 3-panel diagnostic plot for CWT analysis.
    1. Raw fitness + trend
    2. Detrended residual
    3. Scalogram with COI overlay
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Panel 1: Raw fitness + trend
    axs[0].plot(t, y, label="Raw Fitness", color="lightgray")
    axs[0].plot(t, trend, label="Trend (LOESS)", color="red", linewidth=2)
    axs[0].set_title(f"Run {run_id} - Fitness and Trend")
    axs[0].legend()
    axs[0].set_ylabel("Fitness")
    
    # Panel 2: Detrended residual
    axs[1].plot(t, resid, color="blue", linewidth=1)
    axs[1].axhline(0, color="black", linestyle="--", alpha=0.5)
    axs[1].set_title("Detrended Residual")
    axs[1].set_ylabel("Residual")
    
    # Panel 3: Scalogram
    t_mesh, p_mesh = np.meshgrid(t, periods)
    # Use log2 scale for y-axis
    log2_periods = np.log2(p_mesh)
    
    # plot power
    cax = axs[2].contourf(t_mesh, log2_periods, power_masked, levels=100, cmap="viridis")
    
    # Overlay Cone of Influence
    # COI is in periods, so we take log2
    log2_coi = np.log2(coi)
    axs[2].plot(t, log2_coi, color="white", linestyle="--", linewidth=2, label="Cone of Influence")
    axs[2].fill_between(t, log2_coi, log2_periods.max(), color="white", alpha=0.3, hatch="/")
    
    # Format Y-axis to show real periods instead of log2 values
    y_ticks_log2 = np.arange(np.ceil(log2_periods.min()), np.floor(log2_periods.max()) + 1)
    axs[2].set_yticks(y_ticks_log2)
    axs[2].set_yticklabels([f"{2**v:.1f}" for v in y_ticks_log2])
    
    axs[2].set_title("CWT Power Scalogram")
    axs[2].set_ylabel("Period")
    axs[2].set_xlabel("Time Step")
    axs[2].legend(loc="upper right")
    
    fig.colorbar(cax, ax=axs[2], orientation='horizontal', label='Power', pad=0.2)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    
    # Close figure to prevent memory leak
    plt.close(fig)
    return fig

def plot_event_psth(all_events, window, save_path=None):
    """
    Plot the Peri-Stimulus Time Histogram (average trajectory around drug switches).
    """
    from collections import defaultdict
    policy_events = defaultdict(list)
    for ev in all_events:
        policy_events[ev["policy_id"]].append(ev)
        
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    t_axis = np.arange(-window, window + 1)
    
    for policy, events in policy_events.items():
        if not events: continue
        
        # Extract shapes
        resid_stack = np.array([ev["residual_window"] for ev in events])
        sp_stack = np.array([ev["short_power_window"] for ev in events])
        
        # Filter out events that might have incorrect shapes (e.g. at boundaries)
        expected_len = 2 * window + 1
        resid_stack = np.array([r for r in resid_stack if len(r) == expected_len])
        sp_stack = np.array([s for s in sp_stack if len(s) == expected_len])
        
        if len(resid_stack) == 0: continue
        
        resid_mean = np.nanmean(resid_stack, axis=0)
        resid_sem = np.nanstd(resid_stack, axis=0) / np.sqrt(len(resid_stack))
        sp_mean = np.nanmean(sp_stack, axis=0)
        
        p = axs[0].plot(t_axis, resid_mean, label=f"{policy} (n={len(resid_stack)})", linewidth=2)
        color = p[0].get_color()
        axs[0].fill_between(t_axis, resid_mean - resid_sem, resid_mean + resid_sem, color=color, alpha=0.2)
        
        axs[1].plot(t_axis, sp_mean, label=policy, linewidth=2, color=color)
        
    axs[0].axvline(0, color='black', linestyle='--', alpha=0.5)
    axs[0].set_title("Average Detrended Fitness around Drug Switch")
    axs[0].set_ylabel("Fitness Residual")
    axs[0].legend()
    
    axs[1].axvline(0, color='black', linestyle='--', alpha=0.5)
    axs[1].set_title("Average Short-Band Power around Drug Switch")
    axs[1].set_ylabel("Power")
    axs[1].set_xlabel("Time steps relative to switch")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        
    plt.close(fig)
    return fig
