import pandas as pd
import numpy as np

def trajectories_to_dataframe(fitnesses, actions, policy_id, start_run_id=0):
    """
    Convert a list/array of fitness trajectories and action trajectories into
    the standard CWT pipeline DataFrame schema.
    
    Args:
        fitnesses: shape (num_runs, episode_steps)
        actions: shape (num_runs, episode_steps)
        policy_id: string label for the policy
        start_run_id: int offset for run_id
        
    Returns:
        pd.DataFrame with t, fitness, run_id, policy_id, drug_action, switch_flag
    """
    num_runs = len(fitnesses)
    episode_steps = len(fitnesses[0])
    
    # Flatten arrays
    t = np.tile(np.arange(episode_steps), num_runs)
    run_ids = np.repeat(np.arange(start_run_id, start_run_id + num_runs), episode_steps)
    fit_flat = np.array(fitnesses).flatten()
    act_flat = np.array(actions).flatten()
    
    df = pd.DataFrame({
        "t": t,
        "fitness": fit_flat,
        "run_id": run_ids,
        "policy_id": policy_id,
        "drug_action": act_flat
    })
    
    # Compute switch_flag: 1 if action changed from previous step, 0 otherwise
    # Group by run_id so we don't bleed switches across runs
    df["switch_flag"] = df.groupby("run_id")["drug_action"].transform(lambda x: (x != x.shift(1).fillna(x)).astype(int))
    
    return df
