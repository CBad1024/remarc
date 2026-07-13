import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from examples.plotting import plot_four_state_simplex, _load_ppo_policy_fn
from remarc.envs.utils import define_four_state_landscapes

def main():
    landscapes = define_four_state_landscapes()
    
    # Configuration list
    configs = [
        {"sig": "stacked_f3_d0.0_g10_gam0.99_e0.1_b64", "n_frames": 3, "delta": 0.0},
    ]
    
    for cfg in configs:
        policy_path = project_root / "log" / "RL" / f"best_policy_{cfg['sig']}.pth"
        if not policy_path.exists():
            print(f"Skipping {cfg['sig']}, not found.")
            continue
            
        n_frames = cfg['n_frames']
        delta = cfg['delta']
        
        state_dim = 4 * n_frames
            
        base_policy_fn = _load_ppo_policy_fn(str(policy_path), state_dim=state_dim, n_actions=4)
        
        def wrapped_fn(state, nf=n_frames, base_fn=base_policy_fn):
            if nf > 1:
                full_state = np.tile(state, nf)
            else:
                full_state = state
            return base_fn(full_state)
            
        save_path = project_root / "log" / f"{cfg['sig']}_simplex.png"
        print(f"Plotting {cfg['sig']} to {save_path} ...")
        plot_four_state_simplex(
            landscapes=landscapes,
            policy_fn=wrapped_fn,
            resolution=40,
            save_path=str(save_path),
            title=f"Policy Simplex: Frames={n_frames}, Delta={delta}"
        )
        plt.close()

if __name__ == "__main__":
    main()
