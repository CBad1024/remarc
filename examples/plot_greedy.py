import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Ensure remarc is in path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from remarc.envs.wright_fisher_env import WrightFisherEnv
from remarc.agents.greedy_agent import GreedyAgent
from remarc.envs.utils import define_four_state_landscapes
from remarc.core.landscapes import Landscape

def main():
    # Setup Landscapes
    amplification = 5.0
    four_state_data = define_four_state_landscapes(amplification=amplification)
    num_drugs = len(four_state_data)
    v_N = 2

    landscape_list = []
    g_min, g_max = np.min(four_state_data), np.max(four_state_data)
    for i in range(num_drugs):
        landscape_list.append(Landscape(v_N, sigma=0.0, ls=four_state_data[i], g_min=g_min, g_max=g_max))

    # Setup Environment
    env = WrightFisherEnv(
        landscape_list=landscape_list, 
        num_drugs=num_drugs,
        gen_per_step=10,
        seq_length=v_N,
        random_start=False,
        total_generations=1000,
        reward_scale=100.0,
        stochastic=False, # Use deterministic for cleaner trajectories
        delta_multiplier=0.0
    )

    # Setup Agent
    greedy_agent = GreedyAgent(env.drug_landscapes)

    obs, _ = env.reset()
    trajectories = [obs.copy()]
    drugs_applied = []

    done = False
    while not done:
        action = greedy_agent.get_action(obs)
        drugs_applied.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        trajectories.append(obs.copy())
        done = terminated or truncated

    trajectories = np.array(trajectories)

    # Plotting
    plt.figure(figsize=(14, 7))
    
    # Plot Genotype Frequencies
    lines = []
    labels = []
    for i in range(trajectories.shape[1]):
        genotype_str = env.genotypes[i]
        line, = plt.plot(trajectories[:, i], linewidth=2)
        lines.append(line)
        labels.append(f'Genotype {genotype_str}')

    # Add background color blocks for drugs
    colors = ['#ff9999', '#99ff99', '#9999ff', '#ffff99']
    current_drug = drugs_applied[0]
    start_idx = 0
    drug_patches = {}
    
    for i in range(1, len(drugs_applied)):
        if drugs_applied[i] != current_drug:
            patch = plt.axvspan(start_idx, i, alpha=0.3, color=colors[current_drug])
            if current_drug not in drug_patches:
                drug_patches[current_drug] = patch
            current_drug = drugs_applied[i]
            start_idx = i
    
    patch = plt.axvspan(start_idx, len(drugs_applied), alpha=0.3, color=colors[current_drug])
    if current_drug not in drug_patches:
        drug_patches[current_drug] = patch

    # Combine legends
    for d_idx, patch in sorted(drug_patches.items()):
        lines.append(patch)
        labels.append(f'Drug {d_idx} Applied')

    plt.title('Genotype Trajectories under Greedy Policy (Four-State Dataset, Amp=5.0)', fontsize=14)
    plt.xlabel('RL Steps (10 generations per step)', fontsize=12)
    plt.ylabel('Genotype Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(lines, labels, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    
    save_path = str(Path(__file__).resolve().parent / 'greedy_trajectories.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    main()
