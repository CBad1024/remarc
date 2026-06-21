import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Ensure remarc is in path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from remarc.envs.wright_fisher_env import WrightFisherEnv
from remarc.agents.greedy_agent import GreedyAgent
from remarc.core.landscapes import Landscape

def define_trap_landscapes():
    """
    A custom landscape designed to have a collateral sensitivity trap.
    Genotypes: 00 (WT), 01, 10, 11 (Superbug)
    
    Drug 0 (The Trap): 
        Highly effective against WT (0.1). 
        But 01 is resistant (1.5).
        Greedy will pick this at start.
        
    Drug 1: 
        Effective against 01 (0.2).
        But 11 is highly resistant (2.0). 
        If population is at 01, Greedy picks Drug 1, which pushes them to 11.
        
    Drug 2 (The Setup): 
        Mediocre against WT (0.8). 
        But it pushes the population to 10 (fitness 1.2).
        Greedy NEVER picks this at the start because 0.8 > 0.1.
        
    Drug 3 (The Finisher):
        Terrible against WT (1.5).
        But highly effective against 10 (0.1).
        Does NOT push to 11 easily.
        
    Genotype 11 (Superbug): Resistant to all (fitness 1.8 - 2.0).
    """
    raw = np.array([
        # 00,   01,   10,   11
        [0.1,  1.5,  0.8,  1.8], # Drug 0 (Greedy Trap)
        [1.2,  0.2,  1.5,  1.9], # Drug 1 
        [0.8,  0.5,  1.2,  1.8], # Drug 2 (RL Setup)
        [1.5,  1.6,  0.1,  1.8], # Drug 3 (RL Finisher)
    ])
    return raw

def main():
    # Setup Landscapes
    trap_data = define_trap_landscapes()
    num_drugs = len(trap_data)
    v_N = 2

    landscape_list = []
    g_min, g_max = np.min(trap_data), np.max(trap_data)
    for i in range(num_drugs):
        landscape_list.append(Landscape(v_N, sigma=0.0, ls=trap_data[i], g_min=g_min, g_max=g_max))

    # Setup Environment
    env = WrightFisherEnv(
        landscape_list=landscape_list, 
        num_drugs=num_drugs,
        gen_per_step=10,
        seq_length=v_N,
        random_start=False,
        total_generations=1000,
        reward_scale=100.0,
        stochastic=False, 
        delta_multiplier=0.0
    )

    # Setup Agent
    greedy_agent = GreedyAgent(env.drug_landscapes)

    obs, _ = env.reset()
    trajectories = [obs.copy()]
    drugs_applied = []
    fitnesses = []

    done = False
    while not done:
        fitnesses.append(env.get_fitness())
        action = greedy_agent.get_action(obs)
        drugs_applied.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        trajectories.append(obs.copy())
        done = terminated or truncated
        
    fitnesses.append(env.get_fitness())

    trajectories = np.array(trajectories)
    fitnesses = np.array(fitnesses)

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
        
    # Plot Fitness (dashed)
    fit_line, = plt.plot(fitnesses, color='black', linestyle='--', linewidth=2)
    lines.append(fit_line)
    labels.append('Population Fitness')

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

    plt.title('Genotype Trajectories under Greedy Policy (Trap Dataset)', fontsize=14)
    plt.xlabel('RL Steps (10 generations per step)', fontsize=12)
    plt.ylabel('Genotype Frequency / Fitness', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(lines, labels, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    
    save_path = str(Path(__file__).resolve().parent / 'greedy_trap_trajectories.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    main()
