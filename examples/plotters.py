import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from evodm.envs import WrightFisherEnv
from evodm.envs.helpers import define_chen_landscapes
from evodm.core.landscapes import Landscape

def plot_frequency_heatmaps(frequencies_at_times, time_steps, filename="wf_frequency_heatmap.png"):
    num_panels = len(time_steps)
    fig, axes = plt.subplots(1, num_panels, figsize=(3 * num_panels, 3))
    
    if num_panels == 1:
        axes = [axes]
        
    # Find global max for consistent color scaling
    vmax = max([np.max(f) for f in frequencies_at_times]) if frequencies_at_times else 1.0
    
    for idx, (freqs, t) in enumerate(zip(frequencies_at_times, time_steps)):
        ax = axes[idx]
        # Reshape 8 genotypes into 2x4 grid
        grid = freqs.reshape((2, 4))
        im = ax.imshow(grid, cmap='viridis', origin='lower', vmin=0, vmax=vmax)
        ax.set_title(f"Gen {t}")
        ax.set_xticks(range(4))
        ax.set_yticks(range(2))
        
    # Add colorbar at the end
    fig.subplots_adjust(right=0.9)
    if num_panels > 1:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='Frequency')
    else:
        plt.colorbar(im, ax=axes[0], label='Frequency')
    
    plt.suptitle("Wright-Fisher Genotype Frequency Evolution (Drug 0)", fontsize=14)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Heatmap plot saved to {filename}")
    plt.close()

ls = define_chen_landscapes()
v_N = 3
landscape_list = [Landscape(v_N, sigma=0.0, ls=ls[i]) for i in range(len(ls))]

env = WrightFisherEnv(num_drugs=len(landscape_list), seq_length=v_N, landscape_list=landscape_list)
env.reset()
env.current_drug = 0  # Use drug index 0 for Chen dataset

# Get initial fitness map for simulation
fitness = env.get_fitness_map()

# Simulation loop
time_steps = np.arange(0, 101, 5)
captured_frequencies = []
max_time = max(time_steps)

# Capture initial state (t=0)
initial_freqs = np.array([env.pop.get(geno, 0) for geno in env.genotypes]) / env.pop_size
captured_frequencies.append(initial_freqs)

# Start simulation from t=1
for t in range(1, max_time + 1):
    env.time_step(fitness)
    if t in time_steps:
        # Capture current genotype frequencies
        freqs = np.array([env.pop.get(geno, 0) for geno in env.genotypes]) / env.pop_size
        captured_frequencies.append(freqs)

plot_frequency_heatmaps(captured_frequencies, time_steps)
