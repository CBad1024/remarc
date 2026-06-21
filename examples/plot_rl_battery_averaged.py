import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from remarc.core.hyperparameters import Presets as P
from remarc.agents.tianshou_agent import get_ppo_policy, load_best_fn, RandomPolicy
from remarc.envs.wright_fisher_env import WrightFisherEnv
from remarc.agents.greedy_agent import GreedyAgent
from remarc.envs.utils import define_trap_landscapes
from remarc.core.landscapes import Landscape
from tianshou.env import DummyVectorEnv
from tianshou.data import Batch

def run_evaluation_averaged(env, policy, agent_type="RL", num_runs=10, episode_steps=1000):
    all_fitnesses = []
    
    for _ in range(num_runs):
        obs, _ = env.reset()
        fitnesses = []
        
        if agent_type == "Greedy":
            agent = GreedyAgent(env.drug_landscapes)
        elif agent_type == "Random":
            agent = RandomPolicy(env.num_drugs)
        elif agent_type.startswith("Drug"):
            drug_idx = int(agent_type.split()[-1])
            agent = None
        else:
            agent = policy

        done = False
        step = 0
        while not done and step < episode_steps:
            fitnesses.append(env.get_fitness())
            if agent_type == "Greedy":
                action = agent.get_action(obs)
            elif agent_type == "Random":
                batch = Batch(obs=np.array([obs]))
                action = agent(batch).act[0]
            elif agent_type.startswith("Drug"):
                action = drug_idx
            else:
                batch = Batch(obs=np.array([obs]), info={})
                with torch.no_grad():
                    res = agent(batch)
                action = res.act[0]
                
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            
        # Pad if terminated early (should not happen if total_generations is set correctly)
        while len(fitnesses) < episode_steps:
            fitnesses.append(fitnesses[-1])
            
        all_fitnesses.append(fitnesses)
        
    return np.mean(all_fitnesses, axis=0), np.std(all_fitnesses, axis=0)

def main():
    mutation_rates = [1e-6, 1e-4]
    gens_per_steps = [10, 50]
    
    num_runs = 10
    episode_steps = 1000
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Define trap dataset
    trap_data = define_trap_landscapes(amplification=5.0)
    landscape_list = [Landscape(2, sigma=0.0, ls=row, g_min=trap_data.min(), g_max=trap_data.max()) for row in trap_data]
    
    for i, mr in enumerate(mutation_rates):
        for j, gps in enumerate(gens_per_steps):
            print(f"\n--- Evaluating Battery: MR={mr}, GPS={gps} ---")
            sig = f"trap_mr_{mr}_gps_{gps}"
            
            p = P(
                state_shape=(4,),
                num_actions=4,
                buffer_size=20000,
                lr=3e-4,
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=0.03,
                batch_size=32,
                epochs=100,
                train_steps_per_epoch=2000,
                reward_scale=100.0,
                gen_per_step=gps,
                dataset="trap",
                landscape_amplification=5.0,
                test_episodes=10,
                episode_steps=50,
                delta_multiplier=1.0,
                stochastic=False,
                random_start=False
            )
            object.__setattr__(p, 'mutation_rate', mr)
            
            eval_env = WrightFisherEnv(
                landscape_list=landscape_list, 
                num_drugs=4,
                gen_per_step=gps,
                seq_length=2,
                random_start=False,
                total_generations=gps * episode_steps + 100, # Pad total generations slightly
                reward_scale=100.0,
                stochastic=False,
                delta_multiplier=1.0,
                mutation_rate=mr
            )
            eval_env.mutation_matrix = eval_env._build_mutation_matrix()
            
            # Load Policy
            test_envs = DummyVectorEnv([lambda: eval_env])
            policy = get_ppo_policy(p, test_envs).eval()
            
            filename = f"best_policy_{sig}.pth"
            try:
                policy = load_best_fn(policy, filename)
            except Exception as e:
                print(f"Failed to load {filename}, skipping... ({e})")
                continue
            
            ax = axes[i, j]
            
            import pandas as pd
            def process_stats(mean, std):
                # Normalize to % change from 1.0
                m = (mean - 1.0) * 100
                s = std * 100
                # Smooth
                m_smooth = pd.Series(m).rolling(window=20, min_periods=1).mean().values
                s_smooth = pd.Series(s).rolling(window=20, min_periods=1).mean().values
                return m_smooth, s_smooth

            # Plot Single Drugs (Grouped together with thin lines, no shading)
            colors_drugs = ['#ffb3b3', '#b3ffb3', '#b3b3ff', '#ffffb3']
            for d in range(4):
                mean, std = run_evaluation_averaged(eval_env, None, agent_type=f"Drug {d}", num_runs=num_runs, episode_steps=episode_steps)
                m, _ = process_stats(mean, std)
                # Only label the first one to keep legend clean
                label = 'Constant Single Drugs' if d == 0 else ""
                ax.plot(m, label=label, color='gray', linestyle='-', alpha=0.3, linewidth=1)
            
            # Plot Random
            mean, std = run_evaluation_averaged(eval_env, None, agent_type="Random", num_runs=num_runs, episode_steps=episode_steps)
            m, s = process_stats(mean, std)
            ax.plot(m, label='Random Policy', color='#d62728', linestyle=':', linewidth=2)
            
            # Plot Greedy
            mean, std = run_evaluation_averaged(eval_env, None, agent_type="Greedy", num_runs=num_runs, episode_steps=episode_steps)
            m, s = process_stats(mean, std)
            ax.plot(m, label='Greedy Policy', color='#ff7f0e', linestyle='--', linewidth=2.5)
            ax.fill_between(range(episode_steps), m-s, m+s, color='#ff7f0e', alpha=0.15)
            
            # Plot RL
            mean, std = run_evaluation_averaged(eval_env, policy, agent_type="RL", num_runs=num_runs, episode_steps=episode_steps)
            m, s = process_stats(mean, std)
            ax.plot(m, label='Learned Policy', color='#1f77b4', linestyle='-', linewidth=3)
            ax.fill_between(range(episode_steps), m-s, m+s, color='#1f77b4', alpha=0.25)
            
            ax.set_title(f"Mutation Rate: {mr} | Gens per Step: {gps}", fontsize=15, fontweight='bold')
            ax.set_xlabel("RL Steps", fontsize=13)
            if j == 0:
                ax.set_ylabel("Fitness (% Change from Neutral)", fontsize=13)
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # Add a horizontal line at 0 for reference
            ax.axhline(0, color='black', linewidth=1, alpha=0.8)
            
            # Only put legend on the top right plot to reduce clutter
            if i == 0 and j == 1:
                ax.legend(loc='lower right', fontsize=11, framealpha=0.9)

    plt.suptitle("Averaged RL vs Greedy Trajectories (Smoothed & Normalized)", fontsize=20, y=1.02)
    plt.tight_layout()
    save_path = str(Path(__file__).resolve().parent / 'rl_battery_results_averaged_clean.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Clean battery plot saved to {save_path}")

if __name__ == "__main__":
    main()
