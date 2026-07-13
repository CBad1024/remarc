import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from remarc.core.hyperparameters import Presets as P
from remarc.agents.tianshou_agent import train_wf_landscapes, get_ppo_policy, load_best_fn, RandomPolicy
from remarc.envs.wright_fisher_env import WrightFisherEnv
from remarc.agents.greedy_agent import GreedyAgent
from remarc.envs.utils import define_trap_landscapes
from remarc.core.landscapes import Landscape
from tianshou.env import DummyVectorEnv
from tianshou.data import Batch

def run_evaluation(env, policy, agent_type="RL"):
    obs, _ = env.reset()
    fitnesses = []
    
    if agent_type == "Greedy":
        agent = GreedyAgent(env.drug_landscapes)
    elif agent_type == "Random":
        agent = RandomPolicy(env.num_drugs)
    else:
        agent = policy

    done = False
    while not done:
        fitnesses.append(env.get_fitness())
        if agent_type == "Greedy":
            action = agent.get_action(obs)
        elif agent_type == "Random":
            # RandomPolicy acts on batches
            batch = Batch(obs=np.array([obs]))
            action = agent(batch).act[0]
        else:
            # RL Policy
            batch = Batch(obs=np.array([obs]), info={})
            with torch.no_grad():
                res = agent(batch)
            action = res.act[0]
            
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
    fitnesses.append(env.get_fitness())
    return np.array(fitnesses)

def main():
    mutation_rates = [1e-6, 1e-4]
    gens_per_steps = [10, 50]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for i, mr in enumerate(mutation_rates):
        for j, gps in enumerate(gens_per_steps):
            print(f"\n--- Running Battery: MR={mr}, GPS={gps} ---")
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
            
            # Train RL
            train_wf_landscapes(p, signature=sig)
            
            # Create Env for Eval
            trap_data = define_trap_landscapes(amplification=5.0)
            landscape_list = [Landscape(2, sigma=0.0, ls=row, g_min=trap_data.min(), g_max=trap_data.max()) for row in trap_data]
            
            eval_env = WrightFisherEnv(
                landscape_list=landscape_list, 
                num_drugs=4,
                gen_per_step=gps,
                seq_length=2,
                random_start=False,
                total_generations=gps * 50,
                reward_scale=100.0,
                stochastic=False,
                delta_multiplier=1.0,
                mutation_rate=mr
            )
            eval_env.mutation_matrix = eval_env._build_mutation_matrix()
            
            # Load Policy
            test_envs = DummyVectorEnv([lambda: eval_env])
            policy = get_ppo_policy(p, test_envs).eval()
            
            # Tianshou save signature looks like: "best_policy_trap_mr_1e-6_gps_10.pth"
            filename = f"best_policy_{sig}.pth"
            policy = load_best_fn(policy, filename)
            
            # Evaluate RL
            rl_fit = run_evaluation(eval_env, policy, agent_type="RL")
            
            # Evaluate Greedy
            greedy_fit = run_evaluation(eval_env, None, agent_type="Greedy")
            
            # Evaluate Random
            rand_fit = run_evaluation(eval_env, None, agent_type="Random")
            
            # Plot
            ax = axes[i, j]
            ax.plot(rl_fit, label='Learned Policy', color='blue', linewidth=2)
            ax.plot(greedy_fit, label='Greedy Policy', color='gold', linestyle='--', linewidth=2)
            ax.plot(rand_fit, label='Random Policy', color='red', linestyle=':', linewidth=2)
            
            ax.set_title(f"Mutation Rate: {mr} | Gens per Step: {gps}", fontsize=14)
            ax.set_xlabel("RL Steps", fontsize=12)
            ax.set_ylabel("Population Fitness", fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()

    plt.suptitle("RL vs Greedy on Trap Dataset (Amplification=5.0)", fontsize=18)
    plt.tight_layout()
    save_path = str(Path(__file__).resolve().parent / 'rl_battery_results.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Battery plot saved to {save_path}")

if __name__ == "__main__":
    main()
