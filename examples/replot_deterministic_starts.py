import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from remarc.core.hyperparameters import Presets as P
from remarc.agents.tianshou_agent import get_ppo_policy, load_best_fn, RandomPolicy
from remarc.envs.wright_fisher_env import WrightFisherEnv
from remarc.agents.greedy_agent import GreedyAgent
from remarc.envs.utils import define_four_state_landscapes
from remarc.core.landscapes import Landscape
from tianshou.env import DummyVectorEnv
from tianshou.data import Batch

def run_eval(env, policy, agent_type="RL", num_runs=10, episode_steps=1000):
    all_fit = []
    for _ in range(num_runs):
        obs, _ = env.reset()
        fitnesses = []
        if agent_type == "Greedy":
            agent = GreedyAgent(env.drug_landscapes)
        elif agent_type == "Random":
            agent = RandomPolicy(env.num_drugs)
        else:
            agent = policy

        done = False
        step = 0
        while not done and step < episode_steps:
            fitnesses.append(env.get_fitness())
            if agent_type == "Greedy":
                current_state = obs[-env.num_genotypes:]
                action = agent.get_action(current_state)
            elif agent_type == "Random":
                batch = Batch(obs=np.array([obs]))
                action = agent(batch).act[0]
            else:
                batch = Batch(obs=np.array([obs]), info={})
                with torch.no_grad():
                    res = agent(batch)
                action = res.act[0]
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            step += 1
        while len(fitnesses) < episode_steps:
            fitnesses.append(fitnesses[-1])
        all_fit.append(fitnesses)
    return np.mean(all_fit, axis=0), np.std(all_fit, axis=0)

def main():
    base_config = {
        'gps': 10,
        'gamma': 0.99,
        'ent': 0.1,
        'batch': 64,
    }
    
    configurations = [
        {**base_config, 'n_frames': 1, 'delta': 0.0},
        {**base_config, 'n_frames': 1, 'delta': 1.0},
        {**base_config, 'n_frames': 3, 'delta': 0.0},
    ]

    LR            = 3e-4
    MR            = 1e-5
    EPOCHS        = 150
    EPISODE_STEPS = 500
    REWARD_SCALE  = 100.0
    AMP           = 1.0
    EVAL_STEPS    = 1000
    EVAL_RUNS     = 100

    four_state_data = define_four_state_landscapes(amplification=AMP)
    num_drugs = len(four_state_data)
    v_N = 2
    g_min, g_max = np.min(four_state_data), np.max(four_state_data)
    landscape_list = [Landscape(v_N, sigma=0.0, ls=four_state_data[i], g_min=g_min, g_max=g_max)
                      for i in range(num_drugs)]

    for idx, config in enumerate(configurations):
        n_frames = config['n_frames']
        delta    = config['delta']
        gps      = config['gps']
        gamma    = config['gamma']
        ent      = config['ent']
        batch    = config['batch']

        sig = f"stacked_f{n_frames}_d{delta}_g{gps}_gam{gamma}_e{ent}_b{batch}"
        print(f"\n============================================================")
        print(f"Re-evaluating: {sig}")
        print(f"============================================================")

        p = P(
            state_shape=(n_frames * (2**v_N),),
            num_actions=num_drugs,
            buffer_size=20000,
            lr=LR,
            gamma=gamma,
            gae_lambda=0.95,
            ent_coef=ent,
            batch_size=batch,
            epochs=EPOCHS,
            train_steps_per_epoch=10000,
            reward_scale=REWARD_SCALE,
            gen_per_step=gps,
            dataset="four_state",
            landscape_amplification=AMP,
            test_episodes=5,
            episode_steps=EPISODE_STEPS,
            delta_multiplier=delta,
            stochastic=True,
            random_start=True,
            n_frames=n_frames
        )
        object.__setattr__(p, 'mutation_rate', MR)
        object.__setattr__(p, 'n_train_envs', 16)
        object.__setattr__(p, 'n_test_envs', 4)

        eval_env = WrightFisherEnv(
            landscape_list=landscape_list,
            num_drugs=num_drugs,
            gen_per_step=gps,
            seq_length=v_N,
            random_start=False, # DETERMINISTIC START
            total_generations=gps * EVAL_STEPS + 100,
            reward_scale=REWARD_SCALE,
            stochastic=True,
            delta_multiplier=delta,
            mutation_rate=MR,
            n_frames=n_frames
        )

        test_envs = DummyVectorEnv([lambda: eval_env])
        policy = get_ppo_policy(p, test_envs).eval()
        
        try:
            policy = load_best_fn(policy, f"best_policy_{sig}.pth")
        except Exception as e:
            print(f"Failed to load policy for {sig}: {e}")
            continue

        rl_m, rl_std = run_eval(eval_env, policy, "RL",     EVAL_RUNS, EVAL_STEPS)
        gr_m, gr_std = run_eval(eval_env, None,   "Greedy", EVAL_RUNS, EVAL_STEPS)

        rl_mean_fit = np.mean(rl_m)
        gr_mean_fit = np.mean(gr_m)
        delta_fit = (gr_mean_fit - rl_mean_fit) * 100
        
        print(f"RL Mean Fit: {rl_mean_fit:.6f}, Greedy Mean Fit: {gr_mean_fit:.6f}")
        print(f"Advantage: {delta_fit:+.2f}%")
        
        plt.figure(figsize=(10, 6))
        
        steps = np.arange(len(gr_m))

        def norm(arr): return (arr - g_min) / (g_max - g_min)
        def norm_std(arr): return arr / (g_max - g_min)

        gr_m_norm = norm(gr_m)
        gr_std_norm = norm_std(gr_std)
        rl_m_norm = norm(rl_m)
        rl_std_norm = norm_std(rl_std)

        plt.plot(steps, gr_m_norm, color='#ff7f0e', ls='--', lw=2.5, label='Greedy Mean')
        plt.fill_between(steps, gr_m_norm - gr_std_norm, gr_m_norm + gr_std_norm, color='#ff7f0e', alpha=0.2)

        plt.plot(steps, rl_m_norm, color='#1f77b4', lw=3, label='Learned Mean')
        plt.fill_between(steps, rl_m_norm - rl_std_norm, rl_m_norm + rl_std_norm, color='#1f77b4', alpha=0.2)

        plt.grid(True, ls='--', alpha=0.4)
        plt.ylim(0, 1)

        plt.title(f"Normalized Fitness Trajectory over 1000 Steps: {sig}\n(100 episodes, deterministic start)\nAdvantage: {delta_fit:+.2f}%", fontsize=14, fontweight='bold')
        plt.xlabel("RL Steps", fontsize=12)
        plt.ylabel("Normalized Fitness", fontsize=12)
        plt.legend(fontsize=10, loc='upper right')

        plt.tight_layout()
        plot_path = str(project_root / 'log' / f'{sig}_trajectory_det_start.png')
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to {plot_path}")

if __name__ == "__main__":
    main()
