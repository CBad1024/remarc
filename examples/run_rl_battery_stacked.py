"""
RL Battery Test: Frame Stacking without Delta Bonus
Grid: GPS=[1, 10], Gamma=[0.99, 0.999], Ent=[0.03, 0.1], Batch=[32, 64]
Fixed: n_frames=3, Delta=0.0, LR=3e-4, Epochs=200, Episode_Steps=200
"""
import sys
import os
import itertools
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import datetime

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from remarc.core.hyperparameters import Presets as P
from remarc.agents.tianshou_agent import train_wf_landscapes, get_ppo_policy, load_best_fn, RandomPolicy
from remarc.envs.utils import define_four_state_landscapes, define_three_state_landscapes
from remarc.core.landscapes import Landscape
from remarc.envs.wright_fisher_env import WrightFisherEnv, ThreeGenotypeEnv
from remarc.agents.greedy_agent import GreedyAgent
from tianshou.env import DummyVectorEnv
from tianshou.data import Batch

# ── Evaluation ──────────────────────────────────────────────────────────────
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

# ── Main ────────────────────────────────────────────────────────────────────
def main():
    DATASET = "four_state"  # Change to "three_state" to use 3-state system
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

    # Fixed
    LR            = 3e-4
    MR            = 1e-5
    EPOCHS        = 150
    EPISODE_STEPS = 500
    REWARD_SCALE  = 100.0
    AMP           = 1.0
    EVAL_STEPS    = 1000
    EVAL_RUNS     = 100

    # Build landscape once
    if DATASET == "three_state":
        landscape_data = define_three_state_landscapes(amplification=AMP)
    else:
        landscape_data = define_four_state_landscapes(amplification=AMP)
        
    num_drugs = len(landscape_data)
    v_N = 2
    g_min, g_max = np.min(landscape_data), np.max(landscape_data)
    landscape_list = [Landscape(v_N, sigma=0.0, ls=landscape_data[i], g_min=g_min, g_max=g_max)
                      for i in range(num_drugs)]

    print(f"Total configurations: {len(configurations)}")
    
    summary_data = []

    for idx, config in enumerate(configurations):
        gps = config['gps']
        gamma = config['gamma']
        ent = config['ent']
        batch = config['batch']
        n_frames = config['n_frames']
        delta = config['delta']
        
        sig = f"stacked_f{n_frames}_d{delta}_g{gps}_gam{gamma}_e{ent}_b{batch}"
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(configurations)}] Frames={n_frames}, Delta={delta}, GPS={gps}, Gamma={gamma}, Ent={ent}, Batch={batch}")
        print(f"{'='*60}")

        # ── Train ───────────────────────────────────────────────────
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
            dataset=DATASET,
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

        train_wf_landscapes(p, signature=sig)

        # ── Evaluate ────────────────────────────────────────────────
        env_kwargs = dict(
            landscape_list=landscape_list,
            num_drugs=num_drugs,
            gen_per_step=gps,
            seq_length=v_N,
            random_start=True,
            total_generations=gps * EVAL_STEPS + 100,
            reward_scale=REWARD_SCALE,
            stochastic=True,
            delta_multiplier=delta,
            mutation_rate=MR,
            n_frames=n_frames
        )
        if DATASET == "three_state":
            eval_env = ThreeGenotypeEnv(**env_kwargs)
        else:
            eval_env = WrightFisherEnv(**env_kwargs)

        test_envs = DummyVectorEnv([lambda: eval_env])
        policy = get_ppo_policy(p, test_envs).eval()
        policy = load_best_fn(policy, f"best_policy_{sig}.pth")

        rl_m, _     = run_eval(eval_env, policy, "RL",     EVAL_RUNS, EVAL_STEPS)
        gr_m, _     = run_eval(eval_env, None,   "Greedy", EVAL_RUNS, EVAL_STEPS)

        rl_mean_fit = np.mean(rl_m)
        gr_mean_fit = np.mean(gr_m)
        delta_fit = (gr_mean_fit - rl_mean_fit) * 100
        
        summary_data.append({
            "Frames": n_frames,
            "Delta": delta,
            "GPS": gps,
            "Gamma": gamma,
            "Entropy": ent,
            "Batch Size": batch,
            "RL Mean Fit": rl_mean_fit,
            "Greedy Mean Fit": gr_mean_fit,
            "Advantage (%)": delta_fit
        })
        
        print(f"--> Done [{idx+1}/{len(configurations)}]: Advantage = {delta_fit:+.2f}%")
        
        # ── Plot Learning Curve ──────────────────────────────────────────────────
        try:
            metrics_csv = project_root / 'log' / 'metrics' / f"{sig}.csv"
            df_metrics = pd.read_csv(metrics_csv)
            plt.figure(figsize=(10, 6))
            plt.plot(df_metrics['epoch'], df_metrics['mean_reward'], color='purple', lw=2.5, marker='o', markersize=4)
            plt.fill_between(df_metrics['epoch'], 
                             df_metrics['mean_reward'] - df_metrics['std_reward'],
                             df_metrics['mean_reward'] + df_metrics['std_reward'], 
                             color='purple', alpha=0.2)
            plt.title(f"Learning Curve: {sig}", fontsize=14, fontweight='bold')
            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel("Mean Test Reward", fontsize=12)
            plt.grid(True, ls='--', alpha=0.4)
            plt.tight_layout()
            curve_path = str(project_root / 'log' / f'{sig}_learning_curve.png')
            plt.savefig(curve_path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"Saved learning curve to {curve_path}")
        except Exception as e:
            print(f"Could not plot learning curve: {e}")

        # ── Plot RL vs Greedy Eval ────────────────────────────────────────────────
        plt.figure(figsize=(10, 6))
        
        # Normalize to % change from 1.0
        def norm(arr): return (arr - 1.0) * 100

        # Greedy
        gm = norm(gr_m)
        plt.plot(pd.Series(gm).rolling(window=20, min_periods=1).mean().values, color='#ff7f0e', ls='--', lw=2.5, label='Greedy')

        # RL
        rm = norm(rl_m)
        plt.plot(pd.Series(rm).rolling(window=20, min_periods=1).mean().values, color='#1f77b4', lw=3, label='Learned')

        plt.axhline(0, color='black', lw=0.8, alpha=0.6)
        plt.grid(True, ls='--', alpha=0.4)

        plt.title(f"RL vs Greedy: {sig}\nAdvantage: {delta_fit:+.2f}%", fontsize=14, fontweight='bold')
        plt.xlabel("RL Steps", fontsize=12)
        plt.ylabel("Fitness (% Δ)", fontsize=12)
        plt.legend(fontsize=10, loc='upper right')

        plt.tight_layout()
        plot_path = str(project_root / 'log' / f'{sig}_vs_greedy.png')
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to {plot_path}")

    # ── Summary ────────────────────────────────────────────────────────────────
    df = pd.DataFrame(summary_data)
    df = df.sort_values(by="Advantage (%)", ascending=False)
    
    csv_path = str(project_root / 'log' / f'stacked_battery_results_{datetime.datetime.now().strftime("%Y%m%d_%H%M")}.csv')
    df.to_csv(csv_path, index=False)
    
    print("\n\n" + "="*60)
    print("BATTERY COMPLETE. TOP 5 CONFIGURATIONS:")
    print("="*60)
    print(df.head(5).to_string(index=False))
    print(f"\nFull results saved to {csv_path}")

if __name__ == "__main__":
    main()
