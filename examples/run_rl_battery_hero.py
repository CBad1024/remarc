"""
RL Battery Test: Hero Run sweeping Gamma on the Four-State dataset.
Grid: Gamma(0.99, 0.995, 0.999)
Fixed: GPS=10, LR=3e-4, Ent=0.1, MR=1e-5, Delta=1.0, Epochs=300, Episode_Steps=500
"""
import sys
import os
import itertools
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

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
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            step += 1
        while len(fitnesses) < episode_steps:
            fitnesses.append(fitnesses[-1])
        all_fit.append(fitnesses)
    return np.mean(all_fit, axis=0), np.std(all_fit, axis=0)

def smooth(arr, window=20):
    return pd.Series(arr).rolling(window=window, min_periods=1).mean().values

# ── Main ────────────────────────────────────────────────────────────────────
def main():
    # Grid
    gamma_vals = [0.99]
    DATASET = "three_state"  # Change to "three_state" to use 3-state system

    # Fixed
    GPS           = 10
    LR            = 1e-4
    ENT           = 0.1
    MR            = 1e-5
    DELTA         = 1.0
    EPOCHS        = 200
    EPISODE_STEPS = 500
    REWARD_SCALE  = 100.0
    AMP           = 1.0
    EVAL_STEPS    = 1000
    EVAL_RUNS     = 10

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

    print(f"Total configurations: {len(gamma_vals)}")
    results = {}

    for idx, gamma in enumerate(gamma_vals):
        sig = f"hero_gamma{gamma}"
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(gamma_vals)}] Gamma={gamma}")
        print(f"{'='*60}")

        # ── Train ───────────────────────────────────────────────────
        p = P(
            state_shape=(2**v_N,),
            num_actions=num_drugs,
            buffer_size=20000,
            lr=LR,
            gamma=gamma,
            gae_lambda=0.95,
            ent_coef=ENT,
            batch_size=32,
            epochs=EPOCHS,
            train_steps_per_epoch=2000,
            reward_scale=REWARD_SCALE,
            gen_per_step=GPS,
            dataset=DATASET,
            landscape_amplification=AMP,
            test_episodes=10,
            episode_steps=EPISODE_STEPS,
            delta_multiplier=DELTA,
            stochastic=False,
            random_start=False,
            delta_horizon=5,
        )
        object.__setattr__(p, 'mutation_rate', MR)

        train_wf_landscapes(p, signature=sig)

        # ── Evaluate ────────────────────────────────────────────────
        env_kwargs = dict(
            landscape_list=landscape_list,
            num_drugs=num_drugs,
            gen_per_step=GPS,
            seq_length=v_N,
            random_start=False,
            total_generations=GPS * EVAL_STEPS + 100,
            reward_scale=REWARD_SCALE,
            stochastic=False,
            delta_multiplier=DELTA,
            mutation_rate=MR,
            delta_horizon=5,
        )
        if DATASET == "three_state":
            eval_env = ThreeGenotypeEnv(**env_kwargs)
        else:
            eval_env = WrightFisherEnv(**env_kwargs)

        test_envs = DummyVectorEnv([lambda: eval_env])
        policy = get_ppo_policy(p, test_envs).eval()
        policy = load_best_fn(policy, f"best_policy_{sig}.pth")

        rl_m, rl_s     = run_eval(eval_env, policy, "RL",     EVAL_RUNS, EVAL_STEPS)
        gr_m, gr_s     = run_eval(eval_env, None,   "Greedy", EVAL_RUNS, EVAL_STEPS)
        rand_m, rand_s = run_eval(eval_env, None,   "Random", EVAL_RUNS, EVAL_STEPS)

        drug_means = []
        for d in range(num_drugs):
            dm, _ = run_eval(eval_env, None, f"Drug {d}", EVAL_RUNS, EVAL_STEPS)
            drug_means.append(dm)

        results[gamma] = {
            "rl": (rl_m, rl_s), "greedy": (gr_m, gr_s),
            "random": (rand_m, rand_s), "drugs": drug_means
        }

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(gamma_vals), figsize=(20, 6), sharey=True)
    if len(gamma_vals) == 1:
        axes = [axes]

    for col, gamma in enumerate(gamma_vals):
        ax = axes[col]
        r = results[gamma]

        # Normalize to % change from 1.0
        def norm(arr): return (arr - 1.0) * 100

        # Single drugs
        for d in range(num_drugs):
            label = "Single Drugs" if d == 0 else ""
            ax.plot(smooth(norm(r["drugs"][d])), color='gray', alpha=0.25, lw=1, label=label)

        # Random
        ax.plot(smooth(norm(r["random"][0])), color='#d62728', ls=':', lw=1.5, label='Random')

        # Greedy
        gm = norm(r["greedy"][0])
        gs = r["greedy"][1] * 100
        ax.plot(smooth(gm), color='#ff7f0e', ls='--', lw=2.5, label='Greedy')
        ax.fill_between(range(EVAL_STEPS), smooth(gm-gs), smooth(gm+gs), color='#ff7f0e', alpha=0.12)

        # RL
        rm = norm(r["rl"][0])
        rs = r["rl"][1] * 100
        ax.plot(smooth(rm), color='#1f77b4', lw=3, label='Learned')
        ax.fill_between(range(EVAL_STEPS), smooth(rm-rs), smooth(rm+rs), color='#1f77b4', alpha=0.2)

        ax.axhline(0, color='black', lw=0.8, alpha=0.6)
        ax.grid(True, ls='--', alpha=0.4)

        ax.set_title(f"Gamma = {gamma}", fontsize=14, fontweight='bold')
        ax.set_xlabel("RL Steps", fontsize=12)
        if col == 0:
            ax.set_ylabel("Fitness (% Δ)", fontsize=12)
            ax.legend(fontsize=10, loc='upper right', framealpha=0.9)

        # Compute mean RL advantage over greedy
        rl_mean_fit = np.mean(r["rl"][0])
        gr_mean_fit = np.mean(r["greedy"][0])
        delta_fit = (gr_mean_fit - rl_mean_fit) * 100
        marker = "✓" if delta_fit > 0 else "✗"
        ax.text(0.98, 0.02, f"{marker} Δ={delta_fit:+.2f}%",
                transform=ax.transAxes, fontsize=12, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.9),
                fontweight='bold', color='green' if delta_fit > 0 else 'red')

    plt.suptitle("RL vs Greedy — Hero Run: Gamma Sweep (300 Epochs, 500 Steps)", fontsize=18, y=1.05)
    plt.tight_layout()
    save_path = str(Path(__file__).resolve().parent / 'rl_battery_hero.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\nFinal plot saved to {save_path}")

if __name__ == "__main__":
    main()
