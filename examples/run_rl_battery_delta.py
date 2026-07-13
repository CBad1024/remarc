"""
RL Battery Test: Sweep over Delta Multipliers on the Four-State dataset.
Grid: MR(1e-5, 1e-4) x Delta(0.2, 0.5, 1.0, 2.0)
Fixed: GPS=10, LR=3e-4, Ent=0.1
"""
from remarc.envs import ThreeGenotypeEnv
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
from remarc.envs.wright_fisher_env import WrightFisherEnv
from remarc.agents.greedy_agent import GreedyAgent
from remarc.envs.utils import define_three_state_landscapes
from remarc.core.landscapes import Landscape
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
    mr_vals    = [1e-5]
    delta_vals = [1.5, 2.0, 5.0]

    # Fixed
    GPS           = 10
    LR            = 1e-4
    ENT           = 0.1
    EPOCHS        = 200
    EPISODE_STEPS = 1000
    REWARD_SCALE  = 100.0
    AMP           = 1.0
    EVAL_STEPS    = 1000
    EVAL_RUNS     = 100
    BATCH_SIZE    = 64
    GAMMA         = 0.99
    DATASET       = "Three State"

    # Build landscape once
    if DATASET == "Three State":
        data = define_three_state_landscapes(amplification=AMP)
        num_drugs = len(data)
        v_N = 2
        g_min, g_max = np.min(data), np.max(data)
        landscape_list = [Landscape(v_N, sigma=0.0, ls=data[i], g_min=g_min, g_max=g_max)
                      for i in range(num_drugs)]
    elif DATASET == "Four State":
        data = define_four_state_landscapes(amplification=AMP)
        num_drugs = len(data)
        v_N = 2
        g_min, g_max = np.min(data), np.max(data)
        landscape_list = [Landscape(v_N, sigma=0.0, ls=data[i], g_min=g_min, g_max=g_max)
                      for i in range(num_drugs)]

    combos = list(itertools.product(mr_vals, delta_vals))
    print(f"Total configurations: {len(combos)}")

    results = {}

    for idx, (mr, delta) in enumerate(combos):
        sig = f"{DATASET}_d{delta}_g{GPS}_gam{GAMMA}_e{ENT}_b{BATCH_SIZE}"
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(combos)}] MR={mr} Delta={delta}")
        print(f"{'='*60}")

        # ── Train ───────────────────────────────────────────────────
        state_shape_val = 3 if DATASET == "Three State" else (2**v_N)
        dataset_val = "three_state" if DATASET == "Three State" else "four_state"

        p = P(
            state_shape=(state_shape_val,),
            num_actions=num_drugs,
            buffer_size=20000,
            lr=LR,
            gamma=GAMMA,
            gae_lambda=0.95,
            ent_coef=ENT,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            train_steps_per_epoch=2000,
            reward_scale=REWARD_SCALE,
            gen_per_step=GPS,
            dataset=dataset_val,
            landscape_amplification=AMP,
            test_episodes=10,
            episode_steps=EPISODE_STEPS,
            delta_multiplier=delta,
            stochastic=False,
            random_start=False,
        )
        object.__setattr__(p, 'mutation_rate', mr)

        train_wf_landscapes(p, signature=sig)

        # ── Evaluate ────────────────────────────────────────────────
        if DATASET == "Three State":
            eval_env = ThreeGenotypeEnv(
                landscape_list=landscape_list,
                num_drugs=num_drugs,
                gen_per_step=GPS,
                seq_length=v_N,
                random_start=False,
                total_generations=GPS * EVAL_STEPS + 100,
                reward_scale=REWARD_SCALE,
                stochastic=False,
                delta_multiplier=delta,
                mutation_rate=mr,
            )
        else:
            eval_env = WrightFisherEnv(
                landscape_list=landscape_list,
                num_drugs=num_drugs,
                gen_per_step=GPS,
                seq_length=v_N,
                random_start=False,
                total_generations=GPS * EVAL_STEPS + 100,
                reward_scale=REWARD_SCALE,
                stochastic=False,
                delta_multiplier=delta,
                mutation_rate=mr,
            )

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

        results[(mr, delta)] = {
            "rl": (rl_m, rl_s), "greedy": (gr_m, gr_s),
            "random": (rand_m, rand_s), "drugs": drug_means
        }

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(len(mr_vals), len(delta_vals),
                             figsize=(24, 10), sharex=True)

    for row, mr in enumerate(mr_vals):
        for col, delta in enumerate(delta_vals):
            ax = axes[row, col] if len(mr_vals) > 1 else axes[col]
            r = results[(mr, delta)]

            # Normalize to % change from 1.0
            def norm(arr): return (arr - 1.0) * 100

            # Single drugs (faint background)
            for d in range(num_drugs):
                label = "Single Drugs" if d == 0 else ""
                ax.plot(smooth(norm(r["drugs"][d])), color='gray', alpha=0.25, lw=1, label=label)

            # Random
            ax.plot(smooth(norm(r["random"][0])), color='#d62728', ls=':', lw=1.5, label='Random')

            # Greedy
            gm = norm(r["greedy"][0])
            gs = r["greedy"][1] * 100
            ax.plot(smooth(gm), color='#ff7f0e', ls='--', lw=2.5, label='Greedy')
            ax.fill_between(range(EVAL_STEPS), smooth(gm-gs), smooth(gm+gs),
                            color='#ff7f0e', alpha=0.12)

            # RL
            rm = norm(r["rl"][0])
            rs = r["rl"][1] * 100
            ax.plot(smooth(rm), color='#1f77b4', lw=3, label='Learned')
            ax.fill_between(range(EVAL_STEPS), smooth(rm-rs), smooth(rm+rs),
                            color='#1f77b4', alpha=0.2)

            ax.axhline(0, color='black', lw=0.8, alpha=0.6)
            ax.grid(True, ls='--', alpha=0.4)

            # Titles
            if row == 0:
                ax.set_title(f"Delta={delta}", fontsize=13, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f"MR={mr}\nFitness (% Δ)", fontsize=12)
            if row == len(mr_vals) - 1:
                ax.set_xlabel("RL Steps", fontsize=12)

            # Legend only on first subplot
            if row == 0 and col == 0:
                ax.legend(fontsize=9, loc='upper right', framealpha=0.9)

            # Compute mean RL advantage over greedy
            rl_mean_fit = np.mean(r["rl"][0])
            gr_mean_fit = np.mean(r["greedy"][0])
            delta_fit = (gr_mean_fit - rl_mean_fit) * 100
            marker = "✓" if delta_fit > 0 else "✗"
            ax.text(0.98, 0.02, f"{marker} Δ={delta_fit:+.2f}%",
                    transform=ax.transAxes, fontsize=11, ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.9),
                    fontweight='bold',
                    color='green' if delta_fit > 0 else 'red')

    plt.suptitle("RL vs Greedy — Delta Multiplier Sweep (GPS=10, LR=3e-4, Ent=0.1)",
                 fontsize=20, y=1.03)
    plt.tight_layout()
    save_path = str(Path(__file__).resolve().parent / 'rl_battery_delta.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\nFinal plot saved to {save_path}")

if __name__ == "__main__":
    main()
