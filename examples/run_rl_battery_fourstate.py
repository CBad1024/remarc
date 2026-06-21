"""
RL Battery Test: 16 configurations on the Four-State dataset.
Grid: GPS(1,10) x LR(5e-5,3e-4) x Ent(0.01,0.1) x MR(1e-5,1e-4)
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
from remarc.envs.wright_fisher_env import WrightFisherEnv
from remarc.agents.greedy_agent import GreedyAgent
from remarc.envs.utils import define_four_state_landscapes
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
    gps_vals   = [1, 10]
    lr_vals    = [5e-5, 3e-4]
    ent_vals   = [0.01, 0.1]
    mr_vals    = [1e-5, 1e-4]

    # Fixed
    EPOCHS        = 100
    EPISODE_STEPS = 200
    REWARD_SCALE  = 100.0
    AMP           = 5.0
    EVAL_STEPS    = 1000
    EVAL_RUNS     = 10

    # Build landscape once
    four_state_data = define_four_state_landscapes(amplification=AMP)
    num_drugs = len(four_state_data)
    v_N = 2
    g_min, g_max = np.min(four_state_data), np.max(four_state_data)
    landscape_list = [Landscape(v_N, sigma=0.0, ls=four_state_data[i], g_min=g_min, g_max=g_max)
                      for i in range(num_drugs)]

    # All combos: (gps, lr, ent, mr)
    combos = list(itertools.product(gps_vals, lr_vals, ent_vals, mr_vals))
    print(f"Total configurations: {len(combos)}")

    results = {}

    for idx, (gps, lr, ent, mr) in enumerate(combos):
        sig = f"bat_gps{gps}_lr{lr}_ent{ent}_mr{mr}"
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(combos)}] GPS={gps} LR={lr} Ent={ent} MR={mr}")
        print(f"{'='*60}")

        # ── Train ───────────────────────────────────────────────────
        p = P(
            state_shape=(2**v_N,),
            num_actions=num_drugs,
            buffer_size=20000,
            lr=lr,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=ent,
            batch_size=32,
            epochs=EPOCHS,
            train_steps_per_epoch=2000,
            reward_scale=REWARD_SCALE,
            gen_per_step=gps,
            dataset="four_state",
            landscape_amplification=AMP,
            test_episodes=10,
            episode_steps=EPISODE_STEPS,
            delta_multiplier=0.0,
            stochastic=False,
            random_start=False,
        )
        object.__setattr__(p, 'mutation_rate', mr)

        train_wf_landscapes(p, signature=sig)

        # ── Evaluate ────────────────────────────────────────────────
        eval_env = WrightFisherEnv(
            landscape_list=landscape_list,
            num_drugs=num_drugs,
            gen_per_step=gps,
            seq_length=v_N,
            random_start=False,
            total_generations=gps * EVAL_STEPS + 100,
            reward_scale=REWARD_SCALE,
            stochastic=False,
            delta_multiplier=0.0,
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

        results[(gps, lr, ent, mr)] = {
            "rl": (rl_m, rl_s), "greedy": (gr_m, gr_s),
            "random": (rand_m, rand_s), "drugs": drug_means
        }

    # ── Plot ────────────────────────────────────────────────────────────────
    # Layout: rows = env params (GPS x MR = 4), cols = RL params (LR x Ent = 4)
    env_combos = list(itertools.product(gps_vals, mr_vals))
    rl_combos  = list(itertools.product(lr_vals, ent_vals))

    fig, axes = plt.subplots(len(env_combos), len(rl_combos),
                             figsize=(24, 20), sharex=True)

    for row, (gps, mr) in enumerate(env_combos):
        for col, (lr, ent) in enumerate(rl_combos):
            ax = axes[row, col]
            r = results[(gps, lr, ent, mr)]

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
                ax.set_title(f"LR={lr}  Ent={ent}", fontsize=13, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f"GPS={gps}  MR={mr}\nFitness (% Δ)", fontsize=12)
            if row == len(env_combos) - 1:
                ax.set_xlabel("RL Steps", fontsize=12)

            # Legend only on first subplot
            if row == 0 and col == 0:
                ax.legend(fontsize=9, loc='upper right', framealpha=0.9)

            # Compute mean RL advantage over greedy
            rl_mean_fit = np.mean(r["rl"][0])
            gr_mean_fit = np.mean(r["greedy"][0])
            delta = (gr_mean_fit - rl_mean_fit) * 100
            marker = "✓" if delta > 0 else "✗"
            ax.text(0.98, 0.02, f"{marker} Δ={delta:+.2f}%",
                    transform=ax.transAxes, fontsize=11, ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.9),
                    fontweight='bold',
                    color='green' if delta > 0 else 'red')

    plt.suptitle("RL vs Greedy Battery — Four-State Dataset (Amp=5.0, 10 runs avg)",
                 fontsize=20, y=1.01)
    plt.tight_layout()
    save_path = str(Path(__file__).resolve().parent / 'rl_battery_fourstate.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\nFinal plot saved to {save_path}")

if __name__ == "__main__":
    main()
