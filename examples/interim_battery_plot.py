"""Quick interim plot of completed battery models vs Greedy."""
import sys, os, glob
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

def run_eval(env, policy, agent_type="RL", num_runs=10, steps=1000):
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
        while not done and step < steps:
            fitnesses.append(env.get_fitness())
            if agent_type == "Greedy":
                action = agent.get_action(obs)
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
        while len(fitnesses) < steps:
            fitnesses.append(fitnesses[-1])
        all_fit.append(fitnesses)
    return np.mean(all_fit, axis=0), np.std(all_fit, axis=0)

def smooth(arr, w=20):
    return pd.Series(arr).rolling(window=w, min_periods=1).mean().values

def main():
    AMP = 5.0
    EVAL_STEPS = 10000
    EVAL_RUNS = 10

    four_state_data = define_four_state_landscapes(amplification=AMP)
    num_drugs = len(four_state_data)
    v_N = 2
    g_min, g_max = np.min(four_state_data), np.max(four_state_data)
    landscape_list = [Landscape(v_N, sigma=0.0, ls=four_state_data[i], g_min=g_min, g_max=g_max)
                      for i in range(num_drugs)]

    # Find all completed models
    model_dir = project_root / "log" / "RL"
    models = sorted(glob.glob(str(model_dir / "best_policy_bat_*.pth")))
    print(f"Found {len(models)} completed models")

    if not models:
        print("No models found yet!")
        return

    n = len(models)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows), squeeze=False)

    for idx, model_path in enumerate(models):
        fname = Path(model_path).stem.replace("best_policy_", "")
        # Parse: bat_gps1_lr5e-05_ent0.01_mr1e-05
        parts = fname.split("_")
        gps = int(parts[1].replace("gps", ""))
        lr = float(parts[2].replace("lr", ""))
        ent = float(parts[3].replace("ent", ""))
        mr = float(parts[4].replace("mr", ""))

        print(f"\nEvaluating [{idx+1}/{n}]: GPS={gps} LR={lr} Ent={ent} MR={mr}")

        eval_env = WrightFisherEnv(
            landscape_list=landscape_list, num_drugs=num_drugs,
            gen_per_step=gps, seq_length=v_N, random_start=False,
            total_generations=gps * EVAL_STEPS + 100,
            reward_scale=100.0, stochastic=False, delta_multiplier=0.0,
            mutation_rate=mr)

        p = P(state_shape=(4,), num_actions=num_drugs, buffer_size=20000,
              lr=lr, gamma=0.99, gae_lambda=0.95, ent_coef=ent, batch_size=32,
              epochs=100, train_steps_per_epoch=2000, reward_scale=100.0,
              gen_per_step=gps, dataset="four_state", landscape_amplification=AMP,
              test_episodes=10, episode_steps=200, delta_multiplier=0.0,
              stochastic=False, random_start=False)

        test_envs = DummyVectorEnv([lambda: eval_env])
        policy = get_ppo_policy(p, test_envs).eval()
        policy = load_best_fn(policy, Path(model_path).name)

        rl_m, rl_s = run_eval(eval_env, policy, "RL", EVAL_RUNS, EVAL_STEPS)
        gr_m, gr_s = run_eval(eval_env, None, "Greedy", EVAL_RUNS, EVAL_STEPS)
        rand_m, _ = run_eval(eval_env, None, "Random", EVAL_RUNS, EVAL_STEPS)

        ax = axes[idx // cols, idx % cols]
        norm = lambda a: (a - 1.0) * 100

        ax.plot(smooth(norm(rand_m)), color='#d62728', ls=':', lw=1.5, label='Random')
        gm = norm(gr_m)
        ax.plot(smooth(gm), color='#ff7f0e', ls='--', lw=2.5, label='Greedy')
        ax.fill_between(range(EVAL_STEPS), smooth(gm - gr_s*100), smooth(gm + gr_s*100),
                        color='#ff7f0e', alpha=0.12)
        rm = norm(rl_m)
        ax.plot(smooth(rm), color='#1f77b4', lw=3, label='Learned')
        ax.fill_between(range(EVAL_STEPS), smooth(rm - rl_s*100), smooth(rm + rl_s*100),
                        color='#1f77b4', alpha=0.2)

        ax.axhline(0, color='black', lw=0.8, alpha=0.6)
        ax.grid(True, ls='--', alpha=0.4)
        ax.set_title(f"GPS={gps} LR={lr} Ent={ent} MR={mr}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Steps")
        ax.set_ylabel("Fitness (% Δ)")

        delta = (np.mean(gr_m) - np.mean(rl_m)) * 100
        marker = "✓" if delta > 0 else "✗"
        color = 'green' if delta > 0 else 'red'
        ax.text(0.98, 0.02, f"{marker} Δ={delta:+.2f}%", transform=ax.transAxes,
                fontsize=11, ha='right', va='bottom', fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.9))

        if idx == 0:
            ax.legend(fontsize=9, loc='upper right', framealpha=0.9)

    # Hide unused subplots
    for idx in range(n, rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)

    plt.suptitle(f"Interim Battery Results ({n} models completed)", fontsize=16, y=1.02)
    plt.tight_layout()
    save_path = str(Path(__file__).resolve().parent / 'interim_battery.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\nInterim plot saved to {save_path}")

if __name__ == "__main__":
    main()
