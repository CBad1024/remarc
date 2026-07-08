"""
Comparison Plot: 1-Step Delta vs 5-Step Delta vs SHEPHERD
Trains a 1-step delta model, then loads the existing 5-step delta model,
evaluates SHEPHERD, and plots all three on the same fitness trajectory.
"""
import sys
import os
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from remarc.core.hyperparameters import Presets as P
from remarc.agents.tianshou_agent import train_wf_landscapes, get_ppo_policy, load_best_fn
from remarc.envs.utils import define_three_state_landscapes
from remarc.core.landscapes import Landscape
from remarc.envs.wright_fisher_env import ThreeGenotypeEnv
from remarc.agents.greedy_agent import GreedyAgent
from remarc.agents.shepherd_eval import ShepherdMDP
from tianshou.env import DummyVectorEnv
from tianshou.data import Batch

# ── Evaluation ──────────────────────────────────────────────────────────────
def run_eval(env, policy_or_agent, agent_type="RL", num_runs=10, episode_steps=1000):
    all_fit = []
    for _ in range(num_runs):
        obs, _ = env.reset()
        fitnesses = []

        if agent_type == "Greedy":
            agent = GreedyAgent(env.drug_landscapes)
        elif agent_type == "SHEPHERD":
            agent = policy_or_agent  # ShepherdMDP instance
        else:
            agent = policy_or_agent  # RL policy

        done = False
        step = 0
        while not done and step < episode_steps:
            fitnesses.append(env.get_fitness())
            if agent_type == "Greedy":
                action = agent.get_action(obs[-env.num_genotypes:])
            elif agent_type == "SHEPHERD":
                action = agent.get_action(obs[-env.num_genotypes:])
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


def main():
    # ── Shared hyperparameters ───────────────────────────────────────────────
    GPS           = 10
    LR            = 1e-4
    ENT           = 0.1
    MR            = 1e-5
    DELTA         = 1.0
    EPOCHS        = 200
    EPISODE_STEPS = 500
    REWARD_SCALE  = 100.0
    AMP           = 1.0
    GAMMA         = 0.99
    EVAL_STEPS    = 1000
    EVAL_RUNS     = 10
    DATASET       = "three_state"

    # ── Build landscape ──────────────────────────────────────────────────────
    landscape_data = define_three_state_landscapes(amplification=AMP)
    num_drugs = len(landscape_data)
    v_N = 2  # log2(3) ~ seq_length proxy; ThreeGenotypeEnv handles internally

    g_min, g_max = np.min(landscape_data), np.max(landscape_data)
    landscape_list = [Landscape(v_N, sigma=0.0, ls=landscape_data[i], g_min=g_min, g_max=g_max)
                      for i in range(num_drugs)]

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
    )

    # ── Step 1: Train 1-step delta model ────────────────────────────────────
    sig_1step = "hero_delta1"
    print("\n" + "="*60)
    print("Training 1-step delta model...")
    print("="*60)
    p_1step = P(
        state_shape=(3,),
        num_actions=num_drugs,
        buffer_size=20000,
        lr=LR,
        gamma=GAMMA,
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
        delta_horizon=1,  # 1-step delta
    )
    object.__setattr__(p_1step, 'mutation_rate', MR)
    train_wf_landscapes(p_1step, signature=sig_1step)

    # ── Step 2: Load 5-step delta model (already trained) ───────────────────
    sig_5step = "hero_gamma0.99"
    print("\n" + "="*60)
    print("Loading 5-step delta model (pre-trained)...")
    print("="*60)

    # ── Step 3: Build eval env and policies ─────────────────────────────────
    eval_env = ThreeGenotypeEnv(**env_kwargs)
    test_envs = DummyVectorEnv([lambda: ThreeGenotypeEnv(**env_kwargs)])

    # 1-step policy
    p_for_load = P(
        state_shape=(3,),
        num_actions=num_drugs,
        buffer_size=20000,
        lr=LR,
        gamma=GAMMA,
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
        delta_horizon=1,
    )
    object.__setattr__(p_for_load, 'mutation_rate', MR)
    policy_1step = get_ppo_policy(p_for_load, test_envs).eval()
    policy_1step = load_best_fn(policy_1step, f"best_policy_{sig_1step}.pth")

    # 5-step policy (same arch)
    p_5step_load = P(
        state_shape=(3,),
        num_actions=num_drugs,
        buffer_size=20000,
        lr=LR,
        gamma=GAMMA,
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
    object.__setattr__(p_5step_load, 'mutation_rate', MR)
    policy_5step = get_ppo_policy(p_5step_load, test_envs).eval()
    policy_5step = load_best_fn(policy_5step, f"best_policy_{sig_5step}.pth")

    # ── Step 4: Solve SHEPHERD ───────────────────────────────────────────────
    shepherd_cache = Path(project_root) / "log" / "shepherd_L30_three_state.npz"
    print("\nSolving SHEPHERD MDP...")
    tmp_env = ThreeGenotypeEnv(**env_kwargs)
    shepherd = ShepherdMDP.from_env(tmp_env, L=30, discount=GAMMA)
    if shepherd_cache.exists():
        print("Loading cached SHEPHERD policy...")
        shepherd.load(str(shepherd_cache))
    else:
        print("Computing SHEPHERD policy (this will take ~2 minutes)...")
        shepherd.solve()
        shepherd.save(str(shepherd_cache))

    # ── Step 5: Evaluate all three ───────────────────────────────────────────
    print("\nEvaluating all agents...")
    eval_env_1 = ThreeGenotypeEnv(**{**env_kwargs, 'delta_horizon': 1})
    eval_env_5 = ThreeGenotypeEnv(**{**env_kwargs, 'delta_horizon': 5})

    rl1_m, rl1_s = run_eval(eval_env_1, policy_1step, "RL", EVAL_RUNS, EVAL_STEPS)
    rl5_m, rl5_s = run_eval(eval_env_5, policy_5step, "RL", EVAL_RUNS, EVAL_STEPS)
    shep_m, shep_s = run_eval(eval_env_1, shepherd, "SHEPHERD", EVAL_RUNS, EVAL_STEPS)
    gr_m, gr_s = run_eval(eval_env_1, None, "Greedy", EVAL_RUNS, EVAL_STEPS)

    # ── Step 6: Plot ─────────────────────────────────────────────────────────
    def norm(arr): return (arr - 1.0) * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    steps = np.arange(EVAL_STEPS)

    # Greedy baseline
    gm = norm(gr_m)
    ax.plot(smooth(gm), color='#ff9f43', ls='--', lw=2, label='Greedy', zorder=2)
    ax.fill_between(steps, smooth(norm(gr_m - gr_s)), smooth(norm(gr_m + gr_s)), color='#ff9f43', alpha=0.10)

    # SHEPHERD
    sm = norm(shep_m)
    ax.plot(smooth(sm), color='#00d2d3', ls='-.', lw=2.5, label='SHEPHERD (Optimal)', zorder=4)
    ax.fill_between(steps, smooth(norm(shep_m - shep_s)), smooth(norm(shep_m + shep_s)), color='#00d2d3', alpha=0.10)

    # 1-step delta RL
    r1m = norm(rl1_m)
    ax.plot(smooth(r1m), color='#54a0ff', lw=2.5, label='REMARC (Δ=1-step)', zorder=3)
    ax.fill_between(steps, smooth(norm(rl1_m - rl1_s)), smooth(norm(rl1_m + rl1_s)), color='#54a0ff', alpha=0.15)

    # 5-step delta RL
    r5m = norm(rl5_m)
    ax.plot(smooth(r5m), color='#5f27cd', lw=2.5, label='REMARC (Δ=5-step)', zorder=3)
    ax.fill_between(steps, smooth(norm(rl5_m - rl5_s)), smooth(norm(rl5_m + rl5_s)), color='#5f27cd', alpha=0.15)

    ax.axhline(0, color='white', lw=0.7, alpha=0.4, ls=':')
    ax.grid(True, ls='--', alpha=0.2, color='gray')

    ax.set_xlabel("RL Steps", fontsize=13, color='white')
    ax.set_ylabel("Mean Population Fitness (% Δ from baseline)", fontsize=12, color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')

    # Summary stats in corner
    def mean_fit(arr): return np.mean(arr)
    stats_text = (
        f"Mean Fitness (lower = better drug effect)\n"
        f"{'SHEPHERD':<22} {mean_fit(shep_m)*100:.2f}%\n"
        f"{'REMARC Δ=5-step':<22} {mean_fit(rl5_m)*100:.2f}%\n"
        f"{'REMARC Δ=1-step':<22} {mean_fit(rl1_m)*100:.2f}%\n"
        f"{'Greedy':<22} {mean_fit(gr_m)*100:.2f}%"
    )
    ax.text(0.02, 0.04, stats_text, transform=ax.transAxes, fontsize=9,
            va='bottom', ha='left', color='white',
            bbox=dict(boxstyle='round,pad=0.5', fc='#0f3460', ec='#334155', alpha=0.9),
            fontfamily='monospace')

    legend = ax.legend(fontsize=11, loc='upper right', framealpha=0.85,
                       facecolor='#0f3460', edgecolor='#334155', labelcolor='white')

    ax.set_title("1-Step vs 5-Step Delta Reward: Convergence to SHEPHERD\n"
                 "3-State Landscape · γ=0.99 · 200 Epochs",
                 fontsize=14, color='white', pad=15)

    plt.tight_layout()
    save_path = str(Path(__file__).resolve().parent / 'rl_delta_comparison.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"\nComparison plot saved to {save_path}")


if __name__ == "__main__":
    main()
