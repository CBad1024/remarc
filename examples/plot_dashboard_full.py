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
from remarc.agents.tianshou_agent import get_ppo_policy, load_best_fn, RandomPolicy, SingleDrugPolicy
from remarc.envs.wright_fisher_env import WrightFisherEnv
from remarc.agents.greedy_agent import GreedyAgent
from remarc.envs.utils import define_four_state_landscapes
from remarc.core.landscapes import Landscape
from tianshou.env import DummyVectorEnv
from tianshou.data import Batch
from remarc.agents.shepherd_eval import ShepherdMDP

from examples.plotting import (
    plot_simplex_policy_slices,
    plot_policy_fitness_landscape_slices,
    plot_policy_difference_slices,
    plot_policy_magnitude_difference_slices,
    plot_population_density_slices,
    greedy_policy
)

def run_eval(env, policy, agent_type="RL", num_runs=10, episode_steps=1000, drug_idx = 0): # DRUG INDEX NOT USED UNLESS SINGLE DRUG POLICY
    all_fit = []
    all_states = []
    
    for _ in range(num_runs):
        obs, _ = env.reset()
        fitnesses = []
        states = []
        
        if agent_type == "Greedy":
            agent = GreedyAgent(env.drug_landscapes)
        elif agent_type == "Random":
            agent = RandomPolicy(env.num_drugs)
        elif agent_type == "Single Drug":
            agent = SingleDrugPolicy(drug_idx)
        else:
            agent = policy

        done = False
        step = 0
        while not done and step < episode_steps:
            fitnesses.append(env.get_fitness())
            states.append(obs[-env.num_genotypes:])
            
            if agent_type == "Greedy":
                current_state = obs[-env.num_genotypes:]
                action = agent.get_action(current_state)
            elif agent_type == "Random":
                batch = Batch(obs=np.array([obs]))
                action = agent(batch).act[0]
            elif isinstance(agent_type, int): # Single Drug
                action = agent_type
            elif agent_type == "Shepherd":
                action = agent(obs[-env.num_genotypes:])
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
            states.append(states[-1])
            
        all_fit.append(fitnesses)
        all_states.append(states)
        
    return np.mean(all_fit, axis=0), np.std(all_fit, axis=0), all_states

def main():
    LR            = 3e-4
    MR            = 1e-5
    EPOCHS        = 150
    EPISODE_STEPS = 500
    REWARD_SCALE  = 100.0
    AMP           = 1.0
    EVAL_STEPS    = 1000
    EVAL_RUNS     = 500
    n_frames      = 1
    delta         = 1.0
    gps           = 10
    gamma         = 0.99
    ent           = 0.1
    batch         = 64
    
    sig = f"stacked_f{n_frames}_d{delta}_g{gps}_gam{gamma}_e{ent}_b{batch}"

    four_state_data = define_four_state_landscapes(amplification=AMP)
    num_drugs = len(four_state_data)
    v_N = 2
    g_min, g_max = np.min(four_state_data), np.max(four_state_data)
    landscape_list = [Landscape(v_N, sigma=0.0, ls=four_state_data[i], g_min=g_min, g_max=g_max)
                      for i in range(num_drugs)]

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
        random_start=False, 
        total_generations=gps * EVAL_STEPS + 100,
        reward_scale=REWARD_SCALE,
        stochastic=True,
        delta_multiplier=delta,
        mutation_rate=MR,
        n_frames=n_frames
    )

    test_envs = DummyVectorEnv([lambda: eval_env])
    policy = get_ppo_policy(p, test_envs).eval()
    policy = load_best_fn(policy, f"best_policy_{sig}.pth")
    
    shepherd_mdp = ShepherdMDP.from_env(eval_env, L=20, discount=0.99)
    cache_path = project_root / 'log' / 'shepherd_L20.npz'
    
    if cache_path.exists():
        print("Loading cached SHEPHERD MDP (L=20)...")
        shepherd_mdp.load(cache_path)
    else:
        print("Solving Exact SHEPHERD MDP (L=20)...")
        shepherd_mdp.solve()
        shepherd_mdp.save(cache_path)
        print(f"Saved cached SHEPHERD policy to {cache_path}")
    
    def shepherd_fn(state):
        return shepherd_mdp.get_action(state)
        
    print("Evaluating models...")
    rl_m, rl_std, rl_states = run_eval(eval_env, policy, "RL", EVAL_RUNS, EVAL_STEPS)
    gr_m, gr_std, gr_states = run_eval(eval_env, None, "Greedy", EVAL_RUNS, EVAL_STEPS)
    rn_m, rn_std, rn_states = run_eval(eval_env, None, "Random", EVAL_RUNS, EVAL_STEPS)
    sh_m, sh_std, sh_states = run_eval(eval_env, shepherd_fn, "Shepherd", EVAL_RUNS, EVAL_STEPS)
    sd1_m, sd1_std, sd1_states = run_eval(eval_env, None, "Single Drug", EVAL_RUNS, EVAL_STEPS, drug_idx = 0)
    sd2_m, sd2_std, sd2_states = run_eval(eval_env, None, "Single Drug", EVAL_RUNS, EVAL_STEPS, drug_idx = 1)
    sd3_m, sd3_std, sd3_states = run_eval(eval_env, None, "Single Drug", EVAL_RUNS, EVAL_STEPS, drug_idx = 2)
    sd4_m, sd4_std, sd4_states = run_eval(eval_env, None, "Single Drug", EVAL_RUNS, EVAL_STEPS, drug_idx = 3)

    # 1. Fitness Trajectories
    print("Plotting Fitness Trajectories...")
    plt.figure(figsize=(12, 8))
    steps = np.arange(EVAL_STEPS)
    
    def norm(arr): return (arr - g_min) / (g_max - g_min)
    def norm_std(arr): return arr / (g_max - g_min)
    
    # Random
    plt.plot(steps, norm(rn_m), color='gray', ls=':', lw=2, label='Random Mean')
    plt.fill_between(steps, norm(rn_m) - norm_std(rn_std), norm(rn_m) + norm_std(rn_std), color='gray', alpha=0.1)
    
    # Greedy
    plt.plot(steps, norm(gr_m), color='#ff7f0e', ls='--', lw=2.5, label='Greedy Mean')
    plt.fill_between(steps, norm(gr_m) - norm_std(gr_std), norm(gr_m) + norm_std(gr_std), color='#ff7f0e', alpha=0.2)
    
    # SHEPHERD
    plt.plot(steps, norm(sh_m), color='black', ls='-.', lw=2.5, label='SHEPHERD Mean')
    plt.fill_between(steps, norm(sh_m) - norm_std(sh_std), norm(sh_m) + norm_std(sh_std), color='black', alpha=0.2)
    
    # Learned
    plt.plot(steps, norm(rl_m), color='#1f77b4', lw=3, label='Learned Mean')
    plt.fill_between(steps, norm(rl_m) - norm_std(rl_std), norm(rl_m) + norm_std(rl_std), color='#1f77b4', alpha=0.2)
    
    # Single Drugs
    colors = ['#FF0000', '#32CD32', '#8A2BE2', '#FF1493']
    
    plt.plot(steps, norm(sd1_m), color=colors[0], ls='-.', lw=2.5, label=f'Drug 0 Mean', alpha=1.0)
    plt.fill_between(steps, norm(sd1_m) - norm_std(sd1_std), norm(sd1_m) + norm_std(sd1_std), color=colors[0], alpha=0.2)
    plt.plot(steps, norm(sd2_m), color=colors[1], ls='-.', lw=2.5, label=f'Drug 1 Mean', alpha=1.0)
    plt.fill_between(steps, norm(sd2_m) - norm_std(sd2_std), norm(sd2_m) + norm_std(sd2_std), color=colors[1], alpha=0.2)
    plt.plot(steps, norm(sd3_m), color=colors[2], ls='-.', lw=2.5, label=f'Drug 2 Mean', alpha=1.0)
    plt.fill_between(steps, norm(sd3_m) - norm_std(sd3_std), norm(sd3_m) + norm_std(sd3_std), color=colors[2], alpha=0.2)
    plt.plot(steps, norm(sd4_m), color=colors[3], ls='-.', lw=2.5, label=f'Drug 3 Mean', alpha=1.0)
    plt.fill_between(steps, norm(sd4_m) - norm_std(sd4_std), norm(sd4_m) + norm_std(sd4_std), color=colors[3], alpha=0.2)
    
    plt.ylim(0, 1)
    plt.grid(True, ls='--', alpha=0.4)
    plt.title(f"Normalized Fitness Trajectories (500 episodes)\nPolicy: {sig}", fontsize=14, fontweight='bold')
    plt.xlabel("RL Steps", fontsize=12)
    plt.ylabel("Normalized Fitness", fontsize=12)
    plt.legend(fontsize=10, loc='center right', bbox_to_anchor=(1.25, 0.5))
    plt.tight_layout()
    plt.savefig(str(project_root / 'log' / f'{sig}_dashboard_trajectories.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # Define policy functions for plotting
    def rl_policy_fn(state):
        batch = Batch(obs=np.array([state]), info={})
        with torch.no_grad():
            res = policy(batch)
        return res.act[0]
        
    def gr_policy_fn(state):
        return greedy_policy(state, four_state_data)
        
    genotype_labels = ["00", "01", "10", "11"]
    
    # 2. Population distribution simplexes
    print("Plotting Population Distribution Simplexes...")
    fig = plot_population_density_slices(
        state_trajectories=rl_states,
        policy_fn=rl_policy_fn,
        greedy_policy_fn=gr_policy_fn,
        num_drugs=num_drugs
    )
    if fig is not None:
        fig.savefig(str(project_root / 'log' / f'{sig}_dashboard_pop_density.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)

    # 3. Policy simplexes
    print("Plotting Policy Simplexes...")
    fig = plot_simplex_policy_slices(
        policy_fn=rl_policy_fn,
        num_drugs=num_drugs,
        genotype_labels=genotype_labels,
        title=f"Learned Policy: {sig}"
    )
    if fig is not None:
        fig.savefig(str(project_root / 'log' / f'{sig}_dashboard_policy.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)

    fig = plot_simplex_policy_slices(
        policy_fn=gr_policy_fn,
        num_drugs=num_drugs,
        genotype_labels=genotype_labels,
        title="Greedy Policy"
    )
    if fig is not None:
        fig.savefig(str(project_root / 'log' / f'{sig}_dashboard_greedy_policy.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)

    fig = plot_simplex_policy_slices(
        policy_fn=shepherd_fn,
        num_drugs=num_drugs,
        genotype_labels=genotype_labels,
        title="SHEPHERD Policy (L=20)"
    )
    if fig is not None:
        fig.savefig(str(project_root / 'log' / f'{sig}_dashboard_shepherd_policy.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)

    # 4. Population fitness under policy simplexes
    print("Plotting Policy Fitness Landscape Slices...")
    fig = plot_policy_fitness_landscape_slices(
        policy_fn=rl_policy_fn,
        landscapes=four_state_data,
        num_drugs=num_drugs,
        genotype_labels=genotype_labels,
        title="Normalized Fitness Landscape (Learned Policy)"
    )
    if fig is not None:
        fig.savefig(str(project_root / 'log' / f'{sig}_dashboard_fitness_landscape.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)

    # 5. Disagreement plots
    print("Plotting Policy Disagreement...")
    fig = plot_policy_difference_slices(
        policy_fn_1=rl_policy_fn,
        policy_fn_2=gr_policy_fn,
        num_drugs=num_drugs,
        genotype_labels=genotype_labels,
        title="Policy Disagreement (Learned vs Greedy)"
    )
    if fig is not None:
        fig.savefig(str(project_root / 'log' / f'{sig}_dashboard_disagreement.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)

    # 6. Absolute disagreement plots
    print("Plotting Absolute Disagreement (Greedy)...")
    fig = plot_policy_magnitude_difference_slices(
        policy_fn_1=rl_policy_fn,
        policy_fn_2=gr_policy_fn,
        landscapes=four_state_data,
        num_drugs=num_drugs,
        genotype_labels=genotype_labels,
        title="Absolute Policy Fitness Difference (Learned vs Greedy)"
    )
    if fig is not None:
        fig.savefig(str(project_root / 'log' / f'{sig}_dashboard_magnitude_disagreement.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)

    # 7. Disagreement plots (Learned vs SHEPHERD)
    print("Plotting Policy Disagreement (SHEPHERD)...")
    fig = plot_policy_difference_slices(
        policy_fn_1=rl_policy_fn,
        policy_fn_2=shepherd_fn,
        num_drugs=num_drugs,
        genotype_labels=genotype_labels,
        title="Policy Disagreement (Learned vs SHEPHERD)"
    )
    if fig is not None:
        fig.savefig(str(project_root / 'log' / f'{sig}_dashboard_shepherd_disagreement.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)

    # 8. Absolute disagreement plots (Learned vs SHEPHERD)
    print("Plotting Absolute Disagreement (SHEPHERD)...")
    fig = plot_policy_magnitude_difference_slices(
        policy_fn_1=rl_policy_fn,
        policy_fn_2=shepherd_fn,
        landscapes=four_state_data,
        num_drugs=num_drugs,
        genotype_labels=genotype_labels,
        title="Absolute Policy Fitness Difference (Learned vs SHEPHERD)"
    )
    if fig is not None:
        fig.savefig(str(project_root / 'log' / f'{sig}_dashboard_shepherd_magnitude_disagreement.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)

    print("Done!")

if __name__ == "__main__":
    main()
