import sys
import copy
from pathlib import Path
import numpy as np
from itertools import product
from remarc.envs.wright_fisher_env import WrightFisherEnv
from remarc.envs.utils import define_four_state_landscapes
from remarc.core.landscapes import Landscape
from remarc.agents.greedy_agent import GreedyAgent
import time

def run_eval_batched(env_prototype, agent_type="Greedy", num_runs=100, episode_steps=1000, drug_idx=0):
    envs = [copy.deepcopy(env_prototype) for _ in range(num_runs)]
    fitnesses = [[] for _ in range(num_runs)]
    states = [[] for _ in range(num_runs)]
    actions = [[] for _ in range(num_runs)]
    
    if agent_type == "Greedy":
        agent = GreedyAgent(env_prototype.drug_landscapes)
    
    obses = [e.reset()[0] for e in envs]
    dones = [False] * num_runs
    
    for step in range(episode_steps):
        for i, e in enumerate(envs):
            if not dones[i]:
                fitnesses[i].append(e.get_fitness())
                states[i].append(obses[i][-e.num_genotypes :])
        
        active_indices = [i for i, d in enumerate(dones) if not d]
        if not active_indices:
            break
            
        active_obses = [obses[i] for i in active_indices]
        
        acts = [agent.get_action(obs[-env_prototype.num_genotypes :]) for obs in active_obses]
            
        for idx, act in zip(active_indices, acts):
            actions[idx].append(int(act))
            next_obs, _, terminated, truncated, _ = envs[idx].step(act)
            obses[idx] = next_obs
            if terminated or truncated:
                dones[idx] = True
                
    for i in range(num_runs):
        while len(fitnesses[i]) < episode_steps:
            fitnesses[i].append(fitnesses[i][-1])
            states[i].append(states[i][-1])
            actions[i].append(actions[i][-1] if len(actions[i]) > 0 else -1)

    return np.mean(fitnesses, axis=0), np.std(fitnesses, axis=0), states, actions

def test():
    landscape_data = define_four_state_landscapes()
    num_drugs = landscape_data.shape[0]
    v_N = int(np.log2(len(landscape_data[0])))
    g_min, g_max = np.min(landscape_data), np.max(landscape_data)
    landscape_list = [
        Landscape(v_N, sigma=0.0, ls=landscape_data[i], g_min=g_min, g_max=g_max)
        for i in range(num_drugs)
    ]
    env_kwargs = dict(
        landscape_list=landscape_list,
        num_drugs=num_drugs,
        gen_per_step=10,
        seq_length=v_N,
        random_start=False,
        total_generations=50100,
        reward_scale=100.0,
        stochastic=True,
        delta_multiplier=0.5,
        mutation_rate=1e-5,
        n_frames=1,
        delta_horizon=1,
    )
    env = WrightFisherEnv(**env_kwargs)
    
    t0 = time.time()
    run_eval_batched(env, "Greedy", 100, 500)
    print(f"Time batched: {time.time()-t0}")

test()
