import sys
from pathlib import Path
import numpy as np
from itertools import product
from remarc.envs.wright_fisher_env import WrightFisherEnv
from remarc.envs.utils import define_four_state_landscapes
from remarc.core.landscapes import Landscape
from remarc.agents.greedy_agent import GreedyAgent

def run_single(env, agent_type, drug_idx, episode_steps):
    obs, _ = env.reset()
    fitnesses = []
    states = []
    actions = []
    agent = GreedyAgent(env.drug_landscapes)
    done = False
    step = 0
    while not done and step < episode_steps:
        fitnesses.append(env.get_fitness())
        states.append(obs[-env.num_genotypes :])
        current_state = obs[-env.num_genotypes :]
        action = agent.get_action(current_state)
        actions.append(int(action))
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step += 1
    return fitnesses, states, actions

def test():
    from joblib import Parallel, delayed
    import time
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
    
    t0 = time.time()
    res = Parallel(n_jobs=4)(delayed(run_single)(WrightFisherEnv(**env_kwargs), "Greedy", 0, 500) for _ in range(10))
    print(f"Time parallel: {time.time()-t0}")

test()
