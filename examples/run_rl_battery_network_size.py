"""
RL Battery Test: Sweep over Neural Network Sizes for the Three-State dataset.
Grid: Archs = [ [32,32]/[32], [64,64]/[64], [128,128]/[128], [256,256,256]/[128] ]
Fixed: MR=1e-5, Delta=1.0, GPS=10, LR=1e-4, Ent=0.1
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

# ── Main ────────────────────────────────────────────────────────────────────
def main():
    # Grid
    archs = [
        {"hidden": [32, 32], "head": [32], "name": "32"},
        {"hidden": [64, 64], "head": [64], "name": "64"},
        {"hidden": [128, 128], "head": [128], "name": "128"},
        {"hidden": [256, 256, 256], "head": [128], "name": "256"},
    ]

    # Fixed
    mr            = 1e-5
    delta         = 1.0
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

    print(f"Total configurations: {len(archs)}")

    for idx, arch in enumerate(archs):
        sig = f"{DATASET}_net{arch['name']}_d{delta}_g{GPS}_gam{GAMMA}_e{ENT}_b{BATCH_SIZE}"
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(archs)}] Architecture: {arch['name']}")
        print(f"{'='*60}")

        # ── Train ───────────────────────────────────────────────────
        state_shape_val = 3 if DATASET == "Three State" else 4
        dataset_val = "three_state" if DATASET == "Three State" else "four_state"

        p = P(
            state_shape=(state_shape_val,),
            num_actions=3, # Three State has 3 drugs
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
        object.__setattr__(p, 'hidden_sizes', arch["hidden"])
        object.__setattr__(p, 'head_sizes', arch["head"])

        train_wf_landscapes(p, signature=sig)

if __name__ == "__main__":
    main()
