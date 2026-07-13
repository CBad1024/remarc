import argparse
import builtins
import datetime as dt
import logging
import sys
import os
from pathlib import Path

# Add parent directory to path to import from local source instead of installed package
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Ensure log directory exists
(project_root / "log" / "archive").mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from tianshou.data import Batch
from tianshou.policy import BasePolicy
import json
import pickle


from remarc.envs import (
    WrightFisherEnv,
    ThreeGenotypeEnv,
    define_eight_state_landscapes,
    define_four_state_landscapes,
    define_three_state_landscapes,
)
from remarc.core.hyperparameters import Presets
from remarc.core.landscapes import Landscape
from remarc.agents.tianshou_agent import (
    load_best_policy,
    load_random_policy,
    train_wf_landscapes,
)
from remarc.agents.shepherd_eval import ShepherdMDP

# Set up logging
timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            project_root / "log" / "archive" / f"wf_run_{timestamp}.log"
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# Alias all prints as logger.info for consistency
def print(*args, **kwargs):
    logger.info(" ".join(str(arg) for arg in args))


builtins.print = print


def evaluate_best_single_drug(
    landscape: np.ndarray = define_eight_state_landscapes(),
    num_episodes: int = 20,
    seq_length: int = 3,
    episode_length: int = 20,
    gen_per_step: int = 500,
    sigma: float = 0.0,
    random_start: bool = True,
):
    """
    Evaluate the best single drug policy through simulating each individually.

    Returns:
        best_drug: int - index of the best-performing drug
        best_fitness: float - fitness of the best-performing drug
        trajectories: list of pd.DataFrame - trajectories of each drug
    """
    landscape_list = [
        Landscape(N=seq_length, sigma=sigma, ls=landscape[i, :])
        for i in range(len(landscape))
    ]
    env = WrightFisherEnv(
        seq_length=seq_length,
        landscape_list=landscape_list,
        num_drugs=len(landscape_list),
        gen_per_step=gen_per_step,
        random_start=random_start,
    )

    trajectories = []
    best_drug = None
    best_fitness = None
    from remarc.core.simulations import run_one_drug_sim

    for i in range(len(landscape_list)):
        env.reset()
        trajectory = run_one_drug_sim(
            env=env, drug_A=i, num_episodes=num_episodes, episode_length=episode_length
        )
        trajectories.append(trajectory)
        if (
            best_fitness is None or trajectory["Fitness"].mean() < best_fitness
        ):  # find lowest fitness
            best_drug = i
            best_fitness = trajectory["Fitness"].mean()

    return best_drug, best_fitness, trajectories


def log_trajectory_step(signature, episode, step, genotype, fitness, drug):
    if not signature:
        return
    log_dir = project_root / "log" / "trajectories"
    log_dir.mkdir(parents=True, exist_ok=True)
    filename = log_dir / f"{signature}_live.csv"

    if not filename.exists():
        with open(filename, "w") as f:
            f.write("episode,step,genotype,fitness,drug\n")

    with open(filename, "a") as f:
        f.write(f"{episode},{step},{genotype},{fitness},{drug}\n")


def log_policy_snapshot(signature, policy, env):
    if not signature:
        return
    log_dir = project_root / "log" / "policies"
    log_dir.mkdir(parents=True, exist_ok=True)
    filename = log_dir / f"{signature}_live.json"

    n_states = 2**env.seq_length
    state_tensor = torch.FloatTensor(np.identity(n_states))

    # Move state tensor to whatever device the policy is on
    if hasattr(policy, "actor"):
        device = next(policy.actor.parameters()).device
    elif hasattr(policy, "model"):
        device = next(policy.model.parameters()).device
    else:
        device = torch.device("cpu")

    state_tensor = state_tensor.to(device)

    with torch.no_grad():
        if hasattr(policy, "model"):  # DQN
            q_values = policy.model(state_tensor).cpu().numpy().tolist()
        elif hasattr(policy, "actor"):  # PPO
            actor_out = policy.actor(state_tensor)
            if isinstance(actor_out, tuple):
                actor_out = actor_out[0]
            q_values = actor_out.cpu().numpy().tolist()
        else:
            return

    snapshot = {"n_states": n_states, "q_values": q_values}

    with open(filename, "w") as f:
        json.dump(snapshot, f)


def run_sim_tianshou(
    env, policy: BasePolicy, num_episodes=10, episode_length=20, signature=None
):
    """
    Simulates the environment for a number of episodes using a given policy.

    Args:
        env: the WrightFisherEnv environment
        policy: the Tianshou policy to evaluate
        num_episodes (int): The number of simulation episodes.
        episode_length (int): The length of each episode.

    Returns:
        pd.DataFrame: A dataframe containing the simulation history.
    """
    states = []
    actions = []
    time_steps = []
    episodes = []
    fitnesses = []

    log_file = None
    if signature:
        log_dir = project_root / "log" / "trajectories"
        log_dir.mkdir(parents=True, exist_ok=True)
        filename = log_dir / f"{signature}_live.csv"
        log_file = open(filename, "w")
        log_file.write("episode,step,genotype,fitness,drug\n")

    if hasattr(policy, "eval"):
        policy.eval()

    with torch.no_grad():
        for i in range(num_episodes):
            env.reset()
            obs = env.get_obs()

            for j in range(episode_length):
                states.append(obs)

                batch = Batch(obs=[obs], info=Batch())
                action = int(policy(batch).act[0])

                obs, rew, terminated, truncated, info = env.step(action)
                actions.append(int(action))
                fitness = env.get_fitness(raw=True)
                fitnesses.append(fitness)
                time_steps.append(j)
                episodes.append(i)

                # Real-time trajectory logging
                if log_file:
                    log_file.write(
                        f"{i},{j},{int(np.argmax(obs))},{fitness},{int(action)}\n"
                    )
                    if j % 1000 == 0:
                        log_file.flush()

    if log_file:
        log_file.close()

    results_df = pd.DataFrame(
        {
            "Episode": episodes,
            "Time Step": time_steps,
            "State": states,
            "Action": actions,
            "Fitness": fitnesses,
        }
    )
    return results_df


def run_sim_shepherd(
    env, mdp_solver, num_episodes=10, episode_length=20, signature=None
):
    """
    Simulates the environment using the SHEPHERD MDP exact solver policy.
    """
    states = []
    actions = []
    time_steps = []
    episodes = []
    fitnesses = []

    for i in range(num_episodes):
        env.reset()
        obs = env.get_obs()

        for j in range(episode_length):
            states.append(obs)

            # Convert one-hot observation back to a frequency vector on the simplex
            x = np.zeros(mdp_solver.M)
            x[np.argmax(obs)] = 1.0

            # Query the pre-solved MDP policy
            action = mdp_solver.get_action(x)

            obs, rew, terminated, truncated, info = env.step(action)
            actions.append(int(action))
            fitnesses.append(env.get_fitness(raw=True))
            time_steps.append(j)
            episodes.append(i)

            # Real-time trajectory logging
            log_trajectory_step(
                signature,
                i,
                j,
                int(np.argmax(obs)),
                env.get_fitness(raw=True),
                int(action),
            )

    results_df = pd.DataFrame(
        {
            "Episode": episodes,
            "Time Step": time_steps,
            "State": states,
            "Action": actions,
            "Fitness": fitnesses,
        }
    )
    return results_df


def run_wright_fisher(
    train: bool, signature: str | None = None, filename: str | None = None, hp_args=None
):
    p_base = Presets.p1_ls()

    # Determine correct action space size and state shape based on dataset
    v_dataset = hp_args.dataset if hp_args else p_base.dataset
    if v_dataset == "eight_state":
        v_num_drugs = len(define_eight_state_landscapes())
        v_N = 3
    elif v_dataset == "four_state":
        amp = getattr(hp_args, "landscape_amplification", 1.0) if hp_args else 1.0
        v_num_drugs = len(define_four_state_landscapes(amplification=amp))
        v_N = 2
    elif v_dataset == "three_state":
        amp = getattr(hp_args, "landscape_amplification", 1.0) if hp_args else 1.0
        v_num_drugs = len(define_three_state_landscapes(amplification=amp))
        v_N = 2  # Not used directly by ThreeGenotypeEnv since it hardcodes num_genotypes=3
    else:
        v_num_drugs = 10  # synthetic default
        v_N = hp_args.n_mut if hp_args else 4

    v_state_shape = (2**v_N,)  # State shape matches genotype count
    v_num_actions = v_num_drugs

    ##SET HOW MANY EPISODES TO RUN & HOW LONG FOR TESTING
    num_episodes = getattr(hp_args, "test_episodes", 100) if hp_args else 100
    episode_length = getattr(hp_args, "test_episode_length", 100) if hp_args else 100
    v_sigma = hp_args.sigma if hp_args else 0.5

    p = p_base
    if hp_args:
        p = Presets(
            state_shape=v_state_shape,
            num_actions=v_num_actions,
            lr=hp_args.lr or p_base.lr,
            epochs=hp_args.epochs or p_base.epochs,
            train_steps_per_epoch=p_base.train_steps_per_epoch,
            test_episodes=p_base.test_episodes,
            batch_size=hp_args.batch_size or p_base.batch_size,
            buffer_size=p_base.buffer_size,
            activation=hp_args.activation or p_base.activation,
            reward_clip=hp_args.reward_clip,
            dataset=hp_args.dataset if hp_args else "eight_state",
            gen_per_step=hp_args.gen_per_step if hp_args else 500,
            ent_coef=hp_args.ent_coef,
            gamma=hp_args.gamma
            if hp_args and hasattr(hp_args, "gamma")
            else p_base.gamma,
            gae_lambda=hp_args.gae_lambda
            if hp_args and hasattr(hp_args, "gae_lambda")
            else p_base.gae_lambda,
            episode_steps=hp_args.episode_steps,
            reward_scale=hp_args.reward_scale if hp_args else 100.0,
            random_start=getattr(hp_args, "random_start", True),
            landscape_amplification=getattr(hp_args, "landscape_amplification", 1.0),
            stochastic=getattr(hp_args, "stochastic", True),
            delta_multiplier=getattr(hp_args, "delta_multiplier", 0.0),
        )
    else:
        # Even if no hp_args, we should ensure num_actions and state_shape are correct for the dataset
        p = Presets(
            state_shape=v_state_shape,
            num_actions=v_num_actions,
            lr=p_base.lr,
            epochs=p_base.epochs,
            train_steps_per_epoch=p_base.train_steps_per_epoch,
            test_episodes=p_base.test_episodes,
            batch_size=p_base.batch_size,
            buffer_size=p_base.buffer_size,
            activation=p_base.activation,
            reward_clip=p_base.reward_clip,
            dataset=p_base.dataset,
            gen_per_step=p_base.gen_per_step,
            ent_coef=p_base.ent_coef,
            episode_steps=p_base.episode_steps,
            reward_scale=p_base.reward_scale,
            random_start=p_base.random_start,
            landscape_amplification=p_base.landscape_amplification,
            stochastic=p_base.stochastic,
        )

    if train:
        print("Training Wright Fisher...")
        train_wf_landscapes(p=p, signature=signature)

    if filename is None:
        filename = "best_policy.pth"
        if signature:
            filename = f"{Path(filename).stem}_{signature}.pth"

    print(f"Using policy file: {filename}")

    # Try to load shared landscapes from pickle first (for consistent evaluation)
    active_landscapes = None
    if signature:
        pickle_dir = os.path.join(project_root, "log", "RL")
        pickle_file = f"active_landscapes_{signature}.pkl"
        pickle_path = os.path.join(pickle_dir, pickle_file)
        if os.path.exists(pickle_path):
            print(
                f"Loading shared landscapes from {pickle_path} for consistent evaluation."
            )
            with open(pickle_path, "rb") as f:
                active_landscapes = pickle.load(f)

    if active_landscapes is None:
        print("Warning: No pickled landscapes found. Using raw dataset initialization.")
        if v_dataset == "eight_state":
            ls = define_eight_state_landscapes()
        elif v_dataset == "four_state":
            amp = getattr(hp_args, "landscape_amplification", 1.0) if hp_args else 1.0
            ls = define_four_state_landscapes(amplification=amp)
        elif v_dataset == "three_state":
            amp = getattr(hp_args, "landscape_amplification", 1.0) if hp_args else 1.0
            ls = define_three_state_landscapes(amplification=amp)
        else:  # synthetic
            ls = None  # Handled by fallback logic below

        landscape_list = []
        if ls is not None:
            for i in range(len(ls)):
                landscape_obj = Landscape(v_N, sigma=0.0, ls=ls[i])
                landscape_list.append(landscape_obj)
            active_landscapes = landscape_list
        else:
            # Final fallback for synthetic or unknown
            active_landscapes = [
                Landscape(v_N, sigma=v_sigma) for _ in range(v_num_drugs)
            ]

    # WF uses PPO, so we must load as PPO
    v_random_start = getattr(hp_args, "random_start", True) if hp_args else True
    v_stochastic = getattr(hp_args, "stochastic", True) if hp_args else True
    if v_dataset == "three_state":
        env = ThreeGenotypeEnv(
            num_drugs=v_num_drugs,
            seq_length=v_N,
            landscape_list=active_landscapes,
            gen_per_step=hp_args.gen_per_step if hp_args else 500,
            reward_scale=hp_args.reward_scale if hp_args else 100.0,
            random_start=v_random_start,
            stochastic=v_stochastic,
            delta_multiplier=p.delta_multiplier,
        )
    else:
        env = WrightFisherEnv(
            num_drugs=v_num_drugs,
            seq_length=v_N,
            landscape_list=active_landscapes,
            gen_per_step=hp_args.gen_per_step if hp_args else 500,
            reward_scale=hp_args.reward_scale if hp_args else 100.0,
            random_start=v_random_start,
            stochastic=v_stochastic,
            delta_multiplier=p.delta_multiplier,
        )
    best_policy = load_best_policy(p, filename=filename, env_type="wf", ppo=True)
    # Update env with WF specific parameters
    if hp_args:
        env.pop_size = hp_args.pop_size
        env.mutation_rate = hp_args.mutation_rate
        env.switch_interval = hp_args.gen_per_step

    results_df = run_sim_tianshou(
        env=env,
        policy=best_policy,
        num_episodes=num_episodes,
        episode_length=episode_length,
        signature=signature,
    )
    print(results_df.loc[:, ["Episode", "Time Step", "Action", "Fitness"]])

    print("\nAverage WF fitness: ", np.mean(results_df["Fitness"]))

    actions = results_df["Action"]

    action_freq = {i: 0 for i in range(env.num_drugs * env.num_concs)}
    for action in actions:
        action_freq[action] += 1

    sorted_actions = np.array(list(action_freq.keys()))[
        np.argsort(np.array(list(action_freq.values())))
    ][::-1]
    reformatted_actions = [
        f"{(action % v_num_drugs, int(action / v_num_drugs))}: {action_freq[action]}"
        for action in sorted_actions
    ]
    print("Top actions: \n\n", reformatted_actions)

    # Evaluate random policy baseline
    random_results_df_plot = None
    if v_dataset == "three_state":
        random_env = ThreeGenotypeEnv(
            num_drugs=v_num_drugs,
            seq_length=v_N,
            landscape_list=active_landscapes,
            gen_per_step=hp_args.gen_per_step if hp_args else 500,
            reward_scale=hp_args.reward_scale if hp_args else 100.0,
            stochastic=v_stochastic,
            delta_multiplier=p.delta_multiplier,
        )
    else:
        random_env = WrightFisherEnv(
            num_drugs=v_num_drugs,
            seq_length=v_N,
            landscape_list=active_landscapes,
            gen_per_step=hp_args.gen_per_step if hp_args else 500,
            reward_scale=hp_args.reward_scale if hp_args else 100.0,
            stochastic=v_stochastic,
            delta_multiplier=p.delta_multiplier,
        )
    random_results_df_plot = run_sim_tianshou(
        env=random_env,
        policy=load_random_policy(p),
        num_episodes=num_episodes,
        episode_length=episode_length,
    )

    if random_results_df_plot is not None:
        print(
            "\nAverage Random WF fitness: ", np.mean(random_results_df_plot["Fitness"])
        )

    # Evaluate SHEPHERD baseline if computationally feasible (N <= 3)
    shepherd_results_df_plot = None
    if v_N <= 3 and (hp_args and getattr(hp_args, "eval_shepherd", False)):
        print("\nEvaluating SHEPHERD MDP baseline...")
        shepherd_mdp = ShepherdMDP.from_env(
            env, L=getattr(hp_args, "shepherd_resolution", 3), discount=0.99
        )
        shepherd_mdp.solve()
        if v_dataset == "three_state":
            shepherd_env = ThreeGenotypeEnv(
                num_drugs=v_num_drugs,
                seq_length=v_N,
                landscape_list=active_landscapes,
                gen_per_step=hp_args.gen_per_step if hp_args else 500,
                reward_scale=hp_args.reward_scale if hp_args else 100.0,
                stochastic=v_stochastic,
                delta_multiplier=p.delta_multiplier,
            )
        else:
            shepherd_env = WrightFisherEnv(
                num_drugs=v_num_drugs,
                seq_length=v_N,
                landscape_list=active_landscapes,
                gen_per_step=hp_args.gen_per_step if hp_args else 500,
                reward_scale=hp_args.reward_scale if hp_args else 100.0,
                stochastic=v_stochastic,
                delta_multiplier=p.delta_multiplier,
            )
        shepherd_results_df_plot = run_sim_shepherd(
            env=shepherd_env,
            mdp_solver=shepherd_mdp,
            num_episodes=num_episodes,
            episode_length=episode_length,
        )
        print(
            "\nAverage SHEPHERD WF fitness: ",
            np.mean(shepherd_results_df_plot["Fitness"]),
        )

    # Evaluate best single-drug baseline (only if trained with signature)
    if signature:
        print("\nEvaluating best single-drug baseline...")
        try:
            # Use the SAME landscapes the learned policy was trained/evaluated on
            # to ensure fitness values are on the same scale.
            landscapes = np.array([landscape_obj.ls for landscape_obj in active_landscapes])

            # Evaluate all single-drug policies
            best_drug_id, best_fitness, all_trajectories = evaluate_best_single_drug(
                landscape=landscapes,
                num_episodes=num_episodes,
                seq_length=v_N,
                episode_length=episode_length,
                gen_per_step=hp_args.gen_per_step if hp_args else 500,
                sigma=v_sigma,  # Use the same sigma as training (0.0 for empirical)
                random_start=v_random_start,
            )

            assert best_drug_id is not None and best_fitness is not None, (
                "No best drug found"
            )
            print(
                f"Best single drug: #{best_drug_id} with mean fitness: {best_fitness:.4f}"
            )

            # Extract trajectories for ALL single-drug baselines
            all_drug_episodes = {}  # drug_index -> list of episode trajectories
            for drug_idx, traj_df in enumerate(all_trajectories):
                drug_episodes = []
                for i in range(num_episodes):
                    episode_data = traj_df[traj_df["Episode"] == i]
                    drug_episodes.append(episode_data["Fitness"].tolist())
                all_drug_episodes[drug_idx] = drug_episodes

            # Extract trajectories for learned policy
            learned_episodes = []
            learned_states = []
            for i in range(num_episodes):
                episode_data = results_df[results_df["Episode"] == i]
                learned_episodes.append(episode_data["Fitness"].tolist())
                learned_states.append([s.tolist() for s in episode_data["State"]])

            # Extract random policy trajectories
            random_episodes = []
            if random_results_df_plot is not None:
                for i in range(num_episodes):
                    episode_data = random_results_df_plot[
                        random_results_df_plot["Episode"] == i
                    ]
                    random_episodes.append(episode_data["Fitness"].tolist())

            # Extract SHEPHERD policy trajectories
            shepherd_episodes = None
            if shepherd_results_df_plot is not None:
                shepherd_episodes = []
                for i in range(num_episodes):
                    episode_data = shepherd_results_df_plot[
                        shepherd_results_df_plot["Episode"] == i
                    ]
                    shepherd_episodes.append(episode_data["Fitness"].tolist())

            # Save baseline, learned, and random results
            baseline_dir = os.path.join(project_root, "log", "baselines")
            os.makedirs(baseline_dir, exist_ok=True)

            baseline_file = os.path.join(baseline_dir, f"{signature}_baseline.json")
            with open(baseline_file, "w") as f:
                json.dump(
                    {
                        "best_drug": int(best_drug_id),
                        "mean_fitness": float(best_fitness),
                        "num_drugs": len(all_trajectories),
                        "all_drug_trajectories": {
                            str(k): v for k, v in all_drug_episodes.items()
                        },
                        "all_drug_mean_fitness": {
                            str(k): float(np.mean([np.mean(ep) for ep in v]))
                            for k, v in all_drug_episodes.items()
                        },
                        "shepherd_trajectories": shepherd_episodes,
                    },
                    f,
                )

            learned_file = os.path.join(baseline_dir, f"{signature}_learned.json")
            with open(learned_file, "w") as f:
                json.dump(
                    {
                        "mean_fitness": float(
                            np.mean([np.mean(ep) for ep in learned_episodes])
                        ),
                        "trajectories": learned_episodes,
                        "state_trajectories": learned_states,
                    },
                    f,
                )

            random_file = os.path.join(baseline_dir, f"{signature}_random.json")
            with open(random_file, "w") as f:
                json.dump(
                    {
                        "mean_fitness": float(
                            np.mean([np.mean(ep) for ep in random_episodes])
                        ),
                        "trajectories": random_episodes,
                    },
                    f,
                )

            print(f"Saved baseline comparison to: {baseline_dir}")

            # --- PLOTTING ---
            print("\nGenerating performance plot...")
            plt.figure(figsize=(12, 7))

            # Normalize fitness to relative fitness (0 to 1) across all policies
            plot_results_df = results_df.copy()
            plot_all_trajectories = [df.copy() for df in all_trajectories]
            plot_random_results_df = (
                random_results_df_plot.copy()
                if random_results_df_plot is not None
                else None
            )
            plot_shepherd_results_df = (
                shepherd_results_df_plot.copy()
                if shepherd_results_df_plot is not None
                else None
            )

            all_fitness_series = []
            if plot_results_df is not None:
                all_fitness_series.append(plot_results_df["Fitness"])
            for traj_df in plot_all_trajectories:
                all_fitness_series.append(traj_df["Fitness"])
            if plot_random_results_df is not None:
                all_fitness_series.append(plot_random_results_df["Fitness"])
            if plot_shepherd_results_df is not None:
                all_fitness_series.append(plot_shepherd_results_df["Fitness"])

            if all_fitness_series:
                import pandas as pd

                combined = pd.concat(all_fitness_series)
                global_min = combined.min()
                global_max = combined.max()
                denom = global_max - global_min if global_max != global_min else 1.0

                plot_results_df["Fitness"] = (
                    plot_results_df["Fitness"] - global_min
                ) / denom
                for traj_df in plot_all_trajectories:
                    traj_df["Fitness"] = (traj_df["Fitness"] - global_min) / denom
                if plot_random_results_df is not None:
                    plot_random_results_df["Fitness"] = (
                        plot_random_results_df["Fitness"] - global_min
                    ) / denom
                if plot_shepherd_results_df is not None:
                    plot_shepherd_results_df["Fitness"] = (
                        plot_shepherd_results_df["Fitness"] - global_min
                    ) / denom

            # Learned Policy
            learned_mean = plot_results_df.groupby("Time Step")["Fitness"].mean()
            learned_std = plot_results_df.groupby("Time Step")["Fitness"].std()

            plt.plot(
                learned_mean.index,
                learned_mean,
                label="Learned Policy",
                linewidth=2.5,
                color="blue",
            )
            plt.fill_between(
                learned_mean.index,
                learned_mean - learned_std,
                learned_mean + learned_std,
                color="blue",
                alpha=0.1,
            )

            # All Single Drug Policies
            drug_colors = [
                "#e67e22",
                "#27ae60",
                "#8e44ad",
                "#c0392b",
                "#2980b9",
                "#f39c12",
                "#1abc9c",
                "#d35400",
                "#7f8c8d",
                "#2c3e50",
            ]
            for drug_idx, traj_df in enumerate(plot_all_trajectories):
                color = drug_colors[drug_idx % len(drug_colors)]
                drug_mean = traj_df.groupby("Time Step")["Fitness"].mean()
                drug_std = traj_df.groupby("Time Step")["Fitness"].std()

                is_best = drug_idx == best_drug_id
                label = f"Drug {drug_idx}" + (" ★" if is_best else "")
                lw = 2.0 if is_best else 1.2
                alpha_line = 1.0 if is_best else 0.7

                plt.plot(
                    drug_mean.index,
                    drug_mean,
                    label=label,
                    linewidth=lw,
                    color=color,
                    linestyle="--",
                    alpha=alpha_line,
                )
                plt.fill_between(
                    drug_mean.index,
                    drug_mean - drug_std,
                    drug_mean + drug_std,
                    color=color,
                    alpha=0.05,
                )

            # Random Policy
            if plot_random_results_df is not None:
                random_mean = plot_random_results_df.groupby("Time Step")[
                    "Fitness"
                ].mean()
                random_std = plot_random_results_df.groupby("Time Step")[
                    "Fitness"
                ].std()
                plt.plot(
                    random_mean.index,
                    random_mean,
                    label="Random Policy",
                    linewidth=2,
                    color="red",
                    linestyle=":",
                )
                plt.fill_between(
                    random_mean.index,
                    random_mean - random_std,
                    random_mean + random_std,
                    color="red",
                    alpha=0.1,
                )

            # SHEPHERD Policy
            if plot_shepherd_results_df is not None:
                shepherd_mean = plot_shepherd_results_df.groupby("Time Step")[
                    "Fitness"
                ].mean()
                shepherd_std = plot_shepherd_results_df.groupby("Time Step")[
                    "Fitness"
                ].std()
                plt.plot(
                    shepherd_mean.index,
                    shepherd_mean,
                    label="SHEPHERD Policy",
                    linewidth=2,
                    color="black",
                    linestyle="-.",
                )
                plt.fill_between(
                    shepherd_mean.index,
                    shepherd_mean - shepherd_std,
                    shepherd_mean + shepherd_std,
                    color="black",
                    alpha=0.1,
                )

            plt.xlabel("Time Step")
            plt.ylabel("relative fitness")
            plt.title("Policy Performance Over Time (Wright-Fisher)")
            plt.legend(loc="best", fontsize=9)
            plt.grid(True, alpha=0.3)

            plot_dir = os.path.join(project_root, "log", "plots")
            os.makedirs(plot_dir, exist_ok=True)
            plot_file = os.path.join(
                plot_dir, f"performance_comparison_{signature}.png"
            )
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved performance plot to: {plot_file}")

        except Exception as e:
            print(f"Warning: Could not evaluate baseline or plot: {e}")
            import traceback

            traceback.print_exc()

    states_unflattened = np.array(results_df.loc[:, "State"].values)
    states_flat = []
    for i in range(len(states_unflattened)):
        for e in states_unflattened[i]:
            states_flat.append(e)

    states = np.array(states_flat)
    states = np.reshape(states, (num_episodes, episode_length, 2**v_N))
    print(states.shape)


def main(
    wf_test=False,
    wf_train=False,
    signature: str | None = None,
    filename: str | None = None,
    hp_args=None,
):
    if wf_test:
        run_wright_fisher(
            train=wf_train, signature=signature, filename=filename, hp_args=hp_args
        )


def main_wf_landscapes(
    train, signature: str | None = None, filename: str | None = None, hp_args=None
):
    main(
        wf_test=True,
        wf_train=train,
        signature=signature,
        filename=filename,
        hp_args=hp_args,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="REMARC Runner")
    parser.add_argument("--mode", type=str, choices=["wf_ls"], default="wf_ls")
    parser.add_argument(
        "--eval-shepherd", action="store_true", help="Evaluate SHEPHERD baseline"
    )
    parser.add_argument(
        "--no-shepherd",
        action="store_false",
        dest="eval_shepherd",
        help="Do not evaluate SHEPHERD baseline",
    )
    parser.add_argument(
        "--shepherd-resolution",
        type=int,
        default=3,
        help="Lattice resolution L for SHEPHERD exact MDP solver",
    )
    parser.add_argument("--train", action="store_true", help="Train before evaluation")
    parser.add_argument(
        "--no-train",
        action="store_false",
        dest="train",
        help="Skip training (only evaluation)",
    )
    parser.add_argument(
        "--signature",
        type=str,
        default=None,
        help="Signature to append to the policy filename during training",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Explicit policy filename to use during evaluation",
    )

    # Hyperparameters
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument(
        "--epochs", type=int, default=None, help="Number of epochs/episodes"
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Minibatch size")
    parser.add_argument("--n-mut", type=int, default=3, help="Number of mutations (N)")
    parser.add_argument("--sigma", type=float, default=0.0, help="Sigma for landscapes")
    parser.add_argument(
        "--pop-size", type=int, default=10000, help="Population size (for WF)"
    )
    parser.add_argument(
        "--mutation-rate", type=float, default=1e-5, help="Mutation rate (for WF)"
    )
    parser.add_argument(
        "--gen-per-step", type=int, default=500, help="Generations per step"
    )
    parser.add_argument(
        "--reward-scale", type=float, default=100.0, help="Scale for rewards"
    )
    parser.add_argument(
        "--activation",
        type=str,
        default=None,
        help="Activation function (relu, tanh, swish, etc.)",
    )
    parser.add_argument(
        "--reward-clip",
        action="store_true",
        help="Enable reward clipping (default roughly [-5, 5])",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="eight_state",
        choices=["eight_state", "four_state", "three_state", "synthetic"],
        help="Dataset to use (eight_state, four_state, three_state, or synthetic)",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.05,
        help="Entropy coefficient for PPO (default: 0.05)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor (gamma) (default: 0.99)",
    )
    parser.add_argument(
        "--gae-lambda", type=float, default=0.95, help="GAE lambda (default: 0.95)"
    )
    parser.add_argument(
        "--delta-multiplier",
        type=float,
        default=0.0,
        help="Multiplier for the step-to-step delta bonus reward (default: 0.0)",
    )
    parser.add_argument(
        "--episode-steps",
        type=int,
        default=20,
        help="Number of steps per episode (default: 20)",
    )
    parser.add_argument(
        "--random-start",
        action="store_true",
        default=True,
        help="Start episodes from random genotypes (default: True)",
    )
    parser.add_argument(
        "--no-random-start",
        action="store_false",
        dest="random_start",
        help="Start all episodes from genotype 000",
    )
    parser.add_argument(
        "--landscape-amplification",
        type=float,
        default=1.0,
        help="Amplify fitness deviations in Four-State landscape (e.g. 10.0 for 10x selection pressure)",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        default=True,
        help="Use stochastic Wright-Fisher with multinomial sampling (default: True)",
    )
    parser.add_argument(
        "--no-stochastic",
        action="store_false",
        dest="stochastic",
        help="Use deterministic Fokker-Planck mode (no genetic drift)",
    )
    parser.add_argument(
        "--test-episodes",
        type=int,
        default=100,
        help="Number of episodes to run for testing",
    )
    parser.add_argument(
        "--test-episode-length",
        type=int,
        default=100,
        help="Length of each testing episode",
    )

    parser.set_defaults(train=True)

    args = parser.parse_args()

    if args.mode == "wf_ls":
        main_wf_landscapes(
            train=args.train,
            signature=args.signature,
            filename=args.filename,
            hp_args=args,
        )
