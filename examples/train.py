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

from remarc.envs.utils import define_chen_landscapes
from remarc.envs import WrightFisherEnv, define_chen_landscapes, define_four_state_landscapes
from remarc.core.hyperparameters import Presets
from remarc.core.landscapes import Landscape
from remarc.agents.tianshou_agent import load_best_policy, load_random_policy, train_wf_landscapes

# Set up logging
timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / "log" / "archive" / f"wf_run_{timestamp}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Alias all prints as logger.info for consistency
def print(*args, **kwargs):
    logger.info(' '.join(str(arg) for arg in args))


builtins.print = print



def evaluate_best_single_drug(landscape : np.ndarray = define_chen_landscapes(), num_episodes : int = 20, seq_length : int = 3, episode_length : int = 20, gen_per_step: int = 500, sigma: float = 0.0):
    """
    Evaluate the best single drug policy through simulating each individually.

    Returns:
        best_drug: int - index of the best-performing drug
        best_fitness: float - fitness of the best-performing drug
        trajectories: list of pd.DataFrame - trajectories of each drug
    """
    landscape_list = [Landscape(N=seq_length, sigma=sigma, ls=landscape[i, :]) for i in range(len(landscape))]
    env = WrightFisherEnv(seq_length=seq_length, landscape_list=landscape_list, num_drugs=len(landscape_list), gen_per_step=gen_per_step)
    
    trajectories = []
    best_drug = None
    best_fitness = None
    from remarc.core.simulations import run_one_drug_sim
    for i in range(len(landscape_list)):
        env.reset()
        trajectory = run_one_drug_sim(env=env, drug_A=i, num_episodes=num_episodes, episode_length=episode_length)
        trajectories.append(trajectory)
        if best_fitness is None or trajectory['Fitness'].mean() < best_fitness:  # find lowest fitness
            best_drug = i
            best_fitness = trajectory['Fitness'].mean()

    return best_drug, best_fitness, trajectories



def log_trajectory_step(signature, episode, step, genotype, fitness, drug):
    if not signature:
        return
    log_dir = project_root / "log" / "trajectories"
    log_dir.mkdir(parents=True, exist_ok=True)
    filename = log_dir / f"{signature}_live.csv"
    
    # Write header if file doesn't exist
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
    
    n_states = 2**env.N
    state_tensor = torch.FloatTensor(np.identity(n_states))
    
    with torch.no_grad():
        if hasattr(policy, "model"): # DQN
            q_values = policy.model(state_tensor).cpu().numpy().tolist()
        elif hasattr(policy, "actor"): # PPO
            actor_out = policy.actor(state_tensor)
            if isinstance(actor_out, tuple):
                actor_out = actor_out[0]
            q_values = actor_out.cpu().numpy().tolist()
        else:
            return
            
    snapshot = {
        "n_states": n_states,
        "q_values": q_values
    }
    
    with open(filename, "w") as f:
        json.dump(snapshot, f)



def run_sim_tianshou(env, policy: BasePolicy, num_episodes=10, episode_length=20, signature=None):
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

    for i in range(num_episodes):
        env.reset()
        obs = env.get_obs()

        for j in range(episode_length):
            states.append(obs)

            batch = Batch(obs=[obs], info=Batch())
            action = int(policy(batch).act[0])

            obs, rew, terminated, truncated, info = env.step(action)
            actions.append(int(action))
            fitnesses.append(env.get_fitness(raw=True))
            time_steps.append(j)
            episodes.append(i)
            
            # Real-time trajectory logging
            log_trajectory_step(signature, i, j, int(np.argmax(obs)), env.get_fitness(raw=True), int(action))

    results_df = pd.DataFrame(
        {"Episode": episodes, "Time Step": time_steps, "State": states, "Action": actions, "Fitness": fitnesses})
    return results_df


def run_wright_fisher(train: bool, signature: str | None = None, filename: str | None = None, hp_args=None):
    p_base = Presets.p1_ls()
    
    # Determine correct action space size and state shape based on dataset
    v_dataset = hp_args.dataset if hp_args else p_base.dataset
    if v_dataset == "chen":
        v_num_drugs = len(define_chen_landscapes())
        v_N = 3
    elif v_dataset == "four_state":
        v_num_drugs = len(define_four_state_landscapes())
        v_N = 2
    else:
        v_num_drugs = 10  # synthetic default
        v_N = hp_args.n_mut if hp_args else 4
    
    v_state_shape = (2**v_N,)  # State shape matches genotype count
    v_num_actions = v_num_drugs

    ##SET HOW MANY EPISODES TO RUN & HOW LONG FOR TESTING
    num_episodes = 100
    episode_length = 100
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
            dataset=hp_args.dataset if hp_args else "chen",
            gen_per_step=hp_args.gen_per_step if hp_args else 500,
            ent_coef=hp_args.ent_coef,
            episode_steps=hp_args.episode_steps
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
            episode_steps=p_base.episode_steps
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
        pickle_file = "active_landscapes.pkl"
        pickle_path = os.path.join(pickle_dir, pickle_file)
        if os.path.exists(pickle_path):
            print(f"Loading shared landscapes from {pickle_path} for consistent evaluation.")
            with open(pickle_path, "rb") as f:
                active_landscapes = pickle.load(f)
    
    if active_landscapes is None:
        print("Warning: No pickled landscapes found. Using raw dataset initialization.")
        if v_dataset == "chen":
            ls = define_chen_landscapes()
        elif v_dataset == "four_state":
            ls = define_four_state_landscapes()
        else: # synthetic
            ls = None # Handled by fallback logic below
        
        landscape_list = []
        if ls is not None:
            for i in range(len(ls)):
                l = Landscape(v_N, sigma=0.0, ls=ls[i])
                landscape_list.append(l)
            active_landscapes = landscape_list
        else:
            # Final fallback for synthetic or unknown
            active_landscapes = [Landscape(v_N, sigma=v_sigma) for _ in range(v_num_drugs)]

    # WF uses PPO, so we must load as PPO
    env = WrightFisherEnv(num_drugs=v_num_drugs, seq_length=v_N, landscape_list=active_landscapes, gen_per_step=hp_args.gen_per_step if hp_args else 500, reward_scale=hp_args.reward_scale if hp_args else 100.0)
    best_policy = load_best_policy(p, filename=filename, env_type="wf", ppo=True)
    # Update env with WF specific parameters
    if hp_args:
        env.pop_size = hp_args.pop_size
        env.mutation_rate = hp_args.mutation_rate
        env.switch_interval = hp_args.gen_per_step

    results_df = run_sim_tianshou(env=env, policy=best_policy, num_episodes=num_episodes, episode_length=episode_length, signature=signature)
    print(results_df.loc[:, ["Episode", "Time Step", "Action", "Fitness"]])

    print("\nAverage WF fitness: ", np.mean(results_df["Fitness"]))

    actions = results_df["Action"]

    action_freq = {i: 0 for i in range(env.num_drugs * env.num_concs)}
    for action in actions:
        action_freq[action] += 1

    sorted_actions = np.array(list(action_freq.keys()))[np.argsort(np.array(list(action_freq.values())))][::-1]
    reformatted_actions = [f"{(action % v_num_drugs, int(action / v_num_drugs))}: {action_freq[action]}" for action in sorted_actions]
    print("Top actions: \n\n", reformatted_actions)

    # Evaluate random policy baseline
    random_results_df_plot = None
    random_env = WrightFisherEnv(num_drugs=v_num_drugs, seq_length=v_N, landscape_list=active_landscapes, gen_per_step=hp_args.gen_per_step if hp_args else 500, reward_scale=hp_args.reward_scale if hp_args else 100.0)
    random_results_df_plot = run_sim_tianshou(env=random_env, policy=load_random_policy(p), num_episodes=20, episode_length=episode_length)
    
    if random_results_df_plot is not None:
        print("\nAverage Random WF fitness: ", np.mean(random_results_df_plot["Fitness"]))
    
    # Evaluate best single-drug baseline (only if trained with signature)
    if signature:
        print("\nEvaluating best single-drug baseline...")
        try:
            # Use the SAME landscapes the learned policy was trained/evaluated on
            # to ensure fitness values are on the same scale.
            landscapes = np.array([l.ls for l in active_landscapes])
            
            # Evaluate all single-drug policies
            best_drug_id, best_fitness, all_trajectories = evaluate_best_single_drug(
                landscape=landscapes, 
                num_episodes=num_episodes,
                seq_length=v_N,
                episode_length=episode_length,
                gen_per_step=hp_args.gen_per_step if hp_args else 500,
                sigma=v_sigma # Use the same sigma as training (0.0 for empirical)
            )
            
            assert best_drug_id is not None and best_fitness is not None, "No best drug found"
            print(f"Best single drug: #{best_drug_id} with mean fitness: {best_fitness:.4f}")
            
            # Extract trajectories for ALL single-drug baselines
            all_drug_episodes = {}  # drug_index -> list of episode trajectories
            for drug_idx, traj_df in enumerate(all_trajectories):
                drug_episodes = []
                for i in range(20):
                    episode_data = traj_df[traj_df['Episode'] == i]
                    drug_episodes.append(episode_data['Fitness'].tolist())
                all_drug_episodes[drug_idx] = drug_episodes
            
            # Extract trajectories for learned policy
            learned_episodes = []
            for i in range(20):
                episode_data = results_df[results_df['Episode'] == i]
                learned_episodes.append(episode_data['Fitness'].tolist())

            # Extract random policy trajectories
            random_episodes = []
            if random_results_df_plot is not None:
                for i in range(20):
                    episode_data = random_results_df_plot[random_results_df_plot['Episode'] == i]
                    random_episodes.append(episode_data['Fitness'].tolist())

            # Save baseline, learned, and random results
            baseline_dir = os.path.join(project_root, "log", "baselines")
            os.makedirs(baseline_dir, exist_ok=True)
            
            baseline_file = os.path.join(baseline_dir, f"{signature}_baseline.json")
            with open(baseline_file, 'w') as f:
                json.dump({
                    'best_drug': int(best_drug_id),
                    'mean_fitness': float(best_fitness),
                    'num_drugs': len(all_trajectories),
                    'all_drug_trajectories': {str(k): v for k, v in all_drug_episodes.items()},
                    'all_drug_mean_fitness': {
                        str(k): float(np.mean([np.mean(ep) for ep in v]))
                        for k, v in all_drug_episodes.items()
                    }
                }, f)
            
            learned_file = os.path.join(baseline_dir, f"{signature}_learned.json")
            with open(learned_file, 'w') as f:
                json.dump({
                    'mean_fitness': float(np.mean([np.mean(ep) for ep in learned_episodes])),
                    'trajectories': learned_episodes
                }, f)
            
            random_file = os.path.join(baseline_dir, f"{signature}_random.json")
            with open(random_file, 'w') as f:
                json.dump({
                    'mean_fitness': float(np.mean([np.mean(ep) for ep in random_episodes])),
                    'trajectories': random_episodes
                }, f)
            
            print(f"Saved baseline comparison to: {baseline_dir}")

            # --- PLOTTING ---
            print("\nGenerating performance plot...")
            plt.figure(figsize=(12, 7))
            
            # Learned Policy
            learned_mean = results_df.groupby('Time Step')['Fitness'].mean()
            learned_std = results_df.groupby('Time Step')['Fitness'].std()
            
            plt.plot(learned_mean.index, learned_mean, label='Learned Policy', linewidth=2.5, color='blue')
            plt.fill_between(learned_mean.index, 
                             learned_mean - learned_std, 
                             learned_mean + learned_std, 
                             color='blue', alpha=0.1)

            # All Single Drug Policies
            drug_colors = ['#e67e22', '#27ae60', '#8e44ad', '#c0392b', '#2980b9', '#f39c12', '#1abc9c', '#d35400', '#7f8c8d', '#2c3e50']
            for drug_idx, traj_df in enumerate(all_trajectories):
                color = drug_colors[drug_idx % len(drug_colors)]
                drug_mean = traj_df.groupby('Time Step')['Fitness'].mean()
                drug_std = traj_df.groupby('Time Step')['Fitness'].std()
                
                is_best = (drug_idx == best_drug_id)
                label = f'Drug {drug_idx}' + (' ★' if is_best else '')
                lw = 2.0 if is_best else 1.2
                alpha_line = 1.0 if is_best else 0.7
                
                plt.plot(drug_mean.index, drug_mean, label=label, linewidth=lw, 
                         color=color, linestyle='--', alpha=alpha_line)
                plt.fill_between(drug_mean.index,
                                 drug_mean - drug_std,
                                 drug_mean + drug_std,
                                 color=color, alpha=0.05)

            # Random Policy
            if random_results_df_plot is not None:
                random_mean = random_results_df_plot.groupby('Time Step')['Fitness'].mean()
                random_std = random_results_df_plot.groupby('Time Step')['Fitness'].std()
                plt.plot(random_mean.index, random_mean, label='Random Policy', linewidth=2, color='red', linestyle=':')
                plt.fill_between(random_mean.index,
                                    random_mean - random_std,
                                    random_mean + random_std,
                                    color='red', alpha=0.1)

            plt.xlabel('Time Step')
            plt.ylabel('Population Fitness')
            plt.title('Policy Performance Over Time (Wright-Fisher)')
            plt.legend(loc='best', fontsize=9)
            plt.grid(True, alpha=0.3)
            
            plot_dir = os.path.join(project_root, "log", "plots")
            os.makedirs(plot_dir, exist_ok=True)
            plot_file = os.path.join(plot_dir, f"performance_comparison_{signature}.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
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


def main(wf_test=False, wf_train=False, signature: str | None = None, filename: str | None = None, hp_args=None):
    if wf_test:
        run_wright_fisher(train=wf_train, signature=signature, filename=filename, hp_args=hp_args)


def main_wf_landscapes(train, signature: str | None = None, filename: str | None = None, hp_args=None):
    main(wf_test=True, wf_train=train, signature=signature, filename=filename, hp_args=hp_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="REMARC Runner")
    parser.add_argument("--mode", type=str, choices=["wf_ls"], default="wf_ls")
    parser.add_argument("--train", action="store_true", help="Train before evaluation")
    parser.add_argument("--no-train", action="store_false", dest="train", help="Skip training (only evaluation)")
    parser.add_argument("--signature", type=str, default=None, help="Signature to append to the policy filename during training")
    parser.add_argument("--filename", type=str, default=None, help="Explicit policy filename to use during evaluation")

    # Hyperparameters
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs/episodes")
    parser.add_argument("--batch-size", type=int, default=None, help="Minibatch size")
    parser.add_argument("--n-mut", type=int, default=3, help="Number of mutations (N)")
    parser.add_argument("--sigma", type=float, default=0.0, help="Sigma for landscapes")
    parser.add_argument("--pop-size", type=int, default=10000, help="Population size (for WF)")
    parser.add_argument("--mutation-rate", type=float, default=1e-5, help="Mutation rate (for WF)")
    parser.add_argument("--gen-per-step", type=int, default=500, help="Generations per step")
    parser.add_argument("--reward-scale", type=float, default=100.0, help="Scale for rewards")
    parser.add_argument("--activation", type=str, default=None, help="Activation function (relu, tanh, swish, etc.)")
    parser.add_argument("--reward-clip", action="store_true", help="Enable reward clipping (default roughly [-5, 5])")
    parser.add_argument("--dataset", type=str, default="chen", choices=["chen", "four_state", "synthetic"], help="Dataset to use (chen, four_state, or synthetic)")
    parser.add_argument("--ent-coef", type=float, default=0.05, help="Entropy coefficient for PPO (default: 0.05)")
    parser.add_argument("--episode-steps", type=int, default=20, help="Number of steps per episode (default: 20)")

    parser.set_defaults(train=True)

    args = parser.parse_args()

    if args.mode == "wf_ls":
        main_wf_landscapes(train=args.train, signature=args.signature, filename=args.filename, hp_args=args)
