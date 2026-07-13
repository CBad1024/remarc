import argparse
import matplotlib.pyplot as plt
import numpy as np

from tianshou.data import Batch

from remarc.agents import load_testing_envs, ONNXAgent, GreedyAgent


def run_simulation(env, policy, steps=20):
    """
    Runs the environment for a fixed number of steps using the given policy.
    Returns the trajectory of genotype frequencies.
    """
    # WrightFisherEnv uses global np.random for mutations/drift, so we must seed it globally
    np.random.seed(42)
    obs, info = env.reset(seed=42)  # Fixed seed for deterministic behavior
    trajectory = [obs]

    for _ in range(steps):
        # Create a Tianshou-compatible batch just in case
        batch = Batch(obs=np.array([obs]), info={})

        if hasattr(policy, "get_action"):
            # Use our standardized custom agent interface
            action = policy.get_action(obs)
        else:
            # Fallback for raw Tianshou policies
            res = policy(batch)
            action = res.act[0]

        obs, reward, term, trunc, info = env.step(action)
        trajectory.append(obs)

        if term or trunc:
            break

    return np.array(trajectory)


import re


def compare_models(onnx_model_path, policy_path, steps=500, gen_per_step=1):
    print("Loading test environments...")
    envs = load_testing_envs()
    env = envs[0]  # Pick the first test environment

    # Explicitly inject generations per step
    print(
        f"Setting simulation switch interval to {gen_per_step} generations per step..."
    )
    env.switch_interval = gen_per_step

    # Explicitly inject mutation rate from the policy name to avoid any ambiguity
    match = re.search(r"([0-9.]+[eE]?[-+]?[0-9]*)MR", policy_path)
    if match:
        mutation_rate = float(match.group(1))
        print(
            f"Injecting explicit mutation rate {mutation_rate} from policy name into environment..."
        )
        env.mutation_rate = mutation_rate
        env.mutation_matrix = env._build_mutation_matrix()
    else:
        print(
            f"Warning: Could not parse mutation rate from {policy_path}. Using environment default: {env.mutation_rate}"
        )

    print("Initializing Baseline Greedy Agent...")
    greedy_agent = GreedyAgent(env.drug_landscapes)

    print(f"Loading ONNX model from {onnx_model_path}...")
    onnx_agent = ONNXAgent(onnx_model_path)

    print(f"Running simulation for {steps} steps...")
    traj_greedy = run_simulation(env, greedy_agent, steps)
    traj_onnx = run_simulation(env, onnx_agent, steps)

    num_genotypes = traj_greedy.shape[1]

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Plot Greedy baseline
    for i in range(num_genotypes):
        ax1.plot(traj_greedy[:, i], label=f"G{i}", linewidth=3)
    ax1.set_title("Baseline (Greedy) Population Trajectory")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Genotype Frequency")
    ax1.legend()

    # Plot ONNX RL agent
    for i in range(num_genotypes):
        ax2.plot(traj_onnx[:, i], label=f"G{i}", linewidth=3, linestyle="-")
    ax2.set_title("RL Agent (ONNX) Population Trajectory")
    ax2.set_xlabel("Time Step")
    ax2.legend()

    plt.tight_layout()

    plot_path = "model_comparison.png"
    plt.savefig(plot_path)
    print(f"\n✅ Simulation complete! Comparison plot saved to {plot_path}")

    # Optional mathematical comparison of fitness sum (proxy for total tumor load)
    # Lower is better (more cell death)
    fitness_greedy = np.sum([env.drug_landscapes.dot(obs) for obs in traj_greedy])
    fitness_onnx = np.sum([env.drug_landscapes.dot(obs) for obs in traj_onnx])

    print(f"--- Evaluation over {steps} steps ---")
    print(f"Total Tumor Load (Greedy Baseline): {fitness_greedy:.2f}")
    print(f"Total Tumor Load (RL ONNX Agent)  : {fitness_onnx:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default="rl_policy.onnx")
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        help="Original policy file path to extract parameters from",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--gen-per-step", type=int, default=1)
    args = parser.parse_args()

    compare_models(args.onnx, args.policy, args.steps, args.gen_per_step)
