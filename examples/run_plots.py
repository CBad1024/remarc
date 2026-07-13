import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from itertools import product


project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from remarc.core.hyperparameters import Presets as P
from remarc.agents.tianshou_agent import (
    get_ppo_policy,
    load_best_fn,
    RandomPolicy,
    SingleDrugPolicy,
    train_wf_landscapes,
)
from remarc.envs.wright_fisher_env import WrightFisherEnv
from remarc.envs.utils import (
    define_four_state_landscapes,
    define_three_state_landscapes,
    define_eight_state_landscapes,
)
from remarc.core.landscapes import Landscape
from tianshou.env import DummyVectorEnv
from tianshou.data import Batch
from remarc.agents.shepherd_eval import ShepherdMDP
from remarc.agents.greedy_agent import GreedyAgent
from remarc.envs.wright_fisher_env import ThreeGenotypeEnv

from examples.plotting import (
    plot_simplex_policy_slices,
    plot_policy_fitness_landscape_slices,
    plot_policy_difference_slices,
    plot_policy_magnitude_difference_slices,
    plot_population_density_slices,
    greedy_policy,
    plot_dominant_modes,
)


def binary_indices(N):
    return ["".join(bits) for bits in product("01", repeat=N)]


def run_eval(
    env, policy, agent_type="RL", num_runs=10, episode_steps=1000, drug_idx=0
):  # DRUG INDEX NOT USED UNLESS SINGLE DRUG POLICY
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
            states.append(obs[-env.num_genotypes :])

            if agent_type == "Greedy":
                current_state = obs[-env.num_genotypes :]
                action = agent.get_action(current_state)
            elif agent_type == "Random":
                batch = Batch(obs=np.array([obs]))
                action = agent(batch).act[0]
            elif isinstance(agent_type, int):  # Single Drug
                action = agent_type
            elif agent_type == "Shepherd":
                action = agent(obs[-env.num_genotypes :])
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
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--arch", type=str, default="256", choices=["32", "64", "128", "256", "baseline"], help="Network architecture size")
    # args = parser.parse_args()
    LR = 1e-4
    MR = 1e-5
    EPOCHS = 300
    EPISODE_STEPS = 500
    REWARD_SCALE = 100.0
    AMP = 1.0
    EVAL_STEPS = 5000
    EVAL_RUNS = 100
    n_frames = 1
    delta = 0.5
    delta_horizon = 1
    gps = 10
    gamma = 0.99
    ent = 0.1
    batch = 64
    DATASET = "four_state"  # Accepts "three_state", "four_state", or "eight_state"
    L = 3
    TRAIN = False  # Set to True to train the model, False to load existing model

    sig = f"{DATASET}_dh_{delta_horizon}_d{delta}_g{gps}_gam{gamma}_e{ent}_b{batch}"

    # if args.arch == "baseline":
    #     sig = f"{DATASET}_d{delta}_g{gps}_gam{gamma}_e{ent}_b{batch}"
    # else:
    #     sig = f"Three State_net{args.arch}_d{delta}_g{gps}_gam{gamma}_e{ent}_b{batch}"

    # if args.arch == "32":
    #     hidden, head = [32, 32], [32]
    # elif args.arch == "64":
    #     hidden, head = [64, 64], [64]
    # elif args.arch == "128":
    #     hidden, head = [128, 128], [128]
    # elif args.arch == "256":
    #     hidden, head = [256, 256, 256], [128]
    # else:
    hidden, head = [256, 256, 256], [128]  # Baseline uses 256 architecture

    if DATASET == "three_state":
        landscape_data = define_three_state_landscapes(amplification=AMP)
        v_N = 2
    elif DATASET == "four_state":
        landscape_data = define_four_state_landscapes(amplification=AMP)
    elif DATASET == "eight_state":
        landscape_data = define_eight_state_landscapes()
    else:
        raise ValueError(f"Unknown dataset: {DATASET}")

    num_drugs = landscape_data.shape[0]
    if DATASET != "three_state":
        v_N = int(np.log2(len(landscape_data[0])))

    g_min, g_max = np.min(landscape_data), np.max(landscape_data)
    landscape_list = [
        Landscape(v_N, sigma=0.0, ls=landscape_data[i], g_min=g_min, g_max=g_max)
        for i in range(num_drugs)
    ]

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
        dataset=DATASET,
        landscape_amplification=AMP,
        test_episodes=5,
        episode_steps=EPISODE_STEPS,
        delta_multiplier=delta,
        stochastic=True,
        random_start=True,
        n_frames=n_frames,
        delta_horizon=delta_horizon,
    )
    object.__setattr__(p, "mutation_rate", MR)
    object.__setattr__(p, "n_train_envs", 16)
    object.__setattr__(p, "n_test_envs", 4)
    object.__setattr__(p, "hidden_sizes", hidden)
    object.__setattr__(p, "head_sizes", head)

    env_kwargs = dict(
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
        n_frames=n_frames,
        delta_horizon=delta_horizon,
    )
    if DATASET == "three_state":
        eval_env = ThreeGenotypeEnv(**env_kwargs)
    else:
        eval_env = WrightFisherEnv(**env_kwargs)

    test_envs = DummyVectorEnv([lambda: eval_env])

    # Load policy if it exists, else train

    if not TRAIN:
        try:
            policy = get_ppo_policy(p, test_envs).eval()
            policy = load_best_fn(policy, f"best_policy_{sig}.pth")
        except Exception:
            # run the training
            train_wf_landscapes(p, signature=sig)
            policy = get_ppo_policy(p, test_envs).eval()
            policy = load_best_fn(policy, f"best_policy_{sig}.pth")
    else:
        train_wf_landscapes(p, signature=sig)
        policy = get_ppo_policy(p, test_envs).eval()
        policy = load_best_fn(policy, f"best_policy_{sig}.pth")

    shepherd_mdp = ShepherdMDP.from_env(eval_env, L=L, discount=0.99)
    cache_path = project_root / "log" / f"shepherd_L{L}_{DATASET}.npz"

    if cache_path.exists():
        print(f"Loading cached SHEPHERD MDP (L={L})...")
        shepherd_mdp.load(cache_path)
    else:
        print(f"Solving Exact SHEPHERD MDP (L={L})...")
        shepherd_mdp.solve()
        shepherd_mdp.save(cache_path)
        print(f"Saved cached SHEPHERD policy to {cache_path}")

    def shepherd_fn(state):
        return shepherd_mdp.get_action(state)

    print("Evaluating models...")
    rl_m, rl_std, rl_states = run_eval(eval_env, policy, "RL", EVAL_RUNS, EVAL_STEPS)
    gr_m, gr_std, gr_states = run_eval(eval_env, None, "Greedy", EVAL_RUNS, EVAL_STEPS)
    rn_m, rn_std, rn_states = run_eval(eval_env, None, "Random", EVAL_RUNS, EVAL_STEPS)
    sh_m, sh_std, sh_states = run_eval(
        eval_env, shepherd_fn, "Shepherd", EVAL_RUNS, EVAL_STEPS
    )
    sd1_m, sd1_std, sd1_states = run_eval(
        eval_env, None, "Single Drug", EVAL_RUNS, EVAL_STEPS, drug_idx=0
    )
    sd2_m, sd2_std, sd2_states = run_eval(
        eval_env, None, "Single Drug", EVAL_RUNS, EVAL_STEPS, drug_idx=1
    )
    sd3_m, sd3_std, sd3_states = run_eval(
        eval_env, None, "Single Drug", EVAL_RUNS, EVAL_STEPS, drug_idx=2
    )
    sd4_m, sd4_std, sd4_states = run_eval(
        eval_env, None, "Single Drug", EVAL_RUNS, EVAL_STEPS, drug_idx=3
    )

    # 1. Fitness Trajectories
    print("Plotting Fitness Trajectories...")

    def plot_fitness_trajectories(max_steps=None, suffix=""):
        if max_steps is None:
            max_steps = EVAL_STEPS

        plt.figure(figsize=(12, 8))
        steps = np.arange(max_steps)

        def norm(arr):
            return (arr[:max_steps] - g_min) / (g_max - g_min)

        def norm_std(arr):
            return arr[:max_steps] / (g_max - g_min)

        # Random
        plt.plot(steps, norm(rn_m), color="gray", ls=":", lw=2, label="Random Mean")
        plt.fill_between(
            steps,
            norm(rn_m) - norm_std(rn_std),
            norm(rn_m) + norm_std(rn_std),
            color="gray",
            alpha=0.1,
        )

        # Greedy
        plt.plot(
            steps, norm(gr_m), color="#ff7f0e", ls="--", lw=2.5, label="Greedy Mean"
        )
        plt.fill_between(
            steps,
            norm(gr_m) - norm_std(gr_std),
            norm(gr_m) + norm_std(gr_std),
            color="#ff7f0e",
            alpha=0.2,
        )

        # SHEPHERD
        plt.plot(
            steps, norm(sh_m), color="black", ls="-.", lw=2.5, label="SHEPHERD Mean"
        )
        plt.fill_between(
            steps,
            norm(sh_m) - norm_std(sh_std),
            norm(sh_m) + norm_std(sh_std),
            color="black",
            alpha=0.2,
        )

        # Learned
        plt.plot(steps, norm(rl_m), color="#1f77b4", lw=3, label="Learned Mean")
        plt.fill_between(
            steps,
            norm(rl_m) - norm_std(rl_std),
            norm(rl_m) + norm_std(rl_std),
            color="#1f77b4",
            alpha=0.2,
        )

        # Single Drugs
        colors = ["#FF0000", "#32CD32", "#8A2BE2", "#FF1493"]

        for d in range(num_drugs):
            if d == 0:
                m, std = sd1_m, sd1_std
            elif d == 1:
                m, std = sd2_m, sd2_std
            elif d == 2:
                m, std = sd3_m, sd3_std
            else:
                m, std = sd4_m, sd4_std
            plt.plot(
                steps,
                norm(m),
                color=colors[d],
                ls="-.",
                lw=2.5,
                label=f"Drug {d} Mean",
                alpha=1.0,
            )
            plt.fill_between(
                steps,
                norm(m) - norm_std(std),
                norm(m) + norm_std(std),
                color=colors[d],
                alpha=0.2,
            )

        plt.ylim(0, 1)
        plt.grid(True, ls="--", alpha=0.4)
        title_str = (
            f"Normalized Fitness Trajectories ({max_steps} steps)\nPolicy: {sig}"
        )
        if max_steps < EVAL_STEPS:
            title_str = "ZOOMED: " + title_str
        plt.title(title_str, fontsize=14, fontweight="bold")
        plt.xlabel("RL Steps", fontsize=12)
        plt.ylabel("Normalized Fitness", fontsize=12)
        plt.legend(fontsize=10, loc="center right", bbox_to_anchor=(1.25, 0.5))
        plt.tight_layout()
        plt.savefig(
            str(project_root / "log" / f"{sig}_dashboard_trajectories{suffix}.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()

    plot_fitness_trajectories(max_steps=EVAL_STEPS, suffix="")
    plot_fitness_trajectories(max_steps=min(100, EVAL_STEPS), suffix="_zoomed")

    # Define policy functions for plotting
    def rl_policy_fn(state):
        batch = Batch(obs=np.array([state]), info={})
        with torch.no_grad():
            res = policy(batch)
        return res.act[0]

    def gr_policy_fn(state):
        return greedy_policy(state, landscape_data)

    if DATASET == "three_state":
        genotype_labels = ["0", "1", "2"]
        is_three_state = True
    else:
        genotype_labels = binary_indices(v_N)
        is_three_state = False

    # 2. Dominant Modes/PCA Analysis
    print("Plotting Dominant Modes (PCA Analysis)...")

    def fitness_fn(x, a):
        return np.mean(np.dot(landscape_data[a], x))

    fig_rl_pca, fig_rl_simplex, pca_info = plot_dominant_modes(
        state_trajectories=rl_states,
        policy_fn=lambda x: rl_policy_fn(x),
        fitness_fn=fitness_fn,
        burn_in=100,
        drug_colors=[
            "#2ecc71",
            "#e67e22",
            "#5b7cc9",
            "#e84393",
            "#f1c40f",
            "#1abc9c",
            "#9b59b6",
            "#e74c3c",
        ][:num_drugs],
        show_sample_paths=False,
        simplex_view=True,
        title="Dominant Modes (REMARC Policy)",
    )
    fig_rl_pca.savefig(
        str(project_root / "log" / f"{sig}_dominant_modes_remarc_pca.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig_rl_pca)
    if fig_rl_simplex is not None:
        fig_rl_simplex.savefig(
            str(project_root / "log" / f"{sig}_dominant_modes_remarc_simplex.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig_rl_simplex)

    print("REMARC PCA Info:", pca_info)

    fig_gr_pca, fig_gr_simplex, pca_info = plot_dominant_modes(
        state_trajectories=gr_states,
        policy_fn=lambda x: gr_policy_fn(x),
        fitness_fn=fitness_fn,
        burn_in=100,
        drug_colors=[
            "#2ecc71",
            "#e67e22",
            "#5b7cc9",
            "#e84393",
            "#f1c40f",
            "#1abc9c",
            "#9b59b6",
            "#e74c3c",
        ][:num_drugs],
        show_sample_paths=False,
        simplex_view=True,
        title="Dominant Modes (Greedy Policy)",
    )
    fig_gr_pca.savefig(
        str(project_root / "log" / f"{sig}_dominant_modes_greedy_pca.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig_gr_pca)
    if fig_gr_simplex is not None:
        fig_gr_simplex.savefig(
            str(project_root / "log" / f"{sig}_dominant_modes_greedy_simplex.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig_gr_simplex)

    print("Greedy PCA Info:", pca_info)

    fig_sh_pca, fig_sh_simplex, pca_info = plot_dominant_modes(
        state_trajectories=sh_states,
        policy_fn=lambda x: shepherd_fn(x),
        fitness_fn=fitness_fn,
        burn_in=100,
        drug_colors=[
            "#2ecc71",
            "#e67e22",
            "#5b7cc9",
            "#e84393",
            "#f1c40f",
            "#1abc9c",
            "#9b59b6",
            "#e74c3c",
        ][:num_drugs],
        show_sample_paths=False,
        simplex_view=True,
        title="Dominant Modes (SHEPHERD Policy)",
    )
    fig_sh_pca.savefig(
        str(project_root / "log" / f"{sig}_dominant_modes_shepherd_pca.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig_sh_pca)
    if fig_sh_simplex is not None:
        fig_sh_simplex.savefig(
            str(project_root / "log" / f"{sig}_dominant_modes_shepherd_simplex.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig_sh_simplex)

    if DATASET != "eight_state":
        # 2. Population distribution simplexes
        print("Plotting Population Distribution Simplexes...")
        fig = plot_population_density_slices(
            state_trajectories=rl_states,
            policy_fn=rl_policy_fn,
            greedy_policy_fn=gr_policy_fn,
            num_drugs=num_drugs,
            is_three_state=is_three_state,
        )
        if fig is not None:
            fig.savefig(
                str(project_root / "log" / f"{sig}_dashboard_pop_density.png"),
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)

        # 3. Policy simplexes
        print("Plotting Policy Simplexes...")
        fig = plot_simplex_policy_slices(
            policy_fn=rl_policy_fn,
            num_drugs=num_drugs,
            genotype_labels=genotype_labels,
            title=f"Learned Policy: {sig}",
            is_three_state=is_three_state,
        )
        if fig is not None:
            fig.savefig(
                str(project_root / "log" / f"{sig}_dashboard_policy.png"),
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)

        fig = plot_simplex_policy_slices(
            policy_fn=gr_policy_fn,
            num_drugs=num_drugs,
            genotype_labels=genotype_labels,
            title="Greedy Policy",
            is_three_state=is_three_state,
        )
        if fig is not None:
            fig.savefig(
                str(project_root / "log" / f"{sig}_dashboard_greedy_policy.png"),
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)

        fig = plot_simplex_policy_slices(
            policy_fn=shepherd_fn,
            num_drugs=num_drugs,
            genotype_labels=genotype_labels,
            title=f"SHEPHERD Policy (L={L})",
            is_three_state=is_three_state,
        )
        if fig is not None:
            fig.savefig(
                str(project_root / "log" / f"{sig}_dashboard_shepherd_policy.png"),
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)

        # 4. Population fitness under policy simplexes
        print("Plotting Policy Fitness Landscape Slices...")
        fig = plot_policy_fitness_landscape_slices(
            policy_fn=rl_policy_fn,
            landscapes=landscape_data,
            num_drugs=num_drugs,
            genotype_labels=genotype_labels,
            title="Normalized Fitness Landscape (Learned Policy)",
            is_three_state=is_three_state,
        )
        if fig is not None:
            fig.savefig(
                str(project_root / "log" / f"{sig}_dashboard_fitness_landscape.png"),
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)

        # 5. Disagreement plots
        print("Plotting Policy Disagreement...")
        fig = plot_policy_difference_slices(
            policy_fn_1=rl_policy_fn,
            policy_fn_2=gr_policy_fn,
            num_drugs=num_drugs,
            genotype_labels=genotype_labels,
            title="Policy Disagreement (Learned vs Greedy)",
            is_three_state=is_three_state,
        )
        if fig is not None:
            fig.savefig(
                str(project_root / "log" / f"{sig}_dashboard_disagreement.png"),
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)

        # 6. Absolute disagreement plots
        print("Plotting Absolute Disagreement (Greedy)...")
        fig = plot_policy_magnitude_difference_slices(
            policy_fn_1=rl_policy_fn,
            policy_fn_2=gr_policy_fn,
            landscapes=landscape_data,
            num_drugs=num_drugs,
            genotype_labels=genotype_labels,
            title="Absolute Policy Fitness Difference (Learned vs Greedy)",
            is_three_state=is_three_state,
        )
        if fig is not None:
            fig.savefig(
                str(
                    project_root / "log" / f"{sig}_dashboard_magnitude_disagreement.png"
                ),
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)

        # 7. Disagreement plots (Learned vs SHEPHERD)
        print("Plotting Policy Disagreement (SHEPHERD)...")
        fig = plot_policy_difference_slices(
            policy_fn_1=rl_policy_fn,
            policy_fn_2=shepherd_fn,
            num_drugs=num_drugs,
            genotype_labels=genotype_labels,
            title="Policy Disagreement (Learned vs SHEPHERD)",
            is_three_state=is_three_state,
        )
        if fig is not None:
            fig.savefig(
                str(
                    project_root / "log" / f"{sig}_dashboard_shepherd_disagreement.png"
                ),
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)

        # 8. Absolute disagreement plots (Learned vs SHEPHERD)
        print("Plotting Absolute Disagreement (SHEPHERD)...")
        fig = plot_policy_magnitude_difference_slices(
            policy_fn_1=rl_policy_fn,
            policy_fn_2=shepherd_fn,
            landscapes=landscape_data,
            num_drugs=num_drugs,
            genotype_labels=genotype_labels,
            title="Absolute Policy Fitness Difference (Learned vs SHEPHERD)",
            is_three_state=is_three_state,
        )
        if fig is not None:
            fig.savefig(
                str(
                    project_root
                    / "log"
                    / f"{sig}_dashboard_shepherd_magnitude_disagreement.png"
                ),
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)

        # Pop x fitness and Pop x disagreement
        print("Plotting Pop x fitness and Pop x disagreement...")
        from examples.plotting import plot_pop_x_metric_slices
        
        def fitness_metric(x):
            return fitness_fn(x, rl_policy_fn(x))
            
        fig_pop_fit = plot_pop_x_metric_slices(
            state_trajectories=rl_states,
            metric_fn=fitness_metric,
            metric_name="Pop x Fitness",
            num_drugs=num_drugs,
            is_three_state=is_three_state,
        )
        fig_pop_fit.savefig(
            str(project_root / "log" / f"{sig}_dashboard_pop_x_fitness.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig_pop_fit)

        def disagreement_metric(x):
            return 1.0 if rl_policy_fn(x) != gr_policy_fn(x) else 0.0

        fig_pop_dis = plot_pop_x_metric_slices(
            state_trajectories=rl_states,
            metric_fn=disagreement_metric,
            metric_name="Pop x Disagreement",
            num_drugs=num_drugs,
            is_three_state=is_three_state,
        )
        fig_pop_dis.savefig(
            str(project_root / "log" / f"{sig}_dashboard_pop_x_disagreement.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig_pop_dis)

        # 9. Steady state frequencies on decision boundaries
        print("Plotting Steady State Frequencies over Decision Boundary...")
        rl_final = np.array([run[-1] for run in rl_states])
        sh_final = np.array([run[-1] for run in sh_states])
        gr_final = np.array([run[-1] for run in gr_states])

        fig = plot_simplex_policy_slices(
            policy_fn=rl_policy_fn,
            num_drugs=num_drugs,
            genotype_labels=genotype_labels,
            title="Steady State (Learned Policy)",
            is_three_state=is_three_state,
            scatter_states=rl_final,
        )
        if fig is not None:
            fig.savefig(
                str(project_root / "log" / f"{sig}_dashboard_steady_state_RL.png"),
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)

        fig = plot_simplex_policy_slices(
            policy_fn=shepherd_fn,
            num_drugs=num_drugs,
            genotype_labels=genotype_labels,
            title="Steady State (SHEPHERD Policy)",
            is_three_state=is_three_state,
            scatter_states=sh_final,
        )
        if fig is not None:
            fig.savefig(
                str(
                    project_root / "log" / f"{sig}_dashboard_steady_state_SHEPHERD.png"
                ),
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)

        fig = plot_simplex_policy_slices(
            policy_fn=gr_policy_fn,
            num_drugs=num_drugs,
            genotype_labels=genotype_labels,
            title="Steady State (Greedy Policy)",
            is_three_state=is_three_state,
            scatter_states=gr_final,
        )
        if fig is not None:
            fig.savefig(
                str(project_root / "log" / f"{sig}_dashboard_steady_state_Greedy.png"),
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)

    # 10. Frequency trajectories
    print("Plotting Frequency Trajectories...")

    def plot_freq_traj(states_list, title_prefix, filename_suffix):
        states_arr = np.array(states_list)  # (runs, steps, genotypes)
        mean_freqs = np.mean(states_arr, axis=0)  # (steps, genotypes)
        steps = np.arange(mean_freqs.shape[0])
        plt.figure(figsize=(12, 6))
        for i in range(mean_freqs.shape[1]):
            plt.plot(steps, mean_freqs[:, i], label=genotype_labels[i], lw=2)

        plt.ylim(0, 1)
        plt.grid(True, ls="--", alpha=0.4)
        plt.title(
            f"{title_prefix} Genotype Frequencies over Time (Averaged over runs)",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("RL Steps", fontsize=12)
        plt.ylabel("Average Frequency", fontsize=12)
        plt.legend(fontsize=10, loc="center right", bbox_to_anchor=(1.15, 0.5))
        plt.tight_layout()
        plt.savefig(
            str(project_root / "log" / f"{sig}_dashboard_freqs_{filename_suffix}.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()

    plot_freq_traj(rl_states, "Learned Policy", "RL")
    plot_freq_traj(sh_states, "SHEPHERD Policy", "SHEPHERD")
    plot_freq_traj(gr_states, "Greedy Policy", "Greedy")

    # Print out final steady-state fitnesses for Greedy, SHEPHERD, and REMARC and each of the drugs.

    def norm(arr):
        return (arr - g_min) / (g_max - g_min)

    def norm_std(arr):
        return arr / (g_max - g_min)

    print("Final Steady State Fitnesses:")
    print("Greedy:", norm(gr_m[-1]))
    print("SHEPHERD:", norm(sh_m[-1]))
    print("REMARC:", norm(rl_m[-1]))
    print("Single Drug 0:", norm(sd1_m[-1]))
    print("Single Drug 1:", norm(sd2_m[-1]))
    print("Single Drug 2:", norm(sd3_m[-1]))
    print("Single Drug 3:", norm(sd4_m[-1]))

    # Also print out the mean fitnesses for each, over all steps.
    print("Mean Fitnesses:")
    print("Greedy:", norm(np.mean(gr_m)))
    print("SHEPHERD:", norm(np.mean(sh_m)))
    print("REMARC:", norm(np.mean(rl_m)))
    print("Single Drug 0:", norm(np.mean(sd1_m)))
    print("Single Drug 1:", norm(np.mean(sd2_m)))
    print("Single Drug 2:", norm(np.mean(sd3_m)))
    print("Single Drug 3:", norm(np.mean(sd4_m)))
    print("Done!")


if __name__ == "__main__":
    main()
