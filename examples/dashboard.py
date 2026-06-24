import streamlit as st
import subprocess
import sys
import os
import time
import fcntl
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from remarc.envs import define_chen_landscapes, define_chen_landscapes, define_four_state_landscapes

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(page_title="REMARC Playground", layout="wide")

# Custom CSS for Playground feel
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        border-radius: 20px;
    }
    .highlight-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .metric-card {
        text-align: center;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Constants
NUM_TABS = 5
MODES = {"Wright-Fisher Landscapes": "wf_ls"}


DATASET_OPTIONS = {
        "Chen et al.": {"N": 3, "description": "Empirical fitness landscapes from Chen et al.", "cli": "chen"},
    "Four-State": {"N": 2, "description": "Empirical four-state fitness landscapes (4 drugs, N=2).", "cli": "four_state"},
    "Synthetic": {"N": "any", "description": "Randomly generated landscapes with configurable N.", "cli": "synthetic"}
}

# Initialize session state for simulations
if "run_queue" not in st.session_state:
    st.session_state.run_queue = []
if "cooldown_seconds" not in st.session_state:
    st.session_state.cooldown_seconds = 120
if "cooldown_until" not in st.session_state:
    st.session_state.cooldown_until = 0

if "sims" not in st.session_state:
    st.session_state.sims = {}
    for i in range(NUM_TABS):
        st.session_state.sims[i] = {
            "mode": "Wright-Fisher Landscapes",
            "train": True,
            "process": None,
            "logs": "No logs yet. Click 'Run' to start.\n",
            "running": False,
            "exit_code": None,
            "hp": {
                "lr": 0.0001,
                "epochs": 50,
                "batch_size": 128,
                "n_mut": 4,
                "sigma": 0.5,
                "dataset": "Synthetic",
                "pop_size": 10000,
                "mutation_rate": 1e-5,
                "gen_per_step": 25,
                "activation": "relu",
                "reward_clip": False,
                "ent_coef": 0.05,
                "episode_steps": 20,
                "reward_scale": 100.0,
                "random_start": True,
                "landscape_amplification": 1.0,
                "stochastic": True,
                "test_episodes": 100,
                "test_episode_length": 100,
                "plot_shepherd": True
            },
            "starred": False
        }
def copy_settings_callback(tab_id):
    copy_source = st.session_state.get(f"copy_source_{tab_id}")
    if not copy_source:
        return
    source_id = int(copy_source.split(" ")[1]) - 1
    
    tgt_sim = st.session_state.sims[tab_id]
    src_sim = st.session_state.sims[source_id]
    
    import copy
    tgt_sim["hp"] = copy.deepcopy(src_sim["hp"])
    
    fields_to_copy = ["mode", "train", "signature", "last_auto_sig", "selected_policy"]
    for field in fields_to_copy:
        if field in src_sim:
            tgt_sim[field] = copy.deepcopy(src_sim[field])
            
    # Explicitly update widget states to force frontend to reflect new values
    widget_mapping = {
        "lr": "lr", "epochs": "epochs", "batch_size": "batch",
        "n_mut": "n_mut", "sigma": "sigma", "dataset": "dataset_sel",
        "activation": "act", "reward_clip": "clip", "pop_size": "pop",
        "mutation_rate": "mut", "gen_per_step": "gps", "ent_coef": "ent",
        "gamma": "gamma", "gae_lambda": "gae", "delta_multiplier": "delta",
        "episode_steps": "ep_steps", "reward_scale": "reward_scale",
        "random_start": "rstart", "stochastic": "stochastic",
        "test_episodes": "te", "test_episode_length": "tel",
        "plot_shepherd": "shep", "shepherd_resolution": "shep_res",
        "landscape_amplification": "amp"
    }
    for hp_key, widget_prefix in widget_mapping.items():
        if hp_key in tgt_sim["hp"]:
            st.session_state[f"{widget_prefix}_{tab_id}"] = tgt_sim["hp"][hp_key]
            
    st.session_state[f"mode_sel_{tab_id}"] = tgt_sim["mode"]
    st.session_state[f"train_cb_{tab_id}"] = tgt_sim["train"]
    if "signature" in tgt_sim:
        st.session_state[f"sig_input_{tab_id}"] = tgt_sim["signature"]
    if "selected_policy" in tgt_sim:
        st.session_state[f"policy_custom_{tab_id}"] = tgt_sim["selected_policy"]
        
    st.toast(f"Copied settings from {copy_source}!", icon="✅")

def archive_run(signature, starred):
    if not signature:
        return
    import shutil
    
    # If not starred, delete tensorboard logs and move other files to archive
    if not starred:
        tb_dir = project_root / "log" / "tensorboard" / signature
        if tb_dir.exists():
            shutil.rmtree(tb_dir, ignore_errors=True)
            
        archive_dir = project_root / "log" / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        for folder in ["RL", "plots", "baselines", "metrics", "trajectories", "policies"]:
            src_dir = project_root / "log" / folder
            if not src_dir.exists():
                continue
            for file in src_dir.glob(f"*{signature}*"):
                if file.is_file():
                    try:
                        shutil.move(str(file), str(archive_dir / file.name))
                    except Exception:
                        pass

def start_simulation(tab_id):
    sim = st.session_state.sims[tab_id]
    
    # Archive the old run if it wasn't starred
    if sim.get("signature") and not sim.get("starred", False):
        archive_run(sim["signature"], False)
    
    # Reset starred status for the new run
    sim["starred"] = False
    
    # Auto-fix dataset mismatch for evaluation to prevent shape mismatch errors
    if not sim["train"] and sim.get("signature"):
        sig_lower = sim["signature"].lower()
        if "fourstate" in sig_lower:
            sim["hp"]["dataset"] = "Four-State"
            sim["hp"]["n_mut"] = 2
        elif "jun9" in sig_lower or "chen" in sig_lower:
            sim["hp"]["dataset"] = "Chen et al."
            sim["hp"]["n_mut"] = 3

    mode_arg = MODES[sim["mode"]]
    train_arg = "--train" if sim["train"] else "--no-train"
    
    # Use the same python executable as current process
    py_path = sys.executable
    script_path = str(project_root / "examples" / "train.py")
    
    cmd = [
        "caffeinate", "-i",
        sys.executable, "examples/train.py",
        "--mode", mode_arg, 
        train_arg,
        "--lr", str(sim["hp"]["lr"]),
        "--epochs", str(sim["hp"]["epochs"]),
        "--batch-size", str(sim["hp"]["batch_size"]),
        "--n-mut", str(sim["hp"]["n_mut"]),
        "--sigma", str(sim["hp"]["sigma"]),
        "--activation", sim["hp"]["activation"],
        "--dataset", DATASET_OPTIONS[sim["hp"]["dataset"]]["cli"],
        "--ent-coef", str(sim["hp"].get("ent_coef", 0.05)),
        "--gamma", str(sim["hp"].get("gamma", 0.99)),
        "--gae-lambda", str(sim["hp"].get("gae_lambda", 0.95)),
        "--delta-multiplier", str(sim["hp"].get("delta_multiplier", 0.0)),
        "--episode-steps", str(sim["hp"].get("episode_steps", 20)),
        "--reward-scale", str(sim["hp"].get("reward_scale", 100.0)),
        "--landscape-amplification", str(sim["hp"].get("landscape_amplification", 1.0)),
        "--test-episodes", str(sim["hp"].get("test_episodes", 100)),
        "--test-episode-length", str(sim["hp"].get("test_episode_length", 100))
    ]

    if not sim["hp"].get("plot_shepherd", True):
        cmd.append("--no-shepherd")
    else:
        cmd += ["--shepherd-resolution", str(sim["hp"].get("shepherd_resolution", 3))]

    if sim["hp"].get("reward_clip", False):
        cmd += ["--reward-clip"]

    # Mode-specific args
    if mode_arg in ["wf_ls", "wf_ss"]:
        cmd += [
            "--pop-size", str(sim["hp"]["pop_size"]),
            "--mutation-rate", str(sim["hp"]["mutation_rate"]),
            "--gen-per-step", str(sim["hp"]["gen_per_step"])
        ]
        if sim["hp"].get("random_start", True):
            cmd += ["--random-start"]
        else:
            cmd += ["--no-random-start"]
        if sim["hp"].get("stochastic", True):
            cmd += ["--stochastic"]
        else:
            cmd += ["--no-stochastic"]

    if sim["train"] and sim.get("signature"):
        cmd += ["--signature", sim["signature"]]
    elif not sim["train"] and sim.get("selected_policy"):
        cmd += ["--filename", sim["selected_policy"]]
        if sim.get("signature"):
            cmd += ["--signature", sim["signature"]]
    

    
    # Cleanup old live files
    signature = sim.get("signature")
    if signature:
        for folder in ["trajectories", "policies"]:
            fpath = project_root / "log" / folder / f"{signature}_live.{'csv' if folder == 'trajectories' else 'json'}"
            if fpath.exists():
                fpath.unlink()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(project_root),
        env=env,
        bufsize=0,
        text=False
    )
    
    if sys.platform != "win32" and process.stdout is not None:
        fd = process.stdout.fileno()
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

    sim["process"] = process
    sim["running"] = True
    sim["logs"] = f"--- STARTED: {' '.join(cmd)} ---\n\n"
    sim["exit_code"] = None

def update_logs_for_sim(tab_id):
    sim = st.session_state.sims[tab_id]
    changed = False
    max_log_len = 20000 # Keep only last 20k chars to prevent WebSocket saturation
    
    if sim["process"] and sim["process"].stdout is not None:
        fd = sim["process"].stdout.fileno()
        try:
            while True:
                chunk = os.read(fd, 8192)
                if not chunk: break
                sim["logs"] += chunk.decode('utf-8', errors='replace')
                
                # Truncate if too long
                if len(sim["logs"]) > max_log_len:
                    sim["logs"] = "... [truncated] ...\n" + sim["logs"][-max_log_len:]
                
                changed = True
        except (BlockingIOError, IOError):
            pass
        
        exit_code = sim["process"].poll()
        if exit_code is not None:
            sim["running"] = False
            sim["exit_code"] = exit_code
            try:
                while True:
                    chunk = os.read(fd, 8192)
                    if not chunk: break
                    sim["logs"] += chunk.decode('utf-8', errors='replace')
            except: pass
            
            # Final truncation check
            if len(sim["logs"]) > max_log_len:
                sim["logs"] = "... [truncated] ...\n" + sim["logs"][-max_log_len:]
                
            sim["logs"] += f"\n\n--- PROCESS EXITED with code {exit_code} ---\n"
            sim["process"] = None
            st.session_state.cooldown_until = time.time() + st.session_state.cooldown_seconds
            changed = True
            
            if exit_code == 0:
                st.toast(f"✅ Simulation {tab_id+1} Successful!", icon="🎉")
            else:
                st.toast(f"❌ Simulation {tab_id+1} Failed (Code: {exit_code})", icon="⚠️")
            st.rerun()
    return changed

def plot_landscape_heatmap(tab_id):
    sim = st.session_state.sims[tab_id]
    if sim["hp"]["dataset"] == "Chen et al.":
        data = define_chen_landscapes()
        drug_names = ['A', 'B', 'C', 'D']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(data, aspect='auto', cmap="viridis")
        fig.colorbar(im, ax=ax, label="Fitness")
        ax.set_xticks(range(8))
        ax.set_xticklabels([bin(i)[2:].zfill(3) for i in range(8)], rotation=45)
        ax.set_yticks(range(4))
        ax.set_yticklabels(drug_names)
        ax.set_title("Chen et al. Fitness Landscape")
        ax.set_xlabel("Genotypes")
        ax.set_ylabel("Drugs")
        st.pyplot(fig)
    elif sim["hp"]["dataset"] == "Four-State":
        amp = sim["hp"].get("landscape_amplification", 1.0)
        data = define_four_state_landscapes(amplification=amp)
        drug_names = ['A', 'B', 'C', 'D']
        amp_label = f" (Amplification={amp:.0f}x)" if amp != 1.0 else ""
        
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(data, aspect='auto', cmap="viridis")
        fig.colorbar(im, ax=ax, label="Fitness")
        ax.set_xticks(range(4))
        ax.set_xticklabels([bin(i)[2:].zfill(2) for i in range(4)], rotation=45)
        ax.set_yticks(range(4))
        ax.set_yticklabels(drug_names)
        ax.set_title(f"Four-State Fitness Landscape (N=2){amp_label}")
        ax.set_xlabel("Genotypes")
        ax.set_ylabel("Drugs")
        st.pyplot(fig)
    elif sim["hp"]["dataset"] == "Synthetic":
        st.info("Synthetic landscape visualization coming soon (requires generating a sample for N={}).".format(sim["hp"]["n_mut"]))

def plot_live_trajectory(tab_id):
    sim = st.session_state.sims[tab_id]
    signature = sim.get("signature")
    if not signature:
        return

    traj_file = project_root / "log" / "trajectories" / f"{signature}_live.csv"
    if traj_file.exists():
        try:
            df = pd.read_csv(traj_file)
            if not df.empty:
                st.subheader("Live Trajectory (Testing)")
                
                # Get the latest episode
                latest_ep = df['episode'].max()
                ep_df = df[df['episode'] == latest_ep]
                
                # Plot the path on a heatmap-like background or simple line plot
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(ep_df['step'], ep_df['genotype'], 'o-', label=f"Ep {latest_ep}")
                ax.set_yticks(range(2**sim["hp"]["n_mut"]))
                ax.set_yticklabels([bin(i)[2:].zfill(sim["hp"]["n_mut"]) for i in range(2**sim["hp"]["n_mut"])])
                ax.set_xlabel("Time Step")
                ax.set_ylabel("Genotype")
                ax.set_title(f"Tumor Mutation Path (Mean Fitness: {np.mean(ep_df['fitness']):.4f})")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Also show fitness over time
                st.line_chart(ep_df.set_index('step')['fitness'], height=150)
        except Exception as e:
            st.error(f"Error loading trajectory: {e}")

def plot_live_policy(tab_id):
    sim = st.session_state.sims[tab_id]
    signature = sim.get("signature")
    if not signature:
        return

    policy_file = project_root / "log" / "policies" / f"{signature}_live.json"
    if not policy_file.exists():
        return
        
    try:
        with open(policy_file, "r") as f:
            data = json.load(f)
        
        q_values = np.array(data["q_values"]) # Shape (n_states, n_actions)
        
        # Get the landscape for comparison
        is_empirical = sim["hp"]["dataset"] in ["Chen et al.", "Four-State"]
        if is_empirical:
            if sim["hp"]["dataset"] == "Chen et al.":
                true_data = define_chen_landscapes()  # Shape (4 drugs, 8 genotypes)
                dataset_name = "Chen et al."
            else:
                amp = sim["hp"].get("landscape_amplification", 1.0)
                true_data = define_four_state_landscapes(amplification=amp)  # Shape (4 drugs, 4 genotypes)
                dataset_name = "Four-State"
                
            # Transpose to match Q-values orientation (genotypes x drugs)
            true_landscape = true_data.T  # Now (genotypes, drugs)
            
            st.subheader(f"Q-Value vs {dataset_name} Landscape Comparison")
            
            # Calculate correlation
            from scipy.stats import pearsonr
            # Flatten both arrays for correlation
            q_flat = q_values.flatten()
            true_flat = true_landscape.flatten()
            correlation, p_value = pearsonr(q_flat, true_flat)
            
            # Display correlation metric prominently
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Q-Landscape Correlation", f"{correlation:.3f}")
            with col2:
                st.metric("P-value", f"{p_value:.2e}")
            with col3:
                # Interpretation
                if abs(correlation) > 0.7:
                    st.metric("Convergence", "Strong", delta="✓")
                elif abs(correlation) > 0.4:
                    st.metric("Convergence", "Moderate", delta="~")
                else:
                    st.metric("Convergence", "Weak", delta="✗")
            
            # Side-by-side heatmaps
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            # Landscape
            im1 = axes[0].imshow(true_landscape.T, aspect='auto', cmap="viridis")
            fig.colorbar(im1, ax=axes[0], label="Fitness")
            axes[0].set_title(f"True {dataset_name} Landscape")
            axes[0].set_xticks(range(data["n_states"]))
            axes[0].set_xticklabels([bin(i)[2:].zfill(int(np.log2(data["n_states"]))) for i in range(data["n_states"])], rotation=45)
            axes[0].set_ylabel("Drugs")
            axes[0].set_xlabel("Genotypes")
            
            # Learned Q-values
            im2 = axes[1].imshow(q_values.T, aspect='auto', cmap="magma")
            fig.colorbar(im2, ax=axes[1], label="Q-Value")
            axes[1].set_title("Learned Q-Values")
            axes[1].set_xticks(range(data["n_states"]))
            axes[1].set_xticklabels([bin(i)[2:].zfill(int(np.log2(data["n_states"]))) for i in range(data["n_states"])], rotation=45)
            axes[1].set_ylabel("Actions (Drugs)")
            axes[1].set_xlabel("Genotypes")
            
            # Difference map (Q - Fitness)
            # Normalize both to 0-1 range first for fair comparison
            true_norm = (true_landscape - true_landscape.min()) / (true_landscape.max() - true_landscape.min())
            q_norm = (q_values - q_values.min()) / (q_values.max() - q_values.min())
            diff = q_norm - true_norm
            
            im3 = axes[2].imshow(diff.T, aspect='auto', cmap="coolwarm", vmin=-1, vmax=1)
            fig.colorbar(im3, ax=axes[2], label="Difference (Normalized)")
            axes[2].set_title("Difference Map (Q - True)")
            axes[2].set_xticks(range(data["n_states"]))
            axes[2].set_xticklabels([bin(i)[2:].zfill(int(np.log2(data["n_states"]))) for i in range(data["n_states"])], rotation=45)
            axes[2].set_ylabel("Actions (Drugs)")
            axes[2].set_xlabel("Genotypes")
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.caption("💡 High correlation means the agent is learning the true fitness structure. Divergence may indicate strategic optimization beyond immediate fitness.")
        else:
            # For synthetic landscapes, just show Q-values
            st.subheader("Learned Value Landscape (Live)")
            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(q_values.T, aspect='auto', cmap="magma")
            fig.colorbar(im, ax=ax, label="Q-Value / Advantage")
            
            ax.set_xticks(range(data["n_states"]))
            ax.set_xticklabels([bin(i)[2:].zfill(int(np.log2(data["n_states"]))) for i in range(data["n_states"])], rotation=45)
            
            ax.set_xlabel("Genotypes")
            ax.set_ylabel("Actions (Drugs)")
            ax.set_title("Agent's Internal Model of the Landscape")
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Error loading policy snapshot: {e}")


def plot_simplex_section(tab_id):
    """
    Show simplex policy slices for four-state runs.
    Displays the greedy baseline, and the trained RL policy if available.
    """
    sim = st.session_state.sims[tab_id]
    dataset = sim["hp"].get("dataset", "")

    # Only show for four-state datasets
    if DATASET_OPTIONS.get(dataset, {}).get("cli") != "four_state":
        return

    from examples.plotting import plot_simplex_policy_slices, greedy_policy
    from remarc.envs.utils import define_four_state_landscapes

    st.subheader("Simplex Visualization: Greedy Policy")
    st.caption("3D simplex (x₀+x₁+x₂+x₃=1) sliced along x₃. Color shows which drug the policy selects at each population composition.")

    amp = sim["hp"].get("landscape_amplification", 1.0)
    landscapes = define_four_state_landscapes(amplification=amp)

    # --- Greedy baseline ---
    def greedy_fn(state):
        return greedy_policy(state, landscapes)

    fig_greedy = plot_simplex_policy_slices(
        policy_fn=greedy_fn,
        num_drugs=len(landscapes),
        drug_labels=["Drug A", "Drug B", "Drug C", "Drug D"],
        genotype_labels=["genotype 0", "genotype 1", "genotype 2", "genotype 3"],
        resolution=50,
        title="Greedy Policy (minimize immediate fitness)",
    )
    st.pyplot(fig_greedy)
    plt.close(fig_greedy)

    # --- Trained policy (if available) ---
    signature = sim.get("signature")
    if signature:
        mode_arg = MODES[sim["mode"]]
        log_dir = "RL" if mode_arg in ["wf_ls", "wf_ss", "sswm"] else "sswm_dqn"
        policy_path = project_root / "log" / log_dir / f"best_policy_{signature}.pth"
        
        # If the user selected a custom policy via absolute path, use it directly
        if sim.get("selected_policy") and os.path.isabs(sim["selected_policy"]):
            policy_path = Path(sim["selected_policy"])
            
        if policy_path.exists():
            st.subheader("Simplex Visualization: Trained RL Policy")
            try:
                from examples.plotting import _load_ppo_policy_fn
                trained_fn = _load_ppo_policy_fn(str(policy_path), state_dim=4, n_actions=len(landscapes))
                fig_trained = plot_simplex_policy_slices(
                    policy_fn=trained_fn,
                    num_drugs=len(landscapes),
                    drug_labels=["Drug A", "Drug B", "Drug C", "Drug D"],
                    genotype_labels=["genotype 0", "genotype 1", "genotype 2", "genotype 3"],
                    resolution=50,
                    title="Trained RL Policy",
                )
                st.pyplot(fig_trained)
                plt.close(fig_trained)
                
                from examples.plotting import plot_policy_fitness_landscape_slices
                st.subheader("Simplex Visualization: Normalized Fitness (RL Policy)")
                st.caption("Heatmap of normalized fitness under the drug chosen by the RL policy. Lower is better (darker). White dashed lines indicate drug boundaries.")
                fig_fitness = plot_policy_fitness_landscape_slices(
                    policy_fn=trained_fn,
                    landscapes=landscapes,
                    num_drugs=len(landscapes),
                    genotype_labels=["genotype 0", "genotype 1", "genotype 2", "genotype 3"],
                    resolution=50,
                )
                st.pyplot(fig_fitness)
                plt.close(fig_fitness)
                
                # --- SHEPHERD and Difference Plots ---
                if sim["hp"].get("plot_shepherd", True):
                    current_L = sim["hp"].get("shepherd_resolution", 3)
                    if "shepherd_fn" not in sim or sim.get("shepherd_res_cache") != current_L:
                        with st.spinner(f"Solving exact SHEPHERD MDP (L={current_L}) for visualization..."):
                            from remarc.agents.shepherd_eval import ShepherdMDP
                            from remarc.envs.wright_fisher_env import WrightFisherEnv
                            from remarc.core.landscapes import Landscape
                            g_min, g_max = np.min(landscapes), np.max(landscapes)
                            landscape_objs = [Landscape(2, sigma=0.0, ls=ls, g_min=g_min, g_max=g_max) for ls in landscapes]
                            
                            dummy_env = WrightFisherEnv(
                                pop_size=sim["hp"].get("pop_size", 10000),
                                seq_length=2,
                                mutation_rate=sim["hp"].get("mutation_rate", 1e-4),
                                gen_per_step=sim["hp"].get("gen_per_step", 500),
                                num_drugs=len(landscapes),
                                landscape_list=landscape_objs
                            )
                            
                            cache_path = project_root / 'log' / f'shepherd_L{current_L}.npz'
                            shepherd_mdp = ShepherdMDP.from_env(dummy_env, L=current_L, discount=0.99)
                            
                            if cache_path.exists():
                                shepherd_mdp.load(cache_path)
                            else:
                                shepherd_mdp.solve()
                                shepherd_mdp.save(cache_path)
                                
                            def s_fn(state):
                                return shepherd_mdp.get_action(state)
                                
                            sim["shepherd_fn"] = s_fn
                            sim["shepherd_res_cache"] = current_L
                    
                    st.subheader("Simplex Visualization: SHEPHERD Baseline")
                    fig_shepherd = plot_simplex_policy_slices(
                        policy_fn=sim["shepherd_fn"],
                        num_drugs=len(landscapes),
                        drug_labels=["Drug A", "Drug B", "Drug C", "Drug D"],
                        genotype_labels=["genotype 0", "genotype 1", "genotype 2", "genotype 3"],
                        resolution=50,
                        title="SHEPHERD Policy (Exact MDP)",
                    )
                    st.pyplot(fig_shepherd)
                    plt.close(fig_shepherd)
                    
                    from examples.plotting import plot_policy_difference_slices, plot_policy_magnitude_difference_slices
                    
                    st.subheader("Policy Difference: RL vs Greedy")
                    fig_diff1 = plot_policy_difference_slices(
                        policy_fn_1=trained_fn,
                        policy_fn_2=greedy_fn,
                        num_drugs=len(landscapes),
                        resolution=50,
                        title="Disagreement: RL vs Greedy",
                    )
                    st.pyplot(fig_diff1)
                    plt.close(fig_diff1)
                    
                    st.subheader("Magnitude Difference: RL vs Greedy")
                    fig_mag_diff1 = plot_policy_magnitude_difference_slices(
                        policy_fn_1=trained_fn,
                        policy_fn_2=greedy_fn,
                        landscapes=landscapes,
                        num_drugs=len(landscapes),
                        resolution=50,
                        title="Fitness Difference: RL vs Greedy",
                    )
                    st.pyplot(fig_mag_diff1)
                    plt.close(fig_mag_diff1)
                    
                    st.subheader("Policy Difference: RL vs SHEPHERD")
                    fig_diff2 = plot_policy_difference_slices(
                        policy_fn_1=trained_fn,
                        policy_fn_2=sim["shepherd_fn"],
                        num_drugs=len(landscapes),
                        resolution=50,
                        title="Disagreement: RL vs SHEPHERD",
                    )
                    st.pyplot(fig_diff2)
                    plt.close(fig_diff2)
                    
                    st.subheader("Magnitude Difference: RL vs SHEPHERD")
                    fig_mag_diff2 = plot_policy_magnitude_difference_slices(
                        policy_fn_1=trained_fn,
                        policy_fn_2=sim["shepherd_fn"],
                        landscapes=landscapes,
                        num_drugs=len(landscapes),
                        resolution=50,
                        title="Fitness Difference: RL vs SHEPHERD",
                    )
                    st.pyplot(fig_mag_diff2)
                    plt.close(fig_mag_diff2)

            except Exception as e:
                st.warning(f"Could not load trained policy for simplex plot: {e}")
        else:
            st.info("Trained policy not yet available. Will show after training completes.")


def plot_baseline_comparison(tab_id):
    """
    Plot fitness trajectories: Learned vs ALL Single-Drug baselines
    """
    sim = st.session_state.sims[tab_id]
    signature = sim.get("signature")
    if not signature:
        return
    
    baseline_file = project_root / "log" / "baselines" / f"{signature}_baseline.json"
    learned_file = project_root / "log" / "baselines" / f"{signature}_learned.json"
    random_file = project_root / "log" / "baselines" / f"{signature}_random.json"
    
    if not baseline_file.exists() or not learned_file.exists():
        return
    
    try:
        with open(baseline_file) as f:
            baseline_data = json.load(f)
        
        with open(learned_file) as f:
            learned_data = json.load(f)

        random_data = None
        if random_file.exists():
            with open(random_file) as f:
                random_data = json.load(f)
        
        st.subheader("Policy Comparison: Learned vs All Drug Baselines")
        
        # Display summary metrics
        cols = st.columns(4) if random_data else st.columns(3)
        with cols[0]:
            st.metric(
                "Best Single Drug",
                f"Drug #{baseline_data['best_drug']}",
                f"{baseline_data['mean_fitness']:.4f}"
            )
        with cols[1]:
            st.metric(
                "Learned Policy",
                "Multi-Drug Cycling",
                f"{learned_data['mean_fitness']:.4f}"
            )
        
        if random_data:
            with cols[2]:
                st.metric(
                    "Random Policy",
                    "Random Drugs",
                    f"{random_data['mean_fitness']:.4f}"
                )
            with cols[3]:
                improvement = (baseline_data['mean_fitness'] - learned_data['mean_fitness']) / baseline_data['mean_fitness'] * 100
                st.metric(
                    "Improvement vs Best Drug",
                    f"{improvement:.2f}%",
                    delta=f"{improvement:.2f}%",
                    delta_color="inverse" if improvement > 0 else "normal"
                )
        else:
            with cols[2]:
                improvement = (baseline_data['mean_fitness'] - learned_data['mean_fitness']) / baseline_data['mean_fitness'] * 100
                st.metric(
                    "Improvement vs Best Drug",
                    f"{improvement:.2f}%",
                    delta=f"{improvement:.2f}%",
                    delta_color="inverse" if improvement > 0 else "normal"
                )
        
        show_greedy = st.checkbox("Show Greedy Policy Trajectories", value=False)
        
        # Determine data format: new (all_drug_trajectories) or old (trajectories)
        has_all_drugs = 'all_drug_trajectories' in baseline_data
        
        # Plot fitness trajectories
        fig, ax = plt.subplots(figsize=(12, 6))
        
        learned_trajs = learned_data['trajectories']
        random_trajs = random_data['trajectories'] if random_data else []
        shepherd_trajs = baseline_data.get('shepherd_trajectories')
        
        # Normalize trajectories to relative fitness (0 to 1) across all policies
        all_vals = []
        for t in learned_trajs:
            all_vals.extend(t)
        for t in random_trajs:
            all_vals.extend(t)
        if shepherd_trajs:
            for t in shepherd_trajs:
                all_vals.extend(t)
        if has_all_drugs:
            for drug_trajs in baseline_data['all_drug_trajectories'].values():
                for t in drug_trajs:
                    all_vals.extend(t)
        else:
            baseline_trajs = baseline_data.get('trajectories', [])
            for t in baseline_trajs:
                all_vals.extend(t)
                
        if all_vals:
            global_min = min(all_vals)
            global_max = max(all_vals)
            denom = global_max - global_min if global_max != global_min else 1.0
            
            # Map trajectories
            learned_trajs = [[(v - global_min) / denom for v in t] for t in learned_trajs]
            if random_trajs:
                random_trajs = [[(v - global_min) / denom for v in t] for t in random_trajs]
            if shepherd_trajs:
                shepherd_trajs = [[(v - global_min) / denom for v in t] for t in shepherd_trajs]
            
            if has_all_drugs:
                # Create a copy or update baseline_data['all_drug_trajectories']
                baseline_data = dict(baseline_data)
                baseline_data['all_drug_trajectories'] = {
                    k: [[(v - global_min) / denom for v in t] for t in drug_trajs]
                    for k, drug_trajs in baseline_data['all_drug_trajectories'].items()
                }
            else:
                baseline_data = dict(baseline_data)
                if 'trajectories' in baseline_data:
                    baseline_data['trajectories'] = [[(v - global_min) / denom for v in t] for t in baseline_data['trajectories']]
        
        # Compute max_len across all trajectory sources
        all_traj_lengths = [len(t) for t in learned_trajs]
        if random_trajs:
            all_traj_lengths += [len(t) for t in random_trajs]
        if shepherd_trajs:
            all_traj_lengths += [len(t) for t in shepherd_trajs]
        
        if has_all_drugs:
            for drug_trajs in baseline_data['all_drug_trajectories'].values():
                all_traj_lengths += [len(t) for t in drug_trajs]
        
        max_len = max(all_traj_lengths) if all_traj_lengths else 0
        if max_len == 0:
            st.warning("No trajectory data available")
            return
        
        timesteps = np.arange(max_len)
        
        # Learned policy
        learned_padded = np.array([t + [np.nan] * (max_len - len(t)) for t in learned_trajs])
        learned_mean = np.nanmean(learned_padded, axis=0)
        learned_std = np.nanstd(learned_padded, axis=0)
        ax.plot(timesteps, learned_mean, label='Learned Policy', color='blue', linewidth=2.5)
        ax.fill_between(timesteps, learned_mean - learned_std, learned_mean + learned_std, alpha=0.15, color='blue')
        
        # Greedy policy
        if show_greedy:
            if "greedy_trajs" not in sim or len(sim["greedy_trajs"][0]) < max_len:
                with st.spinner("Simulating Greedy Policy..."):
                    from remarc.envs.wright_fisher_env import WrightFisherEnv
                    from remarc.core.landscapes import Landscape
                    from examples.plotting import greedy_policy
                    
                    dataset = sim["hp"].get("dataset", "")
                    env_setup_ok = False
                    
                    if DATASET_OPTIONS.get(dataset, {}).get("cli") == "four_state":
                        from remarc.envs.utils import define_four_state_landscapes
                        amp = sim["hp"].get("landscape_amplification", 1.0)
                        landscapes_array = define_four_state_landscapes(amplification=amp)
                        g_min_val, g_max_val = np.min(landscapes_array), np.max(landscapes_array)
                        landscape_objs = [Landscape(2, sigma=0.0, ls=ls, g_min=g_min_val, g_max=g_max_val) for ls in landscapes_array]
                        
                        env_greedy = WrightFisherEnv(
                            pop_size=sim["hp"].get("pop_size", 10000),
                            seq_length=2,
                            mutation_rate=sim["hp"].get("mutation_rate", 1e-4),
                            gen_per_step=sim["hp"].get("gen_per_step", 500),
                            num_drugs=len(landscapes_array),
                            landscape_list=landscape_objs,
                            random_start=sim["hp"].get("random_start", True),
                            stochastic=sim["hp"].get("stochastic", True)
                        )
                        env_setup_ok = True
                    elif DATASET_OPTIONS.get(dataset, {}).get("cli") == "chen":
                        from remarc.envs.utils import define_chen_landscapes
                        landscapes_array = define_chen_landscapes()
                        g_min_val, g_max_val = np.min(landscapes_array), np.max(landscapes_array)
                        landscape_objs = [Landscape(3, sigma=0.0, ls=ls, g_min=g_min_val, g_max=g_max_val) for ls in landscapes_array]
                        
                        env_greedy = WrightFisherEnv(
                            pop_size=sim["hp"].get("pop_size", 10000),
                            seq_length=3,
                            mutation_rate=sim["hp"].get("mutation_rate", 1e-4),
                            gen_per_step=sim["hp"].get("gen_per_step", 500),
                            num_drugs=len(landscapes_array),
                            landscape_list=landscape_objs,
                            random_start=sim["hp"].get("random_start", True),
                            stochastic=sim["hp"].get("stochastic", True)
                        )
                        env_setup_ok = True
                    
                    if env_setup_ok:
                        greedy_trajs_raw = []
                        num_ep_greedy = sim["hp"].get("test_episodes", 100)
                        ep_len_greedy = max_len
                        
                        for _ in range(num_ep_greedy):
                            obs, _ = env_greedy.reset()
                            ep_fitnesses = []
                            for _ in range(ep_len_greedy):
                                action = greedy_policy(obs, landscapes_array)
                                obs, _, done, truncated, _ = env_greedy.step(action)
                                ep_fitnesses.append(env_greedy.get_fitness(raw=True))
                            greedy_trajs_raw.append(ep_fitnesses)
                        
                        # Normalize fitness values just like other trajectories
                        greedy_trajs_norm = [[(v - global_min) / denom for v in t] for t in greedy_trajs_raw]
                        sim["greedy_trajs"] = greedy_trajs_norm
            
            if "greedy_trajs" in sim and sim["greedy_trajs"]:
                greedy_trajs = sim["greedy_trajs"]
                # Ensure we don't crash if length mismatch
                greedy_padded = np.array([t + [np.nan] * max(0, max_len - len(t)) for t in greedy_trajs])
                # Slice to max_len if it's somehow longer
                greedy_padded = greedy_padded[:, :max_len]
                greedy_mean = np.nanmean(greedy_padded, axis=0)
                greedy_std = np.nanstd(greedy_padded, axis=0)
                
                # Plot
                plot_len = min(len(timesteps), len(greedy_mean))
                ax.plot(timesteps[:plot_len], greedy_mean[:plot_len], label='Greedy Policy', color='#f1c40f', linewidth=2.5, linestyle='--', zorder=10)
                ax.fill_between(timesteps[:plot_len], (greedy_mean - greedy_std)[:plot_len], (greedy_mean + greedy_std)[:plot_len], alpha=0.2, color='#f1c40f', zorder=9)
        
        # All single-drug baselines
        drug_colors = ['#e67e22', '#27ae60', '#8e44ad', '#c0392b', '#2980b9', '#f39c12', '#1abc9c', '#d35400', '#7f8c8d', '#2c3e50']
        best_drug = baseline_data.get('best_drug', -1)
        
        if has_all_drugs:
            for drug_key, drug_trajs in baseline_data['all_drug_trajectories'].items():
                drug_idx = int(drug_key)
                color = drug_colors[drug_idx % len(drug_colors)]
                padded = np.array([t + [np.nan] * (max_len - len(t)) for t in drug_trajs])
                d_mean = np.nanmean(padded, axis=0)
                d_std = np.nanstd(padded, axis=0)
                
                is_best = (drug_idx == best_drug)
                label = f'Drug {drug_idx}' + (' ★' if is_best else '')
                lw = 2.0 if is_best else 1.2
                alpha_line = 1.0 if is_best else 0.7
                
                ax.plot(timesteps, d_mean, label=label, color=color, linewidth=lw, linestyle='--', alpha=alpha_line)
                ax.fill_between(timesteps, d_mean - d_std, d_mean + d_std, alpha=0.05, color=color)
        else:
            # Backward compat: old format with only best drug's trajectories
            baseline_trajs = baseline_data.get('trajectories', [])
            if baseline_trajs:
                padded = np.array([t + [np.nan] * (max_len - len(t)) for t in baseline_trajs])
                b_mean = np.nanmean(padded, axis=0)
                b_std = np.nanstd(padded, axis=0)
                ax.plot(timesteps, b_mean, label=f'Best Single Drug (#{best_drug})', color='orange', linewidth=2, linestyle='--')
                ax.fill_between(timesteps, b_mean - b_std, b_mean + b_std, alpha=0.15, color='orange')
        
        # Random policy
        if random_data and random_trajs:
            random_padded = np.array([t + [np.nan] * (max_len - len(t)) for t in random_trajs])
            random_mean = np.nanmean(random_padded, axis=0)
            random_std = np.nanstd(random_padded, axis=0)
            ax.plot(timesteps, random_mean, label='Random Policy', color='red', linewidth=2, linestyle=':')
            ax.fill_between(timesteps, random_mean - random_std, random_mean + random_std, alpha=0.1, color='red')
            
        # SHEPHERD policy
        if shepherd_trajs:
            shepherd_padded = np.array([t + [np.nan] * (max_len - len(t)) for t in shepherd_trajs])
            shepherd_mean = np.nanmean(shepherd_padded, axis=0)
            shepherd_std = np.nanstd(shepherd_padded, axis=0)
            ax.plot(timesteps, shepherd_mean, label='SHEPHERD Policy', color='black', linewidth=2, linestyle='-.')
            ax.fill_between(timesteps, shepherd_mean - shepherd_std, shepherd_mean + shepherd_std, alpha=0.1, color='black')
        
        ax.set_xlabel('Timestep')
        ax.set_ylabel('relative fitness')
        ax.set_title('Fitness Trajectory Comparison (Mean ± Std)')
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=0.3)
        
        st.pyplot(fig)
        
        st.caption("💡 Lower fitness indicates better drug efficacy. Learned policy should show lower fitness (more effective) than single-drug baselines.")
        
        # Population Density Plot
        if 'state_trajectories' in learned_data and learned_data['state_trajectories']:
            from examples.plotting import plot_population_density_slices
            st.subheader("Population Density (Learned Policy)")
            try:
                dataset = sim["hp"].get("dataset", "")
                if DATASET_OPTIONS.get(dataset, {}).get("cli") == "four_state":
                    from remarc.envs.utils import define_four_state_landscapes
                    amp = sim["hp"].get("landscape_amplification", 1.0)
                    landscapes = define_four_state_landscapes(amplification=amp)
                elif DATASET_OPTIONS.get(dataset, {}).get("cli") == "chen":
                    from remarc.envs.utils import define_chen_landscapes
                    landscapes = define_chen_landscapes()
                else:
                    landscapes = sim.get("landscapes", [])
                
                num_drugs = len(landscapes) if len(landscapes) > 0 else 4
                
                trained_fn = None
                signature = sim.get("signature")
                
                # Check for absolute policy path directly if signature is missing
                policy_path = None
                if sim.get("selected_policy") and os.path.isabs(sim["selected_policy"]):
                    policy_path = Path(sim["selected_policy"])
                elif signature:
                    from examples.plotting import _load_ppo_policy_fn
                    mode_arg = MODES[sim["mode"]]
                    log_dir = "RL" if mode_arg in ["wf_ls", "wf_ss", "sswm"] else "sswm_dqn"
                    policy_path = project_root / "log" / log_dir / f"best_policy_{signature}.pth"
                
                if policy_path and policy_path.exists():
                    from examples.plotting import _load_ppo_policy_fn
                    trained_fn = _load_ppo_policy_fn(str(policy_path), state_dim=4, n_actions=num_drugs)
                else:
                    st.warning(f"Policy file not found or signature missing. Signature: {signature}, Path: {policy_path}")
                
                greedy_fn = None
                if len(landscapes) > 0:
                    from examples.plotting import greedy_policy
                    def g_fn(state):
                        return greedy_policy(state, landscapes)
                    greedy_fn = g_fn
                
                
                fig_density = plot_population_density_slices(
                    state_trajectories=learned_data['state_trajectories'],
                    policy_fn=trained_fn,
                    greedy_policy_fn=greedy_fn,
                    num_drugs=num_drugs,
                    resolution=50,
                )
                st.pyplot(fig_density)
                plt.close(fig_density)
                st.caption("Heatmap of population state frequency during evaluation episodes, binned by small triangles.")
            except Exception as e:
                import traceback
                st.warning(f"Could not generate population density plot: {e}\n\n```python\n{traceback.format_exc()}\n```")
        else:
            st.info("Population density data not found in baseline JSON. Re-run evaluation to generate it.")
            
    except Exception as e:
        st.error(f"Error loading baseline comparison: {e}")



def plot_training_progress(tab_id):
    sim = st.session_state.sims[tab_id]
    signature = sim.get("signature")
    if not signature and not sim["train"] and sim.get("selected_policy"):
        # Try to extract signature from policy filename
        # best_policy_sswm_testing.pth -> testing
        fname = sim["selected_policy"]
        if "sswm_" in fname:
            signature = fname.split("sswm_")[-1].replace(".pth", "")
        elif "policy_" in fname:
            signature = fname.split("policy_")[-1].replace(".pth", "")

    if not signature:
        st.info("No signature available to load training metrics.")
        return

    metrics_file = project_root / "log" / "metrics" / f"{signature}.csv"
    corr_file = project_root / "log" / "metrics" / f"{signature}_correlation.csv"
    
    # Rewards plot
    if metrics_file.exists():
        try:
            df = pd.read_csv(metrics_file)
            if not df.empty:
                st.subheader("Training Progress")
                
                # Reward plot
                plot_df = pd.DataFrame({
                    "Epoch": df["epoch"],
                    "Mean Reward": df["mean_reward"],
                    "Upper Bound": df["mean_reward"] + df["std_reward"],
                    "Lower Bound": df["mean_reward"] - df["std_reward"]
                }).set_index("Epoch")
                
                st.line_chart(plot_df, color=["#1f77b4", "#aec7e8", "#aec7e8"])
                st.caption("Blue: Mean Reward | Shaded: ±1 Std Dev")
                
                # Loss plot
                if "loss" in df.columns and df["loss"].notna().any():
                    st.subheader("Training Loss")
                    #Drop the first 5 values
                    loss_df = df[["epoch", "loss"]].drop(1).dropna().set_index("epoch")
                    if not loss_df.empty:
                        st.line_chart(loss_df, color="#ff7f0e")
                        st.caption("Orange: Training Loss (lower is better)")
            else:
                st.info("Metrics file is empty.")
        except Exception as e:
            st.error(f"Error loading metrics: {e}")
    else:
        if sim["running"]:
            st.info("Waiting for first training metrics...")
        else:
            st.info(f"No metrics found for signature: {signature}")
    



@st.dialog("📜 Execution Logs", width="large")
def show_logs(tab_id):
    # Nested fragment to update logs without closing the dialog
    @st.fragment(run_every=2) # Reduced from 1s to 2s
    def log_viewer():
        # Update logs from process if running
        update_logs_for_sim(tab_id)
        sim = st.session_state.sims[tab_id]
        st.code(sim["logs"] if sim["logs"] else "No logs available.", language="text")
        if sim["running"]:
            st.caption("🔄 Auto-updating logs...")
    log_viewer()

def render_status_logic(tab_id):
    @st.fragment(run_every=5) # Reduced from 2s to 5s for plotting heavy content
    def status_fragment():
        update_logs_for_sim(tab_id)
        sim = st.session_state.sims[tab_id]
        
        # Status indicator at top
        st.divider()
        if sim["running"]:
            st.success(f"🔄 Running (PID: {sim['process'].pid})")
        elif sim["exit_code"] is not None:
            if sim["exit_code"] == 0: st.success("✅ Finished Successfully")
            else: st.error(f"❌ Failed (Exit Code: {sim['exit_code']})")
        else:
            st.info("⏸️ Idle - Configure and click RUN to start")
        
        st.divider()
        
        # Visualization Section

        if sim["train"]:
            plot_live_policy(tab_id)
            st.divider()
            plot_training_progress(tab_id)
            st.divider()
            plot_baseline_comparison(tab_id)
            st.divider()
            plot_simplex_section(tab_id)
            st.divider()
        
        if not sim["train"]:
            plot_baseline_comparison(tab_id)
            st.divider()
            plot_simplex_section(tab_id)
        st.divider()
    status_fragment()

# Header
st.title("🔬 REMARC Playground")
st.markdown("Tinker with Evolutionary Dynamics and Reinforcement Learning right in your browser.")

# UI Layout
def get_tab_label(tab_id):
    sim = st.session_state.sims[tab_id]
    label = f"Simulation {tab_id+1}"
    if sim["running"]:
        return f":orange[{label}]"
    elif sim["exit_code"] is not None:
        if sim["exit_code"] == 0:
            return f":green[{label}]"
        else:
            return f":red[{label}]"
    else:
        return f":gray[{label}]"

tab_labels = [get_tab_label(i) for i in range(NUM_TABS)]
# Use st.radio styled as tabs to prevent Streamlit from losing the active tab state when labels change
st.markdown("<style>.stRadio > div { flex-direction: row; }</style>", unsafe_allow_html=True)
active_tab = st.radio(
    "Select Simulation",
    options=list(range(NUM_TABS)),
    format_func=get_tab_label,
    horizontal=True,
    label_visibility="collapsed",
    key="active_sim_tab"
)

def get_auto_signature(hp):
    ds_cli = DATASET_OPTIONS[hp["dataset"]]["cli"]
    if ds_cli == "four_state": ds = "fourstate"
    elif ds_cli == "chen": ds = "chen"
    else: ds = f"synthetic_N{hp['n_mut']}"
    
    def fmt_e(val):
        if val == 0: return "0"
        s = f"{val:.1e}"
        return s.replace(".0e", "e").replace("e-0", "e-").replace("e+0", "e+")

    def fmt_f(val):
        if isinstance(val, float):
            return f"{val:.4f}".rstrip('0').rstrip('.')
        return str(val)

    lr = f"{fmt_e(hp['lr'])}LR"
    mr = f"{fmt_e(hp['mutation_rate'])}MR"
    gps = f"{hp['gen_per_step']}gps"
    steps = f"{hp.get('episode_steps', 20)}st"
    batch = f"{hp['batch_size']}b"
    epochs = f"{hp['epochs']}ep"
    
    pop = f"{hp.get('pop_size', 10000)}pop"
    ent = f"{fmt_f(hp.get('ent_coef', 0.05))}ent"
    gam = f"{fmt_f(hp.get('gamma', 0.99))}gam"
    gae = f"{fmt_f(hp.get('gae_lambda', 0.95))}gae"
    rsc = f"{fmt_f(hp.get('reward_scale', 100.0))}rsc"
    
    rs = "randomstart" if hp.get("random_start", True) else "norandomstart"
    
    return f"{ds}_{lr}_{mr}_{gps}_{steps}_{batch}_{epochs}_{pop}_{ent}_{gam}_{gae}_{rsc}_{rs}"

def render_tab_content(tab_id):
    sim = st.session_state.sims[tab_id]
    
    # Top Control Bar
    with st.container():
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        cols_top = st.columns([1.5, 1, 1, 1, 1, 1, 1, 1])
        
        with cols_top[0]:
            sim["mode"] = st.selectbox(
                "Evolutionary Regime", 
                list(MODES.keys()), 
                index=list(MODES.keys()).index(sim["mode"]),
                key=f"mode_sel_{tab_id}"
            )
        with cols_top[1]:
            sim["hp"]["lr"] = st.number_input(
                "Learning rate", 
                min_value=1e-8, max_value=1.0, 
                value=sim["hp"].get("lr", 0.0001),
                format="%.1e",
                key=f"lr_{tab_id}"
            )
        with cols_top[2]:
            sim["hp"]["epochs"] = st.number_input("Epochs", 1, 1000, sim["hp"]["epochs"], key=f"epochs_{tab_id}")
        with cols_top[3]:
            sim["hp"]["batch_size"] = st.selectbox("Batch size", [16, 32, 64, 128, 256], index=[16, 32, 64, 128, 256].index(sim["hp"].get("batch_size", 128)), key=f"batch_{tab_id}")
            
        with cols_top[4]:
             can_run = not sim["train"] or (sim["train"] and sim.get("signature", "").strip() != "")
             any_running = any(s["running"] for s in st.session_state.sims.values())
             is_queued = tab_id in st.session_state.run_queue
             
             if not sim["running"] and not is_queued:
                btn_text = "🚀 ENQUEUE" if any_running or time.time() < st.session_state.cooldown_until else "🚀 RUN"
                if st.button(btn_text, key=f"run_btn_{tab_id}", use_container_width=True, type="primary", disabled=not can_run):
                    if any_running or time.time() < st.session_state.cooldown_until:
                        st.session_state.run_queue.append(tab_id)
                        st.toast(f"Simulation {tab_id+1} added to queue", icon="📥")
                    else:
                        start_simulation(tab_id)
                    st.rerun()
             elif is_queued:
                if st.button("❌ DEQUEUE", key=f"dequeue_btn_{tab_id}", use_container_width=True, type="secondary"):
                    st.session_state.run_queue.remove(tab_id)
                    st.rerun()
             else:
                if st.button("🛑 STOP", key=f"stop_btn_{tab_id}", use_container_width=True, type="secondary"):
                    if sim["process"]:
                        sim["process"].terminate()
                    st.rerun()
        
        with cols_top[5]:
            if st.button("📜 LOGS", key=f"logs_btn_{tab_id}", use_container_width=True):
                show_logs(tab_id)

        with cols_top[6]:
            # Add Star Run toggle
            has_sig = sim.get("signature") and sim["exit_code"] == 0
            if has_sig:
                sim["starred"] = st.toggle("⭐ Star Run", value=sim.get("starred", False), key=f"star_run_{tab_id}")
            else:
                st.toggle("⭐ Star Run", disabled=True, key=f"star_run_{tab_id}")

        with cols_top[7]:
            if st.button("🗑️ CLEAR", key=f"clear_tab_btn_{tab_id}", use_container_width=True):
                # Archive the old run if it wasn't starred
                if sim.get("signature") and not sim.get("starred", False):
                    archive_run(sim["signature"], False)
                sim["logs"] = ""
                sim["starred"] = False
                st.rerun()
                
        # Copy Settings
        with st.expander("📋 Copy Settings from another Tab"):
            copy_cols = st.columns([2, 1, 3])
            with copy_cols[0]:
                copy_source = st.selectbox("Source Tab", [f"Sim {i+1}" for i in range(NUM_TABS) if i != tab_id], key=f"copy_source_{tab_id}")
            
            with copy_cols[1]:
                st.button("Copy Settings", key=f"copy_btn_{tab_id}", use_container_width=True, on_click=copy_settings_callback, args=(tab_id,))
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Main Area
    col_env, col_regime = st.columns([1, 1])
    
    with col_env:
        st.subheader("Landscape Config")
        sim["hp"]["n_mut"] = st.slider("Number of mutations (N)", 1, 10, sim["hp"]["n_mut"], key=f"n_mut_{tab_id}")
        
        # Only show Sigma for Synthetic datasets
        is_synthetic = sim["hp"]["dataset"] == "Synthetic"
        if is_synthetic:
            sim["hp"]["sigma"] = st.slider("Sigma (Noise)", 0.0, 1.0, sim["hp"]["sigma"], 0.1, key=f"sigma_{tab_id}")
        else:
            st.info("💡 Sigma (Noise) is fixed to 0.0 for empirical datasets.")
            sim["hp"]["sigma"] = 0.0

        sim["train"] = st.checkbox("Enable Training", value=sim["train"], key=f"train_cb_{tab_id}")
        
        if sim["train"]:
            sig_placeholder = st.empty()
            caption_placeholder = st.empty()
        else:
             mode_arg = MODES[sim["mode"]]
             log_dir = "RL" if mode_arg in ["wf_ls", "wf_ss", "sswm"] else "sswm_dqn"
             path = project_root / "log" / log_dir
             policies = sorted([f.name for f in path.glob("*.pth")]) if path.exists() else []
             
             policy_choice = None
             if policies:
                 policy_options = ["--- Custom Path ---"] + policies
                 default_idx = 0
                 if sim.get("selected_policy") in policies:
                     default_idx = policy_options.index(sim.get("selected_policy"))
                 
                 selected_from_list = st.selectbox("Select Policy", policy_options, index=default_idx, key=f"policy_sel_{tab_id}")
                 if selected_from_list != "--- Custom Path ---":
                     policy_choice = selected_from_list
                     
             custom_policy_path = st.text_input(
                 "Or enter Custom Policy Path",
                 value=sim.get("selected_policy", "") if sim.get("selected_policy", "") not in policies else "",
                 key=f"policy_custom_{tab_id}",
                 help="Enter either a filename in the log directory, or a full absolute path to a .pth policy file."
             )
             
             if custom_policy_path:
                 sim["selected_policy"] = custom_policy_path
             elif policy_choice:
                 sim["selected_policy"] = policy_choice
             else:
                 sim["selected_policy"] = ""

             sim["signature"] = ""
             if sim["selected_policy"]:
                 import os
                 fname = os.path.basename(sim["selected_policy"])
                 if "sswm_" in fname:
                     sim["signature"] = fname.split("sswm_")[-1].replace(".pth", "")
                 elif "policy_" in fname:
                     sim["signature"] = fname.split("policy_")[-1].replace(".pth", "")
                 else:
                     sim["signature"] = fname.replace(".pth", "")

    with col_regime:
        st.subheader("Dataset & Regime")
        
        # Dataset Selection with Filtering Logic
        available_datasets = [name for name, meta in DATASET_OPTIONS.items() if meta["N"] == "any" or meta["N"] == sim["hp"]["n_mut"]]
        
        dataset_key = str(sim["hp"]["dataset"])
        sim["hp"]["dataset"] = st.selectbox(
            "Select Dataset", 
            available_datasets, 
            index=0 if dataset_key not in available_datasets else available_datasets.index(dataset_key),
            key=f"dataset_sel_{tab_id}",
            help=str(DATASET_OPTIONS[dataset_key]["description"]) if dataset_key in DATASET_OPTIONS else ""
        )
        

        
        if sim["hp"]["dataset"] == "Chen et al." and sim["hp"]["n_mut"] != 3:
            st.warning("Chen dataset is strictly for N=3. Please adjust the mutation slider.")

        if sim["hp"]["dataset"] == "Four-State" and sim["hp"]["n_mut"] != 2:
            st.warning("Four-State dataset is strictly for N=2. Please adjust the mutation slider.")
            
        sim["hp"]["activation"] = st.selectbox(
            "Activation Function", 
            ["relu", "tanh", "swish", "sigmoid", "leaky_relu", "elu", "gelu"], 
            index=["relu", "tanh", "swish", "sigmoid", "leaky_relu", "elu", "gelu"].index(sim["hp"]["activation"]),
            key=f"act_{tab_id}"
        )

        sim["hp"]["reward_clip"] = st.checkbox("Enable Reward Clipping", value=sim["hp"].get("reward_clip", False), key=f"clip_{tab_id}", help="Clip rewards to [-5, 5] for stability")
            
        if MODES[sim["mode"]] in ["wf_ls", "wf_ss"]:
            sim["hp"]["pop_size"] = st.number_input("Population Size", 100, 1000000, sim["hp"]["pop_size"], key=f"pop_{tab_id}")
            sim["hp"]["mutation_rate"] = st.number_input("Mutation Rate", 0.0, 1.0, sim["hp"]["mutation_rate"], format="%.1e", key=f"mut_{tab_id}")
            sim["hp"]["gen_per_step"] = st.number_input("Gens per Step", 1, 10000, sim["hp"]["gen_per_step"], key=f"gps_{tab_id}")
            sim["hp"]["ent_coef"] = st.number_input("Entropy Coef.", 0.0, 1.0, sim["hp"].get("ent_coef", 0.05), step=0.01, format="%.4f", key=f"ent_{tab_id}", help="Higher = more exploration")
            sim["hp"]["gamma"] = st.number_input("Discount Factor (gamma)", 0.0, 1.0, sim["hp"].get("gamma", 0.99), step=0.01, format="%.4f", key=f"gamma_{tab_id}", help="Discount factor. Higher = longer horizon")
            sim["hp"]["gae_lambda"] = st.number_input("GAE Lambda", 0.0, 1.0, sim["hp"].get("gae_lambda", 0.95), step=0.01, format="%.4f", key=f"gae_{tab_id}", help="GAE lambda. Higher = higher variance, lower bias")
            sim["hp"]["delta_multiplier"] = st.number_input("Delta Bonus Multiplier", 0.0, 10.0, sim["hp"].get("delta_multiplier", 0.0), step=0.1, key=f"delta_{tab_id}", help="Rewards immediate fitness drops. Set to 0.0 to allow RL to sacrifice short-term fitness for long-term payoffs (avoids greedy trap).")
            sim["hp"]["episode_steps"] = st.number_input("Episode Steps", 1, 1000000, sim["hp"].get("episode_steps", 20), key=f"ep_steps_{tab_id}", help="Number of steps per episode")
            sim["hp"]["reward_scale"] = st.number_input("Reward Scale", 0.1, 10000.0, sim["hp"].get("reward_scale", 100.0), step=10.0, key=f"reward_scale_{tab_id}", help="Scale for rewards (recommended 100 for WF)")
            sim["hp"]["random_start"] = st.checkbox("Random Start", value=sim["hp"].get("random_start", True), key=f"rstart_{tab_id}", help="Start each episode from a random genotype instead of all-zeros")
            sim["hp"]["stochastic"] = st.checkbox("Stochastic (Multinomial Sampling)", value=sim["hp"].get("stochastic", True), key=f"stochastic_{tab_id}", help="ON = Wright-Fisher with genetic drift (multinomial sampling). OFF = Fokker-Planck deterministic mode (faster, no drift noise).")
            
            st.markdown("##### Testing Parameters")
            sim["hp"]["test_episodes"] = st.number_input("Test Episodes", 1, 10000, sim["hp"].get("test_episodes", 100), key=f"te_{tab_id}")
            sim["hp"]["test_episode_length"] = st.number_input("Test Episode Length", 1, 10000, sim["hp"].get("test_episode_length", 100), key=f"tel_{tab_id}")
            sim["hp"]["plot_shepherd"] = st.checkbox("Evaluate SHEPHERD Baseline (N<=3)", value=sim["hp"].get("plot_shepherd", False), key=f"shep_{tab_id}", help="Disable to save computation time")
            if sim["hp"]["plot_shepherd"]:
                sim["hp"]["shepherd_resolution"] = st.number_input("SHEPHERD Lattice Resolution (L)", min_value=2, max_value=20, value=sim["hp"].get("shepherd_resolution", 3), key=f"shep_res_{tab_id}", help="Number of grid points per axis in u-space. Default is 3.")

            if sim["hp"]["dataset"] == "Four-State":
                sim["hp"]["landscape_amplification"] = st.number_input(
                    "Landscape Amplification", 1.0, 50.0,
                    sim["hp"].get("landscape_amplification", 1.0),
                    step=1.0, key=f"amp_{tab_id}",
                    help="Amplify fitness deviations from mean. 1.0 = raw values (~1%% selection). 10.0 = ~10%% selection pressure."
                )
            else:
                sim["hp"]["landscape_amplification"] = 1.0
        else:
            st.info("No specific parameters for this regime.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    current_auto_sig = get_auto_signature(sim["hp"])
    
    if sim.get("last_auto_sig") != current_auto_sig:
        sim["signature"] = current_auto_sig
        sim["last_auto_sig"] = current_auto_sig
        # Force Streamlit to update the widget's internal state
        st.session_state[f"sig_input_{tab_id}"] = current_auto_sig
        
    if sim["train"]:
        new_sig = sig_placeholder.text_input("Run Signature", value=sim["signature"], key=f"sig_input_{tab_id}")
        if new_sig != sim["signature"]:
            sim["signature"] = new_sig
            
        if not sim["signature"]:
            caption_placeholder.caption("⚠️ :red[Run Signature is mandatory for training]")

    # Status indicator
    render_status_logic(tab_id)
    
    # Visualizations removed from here - they're now in render_status_logic



# Render only the active tab's content
render_tab_content(active_tab)

# --- Queue Daemon & Sidebar ---
with st.sidebar:
    st.header("⏱️ Queue Manager")
    st.session_state.cooldown_seconds = st.number_input("Cooldown (seconds)", min_value=0, value=st.session_state.cooldown_seconds, help="Wait time between simulations to prevent overheating.")
    
    if st.session_state.cooldown_until > time.time():
        rem = int(st.session_state.cooldown_until - time.time())
        st.warning(f"Cooldown active: {rem}s remaining")
    
    st.subheader("Current Queue")
    if not st.session_state.run_queue:
        st.info("Queue is empty.")
    else:
        for i, q_tab in enumerate(st.session_state.run_queue):
            st.write(f"{i+1}. Simulation {q_tab+1}")
        if st.button("Clear Queue", use_container_width=True):
            st.session_state.run_queue = []
            st.rerun()
            
    st.subheader("Batch Operations")
    if st.button("Queue All Configured", use_container_width=True, type="primary"):
        for i in range(NUM_TABS):
            s = st.session_state.sims[i]
            can_run = not s["train"] or (s["train"] and s.get("signature", "").strip() != "")
            if can_run and not s["running"] and i not in st.session_state.run_queue:
                st.session_state.run_queue.append(i)
        st.rerun()

@st.fragment(run_every=5)
def queue_daemon():
    # Globally poll running processes so background tasks finish properly 
    # even when they are not the active tab
    for i, s in st.session_state.sims.items():
        if s["running"]:
            update_logs_for_sim(i)
            
    if not st.session_state.run_queue:
        return
        
    any_running = any(s["running"] for s in st.session_state.sims.values())
    if any_running:
        return
        
    if time.time() < st.session_state.cooldown_until:
        return # Still in cooldown
        
    # Ready to run next!
    next_tab = st.session_state.run_queue.pop(0)
    start_simulation(next_tab)
    st.rerun()

queue_daemon()

st.divider()
st.caption("REMARC - Reinforcement-learning based Evolutionary Markovian Resistance Control | support for concurrent executions")
