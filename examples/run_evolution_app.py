"""
Streamlit interface for wf_tumor_evolution.
Allows selecting a trained PPO policy model and renders/displays the resulting interactive 3D evolution HTML.
"""

import os
import sys
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from examples.frequency_evolution_animation import create_frequency_gif

st.set_page_config(layout="wide", page_title="REMARC Tumor Evolution Animation")

st.title("🔬 REMARC Tumor Evolution Interactive Player")
st.markdown(
    "Choose a policy to run the Wright-Fisher evolutionary simulations, generate the 3D fitness landscape trajectory, and interact with the animated result."
)

# Policy directories and lookup
rl_dir = _project_root / "log" / "RL"
pth_files = sorted([f.name for f in rl_dir.glob("*.pth")]) if rl_dir.exists() else []

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Simulation Options")

    # Policy selector
    policy_choice = None
    if pth_files:
        selected_from_list = st.selectbox(
            "Select Trained Policy (.pth)",
            ["--- Custom Path ---"] + pth_files,
            index=1
            if "best_policy_chen_40gps_100st_1000rewsc_64b_100ep_e-3MR.pth" in pth_files
            else 0,
        )
        if selected_from_list != "--- Custom Path ---":
            policy_choice = selected_from_list

    custom_policy = st.text_input(
        "Or enter Custom/Absolute Policy Path",
        value="" if policy_choice else "best_policy_chen.pth",
        help="Full absolute path or name of the policy file",
    )

    chosen_policy = custom_policy if custom_policy else policy_choice

    # Other simulation hyperparameters
    steps = st.number_input("Episode Steps", min_value=1, max_value=500, value=100)
    pop_size = st.number_input(
        "Population Size", min_value=100, max_value=1000000, value=10000, step=1000
    )
    gen_per_step = st.number_input(
        "Generations per Step", min_value=1, max_value=5000, value=50, step=50
    )

with col2:
    st.subheader("Run Simulation")
    st.write(
        "Clicking run will execute the Wright-Fisher simulation across 3 policies: your chosen Optimal Policy, a Random Policy, and the Best Single Drug policy."
    )

    run_btn = st.button(
        "🚀 Run & Render Animation", use_container_width=True, type="primary"
    )

if run_btn and chosen_policy:
    # Resolve absolute path or standard log file name
    final_policy_path = chosen_policy
    if not os.path.isabs(final_policy_path):
        candidate = rl_dir / final_policy_path
        if candidate.exists():
            final_policy_path = str(candidate)

    st.info(f"Simulating using policy: `{final_policy_path}`...")

    output_html_name = "wf_tumor_evolution_chen.html"
    output_path = _project_root / output_html_name

    try:
        with st.spinner("Running Wright-Fisher simulations..."):
            create_frequency_gif(
                episode_steps=steps,
                policy_filename=final_policy_path,
                output_filename=str(output_path),
                pop_size=pop_size,
                gen_per_step=gen_per_step,
            )
        st.success("Simulation complete! Rendered 3D Plotly evolution player below.")
    except Exception as e:
        st.error(f"Error during simulation: {e}")

# Display HTML if it exists
output_html_path = _project_root / "wf_tumor_evolution_chen.html"
if output_html_path.exists():
    with open(output_html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    st.markdown("### Interactive 3D Evolution Player")
    st.caption(
        "You can toggle between policies using the buttons in the player below, play/pause the trajectory, and rotate the 3D surface."
    )
    components.html(html_content, height=950, scrolling=True)
else:
    st.info(
        "No generated simulation player found. Click run above to generate and display the interactive interface."
    )
