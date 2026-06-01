import streamlit as st
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys
import os

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from evodm.envs import WrightFisherEnv
from evodm.envs.helpers import define_chen_landscapes
from evodm.core.landscapes import Landscape
from evodm.core.hyperparameters import Presets

# ---------------------------------------------------------------------------
# Constants & Colors
# ---------------------------------------------------------------------------

DRUG_NAMES = ['Drug_A', 'Drug_B', 'Drug_C', 'Drug_D']
NUM_DRUGS = 4
NUM_GENOTYPES = 8
GENOTYPE_LABELS = [bin(i)[2:].zfill(3) for i in range(NUM_GENOTYPES)]

GLOBAL_FIT_MIN = 0.98
GLOBAL_FIT_MAX = 1.02
MAX_MARKER_SIZE = 45
MIN_MARKER_SIZE = 5

DRUG_FILL_COLORS = [
    "rgba(99,110,250,0.22)", "rgba(239,85,59,0.22)", "rgba(0,204,150,0.22)", "rgba(171,99,250,0.22)"
]

DRUG_SOLID_COLORS = [
    "rgba(99,110,250,0.90)", "rgba(239,85,59,0.90)", "rgba(0,204,150,0.90)", "rgba(171,99,250,0.90)"
]

# ---------------------------------------------------------------------------
# Streamlit Page Config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Interactive Tumor Evolution", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; font-weight: bold; }
    .stRadio>div { color: white; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------

if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.ls = define_chen_landscapes()
    v_N = 3
    landscape_list = [Landscape(v_N, sigma=0.0, ls=st.session_state.ls[i]) for i in range(len(st.session_state.ls))]
    
    st.session_state.env = WrightFisherEnv(
        num_drugs=NUM_DRUGS, 
        seq_length=v_N,
        landscape_list=landscape_list, 
        pop_size=10000,
        gen_per_step=50, # Each action step in environment
        total_generations=50 * 100, # 100 steps total
    )
    st.session_state.env.reset()
    
    # Store history of frames
    # Each frame: {freqs, drug_idx, mean_fit, step}
    initial_obs = st.session_state.env.get_obs()
    st.session_state.history = [{
        "freqs": initial_obs,
        "drug_idx": 0,
        "mean_fit": st.session_state.env.avg_fitness(),
        "step": 0
    }]
    st.session_state.current_period = 0
    st.session_state.max_periods = 100
    st.session_state.step_size = 5

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def advance_simulation(drug_idx):
    env = st.session_state.env
    for _ in range(st.session_state.step_size):
        if st.session_state.current_period >= st.session_state.max_periods:
            break
        
        obs, _, terminated, _, _ = env.step(drug_idx)
        st.session_state.current_period += 1
        
        st.session_state.history.append({
            "freqs": obs,
            "drug_idx": drug_idx,
            "mean_fit": env.avg_fitness(),
            "step": st.session_state.current_period
        })
        
        if terminated:
            break

def reset_simulation():
    st.session_state.env.reset()
    initial_obs = st.session_state.env.get_obs()
    st.session_state.history = [{
        "freqs": initial_obs,
        "drug_idx": 0,
        "mean_fit": st.session_state.env.avg_fitness(),
        "step": 0
    }]
    st.session_state.current_period = 0

# ---------------------------------------------------------------------------
# Visualization Builders
# ---------------------------------------------------------------------------

def build_3d_plot(current_frame, ls):
    freqs = current_frame["freqs"]
    d = current_frame["drug_idx"]
    z_surface = [row.tolist() for row in ls]
    
    fig = go.Figure()
    
    # Fitness Surface
    fig.add_trace(go.Surface(
        x=list(range(NUM_GENOTYPES)),
        y=list(range(NUM_DRUGS)),
        z=z_surface,
        colorscale="Plasma",
        cmin=GLOBAL_FIT_MIN,
        cmax=GLOBAL_FIT_MAX,
        opacity=0.6,
        showscale=True,
        colorbar=dict(title="Fitness", thickness=16, len=0.7, x=1.02),
        name="Fitness Landscape",
        hovertemplate="Drug: %{y}<br>Genotype: %{x}<br>Fitness: %{z:.3f}<extra></extra>",
    ))
    
    # Population Markers
    sq = np.sqrt(np.clip(freqs, 0, 1))
    if sq.max() > 0:
        sq = sq / sq.max()
    sizes = (MIN_MARKER_SIZE + sq * (MAX_MARKER_SIZE - MIN_MARKER_SIZE)).tolist()
    
    z_markers = ls[d].tolist()
    hover = [
        f"Genotype: {GENOTYPE_LABELS[i]}<br>Freq: {freqs[i]:.3f}<br>Fitness: {z_markers[i]:.3f}"
        for i in range(NUM_GENOTYPES)
    ]
    
    fig.add_trace(go.Scatter3d(
        x=list(range(NUM_GENOTYPES)), 
        y=[d] * NUM_GENOTYPES, 
        z=z_markers,
        mode="markers+text",
        marker=dict(
            size=sizes,
            color=ls[d].tolist(), colorscale="Plasma",
            cmin=GLOBAL_FIT_MIN, cmax=GLOBAL_FIT_MAX,
            opacity=0.95, line=dict(color="white", width=1),
        ),
        text=GENOTYPE_LABELS,
        textposition="top center",
        textfont=dict(size=10, color="white"),
        hovertext=hover, hoverinfo="text",
        showlegend=False,
    ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=30, b=0),
        height=700,
        scene=dict(
            xaxis=dict(title="Genotype", tickvals=list(range(NUM_GENOTYPES)), ticktext=GENOTYPE_LABELS),
            yaxis=dict(title="Drug", tickvals=list(range(NUM_DRUGS)), ticktext=DRUG_NAMES),
            zaxis=dict(title="Fitness", range=[GLOBAL_FIT_MIN, GLOBAL_FIT_MAX]),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0)),
            aspectratio=dict(x=1.5, y=1.0, z=0.6),
        ),
        title=dict(
            text=f"Current Drug: <b>{DRUG_NAMES[d]}</b> | Period: {current_frame['step']}/{st.session_state.max_periods}",
            x=0.5, y=0.95, font=dict(size=20)
        )
    )
    return fig

def build_2d_timeline(history):
    steps = [f["step"] for f in history]
    fitness = [f["mean_fit"] for f in history]
    
    fig = go.Figure()
    
    # Fitness line
    fig.add_trace(go.Scatter(
        x=steps, y=fitness,
        mode="lines+markers",
        line=dict(color="white", width=2),
        marker=dict(size=4, color="rgba(255,255,255,0.6)"),
        name="Mean Fitness",
        hovertemplate="Period %{x}<br>Fitness: %{y:.3f}<extra></extra>"
    ))
    
    # Drug intervals
    if len(history) > 1:
        intervals = []
        start_idx = 0
        cur_drug = history[0]["drug_idx"]
        for i in range(1, len(history)):
            if history[i]["drug_idx"] != cur_drug:
                intervals.append((history[start_idx]["step"], history[i-1]["step"], cur_drug))
                start_idx = i
                cur_drug = history[i]["drug_idx"]
        intervals.append((history[start_idx]["step"], history[-1]["step"], cur_drug))
        
        for t0, t1, d_idx in intervals:
            fig.add_vrect(
                x0=t0, x1=t1, 
                fillcolor=DRUG_FILL_COLORS[d_idx], 
                layer="below", line_width=0
            )
            # Label
            if t1 - t0 > 0.5:
                fig.add_annotation(
                    x=(t0+t1)/2, y=min(fitness)*0.9,
                    text=DRUG_NAMES[d_idx],
                    showarrow=False, font=dict(size=10, color=DRUG_SOLID_COLORS[d_idx])
                )

    fig.update_layout(
        template="plotly_dark",
        height=250,
        margin=dict(l=50, r=20, t=10, b=30),
        xaxis=dict(title="Treatment Period", range=[0, st.session_state.max_periods]),
        yaxis=dict(title="Mean Fitness", range=[0.95, 1.05]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,15,30,0.5)",
    )
    return fig

# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

st.title("🔬 Interactive Tumor Evolution Sandbox")
st.subheader("Manually steer the evolutionary trajectory of genotypes")

col_main, col_side = st.columns([4, 1])

with col_side:
    st.markdown("### 💊 Therapy Plan")
    selected_drug_name = st.selectbox(
        "Choose Next Therapy:", 
        DRUG_NAMES, 
        index=0,
        help="Select a drug to apply for the next 5 treatment periods."
    )
    selected_drug_idx = DRUG_NAMES.index(selected_drug_name)
    
    st.info(f"Applying **{selected_drug_name}** will shift the fitness landscape and drive selection for different genotypes.")
    
    advance_disabled = st.session_state.current_period >= st.session_state.max_periods
    if st.button("🚀 Advance 5 Periods", disabled=advance_disabled):
        advance_simulation(selected_drug_idx)
        st.rerun()
    
    if advance_disabled:
        st.warning("Simulation Complete!")
    
    st.markdown("---")
    if st.button("🔄 Reset Simulation"):
        reset_simulation()
        st.rerun()

    st.markdown("### 📊 Metrics")
    curr_frame = st.session_state.history[-1]
    st.metric("Period", f"{curr_frame['step']} / {st.session_state.max_periods}")
    st.metric("Mean Fitness", f"{curr_frame['mean_fit']:.3f}")

with col_main:
    # 3D Visualization
    fig3d = build_3d_plot(st.session_state.history[-1], st.session_state.ls)
    st.plotly_chart(fig3d, use_container_width=True, config={'displayModeBar': False})
    
    # 2D Timeline
    fig2d = build_2d_timeline(st.session_state.history)
    st.plotly_chart(fig2d, use_container_width=True, config={'displayModeBar': False})

st.caption("Developed by Antigravity | Based on Chen et al. Fitness Landscapes")
