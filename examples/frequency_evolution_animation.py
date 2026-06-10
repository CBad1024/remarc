"""
frequency_evolution_animation.py

Interactive animation of tumor/bacterial population evolution overlaid on the
3D fitness landscape surface (Drugs × Genotypes × Fitness) for the Chen et al. dataset.

Three policy modes can be toggled in the browser:
  - Optimal Policy  (trained PPO agent for Chen)
  - Random Policy   (uniform over 4 drugs)
  - Best Single Drug (always uses the drug with the highest mean landscape fitness)

Each mode shows:
  - Full 3D Chen et al. fitness landscape surface (N=3, 4 drugs)
  - Animated Scatter3d markers sized by √(genotype frequency)
  - Fitness-over-time subplot with drug-coloured bands and animated cursor

Output: wf_tumor_evolution_chen.html  (fully interactive, pausable in-browser)
"""

import sys
import os
from pathlib import Path
from typing import Union

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import json
import numpy as np
import plotly.graph_objects as go

from remarc.envs import WrightFisherEnv
from remarc.envs.utils import define_chen_landscapes
from remarc.core.landscapes import Landscape
from remarc.core.hyperparameters import Presets
from tianshou.data import Batch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DRUG_NAMES = ['Drug_A', 'Drug_B', 'Drug_C', 'Drug_D']
NUM_DRUGS = 4
NUM_GENOTYPES = 8
GENOTYPE_LABELS = [bin(i)[2:].zfill(3) for i in range(NUM_GENOTYPES)]

MAX_MARKER_SIZE = 45
MIN_MARKER_SIZE = 3

# Chen fitness values are very close to 1.0 (range ~0.98 to 1.02)
GLOBAL_FIT_MIN = 0.98
GLOBAL_FIT_MAX = 1.02

# ---------------------------------------------------------------------------
# Drug color palette
# ---------------------------------------------------------------------------

DRUG_FILL_COLORS = [
    "rgba(99,110,250,0.22)",
    "rgba(239,85,59,0.22)",
    "rgba(0,204,150,0.22)",
    "rgba(171,99,250,0.22)",
]

DRUG_SOLID_COLORS = [
    "rgba(99,110,250,0.90)",
    "rgba(239,85,59,0.90)",
    "rgba(0,204,150,0.90)",
    "rgba(171,99,250,0.90)",
]

# ---------------------------------------------------------------------------
# Policy classes
# ---------------------------------------------------------------------------

class _FixedDrugPolicy:
    """Always chooses a single drug index."""
    def __init__(self, drug_idx: int):
        self.drug_idx = drug_idx
    def __call__(self, batch):
        import tianshou.data as td
        n = len(batch.obs) if hasattr(batch, "obs") else 1
        return td.Batch(act=np.full(n, self.drug_idx, dtype=int))


class _RandomPolicy:
    """Uniform random over all drugs."""
    def __call__(self, batch):
        import tianshou.data as td
        n = len(batch.obs) if hasattr(batch, "obs") else 1
        return td.Batch(act=np.random.randint(0, NUM_DRUGS, size=n))


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def _collect_frame_data(env, policy, episode_steps: int) -> list:
    frames = []
    env.reset()
    obs = env.get_obs()

    freqs0 = np.array([env.pop.get(g, 0) for g in env.genotypes], dtype=float) / env.pop_size
    frames.append({
        "freqs": freqs0,
        "drug_idx": env.current_drug,
        "mean_fit": float(env.avg_fitness()),
    })

    for _ in range(episode_steps):
        batch = Batch(obs=[obs], info=Batch())
        action = int(policy(batch).act[0])
        obs, _, terminated, _, _ = env.step(action)
        freqs = np.array(obs, dtype=float)
        frames.append({
            "freqs": freqs,
            "drug_idx": env.current_drug,
            "mean_fit": float(env.avg_fitness()),
        })
        if terminated:
            break

    return frames


def _drug_intervals(fdl: list) -> list:
    intervals = []
    start = 0
    cur = fdl[0]["drug_idx"]
    for i in range(1, len(fdl)):
        if fdl[i]["drug_idx"] != cur:
            intervals.append((start, i - 1, cur))
            start = i
            cur = fdl[i]["drug_idx"]
    intervals.append((start, len(fdl) - 1, cur))
    return intervals


# ---------------------------------------------------------------------------
# Plotly trace builders
# ---------------------------------------------------------------------------

def _build_surface(ls_matrix: np.ndarray) -> go.Surface:
    z_lists = [row.tolist() for row in ls_matrix]
    return go.Surface(
        x=list(range(NUM_GENOTYPES)),
        y=list(range(NUM_DRUGS)),
        z=z_lists,
        colorscale="Plasma",
        cmin=GLOBAL_FIT_MIN,
        cmax=GLOBAL_FIT_MAX,
        opacity=0.75,
        showscale=True,
        colorbar=dict(title="Fitness", thickness=16, len=0.7, x=1.02),
        name="Fitness Landscape",
        hovertemplate="Drug: %{y}<br>Genotype: %{x}<br>Fitness: %{z:.3f}<extra></extra>",
    )


def _marker_sizes(freqs: np.ndarray) -> list:
    sq = np.sqrt(np.clip(freqs, 0, 1))
    if sq.max() > 0:
        sq = sq / sq.max()
    return (MIN_MARKER_SIZE + sq * (MAX_MARKER_SIZE - MIN_MARKER_SIZE)).tolist()


def _scatter_for_frame(fd: dict, ls: np.ndarray) -> go.Scatter3d:
    freqs = fd["freqs"]
    d = fd["drug_idx"]
    z = ls[d].tolist()
    hover = [
        f"Genotype: {GENOTYPE_LABELS[i]}<br>Freq: {freqs[i]:.3f}<br>Fitness: {z[i]:.3f}"
        for i in range(NUM_GENOTYPES)
    ]
    return go.Scatter3d(
        x=list(range(NUM_GENOTYPES)), y=[d] * NUM_GENOTYPES, z=z,
        mode="markers+text",
        marker=dict(
            size=_marker_sizes(freqs),
            color=ls[d].tolist(), colorscale="Plasma",
            cmin=GLOBAL_FIT_MIN, cmax=GLOBAL_FIT_MAX,
            opacity=0.92, line=dict(color="white", width=0.8),
        ),
        text=GENOTYPE_LABELS,
        textposition="top center",
        textfont=dict(size=8, color="white"),
        hovertext=hover, hoverinfo="text",
        name=f"Population ({DRUG_NAMES[d]})", showlegend=False,
    )


def _gauge_trace(mean_fit: float) -> go.Scatter3d:
    return go.Scatter3d(
        x=[NUM_GENOTYPES + 0.5], y=[NUM_DRUGS - 1], z=[mean_fit],
        mode="markers+text",
        marker=dict(
            size=22, color=mean_fit,
            colorscale=[[0,"#7b2d8b"],[0.3,"#e05c2c"],[0.6,"#f0c229"],[1,"#f5f5aa"]],
            cmin=GLOBAL_FIT_MIN, cmax=GLOBAL_FIT_MAX,
            symbol="diamond", opacity=1.0,
        ),
        text=[f"<b>μ={mean_fit:.3f}</b>"],
        textposition="middle right",
        textfont=dict(size=13, color="white"),
        hovertext=[f"Mean fitness: {mean_fit:.3f}"], hoverinfo="text",
        name="Mean Fitness", showlegend=False,
    )


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def _build_fig3d(frame_data_list: list, ls: np.ndarray, panel_id: str, include_plotlyjs: Union[str, bool] = False) -> str:
    """Build and return the 3D figure as an HTML div snippet."""
    surface = _build_surface(ls)
    frames_pl = []
    slider_steps = []

    for i, fd in enumerate(frame_data_list):
        d_idx = fd["drug_idx"]
        drug_name = DRUG_NAMES[d_idx]
        frames_pl.append(go.Frame(
            data=[surface, _scatter_for_frame(fd, ls), _gauge_trace(fd["mean_fit"])],
            name=str(i),
            layout=go.Layout(annotations=[
                dict(x=0.02, y=0.97, xref="paper", yref="paper",
                     text=f"<b>Drug: {drug_name}</b>",
                     font=dict(size=18, color="white"),
                     bgcolor="rgba(20,20,50,0.80)",
                     bordercolor=DRUG_SOLID_COLORS[d_idx],
                     borderwidth=2, borderpad=6, showarrow=False, align="left"),
                dict(x=0.98, y=0.97, xref="paper", yref="paper",
                     text=f"Step {i} | μ={fd['mean_fit']:.3f}",
                     font=dict(size=11, color="rgba(200,200,200,0.9)"),
                     showarrow=False, align="right"),
            ]),
        ))
        slider_steps.append(dict(
            method="animate", label=str(i),
            args=[[str(i)], dict(frame=dict(duration=200, redraw=True),
                                 mode="immediate", transition=dict(duration=80))],
        ))

    init_d = frame_data_list[0]["drug_idx"]
    fig = go.Figure(
        data=[surface, _scatter_for_frame(frame_data_list[0], ls), _gauge_trace(frame_data_list[0]["mean_fit"])],
        frames=frames_pl,
    )
    fig.update_layout(
        title=dict(text="Tumor Evolution – Chen et al. Fitness Landscape",
                   font=dict(size=19, color="white"), x=0.5),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="white"),
        scene=dict(
            xaxis=dict(title="Genotype", tickvals=list(range(NUM_GENOTYPES)), ticktext=GENOTYPE_LABELS,
                       tickfont=dict(size=8), gridcolor="rgba(100,100,150,0.3)", backgroundcolor="rgba(10,10,30,0)"),
            yaxis=dict(title="Drug", tickvals=list(range(NUM_DRUGS)), ticktext=DRUG_NAMES,
                       tickfont=dict(size=9), gridcolor="rgba(100,100,150,0.3)", backgroundcolor="rgba(10,10,30,0)"),
            zaxis=dict(title="Fitness", range=[GLOBAL_FIT_MIN, GLOBAL_FIT_MAX],
                       gridcolor="rgba(100,100,150,0.3)", backgroundcolor="rgba(10,10,30,0)"),
            camera=dict(eye=dict(x=1.6, y=-1.8, z=0.9), up=dict(x=0, y=0, z=1)),
            bgcolor="rgba(10,10,30,1)", aspectmode="manual",
            aspectratio=dict(x=1.5, y=1.0, z=0.6),
        ),
        annotations=[
            dict(x=0.02, y=0.97, xref="paper", yref="paper",
                 text=f"<b>Drug: {DRUG_NAMES[init_d]}</b>",
                 font=dict(size=18, color="white"), bgcolor="rgba(20,20,50,0.80)",
                 bordercolor=DRUG_SOLID_COLORS[init_d], borderwidth=2, borderpad=6,
                 showarrow=False, align="left"),
            dict(x=0.98, y=0.97, xref="paper", yref="paper",
                 text=f"Step 0 | μ={frame_data_list[0]['mean_fit']:.3f}",
                 font=dict(size=11, color="rgba(200,200,200,0.9)"), showarrow=False, align="right"),
        ],
        updatemenus=[],
        sliders=[dict(
            active=0, steps=slider_steps,
            x=0.1, len=0.8, xanchor="left", y=0.02, yanchor="top",
            pad=dict(b=10, t=50),
            currentvalue=dict(prefix="Step: ", visible=True, xanchor="center",
                               font=dict(size=13, color="white")),
            transition=dict(duration=100, easing="cubic-in-out"),
            bgcolor="rgba(30,40,80,0.6)", bordercolor="rgba(100,120,200,0.5)",
            tickcolor="white", font=dict(color="white"),
        )],
        margin=dict(l=0, r=0, t=60, b=120),
        height=680,
    )

    return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs, div_id=panel_id + "_3d")


def _build_fig2d(frame_data_list: list, panel_id: str, fit_min_override=None, fit_max_override=None) -> tuple:
    """Build and return the 2D fitness timeline as an HTML div snippet + fitness JSON."""
    all_steps = list(range(len(frame_data_list)))
    all_fitness = [fd["mean_fit"] for fd in frame_data_list]
    fit_min = fit_min_override if fit_min_override is not None else max(0.0, min(all_fitness) - 0.005)
    fit_max = fit_max_override if fit_max_override is not None else min(GLOBAL_FIT_MAX, max(all_fitness) + 0.01)
    intervals = _drug_intervals(frame_data_list)

    fitness_line = go.Scatter(
        x=all_steps, y=all_fitness,
        mode="lines+markers",
        line=dict(color="rgba(255,255,255,0.88)", width=2.5),
        marker=dict(size=5, color="rgba(255,255,255,0.6)"),
        name="Mean Fitness", showlegend=False,
        hovertemplate="Step %{x}<br>Fitness: %{y:.3f}<extra></extra>",
    )
    cursor = go.Scatter(
        x=[0, 0], y=[fit_min, fit_max],
        mode="lines",
        line=dict(color="rgba(255,90,90,0.95)", width=2.5, dash="dot"),
        showlegend=False, hoverinfo="skip", name="Current Step",
    )

    fig2d = go.Figure(data=[fitness_line, cursor])

    for (t0, t1, d_idx) in intervals:
        fig2d.add_vrect(x0=t0 - 0.5, x1=t1 + 0.5,
                        fillcolor=DRUG_FILL_COLORS[d_idx], line_width=0, layer="below")
        fig2d.add_annotation(
            x=(t0 + t1) / 2, y=fit_min + (fit_max - fit_min) * 0.05,
            text=DRUG_NAMES[d_idx],
            font=dict(size=9, color=DRUG_SOLID_COLORS[d_idx]),
            showarrow=False,
        )

    fig2d.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#141824",
        font=dict(color="white"),
        xaxis=dict(title="Policy Step", gridcolor="rgba(100,100,150,0.25)",
                   color="white", showgrid=True, zeroline=False),
        yaxis=dict(title="Mean Fitness", range=[fit_min, fit_max],
                   gridcolor="rgba(100,100,150,0.25)", color="white",
                   showgrid=True, zeroline=False),
        margin=dict(l=60, r=20, t=20, b=50),
        height=280,
    )

    html_2d = fig2d.to_html(full_html=False, include_plotlyjs=False,
                             div_id=panel_id + "_2d", default_height="280px")
    return html_2d, json.dumps(all_fitness), fit_min, fit_max


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def create_frequency_gif(
    episode_steps: int = 75,
    policy_filename: str = "best_policy_chen.pth",
    output_filename: str = "wf_tumor_evolution_chen.html",
    pop_size: int = 10_000,
    gen_per_step: int = 500,
):
    """
    Simulate three policies (optimal, random, best-single-drug) and produce a
    single interactive HTML with a toggle to switch between them.
    """
    print("Setting up Chen et al. landscapes...")
    ls = define_chen_landscapes()  # (4, 8)
    v_N = 3
    landscape_list = [Landscape(v_N, sigma=0.0, ls=ls[i]) for i in range(len(ls))]
    total_gens = gen_per_step * (episode_steps + 2)

    def _make_env():
        return WrightFisherEnv(
            num_drugs=NUM_DRUGS, seq_length=v_N,
            landscape_list=landscape_list, pop_size=pop_size,
            gen_per_step=gen_per_step, total_generations=total_gens,
        )

    # ---- Find best single drug (max mean fitness across all genotypes) ----
    best_drug_idx = int(np.argmax(ls.mean(axis=1)))
    best_drug_name = DRUG_NAMES[best_drug_idx]
    print(f"Best single drug: {best_drug_name} (index {best_drug_idx}, "
          f"mean fitness {ls[best_drug_idx].mean():.3f})")

    # ---- Load optimal policy ----
    from dataclasses import replace
    p_base = Presets.p1_ls()
    p = replace(p_base, state_shape=(NUM_GENOTYPES,), num_actions=NUM_DRUGS, dataset="chen")
    import torch as _torch

    policy_path = policy_filename
    if not os.path.isabs(policy_path):
        candidate = _project_root / "log" / "RL" / policy_filename
        if candidate.exists():
            policy_path = str(candidate)

    opt_policy = None
    opt_loaded = False

    if os.path.exists(policy_path):
        from remarc.agents.tianshou_agent import get_ppo_policy
        _train_envs, _ = WrightFisherEnv.getEnv(2, 2,
                                                   landscape_list=landscape_list)
        opt_policy = get_ppo_policy(p, _train_envs)
        try:
            opt_policy.load_state_dict(_torch.load(policy_path, map_location="cpu"))
            print(f"Loaded policy (strict) from: {policy_path}")
            opt_loaded = True
        except Exception:
            try:
                opt_policy.load_state_dict(
                    _torch.load(policy_path, map_location="cpu"), strict=False)
                print(f"Loaded policy (non-strict) from: {policy_path}")
                opt_loaded = True
            except Exception as e2:
                print(f"Warning: could not load policy ({e2}). Opt policy = random.")
    else:
        print(f"Policy file not found: {policy_path}. Opt policy = random.")

    if not opt_loaded:
        opt_policy = _RandomPolicy()

    # ---- Simulate all three policies ----
    policies = {
        "single": (f"Best Single Drug ({best_drug_name})", _FixedDrugPolicy(best_drug_idx)),
        "random": ("Random Policy",         _RandomPolicy()),
        "opt":    ("Optimal Policy (Chen)",       opt_policy),
    }

    frames_by_policy = {}
    for key, (label, pol) in policies.items():
        print(f"Simulating {episode_steps} steps [{label}]...")
        env = _make_env()
        frames_by_policy[key] = _collect_frame_data(env, pol, episode_steps)
        n = len(frames_by_policy[key])
        print(f"  → {n} frames captured.")

    # ---- Use a shared y-axis range across all three for easy comparison ----
    all_fit = [fd["mean_fit"] for fdl in frames_by_policy.values() for fd in fdl]
    shared_fit_min = max(GLOBAL_FIT_MIN, min(all_fit) - 0.005)
    shared_fit_max = min(GLOBAL_FIT_MAX, max(all_fit) + 0.01)

    print("Building figures...")

    # ---- Build HTML panels per policy ----
    panels = {}
    is_first_panel = True
    for key, (label, _) in policies.items():
        fdl = frames_by_policy[key]
        plotlyjs = "cdn" if is_first_panel else False
        html_3d_div = _build_fig3d(fdl, ls, panel_id=key, include_plotlyjs=plotlyjs)

        html_2d_div, fitness_json, _, _ = _build_fig2d(
            fdl, panel_id=key,
            fit_min_override=shared_fit_min,
            fit_max_override=shared_fit_max,
        )
        panels[key] = {
            "label": label,
            "html_3d": html_3d_div,
            "html_2d": html_2d_div,
            "fitness_json": fitness_json,
            "fit_min": shared_fit_min,
            "fit_max": shared_fit_max,
        }
        is_first_panel = False

    # ---- Assemble full HTML ----
    js_bridges = []
    for key, pd in panels.items():
        js_bridges.append(f"""
    (function() {{
      var fitnessVals = {pd['fitness_json']};
      var fitMin = {pd['fit_min']};
      var fitMax = {pd['fit_max']};
      var gd3 = document.getElementById('{key}_3d');
      var gd2 = document.getElementById('{key}_2d');

      function syncCursor(step) {{
        if (!gd2 || isNaN(step)) return;
        Plotly.restyle(gd2, {{x: [[step, step]], y: [[fitMin, fitMax]]}}, [1]);
      }}

      if (gd3) {{
        gd3.on('plotly_animatingframe', function(e) {{
          if (e && e.name != null) syncCursor(parseInt(e.name, 10));
        }});
        gd3.on('plotly_sliderchange', function(e) {{
          if (e && e.step) syncCursor(parseInt(e.step.label, 10));
        }});
      }}
    }})();""")

    js_bridge_all = "\n".join(js_bridges)

    panel_divs = []
    for key, pd in panels.items():
        display = "block" if key == "opt" else "none"
        panel_divs.append(f"""
    <div id="panel-{key}" class="policy-panel" style="display:{display};">
      <div class="panel-3d">{pd['html_3d']}</div>
      <hr class="sep">
      <div class="panel-2d">{pd['html_2d']}</div>
    </div>""")

    panel_divs_html = "\n".join(panel_divs)

    btn_html_parts = []
    for key, pd in panels.items():
        active_cls = "active" if key == "opt" else ""
        btn_html_parts.append(
            f'<button class="toggle-btn {active_cls}" data-key="{key}">{pd["label"]}</button>'
        )
    btn_html = "\n".join(btn_html_parts)

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tumor Evolution – Chen et al. Landscape</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: #0e1117; font-family: 'Segoe UI', system-ui, sans-serif; color: white; }}

    #toggle-bar {{
      display: flex;
      gap: 10px;
      justify-content: center;
      padding: 14px 10px 8px;
      background: linear-gradient(90deg, #0e1117, #141f38, #0e1117);
      border-bottom: 1px solid rgba(100,120,200,0.3);
      flex-wrap: wrap;
    }}
    .toggle-btn {{
      padding: 8px 20px;
      border-radius: 8px;
      border: 1.5px solid rgba(100,120,200,0.45);
      background: rgba(20,30,70,0.7);
      color: rgba(200,215,255,0.85);
      font-size: 14px;
      font-weight: 500;
      cursor: pointer;
      transition: background 0.2s, border-color 0.2s, color 0.2s, box-shadow 0.2s;
      letter-spacing: 0.3px;
    }}
    .toggle-btn:hover {{
      background: rgba(60,80,160,0.6);
      border-color: rgba(140,160,255,0.7);
      color: white;
    }}
    .toggle-btn.active {{
      background: linear-gradient(135deg, rgba(80,100,220,0.85), rgba(40,60,160,0.9));
      border-color: rgba(160,180,255,0.8);
      color: white;
      box-shadow: 0 0 12px rgba(100,140,255,0.35);
    }}

    .policy-panel {{ width: 100%; }}
    .panel-3d {{ width: 100%; }}
    .panel-2d {{ width: 100%; }}
    .sep {{
      margin: 4px 20px;
      border: none;
      border-top: 1px solid rgba(100,120,200,0.28);
    }}
  </style>
</head>
<body>
  <div id="toggle-bar">
    <span style="color:rgba(180,190,220,0.7);font-size:13px;align-self:center;margin-right:6px;">
      Policy:
    </span>
    {btn_html}
    <div style="width: 1px; height: 24px; background: rgba(100,120,200,0.3); margin: 0 10px; align-self: center;"></div>
    <button id="play-pause-btn" class="toggle-btn" style="min-width: 90px; border-color: rgba(100,200,150,0.5);">▶ Play</button>
  </div>

  <div id="panels-container">
{panel_divs_html}
  </div>

  <script>
  window.addEventListener('load', function() {{
    var hiddenKeys = [];
    document.querySelectorAll('.policy-panel').forEach(function(p) {{
      if (p.style.display === 'none') hiddenKeys.push(p.id.replace('panel-', ''));
    }});

    function initNext(idx) {{
      if (idx >= hiddenKeys.length) return;
      var key = hiddenKeys[idx];
      var panel = document.getElementById('panel-' + key);
      panel.style.visibility = 'hidden';
      panel.style.display = 'block';
      var gd3 = document.getElementById(key + '_3d');
      var gd2 = document.getElementById(key + '_2d');
      setTimeout(function() {{
        if (gd3 && gd3.data) Plotly.redraw(gd3);
        if (gd2 && gd2.data) Plotly.Plots.resize(gd2);
        setTimeout(function() {{
          panel.style.display = 'none';
          panel.style.visibility = '';
          initNext(idx + 1);
        }}, 200);
      }}, 150);
    }}
    setTimeout(function() {{ initNext(0); }}, 600);
  }});

  var buttons = document.querySelectorAll('.toggle-btn[data-key]');
  var playPauseBtn = document.getElementById('play-pause-btn');
  var currentPolicy = 'opt';
  var isPlaying = false;

  function setPlaying(play) {{
    isPlaying = play;
    if (playPauseBtn) {{
      playPauseBtn.innerText = play ? "⏸ Pause" : "▶ Play";
      playPauseBtn.style.background = play ? "rgba(80,40,40,0.7)" : "rgba(20,30,70,0.7)";
      playPauseBtn.style.borderColor = play ? "rgba(200,100,100,0.5)" : "rgba(100,200,150,0.5)";
    }}
  }}

  if (playPauseBtn) {{
    playPauseBtn.addEventListener('click', function() {{
      var gd3 = document.getElementById(currentPolicy + '_3d');
      if (!gd3) return;
      
      if (!isPlaying) {{
        Plotly.animate(gd3, null, {{
          frame: {{duration: 300, redraw: true}},
          fromcurrent: true,
          mode: 'immediate',
          transition: {{duration: 100}}
        }});
        setPlaying(true);
      }} else {{
        Plotly.animate(gd3, [null], {{
          frame: {{duration: 0, redraw: false}},
          mode: 'immediate',
          transition: {{duration: 0}}
        }});
        setPlaying(false);
      }}
    }});
  }}

  buttons.forEach(function(btn) {{
    btn.addEventListener('click', function() {{
      var oldKey = currentPolicy;
      var key = btn.getAttribute('data-key');
      if (oldKey === key) return;

      var oldGd3 = document.getElementById(oldKey + '_3d');
      if (oldGd3 && isPlaying) Plotly.animate(oldGd3, [null], {{mode: 'immediate'}});
      setPlaying(false);

      currentPolicy = key;
      document.querySelectorAll('.policy-panel').forEach(function(p) {{
        p.style.display = 'none';
      }});
      buttons.forEach(function(b) {{ b.classList.remove('active'); }});
      document.getElementById('panel-' + key).style.display = 'block';
      btn.classList.add('active');
      setTimeout(function() {{
        var gd3 = document.getElementById(key + '_3d');
        var gd2 = document.getElementById(key + '_2d');
        if (gd3 && gd3.data) Plotly.redraw(gd3);
        if (gd2 && gd2.data) Plotly.Plots.resize(gd2);
      }}, 80);
    }});
  }});

  {js_bridge_all}
  </script>
</body>
</html>
"""

    out_path = output_filename
    if not os.path.isabs(out_path):
        out_path = str(_project_root / output_filename)

    print(f"Saving to: {out_path}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print("Done! Open the HTML file in your browser.")
    return out_path


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Chen tumor evolution animation")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--policy", type=str, default="best_policy_chen.pth")
    parser.add_argument("--output", type=str, default="wf_tumor_evolution_chen.html")
    parser.add_argument("--pop-size", type=int, default=10000)
    parser.add_argument("--gen-per-step", type=int, default=50)
    args = parser.parse_args()

    create_frequency_gif(
        episode_steps=args.steps,
        policy_filename=args.policy,
        output_filename=args.output,
        pop_size=args.pop_size,
        gen_per_step=args.gen_per_step,
    )
