"""
Plotting utilities for EvoDM / REMARC.

Includes:
- Simplex slice visualization for four-genotype policy (Fig 3 from paper)
- Genotype frequency heatmaps
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.tri import Triangulation
from matplotlib.patches import Patch, FancyArrowPatch
from matplotlib.collections import PolyCollection


# ──────────────────────────────────────────────────────────────────────
#  Simplex Policy Visualization (Four-State System)
# ──────────────────────────────────────────────────────────────────────

_SQRT3_2 = np.sqrt(3) / 2


def _bary_to_cart(lam0, lam1, lam2):
    """Convert barycentric coordinates to 2D Cartesian for an equilateral triangle.
    
    Vertices:  (1,0,0)→(0,0)  (0,1,0)→(1,0)  (0,0,1)→(0.5, √3/2)
    """
    x = lam1 + lam2 * 0.5
    y = lam2 * _SQRT3_2
    return x, y


def _make_simplex_grid(resolution=60):
    """Generate a uniform barycentric grid on the 2-simplex.
    
    Returns:
        bary: (M, 3) array of barycentric coordinates
        x, y: (M,) arrays of Cartesian coordinates
        triangles: (T, 3) array of triangle vertex indices
    """
    pts = []
    idx_map = {}  # (i, j) → index
    
    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            k = resolution - i - j
            idx_map[(i, j)] = len(pts)
            pts.append((i / resolution, j / resolution, k / resolution))
    
    bary = np.array(pts)
    x, y = _bary_to_cart(bary[:, 0], bary[:, 1], bary[:, 2])
    
    # Build triangulation manually (avoids Delaunay edge issues)
    tris = []
    for i in range(resolution):
        for j in range(resolution - i):
            # Upward triangle
            v0 = idx_map[(i, j)]
            v1 = idx_map[(i + 1, j)]
            v2 = idx_map[(i, j + 1)]
            tris.append((v0, v1, v2))
            # Downward triangle (if exists)
            if (i + 1, j + 1) in idx_map:
                v3 = idx_map[(i + 1, j + 1)]
                tris.append((v1, v3, v2))
    
    triangles = np.array(tris)
    return bary, x, y, triangles


def greedy_policy(state, landscapes):
    """Greedy baseline: pick the drug minimizing immediate expected fitness.
    
    Args:
        state: (4,) genotype frequency vector
        landscapes: (num_drugs, 4) fitness landscape array
    
    Returns:
        int: drug index
    """
    expected_fitness = landscapes @ state  # (num_drugs,)
    return int(np.argmin(expected_fitness))


def plot_simplex_policy_slices(
    policy_fn,
    num_drugs=4,
    drug_labels=None,
    drug_colors=None,
    resolution=60,
    x3_slices=None,
    genotype_labels=None,
    title=None,
    figsize=None,
    ax_array=None,
    show_legend=True,
):
    """
    Visualize a drug-switching policy on the 4-genotype simplex.

    For a system with 4 genotypes, the population state lives on the
    3-simplex (x₀ + x₁ + x₂ + x₃ = 1). We slice along x₃ and show
    each cross-section as a colored triangle, where the color at each
    point indicates which drug the policy selects for that population
    composition.

    This recreates the style of Figure 3A from the paper.

    Args:
        policy_fn: callable(state: ndarray(4,)) → int (drug index)
        num_drugs: number of drugs
        drug_labels: list of str, names for each drug
        drug_colors: list of colors for each drug
        resolution: grid density (points per triangle edge)
        x3_slices: list of (x3_min, x3_max) tuples for slicing
        genotype_labels: list of str, names for genotypes 0–3
        title: figure title
        figsize: (width, height) tuple
        ax_array: optional pre-created axes array (length == len(x3_slices))
        show_legend: whether to draw the drug color legend

    Returns:
        fig: matplotlib Figure (None if ax_array was supplied)
    """
    # ── defaults ──
    if drug_labels is None:
        drug_labels = [f"Drug {chr(65 + i)}" for i in range(num_drugs)]
    if drug_colors is None:
        drug_colors = ["#2ecc71", "#e67e22", "#5b7cc9", "#e84393",
                       "#f1c40f", "#1abc9c", "#9b59b6", "#e74c3c"][:num_drugs]
    if genotype_labels is None:
        genotype_labels = [f"genotype {i}" for i in range(4)]
    if x3_slices is None:
        x3_slices = [
            (0.5, 1.0), (0.4, 0.5), (0.3, 0.4),
            (0.2, 0.3), (0.1, 0.2), (0.0, 0.1),
        ]

    n_slices = len(x3_slices)
    if figsize is None:
        figsize = (2.6 * n_slices + 1.5, 3.8)

    # ── figure / axes ──
    fig = None
    if ax_array is None:
        fig, ax_array = plt.subplots(1, n_slices, figsize=figsize)
    if n_slices == 1:
        ax_array = [ax_array]

    # ── pre-compute triangle grid (reuse across slices) ──
    bary, x_cart, y_cart, tri_idx = _make_simplex_grid(resolution)
    cmap = mcolors.ListedColormap(drug_colors[:num_drugs])
    bounds = np.arange(-0.5, num_drugs + 0.5, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    for s_idx, (x3_lo, x3_hi) in enumerate(x3_slices):
        ax = ax_array[s_idx]
        x3_mid = (x3_lo + x3_hi) / 2.0
        S = 1.0 - x3_mid

        # ── query policy at every grid point ──
        actions = np.zeros(len(bary), dtype=int)
        for p_idx in range(len(bary)):
            l0, l1, l2 = bary[p_idx]
            state = np.array([l0 * S, l1 * S, l2 * S, x3_mid], dtype=np.float32)
            actions[p_idx] = policy_fn(state)

        # ── color each triangle by its centroid action ──
        tri_actions = np.zeros(len(tri_idx), dtype=int)
        for t_idx, (v0, v1, v2) in enumerate(tri_idx):
            # majority vote of the three vertices
            votes = [actions[v0], actions[v1], actions[v2]]
            tri_actions[t_idx] = max(set(votes), key=votes.count)

        # Build polygon collection for crisp filled triangles
        verts = np.column_stack([x_cart, y_cart])
        tri_verts = verts[tri_idx]  # (T, 3, 2)
        colors_rgba = [cmap(norm(a)) for a in tri_actions]

        pc = PolyCollection(tri_verts, facecolors=colors_rgba,
                            edgecolors="none", linewidths=0)
        ax.add_collection(pc)

        # ── triangle outline ──
        outline_x = [0, 1, 0.5, 0]
        outline_y = [0, 0, _SQRT3_2, 0]
        ax.plot(outline_x, outline_y, color="black", linewidth=1.2, zorder=5)

        # ── slice label ──
        ax.text(0.5, -0.12, f"[{x3_lo:.1f}, {x3_hi:.1f}]",
                ha="center", va="top", fontsize=8, transform=ax.transAxes)

        # ── vertex labels on first and last slice ──
        if s_idx == 0:
            ax.text(-0.04, -0.06, genotype_labels[0],
                    ha="center", fontsize=7, style="italic")
            ax.text(0.5, _SQRT3_2 + 0.06, genotype_labels[3],
                    ha="center", fontsize=7, style="italic")
        if s_idx == n_slices - 1:
            ax.text(1.04, -0.06, genotype_labels[1],
                    ha="center", fontsize=7, style="italic")
            ax.text(0.5, _SQRT3_2 + 0.06, genotype_labels[2],
                    ha="center", fontsize=7, style="italic")

        ax.set_xlim(-0.08, 1.08)
        ax.set_ylim(-0.15, _SQRT3_2 + 0.15)
        ax.set_aspect("equal")
        ax.axis("off")

    # ── x₃ arrow ──
    if fig is not None:
        fig.text(
            0.10, 0.06,
            f"x₃ = 1",
            ha="left", fontsize=9, color="gray",
        )
        fig.text(
            0.82, 0.06,
            f"x₃ = 0",
            ha="right", fontsize=9, color="gray",
        )
        fig.text(
            0.46, 0.06,
            f"← {genotype_labels[3]} →",
            ha="center", fontsize=9, color="gray",
        )

    # ── legend ──
    if show_legend:
        legend_elements = [
            Patch(facecolor=drug_colors[i], edgecolor="gray",
                  linewidth=0.5, label=drug_labels[i])
            for i in range(num_drugs)
        ]
        target = fig if fig is not None else ax_array[-1]
        if fig is not None:
            fig.legend(
                handles=legend_elements, loc="center right",
                bbox_to_anchor=(0.98, 0.55), fontsize=9,
                framealpha=0.9, edgecolor="lightgray",
            )
        else:
            ax_array[-1].legend(
                handles=legend_elements, loc="upper right",
                fontsize=8, framealpha=0.9,
            )

    if title and fig is not None:
        fig.suptitle(title, fontsize=12, y=0.97)
    if fig is not None:
        fig.subplots_adjust(wspace=0.02, left=0.03, right=0.87, bottom=0.12, top=0.90)

    return fig


def plot_four_state_simplex(
    landscapes=None,
    policy_fn=None,
    policy_path=None,
    resolution=60,
    save_path=None,
    title="Policy on 4-Genotype Simplex (Greedy Baseline)",
):
    """
    Convenience function to plot the simplex visualization for the four-state system.

    If no policy_fn is provided, uses the greedy baseline (minimize immediate fitness).
    If policy_path is provided, loads a trained PPO policy from disk.

    Args:
        landscapes: (4, 4) array of fitness values; loaded from defaults if None
        policy_fn: optional callable(state) → action
        policy_path: optional path to a saved policy .pth file
        resolution: grid density
        save_path: optional path to save the figure
        title: figure title

    Returns:
        fig: matplotlib Figure
    """
    if landscapes is None:
        from remarc.envs.utils import define_four_state_landscapes
        landscapes = define_four_state_landscapes()

    # Build policy function
    if policy_fn is None and policy_path is not None:
        policy_fn = _load_ppo_policy_fn(policy_path, state_dim=4, n_actions=len(landscapes))

    if policy_fn is None:
        # Default: greedy baseline
        def policy_fn(state):
            return greedy_policy(state, landscapes)
        title = title or "Greedy Policy on 4-Genotype Simplex"

    genotype_labels = ["genotype 0", "genotype 1", "genotype 2", "genotype 3"]
    drug_labels = ["Drug A", "Drug B", "Drug C", "Drug D"]

    fig = plot_simplex_policy_slices(
        policy_fn=policy_fn,
        num_drugs=len(landscapes),
        drug_labels=drug_labels,
        genotype_labels=genotype_labels,
        resolution=resolution,
        title=title,
    )

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved simplex plot to {save_path}")

    return fig


def _load_ppo_policy_fn(policy_path, state_dim=4, n_actions=4):
    """Load a trained PPO policy and return a callable for querying it."""
    import torch
    from tianshou.data import Batch

    # Reconstruct network (must match training architecture)
    from remarc.agents.tianshou_agent import build_ppo_agent
    from remarc.core.hyperparameters import Presets

    p = Presets(
        state_shape=(state_dim,),
        num_actions=n_actions,
        lr=1e-4, epochs=1, train_steps_per_epoch=1,
        test_episodes=1, batch_size=8, buffer_size=50,
        dataset="four_state",
    )

    # We need the policy object to load weights into
    from remarc.agents.tianshou_agent import _build_ppo_policy
    policy = _build_ppo_policy(p)
    policy.load_state_dict(torch.load(policy_path, map_location="cpu", weights_only=True))
    policy.eval()

    def fn(state):
        obs = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            result = policy(Batch(obs=obs, info={}))
        return int(result.act.item())

    return fn


# ──────────────────────────────────────────────────────────────────────
#  Genotype Frequency Heatmaps
# ──────────────────────────────────────────────────────────────────────

def plot_frequency_heatmaps(frequencies_at_times, time_steps, filename="wf_frequency_heatmap.png"):
    num_panels = len(time_steps)
    fig, axes = plt.subplots(1, num_panels, figsize=(3 * num_panels, 3))

    if num_panels == 1:
        axes = [axes]

    vmax = max([np.max(f) for f in frequencies_at_times]) if frequencies_at_times else 1.0

    im = None
    for idx, (freqs, t) in enumerate(zip(frequencies_at_times, time_steps)):
        ax = axes[idx]
        grid = freqs.reshape((2, 4))
        im = ax.imshow(grid, cmap="viridis", origin="lower", vmin=0, vmax=vmax)
        ax.set_title(f"Gen {t}")
        ax.set_xticks(range(4))
        ax.set_yticks(range(2))

    if im is None:
        return
    fig.subplots_adjust(right=0.9)
    if num_panels > 1:
        cbar_ax = fig.add_axes((0.92, 0.15, 0.02, 0.7))
        fig.colorbar(im, cax=cbar_ax, label="Frequency")
    else:
        plt.colorbar(im, ax=axes[0], label="Frequency")

    plt.suptitle("Wright-Fisher Genotype Frequency Evolution (Drug 0)", fontsize=14)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Heatmap plot saved to {filename}")
    plt.close()


# ──────────────────────────────────────────────────────────────────────
#  CLI entry point
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate simplex policy plots")
    parser.add_argument("--policy", type=str, default=None, help="Path to trained .pth policy")
    parser.add_argument("--resolution", type=int, default=60, help="Grid resolution")
    parser.add_argument("--output", type=str, default="simplex_policy.png", help="Output file")
    parser.add_argument("--greedy", action="store_true", help="Plot greedy baseline")
    args = parser.parse_args()

    if args.greedy or args.policy is None:
        fig = plot_four_state_simplex(
            resolution=args.resolution,
            save_path=args.output,
            title="Greedy Policy on 4-Genotype Simplex",
        )
    else:
        fig = plot_four_state_simplex(
            policy_path=args.policy,
            resolution=args.resolution,
            save_path=args.output,
            title="Trained RL Policy on 4-Genotype Simplex",
        )
    plt.show()
