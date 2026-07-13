"""
Plotting utilities for EvoDM / REMARC.

Includes:
- Simplex slice visualization for four-genotype policy (Fig 3 from paper)
- Genotype frequency heatmaps
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import to_rgb
import matplotlib as mpl
from matplotlib.tri import Triangulation
from matplotlib.patches import Patch
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle


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
    is_three_state=False,
    scatter_states=None,
):
    """
    Visualize a drug-switching policy on the simplex.

    For a system with 4 genotypes, the population state lives on the
    3-simplex (x₀ + x₁ + x₂ + x₃ = 1). We slice along x₃ and show
    each cross-section as a colored triangle.
    For a system with 3 genotypes, the population state lives exactly on the
    2-simplex (x₀ + x₁ + x₂ = 1), so no slices are needed.
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
        scatter_states: optional ndarray of states to scatter plot on the simplex

    Returns:
        fig: matplotlib Figure (None if ax_array was supplied)
    """
    # ── defaults ──
    if drug_labels is None:
        drug_labels = [f"Drug {chr(65 + i)}" for i in range(num_drugs)]
    if drug_colors is None:
        drug_colors = [
            "#2ecc71",
            "#e67e22",
            "#5b7cc9",
            "#e84393",
            "#f1c40f",
            "#1abc9c",
            "#9b59b6",
            "#e74c3c",
        ][:num_drugs]
    if genotype_labels is None:
        genotype_labels = [f"genotype {i}" for i in range(4)]
    if x3_slices is None and not is_three_state:
        x3_slices = [
            (0.5, 1.0),
            (0.4, 0.5),
            (0.3, 0.4),
            (0.2, 0.3),
            (0.1, 0.2),
            (0.0, 0.1),
        ]
    elif is_three_state:
        x3_slices = [(0.0, 1.0)]  # dummy slice

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
            if is_three_state:
                state = np.array([l0, l1, l2], dtype=np.float32)
            else:
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

        pc = PolyCollection(
            tri_verts, facecolors=colors_rgba, edgecolors="none", linewidths=0
        )
        ax.add_collection(pc)

        if scatter_states is not None:
            if is_three_state:
                sub = np.array(scatter_states)
                l0, l1, l2 = sub[:, 0], sub[:, 1], sub[:, 2]
                S = l0 + l1 + l2
                S[S < 1e-9] = 1.0
                x = 0.5 * (2 * l1 + l2) / S
                y = (_SQRT3_2 * l2) / S
                ax.scatter(
                    x,
                    y,
                    color="white",
                    alpha=0.8,
                    s=15,
                    edgecolors="black",
                    linewidth=0.5,
                    zorder=6,
                )
            else:
                sub = np.array(scatter_states)
                mask = (sub[:, 3] >= x3_lo) & (sub[:, 3] <= x3_hi)
                if np.any(mask):
                    sub = sub[mask]
                    l0, l1, l2 = sub[:, 0], sub[:, 1], sub[:, 2]
                    S = l0 + l1 + l2
                    S[S < 1e-9] = 1.0
                    x = 0.5 * (2 * l1 + l2) / S
                    y = (_SQRT3_2 * l2) / S
                    ax.scatter(
                        x,
                        y,
                        color="white",
                        alpha=0.8,
                        s=15,
                        edgecolors="black",
                        linewidth=0.5,
                        zorder=6,
                    )

        # ── triangle outline ──
        outline_x = [0, 1, 0.5, 0]
        outline_y = [0, 0, _SQRT3_2, 0]
        ax.plot(outline_x, outline_y, color="black", linewidth=1.2, zorder=5)

        # ── slice label ──
        if not is_three_state:
            ax.text(
                0.5,
                -0.12,
                f"[{x3_lo:.1f}, {x3_hi:.1f}]",
                ha="center",
                va="top",
                fontsize=8,
                transform=ax.transAxes,
            )

        # ── vertex labels on first and last slice ──
        if s_idx == 0:
            ax.text(
                -0.04,
                -0.06,
                genotype_labels[0],
                ha="center",
                fontsize=7,
                style="italic",
            )
            top_label = genotype_labels[2] if is_three_state else genotype_labels[3]
            ax.text(
                0.5, _SQRT3_2 + 0.06, top_label, ha="center", fontsize=7, style="italic"
            )
        if s_idx == n_slices - 1:
            ax.text(
                1.04, -0.06, genotype_labels[1], ha="center", fontsize=7, style="italic"
            )
            if not is_three_state:
                ax.text(
                    0.5,
                    _SQRT3_2 + 0.06,
                    genotype_labels[2],
                    ha="center",
                    fontsize=7,
                    style="italic",
                )

        ax.set_xlim(-0.08, 1.08)
        ax.set_ylim(-0.15, _SQRT3_2 + 0.15)
        ax.set_aspect("equal")
        ax.axis("off")

    # ── x₃ arrow ──
    if fig is not None and not is_three_state:
        fig.text(
            0.10,
            0.06,
            "x₃ = 1",
            ha="left",
            fontsize=9,
            color="gray",
        )
        fig.text(
            0.82,
            0.06,
            "x₃ = 0",
            ha="right",
            fontsize=9,
            color="gray",
        )
        fig.text(
            0.46,
            0.06,
            f"← {genotype_labels[3]} →",
            ha="center",
            fontsize=9,
            color="gray",
        )

    # ── legend ──
    if show_legend:
        legend_elements = [
            Patch(
                facecolor=drug_colors[i],
                edgecolor="gray",
                linewidth=0.5,
                label=drug_labels[i],
            )
            for i in range(num_drugs)
        ]
        if fig is not None:
            fig.legend(
                handles=legend_elements,
                loc="center right",
                bbox_to_anchor=(0.98, 0.55),
                fontsize=9,
                framealpha=0.9,
                edgecolor="lightgray",
            )
        else:
            ax_array[-1].legend(
                handles=legend_elements,
                loc="upper right",
                fontsize=8,
                framealpha=0.9,
            )

    if title and fig is not None:
        fig.suptitle(title, fontsize=12, y=0.97)
    if fig is not None:
        fig.subplots_adjust(wspace=0.02, left=0.03, right=0.87, bottom=0.12, top=0.90)

    return fig


def plot_policy_difference_slices(
    policy_fn_1,
    policy_fn_2,
    num_drugs=4,
    resolution=60,
    x3_slices=None,
    genotype_labels=None,
    title="Policy Difference",
    figsize=None,
    is_three_state=False,
):
    """
    Visualize regions of agreement vs disagreement between two policies.
    Green = policies select the SAME drug
    Red = policies select DIFFERENT drugs
    """
    if genotype_labels is None:
        genotype_labels = [f"genotype {i}" for i in range(4)]
    if x3_slices is None and not is_three_state:
        x3_slices = [
            (0.5, 1.0),
            (0.4, 0.5),
            (0.3, 0.4),
            (0.2, 0.3),
            (0.1, 0.2),
            (0.0, 0.1),
        ]
    elif is_three_state:
        x3_slices = [(0.0, 1.0)]

    n_slices = len(x3_slices)
    if figsize is None:
        figsize = (2.6 * n_slices + 1.5, 3.8)

    fig, ax_array = plt.subplots(1, n_slices, figsize=figsize)
    if n_slices == 1:
        ax_array = [ax_array]

    bary, x_cart, y_cart, tri_idx = _make_simplex_grid(resolution)

    # Custom colormap for difference (0 = different = Red, 1 = same = Green)
    cmap = mcolors.ListedColormap(["#e74c3c", "#2ecc71"])
    bounds = [-0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    for s_idx, (x3_lo, x3_hi) in enumerate(x3_slices):
        ax = ax_array[s_idx]
        x3_mid = (x3_lo + x3_hi) / 2.0
        S = 1.0 - x3_mid

        # query policies at every grid point
        actions_1 = np.zeros(len(bary), dtype=int)
        actions_2 = np.zeros(len(bary), dtype=int)
        for p_idx in range(len(bary)):
            l0, l1, l2 = bary[p_idx]
            if is_three_state:
                state = np.array([l0, l1, l2], dtype=np.float32)
            else:
                state = np.array([l0 * S, l1 * S, l2 * S, x3_mid], dtype=np.float32)
            actions_1[p_idx] = policy_fn_1(state)
            actions_2[p_idx] = policy_fn_2(state)

        # color each triangle by agreement
        tri_agreements = np.zeros(len(tri_idx), dtype=int)
        for t_idx, (v0, v1, v2) in enumerate(tri_idx):
            a1_votes = [actions_1[v0], actions_1[v1], actions_1[v2]]
            a2_votes = [actions_2[v0], actions_2[v1], actions_2[v2]]
            # If majority vote matches, it's an agreement
            tri_a1 = max(set(a1_votes), key=a1_votes.count)
            tri_a2 = max(set(a2_votes), key=a2_votes.count)
            tri_agreements[t_idx] = 1 if tri_a1 == tri_a2 else 0

        verts = np.column_stack([x_cart, y_cart])
        tri_verts = verts[tri_idx]
        colors_rgba = [cmap(norm(a)) for a in tri_agreements]

        pc = PolyCollection(
            tri_verts, facecolors=colors_rgba, edgecolors="none", linewidths=0
        )
        ax.add_collection(pc)

        outline_x = [0, 1, 0.5, 0]
        outline_y = [0, 0, _SQRT3_2, 0]
        ax.plot(outline_x, outline_y, color="black", linewidth=1.2, zorder=5)

        ax.text(
            0.5,
            -0.12,
            f"[{x3_lo:.1f}, {x3_hi:.1f}]",
            ha="center",
            va="top",
            fontsize=8,
            transform=ax.transAxes,
        )

        if s_idx == 0:
            ax.text(
                -0.04,
                -0.06,
                genotype_labels[0],
                ha="center",
                fontsize=7,
                style="italic",
            )
            top_label = genotype_labels[2] if is_three_state else genotype_labels[3]
            ax.text(
                0.5, _SQRT3_2 + 0.06, top_label, ha="center", fontsize=7, style="italic"
            )
        if s_idx == n_slices - 1:
            ax.text(
                1.04, -0.06, genotype_labels[1], ha="center", fontsize=7, style="italic"
            )
            ax.text(
                0.5,
                _SQRT3_2 + 0.06,
                genotype_labels[2],
                ha="center",
                fontsize=7,
                style="italic",
            )

        ax.set_xlim(-0.08, 1.08)
        ax.set_ylim(-0.15, _SQRT3_2 + 0.15)
        ax.set_aspect("equal")
        ax.axis("off")

    if not is_three_state:
        fig.text(0.10, 0.06, "x₃ = 1", ha="left", fontsize=9, color="gray")
        fig.text(0.82, 0.06, "x₃ = 0", ha="right", fontsize=9, color="gray")
        fig.text(
            0.46,
            0.06,
            f"← {genotype_labels[3]} →",
            ha="center",
            fontsize=9,
            color="gray",
        )

    legend_elements = [
        Patch(facecolor="#2ecc71", edgecolor="gray", linewidth=0.5, label="Agreement"),
        Patch(
            facecolor="#e74c3c", edgecolor="gray", linewidth=0.5, label="Disagreement"
        ),
    ]
    fig.legend(
        handles=legend_elements,
        loc="center right",
        bbox_to_anchor=(0.98, 0.55),
        fontsize=9,
        framealpha=0.9,
        edgecolor="lightgray",
    )
    fig.suptitle(title, fontsize=12, y=0.97)
    fig.subplots_adjust(wspace=0.02, left=0.03, right=0.87, bottom=0.12, top=0.90)

    return fig


def plot_policy_magnitude_difference_slices(
    policy_fn_1,
    policy_fn_2,
    landscapes,
    num_drugs=4,
    resolution=60,
    x3_slices=None,
    genotype_labels=None,
    title="Policy Magnitude Difference",
    figsize=None,
    is_three_state=False,
):
    """
    Visualize magnitude of disagreement between two policies.
    Intensity of red represents magnitude of difference in fitness between the population under each drug.
    """
    if genotype_labels is None:
        genotype_labels = [f"genotype {i}" for i in range(4)]
    if x3_slices is None and not is_three_state:
        x3_slices = [
            (0.5, 1.0),
            (0.4, 0.5),
            (0.3, 0.4),
            (0.2, 0.3),
            (0.1, 0.2),
            (0.0, 0.1),
        ]
    elif is_three_state:
        x3_slices = [(0.0, 1.0)]

    n_slices = len(x3_slices)
    if figsize is None:
        figsize = (2.6 * n_slices + 1.5, 3.8)

    fig, ax_array = plt.subplots(1, n_slices, figsize=figsize)
    if n_slices == 1:
        ax_array = [ax_array]

    bary, x_cart, y_cart, tri_idx = _make_simplex_grid(resolution)

    # Pre-calculate all differences to find the global max for the colormap
    slice_diffs = []

    # Calculate normalization denominator to match trajectory plots
    g_min = np.min(landscapes)
    g_max = np.max(landscapes)
    denom = (g_max - g_min) if g_max != g_min else 1.0

    for x3_lo, x3_hi in x3_slices:
        x3_mid = (x3_lo + x3_hi) / 2.0
        S = 1.0 - x3_mid

        actions_1 = np.zeros(len(bary), dtype=int)
        actions_2 = np.zeros(len(bary), dtype=int)
        for p_idx in range(len(bary)):
            l0, l1, l2 = bary[p_idx]
            if is_three_state:
                state = np.array([l0, l1, l2], dtype=np.float32)
            else:
                state = np.array([l0 * S, l1 * S, l2 * S, x3_mid], dtype=np.float32)
            actions_1[p_idx] = policy_fn_1(state)
            actions_2[p_idx] = policy_fn_2(state)

        tri_diffs = np.zeros(len(tri_idx), dtype=float)
        for t_idx, (v0, v1, v2) in enumerate(tri_idx):
            a1_votes = [actions_1[v0], actions_1[v1], actions_1[v2]]
            a2_votes = [actions_2[v0], actions_2[v1], actions_2[v2]]
            tri_a1 = max(set(a1_votes), key=a1_votes.count)
            tri_a2 = max(set(a2_votes), key=a2_votes.count)

            if tri_a1 != tri_a2:
                # Approximate state at the triangle's centroid
                cb = (bary[v0] + bary[v1] + bary[v2]) / 3.0
                if is_three_state:
                    state = np.array([cb[0], cb[1], cb[2]])
                else:
                    state = np.array([cb[0] * S, cb[1] * S, cb[2] * S, x3_mid])
                fitness_1 = np.dot(landscapes[tri_a1], state)
                fitness_2 = np.dot(landscapes[tri_a2], state)
                # Normalize difference by the landscape range
                tri_diffs[t_idx] = abs(fitness_1 - fitness_2) / denom
            else:
                tri_diffs[t_idx] = 0.0
        slice_diffs.append(tri_diffs)

    global_max = max([np.max(diffs) for diffs in slice_diffs]) if slice_diffs else 0.0
    if global_max <= 1e-6:
        global_max = 0.1  # fallback if policies are identical

    # Colormap for magnitude difference (White/Yellow -> Red)
    cmap = mpl.colormaps["YlOrRd"]
    norm = mcolors.Normalize(vmin=0.0, vmax=global_max)

    for s_idx, (x3_lo, x3_hi) in enumerate(x3_slices):
        ax = ax_array[s_idx]
        x3_mid = (x3_lo + x3_hi) / 2.0
        tri_diffs = slice_diffs[s_idx]

        verts = np.column_stack([x_cart, y_cart])
        tri_verts = verts[tri_idx]
        colors_rgba = [cmap(norm(a)) for a in tri_diffs]

        pc = PolyCollection(
            tri_verts, facecolors=colors_rgba, edgecolors="none", linewidths=0
        )
        ax.add_collection(pc)

        outline_x = [0, 1, 0.5, 0]
        outline_y = [0, 0, _SQRT3_2, 0]
        ax.plot(outline_x, outline_y, color="black", linewidth=1.2, zorder=5)

        if not is_three_state:
            ax.text(
                0.5,
                -0.12,
                f"[{x3_lo:.1f}, {x3_hi:.1f}]",
                ha="center",
                va="top",
                fontsize=8,
                transform=ax.transAxes,
            )

        if s_idx == 0:
            ax.text(
                -0.04,
                -0.06,
                genotype_labels[0],
                ha="center",
                fontsize=7,
                style="italic",
            )
            top_label = genotype_labels[2] if is_three_state else genotype_labels[3]
            ax.text(
                0.5, _SQRT3_2 + 0.06, top_label, ha="center", fontsize=7, style="italic"
            )
        if s_idx == n_slices - 1:
            ax.text(
                1.04, -0.06, genotype_labels[1], ha="center", fontsize=7, style="italic"
            )
            if not is_three_state:
                ax.text(
                    0.5,
                    _SQRT3_2 + 0.06,
                    genotype_labels[2],
                    ha="center",
                    fontsize=7,
                    style="italic",
                )

        ax.set_xlim(-0.08, 1.08)
        ax.set_ylim(-0.15, _SQRT3_2 + 0.15)
        ax.set_aspect("equal")
        ax.axis("off")

    if not is_three_state:
        fig.text(0.10, 0.06, "x₃ = 1", ha="left", fontsize=9, color="gray")
        fig.text(0.82, 0.06, "x₃ = 0", ha="right", fontsize=9, color="gray")
        fig.text(
            0.46,
            0.06,
            f"← {genotype_labels[3]} →",
            ha="center",
            fontsize=9,
            color="gray",
        )

    # ── colorbar instead of legend ──
    cbar_ax = fig.add_axes([0.90, 0.25, 0.015, 0.5])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Fitness Difference")
    fig.suptitle(title, fontsize=12, y=0.97)
    fig.subplots_adjust(wspace=0.02, left=0.03, right=0.87, bottom=0.12, top=0.90)

    return fig


def plot_policy_fitness_landscape_slices(
    policy_fn,
    landscapes,
    num_drugs=4,
    resolution=60,
    x3_slices=None,
    genotype_labels=None,
    title="Normalized Fitness Landscape (RL Policy)",
    figsize=None,
    is_three_state=False,
):
    """
    Visualize normalized fitness of the population under the drug selected by the policy.
    0.0 = minimum possible fitness across all landscapes (most effective)
    1.0 = maximum possible fitness across all landscapes (least effective)
    """
    if genotype_labels is None:
        genotype_labels = [f"genotype {i}" for i in range(4)]
    if x3_slices is None and not is_three_state:
        x3_slices = [
            (0.5, 1.0),
            (0.4, 0.5),
            (0.3, 0.4),
            (0.2, 0.3),
            (0.1, 0.2),
            (0.0, 0.1),
        ]
    elif is_three_state:
        x3_slices = [(0.0, 1.0)]

    n_slices = len(x3_slices)
    if figsize is None:
        figsize = (2.6 * n_slices + 1.5, 3.8)

    fig, ax_array = plt.subplots(1, n_slices, figsize=figsize)
    if n_slices == 1:
        ax_array = [ax_array]

    bary, x_cart, y_cart, tri_idx = _make_simplex_grid(resolution)

    # Calculate normalization denominator
    g_min = np.min(landscapes)
    g_max = np.max(landscapes)
    denom = (g_max - g_min) if g_max != g_min else 1.0

    # Plasma colormap gives contrast (don't want to overuse viridis haha)
    cmap = mpl.colormaps["plasma"]
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    for s_idx, (x3_lo, x3_hi) in enumerate(x3_slices):
        ax = ax_array[s_idx]
        x3_mid = (x3_lo + x3_hi) / 2.0
        S = 1.0 - x3_mid

        actions = np.zeros(len(bary), dtype=int)
        for p_idx in range(len(bary)):
            l0, l1, l2 = bary[p_idx]
            if is_three_state:
                state = np.array([l0, l1, l2], dtype=np.float32)
            else:
                state = np.array([l0 * S, l1 * S, l2 * S, x3_mid], dtype=np.float32)
            actions[p_idx] = policy_fn(state)

        tri_fitness = np.zeros(len(tri_idx), dtype=float)
        for t_idx, (v0, v1, v2) in enumerate(tri_idx):
            a_votes = [actions[v0], actions[v1], actions[v2]]
            tri_a = max(set(a_votes), key=a_votes.count)

            # Approximate state at the triangle's centroid
            cb = (bary[v0] + bary[v1] + bary[v2]) / 3.0
            if is_three_state:
                state = np.array([cb[0], cb[1], cb[2]])
            else:
                state = np.array([cb[0] * S, cb[1] * S, cb[2] * S, x3_mid])
            fitness = np.dot(landscapes[tri_a], state)

            # Normalize fitness to [0, 1]
            tri_fitness[t_idx] = (fitness - g_min) / denom

        verts = np.column_stack([x_cart, y_cart])
        tri_verts = verts[tri_idx]
        colors_rgba = [cmap(norm(f)) for f in tri_fitness]

        pc = PolyCollection(
            tri_verts, facecolors=colors_rgba, edgecolors="none", linewidths=0
        )
        ax.add_collection(pc)

        # Draw policy boundaries
        levels = np.arange(0.5, num_drugs, 1.0)
        ax.tricontour(
            Triangulation(x_cart, y_cart, tri_idx),
            actions,
            levels=levels,
            colors="red",
            linewidths=1.5,
            linestyles="--",
            zorder=10,
        )

        outline_x = [0, 1, 0.5, 0]
        outline_y = [0, 0, _SQRT3_2, 0]
        ax.plot(outline_x, outline_y, color="black", linewidth=1.2, zorder=5)

        ax.text(
            0.5,
            -0.12,
            f"[{x3_lo:.1f}, {x3_hi:.1f}]",
            ha="center",
            va="top",
            fontsize=8,
            transform=ax.transAxes,
        )

        if s_idx == 0:
            ax.text(
                -0.04,
                -0.06,
                genotype_labels[0],
                ha="center",
                fontsize=7,
                style="italic",
            )
            top_label = genotype_labels[2] if is_three_state else genotype_labels[3]
            ax.text(
                0.5, _SQRT3_2 + 0.06, top_label, ha="center", fontsize=7, style="italic"
            )
        if s_idx == n_slices - 1:
            ax.text(
                1.04, -0.06, genotype_labels[1], ha="center", fontsize=7, style="italic"
            )
            ax.text(
                0.5,
                _SQRT3_2 + 0.06,
                genotype_labels[2],
                ha="center",
                fontsize=7,
                style="italic",
            )

        ax.set_xlim(-0.08, 1.08)
        ax.set_ylim(-0.15, _SQRT3_2 + 0.15)
        ax.set_aspect("equal")
        ax.axis("off")

    if not is_three_state:
        fig.text(0.10, 0.06, "x₃ = 1", ha="left", fontsize=9, color="gray")
        fig.text(0.82, 0.06, "x₃ = 0", ha="right", fontsize=9, color="gray")
        fig.text(
            0.46,
            0.06,
            f"← {genotype_labels[3]} →",
            ha="center",
            fontsize=9,
            color="gray",
        )

    cbar_ax = fig.add_axes([0.90, 0.25, 0.015, 0.5])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Normalized Tumor Fitness")
    fig.suptitle(title, fontsize=12, y=0.97)
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
        policy_fn = _load_ppo_policy_fn(
            policy_path, state_dim=4, n_actions=len(landscapes)
        )

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
        target_fig = fig if fig is not None else plt.gcf()
        target_fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved simplex plot to {save_path}")

    return fig


def _load_ppo_policy_fn(policy_path, state_dim=4, n_actions=4, activation="relu"):
    """Load a trained PPO policy and return a callable for querying it.

    Reconstructs the same network architecture used in
    ``remarc.agents.tianshou_agent.get_ppo_policy`` (Net[64,64] →
    Actor[32] + Critic[32] → PPOPolicy) so we can load saved weights
    without needing a live environment.
    """
    import torch
    from torch.optim import Adam
    from tianshou.data import Batch
    from tianshou.utils.net.common import Net
    from tianshou.utils.net.discrete import Actor, Critic
    from tianshou.policy import PPOPolicy
    from remarc.agents.tianshou_agent import get_activation
    import gymnasium as gym

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    act_cls = get_activation(activation)

    net = Net(
        state_shape=(state_dim,),
        hidden_sizes=[256, 256, 256],
        activation=act_cls,
        device=device,
    )
    actor = Actor(
        preprocess_net=net,
        action_shape=n_actions,
        hidden_sizes=[128],
        device=device,
    ).to(device)
    critic = Critic(
        preprocess_net=net,
        hidden_sizes=[128],
        device=device,
    ).to(device)

    optim = Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-4)

    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=torch.distributions.Categorical,
        action_space=gym.spaces.Discrete(n_actions),
        discount_factor=0.99,
        deterministic_eval=True,
        action_scaling=False,
    )

    policy.load_state_dict(
        torch.load(policy_path, map_location="cpu", weights_only=True)
    )
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


def plot_frequency_heatmaps(
    frequencies_at_times, time_steps, filename="wf_frequency_heatmap.png"
):
    num_panels = len(time_steps)
    fig, axes = plt.subplots(1, num_panels, figsize=(3 * num_panels, 3))

    if num_panels == 1:
        axes = [axes]

    vmax = (
        max([np.max(f) for f in frequencies_at_times]) if frequencies_at_times else 1.0
    )

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
#  Population Density Visualization
# ──────────────────────────────────────────────────────────────────────


def plot_population_density_slices(
    state_trajectories,
    policy_fn=None,
    greedy_policy_fn=None,
    num_drugs=4,
    resolution=60,
    x3_slices=None,
    figsize=None,
    is_three_state=False,
):
    """
    Plot the population density from trajectories as a heatmap on the 2-simplex slices.
    """
    if x3_slices is None and not is_three_state:
        x3_slices = [
            (0.5, 1.0),
            (0.4, 0.5),
            (0.3, 0.4),
            (0.2, 0.3),
            (0.1, 0.2),
            (0.0, 0.1),
        ]
    elif is_three_state:
        x3_slices = [(0.0, 1.0)]

    n_slices = len(x3_slices)
    if figsize is None:
        figsize = (2.6 * n_slices + 1.5, 3.8)

    fig, ax_array = plt.subplots(1, n_slices, figsize=figsize)
    if n_slices == 1:
        ax_array = [ax_array]

    bary, x_cart, y_cart, tri_idx = _make_simplex_grid(resolution)

    # Flatten all states from all episodes into a single array
    all_states = []
    for episode in state_trajectories:
        all_states.extend(episode)
    all_states = np.array(all_states)  # Shape: (N, 4)

    # Build a Triangulation and TriFinder to map Cartesian points to triangles
    triangulation = Triangulation(x_cart, y_cart, tri_idx)
    trifinder = triangulation.get_trifinder()

    # Pre-calculate counts to find global max for the colormap
    slice_counts = []
    for s_idx, (x3_lo, x3_hi) in enumerate(x3_slices):
        # Filter states by x3 slice
        if is_three_state:
            states_in_slice = all_states
        else:
            if s_idx == 0:
                mask = (all_states[:, 3] >= x3_lo) & (all_states[:, 3] <= x3_hi)
            else:
                mask = (all_states[:, 3] >= x3_lo) & (all_states[:, 3] < x3_hi)
            states_in_slice = all_states[mask]

        counts = np.zeros(len(tri_idx), dtype=float)

        if len(states_in_slice) > 0:
            if is_three_state:
                l0 = states_in_slice[:, 0]
                l1 = states_in_slice[:, 1]
                l2 = states_in_slice[:, 2]
            else:
                x3_mid = (x3_lo + x3_hi) / 2.0
                S = 1.0 - x3_mid

                if S > 0:
                    l0 = states_in_slice[:, 0] / S
                    l1 = states_in_slice[:, 1] / S
                    l2 = states_in_slice[:, 2] / S
                else:
                    l0 = np.zeros_like(states_in_slice[:, 0])
                    l1 = np.zeros_like(states_in_slice[:, 1])
                    l2 = np.zeros_like(states_in_slice[:, 2])

                # Normalize just in case of float precision issues
                tot = l0 + l1 + l2
                tot[tot == 0] = 1.0
                l0 /= tot
                l1 /= tot
                l2 /= tot

                cx, cy = _bary_to_cart(l0, l1, l2)
                found_tri_indices = trifinder(cx, cy)

                valid_idx = found_tri_indices[found_tri_indices != -1]
                for tidx in valid_idx:
                    counts[tidx] += 1.0

        slice_counts.append(counts)

    total_states = len(all_states) if len(all_states) > 0 else 1.0
    global_max = (
        max([np.max(c) for c in slice_counts]) / total_states if slice_counts else 0.0
    )

    vmin = 1e-4
    if global_max <= vmin:
        global_max = vmin * 10.0

    cmap = mpl.colormaps["viridis"]
    norm = mcolors.LogNorm(vmin=vmin, vmax=global_max)

    for s_idx, (x3_lo, x3_hi) in enumerate(x3_slices):
        ax = ax_array[s_idx]
        counts = slice_counts[s_idx] / total_states

        # Clip to vmin to avoid log(0) issues and map zeros to the bottom of the colormap
        counts = np.clip(counts, a_min=vmin, a_max=None)

        verts = np.column_stack([x_cart, y_cart])
        tri_verts = verts[tri_idx]
        colors_rgba = [cmap(norm(c)) for c in counts]

        pc = PolyCollection(
            tri_verts, facecolors=colors_rgba, edgecolors="none", linewidths=0
        )
        ax.add_collection(pc)

        if policy_fn is not None or greedy_policy_fn is not None:
            actions = np.zeros(len(bary), dtype=int)
            greedy_actions = np.zeros(len(bary), dtype=int)
            x3_mid = (x3_lo + x3_hi) / 2.0
            S = 1.0 - x3_mid
            for p_idx in range(len(bary)):
                l0, l1, l2 = bary[p_idx]
                if is_three_state:
                    state = np.array([l0, l1, l2], dtype=np.float32)
                else:
                    state = np.array([l0 * S, l1 * S, l2 * S, x3_mid], dtype=np.float32)
                if policy_fn is not None:
                    actions[p_idx] = policy_fn(state)
                if greedy_policy_fn is not None:
                    greedy_actions[p_idx] = greedy_policy_fn(state)

            levels = np.arange(0.5, num_drugs, 1.0)

            if policy_fn is not None:
                # Draw boundary lines where policy changes actions
                ax.tricontour(
                    triangulation,
                    actions,
                    levels=levels,
                    colors="red",
                    linewidths=1.5,
                    linestyles="--",
                    zorder=10,
                )
                # Debug: Add the unique actions found to the title
                unique_a = np.unique(actions)
                ax.set_title(
                    f"Genotype 3\n[{x3_lo:.1f}, {x3_hi:.1f})\nActs: {unique_a}",
                    fontsize=8,
                )
            else:
                unique_a = np.unique(greedy_actions)
                ax.set_title(
                    f"Genotype 3\n[{x3_lo:.1f}, {x3_hi:.1f})\nGreedy Acts: {unique_a}",
                    fontsize=8,
                )
        else:
            ax.set_title(f"Genotype 3\n[{x3_lo:.1f}, {x3_hi:.1f})", fontsize=10)

        outline_x = [0, 1, 0.5, 0]
        outline_y = [0, 0, _SQRT3_2, 0]
        ax.plot(outline_x, outline_y, color="black", linewidth=1.2, zorder=5)

        ax.set_aspect("equal")
        ax.axis("off")

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Relative Density (Log Scale)")

    return fig


## Dimensionality reduction + PCA visualization of trajectories


# HELPER: flatten state trajectories into a single array, with optional burn-in and stride
def flatten_state_trajectories(
    state_trajectories, burn_in=0, stride=1, max_runs=None, max_steps=None
):
    runs = state_trajectories[:max_runs] if max_runs is not None else state_trajectories
    samples = []
    run_ids = []
    time_ids = []

    for r, traj in enumerate(runs):
        arr = np.asarray(traj, dtype=float)
        if max_steps is not None:
            arr = arr[:max_steps]
        arr = arr[burn_in::stride]
        samples.append(arr)
        run_ids.extend([r] * len(arr))
        time_ids.extend(list(range(burn_in, burn_in + stride * len(arr), stride)))

    if len(samples) == 0:
        raise ValueError(
            "No trajectory samples available after burn_in/stride filtering."
        )

    X = np.vstack(samples)
    return X, np.asarray(run_ids), np.asarray(time_ids)


# HELPER: get mean trajectory across runs, optionally with burn-in and stride

def std_trajectory(state_trajectories, burn_in=0, stride=1, max_steps=None):
    arr = np.asarray(state_trajectories, dtype=float)  # (runs, steps, M)
    if max_steps is not None:
        arr = arr[:, :max_steps, :]
    arr = arr[:, burn_in::stride, :]
    return arr.std(axis=0)  # (steps, M)

def mean_trajectory(state_trajectories, burn_in=0, stride=1, max_steps=None):
    arr = np.asarray(state_trajectories, dtype=float)  # (runs, steps, M)
    if max_steps is not None:
        arr = arr[:, :max_steps, :]
    arr = arr[:, burn_in::stride, :]
    return arr.mean(axis=0)  # (steps, M)


# HELPER: Get CLR transformation of compositional data
def clr_transform(X, eps=1e-8):
    X = np.asarray(X, dtype=float)
    X = np.clip(X, eps, None)
    X = X / X.sum(axis=1, keepdims=True)
    logX = np.log(X)
    return logX - logX.mean(axis=1, keepdims=True)


# HELPER: PCA on CLR-transformed data
def run_pca(X, n_components=3):
    X = np.asarray(X, dtype=float)
    X_centered = X - X.mean(axis=0, keepdims=True)

    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    scores = U[:, :n_components] * S[:n_components]
    components = Vt[:n_components]

    var = (S**2) / (X.shape[0] - 1)
    explained_ratio = var / var.sum()
    cumulative = np.cumsum(explained_ratio)

    return {
        "scores": scores,
        "components": components,
        "explained_ratio": explained_ratio[:n_components],
        "cumulative_ratio": cumulative[:n_components],
        "mean": X.mean(axis=0),
        "all_explained_ratio": explained_ratio,
    }


def scores_to_barycentric(scores3, temperature=1.0):
    Z = np.asarray(scores3, dtype=float)
    Z = (Z - Z.mean(axis=0, keepdims=True)) / (Z.std(axis=0, keepdims=True) + 1e-12)
    Z = Z / max(temperature, 1e-12)

    Z = Z - Z.max(axis=1, keepdims=True)
    W = np.exp(Z)
    W /= W.sum(axis=1, keepdims=True)
    return W


def _flatten_aligned_trajectories(
    state_trajectories,
    policy_trajectories=None,
    fitness_trajectories=None,
    burn_in=0,
    stride=1,
    max_runs=None,
    max_steps=None,
):
    runs = state_trajectories[:max_runs] if max_runs is not None else state_trajectories

    X_list = []
    run_ids = []
    time_ids = []
    pol_list = []
    fit_list = []

    for r, traj in enumerate(runs):
        Xr = np.asarray(traj, dtype=float)
        if max_steps is not None:
            Xr = Xr[:max_steps]
        Xr = Xr[burn_in::stride]

        T = len(Xr)
        if T == 0:
            continue

        X_list.append(Xr)
        run_ids.extend([r] * T)
        time_ids.extend(np.arange(burn_in, burn_in + stride * T, stride))

        if policy_trajectories is not None:
            pr = np.asarray(policy_trajectories[r])
            if max_steps is not None:
                pr = pr[:max_steps]
            pr = pr[burn_in::stride]
            pol_list.append(pr)

        if fitness_trajectories is not None:
            fr = np.asarray(fitness_trajectories[r], dtype=float)
            if max_steps is not None:
                fr = fr[:max_steps]
            fr = fr[burn_in::stride]
            fit_list.append(fr)

    if len(X_list) == 0:
        raise ValueError("No samples left after filtering.")

    X = np.vstack(X_list)
    run_ids = np.asarray(run_ids)
    time_ids = np.asarray(time_ids)
    policies = np.concatenate(pol_list) if policy_trajectories is not None else None
    fitness = np.concatenate(fit_list) if fitness_trajectories is not None else None
    return X, run_ids, time_ids, policies, fitness


def _lighten_color(color, amount=0.5):
    c = np.array(to_rgb(color))
    white = np.ones(3)
    return tuple((1 - amount) * white + amount * c)


def _policy_boundary_plot(
    ax,
    x,
    y,
    policies,
    fitness,
    drug_colors,
    gridsize=60,
    min_points=3,
    boundary=True,
):
    x = np.asarray(x)
    y = np.asarray(y)
    policies = np.asarray(policies)
    fitness = np.asarray(fitness, dtype=float)

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    gx = np.linspace(xmin, xmax, gridsize + 1)
    gy = np.linspace(ymin, ymax, gridsize + 1)

    ix = np.clip(np.digitize(x, gx) - 1, 0, gridsize - 1)
    iy = np.clip(np.digitize(y, gy) - 1, 0, gridsize - 1)

    policy_grid = np.full((gridsize, gridsize), -1, dtype=int)
    fit_grid = np.full((gridsize, gridsize), np.nan, dtype=float)
    count_grid = np.zeros((gridsize, gridsize), dtype=int)

    for i in range(gridsize):
        for j in range(gridsize):
            mask = (ix == i) & (iy == j)
            n = mask.sum()
            count_grid[j, i] = n
            if n < min_points:
                continue

            pols = policies[mask].astype(int)
            vals, counts = np.unique(pols, return_counts=True)
            majority_policy = vals[np.argmax(counts)]
            policy_grid[j, i] = majority_policy
            fit_grid[j, i] = fitness[mask].mean()

    valid = np.isfinite(fit_grid)
    if not np.any(valid):
        raise ValueError("No valid bins for policy boundary plot.")

    fmin = np.nanmin(fit_grid)
    fmax = np.nanmax(fit_grid)
    denom = max(fmax - fmin, 1e-12)

    dx = gx[1] - gx[0]
    dy = gy[1] - gy[0]

    for j in range(gridsize):
        for i in range(gridsize):
            pol = policy_grid[j, i]
            if pol < 0:
                continue

            base = drug_colors[pol]
            f = fit_grid[j, i]
            normf = (f - fmin) / denom

            # lower fitness -> lighter, higher fitness -> closer to base color
            face = _lighten_color(base, amount=0.25 + 0.75 * normf)

            rect = Rectangle(
                (gx[i], gy[j]),
                dx,
                dy,
                facecolor=face,
                edgecolor="none",
                linewidth=0,
                zorder=1,
            )
            ax.add_patch(rect)

    if boundary:
        Xc = 0.5 * (gx[:-1] + gx[1:])
        Yc = 0.5 * (gy[:-1] + gy[1:])
        XX, YY = np.meshgrid(Xc, Yc)

        masked = np.ma.masked_where(policy_grid < 0, policy_grid)
        levels = np.arange(np.nanmax(policy_grid) + 2) - 0.5
        ax.contour(
            XX,
            YY,
            masked,
            levels=levels,
            colors="k",
            linewidths=0.8,
            alpha=0.75,
            zorder=2,
        )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    return {
        "policy_grid": policy_grid,
        "fitness_grid": fit_grid,
        "count_grid": count_grid,
        "fitness_min": fmin,
        "fitness_max": fmax,
    }


def clr_inverse(Z):
    Z = np.asarray(Z, dtype=float)
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def pca_inverse_transform(scores, pca):
    scores = np.asarray(scores, dtype=float)
    return scores @ pca["components"] + pca["mean"]


def _mix_with_white(color, t):
    """
    t in [0,1]. t=0 -> white, t=1 -> original color
    """
    rgb = np.array(mcolors.to_rgb(color))
    return tuple((1 - t) * np.ones(3) + t * rgb)


def _latent_policy_grid(
    pca,
    xlim,
    ylim,
    policy_fn,
    fitness_fn,
    grid_n=100,
    pc3_value=0.0,
):
    xs = np.linspace(xlim[0], xlim[1], grid_n)
    ys = np.linspace(ylim[0], ylim[1], grid_n)
    XX, YY = np.meshgrid(xs, ys)

    scores_grid = np.column_stack(
        [
            XX.ravel(),
            YY.ravel(),
            np.full(XX.size, pc3_value, dtype=float),
        ]
    )

    clr_grid = pca_inverse_transform(scores_grid, pca)
    comp_grid = clr_inverse(clr_grid)

    actions = np.array([policy_fn(state) for state in comp_grid], dtype=int)
    fitness = np.array(
        [fitness_fn(state, a) for state, a in zip(comp_grid, actions)], dtype=float
    )

    return XX, YY, actions.reshape(XX.shape), fitness.reshape(XX.shape)


def _draw_latent_policy_map(
    ax,
    XX,
    YY,
    action_grid,
    fitness_grid,
    num_drugs,
    drug_colors,
    show_boundary=True,
):
    valid = np.isfinite(fitness_grid)
    fmin = np.nanmin(fitness_grid[valid])
    fmax = np.nanmax(fitness_grid[valid])
    denom = max(fmax - fmin, 1e-12)

    nx = XX.shape[1]
    ny = XX.shape[0]
    dx = XX[0, 1] - XX[0, 0]
    dy = YY[1, 0] - YY[0, 0]

    for j in range(ny):
        for i in range(nx):
            a = int(action_grid[j, i])
            if a < 0 or a >= num_drugs:
                continue

            # lower fitness -> lighter, higher fitness -> more saturated
            normf = (fitness_grid[j, i] - fmin) / denom
            face = _mix_with_white(drug_colors[a], 0.20 + 0.80 * normf)

            ax.add_patch(
                Rectangle(
                    (XX[j, i] - dx / 2, YY[j, i] - dy / 2),
                    dx,
                    dy,
                    facecolor=face,
                    edgecolor="none",
                    linewidth=0,
                    zorder=1,
                )
            )

    if show_boundary:
        levels = np.arange(-0.5, num_drugs + 0.5, 1)
        ax.contour(
            XX,
            YY,
            action_grid,
            levels=levels,
            colors="k",
            linewidths=0.7,
            alpha=0.7,
            zorder=2,
        )

    return fmin, fmax


def plot_dominant_modes(
    state_trajectories,
    policy_fn=None,
    fitness_fn=None,
    num_drugs=4,
    drug_labels=None,
    drug_colors=None,
    genotype_labels=None,
    burn_in=0,
    stride=1,
    max_runs=None,
    max_steps=None,
    eps=1e-8,
    explained_threshold=0.88,
    show_mean_trajectory=True,
    show_sample_paths=False,
    max_paths=20,
    density=True,
    simplex_view=True,
    title="Dominant Tumor Modes",
    latent_grid_n=100,
    latent_pc3="mean",
    show_policy_boundary=True,
    show_legend=True,
):
    # defaults aligned with plot_simplex_policy_slices
    if drug_labels is None:
        drug_labels = [f"Drug {chr(65 + i)}" for i in range(num_drugs)]
    if drug_colors is None:
        drug_colors = [
            "#2ecc71",
            "#e67e22",
            "#5b7cc9",
            "#e84393",
            "#f1c40f",
            "#1abc9c",
            "#9b59b6",
            "#e74c3c",
        ][:num_drugs]

    X_raw, run_ids, time_ids = flatten_state_trajectories(
        state_trajectories,
        burn_in=burn_in,
        stride=stride,
        max_runs=max_runs,
        max_steps=max_steps,
    )

    X_clr = clr_transform(X_raw, eps=eps)
    pca = run_pca(X_clr, n_components=3)
    scores = pca["scores"]
    evr = pca["explained_ratio"]
    cev = pca["cumulative_ratio"]

    mean_traj = mean_trajectory(
        state_trajectories,
        burn_in=burn_in,
        stride=stride,
        max_steps=max_steps,
    )

    mean_scores = None
    if show_mean_trajectory:
        mean_scores = (clr_transform(mean_traj, eps=eps) - pca["mean"]) @ pca[
            "components"
        ].T

    fig, axes = plt.subplots(1, 1, figsize=(6, 5))
    if simplex_view and cev[2] >= explained_threshold:
        axes = [axes, "dummy"]
    else:
        axes = [axes]

    ax = axes[0]

    use_policy_map = (policy_fn is not None) and (fitness_fn is not None)

    if use_policy_map:
        xpad = 0.05 * max(scores[:, 0].ptp(), 1e-6)
        ypad = 0.05 * max(scores[:, 1].ptp(), 1e-6)
        xlim = (scores[:, 0].min() - xpad, scores[:, 0].max() + xpad)
        ylim = (scores[:, 1].min() - ypad, scores[:, 1].max() + ypad)

        if latent_pc3 == "mean":
            pc3_value = float(np.mean(scores[:, 2]))
        elif latent_pc3 == "median":
            pc3_value = float(np.median(scores[:, 2]))
        else:
            pc3_value = float(latent_pc3)

        XX, YY, action_grid, fitness_grid = _latent_policy_grid(
            pca=pca,
            xlim=xlim,
            ylim=ylim,
            policy_fn=policy_fn,
            fitness_fn=fitness_fn,
            grid_n=latent_grid_n,
            pc3_value=pc3_value,
        )

        fmin, fmax = _draw_latent_policy_map(
            ax,
            XX,
            YY,
            action_grid,
            fitness_grid,
            num_drugs=num_drugs,
            drug_colors=drug_colors,
            show_boundary=show_policy_boundary,
        )

        if show_legend:
            legend_elements = [
                Patch(
                    facecolor=drug_colors[i],
                    edgecolor="gray",
                    linewidth=0.5,
                    label=drug_labels[i],
                )
                for i in range(num_drugs)
            ]
            ax.legend(
                handles=legend_elements,
                loc="upper left",
                fontsize=8,
                framealpha=0.9,
            )

        ax.text(
            0.99,
            0.02,
            f"Fitness shading\nlow={fmin:.3f}  high={fmax:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                alpha=0.85,
                edgecolor="lightgray",
            ),
        )
    else:
        if density:
            hb = ax.hexbin(
                scores[:, 0],
                scores[:, 1],
                gridsize=50,
                cmap="viridis",
                mincnt=1,
                linewidths=0,
                edgecolors="none",
            )
            fig.colorbar(hb, ax=ax, label="Sample density")
        else:
            ax.scatter(scores[:, 0], scores[:, 1], s=6, alpha=0.2, color="tab:blue")

    if show_sample_paths:
        unique_runs = np.unique(run_ids)
        chosen = unique_runs[:max_paths]
        for r in chosen:
            idx = np.where(run_ids == r)[0]
            idx = idx[np.argsort(time_ids[idx])]
            ax.plot(
                scores[idx, 0],
                scores[idx, 1],
                alpha=0.08,
                lw=0.6,
                color="white",
                zorder=3,
            )

    if mean_scores is not None:
        ax.plot(
            mean_scores[:, 0],
            mean_scores[:, 1],
            color="red",
            lw=2.0,
            alpha=0.9,
            label="Mean trajectory",
            zorder=4,
        )
        ax.scatter(
            mean_scores[0, 0],
            mean_scores[0, 1],
            color="white",
            edgecolors="black",
            s=50,
            marker="o",
            zorder=5,
        )
        ax.scatter(
            mean_scores[-1, 0],
            mean_scores[-1, 1],
            color="white",
            edgecolors="black",
            s=70,
            marker="X",
            zorder=5,
        )

    ax.set_xlabel(f"PC1 ({100 * evr[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({100 * evr[1]:.1f}%)")
    ax.set_title(f"{title}\nPC1+PC2+PC3 = {100 * cev[2]:.1f}% variance")

    # Add +/- 1 std boundary for hero's path in PCA
    if mean_scores is not None and len(state_trajectories) > 1:
        std_scores_val = []
        for step in range(len(mean_scores)):
            step_states = []
            for traj in state_trajectories:
                # pad or skip
                if step < len(traj):
                    step_states.append(traj[step])
            if step_states:
                st_clr = clr_transform(step_states, eps=eps)
                sc = (st_clr - pca["mean"]) @ pca["components"].T
                std_scores_val.append(sc.std(axis=0))
            else:
                std_scores_val.append(np.zeros(3))
        std_scores_val = np.array(std_scores_val)
        
        # Plot filled region
        ax.fill_between(
            mean_scores[:, 0],
            mean_scores[:, 1] - std_scores_val[:, 1],
            mean_scores[:, 1] + std_scores_val[:, 1],
            color="red",
            alpha=0.2,
            zorder=3
        )
        # We can't fill_between for x easily, but we can plot upper/lower lines
        ax.plot(mean_scores[:, 0] - std_scores_val[:, 0], mean_scores[:, 1], color="red", lw=0.5, alpha=0.5, ls="--")
        ax.plot(mean_scores[:, 0] + std_scores_val[:, 0], mean_scores[:, 1], color="red", lw=0.5, alpha=0.5, ls="--")

    fig_simplex = None
    if len(axes) > 1:
        ax2 = axes[1]
        # Instead of ax2 in same figure, let's just create a new figure!
        # Actually, let's just keep the code and return both, but it's easier to create a new figure
        fig_simplex, ax2 = plt.subplots(1, 1, figsize=(6, 5))
        
        bary = scores_to_barycentric(scores[:, :3])
        x, y = _bary_to_cart(bary[:, 0], bary[:, 1], bary[:, 2])

        if density:
            ax2.hexbin(
                x,
                y,
                gridsize=30,
                cmap="viridis",
                mincnt=3,
                linewidths=0,
                edgecolors="none",
            )
        else:
            ax2.scatter(x, y, s=6, alpha=0.15)

        if mean_scores is not None:
            mean_bary = scores_to_barycentric(mean_scores[:, :3])
            mx, my = _bary_to_cart(mean_bary[:, 0], mean_bary[:, 1], mean_bary[:, 2])
            
            if len(state_trajectories) > 1:
                # Approximate std boundary in simplex by projecting mean +/- std
                std_upper = mean_scores[:, :3].copy()
                std_lower = mean_scores[:, :3].copy()
                # we have std_scores_val computed earlier
                std_upper[:, 1] += std_scores_val[:, 1]
                std_lower[:, 1] -= std_scores_val[:, 1]
                
                bary_up = scores_to_barycentric(std_upper)
                bary_dn = scores_to_barycentric(std_lower)
                
                ux, uy = _bary_to_cart(bary_up[:, 0], bary_up[:, 1], bary_up[:, 2])
                dx, dy = _bary_to_cart(bary_dn[:, 0], bary_dn[:, 1], bary_dn[:, 2])
                
                ax2.fill_between(mx, dy, uy, color="red", alpha=0.2, zorder=3)
                
                # For x-direction std:
                std_right = mean_scores[:, :3].copy()
                std_left = mean_scores[:, :3].copy()
                std_right[:, 0] += std_scores_val[:, 0]
                std_left[:, 0] -= std_scores_val[:, 0]
                
                bary_r = scores_to_barycentric(std_right)
                bary_l = scores_to_barycentric(std_left)
                rx, ry = _bary_to_cart(bary_r[:, 0], bary_r[:, 1], bary_r[:, 2])
                lx, ly = _bary_to_cart(bary_l[:, 0], bary_l[:, 1], bary_l[:, 2])
                
                ax2.plot(lx, ly, color="red", lw=0.5, alpha=0.5, ls="--")
                ax2.plot(rx, ry, color="red", lw=0.5, alpha=0.5, ls="--")

            ax2.plot(mx, my, color="red", lw=2.0, alpha=0.9, zorder=4)
            ax2.scatter(
                mx[0],
                my[0],
                color="white",
                edgecolors="black",
                s=50,
                marker="o",
                zorder=5,
            )
            ax2.scatter(
                mx[-1],
                my[-1],
                color="white",
                edgecolors="black",
                s=70,
                marker="X",
                zorder=5,
            )

        outline_x = [0, 1, 0.5, 0]
        outline_y = [0, 0, _SQRT3_2, 0]
        ax2.plot(outline_x, outline_y, color="black", lw=1.2)
        ax2.text(-0.03, -0.04, "PC1-mode", fontsize=8)
        ax2.text(1.03, -0.04, "PC2-mode", fontsize=8, ha="right")
        ax2.text(0.5, _SQRT3_2 + 0.04, "PC3-mode", fontsize=8, ha="center")
        ax2.set_aspect("equal")
        ax2.axis("off")
        ax2.set_title("Latent mode simplex")

    fig.tight_layout()
    if len(axes) > 1:
        # We need to remove ax2 from fig if we created a new one, but actually let's just close the big fig and return two new ones? 
        # No, just return fig, fig_simplex, pca
        pass
    
    # Actually, earlier we did:
    # fig_simplex, ax2 = plt.subplots...
    # so we should return (fig, fig_simplex, pca) if simplex_view else (fig, pca)
    if simplex_view and cev[2] >= explained_threshold:
        return fig, fig_simplex, pca
    else:
        return fig, None, pca


# ──────────────────────────────────────────────────────────────────────
#  CLI entry point
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate simplex policy plots")
    parser.add_argument(
        "--policy", type=str, default=None, help="Path to trained .pth policy"
    )
    parser.add_argument("--resolution", type=int, default=60, help="Grid resolution")
    parser.add_argument(
        "--output", type=str, default="simplex_policy.png", help="Output file"
    )
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


def plot_pop_x_metric_slices(
    state_trajectories,
    metric_fn,
    metric_name,
    num_drugs=4,
    resolution=60,
    x3_slices=None,
    figsize=None,
    is_three_state=False,
):
    if x3_slices is None and not is_three_state:
        x3_slices = [
            (0.5, 1.0),
            (0.4, 0.5),
            (0.3, 0.4),
            (0.2, 0.3),
            (0.1, 0.2),
            (0.0, 0.1),
        ]
    elif is_three_state:
        x3_slices = [(0.0, 1.0)]

    n_slices = len(x3_slices)
    if figsize is None:
        figsize = (2.6 * n_slices + 1.5, 3.8)

    fig, ax_array = plt.subplots(1, n_slices, figsize=figsize)
    if n_slices == 1:
        ax_array = [ax_array]

    bary, x_cart, y_cart, tri_idx = _make_simplex_grid(resolution)

    all_states = []
    for episode in state_trajectories:
        all_states.extend(episode)
    all_states = np.array(all_states)

    triangulation = Triangulation(x_cart, y_cart, tri_idx)
    trifinder = triangulation.get_trifinder()

    slice_vals = []
    for s_idx, (x3_lo, x3_hi) in enumerate(x3_slices):
        if is_three_state:
            states_in_slice = all_states
        else:
            if s_idx == 0:
                mask = (all_states[:, 3] >= x3_lo) & (all_states[:, 3] <= x3_hi)
            else:
                mask = (all_states[:, 3] >= x3_lo) & (all_states[:, 3] < x3_hi)
            states_in_slice = all_states[mask]

        vals = np.zeros(len(tri_idx), dtype=float)

        if len(states_in_slice) > 0:
            if is_three_state:
                l0 = states_in_slice[:, 0]
                l1 = states_in_slice[:, 1]
                l2 = states_in_slice[:, 2]
            else:
                x3_mid = (x3_lo + x3_hi) / 2.0
                S = 1.0 - x3_mid
                if S > 0:
                    l0 = states_in_slice[:, 0] / S
                    l1 = states_in_slice[:, 1] / S
                    l2 = states_in_slice[:, 2] / S
                else:
                    l0 = np.zeros_like(states_in_slice[:, 0])
                    l1 = np.zeros_like(states_in_slice[:, 1])
                    l2 = np.zeros_like(states_in_slice[:, 2])

                tot = l0 + l1 + l2
                tot[tot == 0] = 1.0
                l0 /= tot
                l1 /= tot
                l2 /= tot

                cx, cy = _bary_to_cart(l0, l1, l2)
                found_tri_indices = trifinder(cx, cy)

                valid_mask = found_tri_indices != -1
                valid_idx = found_tri_indices[valid_mask]
                valid_states = states_in_slice[valid_mask]

                for i, tidx in enumerate(valid_idx):
                    vals[tidx] += metric_fn(valid_states[i])

        slice_vals.append(vals)

    total_states = len(all_states) if len(all_states) > 0 else 1.0
    global_max = (
        max([np.max(c) for c in slice_vals]) / total_states if slice_vals else 0.0
    )

    vmin = 1e-6
    if global_max <= vmin:
        global_max = vmin * 10.0

    cmap = mpl.colormaps["magma"]
    norm = mcolors.LogNorm(vmin=vmin, vmax=global_max)

    for s_idx, (x3_lo, x3_hi) in enumerate(x3_slices):
        ax = ax_array[s_idx]
        vals = slice_vals[s_idx] / total_states
        
        colors = cmap(norm(vals))
        colors[vals == 0, 3] = 0.0

        verts = np.column_stack([x_cart, y_cart])
        tri_verts = verts[tri_idx]

        pc = PolyCollection(
            tri_verts, facecolors=colors, edgecolors="none", linewidths=0
        )
        ax.add_collection(pc)

        outline_x = [0, 1, 0.5, 0]
        outline_y = [0, 0, _SQRT3_2, 0]
        ax.plot(outline_x, outline_y, color="black", linewidth=1.2, zorder=5)

        if not is_three_state:
            ax.text(
                0.5,
                -0.12,
                f"[{x3_lo:.1f}, {x3_hi:.1f}]",
                ha="center",
                va="top",
                fontsize=8,
                transform=ax.transAxes,
            )

        ax.set_xlim(-0.08, 1.08)
        ax.set_ylim(-0.15, _SQRT3_2 + 0.15)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.subplots_adjust(bottom=0.25, wspace=0.1)
    cbar_ax = fig.add_axes([0.15, 0.12, 0.7, 0.04])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", label=metric_name)

    return fig
