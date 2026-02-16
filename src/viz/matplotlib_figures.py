"""Publication-quality static figures using matplotlib.

Generates Figures 1-6 from precomputed spectral catalog results.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Polygon
import numpy as np

import json

from src.lattices.registry import LATTICE_REGISTRY, list_lattices
from src.io.serialize import load_result
from src.viz.lattice_drawing import (
    COORD_COLORS, DEFAULT_EDGE_COLOR, DEFAULT_FACE_ALPHA, FACE_COLOR,
    coord_color, compute_layout_bounds, edge_segments, edge_segments_periodic,
    face_polygons, harmonic_edge_colors,
)
from src.viz.periodic_edges import classify_edges, classify_faces

# Publication style defaults
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def _build_lattice_for_viz(lattice_name: str, nx: int = 4, ny: int = 4,
                           boundary: str = "periodic"):
    """Build a lattice for visualization."""
    gen = LATTICE_REGISTRY[lattice_name]()
    return gen.build(nx, ny, boundary=boundary)


def _draw_lattice_on_ax(ax, lat, title=None, show_faces=True, edge_color=DEFAULT_EDGE_COLOR):
    """Draw a lattice on a matplotlib axes.

    Periodic (wrap-around) edges are rendered as dashed stubs extending from
    each endpoint toward the lattice boundary.  Periodic faces are skipped.
    """
    pos = lat.positions
    edges = lat.edge_list
    coord = lat.coordination

    # Determine if we can do periodic-aware rendering
    use_periodic = lat.boundary == "periodic"

    if use_periodic:
        a1 = lat.unit_cell.a1
        a2 = lat.unit_cell.a2
        _, _, is_periodic = classify_edges(
            pos, edges, a1, a2, lat.nx_size, lat.ny_size, lat.boundary,
        )
        interior_segs, periodic_stubs = edge_segments_periodic(
            pos, edges, a1, a2, lat.nx_size, lat.ny_size, lat.boundary,
        )
    else:
        is_periodic = None

    # Draw faces (skip periodic faces)
    if show_faces and lat.face_list:
        if use_periodic and is_periodic is not None:
            interior_faces, _ = classify_faces(lat.face_list, edges, is_periodic)
            face_to_draw = interior_faces
        else:
            face_to_draw = lat.face_list

        polys = face_polygons(pos, face_to_draw)
        patches = [Polygon(p, closed=True) for p in polys]
        if patches:
            pc = PatchCollection(patches, facecolor=FACE_COLOR, edgecolor="none",
                                 alpha=DEFAULT_FACE_ALPHA)
            ax.add_collection(pc)

    # Draw edges
    if use_periodic:
        # Interior edges: solid
        int_lines = [np.column_stack((xs, ys)) for xs, ys, _ in interior_segs]
        if int_lines:
            lc = LineCollection(int_lines, colors=edge_color, linewidths=0.8, zorder=1)
            ax.add_collection(lc)

        # Periodic stubs: dashed
        per_lines = []
        for xs_u, ys_u, xs_v, ys_v, _ in periodic_stubs:
            per_lines.append(np.column_stack((xs_u, ys_u)))
            per_lines.append(np.column_stack((xs_v, ys_v)))
        if per_lines:
            lc_per = LineCollection(per_lines, colors="#b0b0b0", linewidths=0.8,
                                    linestyles="dashed", zorder=1)
            ax.add_collection(lc_per)
    else:
        segs = edge_segments(pos, edges)
        lines = [np.column_stack((xs, ys)) for xs, ys in segs]
        lc = LineCollection(lines, colors=edge_color, linewidths=0.8, zorder=1)
        ax.add_collection(lc)

    # Draw vertices colored by coordination
    unique_z = np.unique(coord)
    for z in unique_z:
        mask = coord == z
        ax.scatter(pos[mask, 0], pos[mask, 1], c=coord_color(z),
                   s=15, zorder=2, label=f"z={z}", edgecolors="white", linewidths=0.3)

    xmin, xmax, ymin, ymax = compute_layout_bounds(pos)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontweight="bold")


# ── Figure 1: Lattice Zoo Gallery ──────────────────────────────────────

def figure_lattice_gallery(output_path: str, size: int = 4,
                           boundary: str = "periodic") -> str:
    """8-panel lattice gallery with coordination-colored vertices and shaded faces."""
    names = list_lattices()
    n = len(names)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    axes = axes.flatten()

    for i, name in enumerate(names):
        lat = _build_lattice_for_viz(name, nx=size, ny=size, boundary=boundary)
        title = name.replace("_", " ").title()
        cdist = lat.coordination_distribution
        coord_str = ", ".join(f"z={k}:{v}" for k, v in sorted(cdist.items()))
        _draw_lattice_on_ax(axes[i], lat, title=f"{title}\n({coord_str})")
        axes[i].legend(loc="upper right", markerscale=0.8, framealpha=0.7)

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    bc_label = "periodic" if boundary == "periodic" else "open"
    fig.suptitle(f"ASI Lattice Zoo ({bc_label} BC)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fpath = os.path.join(output_path, f"fig1_lattice_gallery_{bc_label}.png")
    fig.savefig(fpath)
    plt.close(fig)
    return fpath


# ── Figure 2: Eigenvalue Histograms ───────────────────────────────────

def figure_eigenvalue_histograms(
    catalog_dir: str, output_path: str,
    size_label: str = "S", strategy: str = "all",
    boundary: str = "periodic",
) -> str:
    """Eigenvalue histograms of L1 per lattice (zero eigenvalues highlighted)."""
    names = list_lattices()
    n = len(names)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    axes = axes.flatten()

    for i, name in enumerate(names):
        ax = axes[i]
        try:
            data = load_result(catalog_dir, name, size_label, strategy, boundary=boundary)
            evals = data["L1_eigenvalues"]
            zero_mask = np.abs(evals) < 1e-10
            nonzero = evals[~zero_mask]

            ax.hist(nonzero, bins=50, color="#3498db", alpha=0.7, density=True,
                    label="nonzero")
            if np.sum(zero_mask) > 0:
                ax.axvline(0, color="#e74c3c", linewidth=2, linestyle="--",
                           label=f"zero ({np.sum(zero_mask)})")

            beta_1 = data.get("beta_1", np.sum(zero_mask))
            ax.set_title(f"{name.replace('_', ' ').title()}\n"
                         f"$\\beta_1$={beta_1}", fontweight="bold")
            ax.legend(fontsize=7)
        except Exception as e:
            ax.text(0.5, 0.5, f"No data\n{e}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8)
            ax.set_title(name.replace("_", " ").title())

        ax.set_xlabel("$\\lambda$")
        ax.set_ylabel("Density")

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    bc_label = "periodic" if boundary == "periodic" else "open"
    fig.suptitle(f"L$_1$ Eigenvalue Histograms (size={size_label}, faces={strategy}, {bc_label} BC)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fpath = os.path.join(output_path, f"fig2_eigenvalue_histograms_{size_label}_{strategy}_{bc_label}.png")
    fig.savefig(fpath)
    plt.close(fig)
    return fpath


# ── Figure 3: Spectral Density Overlay ────────────────────────────────

def figure_spectral_overlay(
    catalog_dir: str, output_path: str,
    size_label: str = "S", strategy: str = "all",
    boundary: str = "periodic",
) -> str:
    """Overlay spectral densities of all lattices (normalized by n1)."""
    names = list_lattices()
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

    for i, name in enumerate(names):
        try:
            data = load_result(catalog_dir, name, size_label, strategy, boundary=boundary)
            evals = data["L1_eigenvalues"]
            nonzero = evals[np.abs(evals) >= 1e-10]
            if len(nonzero) > 0:
                ax.hist(nonzero, bins=80, density=True, alpha=0.4,
                        color=colors[i], histtype="step", linewidth=1.5,
                        label=name.replace("_", " ").title())
        except Exception:
            continue

    ax.set_xlabel("$\\lambda$ (eigenvalue of L$_1$)")
    ax.set_ylabel("Spectral density (normalized)")
    bc_label = "periodic" if boundary == "periodic" else "open"
    ax.set_title(f"L$_1$ Spectral Density Overlay (size={size_label}, faces={strategy}, {bc_label} BC)",
                 fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fpath = os.path.join(output_path, f"fig3_spectral_overlay_{size_label}_{strategy}_{bc_label}.png")
    fig.savefig(fpath)
    plt.close(fig)
    return fpath


# ── Figure 4: β₁ Scaling ─────────────────────────────────────────────

def figure_beta1_scaling(
    catalog_dir: str, output_path: str,
    sizes: Optional[List[str]] = None,
    boundary: str = "periodic",
) -> str:
    """Log-log beta_1 vs system size for all lattices under both face strategies."""
    if sizes is None:
        sizes = ["XS", "S", "M"]

    from src.spectral.catalog import SIZE_CONFIGS
    names = list_lattices()
    strategies = ["all", "none"]
    markers = {"all": "o", "none": "s"}
    linestyles = {"all": "-", "none": "--"}

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

    for i, name in enumerate(names):
        for strat in strategies:
            ns = []
            betas = []
            for sz in sizes:
                try:
                    data = load_result(catalog_dir, name, sz, strat, boundary=boundary)
                    nx_val = SIZE_CONFIGS[sz][0]
                    ns.append(nx_val)
                    betas.append(data["beta_1"])
                except Exception:
                    continue
            if len(ns) >= 2:
                label = f"{name.replace('_', ' ')} ({strat})"
                ax.plot(ns, betas, marker=markers[strat], linestyle=linestyles[strat],
                        color=colors[i], label=label, markersize=5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("System size N (unit cells per side)")
    ax.set_ylabel("$\\beta_1$")
    bc_label = "periodic" if boundary == "periodic" else "open"
    ax.set_title(f"$\\beta_1$ Scaling with System Size ({bc_label} BC)", fontweight="bold")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fpath = os.path.join(output_path, f"fig4_beta1_scaling_{bc_label}.png")
    fig.savefig(fpath)
    plt.close(fig)
    return fpath


# ── Figure 5: Harmonic Mode Visualization ─────────────────────────────

def figure_harmonic_modes(
    catalog_dir: str, output_path: str,
    lattice_name: str = "square",
    size_label: str = "XS",
    strategy: str = "all",
    max_modes: int = 4,
    boundary: str = "periodic",
) -> str:
    """Lattice with edges colored by harmonic eigenvector components."""
    data = load_result(catalog_dir, lattice_name, size_label, strategy, boundary=boundary)
    lat = _build_lattice_for_viz(lattice_name,
                                  nx=data["nx"], ny=data["ny"],
                                  boundary=boundary)

    harmonic = data.get("harmonic_basis", None)
    bc_label = "periodic" if boundary == "periodic" else "open"
    if harmonic is None or (isinstance(harmonic, np.ndarray) and harmonic.shape[1] == 0):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.text(0.5, 0.5, "No harmonic modes", transform=ax.transAxes,
                ha="center", fontsize=14)
        fpath = os.path.join(output_path,
                             f"fig5_harmonic_{lattice_name}_{size_label}_{strategy}_{bc_label}.png")
        fig.savefig(fpath)
        plt.close(fig)
        return fpath

    n_modes = min(max_modes, harmonic.shape[1])
    fig, axes = plt.subplots(1, n_modes, figsize=(5 * n_modes, 5))
    if n_modes == 1:
        axes = [axes]

    pos = lat.positions
    edges = lat.edge_list
    cmap = plt.cm.RdBu

    # Classify edges for periodic rendering
    use_periodic = lat.boundary == "periodic"
    interior_segs = None
    periodic_stubs = None
    if use_periodic:
        a1 = lat.unit_cell.a1
        a2 = lat.unit_cell.a2
        interior_segs, periodic_stubs = edge_segments_periodic(
            pos, edges, a1, a2, lat.nx_size, lat.ny_size, lat.boundary,
        )

    for idx in range(n_modes):
        ax = axes[idx]
        h = harmonic[:, idx]
        normed, vmax = harmonic_edge_colors(h)

        if use_periodic and interior_segs is not None:
            # Interior edges: solid colored
            int_lines = []
            int_colors = []
            for xs, ys, edge_idx in interior_segs:
                int_lines.append(np.column_stack((xs, ys)))
                int_colors.append(cmap(0.5 + 0.5 * normed[edge_idx]))

            if int_lines:
                lc = LineCollection(int_lines, colors=int_colors, linewidths=2, zorder=1)
                ax.add_collection(lc)

            # Periodic stubs: dashed colored
            per_lines = []
            per_colors = []
            for xs_u, ys_u, xs_v, ys_v, edge_idx in periodic_stubs:
                c = cmap(0.5 + 0.5 * normed[edge_idx])
                per_lines.append(np.column_stack((xs_u, ys_u)))
                per_colors.append(c)
                per_lines.append(np.column_stack((xs_v, ys_v)))
                per_colors.append(c)

            if per_lines:
                lc_per = LineCollection(per_lines, colors=per_colors,
                                        linewidths=2, linestyles="dashed", zorder=1)
                ax.add_collection(lc_per)
        else:
            # Fallback: draw all edges as solid
            lines = []
            colors = []
            for j, (u, v) in enumerate(edges):
                lines.append(np.column_stack(([pos[u, 0], pos[v, 0]],
                                              [pos[u, 1], pos[v, 1]])))
                colors.append(cmap(0.5 + 0.5 * normed[j]))

            lc = LineCollection(lines, colors=colors, linewidths=2, zorder=1)
            ax.add_collection(lc)

        # Draw vertices
        ax.scatter(pos[:, 0], pos[:, 1], c="black", s=8, zorder=2)

        xmin, xmax, ymin, ymax = compute_layout_bounds(pos)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Mode {idx + 1}", fontweight="bold")

    fig.suptitle(f"Harmonic Modes: {lattice_name.replace('_', ' ').title()} "
                 f"({size_label}, {strategy}, {bc_label} BC)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fpath = os.path.join(output_path,
                         f"fig5_harmonic_{lattice_name}_{size_label}_{strategy}_{bc_label}.png")
    fig.savefig(fpath)
    plt.close(fig)
    return fpath


# ── Figure 6: L1_down vs L1 Spectral Comparison ──────────────────────

def figure_l1_decomposition(
    catalog_dir: str, output_path: str,
    size_label: str = "S", strategy: str = "all",
    boundary: str = "periodic",
) -> str:
    """Compare L1, L1_down, L1_up eigenvalues for each lattice."""
    names = list_lattices()
    n = len(names)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    axes = axes.flatten()

    for i, name in enumerate(names):
        ax = axes[i]
        try:
            data = load_result(catalog_dir, name, size_label, strategy, boundary=boundary)
            for key, label, color in [
                ("L1_eigenvalues", "$L_1$", "#2c3e50"),
                ("L1_down_eigenvalues", "$L_1^{\\downarrow}$", "#3498db"),
                ("L1_up_eigenvalues", "$L_1^{\\uparrow}$", "#e74c3c"),
            ]:
                if key in data:
                    evals = data[key]
                    ax.hist(evals, bins=40, alpha=0.4, color=color,
                            density=True, label=label, histtype="step",
                            linewidth=1.5)
            ax.legend(fontsize=7)
            ax.set_title(name.replace("_", " ").title(), fontweight="bold")
        except Exception:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", fontsize=10)
            ax.set_title(name.replace("_", " ").title())

        ax.set_xlabel("$\\lambda$")

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    bc_label = "periodic" if boundary == "periodic" else "open"
    fig.suptitle(f"Hodge Decomposition: $L_1$ vs $L_1^{{\\downarrow}}$ vs $L_1^{{\\uparrow}}$ "
                 f"(size={size_label}, faces={strategy}, {bc_label} BC)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fpath = os.path.join(output_path, f"fig6_l1_decomposition_{size_label}_{strategy}_{bc_label}.png")
    fig.savefig(fpath)
    plt.close(fig)
    return fpath


# ── Helpers for ice manifold figures ──────────────────────────────────

def _load_ice_manifold_entries(catalog_dir: str) -> List[Dict]:
    """Read catalog_index.json, deduplicate by (lattice, size, boundary)."""
    index_path = os.path.join(catalog_dir, "catalog_index.json")
    with open(index_path) as f:
        results = json.load(f)

    seen: dict = {}
    for entry in results:
        key = (entry["lattice_name"], entry["size_label"],
               entry.get("boundary", "periodic"))
        if key in seen:
            continue
        n_v = entry["n_vertices"]
        n_e = entry["n_edges"]
        ice_dim = n_e - n_v + 1
        seen[key] = {
            "lattice_name": entry["lattice_name"],
            "size_label": entry["size_label"],
            "boundary": entry.get("boundary", "periodic"),
            "n_vertices": n_v,
            "n_edges": n_e,
            "ice_manifold_dim": ice_dim,
            "ice_manifold_fraction": ice_dim / n_e if n_e > 0 else 0.0,
        }
    return list(seen.values())


# ── Figure 7: Ice Manifold Dimension Scaling ──────────────────────────

def figure_ice_manifold_scaling(
    catalog_dir: str, output_path: str,
    sizes: Optional[List[str]] = None,
) -> str:
    """Log-log ice manifold dim vs n_vertices, both BCs overlaid."""
    if sizes is None:
        sizes = ["XS", "S", "M", "L"]

    entries = _load_ice_manifold_entries(catalog_dir)
    names = list_lattices()
    boundaries = ["periodic", "open"]
    bc_styles = {"periodic": "-", "open": "--"}
    bc_markers = {"periodic": "o", "open": "s"}

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

    for i, name in enumerate(names):
        for bc in boundaries:
            subset = [e for e in entries
                      if e["lattice_name"] == name
                      and e["boundary"] == bc
                      and e["size_label"] in sizes]
            subset.sort(key=lambda e: e["n_vertices"])
            if len(subset) < 2:
                continue

            n_vals = np.array([e["n_vertices"] for e in subset], dtype=float)
            dim_vals = np.array([e["ice_manifold_dim"] for e in subset], dtype=float)

            display = name.replace("_", " ")
            label = f"{display} ({bc})"
            ax.plot(n_vals, dim_vals,
                    marker=bc_markers[bc], linestyle=bc_styles[bc],
                    color=colors[i], label=label, markersize=5)

            # Power-law fit
            mask = dim_vals > 0
            if np.sum(mask) >= 2:
                log_n = np.log10(n_vals[mask])
                log_d = np.log10(dim_vals[mask])
                coeffs = np.polyfit(log_n, log_d, 1)
                n_fit = np.logspace(np.log10(n_vals.min() * 0.8),
                                    np.log10(n_vals.max() * 1.2), 50)
                d_fit = 10 ** coeffs[1] * n_fit ** coeffs[0]
                ax.plot(n_fit, d_fit, linestyle=":", color=colors[i],
                        linewidth=0.8, alpha=0.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$N$ (vertices)")
    ax.set_ylabel("Ice manifold dimension")
    ax.set_title("Ice Manifold Dimension Scaling (both BCs)", fontweight="bold")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fpath = os.path.join(output_path, "fig7_ice_manifold_scaling.png")
    fig.savefig(fpath)
    plt.close(fig)
    return fpath


# ── Figure 8: Ice Manifold Fraction ──────────────────────────────────

def figure_ice_manifold_fraction(
    catalog_dir: str, output_path: str,
    sizes: Optional[List[str]] = None,
) -> str:
    """Ice manifold fraction (dim/n_edges) vs n_vertices, log x / linear y."""
    if sizes is None:
        sizes = ["XS", "S", "M", "L"]

    entries = _load_ice_manifold_entries(catalog_dir)
    names = list_lattices()
    boundaries = ["periodic", "open"]
    bc_styles = {"periodic": "-", "open": "--"}
    bc_markers = {"periodic": "o", "open": "s"}

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

    for i, name in enumerate(names):
        for bc in boundaries:
            subset = [e for e in entries
                      if e["lattice_name"] == name
                      and e["boundary"] == bc
                      and e["size_label"] in sizes]
            subset.sort(key=lambda e: e["n_vertices"])
            if len(subset) < 2:
                continue

            n_vals = np.array([e["n_vertices"] for e in subset], dtype=float)
            frac_vals = np.array([e["ice_manifold_fraction"] for e in subset])

            display = name.replace("_", " ")
            label = f"{display} ({bc})"
            ax.plot(n_vals, frac_vals,
                    marker=bc_markers[bc], linestyle=bc_styles[bc],
                    color=colors[i], label=label, markersize=5)

    ax.set_xscale("log")
    ax.set_xlabel("$N$ (vertices)")
    ax.set_ylabel("Ice manifold fraction (dim / $n_1$)")
    ax.set_title("Ice Manifold Fraction vs System Size (both BCs)", fontweight="bold")
    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fpath = os.path.join(output_path, "fig8_ice_manifold_fraction.png")
    fig.savefig(fpath)
    plt.close(fig)
    return fpath


# ── Figure 9: Ice State Samples ──────────────────────────────────────

def figure_ice_states(
    output_path: str,
    lattice_name: str = "square",
    size: int = 4,
    boundary: str = "periodic",
    n_samples: int = 6,
) -> str:
    """2x3 grid of ice state samples with arrows on edges.

    Blue arrows for sigma=+1, red for sigma=-1. Arrows at edge midpoints
    indicate spin direction.
    """
    from src.topology.incidence import build_B1
    from src.topology.ice_sampling import sample_ice_states

    lat = _build_lattice_for_viz(lattice_name, nx=size, ny=size, boundary=boundary)
    B1 = build_B1(lat.n_vertices, lat.edge_list)
    states = sample_ice_states(B1, lat.coordination, n_samples=n_samples,
                               n_flips_between=20, edge_list=lat.edge_list)

    pos = lat.positions
    edges = lat.edge_list
    use_periodic = lat.boundary == "periodic"

    if use_periodic:
        a1 = lat.unit_cell.a1
        a2 = lat.unit_cell.a2
        interior_segs, periodic_stubs = edge_segments_periodic(
            pos, edges, a1, a2, lat.nx_size, lat.ny_size, lat.boundary,
        )
    else:
        interior_segs = None
        periodic_stubs = None

    nrows = 2
    ncols = 3
    n_panels = min(n_samples, nrows * ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = axes.flatten()

    spin_plus_color = "#3498db"
    spin_minus_color = "#e74c3c"

    for panel_idx in range(n_panels):
        ax = axes[panel_idx]
        sigma = states[panel_idx]

        if use_periodic and interior_segs is not None:
            # Interior edges colored by spin
            lines_plus, lines_minus = [], []
            for xs, ys, edge_idx in interior_segs:
                seg = np.column_stack((xs, ys))
                if sigma[edge_idx] > 0:
                    lines_plus.append(seg)
                else:
                    lines_minus.append(seg)

            if lines_plus:
                lc_p = LineCollection(lines_plus, colors=spin_plus_color,
                                      linewidths=1.5, zorder=1)
                ax.add_collection(lc_p)
            if lines_minus:
                lc_m = LineCollection(lines_minus, colors=spin_minus_color,
                                      linewidths=1.5, zorder=1)
                ax.add_collection(lc_m)

            # Periodic stubs: gray dashed
            per_lines = []
            for xs_u, ys_u, xs_v, ys_v, _ in periodic_stubs:
                per_lines.append(np.column_stack((xs_u, ys_u)))
                per_lines.append(np.column_stack((xs_v, ys_v)))
            if per_lines:
                lc_per = LineCollection(per_lines, colors="#b0b0b0",
                                        linewidths=0.8, linestyles="dashed", zorder=1)
                ax.add_collection(lc_per)

            # Arrows on interior edges
            for xs, ys, edge_idx in interior_segs:
                _draw_spin_arrow(ax, pos, edges[edge_idx], sigma[edge_idx])

        else:
            # Non-periodic: draw all edges
            lines_plus, lines_minus = [], []
            for j, (u, v) in enumerate(edges):
                seg = np.column_stack(([pos[u, 0], pos[v, 0]],
                                       [pos[u, 1], pos[v, 1]]))
                if sigma[j] > 0:
                    lines_plus.append(seg)
                else:
                    lines_minus.append(seg)

            if lines_plus:
                ax.add_collection(LineCollection(lines_plus, colors=spin_plus_color,
                                                  linewidths=1.5, zorder=1))
            if lines_minus:
                ax.add_collection(LineCollection(lines_minus, colors=spin_minus_color,
                                                  linewidths=1.5, zorder=1))

            for j, (u, v) in enumerate(edges):
                _draw_spin_arrow(ax, pos, (u, v), sigma[j])

        # Small vertex dots
        ax.scatter(pos[:, 0], pos[:, 1], c="black", s=8, zorder=3)

        xmin, xmax, ymin, ymax = compute_layout_bounds(pos)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        label = "Seed" if panel_idx == 0 else f"Sample {panel_idx}"
        ax.set_title(label, fontweight="bold")

    # Hide unused panels
    for j in range(n_panels, len(axes)):
        axes[j].set_visible(False)

    bc_label = "periodic" if boundary == "periodic" else "open"
    display = lattice_name.replace("_", " ").title()
    fig.suptitle(f"Ice States: {display} (size={size}, {bc_label} BC)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fpath = os.path.join(
        output_path,
        f"fig9_ice_states_{lattice_name}_{size}_{bc_label}.png",
    )
    fig.savefig(fpath)
    plt.close(fig)
    return fpath


def _draw_spin_arrow(ax, positions, edge, spin_val):
    """Draw a small arrow at the midpoint of an edge indicating spin direction."""
    u, v = edge
    pu = positions[u]
    pv = positions[v]
    mid = 0.5 * (pu + pv)

    # Direction: tail->head (+1) or head->tail (-1)
    if spin_val > 0:
        dx = pv[0] - pu[0]
        dy = pv[1] - pu[1]
    else:
        dx = pu[0] - pv[0]
        dy = pu[1] - pv[1]

    length = np.sqrt(dx**2 + dy**2)
    if length < 1e-10:
        return

    scale = 0.12 * length
    dx_n = dx / length * scale
    dy_n = dy / length * scale

    color = "#3498db" if spin_val > 0 else "#e74c3c"
    ax.annotate(
        "",
        xy=(mid[0] + dx_n, mid[1] + dy_n),
        xytext=(mid[0] - dx_n, mid[1] - dy_n),
        arrowprops=dict(
            arrowstyle="->",
            color=color,
            lw=1.2,
        ),
        zorder=2,
    )
