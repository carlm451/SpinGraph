#!/usr/bin/env python
"""Generate paper figures for Section 5.8 counting table.

Produces 24 PNGs (3 per row x 8 rows): undirected graph, B1 reference
orientation, and a sampled ice-manifold state for each lattice/size/BC
combination.

Output: results/paper_figures/
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Publication style
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

from src.lattices.registry import get_generator
from src.topology.incidence import build_B1
from src.topology.ice_sampling import sample_ice_states, verify_ice_state
from src.viz.paper_figures import (
    draw_undirected_graph,
    draw_reference_orientation,
    draw_ice_state,
)

# ── Table rows ───────────────────────────────────────────────────────

TABLE_ROWS = [
    # (lattice_name, nx, ny, boundary)
    ("square",   4, 4, "open"),
    ("square",   4, 4, "periodic"),
    ("kagome",   2, 2, "open"),
    ("kagome",   2, 2, "periodic"),
    ("santa_fe", 2, 2, "open"),
    ("shakti",   2, 2, "open"),
    ("tetris",   2, 2, "open"),
    ("tetris",   3, 3, "open"),
]

OUTPUT_DIR = os.path.join("results", "paper_figures")


def _pretty_name(name: str) -> str:
    return name.replace("_", " ").title()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    generated = []

    for lattice_name, nx, ny, bc in TABLE_ROWS:
        label = f"{_pretty_name(lattice_name)} {nx}x{ny} {bc}"
        prefix = f"{lattice_name}_{nx}x{ny}_{bc}"
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

        # Build lattice
        gen = get_generator(lattice_name)
        lat = gen.build(nx, ny, boundary=bc)

        # Print verification counts
        n_odd = int(np.sum(lat.coordination % 2 == 1))
        cdist = lat.coordination_distribution
        cdist_str = ", ".join(f"z={k}:{v}" for k, v in sorted(cdist.items()))
        print(f"  n_vertices = {lat.n_vertices}")
        print(f"  n_edges    = {lat.n_edges}")
        print(f"  n_odd      = {n_odd}")
        print(f"  coord dist = {cdist_str}")

        # Build B1
        B1 = build_B1(lat.n_vertices, lat.edge_list)

        # Sample ice state
        states = sample_ice_states(
            B1, lat.coordination, n_samples=2,
            n_flips_between=50, seed=42, edge_list=lat.edge_list,
        )
        sigma = states[1]  # second state, after decorrelation

        # Verify ice rule
        ok = verify_ice_state(B1, sigma, lat.coordination)
        print(f"  ice rule   = {'PASS' if ok else 'FAIL'}")
        assert ok, f"Ice rule violation for {label}!"

        # Compute beta_1 for reporting
        from scipy import sparse
        rank_B1 = np.linalg.matrix_rank(B1.toarray())
        beta_1 = lat.n_edges - rank_B1  # no faces filled
        print(f"  beta_1     = {beta_1} (no faces)")

        # ── Generate 3 figures ───────────────────────────────────────

        # 1. Undirected graph
        fig, ax = plt.subplots(figsize=(5, 5))
        draw_undirected_graph(ax, lat)
        ax.set_title(label, fontweight="bold", fontsize=12)
        fpath = os.path.join(OUTPUT_DIR, f"{prefix}_graph.png")
        fig.savefig(fpath)
        plt.close(fig)
        generated.append(fpath)
        print(f"  -> {fpath}")

        # 2. Reference orientation
        fig, ax = plt.subplots(figsize=(5, 5))
        draw_reference_orientation(ax, lat)
        ax.set_title(f"{label} — B1 orientation", fontweight="bold",
                     fontsize=12)
        fpath = os.path.join(OUTPUT_DIR, f"{prefix}_orientation.png")
        fig.savefig(fpath)
        plt.close(fig)
        generated.append(fpath)
        print(f"  -> {fpath}")

        # 3. Ice state
        fig, ax = plt.subplots(figsize=(5, 5))
        draw_ice_state(ax, lat, sigma, B1)
        ax.set_title(f"{label} — ice state", fontweight="bold",
                     fontsize=12)
        fpath = os.path.join(OUTPUT_DIR, f"{prefix}_ice_state.png")
        fig.savefig(fpath)
        plt.close(fig)
        generated.append(fpath)
        print(f"  -> {fpath}")

    print(f"\n{'='*60}")
    print(f"  Done: {len(generated)} figures saved to {OUTPUT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
