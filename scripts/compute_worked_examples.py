#!/usr/bin/env python3
"""Compute all matrices, spectra, and ice states for the worked-examples HTML document.

Builds each lattice at its minimum valid periodic size (and open BC variant),
extracts B1, B2 as dense arrays, computes Laplacians, eigenvalues, Betti numbers,
ice states, and S matrices. Outputs JSON with all values needed for the HTML.

Minimum valid periodic sizes (avoiding multi-edges / self-loops):
  Square:  3x3  -> n0=9, n1=18, n2=9
  Kagome:  2x2  -> n0=8, n1=12, n2=4
  Shakti:  1x1  -> n0=16, n1=24, n2=8
  Tetris:  2x1  -> n0=16, n1=24, n2=8
  Santa Fe: 1x2 -> n0=12, n1=18, n2=6
"""
import json
import sys
import os

import numpy as np
from scipy import sparse
from scipy.linalg import eigh

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.lattices.registry import get_generator
from src.topology.incidence import build_B1, build_B2, verify_chain_complex
from src.topology.laplacians import build_all_laplacians
from src.topology.ice_sampling import find_seed_ice_state, verify_ice_state, pauling_estimate
from src.spectral.eigensolve import eigendecompose, extract_harmonic_basis, count_zero_eigenvalues

# Lattice configs: name -> (nx, ny) for minimum valid periodic tiling
LATTICE_CONFIGS = {
    "square": (3, 3),
    "kagome": (2, 2),
    "shakti": (1, 1),
    "tetris": (2, 1),
    "santa_fe": (1, 2),
}

ZERO_TOL = 1e-10


def format_dense_matrix(M, precision=0):
    """Convert dense matrix to nested list for JSON serialization."""
    if precision == 0:
        return [[int(round(x)) for x in row] for row in M]
    return [[round(float(x), precision) for x in row] for row in M]


def format_eigenvalues(evals, precision=6):
    """Format eigenvalues for display."""
    return [round(float(v), precision) for v in evals]


def compute_S_matrix(B1_dense, sigma):
    """Compute Nisoli's antisymmetrized S matrix.

    S = (1/2)(|B1| @ D_sigma @ B1^T - B1 @ D_sigma @ |B1|^T)

    where |B1| is the element-wise absolute value and D_sigma = diag(sigma).
    """
    abs_B1 = np.abs(B1_dense)
    D_sigma = np.diag(sigma)
    term1 = abs_B1 @ D_sigma @ B1_dense.T
    term2 = B1_dense @ D_sigma @ abs_B1.T
    S = 0.5 * (term1 - term2)
    return S


def compute_lattice_data(name, nx, ny, boundary="periodic"):
    """Compute all data for one lattice configuration."""
    gen = get_generator(name)
    lat = gen.build(nx, ny, boundary=boundary)

    # Build incidence matrices
    B1 = build_B1(lat.n_vertices, lat.edge_list)
    B2 = build_B2(lat.n_edges, lat.face_list, lat.edge_list)

    # Verify chain complex
    chain_valid = verify_chain_complex(B1, B2)

    # Dense versions for display
    B1_dense = B1.toarray()
    B2_dense = B2.toarray()

    # Build Laplacians
    laps = build_all_laplacians(B1, B2)

    # Eigendecompose
    L0_result = eigendecompose(laps["L0"], method="dense")
    L1_result = eigendecompose(laps["L1"], method="dense")
    L1_down_result = eigendecompose(laps["L1_down"], method="dense")
    L1_up_result = eigendecompose(laps["L1_up"], method="dense")

    L0_evals = L0_result["eigenvalues"]
    L1_evals = L1_result["eigenvalues"]
    L1_down_evals = L1_down_result["eigenvalues"]
    L1_up_evals = L1_up_result["eigenvalues"]

    # Betti numbers
    beta_0 = count_zero_eigenvalues(L0_evals, ZERO_TOL)
    beta_1 = count_zero_eigenvalues(L1_evals, ZERO_TOL)
    # beta_2 from rank formula
    rank_B1 = lat.n_vertices - beta_0
    rank_B2 = lat.n_edges - rank_B1 - beta_1
    beta_2 = lat.n_faces - rank_B2

    # Euler characteristic
    euler_simplex = lat.n_vertices - lat.n_edges + lat.n_faces
    euler_betti = beta_0 - beta_1 + beta_2
    euler_consistent = (euler_simplex == euler_betti)

    # Harmonic basis
    harmonic = extract_harmonic_basis(L1_result["eigenvectors"], L1_evals, ZERO_TOL)

    # Validate harmonic vectors
    harmonic_valid = []
    for i in range(harmonic.shape[1]):
        h = harmonic[:, i]
        div_norm = float(np.linalg.norm(B1_dense @ h))
        curl_norm = float(np.linalg.norm(B2_dense.T @ h))
        L1h_norm = float(np.linalg.norm(laps["L1"].toarray() @ h))
        harmonic_valid.append({
            "mode": i,
            "div_norm": div_norm,
            "curl_norm": curl_norm,
            "L1h_norm": L1h_norm,
            "is_harmonic": (L1h_norm < 1e-8),
        })

    # Spectral gaps
    nonzero_L0 = L0_evals[L0_evals > ZERO_TOL]
    nonzero_L1 = L1_evals[L1_evals > ZERO_TOL]
    L0_gap = float(nonzero_L0[0]) if len(nonzero_L0) > 0 else 0.0
    L1_gap = float(nonzero_L1[0]) if len(nonzero_L1) > 0 else 0.0

    # Ice state
    np.random.seed(42)
    sigma = find_seed_ice_state(B1, lat.coordination, edge_list=lat.edge_list)
    ice_valid = verify_ice_state(B1, sigma, lat.coordination)
    charge = (B1_dense @ sigma).astype(int)

    # Compute S matrix
    S = compute_S_matrix(B1_dense, sigma)

    # Monopole excitation: flip edge 0
    sigma_flip = sigma.copy()
    sigma_flip[0] *= -1.0
    charge_flip = (B1_dense @ sigma_flip).astype(int)

    # Pauling estimate
    pauling = pauling_estimate(lat.coordination)

    # L0 dense matrix
    L0_dense = laps["L0"].toarray()

    # Edge table with metadata
    edge_table = []
    for idx, (u, v) in enumerate(lat.edge_list):
        # Determine if wrap-around: check if distance > 1 unit cell
        pos_u = lat.positions[u]
        pos_v = lat.positions[v]
        dist = np.linalg.norm(pos_v - pos_u)
        # For periodic lattices, wrap edges have large distances in Cartesian coords
        uc = lat.unit_cell
        max_dim = max(abs(uc.a1[0]) * nx, abs(uc.a1[1]) * nx,
                      abs(uc.a2[0]) * ny, abs(uc.a2[1]) * ny)
        is_wrap = dist > max_dim * 0.5
        edge_table.append({
            "index": idx,
            "tail": int(u),
            "head": int(v),
            "wrap": is_wrap,
        })

    # Face table
    face_table = []
    for idx, face_verts in enumerate(lat.face_list):
        face_table.append({
            "index": idx,
            "vertices": [int(v) for v in face_verts],
            "n_vertices": len(face_verts),
        })

    # Coordination distribution
    coord_dist = lat.coordination_distribution

    result = {
        "name": name,
        "nx": nx,
        "ny": ny,
        "boundary": boundary,
        "n_vertices": lat.n_vertices,
        "n_edges": lat.n_edges,
        "n_faces": lat.n_faces,
        "coordination_distribution": {str(k): v for k, v in coord_dist.items()},
        "positions": [[round(float(x), 4), round(float(y), 4)]
                       for x, y in lat.positions],
        "edge_table": edge_table,
        "face_table": face_table,
        "B1": format_dense_matrix(B1_dense),
        "B2": format_dense_matrix(B2_dense),
        "chain_complex_valid": chain_valid,
        "L0": format_dense_matrix(L0_dense),
        "L0_eigenvalues": format_eigenvalues(L0_evals),
        "L1_eigenvalues": format_eigenvalues(L1_evals),
        "L1_down_eigenvalues": format_eigenvalues(L1_down_evals),
        "L1_up_eigenvalues": format_eigenvalues(L1_up_evals),
        "L0_spectral_gap": round(L0_gap, 6),
        "L1_spectral_gap": round(L1_gap, 6),
        "beta_0": beta_0,
        "beta_1": beta_1,
        "beta_2": beta_2,
        "rank_B1": rank_B1,
        "rank_B2": rank_B2,
        "euler_simplex": euler_simplex,
        "euler_betti": euler_betti,
        "euler_consistent": euler_consistent,
        "harmonic_basis": format_dense_matrix(harmonic, precision=6),
        "harmonic_validation": harmonic_valid,
        "sigma": [int(s) for s in sigma],
        "charge": [int(c) for c in charge],
        "ice_valid": ice_valid,
        "sigma_flipped": [int(s) for s in sigma_flip],
        "charge_flipped": [int(c) for c in charge_flip],
        "S_matrix": format_dense_matrix(S),
        "pauling_estimate": round(pauling, 2) if pauling != float("inf") else "inf",
    }

    return result


def main():
    all_results = {}

    for name, (nx, ny) in LATTICE_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Computing {name} ({nx}x{ny} periodic)...")
        print(f"{'='*60}")

        # Periodic
        data = compute_lattice_data(name, nx, ny, "periodic")
        key = f"{name}_periodic"
        all_results[key] = data

        print(f"  n0={data['n_vertices']}, n1={data['n_edges']}, n2={data['n_faces']}")
        print(f"  B1: {data['n_vertices']}x{data['n_edges']}")
        print(f"  B2: {data['n_edges']}x{data['n_faces']}")
        print(f"  Chain complex valid: {data['chain_complex_valid']}")
        print(f"  beta_0={data['beta_0']}, beta_1={data['beta_1']}, beta_2={data['beta_2']}")
        print(f"  Euler: simplex={data['euler_simplex']}, betti={data['euler_betti']}, consistent={data['euler_consistent']}")
        print(f"  L0 gap={data['L0_spectral_gap']}, L1 gap={data['L1_spectral_gap']}")
        print(f"  Ice state valid: {data['ice_valid']}")
        print(f"  Pauling estimate: {data['pauling_estimate']}")
        print(f"  Coord dist: {data['coordination_distribution']}")
        print(f"  L0 eigenvalues: {data['L0_eigenvalues']}")
        print(f"  L1 eigenvalues: {data['L1_eigenvalues']}")

        # Open BC
        print(f"\nComputing {name} ({nx}x{ny} open)...")
        data_open = compute_lattice_data(name, nx, ny, "open")
        key_open = f"{name}_open"
        all_results[key_open] = data_open

        print(f"  n0={data_open['n_vertices']}, n1={data_open['n_edges']}, n2={data_open['n_faces']}")
        print(f"  beta_0={data_open['beta_0']}, beta_1={data_open['beta_1']}, beta_2={data_open['beta_2']}")
        print(f"  Euler: simplex={data_open['euler_simplex']}, consistent={data_open['euler_consistent']}")

    # Also compute XS (4x4) dimensions for comparison table
    xs_dims = {}
    for name in LATTICE_CONFIGS:
        gen = get_generator(name)
        lat = gen.build(4, 4, boundary="periodic")
        B1 = build_B1(lat.n_vertices, lat.edge_list)
        B2 = build_B2(lat.n_edges, lat.face_list, lat.edge_list)
        xs_dims[name] = {
            "n_vertices": lat.n_vertices,
            "n_edges": lat.n_edges,
            "n_faces": lat.n_faces,
            "B1_dims": f"{lat.n_vertices}x{lat.n_edges}",
            "B2_dims": f"{lat.n_edges}x{lat.n_faces}",
            "entries_B1": lat.n_vertices * lat.n_edges,
        }
    all_results["xs_4x4_dimensions"] = xs_dims

    # Save
    output_path = os.path.join(os.path.dirname(__file__), "..", "results", "worked_examples.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return all_results


if __name__ == "__main__":
    main()
