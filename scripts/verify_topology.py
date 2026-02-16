#!/usr/bin/env python3
"""Quick end-to-end topology check for corrected tetris and santa_fe."""
import sys
sys.path.insert(0, "/Users/carlmerrigan/DeckerCode/SpinIceTDL")

import numpy as np
from scipy.linalg import eigh
from src.lattices.tetris import TetrisGenerator
from src.lattices.santa_fe import SantaFeGenerator
from src.topology.incidence import build_B1, build_B2
from src.topology.laplacians import build_all_laplacians


def check_lattice(name, generator, nx=4, ny=4):
    print(f"\n{'='*60}")
    print(f"  {name} ({nx}x{ny} periodic)")
    print(f"{'='*60}")

    result = generator.build(nx, ny, boundary="periodic")
    print(f"  V={result.n_vertices}, E={result.n_edges}, F={result.n_faces}")
    print(f"  Coordination: {result.coordination_distribution}")

    n_cells = nx * ny
    print(f"  Per cell: V={result.n_vertices/n_cells:.1f}, "
          f"E={result.n_edges/n_cells:.1f}, F={result.n_faces/n_cells:.1f}")

    # Build incidence matrices
    B1 = build_B1(result.n_vertices, result.edge_list)
    B2 = build_B2(result.n_edges, result.face_list, result.edge_list)
    print(f"  B1: {B1.shape}, B2: {B2.shape}")

    # Chain complex: B1 @ B2 = 0
    if B2.shape[1] > 0:
        product = (B1 @ B2).toarray()
        print(f"  B1@B2=0 check: max|err| = {np.abs(product).max():.2e}")

    # Build Laplacians
    laps = build_all_laplacians(B1, B2)
    L0 = laps['L0'].toarray()
    L1 = laps['L1'].toarray()

    # Eigenvalues
    evals_L0 = eigh(L0, eigvals_only=True)
    evals_L1 = eigh(L1, eigvals_only=True)

    tol = 1e-10
    beta0 = int(np.sum(np.abs(evals_L0) < tol))
    beta1 = int(np.sum(np.abs(evals_L1) < tol))

    # Rank-nullity Betti numbers
    rank_B1 = np.linalg.matrix_rank(B1.toarray(), tol=1e-10)
    rank_B2 = np.linalg.matrix_rank(B2.toarray(), tol=1e-10) if B2.shape[1] > 0 else 0
    beta1_rn = result.n_edges - rank_B1 - rank_B2

    chi_betti = beta0 - beta1
    chi_count = result.n_vertices - result.n_edges + result.n_faces
    print(f"  beta0={beta0}, beta1(spectral)={beta1}, beta1(rank-nullity)={beta1_rn}")
    print(f"  Euler: betti={chi_betti}, counts={chi_count} (should be 0)")

    nonzero = evals_L1[np.abs(evals_L1) >= tol]
    gap = nonzero.min() if len(nonzero) > 0 else float('inf')
    print(f"  L1 spectral gap: {gap:.6f}")
    print(f"  beta1 matches? {beta1 == beta1_rn}")


if __name__ == "__main__":
    check_lattice("Tetris", TetrisGenerator())
    check_lattice("Santa Fe", SantaFeGenerator())
