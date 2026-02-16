"""Orchestrate full spectral catalog computation across all lattice configurations."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.lattices.registry import LATTICE_REGISTRY
from src.topology.incidence import build_B1, build_B2, verify_chain_complex
from src.topology.laplacians import build_all_laplacians
from src.spectral.eigensolve import eigendecompose, count_zero_eigenvalues, extract_harmonic_basis
from src.spectral.betti import compute_betti_numbers, validate_harmonic_vectors


SIZE_CONFIGS = {
    "XS": (4, 4),
    "S": (10, 10),
    "M": (20, 20),
    "L": (50, 50),
    "XL": (100, 100),
}


@dataclass
class SpectralResult:
    """Bundle of all computed spectral quantities for one lattice configuration."""
    lattice_name: str
    size_label: str
    nx: int
    ny: int
    boundary: str
    face_strategy: str  # 'all' or 'none'

    # Counts
    n_vertices: int = 0
    n_edges: int = 0
    n_faces: int = 0
    coordination_dist: Dict[int, int] = field(default_factory=dict)

    # Betti numbers
    beta_0: int = 0
    beta_1: int = 0
    beta_2: int = 0
    euler_consistent: bool = False
    method_agreement: bool = False

    # Spectral data
    L0_eigenvalues: Optional[np.ndarray] = None
    L1_eigenvalues: Optional[np.ndarray] = None
    L1_down_eigenvalues: Optional[np.ndarray] = None
    L1_up_eigenvalues: Optional[np.ndarray] = None
    harmonic_basis: Optional[np.ndarray] = None

    # Spectral gaps
    L0_spectral_gap: float = 0.0
    L1_spectral_gap: float = 0.0

    # Validation
    chain_complex_valid: bool = False
    harmonic_validation: Optional[List[Dict]] = None

    # Timing
    compute_time_seconds: float = 0.0


def _spectral_gap(eigenvalues: np.ndarray, tol: float = 1e-10) -> float:
    """Smallest nonzero eigenvalue."""
    nonzero = eigenvalues[np.abs(eigenvalues) >= tol]
    if len(nonzero) == 0:
        return 0.0
    return float(np.min(np.abs(nonzero)))


def run_single(
    lattice_name: str,
    size_label: str,
    face_strategy: str = "all",
    boundary: str = "periodic",
    eigensolve_method: str = "auto",
    eigensolve_k: Optional[int] = None,
) -> SpectralResult:
    """Compute full spectral characterization for one lattice configuration."""
    t0 = time.time()

    nx, ny = SIZE_CONFIGS[size_label]
    gen = LATTICE_REGISTRY[lattice_name]()
    lat = gen.build(nx, ny, boundary=boundary)

    result = SpectralResult(
        lattice_name=lattice_name,
        size_label=size_label,
        nx=nx,
        ny=ny,
        boundary=boundary,
        face_strategy=face_strategy,
        n_vertices=lat.n_vertices,
        n_edges=lat.n_edges,
    )

    # Coordination distribution
    unique, counts = np.unique(lat.coordination, return_counts=True)
    result.coordination_dist = {int(u): int(c) for u, c in zip(unique, counts)}

    # Build incidence matrices
    B1 = build_B1(lat.n_vertices, lat.edge_list)

    if face_strategy == "all":
        B2 = build_B2(lat.n_edges, lat.face_list, lat.edge_list)
        result.n_faces = lat.n_faces
    else:
        B2 = build_B2(lat.n_edges, [], lat.edge_list)
        result.n_faces = 0

    result.chain_complex_valid = verify_chain_complex(B1, B2)

    # Build Laplacians
    laps = build_all_laplacians(B1, B2)

    # Eigendecomposition
    L0_result = eigendecompose(laps["L0"], method=eigensolve_method, k=eigensolve_k)
    L1_result = eigendecompose(laps["L1"], method=eigensolve_method, k=eigensolve_k)

    result.L0_eigenvalues = L0_result["eigenvalues"]
    result.L1_eigenvalues = L1_result["eigenvalues"]

    # Also decompose L1_down and L1_up for spectral comparison
    # Skip for large matrices to save time and memory
    n1 = lat.n_edges
    if n1 <= 20000:
        L1d_result = eigendecompose(laps["L1_down"], method=eigensolve_method, k=eigensolve_k)
        result.L1_down_eigenvalues = L1d_result["eigenvalues"]

        L1u_result = eigendecompose(laps["L1_up"], method=eigensolve_method, k=eigensolve_k)
        result.L1_up_eigenvalues = L1u_result["eigenvalues"]

    # Spectral gaps
    result.L0_spectral_gap = _spectral_gap(result.L0_eigenvalues)
    result.L1_spectral_gap = _spectral_gap(result.L1_eigenvalues)

    # Betti numbers
    betti = compute_betti_numbers(
        B1, B2, result.L0_eigenvalues, result.L1_eigenvalues,
        boundary=boundary,
    )
    result.beta_0 = betti["beta_0"]
    result.beta_1 = betti["beta_1"]
    result.beta_2 = betti["beta_2"]
    result.euler_consistent = betti["euler_consistent"]
    result.method_agreement = betti["method_agreement"]

    # Harmonic basis
    result.harmonic_basis = extract_harmonic_basis(
        L1_result["eigenvectors"], L1_result["eigenvalues"]
    )

    # Validate harmonic vectors
    if result.harmonic_basis.shape[1] > 0:
        result.harmonic_validation = validate_harmonic_vectors(
            B1, B2, result.harmonic_basis
        )

    result.compute_time_seconds = time.time() - t0
    return result


def run_full_catalog(
    lattices: Optional[List[str]] = None,
    sizes: Optional[List[str]] = None,
    strategies: Optional[List[str]] = None,
    boundary: str = "periodic",
    save_dir: Optional[str] = None,
) -> List[SpectralResult]:
    """Run catalog across all specified configurations.

    Args:
        lattices: List of lattice names (default: all).
        sizes: List of size labels (default: ['XS', 'S', 'M']).
        strategies: List of face strategies (default: ['all', 'none']).
        boundary: Boundary condition.
        save_dir: If provided, save each result immediately after computation.
            This prevents loss of results if the process is killed.

    Returns:
        List of SpectralResult objects.
    """
    if lattices is None:
        lattices = sorted(LATTICE_REGISTRY.keys())
    if sizes is None:
        sizes = ["XS", "S", "M"]
    if strategies is None:
        strategies = ["all", "none"]

    if save_dir is not None:
        import os
        from src.io.serialize import save_result
        os.makedirs(save_dir, exist_ok=True)

    results = []
    total = len(lattices) * len(sizes) * len(strategies)
    count = 0

    for lattice_name in lattices:
        for size_label in sizes:
            for strategy in strategies:
                count += 1
                print(f"[{count}/{total}] {lattice_name} {size_label} {strategy}...", end=" ", flush=True)
                try:
                    result = run_single(
                        lattice_name=lattice_name,
                        size_label=size_label,
                        face_strategy=strategy,
                        boundary=boundary,
                    )
                    print(f"beta_1={result.beta_1}, {result.compute_time_seconds:.1f}s")
                    results.append(result)
                    if save_dir is not None:
                        save_result(result, save_dir)
                except Exception as e:
                    print(f"FAILED: {e}")

    return results
