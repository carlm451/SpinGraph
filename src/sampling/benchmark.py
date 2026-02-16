"""MCMC benchmark suite for ice state sampling.

Wraps existing sample_ice_states() with timing and quality metrics.
Tests multiple n_flips_between values to characterize mixing.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy import sparse

from src.lattices.registry import get_generator
from src.neural.metrics import (
    batch_ice_rule_violation,
    mean_hamming_distance,
)
from src.topology.ice_sampling import sample_ice_states, verify_ice_state
from src.topology.incidence import build_B1

logger = logging.getLogger(__name__)


# Size configs matching spectral catalog
SIZE_CONFIGS = {
    "XS": (4, 4),
    "S": (10, 10),
    "M": (20, 20),
    "L": (50, 50),
}


@dataclass
class MCMCBenchmarkResult:
    """Result from a single MCMC benchmark run."""

    lattice_name: str
    size_label: str
    boundary: str
    n_samples: int
    n_flips_between: int
    wall_time_seconds: float
    time_per_sample: float
    mean_energy: float
    energy_std: float
    ice_rule_violation_rate: float
    mean_hamming_distance: float
    hamming_std: float
    unique_fraction: float
    n_vertices: int
    n_edges: int
    autocorrelation_time: Optional[float] = None


def estimate_autocorrelation_time(
    samples: np.ndarray,
    max_lag: int = 50,
) -> float:
    """Estimate autocorrelation time from Hamming overlap series.

    Computes C(t) = <sigma_i . sigma_{i+t}> / n_edges averaged over i,
    then integrates to get tau = 1 + 2 * sum_{t>0} C(t)/C(0).

    Parameters
    ----------
    samples : array (n_samples, n_edges) of +1/-1
    max_lag : int

    Returns
    -------
    tau : float
        Integrated autocorrelation time in units of samples.
    """
    n_samples, n_edges = samples.shape
    if n_samples < 3:
        return float("nan")

    max_lag = min(max_lag, n_samples // 3)

    # Compute autocorrelation via dot products
    # C(t) = mean_i [sigma_i . sigma_{i+t}] / n_edges
    correlations = np.zeros(max_lag + 1)
    for t in range(max_lag + 1):
        n_pairs = n_samples - t
        if n_pairs <= 0:
            break
        dots = np.sum(samples[:n_pairs] * samples[t : t + n_pairs], axis=1)
        correlations[t] = np.mean(dots) / n_edges

    # Normalize by C(0)
    if abs(correlations[0]) < 1e-10:
        return float("nan")
    correlations = correlations / correlations[0]

    # Integrate: tau = 1 + 2 * sum_{t=1}^{max_lag} C(t)
    # Cut off when C(t) < 0 (noise regime)
    tau = 1.0
    for t in range(1, max_lag + 1):
        if correlations[t] < 0:
            break
        tau += 2.0 * correlations[t]

    return tau


def run_mcmc_benchmark(
    lattice_name: str,
    size_label: str,
    boundary: str = "periodic",
    n_samples: int = 1000,
    n_flips_between: int = 20,
    seed: Optional[int] = 42,
) -> MCMCBenchmarkResult:
    """Run MCMC benchmark for a single lattice configuration.

    Parameters
    ----------
    lattice_name : str
    size_label : str (XS, S, M, L)
    boundary : str
    n_samples : int
    n_flips_between : int
    seed : int, optional

    Returns
    -------
    MCMCBenchmarkResult
    """
    nx_size, ny_size = SIZE_CONFIGS[size_label]

    # Build lattice
    gen = get_generator(lattice_name)
    result = gen.build(nx_size, ny_size, boundary=boundary)

    B1 = build_B1(result.n_vertices, result.edge_list)

    logger.info(
        f"Benchmarking {lattice_name} {size_label} ({boundary}): "
        f"n0={result.n_vertices}, n1={result.n_edges}, "
        f"n_samples={n_samples}, n_flips={n_flips_between}"
    )

    # Time the sampling
    t0 = time.perf_counter()
    states = sample_ice_states(
        B1,
        result.coordination,
        n_samples=n_samples,
        n_flips_between=n_flips_between,
        seed=seed,
        edge_list=result.edge_list,
    )
    wall_time = time.perf_counter() - t0

    samples = np.array(states)

    # Quality metrics
    violation = batch_ice_rule_violation(samples, B1, result.coordination)
    h_mean, h_std = mean_hamming_distance(samples)

    # Unique fraction
    unique_set = set()
    for s in samples:
        unique_set.add(tuple(s.astype(np.int8)))
    unique_frac = len(unique_set) / len(samples)

    # Energy (sigma^T L_equ sigma -- should be 0 for ice states)
    L_equ = B1.T @ B1
    energies = np.array([float(s @ L_equ @ s) for s in samples])

    # Autocorrelation
    tau = estimate_autocorrelation_time(samples)

    return MCMCBenchmarkResult(
        lattice_name=lattice_name,
        size_label=size_label,
        boundary=boundary,
        n_samples=n_samples,
        n_flips_between=n_flips_between,
        wall_time_seconds=wall_time,
        time_per_sample=wall_time / n_samples,
        mean_energy=float(np.mean(energies)),
        energy_std=float(np.std(energies)),
        ice_rule_violation_rate=violation,
        mean_hamming_distance=h_mean,
        hamming_std=h_std,
        unique_fraction=unique_frac,
        n_vertices=result.n_vertices,
        n_edges=result.n_edges,
        autocorrelation_time=tau,
    )


def run_full_benchmark_suite(
    lattices: Optional[List[str]] = None,
    sizes: Optional[List[str]] = None,
    boundary: str = "periodic",
    n_samples: int = 1000,
    flip_values: Optional[List[int]] = None,
    seed: int = 42,
) -> Dict[str, List[MCMCBenchmarkResult]]:
    """Run MCMC benchmarks across lattices, sizes, and flip values.

    Parameters
    ----------
    lattices : list of lattice names (default: all 5)
    sizes : list of size labels (default: XS, S, M)
    boundary : boundary condition
    n_samples : samples per run
    flip_values : list of n_flips_between values to test
    seed : random seed

    Returns
    -------
    results : dict mapping "lattice_size" -> list of MCMCBenchmarkResult
    """
    if lattices is None:
        lattices = ["square", "kagome", "shakti", "tetris", "santa_fe"]
    if sizes is None:
        sizes = ["XS", "S", "M"]
    if flip_values is None:
        flip_values = [10, 20, 50, 100]

    all_results = {}

    for lattice in lattices:
        for size in sizes:
            key = f"{lattice}_{size}"
            all_results[key] = []

            for n_flips in flip_values:
                logger.info(f"Running: {key}, n_flips_between={n_flips}")
                try:
                    result = run_mcmc_benchmark(
                        lattice, size, boundary,
                        n_samples=n_samples,
                        n_flips_between=n_flips,
                        seed=seed,
                    )
                    all_results[key].append(result)
                    logger.info(
                        f"  -> {result.wall_time_seconds:.2f}s, "
                        f"hamming={result.mean_hamming_distance:.4f}, "
                        f"unique={result.unique_fraction:.4f}, "
                        f"tau={result.autocorrelation_time:.2f}"
                    )
                except Exception as e:
                    logger.error(f"  -> FAILED: {e}")

    return all_results


def compare_samplers(
    mcmc_results: Dict[str, MCMCBenchmarkResult],
    neural_results: Dict[str, dict],
) -> Dict[str, dict]:
    """Head-to-head comparison of MCMC vs neural sampling.

    Parameters
    ----------
    mcmc_results : dict mapping config key -> MCMCBenchmarkResult
    neural_results : dict mapping config key -> dict with keys:
        'wall_time', 'hamming_mean', 'unique_fraction', 'kl', 'ess'

    Returns
    -------
    comparison : dict mapping config key -> comparison dict
    """
    comparison = {}

    for key in mcmc_results:
        if key not in neural_results:
            continue

        mcmc = mcmc_results[key]
        neural = neural_results[key]

        comparison[key] = {
            "lattice": mcmc.lattice_name,
            "size": mcmc.size_label,
            "mcmc_time": mcmc.wall_time_seconds,
            "neural_time": neural.get("wall_time", float("nan")),
            "speedup": mcmc.wall_time_seconds / max(neural.get("wall_time", 1e-10), 1e-10),
            "mcmc_hamming": mcmc.mean_hamming_distance,
            "neural_hamming": neural.get("hamming_mean", float("nan")),
            "mcmc_unique": mcmc.unique_fraction,
            "neural_unique": neural.get("unique_fraction", float("nan")),
            "neural_kl": neural.get("kl", float("nan")),
            "neural_ess": neural.get("ess", float("nan")),
            "mcmc_violation": mcmc.ice_rule_violation_rate,
            "mcmc_tau": mcmc.autocorrelation_time,
        }

    return comparison
