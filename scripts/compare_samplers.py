#!/usr/bin/env python3
"""Head-to-head comparison of MCMC vs neural sampling.

Usage:
    python -m scripts.compare_samplers [--lattice square] [--size XS]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np
import torch

from src.lattices.registry import get_generator
from src.neural.enumeration import enumerate_all_ice_states
from src.neural.loop_basis import (
    compute_loop_ordering,
    extract_loop_basis,
)
from src.neural.loop_mpvan import LoopMPVAN
from src.neural.metrics import (
    batch_ice_rule_violation,
    effective_sample_size,
    kl_from_samples,
    mean_hamming_distance,
)
from src.neural.operators import build_eign_operators
from src.neural.training import TrainingConfig, build_inv_features, train
from src.sampling.benchmark import run_mcmc_benchmark
from src.topology.ice_sampling import find_seed_ice_state
from src.topology.incidence import build_B1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Compare MCMC vs neural sampling")
    parser.add_argument("--lattice", default="square")
    parser.add_argument("--size", default="XS")
    parser.add_argument("--boundary", default="open")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger.info(f"=== Comparing samplers: {args.lattice} {args.size} ({args.boundary}) ===")

    # Build lattice
    gen = get_generator(args.lattice)
    nx_size, ny_size = {"XS": (4, 4), "S": (10, 10), "M": (20, 20)}[args.size]
    lattice = gen.build(nx_size, ny_size, boundary=args.boundary)
    B1 = build_B1(lattice.n_vertices, lattice.edge_list)

    sigma_seed = find_seed_ice_state(B1, lattice.coordination, edge_list=lattice.edge_list)

    # === MCMC Benchmark ===
    logger.info("\n--- MCMC Sampling ---")
    mcmc_result = run_mcmc_benchmark(
        args.lattice, args.size, args.boundary,
        n_samples=args.n_samples, n_flips_between=20, seed=args.seed,
    )
    logger.info(f"MCMC: {mcmc_result.wall_time_seconds:.2f}s, "
                f"hamming={mcmc_result.mean_hamming_distance:.4f}, "
                f"unique={mcmc_result.unique_fraction:.4f}")

    # === Neural Setup & Training ===
    logger.info("\n--- Neural Sampler ---")
    ops = build_eign_operators(B1)
    loop_basis = extract_loop_basis(lattice.graph, B1, lattice.edge_list)
    loop_basis.ordering = compute_loop_ordering(
        loop_basis, strategy="spatial_bfs",
        positions=lattice.positions, edge_list=lattice.edge_list,
    )

    # Exact enumeration if feasible (directed-cycle-aware)
    exact_states = None
    if loop_basis.n_loops <= 20:
        exact_states = enumerate_all_ice_states(
            sigma_seed, loop_basis.loop_indicators.numpy(),
            B1, lattice.coordination,
            cycle_edge_lists=loop_basis.cycle_edge_lists,
            ordering=loop_basis.ordering,
        )
        logger.info(f"Exact enumeration: {len(exact_states)} reachable states")

    model = LoopMPVAN(
        operators=ops, loop_basis=loop_basis,
        n_layers=3, equ_dim=16, inv_dim=8, head_hidden=32,
        B1_csc=B1,
    )
    inv_features = build_inv_features(lattice.edge_list, lattice.coordination)

    config = TrainingConfig(
        n_epochs=args.epochs, batch_size=64, lr=1e-3, seed=args.seed,
        eval_every=max(1, args.epochs // 5),
    )

    t_train_start = time.perf_counter()
    train_result = train(
        model, sigma_seed, inv_features,
        B1, lattice.coordination, config,
        exact_states=exact_states,
    )
    train_time = time.perf_counter() - t_train_start

    # Neural sampling (post-training)
    model.eval()
    seed_tensor = torch.from_numpy(sigma_seed.astype(np.float32))

    t_sample_start = time.perf_counter()
    with torch.no_grad():
        neural_sigmas, neural_log_probs = model.sample(
            seed_tensor, inv_features, n_samples=args.n_samples,
        )
    neural_sample_time = time.perf_counter() - t_sample_start

    neural_np = neural_sigmas.numpy()
    neural_lp = neural_log_probs.numpy()

    # Neural metrics
    n_violation = batch_ice_rule_violation(neural_np, B1, lattice.coordination)
    n_hamming, n_h_std = mean_hamming_distance(neural_np)
    n_ess = effective_sample_size(neural_lp)

    unique_set = set()
    for s in neural_np:
        unique_set.add(tuple(s.astype(np.int8)))
    n_unique = len(unique_set) / len(neural_np)

    n_kl = float("nan")
    if exact_states is not None:
        n_kl = kl_from_samples(neural_np, exact_states)

    # === Comparison ===
    logger.info("\n=== Comparison Results ===")
    logger.info(f"{'Metric':<25} {'MCMC':>15} {'Neural':>15}")
    logger.info("-" * 55)
    logger.info(f"{'Wall time (sampling)':<25} {mcmc_result.wall_time_seconds:>15.3f} {neural_sample_time:>15.3f}")
    logger.info(f"{'Time per sample':<25} {mcmc_result.time_per_sample:>15.6f} {neural_sample_time/args.n_samples:>15.6f}")
    logger.info(f"{'Mean Hamming dist':<25} {mcmc_result.mean_hamming_distance:>15.4f} {n_hamming:>15.4f}")
    logger.info(f"{'Unique fraction':<25} {mcmc_result.unique_fraction:>15.4f} {n_unique:>15.4f}")
    logger.info(f"{'Ice violations':<25} {mcmc_result.ice_rule_violation_rate:>15.6f} {n_violation:>15.6f}")
    logger.info(f"{'ESS':<25} {'N/A':>15} {n_ess:>15.1f}")
    if exact_states is not None:
        logger.info(f"{'KL(q || uniform)':<25} {'N/A':>15} {n_kl:>15.6f}")
    logger.info(f"\n{'Neural training time':<25} {train_time:>15.1f}s")
    logger.info(f"{'Neural parameters':<25} {model.count_parameters():>15}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
