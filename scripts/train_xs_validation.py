#!/usr/bin/env python3
"""Train LoopMPVAN on square XS and validate against exact enumeration.

Usage:
    python -m scripts.train_xs_validation [--boundary open|periodic] [--epochs 2000]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch

from src.lattices.registry import get_generator
from src.neural.checkpointing import RunMetadata, generate_run_id, save_training_run
from src.neural.enumeration import enumerate_all_ice_states
from src.neural.loop_basis import (
    LoopBasis,
    compute_loop_ordering,
    extract_loop_basis,
)
from src.neural.loop_mpvan import LoopMPVAN
from src.neural.metrics import (
    batch_ice_rule_violation,
    kl_from_samples,
    mean_hamming_distance,
)
from src.neural.operators import build_eign_operators
from src.neural.training import TrainingConfig, build_inv_features, train
from src.topology.ice_sampling import find_seed_ice_state, verify_ice_state
from src.topology.incidence import build_B1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train on square XS, compare to exact enumeration")
    parser.add_argument("--boundary", default="open", choices=["open", "periodic"])
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--equ-dim", type=int, default=16)
    parser.add_argument("--inv-dim", type=int, default=8)
    parser.add_argument("--run-id", type=str, default=None,
                        help="Run ID (default: auto-generated from lattice+boundary+timestamp)")
    args = parser.parse_args()

    logger.info(f"=== Training LoopMPVAN on square XS ({args.boundary} BC) ===")

    # Build lattice
    gen = get_generator("square")
    lattice = gen.build(4, 4, boundary=args.boundary)
    logger.info(f"Lattice: n0={lattice.n_vertices}, n1={lattice.n_edges}")

    # Build incidence matrix
    B1 = build_B1(lattice.n_vertices, lattice.edge_list)

    # Find seed ice state
    sigma_seed = find_seed_ice_state(B1, lattice.coordination, edge_list=lattice.edge_list)
    assert verify_ice_state(B1, sigma_seed, lattice.coordination), "Seed state invalid!"
    logger.info("Seed ice state found and verified")

    # Build EIGN operators
    ops = build_eign_operators(B1)
    logger.info(f"EIGN operators built: n0={ops.n0}, n1={ops.n1}")

    # Extract loop basis
    loop_basis = extract_loop_basis(lattice.graph, B1, lattice.edge_list)
    loop_basis.ordering = compute_loop_ordering(
        loop_basis, strategy="spatial_bfs",
        positions=lattice.positions, edge_list=lattice.edge_list,
    )
    logger.info(f"Loop basis: {loop_basis.n_loops} loops (beta_1)")

    # Exact enumeration (directed-cycle-aware)
    logger.info(f"Enumerating reachable ice states (beta_1={loop_basis.n_loops})...")
    t0 = time.perf_counter()
    exact_states = enumerate_all_ice_states(
        sigma_seed,
        loop_basis.loop_indicators.numpy(),
        B1, lattice.coordination,
        cycle_edge_lists=loop_basis.cycle_edge_lists,
        ordering=loop_basis.ordering,
    )
    enum_time = time.perf_counter() - t0
    logger.info(f"Enumeration complete: {len(exact_states)} reachable states in {enum_time:.2f}s")

    # Build model
    model = LoopMPVAN(
        operators=ops,
        loop_basis=loop_basis,
        n_layers=args.n_layers,
        equ_dim=args.equ_dim,
        inv_dim=args.inv_dim,
        head_hidden=32,
        B1_csc=B1,
    )
    logger.info(f"Model: {model.count_parameters()} parameters")

    # Build invariant features
    inv_features = build_inv_features(lattice.edge_list, lattice.coordination)

    # Training config
    config = TrainingConfig(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        eval_every=max(1, args.epochs // 10),
    )

    # Train
    logger.info("Starting training...")
    t0 = time.perf_counter()
    result = train(
        model, sigma_seed, inv_features,
        B1, lattice.coordination, config,
        exact_states=exact_states,
    )
    train_time = time.perf_counter() - t0
    logger.info(f"Training complete in {train_time:.1f}s")

    # Final evaluation
    logger.info("\n=== Final Evaluation ===")
    model.eval()
    seed_tensor = torch.from_numpy(sigma_seed.astype(np.float32))

    t0_sample = time.perf_counter()
    with torch.no_grad():
        eval_sigmas, eval_log_probs = model.sample(
            seed_tensor, inv_features, n_samples=min(2000, 2 ** loop_basis.n_loops * 4)
        )
    sample_time = time.perf_counter() - t0_sample

    eval_np = eval_sigmas.numpy()
    eval_lp_np = eval_log_probs.numpy()

    # Check all samples valid
    violation = batch_ice_rule_violation(eval_np, B1, lattice.coordination)
    logger.info(f"Ice rule violation rate: {violation:.6f}")

    # Hamming distance
    h_mean, h_std = mean_hamming_distance(eval_np)
    logger.info(f"Mean Hamming distance: {h_mean:.4f} (+/- {h_std:.4f})")

    # KL divergence
    kl = kl_from_samples(eval_np, exact_states)
    logger.info(f"KL(empirical || uniform): {kl:.6f}")

    # State coverage
    unique_set = set()
    for s in eval_np:
        unique_set.add(tuple(s.astype(np.int8)))
    coverage = len(unique_set) / len(exact_states)
    logger.info(f"State coverage: {len(unique_set)}/{len(exact_states)} = {coverage:.4f}")

    # Summary
    logger.info("\n=== Summary ===")
    logger.info(f"Lattice: square XS ({args.boundary})")
    logger.info(f"beta_1 = {loop_basis.n_loops}")
    logger.info(f"Total ice states = {len(exact_states)}")
    logger.info(f"Model parameters = {model.count_parameters()}")
    logger.info(f"Training time = {train_time:.1f}s")
    logger.info(f"Final KL = {kl:.6f}")
    logger.info(f"Final Hamming = {h_mean:.4f}")
    logger.info(f"Ice violations = {violation:.6f}")
    logger.info(f"State coverage = {coverage:.4f}")

    # Save training run
    run_id = args.run_id or generate_run_id("square", args.boundary)
    run_dir = os.path.join("results", "neural_training", run_id)

    metadata = RunMetadata(
        run_id=run_id,
        lattice_name="square",
        size_label="XS",
        boundary=args.boundary,
        n_vertices=lattice.n_vertices,
        n_edges=lattice.n_edges,
        beta_1=loop_basis.n_loops,
        n_reachable_states=len(exact_states),
        n_model_params=model.count_parameters(),
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_scheduler=config.lr_scheduler,
        entropy_bonus=config.entropy_bonus,
        grad_clip=config.grad_clip,
        eval_every=config.eval_every,
        seed=args.seed,
        n_layers=args.n_layers,
        equ_dim=args.equ_dim,
        inv_dim=args.inv_dim,
        head_hidden=32,
        enum_time_s=enum_time,
        train_time_s=train_time,
        sample_time_s=sample_time,
        timestamp=datetime.now().isoformat(),
    )

    save_training_run(
        run_dir=run_dir,
        metadata=metadata,
        result=result,
        model=model,
        exact_states=exact_states,
        final_samples=eval_np,
        final_log_probs=eval_lp_np,
        sigma_seed=sigma_seed,
        positions=lattice.positions,
        edge_list=lattice.edge_list,
        coordination=lattice.coordination,
    )
    logger.info(f"\nRun saved to: {run_dir}")

    return 0 if violation == 0.0 else 1


if __name__ == "__main__":
    sys.exit(main())
