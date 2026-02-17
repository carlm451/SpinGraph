#!/usr/bin/env python3
"""Train LoopMPVAN on any lattice from the zoo at any size.

For small beta_1 (<=25), validates against exact enumeration.
For larger beta_1, evaluates via ice-rule violation, Hamming distance, and ESS.

Usage:
    python -m scripts.train_square --lattice square --nx 6 --ny 6 [--boundary open] [--epochs 2000]
    python -m scripts.train_square --lattice shakti --nx 2 --ny 2 --boundary open --epochs 2000
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
from src.neural.loop_basis import (
    compute_loop_ordering,
    extract_loop_basis,
)
from src.neural.loop_mpvan import LoopMPVAN
from src.neural.metrics import (
    batch_ice_rule_violation,
    kl_from_samples,
    mean_hamming_distance,
    effective_sample_size,
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

# Maximum beta_1 for exact enumeration
MAX_ENUM_BETA1 = 25


def _size_label(nx: int) -> str:
    """Map grid dimension to size label."""
    if nx <= 4:
        return "XS"
    elif nx <= 6:
        return "S-"
    elif nx <= 10:
        return "S"
    elif nx <= 20:
        return "M"
    elif nx <= 50:
        return "L"
    else:
        return "XL"


def main():
    parser = argparse.ArgumentParser(
        description="Train LoopMPVAN on any lattice from the zoo"
    )
    parser.add_argument("--lattice", default="square",
                        help="Lattice type (square, kagome, shakti, tetris, etc.)")
    parser.add_argument("--nx", type=int, required=True, help="Grid width")
    parser.add_argument("--ny", type=int, default=None, help="Grid height (default: same as nx)")
    parser.add_argument("--boundary", default="open", choices=["open", "periodic"])
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--equ-dim", type=int, default=16)
    parser.add_argument("--inv-dim", type=int, default=8)
    parser.add_argument("--head-hidden", type=int, default=32)
    parser.add_argument("--entropy-bonus", type=float, default=0.01)
    parser.add_argument("--lr-min-factor", type=float, default=0.01,
                        help="Cosine LR schedule minimum as fraction of initial LR")
    parser.add_argument("--n-eval-samples", type=int, default=2000,
                        help="Number of samples for final evaluation")
    parser.add_argument("--skip-enum", action="store_true",
                        help="Skip exact enumeration even if beta_1 is small enough")
    parser.add_argument("--no-randomize-ordering", action="store_true",
                        help="Disable per-batch loop ordering randomization")
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    ny = args.ny or args.nx
    lattice_name = args.lattice
    size_label = _size_label(args.nx)

    logger.info(f"=== Training LoopMPVAN on {lattice_name} {args.nx}x{ny} ({args.boundary} BC) ===")

    # Build lattice
    gen = get_generator(lattice_name)
    lattice = gen.build(args.nx, ny, boundary=args.boundary)
    logger.info(f"Lattice: n0={lattice.n_vertices}, n1={lattice.n_edges}")
    logger.info(f"Coordination: {lattice.coordination_distribution}")

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

    # Exact enumeration (only if beta_1 small enough)
    exact_states = None
    enum_time = 0.0
    if loop_basis.n_loops <= MAX_ENUM_BETA1 and not args.skip_enum:
        from src.neural.enumeration import enumerate_all_ice_states, enumerate_multi_ordering

        logger.info(f"Enumerating reachable ice states (beta_1={loop_basis.n_loops})...")
        t0 = time.perf_counter()
        if not args.no_randomize_ordering:
            # With ordering randomization, enumerate across multiple orderings
            # to get the full set of states the model can discover
            exact_states, cumulative = enumerate_multi_ordering(
                sigma_seed,
                loop_basis.loop_indicators.numpy(),
                B1, lattice.coordination,
                cycle_edge_lists=loop_basis.cycle_edge_lists,
                n_orderings=200,
                seed=args.seed,
            )
            enum_time = time.perf_counter() - t0
            logger.info(
                f"Multi-ordering enumeration: {len(exact_states)} reachable states "
                f"across 200 orderings in {enum_time:.2f}s"
            )
        else:
            exact_states = enumerate_all_ice_states(
                sigma_seed,
                loop_basis.loop_indicators.numpy(),
                B1, lattice.coordination,
                cycle_edge_lists=loop_basis.cycle_edge_lists,
                ordering=loop_basis.ordering,
            )
            enum_time = time.perf_counter() - t0
            logger.info(f"Enumeration complete: {len(exact_states)} reachable states in {enum_time:.2f}s")
    else:
        logger.info(
            f"Skipping enumeration (beta_1={loop_basis.n_loops} > {MAX_ENUM_BETA1})"
            if not args.skip_enum else "Enumeration skipped by user"
        )

    # Build model
    model = LoopMPVAN(
        operators=ops,
        loop_basis=loop_basis,
        n_layers=args.n_layers,
        equ_dim=args.equ_dim,
        inv_dim=args.inv_dim,
        head_hidden=args.head_hidden,
        B1_csc=B1,
    )
    logger.info(f"Model: {model.count_parameters()} parameters")

    # Build invariant features
    inv_features = build_inv_features(lattice.edge_list, lattice.coordination)

    # Training config
    checkpoint_every = max(1, args.epochs // 10)  # Checkpoint at eval intervals
    config = TrainingConfig(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        entropy_bonus=args.entropy_bonus,
        lr_min_factor=args.lr_min_factor,
        seed=args.seed,
        eval_every=max(1, args.epochs // 10),
        checkpoint_every=checkpoint_every,
        randomize_ordering=not args.no_randomize_ordering,
    )

    # Create run directory before training (for checkpoints)
    run_id = args.run_id or generate_run_id(lattice_name, args.boundary)
    run_dir = os.path.join("results", "neural_training", run_id)
    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"Run directory: {run_dir}")

    # Train
    logger.info("Starting training...")
    t0 = time.perf_counter()
    result = train(
        model, sigma_seed, inv_features,
        B1, lattice.coordination, config,
        exact_states=exact_states,
        checkpoint_dir=run_dir,
    )
    train_time = time.perf_counter() - t0
    logger.info(f"Training complete in {train_time:.1f}s")

    # Final evaluation
    logger.info("\n=== Final Evaluation ===")
    model.eval()
    seed_tensor = torch.from_numpy(sigma_seed.astype(np.float32))
    n_loops_eval = loop_basis.n_loops

    n_eval = args.n_eval_samples
    t0_sample = time.perf_counter()
    with torch.no_grad():
        if config.randomize_ordering:
            # Sample with multiple random orderings to measure full coverage
            n_orderings_eval = 20
            samples_per = max(1, n_eval // n_orderings_eval)
            eval_sigmas_list = []
            eval_lp_list = []
            for _ in range(n_orderings_eval):
                eval_ord = np.random.permutation(n_loops_eval).tolist()
                s, lp = model.sample_batch(
                    seed_tensor, inv_features, n_samples=samples_per,
                    ordering=eval_ord,
                )
                eval_sigmas_list.append(s)
                eval_lp_list.append(lp)
            eval_sigmas = torch.cat(eval_sigmas_list, dim=0)
            eval_log_probs = torch.cat(eval_lp_list, dim=0)
            logger.info(
                f"Eval: {len(eval_sigmas)} samples across {n_orderings_eval} random orderings"
            )
        else:
            eval_sigmas, eval_log_probs = model.sample(
                seed_tensor, inv_features, n_samples=n_eval
            )
    sample_time = time.perf_counter() - t0_sample

    eval_np = eval_sigmas.numpy()
    eval_lp_np = eval_log_probs.numpy()

    # Ice rule violation
    violation = batch_ice_rule_violation(eval_np, B1, lattice.coordination)
    logger.info(f"Ice rule violation rate: {violation:.6f}")

    # Hamming distance
    h_mean, h_std = mean_hamming_distance(eval_np)
    logger.info(f"Mean Hamming distance: {h_mean:.4f} (+/- {h_std:.4f})")

    # ESS
    ess = effective_sample_size(eval_lp_np)
    logger.info(f"Effective sample size: {ess:.1f}")

    # KL divergence (only if we have exact states)
    kl = float("nan")
    if exact_states is not None:
        kl = kl_from_samples(eval_np, exact_states)
        logger.info(f"KL(empirical || uniform): {kl:.6f}")

    # State coverage
    unique_set = set()
    for s in eval_np:
        unique_set.add(tuple(s.astype(np.int8)))
    n_unique = len(unique_set)
    if exact_states is not None:
        coverage = n_unique / len(exact_states)
        logger.info(f"State coverage: {n_unique}/{len(exact_states)} = {coverage:.4f}")
    else:
        coverage = float("nan")
        logger.info(f"Unique states sampled: {n_unique}/{n_eval}")

    # Summary
    n_reachable = len(exact_states) if exact_states is not None else -1
    logger.info("\n=== Summary ===")
    logger.info(f"Lattice: {lattice_name} {args.nx}x{ny} ({args.boundary})")
    logger.info(f"Size label: {size_label}")
    logger.info(f"beta_1 = {loop_basis.n_loops}")
    logger.info(f"Reachable states = {n_reachable if n_reachable > 0 else 'not enumerated'}")
    logger.info(f"Model parameters = {model.count_parameters()}")
    logger.info(f"Training time = {train_time:.1f}s")
    logger.info(f"Final KL = {kl:.6f}" if not np.isnan(kl) else "Final KL = N/A (no enumeration)")
    logger.info(f"Final Hamming = {h_mean:.4f}")
    logger.info(f"Final ESS = {ess:.1f}")
    logger.info(f"Ice violations = {violation:.6f}")
    logger.info(f"Unique states = {n_unique}")
    logger.info(f"Sample time ({n_eval} samples) = {sample_time:.2f}s")

    # Save training run
    metadata = RunMetadata(
        run_id=run_id,
        lattice_name=lattice_name,
        size_label=size_label,
        boundary=args.boundary,
        n_vertices=lattice.n_vertices,
        n_edges=lattice.n_edges,
        beta_1=loop_basis.n_loops,
        n_reachable_states=n_reachable,
        n_model_params=model.count_parameters(),
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_scheduler=config.lr_scheduler,
        entropy_bonus=config.entropy_bonus,
        lr_min_factor=config.lr_min_factor,
        grad_clip=config.grad_clip,
        eval_every=config.eval_every,
        seed=args.seed,
        n_layers=args.n_layers,
        equ_dim=args.equ_dim,
        inv_dim=args.inv_dim,
        head_hidden=args.head_hidden,
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
