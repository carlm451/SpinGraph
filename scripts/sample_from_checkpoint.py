#!/usr/bin/env python3
"""Generate new samples from a saved training run or intermediate checkpoint.

Rebuilds the LoopMPVAN model from saved config + lattice geometry, loads
model weights from a checkpoint, generates samples, and reports metrics.

Usage:
    # Use final model weights:
    python -m scripts.sample_from_checkpoint --run-dir results/neural_training/square_open_20260216_164125

    # Use an intermediate checkpoint:
    python -m scripts.sample_from_checkpoint --run-dir results/neural_training/square_open_20260216_164125 --checkpoint model_epoch_1000.pt

    # Generate more samples with a different seed:
    python -m scripts.sample_from_checkpoint --run-dir results/neural_training/square_open_20260216_164125 --n-samples 5000 --seed 123
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch

from src.lattices.registry import get_generator
from src.neural.loop_basis import extract_loop_basis, compute_loop_ordering
from src.neural.loop_mpvan import LoopMPVAN
from src.neural.metrics import (
    batch_ice_rule_violation,
    effective_sample_size,
    kl_from_samples,
    mean_hamming_distance,
)
from src.neural.operators import build_eign_operators
from src.neural.training import build_inv_features
from src.topology.ice_sampling import find_seed_ice_state, verify_ice_state
from src.topology.incidence import build_B1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _derive_grid_size(lattice_name: str, boundary: str, n_vertices: int, n_edges: int):
    """Derive nx, ny by trial-building lattices from the registry."""
    gen = get_generator(lattice_name)
    for nx in range(1, 200):
        try:
            lat = gen.build(nx, nx, boundary=boundary)
        except Exception:
            continue
        if lat.n_vertices == n_vertices and lat.n_edges == n_edges:
            return nx, nx
    raise ValueError(
        f"Cannot derive grid size for {lattice_name} ({boundary}) "
        f"with n_vertices={n_vertices}, n_edges={n_edges}"
    )


def rebuild_model(run_dir: str, checkpoint_name: str = "model_final.pt"):
    """Rebuild LoopMPVAN from a saved run and load checkpoint weights.

    Parameters
    ----------
    run_dir : str
        Path to the saved training run directory.
    checkpoint_name : str
        Filename of the checkpoint to load (default: model_final.pt).

    Returns
    -------
    model : LoopMPVAN
    sigma_seed : np.ndarray
    inv_features : torch.Tensor
    B1 : scipy sparse
    coordination : np.ndarray
    config : dict
    exact_states : np.ndarray or None
    """
    # Load config
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    lattice_name = config["lattice_name"]
    boundary = config["boundary"]
    n_vertices = config["n_vertices"]
    n_edges = config["n_edges"]

    logger.info(f"Rebuilding model for {lattice_name} ({boundary}), "
                f"n_vertices={n_vertices}, n_edges={n_edges}")

    # Derive grid size and rebuild lattice
    nx, ny = _derive_grid_size(lattice_name, boundary, n_vertices, n_edges)
    logger.info(f"Derived grid size: {nx}x{ny}")

    gen = get_generator(lattice_name)
    lattice = gen.build(nx, ny, boundary=boundary)

    # Build incidence and operators
    B1 = build_B1(lattice.n_vertices, lattice.edge_list)
    ops = build_eign_operators(B1)

    # Find seed ice state (deterministic from B1 + edge_list)
    sigma_seed = find_seed_ice_state(B1, lattice.coordination, edge_list=lattice.edge_list)
    assert verify_ice_state(B1, sigma_seed, lattice.coordination), "Seed state invalid!"

    # Extract loop basis
    loop_basis = extract_loop_basis(lattice.graph, B1, lattice.edge_list)
    loop_basis.ordering = compute_loop_ordering(
        loop_basis, strategy="spatial_bfs",
        positions=lattice.positions, edge_list=lattice.edge_list,
    )
    logger.info(f"Loop basis: {loop_basis.n_loops} loops (beta_1={config['beta_1']})")
    assert loop_basis.n_loops == config["beta_1"], (
        f"Loop basis mismatch: got {loop_basis.n_loops}, expected {config['beta_1']}"
    )

    # Build model
    model = LoopMPVAN(
        operators=ops,
        loop_basis=loop_basis,
        n_layers=config["n_layers"],
        equ_dim=config["equ_dim"],
        inv_dim=config["inv_dim"],
        head_hidden=config["head_hidden"],
        B1_csc=B1,
    )

    # Load weights
    ckpt_path = os.path.join(run_dir, checkpoint_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(f"Loaded checkpoint: {ckpt_path} ({model.count_parameters()} params)")

    # Invariant features
    inv_features = build_inv_features(lattice.edge_list, lattice.coordination)

    # Load exact states if available
    exact_path = os.path.join(run_dir, "exact_states.npy")
    exact_states = np.load(exact_path) if os.path.exists(exact_path) else None

    return model, sigma_seed, inv_features, B1, lattice.coordination, config, exact_states


def main():
    parser = argparse.ArgumentParser(
        description="Generate samples from a saved LoopMPVAN checkpoint"
    )
    parser.add_argument("--run-dir", required=True,
                        help="Path to saved training run directory")
    parser.add_argument("--checkpoint", default="model_final.pt",
                        help="Checkpoint filename (default: model_final.pt)")
    parser.add_argument("--n-samples", type=int, default=2000,
                        help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    parser.add_argument("--save-samples", action="store_true",
                        help="Save generated samples to .npy file")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Rebuild model and load weights
    model, sigma_seed, inv_features, B1, coordination, config, exact_states = \
        rebuild_model(args.run_dir, args.checkpoint)

    seed_tensor = torch.from_numpy(sigma_seed.astype(np.float32))

    # Generate samples
    logger.info(f"Generating {args.n_samples} samples...")
    t0 = time.perf_counter()
    with torch.no_grad():
        sigmas, log_probs = model.sample(
            seed_tensor, inv_features, n_samples=args.n_samples
        )
    sample_time = time.perf_counter() - t0

    samples_np = sigmas.numpy()
    log_probs_np = log_probs.numpy()

    # Metrics
    violation = batch_ice_rule_violation(samples_np, B1, coordination)
    h_mean, h_std = mean_hamming_distance(samples_np)
    ess = effective_sample_size(log_probs_np)

    kl = float("nan")
    if exact_states is not None:
        kl = kl_from_samples(samples_np, exact_states)

    # State coverage
    unique_set = set()
    for s in samples_np:
        unique_set.add(tuple(s.astype(np.int8)))
    n_unique = len(unique_set)

    # Report
    logger.info(f"\n=== Inference Results ===")
    logger.info(f"Run: {args.run_dir}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Lattice: {config['lattice_name']} ({config['boundary']})")
    logger.info(f"beta_1 = {config['beta_1']}")
    logger.info(f"Samples: {args.n_samples} in {sample_time:.2f}s "
                f"({args.n_samples / sample_time:.0f} samples/s)")
    logger.info(f"Ice rule violations: {violation:.6f}")
    logger.info(f"Mean Hamming distance: {h_mean:.4f} (+/- {h_std:.4f})")
    logger.info(f"ESS: {ess:.1f}")
    if not np.isnan(kl):
        logger.info(f"KL(empirical || uniform): {kl:.6f}")
    if exact_states is not None:
        logger.info(f"State coverage: {n_unique}/{len(exact_states)}")
    else:
        logger.info(f"Unique states: {n_unique}")

    # Optionally save samples
    if args.save_samples:
        ckpt_stem = os.path.splitext(args.checkpoint)[0]
        out_path = os.path.join(args.run_dir, f"samples_{ckpt_stem}_n{args.n_samples}.npy")
        np.save(out_path, samples_np)
        logger.info(f"Samples saved to: {out_path}")

    return 0 if violation == 0.0 else 1


if __name__ == "__main__":
    sys.exit(main())
