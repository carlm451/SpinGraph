#!/usr/bin/env python3
"""Compute multi-ordering reachable ice state counts.

Runs enumerate_multi_ordering() with K=200 random orderings for
lattices that are missing multi-ordering values in the §5.8 table.

Usage:
    python -m scripts.enumerate_multi_ordering
"""
import logging
import sys

import numpy as np

from src.lattices.registry import get_generator
from src.topology.incidence import build_B1
from src.topology.ice_sampling import find_seed_ice_state, verify_ice_state
from src.neural.loop_basis import extract_loop_basis
from src.neural.enumeration import enumerate_multi_ordering

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


LATTICES = [
    ("shakti", 2, 2, "open"),
    ("tetris", 3, 3, "open"),
]


def main():
    for lattice_name, nx, ny, boundary in LATTICES:
        logger.info(f"\n{'='*60}")
        logger.info(f"{lattice_name} {nx}x{ny} ({boundary} BC)")
        logger.info(f"{'='*60}")

        # Build lattice
        gen = get_generator(lattice_name)
        lattice = gen.build(nx, ny, boundary=boundary)
        logger.info(f"  n0={lattice.n_vertices}, n1={lattice.n_edges}")

        # Build incidence matrix
        B1 = build_B1(lattice.n_vertices, lattice.edge_list)

        # Find seed ice state
        sigma_seed = find_seed_ice_state(
            B1, lattice.coordination, edge_list=lattice.edge_list
        )
        assert verify_ice_state(B1, sigma_seed, lattice.coordination)
        logger.info("  Seed ice state found and verified")

        # Extract loop basis
        loop_basis = extract_loop_basis(lattice.graph, B1, lattice.edge_list)
        beta_1 = loop_basis.n_loops
        logger.info(f"  beta_1 = {beta_1}")

        if beta_1 > 25:
            logger.info(f"  SKIPPED: beta_1={beta_1} too large for enumeration")
            continue

        # Single ordering (natural order)
        from src.neural.enumeration import enumerate_all_ice_states
        states_single = enumerate_all_ice_states(
            sigma_seed,
            loop_basis.loop_indicators.numpy(),
            B1,
            lattice.coordination,
            cycle_edge_lists=loop_basis.cycle_edge_lists,
            ordering=list(range(beta_1)),
        )
        logger.info(f"  Single-ordering DFS: {len(states_single)} states")

        # Multi-ordering enumeration
        states_multi, cumulative = enumerate_multi_ordering(
            sigma_seed,
            loop_basis.loop_indicators.numpy(),
            B1,
            lattice.coordination,
            cycle_edge_lists=loop_basis.cycle_edge_lists,
            n_orderings=200,
            seed=42,
        )
        logger.info(f"  Multi-ordering (K=200): {len(states_multi)} states")
        logger.info(f"  Growth curve (last 5): {cumulative[-5:]}")

    logger.info(f"\n{'='*60}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
