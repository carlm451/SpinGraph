"""Test how multi-ordering enumeration improves ice-state coverage.

Runs DFS enumeration with K random loop orderings on several lattices
and reports how the discovered state set grows. Compares to the verified
total ε(G) from scripts/verify_ice_counts.py.
"""
import sys
import time
import logging

import numpy as np

from src.lattices.registry import get_generator
from src.topology.incidence import build_B1
from src.topology.ice_sampling import find_seed_ice_state
from src.neural.loop_basis import extract_loop_basis
from src.neural.enumeration import enumerate_multi_ordering, enumerate_all_ice_states

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# Lattices to test, with verified ε(G) from verify_ice_counts.py
TEST_CASES = [
    # (name, nx, ny, bc, verified_total, description)
    ("square", 4, 4, "open", 2768, "Square 4×4 open"),
    ("square", 4, 4, "periodic", 2970, "Square 4×4 periodic"),
    ("kagome", 2, 2, "open", 172, "Kagome 2×2 open"),
    ("kagome", 2, 2, "periodic", 600, "Kagome 2×2 periodic"),
    ("santa_fe", 2, 2, "open", 1312, "Santa Fe 2×2 open"),
    ("shakti", 1, 1, "open", 46, "Shakti 1×1 open"),
    ("tetris", 2, 2, "open", 86560, "Tetris 2×2 open"),  # skip if β₁ > 25
]

N_ORDERINGS = 200

print("=" * 90)
print("MULTI-ORDERING ENUMERATION: COVERAGE IMPROVEMENT TEST")
print(f"Testing {N_ORDERINGS} random orderings per lattice")
print("=" * 90)
print()

for name, nx, ny, bc, verified_total, desc in TEST_CASES:
    gen = get_generator(name)
    result = gen.build(nx, ny, boundary=bc)
    B1 = build_B1(result.n_vertices, result.edge_list)
    sigma_seed = find_seed_ice_state(B1, result.coordination, edge_list=result.edge_list)
    loop_basis = extract_loop_basis(result.graph, B1, result.edge_list)
    n_loops = loop_basis.n_loops

    if n_loops > 25:
        print(f"{desc}: β₁={n_loops} > 25, skipping (too large for enumeration)")
        print()
        continue

    # Single fixed ordering (natural)
    t0 = time.time()
    fixed_states = enumerate_all_ice_states(
        sigma_seed,
        loop_basis.loop_indicators.numpy(),
        B1, result.coordination,
        cycle_edge_lists=loop_basis.cycle_edge_lists,
        ordering=list(range(n_loops)),
    )
    t_fixed = time.time() - t0
    n_fixed = len(fixed_states)

    # Multi-ordering enumeration
    t0 = time.time()
    multi_states, cumulative = enumerate_multi_ordering(
        sigma_seed,
        loop_basis.loop_indicators.numpy(),
        B1, result.coordination,
        cycle_edge_lists=loop_basis.cycle_edge_lists,
        n_orderings=N_ORDERINGS,
        seed=42,
    )
    t_multi = time.time() - t0
    n_multi = len(multi_states)

    # Report
    print(f"{'─' * 90}")
    print(f"{desc}: β₁={n_loops}, ε(G)={verified_total}")
    print(f"  Fixed ordering (natural):  {n_fixed:>6} states ({n_fixed/verified_total*100:6.2f}% of ε(G))  [{t_fixed:.1f}s]")
    print(f"  Multi-ordering ({N_ORDERINGS:>3} rand): {n_multi:>6} states ({n_multi/verified_total*100:6.2f}% of ε(G))  [{t_multi:.1f}s]")
    print(f"  Improvement: {n_multi/max(n_fixed,1):.1f}× more states")

    # Print growth curve milestones
    milestones = [1, 5, 10, 20, 50, 100, 200]
    milestone_str = "  Growth: "
    for m in milestones:
        if m <= len(cumulative):
            milestone_str += f"K={m}→{cumulative[m-1]}  "
    print(milestone_str)
    print()

print("=" * 90)
print("SUMMARY")
print("=" * 90)
