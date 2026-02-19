#!/usr/bin/env python3
"""Verify topological invariants for Table VI (topology zoo).

Builds all 12 lattice configurations, computes Betti numbers under both
face-filling strategies (all faces / no faces), and validates:
  - Euler characteristic consistency: χ = n₀ − n₁ + n₂ = β₀ − β₁ + β₂
  - β₁(none) = n₁ − n₀ + 1 for connected graphs
  - Chain complex property: B₁ @ B₂ = 0
  - Open BC → β₁(all) = 0, β₂ = 0
  - Periodic BC → β₁(all) = 2, β₂ = 1
"""
import sys
sys.path.insert(0, "/Users/carlmerrigan/DeckerCode/SpinIceTDL")

import numpy as np
from scipy.linalg import eigh
from src.lattices.registry import get_generator
from src.topology.incidence import build_B1, build_B2, verify_chain_complex
from src.topology.laplacians import build_all_laplacians

# ── Configurations matching Table V (plus periodic counterparts) ────────────
CONFIGS = [
    ("square",   4, 4, "open"),
    ("square",   4, 4, "periodic"),
    ("square",   6, 6, "open"),
    ("square",   6, 6, "periodic"),
    ("kagome",   2, 2, "open"),
    ("kagome",   2, 2, "periodic"),
    ("kagome",   3, 3, "open"),
    ("kagome",   3, 3, "periodic"),
    ("santa_fe", 2, 2, "open"),
    ("santa_fe", 2, 2, "periodic"),
    ("santa_fe", 3, 3, "open"),
    ("santa_fe", 3, 3, "periodic"),
    ("shakti",   2, 2, "open"),
    ("shakti",   2, 2, "periodic"),
    ("shakti",   3, 3, "open"),
    ("shakti",   3, 3, "periodic"),
    ("tetris",   2, 2, "open"),
    ("tetris",   2, 2, "periodic"),
    ("tetris",   3, 3, "open"),
    ("tetris",   3, 3, "periodic"),
]

# Expected values from the plan (for cross-check)
EXPECTED = {
    ("square",   4, 4, "open"):     dict(n0=16, n1=24, n2=9,  chi=1, b0=1, b2=0, b1_all=0,  b1_none=9),
    ("square",   4, 4, "periodic"): dict(n0=16, n1=32, n2=16, chi=0, b0=1, b2=1, b1_all=2,  b1_none=17),
    ("kagome",   2, 2, "open"):     dict(n0=8,  n1=7,  n2=0,  chi=1, b0=1, b2=0, b1_all=0,  b1_none=0),
    ("kagome",   2, 2, "periodic"): dict(n0=8,  n1=12, n2=4,  chi=0, b0=1, b2=1, b1_all=2,  b1_none=5),
    ("kagome",   3, 3, "open"):     dict(n0=18, n1=19, n2=2,  chi=1, b0=1, b2=0, b1_all=0,  b1_none=2),
    ("kagome",   3, 3, "periodic"): dict(n0=18, n1=27, n2=9,  chi=0, b0=1, b2=1, b1_all=2,  b1_none=10),
    ("santa_fe", 2, 2, "open"):     dict(n0=24, n1=30, n2=7,  chi=1, b0=1, b2=0, b1_all=0,  b1_none=7),
    ("santa_fe", 2, 2, "periodic"): dict(n0=24, n1=36, n2=12, chi=0, b0=1, b2=1, b1_all=2,  b1_none=13),
    ("shakti",   2, 2, "open"):     dict(n0=64, n1=81, n2=18, chi=1, b0=1, b2=0, b1_all=0,  b1_none=18),
    ("shakti",   2, 2, "periodic"): dict(n0=64, n1=96, n2=32, chi=0, b0=1, b2=1, b1_all=2,  b1_none=33),
    ("tetris",   2, 2, "open"):     dict(n0=32, n1=38, n2=7,  chi=1, b0=1, b2=0, b1_all=0,  b1_none=7),
    ("tetris",   2, 2, "periodic"): dict(n0=32, n1=48, n2=16, chi=0, b0=1, b2=1, b1_all=2,  b1_none=17),
    ("tetris",   3, 3, "open"):     dict(n0=72, n1=93, n2=22, chi=1, b0=1, b2=0, b1_all=0,  b1_none=22),
    ("tetris",   3, 3, "periodic"): dict(n0=72, n1=108,n2=36, chi=0, b0=1, b2=1, b1_all=2,  b1_none=37),
}

TOL = 1e-10


def compute_betti_from_spectrum(L_evals, tol=TOL):
    """Count zero eigenvalues as the Betti number."""
    return int(np.sum(np.abs(L_evals) < tol))


def run_config(name, nx, ny, bc):
    """Build lattice, compute invariants under both face strategies."""
    gen = get_generator(name)
    lattice = gen.build(nx, ny, boundary=bc)

    n0 = lattice.n_vertices
    n1 = lattice.n_edges
    n2_all = lattice.n_faces  # all faces filled

    # ── Strategy 1: all faces filled ────────────────────────────────────
    B1 = build_B1(n0, lattice.edge_list)
    B2_all = build_B2(n1, lattice.face_list, lattice.edge_list)
    assert verify_chain_complex(B1, B2_all), f"B1@B2 ≠ 0 for {name} {nx}x{ny} {bc} (all faces)"

    laps_all = build_all_laplacians(B1, B2_all)
    evals_L0 = eigh(laps_all['L0'].toarray(), eigvals_only=True)
    evals_L1_all = eigh(laps_all['L1'].toarray(), eigvals_only=True)

    b0 = compute_betti_from_spectrum(evals_L0)
    b1_all = compute_betti_from_spectrum(evals_L1_all)

    # β₂ via rank-nullity: β₂ = n₂ − rank(B₂)
    rank_B2 = np.linalg.matrix_rank(B2_all.toarray(), tol=TOL) if n2_all > 0 else 0
    b2 = n2_all - rank_B2

    chi_simplex = n0 - n1 + n2_all
    chi_betti = b0 - b1_all + b2

    # ── Strategy 2: no faces ────────────────────────────────────────────
    B2_none = build_B2(n1, [], lattice.edge_list)  # empty face list
    laps_none = build_all_laplacians(B1, B2_none)
    evals_L1_none = eigh(laps_none['L1'].toarray(), eigvals_only=True)
    b1_none = compute_betti_from_spectrum(evals_L1_none)

    # Cross-check: β₁(none) = n₁ − n₀ + β₀ = n₁ − n₀ + 1 for connected
    b1_none_formula = n1 - n0 + b0

    return dict(
        name=name, nx=nx, ny=ny, bc=bc,
        n0=n0, n1=n1, n2=n2_all,
        chi_simplex=chi_simplex, chi_betti=chi_betti,
        b0=b0, b1_all=b1_all, b2=b2, b1_none=b1_none,
        b1_none_formula=b1_none_formula,
    )


def main():
    results = []
    all_ok = True

    print("=" * 100)
    print("Table VI Verification: Topological Invariants Across the Lattice Zoo")
    print("=" * 100)

    for name, nx, ny, bc in CONFIGS:
        r = run_config(name, nx, ny, bc)
        results.append(r)

        key = (name, nx, ny, bc)
        exp = EXPECTED.get(key)
        errors = []

        # Check against expected values (if available)
        if exp is not None:
            for field in ['n0', 'n1', 'n2', 'b0', 'b2', 'b1_all', 'b1_none']:
                got = r[field]
                want = exp[field]
                if got != want:
                    errors.append(f"  {field}: got {got}, expected {want}")
            if r['chi_simplex'] != exp['chi']:
                errors.append(f"  χ={r['chi_simplex']}, expected {exp['chi']}")

        # Euler consistency (always checked)
        if r['chi_simplex'] != r['chi_betti']:
            errors.append(f"  Euler mismatch: simplex χ={r['chi_simplex']}, betti χ={r['chi_betti']}")

        # Structural checks based on BC
        if bc == "open":
            if r['b1_all'] != 0:
                errors.append(f"  Open BC should have β₁(all)=0, got {r['b1_all']}")
            if r['b2'] != 0:
                errors.append(f"  Open BC should have β₂=0, got {r['b2']}")
            if r['chi_simplex'] != 1:
                errors.append(f"  Open BC should have χ=1, got {r['chi_simplex']}")
        elif bc == "periodic":
            if r['b1_all'] != 2:
                errors.append(f"  Periodic BC should have β₁(all)=2, got {r['b1_all']}")
            if r['b2'] != 1:
                errors.append(f"  Periodic BC should have β₂=1, got {r['b2']}")
            if r['chi_simplex'] != 0:
                errors.append(f"  Periodic BC should have χ=0, got {r['chi_simplex']}")

        # β₁(none) formula check
        if r['b1_none'] != r['b1_none_formula']:
            errors.append(f"  β₁(none)={r['b1_none']} ≠ formula n₁−n₀+1={r['b1_none_formula']}")

        status = "PASS" if not errors else "FAIL"
        if errors:
            all_ok = False
        label = f"{name} {nx}x{ny} {bc}"
        print(f"\n  [{status}] {label}")
        print(f"    n₀={r['n0']}, n₁={r['n1']}, n₂={r['n2']}, χ={r['chi_simplex']}")
        print(f"    β₀={r['b0']}, β₂={r['b2']}, β₁(all)={r['b1_all']}, β₁(none)={r['b1_none']}")
        if errors:
            for e in errors:
                print(f"    ERROR: {e}")

    # ── Summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("Summary Table")
    print("=" * 100)
    header = f"{'Lattice':<12} {'Size':<6} {'BC':<10} {'n₀':>4} {'n₁':>4} {'n₂':>4} {'χ':>3} {'β₀':>3} {'β₂':>3} {'β₁(all)':>8} {'β₁(none)':>9}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['name']:<12} {r['nx']}x{r['ny']:<4} {r['bc']:<10} {r['n0']:>4} {r['n1']:>4} {r['n2']:>4} {r['chi_simplex']:>3} {r['b0']:>3} {r['b2']:>3} {r['b1_all']:>8} {r['b1_none']:>9}")

    print("\n" + ("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
