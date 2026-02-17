"""Count charge sectors for the §5.8 counting table.

For each row of the ice-manifold counting table, determines:
- n_odd: number of odd-degree (boundary) vertices
- Number of distinct charge sectors (Coulomb classes)
- Sector sizes where available

Methods:
- Full backtracking: for lattices where count_ice_states_backtracking completes
- Targeted enumeration: for medium-sized lattices where candidate count is manageable
- Random sampling: for large lattices, reports a lower bound

Usage:
    source venv/bin/activate
    python -m scripts.count_charge_sectors [--timeout 300]
"""
from __future__ import annotations

import argparse
import sys
import time
from math import comb

import numpy as np

sys.path.insert(0, ".")
from src.lattices.registry import get_generator
from src.topology.incidence import build_B1
from src.topology.charge_sectors import (
    count_charge_sectors,
    enumerate_feasible_sectors,
    discover_sectors_by_sampling,
)


# Table rows: (lattice, nx, ny, bc)
TABLE_ROWS = [
    ("square", 4, 4, "open"),
    ("square", 4, 4, "periodic"),
    ("kagome", 2, 2, "open"),
    ("kagome", 2, 2, "periodic"),
    ("santa_fe", 2, 2, "open"),
    ("shakti", 2, 2, "open"),
    ("tetris", 2, 2, "open"),
    ("tetris", 3, 3, "open"),
]

# Thresholds for method selection
FULL_BACKTRACK_MAX_N1 = 45   # full enumeration feasible
ENUMERATE_MAX_CANDIDATES = 20_000  # targeted feasibility feasible


def main():
    parser = argparse.ArgumentParser(
        description="Count charge sectors for §5.8 table"
    )
    parser.add_argument(
        "--timeout", type=float, default=300,
        help="Timeout per lattice in seconds (default: 300)"
    )
    parser.add_argument(
        "--n-samples", type=int, default=5000,
        help="Number of random samples for sampling method (default: 5000)"
    )
    args = parser.parse_args()

    print("=" * 110)
    print("CHARGE SECTOR ANALYSIS FOR §5.8 TABLE")
    print("=" * 110)
    print()

    header = (
        f"{'Lattice':<10} {'Size':>5} {'BC':<9} "
        f"{'n1':>4} {'n_odd':>5} {'C(n_odd,n_odd/2)':>16} "
        f"{'|Q_min|':>8} {'Method':<14} "
        f"{'|I|':>10} {'Largest':>10} "
        f"{'Time':>8}"
    )
    print(header)
    print("-" * len(header))

    results = []

    for lattice, nx, ny, bc in TABLE_ROWS:
        gen = get_generator(lattice)
        result = gen.build(nx, ny, boundary=bc)
        G = result.graph
        n0 = G.number_of_nodes()
        n1 = G.number_of_edges()
        edges = result.edge_list
        coord = result.coordination

        B1_sparse = build_B1(n0, edges)
        B1_dense = B1_sparse.toarray()

        n_odd = int(np.sum(coord % 2 != 0))
        max_candidates = comb(n_odd, n_odd // 2) if n_odd > 0 else 1

        row_result = {
            "lattice": lattice, "nx": nx, "ny": ny, "bc": bc,
            "n1": n1, "n_odd": n_odd, "max_candidates": max_candidates,
        }

        # Choose method
        if n_odd == 0:
            # All even degree — exactly 1 sector
            row_result.update({
                "n_sectors": 1, "method": "all-even",
                "total": None, "largest": None,
                "elapsed": 0.0, "sector_counts": {(): None},
                "note": "",
            })
        elif n1 <= FULL_BACKTRACK_MAX_N1:
            # Full backtracking — get exact sector counts
            res = count_charge_sectors(
                B1_dense, coord, timeout=args.timeout
            )
            row_result.update({
                "n_sectors": res["n_sectors"],
                "method": "full-backtrack",
                "total": res["total"],
                "largest": res["largest_sector"],
                "elapsed": res["elapsed"],
                "sector_counts": res["sector_counts"],
                "note": "TIMEOUT" if res["timed_out"] else "",
            })
        elif max_candidates <= ENUMERATE_MAX_CANDIDATES:
            # Targeted enumeration of each candidate charge pattern
            res = enumerate_feasible_sectors(
                B1_dense, coord,
                timeout_per_candidate=10.0,
                total_timeout=args.timeout,
            )
            row_result.update({
                "n_sectors": res["n_feasible"],
                "method": "enumerate",
                "total": None,
                "largest": None,
                "elapsed": res["elapsed"],
                "sector_counts": {k: None for k in res["feasible_sectors"]},
                "note": "TIMEOUT" if res["timed_out"] else "",
            })
        else:
            # Random sampling — lower bound
            res = discover_sectors_by_sampling(
                B1_sparse, coord,
                edge_list=edges,
                n_samples=args.n_samples,
            )
            row_result.update({
                "n_sectors": res["n_discovered"],
                "method": "sampling",
                "total": None,
                "largest": None,
                "elapsed": res["elapsed"],
                "sector_counts": {s: None for s in res["discovered_sectors"]},
                "note": (
                    f"≥{res['n_discovered']} (lower bound, "
                    f"{res.get('n_valid', '?')}/{args.n_samples} valid)"
                ),
            })

        results.append(row_result)

        # Print row
        n_sec_str = str(row_result["n_sectors"])
        if row_result["method"] == "sampling":
            n_sec_str = f"≥{row_result['n_sectors']}"

        total_str = (
            f"{row_result['total']:,}" if row_result["total"] is not None
            else "—"
        )
        largest_str = (
            f"{row_result['largest']:,}" if row_result["largest"] is not None
            else "—"
        )

        print(
            f"{lattice:<10} {nx}x{ny:>2} {bc:<9} "
            f"{n1:>4} {n_odd:>5} {max_candidates:>16,} "
            f"{n_sec_str:>8} {row_result['method']:<14} "
            f"{total_str:>10} {largest_str:>10} "
            f"{row_result['elapsed']:>7.1f}s"
        )
        if row_result["note"]:
            print(f"  NOTE: {row_result['note']}")

    # ── Detailed sector breakdown ──────────────────────────────────────
    print()
    print("=" * 80)
    print("SECTOR BREAKDOWN")
    print("=" * 80)

    for r in results:
        sc = r["sector_counts"]
        n_sec = r["n_sectors"]
        label = f"{r['lattice']} {r['nx']}x{r['ny']} {r['bc']}"

        print(f"\n{label}  (n_odd={r['n_odd']}, {n_sec} sectors)")

        if r["n_odd"] == 0:
            print("  All vertices even-degree → single sector (Q = 0 everywhere)")
            continue

        if r["method"] == "full-backtrack" and not r.get("note"):
            # Show sizes sorted descending
            sizes = sorted(sc.values(), reverse=True)
            print(f"  Sector sizes: {sizes}")
            print(f"  Sum = {sum(sizes)} (should equal |I| = {r['total']})")
            assert sum(sizes) == r["total"], "Sum mismatch!"
        elif r["method"] == "enumerate":
            print(f"  {n_sec} feasible of {r['max_candidates']:,} candidates")
        elif r["method"] == "sampling":
            print(
                f"  Discovered {n_sec} sectors from {args.n_samples} samples "
                f"(max possible: {r['max_candidates']:,})"
            )

    # ── Validation checks ──────────────────────────────────────────────
    print()
    print("=" * 80)
    print("VALIDATION")
    print("=" * 80)

    all_ok = True
    for r in results:
        label = f"{r['lattice']} {r['nx']}x{r['ny']} {r['bc']}"

        # Check n_odd is even
        if r["n_odd"] % 2 != 0:
            print(f"  FAIL: {label} has n_odd={r['n_odd']} (should be even)")
            all_ok = False

        # Even-degree periodic → 1 sector
        if r["n_odd"] == 0:
            if r["n_sectors"] != 1:
                print(f"  FAIL: {label} all-even but {r['n_sectors']} sectors")
                all_ok = False
            else:
                print(f"  OK: {label} — all-even, 1 sector")

        # Full-backtrack: sum of sectors = total
        elif r["method"] == "full-backtrack" and not r.get("note"):
            s = sum(r["sector_counts"].values())
            if s == r["total"]:
                print(f"  OK: {label} — Σsectors = |I| = {r['total']}")
            else:
                print(f"  FAIL: {label} — Σsectors={s} ≠ |I|={r['total']}")
                all_ok = False

        else:
            print(f"  INFO: {label} — {r['method']}, {r['n_sectors']} sectors")

    # Specific validation: kagome 2x2 open should have 6 sectors
    kag = next(
        (r for r in results
         if r["lattice"] == "kagome" and r["nx"] == 2 and r["bc"] == "open"),
        None,
    )
    if kag and kag["method"] == "full-backtrack":
        sizes = sorted(kag["sector_counts"].values(), reverse=True)
        expected = [34, 34, 32, 32, 20, 20]
        if sizes == expected:
            print(f"  OK: Kagome 2x2 open sectors = {sizes} (matches §5.8)")
        else:
            print(f"  WARN: Kagome 2x2 open sectors = {sizes}, expected {expected}")

    if all_ok:
        print("\nAll checks passed.")

    # ── Summary for HTML table ─────────────────────────────────────────
    print()
    print("=" * 80)
    print("HTML TABLE VALUES (n_odd and |Q_min| columns)")
    print("=" * 80)
    print(f"{'Lattice':<10} {'Size':>5} {'BC':<9} {'n_odd':>5} {'|Q_min|':>10}")
    print("-" * 50)
    for r in results:
        n_sec_str = str(r["n_sectors"])
        if r["method"] == "sampling":
            n_sec_str = f"≥{r['n_sectors']}"
        print(
            f"{r['lattice']:<10} {r['nx']}x{r['ny']:>2} {r['bc']:<9} "
            f"{r['n_odd']:>5} {n_sec_str:>10}"
        )


if __name__ == "__main__":
    main()
