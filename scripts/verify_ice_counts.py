"""Independent verification of ice-state counts via backtracking.

Counts ALL ice states on each lattice using constraint-propagation
backtracking (no dependency on the loop-basis or DFS enumeration code).
Compares to the reachable counts from training-experiments.md.

For even-degree lattices: total ice states = reachable states (all in one
charge sector), so this directly validates the DFS enumeration.

For odd-degree lattices: total >= reachable (multiple charge sectors),
and we also count per-sector to get the exact comparison.

Usage:
    source venv/bin/activate
    python -m scripts.verify_ice_counts [--timeout 300]
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse

sys.path.insert(0, ".")
from src.lattices.registry import get_generator
from src.topology.incidence import build_B1


# ── Backtracking ice-state counter ──────────────────────────────────────

def count_ice_states_backtracking(
    B1: np.ndarray,
    coordination: np.ndarray,
    timeout: float = 300.0,
) -> Optional[Dict]:
    """Count all ice states via backtracking with constraint propagation.

    Independent of the loop-basis / DFS enumeration code.

    Returns dict with:
        total: total ice states found
        by_charge: dict mapping charge-pattern tuple -> count
        timed_out: whether the search was cut short
    """
    n0, n1 = B1.shape
    # For each vertex, the target charge
    # even z: Q_v = 0; odd z: |Q_v| = 1 (could be +1 or -1)
    target_zero = np.array([z % 2 == 0 for z in coordination])

    # For each edge, which vertices it touches and with what sign
    # edge_vertices[e] = list of (vertex, sign) pairs
    edge_vertices: List[List[Tuple[int, int]]] = [[] for _ in range(n1)]
    vertex_edges: List[List[int]] = [[] for _ in range(n0)]
    for v in range(n0):
        for e in range(n1):
            if B1[v, e] != 0:
                edge_vertices[e].append((v, int(B1[v, e])))
                vertex_edges[v].append(e)

    # Edge ordering: BFS from vertex 0 for good constraint propagation
    edge_order = _bfs_edge_order(vertex_edges, n0, n1)

    # Track partial charges at each vertex
    partial_charge = np.zeros(n0, dtype=np.int32)
    # Track how many edges remain unassigned at each vertex
    remaining = coordination.copy().astype(np.int32)
    sigma = np.zeros(n1, dtype=np.int8)

    total = 0
    charge_counts: Dict[tuple, int] = {}
    start_time = time.time()
    timed_out = False

    def _can_satisfy(v: int) -> bool:
        """Check if vertex v can still reach a valid charge."""
        q = partial_charge[v]
        r = remaining[v]
        if r == 0:
            if target_zero[v]:
                return q == 0
            else:
                return abs(q) == 1
        # With r remaining edges, charge can change by at most r
        # and parity is fixed: final charge has same parity as q + r
        if target_zero[v]:
            # Need q_final = 0, so |q| <= r and q ≡ r (mod 2)
            return abs(q) <= r and (q % 2 == r % 2)
        else:
            # Need |q_final| = 1, so can reach +1 or -1
            can_reach_plus1 = (abs(q - 1) <= r) and ((q - 1) % 2 == r % 2)
            can_reach_minus1 = (abs(q + 1) <= r) and ((q + 1) % 2 == r % 2)
            return can_reach_plus1 or can_reach_minus1

    def _backtrack(idx: int):
        nonlocal total, timed_out

        if timed_out:
            return

        if time.time() - start_time > timeout:
            timed_out = True
            return

        if idx == n1:
            # All edges assigned — record this ice state
            total += 1
            charge_key = tuple(partial_charge.copy())
            charge_counts[charge_key] = charge_counts.get(charge_key, 0) + 1
            return

        e = edge_order[idx]

        for s in (+1, -1):
            sigma[e] = s
            # Update partial charges
            valid = True
            for v, sign in edge_vertices[e]:
                partial_charge[v] += sign * s
                remaining[v] -= 1
                if not _can_satisfy(v):
                    valid = False
                    break

            if valid:
                # Check all affected vertices (not just those that failed)
                all_ok = True
                for v, sign in edge_vertices[e]:
                    if not _can_satisfy(v):
                        all_ok = False
                        break
                if all_ok:
                    _backtrack(idx + 1)

            # Undo
            for v, sign in edge_vertices[e]:
                partial_charge[v] -= sign * s
                remaining[v] += 1
            # Stop undoing if we broke early (only undo what was done)
            # Actually the loop above undoes everything since we iterate
            # all edge_vertices[e] in both the forward and undo passes.
            # But if we broke early in the forward pass, some weren't updated.
            # Let me fix this...

        sigma[e] = 0

    # Actually, let me rewrite more carefully to handle early break
    def _backtrack_v2(idx: int):
        nonlocal total, timed_out

        if timed_out:
            return

        if time.time() - start_time > timeout:
            timed_out = True
            return

        if idx == n1:
            total += 1
            charge_key = tuple(partial_charge.copy())
            charge_counts[charge_key] = charge_counts.get(charge_key, 0) + 1
            return

        e = edge_order[idx]

        for s in (+1, -1):
            # Apply assignment
            for v, sign in edge_vertices[e]:
                partial_charge[v] += sign * s
                remaining[v] -= 1

            # Check feasibility at all affected vertices
            feasible = all(_can_satisfy(v) for v, _ in edge_vertices[e])

            if feasible:
                _backtrack_v2(idx + 1)

            # Undo assignment
            for v, sign in edge_vertices[e]:
                partial_charge[v] -= sign * s
                remaining[v] += 1

    _backtrack_v2(0)

    elapsed = time.time() - start_time

    # Group by charge pattern for odd-degree analysis
    # For odd-degree vertices, the charge sector is the sign pattern of Q_v
    sector_counts: Dict[tuple, int] = {}
    for charge_key, count in charge_counts.items():
        # Sector = tuple of charges at odd-degree vertices only
        sector = tuple(
            charge_key[v] for v in range(n0) if not target_zero[v]
        )
        sector_counts[sector] = sector_counts.get(sector, 0) + count

    return {
        "total": total,
        "n_sectors": len(sector_counts),
        "sector_counts": sector_counts,
        "largest_sector": max(sector_counts.values()) if sector_counts else total,
        "timed_out": timed_out,
        "elapsed": elapsed,
    }


def _bfs_edge_order(
    vertex_edges: List[List[int]], n0: int, n1: int
) -> List[int]:
    """BFS edge ordering for good constraint propagation."""
    visited_edges = set()
    visited_vertices = set()
    order = []
    queue = [0]  # Start from vertex 0
    visited_vertices.add(0)

    while queue:
        v = queue.pop(0)
        for e in vertex_edges[v]:
            if e not in visited_edges:
                visited_edges.add(e)
                order.append(e)
                # Add the other vertex of this edge to the queue
                for u in range(n0):
                    # This is slow but correct; we only do it once
                    pass
        # Add neighbors
        for e in vertex_edges[v]:
            # Find the other endpoint -- look at edge_vertices
            # Actually we don't have B1 here. Let me use vertex_edges differently.
            pass

    # Simpler: just use the natural order if BFS is tricky without B1
    # Actually, let me build adjacency from vertex_edges
    # Two vertices share an edge if they appear in the same edge's vertex list
    # But we don't have that info here. Let me just order edges by
    # how many already-visited vertices they touch.

    # Greedy: pick the edge that shares the most vertices with already-assigned edges
    remaining_edges = set(range(n1))
    touched_vertices: set = set()
    order = []

    # Start with edges at vertex 0
    if vertex_edges[0]:
        e0 = vertex_edges[0][0]
        order.append(e0)
        remaining_edges.remove(e0)
        touched_vertices.update(v for v in range(n0)
                                if e0 in vertex_edges[v])

    while remaining_edges:
        # Pick edge that touches the most already-touched vertices
        best_e = None
        best_score = -1
        for e in remaining_edges:
            score = sum(1 for v in range(n0)
                        if e in vertex_edges[v] and v in touched_vertices)
            if score > best_score:
                best_score = score
                best_e = e
        order.append(best_e)
        remaining_edges.remove(best_e)
        touched_vertices.update(v for v in range(n0)
                                if best_e in vertex_edges[v])

    return order


# ── Faster BFS edge ordering using B1 ──────────────────────────────────

def bfs_edge_order_from_B1(B1: np.ndarray, n0: int, n1: int) -> List[int]:
    """BFS edge ordering using the incidence matrix directly."""
    # Build edge-to-vertices mapping
    edge_verts = {}
    for e in range(n1):
        verts = list(np.nonzero(B1[:, e])[0])
        edge_verts[e] = verts

    # Build vertex-to-edges mapping
    vert_edges = [[] for _ in range(n0)]
    for e, verts in edge_verts.items():
        for v in verts:
            vert_edges[v].append(e)

    # BFS from vertex 0
    visited_v = {0}
    visited_e = set()
    queue = [0]
    order = []

    while queue:
        v = queue.pop(0)
        for e in vert_edges[v]:
            if e not in visited_e:
                visited_e.add(e)
                order.append(e)
                for u in edge_verts[e]:
                    if u not in visited_v:
                        visited_v.add(u)
                        queue.append(u)

    # Any remaining disconnected edges (shouldn't happen for connected graphs)
    for e in range(n1):
        if e not in visited_e:
            order.append(e)

    return order


# ── Brute-force counter (for cross-checking, n1 <= 24) ─────────────────

def count_ice_states_bruteforce(
    B1: np.ndarray,
    coordination: np.ndarray,
    max_n1: int = 24,
) -> Optional[int]:
    """Count ice states by exhaustive enumeration of all 2^n1 configurations.

    Only feasible for n1 <= ~24.  Returns None if n1 > max_n1.
    """
    n0, n1 = B1.shape
    if n1 > max_n1:
        return None

    target_zero = np.array([z % 2 == 0 for z in coordination])
    count = 0

    for bits in range(2 ** n1):
        # Convert integer to sigma in {+1, -1}^n1
        sigma = np.array(
            [1 - 2 * ((bits >> e) & 1) for e in range(n1)],
            dtype=np.float64,
        )
        # Compute charges
        Q = B1 @ sigma
        # Check ice rule at every vertex
        valid = True
        for v in range(n0):
            if target_zero[v]:
                if Q[v] != 0:
                    valid = False
                    break
            else:
                if abs(Q[v]) != 1:
                    valid = False
                    break
        if valid:
            count += 1

    return count


# ── Main ────────────────────────────────────────────────────────────────

# Expected reachable counts from training-experiments.md
EXPECTED = {
    ("square", 4, 4, "open"): {"beta1": 9, "reachable": 25},
    ("square", 4, 4, "periodic"): {"beta1": 17, "reachable": 40},
    ("kagome", 2, 2, "open"): {"beta1": 6, "reachable": 32},
    ("kagome", 2, 2, "periodic"): {"beta1": 13, "reachable": 172},
    ("santa_fe", 2, 2, "open"): {"beta1": 7, "reachable": 8},
    ("santa_fe", 3, 3, "open"): {"beta1": 19, "reachable": 48},
    ("shakti", 1, 1, "open"): {"beta1": 2, "reachable": 3},
    ("shakti", 2, 2, "open"): {"beta1": 18, "reachable": 112},
    ("tetris", 2, 2, "open"): {"beta1": 7, "reachable": 12},
    ("tetris", 3, 3, "open"): {"beta1": 22, "reachable": 4},
    ("kagome", 3, 3, "open"): {"beta1": 17, "reachable": 4352},
    ("square", 6, 6, "open"): {"beta1": 25, "reachable": 3029},
}


def main():
    parser = argparse.ArgumentParser(
        description="Verify ice-state counts via independent backtracking"
    )
    parser.add_argument(
        "--timeout", type=float, default=300,
        help="Timeout per lattice in seconds (default: 300)"
    )
    parser.add_argument(
        "--lattice", type=str, default=None,
        help="Run only this lattice type (e.g., 'square')"
    )
    args = parser.parse_args()

    cases = [
        # Small validation cases first
        ("square", 2, 2, "periodic"),
        ("square", 3, 3, "periodic"),
        # Training-experiments cases, sorted by n1
        ("shakti", 1, 1, "open"),
        ("kagome", 2, 2, "open"),
        ("square", 4, 4, "open"),
        ("tetris", 2, 2, "open"),
        ("santa_fe", 2, 2, "open"),
        ("kagome", 2, 2, "periodic"),
        ("square", 4, 4, "periodic"),
        ("kagome", 3, 3, "open"),
        ("santa_fe", 3, 3, "open"),
        ("shakti", 2, 2, "open"),
        ("tetris", 3, 3, "open"),
        ("square", 6, 6, "open"),
    ]

    if args.lattice:
        cases = [c for c in cases if c[0] == args.lattice]

    print("=" * 100)
    print("ICE-STATE COUNT VERIFICATION")
    print("Method: constraint-propagation backtracking (independent of loop-basis DFS)")
    print(f"Timeout: {args.timeout}s per lattice")
    print("=" * 100)
    print()

    header = (
        f"{'Lattice':<12} {'Size':>5} {'BC':<8} "
        f"{'n0':>4} {'n1':>4} {'β1':>4} "
        f"{'Even z':>6} "
        f"{'Backtrack':>10} {'BruteForce':>10} {'Agree?':>6} "
        f"{'MaxSector':>10} "
        f"{'DFS reach':>10} {'Match?':>8} "
        f"{'Time':>8}"
    )
    print(header)
    print("-" * len(header))

    for lattice, nx, ny, bc in cases:
        gen = get_generator(lattice)
        result = gen.build(nx, ny, boundary=bc)
        G = result.graph
        n0 = G.number_of_nodes()
        n1 = G.number_of_edges()

        # Use the SAME edge list as the lattice generator (canonical ordering)
        edges = result.edge_list
        B1_sparse = build_B1(n0, edges)
        B1 = B1_sparse.toarray()

        coord = np.array([G.degree(v) for v in sorted(G.nodes())])
        all_even = all(z % 2 == 0 for z in coord)
        beta1 = n1 - n0 + 1

        expected = EXPECTED.get((lattice, nx, ny, bc), {})
        exp_reachable = expected.get("reachable", "?")

        # Use BFS edge ordering from B1
        # Replace the naive ordering in count_ice_states_backtracking
        edge_order = bfs_edge_order_from_B1(B1, n0, n1)

        # Build edge_vertices for the backtracking
        edge_vertices: List[List[Tuple[int, int]]] = [[] for _ in range(n1)]
        for v in range(n0):
            for e in range(n1):
                if B1[v, e] != 0:
                    edge_vertices[e].append((v, int(B1[v, e])))

        target_zero = np.array([z % 2 == 0 for z in coord])
        partial_charge = np.zeros(n0, dtype=np.int32)
        remaining_at_v = coord.copy().astype(np.int32)

        # Inline backtracking for speed (avoid function call overhead)
        total_count = 0
        sector_counts: Dict[tuple, int] = {}
        start_time = time.time()
        timed_out_flag = False

        def can_satisfy(v: int) -> bool:
            q = partial_charge[v]
            r = remaining_at_v[v]
            if r == 0:
                return (q == 0) if target_zero[v] else (abs(q) == 1)
            if target_zero[v]:
                return abs(q) <= r and (q % 2 == r % 2)
            else:
                return (
                    (abs(q - 1) <= r and (q - 1) % 2 == r % 2) or
                    (abs(q + 1) <= r and (q + 1) % 2 == r % 2)
                )

        def backtrack(idx: int):
            nonlocal total_count, timed_out_flag

            if timed_out_flag:
                return

            if idx == n1:
                total_count += 1
                sector = tuple(
                    partial_charge[v]
                    for v in range(n0) if not target_zero[v]
                )
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
                return

            if total_count % 100000 == 0 and total_count > 0:
                if time.time() - start_time > args.timeout:
                    timed_out_flag = True
                    return

            e = edge_order[idx]

            for s in (+1, -1):
                for v, sign in edge_vertices[e]:
                    partial_charge[v] += sign * s
                    remaining_at_v[v] -= 1

                if all(can_satisfy(v) for v, _ in edge_vertices[e]):
                    backtrack(idx + 1)

                for v, sign in edge_vertices[e]:
                    partial_charge[v] -= sign * s
                    remaining_at_v[v] += 1

        backtrack(0)
        elapsed = time.time() - start_time

        n_sectors = len(sector_counts)
        max_sector = max(sector_counts.values()) if sector_counts else 0

        # Brute-force cross-check for small lattices
        bf_count = count_ice_states_bruteforce(B1, coord, max_n1=24)
        bf_str = str(bf_count) if bf_count is not None else "—"
        if bf_count is not None and not timed_out_flag:
            agree = "✓" if bf_count == total_count else "✗"
        else:
            agree = "—"

        # Determine match vs DFS reachable
        if timed_out_flag:
            match_str = "TIMEOUT"
            total_str = f">{total_count}"
        else:
            total_str = str(total_count)
            if all_even:
                # Even degree: total should equal reachable
                if total_count == exp_reachable:
                    match_str = "✓ EXACT"
                else:
                    match_str = f"✗ {total_count}≠{exp_reachable}"
            else:
                # Odd degree: check if reachable matches a sector
                sector_vals = sorted(sector_counts.values(), reverse=True)
                if exp_reachable in sector_vals:
                    match_str = "✓ SECTOR"
                elif isinstance(exp_reachable, int) and total_count >= exp_reachable:
                    match_str = f"~ total≥"
                else:
                    match_str = f"✗ DIFF"

        print(
            f"{lattice:<12} {nx}x{ny:>2} {bc:<8} "
            f"{n0:>4} {n1:>4} {beta1:>4} "
            f"{'yes' if all_even else 'no':>6} "
            f"{total_str:>10} {bf_str:>10} {agree:>6} "
            f"{max_sector:>10} "
            f"{exp_reachable:>10} {match_str:>8} "
            f"{elapsed:>7.1f}s"
        )

        # Print sector breakdown for odd-degree lattices with few sectors
        if not all_even and not timed_out_flag and n_sectors <= 10:
            for sector, count in sorted(
                sector_counts.items(), key=lambda x: -x[1]
            ):
                # Simplify sector display
                sector_clean = tuple(int(x) for x in sector)
                marker = " ← matches DFS" if count == exp_reachable else ""
                print(f"  {'':>40} sector {sector_clean}: {count} states{marker}")

    print()
    print("Legend:")
    print("  ✓ EXACT  = total ice states = DFS reachable (even-degree lattice)")
    print("  ✓ SECTOR = largest charge sector = DFS reachable (odd-degree lattice)")
    print("  ~ total≥ = total ≥ reachable (consistent, sector breakdown shows detail)")
    print("  ✗ DIFF   = mismatch (investigate!)")
    print("  TIMEOUT  = search exceeded time limit")


if __name__ == "__main__":
    main()
