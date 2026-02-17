"""Charge-sector analysis for the ice manifold.

Identifies and counts the distinct charge sectors (Coulomb classes) within
the ice manifold of a lattice.  A charge sector is defined by the pattern
of vertex charges Q_v = (B1 @ sigma)_v at odd-degree vertices.  Even-degree
vertices always have Q_v = 0 in the ice manifold, so they don't split sectors.

Three approaches in order of computational cost:

1. Full backtracking (small lattices, n1 <= ~38): counts all ice states
   per sector exactly.
2. Targeted feasibility (medium): for each candidate charge pattern, runs
   backtracking that stops at the first solution found.
3. Random sampling (large lattices): generates many ice states via random
   restarts and collects the distinct charge patterns observed.
"""
from __future__ import annotations

import time
from collections import deque
from math import comb
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse


# ── BFS edge ordering ──────────────────────────────────────────────────

def _bfs_edge_order(B1: np.ndarray, n0: int, n1: int) -> List[int]:
    """BFS edge ordering from B1 for good constraint propagation."""
    edge_verts: Dict[int, List[int]] = {}
    for e in range(n1):
        edge_verts[e] = list(np.nonzero(B1[:, e])[0])

    vert_edges: List[List[int]] = [[] for _ in range(n0)]
    for e, verts in edge_verts.items():
        for v in verts:
            vert_edges[v].append(e)

    visited_v = {0}
    visited_e: set = set()
    queue = deque([0])
    order: List[int] = []

    while queue:
        v = queue.popleft()
        for e in vert_edges[v]:
            if e not in visited_e:
                visited_e.add(e)
                order.append(e)
                for u in edge_verts[e]:
                    if u not in visited_v:
                        visited_v.add(u)
                        queue.append(u)

    for e in range(n1):
        if e not in visited_e:
            order.append(e)

    return order


# ── Full backtracking sector count ─────────────────────────────────────

def count_charge_sectors(
    B1: np.ndarray,
    coordination: np.ndarray,
    timeout: float = 300.0,
) -> Dict:
    """Count charge sectors via full backtracking enumeration.

    Enumerates all ice states and groups them by charge sector.

    Parameters
    ----------
    B1 : dense array (n0, n1)
    coordination : array (n0,)
    timeout : seconds before aborting

    Returns
    -------
    dict with keys: total, n_sectors, sector_counts, largest_sector,
                    timed_out, elapsed, n_odd
    """
    n0, n1 = B1.shape
    target_zero = np.array([z % 2 == 0 for z in coordination])
    odd_vertices = [v for v in range(n0) if not target_zero[v]]
    n_odd = len(odd_vertices)

    edge_vertices: List[List[Tuple[int, int]]] = [[] for _ in range(n1)]
    for v in range(n0):
        for e in range(n1):
            if B1[v, e] != 0:
                edge_vertices[e].append((v, int(B1[v, e])))

    edge_order = _bfs_edge_order(B1, n0, n1)

    partial_charge = np.zeros(n0, dtype=np.int32)
    remaining = coordination.copy().astype(np.int32)

    total = 0
    sector_counts: Dict[tuple, int] = {}
    start_time = time.time()
    timed_out = False

    def can_satisfy(v: int) -> bool:
        q = partial_charge[v]
        r = remaining[v]
        if r == 0:
            return (q == 0) if target_zero[v] else (abs(q) == 1)
        if target_zero[v]:
            return abs(q) <= r and (q % 2 == r % 2)
        else:
            return (
                (abs(q - 1) <= r and (q - 1) % 2 == r % 2)
                or (abs(q + 1) <= r and (q + 1) % 2 == r % 2)
            )

    def backtrack(idx: int):
        nonlocal total, timed_out

        if timed_out:
            return

        if idx == n1:
            total += 1
            sector = tuple(int(partial_charge[v]) for v in odd_vertices)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            return

        if total % 100_000 == 0 and total > 0:
            if time.time() - start_time > timeout:
                timed_out = True
                return

        e = edge_order[idx]
        for s in (+1, -1):
            for v, sign in edge_vertices[e]:
                partial_charge[v] += sign * s
                remaining[v] -= 1

            if all(can_satisfy(v) for v, _ in edge_vertices[e]):
                backtrack(idx + 1)

            for v, sign in edge_vertices[e]:
                partial_charge[v] -= sign * s
                remaining[v] += 1

    backtrack(0)
    elapsed = time.time() - start_time

    return {
        "total": total,
        "n_sectors": len(sector_counts),
        "sector_counts": sector_counts,
        "largest_sector": max(sector_counts.values()) if sector_counts else 0,
        "timed_out": timed_out,
        "elapsed": elapsed,
        "n_odd": n_odd,
    }


# ── Targeted feasibility check ─────────────────────────────────────────

def check_sector_feasibility(
    B1: np.ndarray,
    coordination: np.ndarray,
    target_charge: np.ndarray,
    timeout: float = 10.0,
) -> Optional[np.ndarray]:
    """Check if any ice state exists with the given vertex charge pattern.

    Runs targeted backtracking that stops at the first valid configuration.

    Parameters
    ----------
    B1 : dense array (n0, n1)
    coordination : array (n0,)
    target_charge : array (n0,), the desired Q_v at each vertex.
        Even-degree vertices must have 0; odd-degree vertices must have +1 or -1.
    timeout : seconds before aborting

    Returns
    -------
    sigma : array of +1/-1 if feasible, None if infeasible or timeout.
    """
    n0, n1 = B1.shape

    edge_vertices: List[List[Tuple[int, int]]] = [[] for _ in range(n1)]
    for v in range(n0):
        for e in range(n1):
            if B1[v, e] != 0:
                edge_vertices[e].append((v, int(B1[v, e])))

    edge_order = _bfs_edge_order(B1, n0, n1)

    partial_charge = np.zeros(n0, dtype=np.int32)
    remaining = coordination.copy().astype(np.int32)
    sigma = np.zeros(n1, dtype=np.int8)
    start_time = time.time()
    found = [None]  # mutable container for nested function

    def can_satisfy(v: int) -> bool:
        q = partial_charge[v]
        r = remaining[v]
        t = int(target_charge[v])
        if r == 0:
            return q == t
        diff = t - q
        return abs(diff) <= r and (diff % 2 == r % 2)

    def backtrack(idx: int):
        if found[0] is not None:
            return
        if time.time() - start_time > timeout:
            return

        if idx == n1:
            found[0] = sigma.copy().astype(np.float64)
            found[0][found[0] == 0] = 1.0  # shouldn't happen, safety
            return

        e = edge_order[idx]
        for s in (+1, -1):
            sigma[e] = s
            for v, sign in edge_vertices[e]:
                partial_charge[v] += sign * s
                remaining[v] -= 1

            if all(can_satisfy(v) for v, _ in edge_vertices[e]):
                backtrack(idx + 1)

            for v, sign in edge_vertices[e]:
                partial_charge[v] -= sign * s
                remaining[v] += 1

        sigma[e] = 0

    backtrack(0)
    return found[0]


def enumerate_feasible_sectors(
    B1: np.ndarray,
    coordination: np.ndarray,
    timeout_per_candidate: float = 10.0,
    total_timeout: float = 300.0,
) -> Dict:
    """Find all feasible charge sectors by testing each candidate.

    For each possible pattern of +-1 charges at odd-degree vertices
    (subject to sum = 0), tests feasibility via targeted backtracking.

    Parameters
    ----------
    B1 : dense array (n0, n1)
    coordination : array (n0,)
    timeout_per_candidate : seconds per candidate
    total_timeout : total time budget

    Returns
    -------
    dict with: n_feasible, n_candidates, feasible_sectors, timed_out, elapsed
    """
    n0, n1 = B1.shape
    target_zero = np.array([z % 2 == 0 for z in coordination])
    odd_vertices = [v for v in range(n0) if not target_zero[v]]
    n_odd = len(odd_vertices)

    if n_odd == 0:
        return {
            "n_feasible": 1,
            "n_candidates": 1,
            "feasible_sectors": {(): None},
            "timed_out": False,
            "elapsed": 0.0,
            "n_odd": 0,
        }

    # Number of candidates: C(n_odd, n_odd/2) — those with equal +1 and -1
    n_candidates = comb(n_odd, n_odd // 2)

    feasible_sectors: Dict[tuple, Optional[np.ndarray]] = {}
    start_time = time.time()
    timed_out = False

    # Generate all binary patterns with n_odd//2 ones
    from itertools import combinations

    half = n_odd // 2
    for combo in combinations(range(n_odd), half):
        if time.time() - start_time > total_timeout:
            timed_out = True
            break

        target = np.zeros(n0, dtype=np.int32)
        # Set odd vertices: +1 for those in combo, -1 for others
        plus_set = set(combo)
        for i, v in enumerate(odd_vertices):
            target[v] = +1 if i in plus_set else -1

        sigma = check_sector_feasibility(
            B1, coordination, target,
            timeout=timeout_per_candidate,
        )
        if sigma is not None:
            sector_key = tuple(int(target[v]) for v in odd_vertices)
            feasible_sectors[sector_key] = sigma

    elapsed = time.time() - start_time
    return {
        "n_feasible": len(feasible_sectors),
        "n_candidates": n_candidates,
        "feasible_sectors": feasible_sectors,
        "timed_out": timed_out,
        "elapsed": elapsed,
        "n_odd": n_odd,
    }


# ── Random-sampling sector discovery ───────────────────────────────────

def discover_sectors_by_sampling(
    B1_sparse: sparse.spmatrix,
    coordination: np.ndarray,
    edge_list: Optional[List[Tuple[int, int]]] = None,
    n_samples: int = 5000,
    seed: int = 42,
) -> Dict:
    """Discover charge sectors by generating random ice states.

    Uses random +/-1 starting vectors with greedy repair to generate
    ice states landing in diverse charge sectors.  The Eulerian seed
    is deterministic and always lands in the same sector, so we bypass
    it and rely on random restarts to explore different sectors.

    Parameters
    ----------
    B1_sparse : sparse matrix (n0, n1)
    coordination : array (n0,)
    edge_list : optional (unused, kept for API compatibility)
    n_samples : number of random ice states to generate
    seed : random seed

    Returns
    -------
    dict with: n_discovered, discovered_sectors (set of tuples),
               n_odd, max_possible, elapsed, n_valid
    """
    from src.topology.ice_sampling import find_seed_ice_state

    n0, n1 = B1_sparse.shape
    target_zero = np.array([z % 2 == 0 for z in coordination])
    odd_vertices = [v for v in range(n0) if not target_zero[v]]
    n_odd = len(odd_vertices)

    if n_odd == 0:
        return {
            "n_discovered": 1,
            "discovered_sectors": {()},
            "n_odd": 0,
            "max_possible": 1,
            "elapsed": 0.0,
            "n_valid": 1,
        }

    max_possible = comb(n_odd, n_odd // 2)

    np.random.seed(seed)
    B1_csr = sparse.csr_matrix(B1_sparse)
    target_abs = coordination.astype(np.float64) % 2
    discovered: set = set()
    n_valid = 0
    start_time = time.time()

    for _ in range(n_samples):
        # Use find_seed_ice_state with edge_list=None to skip the
        # deterministic Eulerian path and force random-start greedy repair
        sigma = find_seed_ice_state(
            B1_sparse, coordination,
            edge_list=None,
            max_attempts=3,
        )
        charge = np.asarray(B1_csr @ sigma).ravel()
        if np.all(np.abs(np.abs(charge) - target_abs) < 0.5):
            n_valid += 1
            sector = tuple(int(round(charge[v])) for v in odd_vertices)
            discovered.add(sector)

    elapsed = time.time() - start_time
    return {
        "n_discovered": len(discovered),
        "discovered_sectors": discovered,
        "n_odd": n_odd,
        "max_possible": max_possible,
        "elapsed": elapsed,
        "n_valid": n_valid,
    }
