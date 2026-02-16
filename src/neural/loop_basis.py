"""Loop basis extraction and directed loop-flip state representation.

Extracts beta_1 independent cycles from the lattice graph using
networkx.cycle_basis(), orients them to verify B1 @ c = 0, and provides
the directed loop-flip machinery.

Key physics: flipping all edges on a cycle only preserves the ice rule
if the cycle is a DIRECTED cycle in the current spin configuration --
meaning at every vertex on the cycle, one cycle edge flows in and one
flows out. This is checked at each autoregressive step.

The undirected cycle basis provides the set of candidate loops. At each
step, if the current loop is directed in the current sigma, the model
decides whether to flip it. If not directed, the flip is skipped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from scipy import sparse


@dataclass
class LoopBasis:
    """Extracted loop basis for the ice manifold."""

    loop_indicators: torch.Tensor   # (beta_1, n1) binary 0/1 -- which edges each loop touches
    loop_oriented: np.ndarray       # (beta_1, n1) signed +1/-1/0 -- oriented cycle vectors
    n_loops: int                    # = beta_1
    n_edges: int                    # = n1
    ordering: List[int] = field(default_factory=list)  # autoregressive loop ordering
    cycle_edge_lists: List[List[int]] = field(default_factory=list)  # edge indices per cycle


def _build_edge_lookup(
    edge_list: List[Tuple[int, int]],
) -> Dict[Tuple[int, int], int]:
    """Build bidirectional edge lookup: (u,v)->index and (v,u)->index."""
    lookup = {}
    for idx, (u, v) in enumerate(edge_list):
        lookup[(u, v)] = idx
        lookup[(v, u)] = idx
    return lookup


def orient_cycle(
    cycle_vertices: List[int],
    edge_list: List[Tuple[int, int]],
    edge_lookup: Dict[Tuple[int, int], int],
    B1: sparse.spmatrix,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Orient a vertex cycle to produce a signed edge vector satisfying B1 @ c = 0.

    Parameters
    ----------
    cycle_vertices : list of vertex indices forming the cycle
    edge_list : canonical edge list
    edge_lookup : (u,v)->edge_index mapping
    B1 : vertex-edge incidence matrix

    Returns
    -------
    oriented_vec : array (n_edges,) with +1/-1/0
        Signed cycle vector (B1 @ oriented_vec should be zero).
    indicator_vec : array (n_edges,) with 0/1
        Binary indicator of which edges belong to this cycle.
    cycle_edges : list of edge indices in this cycle.
    """
    n_edges = len(edge_list)
    oriented = np.zeros(n_edges, dtype=np.float64)
    indicator = np.zeros(n_edges, dtype=np.float64)
    cycle_edges = []

    n_cycle = len(cycle_vertices)
    for i in range(n_cycle):
        u = cycle_vertices[i]
        v = cycle_vertices[(i + 1) % n_cycle]

        edge_idx = edge_lookup.get((u, v))
        if edge_idx is None:
            raise ValueError(
                f"Edge ({u}, {v}) not found in edge_list. "
                f"Cycle vertices: {cycle_vertices}"
            )

        indicator[edge_idx] = 1.0
        cycle_edges.append(edge_idx)

        # Canonical edge direction is (min, max) with B1[min,e]=-1, B1[max,e]=+1
        eu, ev = edge_list[edge_idx]
        if u == eu:
            oriented[edge_idx] = +1.0
        else:
            oriented[edge_idx] = -1.0

    return oriented, indicator, cycle_edges


def is_directed_cycle(
    sigma: np.ndarray,
    cycle_edge_indices: List[int],
    B1_csc: sparse.spmatrix,
) -> bool:
    """Check if a cycle is a directed cycle in the given spin configuration.

    A cycle is directed if at every vertex on the cycle, one cycle edge
    flows in and one flows out. This guarantees that flipping all cycle
    edges preserves the ice rule.

    Parameters
    ----------
    sigma : array (n_edges,) of +1/-1
    cycle_edge_indices : list of edge indices in the cycle
    B1_csc : sparse CSC matrix (n0, n1)

    Returns
    -------
    bool
        True if the cycle is directed in sigma.
    """
    cycle_set = set(cycle_edge_indices)

    # Find vertices on the cycle and their incident cycle edges
    vertex_cycle_flows = {}  # vertex -> list of flow values (B1[v,e]*sigma[e])
    for e in cycle_edge_indices:
        col = B1_csc.getcol(e)
        for v, b1_val in zip(col.indices, col.data):
            flow = b1_val * sigma[e]  # positive = inflow at v
            vertex_cycle_flows.setdefault(v, []).append(flow)

    # At each cycle vertex, check that cycle edges have one in and one out
    for v, flows in vertex_cycle_flows.items():
        if len(flows) != 2:
            return False  # Vertex should have exactly 2 cycle edges
        if flows[0] * flows[1] >= 0:
            # Both in or both out -> not a directed cycle
            return False

    return True


def is_directed_cycle_torch(
    sigma: torch.Tensor,
    cycle_edge_indices: List[int],
    B1_csc: sparse.spmatrix,
) -> bool:
    """Torch-compatible version of is_directed_cycle."""
    return is_directed_cycle(
        sigma.detach().numpy() if isinstance(sigma, torch.Tensor) else sigma,
        cycle_edge_indices,
        B1_csc,
    )


def extract_loop_basis(
    G: nx.Graph,
    B1_scipy: sparse.spmatrix,
    edge_list: List[Tuple[int, int]],
) -> LoopBasis:
    """Extract beta_1 independent cycles and build the loop basis.

    Parameters
    ----------
    G : networkx Graph
        The lattice graph.
    B1_scipy : sparse matrix (n0, n1)
        Signed vertex-edge incidence matrix.
    edge_list : list of (int, int)
        Canonical edge list (lower index first).

    Returns
    -------
    LoopBasis
        Contains loop indicators, oriented vectors, cycle edge lists,
        and default ordering.
    """
    edge_lookup = _build_edge_lookup(edge_list)
    n_edges = len(edge_list)

    # nx.cycle_basis returns a list of vertex cycles
    cycles = nx.cycle_basis(G)

    oriented_list = []
    indicator_list = []
    cycle_edge_lists = []

    for cycle_verts in cycles:
        oriented, indicator, cycle_edges = orient_cycle(
            cycle_verts, edge_list, edge_lookup, B1_scipy
        )
        oriented_list.append(oriented)
        indicator_list.append(indicator)
        cycle_edge_lists.append(cycle_edges)

    n_loops = len(cycles)

    if n_loops == 0:
        return LoopBasis(
            loop_indicators=torch.zeros(0, n_edges, dtype=torch.float32),
            loop_oriented=np.zeros((0, n_edges), dtype=np.float64),
            n_loops=0,
            n_edges=n_edges,
            ordering=[],
            cycle_edge_lists=[],
        )

    loop_oriented = np.array(oriented_list)
    loop_indicators_np = np.array(indicator_list)
    loop_indicators = torch.from_numpy(loop_indicators_np.astype(np.float32))

    ordering = list(range(n_loops))

    return LoopBasis(
        loop_indicators=loop_indicators,
        loop_oriented=loop_oriented,
        n_loops=n_loops,
        n_edges=n_edges,
        ordering=ordering,
        cycle_edge_lists=cycle_edge_lists,
    )


def compute_loop_ordering(
    loop_basis: LoopBasis,
    strategy: str = "spatial_bfs",
    positions: Optional[np.ndarray] = None,
    edge_list: Optional[List[Tuple[int, int]]] = None,
) -> List[int]:
    """Compute autoregressive ordering for loops.

    Parameters
    ----------
    loop_basis : LoopBasis
    strategy : one of 'natural', 'spatial_bfs', 'random', 'size_ascending'
    positions : vertex positions (n0, 2), needed for spatial_bfs
    edge_list : edge list, needed for spatial_bfs

    Returns
    -------
    ordering : list of loop indices
    """
    n = loop_basis.n_loops
    if n == 0:
        return []

    if strategy == "natural":
        return list(range(n))

    elif strategy == "random":
        perm = np.random.permutation(n).tolist()
        return perm

    elif strategy == "size_ascending":
        sizes = loop_basis.loop_indicators.sum(dim=1).numpy()
        return np.argsort(sizes).tolist()

    elif strategy == "spatial_bfs":
        indicators = loop_basis.loop_indicators.numpy()
        overlap = indicators @ indicators.T
        np.fill_diagonal(overlap, 0)

        adj = {i: [] for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                if overlap[i, j] > 0:
                    adj[i].append(j)
                    adj[j].append(i)

        start = 0
        if positions is not None and edge_list is not None:
            center = positions.mean(axis=0)
            best_dist = float("inf")
            for i in range(n):
                edge_mask = indicators[i] > 0
                edge_indices = np.where(edge_mask)[0]
                verts = set()
                for ei in edge_indices:
                    verts.add(edge_list[ei][0])
                    verts.add(edge_list[ei][1])
                if verts:
                    loop_center = positions[list(verts)].mean(axis=0)
                    dist = np.linalg.norm(loop_center - center)
                    if dist < best_dist:
                        best_dist = dist
                        start = i

        visited = set()
        order = []
        queue = [start]
        visited.add(start)
        while queue:
            node = queue.pop(0)
            order.append(node)
            for nbr in adj[node]:
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)

        for i in range(n):
            if i not in visited:
                order.append(i)

        return order

    else:
        raise ValueError(f"Unknown ordering strategy: {strategy}")


def flip_single_loop(
    sigma: torch.Tensor,
    loop_indicators: torch.Tensor,
    loop_idx: int,
) -> torch.Tensor:
    """Flip a single loop: negate all edges in the loop.

    Parameters
    ----------
    sigma : (n_edges,) tensor of +1/-1
    loop_indicators : (beta_1, n_edges) binary tensor
    loop_idx : index of the loop to flip

    Returns
    -------
    sigma_new : (n_edges,) tensor of +1/-1
    """
    mask = loop_indicators[loop_idx]  # (n_edges,) binary
    sign_flip = 1.0 - 2.0 * mask     # +1 where mask=0, -1 where mask=1
    return sigma * sign_flip


def apply_loop_flips_sequential(
    sigma_seed: torch.Tensor,
    loop_indicators: torch.Tensor,
    alpha: torch.Tensor,
    ordering: List[int],
    cycle_edge_lists: List[List[int]],
    B1_csc: sparse.spmatrix,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply loop flips sequentially, checking directed-cycle condition at each step.

    At each step, if the loop is a directed cycle in the current sigma,
    the flip decision from alpha is applied. If not directed, the flip
    is skipped (alpha forced to 0).

    Parameters
    ----------
    sigma_seed : (n_edges,) of +1/-1
    loop_indicators : (beta_1, n_edges) binary
    alpha : (beta_1,) binary desired flips
    ordering : loop ordering
    cycle_edge_lists : edge index lists per cycle
    B1_csc : sparse CSC incidence matrix

    Returns
    -------
    sigma : (n_edges,) final state
    effective_alpha : (beta_1,) actual flips applied (some may be forced to 0)
    """
    sigma = sigma_seed.clone()
    effective_alpha = torch.zeros_like(alpha)

    for loop_idx in ordering:
        if alpha[loop_idx] > 0.5:
            # Check if this loop is a directed cycle in current sigma
            if is_directed_cycle_torch(sigma, cycle_edge_lists[loop_idx], B1_csc):
                sigma = flip_single_loop(sigma, loop_indicators, loop_idx)
                effective_alpha[loop_idx] = 1.0

    return sigma, effective_alpha


def apply_loop_flips(
    sigma_seed: torch.Tensor,
    loop_indicators: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Apply loop flips via XOR composition (unchecked).

    WARNING: This does NOT check the directed-cycle condition. Use
    apply_loop_flips_sequential() for ice-rule-safe flips.

    This function is kept for GF(2) operations (alpha recovery, etc.)
    where ice rule preservation is not required.
    """
    flip_count = alpha @ loop_indicators
    flip_parity = flip_count % 2
    sign_flip = 1.0 - 2.0 * flip_parity
    return sigma_seed * sign_flip


def apply_partial_flips(
    sigma_seed: torch.Tensor,
    loop_indicators: torch.Tensor,
    alpha_partial: torch.Tensor,
    ordering: List[int],
) -> torch.Tensor:
    """Apply partial flips via XOR (unchecked, for internal use)."""
    active_loops = ordering[: len(alpha_partial)]
    active_indicators = loop_indicators[active_loops]
    flip_count = alpha_partial @ active_indicators
    flip_parity = flip_count % 2
    sign_flip = 1.0 - 2.0 * flip_parity
    return sigma_seed * sign_flip


def recover_alpha(
    sigma: torch.Tensor,
    sigma_seed: torch.Tensor,
    loop_indicators: torch.Tensor,
) -> torch.Tensor:
    """Recover the alpha vector from a sigma configuration.

    Solves the GF(2) linear system: L @ alpha = diff (mod 2)
    where diff[e] = 1 if sigma[e] != sigma_seed[e], else 0.

    Uses Gaussian elimination over GF(2).

    Parameters
    ----------
    sigma : (n_edges,) or (batch, n_edges) tensor of +1/-1
    sigma_seed : (n_edges,) tensor of +1/-1
    loop_indicators : (beta_1, n_edges) binary tensor

    Returns
    -------
    alpha : (beta_1,) or (batch, beta_1) binary tensor
    """
    batched = sigma.dim() == 2
    if not batched:
        sigma = sigma.unsqueeze(0)

    diff = ((sigma * sigma_seed.unsqueeze(0)) < 0).float()

    L = loop_indicators.numpy().astype(np.int64)
    n_loops, n_edges = L.shape

    results = []
    for b in range(diff.shape[0]):
        d = diff[b].numpy().astype(np.int64)
        alpha = _solve_gf2(L, d)
        results.append(alpha)

    alpha_tensor = torch.from_numpy(np.array(results, dtype=np.float32))
    if not batched:
        alpha_tensor = alpha_tensor.squeeze(0)
    return alpha_tensor


def _solve_gf2(L: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Solve L^T @ alpha = d over GF(2) via Gaussian elimination."""
    n_loops, n_edges = L.shape
    A = np.zeros((n_edges, n_loops + 1), dtype=np.int64)
    A[:, :n_loops] = L.T
    A[:, n_loops] = d

    pivot_row = 0
    pivot_cols = []
    for col in range(n_loops):
        found = -1
        for row in range(pivot_row, n_edges):
            if A[row, col] == 1:
                found = row
                break
        if found == -1:
            continue
        pivot_cols.append(col)
        A[[pivot_row, found]] = A[[found, pivot_row]]
        for row in range(n_edges):
            if row != pivot_row and A[row, col] == 1:
                A[row] = (A[row] + A[pivot_row]) % 2
        pivot_row += 1

    alpha = np.zeros(n_loops, dtype=np.int64)
    for i, col in enumerate(pivot_cols):
        alpha[col] = A[i, n_loops]

    return alpha
