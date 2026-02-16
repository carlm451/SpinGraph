"""Ice manifold state sampling via directed loop algorithm.

Computes binary spin configurations sigma in {+1, -1}^n_edges that satisfy
the ice rule: at each vertex v, the vertex charge Q_v = (B1 @ sigma)_v equals
z_v mod 2 in absolute value, where z_v is the vertex coordination number.

For even-degree vertices the ice rule requires Q_v = 0 (divergence-free).
For odd-degree vertices the ice rule requires |Q_v| = 1 (minimal violation).

The sampling algorithm:
1. Construct a seed ice state via greedy repair from the all-+1 configuration.
2. Generate new states by random directed loop flips that preserve vertex charges.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy import sparse
from math import comb


def pauling_estimate(coordination: np.ndarray) -> float:
    """Pauling approximation for the number of ice-rule-satisfying states.

    For each vertex with coordination z_v:
    - Even z: fraction = C(z, z/2) / 2^z
    - Odd z: fraction = 2 * C(z, floor(z/2)) / 2^z  (both charge signs)

    Returns 2^n_edges * product_v(fraction_v).  Returns ``float('inf')``
    when the count exceeds float64 range (~10^308).
    """
    log2_count = 0.0
    for z in coordination:
        z = int(z)
        if z % 2 == 0:
            frac = comb(z, z // 2) / (2 ** z)
        else:
            frac = 2 * comb(z, z // 2) / (2 ** z)
        log2_count += np.log2(frac)
    # Add n_edges (one binary choice per edge)
    # But n_edges is sum(z_v)/2 for the graph
    n_edges = int(np.sum(coordination)) // 2
    log2_count += n_edges
    if log2_count > 1023:
        return float("inf")
    return 2.0 ** log2_count


def _eulerian_seed(
    edge_list: List[Tuple[int, int]],
    n_vertices: int,
    n_edges: int,
    coordination: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Construct an ice state via Eulerian circuit orientation.

    For all-even-degree graphs, an Euler circuit directly gives z/2 in and
    z/2 out at every vertex (charge = 0).

    For graphs with odd-degree vertices (boundary vertices under open BCs),
    odd-degree vertices are paired via greedy nearest-neighbor matching and
    connected by virtual edges to make all degrees even.  The Euler circuit
    is then run on the augmented graph, and virtual edges are discarded.
    Odd-degree vertices end up with |charge| = 1 as required by the ice rule.

    Returns sigma array or None on failure.
    """
    import networkx as nx

    G = nx.MultiGraph()
    G.add_nodes_from(range(n_vertices))
    edge_index_map = {}
    for idx, (u, v) in enumerate(edge_list):
        key = G.add_edge(u, v)
        edge_index_map[(u, v, key)] = idx

    # Identify odd-degree vertices
    odd_verts = [v for v in range(n_vertices) if G.degree(v) % 2 == 1]

    virtual_keys = set()
    if odd_verts:
        if len(odd_verts) % 2 != 0:
            return None  # Can't pair odd number of odd-degree vertices

        # Greedy nearest-neighbor pairing using BFS shortest paths
        from collections import deque
        remaining = set(odd_verts)
        while remaining:
            u = remaining.pop()
            # BFS from u to find nearest other odd-degree vertex
            visited = {u}
            queue = deque([u])
            found = None
            while queue and found is None:
                v = queue.popleft()
                for w in G.neighbors(v):
                    if w not in visited:
                        visited.add(w)
                        if w in remaining:
                            found = w
                            break
                        queue.append(w)
            if found is None:
                return None
            remaining.discard(found)
            # Add virtual edge
            key = G.add_edge(u, found)
            virtual_keys.add((u, found, key))
            virtual_keys.add((found, u, key))

    if not nx.is_eulerian(G):
        return None

    sigma = np.ones(n_edges, dtype=np.float64)

    try:
        for u, v, key in nx.eulerian_circuit(G, keys=True):
            # Skip virtual edges
            if (u, v, key) in virtual_keys:
                continue

            idx = edge_index_map.get((u, v, key))
            if idx is None:
                idx = edge_index_map.get((v, u, key))
            if idx is None:
                continue
            eu, ev = edge_list[idx]
            # Circuit traverses u -> v.
            # B1 convention: tail=eu (lower idx, B1=-1), head=ev (higher idx, B1=+1)
            # sigma=+1 means flow from tail to head (eu -> ev)
            # sigma=-1 means flow from head to tail (ev -> eu)
            if u == eu:
                sigma[idx] = +1.0
            else:
                sigma[idx] = -1.0
    except nx.NetworkXError:
        return None

    return sigma


def _build_adjacency(B1_csr: sparse.csr_matrix, n_vertices: int) -> List[List[Tuple[int, int]]]:
    """Build adjacency list: adj[v] = [(neighbor, edge_index), ...]."""
    adj: List[List[Tuple[int, int]]] = [[] for _ in range(n_vertices)]
    n_edges = B1_csr.shape[1]
    B1_csc = sparse.csc_matrix(B1_csr)
    for e in range(n_edges):
        col = B1_csc.getcol(e)
        verts = col.indices
        if len(verts) == 2:
            adj[verts[0]].append((verts[1], e))
            adj[verts[1]].append((verts[0], e))
    return adj


def _path_flip_repair(
    sigma: np.ndarray,
    charge: np.ndarray,
    target_abs_charge: np.ndarray,
    adj: List[List[Tuple[int, int]]],
    edge_endpoints: List[Tuple[np.ndarray, np.ndarray]],
) -> None:
    """Fix residual charge-violation pairs by flipping edges along BFS paths.

    When greedy repair leaves an even number of violations (e.g. one vertex
    at +2 and one at -2), single edge flips cannot fix them because both
    endpoints change simultaneously.  Flipping all edges on a *path* between
    such a pair transfers charge along the path, fixing both endpoints while
    leaving intermediate vertices unchanged (they see one flip in and one out).

    Modifies sigma and charge in-place.
    """
    from collections import deque

    for _ in range(20):  # repeat in case there are multiple pairs
        violations = np.abs(np.abs(charge) - target_abs_charge) > 0.5
        viol_idx = np.where(violations)[0]
        if len(viol_idx) < 2:
            break

        # Find a positive-excess and negative-excess pair
        pos_verts = [v for v in viol_idx if charge[v] > target_abs_charge[v]]
        neg_verts = [v for v in viol_idx if charge[v] < -target_abs_charge[v]]

        if not pos_verts or not neg_verts:
            break

        source = pos_verts[0]
        target_v = neg_verts[0]

        # BFS to find shortest path from source to target_v
        parent = {source: (None, None)}  # vertex -> (parent_vertex, edge_idx)
        queue = deque([source])
        found = False
        while queue and not found:
            v = queue.popleft()
            for w, e in adj[v]:
                if w not in parent:
                    parent[w] = (v, e)
                    if w == target_v:
                        found = True
                        break
                    queue.append(w)

        if not found:
            break

        # Extract path edges
        path_edges = []
        v = target_v
        while parent[v][0] is not None:
            _, e = parent[v]
            path_edges.append(e)
            v = parent[v][0]

        # Flip all edges on the path
        for e in path_edges:
            verts, signs = edge_endpoints[e]
            for j, w in enumerate(verts):
                charge[w] -= 2.0 * signs[j] * sigma[e]
            sigma[e] *= -1.0


def find_seed_ice_state(
    B1: sparse.spmatrix,
    coordination: np.ndarray,
    edge_list: Optional[List[Tuple[int, int]]] = None,
    max_attempts: int = 30,
) -> np.ndarray:
    """Construct a seed ice-rule-satisfying spin configuration.

    Strategy:
    1. If all vertex degrees are even, use an Eulerian circuit orientation
       (guaranteed O(n) construction, always valid).
    2. Otherwise, start from random +/-1, greedy single-edge repair to reduce
       charge violations, then path-flip repair to fix residual pairs that
       greedy cannot resolve (e.g. +2/-2 pairs on a torus).

    Parameters
    ----------
    B1 : sparse matrix, shape (n_vertices, n_edges)
        Vertex-edge incidence matrix.
    coordination : array, shape (n_vertices,)
        Per-vertex coordination number (degree).
    edge_list : list of (int, int), optional
        Edge list (needed for Eulerian construction). If None, falls back to
        greedy repair only.
    max_attempts : int
        Number of random restarts to try.

    Returns
    -------
    sigma : array of +1/-1, shape (n_edges,)
    """
    n_vertices, n_edges = B1.shape
    target_abs_charge = coordination.astype(np.float64) % 2
    B1_csr = sparse.csr_matrix(B1)

    # Fast path: Eulerian circuit orientation (works for even and mixed degree)
    if edge_list is not None:
        sigma = _eulerian_seed(edge_list, n_vertices, n_edges, coordination)
        if sigma is not None:
            charge = np.asarray(B1_csr @ sigma).ravel()
            if np.all(np.abs(np.abs(charge) - target_abs_charge) < 0.5):
                return sigma

    # Precompute edge endpoints from B1
    edge_endpoints = []
    B1_csc = sparse.csc_matrix(B1)
    for e in range(n_edges):
        col = B1_csc.getcol(e)
        verts = col.indices
        signs = col.data
        edge_endpoints.append((verts, signs))

    # Build adjacency for path-flip repair
    adj = _build_adjacency(B1_csr, n_vertices)

    best_sigma = None
    best_n_violations = n_vertices + 1

    for attempt in range(max_attempts):
        # Start from random +/-1
        sigma = np.sign(np.random.randn(n_edges))
        sigma[sigma == 0] = 1.0

        charge = np.asarray(B1_csr @ sigma).ravel()

        # Phase 1: Greedy single-edge repair
        for _pass in range(100):
            violations = np.abs(np.abs(charge) - target_abs_charge) > 0.5
            n_violations = int(np.sum(violations))

            if n_violations == 0:
                return sigma

            improved = False
            violating_verts = np.where(violations)[0]
            np.random.shuffle(violating_verts)

            for v in violating_verts:
                current_err = abs(abs(charge[v]) - target_abs_charge[v])
                if current_err < 0.5:
                    continue

                row = B1_csr.getrow(v)
                edge_indices = row.indices
                edge_signs = row.data

                best_edge = -1
                best_total = -1e10

                perm = np.random.permutation(len(edge_indices))

                for k in perm:
                    e = edge_indices[k]
                    b1_ve = edge_signs[k]
                    delta_v = -2.0 * b1_ve * sigma[e]
                    new_charge_v = charge[v] + delta_v
                    new_err_v = abs(abs(new_charge_v) - target_abs_charge[v])

                    total_improvement = current_err - new_err_v

                    verts, signs = edge_endpoints[e]
                    for j, w in enumerate(verts):
                        if w == v:
                            continue
                        delta_w = -2.0 * signs[j] * sigma[e]
                        new_charge_w = charge[w] + delta_w
                        old_err_w = abs(abs(charge[w]) - target_abs_charge[w])
                        new_err_w = abs(abs(new_charge_w) - target_abs_charge[w])
                        total_improvement += old_err_w - new_err_w

                    if total_improvement > best_total:
                        best_total = total_improvement
                        best_edge = e

                if best_edge >= 0 and best_total >= 0:
                    verts, signs = edge_endpoints[best_edge]
                    for j, w in enumerate(verts):
                        charge[w] -= 2.0 * signs[j] * sigma[best_edge]
                    sigma[best_edge] *= -1.0
                    if best_total > 0:
                        improved = True

            if not improved:
                break

        # Phase 2: Path-flip repair for residual violation pairs
        _path_flip_repair(sigma, charge, target_abs_charge, adj, edge_endpoints)

        # Phase 3: One more greedy pass after path repair
        for _pass in range(20):
            violations = np.abs(np.abs(charge) - target_abs_charge) > 0.5
            n_violations = int(np.sum(violations))
            if n_violations == 0:
                return sigma

            improved = False
            violating_verts = np.where(violations)[0]
            np.random.shuffle(violating_verts)

            for v in violating_verts:
                current_err = abs(abs(charge[v]) - target_abs_charge[v])
                if current_err < 0.5:
                    continue

                row = B1_csr.getrow(v)
                edge_indices = row.indices
                edge_signs = row.data
                best_edge = -1
                best_total = -1e10
                perm = np.random.permutation(len(edge_indices))
                for k in perm:
                    e = edge_indices[k]
                    b1_ve = edge_signs[k]
                    delta_v = -2.0 * b1_ve * sigma[e]
                    new_charge_v = charge[v] + delta_v
                    new_err_v = abs(abs(new_charge_v) - target_abs_charge[v])
                    total_improvement = current_err - new_err_v
                    verts, signs = edge_endpoints[e]
                    for j, w in enumerate(verts):
                        if w == v:
                            continue
                        delta_w = -2.0 * signs[j] * sigma[e]
                        new_charge_w = charge[w] + delta_w
                        old_err_w = abs(abs(charge[w]) - target_abs_charge[w])
                        new_err_w = abs(abs(new_charge_w) - target_abs_charge[w])
                        total_improvement += old_err_w - new_err_w
                    if total_improvement > best_total:
                        best_total = total_improvement
                        best_edge = e
                if best_edge >= 0 and best_total >= 0:
                    verts, signs = edge_endpoints[best_edge]
                    for j, w in enumerate(verts):
                        charge[w] -= 2.0 * signs[j] * sigma[best_edge]
                    sigma[best_edge] *= -1.0
                    if best_total > 0:
                        improved = True
            if not improved:
                break

        # Check final state
        violations = np.abs(np.abs(charge) - target_abs_charge) > 0.5
        n_violations = int(np.sum(violations))
        if n_violations == 0:
            return sigma
        if n_violations < best_n_violations:
            best_n_violations = n_violations
            best_sigma = sigma.copy()

    return best_sigma if best_sigma is not None else sigma


def _vertex_edge_map(B1_csr: sparse.csr_matrix) -> List[np.ndarray]:
    """Return list mapping each vertex to its incident edge indices."""
    n_verts = B1_csr.shape[0]
    result = []
    for v in range(n_verts):
        row = B1_csr.getrow(v)
        result.append(row.indices.copy())
    return result


def random_directed_loop(
    B1_csr: sparse.csr_matrix,
    sigma: np.ndarray,
    vertex_edges: List[np.ndarray],
    max_steps: int = 500,
) -> Optional[List[int]]:
    """Find a random directed loop in the current spin configuration.

    At each vertex, edges are classified as incoming (B1[v,e]*sigma[e] > 0)
    or outgoing (B1[v,e]*sigma[e] < 0). A random walk follows outgoing edges
    until a vertex is revisited, at which point the closed sub-loop is extracted.

    Parameters
    ----------
    B1_csr : sparse CSR matrix, shape (n_vertices, n_edges)
    sigma : array of +1/-1, shape (n_edges,)
    vertex_edges : list of arrays, vertex_edges[v] = edge indices incident to v
    max_steps : int
        Maximum walk length before giving up.

    Returns
    -------
    loop_edges : list of edge indices forming the loop, or None if no loop found.
    """
    n_vertices = B1_csr.shape[0]
    start = np.random.randint(n_vertices)

    # Walk
    visited = {}  # vertex -> step index
    path_verts = []
    path_edges = []
    current = start

    for step in range(max_steps):
        if current in visited:
            # Extract sub-loop from the first visit of current to now
            loop_start = visited[current]
            return path_edges[loop_start:]

        visited[current] = step
        path_verts.append(current)

        # Find outgoing edges at current vertex
        edges = vertex_edges[current]
        row = B1_csr.getrow(current)
        row_data = dict(zip(row.indices, row.data))

        outgoing = []
        next_verts = []
        for e in edges:
            b1_val = row_data.get(e, 0.0)
            if b1_val * sigma[e] < 0:
                # This is an outgoing edge: spin flows away from current
                # Find the other endpoint
                col = B1_csr[:, e].toarray().ravel()
                endpoints = np.where(np.abs(col) > 0.5)[0]
                for w in endpoints:
                    if w != current:
                        outgoing.append(e)
                        next_verts.append(w)
                        break

        if not outgoing:
            # Dead end (shouldn't happen in a valid ice state), restart
            return None

        choice = np.random.randint(len(outgoing))
        path_edges.append(outgoing[choice])
        current = next_verts[choice]

    return None


def sample_ice_states(
    B1: sparse.spmatrix,
    coordination: np.ndarray,
    n_samples: int = 12,
    n_flips_between: int = 20,
    seed: Optional[int] = None,
    edge_list: Optional[List[Tuple[int, int]]] = None,
) -> List[np.ndarray]:
    """Sample ice-rule-satisfying spin configurations via directed loop flips.

    Parameters
    ----------
    B1 : sparse matrix, shape (n_vertices, n_edges)
        Vertex-edge incidence matrix.
    coordination : array, shape (n_vertices,)
        Per-vertex coordination numbers.
    n_samples : int
        Number of states to collect.
    n_flips_between : int
        Number of loop flip attempts between successive samples.
    seed : int, optional
        Random seed for reproducibility.
    edge_list : list of (int, int), optional
        Edge list (enables fast Eulerian seed for even-degree graphs).

    Returns
    -------
    states : list of arrays, each shape (n_edges,) with entries +1/-1.
        The first state is the seed state.
    """
    if seed is not None:
        np.random.seed(seed)

    B1_csr = sparse.csr_matrix(B1)
    vertex_edges = _vertex_edge_map(B1_csr)

    sigma = find_seed_ice_state(B1, coordination, edge_list=edge_list)
    states = [sigma.copy()]

    for _ in range(n_samples - 1):
        flips_done = 0
        attempts = 0
        max_attempts = n_flips_between * 5
        while flips_done < n_flips_between and attempts < max_attempts:
            loop = random_directed_loop(B1_csr, sigma, vertex_edges)
            if loop is not None and len(loop) > 0:
                for e in loop:
                    sigma[e] *= -1.0
                flips_done += 1
            attempts += 1
        states.append(sigma.copy())

    return states


def verify_ice_state(
    B1: sparse.spmatrix,
    sigma: np.ndarray,
    coordination: np.ndarray,
    tol: float = 0.5,
) -> bool:
    """Verify that sigma satisfies the ice rule at every vertex.

    Returns True if |Q_v| = z_v mod 2 for all vertices.
    """
    charge = np.asarray((sparse.csr_matrix(B1) @ sigma)).ravel()
    target = coordination.astype(np.float64) % 2
    return bool(np.all(np.abs(np.abs(charge) - target) < tol))
