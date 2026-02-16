"""Exact enumeration of ice states via directed loop-flip tree search.

For small systems, exhaustively enumerate all ice states reachable by
directed loop flips from a seed state. At each step in the autoregressive
ordering, a loop can only be flipped if it is a directed cycle in the
current sigma (one cycle edge in, one out at every cycle vertex).

The total number of reachable states may be less than 2^beta_1 because
some loops may not be directed in certain configurations.
"""
from __future__ import annotations

from typing import List

import numpy as np
from scipy import sparse

from src.neural.loop_basis import (
    is_directed_cycle,
)


def enumerate_reachable_ice_states(
    sigma_seed: np.ndarray,
    loop_indicators: np.ndarray,
    cycle_edge_lists: List[List[int]],
    ordering: List[int],
    B1: sparse.spmatrix,
    coordination: np.ndarray,
) -> np.ndarray:
    """Enumerate all ice states reachable by directed loop flips.

    Uses DFS through the autoregressive decision tree. At each level
    (loop in ordering), if the loop is a directed cycle in the current
    sigma, both flip and no-flip branches are explored. If not directed,
    only no-flip is explored.

    Parameters
    ----------
    sigma_seed : array (n_edges,) of +1/-1
    loop_indicators : array (beta_1, n_edges) of 0/1
    cycle_edge_lists : list of lists of edge indices per cycle
    ordering : autoregressive loop ordering
    B1 : sparse matrix (n0, n1)
    coordination : array (n0,)

    Returns
    -------
    states : array (n_states, n_edges) of +1/-1
        All distinct reachable ice states.
    """
    from src.topology.ice_sampling import verify_ice_state

    B1_csc = sparse.csc_matrix(B1)
    n_loops = len(ordering)
    n_edges = len(sigma_seed)

    if n_loops > 30:
        raise ValueError(
            f"beta_1 = {n_loops} is too large for enumeration "
            f"(up to 2^{n_loops} states). Maximum recommended: 25."
        )

    states = []
    seen = set()

    def _dfs(step: int, sigma: np.ndarray):
        if step == n_loops:
            key = tuple(sigma.astype(np.int8))
            if key not in seen:
                seen.add(key)
                assert verify_ice_state(B1, sigma, coordination), (
                    f"Generated state violates ice rule!"
                )
                states.append(sigma.copy())
            return

        loop_idx = ordering[step]

        # Branch 1: don't flip
        _dfs(step + 1, sigma)

        # Branch 2: flip (only if directed cycle in current sigma)
        if is_directed_cycle(sigma, cycle_edge_lists[loop_idx], B1_csc):
            sigma_flipped = sigma.copy()
            for e in cycle_edge_lists[loop_idx]:
                sigma_flipped[e] *= -1.0
            _dfs(step + 1, sigma_flipped)

    _dfs(0, sigma_seed.copy())
    return np.array(states)


def enumerate_all_ice_states(
    sigma_seed: np.ndarray,
    loop_indicators: np.ndarray,
    B1: sparse.spmatrix,
    coordination: np.ndarray,
    cycle_edge_lists: List[List[int]] = None,
    ordering: List[int] = None,
) -> np.ndarray:
    """Enumerate all ice states reachable by directed loop flips.

    Wrapper that handles both the directed-cycle-aware enumeration and
    provides a convenient interface.

    Parameters
    ----------
    sigma_seed : array (n_edges,) of +1/-1
    loop_indicators : array (beta_1, n_edges) of 0/1
    B1 : sparse matrix (n0, n1)
    coordination : array (n0,)
    cycle_edge_lists : list of edge index lists per cycle (optional)
    ordering : loop ordering (optional, defaults to natural)

    Returns
    -------
    states : array (n_states, n_edges) of +1/-1
    """
    n_loops = loop_indicators.shape[0]

    if n_loops > 25:
        raise ValueError(
            f"beta_1 = {n_loops} is too large for exact enumeration "
            f"(2^{n_loops} = {2**n_loops} states). Maximum beta_1 = 25."
        )

    if ordering is None:
        ordering = list(range(n_loops))

    if cycle_edge_lists is None:
        # Derive from loop_indicators
        cycle_edge_lists = []
        for i in range(n_loops):
            edges = list(np.where(loop_indicators[i] > 0.5)[0])
            cycle_edge_lists.append(edges)

    return enumerate_reachable_ice_states(
        sigma_seed, loop_indicators, cycle_edge_lists,
        ordering, B1, coordination,
    )
