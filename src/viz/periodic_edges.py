"""Classify edges as interior vs periodic (wrap-around) for visualization.

Uses the minimum image convention to detect edges that wrap around periodic
boundaries.  Periodic edges are split into two half-length "stubs" -- one
extending from each endpoint toward the lattice boundary -- so the torus
topology is visible when the lattice is drawn on a flat plane.
"""
from __future__ import annotations

from typing import List, NamedTuple, Optional, Set, Tuple

import numpy as np


class PeriodicEdge(NamedTuple):
    """A periodic (wrap-around) edge split into two visualization stubs.

    Attributes:
        u, v: Global vertex indices (same ordering as in edge_list).
        edge_index: Position of this edge in the original edge_list.
        stub_u: (start, end) coordinate pairs for the stub at vertex u.
        stub_v: (start, end) coordinate pairs for the stub at vertex v.
    """
    u: int
    v: int
    edge_index: int
    stub_u: Tuple[np.ndarray, np.ndarray]  # (start_xy, end_xy)
    stub_v: Tuple[np.ndarray, np.ndarray]  # (start_xy, end_xy)


def classify_edges(
    positions: np.ndarray,
    edge_list: List[Tuple[int, int]],
    a1: Tuple[float, float],
    a2: Tuple[float, float],
    nx_size: int,
    ny_size: int,
    boundary: str = "periodic",
) -> Tuple[List[Tuple[int, int, int]], List[PeriodicEdge], np.ndarray]:
    """Classify each edge as interior or periodic using minimum image convention.

    Parameters
    ----------
    positions : (n_vertices, 2)
        Real-space vertex coordinates.
    edge_list : list of (u, v)
        Edges as global vertex-index pairs.
    a1, a2 : (float, float)
        Unit-cell lattice vectors.
    nx_size, ny_size : int
        Number of unit-cell repetitions in each direction.
    boundary : str
        ``'periodic'`` or ``'open'``.

    Returns
    -------
    interior_edges : list of (u, v, edge_index)
        Edges that do not wrap around a periodic boundary.
    periodic_edges : list of PeriodicEdge
        Edges that wrap, each with two visualization stubs.
    is_periodic : bool array, shape (n_edges,)
        Mask: True for periodic edges.
    """
    if boundary != "periodic":
        # Open BCs: nothing wraps
        interior = [(u, v, i) for i, (u, v) in enumerate(edge_list)]
        return interior, [], np.zeros(len(edge_list), dtype=bool)

    a1 = np.asarray(a1, dtype=float)
    a2 = np.asarray(a2, dtype=float)

    # Full supercell translation vectors
    T1 = nx_size * a1
    T2 = ny_size * a2

    interior: List[Tuple[int, int, int]] = []
    periodic: List[PeriodicEdge] = []
    is_periodic = np.zeros(len(edge_list), dtype=bool)

    for idx, (u, v) in enumerate(edge_list):
        d_naive = positions[v] - positions[u]

        # Minimum image: check all 9 shifts
        best_d = d_naive.copy()
        best_len2 = np.dot(d_naive, d_naive)

        for n1 in (-1, 0, 1):
            for n2 in (-1, 0, 1):
                if n1 == 0 and n2 == 0:
                    continue
                d_shifted = d_naive - n1 * T1 - n2 * T2
                len2 = np.dot(d_shifted, d_shifted)
                if len2 < best_len2 - 1e-12:
                    best_d = d_shifted
                    best_len2 = len2

        # If the best displacement differs from naive, it's periodic
        if np.linalg.norm(best_d - d_naive) > 1e-8:
            is_periodic[idx] = True

            # Stubs: half the true displacement from each endpoint
            pos_u = positions[u]
            pos_v = positions[v]
            stub_u = (pos_u, pos_u + 0.5 * best_d)
            stub_v = (pos_v, pos_v - 0.5 * best_d)

            periodic.append(PeriodicEdge(
                u=u, v=v, edge_index=idx,
                stub_u=stub_u, stub_v=stub_v,
            ))
        else:
            interior.append((u, v, idx))

    return interior, periodic, is_periodic


def classify_faces(
    face_list: List[List[int]],
    edge_list: List[Tuple[int, int]],
    is_periodic: np.ndarray,
) -> Tuple[List[List[int]], List[List[int]]]:
    """Split faces into interior and periodic based on their edges.

    A face is "periodic" if any of its edges is periodic.  Periodic faces
    should generally be skipped in rendering because their vertex positions
    span the entire lattice (producing giant polygons).

    Parameters
    ----------
    face_list : list of list[int]
        Faces as ordered vertex-index lists.
    edge_list : list of (u, v)
        Full edge list (same ordering as is_periodic).
    is_periodic : bool array, shape (n_edges,)
        Periodic mask from :func:`classify_edges`.

    Returns
    -------
    interior_faces : list of list[int]
        Faces with no periodic edges.
    periodic_faces : list of list[int]
        Faces with at least one periodic edge.
    """
    # Build lookup: edge canonical form -> index
    edge_to_idx = {}
    for i, (u, v) in enumerate(edge_list):
        edge_to_idx[(min(u, v), max(u, v))] = i

    interior_faces: List[List[int]] = []
    periodic_faces: List[List[int]] = []

    for face in face_list:
        n_fv = len(face)
        has_periodic = False
        for k in range(n_fv):
            u, v = face[k], face[(k + 1) % n_fv]
            e_key = (min(u, v), max(u, v))
            e_idx = edge_to_idx.get(e_key)
            if e_idx is not None and is_periodic[e_idx]:
                has_periodic = True
                break
        if has_periodic:
            periodic_faces.append(face)
        else:
            interior_faces.append(face)

    return interior_faces, periodic_faces
