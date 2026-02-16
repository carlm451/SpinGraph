"""Construct incidence (boundary) matrices B1 and B2 for simplicial complexes."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse


def build_B1(
    n_vertices: int,
    edge_list: List[Tuple[int, int]],
) -> sparse.csc_matrix:
    """Build the vertex-edge incidence matrix B1.

    B1 is (n_vertices × n_edges) with:
      B1[head, e] = +1
      B1[tail, e] = -1

    Convention: edge (u, v) with u < v has tail=u, head=v.
    So B1[v, e] = +1 and B1[u, e] = -1.
    """
    n_edges = len(edge_list)
    rows = []
    cols = []
    data = []

    for e_idx, (u, v) in enumerate(edge_list):
        # tail = u (lower index), head = v (higher index)
        rows.append(u)
        cols.append(e_idx)
        data.append(-1.0)  # tail

        rows.append(v)
        cols.append(e_idx)
        data.append(+1.0)  # head

    B1 = sparse.csc_matrix(
        (data, (rows, cols)),
        shape=(n_vertices, n_edges),
        dtype=np.float64,
    )
    return B1


def build_B2(
    n_edges: int,
    face_list: List[List[int]],
    edge_list: List[Tuple[int, int]],
) -> sparse.csc_matrix:
    """Build the edge-face incidence matrix B2.

    B2 is (n_edges × n_faces) with:
      B2[e, f] = +1 if edge e appears in face f with consistent orientation
      B2[e, f] = -1 if opposite

    Each face is an ordered vertex cycle (counterclockwise convention).
    """
    n_faces = len(face_list)
    if n_faces == 0:
        return sparse.csc_matrix((n_edges, 0), dtype=np.float64)

    # Build edge lookup: (min(u,v), max(u,v)) -> edge_index
    edge_lookup: Dict[Tuple[int, int], int] = {}
    for e_idx, (u, v) in enumerate(edge_list):
        edge_lookup[(min(u, v), max(u, v))] = e_idx

    rows = []
    cols = []
    data = []

    for f_idx, face_verts in enumerate(face_list):
        n_fv = len(face_verts)
        for k in range(n_fv):
            u = face_verts[k]
            v = face_verts[(k + 1) % n_fv]

            canon = (min(u, v), max(u, v))
            e_idx = edge_lookup[canon]

            # Canonical edge direction is low->high.
            # Face traversal u->v: if u < v, matches canonical => +1; else -1
            sign = +1.0 if u < v else -1.0

            rows.append(e_idx)
            cols.append(f_idx)
            data.append(sign)

    B2 = sparse.csc_matrix(
        (data, (rows, cols)),
        shape=(n_edges, n_faces),
        dtype=np.float64,
    )
    return B2


def verify_chain_complex(
    B1: sparse.spmatrix,
    B2: sparse.spmatrix,
    tol: float = 1e-12,
) -> bool:
    """Verify that B1 @ B2 = 0 (chain complex property).

    Returns True if max |B1 @ B2| < tol.
    """
    if B2.shape[1] == 0:
        return True  # No faces, trivially satisfied
    product = B1 @ B2
    if sparse.issparse(product):
        max_val = abs(product).max()
    else:
        max_val = np.max(np.abs(product))
    return float(max_val) < tol
