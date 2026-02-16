"""Construct Hodge Laplacians from boundary operators."""
from __future__ import annotations

from typing import Dict

from scipy import sparse


def build_all_laplacians(
    B1: sparse.spmatrix,
    B2: sparse.spmatrix,
) -> Dict[str, sparse.csc_matrix]:
    """Build all four Laplacians from boundary operators B1 and B2.

    Returns dict with keys: 'L0', 'L1', 'L1_down', 'L1_up'

    L0 = B1 @ B1^T          (graph Laplacian, n0 × n0)
    L1_down = B1^T @ B1     (lower edge Laplacian, n1 × n1)
    L1_up = B2 @ B2^T       (upper edge Laplacian, n1 × n1)
    L1 = L1_down + L1_up    (full Hodge 1-Laplacian, n1 × n1)
    """
    B1T = B1.T.tocsc()
    B2T = B2.T.tocsc()

    L0 = (B1 @ B1T).tocsc()
    L1_down = (B1T @ B1).tocsc()

    if B2.shape[1] > 0:
        L1_up = (B2 @ B2T).tocsc()
    else:
        L1_up = sparse.csc_matrix(L1_down.shape)

    L1 = (L1_down + L1_up).tocsc()

    return {
        "L0": L0,
        "L1": L1,
        "L1_down": L1_down,
        "L1_up": L1_up,
    }
