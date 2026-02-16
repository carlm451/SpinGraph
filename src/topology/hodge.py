"""Hodge decomposition projectors."""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import sparse
from scipy.linalg import pinvh


def build_hodge_projectors(
    B1: sparse.spmatrix,
    B2: sparse.spmatrix,
) -> Dict[str, np.ndarray]:
    """Build the three Hodge projectors as dense matrices.

    P_grad: projection onto im(B1^T) (gradient subspace)
    P_curl: projection onto im(B2) (curl subspace)
    P_harm: projection onto ker(L1) (harmonic subspace)

    Only use for small systems (n1 < ~5000) since these are dense.
    """
    n1 = B1.shape[1]
    B1_dense = B1.toarray()
    B2_dense = B2.toarray() if B2.shape[1] > 0 else np.zeros((n1, 0))

    # P_grad = B1^T @ pinv(B1 @ B1^T) @ B1
    L0 = B1_dense @ B1_dense.T
    L0_pinv = pinvh(L0)
    P_grad = B1_dense.T @ L0_pinv @ B1_dense

    # P_curl = B2 @ pinv(B2^T @ B2) @ B2^T
    if B2_dense.shape[1] > 0:
        B2TB2 = B2_dense.T @ B2_dense
        B2TB2_pinv = pinvh(B2TB2)
        P_curl = B2_dense @ B2TB2_pinv @ B2_dense.T
    else:
        P_curl = np.zeros((n1, n1))

    P_harm = np.eye(n1) - P_grad - P_curl

    return {
        "P_grad": P_grad,
        "P_curl": P_curl,
        "P_harm": P_harm,
    }
