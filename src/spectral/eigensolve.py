"""Eigendecomposition dispatcher for dense and sparse Laplacians."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy import sparse
from scipy.linalg import eigh as dense_eigh
from scipy.sparse.linalg import eigsh


DEFAULT_ZERO_TOL = 1e-10
DENSE_THRESHOLD = 15000
# Above this size, avoid shift-invert (LU factorization memory scales poorly).
# For 2D lattice Laplacians the fill-in is O(n*sqrt(n)), so shift-invert is
# safe well beyond 20k.  See docs/eigensolver-research.md for analysis.
SHIFT_INVERT_THRESHOLD = 50000


def eigendecompose(
    L: sparse.spmatrix,
    method: str = "auto",
    k: Optional[int] = None,
) -> Dict:
    """Compute eigenvalues and eigenvectors of a symmetric PSD Laplacian.

    Args:
        L: Symmetric positive semi-definite sparse matrix.
        method: 'dense', 'sparse', or 'auto'.
        k: Number of eigenvalues for sparse solver (default: min(100, n-1)).

    Returns:
        dict with 'eigenvalues' (sorted ascending) and 'eigenvectors' (columns).
    """
    n = L.shape[0]

    # Handle trivial zero matrix (e.g., L1_up when no faces)
    if sparse.issparse(L) and L.nnz == 0:
        eigenvalues = np.zeros(n)
        # Don't create full n*n identity for large matrices
        if n <= DENSE_THRESHOLD:
            eigenvectors = np.eye(n)
        else:
            eigenvectors = np.eye(n, min(n, 100))  # only first 100 columns
        return {"eigenvalues": eigenvalues, "eigenvectors": eigenvectors}

    if method == "auto":
        method = "dense" if n <= DENSE_THRESHOLD else "sparse"

    if method == "dense":
        L_dense = L.toarray() if sparse.issparse(L) else L
        # Symmetrize to avoid numerical issues
        L_dense = 0.5 * (L_dense + L_dense.T)
        # driver='evd' uses LAPACK DSYEVD (divide-and-conquer), which is
        # 2-4x faster than the default 'evr' for full eigendecomposition.
        # Uses ~50% more workspace memory but well within limits at n<=15k.
        eigenvalues, eigenvectors = dense_eigh(L_dense, driver="evd")
        return {"eigenvalues": eigenvalues, "eigenvectors": eigenvectors}

    # Sparse method
    if k is None:
        k = min(100, n - 2) if n > 2 else 1

    if k >= n - 1:
        # Fall back to dense if k is close to n
        return eigendecompose(L, method="dense")

    # For large matrices, skip shift-invert to avoid OOM from LU factorization
    if n <= SHIFT_INVERT_THRESHOLD:
        # Try shift-invert first (best for finding near-zero eigenvalues)
        try:
            eigenvalues, eigenvectors = eigsh(
                L, k=k, sigma=0.0, which="LM",
                mode="normal", tol=1e-12,
            )
            order = np.argsort(eigenvalues)
            return {
                "eigenvalues": eigenvalues[order],
                "eigenvectors": eigenvectors[:, order],
            }
        except Exception:
            pass  # fall through to SM method

    # Direct smallest-magnitude method (no LU factorization needed)
    try:
        eigenvalues, eigenvectors = eigsh(
            L, k=k, which="SM", tol=1e-10,
            maxiter=n * 10,
        )
    except Exception:
        # Last resort: dense (only if matrix isn't huge)
        if n <= DENSE_THRESHOLD * 2:
            return eigendecompose(L, method="dense")
        raise RuntimeError(
            f"Sparse eigensolve failed for {n}x{n} matrix and "
            f"matrix too large for dense fallback"
        )

    # Sort by eigenvalue
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    return {"eigenvalues": eigenvalues, "eigenvectors": eigenvectors}


def count_zero_eigenvalues(
    eigenvalues: np.ndarray,
    tol: float = DEFAULT_ZERO_TOL,
) -> int:
    """Count eigenvalues that are effectively zero."""
    return int(np.sum(np.abs(eigenvalues) < tol))


def extract_harmonic_basis(
    eigenvectors: np.ndarray,
    eigenvalues: np.ndarray,
    tol: float = DEFAULT_ZERO_TOL,
) -> np.ndarray:
    """Extract eigenvectors corresponding to zero eigenvalues.

    Returns:
        Array of shape (n, beta_1) where columns are harmonic basis vectors.
        Returns shape (n, 0) if no zero eigenvalues found.
    """
    mask = np.abs(eigenvalues) < tol
    if not np.any(mask):
        return eigenvectors[:, :0]  # empty but correct shape
    return eigenvectors[:, mask]
