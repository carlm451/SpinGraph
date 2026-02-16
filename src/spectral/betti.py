"""Betti number computation and validation."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy import sparse

from .eigensolve import count_zero_eigenvalues, DEFAULT_ZERO_TOL

# Threshold beyond which we avoid dense rank computation
_SPARSE_RANK_THRESHOLD = 8000


def _sparse_matrix_rank(M: sparse.spmatrix, tol: float = 1e-10) -> int:
    """Compute rank of a sparse matrix without dense conversion.

    Uses sparse SVD to find singular values, computing enough to identify
    the numerical rank. For very sparse boundary matrices, the rank is
    close to min(m, n), so we search from the small end.
    """
    m, n = M.shape
    if m == 0 or n == 0:
        return 0

    min_dim = min(m, n)
    max_k = min_dim - 1  # svds requires k < min(m,n)

    if max_k <= 0:
        # 1x1 or degenerate
        if sparse.issparse(M):
            return int(np.linalg.matrix_rank(M.toarray(), tol=tol))
        return int(np.linalg.matrix_rank(M, tol=tol))

    # Strategy: compute a moderate number of smallest singular values.
    # If all are > tol, rank = min_dim. Otherwise, count the zero ones.
    from scipy.sparse.linalg import svds

    # Compute smallest singular values to find nullity
    k_probe = min(50, max_k)
    try:
        _, s_small, _ = svds(M.astype(np.float64).tocsc(), k=k_probe, which='SM')
        nullity = int(np.sum(s_small < tol))
        # If all k_probe smallest are nonzero, rank = min_dim
        # If some are zero, nullity = count of zero ones (but could be more)
        if nullity == 0:
            return min_dim
        elif nullity < k_probe:
            # Found the boundary: some zero, some not
            return min_dim - nullity
        else:
            # All k_probe were zero, need more
            k_probe = min(nullity * 2, max_k)
            _, s_small, _ = svds(M.astype(np.float64).tocsc(), k=k_probe, which='SM')
            nullity = int(np.sum(s_small < tol))
            return min_dim - nullity
    except Exception:
        # Fallback: use analytical estimate if we can
        raise


def compute_betti_numbers(
    B1: sparse.spmatrix,
    B2: sparse.spmatrix,
    L0_evals: np.ndarray,
    L1_evals: np.ndarray,
    tol: float = DEFAULT_ZERO_TOL,
    boundary: str = "periodic",
) -> Dict:
    """Compute Betti numbers via analytical, rank-nullity, and spectral methods.

    For periodic BCs on a torus, uses analytical formulas (fastest, exact):
      beta_0 = 1, beta_2 = 1 (or 0 if no faces), rank(B1) = n0-1, rank(B2) = n2-1.

    For other cases or smaller matrices, uses rank-nullity with dense or sparse
    SVD depending on matrix size.

    Returns dict with: beta_0, beta_1, beta_2, euler_char,
    euler_check (n0-n1+n2), rank_B1, rank_B2, method_agreement.
    """
    n0 = B1.shape[0]
    n1 = B1.shape[1]
    n2 = B2.shape[1]

    # Method 1: Spectral (count zero eigenvalues from whatever was computed)
    beta_0_spectral = count_zero_eigenvalues(L0_evals, tol)
    beta_1_spectral = count_zero_eigenvalues(L1_evals, tol)

    # Method 2: Rank-based computation
    # For periodic BCs on a torus, we can use exact analytical formulas:
    #   Connected graph -> beta_0 = 1 -> rank(B1) = n0 - 1
    #   Closed orientable surface -> beta_2 = 1 -> rank(B2) = n2 - 1 (when n2 > 0)
    use_analytical = (boundary == "periodic")
    use_sparse = max(n0, n1, n2) > _SPARSE_RANK_THRESHOLD

    if use_analytical:
        # Analytical formulas for periodic torus
        rank_B1 = n0 - 1 if n1 > 0 else 0
        rank_B2 = (n2 - 1) if n2 > 0 else 0
        rank_method = "analytical"
    elif use_sparse:
        # Sparse SVD for large matrices with non-periodic BCs
        rank_B1 = _sparse_matrix_rank(B1, tol=tol) if n1 > 0 else 0
        rank_B2 = _sparse_matrix_rank(B2, tol=tol) if n2 > 0 else 0
        rank_method = "sparse_svd"
    else:
        # Dense rank for small matrices
        rank_B1 = np.linalg.matrix_rank(B1.toarray(), tol=tol) if n1 > 0 else 0
        rank_B2 = np.linalg.matrix_rank(B2.toarray(), tol=tol) if n2 > 0 else 0
        rank_method = "dense"

    beta_0_rank = n0 - rank_B1
    beta_1_rank = n1 - rank_B1 - rank_B2
    beta_2_rank = n2 - rank_B2

    # Euler characteristic
    euler_betti = beta_0_rank - beta_1_rank + beta_2_rank
    euler_simplex = n0 - n1 + n2

    beta_2 = beta_2_rank

    # Check agreement between methods
    # For sparse eigensolve, L1_evals may only contain k smallest eigenvalues,
    # so spectral beta_1 could undercount if beta_1 > k. Only check agreement when
    # we have full spectra (i.e., len(L1_evals) == n1).
    full_spectrum = (len(L0_evals) == n0 and len(L1_evals) == n1)
    if full_spectrum:
        method_agreement = (
            beta_0_spectral == beta_0_rank
            and beta_1_spectral == beta_1_rank
        )
    else:
        # Partial spectrum: spectral count is a lower bound
        method_agreement = (
            beta_0_spectral <= beta_0_rank
            and beta_1_spectral <= beta_1_rank
        )

    return {
        "beta_0": beta_0_rank,
        "beta_1": beta_1_rank,
        "beta_2": beta_2,
        "beta_0_spectral": beta_0_spectral,
        "beta_1_spectral": beta_1_spectral,
        "beta_0_rank": beta_0_rank,
        "beta_1_rank": beta_1_rank,
        "beta_2_rank": beta_2_rank,
        "rank_B1": rank_B1,
        "rank_B2": rank_B2,
        "rank_method": rank_method,
        "euler_betti": euler_betti,
        "euler_simplex": euler_simplex,
        "euler_consistent": euler_betti == euler_simplex,
        "method_agreement": method_agreement,
        "full_spectrum": full_spectrum,
        "n0": n0,
        "n1": n1,
        "n2": n2,
    }


def validate_harmonic_vectors(
    B1: sparse.spmatrix,
    B2: sparse.spmatrix,
    harmonic_basis: np.ndarray,
    tol: float = 1e-8,
) -> List[Dict]:
    """Validate that harmonic vectors are divergence-free and curl-free.

    Harmonic means h in ker(L1) = ker(B1) intersection ker(B2^T).
    L1 h = 0 means (B1^T B1 + B2 B2^T) h = 0.
    Since both terms are PSD:
      B1^T B1 h = 0 => ||B1 h||^2 = 0 => B1 h = 0 (divergence-free)
      B2 B2^T h = 0 => ||B2^T h||^2 = 0 => B2^T h = 0 (curl-free)
    """
    n_modes = harmonic_basis.shape[1]
    results = []

    for i in range(n_modes):
        h = harmonic_basis[:, i]

        # Divergence check: B1 @ h ~ 0
        div = B1 @ h
        div_norm = float(np.linalg.norm(div))

        # Curl check: B2^T @ h ~ 0
        if B2.shape[1] > 0:
            curl = B2.T @ h
            curl_norm = float(np.linalg.norm(curl))
        else:
            curl_norm = 0.0

        results.append({
            "mode_index": i,
            "divergence_norm": div_norm,
            "curl_norm": curl_norm,
            "divergence_free": div_norm < tol,
            "curl_free": curl_norm < tol,
            "is_harmonic": div_norm < tol and curl_norm < tol,
        })

    return results
