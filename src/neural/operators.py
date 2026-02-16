"""EIGN operators: scipy sparse -> PyTorch sparse, four operator products.

The EIGN (Edge-level Information-Gated Network) framework uses four operator
products derived from B1 (signed incidence) and |B1| (unsigned incidence):

  L_equ     = B1^T @ B1       (equivariant channel, = L1_down / Hamiltonian)
  L_inv     = |B1|^T @ |B1|   (invariant channel)
  equ_to_inv = |B1|^T @ B1    (cross-channel: equivariant -> invariant)
  inv_to_equ = B1^T @ |B1|    (cross-channel: invariant -> equivariant)

All products are precomputed once in scipy (mature sparse matmul) and then
converted to PyTorch sparse COO tensors for use in the training loop.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy import sparse


@dataclass
class EIGNOperators:
    """Precomputed EIGN operator matrices in PyTorch sparse format."""

    B1: torch.Tensor           # sparse (n0, n1), signed incidence
    B1_abs: torch.Tensor       # sparse (n0, n1), |B1|
    L_equ: torch.Tensor        # sparse (n1, n1), B1^T @ B1
    L_inv: torch.Tensor        # sparse (n1, n1), |B1|^T @ |B1|
    equ_to_inv: torch.Tensor   # sparse (n1, n1), |B1|^T @ B1
    inv_to_equ: torch.Tensor   # sparse (n1, n1), B1^T @ |B1|
    n0: int
    n1: int


def scipy_csc_to_torch_sparse(
    M: sparse.spmatrix,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Convert a scipy sparse matrix to a PyTorch sparse COO tensor.

    Parameters
    ----------
    M : scipy sparse matrix
        Any scipy sparse format (CSC, CSR, COO).
    device : torch device or string

    Returns
    -------
    torch.Tensor
        Sparse COO tensor with float32 dtype.
    """
    coo = sparse.coo_matrix(M)
    indices = np.vstack([coo.row, coo.col])
    indices = torch.from_numpy(indices.astype(np.int64))
    values = torch.from_numpy(coo.data.astype(np.float32))
    shape = torch.Size(coo.shape)
    t = torch.sparse_coo_tensor(indices, values, shape, device=device)
    return t.coalesce()


def build_eign_operators(
    B1_scipy: sparse.spmatrix,
    device: torch.device | str = "cpu",
) -> EIGNOperators:
    """Build all four EIGN operator products from B1.

    Computes products in scipy (efficient sparse matmul), then converts
    each result to a PyTorch sparse tensor.

    Parameters
    ----------
    B1_scipy : scipy sparse matrix, shape (n0, n1)
        Signed vertex-edge incidence matrix from build_B1().
    device : torch device or string

    Returns
    -------
    EIGNOperators
        Dataclass with all six matrices and dimension info.
    """
    B1_csc = sparse.csc_matrix(B1_scipy, dtype=np.float64)
    B1_abs_csc = sparse.csc_matrix(np.abs(B1_csc), dtype=np.float64)

    # Four products (computed in scipy)
    L_equ_sp = B1_csc.T @ B1_csc           # B1^T @ B1
    L_inv_sp = B1_abs_csc.T @ B1_abs_csc   # |B1|^T @ |B1|
    equ_to_inv_sp = B1_abs_csc.T @ B1_csc  # |B1|^T @ B1
    inv_to_equ_sp = B1_csc.T @ B1_abs_csc  # B1^T @ |B1|

    n0, n1 = B1_csc.shape

    return EIGNOperators(
        B1=scipy_csc_to_torch_sparse(B1_csc, device),
        B1_abs=scipy_csc_to_torch_sparse(B1_abs_csc, device),
        L_equ=scipy_csc_to_torch_sparse(L_equ_sp, device),
        L_inv=scipy_csc_to_torch_sparse(L_inv_sp, device),
        equ_to_inv=scipy_csc_to_torch_sparse(equ_to_inv_sp, device),
        inv_to_equ=scipy_csc_to_torch_sparse(inv_to_equ_sp, device),
        n0=n0,
        n1=n1,
    )
