"""Tests for EIGN operators (Phase 1)."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy import sparse

from src.lattices.registry import get_generator
from src.neural.operators import (
    EIGNOperators,
    build_eign_operators,
    scipy_csc_to_torch_sparse,
)
from src.topology.ice_sampling import find_seed_ice_state
from src.topology.incidence import build_B1
from src.topology.laplacians import build_all_laplacians


@pytest.fixture
def square_xs():
    gen = get_generator("square")
    return gen.build(4, 4, boundary="periodic")


@pytest.fixture
def square_xs_open():
    gen = get_generator("square")
    return gen.build(4, 4, boundary="open")


@pytest.fixture
def kagome_xs():
    gen = get_generator("kagome")
    return gen.build(4, 4, boundary="periodic")


class TestScipyToTorch:
    def test_identity(self):
        M = sparse.eye(5, format="csc")
        T = scipy_csc_to_torch_sparse(M)
        assert T.shape == (5, 5)
        assert T.is_sparse
        dense = T.to_dense()
        np.testing.assert_allclose(dense.numpy(), np.eye(5), atol=1e-7)

    def test_signed_matrix(self):
        data = np.array([1.0, -1.0, 1.0, -1.0])
        row = np.array([0, 1, 0, 1])
        col = np.array([0, 0, 1, 1])
        M = sparse.csc_matrix((data, (row, col)), shape=(2, 2))
        T = scipy_csc_to_torch_sparse(M)
        dense = T.to_dense().numpy()
        expected = np.array([[1.0, 1.0], [-1.0, -1.0]])
        np.testing.assert_allclose(dense, expected, atol=1e-7)

    def test_preserves_shape(self):
        M = sparse.random(10, 20, density=0.3, format="csc")
        T = scipy_csc_to_torch_sparse(M)
        assert T.shape == (10, 20)


class TestBuildEIGNOperators:
    def test_shapes(self, square_xs):
        B1 = build_B1(square_xs.n_vertices, square_xs.edge_list)
        ops = build_eign_operators(B1)
        assert ops.n0 == square_xs.n_vertices
        assert ops.n1 == square_xs.n_edges
        assert ops.B1.shape == (ops.n0, ops.n1)
        assert ops.B1_abs.shape == (ops.n0, ops.n1)
        assert ops.L_equ.shape == (ops.n1, ops.n1)
        assert ops.L_inv.shape == (ops.n1, ops.n1)
        assert ops.equ_to_inv.shape == (ops.n1, ops.n1)
        assert ops.inv_to_equ.shape == (ops.n1, ops.n1)

    def test_L_equ_matches_L1_down(self, square_xs):
        """L_equ = B1^T @ B1 must match L1_down from build_all_laplacians."""
        B1 = build_B1(square_xs.n_vertices, square_xs.edge_list)
        # Need B2 for build_all_laplacians (with no faces for simplicity)
        from src.topology.incidence import build_B2
        B2 = build_B2(square_xs.n_edges, [], square_xs.edge_list)
        laps = build_all_laplacians(B1, B2)

        ops = build_eign_operators(B1)
        L_equ_dense = ops.L_equ.to_dense().numpy()
        L1_down_dense = laps["L1_down"].toarray()

        np.testing.assert_allclose(L_equ_dense, L1_down_dense, atol=1e-6)

    def test_L_equ_annihilates_ice_state(self, square_xs):
        """L_equ @ sigma_ice = 0 for valid ice states."""
        B1 = build_B1(square_xs.n_vertices, square_xs.edge_list)
        ops = build_eign_operators(B1)

        sigma = find_seed_ice_state(B1, square_xs.coordination, edge_list=square_xs.edge_list)
        sigma_t = torch.from_numpy(sigma.astype(np.float32))

        result = torch.sparse.mm(ops.L_equ, sigma_t.unsqueeze(-1)).squeeze()
        assert result.norm().item() < 1e-5, f"L_equ @ sigma = {result.norm().item()}"

    def test_L_equ_annihilates_ice_kagome(self, kagome_xs):
        """L_equ @ sigma_ice = 0 for kagome ice states."""
        B1 = build_B1(kagome_xs.n_vertices, kagome_xs.edge_list)
        ops = build_eign_operators(B1)

        sigma = find_seed_ice_state(B1, kagome_xs.coordination, edge_list=kagome_xs.edge_list)
        sigma_t = torch.from_numpy(sigma.astype(np.float32))

        result = torch.sparse.mm(ops.L_equ, sigma_t.unsqueeze(-1)).squeeze()
        assert result.norm().item() < 1e-5

    def test_operators_symmetric(self, square_xs):
        """L_equ and L_inv should be symmetric."""
        B1 = build_B1(square_xs.n_vertices, square_xs.edge_list)
        ops = build_eign_operators(B1)

        L_equ_d = ops.L_equ.to_dense().numpy()
        np.testing.assert_allclose(L_equ_d, L_equ_d.T, atol=1e-6)

        L_inv_d = ops.L_inv.to_dense().numpy()
        np.testing.assert_allclose(L_inv_d, L_inv_d.T, atol=1e-6)

    def test_cross_operators_transpose(self, square_xs):
        """equ_to_inv = inv_to_equ^T."""
        B1 = build_B1(square_xs.n_vertices, square_xs.edge_list)
        ops = build_eign_operators(B1)

        e2i = ops.equ_to_inv.to_dense().numpy()
        i2e = ops.inv_to_equ.to_dense().numpy()
        np.testing.assert_allclose(e2i, i2e.T, atol=1e-6)
