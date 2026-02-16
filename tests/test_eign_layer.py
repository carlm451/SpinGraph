"""Tests for EIGN layer module (Phase 3)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.lattices.registry import get_generator
from src.neural.eign_layer import EIGNLayer
from src.neural.operators import build_eign_operators
from src.topology.ice_sampling import find_seed_ice_state
from src.topology.incidence import build_B1


@pytest.fixture
def square_setup():
    gen = get_generator("square")
    lat = gen.build(4, 4, boundary="periodic")
    B1 = build_B1(lat.n_vertices, lat.edge_list)
    ops = build_eign_operators(B1)
    return lat, B1, ops


class TestEIGNLayerShapes:
    def test_output_shapes(self, square_setup):
        lat, B1, ops = square_setup
        equ_dim, inv_dim = 16, 8
        layer = EIGNLayer(equ_dim, inv_dim, ops)

        X_equ = torch.randn(ops.n1, equ_dim)
        X_inv = torch.randn(ops.n1, inv_dim)

        X_equ_out, X_inv_out = layer(X_equ, X_inv)
        assert X_equ_out.shape == (ops.n1, equ_dim)
        assert X_inv_out.shape == (ops.n1, inv_dim)

    def test_different_dims(self, square_setup):
        lat, B1, ops = square_setup
        for equ_dim, inv_dim in [(8, 4), (32, 16), (64, 32)]:
            layer = EIGNLayer(equ_dim, inv_dim, ops)
            X_equ = torch.randn(ops.n1, equ_dim)
            X_inv = torch.randn(ops.n1, inv_dim)
            X_equ_out, X_inv_out = layer(X_equ, X_inv)
            assert X_equ_out.shape == (ops.n1, equ_dim)
            assert X_inv_out.shape == (ops.n1, inv_dim)


class TestEIGNLayerGradients:
    def test_gradients_flow_all_weights(self, square_setup):
        """All 6 weight matrices should receive gradients."""
        lat, B1, ops = square_setup
        equ_dim, inv_dim = 16, 8
        layer = EIGNLayer(equ_dim, inv_dim, ops)

        X_equ = torch.randn(ops.n1, equ_dim, requires_grad=True)
        X_inv = torch.randn(ops.n1, inv_dim, requires_grad=True)

        X_equ_out, X_inv_out = layer(X_equ, X_inv)
        loss = X_equ_out.sum() + X_inv_out.sum()
        loss.backward()

        for name in ["W1", "W2", "W3", "W4", "W5", "W6"]:
            W = getattr(layer, name)
            assert W.weight.grad is not None, f"{name} has no gradient"
            assert W.weight.grad.norm().item() > 0, f"{name} gradient is zero"

    def test_input_gradients(self, square_setup):
        lat, B1, ops = square_setup
        layer = EIGNLayer(16, 8, ops)

        X_equ = torch.randn(ops.n1, 16, requires_grad=True)
        X_inv = torch.randn(ops.n1, 8, requires_grad=True)

        X_equ_out, X_inv_out = layer(X_equ, X_inv)
        loss = X_equ_out.sum() + X_inv_out.sum()
        loss.backward()

        assert X_equ.grad is not None
        assert X_inv.grad is not None


class TestEIGNLayerIceState:
    def test_L_equ_channel_zero_for_ice(self, square_setup):
        """L_equ annihilates ice states, so the L_equ message should be zero-ish."""
        lat, B1, ops = square_setup

        sigma = find_seed_ice_state(B1, lat.coordination, edge_list=lat.edge_list)
        sigma_t = torch.from_numpy(sigma.astype(np.float32))

        # L_equ @ sigma = 0 for ice states
        result = torch.sparse.mm(ops.L_equ, sigma_t.unsqueeze(-1)).squeeze()
        assert result.norm().item() < 1e-5

        # The L_equ message on a single-feature ice state should be near zero
        # (before the learnable transform -- just the sparse matmul part)
        X_equ = sigma_t.unsqueeze(-1)  # (n1, 1) -- single feature = spin value
        msg = torch.sparse.mm(ops.L_equ, X_equ)  # (n1, 1)
        assert msg.norm().item() < 1e-5
