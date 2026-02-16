"""Tests for loop basis extraction (Phase 2)."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy import sparse

from src.lattices.registry import get_generator
from src.neural.loop_basis import (
    LoopBasis,
    apply_loop_flips,
    apply_loop_flips_sequential,
    apply_partial_flips,
    compute_loop_ordering,
    extract_loop_basis,
    flip_single_loop,
    is_directed_cycle,
    recover_alpha,
)
from src.topology.ice_sampling import find_seed_ice_state, verify_ice_state
from src.topology.incidence import build_B1


@pytest.fixture
def square_xs_periodic():
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


class TestExtractLoopBasis:
    def test_loop_count_square_periodic(self, square_xs_periodic):
        lat = square_xs_periodic
        B1 = build_B1(lat.n_vertices, lat.edge_list)
        lb = extract_loop_basis(lat.graph, B1, lat.edge_list)

        expected_beta1 = lat.n_edges - lat.n_vertices + 1
        assert lb.n_loops == expected_beta1, (
            f"Expected {expected_beta1} loops, got {lb.n_loops}"
        )

    def test_loop_count_square_open(self, square_xs_open):
        lat = square_xs_open
        B1 = build_B1(lat.n_vertices, lat.edge_list)
        lb = extract_loop_basis(lat.graph, B1, lat.edge_list)

        expected_beta1 = lat.n_edges - lat.n_vertices + 1
        assert lb.n_loops == expected_beta1

    def test_loop_count_kagome(self, kagome_xs):
        lat = kagome_xs
        B1 = build_B1(lat.n_vertices, lat.edge_list)
        lb = extract_loop_basis(lat.graph, B1, lat.edge_list)

        expected_beta1 = lat.n_edges - lat.n_vertices + 1
        assert lb.n_loops == expected_beta1

    def test_shapes(self, square_xs_open):
        lat = square_xs_open
        B1 = build_B1(lat.n_vertices, lat.edge_list)
        lb = extract_loop_basis(lat.graph, B1, lat.edge_list)

        assert lb.loop_indicators.shape == (lb.n_loops, lat.n_edges)
        assert lb.loop_oriented.shape == (lb.n_loops, lat.n_edges)
        assert lb.n_edges == lat.n_edges
        assert len(lb.cycle_edge_lists) == lb.n_loops

    def test_oriented_cycles_divergence_free(self, square_xs_open):
        """B1 @ oriented_cycle = 0 for all cycles."""
        lat = square_xs_open
        B1 = build_B1(lat.n_vertices, lat.edge_list)
        lb = extract_loop_basis(lat.graph, B1, lat.edge_list)

        for i in range(lb.n_loops):
            div = B1 @ lb.loop_oriented[i]
            norm = np.linalg.norm(div)
            assert norm < 1e-10, f"Cycle {i} has non-zero divergence: norm={norm}"

    def test_oriented_cycles_divergence_free_periodic(self, square_xs_periodic):
        lat = square_xs_periodic
        B1 = build_B1(lat.n_vertices, lat.edge_list)
        lb = extract_loop_basis(lat.graph, B1, lat.edge_list)

        for i in range(lb.n_loops):
            div = B1 @ lb.loop_oriented[i]
            norm = np.linalg.norm(div)
            assert norm < 1e-10

    def test_indicators_binary(self, square_xs_open):
        lat = square_xs_open
        B1 = build_B1(lat.n_vertices, lat.edge_list)
        lb = extract_loop_basis(lat.graph, B1, lat.edge_list)

        vals = lb.loop_indicators.unique()
        assert set(vals.numpy().tolist()).issubset({0.0, 1.0})


class TestDirectedCycleCheck:
    def test_detects_directed_cycles(self, square_xs_periodic):
        """At least some cycles should be directed in the seed state."""
        lat = square_xs_periodic
        B1 = build_B1(lat.n_vertices, lat.edge_list)
        B1_csc = sparse.csc_matrix(B1)
        lb = extract_loop_basis(lat.graph, B1, lat.edge_list)

        sigma = find_seed_ice_state(B1, lat.coordination, edge_list=lat.edge_list)

        n_directed = sum(
            is_directed_cycle(sigma, lb.cycle_edge_lists[i], B1_csc)
            for i in range(lb.n_loops)
        )
        assert n_directed > 0, "No directed cycles found in seed state"

    def test_directed_flip_preserves_ice(self, square_xs_open):
        """Flipping a directed cycle always preserves ice rule."""
        lat = square_xs_open
        B1 = build_B1(lat.n_vertices, lat.edge_list)
        B1_csc = sparse.csc_matrix(B1)
        lb = extract_loop_basis(lat.graph, B1, lat.edge_list)

        sigma = find_seed_ice_state(B1, lat.coordination, edge_list=lat.edge_list)
        sigma_t = torch.from_numpy(sigma.astype(np.float32))

        for i in range(lb.n_loops):
            if is_directed_cycle(sigma, lb.cycle_edge_lists[i], B1_csc):
                sigma_new = flip_single_loop(sigma_t, lb.loop_indicators, i)
                assert verify_ice_state(B1, sigma_new.numpy(), lat.coordination), (
                    f"Directed flip of loop {i} violated ice rule"
                )


class TestSequentialFlips:
    def test_sequential_always_valid(self, square_xs_open):
        """Sequential flips with directed checking always produce valid states."""
        lat = square_xs_open
        B1 = build_B1(lat.n_vertices, lat.edge_list)
        B1_csc = sparse.csc_matrix(B1)
        lb = extract_loop_basis(lat.graph, B1, lat.edge_list)
        lb.ordering = list(range(lb.n_loops))

        sigma_seed = find_seed_ice_state(B1, lat.coordination, edge_list=lat.edge_list)
        seed_t = torch.from_numpy(sigma_seed.astype(np.float32))

        # Try several random alpha vectors
        for _ in range(20):
            alpha = torch.bernoulli(torch.full((lb.n_loops,), 0.5))
            sigma, eff_alpha = apply_loop_flips_sequential(
                seed_t, lb.loop_indicators, alpha,
                lb.ordering, lb.cycle_edge_lists, B1_csc,
            )
            assert verify_ice_state(B1, sigma.numpy(), lat.coordination), (
                f"Sequential flip violated ice rule (alpha={alpha})"
            )

    def test_double_flip_identity(self, square_xs_open):
        """No flips applied returns seed state."""
        lat = square_xs_open
        B1 = build_B1(lat.n_vertices, lat.edge_list)
        B1_csc = sparse.csc_matrix(B1)
        lb = extract_loop_basis(lat.graph, B1, lat.edge_list)
        lb.ordering = list(range(lb.n_loops))

        sigma_seed = find_seed_ice_state(B1, lat.coordination, edge_list=lat.edge_list)
        seed_t = torch.from_numpy(sigma_seed.astype(np.float32))

        alpha = torch.zeros(lb.n_loops)
        sigma, eff_alpha = apply_loop_flips_sequential(
            seed_t, lb.loop_indicators, alpha,
            lb.ordering, lb.cycle_edge_lists, B1_csc,
        )
        np.testing.assert_allclose(sigma.numpy(), sigma_seed, atol=1e-7)


class TestRecoverAlpha:
    def test_roundtrip_with_sequential(self, square_xs_open):
        """Recover alpha from states generated by sequential flips."""
        lat = square_xs_open
        B1 = build_B1(lat.n_vertices, lat.edge_list)
        B1_csc = sparse.csc_matrix(B1)
        lb = extract_loop_basis(lat.graph, B1, lat.edge_list)
        lb.ordering = list(range(lb.n_loops))

        sigma_seed = find_seed_ice_state(B1, lat.coordination, edge_list=lat.edge_list)
        seed_t = torch.from_numpy(sigma_seed.astype(np.float32))

        for _ in range(10):
            alpha = torch.bernoulli(torch.full((lb.n_loops,), 0.5))
            sigma, eff_alpha = apply_loop_flips_sequential(
                seed_t, lb.loop_indicators, alpha,
                lb.ordering, lb.cycle_edge_lists, B1_csc,
            )
            # Recover should match effective_alpha (the GF(2) diff)
            recovered = recover_alpha(sigma, seed_t, lb.loop_indicators)
            # The recovered alpha should produce the same sigma via XOR
            sigma_check = apply_loop_flips(seed_t, lb.loop_indicators, recovered)
            np.testing.assert_allclose(
                sigma_check.numpy(), sigma.numpy(), atol=1e-5,
            )


class TestLoopOrdering:
    def test_natural(self, square_xs_open):
        lat = square_xs_open
        B1 = build_B1(lat.n_vertices, lat.edge_list)
        lb = extract_loop_basis(lat.graph, B1, lat.edge_list)

        order = compute_loop_ordering(lb, strategy="natural")
        assert order == list(range(lb.n_loops))

    def test_size_ascending(self, square_xs_open):
        lat = square_xs_open
        B1 = build_B1(lat.n_vertices, lat.edge_list)
        lb = extract_loop_basis(lat.graph, B1, lat.edge_list)

        order = compute_loop_ordering(lb, strategy="size_ascending")
        sizes = lb.loop_indicators.sum(dim=1).numpy()
        for i in range(len(order) - 1):
            assert sizes[order[i]] <= sizes[order[i + 1]]

    def test_spatial_bfs_covers_all(self, square_xs_open):
        lat = square_xs_open
        B1 = build_B1(lat.n_vertices, lat.edge_list)
        lb = extract_loop_basis(lat.graph, B1, lat.edge_list)

        order = compute_loop_ordering(
            lb, strategy="spatial_bfs",
            positions=lat.positions, edge_list=lat.edge_list,
        )
        assert sorted(order) == list(range(lb.n_loops))
