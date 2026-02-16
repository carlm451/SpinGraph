"""Tests for training and exact enumeration (Phase 5)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.lattices.registry import get_generator
from src.neural.enumeration import enumerate_all_ice_states
from src.neural.loop_basis import extract_loop_basis
from src.neural.loop_mpvan import LoopMPVAN
from src.neural.metrics import (
    effective_sample_size,
    energy,
    ice_rule_violation,
    kl_divergence_exact,
    kl_from_samples,
    mean_hamming_distance,
)
from src.neural.operators import build_eign_operators
from src.neural.training import TrainingConfig, build_inv_features, train
from src.topology.ice_sampling import find_seed_ice_state, verify_ice_state
from src.topology.incidence import build_B1


@pytest.fixture
def square_open_setup():
    gen = get_generator("square")
    lat = gen.build(4, 4, boundary="open")
    B1 = build_B1(lat.n_vertices, lat.edge_list)
    sigma_seed = find_seed_ice_state(B1, lat.coordination, edge_list=lat.edge_list)
    return lat, B1, sigma_seed


@pytest.fixture
def square_periodic_setup():
    gen = get_generator("square")
    lat = gen.build(4, 4, boundary="periodic")
    B1 = build_B1(lat.n_vertices, lat.edge_list)
    sigma_seed = find_seed_ice_state(B1, lat.coordination, edge_list=lat.edge_list)
    return lat, B1, sigma_seed


class TestEnumeration:
    def test_square_xs_open_reachable(self, square_open_setup):
        """Square XS open: directed enumeration finds valid ice states."""
        lat, B1, sigma_seed = square_open_setup
        lb = extract_loop_basis(lat.graph, B1, lat.edge_list)

        assert lb.n_loops == 9

        states = enumerate_all_ice_states(
            sigma_seed, lb.loop_indicators.numpy(),
            B1, lat.coordination,
            cycle_edge_lists=lb.cycle_edge_lists,
            ordering=lb.ordering,
        )
        # Should find at least 2 states (seed + at least one flip)
        assert len(states) >= 2
        # Should find at most 2^9 = 512 states
        assert len(states) <= 512

    def test_all_enumerated_states_valid(self, square_open_setup):
        lat, B1, sigma_seed = square_open_setup
        lb = extract_loop_basis(lat.graph, B1, lat.edge_list)

        states = enumerate_all_ice_states(
            sigma_seed, lb.loop_indicators.numpy(),
            B1, lat.coordination,
            cycle_edge_lists=lb.cycle_edge_lists,
        )
        for i in range(len(states)):
            assert verify_ice_state(B1, states[i], lat.coordination), (
                f"State {i} is not a valid ice state"
            )

    def test_enumerated_states_distinct(self, square_open_setup):
        """All enumerated states should be distinct."""
        lat, B1, sigma_seed = square_open_setup
        lb = extract_loop_basis(lat.graph, B1, lat.edge_list)

        states = enumerate_all_ice_states(
            sigma_seed, lb.loop_indicators.numpy(),
            B1, lat.coordination,
            cycle_edge_lists=lb.cycle_edge_lists,
        )
        unique_set = set()
        for s in states:
            unique_set.add(tuple(s.astype(np.int8)))
        assert len(unique_set) == len(states)

    def test_too_large_raises(self, square_open_setup):
        lat, B1, sigma_seed = square_open_setup
        fake_indicators = np.random.randint(0, 2, size=(30, lat.n_edges)).astype(np.float64)
        with pytest.raises(ValueError, match="too large"):
            enumerate_all_ice_states(sigma_seed, fake_indicators, B1, lat.coordination)


class TestMetrics:
    def test_kl_uniform(self):
        """KL = 0 for uniform distribution."""
        n = 100
        log_probs = np.full(n, -np.log(n))
        kl = kl_divergence_exact(log_probs, n)
        assert abs(kl) < 1e-10

    def test_kl_nonuniform(self):
        """KL > 0 for non-uniform distribution."""
        n = 100
        log_probs = np.full(n, -10.0)
        log_probs[0] = 0.0
        kl = kl_divergence_exact(log_probs, n)
        assert kl > 0

    def test_hamming_identical(self):
        """Identical samples -> Hamming distance = 0."""
        samples = np.ones((10, 20))
        mean, std = mean_hamming_distance(samples)
        assert mean == 0.0

    def test_hamming_opposite(self):
        """Opposite samples -> Hamming distance = 1."""
        samples = np.array([[1, 1, 1, 1], [-1, -1, -1, -1]], dtype=np.float64)
        mean, std = mean_hamming_distance(samples)
        assert abs(mean - 1.0) < 1e-10

    def test_ess_uniform_weights(self):
        """Uniform weights -> ESS = n."""
        n = 100
        log_q = np.zeros(n)
        ess = effective_sample_size(log_q)
        assert abs(ess - n) < 1e-5

    def test_energy_periodic_ice_state(self, square_periodic_setup):
        """Energy should be 0 for periodic ice states (all even degree)."""
        lat, B1, sigma_seed = square_periodic_setup
        L_equ = B1.T @ B1
        e = energy(sigma_seed, L_equ)
        assert abs(e) < 1e-10

    def test_energy_open_ice_state(self, square_open_setup):
        """Energy for open BC ice state = sum of squared charges at odd-degree vertices."""
        lat, B1, sigma_seed = square_open_setup
        L_equ = B1.T @ B1
        e = energy(sigma_seed, L_equ)
        # Open BC has odd-degree vertices with |Q|=1, so E = number of odd-degree vertices
        n_odd = int(np.sum(lat.coordination % 2 == 1))
        assert abs(e - n_odd) < 1e-10

    def test_ice_violation_valid(self, square_open_setup):
        lat, B1, sigma_seed = square_open_setup
        v = ice_rule_violation(sigma_seed, B1, lat.coordination)
        assert v == 0.0


class TestTrainingSmoke:
    def test_short_training_runs(self, square_open_setup):
        """Smoke test: training runs without error for a few epochs."""
        lat, B1, sigma_seed = square_open_setup
        ops = build_eign_operators(B1)
        lb = extract_loop_basis(lat.graph, B1, lat.edge_list)
        lb.ordering = list(range(lb.n_loops))

        model = LoopMPVAN(
            operators=ops, loop_basis=lb,
            n_layers=2, equ_dim=8, inv_dim=4, head_hidden=16,
            B1_csc=B1,
        )
        inv_features = build_inv_features(lat.edge_list, lat.coordination)

        config = TrainingConfig(
            n_epochs=5,
            batch_size=4,
            lr=1e-3,
            eval_every=5,
            seed=42,
        )

        result = train(
            model, sigma_seed, inv_features,
            B1, lat.coordination, config,
        )

        assert len(result.loss_history) == 5
        assert len(result.eval_epochs) >= 1
        assert all(np.isfinite(result.loss_history))
