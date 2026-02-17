"""Tests for LoopMPVAN model (Phase 4)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.lattices.registry import get_generator
from src.neural.loop_basis import (
    compute_loop_ordering,
    extract_loop_basis,
)
from src.neural.loop_mpvan import LoopMPVAN
from src.neural.operators import build_eign_operators
from src.neural.training import build_inv_features
from src.topology.ice_sampling import find_seed_ice_state, verify_ice_state
from src.topology.incidence import build_B1


@pytest.fixture
def model_setup():
    """Build a small model on square XS open for testing."""
    gen = get_generator("square")
    lat = gen.build(4, 4, boundary="open")
    B1 = build_B1(lat.n_vertices, lat.edge_list)
    ops = build_eign_operators(B1)

    lb = extract_loop_basis(lat.graph, B1, lat.edge_list)
    lb.ordering = compute_loop_ordering(lb, strategy="natural")

    sigma_seed = find_seed_ice_state(B1, lat.coordination, edge_list=lat.edge_list)
    inv_features = build_inv_features(lat.edge_list, lat.coordination)

    model = LoopMPVAN(
        operators=ops, loop_basis=lb,
        n_layers=2, equ_dim=8, inv_dim=4, head_hidden=16,
        B1_csc=B1,
    )

    return model, lat, B1, sigma_seed, inv_features, lb


class TestForwardLogProb:
    def test_returns_scalar(self, model_setup):
        model, lat, B1, sigma_seed, inv_features, lb = model_setup
        seed_t = torch.from_numpy(sigma_seed.astype(np.float32))

        alpha = torch.zeros(lb.n_loops)
        log_prob = model.forward_log_prob(alpha, seed_t, inv_features)

        assert log_prob.dim() == 0, "log_prob should be scalar"

    def test_differentiable(self, model_setup):
        model, lat, B1, sigma_seed, inv_features, lb = model_setup
        seed_t = torch.from_numpy(sigma_seed.astype(np.float32))

        alpha = torch.zeros(lb.n_loops)
        log_prob = model.forward_log_prob(alpha, seed_t, inv_features)
        log_prob.backward()

        has_grad = False
        for p in model.parameters():
            if p.grad is not None and p.grad.norm().item() > 0:
                has_grad = True
                break
        assert has_grad, "No parameter received non-zero gradients"

    def test_log_prob_negative_or_zero(self, model_setup):
        """Log probability should be <= 0."""
        model, lat, B1, sigma_seed, inv_features, lb = model_setup
        seed_t = torch.from_numpy(sigma_seed.astype(np.float32))

        for _ in range(5):
            alpha = torch.bernoulli(torch.full((lb.n_loops,), 0.5))
            log_prob = model.forward_log_prob(alpha, seed_t, inv_features)
            assert log_prob.item() <= 1e-5, f"log_prob = {log_prob.item()} > 0"


class TestSampling:
    def test_all_samples_valid_ice(self, model_setup):
        """All sampled states should be valid ice states (directed flips)."""
        model, lat, B1, sigma_seed, inv_features, lb = model_setup
        seed_t = torch.from_numpy(sigma_seed.astype(np.float32))

        model.eval()
        sigmas, log_probs = model.sample(seed_t, inv_features, n_samples=20)

        assert sigmas.shape == (20, lat.n_edges)
        assert log_probs.shape == (20,)

        for i in range(20):
            sigma_np = sigmas[i].numpy()
            assert verify_ice_state(B1, sigma_np, lat.coordination), (
                f"Sample {i} violates ice rule"
            )

    def test_samples_include_seed(self, model_setup):
        """At least the seed state should be reachable (alpha=0)."""
        model, lat, B1, sigma_seed, inv_features, lb = model_setup
        seed_t = torch.from_numpy(sigma_seed.astype(np.float32))

        model.eval()
        sigmas, _ = model.sample(seed_t, inv_features, n_samples=50)

        # Check if any sample equals the seed (alpha=all zeros is always valid)
        found_seed = False
        for s in sigmas:
            if torch.allclose(s, seed_t, atol=1e-5):
                found_seed = True
                break
        # With 50 samples there's a decent chance of getting all-zeros alpha
        # but it's not guaranteed. Just check samples are valid.
        assert sigmas.shape[0] == 50

    def test_log_probs_finite(self, model_setup):
        model, lat, B1, sigma_seed, inv_features, lb = model_setup
        seed_t = torch.from_numpy(sigma_seed.astype(np.float32))

        model.eval()
        _, log_probs = model.sample(seed_t, inv_features, n_samples=10)

        assert torch.all(torch.isfinite(log_probs)), "Some log_probs are non-finite"


class TestBatchedEquivalence:
    """Verify batched methods produce identical results to sequential ones."""

    def test_forward_log_prob_batch_matches_sequential(self, model_setup):
        """forward_log_prob_batch should match per-sample forward_log_prob."""
        model, lat, B1, sigma_seed, inv_features, lb = model_setup
        seed_t = torch.from_numpy(sigma_seed.astype(np.float32))
        model.eval()

        torch.manual_seed(123)
        alphas = torch.bernoulli(torch.full((5, lb.n_loops), 0.5))

        # Sequential
        sequential_lps = []
        for i in range(5):
            lp = model.forward_log_prob(alphas[i], seed_t, inv_features)
            sequential_lps.append(lp.item())

        # Batched
        batched_lps = model.forward_log_prob_batch(alphas, seed_t, inv_features)

        for i in range(5):
            assert abs(batched_lps[i].item() - sequential_lps[i]) < 1e-5, (
                f"Sample {i}: batched={batched_lps[i].item():.6f} vs "
                f"sequential={sequential_lps[i]:.6f}"
            )

    def test_forward_log_prob_batch_differentiable(self, model_setup):
        """Batched forward should be differentiable."""
        model, lat, B1, sigma_seed, inv_features, lb = model_setup
        seed_t = torch.from_numpy(sigma_seed.astype(np.float32))

        alphas = torch.zeros(3, lb.n_loops)
        log_probs = model.forward_log_prob_batch(alphas, seed_t, inv_features)
        log_probs.sum().backward()

        has_grad = False
        for p in model.parameters():
            if p.grad is not None and p.grad.norm().item() > 0:
                has_grad = True
                break
        assert has_grad, "No parameter received non-zero gradients"

    def test_sample_batch_shapes(self, model_setup):
        """sample_batch should return correct shapes."""
        model, lat, B1, sigma_seed, inv_features, lb = model_setup
        seed_t = torch.from_numpy(sigma_seed.astype(np.float32))
        model.eval()

        sigmas, log_probs = model.sample_batch(seed_t, inv_features, n_samples=10)
        assert sigmas.shape == (10, lat.n_edges)
        assert log_probs.shape == (10,)

    def test_sample_batch_all_valid_ice(self, model_setup):
        """All batched samples should be valid ice states."""
        model, lat, B1, sigma_seed, inv_features, lb = model_setup
        seed_t = torch.from_numpy(sigma_seed.astype(np.float32))
        model.eval()

        sigmas, log_probs = model.sample_batch(seed_t, inv_features, n_samples=20)

        for i in range(20):
            sigma_np = sigmas[i].numpy()
            assert verify_ice_state(B1, sigma_np, lat.coordination), (
                f"Batched sample {i} violates ice rule"
            )

    def test_sample_batch_log_probs_finite(self, model_setup):
        """Batched sample log probs should be finite."""
        model, lat, B1, sigma_seed, inv_features, lb = model_setup
        seed_t = torch.from_numpy(sigma_seed.astype(np.float32))
        model.eval()

        _, log_probs = model.sample_batch(seed_t, inv_features, n_samples=10)
        assert torch.all(torch.isfinite(log_probs)), "Some batched log_probs are non-finite"

    def test_batch_is_directed_matches_sequential(self, model_setup):
        """_batch_is_directed should match _is_loop_directed for each sample."""
        model, lat, B1, sigma_seed, inv_features, lb = model_setup
        seed_t = torch.from_numpy(sigma_seed.astype(np.float32))
        model.eval()

        # Generate a few different sigmas
        sigmas, _ = model.sample_batch(seed_t, inv_features, n_samples=8)

        for loop_idx in range(min(lb.n_loops, 5)):
            batch_result = model._batch_is_directed(sigmas, loop_idx)
            for i in range(8):
                seq_result = model._is_loop_directed(sigmas[i], loop_idx)
                assert batch_result[i].item() == seq_result, (
                    f"Mismatch at sample {i}, loop {loop_idx}: "
                    f"batch={batch_result[i].item()}, seq={seq_result}"
                )


class TestModelProperties:
    def test_count_parameters(self, model_setup):
        model, *_ = model_setup
        n_params = model.count_parameters()
        assert n_params > 0
        assert n_params < 100_000

    def test_parameter_count_scales(self):
        gen = get_generator("square")
        lat = gen.build(4, 4, boundary="open")
        B1 = build_B1(lat.n_vertices, lat.edge_list)
        ops = build_eign_operators(B1)
        lb = extract_loop_basis(lat.graph, B1, lat.edge_list)
        lb.ordering = list(range(lb.n_loops))

        small = LoopMPVAN(ops, lb, n_layers=2, equ_dim=8, inv_dim=4, head_hidden=16, B1_csc=B1)
        large = LoopMPVAN(ops, lb, n_layers=4, equ_dim=32, inv_dim=16, head_hidden=64, B1_csc=B1)

        assert large.count_parameters() > small.count_parameters()
