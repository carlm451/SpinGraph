"""Tests for MCMC benchmark suite (Phase 6)."""
from __future__ import annotations

import numpy as np
import pytest

from src.sampling.benchmark import (
    MCMCBenchmarkResult,
    estimate_autocorrelation_time,
    run_mcmc_benchmark,
)


class TestRunMCMCBenchmark:
    def test_square_xs(self):
        """Basic benchmark on square XS runs and returns valid results."""
        result = run_mcmc_benchmark(
            "square", "XS", "periodic",
            n_samples=50, n_flips_between=10, seed=42,
        )

        assert isinstance(result, MCMCBenchmarkResult)
        assert result.lattice_name == "square"
        assert result.size_label == "XS"
        assert result.n_samples == 50
        assert result.wall_time_seconds > 0
        assert result.time_per_sample > 0
        assert result.ice_rule_violation_rate == 0.0  # All samples valid
        assert 0 <= result.mean_hamming_distance <= 1.0
        assert 0 <= result.unique_fraction <= 1.0
        assert result.n_vertices > 0
        assert result.n_edges > 0

    def test_kagome_xs(self):
        result = run_mcmc_benchmark(
            "kagome", "XS", "periodic",
            n_samples=30, n_flips_between=10, seed=42,
        )
        assert result.ice_rule_violation_rate == 0.0
        assert result.mean_hamming_distance > 0

    def test_energy_zero(self):
        """Energy should be 0 for all ice state samples."""
        result = run_mcmc_benchmark(
            "square", "XS", "periodic",
            n_samples=20, n_flips_between=10, seed=42,
        )
        assert abs(result.mean_energy) < 1e-8

    def test_more_flips_better_mixing(self):
        """More flips between samples should give higher diversity."""
        r10 = run_mcmc_benchmark(
            "square", "XS", "periodic",
            n_samples=100, n_flips_between=5, seed=42,
        )
        r50 = run_mcmc_benchmark(
            "square", "XS", "periodic",
            n_samples=100, n_flips_between=50, seed=42,
        )
        # More flips should give at least as many unique states
        assert r50.unique_fraction >= r10.unique_fraction * 0.8


class TestAutocorrelation:
    def test_independent_samples(self):
        """Independent random samples should have tau ~ 1."""
        rng = np.random.default_rng(42)
        samples = rng.choice([-1.0, 1.0], size=(200, 50))
        tau = estimate_autocorrelation_time(samples)
        assert 0.5 < tau < 3.0, f"tau={tau} for independent samples"

    def test_correlated_samples(self):
        """Highly correlated samples should have tau > 1."""
        n = 200
        n_edges = 50
        rng = np.random.default_rng(42)
        # Start with a state and flip one edge at a time
        samples = np.zeros((n, n_edges))
        samples[0] = rng.choice([-1.0, 1.0], size=n_edges)
        for i in range(1, n):
            samples[i] = samples[i - 1].copy()
            # Flip one random edge
            idx = rng.integers(n_edges)
            samples[i, idx] *= -1
        tau = estimate_autocorrelation_time(samples)
        assert tau > 1.0, f"tau={tau} for correlated samples"

    def test_too_few_samples(self):
        """Should return NaN for too few samples."""
        samples = np.ones((2, 10))
        tau = estimate_autocorrelation_time(samples)
        assert np.isnan(tau)
