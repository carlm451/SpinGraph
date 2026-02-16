"""Quality metrics for ice state samplers.

Provides metrics for evaluating neural and MCMC samplers:
  - KL divergence against exact uniform (for small systems)
  - Mean pairwise Hamming distance (diversity measure)
  - Effective sample size (importance-weight-based)
  - Energy (should be 0 for all Mode A samples)
  - Ice rule violation rate (should be 0 for Mode A)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from scipy import sparse


def kl_divergence_exact(
    log_probs_model: np.ndarray,
    n_total_states: int,
) -> float:
    """KL divergence D_KL(q_theta || uniform) for exact enumeration.

    Parameters
    ----------
    log_probs_model : array (n_states,)
        Log probabilities assigned by the model to each state.
        Must cover all 2^beta_1 states.
    n_total_states : int
        Total number of ice states (= 2^beta_1).

    Returns
    -------
    kl : float
        D_KL(q || p_uniform) = sum_x q(x) [log q(x) - log p(x)]
        = sum_x q(x) log q(x) + log(n_total_states)
        = -H(q) + log(n_total_states)
    """
    # Normalize log probs
    log_probs = log_probs_model - np.logaddexp.reduce(log_probs_model)
    probs = np.exp(log_probs)

    # H(q) = -sum q log q
    entropy = -np.sum(probs * log_probs)

    # KL = log(N) - H(q)
    kl = np.log(n_total_states) - entropy
    return max(0.0, kl)  # Clamp to non-negative (numerical noise)


def kl_from_samples(
    samples: np.ndarray,
    exact_states: np.ndarray,
) -> float:
    """Estimate KL divergence from samples against exact uniform distribution.

    Bins sampled states and compares empirical distribution to uniform.

    Parameters
    ----------
    samples : array (n_samples, n_edges) of +1/-1
    exact_states : array (n_exact, n_edges) of +1/-1

    Returns
    -------
    kl : float
        Estimated D_KL(empirical || uniform)
    """
    n_exact = exact_states.shape[0]
    n_samples = samples.shape[0]

    # Map each exact state to an index
    state_to_idx = {}
    for i in range(n_exact):
        key = tuple(exact_states[i].astype(np.int8))
        state_to_idx[key] = i

    # Count occurrences of each state in samples
    counts = np.zeros(n_exact, dtype=np.int64)
    for s in samples:
        key = tuple(s.astype(np.int8))
        idx = state_to_idx.get(key)
        if idx is not None:
            counts[idx] += 1

    # Empirical distribution (with Laplace smoothing)
    q = (counts + 1.0) / (n_samples + n_exact)
    p = np.ones(n_exact) / n_exact

    # KL(q || p) = sum q * log(q / p)
    kl = np.sum(q * np.log(q / p))
    return max(0.0, kl)


def mean_hamming_distance(samples: np.ndarray) -> tuple[float, float]:
    """Compute mean pairwise Hamming distance (normalized by n_edges).

    Parameters
    ----------
    samples : array (n_samples, n_edges) of +1/-1

    Returns
    -------
    mean_dist : float
        Mean d_H / n_edges. For uniform sampling, expect ~0.5.
    std_dist : float
        Standard deviation of normalized Hamming distances.
    """
    n_samples, n_edges = samples.shape
    if n_samples < 2:
        return 0.0, 0.0

    # Pairwise Hamming = number of positions where spins differ
    # For +1/-1 encoding: d_H(a, b) = sum(a != b) = sum(a * b == -1)
    # = (n_edges - a . b) / 2
    dots = samples @ samples.T  # (n_samples, n_samples)
    hamming = (n_edges - dots) / 2.0
    normalized = hamming / n_edges

    # Extract upper triangle
    mask = np.triu(np.ones((n_samples, n_samples), dtype=bool), k=1)
    distances = normalized[mask]

    return float(np.mean(distances)), float(np.std(distances))


def effective_sample_size(
    log_q: np.ndarray,
    log_p: Optional[np.ndarray] = None,
) -> float:
    """Importance-weight-based effective sample size.

    ESS = (sum w_i)^2 / sum w_i^2
    where w_i = p(x_i) / q(x_i).

    For uniform target: w_i = 1 / (N * q(x_i)) where N = total states.

    Parameters
    ----------
    log_q : array (n_samples,)
        Log probabilities under the model for each sample.
    log_p : array (n_samples,), optional
        Log probabilities under the target. If None, assumes uniform.

    Returns
    -------
    ess : float
        Effective sample size (between 1 and n_samples).
    """
    n = len(log_q)
    if log_p is None:
        # Uniform target: log w = -log q + const (const cancels in ESS ratio)
        log_w = -log_q
    else:
        log_w = log_p - log_q

    # Stabilize
    log_w = log_w - np.max(log_w)
    w = np.exp(log_w)

    ess = (np.sum(w) ** 2) / np.sum(w ** 2)
    return float(ess)


def energy(
    sigma: np.ndarray,
    L_equ_scipy: sparse.spmatrix,
) -> float:
    """Compute ice Hamiltonian energy E = sigma^T @ L_equ @ sigma.

    For valid ice states this should be exactly 0 (sigma in kernel of L1_down).

    Parameters
    ----------
    sigma : array (n_edges,) of +1/-1
    L_equ_scipy : sparse matrix (n1, n1), = B1^T @ B1

    Returns
    -------
    energy : float
    """
    return float(sigma @ L_equ_scipy @ sigma)


def ice_rule_violation(
    sigma: np.ndarray,
    B1: sparse.spmatrix,
    coordination: np.ndarray,
) -> float:
    """Fraction of vertices violating the ice rule.

    Parameters
    ----------
    sigma : array (n_edges,) of +1/-1
    B1 : sparse (n0, n1)
    coordination : array (n0,)

    Returns
    -------
    violation_rate : float
        Fraction of vertices where |Q_v| != z_v mod 2.
    """
    charge = np.asarray((sparse.csr_matrix(B1) @ sigma)).ravel()
    target = coordination.astype(np.float64) % 2
    violations = np.abs(np.abs(charge) - target) > 0.5
    return float(np.mean(violations))


def batch_ice_rule_violation(
    samples: np.ndarray,
    B1: sparse.spmatrix,
    coordination: np.ndarray,
) -> float:
    """Fraction of samples that have any ice rule violation.

    Parameters
    ----------
    samples : array (n_samples, n_edges) of +1/-1
    B1 : sparse (n0, n1)
    coordination : array (n0,)

    Returns
    -------
    violation_rate : float
        Fraction of samples with at least one vertex violation.
    """
    n_violations = 0
    for sigma in samples:
        if ice_rule_violation(sigma, B1, coordination) > 0:
            n_violations += 1
    return n_violations / len(samples)
