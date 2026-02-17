"""REINFORCE training loop for LoopMPVAN.

Training objective (T=0 case): All ice states have H=0, so variational free
energy = T * E_q[ln q_theta]. Minimizing this = maximizing entropy = learning
to sample uniformly.

REINFORCE gradient: grad = E_q[(ln q - baseline) * grad_theta ln q]

Training schedule: Adam optimizer, cosine LR annealing, running-mean REINFORCE
baseline, gradient clipping.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.optim as optim

from .loop_basis import LoopBasis, apply_loop_flips, recover_alpha
from .loop_mpvan import LoopMPVAN
from .metrics import (
    batch_ice_rule_violation,
    effective_sample_size,
    kl_from_samples,
    mean_hamming_distance,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for REINFORCE training."""

    n_epochs: int = 2000
    batch_size: int = 64
    lr: float = 1e-3
    lr_scheduler: str = "cosine"
    baseline_momentum: float = 0.99
    entropy_bonus: float = 0.01
    grad_clip: float = 1.0
    eval_every: int = 200
    seed: int = 42


@dataclass
class TrainingResult:
    """Results from training."""

    loss_history: List[float] = field(default_factory=list)
    entropy_history: List[float] = field(default_factory=list)
    kl_history: List[float] = field(default_factory=list)
    hamming_history: List[float] = field(default_factory=list)
    ess_history: List[float] = field(default_factory=list)
    eval_epochs: List[int] = field(default_factory=list)
    grad_norm_history: List[float] = field(default_factory=list)
    advantage_var_history: List[float] = field(default_factory=list)
    final_model_state: Optional[Dict] = None


def build_inv_features(
    edge_list: list,
    coordination: np.ndarray,
) -> torch.Tensor:
    """Build invariant input features: endpoint coordination numbers.

    Parameters
    ----------
    edge_list : list of (int, int)
    coordination : array (n0,)

    Returns
    -------
    inv_features : tensor (n1, 2) with normalized coordinations
    """
    n_edges = len(edge_list)
    features = np.zeros((n_edges, 2), dtype=np.float32)
    max_z = float(coordination.max()) if len(coordination) > 0 else 1.0

    for e, (u, v) in enumerate(edge_list):
        features[e, 0] = coordination[u] / max_z
        features[e, 1] = coordination[v] / max_z

    return torch.from_numpy(features)


def train(
    model: LoopMPVAN,
    sigma_seed: np.ndarray,
    inv_features: torch.Tensor,
    B1_scipy,
    coordination: np.ndarray,
    config: TrainingConfig,
    exact_states: Optional[np.ndarray] = None,
) -> TrainingResult:
    """Train LoopMPVAN via REINFORCE to maximize entropy.

    Parameters
    ----------
    model : LoopMPVAN
    sigma_seed : array (n_edges,) of +1/-1
    inv_features : tensor (n1, 2)
    B1_scipy : scipy sparse (n0, n1)
    coordination : array (n0,)
    config : TrainingConfig
    exact_states : optional array (n_states, n_edges) for KL evaluation

    Returns
    -------
    TrainingResult
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    seed_tensor = torch.from_numpy(sigma_seed.astype(np.float32))
    indicators = model.loop_basis.loop_indicators
    n_loops = model.loop_basis.n_loops

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    if config.lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.n_epochs, eta_min=config.lr * 0.01
        )
    else:
        scheduler = None

    baseline = None  # Initialize from first batch (C5 cold-start fix)
    result = TrainingResult()

    logger.info(
        f"Starting training: {config.n_epochs} epochs, "
        f"batch_size={config.batch_size}, lr={config.lr}, "
        f"n_loops={n_loops}, n_params={model.count_parameters()}"
    )

    for epoch in range(config.n_epochs):
        model.train()
        optimizer.zero_grad()

        # Sample a batch of ice states
        with torch.no_grad():
            sigmas, log_probs_sample = model.sample(
                seed_tensor, inv_features, n_samples=config.batch_size
            )

        # Recover alpha vectors for each sample
        alphas = recover_alpha(sigmas, seed_tensor, indicators)  # (batch, n_loops)

        # Compute log probabilities with gradients (teacher forcing)
        batch_log_probs = []
        for b in range(config.batch_size):
            log_q = model.forward_log_prob(
                alphas[b], seed_tensor, inv_features
            )
            batch_log_probs.append(log_q)

        log_probs = torch.stack(batch_log_probs)  # (batch,)

        # REINFORCE: maximize entropy = minimize E[log q]
        # Reward = -log_q (lower log_q = higher entropy = better)
        # Advantage = reward - baseline = -log_q - baseline
        rewards = -log_probs.detach()
        if baseline is None:
            baseline = rewards.mean().item()
        else:
            baseline = config.baseline_momentum * baseline + (1 - config.baseline_momentum) * rewards.mean().item()
        advantages = rewards - baseline

        # Policy gradient loss
        # grad J = E[advantage * grad log q]
        # Loss = -E[advantage * log q] (negative for gradient descent)
        policy_loss = -(advantages * log_probs).mean()

        # Entropy bonus to prevent mode collapse
        entropy = -log_probs.mean()
        loss = policy_loss - config.entropy_bonus * entropy

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        # Record gradient diagnostics (C2)
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        result.grad_norm_history.append(total_norm)
        result.advantage_var_history.append(advantages.var().item())

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        result.loss_history.append(loss.item())
        result.entropy_history.append(entropy.item())

        # Evaluation
        if (epoch + 1) % config.eval_every == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                eval_sigmas, eval_log_probs = model.sample(
                    seed_tensor, inv_features, n_samples=min(256, config.batch_size * 4)
                )

            eval_np = eval_sigmas.numpy()
            log_probs_np = eval_log_probs.numpy()

            # Hamming distance
            h_mean, h_std = mean_hamming_distance(eval_np)

            # ESS
            ess = effective_sample_size(log_probs_np)

            # Ice rule violation
            violation = batch_ice_rule_violation(eval_np, B1_scipy, coordination)

            # KL if exact states available
            kl = float("nan")
            if exact_states is not None:
                kl = kl_from_samples(eval_np, exact_states)

            result.hamming_history.append(h_mean)
            result.ess_history.append(ess)
            result.kl_history.append(kl)
            result.eval_epochs.append(epoch + 1)

            logger.info(
                f"Epoch {epoch+1}/{config.n_epochs}: "
                f"loss={loss.item():.4f}, entropy={entropy.item():.4f}, "
                f"hamming={h_mean:.4f}, ESS={ess:.1f}, "
                f"violation={violation:.4f}, KL={kl:.4f}, "
                f"grad_norm={total_norm:.4f}, adv_var={advantages.var().item():.4f}"
            )

    # Save final model state
    result.final_model_state = {
        k: v.cpu().clone() for k, v in model.state_dict().items()
    }

    return result
