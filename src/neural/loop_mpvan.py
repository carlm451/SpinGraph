"""LoopMPVAN: autoregressive model for ice state sampling via loop flips.

Mode A: Loop-Basis sampling with directed-cycle checking. Every generated
sample is a valid ice state by construction. At each autoregressive step,
the cycle is checked for directedness in the current sigma. Only directed
cycles can be flipped; non-directed cycles are skipped.

Architecture:
  1. Input: current spin config sigma -> edge features
  2. Stack of K shared EIGNLayers (unmasked, full lattice view)
  3. LoopOutputHead: pool edge features over loop edges -> MLP -> sigmoid -> p_i

At each autoregressive step i:
  - Current spin config reflects all prior directed flips
  - Check if loop i is a directed cycle in current sigma
  - If directed: EIGN stack -> pool -> MLP -> p_i, sample flip
  - If not directed: skip (alpha_i forced to 0)
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from scipy import sparse

from .eign_layer import EIGNLayer
from .loop_basis import (
    LoopBasis,
    flip_single_loop,
    is_directed_cycle_torch,
)
from .operators import EIGNOperators


class LoopOutputHead(nn.Module):
    """Pool EIGN features over loop edges -> MLP -> sigmoid -> probability."""

    def __init__(self, equ_dim: int, inv_dim: int, hidden_dim: int):
        super().__init__()
        input_dim = equ_dim + inv_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        X_equ: torch.Tensor,
        X_inv: torch.Tensor,
        loop_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute flip probability for one loop.

        Parameters
        ----------
        X_equ : (n1, equ_dim)
        X_inv : (n1, inv_dim)
        loop_mask : (n1,) binary mask

        Returns
        -------
        p : scalar tensor in (0, 1)
        """
        X_cat = torch.cat([X_equ, X_inv], dim=-1)
        mask = loop_mask.unsqueeze(-1)
        n_edges_in_loop = mask.sum().clamp(min=1.0)
        pooled = (X_cat * mask).sum(dim=0) / n_edges_in_loop
        logit = self.mlp(pooled).squeeze(-1)
        return torch.sigmoid(logit)


class LoopMPVAN(nn.Module):
    """Autoregressive loop-flip model for ice state sampling.

    Parameters
    ----------
    operators : EIGNOperators
        Precomputed EIGN operator matrices.
    loop_basis : LoopBasis
        Extracted loop basis with ordering.
    B1_csc : scipy sparse CSC matrix
        For directed-cycle checking. If None, uses operators.B1 converted back.
    n_layers : int
        Number of EIGN layers.
    equ_dim : int
        Equivariant feature dimension.
    inv_dim : int
        Invariant feature dimension.
    head_hidden : int
        Hidden dimension of the output head MLP.
    """

    def __init__(
        self,
        operators: EIGNOperators,
        loop_basis: LoopBasis,
        n_layers: int = 4,
        equ_dim: int = 32,
        inv_dim: int = 16,
        head_hidden: int = 64,
        B1_csc: Optional[sparse.spmatrix] = None,
    ):
        super().__init__()
        self.ops = operators
        self.loop_basis = loop_basis
        self.n_layers = n_layers
        self.equ_dim = equ_dim
        self.inv_dim = inv_dim

        # Store B1_csc for directed-cycle checking
        if B1_csc is not None:
            self._B1_csc = B1_csc
        else:
            # Convert from torch sparse back to scipy
            B1_dense = operators.B1.to_dense().numpy()
            self._B1_csc = sparse.csc_matrix(B1_dense)

        # Precompute per-cycle (edge_pairs, b1_signs) tensors for fast
        # batched directedness checking (eliminates scipy column extraction)
        self._cycle_data = []
        for loop_idx in range(loop_basis.n_loops):
            cycle_edges = loop_basis.cycle_edge_lists[loop_idx]
            vertex_edges: dict[int, list[tuple[int, float]]] = {}
            for e in cycle_edges:
                col = self._B1_csc.getcol(e)
                for v, b1_val in zip(col.indices, col.data):
                    vertex_edges.setdefault(v, []).append((e, b1_val))
            edge_pairs = []
            b1_signs = []
            for v, pairs in sorted(vertex_edges.items()):
                if len(pairs) == 2:
                    edge_pairs.append([pairs[0][0], pairs[1][0]])
                    b1_signs.append([pairs[0][1], pairs[1][1]])
            self._cycle_data.append({
                'edge_pairs': torch.tensor(edge_pairs, dtype=torch.long),
                'b1_signs': torch.tensor(b1_signs, dtype=torch.float32),
            })

        # Input projections
        self.equ_input = nn.Linear(1, equ_dim)
        self.inv_input = nn.Linear(2, inv_dim)

        # Shared EIGN layers
        self.layers = nn.ModuleList([
            EIGNLayer(equ_dim, inv_dim, operators)
            for _ in range(n_layers)
        ])

        # Shared output head
        self.output_head = LoopOutputHead(equ_dim, inv_dim, head_hidden)

    def _build_features(
        self,
        sigma: torch.Tensor,
        inv_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build input features and run through EIGN stack."""
        X_equ = self.equ_input(sigma.unsqueeze(-1))
        X_inv = self.inv_input(inv_features)

        for layer in self.layers:
            X_equ, X_inv = layer(X_equ, X_inv)

        return X_equ, X_inv

    def _is_loop_directed(self, sigma: torch.Tensor, loop_idx: int) -> bool:
        """Check if loop is a directed cycle in current sigma."""
        data = self._cycle_data[loop_idx]
        sigma_at_edges = sigma[data['edge_pairs']]  # (V, 2)
        flows = data['b1_signs'] * sigma_at_edges    # (V, 2)
        return (flows[:, 0] * flows[:, 1] < 0).all().item()

    def forward_log_prob(
        self,
        alpha: torch.Tensor,
        sigma_seed: torch.Tensor,
        inv_features: torch.Tensor,
        ordering: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """Compute log q_theta(alpha) via teacher forcing with directed-cycle checks.

        At each step, if the loop is not a directed cycle in the current sigma,
        alpha_i is forced to 0 and contributes 0 to log_prob.

        Parameters
        ----------
        alpha : (beta_1,) binary tensor (desired flips, may be overridden)
        sigma_seed : (n1,) seed ice state +1/-1
        inv_features : (n1, 2) invariant input features
        ordering : optional loop processing order (defaults to self.loop_basis.ordering)

        Returns
        -------
        log_prob : scalar tensor (differentiable)
        """
        if ordering is None:
            ordering = self.loop_basis.ordering
        indicators = self.loop_basis.loop_indicators
        log_prob = torch.tensor(0.0, dtype=torch.float32)
        sigma = sigma_seed.clone()

        for loop_idx in ordering:
            # Check directedness
            directed = self._is_loop_directed(sigma, loop_idx)

            if not directed:
                # Loop not flippable -> forced to no-flip, contributes 0 to log_prob
                continue

            # Run EIGN stack on current sigma
            X_equ, X_inv = self._build_features(sigma, inv_features)

            # Get probability for this loop
            loop_mask = indicators[loop_idx]
            p_i = self.output_head(X_equ, X_inv, loop_mask)
            p_i = p_i.clamp(1e-6, 1.0 - 1e-6)

            a_i = alpha[loop_idx]
            log_prob = log_prob + a_i * torch.log(p_i) + (1 - a_i) * torch.log(1 - p_i)

            # Apply the flip if alpha says to
            if a_i > 0.5:
                sigma = flip_single_loop(sigma, indicators, loop_idx)

        return log_prob

    @torch.no_grad()
    def sample(
        self,
        sigma_seed: torch.Tensor,
        inv_features: torch.Tensor,
        n_samples: int = 1,
        ordering: Optional[Sequence[int]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample ice states autoregressively with directed-cycle checks.

        Parameters
        ----------
        sigma_seed : (n1,) seed ice state +1/-1
        inv_features : (n1, 2) invariant input features
        n_samples : int
        ordering : optional loop processing order (defaults to self.loop_basis.ordering)

        Returns
        -------
        sigmas : (n_samples, n1) tensor of ice states
        log_probs : (n_samples,) tensor of log q_theta values
        """
        if ordering is None:
            ordering = self.loop_basis.ordering
        indicators = self.loop_basis.loop_indicators

        all_sigmas = []
        all_log_probs = []

        for _ in range(n_samples):
            sigma = sigma_seed.clone()
            log_prob = 0.0

            for loop_idx in ordering:
                # Check directedness
                directed = self._is_loop_directed(sigma, loop_idx)

                if not directed:
                    # Skip this loop
                    continue

                # Run EIGN stack
                X_equ, X_inv = self._build_features(sigma, inv_features)

                # Get probability
                loop_mask = indicators[loop_idx]
                p_i = self.output_head(X_equ, X_inv, loop_mask)
                p_i = p_i.clamp(1e-6, 1.0 - 1e-6)

                # Sample
                a_i = torch.bernoulli(p_i).item()
                log_prob += a_i * torch.log(p_i).item() + (1 - a_i) * torch.log(1 - p_i).item()

                if a_i > 0.5:
                    sigma = flip_single_loop(sigma, indicators, loop_idx)

            all_sigmas.append(sigma)
            all_log_probs.append(log_prob)

        return torch.stack(all_sigmas), torch.tensor(all_log_probs)

    def _build_features_batch(
        self,
        sigmas: torch.Tensor,
        inv_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build input features and run through EIGN stack (batched).

        Parameters
        ----------
        sigmas : (B, n1) spin configurations
        inv_features : (n1, 2) invariant features (shared across batch)

        Returns
        -------
        X_equ : (B, n1, equ_dim)
        X_inv : (B, n1, inv_dim)
        """
        B = sigmas.shape[0]
        X_equ = self.equ_input(sigmas.unsqueeze(-1))  # (B, n1, equ_dim)
        X_inv = self.inv_input(inv_features.unsqueeze(0).expand(B, -1, -1))  # (B, n1, inv_dim)

        for layer in self.layers:
            X_equ, X_inv = layer.forward_batch(X_equ, X_inv)

        return X_equ, X_inv

    def _batch_is_directed(
        self,
        sigmas: torch.Tensor,
        loop_idx: int,
    ) -> torch.Tensor:
        """Check directedness for a batch of sigmas at a given loop.

        Parameters
        ----------
        sigmas : (B, n1) spin configurations
        loop_idx : index of loop to check

        Returns
        -------
        directed : (B,) bool tensor
        """
        data = self._cycle_data[loop_idx]
        # data['edge_pairs']: (V, 2), data['b1_signs']: (V, 2)
        sigma_at_edges = sigmas[:, data['edge_pairs']]  # (B, V, 2)
        flows = data['b1_signs'].unsqueeze(0) * sigma_at_edges  # (B, V, 2)
        # Directed if all vertex pairs have opposite flow signs
        return (flows[:, :, 0] * flows[:, :, 1] < 0).all(dim=1)  # (B,)

    def forward_log_prob_batch(
        self,
        alphas: torch.Tensor,
        sigma_seed: torch.Tensor,
        inv_features: torch.Tensor,
        ordering: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """Compute log q_theta(alpha) for a batch via teacher forcing.

        Parameters
        ----------
        alphas : (B, beta_1) binary tensor
        sigma_seed : (n1,) seed ice state +1/-1
        inv_features : (n1, 2) invariant input features
        ordering : optional loop processing order (defaults to self.loop_basis.ordering)

        Returns
        -------
        log_probs : (B,) tensor (differentiable)
        """
        B = alphas.shape[0]
        if ordering is None:
            ordering = self.loop_basis.ordering
        indicators = self.loop_basis.loop_indicators

        log_probs = torch.zeros(B, dtype=torch.float32)
        sigmas = sigma_seed.unsqueeze(0).expand(B, -1).clone()  # (B, n1)

        for loop_idx in ordering:
            # Batched directedness check
            directed = self._batch_is_directed(sigmas, loop_idx)  # (B,)

            if not directed.any():
                continue

            # Batched EIGN forward (only need to process directed samples,
            # but for simplicity process all and mask — avoids gather/scatter)
            X_equ, X_inv = self._build_features_batch(sigmas, inv_features)

            # Batched pooling over loop edges
            loop_mask = indicators[loop_idx]  # (n1,)
            n_loop_edges = loop_mask.sum().clamp(min=1.0)
            mask = loop_mask.unsqueeze(0).unsqueeze(-1)  # (1, n1, 1)

            X_cat = torch.cat([X_equ, X_inv], dim=-1)  # (B, n1, d)
            pooled = (X_cat * mask).sum(dim=1) / n_loop_edges  # (B, d)
            logit = self.output_head.mlp(pooled).squeeze(-1)  # (B,)
            p_i = torch.sigmoid(logit).clamp(1e-6, 1.0 - 1e-6)  # (B,)

            a_i = alphas[:, loop_idx]  # (B,)
            step_log_prob = a_i * torch.log(p_i) + (1 - a_i) * torch.log(1 - p_i)

            # Only accumulate for directed samples
            log_probs = log_probs + step_log_prob * directed.float()

            # Apply flips where alpha=1 and directed
            flip_mask = (a_i > 0.5) & directed  # (B,)
            if flip_mask.any():
                loop_edges = indicators[loop_idx].unsqueeze(0)  # (1, n1)
                sign_flip = 1.0 - 2.0 * loop_edges  # (1, n1): +1 for non-loop, -1 for loop
                # Only flip for samples where flip_mask is True
                flip_factor = torch.where(
                    flip_mask.unsqueeze(-1),  # (B, 1)
                    sign_flip,                # flip loop edges
                    torch.ones_like(sign_flip) # no flip
                )
                sigmas = sigmas * flip_factor

        return log_probs

    @torch.no_grad()
    def sample_batch(
        self,
        sigma_seed: torch.Tensor,
        inv_features: torch.Tensor,
        n_samples: int = 1,
        ordering: Optional[Sequence[int]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample ice states autoregressively (batched).

        Parameters
        ----------
        sigma_seed : (n1,) seed ice state +1/-1
        inv_features : (n1, 2) invariant input features
        n_samples : int
        ordering : optional loop processing order (defaults to self.loop_basis.ordering)

        Returns
        -------
        sigmas : (n_samples, n1) tensor of ice states
        log_probs : (n_samples,) tensor of log q_theta values
        """
        if ordering is None:
            ordering = self.loop_basis.ordering
        indicators = self.loop_basis.loop_indicators

        sigmas = sigma_seed.unsqueeze(0).expand(n_samples, -1).clone()  # (B, n1)
        log_probs = torch.zeros(n_samples, dtype=torch.float32)

        for loop_idx in ordering:
            # Batched directedness check
            directed = self._batch_is_directed(sigmas, loop_idx)  # (B,)

            if not directed.any():
                continue

            # Batched EIGN forward
            X_equ, X_inv = self._build_features_batch(sigmas, inv_features)

            # Batched pooling
            loop_mask = indicators[loop_idx]  # (n1,)
            n_loop_edges = loop_mask.sum().clamp(min=1.0)
            mask = loop_mask.unsqueeze(0).unsqueeze(-1)  # (1, n1, 1)

            X_cat = torch.cat([X_equ, X_inv], dim=-1)  # (B, n1, d)
            pooled = (X_cat * mask).sum(dim=1) / n_loop_edges  # (B, d)
            logit = self.output_head.mlp(pooled).squeeze(-1)  # (B,)
            p_i = torch.sigmoid(logit).clamp(1e-6, 1.0 - 1e-6)  # (B,)

            # Sample decisions
            a_i = torch.bernoulli(p_i)  # (B,)

            # Accumulate log probs (only for directed samples)
            step_log_prob = a_i * torch.log(p_i) + (1 - a_i) * torch.log(1 - p_i)
            log_probs = log_probs + step_log_prob * directed.float()

            # Force non-directed to no-flip
            a_i = a_i * directed.float()

            # Apply flips
            flip_mask = a_i > 0.5  # (B,)
            if flip_mask.any():
                loop_edges = indicators[loop_idx].unsqueeze(0)  # (1, n1)
                sign_flip = 1.0 - 2.0 * loop_edges
                flip_factor = torch.where(
                    flip_mask.unsqueeze(-1),
                    sign_flip,
                    torch.ones_like(sign_flip)
                )
                sigmas = sigmas * flip_factor

        return sigmas, log_probs

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
