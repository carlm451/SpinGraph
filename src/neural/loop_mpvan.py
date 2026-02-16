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

from typing import List, Optional

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
        return is_directed_cycle_torch(
            sigma,
            self.loop_basis.cycle_edge_lists[loop_idx],
            self._B1_csc,
        )

    def forward_log_prob(
        self,
        alpha: torch.Tensor,
        sigma_seed: torch.Tensor,
        inv_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log q_theta(alpha) via teacher forcing with directed-cycle checks.

        At each step, if the loop is not a directed cycle in the current sigma,
        alpha_i is forced to 0 and contributes 0 to log_prob.

        Parameters
        ----------
        alpha : (beta_1,) binary tensor (desired flips, may be overridden)
        sigma_seed : (n1,) seed ice state +1/-1
        inv_features : (n1, 2) invariant input features

        Returns
        -------
        log_prob : scalar tensor (differentiable)
        """
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample ice states autoregressively with directed-cycle checks.

        Parameters
        ----------
        sigma_seed : (n1,) seed ice state +1/-1
        inv_features : (n1, 2) invariant input features
        n_samples : int

        Returns
        -------
        sigmas : (n_samples, n1) tensor of ice states
        log_probs : (n_samples,) tensor of log q_theta values
        """
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

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
