"""EIGN Layer: dual-channel edge-level message passing.

Implements the EIGN update equations with two feature channels:
  - Equivariant channel (X_equ): carries sign-sensitive spin information
  - Invariant channel (X_inv): carries geometric / charge information

Each layer has 6 learnable weight matrices:
  W1: equ->equ via L_equ (Hamiltonian message passing)
  W2: inv->equ via inv_to_equ = B1^T|B1| (cross-channel)
  W3: inv->inv via L_inv = |B1|^T|B1| (invariant message passing)
  W4: equ->inv via equ_to_inv = |B1|^TB1 (charge monitor)
  W5: equ->equ skip connection
  W6: inv->inv skip connection
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .operators import EIGNOperators


class EIGNLayer(nn.Module):
    """Single EIGN dual-channel update layer.

    Parameters
    ----------
    equ_dim : int
        Dimension of equivariant features per edge.
    inv_dim : int
        Dimension of invariant features per edge.
    operators : EIGNOperators
        Precomputed sparse operator matrices.
    activation : str
        Activation function ('gelu' or 'relu').
    """

    def __init__(
        self,
        equ_dim: int,
        inv_dim: int,
        operators: EIGNOperators,
        activation: str = "gelu",
    ):
        super().__init__()
        self.equ_dim = equ_dim
        self.inv_dim = inv_dim
        self.ops = operators

        # Message-passing weights
        self.W1 = nn.Linear(equ_dim, equ_dim, bias=False)  # equ->equ via L_equ
        self.W2 = nn.Linear(inv_dim, equ_dim, bias=False)  # inv->equ via inv_to_equ
        self.W3 = nn.Linear(inv_dim, inv_dim, bias=False)  # inv->inv via L_inv
        self.W4 = nn.Linear(equ_dim, inv_dim, bias=False)  # equ->inv via equ_to_inv

        # Skip connections
        self.W5 = nn.Linear(equ_dim, equ_dim, bias=False)  # equ skip
        self.W6 = nn.Linear(inv_dim, inv_dim, bias=False)  # inv skip

        # Normalization
        self.norm_equ = nn.LayerNorm(equ_dim)
        self.norm_inv = nn.LayerNorm(inv_dim)

        # Activation
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self._init_weights()

    def _init_weights(self):
        """Initialize weights: Xavier for message-passing, near-identity for skip."""
        for W in [self.W1, self.W2, self.W3, self.W4]:
            nn.init.xavier_uniform_(W.weight)

        # Skip connections start as near-identity (pass-through initially)
        nn.init.eye_(self.W5.weight)
        nn.init.eye_(self.W6.weight)
        # Add small noise to break symmetry
        with torch.no_grad():
            self.W5.weight.add_(torch.randn_like(self.W5.weight) * 0.01)
            self.W6.weight.add_(torch.randn_like(self.W6.weight) * 0.01)

    def forward(
        self,
        X_equ: torch.Tensor,
        X_inv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: dual-channel EIGN update.

        Parameters
        ----------
        X_equ : (n1, equ_dim) equivariant edge features
        X_inv : (n1, inv_dim) invariant edge features

        Returns
        -------
        X_equ_new : (n1, equ_dim)
        X_inv_new : (n1, inv_dim)
        """
        # NOTE on "deaf Hamiltonian" (C1 from physics review):
        # For ice states sigma, L_equ @ sigma = B1^T B1 @ sigma = B1^T Q = 0 since
        # Q = B1 @ sigma = 0 (ice rule). So the equ->equ channel (W1) and the
        # equ->inv channel (W4) receive zero input in layer 1 when X_equ = sigma.
        # However, training still works because:
        #   1. Skip connection W5 passes sigma through unchanged
        #   2. GELU activation makes layer-1 output nonlinear in sigma
        #   3. From layer 2 onward, L_equ @ GELU(...) != 0 — Hamiltonian is active
        #   4. inv->inv (W3) and inv->equ (W2) channels are active in all layers
        # If periodic BC + even coordination causes training failure, consider
        # adding depth (n_layers=5) or a nonlinear input expansion.

        # Message-passing terms (sparse @ dense)
        # L_equ @ X_equ @ W1: Hamiltonian MP in equivariant channel
        msg_equ_equ = torch.sparse.mm(self.ops.L_equ, X_equ)
        msg_equ_equ = self.W1(msg_equ_equ)

        # inv_to_equ @ X_inv @ W2: cross-channel invariant -> equivariant
        msg_inv_equ = torch.sparse.mm(self.ops.inv_to_equ, X_inv)
        msg_inv_equ = self.W2(msg_inv_equ)

        # L_inv @ X_inv @ W3: invariant MP
        msg_inv_inv = torch.sparse.mm(self.ops.L_inv, X_inv)
        msg_inv_inv = self.W3(msg_inv_inv)

        # equ_to_inv @ X_equ @ W4: cross-channel equivariant -> invariant
        msg_equ_inv = torch.sparse.mm(self.ops.equ_to_inv, X_equ)
        msg_equ_inv = self.W4(msg_equ_inv)

        # Skip connections
        skip_equ = self.W5(X_equ)
        skip_inv = self.W6(X_inv)

        # Combine
        X_equ_new = msg_equ_equ + msg_inv_equ + skip_equ
        X_inv_new = msg_inv_inv + msg_equ_inv + skip_inv

        # Normalize and activate
        X_equ_new = self.act(self.norm_equ(X_equ_new))
        X_inv_new = self.act(self.norm_inv(X_inv_new))

        return X_equ_new, X_inv_new

    def forward_batch(
        self,
        X_equ: torch.Tensor,
        X_inv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched forward pass: dual-channel EIGN update.

        Uses reshape trick for batched sparse matmul:
        (B, n1, d) -> transpose -> (n1, B*d) -> sparse_mm -> reshape -> (B, n1, d)

        Parameters
        ----------
        X_equ : (B, n1, equ_dim) equivariant edge features
        X_inv : (B, n1, inv_dim) invariant edge features

        Returns
        -------
        X_equ_new : (B, n1, equ_dim)
        X_inv_new : (B, n1, inv_dim)
        """
        def _bsmm(sparse_mat, dense_batch):
            B, n, d = dense_batch.shape
            flat = dense_batch.transpose(0, 1).reshape(n, B * d)
            out = torch.sparse.mm(sparse_mat, flat)
            return out.reshape(n, B, d).transpose(0, 1)

        # Message-passing terms (batched sparse @ dense)
        msg_equ_equ = self.W1(_bsmm(self.ops.L_equ, X_equ))
        msg_inv_equ = self.W2(_bsmm(self.ops.inv_to_equ, X_inv))
        msg_inv_inv = self.W3(_bsmm(self.ops.L_inv, X_inv))
        msg_equ_inv = self.W4(_bsmm(self.ops.equ_to_inv, X_equ))

        # Skip connections
        skip_equ = self.W5(X_equ)
        skip_inv = self.W6(X_inv)

        # Combine
        X_equ_new = msg_equ_equ + msg_inv_equ + skip_equ
        X_inv_new = msg_inv_inv + msg_equ_inv + skip_inv

        # Normalize and activate
        X_equ_new = self.act(self.norm_equ(X_equ_new))
        X_inv_new = self.act(self.norm_inv(X_inv_new))

        return X_equ_new, X_inv_new
