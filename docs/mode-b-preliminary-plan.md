# Mode B: Direct Edge Sampling — Preliminary Plan

> High-level implementation plan for the Edge-MPVAN direct edge sampler.
> Based on [Edge-MPVAN proposal](edge-mpvan-proposal.html), the VAN paper
> (Wu et al. 2019), and lessons learned from Mode A development.

---

## 1. What Mode B Is

Mode B is a **standard variational autoregressive network** (VAN) operating on edge spins.
It factorizes the joint distribution as:

```
q_θ(σ) = Π_{e=1}^{n₁} q_θ(σ_e | σ_{<e})
```

Each edge spin σ_e ∈ {+1, -1} is sampled sequentially, conditioned on all
previously assigned edges. The EIGN dual-channel message passing provides the
Hamiltonian-informed context at each step.

**Key difference from Mode A:** Mode B builds configurations edge-by-edge from
scratch. There is no seed ice state, no loop basis, and no requirement that
every sample satisfy the ice rule. Ice-rule violations (monopoles) are permitted
but energetically penalized, enabling **finite-temperature Boltzmann sampling**.

---

## 2. Why Mode B Matters

| Capability | Mode A | Mode B |
|------------|--------|--------|
| Temperature range | T = 0 only | Any T |
| Ice-rule guarantee | Exact (by construction) | Soft (via penalty) |
| Monopole physics | Cannot study | Natural |
| Phase transitions | Cannot access | Can sweep C(T) |
| Sequence length | β₁ | n₁ (2-3× longer) |

Mode B complements Mode A by accessing the full thermodynamic phase space,
including the paramagnetic-to-ice crossover and monopole pair creation.

---

## 3. Key Architectural Insights

### 3a. No Seed State Required

Unlike Mode A, which transforms a complete seed ice state via loop flips,
Mode B follows the standard VAN approach (Wu et al. 2019):

- **Input at step k:** A partial assignment vector σ̃ ∈ ℝ^{n₁} with assigned
  edges set to ±1 and unassigned edges set to **0**.
- **Step 1 (first edge):** The input is all zeros. The first conditional
  q_θ(σ_{e₁}) is unconditional — it depends only on the static lattice geometry
  (invariant features), not on any spin values.
- **No preprocessing dependency** on ice state construction or loop basis.

This is simpler than Mode A's seed + loop-flip pipeline and avoids the
Eulerian circuit construction entirely.

### 3b. Deaf Hamiltonian Is Not a Problem

The "deaf Hamiltonian" issue (C1 from physics review) that affects Mode A
does **not** affect Mode B:

- In Mode A, every intermediate state is a complete ice state, so
  L_equ @ σ = B₁ᵀB₁σ = B₁ᵀQ = 0 (since Q = B₁σ = 0 for all ice states).
  The equ→equ channel receives zero input in layer 1.
- In Mode B, partial assignments are **not** ice states. Vertices incident to
  unassigned edges generically have Q_v ≠ 0, so B₁σ̃ ≠ 0 and
  L_equ @ σ̃ ≠ 0. The Hamiltonian channel is **active from layer 1**.

This means Mode B's EIGN layers extract richer information from the start,
without needing the skip+GELU workaround that Mode A relies on.

### 3c. Charge Monitor for Free

The equ→inv channel (|B₁|ᵀB₁) computes unsigned vertex charge accumulation.
During autoregressive generation, this provides the network with a **real-time
ice-rule violation detector** — it can "see" where charges are building up
as each spin is assigned and adjust subsequent decisions to compensate.

---

## 4. Implementation Phases

### Phase B1: Causal Masking of EIGN Operators

The core engineering challenge. Each edge must only receive messages from
edges earlier in the ordering.

**What to build:**
- `build_causal_masks(edge_ordering, B1)` — precompute lower-triangular
  masked versions of all four EIGN operator products
- For self-modal operators (L_equ, L_inv): zero entries (L)_{ee'} where e' ≥ e
- For cross-modal operators: mask at the vertex aggregation step
- `MaskedEIGNLayer(nn.Module)` — wraps `EIGNLayer` with per-step masked operators
- Store masked operators as sparse tensors (same sparsity pattern, some entries zeroed)

**Validation:**
- Apply masked operator to one-hot edge vector → output should depend only on
  earlier edges in the ordering
- Compare masked layer output to full layer output for the causal subset

**Open question:** Should we precompute n₁ different masked operators (one per
step), or use a single lower-triangular mask? The proposal suggests precomputing
the masked sparse matrices once per ordering since the operators are already
sparse (at most 2z_max - 1 nonzero entries per row).

### Phase B2: Edge Ordering Strategies

The ordering of edges in the autoregressive decomposition affects conditional
prediction quality. Implement four strategies:

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| **BFS** (default) | BFS from seed vertex, order incident edges | General purpose, spatially coherent |
| **Spectral** | Sort by Fiedler vector of L₁_down | Captures long-range structure |
| **Backbone-first** | High-coordination edges first | Heterogeneous lattices (shakti, tetris) |
| **Random** | Random permutation per epoch (à la MADE) | Order-agnostic training |

**What to build:**
- `EdgeOrderingStrategy` enum + `compute_edge_ordering(lattice, strategy)` function
- BFS ordering: start at a vertex, order its incident edges, expand to neighbors
- Spectral ordering: use Fiedler vector from spectral catalog
- Backbone-first: sort by max(z_u, z_v) descending

**Start with BFS** — it maximizes connected context (each new edge shares a
vertex with an already-assigned edge).

### Phase B3: EdgeMPVAN Model

The autoregressive model itself.

**What to build:**
- `EdgeMPVAN(nn.Module)` with:
  - K masked EIGN layers (reusing `EIGNLayer` weights with causal masks)
  - Equivariant input: partial spin vector σ̃ (assigned = ±1, unassigned = 0)
  - Invariant input: static geometry features (z_u, z_v, ℓ_e) — always fully
    available, no masking needed
  - Output head: sigmoid on equivariant features at edge e_k → p(σ_{e_k} = +1)
- `sample()` method: sequential forward passes, one edge per step
- `forward_log_prob(σ)` method: teacher-forced computation of log q_θ(σ) for
  training (can be parallelized across edges using masked operators)

**Key design decision — teacher forcing:**
During training, the ground-truth configuration is known. Use teacher forcing:
at step k, the input contains the true values σ_{e₁}...σ_{e_{k-1}} (not the
model's own samples). This enables parallel computation of all n₁ conditionals
in a single forward pass through the masked EIGN layers, dramatically faster
than sequential sampling. This is the standard VAN training trick.

### Phase B4: Soft Ice-Rule Constraint + Finite-T Training

**What to build:**
- Modified loss function:
  ```
  L = F_θ + λ(T) · E_q[‖B₁σ‖²]
  ```
  where F_θ = E_q[H(σ)] + T·E_q[ln q_θ(σ)] is the variational free energy
- λ(T) schedule: λ ∝ 1/T (strong enforcement at low T, relaxed at high T)
- Temperature annealing: geometric schedule T(epoch) = T_max · (T_min/T_max)^{epoch/total}
- **Lattice-adapted calibration:** T_max ~ ε·Δ₁·n₁ (nearly uniform over
  low-energy states), T_min ≪ ε·Δ₁ (ground-state dominated). Use Δ₁ from
  spectral catalog.

**Validation:**
- At low T: ice-rule violation → 0, energy → 0
- At high T: samples approach uniform
- Specific heat C(T) = (⟨H²⟩ - ⟨H⟩²)/T² has correct peak location

---

## 5. What We Reuse from Mode A

Mode B shares extensive infrastructure with Mode A:

| Component | Source | Reuse Status |
|-----------|--------|-------------|
| EIGN operators (B₁ → torch sparse) | `src/neural/operators.py` | Direct reuse |
| EIGNLayer (forward pass) | `src/neural/eign_layer.py` | Wrap with masking |
| REINFORCE training loop | `src/neural/training.py` | Adapt for edge-level AR |
| Metrics (KL, Hamming, ESS, energy) | `src/neural/metrics.py` | Direct reuse + add violation rate |
| Checkpointing | `src/neural/checkpointing.py` | Direct reuse |
| Training plots (Panels 1-5) | `src/neural/training_plots.py` | Direct reuse + add violation panel |
| Lattice pipeline | `src/lattices/` | Direct reuse |
| MCMC baseline | `src/sampling/benchmark.py` | Direct reuse |
| Loop basis (for validation) | `src/neural/loop_basis.py` | Project Mode B samples onto loop basis |

**New modules needed:**
- `src/neural/causal_masking.py` — mask precomputation + `MaskedEIGNLayer`
- `src/neural/edge_ordering.py` — ordering strategies
- `src/neural/edge_mpvan.py` — `EdgeMPVAN` model (sampling + teacher forcing)

---

## 6. Experimental Progression

### Tier 0: Proof of Concept (Square XS, open BC)

- n₁ = 24, β₁ = 9, 25 reachable ice states
- Train at low T, verify ice-rule violation → 0
- Compare learned distribution to exact enumeration
- **This validates causal masking + edge ordering independently of scale**

### Tier 1: First Scaling Test (Square S / Kagome S)

- n₁ = 200-600, compare against MCMC baselines
- Test BFS vs spectral ordering
- Monitor gradient diagnostics (reuse C2 infrastructure)

### Tier 2: Finite-Temperature Physics

- Temperature sweep on Square S: compute C(T), locate crossover
- Compare to MCMC specific heat
- Monopole density vs T: ⟨Q²⟩(T) = ⟨‖B₁σ‖²/n₀⟩

### Tier 3: Frustrated Lattices

- Tetris M (n₁ = 4800), Shakti M (n₁ = 9600)
- Mixed coordination → backbone-first ordering may help
- Compare to Mode A at T = 0 (cross-validation)

---

## 7. Metrics to Add (Beyond Mode A)

| Metric | Formula | What It Measures |
|--------|---------|-----------------|
| Ice-rule violation rate | ‖B₁σ‖² / n₀ | Monopole density |
| Vertex charge histogram | histogram of Q_v = (B₁σ)_v | Charge distribution |
| Specific heat | C(T) = (⟨H²⟩ - ⟨H⟩²) / T² | Phase transition location |
| Monopole pair correlation | ⟨Q_u Q_v⟩ vs distance | String tension |
| Loop-basis projection | Project σ onto ker(B₁), measure residual | How close to ice manifold |

---

## 8. Open Questions and Risks

### Does causal masking degrade EIGN message passing?
Masking breaks operator symmetry — L₁_down becomes lower-triangular, losing
some neighbor information. BFS ordering mitigates this (most neighbors appear
before the current edge). Empirically: compare masked vs unmasked EIGN features
at convergence. If degradation is severe, consider bidirectional context
(reverse pass after forward pass, as in bidirectional RNNs).

### Sequence length scaling
Mode B's sequence length is n₁ (2-3× longer than Mode A's β₁). At Shakti M,
n₁ = 9600 — each sample requires 9600 sequential forward passes. Teacher
forcing parallelizes training, but sampling is inherently sequential. For
large systems, this may be the practical bottleneck.

**Mitigation options:**
- Blocked autoregression: assign groups of spatially distant edges simultaneously
- KV-caching: cache EIGN layer activations, only recompute for newly assigned edges
- Hierarchical: use Mode A for ice backbone, Mode B only for excitations
  (Conjecture 1 from proposal — hierarchical decomposition)

### λ(T) tuning
The soft ice-rule penalty weight λ(T) has no principled value. Too small → many
violations at low T. Too large → gradient signal from energy term overwhelmed.
Start with λ(T) = c/T where c is tuned on XS to achieve <1% violation rate
at the target T_min.

### Edge ordering sensitivity
The proposal lists 4 strategies. Systematic comparison needed on at least one
lattice before committing to a default. If ordering matters a lot, the
random-permutation (MADE-style) approach may be best for robustness, at the
cost of per-ordering quality.

---

## 9. Deferred Items

| Item | Why Deferred | Prerequisite |
|------|-------------|-------------|
| Hierarchical decomposition (Mode A + B) | Conjecture, needs Mode B working first | Tier 2 results |
| Magnetic Laplacian extension | Applied fields, symmetry breaking | Both modes validated |
| Cross-lattice transfer learning | Train on one lattice, test on another | Tier 3 results |
| CUDA kernel optimization | Not needed at current sizes (n₁ ≤ 9600) | Tier 3 scaling limits |
| Parallel tempering comparison | State-of-the-art MCMC baseline | Tier 2 C(T) results |

---

## 10. Summary: Mode A vs Mode B at a Glance

```
Mode A (Loop-MPVAN)          Mode B (Edge-MPVAN)
─────────────────────        ─────────────────────
Seed ice state required      No seed — build from scratch
Loop-flip decisions (β₁)    Edge spin decisions (n₁)
Unmasked EIGN operators     Causally masked EIGN operators
Ice rule: exact             Ice rule: soft penalty
T = 0 only                  Any temperature
Deaf Hamiltonian in L1      Hamiltonian active from L1
Shorter sequences           Longer sequences (2-3×)
Implemented ✓               To be implemented
```

**Bottom line:** Mode B is the standard VAN approach applied to edge spins with
EIGN message passing. It is conceptually simpler than Mode A (no loop basis,
no seed, no directed-cycle gating) but engineering-harder (causal masking,
longer sequences, temperature annealing). The deaf Hamiltonian is not a concern.
Mode A infrastructure carries forward almost entirely.
