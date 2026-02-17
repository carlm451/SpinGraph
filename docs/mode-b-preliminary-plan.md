# Mode B: Direct Edge Sampling — Preliminary Plan

> High-level implementation plan for the Edge-MPVAN direct edge sampler.
> Based on [Edge-MPVAN proposal](edge-mpvan-proposal.html), the VAN paper
> (Wu et al. 2019), and lessons learned from Mode A development.
>
> **Revised 2026-02-17** with empirical findings from Mode A Tier 0/1 ensemble
> experiments (see `docs/mode-a-walkthrough.md` §8 for the charge sector analysis
> that motivates this revision).

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
| **Charge sectors** | **Single sector only** | **All sectors** |
| **Full manifold** | **0.02–20% of ice states** | **100% accessible** |
| Monopole physics | Cannot study | Natural |
| Phase transitions | Cannot access | Can sweep C(T) |
| Sequence length | β₁ | n₁ (2-3× longer) |

**The charge sector problem is Mode B's primary motivation.** Mode A Tier 0/1
experiments (5-run ensembles on 7 lattice/BC combinations) revealed that loop
flips preserve vertex charge (B₁c = 0), confining Mode A to the seed state's
Coulomb class. On odd-degree lattices, this is a tiny fraction of the full ice
manifold — as low as 0.02% on tetris 2×2 open (17 of 86,560 states). Even on
even-degree lattices where only one charge sector exists, the autoregressive
ordering gap limits coverage (38 of 2,768 states on square 4×4 open). See
`docs/mode-a-walkthrough.md` §8 and `docs/tdl-spinice-correspondence.html` §5
for the full analysis.

Mode B builds configurations edge-by-edge without a seed, so it naturally
accesses all charge sectors and the full thermodynamic phase space.

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

**Empirical confirmation (Mode A experiment 0b):** Square 4×4 periodic failed
catastrophically (KL = 1.433, coverage = 39%, gradient norms *increasing* over
4000 epochs) precisely because the deaf Hamiltonian kills the equivariant
channel on periodic lattices where every intermediate state is ice-rule
satisfying. Mode B's partial assignments break this symmetry — the Hamiltonian
channel is live from step 1, so we expect Mode B to handle periodic lattices
without difficulty.

### 3c. Charge Monitor for Free

The equ→inv channel (|B₁|ᵀB₁) computes unsigned vertex charge accumulation.
During autoregressive generation, this provides the network with a **real-time
ice-rule violation detector** — it can "see" where charges are building up
as each spin is assigned and adjust subsequent decisions to compensate.

### 3d. Full Manifold Access via Edge-Level Autoregression

Mode A's confinement to a single charge sector is a topological constraint on
loop-flip dynamics, not a neural network limitation. Loop flips preserve vertex
charge by definition (B₁c = 0 for any cycle c), so no amount of training
improvement can fix it.

Mode B sidesteps this entirely because it assigns individual edge spins, not
loop flips. An edge flip *does* change vertex charge — flipping edge (u,v)
changes Q_u and Q_v by ±2. The autoregressive decomposition q(σ) = Π q(σ_e | σ_{<e})
places no topological restriction on which configurations can be reached.
At sufficiently low T, the soft ice-rule penalty drives Q_v → 0 at every
vertex, recovering ice states from *all* charge sectors simultaneously.

This is the key structural advantage that makes Mode B the path to full-manifold
sampling, even though Mode A remains superior for within-sector uniformity at T=0.

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

**Lesson from Mode A: randomize from the start.** Mode A's single-ordering
coverage was 17–50% of single-sector states; random permutation per batch
(XLNet-style) gave 1.5–5.8× improvement. For Mode B, use **MADE-style random
permutation per batch as the default**, with BFS as the fallback for inference.
This requires precomputing causal masks per batch (or using a single
lower-triangular mask in permuted edge index space). The causal mask
precomputation cost is negligible compared to the EIGN forward pass.

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

**Efficiency gain over Mode A:** In Mode A, both sampling AND teacher forcing
are sequential (β₁ forward passes each). Mode B sampling is still sequential
(n₁ passes), but teacher-forced log-prob computation is a single parallel
forward pass. Since REINFORCE requires: (1) sample batch → sequential, (2)
compute log_prob → parallel via teacher forcing, (3) gradient update, the
teacher forcing parallelism means Mode B's training cost per epoch is dominated
by the sampling step, not the gradient computation. This is a structural
advantage over Mode A's fully sequential training pipeline.

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

## 6. Lessons from Mode A Tier 0/1 Experiments

The following empirical findings from Mode A's 5-run ensemble tests directly
inform Mode B's design and validation strategy.

### 6a. Scorecard summary

| Exp | Lattice | BC | β₁ | Sector States | KL | Sector Cov | Grade |
|-----|---------|----|----|---------------|-----|------------|-------|
| 0a | Square 4×4 | open | 9 | 38 | 0.325 | 0.816 | MARGINAL |
| 0b | Square 4×4 | periodic | 17 | 299 | 1.433 | 0.387 | FAIL |
| 0c | Kagome 2×2 | periodic | 13 | 355 | 0.463 | 0.608 | MARGINAL |
| 1b | Kagome 2×2 | open | 6 | 34 | 0.042 | 1.000 | PASS |
| 1c | Santa Fe 2×2 | open | 7 | 23 | 0.286 | 1.000 | MARGINAL |
| 1d | Tetris 2×2 | open | 7 | 17 | 0.168 | 1.000 | MARGINAL |
| 1e | Shakti 1×1 | open | 2 | 3 | 0.002 | 1.000 | PASS |

"Sector States" = states reachable within one Coulomb class via multi-ordering
DFS. "Sector Cov" = fraction of sector states found. These are NOT full-manifold
numbers.

### 6b. Key takeaways for Mode B

1. **Charge sector confinement is the dominant limitation.** Mode A achieves
   100% sector coverage on 5/7 test cases, but each sector is a tiny fraction
   of the full manifold (e.g., tetris: 17/86,560 = 0.02%). Mode B must be
   validated against full-manifold state counts, not sector counts.

2. **Deaf Hamiltonian kills periodic lattices in Mode A.** Experiment 0b
   (square periodic) failed with increasing gradient norms — L_equ @ σ = 0
   for all ice states made the equivariant channel deaf. Mode B does not have
   this problem (partial assignments have Q ≠ 0), so periodic lattices should
   work from the start.

3. **Ordering randomization is essential, not optional.** Single-ordering DFS
   found 25 states on square 4×4 open; 200-ordering DFS found 38 (1.5×). On
   square 4×4 periodic: 40 → 299 (7.5×). Mode B should use random edge
   orderings per batch from day one.

4. **REINFORCE gradient noise is inherent.** Gradient norms plateau at 0.2–0.4
   and do not converge to zero — this is expected REINFORCE behavior, not a
   training failure. Advantage variance is the better diagnostic for gradient
   signal quality.

5. **Small lattices are easy; the challenge is scaling.** β₁ ≤ 7 with open BC
   all achieve 100% sector coverage. The real test is n₁ > 30 and periodic BC.
   Mode B's Tier 0 should target these harder cases where Mode A struggled.

6. **Training time scales with state space, not just sequence length.** Mode A:
   ~25s for 3 states (shakti 1×1), ~60s for 34 states (kagome 2×2), ~200s for
   355 states (kagome 2×2 periodic). Mode B will have longer sequences (n₁ vs
   β₁) but teacher forcing parallelizes the log-prob computation.

### 6c. Cross-validation targets from §5.8

Full ice manifold counts from exact backtracking enumeration provide ground
truth for Mode B validation:

| Lattice | BC | n₁ | |I| (full manifold) | Mode A sector | Mode B target |
|---------|-----|-----|--------------------|--------------:|:-------------:|
| Square 4×4 | open | 24 | 2,768 | 38 | 2,768 |
| Square 4×4 | periodic | 32 | 2,970 | 299 | 2,970 |
| Kagome 2×2 | open | 17 | 172 | 34 | 172 |
| Kagome 2×2 | periodic | 24 | 600 | 355 | 600 |
| Santa Fe 2×2 | open | 30 | 1,312 | 23 | 1,312 |
| Tetris 2×2 | open | 38 | 86,560 | 17 | 86,560 |
| Shakti 2×2 | open | 81 | >1.4M | 112 | >1.4M |

At low T, Mode B should produce zero-violation samples spanning all charge
sectors. Coverage of the full |I| is the validation metric — not just one
sector.

---

## 7. Experimental Progression

### Tier 0: Proof of Concept — Cross-Validate Against Mode A

Start with the same lattices Mode A was tested on, at low T (targeting ice
states). This validates causal masking, edge ordering, and soft ice-rule
enforcement before introducing finite-T physics.

**Tier 0a: Kagome 2×2 open (n₁ = 17, |I| = 172)**
- Smallest edge count in our test suite, fastest iteration
- Mode A found 34/172 states (one sector). Mode B target: all 172
- Success criterion: violation rate < 1%, coverage > 50% of full |I|
- Cross-validate: samples in Mode A's sector should have similar frequency

**Tier 0b: Square 4×4 open (n₁ = 24, |I| = 2,768)**
- Single charge sector (even degree), so Mode A's 38 states = ordering gap only
- Mode B should find ≫ 38 unique ice states (ideally approaching 2,768)
- Tests whether Mode B closes the ordering gap Mode A couldn't

**Tier 0c: Square 4×4 periodic (n₁ = 32, |I| = 2,970)**
- **Critical test:** Mode A FAILED here (deaf Hamiltonian). Mode B should succeed
- If Mode B passes where Mode A failed, this is the strongest validation of
  the architectural advantage
- Success criterion: KL < 1.0, coverage > 50%, violations < 1%

### Tier 1: Multi-Sector Validation on Frustrated Lattices

**Tier 1a: Santa Fe 2×2 open (n₁ = 30, |I| = 1,312)**
- Mode A found 23 states (one sector of a multi-sector manifold)
- Mode B should discover states in multiple charge sectors
- Validate by checking vertex charge histograms across samples

**Tier 1b: Tetris 2×2 open (n₁ = 38, |I| = 86,560)**
- Extreme case: Mode A saw 0.02% of the manifold (17/86,560)
- The tetris "sliding phase" creates massive degeneracy across sectors
- Even partial coverage (>1000 unique states) would be a major improvement

### Tier 2: Finite-Temperature Physics

- Temperature sweep on Square S (n₁ ~ 200): compute C(T), locate crossover
- Compare to MCMC specific heat
- Monopole density vs T: ⟨Q²⟩(T) = ⟨‖B₁σ‖²/n₀⟩
- Charge sector population vs T: which sectors dominate at each temperature?

### Tier 3: Frustrated Lattices at Scale

- Tetris M (n₁ = 4800), Shakti M (n₁ = 9600)
- Mixed coordination → backbone-first ordering may help
- Compare to Mode A at T = 0 within each sector (cross-validation)

---

## 8. Metrics to Add (Beyond Mode A)

| Metric | Formula | What It Measures |
|--------|---------|-----------------|
| Ice-rule violation rate | ‖B₁σ‖² / n₀ | Monopole density |
| **Full-manifold coverage** | |unique ice states| / |I| | Fraction of all ice states found (not just one sector) |
| **Charge sector histogram** | Count samples per Coulomb class | Whether model accesses multiple sectors |
| **Cross-sector KL** | KL(empirical sector dist \|\| uniform over sectors) | Uniformity across charge sectors |
| Vertex charge histogram | histogram of Q_v = (B₁σ)_v | Charge distribution |
| Specific heat | C(T) = (⟨H²⟩ - ⟨H⟩²) / T² | Phase transition location |
| Monopole pair correlation | ⟨Q_u Q_v⟩ vs distance | String tension |
| Loop-basis projection | Project σ onto ker(B₁), measure residual | How close to ice manifold |

**Charge sector identification:** Given a sample σ, compute Q = B₁σ. The charge
vector Q identifies the Coulomb class. Two samples are in the same sector iff
their charge vectors are identical. At low T (near ice manifold), Q should be
close to an ice-rule-satisfying charge pattern; the specific pattern of ±1
charges on odd-degree vertices determines the sector.

**Key difference from Mode A metrics:** Mode A's coverage metric was
|unique states| / |I_sector|. Mode B's primary coverage metric must be
|unique ice states| / |I| — the full manifold. We should also report per-sector
coverage for cross-validation against Mode A results.

---

## 9. Open Questions and Risks

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

**Mode A timing reference:** At comparable state-space sizes, Mode A training
took ~60s (34 states) to ~200s (355 states) per run on CPU. Mode B's per-sample
cost is higher (n₁ sequential steps vs β₁), but teacher forcing parallelizes
the gradient computation. Expect Tier 0 runs (n₁ ≤ 38) to complete in minutes,
not hours.

### λ(T) tuning
The soft ice-rule penalty weight λ(T) has no principled value. Too small → many
violations at low T. Too large → gradient signal from energy term overwhelmed.
Start with λ(T) = c/T where c is tuned on XS to achieve <1% violation rate
at the target T_min.

### Edge ordering sensitivity — partially resolved
Mode A's experience strongly favors random permutation per batch. The 4
strategies (BFS, spectral, backbone-first, random) should still be available
for inference, but **training should default to MADE-style random ordering**.
The remaining question is whether random ordering requires per-batch mask
recomputation (expensive) or can be handled via permuted indexing (cheap).

### REINFORCE gradient noise — resolved
Mode A experiments showed gradient norms plateauing at 0.2–0.4, which initially
appeared concerning but is **expected REINFORCE behavior**. The gradient
estimator ∇_θ E_q[f(σ) log q_θ(σ)] has inherent variance that does not vanish
at convergence. Advantage variance (tracked via C2 diagnostics) is the better
signal for training health. Do not waste time trying to eliminate gradient noise
— focus on loss and KL convergence instead.

---

## 10. Deferred Items

| Item | Why Deferred | Prerequisite |
|------|-------------|-------------|
| Hierarchical decomposition (Mode A + B) | Conjecture, needs Mode B working first | Tier 2 results |
| Magnetic Laplacian extension | Applied fields, symmetry breaking | Both modes validated |
| Cross-lattice transfer learning | Train on one lattice, test on another | Tier 3 results |
| CUDA kernel optimization | Not needed at current sizes (n₁ ≤ 9600) | Tier 3 scaling limits |
| Parallel tempering comparison | State-of-the-art MCMC baseline | Tier 2 C(T) results |

---

## 11. Summary: Mode A vs Mode B at a Glance

```
Mode A (Loop-MPVAN)          Mode B (Edge-MPVAN)
─────────────────────        ─────────────────────
Seed ice state required      No seed — build from scratch
Loop-flip decisions (β₁)    Edge spin decisions (n₁)
Unmasked EIGN operators     Causally masked EIGN operators
Ice rule: exact             Ice rule: soft penalty
T = 0 only                  Any temperature
Single charge sector        All charge sectors
0.02–20% of manifold        Full manifold accessible
Deaf Hamiltonian in L1      Hamiltonian active from L1
Shorter sequences           Longer sequences (2-3×)
Sequential teacher forcing  Parallel teacher forcing
Implemented ✓               To be implemented
```

**Bottom line:** Mode B is the standard VAN approach applied to edge spins with
EIGN message passing. It is conceptually simpler than Mode A (no loop basis,
no seed, no directed-cycle gating) but engineering-harder (causal masking,
longer sequences, temperature annealing). The deaf Hamiltonian is not a concern
(empirically confirmed by Mode A's 0b failure on periodic lattices). The charge
sector confinement discovered in Mode A experiments is the primary motivation:
Mode B is the only path to full-manifold sampling across all Coulomb classes.
Mode A infrastructure carries forward almost entirely.
