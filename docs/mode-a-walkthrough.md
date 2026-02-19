# Mode A (LoopMPVAN) — Architecture & Training Walkthrough

A step-by-step guide to how the autoregressive loop-flip sampler works, from data preparation through sampling and training. Each section references exact lines in the source code.

---

## 1. Problem Setup

**Goal:** Learn to sample uniformly from ice states reachable by loop flips from a seed configuration. Because loop flips preserve vertex charge (see Section 8), this samples uniformly within a single **charge sector** of the ice manifold, not the full manifold.

**Key insight:** The ice manifold is a discrete set connected by *directed loop flips* within each charge sector. Starting from any valid ice state (the "seed"), every other ice state in the same charge sector can be reached by flipping some subset of β₁ independent cycles. This means we can parameterize the sector as β₁ binary decisions (flip or don't flip each loop), and train an autoregressive model over these decisions.

---

## 2. Data Preparation (before training)

### 2a. Build the lattice

The lattice generator produces vertex positions, edge list, and coordination numbers.

> `scripts/train_lattice.py:95-98` — build lattice from registry
> `src/lattices/base.py:40-54` — `LatticeResult` dataclass with graph, positions, edge_list, coordination

### 2b. Build the incidence matrix B₁

B₁ is the (n₀ × n₁) signed vertex-edge incidence matrix. Convention: for edge e = (u,v) with u < v, B₁[u,e] = -1 (tail), B₁[v,e] = +1 (head).

> `src/topology/incidence.py` — `build_B1(n_vertices, edge_list)` → scipy sparse CSC

### 2c. Find a seed ice state

Constructs a valid ice state σ_seed where every vertex satisfies the ice rule: |Q_v| = z_v mod 2. Uses Eulerian circuit construction for even-degree subgraphs and greedy repair for odd-degree vertices.

> `scripts/train_lattice.py:101-103` — find and verify seed
> `src/topology/ice_sampling.py` — `find_seed_ice_state(B1, coordination, edge_list)`

### 2d. Build EIGN operators

Four sparse matrix products derived from B₁, precomputed once:

| Operator | Formula | Shape | Physics meaning |
|----------|---------|-------|-----------------|
| L_equ | B₁ᵀ B₁ | (n₁, n₁) | Lower Hodge Laplacian = ASI Hamiltonian |
| L_inv | \|B₁\|ᵀ \|B₁\| | (n₁, n₁) | Unsigned geometry (coordination structure) |
| equ_to_inv | \|B₁\|ᵀ B₁ | (n₁, n₁) | Charge monitor: spins → vertex charge info |
| inv_to_equ | B₁ᵀ \|B₁\| | (n₁, n₁) | Geometry modulates spin signals |

> `src/neural/operators.py:63-103` — `build_eign_operators(B1_scipy)` computes all four in scipy, converts to PyTorch sparse COO
> `src/neural/operators.py:87-90` — the four matrix products
> `src/neural/operators.py:24-34` — `EIGNOperators` dataclass

### 2e. Extract loop basis

Finds β₁ = n₁ - rank(B₁) independent cycles using `networkx.cycle_basis()`. Each cycle is oriented so that B₁ · c = 0 (divergence-free), producing a signed edge vector.

> `src/neural/loop_basis.py:159-224` — `extract_loop_basis(G, B1, edge_list)`
> `src/neural/loop_basis.py:185` — `nx.cycle_basis(G)` returns vertex cycles
> `src/neural/loop_basis.py:50-100` — `orient_cycle()` produces signed vector + binary indicator + edge list
> `src/neural/loop_basis.py:27-36` — `LoopBasis` dataclass: `loop_indicators` (β₁ × n₁ binary), `loop_oriented` (β₁ × n₁ signed), `cycle_edge_lists`, `ordering`

### 2f. Compute loop ordering

The autoregressive ordering determines which loop is decided first, second, etc. A default ordering is computed using the `spatial_bfs` strategy: build an overlap graph of loops (loops that share edges are adjacent), start from the most central loop, and BFS outward.

**Important:** During training, this default ordering is overridden by a random permutation generated fresh for each batch (see Section 5b). This "ordering randomization" is analogous to XLNet's permutation training and dramatically expands the set of reachable states — a fixed ordering can miss 30-99% of states on periodic lattices due to the autoregressive ordering gap (see `docs/tdl-spinice-correspondence.html` §8.8). The default ordering is still used when `--no-randomize-ordering` is passed, or as a fallback for inference without explicit ordering.

> `src/neural/loop_basis.py:227-310` — `compute_loop_ordering(loop_basis, strategy, positions, edge_list)`
> `src/neural/loop_basis.py:261-307` — spatial_bfs: overlap graph → BFS from center

### 2g. Exact enumeration (small systems only)

For β₁ ≤ 25, DFS through the autoregressive decision tree. At each node, if the current loop is a directed cycle, both branches (flip / no-flip) are explored. If not directed, only no-flip. This prunes the 2^β₁ tree dramatically.

**Multi-ordering enumeration:** Because the DFS tree depends on the loop ordering, a single ordering may miss reachable states (the "autoregressive ordering gap"). When ordering randomization is enabled, `enumerate_multi_ordering()` runs DFS with K random orderings (default 200) and unions all discovered states. This gives a tighter lower bound on the reachable ice manifold. For example, square 4×4 periodic discovers 52 states with a fixed ordering but 299 across 200 random orderings (5.8× more).

> `src/neural/enumeration.py:28-96` — `enumerate_reachable_ice_states()` DFS
> `src/neural/enumeration.py:89-93` — the branching logic (no-flip always; flip only if directed)
> `src/neural/enumeration.py:149-225` — `enumerate_multi_ordering()` multi-ordering union

---

## 3. Network Architecture

### 3a. Input projections

At each autoregressive step, the current spin configuration σ (n₁ values of +1/-1) is the equivariant input. The invariant input is the normalized endpoint coordinations (n₁ × 2).

> `src/neural/loop_mpvan.py:124-125` — `equ_input`: Linear(1 → equ_dim), `inv_input`: Linear(2 → inv_dim)
> `src/neural/loop_mpvan.py:142-143` — σ.unsqueeze(-1) → (n₁, 1) → Linear → (n₁, equ_dim)

### 3b. EIGN stack (K shared layers)

K identical EIGN layers process the full lattice. Each layer has 6 learnable weight matrices and performs dual-channel message passing:

**Equivariant update:**
```
X_equ^(ℓ+1) = GELU(LayerNorm(L_equ·X_equ·W1 + inv_to_equ·X_inv·W2 + X_equ·W5))
```

**Invariant update:**
```
X_inv^(ℓ+1) = GELU(LayerNorm(L_inv·X_inv·W3 + equ_to_inv·X_equ·W4 + X_inv·W6))
```

The sparse operator multiply (e.g. L_equ · X_equ) is the message-passing step — it aggregates features from neighboring edges via the Laplacian. The linear transforms (W1-W4) are learned projections. W5, W6 are skip connections initialized near identity.

> `src/neural/eign_layer.py:23-145` — full `EIGNLayer` class
> `src/neural/eign_layer.py:51-58` — 6 weight matrices
> `src/neural/eign_layer.py:104-114` — "deaf Hamiltonian" note (see below)
> `src/neural/eign_layer.py:118-131` — 4 message-passing terms (sparse @ dense @ linear)
> `src/neural/eign_layer.py:134-135` — skip connections
> `src/neural/eign_layer.py:138-143` — combine, LayerNorm, GELU
> `src/neural/eign_layer.py:74-85` — weight init: Xavier for MP, near-identity for skip

**Deaf Hamiltonian (C1):** For ice states σ, `L_equ @ σ = B₁ᵀB₁σ = B₁ᵀQ = 0` since Q = B₁σ = 0 (ice rule). This means the equ→equ (W1) and equ→inv (W4) channels receive zero input in layer 1 when X_equ = σ. Training still works because: (1) skip connection W5 passes σ through unchanged, (2) GELU makes layer-1 output nonlinear, (3) from layer 2 onward L_equ operates on GELU output which is non-zero, and (4) the inv→inv and inv→equ channels are active in all layers.

**Critical design point:** No causal masking. Unlike Mode B (planned), the EIGN stack sees the **fully-assigned** spin configuration at every autoregressive step. This works because each step operates on a complete valid ice state — the autoregression is over loop-flip decisions, not individual edge assignments.

> `src/neural/loop_mpvan.py:128-131` — shared layer stack
> `src/neural/loop_mpvan.py:136-148` — `_build_features()`: input projection + full EIGN stack

### 3c. Output head (LoopOutputHead)

After the EIGN stack produces per-edge features (X_equ, X_inv), the output head computes a single flip probability p_i for the current loop:

1. **Concatenate** equivariant + invariant features: (n₁, equ_dim + inv_dim)
2. **Pool** over loop edges only (masked mean): sum features on loop edges / num loop edges
3. **MLP** (3-layer: Linear → GELU → Linear → GELU → Linear → sigmoid)
4. **Output:** scalar p_i ∈ (0, 1) — probability of flipping this loop

> `src/neural/loop_mpvan.py:37-74` — `LoopOutputHead`
> `src/neural/loop_mpvan.py:69-72` — concat, mask, pool
> `src/neural/loop_mpvan.py:43-49` — MLP architecture
> `src/neural/loop_mpvan.py:73-74` — logit → sigmoid → p_i

---

## 4. Autoregressive Sampling

For each sample, the model processes loops one at a time in a given ordering. The ordering is either a random permutation (during training) or a specified/default ordering (during inference). Starting from σ_seed:

```
For each loop i in ordering:                  # ordering is a permutation of [0, β₁)
    1. Is loop i a directed cycle in current σ?
       - Check: at every vertex on the cycle, one cycle edge flows in, one flows out
       - If NO: skip this loop (α_i = 0, no contribution to log_prob)
       - If YES: continue to step 2

    2. Run full EIGN stack on current σ → (X_equ, X_inv)

    3. Pool features over loop i's edges → MLP → sigmoid → p_i

    4. Sample: α_i ~ Bernoulli(p_i)

    5. Accumulate: log_prob += α_i·log(p_i) + (1-α_i)·log(1-p_i)

    6. If α_i = 1: flip loop i → σ = σ ⊕ loop_i
       (negate all spins on loop edges)

Return: final σ (guaranteed valid ice state), log_prob
```

The `ordering` parameter is accepted by all sampling and log-prob methods. When `None`, it defaults to the spatial_bfs ordering stored in `loop_basis.ordering`. During training, a fresh random permutation is generated per batch (see Section 5b).

> `src/neural/loop_mpvan.py:232-291` — `sample()` method
> `src/neural/loop_mpvan.py:254-255` — ordering defaults to `self.loop_basis.ordering`
> `src/neural/loop_mpvan.py:265-271` — directed-cycle check and skip
> `src/neural/loop_mpvan.py:412-483` — `sample_batch()`: batched version with ordering param
> `src/neural/loop_basis.py:313-332` — `flip_single_loop()`: mask → sign_flip → σ * sign_flip
> `src/neural/loop_basis.py:103-143` — `is_directed_cycle()`: check flow at each cycle vertex

**Why this guarantees valid ice states:** The seed state satisfies the ice rule. Flipping a directed cycle preserves the ice rule at every vertex (one-in-one-out on the cycle means the net charge doesn't change). Skipping non-directed cycles prevents invalid flips. Therefore every output σ is a valid ice state.

---

## 5. Training Loop (REINFORCE)

### 5a. Objective

At T = 0, all ice states have energy H = 0 (they're ground states). The variational free energy reduces to:

```
F_θ = T · E_q[ln q_θ(σ)] = -T · H(q_θ)
```

Minimizing F_θ = maximizing entropy H(q_θ) = learning to sample **uniformly**.

**Why train at all if every sample is already valid?** Zero ice-rule violations are guaranteed by construction (Section 4) — an untrained model produces 100% valid ice states. What training achieves is *uniform frequency* across those valid states. An untrained model might sample state #3 fifty times and state #17 never. The REINFORCE objective pushes the model toward equal probability for every reachable state. This matters because thermodynamic observables (correlation functions, structure factors) are averages over the ice manifold weighted equally at T=0. Biased sampling gives wrong expectation values. KL(empirical || uniform) measures exactly this: how far the sampling distribution is from the target uniform distribution.

### 5b. One epoch step-by-step

> `src/neural/training.py:146-268` — main training loop

**Step 0: Generate batch ordering**

If ordering randomization is enabled (default), generate a fresh random permutation of the β₁ loop indices for this batch. This is the key mechanism for closing the autoregressive ordering gap — each batch explores different parts of the ice manifold by processing loops in a different sequence.

```python
if config.randomize_ordering:
    batch_ordering = np.random.permutation(n_loops).tolist()
else:
    batch_ordering = None  # uses model's default ordering
```

> `src/neural/training.py:150-154` — per-batch ordering generation

**Step 1: Sample a batch (no gradients)**

Generate B ice states by running the autoregressive sampler B times, using the batch ordering.

> `src/neural/training.py:156-161` — `model.sample_batch(seed, inv_features, n_samples=batch_size, ordering=batch_ordering)` under `torch.no_grad()`

**Step 2: Recover α vectors**

Each sampled σ came from some sequence of flip decisions α = (α₁, ..., α_{β₁}). But sampling was done without gradients. To get differentiable log_probs, we need to recover which α produced each σ. This is a GF(2) linear algebra problem:

```
diff[e] = 1 if σ[e] ≠ σ_seed[e], else 0
Solve: L^T · α = diff  (mod 2)    where L = loop_indicators
```

Note: α recovery is ordering-independent — it recovers the same α regardless of what ordering was used during sampling.

> `src/neural/training.py:163-164` — `recover_alpha(sigmas, seed_tensor, indicators)`
> `src/neural/loop_basis.py:410-451` — `recover_alpha()`: compute diff, call GF(2) solver per sample
> `src/neural/loop_basis.py:453-481` — `_solve_gf2()`: Gaussian elimination over GF(2)

**Step 3: Teacher-forced log_prob (with gradients)**

Re-run the autoregressive model with the *known* α sequence (teacher forcing), using the **same batch ordering** as Step 1. This is critical — the log probability depends on the ordering because directedness checks happen in sequence. Using a different ordering would compute incorrect log_probs.

For each sample b:
- Walk through loops in the batch ordering
- At each directed loop: compute p_i, accumulate log q = Σ [α_i·log(p_i) + (1-α_i)·log(1-p_i)]
- At each non-directed loop: skip (contributes 0)

> `src/neural/training.py:166-171` — `model.forward_log_prob_batch(alphas, seed, inv_features, ordering=batch_ordering)`
> `src/neural/loop_mpvan.py:342-410` — `forward_log_prob_batch()`: batched teacher forcing with ordering param
> `src/neural/loop_mpvan.py:178-230` — `forward_log_prob()`: single-sample version

**Step 4: REINFORCE gradient**

```python
rewards     = -log_probs.detach()           # lower log_prob = higher entropy = better
if baseline is None:                         # C5 fix: init from first batch
    baseline = mean(rewards)
else:
    baseline = 0.99 * baseline + 0.01 * mean(rewards)   # running mean
advantages  = rewards - baseline            # centered rewards
policy_loss = -mean(advantages * log_probs) # REINFORCE estimator
```

The baseline is initialized to `None` and set from the first batch's mean reward (C5 fix). This avoids the cold-start problem where `baseline = 0` causes a large initial spike in advantages and loss.

The key identity: ∇_θ E_q[f(σ)] = E_q[f(σ) · ∇_θ log q_θ(σ)]. Here f(σ) = -log q_θ(σ) (the reward), so the gradient pushes the model to increase probability of high-entropy samples.

> `src/neural/training.py:162-175` — REINFORCE: rewards, baseline init/update, advantages, policy_loss

**Step 5: Entropy bonus**

An explicit entropy bonus `-0.01 * mean(-log_probs)` is added to the loss. This directly encourages high-entropy distributions and prevents mode collapse early in training.

```python
entropy = -log_probs.mean()
loss = policy_loss - entropy_bonus * entropy
```

> `src/neural/training.py:177-179` — entropy bonus and final loss

**Step 6: Backprop, gradient diagnostics, and update**

```python
loss.backward()
clip_grad_norm_(model.parameters(), 1.0)
# C2: Record gradient diagnostics (post-clip)
total_norm = sqrt(sum(p.grad.norm(2)^2 for p in parameters))
grad_norm_history.append(total_norm)
advantage_var_history.append(var(advantages))
optimizer.step()
scheduler.step()
```

After gradient clipping, the total gradient norm and advantage variance are recorded each epoch (C2 gradient diagnostics). These are saved to `metrics.npz` and plotted in Panel 5 of the diagnostic plots. High advantage variance signals high REINFORCE gradient variance, which may require larger batch sizes.

> `src/neural/training.py:181-197` — backward, clip, gradient diagnostics, step, schedule

### 5c. Evaluation checkpoints

Every `eval_every` epochs, generate fresh samples and compute metrics. When ordering randomization is enabled, eval samples also use a fresh random ordering to measure coverage across the full manifold.

- **Ice violations:** should always be 0.0 (Mode A guarantee)
- **Mean Hamming distance:** pairwise distance between samples (~0.5 for uniform)
- **ESS:** effective sample size from importance weights (higher = more diverse)
- **KL divergence:** KL(empirical || uniform) if exact states available
- **Gradient norm:** post-clip gradient norm (logged with eval metrics)
- **Advantage variance:** variance of REINFORCE advantages (logged with eval metrics)

> `src/neural/training.py:213-255` — evaluation block
> `src/neural/training.py:217-221` — random ordering for eval sampling
> `src/neural/metrics.py` — all metric implementations

---

## 6. Putting It All Together

The complete data flow for one training epoch:

```
σ_seed ──────────────────────────────────────────────────────────────┐
                                                                     │
┌── ORDERING (per batch) ────────────────────────────────────────┐   │
│  batch_ordering = random_permutation([0, β₁))                  │   │
│  (closes autoregressive gap — each batch explores differently)  │   │
└── batch_ordering ──────────────────────────────────────────────┘   │
         │                                                            │
         ▼                                                            │
┌── SAMPLING (no grad) ──────────────────────────────────────────┐   │
│                                                                 │   │
│  For each of B samples:                                         │   │
│    σ = σ_seed                                                   │   │
│    For each loop i in batch_ordering:                           │   │
│      if is_directed_cycle(σ, loop_i):                           │   │
│        σ → [equ_input] → [EIGN×K] → [pool loop_i] → [MLP] → p_i│  │
│        α_i ~ Bernoulli(p_i)                                     │   │
│        if α_i: σ = flip(σ, loop_i)                              │   │
│    collect σ_b                                                   │   │
│                                                                 │   │
└── B samples: {σ_1, ..., σ_B} ─────────────────────────────────┘   │
         │                                                            │
         ▼                                                            │
┌── ALPHA RECOVERY (ordering-independent) ───────────────────────┐   │
│  For each σ_b:                                                  │   │
│    diff = (σ_b ≠ σ_seed)                                        │   │
│    Solve L^T · α_b = diff over GF(2)                            │   │
└── {α_1, ..., α_B} ────────────────────────────────────────────┘   │
         │                                                            │
         ▼                                                            │
┌── TEACHER FORCING (with grad, SAME batch_ordering) ────────────┐   │
│  For each α_b:                                                  │   │
│    σ = σ_seed                                                   │◄──┘
│    log_q = 0                                                    │
│    For each loop i in batch_ordering:   ◄── must match sampling │
│      if is_directed_cycle(σ, loop_i):                           │
│        σ → [equ_input] → [EIGN×K] → [pool loop_i] → [MLP] → p_i│
│        log_q += α_i·log(p_i) + (1-α_i)·log(1-p_i)  ◄── GRAD   │
│        if α_i: σ = flip(σ, loop_i)                              │
│    collect log_q_b                                              │
└── {log_q_1, ..., log_q_B} ────────────────────────────────────┘
         │
         ▼
┌── REINFORCE UPDATE ────────────────────────────────────────────┐
│  rewards    = -log_q.detach()                                   │
│  if baseline is None:  baseline = mean(rewards)   # C5: 1st batch│
│  else: baseline = 0.99·baseline + 0.01·mean(rewards)           │
│  advantages = rewards - baseline                                │
│  loss       = -mean(advantages · log_q) - 0.01·mean(-log_q)    │
│  loss.backward() → clip_grad_norm                               │
│  record grad_norm, advantage_var  ◄── C2 diagnostics            │
│  Adam step → cosine LR                                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Key Design Decisions

### Why loop-basis instead of edge-by-edge?

- **Sequence length:** β₁ steps instead of n₁ steps. For square: β₁/n₁ ≈ 37% (9/24). For shakti: 22% (18/81).
- **Guaranteed validity:** Every sample is a valid ice state. No rejection, no soft penalty.
- **No causal masking:** Full lattice view at every step (all edges assigned).

### Why REINFORCE instead of maximum likelihood?

- We don't have target samples from the uniform distribution.
- REINFORCE lets us optimize an objective (entropy) using only samples from our own model.
- The reward signal (-log q) directly measures how uniform the distribution is.

### Why recover α via GF(2)?

- Sampling uses `torch.no_grad()` (fast, no graph).
- Teacher forcing needs gradients, so we re-run with known α.
- The GF(2) solver recovers the unique α that produces each σ from σ_seed.
- This decouples the fast sampling step from the gradient computation.

### Why directed-cycle checking?

- Not every cycle in the basis is flippable in every configuration.
- A cycle can only be flipped if it's a *directed* cycle: at each cycle vertex, one cycle edge points in and one points out.
- This constraint means the reachable state space is often much smaller than 2^β₁ (e.g., tetris 3×3: only 4 out of 2^22 ≈ 4M).
- The model learns to assign p ≈ 0.5 for flippable loops and never sees non-flippable ones.

### Why randomize the loop ordering per batch?

- **The autoregressive ordering gap:** With a fixed loop ordering, the DFS decision tree can only reach a subset of all valid ice states. Whether a loop is directed depends on all prior flip decisions — a different ordering changes the set of states reachable through the directed-cycle gates. On square 4×4 periodic, a single fixed ordering discovers only 52 of the 299 reachable states (17%).
- **XLNet-style permutation training:** Each batch uses a fresh random permutation of the β₁ loop indices. Over many training batches, the model is exposed to many different orderings, learning to handle any permutation. This is analogous to XLNet's permutation language modeling.
- **Empirical impact:** Multi-ordering enumeration (200 orderings) discovers 1.5-5.8× more states than a single ordering. Training with randomized ordering achieves 2-3× more unique states in final evaluation.
- **Consistency requirement:** The same ordering must be used for both sampling (Step 1) and teacher forcing (Step 3) within each batch. The α recovery (Step 2) is ordering-independent.
- Controlled via `TrainingConfig.randomize_ordering` (default `True`) and the `--no-randomize-ordering` CLI flag.

---

## 8. Fundamental Limitation: Charge Sector Confinement

Mode A samples uniformly *within a single charge sector* of the ice manifold. It cannot sample the full manifold. This is a mathematical constraint, not a training limitation.

### 8a. Why loop flips preserve charge

A loop flip applies a divergence-free perturbation to the spin configuration: if c is the oriented cycle vector, then B₁c = 0 by definition of a cycle. The vertex charge is Q = B₁σ, so after flipping:

```
Q_new = B₁(σ + 2c) = B₁σ + 2B₁c = Q + 0 = Q
```

Every loop flip — and therefore every autoregressive decision Mode A makes — preserves the vertex charge at every vertex. The model is confined to the **Coulomb class** of the seed state σ_seed.

### 8b. Charge sectors on different lattice types

**Even-degree lattices** (square, all z = 4): The ice rule forces Q_v = 0 at every vertex. There is only one charge sector (the zero-charge sector), so all ice states are reachable by loop flips in principle. The practical limitation here is the *autoregressive ordering gap*, not the charge sector.

**Odd-degree lattices** (kagome z=3, and mixed-coordination lattices like shakti, tetris, santa_fe): The ice rule allows Q_v = ±1 at odd-degree vertices. Different assignments of these ±1 charges define distinct charge sectors. Loop flips cannot change any vertex charge, so the ice manifold is partitioned into multiple disconnected sectors that Mode A cannot bridge.

### 8c. Impact on coverage

The "100% coverage" reported in Mode A experiments means 100% of states reachable *within the seed's charge sector*, not 100% of the full ice manifold. The table below shows how the sector relates to the full manifold for our test lattices (from `docs/tdl-spinice-correspondence.html` §5.8):

| Lattice | BC | |I_sector| (Mode A sees) | |I| (full manifold) | Fraction |
|---------|-----|------------------------|---------------------|----------|
| Square 4×4 | open | 38 | 2,768 | 1.4% |
| Square 4×4 | periodic | 299 | 2,768 | 10.8% |
| Kagome 2×2 | open | 1 | 18 | 5.6% |
| Kagome 2×2 | periodic | 2 | 450 | 0.4% |
| Santa Fe 2×2 | open | 23 | 23+ | ~100%* |
| Tetris 2×2 | open | 17 | 86,560 | 0.02% |
| Shakti 1×1 | open | 3 | 3+ | ~100%* |

*For Santa Fe and Shakti at these tiny sizes, exact full-manifold counts are pending; the sector may coincide with the full manifold at minimal system size.

For larger or more frustrated lattices, Mode A accesses only a tiny fraction of the full ice manifold. This is not a failure of the neural network — it is a topological constraint on loop-flip dynamics.

### 8d. Implications and path forward

Mode A is best understood as a **within-sector sampler**: given a charge sector (defined by the seed state), it learns to sample uniformly over that sector's ice states. This is useful for:

- Computing sector-conditioned observables
- Studying the structure of individual Coulomb classes
- Validating the loop-flip dynamics and EIGN architecture

To sample across the *full* ice manifold — including all charge sectors — requires moves that change vertex charges. This is the motivation for **Mode B** (direct edge sampling), which autoregressively assigns individual edge spins and can naturally access all charge sectors. Mode B trades Mode A's guaranteed validity for generality: it uses a soft ice-rule penalty rather than hard enforcement, but can reach any spin configuration including those in different charge sectors.

See `docs/tdl-spinice-correspondence.html` §5 for the full mathematical treatment of Coulomb classes and their connection to the Hodge decomposition.

---

## 9. Inference: Using the Trained Model

After training, the saved checkpoint contains everything needed to generate new ice state samples. This section describes how to load a trained model and use it for inference.

### 9a. What's saved in a training run

Each run is stored in `results/neural_training/{run_id}/` with these artifacts:

| File | Contents | Needed for inference? |
|------|----------|----------------------|
| `config.json` | Hyperparameters (n_layers, equ_dim, inv_dim, head_hidden, lattice, boundary, etc.) | Yes — to reconstruct model |
| `model_final.pt` | `state_dict()` of the trained LoopMPVAN | Yes — the learned weights |
| `seed_state.npy` | σ_seed (n₁,) array of +1/-1 | Yes — starting point for sampling |
| `positions.npy` | Vertex positions (n₀, 2) | Yes — for loop ordering |
| `edge_list.npy` | Edge pairs (n₁, 2) | Yes — to rebuild B₁ and operators |
| `coordination.npy` | Vertex coordinations (n₀,) | Yes — for invariant features |
| `metrics.npz` | Training curves (loss, KL, Hamming, ESS, grad norms) | No — diagnostic only |
| `final_samples.npy` | Samples from final evaluation | No — can regenerate |
| `final_log_probs.npy` | Log-probs of final samples | No — can regenerate |
| `exact_states.npy` | All enumerated ice states (if β₁ ≤ 25) | No — validation only |

> `src/neural/checkpointing.py:60-138` — `save_training_run()`
> `src/neural/checkpointing.py:141-190` — `load_training_run()`

### 9b. Reconstruction procedure

To generate new samples from a saved run, rebuild the model from the checkpoint:

```
1. Load config.json → extract lattice name, boundary, nx, ny, model hyperparams
2. Rebuild lattice → gen.build(nx, ny, boundary)
3. Build B₁ from edge_list (or rebuild from lattice)
4. Build EIGN operators from B₁
5. Extract loop basis + compute ordering (spatial_bfs)
6. Construct LoopMPVAN with saved hyperparams (n_layers, equ_dim, inv_dim, head_hidden)
7. Load state_dict from model_final.pt
8. Load seed_state.npy → σ_seed tensor
9. Build invariant features from edge_list + coordination
```

Steps 2–5 are deterministic given the lattice parameters, so the loop basis will be identical to the one used during training. With ordering randomization (default), the model is trained on all orderings, so inference can use any ordering — the default spatial_bfs ordering or random permutations for better coverage.

> `scripts/train_lattice.py:97-121` — the same setup pipeline used before training
> `src/neural/training.py` — `build_inv_features(edge_list, coordination)` for step 9

### 9c. Generating samples

Once the model is reconstructed:

```python
model.eval()
seed_tensor = torch.from_numpy(sigma_seed.astype(np.float32))
inv_features = build_inv_features(edge_list, coordination)

with torch.no_grad():
    # Single ordering (fast, uses default spatial_bfs ordering):
    sigmas, log_probs = model.sample_batch(seed_tensor, inv_features, n_samples=N)

    # Multi-ordering (better coverage, especially for periodic lattices):
    n_orderings = 20
    samples_per = N // n_orderings
    all_sigmas, all_lp = [], []
    for _ in range(n_orderings):
        ordering = np.random.permutation(loop_basis.n_loops).tolist()
        s, lp = model.sample_batch(seed_tensor, inv_features,
                                    n_samples=samples_per, ordering=ordering)
        all_sigmas.append(s); all_lp.append(lp)
    sigmas = torch.cat(all_sigmas)
    log_probs = torch.cat(all_lp)
```

This returns:
- `sigmas`: (N, n₁) tensor of ice state configurations, each ±1
- `log_probs`: (N,) tensor of log q_θ(σ) for each sample

Every sample is a valid ice state by construction. The samples are **independent** (no autocorrelation, unlike MCMC) — each one is a fresh forward pass through the autoregressive loop. Multi-ordering sampling at inference improves state coverage by exploring different parts of the ice manifold (the same mechanism that helps during training).

> `src/neural/loop_mpvan.py:412-483` — `sample_batch()` method with ordering parameter
> `src/neural/loop_mpvan.py:232-291` — `sample()` single-sample version

### 9d. What you can compute from samples

**Sample quality metrics** (same as training evaluation):

```python
from src.neural.metrics import (
    mean_hamming_distance, effective_sample_size,
    kl_from_samples, batch_ice_rule_violation, energy,
)

# Diversity: should be ~0.5 for uniform sampling
h_mean, h_std = mean_hamming_distance(sigmas.numpy())

# Effective sample size: higher = more uniform
ess = effective_sample_size(log_probs.numpy())

# KL divergence (only if exact_states available, β₁ ≤ 25)
kl = kl_from_samples(sigmas.numpy(), exact_states)

# Sanity checks (should always be 0.0 for Mode A)
violation = batch_ice_rule_violation(sigmas.numpy(), B1, coordination)
```

**Importance-weighted observables:** Because log q_θ(σ) is known for each sample, you can compute importance-weighted estimates of any observable O(σ) under the true uniform distribution:

```
w_i = 1 / q_θ(σ_i)                    (uniform target)
⟨O⟩_uniform ≈ Σ w_i O(σ_i) / Σ w_i
```

This corrects for any remaining non-uniformity in the learned distribution. The ESS measures how effective this correction is — when ESS ≈ N, the model is near-uniform and importance weights are approximately equal.

> `src/neural/metrics.py:128-163` — `effective_sample_size(log_q, log_p)`

### 9e. Sampling cost

Each sample requires β₁ sequential passes through the EIGN stack (one per loop in the ordering). At each step:
1. **Directed-cycle check** — O(loop_length) vertex inspections
2. **EIGN forward pass** — K sparse-dense matrix multiplies over all n₁ edges
3. **Output head** — pool over loop edges + 3-layer MLP → scalar

Non-directed loops are skipped (no EIGN pass), so the effective number of forward passes is typically less than β₁.

**Benchmark (square XS, 24 edges, β₁ = 9):** 2000 samples in ~6 seconds on CPU.

The key advantage over MCMC: samples are **independent by construction**. MCMC requires τ_corr flip sweeps between independent samples, where τ_corr grows with system size. The neural sampler's cost per independent sample is fixed at one forward pass, regardless of correlation structure.

### 9f. Diagnostic plots from saved runs

The plotting pipeline generates all diagnostic panels from saved artifacts without reloading the model:

```bash
python -m scripts.plot_training_diagnostics results/neural_training/{run_id}
```

This reads `metrics.npz`, `final_samples.npy`, `seed_state.npy`, `positions.npy`, `edge_list.npy`, and `coordination.npy` to produce:
- Panel 1: Training loss curve
- Panel 2: Sampling quality (KL, Hamming, ESS over training)
- Panel 3: Sample gallery (spin arrows + monopole markers on lattice)
- Panel 4: Summary card (key metrics at a glance)
- Panel 5: Gradient diagnostics (grad norm + advantage variance)

> `src/neural/training_plots.py` — `generate_all_panels(run_dir)` orchestrates all panels
