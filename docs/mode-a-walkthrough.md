# Mode A (LoopMPVAN) — Architecture & Training Walkthrough

A step-by-step guide to how the autoregressive loop-flip sampler works, from data preparation through sampling and training. Each section references exact lines in the source code.

---

## 1. Problem Setup

**Goal:** Learn to sample uniformly from the ice manifold — the set of all spin configurations satisfying the ice rule (divergence-free at every vertex).

**Key insight:** The ice manifold is a discrete set connected by *directed loop flips*. Starting from any valid ice state (the "seed"), every other reachable ice state can be reached by flipping some subset of β₁ independent cycles. This means we can parameterize the manifold as β₁ binary decisions (flip or don't flip each loop), and train an autoregressive model over these decisions.

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

The autoregressive ordering determines which loop is decided first, second, etc. The default strategy is `spatial_bfs`: build an overlap graph of loops (loops that share edges are adjacent), start from the most central loop, and BFS outward.

> `src/neural/loop_basis.py:227-310` — `compute_loop_ordering(loop_basis, strategy, positions, edge_list)`
> `src/neural/loop_basis.py:261-307` — spatial_bfs: overlap graph → BFS from center

### 2g. Exact enumeration (small systems only)

For β₁ ≤ 25, DFS through the autoregressive decision tree. At each node, if the current loop is a directed cycle, both branches (flip / no-flip) are explored. If not directed, only no-flip. This prunes the 2^β₁ tree dramatically.

> `src/neural/enumeration.py:23-91` — `enumerate_reachable_ice_states()` DFS
> `src/neural/enumeration.py:80-88` — the branching logic (no-flip always; flip only if directed)

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

> `src/neural/eign_layer.py:23-133` — full `EIGNLayer` class
> `src/neural/eign_layer.py:51-58` — 6 weight matrices
> `src/neural/eign_layer.py:106-119` — 4 message-passing terms (sparse @ dense @ linear)
> `src/neural/eign_layer.py:122-123` — skip connections
> `src/neural/eign_layer.py:126-131` — combine, LayerNorm, GELU
> `src/neural/eign_layer.py:74-85` — weight init: Xavier for MP, near-identity for skip

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

For each sample, the model processes loops one at a time in the fixed ordering. Starting from σ_seed:

```
For each loop i in ordering:
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

> `src/neural/loop_mpvan.py:209-265` — `sample()` method
> `src/neural/loop_mpvan.py:239-245` — directed-cycle check and skip
> `src/neural/loop_mpvan.py:248-253` — EIGN stack → pool → p_i
> `src/neural/loop_mpvan.py:256-257` — Bernoulli sample, log_prob accumulation
> `src/neural/loop_mpvan.py:259-260` — conditional flip
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

### 5b. One epoch step-by-step

> `src/neural/training.py:137-183` — main training loop

**Step 1: Sample a batch (no gradients)**

Generate B ice states by running the autoregressive sampler B times.

> `src/neural/training.py:141-145` — `model.sample(seed, inv_features, n_samples=batch_size)` under `torch.no_grad()`

**Step 2: Recover α vectors**

Each sampled σ came from some sequence of flip decisions α = (α₁, ..., α_{β₁}). But sampling was done without gradients. To get differentiable log_probs, we need to recover which α produced each σ. This is a GF(2) linear algebra problem:

```
diff[e] = 1 if σ[e] ≠ σ_seed[e], else 0
Solve: L^T · α = diff  (mod 2)    where L = loop_indicators
```

> `src/neural/training.py:147-148` — `recover_alpha(sigmas, seed_tensor, indicators)`
> `src/neural/loop_basis.py:410-451` — `recover_alpha()`: compute diff, call GF(2) solver per sample
> `src/neural/loop_basis.py:453-481` — `_solve_gf2()`: Gaussian elimination over GF(2)

**Step 3: Teacher-forced log_prob (with gradients)**

Re-run the autoregressive model with the *known* α sequence (teacher forcing). This time gradients flow through the EIGN stack and output head.

For each sample b:
- Walk through loops in order
- At each directed loop: compute p_i, accumulate log q = Σ [α_i·log(p_i) + (1-α_i)·log(1-p_i)]
- At each non-directed loop: skip (contributes 0)

> `src/neural/training.py:150-158` — loop over batch, call `model.forward_log_prob(alpha, seed, inv_features)`
> `src/neural/loop_mpvan.py:158-207` — `forward_log_prob()`: teacher forcing with directed-cycle checks
> `src/neural/loop_mpvan.py:184-206` — the autoregressive loop (same structure as sampling, but uses known α instead of Bernoulli)

**Step 4: REINFORCE gradient**

```python
rewards     = -log_probs.detach()           # lower log_prob = higher entropy = better
baseline    = 0.99 * baseline + 0.01 * mean(rewards)   # running mean
advantages  = rewards - baseline            # centered rewards
policy_loss = -mean(advantages * log_probs) # REINFORCE estimator
```

The key identity: ∇_θ E_q[f(σ)] = E_q[f(σ) · ∇_θ log q_θ(σ)]. Here f(σ) = -log q_θ(σ) (the reward), so the gradient pushes the model to increase probability of high-entropy samples.

> `src/neural/training.py:160-170` — REINFORCE: rewards, baseline update, advantages, policy_loss

**Step 5: Entropy bonus**

An explicit entropy bonus `-0.01 * mean(-log_probs)` is added to the loss. This directly encourages high-entropy distributions and prevents mode collapse early in training.

```python
entropy = -log_probs.mean()
loss = policy_loss - entropy_bonus * entropy
```

> `src/neural/training.py:172-174` — entropy bonus and final loss

**Step 6: Backprop and update**

```python
loss.backward()
clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
scheduler.step()
```

> `src/neural/training.py:176-183` — backward, clip, step, schedule

### 5c. Evaluation checkpoints

Every `eval_every` epochs, generate fresh samples and compute:
- **Ice violations:** should always be 0.0 (Mode A guarantee)
- **Mean Hamming distance:** pairwise distance between samples (~0.5 for uniform)
- **ESS:** effective sample size from importance weights (higher = more diverse)
- **KL divergence:** KL(empirical || uniform) if exact states available

> `src/neural/training.py:188-223` — evaluation block
> `src/neural/metrics.py` — all metric implementations

---

## 6. Putting It All Together

The complete data flow for one training epoch:

```
σ_seed ──────────────────────────────────────────────────────────────┐
                                                                     │
┌── SAMPLING (no grad) ──────────────────────────────────────────┐   │
│                                                                 │   │
│  For each of B samples:                                         │   │
│    σ = σ_seed                                                   │   │
│    For each loop i in ordering:                                 │   │
│      if is_directed_cycle(σ, loop_i):                           │   │
│        σ → [equ_input] → [EIGN×K] → [pool loop_i] → [MLP] → p_i│  │
│        α_i ~ Bernoulli(p_i)                                     │   │
│        if α_i: σ = flip(σ, loop_i)                              │   │
│    collect σ_b                                                   │   │
│                                                                 │   │
└── B samples: {σ_1, ..., σ_B} ─────────────────────────────────┘   │
         │                                                            │
         ▼                                                            │
┌── ALPHA RECOVERY ──────────────────────────────────────────────┐   │
│  For each σ_b:                                                  │   │
│    diff = (σ_b ≠ σ_seed)                                        │   │
│    Solve L^T · α_b = diff over GF(2)                            │   │
└── {α_1, ..., α_B} ────────────────────────────────────────────┘   │
         │                                                            │
         ▼                                                            │
┌── TEACHER FORCING (with grad) ─────────────────────────────────┐   │
│  For each α_b:                                                  │   │
│    σ = σ_seed                                                   │◄──┘
│    log_q = 0                                                    │
│    For each loop i in ordering:                                 │
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
│  baseline   = 0.99·baseline + 0.01·mean(rewards)               │
│  advantages = rewards - baseline                                │
│  loss       = -mean(advantages · log_q) - 0.01·mean(-log_q)    │
│  loss.backward() → clip_grad_norm → Adam step → cosine LR      │
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
