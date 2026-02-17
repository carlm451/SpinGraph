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

> `src/neural/training.py:139-197` — main training loop

**Step 1: Sample a batch (no gradients)**

Generate B ice states by running the autoregressive sampler B times.

> `src/neural/training.py:143-147` — `model.sample(seed, inv_features, n_samples=batch_size)` under `torch.no_grad()`

**Step 2: Recover α vectors**

Each sampled σ came from some sequence of flip decisions α = (α₁, ..., α_{β₁}). But sampling was done without gradients. To get differentiable log_probs, we need to recover which α produced each σ. This is a GF(2) linear algebra problem:

```
diff[e] = 1 if σ[e] ≠ σ_seed[e], else 0
Solve: L^T · α = diff  (mod 2)    where L = loop_indicators
```

> `src/neural/training.py:149-150` — `recover_alpha(sigmas, seed_tensor, indicators)`
> `src/neural/loop_basis.py:410-451` — `recover_alpha()`: compute diff, call GF(2) solver per sample
> `src/neural/loop_basis.py:453-481` — `_solve_gf2()`: Gaussian elimination over GF(2)

**Step 3: Teacher-forced log_prob (with gradients)**

Re-run the autoregressive model with the *known* α sequence (teacher forcing). This time gradients flow through the EIGN stack and output head.

For each sample b:
- Walk through loops in order
- At each directed loop: compute p_i, accumulate log q = Σ [α_i·log(p_i) + (1-α_i)·log(1-p_i)]
- At each non-directed loop: skip (contributes 0)

> `src/neural/training.py:152-160` — loop over batch, call `model.forward_log_prob(alpha, seed, inv_features)`
> `src/neural/loop_mpvan.py:158-207` — `forward_log_prob()`: teacher forcing with directed-cycle checks
> `src/neural/loop_mpvan.py:184-206` — the autoregressive loop (same structure as sampling, but uses known α instead of Bernoulli)

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

Every `eval_every` epochs, generate fresh samples and compute:
- **Ice violations:** should always be 0.0 (Mode A guarantee)
- **Mean Hamming distance:** pairwise distance between samples (~0.5 for uniform)
- **ESS:** effective sample size from importance weights (higher = more diverse)
- **KL divergence:** KL(empirical || uniform) if exact states available
- **Gradient norm:** post-clip gradient norm (logged with eval metrics)
- **Advantage variance:** variance of REINFORCE advantages (logged with eval metrics)

> `src/neural/training.py:202-238` — evaluation block
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

---

## 8. Inference: Using the Trained Model

After training, the saved checkpoint contains everything needed to generate new ice state samples. This section describes how to load a trained model and use it for inference.

### 8a. What's saved in a training run

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

### 8b. Reconstruction procedure

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

Steps 2–5 are deterministic given the lattice parameters, so the loop basis and ordering will be identical to the ones used during training. This is essential — the model's weights are tied to a specific loop ordering.

> `scripts/train_lattice.py:97-121` — the same setup pipeline used before training
> `src/neural/training.py` — `build_inv_features(edge_list, coordination)` for step 9

### 8c. Generating samples

Once the model is reconstructed:

```python
model.eval()
seed_tensor = torch.from_numpy(sigma_seed.astype(np.float32))
inv_features = build_inv_features(edge_list, coordination)

with torch.no_grad():
    sigmas, log_probs = model.sample(seed_tensor, inv_features, n_samples=N)
```

This returns:
- `sigmas`: (N, n₁) tensor of ice state configurations, each ±1
- `log_probs`: (N,) tensor of log q_θ(σ) for each sample

Every sample is a valid ice state by construction. The samples are **independent** (no autocorrelation, unlike MCMC) — each one is a fresh forward pass through the autoregressive loop.

> `src/neural/loop_mpvan.py:209-265` — `sample()` method (runs under `@torch.no_grad()`)

### 8d. What you can compute from samples

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

### 8e. Sampling cost

Each sample requires β₁ sequential passes through the EIGN stack (one per loop in the ordering). At each step:
1. **Directed-cycle check** — O(loop_length) vertex inspections
2. **EIGN forward pass** — K sparse-dense matrix multiplies over all n₁ edges
3. **Output head** — pool over loop edges + 3-layer MLP → scalar

Non-directed loops are skipped (no EIGN pass), so the effective number of forward passes is typically less than β₁.

**Benchmark (square XS, 24 edges, β₁ = 9):** 2000 samples in ~6 seconds on CPU.

The key advantage over MCMC: samples are **independent by construction**. MCMC requires τ_corr flip sweeps between independent samples, where τ_corr grows with system size. The neural sampler's cost per independent sample is fixed at one forward pass, regardless of correlation structure.

### 8f. Diagnostic plots from saved runs

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
