# Mode A Scalability Analysis: Honest Assessment

## The Bottom Line

Mode A (LoopMPVAN) is an elegant proof of concept that works beautifully at small sizes. **It will not scale to L-size lattices (50×50) in its current form.** The bottleneck is fundamental to the architecture, not an implementation detail. Mode B faces the same core bottleneck but has a clearer path to GPU acceleration via batching.

---

## 1. The Computational Wall

### Per-sample cost

Each sample requires β₁ sequential EIGN forward passes (one per loop in the autoregressive ordering). Each EIGN pass runs K layers through the full n₁-edge lattice. Nothing can be parallelized across these steps because each step depends on the result of the previous one (the spin configuration σ changes after each flip).

**Cost per sample:** O(β₁ × K × nnz(L_equ) × d)

For 2D lattices: β₁ ~ L², n₁ ~ 2L², nnz ~ 5n₁ ~ 10L². So:

**Cost per sample ~ O(L⁴ × K × d)**

This is the killer. It's quartic in the linear system size.

### Projected wall-clock times (CPU, M1 Pro)

Based on empirical measurements from XS and S- runs (3064 and 1588 EIGN passes/sec respectively):

| Size | L | n₁ | β₁ | EIGN passes/sample | **CPU time (2000 ep, B=64)** |
|------|---|-----|------|-------------------|------------------------------|
| XS | 4 | 24 | 9 | 36 | **2 min** |
| S- | 6 | 60 | 25 | 100 | **1.1 hr** |
| S | 10 | 180 | 81 | 324 | **11 hr** |
| M | 20 | 760 | 361 | 1,444 | **8.5 days** |
| L | 50 | 4,900 | 2,401 | 9,604 | **366 days** |

The training time grows as ~L⁴. Every doubling of L multiplies the time by 16×.

### Why this can't be fixed with simple optimizations

The sequential nature of the autoregressive loop is the fundamental constraint. At each step i:

1. σ has been modified by all prior flips → must re-run full EIGN stack on the new σ
2. Directed-cycle check depends on current σ → can't pre-compute
3. The flip decision at step i changes σ for step i+1 → fully serial dependency

You cannot amortize, cache, or skip the EIGN computation across steps. Every step is a full forward pass through the network on a different input.

---

## 2. Five Specific Bottlenecks

### Bottleneck 1: Sequential EIGN passes (dominant)

β₁ sequential forward passes per sample. Each pass: 4 sparse matmuls + 6 linear transforms + 2 LayerNorms, through K layers.

- **XS (β₁=9):** 36 passes/sample — fine
- **L (β₁=2401):** 9,604 passes/sample — catastrophic

**Can batching help?** Partially. You could batch the B=64 samples at the *same* autoregressive step (same loop index), running 64 EIGN forward passes as a single batched operation on GPU. This gives ~64× speedup on the EIGN computation. But the β₁ sequential steps remain serial.

With perfect GPU batching: L-size drops from 366 days to ~6 days. Better, but still not practical for a research iteration cycle.

### Bottleneck 2: GF(2) alpha recovery

After sampling (no grad), we need to recover the α vector for teacher forcing. This requires Gaussian elimination over GF(2):

**Cost per sample:** O(n₁ × β₁²)

| Size | GF(2) ops/sample | GF(2) ops/epoch (B=64) |
|------|-------------------|------------------------|
| XS | 1.9 × 10³ | 1.2 × 10⁵ |
| S | 1.2 × 10⁶ | 7.6 × 10⁷ |
| M | 9.9 × 10⁷ | 6.3 × 10⁹ |
| L | 2.8 × 10¹⁰ | 1.8 × 10¹² |

At L-size, the GF(2) recovery alone costs ~10¹² integer ops per epoch. This is a second wall, independent of the EIGN computation.

### Bottleneck 3: Directed-cycle checking (Python overhead)

Each loop at each step requires `is_directed_cycle()`, which does scipy sparse column lookups in a Python loop. This is O(|cycle_edges|) per check, O(β₁ × avg_cycle_length) per sample. Not the dominant term, but the Python/scipy overhead adds constant-factor pain.

### Bottleneck 4: REINFORCE variance scaling

The REINFORCE gradient estimator has variance proportional to the sequence length. Longer autoregressive sequences → noisier gradients → more epochs needed to converge. The running-mean baseline helps, but variance still grows with β₁.

At XS (β₁=9), 500 epochs suffice. At S- (β₁=25), 2000 epochs get close. At L (β₁=2401), we may need significantly more epochs, compounding the per-epoch cost.

### Bottleneck 5: Evaluation without enumeration

For β₁ > 25, exact enumeration is impossible. We lose our ability to compute KL(empirical || uniform) and state coverage. The only metrics available are:
- ESS (proxy for diversity, but doesn't detect systematic biases)
- Hamming distance (necessary condition for uniformity, not sufficient)
- Unique state count (lower bound on coverage)

Without ground truth, we can't verify that the model is actually sampling uniformly. It might learn to cover a large but biased subset of the ice manifold and we'd have no way to detect this.

---

## 3. What Would It Take to Reach L-Size?

### Option A: Brute-force GPU acceleration

Batch across samples at each autoregressive step. Move all ops to GPU. Implement custom CUDA kernels for the sparse-dense products.

- **Speedup:** ~64× from batching, ~10× from GPU arithmetic = ~640×
- **L-size estimate:** 366 days / 640 ≈ **14 hours**
- **Feasibility:** Possible but requires significant engineering (batched sparse ops, compiled autoregressive loop, GF(2) on GPU)
- **Problem:** Still serial over β₁=2401 steps. GPU utilization will be low because each step is a small operation.

### Option B: Reduce autoregressive sequence length

Instead of deciding each loop independently, group loops into blocks and make block-level decisions. Or use a hierarchical model: coarse-grained loop groups → fine-grained individual loops.

- **Potential speedup:** If you can reduce from β₁ steps to √β₁ steps, that's 49× for L-size
- **Problem:** Grouping loops while maintaining ice-rule guarantees is non-trivial. The directed-cycle constraint couples loops in complex ways.

### Option C: Non-autoregressive sampling

Train a model that outputs all β₁ flip decisions simultaneously, then project onto the valid ice manifold. Or use a flow-based model (normalizing flow on the loop-flip space).

- **Potential:** O(1) forward passes per sample instead of O(β₁)
- **Problem:** No longer guaranteed valid. Need the directed-cycle constraint somehow. The discrete binary nature of α makes continuous normalizing flows difficult.

### Option D: Entirely different approach

MCMC sampling with learned proposals. The model doesn't need to sample autoregressively — it just needs to propose good moves for a Metropolis-Hastings chain. Each proposal could be a single EIGN forward pass → suggest which loop to flip.

- **Cost:** O(K × n₁ × d) per MCMC step (one EIGN pass, not β₁)
- **Mixing time:** Unknown, but the model can learn to make large moves
- **Validity:** Each accepted move is a single directed loop flip → always valid

This reduces the per-step cost from O(β₁) EIGN passes to O(1). The question becomes whether the mixing time scales favorably.

---

## 4. Mode B: Better or Worse?

### Sequence length

Mode B autoregressively samples n₁ individual edges. For square: n₁ ≈ 2β₁, so the sequence is ~2× longer than Mode A.

**At first glance, Mode B is worse.** But:

### Batching advantage (the key difference)

In Mode A, different samples may take different paths through the autoregressive tree (some loops are directed, some aren't, depending on the sample's history). This makes batching across samples awkward — samples diverge.

In Mode B, **all samples follow the same fixed edge ordering.** At step t, every sample has assigned edges 1..t-1 and is deciding edge t. The causal mask is the same for all samples. This means:

- The masked EIGN forward pass can be trivially batched as `(B, n₁, d)` through the same masked operators
- No divergent paths, no directed-cycle branching
- One batched EIGN pass serves all B samples simultaneously

**Effective cost comparison (GPU, batched):**

| Mode | Steps per batch | EIGN passes per batch | Per-batch cost |
|------|-----------------|----------------------|----------------|
| A (current) | β₁ × B | β₁ × B | O(β₁ × B × K × n₁ × d) |
| A (batched) | β₁ | β₁ (each batched over B) | O(β₁ × K × n₁ × d × B/GPU_parallelism) |
| B (batched) | n₁ | n₁ (each batched over B) | O(n₁ × K × n₁ × d × B/GPU_parallelism) |

Mode B has ~2× more steps but the same batching benefit. The difference is modest.

### No GF(2) recovery

Mode B eliminates the alpha recovery bottleneck entirely. The model directly outputs log q_θ(σ) as it samples — no need to solve a linear system to recover what decisions were made.

### Soft ice-rule penalty: a weakness

Mode B does **not** guarantee valid ice states. It relies on:
```
loss = E[H(σ)] + T·E[ln q] + λ(T)·‖B₁σ‖²
```

The penalty λ·‖B₁σ‖² pushes toward ice-rule satisfaction but doesn't enforce it. At finite temperature, some fraction of samples will have monopole excitations. This is physically meaningful (real ASI has monopoles at T > 0) but makes the T → 0 limit harder to achieve cleanly.

### Causal masking overhead

Mode B requires precomputing lower-triangular masks for each of the 4 EIGN operators. This is a one-time cost and doesn't affect the scaling, but adds implementation complexity and potentially reduces the information available at each step (partial lattice view vs. Mode A's full lattice view).

### Mode B verdict

Mode B is **not fundamentally faster** than Mode A (same O(seq_len × K × n₁ × d) scaling), but it's **much easier to batch efficiently on GPU** because all samples follow the same execution path. It also eliminates the GF(2) bottleneck. The cost is losing the ice-rule guarantee and needing temperature annealing.

For a GPU-targeted implementation, Mode B is likely the more practical path to larger sizes.

---

## 5. The Honest Summary

### What Mode A is good for

- **Proof of concept:** Demonstrates that EIGN message passing on ASI lattices can learn the ice manifold structure
- **Exact validation at small sizes:** With enumeration, we can verify true uniformity
- **Zero-violation guarantee:** Every sample is a valid ice state, period
- **Scientific insight:** The reachable-state counts (e.g., tetris 3×3 has only 4 states out of 2²²) reveal the strength of the directed-cycle constraint

### What Mode A cannot do

- **Scale to L-size (50×50):** The O(L⁴) per-sample cost makes this impractical even on GPU
- **Scale to M-size (20×20) on CPU:** ~8.5 days per training run
- **Verify uniformity at large sizes:** No enumeration → no ground truth

### The fundamental tension

Autoregressive models over structured discrete spaces face a dilemma:
- **Sequential decisions preserve structure** (ice rule) but create O(seq_len) serial dependencies
- **Parallel decisions are fast** but don't respect the constraints
- The ice manifold's structure (directed-cycle coupling between loops) resists parallelization

### Most promising path forward

1. **Short term:** GPU-batched Mode A or Mode B for S/M sizes. Sufficient for demonstrating the EIGN architecture works on frustrated lattices with mixed coordination.

2. **Medium term:** Learned MCMC proposals (one EIGN pass → suggest a loop flip → Metropolis accept/reject). This is O(1) EIGN passes per move instead of O(β₁), with mixing time as the open question.

3. **Long term:** Non-autoregressive models (discrete flows, score-based diffusion on binary spaces) that can generate all β₁ decisions in parallel.

The current Mode A results at XS and S- are scientifically valuable as-is — they validate the EIGN-ASI connection and demonstrate exact uniform sampling on frustrated topologies. Scaling to L-size would require a fundamentally different sampling algorithm, not just faster hardware.
