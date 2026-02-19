# Physics Review: Edge-MPVAN / LoopMPVAN vs. VAN Framework

## Context

Review of our Edge-MPVAN / LoopMPVAN implementation against the foundational VAN paper (Wu et al. 2019, "Solving Statistical Mechanics Using Variational Autoregressive Networks") and the MPVAN extension (Hao et al.). Goal: verify that our adaptation of EIGN edge-level message passing to the VAN framework for ASI sampling is physically sound and identify any bad design choices.

**Key documents reviewed:**
- `KeyPapers/SolvingStatMechUsingVAN.pdf` (Wu et al. 2019)
- `docs/edge-mpvan-proposal.html` (our full design document)
- `docs/mode-a-walkthrough.md` (implementation walkthrough)
- `src/neural/loop_mpvan.py`, `training.py`, `eign_layer.py`, `metrics.py` (source code)

---

## Verdict: The Physics Is Sound

The core adaptation is physically correct. The EIGN-ASI operator identity (L_equ = B1^T B1 = ASI Hamiltonian) is a mathematical fact, not an approximation. The autoregressive loop-basis factorization is a valid reparametrization. The REINFORCE training objective correctly reduces to entropy maximization at T=0. The directed-cycle constraint properly preserves probability normalization.

However, there are several design issues ranging from a structural limitation in the message-passing architecture to a scalability concern with gradient variance. None invalidate the XS proof-of-concept, but several will matter at larger sizes.

---

## (A) Correct and Well-Designed

### 1. EIGN = ASI Hamiltonian identity (mathematically exact)
The claim that MPVAN's "Hamiltonians MP" principle is satisfied by construction is correct. In MPVAN, the best strategy is to use Hamiltonian couplings J_ij as message weights. For edge-level ASI, the coupling between edges e and e' is `(B1^T B1)_{ee'}`, which is identically EIGN's equ-to-equ operator L_equ. No separate Hamiltonian encoding is needed — the physics IS the architecture.

### 2. Autoregressive factorization over loop flips is valid
The factorization `q(alpha) = prod_i q(alpha_i | alpha_{<i})` where alpha_i are loop-flip decisions is a correct autoregressive decomposition. When a loop is not directed (forced alpha_i = 0), it contributes factor 1 to the probability (log_prob += 0), which is correct — a deterministic event has probability 1. The product over all loops (directed and non-directed) properly normalizes to 1 across all possible outcomes.

### 3. REINFORCE gradient is correctly implemented
Wu et al. Eq. (5)/(14): `grad F = (1/beta) E[(R(s) - b) grad ln q(s)]` with `R(s) = beta*E(s) + ln q(s)`.

At T=0 with E=0 for all ice states: R(s) = ln q(s). We want to minimize E[ln q] (maximize entropy).

Our code: `rewards = -log_probs.detach()`, `policy_loss = -(advantages * log_probs).mean()`. This correctly implements REINFORCE for entropy maximization. Gradient descent on this loss pushes high-entropy (diverse) samples to be more probable.

### 4. Teacher-forcing + GF(2) recovery is consistent
The two-pass approach (sample without grad, recover alpha via GF(2), teacher-force with grad) produces identical log-probabilities to single-pass REINFORCE. The GF(2) solution is unique (cycle basis guarantees linear independence over GF(2)), and teacher forcing replays the exact same sigma sequence with the same directed-cycle decisions. Verified by tracing through overlapping-loop scenarios — no mismatch is possible.

### 5. Ice-rule guarantee by directed-cycle gating
Every sample is a valid ice state by construction. This eliminates the need for soft penalty terms (contrast with Mode B's planned lambda * ||B1 sigma||^2). Wu et al.'s VAN has no constraint mechanism at all — it relies on energy penalties to push toward ground states. Mode A's hard constraint is a genuine architectural advantage for T=0 sampling.

### 6. Skip connections correctly preserve kernel information
The near-identity initialization of W5, W6 (lines 80-85 of `eign_layer.py`) ensures that sigma features pass through the EIGN stack even when the message-passing channels are uninformative (see issue C1 below). Without skip connections, ice-state information would be annihilated by L_equ in the first layer.

---

## (B) Acceptable Tradeoffs (Not Bugs)

### 1. No Z2 symmetry (Wu et al. use q_Z2(s) = 0.5*(q(s) + q(-s)))
The ice manifold has sigma <-> -sigma symmetry, but in the loop-basis representation, the global flip corresponds to a specific (non-obvious) alpha vector, not a simple per-component flip. Implementing Z2 would require identifying this vector per lattice, which is non-trivial. The network can learn the symmetry through training. For XS (25 states), it demonstrably does (100% coverage includes both sigma and -sigma for each state). **Monitor at larger sizes** — if coverage becomes asymmetric, consider adding Z2.

### 2. No temperature annealing (correct for T=0)
Wu et al. anneal beta from 0 (uniform) to target. We don't need this because our target IS uniform (T=0, all ice states have E=0). The entropy bonus (0.01 coefficient) serves an analogous anti-collapse role. **Will be needed for Mode B** at finite temperature.

### 3. Small batch size (64 vs Wu et al.'s 1000)
Higher REINFORCE variance, but compensated by: running-mean baseline (smooths across epochs), smaller action space (beta_1=9 vs N=256), gradient clipping. Adequate for XS validation. **Will need scaling** for larger beta_1 (see issue C4).

### 4. Entropy bonus is "double-counting" but harmless
`loss = policy_loss - 0.01 * entropy` adds a direct gradient for entropy maximization on top of the REINFORCE signal. The REINFORCE signal is unbiased but high-variance; the entropy bonus is biased (uses old-policy samples) but low-variance. At coefficient 0.01, the bonus stabilizes early training without dominating. Standard practice in policy gradient methods (A2C, PPO).

### 5. No single-pass forward (beta_1 sequential passes per sample)
Wu et al. get all N conditionals in one masked-convolution forward pass. We need one full EIGN pass per loop decision. This is inherent to Mode A's design (each step conditions on a different fully-assigned sigma, not a partially-masked input). The cost is O(beta_1) passes per sample vs O(1). Acceptable for proof-of-concept; the scalability analysis already documents this as the dominant bottleneck.

---

## (C) Genuine Issues to Fix or Investigate

### C1. "Deaf Hamiltonian" — L_equ produces zero for ice states in layer 1 (MEDIUM)

**The problem:** The equivariant input is `X_equ = Linear(sigma)`, where each edge's feature is `sigma_e * w + bias`. For ice states, `B1 @ sigma = 0`. Since `Linear(1, d)` applies the SAME weight to every edge, column j of X_equ is `sigma * w_j + bias_j`. Then:

```
L_equ @ X_equ[:, j] = B1^T @ B1 @ (sigma * w_j + bias_j)
                     = w_j * B1^T @ (B1 @ sigma) + bias_j * B1^T @ (B1 @ 1)
                     = w_j * 0 + bias_j * B1^T @ (B1 @ 1)
```

The first term (sigma-dependent) is **exactly zero**. The second term (bias-dependent) is constant regardless of which ice state sigma is. **The Hamiltonian message-passing channel carries zero information about the current ice state in layer 1.**

The equ-to-inv cross-channel has the same problem: `|B1|^T @ B1 @ sigma = |B1|^T @ 0 = 0` for periodic-BC ice states.

**Impact:** For periodic BC with even coordination (the "clean" case), the network is functionally blind to sigma through the L_equ and equ_to_inv channels in layer 1. Only the skip connection (W5) and the inv-to-equ cross-channel (which carries static geometry, not sigma) survive. From layer 2 onward, the GELU nonlinearity breaks the kernel membership, and L_equ becomes informative.

**Mitigation:** For open BC (our current experiments), boundary vertices have |Q_v| = 1, so B1 @ sigma != 0, and L_equ IS informative in layer 1 near boundaries. This means our XS open-BC validation sidesteps this issue.

**Action needed:** Run a periodic-BC experiment (e.g., square 4x4 periodic) to test whether training still converges when the Hamiltonian channel is truly deaf in layer 1. If it fails or converges slowly, consider:
- Adding edge-pair features (sigma_e * sigma_e' for neighbors) that are NOT in ker(L_equ)
- Increasing network depth (gives more layers where L_equ is active)
- Using a different input representation that breaks the kernel structure

### C2. REINFORCE variance scales linearly with beta_1 (HIGH for scaling)

**The problem:** The reward `R = -ln q` is a sum over beta_1 directed-loop contributions: `ln q = sum_{directed loops} [alpha_i ln p_i + (1-alpha_i) ln(1-p_i)]`. By CLT, `Var(ln q) ~ beta_1 * var_per_loop`. The REINFORCE gradient variance is proportional to `Var(R) * E[||grad ln q||^2]`, which grows at least linearly with beta_1.

**Impact:** At beta_1 = 9 (square XS), variance is manageable. At beta_1 = 25 (square S-), it's ~3x worse. At beta_1 = 101 (kagome S), ~11x worse. This means:
- More epochs needed for convergence (compounding the per-epoch cost increase)
- Larger batch sizes needed to control gradient noise
- Training may fail to converge at all without variance reduction

**Comparison to Wu et al.:** Their masked-convolution architecture computes per-site gradients, effectively averaging over N sites. Our scalar-log-prob REINFORCE averages over only the batch dimension.

**Action needed:** Before attempting Tier 2 experiments (beta_1 > 15):
- Implement per-loop baselines (a learned baseline b_i for each loop position)
- Consider local reward shaping (credit assignment per loop rather than global reward)
- Monitor gradient variance during training and log it as a diagnostic
- Increase batch size proportionally to sqrt(beta_1) at minimum

### C3. Cycle basis from nx.cycle_basis may not be optimal (LOW-MEDIUM)

**The problem:** `nx.cycle_basis(G)` returns fundamental cycles from a spanning tree. These are typically short (each contains one non-tree edge) but not necessarily the most physically natural basis. For lattices with well-defined plaquette structure (square, kagome), the minimal face loops would be a more natural basis.

**Impact:** A suboptimal basis could:
- Create unnecessary overlap between loops (complicating the directed-cycle constraints)
- Make the autoregressive factorization harder to learn (spatially disjoint loops are easier)
- Reduce the fraction of loops that are directed in any given state

**Evidence:** For square XS, the current basis achieves 25/25 coverage and KL = 0.006, suggesting the basis is adequate. But the tetris 3x3 result (only 4 reachable states out of 2^22) shows that the directed-cycle constraint can dramatically prune the reachable space, and the severity of pruning likely depends on the basis choice.

**Action needed:** For lattices where training underperforms, try alternative basis construction (e.g., face-based loops from planar embedding). Compare reachable state counts across bases.

### C4. Validation is only on open BC — deaf Hamiltonian not stress-tested (MEDIUM)

**The problem:** All validation so far (square XS, shakti 2x2 partial) uses open boundary conditions. Open BC has odd-degree boundary vertices where B1 @ sigma != 0, meaning the L_equ channel IS informative near boundaries. This means our validation does not exercise the worst case (periodic BC, all even coordination) where L_equ is truly deaf in layer 1.

**Action needed:** Add periodic-BC experiments to the training experiments document. Compare convergence speed and final KL between open and periodic BC for the same lattice size.

### C5. Running-mean baseline has cold-start bias (LOW)

**The problem:** `baseline` starts at 0.0 (line 128 of `training.py`), while actual rewards at initialization are typically large (high entropy → large -ln q). The baseline takes ~100 epochs (at momentum 0.99) to converge to the true reward mean. During this warmup, advantages are biased, potentially causing wasteful early-training gradient steps.

**Action needed:** Initialize baseline to `mean(first_batch_rewards)` instead of 0.0. This is a one-line fix:
```python
# In train(), after first batch:
if epoch == 0:
    baseline = rewards.mean().item()
```

---

## Summary: Comparison to VAN Framework

| Wu et al. VAN principle | Our Mode A adaptation | Status |
|---|---|---|
| Autoregressive factorization q = prod q(s_i\|s_{<i}) | q = prod q(alpha_i\|alpha_{<i}) over loop flips | **Correct** — valid reparametrization |
| Variational free energy F = E[beta*E + ln q] | F = T * E[ln q] (since E=0 for ice states) | **Correct** — proper T=0 reduction |
| REINFORCE gradient with baseline | REINFORCE with running-mean baseline | **Correct** — same estimator |
| Masked convolutions for autoregressive property | No masking needed (full state at each step) | **Correct** — Mode A sees complete sigma |
| Hamiltonian couplings as message weights (MPVAN) | L_equ = B1^T B1 IS the Hamiltonian | **Correct** — by mathematical identity |
| Temperature annealing to avoid mode collapse | Entropy bonus (0.01 coefficient) | **Acceptable** for T=0 |
| Z2 symmetry q_Z2 = 0.5*(q(s) + q(-s)) | Not implemented | **Acceptable** — non-trivial in loop basis |
| Single forward pass for all conditionals | beta_1 sequential forward passes | **Acceptable** tradeoff for no-masking design |
| Batch size ~1000 | Batch size 64 | **Needs scaling** with beta_1 |
| Per-site gradient information | Scalar log_prob REINFORCE | **Needs improvement** for variance control |

---

## Recommended Priority Actions

1. **Run periodic-BC validation** (square 4x4 periodic) to test the deaf-Hamiltonian scenario
2. **Initialize baseline from first batch** (one-line fix in training.py)
3. **Add gradient variance logging** as a training diagnostic
4. **Before Tier 2 experiments:** implement per-loop baselines or increase batch size proportionally to beta_1
5. **Document the deaf-Hamiltonian property** in the Mode A walkthrough — it's a fundamental structural property, not a bug
