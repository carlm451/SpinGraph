# Mode A Training Experiments

## Overview

Systematic training of LoopMPVAN across the lattice zoo. All experiments use `scripts/train_lattice.py` (generalized to any lattice type). Experiments are organized into tiers of increasing difficulty, with pass/fail criteria and pre-tier checklists to gate progression.

**Activate venv first:**
```bash
source venv/bin/activate
```

**Generate plots after any run:**
```bash
python -m scripts.plot_training_diagnostics --run-dir results/neural_training/{run_id}
```

---

## Pass/Fail Criteria

Every run is evaluated against these metrics:

| Metric | Pass | Marginal | Fail | Notes |
|--------|------|----------|------|-------|
| Ice violations | 0.0 | — | > 0 | Mode A guarantee; any violation is a bug |
| KL(empirical \|\| uniform) | < 0.05 | 0.05–0.10 | > 0.10 | Measures deviation from uniform sampling |
| State coverage | 1.0 | > 0.90 | < 0.90 | Fraction of reachable states sampled (when enumerable) |
| ESS | > batch_size | > 0.5 × batch_size | < 0.5 × batch_size | Effective sample size |
| Entropy | > 0.95 × ln(N) | > 0.90 × ln(N) | < 0.90 × ln(N) | Fraction of maximum entropy |
| Grad norm (final 100 ep) | < 2 × grad_clip | < 5 × grad_clip | Diverging / NaN | Post-clip norm should be stable |
| Advantage variance | Decreasing trend | Flat | Increasing / exploding | High variance → high REINFORCE gradient variance |

---

## Lattice Survey

| Lattice   | Size  | BC   | n₀   | n₁  | β₁ | Reachable States | Enumerable | Est. Train Time (2000 ep) |
|-----------|-------|------|------|-----|-----|-----------------|------------|---------------------------|
| square    | 4×4   | open | 16   | 24  | 9   | 25              | yes        | ~6 min                    |
| square    | 4×4   | periodic | 25 | 40 | 16 | ~2,000         | yes        | ~12 min                   |
| square    | 6×6   | open | 36   | 60  | 25  | 3,029           | yes        | ~70 min                   |
| kagome    | 2×2   | open | 12   | 17  | 6   | 32              | yes        | ~2 min                    |
| kagome    | 2×2   | periodic | 12 | 24 | 13 | ~256           | yes        | ~8 min                    |
| kagome    | 3×3   | open | 27   | 43  | 17  | 4,352           | yes        | ~30 min                   |
| santa_fe  | 2×2   | open | 24   | 30  | 7   | 8               | yes        | ~4 min                    |
| santa_fe  | 3×3   | open | 54   | 72  | 19  | 48              | yes        | ~35 min                   |
| shakti    | 1×1   | open | 16   | 18  | 2   | 4               | yes        | ~1 min                    |
| shakti    | 2×2   | open | 64   | 81  | 18  | 112             | yes        | ~30 min                   |
| tetris    | 2×2   | open | 32   | 38  | 7   | 12              | yes        | ~4 min                    |
| tetris    | 3×3   | open | 72   | 93  | 22  | 4               | yes        | ~40 min                   |

Time estimates based on square XS (β₁=9, 500ep = 188s) scaling roughly as β₁ × n₁ × epochs.

---

## Tier 0: Diagnostic Experiments (run first)

These experiments validate the C1–C5 fixes before scaling up. Each takes ~2–6 min.

### Pre-Tier 0 Checklist

- [ ] All 57+ tests pass: `python -m pytest tests/ -v`
- [ ] C5 baseline fix applied (`baseline = None` in training.py)
- [ ] Gradient diagnostics recording verified (new fields in TrainingResult)

### 0a. Square 4×4 open — re-baseline (C5 validation)

Re-run the original validation experiment with the C5 baseline fix.

```bash
python -m scripts.train_lattice --lattice square --nx 4 --ny 4 --boundary open --epochs 500
```

**Tests:** C5 fix — `loss[0]` should be comparable to `loss[50]` (no cold-start spike).

**Pass criteria:**
- KL < 0.01
- `loss[0] < 5 × loss[50]` (old run had `loss[0]` ~10× higher due to baseline=0 cold start)
- `grad_norm_history` and `advantage_var_history` populated and finite

**Interpretation:** Compare `loss_history[0]` to the old run (`square_open_20260216_164125`). The first-epoch loss should be substantially lower with the C5 fix since the baseline is initialized from the first batch instead of 0.

### 0b. Square 4×4 periodic — deaf Hamiltonian test (C1/C4)

Test periodic BC where all vertices have even coordination (z=4). The deaf Hamiltonian (L_equ @ sigma = 0 for ice states) affects the equ→equ and equ→inv channels in layer 1.

```bash
python -m scripts.train_lattice --lattice square --nx 4 --ny 4 --boundary periodic --epochs 2000
```

**Tests:**
- C1: Does the skip+GELU mechanism compensate for the deaf Hamiltonian?
- C4: Does periodic BC (toroidal topology, β₁ includes non-contractible loops) train correctly?

**Pass criteria:**
- KL < 0.05
- State coverage > 0.90
- No NaN/Inf in gradient norms

**Interpretation:** If this fails but 0c passes, the deaf Hamiltonian is a real problem on periodic even-coordination lattices. Follow-up: add depth (n_layers=5) or nonlinear input expansion.

### 0c. Kagome 2×2 periodic — control (C4)

Control experiment: kagome has z=3 (odd coordination), so L_equ @ sigma != 0 even for ice states (since |Q_v| = 1 on z=3 vertices). The equ→equ channel is NOT deaf.

```bash
python -m scripts.train_lattice --lattice kagome --nx 2 --ny 2 --boundary periodic --epochs 1000
```

**Tests:** Control for C1 — odd coordination means Hamiltonian channel is active from layer 1.

**Pass criteria:**
- KL < 0.03
- State coverage > 0.95

**Interpretation:** If 0b fails and 0c passes, the deaf Hamiltonian is the cause. If both fail, the issue is with periodic BC handling generally, not the Hamiltonian channel.

### Tier 0 Results Template

After running all three experiments, fill in:

| Exp | KL | Coverage | ESS | Grad norm (final) | Adv var (final) | Pass? |
|-----|-----|----------|-----|--------------------|-----------------|-------|
| 0a  |     |          |     |                    |                 |       |
| 0b  |     |          |     |                    |                 |       |
| 0c  |     |          |     |                    |                 |       |

---

## Tier 1: Quick Runs (~2–6 min each)

Small β₁, fast convergence, good for validating the pipeline across lattice types.

### Pre-Tier 1 Checklist

- [ ] Tier 0 experiments all pass
- [ ] Gradient diagnostics look healthy (no NaN, no explosion)

### 1a. Square 4×4 open (baseline, already complete)
```bash
python -m scripts.train_lattice --lattice square --nx 4 --ny 4 --boundary open --epochs 500
```
**Expected:** KL < 0.01, 100% coverage, 0 violations. **Status: DONE** (run: `square_open_20260216_164125`)

### 1b. Kagome 2×2 open
```bash
python -m scripts.train_lattice --lattice kagome --nx 2 --ny 2 --boundary open --epochs 1000
```
**Expected:** β₁=6, 32 states. Should converge easily to near-uniform.

**Pass:** KL < 0.03, coverage = 1.0

### 1c. Santa Fe 2×2 open
```bash
python -m scripts.train_lattice --lattice santa_fe --nx 2 --ny 2 --boundary open --epochs 1000
```
**Expected:** β₁=7, only 8 states. Very small state space, fast convergence.

**Pass:** KL < 0.02, coverage = 1.0

### 1d. Tetris 2×2 open
```bash
python -m scripts.train_lattice --lattice tetris --nx 2 --ny 2 --boundary open --epochs 1000
```
**Expected:** β₁=7, 12 states. Similar to santa_fe.

**Pass:** KL < 0.03, coverage = 1.0

### 1e. Shakti 1×1 open
```bash
python -m scripts.train_lattice --lattice shakti --nx 1 --ny 1 --boundary open --epochs 500
```
**Expected:** β₁=2, trivial. Smoke test for shakti pipeline.

**Pass:** KL < 0.01, coverage = 1.0

### Tier 1 Results Template

| Exp | Lattice | β₁ | States | KL | Coverage | ESS | Grad norm | Time | Pass? |
|-----|---------|-----|--------|-----|----------|-----|-----------|------|-------|
| 1a  | square  | 9   | 25     |     |          |     |           |      | DONE  |
| 1b  | kagome  | 6   | 32     |     |          |     |           |      |       |
| 1c  | santa_fe| 7   | 8      |     |          |     |           |      |       |
| 1d  | tetris  | 7   | 12     |     |          |     |           |      |       |
| 1e  | shakti  | 2   | 4      |     |          |     |           |      |       |

---

## Tier 2: Medium Runs (~20–40 min each)

Larger β₁ (15–22), meaningful state spaces. These test whether the model can learn to cover hundreds–thousands of states.

### Pre-Tier 2 Checklist

- [ ] All Tier 1 experiments pass
- [ ] Review gradient diagnostics from Tier 1: no divergence, advantage variance decreasing
- [ ] Consider increasing batch_size for β₁ > 15 (default 64 may have high REINFORCE variance)
- [ ] Batch size scaling guidance: `batch_size >= 4 × β₁` as a starting heuristic

### Gradient Variance Monitoring (C2)

For Tier 2 runs, inspect Panel 5 (gradient diagnostics) after each run:

- **Healthy:** Grad norm stable or slowly decreasing. Advantage variance decreasing.
- **Warning:** Grad norm has spikes > 5× the clip value. Advantage variance flat or increasing.
- **Action if warning:** Increase batch_size (128 or 256), or reduce learning rate.

### Cycle Basis Monitoring (C3)

At β₁ > 15, the cycle basis ordering may affect convergence speed. The current implementation uses a fixed basis from the kernel of the incidence matrix. If Tier 2 runs show slow convergence:

- Check whether early loops in the ordering have high directedness rates
- Consider sorting loops by directedness (most often directed → earliest in sequence)
- This is monitoring only — no code changes until evidence shows it matters

### 2a. Shakti 2×2 open
```bash
python -m scripts.train_lattice --lattice shakti --nx 2 --ny 2 --boundary open --epochs 2000
```
**Expected:** β₁=18, 112 states. Mixed coordination (z=1,2,3,4). Key test of EIGN on frustrated lattice.

**Pass:** KL < 0.05, coverage > 0.95

### 2b. Kagome 3×3 open
```bash
python -m scripts.train_lattice --lattice kagome --nx 3 --ny 3 --boundary open --epochs 2000
```
**Expected:** β₁=17, 4,352 states. Large state space — good stress test for coverage.

**Pass:** KL < 0.08, coverage > 0.90

### 2c. Santa Fe 3×3 open
```bash
python -m scripts.train_lattice --lattice santa_fe --nx 3 --ny 3 --boundary open --epochs 2000
```
**Expected:** β₁=19, only 48 states. Compact manifold despite moderate β₁.

**Pass:** KL < 0.05, coverage > 0.95

### 2d. Tetris 3×3 open
```bash
python -m scripts.train_lattice --lattice tetris --nx 3 --ny 3 --boundary open --epochs 2000
```
**Expected:** β₁=22, only 4 states! Extreme directed-cycle constraint. Interesting test case.

**Pass:** KL < 0.03, coverage = 1.0

### Tier 2 Results Template

| Exp | Lattice | β₁ | States | KL | Coverage | ESS | Grad norm (avg last 100) | Adv var (avg last 100) | Time | Pass? |
|-----|---------|-----|--------|-----|----------|-----|--------------------------|------------------------|------|-------|
| 2a  | shakti  | 18  | 112    |     |          |     |                          |                        |      |       |
| 2b  | kagome  | 17  | 4,352  |     |          |     |                          |                        |      |       |
| 2c  | santa_fe| 19  | 48     |     |          |     |                          |                        |      |       |
| 2d  | tetris  | 22  | 4      |     |          |     |                          |                        |      |       |

---

## Tier 3: Long Runs (~60–90 min each)

Pushing β₁ to 25. May require batch size adjustments if REINFORCE variance is problematic.

### Pre-Tier 3 Checklist

- [ ] All Tier 2 experiments pass
- [ ] Review gradient diagnostics from Tier 2: confirm advantage variance is manageable
- [ ] If any Tier 2 run showed gradient issues, increase batch_size to 128 or 256 for Tier 3
- [ ] Consider per-loop baselines if advantage variance is high (see deferred C2 item 4)

### 3a. Square 6×6 open
```bash
python -m scripts.train_lattice --lattice square --nx 6 --ny 6 --boundary open --epochs 2000
```
**Expected:** β₁=25, 3,029 states. Partial run showed KL=0.028 at epoch 200 (entropy nearly converged).

**Pass:** KL < 0.05, coverage > 0.90

---

## GPU Feasibility Notes

If CPU training is too slow for Tier 2/3:

1. **Current bottleneck:** Each epoch does `batch_size` sequential forward passes through the EIGN stack (one per autoregressive loop step × batch). No batching across samples in the current implementation.

2. **GPU would help if:** We batch the EIGN forward pass across samples. Currently the `sample()` and `forward_log_prob()` loops are sample-by-sample. Batching the EIGN stack (which is just sparse matrix multiplies + linear layers) across B samples would give ~B× speedup on GPU.

3. **Quick GPU test:** If you have CUDA available, the model already uses PyTorch and should move to GPU with `.to('cuda')`. The bottleneck is the Python loop over samples, not the tensor ops.

4. **Estimated speedup from batching:** For β₁=25, batch_size=64: current approach does 25×64=1600 sequential EIGN forward passes per epoch. With batched implementation: 25 batched passes. ~64× speedup on the forward pass.

---

## Cross-Lattice Comparison

After completing Tier 1+2 runs, compare:

- **KL convergence rate** across lattice types at similar β₁
- **State coverage** vs number of reachable states
- **Training time** vs β₁ (scaling law)
- **Effect of coordination mixing** (square=uniform z=4 vs shakti=mixed z=1-4)
- **Directed-cycle constraint strength** (tetris 3×3 has only 4/2^22 states reachable!)
- **Gradient diagnostics** across lattice types (does mixed coordination help or hurt REINFORCE variance?)

---

## Deferred Items

These are not yet implemented but may be needed based on Tier 2+ results:

| Item | Trigger | Action |
|------|---------|--------|
| Per-loop baselines (C2) | Advantage variance stays high at β₁ > 15 | Implement loop-specific running-mean baselines |
| Cycle basis reordering (C3) | Slow convergence correlated with loop ordering | Sort loops by directedness frequency |
| Nonlinear input expansion (C1) | Periodic BC + even coordination fails to train | Add `sigma -> [sigma, sigma^2, ...]` input layer |
| Increased depth (C1) | Skip+GELU insufficient for Hamiltonian info | Try n_layers=5 instead of default 3 |
