# Mode A Training Experiments

## Overview

Systematic training of LoopMPVAN across the lattice zoo. All experiments use `scripts/train_lattice.py` (generalized to any lattice type). Experiments are organized into tiers of increasing difficulty, with pass/fail criteria and pre-tier checklists to gate progression.

**Key change (v2):** All training now uses **per-batch ordering randomization** (default). This closes the autoregressive ordering gap where a fixed loop ordering misses reachable ice states. State counts in the lattice survey below reflect multi-ordering enumeration (200 random orderings). See `docs/mode-a-walkthrough.md` Section 7 for details.

**Ensemble averaging:** Each experiment should be run N=5 times with different seeds. Results tables report mean +/- std across runs. Use `scripts/train_ensemble.py` for automated ensemble runs.

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
| State coverage | > 0.90 | 0.80–0.90 | < 0.80 | Fraction of reachable states sampled (when enumerable) |
| ESS | > batch_size | > 0.5 × batch_size | < 0.5 × batch_size | Effective sample size |
| Entropy | > 0.95 × ln(N) | > 0.90 × ln(N) | < 0.90 × ln(N) | Fraction of maximum entropy |
| Grad norm (final 100 ep) | < 2 × grad_clip | < 5 × grad_clip | Diverging / NaN | Post-clip norm should be stable |
| Advantage variance | Decreasing trend | Flat | Increasing / exploding | High variance → high REINFORCE gradient variance |

**Coverage note:** With ordering randomization, final evaluation samples across 20 random orderings. Coverage is measured against the multi-ordering state count (which is itself a lower bound on true ε(G)). Coverage thresholds have been relaxed from 1.0/0.90 to 0.90/0.80 to account for the larger state space.

---

## Lattice Survey

State counts are from multi-ordering enumeration (200 random orderings). The "DFS (fixed)" column shows how many states a single fixed ordering finds, for comparison.

| Lattice   | Size  | BC       | n₀  | n₁  | β₁ | States (multi-ordering) | DFS (fixed) | Enumerable | Est. Train Time |
|-----------|-------|----------|-----|-----|-----|------------------------|-------------|------------|-----------------|
| square    | 4×4   | open     | 16  | 24  | 9   | 38                     | 25          | yes        | ~2 min          |
| square    | 4×4   | periodic | 16  | 32  | 17  | 299                    | 52          | yes        | ~6 min          |
| square    | 6×6   | open     | 36  | 60  | 25  | TBD                    | 3,029       | yes        | ~25 min         |
| kagome    | 2×2   | open     | 12  | 17  | 6   | TBD                    | 32          | yes        | ~1 min          |
| kagome    | 2×2   | periodic | 12  | 24  | 13  | 355                    | 189         | yes        | ~5 min          |
| kagome    | 3×3   | open     | 27  | 43  | 17  | TBD                    | 4,352       | yes        | ~10 min         |
| santa_fe  | 2×2   | open     | 24  | 30  | 7   | TBD                    | 8           | yes        | ~2 min          |
| santa_fe  | 3×3   | open     | 54  | 72  | 19  | TBD                    | 48          | yes        | ~15 min         |
| shakti    | 1×1   | open     | 16  | 18  | 2   | TBD                    | 3           | yes        | ~1 min          |
| shakti    | 2×2   | open     | 64  | 81  | 18  | TBD                    | 112         | yes        | ~10 min         |
| tetris    | 2×2   | open     | 32  | 38  | 7   | TBD                    | 12          | yes        | ~2 min          |
| tetris    | 3×3   | open     | 72  | 93  | 22  | TBD                    | 4           | yes        | ~10 min         |

Multi-ordering counts for TBD entries will be filled as tier experiments run. Time estimates updated for batched forward pass + ordering randomization.

---

## Tier 0: Diagnostic Experiments (run first)

These experiments validate the training pipeline before scaling up.

### Pre-Tier 0 Checklist

- [ ] All tests pass: `python -m pytest tests/ -v`
- [ ] Ordering randomization enabled by default in TrainingConfig

### 0a. Square 4×4 open — baseline

```bash
python -m scripts.train_lattice --lattice square --nx 4 --ny 4 --boundary open \
  --epochs 2000 --run-id tier0-v3-0a
```

**Pass criteria:**
- KL < 0.05
- Coverage > 0.80 (of 38 multi-ordering states)
- 0 ice violations

### 0b. Square 4×4 periodic — deaf Hamiltonian test

```bash
python -m scripts.train_lattice --lattice square --nx 4 --ny 4 --boundary periodic \
  --epochs 4000 --run-id tier0-v3-0b
```

**Pass criteria:**
- KL < 0.10
- Coverage > 0.50 (of 299 multi-ordering states — large state space for XS)
- 0 ice violations

### 0c. Kagome 2×2 periodic — control

```bash
python -m scripts.train_lattice --lattice kagome --nx 2 --ny 2 --boundary periodic \
  --epochs 3000 --run-id tier0-v3-0c
```

**Pass criteria:**
- KL < 0.10
- Coverage > 0.50 (of 355 multi-ordering states)
- 0 ice violations

### Tier 0 Results

*To be filled after running experiments.*

---

## Tier 1: Quick Runs (~1–5 min each)

Small β₁, fast convergence, validates the pipeline across lattice types.

### Pre-Tier 1 Checklist

- [ ] Tier 0 experiments all pass
- [ ] Gradient diagnostics look healthy (no NaN, no explosion)

### 1a. Square 4×4 open (same as 0a, included for completeness)
### 1b. Kagome 2×2 open
### 1c. Santa Fe 2×2 open
### 1d. Tetris 2×2 open
### 1e. Shakti 1×1 open

### Tier 1 Results

*To be filled after running experiments.*

---

## Tier 2: Medium Runs (~5–20 min each)

Larger β₁ (15–22), meaningful state spaces.

### Pre-Tier 2 Checklist

- [ ] All Tier 1 experiments pass
- [ ] Review gradient diagnostics from Tier 1: no divergence, advantage variance decreasing
- [ ] Batch size scaling guidance: `batch_size >= 4 × β₁` as a starting heuristic

### 2a. Shakti 2×2 open
### 2b. Kagome 3×3 open
### 2c. Santa Fe 3×3 open
### 2d. Tetris 3×3 open

### Tier 2 Results

*To be filled after running experiments.*

---

## Tier 3: Cross-Lattice Scaling

Push each lattice type to higher β₁. Some experiments are beyond enumeration (β₁ > 25) and rely on entropy, ESS, and stability metrics.

### Pre-Tier 3 Checklist

- [ ] All Tier 2 experiments pass
- [ ] Gradient diagnostics healthy
- [ ] For β₁ > 32, consider `batch_size=256`

### Pass Criteria (non-enumerable, β₁ > 25)

| Metric | Pass | Fail | Notes |
|--------|------|------|-------|
| Ice violations | 0.0 | > 0 | Guaranteed by construction |
| Entropy | Plateaus and stays high | Decreasing or collapsed | No target value without enumeration |
| ESS | > batch_size | < 0.5 × batch_size | Low ESS = mode collapse |
| Grad norm (final 100) | Stable, < 2 × grad_clip | Diverging / NaN | |
| Hamming distance | Reasonable spread (peak 0.2-0.5) | Collapsed to 0 or bimodal | Indicates diversity |

### 3a. Square 6×6 open (enumerable)
### 3b. Kagome 3×3 periodic (non-enumerable)
### 3c. Kagome 4×4 open (non-enumerable)
### 3d. Santa Fe 4×4 open (non-enumerable)
### 3e. Tetris 4×4 open (non-enumerable)

### Tier 3 Results

*To be filled after running experiments.*

---

## Cross-Lattice Comparison

After completing all tiers, compare:

- **KL convergence rate** across lattice types at similar β₁
- **State coverage** vs number of reachable states
- **Training time** vs β₁ (scaling law)
- **Effect of coordination mixing** (square=uniform z=4 vs shakti=mixed z=1-4)
- **Directed-cycle constraint strength** (tetris 3×3 has only 4/2^22 states reachable!)
- **Gradient diagnostics** across lattice types
- **Ensemble variance** — which lattices/sizes show consistent vs variable training outcomes?

---

## KL Noise Floor Reference

The empirical KL estimator KL(empirical || uniform) has a bias from finite sampling of approximately (N-1)/(2n) where N = number of states and n = number of samples:

| States (N) | n=256 noise floor | n=2000 noise floor |
|------------|-------------------|--------------------|
| 4          | 0.006             | 0.001              |
| 38         | 0.072             | 0.009              |
| 112        | 0.217             | 0.028              |
| 299        | 0.582             | 0.075              |
| 355        | 0.692             | 0.089              |
| 4,352      | 8.50              | 1.088              |

For lattices with >50 states, the 256-sample training-time KL is dominated by sampling noise. The 2000-sample final evaluation is the reliable metric. Entropy is a better training-time diagnostic.

---

## Deferred Items

| Item | Trigger | Action |
|------|---------|--------|
| Per-loop baselines (C2) | Advantage variance stays high at β₁ > 15 | Implement loop-specific running-mean baselines |
| Cycle basis reordering (C3) | Slow convergence correlated with loop ordering | Sort loops by directedness frequency |
| Nonlinear input expansion (C1) | Periodic BC + even coordination fails to train | Add `sigma -> [sigma, sigma^2, ...]` input layer |
| Increased depth (C1) | Skip+GELU insufficient for Hamiltonian info | Try n_layers=5 instead of default 3 |
