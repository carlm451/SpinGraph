# Mode A Training Experiments

## Overview

Systematic training of LoopMPVAN across the lattice zoo at small sizes. All experiments use `scripts/train_lattice.py` (generalized to any lattice type).

**Activate venv first:**
```bash
source venv/bin/activate
```

**Generate plots after any run:**
```bash
python -m scripts.plot_training_diagnostics --run-dir results/neural_training/{run_id}
```

---

## Lattice Survey at Small Sizes

| Lattice   | Size  | n₀   | n₁  | β₁ | Reachable States | Enumerable | Est. Train Time (2000 ep) |
|-----------|-------|------|-----|-----|-----------------|------------|---------------------------|
| square    | 4×4   | 16   | 24  | 9   | 25              | yes        | ~6 min                    |
| square    | 6×6   | 36   | 60  | 25  | 3,029           | yes        | ~70 min                   |
| kagome    | 2×2   | 12   | 17  | 6   | 32              | yes        | ~2 min                    |
| kagome    | 3×3   | 27   | 43  | 17  | 4,352           | yes        | ~30 min                   |
| santa_fe  | 2×2   | 24   | 30  | 7   | 8               | yes        | ~4 min                    |
| santa_fe  | 3×3   | 54   | 72  | 19  | 48              | yes        | ~35 min                   |
| shakti    | 2×2   | 64   | 81  | 18  | 112             | yes        | ~30 min                   |
| tetris    | 2×2   | 32   | 38  | 7   | 12              | yes        | ~4 min                    |
| tetris    | 3×3   | 72   | 93  | 22  | 4               | yes        | ~40 min                   |

Time estimates based on square XS (β₁=9, 500ep = 188s) scaling roughly as β₁ × n₁ × epochs.

---

## Tier 1: Quick Runs (~2-6 min each)

Small β₁, fast convergence, good for validating the pipeline works across lattice types.

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

### 1c. Santa Fe 2×2 open
```bash
python -m scripts.train_lattice --lattice santa_fe --nx 2 --ny 2 --boundary open --epochs 1000
```
**Expected:** β₁=7, only 8 states. Very small state space, fast convergence.

### 1d. Tetris 2×2 open
```bash
python -m scripts.train_lattice --lattice tetris --nx 2 --ny 2 --boundary open --epochs 1000
```
**Expected:** β₁=7, 12 states. Similar to santa_fe.

### 1e. Shakti 1×1 open
```bash
python -m scripts.train_lattice --lattice shakti --nx 1 --ny 1 --boundary open --epochs 500
```
**Expected:** β₁=2, trivial. Smoke test for shakti pipeline.

---

## Tier 2: Medium Runs (~20-40 min each)

Larger β₁, meaningful state spaces. These test whether the model can learn to cover hundreds–thousands of states.

### 2a. Shakti 2×2 open
```bash
python -m scripts.train_lattice --lattice shakti --nx 2 --ny 2 --boundary open --epochs 2000
```
**Expected:** β₁=18, 112 states. Mixed coordination (z=1,2,3,4). Key test of EIGN on frustrated lattice.

### 2b. Kagome 3×3 open
```bash
python -m scripts.train_lattice --lattice kagome --nx 3 --ny 3 --boundary open --epochs 2000
```
**Expected:** β₁=17, 4,352 states. Large state space — good stress test for coverage.

### 2c. Santa Fe 3×3 open
```bash
python -m scripts.train_lattice --lattice santa_fe --nx 3 --ny 3 --boundary open --epochs 2000
```
**Expected:** β₁=19, only 48 states. Compact manifold despite moderate β₁.

### 2d. Tetris 3×3 open
```bash
python -m scripts.train_lattice --lattice tetris --nx 3 --ny 3 --boundary open --epochs 2000
```
**Expected:** β₁=22, only 4 states! Extreme directed-cycle constraint. Interesting test case.

---

## Tier 3: Long Runs (~60-90 min each)

Pushing β₁ to 25. These may require GPU or reduced epochs if CPU is too slow.

### 3a. Square 6×6 open
```bash
python -m scripts.train_lattice --lattice square --nx 6 --ny 6 --boundary open --epochs 2000
```
**Expected:** β₁=25, 3,029 states. Partial run showed KL=0.028 at epoch 200 (entropy nearly converged).

---

## Key Metrics to Track

For each run, record:

| Metric | Target | Notes |
|--------|--------|-------|
| Ice violations | 0.0 | Should always be zero (Mode A guarantee) |
| KL(empirical \|\| uniform) | < 0.05 | Lower is better; measures uniformity |
| State coverage | 1.0 | Fraction of reachable states sampled at least once |
| ESS | > batch_size | Effective sample size; > 2× batch is good |
| Entropy | → ln(N_states) | Should approach target for uniform distribution |
| Training time | — | Wall clock for timing feasibility |
| Sample time | — | Time to generate 2000 post-training samples |

---

## GPU Feasibility Notes

If CPU training is too slow for Tier 2/3:

1. **Current bottleneck:** Each epoch does `batch_size` sequential forward passes through the EIGN stack (one per autoregressive loop step × batch). No batching across samples in the current implementation.

2. **GPU would help if:** We batch the EIGN forward pass across samples. Currently the `sample()` and `forward_log_prob()` loops are sample-by-sample. Batching the EIGN stack (which is just sparse matrix multiplies + linear layers) across B samples would give ~B× speedup on GPU.

3. **Quick GPU test:** If you have CUDA available, the model already uses PyTorch and should move to GPU with `.to('cuda')`. The bottleneck is the Python loop over samples, not the tensor ops.

4. **Estimated speedup from batching:** For β₁=25, batch_size=64: current approach does 25×64=1600 sequential EIGN forward passes per epoch. With batched implementation: 25 batched passes. ~64× speedup on the forward pass.

---

## Comparison Across Lattices

After completing Tier 1+2 runs, compare:

- **KL convergence rate** across lattice types at similar β₁
- **State coverage** vs number of reachable states
- **Training time** vs β₁ (scaling law)
- **Effect of coordination mixing** (square=uniform z=4 vs shakti=mixed z=1-4)
- **Directed-cycle constraint strength** (tetris 3×3 has only 4/2^22 states reachable!)
