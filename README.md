# SpinIceTDL

Topological deep learning meets artificial spin ice: neural variational samplers for frustrated lattice systems, built on the spectral and homological structure of the Hodge Laplacian.

## What This Is

This project connects two ideas:

1. **Artificial spin ice (ASI)** lattices have large harmonic subspaces (high first Betti number β₁) due to geometric frustration — the ice rule forces extensive ground-state degeneracy.
2. **Graph neural networks** suffer from oversmoothing because Laplacian diffusion kills information — but signals in the harmonic subspace of the Hodge 1-Laplacian are mathematically immune (L₁h = 0).

We build computational infrastructure to exploit this connection: a spectral catalog of frustrated lattice topologies (Stage 1), and neural variational samplers that learn to generate ice-rule-satisfying spin configurations (Mode A, complete) and finite-temperature Boltzmann samples (Mode B, planned).

## Current State

### Stage 1: Spectral Catalog (Complete)

Eight ASI lattice types (square, kagome, shakti, tetris, pinwheel, santa_fe, staggered_shakti, staggered_brickwork) at sizes XS–XL. Full eigendecomposition of L₀ and L₁, β₁ scaling laws, spectral gap fits. Interactive Dash dashboard for exploration.

### Mode A: LoopMPVAN Ice-Manifold Sampler (Complete)

Autoregressive sampling over β₁ loop-flip decisions. Every sample is a valid ice state by construction — no rejection needed.

**Architecture:** EIGN dual-channel message passing (equivariant + invariant) on fully-assigned spin configurations → pool over loop edges → MLP → flip probability. Directed-cycle gating ensures only physically flippable loops are considered.

**Validation on square XS (4×4, open BC):**
- 25/25 reachable states covered (100%)
- Zero ice-rule violations
- KL(empirical || uniform) ≈ 0.006
- 57/57 tests passing

Key modules: `src/neural/` (model, training, metrics, enumeration, checkpointing, diagnostics), `src/sampling/` (MCMC benchmarks).

### Mode B: Direct Edge Sampler (Planned)

Autoregressive sampling of individual edge spins with causal masking and soft ice-rule penalty. Enables finite-temperature Boltzmann sampling where monopole excitations are permitted but energetically penalized. Temperature annealing from high-T (disordered) to low-T (ice-rule-enforced).

## Quick Start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Train Mode A on square XS
python -m scripts.train_xs_validation --boundary open --epochs 500 --batch-size 32

# Generate diagnostic plots from a training run
python -m scripts.plot_training_diagnostics --run-dir results/neural_training/{run_id}

# Compare neural vs MCMC sampling
python -m scripts.compare_samplers --lattice square --size XS --boundary open

# Launch spectral catalog dashboard
python -m dashboard.app
```

## Repository Structure

```
src/
  lattices/       # Lattice zoo: 8 ASI lattice generators with open/periodic BC
  topology/       # B₁, B₂ incidence matrices, Laplacians, Hodge decomposition, ice sampling
  spectral/       # Eigensolvers, Betti numbers, spectral catalog
  neural/         # Mode A: LoopMPVAN, EIGN layers, training, metrics, checkpointing, plots
  sampling/       # MCMC benchmarks and comparison infrastructure
  viz/            # Matplotlib publication figures
scripts/          # Training, plotting, benchmarking CLI entrypoints
dashboard/        # Interactive Dash/Plotly spectral explorer
tests/            # Unit tests for all modules
results/          # Spectral catalog, training runs, figures
```

## References

- Morrison, Nelson & Nisoli (2013), New J. Phys. 15, 045009 — ASI lattice zoo
- Nisoli (2020), New J. Phys. 22, 103052 — Charge framework and Laplacian connection
- Li, Han & Wu (2018), AAAI — GCN oversmoothing as Laplacian smoothing
- Yang, Isufi & Leus (2022), IEEE TSP — Simplicial convolutional neural networks
