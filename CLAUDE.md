# CLAUDE-SPINICE.md
## Essential Context for the ASI Topology + Neural Network Oversmoothing Project

> **Purpose of this file:** This is a context document for AI coding agents working on this repository. It contains distilled background from extensive research conversations that produced the companion documents (`spin-ice-tdl-research-sketch.md`, `pillar1-stages-1-2-research-plan.md`). Read this first to understand what we're building and why.

---

## What This Project Is

We are building computational tools to test whether **frustrated lattice topologies from artificial spin ice (ASI) physics** can serve as principled graph substrates that resist **oversmoothing** in graph neural networks.

The central claim: frustrated lattices have large harmonic subspaces (high first Betti number β₁), and signals in the harmonic subspace of the Hodge 1-Laplacian are **mathematically immune to Laplacian-based smoothing** — they sit in the kernel, so L₁h = 0 means zero decay rate through any number of message-passing layers. This is a theorem, not a heuristic.

The project proceeds in stages:
- **Stage 1** (this repo's starting point): Build the ASI lattice zoo as simplicial complexes, compute all spectral properties, catalog β₁ values
- **Stage 2**: Propagate random features through GCN operators on these lattices, measure oversmoothing rates vs. matched unfrustrated controls
- **Stage 3**: Train actual GCNs on a synthetic node classification task (CSBM-on-lattice), produce accuracy-vs-depth curves
- **Stage 4** (future): Move to edge features and Simplicial Convolutional Neural Networks (SCNNs), test harmonic protection directly

---

## The Physics You Need to Know

### Artificial Spin Ice (ASI)

ASI systems place binary Ising-like spins on the **edges** of a 2D lattice. Each spin points along its edge (two possible orientations). At each vertex, the "ice rule" says the number of spins pointing in should equal the number pointing out (or be as close as possible). When the lattice topology makes it impossible to satisfy the ice rule at every vertex simultaneously, we have **frustration**.

Key paper: **Morrison, Nelson & Nisoli, New J. Phys. 15, 045009 (2013)** — introduced a family of vertex-frustrated lattice designs by mixing vertex coordination numbers (z=2, 3, 4) in specific geometric patterns. This is our "lattice zoo."

### The Lattice Zoo

Eight lattice types with distinct frustration properties:

| Lattice | Coordination | Frustration | Key Property |
|---|---|---|---|
| **Square** | All z=4 | Geometric only | Clean baseline, ordered ground state |
| **Kagome** | All z=3 | Geometric | Extensive degeneracy, flat bands, Dirac cones |
| **Shakti** | Mixed z=2,3,4 | Vertex-frustrated, maximal | Extensive degeneracy, topological order, monopole crystallization |
| **Tetris** | Mixed z=2,3,4 | Vertex-frustrated, maximal | "Sliding phase" — ordered in one direction, disordered in other |
| **Pinwheel** | Mixed z=3,4 | Partial (T-loops) | Mixed frustrated/unfrustrated loops |
| **Santa Fe** | Mixed z=2,3,4 | Both types | Polymer-like strings of unhappy vertices |
| **Staggered Shakti** | Mixed z=2,3,4 | Vertex-frustrated | Striped ground state with forced defects |
| **Staggered Brickwork** | Mixed z=2,3 | Trivial | Degeneracy reduces to independent spins — **null model** |

**Why this zoo matters for ML:** These lattices provide a principled design space parametrized by coordination mix, loop topology, frustration type, and ground-state structure. Unlike random graphs (Erdős–Rényi, Barabási–Albert) or application graphs (molecules, social networks), these have *analytically characterized* spectral and homological properties. We know what β₁ should be before we compute it.

### Nisoli's Charge Framework

Nisoli (2020) showed that the entropic interaction between vertex charges (ice-rule violations) in ASI is mediated by the pseudoinverse of the graph Laplacian: the interaction kernel is T·L₀⁻¹. This is the same operator that appears in GCN aggregation. The connection is structural, not analogical.

Key correspondence:
- Spin configuration on edges ↔ edge feature vector (1-cochain)
- Vertex charge Q_v = Σ(spins in) - Σ(spins out) ↔ divergence B₁ᵀ · f
- Ice rule (Q_v = 0) ↔ divergence-free signal (f ∈ ker(B₁ᵀ))
- Ice manifold (degenerate ground states) ↔ ker(L₁) = harmonic subspace
- Monopole excitation ↔ non-zero divergence at a vertex

---

## The Math You Need to Know

### Simplicial Complexes and Boundary Operators

A simplicial complex on a graph has three levels for our purposes:
- **0-simplices**: vertices (n₀ of them)
- **1-simplices**: oriented edges (n₁ of them)
- **2-simplices**: faces / filled loops (n₂ of them)

**B₁** (n₀ × n₁): vertex-edge incidence matrix
```
B₁[v, e] = +1 if v is the head of edge e
B₁[v, e] = -1 if v is the tail of edge e
```

**B₂** (n₁ × n₂): edge-face incidence matrix
```
B₂[e, f] = +1 if edge e appears in face f with consistent orientation
B₂[e, f] = -1 if edge e appears in face f with opposite orientation
```

### Laplacians

```python
L0 = B1 @ B1.T           # Graph (vertex) Laplacian, shape (n0, n0)
L1_down = B1.T @ B1      # Lower edge Laplacian, shape (n1, n1)
L1_up = B2 @ B2.T        # Upper edge Laplacian, shape (n1, n1)
L1 = L1_down + L1_up     # Full Hodge 1-Laplacian, shape (n1, n1)
```

All are symmetric positive semi-definite.

### The Hodge Decomposition (the core of everything)

The edge signal space ℝ^n₁ decomposes into three orthogonal subspaces:

```
ℝ^n₁ = im(B₁ᵀ) ⊕ im(B₂) ⊕ ker(L₁)
        ────────   ──────   ────────
        gradient    curl    harmonic
```

- **Gradient** (im(B₁ᵀ)): edge signals that are gradients of vertex potentials. Dimension = n₀ - β₀.
- **Curl** (im(B₂)): edge signals that are boundaries of face signals. Dimension = n₂ - β₂.
- **Harmonic** (ker(L₁)): edge signals in the kernel of L₁. Dimension = **β₁** (first Betti number).

β₁ = n₁ - rank(B₁) - rank(B₂). It counts independent cycles not bounded by faces.

**Euler characteristic check:** β₀ - β₁ + β₂ = n₀ - n₁ + n₂

### Why β₁ Matters for Oversmoothing

Any polynomial filter p(L₁) applied to an edge signal f = f_grad + f_curl + f_harm gives:
```
p(L₁)f = p(L₁)f_grad + p(L₁)f_curl + p(0)·f_harm
```

The gradient and curl components get filtered (smoothed/attenuated) by nonzero eigenvalues. The harmonic component is multiplied by the scalar p(0). For standard GCN-style filters where p(0) ≠ 0, **the harmonic component passes through unchanged regardless of network depth.** This is the "topologically protected channel."

Lattices with larger β₁ → more dimensions of protected channel → more information survives deep networks.

### Projectors

```python
P_grad = B1.T @ pinv(B1 @ B1.T) @ B1           # onto im(B₁ᵀ)
P_curl = B2 @ pinv(B2.T @ B2) @ B2.T            # onto im(B₂)
P_harm = I - P_grad - P_curl                     # onto ker(L₁)
```

These will be needed in Stage 4 to decompose learned representations. For Stage 1, we just need the eigendecomposition.

---

## The Oversmoothing Problem (ML Context)

**The problem:** GCN convolution is Laplacian smoothing (Li et al. 2018). Repeated application drives all vertex features toward the leading eigenvector of the normalized adjacency. Representations become indistinguishable. Practical GNN depth is limited to ~2-5 layers.

**On Cora:** accuracy drops from 81% (2 layers) to 20% (6 layers) — worse than random.

**Why depth matters:** Many tasks require information from distant nodes (long-range dependencies). Without depth, the receptive field is too small. But with depth, oversmoothing kills you.

**Existing mitigations:** Residual connections (JKNet), DropEdge, PairNorm, graph sparsification. All treat the symptom by modifying the message-passing mechanism. None address the topological root cause.

**Our approach:** Attack the topology itself. Standard graphs lack a protected zero-eigenvalue subspace for edge signals. ASI frustrated topologies provide one, with β₁ as the tunable control parameter.

---

## Stage 1: What We're Building

### Goal

For each lattice in the zoo, at multiple system sizes: construct the simplicial complex, compute all Laplacians, perform full eigendecomposition, extract β₁ and harmonic basis vectors. Produce a complete spectral catalog.

### Lattice Construction

Each lattice is defined by a **unit cell** (vertices + edges in a tileable pattern). The unit cell is repeated Nx × Ny times with periodic or open boundary conditions. Implementation should be a `LatticeGenerator` class with a method per lattice type.

**Critical implementation detail:** The minimal loops (faces/2-cells) of a planar lattice correspond to the faces of the planar embedding. For planar graphs, these can be found via face-finding on the planar embedding. Each face is a candidate 2-cell for the simplicial complex.

**Two face-filling strategies bracket the range of β₁:**
- **Fill all faces** → maximal n₂, minimal β₁ (most cycles become trivial in homology)
- **Fill no faces** → n₂ = 0, L₁_up = 0, maximal β₁ (all independent cycles are homologically nontrivial)

### System Sizes

| Label | Unit Cells | Approx n₀ | Approx n₁ | Method |
|---|---|---|---|---|
| XS | 4×4 | 16-128 | 32-200 | Dense eigh (debug/viz) |
| S | 10×10 | 100-800 | 200-1200 | Dense eigh |
| M | 20×20 | 400-3200 | 800-4800 | Dense eigh (all under 15k threshold) |
| L | 50×50 | 2500-20000 | 5000-30000 | Dense for n≤15k, sparse otherwise |
| XL | 100×100 | 10000-80000 | 20000-120000 | Dense for n≤15k (square L0), sparse otherwise |

**Full spectrum availability** (with DENSE_THRESHOLD=15,000):

| Lattice | Full L0 sizes | Full L1 sizes |
|---|---|---|
| Square | S, M, L, XL (n₀≤10k) | S, M, L (n₁≤5k) |
| Kagome | S, M, L (n₀≤7.5k) | S, M, L (n₁≤15k) |
| Shakti | S, M (n₀≤6.4k) | S, M (n₁≤9.6k) |
| Tetris | S, M (n₀≤3.2k) | S, M (n₁≤4.8k) |
| Santa Fe | S, M, L (n₀≤15k) | S, M (n₁≤3.6k) |

### What to Compute and Store

For each lattice × size × face strategy:

1. **Incidence matrices** B₁, B₂ (sparse, CSC format)
2. **All four Laplacians** L₀, L₁, L₁_down, L₁_up (sparse)
3. **Full eigenvalue spectrum** (dense for S/M, k-smallest for L/XL)
4. **Betti numbers** β₀, β₁ (count eigenvalues < 1e-10)
5. **Spectral gaps** of L₀ and L₁ (smallest nonzero eigenvalue)
6. **Harmonic basis vectors** (eigenvectors of L₁ with eigenvalue ≈ 0)
7. **β₁ scaling** — how β₁ grows with system size (extensive? boundary? topological?)

### Validation Checks

- β₀ + β₂ - β₁ = n₀ - n₁ + n₂ (Euler characteristic)
- β₁ = n₁ - rank(B₁) - rank(B₂)
- L₁ @ h ≈ 0 for each harmonic eigenvector h
- B₁.T @ h ≈ 0 and B₂.T @ h ≈ 0 for each harmonic h
- β₀ = 1 for connected graphs (periodic BCs)

### Key Outputs

**Table: Lattice Zoo Summary** — per-unit-cell counts, coordination distribution, β₁ under both face strategies

**Table: Spectral Properties at Size M** — β₁, spectral gaps, fraction harmonic (β₁/n₁)

**Table: β₁ Scaling Laws** — β₁ vs. system size, fitted scaling exponent

**Figures:** Lattice gallery (colored by coordination), eigenvalue histograms of L₁, spectral density overlay across lattices, β₁ scaling plots, harmonic mode visualizations on the lattice

---

## Tech Stack

- **Python 3.10+** in a venv
- **numpy** — array operations
- **scipy** — sparse matrices (`scipy.sparse`), eigensolvers (`scipy.sparse.linalg.eigsh` for large, `scipy.linalg.eigh` for dense)
- **networkx** — graph construction, planar embedding, face finding, configuration model for controls
- **matplotlib** — all visualization
- **pandas** — results tables
- **torch + torch_geometric** — needed for Stages 2-3 (GCN training), not Stage 1

---

## Repo Structure (Suggested)

```
spinice-topology/
├── CLAUDE-SPINICE.md          ← this file
├── docs/
│   ├── spin-ice-tdl-research-sketch.md   ← full research proposal
│   └── pillar1-stages-1-2-research-plan.md ← detailed Stage 1-3 plan
├── src/
│   ├── lattices/
│   │   ├── __init__.py
│   │   ├── generator.py       ← LatticeGenerator class
│   │   ├── square.py          ← unit cell definitions
│   │   ├── kagome.py
│   │   ├── shakti.py
│   │   ├── tetris.py
│   │   ├── pinwheel.py
│   │   ├── santa_fe.py
│   │   ├── staggered_shakti.py
│   │   └── staggered_brickwork.py
│   ├── topology/
│   │   ├── __init__.py
│   │   ├── incidence.py       ← B₁, B₂ construction from graph + faces
│   │   ├── laplacians.py      ← L₀, L₁, L₁_down, L₁_up
│   │   └── hodge.py           ← Hodge projectors, harmonic basis
│   ├── spectral/
│   │   ├── __init__.py
│   │   ├── eigensolve.py      ← dense/sparse eigendecomposition
│   │   ├── betti.py           ← β₁ computation, validation checks
│   │   └── catalog.py         ← run full catalog, save results
│   └── viz/
│       ├── __init__.py
│       ├── lattice_plots.py   ← graph drawing, coordination coloring
│       └── spectral_plots.py  ← eigenvalue histograms, scaling plots
├── notebooks/
│   └── 01_spectral_catalog.ipynb  ← interactive exploration
├── results/                   ← saved spectra, tables, figures
├── tests/
│   ├── test_lattices.py       ← unit cell counts match Morrison et al.
│   ├── test_incidence.py      ← B₁, B₂ properties (B₁B₂ = 0, etc.)
│   ├── test_euler.py          ← Euler characteristic check
│   └── test_harmonic.py       ← L₁h ≈ 0 verification
└── requirements.txt
```

---

## Common Pitfalls and Gotchas

### Lattice Construction
- **Edge orientation matters.** B₁ and B₂ depend on a consistent global edge orientation. Choose a convention (e.g., left→right, bottom→top, lower-index→higher-index) and stick to it.
- **Periodic boundary conditions** create wrap-around edges. These change the topology (a periodic square lattice is a torus, β₁ = 2 even with all faces filled). Be explicit about boundary conditions in all results.
- **Morrison et al. figures** are the ground truth for unit cell definitions. Some lattices (shakti, pinwheel) have intricate connectivity that's easy to get wrong. Validate by checking n₀/cell, n₁/cell, and coordination distribution against the paper.

### Face Finding
- **Planar embedding is required** to correctly identify faces. NetworkX has `nx.check_planarity()` which returns a planar embedding if one exists. The faces of the embedding are the minimal cycles.
- **Outer face:** Planar embeddings include an infinite outer face. Exclude it from the 2-cell list (or handle it carefully with periodic BCs where there is no outer face).
- **With periodic BCs on a torus**, the graph is not planar. You need to identify faces from the unit cell structure directly, not from planarity. The unit cell's faces tile periodically.

### Numerical Issues
- **Null space detection:** Eigenvalues "equal to zero" will be ~1e-14 in floating point. Use a threshold (1e-10 recommended). Count eigenvalues below threshold as zero to get β₁.
- **Large sparse eigensolves:** Use `scipy.sparse.linalg.eigsh(L1, k=50, which='SM')` (smallest magnitude) for the bottom of the spectrum. For the null space specifically, shift-invert mode (`sigma=0`) is more reliable.
- **Pseudoinverse** for projectors: use `scipy.linalg.pinvh` for symmetric matrices, or compute from the eigendecomposition directly.

### Eigensolver Tuning (current settings in `src/spectral/eigensolve.py`)
- **DENSE_THRESHOLD = 15,000**: Matrices up to 15k×15k use dense `scipy.linalg.eigh` for full spectrum. Raised from 5,000 after benchmarking on M1 Pro (16 GB). At n=15k, dense takes ~100-200s and ~3.4 GB RAM. See `docs/eigensolver-research.md` for full analysis.
- **driver='evd'**: Dense solver uses LAPACK DSYEVD (divide-and-conquer), 2-4x faster than the default 'evr' for full eigendecomposition. Uses ~50% more workspace but well within limits at n≤15k.
- **SHIFT_INVERT_THRESHOLD = 50,000**: Shift-invert mode (`sigma=0`) uses LU factorization which can OOM on large matrices. For 2D lattice Laplacians the fill-in is O(n·sqrt(n)), so shift-invert is safe up to ~50k. Above this, falls back to direct `which='SM'` method.
- **Sparse solver** (`eigsh`, k=100): For matrices above DENSE_THRESHOLD, only the 100 smallest eigenvalues are computed. This means **no full spectrum** at L/XL sizes for most lattices.
- **Future options** for pushing full spectra to larger sizes: spectrum slicing (divide eigenvalue range into intervals, shift-invert at each), PRIMME library, or KPM for approximate DOS. Details in `docs/eigensolver-research.md`.

### Controls (for Stages 2-3)
- **Configuration model** (`nx.configuration_model(degree_sequence)`) preserves the exact degree sequence but produces multigraphs with self-loops. Remove self-loops and multi-edges, which slightly perturbs the degree sequence.
- **Maslov-Sneppen rewiring** is a better control: start from the real lattice, perform random degree-preserving edge swaps. This destroys spatial/frustration structure while keeping the graph connected and simple.
- Generate 10 random realizations of each control for error bars.

---

## Key References

Computational and theoretical references most relevant to implementation:

1. **Morrison, Nelson & Nisoli** (2013), New J. Phys. 15, 045009 — The lattice zoo. Unit cell definitions, coordination tables, frustration classification.
2. **Nisoli** (2020), New J. Phys. 22, 103052 — Charge framework, L₀⁻¹ interaction kernel. Establishes the Laplacian connection.
3. **Lao et al.** (2018), Nature Physics 14, 723 — Topological order in shakti ice. Experimental observation of the protected defects our math predicts.
4. **Yang, Isufi & Leus** (2022), IEEE TSP — SCNN architecture. Defines the L₁_down/L₁_up filter decomposition we will use in Stage 4.
5. **Papillon et al.** (2024), JMLR — TopoX library. Reference implementation of simplicial complexes and Hodge Laplacians in Python. May be useful but we may want our own lighter implementation.
6. **Li, Han & Wu** (2018), AAAI — First identification of GCN oversmoothing as Laplacian smoothing. The 81%→20% Cora result.
7. **Rusch et al.** (2023), arXiv — Comprehensive survey/formal treatment of oversmoothing. Covers spectral analysis.
8. **Duranthon & Zdeborová** (2025), Phys. Rev. X — Statistical physics analysis of deep GCNs on CSBM. Shows depth scaling with spectral gap. Most relevant recent theoretical work.
9. **Wu et al.** (2022), ICLR — Non-asymptotic oversmoothing analysis on CSBM. First finite-depth theory.

---

## What Success Looks Like

**Stage 1 is complete when we have:**
1. All 8 lattice types generating correctly at sizes XS through XL
2. B₁, B₂ matrices validated (B₁B₂ = 0, correct dimensions)
3. Euler characteristic checks passing for all lattices
4. Complete eigenvalue spectra for L₀ and L₁ at sizes S and M
5. β₁ values tabulated for all lattices under both face strategies
6. β₁ scaling laws fitted (β₁ vs. N for each lattice)
7. Harmonic basis vectors extracted and verified (L₁h ≈ 0)
8. Gallery figure showing all lattices, spectral comparison figure showing eigenvalue distributions

**The key scientific question Stage 1 answers:** Which lattices have large β₁ (extensive scaling with system size) and which have small β₁? This tells us which topologies have the most "protected channel capacity" for the later stages. We expect shakti and tetris (maximally frustrated) to have the largest β₁, and staggered brickwork (trivially frustrated) to have the smallest. Confirming this computationally — and quantifying the exact scaling — is the deliverable.

---

## Current Status and Dashboard

### Spectral Catalog
The spectral catalog (`results/catalog/`) stores precomputed results as `{key}.npz` + `{key}_meta.json` pairs. The catalog covers 5 lattices (square, kagome, shakti, tetris, santa_fe) at sizes XS through XL, both face strategies (all/none), and both boundary conditions (periodic/open). Recomputed with DENSE_THRESHOLD=15,000 and driver='evd'.

### Interactive Dashboard (`dashboard/`)
A Dash/Plotly dashboard for exploring results, run via `python -m dashboard.app`:

- **Home** (`/`): Lattice zoo overview, β₁ summary table
- **Spectra** (`/spectra`): Eigenvalue histograms for individual lattices, spectral overlay across lattice types, and **DOS convergence** (spectral distribution function: sorted eigenvalues vs i/n) across system sizes
- **Spectral Gap** (`/spectral-gap`): Spectral gap scaling Δ ~ c/L², fits for stiffness c and normalized c* = c/(4π²), bar chart comparison across lattices

### Key Spectral Results
- **Spectral distribution plots** (`results/dos_*.png`): Sorted eigenvalues vs normalized index (i/n₀ or i/n₁). Convergence of these curves = approach to thermodynamic-limit integrated DOS. Generated for all lattices with 2+ full-spectrum sizes.
- **Spectral gap scaling**: c* ≈ 1.0 for square (matching analytical 1D chain result). XS (L=4) excluded from fits due to strong finite-size corrections.
- **Finite-size effects**: XS points have O(1/L⁴) corrections that bias linear fits in 1/L². Standard practice: exclude L < 10 from scaling fits.
- **Flat bands**: Kagome L0 shows exact eigenvalue = 6 for top ~33% of spectrum (z=3 coordination). Shakti shows sub-band structure from mixed coordination z=2,3,4.
