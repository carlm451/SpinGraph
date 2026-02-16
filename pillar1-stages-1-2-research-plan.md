# Pillar 1, Stages 1–3: Detailed Research Plan
## Spectral Catalog, Oversmoothing Baseline, and Trained GCN Experiments

*C. M. — February 2026*

---

## Overview

This document breaks down the first three stages of Pillar 1 into implementable steps. Stages 1 and 2 are purely computational — no training data required. Stage 1 builds the lattice zoo as simplicial complexes and computes their complete spectral characterization. Stage 2 feeds random features through GCN layers on these lattices (and matched unfrustrated controls) and measures how fast feature diversity dies as a function of depth. Stage 3 graduates to actual PyTorch training: we plant a standard synthetic node classification task (adapted from the Contextual Stochastic Block Model) on the ASI lattice topologies, train real GCNs, and produce the classic "accuracy vs. depth" curves that directly show whether frustrated topologies allow deeper networks. The combined output establishes the relationship between lattice topology, frustration, spectral structure, and oversmoothing rate — the empirical foundation on which Stage 4 (SCNN harmonic protection on edge features) builds.

**What we need before starting:** Python environment with `networkx`, `scipy` (sparse linear algebra), `numpy`, `torch`, `torch_geometric`. No GPU required for Stage 1. Stage 2 uses GPU for GCN forward passes but is lightweight.

**Mathematical reference:** The exact correspondences between Nisoli's spin ice formalism and TDL are derived with full proofs and worked examples in `docs/tdl-spinice-correspondence.html`. That document establishes the identities (Q = B₁σ, H = (ε/2)σᵀL₁ᵈᵒʷⁿσ, ker(B₁) = im(B₂) ⊕ ker(L₁), etc.) that this plan implements computationally.

---

## Stage 1 — Spectral Catalog

### 1.1 Lattice Construction

Each ASI lattice is defined by a unit cell: a set of vertices and oriented edges in a 2D tile that can be repeated on a periodic grid. The construction for each lattice type is:

#### 1.1.1 Square Ice

The simplest case. Vertices sit on a square grid. Edges connect nearest neighbors (horizontal and vertical). Every vertex has coordination z=4.

```
Unit cell: 1 vertex, 2 edges (one horizontal, one vertical)
Tiling: Nx × Ny repetitions
Resulting graph: Nx·Ny vertices, 2·Nx·Ny edges (with periodic BCs)
                 or (2·Nx·Ny - Nx - Ny) edges (open BCs)
Minimal loops: all squares (4-cycles), one per unit cell
```

#### 1.1.2 Kagome Ice

Vertices sit at the midpoints of the edges of a triangular lattice. Equivalently: corner-sharing triangles. Every vertex has z=3.

```
Unit cell: 3 vertices, 6 edges
Tiling: Nx × Ny repetitions of the unit cell
Resulting graph: 3·Nx·Ny vertices, 6·Nx·Ny edges (periodic)
Minimal loops: triangles (3-cycles) and hexagons (6-cycles)
```

#### 1.1.3 Shakti

The key vertex-frustrated lattice. Built from a square lattice with additional vertices inserted at the centers of alternating squares, connected to all four corners of that square. This creates a mix of z=4 (corner vertices), z=4 (center vertices in connected squares), z=3 (some edges), and z=2 (some bridge vertices). The exact construction follows Morrison et al. (2013) Fig. 2.

```
Unit cell: ~8 vertices, ~12 edges (varies by exact definition)
Coordination mix: z=2, z=3, z=4 vertices present
Tiling: Nx × Ny
Minimal loops: squares (4-cycles), triangular sub-loops, and composite loops
Frustrated loops: those containing a vertex where ice rules cannot be simultaneously satisfied
```

#### 1.1.4 Tetris

Similar to shakti but with the additional vertices inserted in a striped pattern (every other row). Creates alternating "active" rows (with extra connectivity) and "passive" rows (standard square).

```
Unit cell: ~6 vertices, ~9 edges
Coordination mix: z=2, z=3, z=4
Minimal loops: mixed sizes
Key feature: translation symmetry broken in one direction → "sliding phase"
```

#### 1.1.5 Remaining Lattices

Pinwheel, Santa Fe, staggered shakti, and staggered brickwork are similarly defined by their unit cells from Morrison et al. (2013) Figures 2–4. Each requires reading the vertex positions and edge connectivity from the paper figures. The staggered brickwork is the simplest non-trivial case (z=2,3 only) and serves as a "trivially frustrated" null model.

#### 1.1.6 Implementation Strategy

Write a single `LatticeGenerator` class with a method per lattice type:

```python
class LatticeGenerator:
    def __init__(self, lattice_type: str, nx: int, ny: int, 
                 boundary: str = 'periodic'):
        """
        lattice_type: 'square', 'kagome', 'shakti', 'tetris', 
                      'pinwheel', 'santa_fe', 'staggered_shakti', 
                      'staggered_brickwork'
        nx, ny: number of unit cell repetitions
        boundary: 'periodic' or 'open'
        """
        
    def build_graph(self) -> nx.Graph:
        """Returns networkx graph with positions as node attributes."""
        
    def get_vertices(self) -> np.ndarray:
        """Vertex coordinates, shape (n_vertices, 2)."""
        
    def get_edges(self) -> np.ndarray:
        """Edge list with orientations, shape (n_edges, 2)."""
        
    def get_coordination_numbers(self) -> dict:
        """Returns {vertex_id: z} for all vertices."""
        
    def get_minimal_loops(self) -> list:
        """Returns list of minimal cycles (as ordered edge lists).
        Each cycle is a candidate 2-cell for the simplicial complex."""
        
    def summary(self) -> dict:
        """Returns {n_vertices, n_edges, n_loops, 
                    coordination_distribution, mean_loop_length}."""
```

**Minimal loop detection:** For periodic planar lattices, the minimal loops (faces of the planar graph) can be found by the planar face-finding algorithm. For each lattice embedded in the plane, compute the dual graph — each face of the planar embedding becomes a 2-cell. In `networkx`, this is available via `nx.planar_layout` and face traversal, or more robustly via computing the cycle basis and identifying the minimal (facial) cycles of the planar embedding.

**Validation:** For each lattice type, print the summary statistics and visually inspect the generated graph at small sizes (4×4 unit cells) using `matplotlib`. Cross-reference vertex counts, edge counts, and coordination distributions against Morrison et al. Table 1.

### 1.2 Simplicial Complex Construction

From each graph + its minimal loops, build the simplicial complex:

```
0-simplices: vertices           → n₀ = |V|
1-simplices: oriented edges     → n₁ = |E|
2-simplices: minimal loops      → n₂ = |F| (faces)
```

Choose a global edge orientation (e.g., left-to-right for horizontal edges, bottom-to-top for vertical, breaking ties by vertex index). This fixes the incidence matrices.

#### 1.2.1 Incidence Matrices

**B₁ (vertex-edge incidence), shape (n₀ × n₁):**
```
B₁[v, e] = +1  if vertex v is the head of oriented edge e
B₁[v, e] = -1  if vertex v is the tail of oriented edge e
B₁[v, e] =  0  otherwise
```

**B₂ (edge-face incidence), shape (n₁ × n₂):**
```
For each face f (a minimal loop traversed in a chosen orientation):
B₂[e, f] = +1  if edge e appears in f with consistent orientation
B₂[e, f] = -1  if edge e appears in f with opposite orientation
B₂[e, f] =  0  if edge e is not in face f
```

**Implementation:** Store B₁ and B₂ as `scipy.sparse.csc_matrix`. For lattices up to 100×100 unit cells (~10⁴ vertices, ~2×10⁴ edges), these fit comfortably in memory.

**Critical property:** B₁ B₂ = 0 (the chain complex condition). This must be verified for every lattice as a basic correctness check.

#### 1.2.2 Connection to Nisoli's Spin Ice Framework

The incidence matrix B₁ is the bridge between TDL and Nisoli's spin ice formalism. The exact correspondences (derived in detail in `docs/tdl-spinice-correspondence.html`) are:

**Topological charge as divergence.** A spin configuration σ ∈ {±1}^n₁ assigns a binary orientation to each edge. The topological charge at vertex v is:

```
Q_v = (B₁ σ)_v = Σ_e [B₁]_{v,e} · σ_e
```

This is Nisoli's Eq. (1) rewritten in matrix form. The ice rule (Q_v = 0 at all vertices) becomes σ ∈ ker(B₁).

**Ice manifold — even vs. odd degree.** For vertices with even coordination z_v, the ice rule gives Q_v = 0. For odd-degree vertices (z_v = 3), the minimum achievable |Q_v| = 1, not 0. The general ice manifold is defined by |Q_v|_min = z_v mod 2 at each vertex. This matters for kagome (all z=3) and mixed-coordination lattices (shakti, tetris, pinwheel, santa fe) where odd-degree vertices are present. For even-degree lattices, the ice manifold = ker(B₁) exactly; for odd-degree lattices it is an affine coset σ₀ + ker(B₁).

**Nisoli's S matrix reconstruction.** Nisoli's antisymmetric spin matrix S_{vv'} can be reconstructed from B₁ and σ via:

```
S = skew(|B₁| · diag(σ) · B₁ᵀ)  =  ½(M − Mᵀ)
where M = |B₁| · diag(σ) · B₁ᵀ
```

The same matrix M also encodes the charge vector on its diagonal: diag(M) = Q. This unification (S and Q are the antisymmetric and diagonal parts of the same matrix M) is a key structural insight.

**Hamiltonian as L₁ᵈᵒʷⁿ quadratic form.** The dumbbell Hamiltonian is:

```
H[σ] = (ε/2) Σ_v Q_v² = (ε/2) ‖B₁ σ‖² = (ε/2) σᵀ L₁ᵈᵒʷⁿ σ
```

where L₁ᵈᵒʷⁿ = B₁ᵀ B₁. Ground states (ice configurations) sit in ker(L₁ᵈᵒʷⁿ) = ker(B₁), and the spectral gap of L₁ᵈᵒʷⁿ sets the minimum monopole excitation energy.

#### 1.2.3 Face Selection Strategy: "Fill All" vs. "Leave Loops Open"

For Stage 1, we compute the spectral catalog under two face-filling strategies. The natural candidates for 2-cells are Morrison et al.'s "minimal loops" — closed chains of edges that contain no vertices in their interior (Morrison et al. 2013, §3.1). These are the smallest polygon faces of the lattice embedding and correspond to the plaquettes around which local spin flips can occur.

**Strategy A — Fill all minimal loops:** Every minimal loop becomes a 2-cell. This gives the maximal number of faces and the *smallest* β₁ (most independent cycles become boundaries of faces, hence trivial in homology).

**Strategy B — Fill no loops:** B₂ is empty (n₂ = 0). This makes L₁ᵘᵖ = 0, so L₁ = L₁ᵈᵒʷⁿ = B₁ᵀB₁. Every independent cycle is homologically nontrivial, giving the *largest possible* β₁ = n₁ − n₀ + β₀.

**Strategy C (for Stage 3, not Stage 1) — Partial filling:** Selectively fill some loops and not others, tuning β₁ between the extremes. We defer this to Stage 3.

**Key point:** The divergence-free subspace ker(B₁) is independent of face-filling — it depends only on the graph. What changes is the internal decomposition ker(B₁) = im(B₂) ⊕ ker(L₁): filling more faces enlarges the curl component im(B₂) at the expense of the harmonic component ker(L₁), reducing β₁.

For Stage 1, computing both A and B for every lattice gives us the full range of β₁ attainable on each topology, which is the key information for planning Stage 3.

#### 1.2.4 Boundary Conditions and β₁

Boundary conditions have a dramatic effect on β₁. Example: 3×3 square lattice.

| Property | Periodic (torus) | Open (planar patch) |
|---|---|---|
| n₀ / n₁ / n₂ (all filled) | 9 / 18 / 9 | 9 / 12 / 4 |
| Euler characteristic χ | 0 (torus) | 1 (disk) |
| β₁ (all faces filled) | **2** | **0** |
| β₁ (no faces filled) | **10** | **4** |

Periodic BCs (torus) guarantee β₁ ≥ 2 even with all faces filled, because the two non-contractible winding loops (horizontal and vertical) around the torus cannot be expressed as combinations of face boundaries. Open BCs can drive β₁ to zero. **All Stage 1 computations use periodic BCs** to ensure a nonzero topological protection floor.

### 1.3 Laplacian Construction and Eigendecomposition

From B₁ and B₂, construct:

```python
# Vertex (graph) Laplacian
L0 = B1 @ B1.T                    # shape (n0, n0)

# Lower edge Laplacian (aggregation through shared vertices)
L1_down = B1.T @ B1               # shape (n1, n1)

# Upper edge Laplacian (aggregation through shared faces)
L1_up = B2 @ B2.T                 # shape (n1, n1)

# Full Hodge 1-Laplacian
L1 = L1_down + L1_up              # shape (n1, n1)
```

All four are symmetric positive semi-definite sparse matrices.

**Physical significance of L₁ᵈᵒʷⁿ:** This is the central matrix of the project. It simultaneously serves as:
1. The **Hamiltonian operator**: H[σ] = (ε/2) σᵀ L₁ᵈᵒʷⁿ σ (energy of spin configuration)
2. The **ground state projector**: ker(L₁ᵈᵒʷⁿ) = ker(B₁) = divergence-free spin space
3. The **monopole spectrum**: eigenvalues of L₁ᵈᵒʷⁿ set excitation energies above ice manifold
4. The **oversmoothing operator** (with appropriate normalization): spectral gap controls signal decay rate

**Connection to L₁:** The full Hodge Laplacian L₁ = L₁ᵈᵒʷⁿ + L₁ᵘᵖ additionally captures face (curl) structure. The Hodge decomposition ker(B₁) = im(B₂) ⊕ ker(L₁) splits the ice manifold into locally trivial circulations (curl modes, killed by L₁ᵘᵖ) and topologically protected winding modes (harmonic modes, in ker(L₁)). The β₁ = dim(ker(L₁)) harmonic modes are immune to any polynomial filter p(L₁) — this is the mathematical core of the oversmoothing protection claim.

#### 1.3.1 Eigendecomposition

For lattices up to ~10⁴ edges, use `scipy.sparse.linalg.eigsh` for the smallest k eigenvalues and `scipy.linalg.eigh` for dense decomposition of smaller lattices.

**Preferred computational target: L₁ᵈᵒʷⁿ** (not B₁ directly). Since ker(L₁ᵈᵒʷⁿ) = ker(B₁) (standard: B₁ᵀB₁x = 0 ⟺ ‖B₁x‖² = 0 ⟺ B₁x = 0), the null space is the same, but L₁ᵈᵒʷⁿ has computational advantages:
- Symmetric PSD → efficient eigensolvers (dense `eigh`, sparse `eigsh` with shift-invert at `sigma=0`)
- Full spectrum needed anyway for Hamiltonian/monopole analysis — β₁ count falls out of the same eigendecomposition
- Matrix-free application: x ↦ B₁ᵀ(B₁ x) exploits B₁'s extreme sparsity (exactly 2 nonzeros per column)
- Clean threshold: all eigenvalues ≥ 0, so count λᵢ < 10⁻¹⁰ to get β₁

```python
# Dense (sizes S/M): full eigendecomposition
eigenvalues, eigenvectors = scipy.linalg.eigh(L1_down.toarray())
beta_1 = np.sum(eigenvalues < 1e-10)
harmonic_basis = eigenvectors[:, eigenvalues < 1e-10]

# Sparse (sizes L/XL): bottom of spectrum only
eigenvalues_k, eigenvectors_k = scipy.sparse.linalg.eigsh(
    L1_down, k=50, which='SM', sigma=0)
```

**What to compute for each lattice × size × face strategy:**

| Quantity | Definition | Why We Need It |
|---|---|---|
| **Full spectrum of L₀** | All eigenvalues λ₀⁽¹⁾ ≤ λ₀⁽²⁾ ≤ ... | Vertex-level oversmoothing rate; algebraic connectivity (Fiedler value); charge interaction kernel L₀⁻¹ (Nisoli's entropic field theory) |
| **Full spectrum of L₁** | All eigenvalues λ₁⁽¹⁾ ≤ λ₁⁽²⁾ ≤ ... | Edge-level oversmoothing rate; harmonic subspace; Hodge decomposition |
| **Spectrum of L₁ᵈᵒʷⁿ** | Eigenvalues of B₁ᵀB₁ | **Spin ice Hamiltonian spectrum.** Ground state manifold, monopole excitation energies, gradient-channel smoothing rate |
| **Spectrum of L₁ᵘᵖ** | Eigenvalues of B₂B₂ᵀ | Curl-channel smoothing rate; controls how face-filling affects signal propagation |
| **β₀** | dim(ker(L₀)) = # connected components | Should be 1 for periodic BCs. Enters dimension formulas: rank(B₁) = n₀ − β₀ |
| **β₁** | dim(ker(L₁)) | **The key quantity.** Dimension of the harmonic/topologically protected subspace. Controls oversmoothing-resistant channel capacity |
| **β₂** | dim(ker(L₂)) where L₂ = B₂ᵀB₂ | 1 for torus (periodic BCs), 0 for open BCs. Enters Euler check: β₀ − β₁ + β₂ = n₀ − n₁ + n₂ |
| **Spectral gap of L₀** | λ₀⁽²⁾ (smallest nonzero eigenvalue) | Controls vertex oversmoothing rate; related to Nisoli's screening length ξ |
| **Spectral gap of L₁ᵈᵒʷⁿ** | Smallest nonzero eigenvalue Δ of B₁ᵀB₁ | **Monopole gap**: minimum energy (ε/2)Δ to create a charge excitation above the ice manifold |
| **Spectral gap of L₁** | Smallest nonzero eigenvalue of L₁ | Controls edge oversmoothing rate for non-harmonic components |
| **Harmonic basis** | Eigenvectors of L₁ with eigenvalue ≈ 0 | For projecting signals into/out of the protected subspace in Stages 3–4. On the torus, should correspond to horizontal/vertical winding modes |

#### 1.3.2 Numerical Considerations

The null space of L₁ must be computed carefully. Eigenvalues near zero (say < 10⁻¹⁰) should be counted as zero. For large lattices, use iterative solvers (`eigsh` with `sigma=0`, shift-invert mode) to find the k smallest eigenvalues efficiently.

**Validation checks (all must pass for every lattice × size × face strategy):**
- B₁ B₂ = 0 (chain complex property — fundamental correctness check)
- β₀ − β₁ + β₂ = n₀ − n₁ + n₂ (Euler characteristic)
- β₁ = n₁ − rank(B₁) − rank(B₂) (rank-nullity)
- β₀ = 1 for all periodic (connected) lattices
- β₂ = 1 for torus with all faces filled; β₂ = 0 for open BCs or no faces filled
- L₁ · h ≈ 0 for each computed harmonic eigenvector h
- B₁ · h ≈ 0 (divergence-free) and B₂ᵀ · h ≈ 0 (curl-free) for each harmonic h
- L₀ = D − A (graph Laplacian equals degree matrix minus adjacency)
- For a known ice configuration σ (e.g., all-clockwise on squares): verify B₁ σ = 0 and σᵀ L₁ᵈᵒʷⁿ σ = 0

### 1.4 System Sizes

Run the full spectral catalog at these system sizes:

| Size Label | Unit Cells | Approx. Vertices | Approx. Edges | Compute Time (est.) |
|---|---|---|---|---|
| **XS** | 4×4 | 16–128 | 32–200 | < 1 sec (full dense eigh) |
| **S** | 10×10 | 100–800 | 200–1200 | < 1 sec |
| **M** | 20×20 | 400–3200 | 800–4800 | ~seconds |
| **L** | 50×50 | 2500–20000 | 5000–30000 | ~minutes (sparse eigsh) |
| **XL** | 100×100 | 10000–80000 | 20000–120000 | ~10 min (sparse, k smallest only) |

For XS and S: full dense eigendecomposition (all eigenvalues).
For M: full dense or sparse depending on exact count.
For L and XL: sparse iterative for the 100 smallest eigenvalues + explicit null-space computation.

The XS size is for debugging and visualization. S and M are for the complete spectral catalog. L and XL are for verifying scaling laws (how does β₁ scale with system size?).

### 1.5 Stage 1 Outputs

#### 1.5.1 Tables

**Table 1: Lattice Zoo Summary**

| Lattice | z-distribution | n₀/cell | n₁/cell | n₂/cell (all filled) | β₁/cell (all filled) | β₁/cell (none filled) |
|---|---|---|---|---|---|---|
| Square | {4: 100%} | 1 | 2 | 1 | ? | ? |
| Kagome | {3: 100%} | 3 | 6 | 3 | ? | ? |
| Shakti | {2: a%, 3: b%, 4: c%} | ~8 | ~12 | ~5 | ? | ? |
| ... | ... | ... | ... | ... | ... | ... |

The "?" entries are the actual measurements — this table is the first deliverable.

**Table 2: Spectral Properties at Size M (20×20)**

| Lattice | Face Strategy | β₁ | Gap(L₀) | Gap(L₁ᵈᵒʷⁿ) | Gap(L₁) | β₁/n₁ | dim(ker B₁) | Odd-z vertices |
|---|---|---|---|---|---|---|---|---|
| Square | All filled | | | | | | | 0 |
| Square | None filled | | | | | | | 0 |
| Kagome | All filled | | | | | | | all |
| Shakti | All filled | | | | | | | mixed |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

Notes: Gap(L₁ᵈᵒʷⁿ) = monopole gap (minimum charge excitation energy in units of ε/2). dim(ker B₁) = total ice manifold dimension = β₁ + rank(B₂), independent of face strategy. "Odd-z vertices" flags lattices where the ice rule gives |Q_v| = 1 (not 0) at some vertices — relevant for interpreting ker(B₁) vs. the physical ice manifold.

**Table 3: β₁ Scaling**

| Lattice | Face Strategy | β₁(S) | β₁(M) | β₁(L) | β₁(XL) | Scaling Law (fit) |
|---|---|---|---|---|---|---|
| Square | All filled | | | | | β₁ ~ aN² + bN + c |
| Shakti | All filled | | | | | β₁ ~ ? |
| ... | ... | ... | ... | ... | ... | ... |

#### 1.5.2 Figures

**Figure 1: Lattice Zoo Visual Gallery.** For each of the 8 lattice types at size XS (4×4): plot the graph with vertices colored by coordination number, edges shown, and 2-cells (filled loops) shaded. One panel per lattice.

**Figure 2: Eigenvalue Histograms.** For each lattice at size M: histogram of L₁ eigenvalues, with the zero eigenvalues (harmonic modes) highlighted. One panel per lattice, both face strategies overlaid.

**Figure 3: Spectral Density Comparison.** Overlay the eigenvalue distributions of L₁ for all lattices on a single plot (normalized by n₁), showing how the spectral weight shifts toward zero for frustrated lattices.

**Figure 4: β₁ Scaling Laws.** Plot β₁ vs. system size (N = nx = ny) for each lattice type under both face strategies. Log-log scale to identify power-law exponents. Key question: does β₁ scale as N² (extensive), N (boundary), or O(1) (topological)?

**Figure 5: Harmonic Modes Visualization.** For each lattice at size S, visualize the first 3–4 harmonic eigenvectors of L₁ as edge colorings on the lattice. This shows where the protected information lives spatially — are harmonic modes localized or delocalized? Do they concentrate on frustrated loops?

**Figure 6 (new): L₁ᵈᵒʷⁿ vs. L₁ Spectral Comparison.** For each lattice at size M, overlay the eigenvalue spectra of L₁ᵈᵒʷⁿ and L₁. The difference reveals the contribution of L₁ᵘᵖ (face structure). For Strategy B (no faces), L₁ = L₁ᵈᵒʷⁿ and the spectra are identical.

#### 1.5.3 Reference Document

The mathematical correspondences underlying Stage 1 are derived in full detail (with proofs and worked examples) in `docs/tdl-spinice-correspondence.html`. Key results that inform the computation:

| Correspondence | HTML Section | Implementation Impact |
|---|---|---|
| Q = B₁ σ (charge = divergence) | §4 | Compute charges via sparse mat-vec, not vertex loops |
| S = skew(\|B₁\| diag(σ) B₁ᵀ) | §3 | Reconstruct Nisoli's S matrix from B₁ + σ |
| H = (ε/2) σᵀ L₁ᵈᵒʷⁿ σ (Hamiltonian) | §6 | L₁ᵈᵒʷⁿ eigendecomposition gives full energy spectrum |
| ker(B₁) = im(B₂) ⊕ ker(L₁) (Hodge) | §5 | Face-filling controls curl/harmonic partition |
| L₁ᵈᵒʷⁿ preferred over B₁ for null space | §6.6 | Use eigh(L₁ᵈᵒʷⁿ) not SVD(B₁) |
| Ice manifold: even z → Q=0, odd z → \|Q\|=1 | §5.1 | Kagome/mixed lattices need affine coset treatment |
| Periodic BCs guarantee β₁ ≥ 2 | §2.2 | Always use periodic BCs for spectral catalog |

---

## Stage 2 — GCN Oversmoothing Baseline

### 2.1 Purpose

Stage 2 answers a simpler, prerequisite question before we touch edge features or SCNNs: **Does the graph topology of frustrated lattices, by itself, slow down vertex-level oversmoothing in standard GCNs?**

This is worth checking because:
- If frustration already helps at the GCN level, that's interesting and publishable on its own
- If frustration does NOT help at the GCN level (because the GCN only sees L₀, which may not be affected by frustration in the same way as L₁), that sharpens the claim for Stage 3: the protection is specifically a *simplicial/edge-level* phenomenon that requires the Hodge decomposition to access
- Either way, it establishes the quantitative baseline against which Stage 3's SCNN results are compared

### 2.2 Experimental Design

#### 2.2.1 The Core Experiment: Feature Propagation Without Training

The cleanest version of this experiment does not involve training at all. We simply propagate random features through GCN-style Laplacian smoothing and measure how fast they converge. This isolates the topology's effect from any confounds due to training dynamics, loss functions, or learned weights.

**Protocol:**

1. For a given lattice graph G with n₀ vertices:
   - Generate random initial vertex features: H⁽⁰⁾ ∈ ℝ^(n₀ × d), entries drawn iid from N(0, 1)
   - Feature dimension d = 64 (check sensitivity to d ∈ {16, 32, 64, 128})

2. Define the GCN propagation operator:
   - Â = D̃⁻¹/² Ã D̃⁻¹/², where Ã = A + I (self-loops), D̃ = degree matrix of Ã
   - This is the standard Kipf & Welling (2017) normalized adjacency

3. Apply repeated propagation without learned weights or nonlinearity:
   - H⁽ℓ+1⁾ = Â H⁽ℓ⁾ for ℓ = 0, 1, 2, ..., L_max
   - L_max = 64 (far beyond where any practical GCN would be used — we want to see the full decay curve)

4. At each layer ℓ, compute the oversmoothing metrics (Section 2.3)

5. Repeat steps 1–4 for 20 random initializations of H⁽⁰⁾ and report mean ± std of each metric

**Why no weights or nonlinearity?** Li et al. (2018) showed that GCN convolution is fundamentally Laplacian smoothing. Adding learned weights W⁽ℓ⁾ and ReLU nonlinearity does not prevent oversmoothing — it can only slow it or (with pathological weights) accelerate it. The weight-free propagation gives us the *intrinsic smoothing rate of the topology*, which is what we want to compare across lattice types. We do a secondary experiment with weights+ReLU (Section 2.6) to confirm the trends hold.

#### 2.2.2 Lattice Conditions (8 frustrated + 8 matched controls = 16 conditions)

For each of the 8 ASI lattice types, we also construct a **matched unfrustrated control** — a graph with the same degree sequence (coordination number distribution) but without the frustrated loop topology. The comparison isolates the effect of *frustration structure* from the trivial effect of *degree distribution*.

**How to construct matched controls:**

| ASI Lattice | Coordination | Matched Control | Construction |
|---|---|---|---|
| **Square** | All z=4 | Regular 4-grid (same graph) | Square ice IS the regular grid — it serves as its own control |
| **Kagome** | All z=3 | Random 3-regular graph | `networkx.random_regular_graph(3, n)` |
| **Shakti** | Mixed z=2,3,4 | Configuration-model graph | `networkx.configuration_model(degree_sequence)` using shakti's exact degree sequence, then remove self-loops and multi-edges |
| **Tetris** | Mixed z=2,3,4 | Configuration-model graph | Same approach with tetris degree sequence |
| **Pinwheel** | Mixed z=3,4 | Configuration-model graph | Same approach |
| **Santa Fe** | Mixed z=2,3,4 | Configuration-model graph | Same approach |
| **Staggered Shakti** | Mixed z=2,3,4 | Configuration-model graph | Same approach |
| **Staggered Brickwork** | Mixed z=2,3 | Configuration-model graph | Same approach |

The configuration model preserves the exact degree sequence but randomizes the edge placement, destroying any systematic loop structure or frustration. Multiple random realizations (10 per lattice) should be generated to get error bars on the control.

**Important subtlety:** The configuration model may produce disconnected graphs or graphs with very different loop statistics. For a cleaner control, consider also using **rewired lattices**: start from the ASI lattice and perform random edge swaps (Maslov-Sneppen rewiring) that preserve the degree sequence while destroying the spatial/frustrated structure. This gives a control that is more "graph-like" (connected, locally tree-like near rewired edges) rather than fully random.

Both controls should be tested. The rewired control is the more stringent comparison.

#### 2.2.3 System Sizes

Run Stage 2 at sizes S (10×10) and M (20×20). Stage 2 is fast (no training, just sparse matrix-vector products), so M should complete in seconds per lattice per random seed.

### 2.3 Oversmoothing Metrics

At each propagation depth ℓ, compute four complementary metrics:

#### 2.3.1 Dirichlet Energy

$$E_{\text{Dir}}(\ell) = \text{tr}\left(H^{(\ell)\top} L_0 \, H^{(\ell)}\right)$$

This measures the total "roughness" of the feature field on the graph. Laplacian smoothing monotonically decreases Dirichlet energy — features become smoother with each application. Oversmoothing = Dirichlet energy → 0.

**Normalized version:** Divide by the initial value to get relative Dirichlet energy: $\hat{E}_{\text{Dir}}(\ell) = E_{\text{Dir}}(\ell) / E_{\text{Dir}}(0)$. This starts at 1 and decays toward 0, allowing comparison across lattices with different numbers of edges.

#### 2.3.2 Mean Average Distance (MAD)

$$\text{MAD}(\ell) = \frac{1}{n_0(n_0-1)} \sum_{i \neq j} \| h_i^{(\ell)} - h_j^{(\ell)} \|_2$$

where $h_i^{(\ell)}$ is the feature vector of vertex i at layer ℓ. This directly measures how distinguishable vertex representations are. Oversmoothing = MAD → 0. Computationally expensive for large graphs (O(n₀²)); for size M+, subsample 500 random vertex pairs.

**Normalized:** $\widehat{\text{MAD}}(\ell) = \text{MAD}(\ell) / \text{MAD}(0)$.

#### 2.3.3 Effective Rank

$$r_{\text{eff}}(\ell) = \exp\left(- \sum_{i=1}^{d} p_i \log p_i\right), \quad p_i = \sigma_i / \sum_j \sigma_j$$

where σ₁ ≥ σ₂ ≥ ... ≥ σ_d are the singular values of H⁽ℓ⁾ ∈ ℝ^(n₀ × d). This measures the "effective dimensionality" of the feature matrix. When all features collapse to rank 1 (oversmoothed), r_eff → 1. When features span the full d-dimensional space, r_eff → d.

This is arguably a cleaner metric than MAD because it's invariant to the overall scale of the features and captures dimensional collapse directly.

#### 2.3.4 Layer-wise Cosine Similarity

$$\text{CosSim}(\ell) = \frac{1}{|E|} \sum_{(i,j) \in E} \frac{h_i^{(\ell)} \cdot h_j^{(\ell)}}{\|h_i^{(\ell)}\| \, \|h_j^{(\ell)}\|}$$

Average cosine similarity between features of adjacent vertices. Oversmoothing drives this toward 1 (all neighbors become identical). This is local (computed over edges, not all pairs) and cheap to compute.

### 2.4 Theoretical Predictions

Before running the experiments, we can write down what the theory predicts.

The propagation H⁽ℓ⁾ = Â^ℓ H⁽⁰⁾ has an exact spectral decomposition. Let Â = UΛUᵀ be the eigendecomposition with eigenvalues 1 = μ₁ ≥ μ₂ ≥ ... ≥ μₙ ≥ -1. Then:

$$H^{(\ell)} = U \, \text{diag}(\mu_1^\ell, \mu_2^\ell, \ldots, \mu_n^\ell) \, U^\top \, H^{(0)}$$

As ℓ → ∞, only the eigencomponents with |μᵢ| close to 1 survive. For a connected graph, μ₁ = 1 is simple, so asymptotically H⁽ℓ⁾ → u₁u₁ᵀH⁽⁰⁾ (rank-1 projection onto the leading eigenvector — complete oversmoothing). The **rate** of convergence is controlled by |μ₂|, the second-largest eigenvalue magnitude:

$$\hat{E}_{\text{Dir}}(\ell) \approx c \cdot |\mu_2|^{2\ell}$$

So the **spectral gap** (1 − |μ₂|) directly controls the oversmoothing rate. Graphs with smaller spectral gaps oversmooth more slowly.

**Key predictions for Stage 2:**

| Lattice | Expected Spectral Gap (L₀) | Expected Oversmoothing Rate | Reasoning |
|---|---|---|---|
| **Square** | Moderate (~2π²/N² for periodic) | Moderate — clean exponential decay | Regular, no frustration |
| **Kagome** | Small (flat bands near zero) | Slow — many modes near eigenvalue 1 of Â | Flat bands → many near-degenerate modes |
| **Shakti** | Small (mixed coordination) | Slow | Mixed z creates irregular spectral density |
| **Tetris** | Small (1D-like bands) | Slow in one direction, fast in orthogonal | Reduced dimensionality → directional smoothing |
| **Staggered Brickwork** | Larger (trivial frustration) | Faster than shakti/tetris | Approaches regular behavior |
| **Config-model control** | Typically larger (expander-like) | Faster | Random graphs tend to be good expanders |
| **Rewired control** | Similar to config-model | Faster than structured lattice | Loop structure destroyed |

**The nuanced prediction:** At the GCN level (vertex features, L₀), frustrated lattices may oversmooth *somewhat* more slowly than their controls because mixed coordination creates a broader eigenvalue distribution. But this effect is indirect and may be small, because the frustration is an *edge-level* (L₁) phenomenon that L₀ doesn't directly see. The real payoff should come in Stage 3 when we move to edge features and the Hodge Laplacian. Stage 2 quantifies the size of the vertex-level effect as a baseline.

### 2.5 Analysis Protocol

For each of the 16 conditions (8 lattices + 8 controls) × 2 sizes × 20 random seeds:

1. **Compute and store** all four metrics at every layer ℓ = 0, 1, ..., 64
2. **Fit exponential decay** to the normalized Dirichlet energy: $\hat{E}_{\text{Dir}}(\ell) = a \cdot e^{-\alpha \ell}$. The decay constant α is the "oversmoothing rate." Report α for each condition.
3. **Identify the critical depth** ℓ* where the effective rank drops below d/2 (feature space half-collapsed). This is the practical "depth budget" of the topology.
4. **Correlate with spectral properties** from Stage 1: plot α vs. spectral gap of L₀ for each lattice. Does the spectral gap predict the oversmoothing rate?

### 2.6 Secondary Experiment: GCN with Learned Weights and Nonlinearity

After the pure-propagation experiment, run a lighter version with actual GCN layers to verify the trends hold under realistic conditions.

**Protocol:**

1. Use the same lattice conditions (8 lattices + 8 controls, size M)
2. Construct GCN networks of depth L ∈ {2, 4, 8, 16, 32}
3. Each layer: H⁽ℓ+1⁾ = ReLU(Â H⁽ℓ⁾ W⁽ℓ⁾) with W⁽ℓ⁾ ∈ ℝ^(d × d), d = 64
4. Initialize all W⁽ℓ⁾ with Xavier initialization
5. **No training** — just measure the metrics on the forward pass with random weights
6. Repeat for 20 random W initializations
7. Compare metric decay curves against the weight-free experiment

**Why random weights without training?** We want to measure the topology's *intrinsic* effect on feature diversity, not the effect of a particular learned task. Training on a task would confound the topology effect with the task effect. The random-weight experiment measures: given an uninitialized GCN of depth L on this topology, how much feature diversity survives to the output layer? This is the "starting point" from which any training must work.

**Optional extension:** If resources allow, train on a simple synthetic task (random binary node classification with class labels correlated with graph structure) and measure test accuracy vs. depth. This connects the abstract metrics to a concrete performance measure but is lower priority for Stage 2.

### 2.7 Stage 2 Outputs

#### 2.7.1 Tables

**Table 4: Oversmoothing Rates**

| Lattice | Control Type | α (Dirichlet) | α (MAD) | ℓ* (half-rank depth) | Spectral Gap (L₀) |
|---|---|---|---|---|---|
| Square | (self) | | | | |
| Square | Rewired | | | | |
| Kagome | (self) | | | | |
| Kagome | 3-regular random | | | | |
| Kagome | Rewired | | | | |
| Shakti | (self) | | | | |
| Shakti | Config-model | | | | |
| Shakti | Rewired | | | | |
| ... | ... | ... | ... | ... | ... |

**Table 5: Effect Sizes**

| Lattice | Δα vs. config-model | Δα vs. rewired | Δℓ* vs. config-model | Δℓ* vs. rewired |
|---|---|---|---|---|
| Shakti | | | | |
| Tetris | | | | |
| ... | ... | ... | ... | ... |

How much slower (or faster) does each frustrated lattice oversmooth compared to its matched control?

#### 2.7.2 Figures

**Figure 6: Normalized Dirichlet Energy vs. Depth.** 8 panels (one per lattice type). Each panel shows: the ASI lattice curve, the config-model control curve (with error band over 10 realizations), and the rewired control curve (with error band). X-axis: layer depth ℓ. Y-axis: $\hat{E}_{\text{Dir}}(\ell)$ on log scale. Shaded bands = ±1 std over 20 random initial feature seeds.

**Figure 7: Effective Rank vs. Depth.** Same layout as Figure 6, using effective rank instead of Dirichlet energy.

**Figure 8: Summary — Oversmoothing Rate vs. Spectral Gap.** Scatter plot with one point per lattice condition. X-axis: spectral gap of L₀ (from Stage 1). Y-axis: fitted decay constant α. Color: frustrated (red) vs. control (blue). If the spectral gap is a good predictor, points should fall on a line. Deviations from the line suggest additional topological effects beyond what the spectral gap captures.

**Figure 9: Summary — Critical Depth vs. β₁ (preview of Stage 3).** Scatter plot. X-axis: β₁ from Stage 1 (even though Stage 2 uses vertex features, this previews whether β₁ is already correlated with vertex-level robustness). Y-axis: ℓ* (half-rank depth). This figure sets up the Stage 3 hypothesis: if β₁ predicts ℓ* even at the GCN level, the harmonic protection effect extends beyond the edge signal level.

---

## Workflow Summary

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1                                                         │
│                                                                 │
│  1. Implement LatticeGenerator for all 8 lattice types          │
│  2. Generate graphs at sizes XS, S, M, L, XL                   │
│  3. Construct B₁, B₂ (both face strategies)                    │
│  4. Compute L₀, L₁, L₁_down, L₁_up                            │
│  5. Eigendecompose → spectra, β₁, harmonic bases               │
│  6. Produce Tables 1–3, Figures 1–5                             │
│                                                                 │
│  Estimated time: 1–2 weeks coding + computation                 │
│  Dependencies: networkx, scipy, numpy, matplotlib               │
│  No GPU needed                                                  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2                                                         │
│                                                                 │
│  7. Construct matched controls (config-model + rewired)         │
│  8. Generate random features H⁰ ∈ ℝ^(n₀ × 64)                 │
│  9. Propagate: H^(ℓ+1) = Â H^(ℓ) for ℓ = 0..64                │
│  10. At each ℓ: compute Dirichlet energy, MAD, effective rank,  │
│      cosine similarity                                          │
│  11. Repeat 20 seeds × 16 conditions × 2 sizes                 │
│  12. Fit exponential decays, compute critical depths            │
│  13. Secondary: repeat with ReLU + random weights               │
│  14. Produce Tables 4–5, Figures 6–9                            │
│                                                                 │
│  Estimated time: 1 week coding + computation                    │
│  Dependencies: + torch, torch_geometric (for secondary exp)     │
│  GPU: optional, helpful for secondary experiment                │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ Decision Point                                                  │
│                                                                 │
│  If frustrated lattices oversmooth slower than controls:         │
│  → Frustration has a vertex-level effect. Stage 3 should        │
│    show an even stronger edge-level effect via β₁.              │
│                                                                 │
│  If frustrated lattices oversmooth at same rate as controls:     │
│  → Frustration is purely an edge-level phenomenon.              │
│    Stage 3 is the real test. Sharpen the claim:                 │
│    "you MUST use edge features and Hodge decomposition           │
│    to access topological protection."                           │
│                                                                 │
│  Either outcome is informative and publishable.                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Appendix: Quick Reference for Spectral Relationships

For a simplicial complex with boundary operators B₁ (n₀ × n₁) and B₂ (n₁ × n₂):

```
Hodge decomposition of ℝ^n₁:

    ℝ^n₁ = im(B₁ᵀ) ⊕ im(B₂) ⊕ ker(L₁)
            ────────   ──────   ────────
            gradient    curl    harmonic
            
    dim(im(B₁ᵀ)) = rank(B₁) = n₀ - β₀
    dim(im(B₂))   = rank(B₂) = n₂ - β₂
    dim(ker(L₁))  = β₁ = n₁ - rank(B₁) - rank(B₂)
    
    Check: (n₀ - β₀) + (n₂ - β₂) + β₁ = n₁  ✓
    Euler: β₀ - β₁ + β₂ = n₀ - n₁ + n₂       ✓
```

The projectors onto each subspace:

```
P_grad    = B₁ᵀ (B₁ B₁ᵀ)⁺ B₁           projects onto im(B₁ᵀ)
P_curl    = B₂ (B₂ᵀ B₂)⁺ B₂ᵀ           projects onto im(B₂)
P_harmonic = I - P_grad - P_curl          projects onto ker(L₁)
```

where (·)⁺ denotes pseudoinverse.

Oversmoothing under L₁-polynomial filters p(L₁):

```
For f = f_grad + f_curl + f_harm:

    p(L₁) f = p(L₁) f_grad + p(L₁) f_curl + p(0) f_harm
    
    The gradient and curl components are filtered by nonzero eigenvalues.
    The harmonic component is multiplied by p(0) — a constant.
    
    If p(0) ≠ 0, the harmonic component is preserved exactly.
    If p(0) = 0, the harmonic component is killed.
    
    Standard GCN-style filters have p(0) ≠ 0 (they include the identity),
    so the harmonic component persists through arbitrarily many layers.
```

This is the mathematical core of the Pillar 1 claim. Stage 1 tells us how large the harmonic subspace is for each lattice. Stage 2 establishes the vertex-level propagation baseline. Stage 3 (below) trains real GCNs and shows whether the spectral differences translate into actual task performance differences.

---

## Stage 3 — Trained GCN Node Classification: The "How Deep Can You Go" Experiment

### 3.1 Purpose

Stages 1 and 2 are analytical: they measure spectral properties and untrained propagation dynamics. Stage 3 is the experiment that produces the money plot — **test accuracy vs. network depth** — on a real classification task, comparing frustrated ASI lattices against matched unfrustrated controls. This is the standard experiment format in the oversmoothing literature, from Li et al. (2018) through the CSBM analyses of Wu et al. (2022) and the Phys. Rev. X paper of Duranthon & Zdeborová (2025). Our contribution is running it on a principled family of topologies with controlled frustration.

The key question: **On which lattice topologies can you train deeper GCNs before accuracy collapses?** If the answer correlates with β₁ or the spectral gap from Stage 1, we have the causal chain: frustrated topology → spectral properties → deeper effective depth.

### 3.2 The Task: Spatial Node Classification (CSBM-on-Lattice)

The Contextual Stochastic Block Model is the standard synthetic benchmark for GNN node classification. In the standard CSBM, the graph itself is generated randomly from a stochastic block model (edges are denser within communities). We adapt this by keeping the graph topology fixed (it's our ASI lattice) and planting the community structure through spatial partitions and noisy node features.

#### 3.2.1 Data Generation

For a given lattice graph G = (V, E) with n₀ vertices, each having a 2D spatial coordinate (xᵢ, yᵢ):

**Step 1 — Assign class labels.** Partition the vertices into C classes based on their spatial position. Three partition schemes, from easy to hard:

**Scheme A — "Halves" (C=2):**
Vertex i gets label yᵢ = 0 if xᵢ < x_median, else yᵢ = 1. This creates a single straight boundary cutting the lattice roughly in half. The boundary is local — only vertices near the midline are ambiguous. A shallow GCN should handle this, so the performance difference between topologies may be small.

**Scheme B — "Quadrants" (C=4):**
Labels based on (x < x_median, y < y_median) → four spatial quadrants. Two perpendicular boundaries. Slightly harder — corner regions require information from two boundaries.

**Scheme C — "Stripes" (C=2, tunable wavelength λ):**
Label yᵢ = 0 if ⌊xᵢ / λ⌋ is even, else yᵢ = 1. This creates alternating vertical stripes of width λ. The wavelength λ controls task difficulty:

- Large λ (few stripes): boundaries are far apart, most vertices are deep inside their class region → easy, shallow GCN suffices
- Small λ (many stripes): boundaries are dense, most vertices are near a boundary → hard, need good spatial resolution
- Very small λ (sub-unit-cell stripes): every vertex is near a boundary → maximally hard, deeper networks needed but oversmoothing kills them

Scheme C is the most informative because λ gives us a continuous control knob for how much depth the task demands. We can sweep λ and find the critical wavelength at which each topology's GCN breaks.

**Step 2 — Generate node features.** Each vertex gets a feature vector:

$$x_i = \mu_{y_i} + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2 I_d)$$

where μ_c ∈ ℝ^d is a class-specific mean vector. For C=2, set μ₀ = +μ·e₁ and μ₁ = −μ·e₁ (features point in opposite directions along the first coordinate). The signal-to-noise ratio SNR = μ/σ controls how informative the features are:

- High SNR (μ >> σ): features alone are almost enough to classify → GCN barely needed
- Low SNR (μ << σ): features are very noisy → GCN must aggregate over many neighbors to denoise → depth matters → oversmoothing becomes the bottleneck

**Recommended default:** d = 64, SNR ∈ {0.1, 0.5, 1.0, 2.0}. The SNR = 0.5 regime is the sweet spot where the task is hard enough that depth matters but not so hard that it's unsolvable.

**Step 3 — Train/validation/test split.** Random 60/20/20 split of vertices. Since the lattice is spatially structured, ensure the split is random (not spatial) so that train/test vertices are interleaved.

#### 3.2.2 Why This Works as an Oversmoothing Testbed

The logic is exactly the CSBM logic adapted to fixed topology:

1. Features are noisy: no single vertex can be classified reliably from its own features alone (at low SNR)
2. Neighbors in the same class have the same mean μ_c: aggregating features from same-class neighbors averages out noise → *this is what makes GCNs useful* (Laplacian smoothing IS denoising when the label field is smooth)
3. Neighbors across a class boundary have different means: aggregating from them adds *noise* rather than signal → this is the "mixing effect"
4. As depth increases: the receptive field grows, more same-class neighbors contribute (denoising improves), but eventually the receptive field crosses enough boundaries that the mixing effect dominates → accuracy drops → **oversmoothing**
5. The "sweet spot" depth — where denoising is maximized before mixing takes over — depends on the graph topology

On ASI lattices, the key hypothesis is that **frustrated topologies shift the sweet spot to greater depth** because their broader spectral density (smaller spectral gap) slows down the rate at which the mixing effect grows.

### 3.3 Network Architecture

Keep it as simple as possible — we want to isolate the topology effect, not confound it with architectural tricks.

#### 3.3.1 Vanilla GCN (primary)

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class VanillaGCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, out_dim))
    
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index)
        return x  # raw logits for cross-entropy
```

**Hyperparameters:**
- hidden_dim = 64
- num_layers ∈ {2, 4, 8, 16, 32, 64} — this is the variable we sweep
- No dropout, no batch norm, no residual connections — we want the raw topology effect
- Optimizer: Adam, lr = 0.01, weight_decay = 5e-4
- Train for 200 epochs (these are small graphs, convergence is fast)
- Report best validation accuracy, corresponding test accuracy

#### 3.3.2 GCN + Residual Connection (secondary comparison)

```python
class ResGCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        self.input_proj = torch.nn.Linear(in_dim, hidden_dim)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.output_proj = torch.nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x, edge_index):
        x = F.relu(self.input_proj(x))
        for conv in self.convs:
            x = x + F.relu(conv(x, edge_index))  # residual
        x = self.output_proj(x)
        return x
```

The residual GCN is the standard oversmoothing mitigation. Including it lets us answer: does ASI frustrated topology provide benefit *on top of* residual connections, or only for the vanilla case?

#### 3.3.3 No Other Mitigations

We deliberately do NOT use DropEdge, PairNorm, JKNet, or other anti-oversmoothing techniques. The point is to measure the topology's naked effect. These can be added later as additional comparisons.

### 3.4 Experimental Matrix

#### 3.4.1 Full Sweep

| Factor | Values | Count |
|---|---|---|
| Lattice type | square, kagome, shakti, tetris, pinwheel, santa_fe, staggered_shakti, staggered_brickwork | 8 |
| Control type | ASI lattice, config-model, rewired | 3 per lattice = 24 total |
| System size | M (20×20 unit cells) | 1 |
| Partition scheme | Stripes with λ ∈ {2, 4, 8, 16} unit cells | 4 |
| SNR | 0.5, 1.0 | 2 |
| Network depth | 2, 4, 8, 16, 32, 64 | 6 |
| Architecture | Vanilla GCN, ResGCN | 2 |
| Random seeds | 10 (covering feature noise, weight init, train/test split) | 10 |

**Total runs:** 24 conditions × 4 λ × 2 SNR × 6 depths × 2 architectures × 10 seeds = **23,040 training runs**

Each run is tiny (200 epochs, ~1000 vertices, ~2000 edges, ≤64 layers), so this fits comfortably on a single GPU. Estimated wall time: ~10 seconds per run → ~64 GPU-hours total → ~3 days on one GPU, or ~8 hours on 8 GPUs. Very tractable.

#### 3.4.2 Quick Pilot (start here)

For initial debugging and sanity-checking, run only:

| Factor | Values |
|---|---|
| Lattice | square, shakti, staggered_brickwork |
| Control | ASI lattice only (no controls yet) |
| Size | S (10×10) |
| Partition | Stripes, λ = 4 |
| SNR | 1.0 |
| Depth | 2, 4, 8, 16, 32 |
| Architecture | Vanilla GCN |
| Seeds | 3 |

**Pilot runs:** 3 × 5 × 3 = 45 runs → ~10 minutes. This immediately shows whether the accuracy-vs-depth curves look different across the three lattices (they should — square and shakti have very different spectral properties, staggered brickwork is the null model).

### 3.5 Metrics

For each run, record:

1. **Test accuracy** at the epoch with best validation accuracy (standard protocol)
2. **Train accuracy** at the same epoch (to detect overfitting vs. underfitting)
3. **Convergence epoch** — when validation accuracy stabilizes (within 0.5% of final value)
4. **Dirichlet energy** of the final-layer representations (connect to Stage 2 metrics)
5. **Effective rank** of the final-layer feature matrix (connect to Stage 2 metrics)

Metrics 4 and 5 bridge Stages 2 and 3: they let us verify that the topology-dependent oversmoothing measured with random features (Stage 2) actually corresponds to the topology-dependent accuracy degradation during training (Stage 3).

### 3.6 Predicted Outcomes and What They Mean

#### 3.6.1 The Money Plot: Accuracy vs. Depth

For each lattice/control condition, the accuracy-vs-depth curve should have the classic shape:

```
Accuracy
   │
   │        ╭──── frustrated lattice (shakti, tetris)
   │      ╱    ╲  
   │    ╱        ╲─────────── slow decline
   │  ╱    ╭─── unfrustrated control
   │╱    ╱   ╲
   │   ╱       ╲──────────── fast decline
   │ ╱
   │╱
   └────────────────────────────── Depth
   2    4    8    16    32    64
```

Three key measurements from each curve:

1. **Peak accuracy** — the best achievable accuracy (at optimal depth). May or may not differ between frustrated and control.
2. **Optimal depth** (ℓ*) — the depth at which peak accuracy occurs. Prediction: ℓ* is larger for frustrated lattices.
3. **Decay rate** after peak — how fast accuracy drops beyond ℓ*. Prediction: slower decay for frustrated lattices.

#### 3.6.2 Interpretation Table

| Outcome | What It Means |
|---|---|
| **Frustrated lattices have larger ℓ* and slower decay** | Core hypothesis confirmed at vertex level. Topology controls oversmoothing rate. |
| **Same ℓ* but higher peak accuracy for frustrated** | Frustration improves representation quality but doesn't extend depth. Interesting but different from our thesis. |
| **No difference at all** | Frustration has no vertex-level effect. Stage 4 (SCNN on edge features) becomes the first real test. This outcome *sharpens* the edge-level claim. |
| **Frustrated lattices are WORSE** | Possible if mixed coordination creates bottlenecks that hurt message passing. Would need investigation. Unlikely but informative. |
| **Controls show high variance, frustrated lattices are stable** | Interesting side finding: structured frustrated topology regularizes performance. |

#### 3.6.3 Connecting to Stage 1 Spectral Data

The most important analysis: **correlate ℓ* and decay rate with spectral properties from Stage 1** across all lattice types and controls.

Plot 1: ℓ* vs. spectral gap of L₀. If this is a clean negative correlation (smaller gap → larger ℓ*), the spectral gap fully explains the oversmoothing rate and there's no additional "frustration magic" at the vertex level.

Plot 2: ℓ* vs. β₁ (even though we're doing vertex-level GCN, not edge-level SCNN). If β₁ predicts ℓ* even at the vertex level, that's a preview of the Stage 4 result and suggests the harmonic subspace has indirect vertex-level consequences.

Plot 3: Decay rate vs. spectral gap. The decay rate should be directly predicted by |μ₂|^2 from Stage 2's spectral analysis.

### 3.7 Secondary Experiment: Signal Denoising (Regression Task)

If time permits, run a regression variant that's even closer to the Stage 2 analysis:

**Task:** The "ground truth" vertex signal is the k-th eigenmode of L₀ (a smooth spatial pattern). The input is this eigenmode plus Gaussian noise. Train a GCN to recover the clean signal (MSE loss). Vary k (which eigenmode) and depth.

```python
# Ground truth: k-th eigenmode of L0
eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(L0, k=20, which='SM')
target = eigenvectors[:, k]  # shape (n0,)

# Input: target + noise
x = target.unsqueeze(1) + sigma * torch.randn(n0, d)

# Task: regress from x to target
# Loss: MSE
```

**Why this is nice:** The theory predicts exactly what should happen. The GCN's ability to recover the k-th eigenmode depends on whether the corresponding eigenvalue λ_k is amplified or suppressed by the Laplacian smoothing. Low-frequency modes (small k, small λ_k) should be easy — they're what the GCN naturally preserves. High-frequency modes (large k, large λ_k) should be progressively harder — they're what gets killed by smoothing.

The frustration connection: frustrated lattices have more eigenvalues clustered near zero (broader spectral density). This means more eigenmodes in the "easy" regime → the GCN can recover more spatial structure before depth destroys it.

**Predicted outcome:** Plot MSE vs. depth for each eigenmode k and each lattice. On frustrated lattices, higher-k modes should remain recoverable at greater depth.

### 3.8 Implementation Notes

#### 3.8.1 Converting Lattices to PyTorch Geometric Format

```python
from torch_geometric.data import Data

def lattice_to_pyg(lattice_generator, partition_scheme, snr, seed):
    """Convert a LatticeGenerator instance to a PyG Data object."""
    G = lattice_generator.build_graph()
    
    # Edge index (COO format for PyG)
    edges = list(G.edges())
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    # Make undirected
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    # Node positions
    pos = torch.tensor([G.nodes[v]['pos'] for v in G.nodes()], 
                        dtype=torch.float)
    
    # Class labels from partition scheme
    y = partition_scheme(pos)  # returns LongTensor of class labels
    
    # Noisy features
    rng = torch.Generator().manual_seed(seed)
    mu = class_means(y, d=64)  # (n_nodes, 64)
    x = mu + (1.0/snr) * torch.randn(len(y), 64, generator=rng)
    
    # Train/val/test masks (60/20/20 random)
    idx = torch.randperm(len(y), generator=rng)
    n = len(y)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[idx[:int(0.6*n)]] = True
    val_mask[idx[int(0.6*n):int(0.8*n)]] = True
    test_mask[idx[int(0.8*n):]] = True
    
    return Data(x=x, edge_index=edge_index, y=y, pos=pos,
                train_mask=train_mask, val_mask=val_mask, 
                test_mask=test_mask)
```

#### 3.8.2 Training Loop

```python
def train_and_evaluate(model, data, epochs=200, lr=0.01, wd=5e-4):
    """Standard PyG training loop. Returns metrics dict."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, 
                                 weight_decay=wd)
    
    best_val_acc = 0
    best_test_acc = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], 
                               data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        # Eval
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            
            val_acc = (pred[data.val_mask] == data.y[data.val_mask]
                      ).float().mean().item()
            test_acc = (pred[data.test_mask] == data.y[data.test_mask]
                       ).float().mean().item()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
    
    # Final layer diagnostics
    model.eval()
    with torch.no_grad():
        H = get_final_layer_features(model, data)  # hook
        dirichlet_energy = compute_dirichlet_energy(H, data.edge_index)
        eff_rank = compute_effective_rank(H)
    
    return {
        'best_val_acc': best_val_acc,
        'best_test_acc': best_test_acc,
        'final_dirichlet_energy': dirichlet_energy,
        'final_effective_rank': eff_rank,
    }
```

#### 3.8.3 Main Experiment Script

```python
results = []

for lattice_type in LATTICE_TYPES:
    for control_type in ['native', 'config_model', 'rewired']:
        for lam in [2, 4, 8, 16]:
            for snr in [0.5, 1.0]:
                for depth in [2, 4, 8, 16, 32, 64]:
                    for arch in ['vanilla', 'residual']:
                        for seed in range(10):
                            # Build graph
                            gen = LatticeGenerator(
                                lattice_type, nx=20, ny=20)
                            if control_type == 'config_model':
                                G = make_config_model(gen)
                            elif control_type == 'rewired':
                                G = maslov_sneppen_rewire(gen)
                            else:
                                G = gen
                            
                            # Build PyG data
                            partition = StripePartition(lam)
                            data = lattice_to_pyg(G, partition, 
                                                   snr, seed)
                            
                            # Build model
                            if arch == 'vanilla':
                                model = VanillaGCN(
                                    64, 64, partition.n_classes, 
                                    depth)
                            else:
                                model = ResGCN(
                                    64, 64, partition.n_classes, 
                                    depth)
                            
                            # Train
                            metrics = train_and_evaluate(model, data)
                            metrics.update({
                                'lattice': lattice_type,
                                'control': control_type,
                                'lambda': lam,
                                'snr': snr,
                                'depth': depth,
                                'arch': arch,
                                'seed': seed,
                            })
                            results.append(metrics)

df = pd.DataFrame(results)
df.to_csv('stage3_results.csv', index=False)
```

### 3.9 Stage 3 Outputs

#### 3.9.1 Tables

**Table 6: Optimal Depth and Peak Accuracy by Lattice**

| Lattice | Control | SNR | λ | ℓ* (Vanilla) | Peak Acc (Vanilla) | ℓ* (Res) | Peak Acc (Res) |
|---|---|---|---|---|---|---|---|
| Square | native | 0.5 | 4 | | | | |
| Square | rewired | 0.5 | 4 | | | | |
| Shakti | native | 0.5 | 4 | | | | |
| Shakti | rewired | 0.5 | 4 | | | | |
| ... | ... | ... | ... | ... | ... | ... | ... |

**Table 7: Depth Gain from Frustrated Topology**

| Lattice | Δℓ* vs. rewired control | ΔPeak Acc vs. rewired | β₁ (from Stage 1) | Spectral Gap L₀ |
|---|---|---|---|---|
| Shakti | | | | |
| Tetris | | | | |
| ... | ... | ... | ... | ... |

#### 3.9.2 Figures

**Figure 10: The Money Plot — Accuracy vs. Depth Curves.**
Grid of panels: rows = lattice types (shakti, tetris, square, kagome, staggered_brickwork), columns = SNR values. Each panel shows test accuracy (y-axis) vs. network depth (x-axis, log scale) with three curves: ASI lattice (red), config-model control (blue, with error band), rewired control (green, with error band). Shaded bands = ±1 std over 10 seeds.

**Figure 11: Optimal Depth vs. Spectral Gap.**
Scatter plot. One point per lattice × control condition. X-axis: spectral gap of L₀. Y-axis: ℓ* (optimal depth). Color: frustrated ASI (red) vs. controls (blue). Annotate each point with lattice name. If the relationship is clean, fit a power law.

**Figure 12: Optimal Depth vs. β₁.**
Same format as Figure 11 but x-axis is β₁ from Stage 1. This is the key preview figure for Stage 4: does β₁ predict oversmoothing resistance even at the vertex level?

**Figure 13: Accuracy vs. Stripe Wavelength λ at Fixed Depth.**
For depth = 16 (deep enough that oversmoothing matters): plot accuracy vs. λ for each lattice. This shows the interaction between task difficulty (controlled by λ) and topology. Frustrated lattices should maintain high accuracy at smaller λ (harder tasks requiring more spatial resolution).

**Figure 14: Dirichlet Energy vs. Depth (Trained vs. Untrained).**
Overlay the Dirichlet energy curves from Stage 2 (untrained propagation, dashed lines) with Stage 3 (trained GCN, solid lines) for the same lattice conditions. This directly validates that the untrained propagation analysis (Stage 2) predicts the trained behavior (Stage 3).

---

## Updated Workflow Summary

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: Spectral Catalog                                       │
│                                                                 │
│  Build lattice zoo → compute B₁, B₂, Laplacians → eigensolve   │
│  Output: β₁, spectral gaps, harmonic bases for all lattices     │
│  Time: ~1–2 weeks                                               │
│  Hardware: CPU only                                              │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: Propagation Baseline (no training)                     │
│                                                                 │
│  Random features → repeated Â multiplication → metric decay     │
│  Output: oversmoothing rates for all lattices + controls        │
│  Time: ~1 week                                                  │
│  Hardware: CPU only                                              │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 3: Trained GCN Experiments                                │
│                                                                 │
│  CSBM-on-lattice task → train GCNs → accuracy-vs-depth curves  │
│  Output: optimal depth ℓ* for each topology, correlation with   │
│          spectral properties from Stages 1–2                    │
│  Time: ~1 week coding + ~3 days GPU                             │
│  Hardware: single GPU                                           │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ Decision Point                                                  │
│                                                                 │
│  Frustrated lattices allow deeper GCNs:                         │
│  → Write up Stages 1–3 as self-contained result.                │
│  → Stage 4 (SCNN) should show even stronger effect              │
│    via β₁ and harmonic protection on edge features.             │
│                                                                 │
│  No vertex-level difference:                                    │
│  → Frustration is purely an edge-level phenomenon.              │
│  → Stage 4 is the real test. Frame the paper as:                │
│    "edge-level Hodge structure provides protection              │
│     invisible to standard vertex-level GCNs."                   │
│                                                                 │
│  Total project time through Stage 3: ~5–6 weeks.               │
│  Everything runs on one machine with one GPU.                   │
└─────────────────────────────────────────────────────────────────┘
```
