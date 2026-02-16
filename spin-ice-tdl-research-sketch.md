# Frustrated Topologies as Neural Network Substrates
## A Research Sketch at the Intersection of Artificial Spin Ice and Topological Deep Learning

*C. M. — February 2026*

---

## 1. The Core Insight: Why These Two Fields Should Talk

Two independent lines of research have converged on the same mathematical objects — the graph Laplacian, the incidence matrix, and the Hodge decomposition — but from opposite directions and for completely different purposes:

**Artificial Spin Ice (ASI)** places Ising-like binary spins on the *edges* of a lattice. The topology of the lattice dictates which vertices are frustrated (unable to satisfy the local energy-minimizing "ice rule"), creating degenerate ground-state manifolds with rich emergent physics: monopole excitations, topological order, Coulomb phases, and entropic screening. Nisoli (2020) showed that the kernel of the entropic charge–charge interaction is precisely L₀⁻¹, the pseudoinverse of the graph Laplacian. Morrison, Nelson & Nisoli (2013) showed that by engineering the lattice topology — mixing coordination numbers, creating vertex-frustrated loops — one can design *a virtually infinite variety* of degenerate geometries with controlled frustration.

**Topological Deep Learning (TDL)** places feature vectors on simplices of any rank (vertices, edges, faces, ...) and uses Hodge Laplacians and boundary operators to define message-passing neural networks. The Simplicial Convolutional Neural Network (SCNN) of Yang, Isufi & Leus (2022) decomposes edge-signal aggregation into independent lower and upper channels via L₁ = L₁ᵈᵒʷⁿ + L₁ᵘᵖ, with separately learnable filters for the gradient, curl, and harmonic subspaces.

**The structural parallel is exact:**

| Spin Ice Concept | Graph/TDL Concept |
|---|---|
| Ising spins on edges | 1-cochains (edge feature vectors) |
| Vertex charge Qᵥ = #in − #out | Divergence: B₁ᵀ · (edge signal) |
| Ice rule: Qᵥ = 0 ∀v | Divergence-free flow: signal ∈ ker(B₁ᵀ) |
| Entropic interaction kernel = TL₀⁻¹ | GCN aggregation operator ∝ L₀⁻¹ or Â |
| Topological charge (monopole) | Non-zero divergence → vertex excitation |
| Ice manifold degeneracy | dim(ker(L₁)) = β₁ (Betti number) |
| Frustrated loop hosting "unhappy vertex" | Non-trivial cycle in homology |
| Mixed coordination lattice (2,3,4-leg) | Heterogeneous graph with variable degree |

This project pursues two complementary questions that arise from this correspondence:

**Pillar 1 — Architecture design (ASI → ML):** Can the topological properties of frustrated lattices — specifically their large harmonic subspaces — be imported as a *design principle* for building more robust simplicial neural networks?

**Pillar 2 — Physics discovery (ML → ASI):** Can SCNN architectures, combined with Monte Carlo simulation, reveal new physical insights about spin ice — in particular, whether the Hodge decomposition (gradient / curl / harmonic) maps onto physically distinct channels (charges / frustration / ice manifold)?

---

## 2. The Lattice Zoo: A Menu of ASI Topologies

Morrison et al. (2013) introduced a rich family of lattice designs, each with distinct topological and frustration properties. These serve as both experimental substrates (Pillar 2) and a catalog of topologies with known spectral/homological properties (Pillar 1):

### 2.1 Classical Lattices (Baselines)

| Lattice | Coordination | Frustration Type | Ground State | Key Spectral Property |
|---|---|---|---|---|
| **Square** | All z=4 | Geometric (dipolar anisotropy lifts degeneracy) | Ordered AFM | Clean spectral gap, Dirac-like dispersion |
| **Kagome / Honeycomb** | All z=3 | Geometric (pseudo-ice rule) | Extensive degeneracy | Flat bands, Dirac cones |

### 2.2 Vertex-Frustrated Lattices (Morrison et al.)

| Lattice | Coordination | Frustration Type | Ground State | Interesting Properties |
|---|---|---|---|---|
| **Shakti** | Mixed z=2,3,4 | Vertex-frustrated, maximally frustrated | Extensive degeneracy + monopole crystallization | Emergent dimer-cover model, topological order (Lao et al. 2018) |
| **Tetris** | Mixed z=2,3,4 | Vertex-frustrated, maximally frustrated | Sliding phase: ordered/disordered bands | Emergent 1D Ising chains, reduced dimensionality |
| **Pinwheel** | Mixed z=3,4 | Vertex-frustrated (T-loops), partially frustrated | Extensive degeneracy, nontrivial | Square + T-shaped loops, mixed frustration |
| **Santa Fe** | Mixed z=2,3,4 | Both frustrated and unfrustrated loops | Polymer-like unhappy-vertex strings | Strings of defects threading between loop types |
| **Staggered Shakti** | Mixed z=2,3,4 | Vertex-frustrated | Striped ground state, forced unhappy vertices | Mandatory defect sites |
| **Staggered Brickwork** | Mixed z=2,3 | Trivially frustrated | Degeneracy reduces to independent spins | "Trivial" — useful null model |

### 2.3 Why This Design Space Matters

Standard GNN experiments use random graphs (Erdős–Rényi), scale-free graphs (Barabási–Albert), small-world graphs (Watts–Strogatz), or application-specific graphs (molecules, social networks). None of these are designed with controlled frustration or topological protection. The ASI lattice zoo offers a *principled* design space parametrized by:

- **Coordination mix** — which vertex types are present (z=2,3,4)
- **Loop topology** — which loops are frustrated vs. unfrustrated
- **Maximal vs. partial frustration** — how many loops host unhappy vertices
- **Ground-state structure** — ordered, disordered, sliding, striped, polymer-like
- **β₁ (independent loops)** — dimension of the harmonic subspace

For Pillar 1, these are topologies with *known, analytically characterized* spectral and homological properties. For Pillar 2, they are physical systems with *known, experimentally verified* collective behavior. The lattice zoo gives us ground truth on both sides.

---

## 3. Pillar 1 — The Harmonic Bottleneck: Topologically Protected Information Channels in SCNNs

### 3.1 The Problem: Oversmoothing Kills Deep GNNs

The dominant failure mode of deep graph neural networks is oversmoothing: as layers accumulate, repeated Laplacian smoothing drives all vertex features toward the graph's leading eigenvector. Representations become indistinguishable. This limits practical GNN depth to ~2–5 layers.

The same problem exists in principle for SCNNs operating on edge features — repeated application of L₁-based filters should similarly homogenize edge representations. But L₁ has a structural property that L₀ typically lacks: a potentially large kernel.

### 3.2 The Key Observation: ker(L₁) Is Immune by Construction

In a simplicial complex with β₁ > 0, the harmonic 1-cochains (ker L₁) form a subspace that is:

- **Invisible to both boundary operators:** B₁ᵀh = 0 (no divergence) and B₁₂ᵀh = 0 (no curl)
- **Invariant under L₁-based diffusion:** eigenvalue zero → infinite diffusion time → zero smoothing rate
- **Topologically determined:** dim(ker L₁) = β₁ depends only on the topology of the complex, not on feature values or learned weights

This is not an analogy. It is a theorem: if h ∈ ker(L₁), then any polynomial filter p(L₁) applied to h returns p(0)·h = constant·h. No amount of Laplacian-based message passing can destroy or distort information stored in the harmonic subspace. It is a **topologically protected channel** through the network.

### 3.3 The Design Principle

This yields a concrete architectural prescription: **engineer the topology of your SCNN's underlying simplicial complex to control β₁ — the dimensionality of the protected information channel.**

- Want more protected capacity? Choose a topology with larger β₁ (more independent loops not bounded by faces)
- Want to control which information is protected? Initialize features with deliberate harmonic components
- Want to tune the tradeoff between smoothing and protection? Adjust the ratio of gradient/curl/harmonic subspace dimensions via the number and arrangement of 2-cells (faces)

The ASI lattice zoo is the ideal testbed because these lattices have *known, analytically characterized* β₁ values that vary systematically with the frustration design. The shakti and tetris lattices have extensive degeneracy (large β₁ scaling with system size), while the staggered brickwork has trivial degeneracy (small β₁). This gives us a controlled experimental ladder.

### 3.4 The Connection to ASI Physics

In spin ice, the harmonic subspace literally *is* the ice manifold. The extensively degenerate ground states of frustrated lattices correspond to large harmonic subspaces. The topological order observed by Lao et al. (2018) in shakti ice — where excitations are topologically protected defects with long lifetimes — is the condensed matter manifestation of the same mathematical structure. Our claim is that this topological protection transfers directly to the neural network setting: information in ker(L₁) has "long lifetime" (persists through many layers) for exactly the same algebraic reason that shakti monopoles have long lifetime (can only be created/annihilated in topologically constrained pairs).

### 3.5 Experimental Plan

**Stage 1 — Spectral catalog (baseline characterization):**

For each lattice in the zoo at several system sizes (20×20 to 100×100 unit cells):
1. Construct the simplicial complex (vertices, edges, and minimal loops as 2-cells)
2. Compute L₀, L₁, L₁ᵈᵒʷⁿ, L₁ᵘᵖ
3. Full eigendecomposition: eigenvalue distributions, spectral gaps, β₁
4. Explicit harmonic basis vectors for ker(L₁)
5. Record how β₁ scales with system size for each lattice type

**Stage 2 — GCN-level oversmoothing baseline:**

Before touching the SCNN, measure oversmoothing on vertex features using standard GCNs:
1. Construct GCN-style networks on each lattice topology and on matched non-frustrated controls (same coordination numbers, no frustration)
2. Measure **Dirichlet energy** E(H) = tr(HᵀL₀H) as a function of layer depth
3. Measure **mean average distance** (MAD) between vertex feature vectors vs. depth
4. Establish: do frustrated lattices show slower decay of feature diversity at the GCN level?

This stage answers a simpler question (does frustration help even for vertex-level GCNs?) and establishes the baseline against which the SCNN harmonic protection result is compared.

**Stage 3 — SCNN harmonic protection (the core experiment):**

1. Train SCNNs of varying depth (2, 4, 8, 16, 32 layers) on each lattice topology
2. At each layer, decompose internal edge representations into gradient + curl + harmonic components using the precomputed Hodge projectors
3. Track the **signal energy** in each subspace through depth: E_grad(ℓ), E_curl(ℓ), E_harm(ℓ)
4. Track the **signal-to-noise ratio** in each subspace under input perturbation
5. Compare lattices with systematically different β₁

**Predicted outcome:** The gradient and curl components should decay (oversmooth) at rates controlled by the nonzero eigenvalues of L₁ᵈᵒʷⁿ and L₁ᵘᵖ respectively. The harmonic component should remain exactly constant (zero eigenvalue → zero decay rate). On lattices with large β₁ (shakti, tetris), a larger fraction of the total signal energy is in the protected channel, so the network should maintain more feature diversity at depth. On lattices with small β₁ (staggered brickwork), nearly all energy is in the gradient/curl subspaces and oversmoothing proceeds as usual.

**Stage 4 — From observation to design:**

If Stage 3 confirms the harmonic protection effect, the question becomes: can we exploit it for arbitrary ML tasks, not just on ASI lattices?

1. **Topology augmentation:** Given a graph from a standard benchmark (molecular, social, etc.), lift it to a simplicial complex by filling in cliques as faces. Different fill strategies yield different β₁ values. Test whether higher β₁ improves deep-network performance.
2. **Learnable topology:** Can we make the 2-cell structure learnable? Add/remove faces during training to optimize β₁ for the task at hand.
3. **Harmonic initialization:** Seed specific features into the harmonic subspace at input, forcing the network to preserve task-critical information through depth.

### 3.6 Deliverable

A demonstrated, quantified correspondence: **β₁ of the underlying simplicial complex predicts the depth at which an SCNN loses feature diversity**, with frustrated ASI topologies providing the highest protection. This establishes the first concrete architectural design principle imported from condensed matter frustration physics into neural network design: *topological frustration → large β₁ → deep-network robustness.*

### 3.7 Oversmoothing in Practice: Concrete Use Cases and ASI Topology Augmentation Experiments

The oversmoothing problem is not an abstraction — it is the primary bottleneck limiting GNN depth across multiple application domains. Li, Han & Wu (2018) first showed that GCN graph convolution is a special form of Laplacian smoothing, and that accuracy on the Cora citation network drops from ~81% at 2 layers to ~20% at just 6 layers. Rusch, Bronstein & Mishra (2023) provide a comprehensive formal treatment, demonstrating the effect across small-, medium-, and large-scale graphs and establishing that mitigating oversmoothing is necessary but not sufficient for building expressive deep GNNs. Existing countermeasures — residual connections (JKNet, Xu et al. 2018), edge dropout (DropEdge, Rong et al. 2019), normalization (PairNorm, Zhao & Akoglu 2019; DGN, Zhou et al. 2020), spectral adaptation (Laplacian-LoRA, 2025), and graph sparsification (TGS, Hossain et al. 2024) — all treat the symptom by modifying message-passing dynamics or pruning edges. None of them address the *topological* root cause: the graph Laplacian's spectral structure determines the oversmoothing rate, and standard graphs lack a topologically protected zero-eigenvalue subspace for edge signals. ASI-inspired topology augmentation attacks this root cause directly.

Below we identify five domains where oversmoothing is a documented, performance-limiting bottleneck and sketch how ASI frustrated topology augmentation could be tested as a remedy in each.

#### Use Case 1: Citation Network Node Classification (Cora, CiteSeer, PubMed)

**The problem.** The canonical setting where oversmoothing was discovered. These are semi-supervised node classification tasks on academic citation graphs (2,708 / 3,327 / 19,717 nodes). GCN accuracy peaks at 2 layers and degrades rapidly beyond 4 layers. Chen et al. (2020) quantified this via the MADGap metric and showed that topology adjustment (adding intra-class edges, removing inter-class edges) directly alleviates oversmoothing — demonstrating that the *graph structure itself* is a key lever.

**ASI augmentation experiment.** Lift the citation graph to a simplicial complex by identifying 3-cliques (triangles) and designating them as 2-cells. Then *selectively remove* some 2-cells to increase β₁ — creating "frustrated loops" analogous to ASI lattices where loops are not bounded by faces. Specifically:

1. Compute the full clique complex of the citation graph
2. Apply three face-removal strategies inspired by the ASI zoo: (a) *shakti-like* — remove faces from mixed-degree neighborhoods preferentially, (b) *random* — remove faces uniformly at random to a target β₁, (c) *spectral* — remove faces that most increase β₁ per removal (greedy)
3. Train SCNNs of depth 2, 4, 8, 16 on the resulting simplicial complexes
4. Compare node classification accuracy vs. depth against GCN baselines and existing oversmoothing mitigations (DropEdge, PairNorm, JKNet)

**Predicted outcome.** Higher β₁ from frustrated face removal → larger harmonic subspace → slower accuracy decay with depth. The spectral-greedy strategy should perform best, but even shakti-like heuristic removal should outperform the full clique complex (β₁ ≈ 0) at depth ≥ 8.

#### Use Case 2: Molecular Property Prediction (QM9, OGB-PCQM4Mv2)

**The problem.** Predicting quantum chemical properties of molecules from their 3D graph structure. Godwin et al. (2021) showed that oversmoothing limits deep GNNs on these tasks and proposed "Noisy Nodes" — corrupting input positions with noise and adding a denoising loss — as a regularizer that achieved state-of-the-art results on QM9 and the OGB-PCQM4Mv1 leaderboard. The underlying issue: molecular properties like HOMO-LUMO gap depend on delocalized electronic structure (a long-range, global property), but stacking enough GNN layers to capture this global structure destroys local atomic identity.

**ASI augmentation experiment.** Molecular graphs have natural ring structures (benzene, cyclopentane, etc.) that can be promoted to 2-cells. The key design choice: *which rings to fill as faces and which to leave open.*

1. For each molecule, enumerate all rings up to size 8
2. Apply face-assignment strategies: (a) *fill all rings* (standard clique complex — minimal β₁), (b) *fill only non-aromatic rings* (aromatic rings stay open → frustrated → β₁ increases), (c) *ASI-inspired mixed filling* — fill small rings (3,4-membered) as faces, leave larger rings open, mimicking the mixed coordination of shakti/tetris lattices
3. Train SCNN with Hodge-decomposed filters on QM9 regression tasks (dipole moment, HOMO, LUMO, gap, etc.) at depths 4, 8, 12, 16
4. Compare against GCN baselines, Noisy Nodes regularization, and DimeNet/SchNet (dedicated molecular architectures)

**Predicted outcome.** The aromatic-ring-open strategy should be particularly interesting: aromatic rings are precisely the molecular structures where electronic delocalization matters most, and leaving them as open cycles creates harmonic edge signals that the SCNN can use to represent delocalized electronic density. This is a case where the physics of the application (aromaticity) aligns naturally with the topology of the protection mechanism (unsaturated cycles → nonzero β₁ → harmonic channels).

#### Use Case 3: Traffic Flow Forecasting (METR-LA, PEMS-BAY)

**The problem.** Spatio-temporal GNNs on road sensor networks forecast traffic speed/flow at sensor locations. The graph is a spatial network of ~200–300 sensors connected by road proximity. Capturing long-range spatial dependencies (e.g., a highway accident 20 km upstream affecting downstream flow) requires deep spatial aggregation, but oversmoothing limits GNN depth to ~2 layers, forcing researchers to use temporal models (LSTMs, Neural ODEs) to compensate for spatial shallowness. The STGODE model (Fang et al. 2021) introduced Neural ODEs specifically to build deeper spatial GNNs without oversmoothing.

**ASI augmentation experiment.** Road networks have natural loop structure (city blocks, highway interchanges). These are physically meaningful — traffic circulates around blocks, and congestion propagates through loop topology.

1. Identify all minimal cycles (city blocks) in the road network graph
2. Apply face-assignment: (a) *fill all blocks* (standard — low β₁), (b) *fill only short blocks* (≤4 edges), leave large blocks open (shakti-like mixed frustration), (c) *leave highway interchange loops open* (these are the high-coordination vertices analogous to z=4 vertices in ASI)
3. Train SCNN on traffic forecasting: edge features = traffic speed on road segments (naturally an edge signal!), with temporal encoding
4. Compare against STGODE, DCRNN, and standard GCN-LSTM baselines

**Predicted outcome.** Traffic flow on road segments is *literally an edge signal* — making this a natural SCNN application even without the oversmoothing angle. The frustrated topology should help because congestion propagation through loops creates circulation patterns (curl component) that a standard GCN operating on vertex features cannot capture. Leaving interchange loops open provides harmonic channels that preserve long-range spatial information through depth.

#### Use Case 4: 3D Point Cloud Classification and Segmentation (ModelNet40, ShapeNet)

**The problem.** 3D point clouds from LiDAR or depth sensors are represented as KNN graphs, with each point as a vertex connected to its k nearest spatial neighbors. Classification and segmentation require both fine local geometry (edges, corners) and global shape understanding (overall object identity). DeepGCN (Li et al. 2019) showed that residual connections and dilated convolution can push GNN depth to 56 layers for point clouds, but the oversmoothing problem means that without these workarounds, accuracy degrades rapidly beyond ~3 layers. MLGCN (2024) circumvents the problem entirely by using multi-branch shallow GNNs rather than deep ones.

**ASI augmentation experiment.** KNN graphs on point clouds naturally contain many triangles (three mutually close points). The question is which triangles to promote to 2-cells.

1. Construct the KNN graph (k=20) for each point cloud
2. Identify all triangles. Apply face-assignment strategies: (a) *fill all triangles* (full clique complex), (b) *fill only triangles with edge lengths below median* (tight clusters get faces, loose connections stay open), (c) *ASI-inspired coordination mixing* — at high-degree vertices (analogous to z=4 in ASI), remove adjacent faces to create frustration; at low-degree boundary vertices (analogous to z=2), keep faces
3. Train SCNN at depths 4, 8, 16, 32 on ModelNet40 classification
4. Compare against DeepGCN (with residual + dilated convolution), DGCNN, and PointNet++ baselines

**Predicted outcome.** The coordination-mixing strategy should prevent oversmoothing at high-degree interior vertices (where it's worst, per the truss-based analysis of Hossain et al. 2024) while preserving smoothing at boundary vertices (where it's beneficial for capturing surface geometry). β₁ should be highest near object boundaries and topological features (handles, holes), providing protected channels for precisely the geometric information that matters for classification.

#### Use Case 5: Long-Range Graph Benchmark (LRGB: Peptides-func, Peptides-struct, PCQM-Contact)

**The problem.** The LRGB (Dwivedi et al. 2022) was explicitly designed to test whether GNN architectures can capture long-range interactions. The five datasets have large graphs (average shortest path lengths of 20+) where the learning task requires information exchange across the full graph diameter. Standard MP-GNNs perform poorly because they would need ~20+ layers to propagate information end-to-end, but oversmoothing kills them long before that depth. Graph Transformers (which bypass message passing via full attention) currently dominate these benchmarks — precisely because they avoid the oversmoothing problem entirely.

**ASI augmentation experiment.** This is the most direct test of the Pillar 1 thesis. If ASI-inspired topology augmentation can make deep SCNNs competitive with Graph Transformers on LRGB, that is a strong result.

1. For Peptides-func and Peptides-struct (molecular graphs of peptide chains), apply the ring-based augmentation from Use Case 2
2. For PCQM-Contact (predicting which atoms in a molecule are in 3D contact despite being far apart in the molecular graph), additionally add "shortcut" edges between predicted-contact atoms and promote the resulting triangles selectively — creating frustrated loops that bridge distant graph regions
3. For PascalVOC-SP (superpixel graphs from image segmentation), construct the Delaunay triangulation as 2-cells, then selectively remove faces at superpixel boundaries (where different object classes meet) to create frustrated boundaries with high β₁
4. Train SCNN at depths 4, 8, 16, 32 on all LRGB tasks
5. Compare against GCN, GIN, GraphGPS, SAN, and other LRGB leaderboard entries

**Predicted outcome.** The LRGB is where the oversmoothing-via-topology thesis gets its hardest test and its biggest potential payoff. If topology augmentation with controlled β₁ allows a message-passing SCNN to match or approach Graph Transformer performance on these benchmarks — without requiring full O(n²) attention — that is a significant result with practical implications for scaling GNNs to large graphs.

#### Summary of Use Case Experiments

| Use Case | Graph Type | Native Edge Signal? | Oversmoothing Evidence | ASI Augmentation Strategy | Key Prediction |
|---|---|---|---|---|---|
| Citation networks | Social/academic | No (vertex features) | Li et al. 2018: accuracy drops 81%→20% at 6 layers | Selective face removal from clique complex | β₁-controlled depth robustness |
| Molecular properties | Chemical | Partially (bond features) | Godwin et al. 2021: Noisy Nodes needed for depth | Aromatic rings left open as frustrated cycles | Harmonic channels ≈ delocalized electrons |
| Traffic forecasting | Spatial/road | Yes (flow on segments) | Fang et al. 2021: Neural ODEs needed for depth | City blocks as faces, interchange loops open | Circulation patterns in curl/harmonic |
| Point clouds | KNN/geometric | No (point features) | DeepGCN needs residual+dilated for >3 layers | Coordination mixing at high-degree vertices | Oversmoothing reduced where it's worst |
| LRGB | Mixed/large | Varies | Dwivedi et al. 2022: MP-GNNs fail at long range | Task-specific frustrated augmentation | SCNN competitive with Graph Transformers |

---

## 4. Pillar 2 — SCNN + Monte Carlo for Spin Ice Physics Discovery

### 4.1 The Opportunity: Edge Features on Their Natural Habitat

The SCNN was designed for signals on edges of a simplicial complex. Spin ice *literally is* a system of binary signals on edges. The mathematical formalism of TDL was built for exactly this physical setting — yet no one has applied it there. Meanwhile, the ASI community studies these systems with Monte Carlo simulation, mean-field theory, and analytic methods from graph theory, but has not used simplicial neural networks.

The core physics question: **Does the SCNN's Hodge decomposition (gradient / curl / harmonic) map onto the physically meaningful decomposition in spin ice (charge excitations / frustrated circulation / ice manifold)?**

If yes, this means the SCNN architecture has the right inductive bias to separate the relevant physics — and that simplicial neural networks could become a practical tool for studying frustrated systems, complementing Monte Carlo methods.

### 4.2 Experimental Setup

**Data generation:**

For each lattice in the zoo, generate ~10⁴ spin-ice configurations via Metropolis Monte Carlo at a range of temperatures spanning ordered → ice manifold → disordered phases. For each configuration, record:
- The binary spin state on every edge (the 1-cochain)
- The vertex charge field Qᵥ = Σ(spins in) − Σ(spins out) at every vertex
- The total system energy (dipolar Hamiltonian)
- The phase label (determined by order parameters: staggered magnetization, charge-charge correlation, etc.)

**Architecture comparison:**

Three architectures, tested on identical data across all lattice topologies:
- **GCN** — vertex features only (spin states averaged onto vertices as scalar features). Aggregation via normalized adjacency Â.
- **SNN** — edge features (raw spin states). Aggregation via combined Hodge Laplacian L₁ = L₁ᵈᵒʷⁿ + L₁ᵘᵖ (Ebli et al. 2020). Gradient and curl channels mixed.
- **SCNN** — edge features. Aggregation via *separate* L₁ᵈᵒʷⁿ and L₁ᵘᵖ with independent learnable filter weights (Yang et al. 2022). Gradient and curl channels decoupled.

### 4.3 Task Suite

**Task 1 — Phase Classification**

Given a spin configuration, classify the thermodynamic phase. This tests whether the network can distinguish order from disorder and identify the ice manifold.

*Physics interest:* Phase classification on the shakti and tetris lattices is nontrivial because these lattices have *topological* phase transitions (Lao et al. 2018), not conventional symmetry-breaking transitions. The order parameter is a topological charge rather than a magnetization. If the SCNN succeeds where the GCN fails, it's because the Hodge decomposition separates the topological from the conventional.

**Task 2 — Charge Prediction from Partial Observations**

Given spin states on a random subset of edges (say 60–80%), predict the vertex charge Qᵥ at all vertices.

*Physics interest:* This is directly relevant to experimental ASI, where MFM imaging may not resolve every nanomagnet. It also tests whether the network learns the divergence relationship Qᵥ = B₁ᵀ · s, which is the defining equation connecting edge spins to vertex charges.

**Task 3 — Energy Regression**

Given a full spin configuration, predict the total system energy.

*Physics interest:* The energy is a sum of pairwise dipolar interactions along edges. A GCN can learn this in principle, but the SCNN should learn it more efficiently because the energy factorizes naturally into contributions from the gradient and curl subspaces (Hodge decomposition of the energy functional). Comparing learned filter weights across lattice types reveals how the energy landscape changes with topology.

**Task 4 — Ground-State Completion**

Given a partial ground-state configuration, predict the remaining spins consistent with ice rules.

*Physics interest:* On the extensively degenerate shakti and tetris lattices, this is a constraint-satisfaction problem with many valid solutions. This task becomes more interesting with an ice-rule projection layer — projecting the SCNN output onto ker(B₁ᵀ) after each layer via P = I − B₁(B₁ᵀB₁)⁻¹B₁ᵀ, so the output is guaranteed divergence-free by construction. Comparing constrained vs. unconstrained SCNNs measures whether hard-coding the ice rule as architectural inductive bias helps or hurts generalization.

### 4.4 Analysis: What Do the Learned Filters Tell Us About the Physics?

Beyond task accuracy, the key scientific deliverable is **interpreting the learned SCNN filters through the Hodge decomposition:**

**Filter analysis:**
- After training, extract the learned polynomial coefficients γⱼ (lower Laplacian weights) and θⱼ (upper Laplacian weights) from each SCNN layer
- The ratio |γ|/|θ| tells us how much the network relies on vertex-mediated vs. face-mediated communication
- If |γ| >> |θ|, the physics is dominated by vertex-level (charge) interactions
- If |θ| >> |γ|, the physics is dominated by face-level (loop/frustration) interactions
- How does this ratio change across the lattice zoo?

**Prediction:** On the square lattice (where frustration is geometric, not topological), the lower and upper filters should learn similar weights — the physics doesn't strongly distinguish the two channels. On the shakti lattice (where frustration is topological and the ice manifold has extensive degeneracy), the upper filter should be much more important, because the face structure is what encodes the frustrated loops and the dimer-cover topology.

**Representation probing — Does the SCNN discover charges?**

Nisoli (2020) showed that the natural degrees of freedom in spin ice are not the binary edge spins but the *topological vertex charges*, interacting via TL₀⁻¹. This is a dramatic dimensionality reduction: the spins can be "integrated out," leaving an effective charge field theory.

We test whether the SCNN discovers this representation spontaneously:
1. Train the SCNN on energy regression (no explicit charge supervision)
2. Apply **linear probing classifiers** to intermediate layer representations: can a linear probe on layer k's vertex-aggregated features predict the charge Qᵥ at each vertex?
3. If yes: the SCNN has learned to internally represent charges without being told about them — its Hodge-decomposed architecture is biased toward the physically correct effective degrees of freedom
4. Compare SCNN vs. SNN vs. GCN: does the Hodge decomposition make charge discovery easier?

**Prediction:** The SCNN should discover charges earlier (in shallower layers) and more cleanly than the SNN, because its separate gradient filter directly processes the divergence structure. The GCN, operating on vertex features, may learn charges trivially (they're already on vertices) but will miss the curl/harmonic physics entirely.

### 4.5 Extension: Monopole Dynamics Prediction

If the static experiments succeed, the natural next step is dynamics. Magnetic monopoles in spin ice are emergent quasiparticles that live on vertices and hop via single spin flips on edges. Their dynamics is governed by the lattice topology and the entropic/energetic landscape.

1. Generate kinetic Monte Carlo trajectories: sequences of spin-flip events with timestamps
2. Frame each spin flip as a "message" along an edge, changing both adjacent vertex charges by ±1
3. Train SCNN to predict: (a) which edge flips next, (b) monopole trajectory over next N steps, (c) time to reach equilibrium

*Physics interest:* Monopole dynamics on the shakti lattice exhibits *topological constraints* — excitations are topologically protected and can only be created/annihilated in pairs (Lao et al. 2018). A network that learns monopole dynamics must implicitly learn these conservation laws. This is a direct test of whether SCNNs can discover topological invariants from dynamical data. The precedent is Roddenberry et al. (2021), who used simplicial networks for trajectory prediction on street networks — the same architecture applied to ASI monopole trajectories on frustrated lattices.

### 4.6 Deliverable

A systematic benchmark of GCN vs. SNN vs. SCNN across the ASI lattice zoo, demonstrating that:
1. The Hodge decomposition is physically meaningful — it separates charge physics (gradient) from frustration physics (curl) from ice manifold physics (harmonic)
2. The SCNN's independent lower/upper filters provide measurable accuracy gains on frustrated lattices, with the advantage scaling with the "topological complexity" of the lattice (quantified by β₁ and the frustration structure)
3. Learned filter weights encode interpretable physics: the γ/θ ratio maps onto the relative importance of vertex-level vs. face-level interactions
4. The SCNN spontaneously discovers the charge representation when trained on energy regression — confirming Nisoli's field theory via a completely independent computational method

---

## 5. Shared Infrastructure

Both pillars require the same computational foundation:

### 5.1 Lattice Generation
All ASI lattices can be generated programmatically from their unit cells. Typical simulation sizes: 20×20 to 100×100 unit cells → graphs with 10²–10⁴ vertices. Each lattice is represented as:
- Graph: adjacency matrix A, vertex list, edge list with orientations
- Simplicial complex: + incidence matrices B₁ (vertex-edge), B₁₂ (edge-face), with minimal loops/plaquettes as 2-cells
- Laplacians: L₀, L₁, L₁ᵈᵒʷⁿ, L₁ᵘᵖ, eigendecompositions, harmonic projectors

### 5.2 Monte Carlo Engine
Standard Metropolis algorithm with nearest-neighbor or dipolar Hamiltonian on arbitrary graph topologies. Validated against known results: Pauling entropy of square/kagome ice, ground states of shakti/tetris, monopole crystallization temperatures.

### 5.3 Software Stack
- `TopoModelX` / `TopoNetX` — Papillon et al. (2023) software suite for SCNN, SNN, CWN architectures
- `PyTorch Geometric` — baseline GCN/GAT comparisons
- Custom Monte Carlo code (Python + numba)
- `NetworkX` + `SciPy` for graph construction and sparse eigendecomposition

### 5.4 Computational Requirements
All experiments fit on a single GPU. Lattice eigendecompositions are the computational bottleneck but tractable up to ~10⁴ elements with SciPy sparse solvers. Monte Carlo generation is embarrassingly parallel.

---

## 6. Timeline

### Phase 1 — Foundation (2–3 months)
Build shared infrastructure:
1. Programmatic generation of all ASI lattice topologies as graphs and simplicial complexes
2. Spectral catalog: eigenspectra, Betti numbers, harmonic bases for each lattice at several sizes
3. Monte Carlo simulator validated against known results
4. Integration with TopoModelX for SCNN training

### Phase 2 — Core results (3–4 months)
Run both pillars in parallel:
5. **Pillar 1, Stages 2–3:** GCN oversmoothing baseline + SCNN harmonic protection experiments across the lattice zoo
6. **Pillar 2, Tasks 1–3:** Phase classification, charge prediction, energy regression benchmarks (GCN vs. SNN vs. SCNN)

### Phase 3 — Deep analysis (3–4 months)
7. **Pillar 1, Stage 4:** Topology augmentation experiments on standard benchmarks — citation networks (Cora/CiteSeer/PubMed), molecular graphs (QM9), traffic networks (METR-LA), point clouds (ModelNet40)
8. **Pillar 2, Filter analysis + probing:** Interpret learned SCNN filters, test for spontaneous charge discovery
9. **Pillar 2, Task 4:** Ground-state completion with ice-rule projection layer

### Phase 4 — Extensions
10. **Pillar 1, LRGB challenge:** Test ASI topology augmentation on Peptides-func/struct and PCQM-Contact — the hardest test of whether frustrated topology can make deep SCNNs competitive with Graph Transformers
11. Pillar 2: Monopole dynamics prediction (SCNN on kinetic Monte Carlo trajectories)
12. Write-up, targeting dual submission: ML venue (ICML/NeurIPS) for Pillar 1, physics venue (Phys. Rev. X / Phys. Rev. E) for Pillar 2

---

## 7. Key References

### Artificial Spin Ice
1. Morrison, M. J., Nelson, T. R. & Nisoli, C. "Unhappy vertices in artificial spin ice: new degeneracies from vertex frustration." *New J. Phys.* **15**, 045009 (2013).
2. Nisoli, C. "The concept of spin ice graphs and a field theory for their charges." *AIP Advances* **10**, 115102 (2020).
3. Gilbert, I. et al. "Emergent ice rule and magnetic charge screening from vertex frustration in artificial spin ice." *Nature Phys.* **10**, 670–675 (2014).
4. Gilbert, I. et al. "Emergent reduced dimensionality by vertex frustration in artificial spin ice." *Nature Phys.* **12**, 162–165 (2016).
5. Lao, Y. et al. "Classical topological order in the kinetics of artificial spin ice." *Nature Phys.* **14**, 723–727 (2018).
6. Nisoli, C., Moessner, R. & Schiffer, P. "Colloquium: Artificial spin ice: designing and imaging magnetic frustration." *Rev. Mod. Phys.* **85**, 1473 (2013).
7. Nisoli, C., Kapaklis, V. & Schiffer, P. "Deliberate exotic magnetism via frustration and topology." *Nature Phys.* **13**, 200–203 (2017).
8. Skjærvø, S. H. et al. "Advances in artificial spin ice." *Nature Rev. Phys.* **2**, 13–28 (2020).

### Topological Deep Learning / Simplicial Neural Networks
9. Hajij, M. et al. "Topological deep learning: going beyond graph data." arXiv:2206.00606 (2022).
10. Ebli, S., Defferrard, M. & Spreemann, G. "Simplicial neural networks." *NeurIPS TDA Workshop* (2020).
11. Yang, M., Isufi, E. & Leus, G. "Simplicial convolutional neural networks." *ICASSP* (2022).
12. Yang, M., Isufi, E., Schaub, M. T. & Leus, G. "Simplicial convolutional filters." *IEEE Trans. Signal Processing* **70**, 4633–4648 (2022).
13. Bodnar, C. et al. "Weisfeiler and Leman go topological: Message passing simplicial networks." *ICML* (2021).
14. Roddenberry, T. M., Glaze, N. & Segarra, S. "Principled simplicial neural networks for trajectory prediction." *ICML* (2021).
15. Schaub, M. T. et al. "Random walks on simplicial complexes and the normalized Hodge 1-Laplacian." *SIAM Review* **62**, 353–391 (2020).
16. Papillon, M. et al. "Architectures of topological deep learning: a survey of message-passing topological neural networks." arXiv:2304.10031 (2023).
17. Yang, M., Leus, G. & Isufi, E. "Hodge-aware convolutional learning on simplicial complexes." *TMLR* (2025).
18. Einizade, A. et al. "Continuous simplicial neural networks (COSIMO)." arXiv:2503.12919 (2025).

### ML for Frustrated/Spin Systems
19. Carrasquilla, J. & Melko, R. G. "Machine learning phases of matter." *Nature Phys.* **13**, 431 (2017).

### Oversmoothing in GNNs
20. Li, Q., Han, Z. & Wu, X.-M. "Deeper insights into graph convolutional networks for semi-supervised learning." *AAAI* (2018). — First identification of GCN convolution as Laplacian smoothing; oversmoothing demonstrated on Cora/CiteSeer/PubMed.
21. Rusch, T. K., Bronstein, M. M. & Mishra, S. "A survey on oversmoothing in graph neural networks." arXiv:2303.10993 (2023). — Comprehensive formal treatment; axiomatic definition; empirical evaluation across scales.
22. Chen, D. et al. "Measuring and relieving the over-smoothing problem for graph neural networks from the topological view." *AAAI* **34**(04), 3438–3445 (2020). — MADGap metric; topology adjustment experiments on Cora/CiteSeer/PubMed.
23. Rong, Y. et al. "DropEdge: towards deep graph convolutional networks on node classification." *ICLR* (2020). — Random edge removal to slow oversmoothing.
24. Zhao, L. & Akoglu, L. "PairNorm: tackling oversmoothing in GNNs." *ICLR* (2020). — Normalization-based mitigation.
25. Xu, K. et al. "Representation learning on graphs with jumping knowledge networks." *ICML* (2018). — Skip connections aggregating across layers.
26. Godwin, J. et al. "Simple GNN regularisation for 3D molecular property prediction and beyond." arXiv:2106.07971 (2021). — Noisy Nodes regularizer; SOTA on QM9 and OGB-PCQM4Mv1.
27. Dwivedi, V. P. et al. "Long range graph benchmark." *NeurIPS Datasets and Benchmarks Track* (2022). — Five datasets requiring long-range reasoning; MP-GNNs underperform Graph Transformers.
28. Hossain, T. et al. "Tackling oversmoothing in GNN via graph sparsification." *ECML-PKDD* (2024). — Truss-based analysis; dense regions oversmooth first.
29. Fang, Z. et al. "Spatial-temporal graph ODE networks for traffic forecasting." *KDD* (2021). — Neural ODEs for deeper spatial GNNs on road networks.
30. Li, G. et al. "DeepGCNs: can GCNs go as deep as CNNs?" *ICCV* (2019). — Residual + dilated convolution for 56-layer GCN on point clouds.

---

## 8. Why This Matters

The two pillars tell a single coherent story from opposite ends:

**From the ML side:** Current SCNN architecture design has no principled way to choose the topology of the underlying simplicial complex. Pillar 1 provides one: **maximize β₁ to maximize the dimensionality of the topologically protected information channel.** The ASI lattice zoo, with its analytically characterized frustration structures and known Betti numbers, is the ideal laboratory for demonstrating and quantifying this principle. If it works on ASI lattices, the prescription generalizes: for any domain where you can choose or augment the simplicial structure, you can engineer β₁ to control depth-robustness.

**From the physics side:** The SCNN's Hodge-decomposed filters provide a new computational lens for studying frustrated spin systems. Unlike Monte Carlo, which samples configurations, or analytic field theory, which derives effective Hamiltonians, the SCNN *learns* the relevant decomposition from data. If the learned gradient/curl/harmonic filters align with the known charge/frustration/manifold physics, that's independent computational confirmation of the theoretical picture. If the filters reveal something *unexpected* — say, that certain lattice topologies have physics that doesn't cleanly decompose into the three Hodge subspaces — that's a new physical insight pointing to interactions between the subspaces that the standard theory doesn't capture.

**The deep connection:** The same quantity, β₁, simultaneously measures the size of the ice manifold in ASI (physics) and the capacity of the protected information channel in SCNNs (ML). This isn't a metaphor — it's the same vector space. Understanding it from both sides creates a feedback loop: physics insights about which topologies produce large β₁ inform neural network design, while neural network experiments on those topologies reveal new physics about how the Hodge subspaces interact under learnable nonlinear transformations.
