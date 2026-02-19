# Eigensolver Research: Pushing Full Spectrum Computation to Larger Sizes

*Date: 2026-02-15*
*Hardware: Apple M1 Pro, 16 GB RAM, OpenBLAS 0.3.30, Python 3.13*

## Current Setup

The spectral catalog computes eigenvalues of graph Laplacians (L0, L1, L1_down, L1_up) which are sparse, symmetric, positive semi-definite matrices with 4-8 nonzeros per row (bounded coordination number) and eigenvalues in [0, 2*z_max].

| Method | Condition | Returns |
|--------|-----------|---------|
| Dense `scipy.linalg.eigh` | n <= 5,000 (DENSE_THRESHOLD) | ALL n eigenvalues |
| Sparse `scipy.sparse.linalg.eigsh` | n > 5,000 | k=100 smallest only |

The full eigenvalue spectrum is needed for density-of-states (DOS) plots. The partial (k=100) spectrum suffices for spectral gaps and Betti numbers but not for histogram/distribution analysis.

---

## Changes Implemented (2026-02-15)

1. **Raised DENSE_THRESHOLD from 5,000 to 15,000** -- unlocks full spectra for Kagome L (n_e=7,500), Shakti M (n_e=9,600), and all L0 matrices through L size.

2. **Added `driver='evd'`** to `scipy.linalg.eigh` calls -- uses LAPACK's DSYEVD (divide-and-conquer) which is 2-4x faster than the default DSYEVR for full eigendecomposition. Uses ~50% more memory (3n^2 vs 2n^2 bytes) but well within limits.

3. **Raised SHIFT_INVERT_THRESHOLD from 20,000 to 50,000** -- 2D lattice Laplacians have well-behaved LU fill-in (O(n*sqrt(n)) for 2D mesh graphs), so shift-invert is safe at much larger sizes.

---

## Dense Solver Scaling on This Hardware

`scipy.linalg.eigh` with OpenBLAS on M1 Pro:

| n | Memory (matrix + eigvecs) | Estimated Time (evr) | Estimated Time (evd) |
|---|--------------------------|----------------------|---------------------|
| 5,000 | 400 MB | 10-15 s | 5-8 s |
| 10,000 | 1.6 GB | 1.5-2 min | 40-60 s |
| 15,000 | 3.6 GB (evr) / 5.4 GB (evd) | 5-7 min | 2-3 min |
| 20,000 | 6.4 GB (evr) / 9.6 GB (evd) | 10-16 min | 4-7 min |
| 30,000 | 14.4 GB | 36-54 min | swap/OOM |

**Hard ceiling**: n = 20,000 with `evr` driver, n = 15,000 with `evd` driver (16 GB RAM).

---

## Spectrum Slicing (Next Step for n = 15,000-30,000)

For matrices too large for dense decomposition, the full spectrum can be computed by partitioning the known eigenvalue range [0, lambda_max] into intervals and using shift-invert `eigsh` at each interval center.

### Algorithm

```python
def spectrum_slice(L, n_intervals=15, k_per_interval=2000):
    """Compute full spectrum via shift-invert spectrum slicing."""
    # 1. Estimate lambda_max
    lam_max = eigsh(L, k=1, which='LM', return_eigenvectors=False)[0]

    # 2. Define interval centers
    delta = lam_max / n_intervals
    sigmas = [delta * (i + 0.5) for i in range(n_intervals)]

    # 3. Compute eigenvalues near each sigma
    all_evals = []
    for sigma in sigmas:
        evals, _ = eigsh(L, k=k_per_interval, sigma=sigma, which='LM')
        all_evals.append(evals)

    # 4. Concatenate and deduplicate
    all_evals = np.concatenate(all_evals)
    all_evals = np.sort(all_evals)
    # Remove duplicates within tolerance
    unique = [all_evals[0]]
    for v in all_evals[1:]:
        if abs(v - unique[-1]) > 1e-10:
            unique.append(v)
    return np.array(unique)
```

### Expected Performance

| n | Memory | Time | Notes |
|---|--------|------|-------|
| 20,000 | ~500 MB | 5-15 min | 15 intervals, k=2000 each |
| 30,000 | ~1 GB | 15-45 min | 15 intervals, k=2000 each |
| 60,000 | ~2-4 GB | 1-3 hours | 15-20 intervals, k=4000 each |

### Caveats

- Completeness not guaranteed: eigenvalues in sparse spectral regions could be missed. Use overlapping intervals and validate total count against matrix dimension.
- Each interval requires an LU factorization of (L - sigma*I). For 2D lattice Laplacians, fill-in is manageable: O(n * sqrt(n)).
- Consider using eigenvalue density estimates (from Chebyshev moments or Sylvester's inertia theorem) to adaptively size k_per_interval.

---

## Alternative Solvers

### PRIMME (Recommended for spectrum slicing improvement)

**Installation**: `pip install primme`

Drop-in replacement for `scipy.sparse.linalg.eigsh` with better convergence for interior eigenvalues (critical for spectrum slicing where sigma is far from the extremes).

```python
import primme
evals, evecs = primme.eigsh(L, k=k, sigma=sigma, which='LM')
```

Key advantages:
- Jacobi-Davidson methods converge faster than ARPACK's Lanczos for interior eigenvalues
- Block methods find multiple eigenvalues simultaneously
- Preconditioning support for further acceleration
- More robust near-zero eigenvalue finding (`which='SA'`)

**Estimated benefit**: 2-5x speedup over scipy eigsh for spectrum slicing intervals.

### FEAST (Contour Integration)

The FEAST algorithm computes all eigenvalues in a user-specified interval [a, b] using contour integration (Cauchy's residue theorem). This is the most mathematically elegant approach to spectrum slicing.

- **Python access**: PyFEAST (https://github.com/empter/PYFEAST) -- requires MKL or OpenBLAS + Cython
- **Caveat**: MKL unavailable on Apple Silicon; PyFEAST has limited maintenance
- **Advantage**: Each interval solve is embarrassingly parallel (across intervals and within each solve)
- **FEAST v4.0**: 3-4x faster than v3.0 thanks to inverse residual iteration

Worth investigating if spectrum slicing with eigsh proves too slow or unreliable.

### SLEPc / slepc4py

Gold standard for large-scale sparse eigenvalue problems. Features Krylov-Schur, Contour Integral Spectrum Slicing (CISS), Jacobi-Davidson. Full MPI parallel support.

- **Installation**: Complex -- requires building PETSc, then SLEPc, then Python bindings
- **Verdict**: Overkill for n <= 60,000 on a single laptop. Worth it for n > 100,000 or distributed computing.

### Kernel Polynomial Method (KPM)

Approximates the density of states directly in O(n * m) time using Chebyshev moment expansion, where m = number of moments (100-500). Does NOT compute individual eigenvalues.

**When useful**: If we only need approximate DOS histograms (not exact eigenvalues), KPM can estimate the spectral density for n = 100,000+ in minutes. Each moment requires one sparse matrix-vector product.

**Implementation**: ~100-200 lines of Python using sparse matrix-vector products. No special libraries needed.

**Verdict**: Best option if the goal is only DOS visualization at very large sizes. Not suitable if exact eigenvalues are needed for other analyses.

### Not Viable on This Hardware

| Solver | Why Not |
|--------|---------|
| CuPy / GPU solvers | No NVIDIA GPU (M1 Pro has no CUDA) |
| Intel MKL | x86-only binary, not available on Apple Silicon |
| JAX eigh | No GPU benefit on this hardware, same LAPACK backend |
| ELPA | HPC-only, MPI + GPU cluster oriented |
| ChASE | Distributed GPU cluster (144 nodes, 526 A100s in benchmarks) |

---

## Lattice Size Reference

Matrix dimensions at each catalog size (periodic BCs, all faces):

| Lattice | n_v (XS/S/M/L/XL) | n_e (XS/S/M/L/XL) |
|---------|-------------------|-------------------|
| Square | 16/100/400/2500/10000 | 32/200/800/5000/20000 |
| Kagome | 32/200/800/5000/20000 | 48/300/1200/7500/30000 |
| Shakti | 256/1600/6400/40000/160000 | 384/2400/9600/60000/240000 |
| Tetris | 128/800/3200/20000/80000 | 192/1200/4800/30000/120000 |
| Santa Fe | 96/600/2400/15000/60000 | 192/1200/4800/22500/90000 |

With DENSE_THRESHOLD = 15,000, full spectra available for:
- **L0**: All lattices through L size (max n_v = 40,000 for shakti, but we use sparse there; square/kagome/tetris/santa_fe all under 15,000 at L)
- **L1**: Square through L (5,000), Kagome through L (7,500), Shakti through M (9,600), Tetris through M (4,800), Santa Fe through M (4,800)

---

## References

- scipy.linalg.eigh: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html
- scipy.sparse.linalg.eigsh: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html
- PRIMME: https://pypi.org/project/primme/
- FEAST: https://www.feast-solver.org/
- SLEPc: https://slepc.upv.es/
- Spectrum slicing shift selection: https://dl.acm.org/doi/fullHtml/10.1145/3409571
- scipy eigh performance: https://github.com/scipy/scipy/issues/9212
