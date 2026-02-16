"""Load precomputed spectral catalog results with LRU caching.

Provides cached accessors for the catalog index, individual spectral results,
and on-the-fly lattice construction for visualization. All functions assume
the project is run from the repository root so that ``src`` is importable.
"""
from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so ``src.*`` imports resolve
# regardless of where the dashboard process is launched from.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.io.serialize import load_result as _raw_load_result
from src.lattices.registry import LATTICE_REGISTRY


# ---------------------------------------------------------------------------
# Catalog directory
# ---------------------------------------------------------------------------

def get_catalog_dir() -> str:
    """Return the absolute path to the catalog directory.

    Defaults to ``<project_root>/results/catalog``.  Override by setting the
    environment variable ``SPINICE_CATALOG_DIR``.
    """
    default = os.path.join(str(_PROJECT_ROOT), "results", "catalog")
    return os.environ.get("SPINICE_CATALOG_DIR", default)


# ---------------------------------------------------------------------------
# Catalog index
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_available_results() -> List[Dict]:
    """Read ``catalog_index.json`` and return the list of result entries.

    Each entry is a dict with keys: lattice_name, size_label, face_strategy,
    boundary, beta_1, n_vertices, n_edges, n_faces.

    The result is cached; call ``get_available_results.cache_clear()`` to
    force a re-read after the catalog is updated on disk.
    """
    index_path = os.path.join(get_catalog_dir(), "catalog_index.json")
    with open(index_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Single result loader
# ---------------------------------------------------------------------------

@lru_cache(maxsize=128)
def load_result(
    lattice_name: str,
    size_label: str,
    face_strategy: str,
    boundary: str = "periodic",
) -> Dict:
    """Load a single precomputed spectral result.

    Returns a dict merging the JSON metadata and the numpy arrays stored in
    the companion ``.npz`` file.  Keys include ``lattice_name``,
    ``size_label``, ``face_strategy``, ``beta_1``, ``L0_eigenvalues``,
    ``L1_eigenvalues``, ``harmonic_basis``, etc.

    Parameters
    ----------
    lattice_name : str
        E.g. ``'square'``, ``'kagome'``, ``'shakti'``.
    size_label : str
        One of ``'XS'``, ``'S'``, ``'M'``, ``'L'``, ``'XL'``.
    face_strategy : str
        ``'all'`` (fill all minimal faces) or ``'none'`` (no faces).
    boundary : str
        ``'periodic'`` or ``'open'``.
    """
    return _raw_load_result(
        directory=get_catalog_dir(),
        lattice_name=lattice_name,
        size_label=size_label,
        face_strategy=face_strategy,
        boundary=boundary,
    )


# ---------------------------------------------------------------------------
# Lattice builder for visualization
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32)
def load_lattice_for_viz(
    lattice_name: str,
    nx: int,
    ny: int,
    boundary: str = "periodic",
) -> Dict:
    """Build a lattice on-the-fly and return geometry for visualization.

    Parameters
    ----------
    lattice_name : str
        Must be a key in ``LATTICE_REGISTRY``.
    nx, ny : int
        Number of unit-cell repetitions in each direction.
    boundary : str
        ``'periodic'`` or ``'open'``.

    Returns
    -------
    dict with keys:
        positions : np.ndarray, shape (n_vertices, 2)
        edge_list : list of (int, int) tuples
        face_list : list of list[int]
        coordination : np.ndarray, shape (n_vertices,)
        n_vertices : int
        n_edges : int
        n_faces : int
        coordination_distribution : dict mapping int -> int
    """
    if lattice_name not in LATTICE_REGISTRY:
        available = ", ".join(sorted(LATTICE_REGISTRY.keys()))
        raise KeyError(
            f"Unknown lattice '{lattice_name}'. Available: {available}"
        )

    generator = LATTICE_REGISTRY[lattice_name]()
    result = generator.build(nx, ny, boundary=boundary)

    return {
        "positions": result.positions,
        "edge_list": result.edge_list,
        "face_list": result.face_list,
        "coordination": result.coordination,
        "n_vertices": result.n_vertices,
        "n_edges": result.n_edges,
        "n_faces": result.n_faces,
        "coordination_distribution": result.coordination_distribution,
        "a1": result.unit_cell.a1,
        "a2": result.unit_cell.a2,
        "nx_size": result.nx_size,
        "ny_size": result.ny_size,
        "boundary": result.boundary,
    }


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def get_lattice_names() -> List[str]:
    """Return sorted list of unique lattice names present in the catalog."""
    results = get_available_results()
    return sorted({entry["lattice_name"] for entry in results})


def get_ice_manifold_data() -> List[Dict]:
    """Return ice manifold dimensions for each (lattice, size, boundary) combo.

    The ice manifold dimension is ``n_edges - n_vertices + 1`` for connected
    graphs (beta_0 = 1).  It depends only on the graph, not on face strategy,
    so results are deduplicated by ``(lattice_name, size_label, boundary)``.

    Each returned dict contains: lattice_name, size_label, boundary,
    n_vertices, n_edges, ice_manifold_dim, ice_manifold_fraction.
    """
    results = get_available_results()
    seen: Dict[Tuple[str, str, str], Dict] = {}
    for entry in results:
        key = (entry["lattice_name"], entry["size_label"],
               entry.get("boundary", "periodic"))
        if key in seen:
            continue
        n_v = entry["n_vertices"]
        n_e = entry["n_edges"]
        ice_dim = n_e - n_v + 1
        seen[key] = {
            "lattice_name": entry["lattice_name"],
            "size_label": entry["size_label"],
            "boundary": entry.get("boundary", "periodic"),
            "n_vertices": n_v,
            "n_edges": n_e,
            "ice_manifold_dim": ice_dim,
            "ice_manifold_fraction": ice_dim / n_e if n_e > 0 else 0.0,
        }
    return list(seen.values())


def get_spectral_gap_data() -> List[Dict]:
    """Collect spectral gap data across all catalog entries.

    For each entry in the catalog index, loads the full result to extract
    L0_spectral_gap, L1_spectral_gap (from metadata), and L1_down_spectral_gap
    (computed from L1_down_eigenvalues array if available).

    Returns a list of dicts with keys: lattice_name, size_label, boundary,
    face_strategy, n_vertices, n_edges, nx, L0_spectral_gap, L1_spectral_gap,
    L1_down_spectral_gap (may be None if eigenvalues not stored).
    """
    results = get_available_results()
    gap_data: List[Dict] = []
    tol = 1e-10

    for entry in results:
        lattice = entry["lattice_name"]
        size = entry["size_label"]
        face = entry["face_strategy"]
        bc = entry.get("boundary", "periodic")

        try:
            full = load_result(lattice, size, face, bc)
        except Exception:
            continue

        l1_down_gap = None
        l1_down_eigs = full.get("L1_down_eigenvalues")
        if l1_down_eigs is not None and len(l1_down_eigs) > 0:
            nonzero = l1_down_eigs[np.abs(l1_down_eigs) >= tol]
            if len(nonzero) > 0:
                l1_down_gap = float(np.min(np.abs(nonzero)))
            elif len(l1_down_eigs) < entry["n_edges"]:
                # Sparse solver stored fewer eigenvalues than edges exist;
                # all captured eigenvalues sit in the null space so the
                # actual gap was not reached.  Mark as unknown.
                l1_down_gap = None
            else:
                l1_down_gap = 0.0

        # Fallback: when L1_down eigenvalues are partial or missing, use
        # the metadata L1_spectral_gap.  For these lattices the Hodge gap
        # is always determined by L1_down (gradient sector), so the two
        # values agree wherever both are available.
        if l1_down_gap is None:
            meta_gap = full.get("L1_spectral_gap")
            if meta_gap is not None and meta_gap > 0:
                l1_down_gap = float(meta_gap)

        gap_data.append({
            "lattice_name": lattice,
            "size_label": size,
            "boundary": bc,
            "face_strategy": face,
            "n_vertices": entry["n_vertices"],
            "n_edges": entry["n_edges"],
            "nx": full.get("nx"),
            "L0_spectral_gap": full.get("L0_spectral_gap"),
            "L1_spectral_gap": full.get("L1_spectral_gap"),
            "L1_down_spectral_gap": l1_down_gap,
        })

    return gap_data


def get_available_sizes(lattice_name: str, boundary: str = "periodic") -> List[str]:
    """Return list of available size labels for a given lattice name.

    The returned list preserves the canonical ordering
    ``['XS', 'S', 'M', 'L', 'XL']`` and only includes sizes that actually
    exist in the catalog index.
    """
    canonical_order = ["XS", "S", "M", "L", "XL"]
    results = get_available_results()
    available = {
        entry["size_label"]
        for entry in results
        if entry["lattice_name"] == lattice_name
        and entry.get("boundary", "periodic") == boundary
    }
    return [s for s in canonical_order if s in available]
