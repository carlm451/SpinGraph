"""Save/load SpectralResult as .npz + .json files."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.spectral.catalog import SpectralResult


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _result_key(result: SpectralResult) -> str:
    boundary = getattr(result, 'boundary', 'periodic')
    if boundary == "periodic":
        return f"{result.lattice_name}_{result.size_label}_{result.face_strategy}"
    return f"{result.lattice_name}_{result.size_label}_{result.face_strategy}_{boundary}"


def save_result(result: SpectralResult, directory: str) -> None:
    """Save a SpectralResult to .npz (arrays) + _meta.json (metadata)."""
    os.makedirs(directory, exist_ok=True)
    key = _result_key(result)

    # Save arrays
    arrays = {}
    if result.L0_eigenvalues is not None:
        arrays["L0_eigenvalues"] = result.L0_eigenvalues
    if result.L1_eigenvalues is not None:
        arrays["L1_eigenvalues"] = result.L1_eigenvalues
    if result.L1_down_eigenvalues is not None:
        arrays["L1_down_eigenvalues"] = result.L1_down_eigenvalues
    if result.L1_up_eigenvalues is not None:
        arrays["L1_up_eigenvalues"] = result.L1_up_eigenvalues
    if result.harmonic_basis is not None and result.harmonic_basis.shape[1] > 0:
        arrays["harmonic_basis"] = result.harmonic_basis

    np.savez_compressed(os.path.join(directory, f"{key}.npz"), **arrays)

    # Save metadata
    meta = {
        "lattice_name": result.lattice_name,
        "size_label": result.size_label,
        "nx": result.nx,
        "ny": result.ny,
        "boundary": result.boundary,
        "face_strategy": result.face_strategy,
        "n_vertices": result.n_vertices,
        "n_edges": result.n_edges,
        "n_faces": result.n_faces,
        "coordination_dist": result.coordination_dist,
        "beta_0": result.beta_0,
        "beta_1": result.beta_1,
        "beta_2": result.beta_2,
        "euler_consistent": result.euler_consistent,
        "method_agreement": result.method_agreement,
        "chain_complex_valid": result.chain_complex_valid,
        "L0_spectral_gap": result.L0_spectral_gap,
        "L1_spectral_gap": result.L1_spectral_gap,
        "compute_time_seconds": result.compute_time_seconds,
    }
    if result.harmonic_validation:
        meta["harmonic_validation"] = result.harmonic_validation

    with open(os.path.join(directory, f"{key}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, cls=_NumpyEncoder)


def load_result(
    directory: str,
    lattice_name: str,
    size_label: str,
    face_strategy: str,
    boundary: str = "periodic",
) -> Dict:
    """Load a saved result as a dict with arrays and metadata."""
    if boundary == "periodic":
        key = f"{lattice_name}_{size_label}_{face_strategy}"
    else:
        key = f"{lattice_name}_{size_label}_{face_strategy}_{boundary}"

    # Load metadata
    meta_path = os.path.join(directory, f"{key}_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    # Load arrays
    npz_path = os.path.join(directory, f"{key}.npz")
    arrays = dict(np.load(npz_path))

    return {**meta, **arrays}


def update_catalog_index(directory: str) -> None:
    """Scan directory and write catalog_index.json listing all results."""
    entries = []
    for fname in sorted(os.listdir(directory)):
        if fname.endswith("_meta.json"):
            meta_path = os.path.join(directory, fname)
            with open(meta_path) as f:
                meta = json.load(f)
            entries.append({
                "lattice_name": meta["lattice_name"],
                "size_label": meta["size_label"],
                "face_strategy": meta["face_strategy"],
                "boundary": meta.get("boundary", "periodic"),
                "beta_1": meta["beta_1"],
                "n_vertices": meta["n_vertices"],
                "n_edges": meta["n_edges"],
                "n_faces": meta["n_faces"],
            })

    with open(os.path.join(directory, "catalog_index.json"), "w") as f:
        json.dump(entries, f, indent=2)
