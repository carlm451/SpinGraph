"""Save and load training run outputs for diagnostic analysis.

Stores all artifacts from a single training run in a structured directory:
  config.json, metrics.npz, model_final.pt, exact_states.npy,
  final_samples.npy, final_log_probs.npy, seed_state.npy,
  positions.npy, edge_list.npy, coordination.npy
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import torch


@dataclass
class RunMetadata:
    """Metadata for a single training run."""

    run_id: str
    lattice_name: str
    size_label: str  # "XS", "S", etc.
    boundary: str
    n_vertices: int
    n_edges: int
    beta_1: int
    n_reachable_states: int  # from enumeration (or -1 if not enumerated)
    n_model_params: int
    # Training config fields
    n_epochs: int
    batch_size: int
    lr: float
    lr_scheduler: str
    entropy_bonus: float
    grad_clip: float
    eval_every: int
    seed: int
    # Model hyperparams
    n_layers: int
    equ_dim: int
    inv_dim: int
    head_hidden: int
    # Timing
    enum_time_s: float
    train_time_s: float
    sample_time_s: float
    timestamp: str


def generate_run_id(lattice_name: str, boundary: str) -> str:
    """Generate a run ID from lattice name, boundary, and timestamp."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{lattice_name}_{boundary}_{ts}"


def save_training_run(
    run_dir: str,
    metadata: RunMetadata,
    result,  # TrainingResult
    model,  # LoopMPVAN
    exact_states: Optional[np.ndarray],
    final_samples: np.ndarray,
    final_log_probs: np.ndarray,
    sigma_seed: np.ndarray,
    positions: np.ndarray,
    edge_list: list,
    coordination: np.ndarray,
) -> str:
    """Save all training run artifacts to run_dir.

    Parameters
    ----------
    run_dir : str
        Directory to write outputs into (created if needed).
    metadata : RunMetadata
    result : TrainingResult
        Contains loss_history, entropy_history, kl_history, etc.
    model : LoopMPVAN
        For saving state_dict.
    exact_states : array (n_states, n_edges) or None
    final_samples : array (n_samples, n_edges)
    final_log_probs : array (n_samples,)
    sigma_seed : array (n_edges,)
    positions : array (n_vertices, 2)
    edge_list : list of (int, int)
    coordination : array (n_vertices,)

    Returns
    -------
    run_dir : str
        The directory where outputs were saved.
    """
    os.makedirs(run_dir, exist_ok=True)

    # config.json
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(asdict(metadata), f, indent=2)

    # metrics.npz
    metrics_path = os.path.join(run_dir, "metrics.npz")
    metrics_dict = {
        "loss_history": np.array(result.loss_history, dtype=np.float64),
        "entropy_history": np.array(result.entropy_history, dtype=np.float64),
        "kl_history": np.array(result.kl_history, dtype=np.float64),
        "hamming_history": np.array(result.hamming_history, dtype=np.float64),
        "ess_history": np.array(result.ess_history, dtype=np.float64),
        "eval_epochs": np.array(result.eval_epochs, dtype=np.int64),
    }
    np.savez_compressed(metrics_path, **metrics_dict)

    # model_final.pt
    model_path = os.path.join(run_dir, "model_final.pt")
    torch.save(model.state_dict(), model_path)

    # exact_states.npy (if available)
    if exact_states is not None:
        np.save(os.path.join(run_dir, "exact_states.npy"), exact_states)

    # final_samples.npy, final_log_probs.npy
    np.save(os.path.join(run_dir, "final_samples.npy"), final_samples)
    np.save(os.path.join(run_dir, "final_log_probs.npy"), final_log_probs)

    # seed_state.npy
    np.save(os.path.join(run_dir, "seed_state.npy"), sigma_seed)

    # Lattice geometry for plotting
    np.save(os.path.join(run_dir, "positions.npy"), positions)
    np.save(os.path.join(run_dir, "edge_list.npy"), np.array(edge_list, dtype=np.int64))
    np.save(os.path.join(run_dir, "coordination.npy"), coordination)

    return run_dir


def load_training_run(run_dir: str) -> dict:
    """Load all training run artifacts from run_dir.

    Returns
    -------
    data : dict with keys:
        'metadata' : dict (from config.json)
        'loss_history', 'entropy_history', 'kl_history',
        'hamming_history', 'ess_history', 'eval_epochs' : arrays
        'final_samples', 'final_log_probs', 'seed_state' : arrays
        'exact_states' : array or None
        'positions', 'edge_list', 'coordination' : arrays
    """
    # config.json
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path) as f:
        metadata = json.load(f)

    # metrics.npz
    metrics_path = os.path.join(run_dir, "metrics.npz")
    metrics = np.load(metrics_path)

    data = {
        "metadata": metadata,
        "loss_history": metrics["loss_history"],
        "entropy_history": metrics["entropy_history"],
        "kl_history": metrics["kl_history"],
        "hamming_history": metrics["hamming_history"],
        "ess_history": metrics["ess_history"],
        "eval_epochs": metrics["eval_epochs"],
    }

    # final_samples, final_log_probs, seed_state
    data["final_samples"] = np.load(os.path.join(run_dir, "final_samples.npy"))
    data["final_log_probs"] = np.load(os.path.join(run_dir, "final_log_probs.npy"))
    data["seed_state"] = np.load(os.path.join(run_dir, "seed_state.npy"))

    # exact_states (optional)
    exact_path = os.path.join(run_dir, "exact_states.npy")
    data["exact_states"] = np.load(exact_path) if os.path.exists(exact_path) else None

    # Lattice geometry
    data["positions"] = np.load(os.path.join(run_dir, "positions.npy"))
    data["edge_list"] = np.load(os.path.join(run_dir, "edge_list.npy"))
    data["coordination"] = np.load(os.path.join(run_dir, "coordination.npy"))

    return data
