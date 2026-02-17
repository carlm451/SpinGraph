#!/usr/bin/env python3
"""Generate ensemble-averaged diagnostic plots from a set of training runs.

Loads metrics.npz from each run in an ensemble, computes mean +/- std across
runs, and produces overlay + shaded-band versions of the key diagnostic panels.

Usage:
    python -m scripts.plot_ensemble_diagnostics \
        --ensemble-dir results/neural_training/ensemble-square-4-open

    # Or specify run dirs explicitly:
    python -m scripts.plot_ensemble_diagnostics \
        --run-dirs results/neural_training/run1 results/neural_training/run2 ...
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.neural.checkpointing import load_training_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Publication style
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

_BLUE = "#3498db"
_RED = "#e74c3c"
_GRAY = "#95a5a6"
_GREEN = "#27ae60"
_COLORS = ["#3498db", "#e74c3c", "#27ae60", "#f39c12", "#9b59b6",
           "#1abc9c", "#e67e22", "#2ecc71"]


def _truncate_to_common_length(*arrays: np.ndarray) -> list[np.ndarray]:
    """Truncate arrays to the shortest common length."""
    min_len = min(len(a) for a in arrays)
    return [a[:min_len] for a in arrays]


def _compute_band(arrays: list[np.ndarray]):
    """Compute mean and std from a list of arrays, truncated to common length."""
    min_len = min(len(a) for a in arrays)
    stacked = np.array([a[:min_len] for a in arrays])
    mean = np.mean(stacked, axis=0)
    std = np.std(stacked, axis=0)
    return mean, std, min_len


def plot_ensemble_training_curves(
    all_run_data: list[dict],
    output_path: str,
    ensemble_label: str = "",
) -> str:
    """Panel 1: Ensemble-averaged training curves with mean +/- std bands.

    4 subplots: (a) Loss, (b) Entropy, (c) KL, (d) ESS
    Individual runs shown as thin transparent lines, ensemble mean as thick line,
    std as shaded band.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    n_runs = len(all_run_data)

    # --- (a) Loss ---
    ax = axes[0, 0]
    losses = [rd["loss_history"] for rd in all_run_data]
    mean, std, n = _compute_band(losses)
    epochs = np.arange(1, n + 1)
    for i, l in enumerate(losses):
        ax.plot(np.arange(1, len(l) + 1), l, color=_COLORS[i % len(_COLORS)],
                linewidth=0.4, alpha=0.3)
    ax.plot(epochs, mean, color=_BLUE, linewidth=1.5, label=f"mean (n={n_runs})")
    ax.fill_between(epochs, mean - std, mean + std, color=_BLUE, alpha=0.2, label="+/- 1 std")
    ax.axhline(0, color=_GRAY, linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("(a) Policy Loss", fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- (b) Entropy ---
    ax = axes[0, 1]
    entropies = [rd["entropy_history"] for rd in all_run_data]
    mean, std, n = _compute_band(entropies)
    epochs = np.arange(1, n + 1)
    for i, e in enumerate(entropies):
        ax.plot(np.arange(1, len(e) + 1), e, color=_COLORS[i % len(_COLORS)],
                linewidth=0.4, alpha=0.3)
    ax.plot(epochs, mean, color=_BLUE, linewidth=1.5, label=f"mean (n={n_runs})")
    ax.fill_between(epochs, mean - std, mean + std, color=_BLUE, alpha=0.2)
    # Target entropy lines: multi-ordering (full manifold) and single-ordering (per-batch ceiling)
    n_reachable = all_run_data[0]["metadata"].get("n_reachable_states", -1)
    n_single = all_run_data[0]["metadata"].get("n_reachable_single_ordering", -1)
    if n_reachable > 0:
        target_ent = np.log(n_reachable)
        ax.axhline(target_ent, color=_GRAY, linestyle="--", linewidth=1.2,
                    label=f"multi-ordering: ln({n_reachable}) = {target_ent:.2f}")
    if n_single > 0:
        single_ent = np.log(n_single)
        ax.axhline(single_ent, color=_GREEN, linestyle=":", linewidth=1.5,
                    label=f"per-batch ceiling: ln({n_single}) = {single_ent:.2f}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Entropy")
    ax.set_title("(b) Entropy", fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- (c) KL divergence ---
    ax = axes[1, 0]
    kls = [rd["kl_history"] for rd in all_run_data]
    eval_epochs_list = [rd["eval_epochs"] for rd in all_run_data]
    kl_mean, kl_std, kl_n = _compute_band(kls)
    eval_ep_mean = all_run_data[0]["eval_epochs"][:kl_n]
    for i, (kl, ee) in enumerate(zip(kls, eval_epochs_list)):
        ax.plot(ee[:len(kl)], kl, color=_COLORS[i % len(_COLORS)],
                marker=".", markersize=2, linewidth=0.4, alpha=0.3)
    ax.plot(eval_ep_mean, kl_mean, color=_BLUE, marker="o", markersize=3,
            linewidth=1.5, label=f"mean (n={n_runs})")
    ax.fill_between(eval_ep_mean, kl_mean - kl_std, kl_mean + kl_std,
                     color=_BLUE, alpha=0.2)
    ax.axhline(0, color=_GRAY, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL divergence")
    ax.set_title("(c) KL(empirical || uniform)", fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- (d) ESS ---
    ax = axes[1, 1]
    ess_list = [rd["ess_history"] for rd in all_run_data]
    ess_mean, ess_std, ess_n = _compute_band(ess_list)
    eval_ep_ess = all_run_data[0]["eval_epochs"][:ess_n]
    for i, (ess, ee) in enumerate(zip(ess_list, eval_epochs_list)):
        ax.plot(ee[:len(ess)], ess, color=_COLORS[i % len(_COLORS)],
                marker=".", markersize=2, linewidth=0.4, alpha=0.3)
    ax.plot(eval_ep_ess, ess_mean, color=_BLUE, marker="o", markersize=3,
            linewidth=1.5, label=f"mean (n={n_runs})")
    ax.fill_between(eval_ep_ess, ess_mean - ess_std, ess_mean + ess_std,
                     color=_BLUE, alpha=0.2)
    batch_size = all_run_data[0]["metadata"].get("batch_size", 64)
    ax.axhline(batch_size, color=_GRAY, linestyle="--", linewidth=1.2,
               label=f"batch_size = {batch_size}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ESS")
    ax.set_title("(d) Effective Sample Size", fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    title = f"Ensemble Training Curves ({n_runs} runs)"
    if ensemble_label:
        title += f" — {ensemble_label}"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fpath = os.path.join(output_path, "ensemble_panel1_training_curves.png")
    fig.savefig(fpath)
    plt.close(fig)
    return fpath


def plot_ensemble_gradient_diagnostics(
    all_run_data: list[dict],
    output_path: str,
    ensemble_label: str = "",
) -> str:
    """Panel 5: Ensemble-averaged gradient diagnostics.

    (a) Gradient norm (post-clip) with mean +/- std
    (b) Advantage variance with mean +/- std
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    n_runs = len(all_run_data)

    # (a) Gradient norm
    ax = axes[0]
    gnorms = [rd["grad_norm_history"] for rd in all_run_data if len(rd["grad_norm_history"]) > 0]
    if gnorms:
        mean, std, n = _compute_band(gnorms)
        epochs = np.arange(1, n + 1)
        for i, g in enumerate(gnorms):
            ax.plot(np.arange(1, len(g) + 1), g, color=_COLORS[i % len(_COLORS)],
                    linewidth=0.3, alpha=0.2)
        ax.plot(epochs, mean, color=_BLUE, linewidth=1.5, label=f"mean (n={len(gnorms)})")
        ax.fill_between(epochs, mean - std, mean + std, color=_BLUE, alpha=0.2)
        # Smoothed mean
        if n >= 50:
            kernel = np.ones(50) / 50
            smoothed = np.convolve(mean, kernel, mode="valid")
            ax.plot(np.arange(25, 25 + len(smoothed)), smoothed,
                    color=_RED, linewidth=1.5, label="50-epoch smoothed mean")
        ax.legend(fontsize=7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient norm (post-clip)")
    ax.set_title("(a) Gradient Norm", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # (b) Advantage variance
    ax = axes[1]
    advvars = [rd["advantage_var_history"] for rd in all_run_data if len(rd["advantage_var_history"]) > 0]
    if advvars:
        mean, std, n = _compute_band(advvars)
        epochs = np.arange(1, n + 1)
        for i, av in enumerate(advvars):
            ax.plot(np.arange(1, len(av) + 1), av, color=_COLORS[i % len(_COLORS)],
                    linewidth=0.3, alpha=0.2)
        ax.plot(epochs, mean, color=_BLUE, linewidth=1.5, label=f"mean (n={len(advvars)})")
        ax.fill_between(epochs, mean - std, mean + std, color=_BLUE, alpha=0.2)
        if n >= 50:
            kernel = np.ones(50) / 50
            smoothed = np.convolve(mean, kernel, mode="valid")
            ax.plot(np.arange(25, 25 + len(smoothed)), smoothed,
                    color=_RED, linewidth=1.5, label="50-epoch smoothed mean")
        ax.legend(fontsize=7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Var(advantage)")
    ax.set_title("(b) Advantage Variance", fontweight="bold")
    ax.grid(True, alpha=0.3)

    title = f"Ensemble Gradient Diagnostics ({n_runs} runs)"
    if ensemble_label:
        title += f" — {ensemble_label}"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fpath = os.path.join(output_path, "ensemble_panel5_gradient_diagnostics.png")
    fig.savefig(fpath)
    plt.close(fig)
    return fpath


def plot_ensemble_sampling_quality(
    all_run_data: list[dict],
    output_path: str,
    ensemble_label: str = "",
) -> str:
    """Panel 2: Ensemble sampling quality.

    (a) State frequency histogram aggregated across all runs
    (b) Per-run coverage bar chart with mean +/- std line
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    n_runs = len(all_run_data)

    # (a) Aggregated state frequency
    ax = axes[0]
    exact_states = all_run_data[0].get("exact_states")
    if exact_states is not None and len(exact_states) > 0:
        n_exact = len(exact_states)
        state_to_idx = {}
        for i in range(n_exact):
            key = tuple(exact_states[i].astype(np.int8))
            state_to_idx[key] = i

        # Count across all runs
        total_counts = np.zeros(n_exact, dtype=np.float64)
        total_samples = 0
        per_run_coverage = []

        for rd in all_run_data:
            samples = rd["final_samples"]
            counts = np.zeros(n_exact, dtype=np.int64)
            for s in samples:
                key = tuple(s.astype(np.int8))
                idx = state_to_idx.get(key)
                if idx is not None:
                    counts[idx] += 1
            total_counts += counts
            total_samples += len(samples)
            n_found = np.sum(counts > 0)
            per_run_coverage.append(n_found / n_exact)

        # Normalize to mean counts per run
        mean_counts = total_counts / n_runs
        sorted_counts = np.sort(mean_counts)[::-1]
        uniform_expected = (total_samples / n_runs) / n_exact

        ax.barh(range(n_exact), sorted_counts, color=_BLUE, alpha=0.7, height=1.0)
        ax.axvline(uniform_expected, color=_GRAY, linestyle="--", linewidth=1.5,
                    label=f"Uniform = {uniform_expected:.1f}")
        ax.set_xlabel("Mean sample count per run")
        ax.set_ylabel("State (sorted by frequency)")
        ax.set_title("(a) Aggregated State Frequency", fontweight="bold")
        ax.legend(fontsize=7)
    else:
        # No exact states — show unique state counts per run
        unique_counts = []
        for rd in all_run_data:
            states = set(tuple(s.astype(np.int8)) for s in rd["final_samples"])
            unique_counts.append(len(states))
        ax.bar(range(n_runs), unique_counts, color=_BLUE, alpha=0.7)
        ax.set_xlabel("Run")
        ax.set_ylabel("Unique states")
        ax.set_title("(a) Unique States per Run", fontweight="bold")
        per_run_coverage = None

    # (b) Per-run coverage bar chart
    ax = axes[1]
    if exact_states is not None and len(exact_states) > 0 and per_run_coverage is not None:
        bars = ax.bar(range(n_runs), per_run_coverage,
                       color=[_COLORS[i % len(_COLORS)] for i in range(n_runs)],
                       alpha=0.7)
        mean_cov = np.mean(per_run_coverage)
        std_cov = np.std(per_run_coverage)
        ax.axhline(mean_cov, color=_BLUE, linewidth=2, linestyle="-",
                    label=f"mean = {mean_cov:.3f}")
        ax.axhspan(mean_cov - std_cov, mean_cov + std_cov, color=_BLUE, alpha=0.1,
                    label=f"+/- std = {std_cov:.3f}")
        # Pass/fail thresholds from training-experiments.md
        ax.axhline(0.90, color=_GREEN, linewidth=1, linestyle="--",
                    label="pass = 0.90", alpha=0.7)
        ax.axhline(0.80, color=_RED, linewidth=1, linestyle="--",
                    label="marginal = 0.80", alpha=0.7)
        ax.set_xlabel("Run")
        ax.set_ylabel("Coverage (fraction of reachable states)")
        ax.set_title(f"(b) Per-Run Coverage ({n_exact} states)", fontweight="bold")
        ax.set_xticks(range(n_runs))
        ax.set_xticklabels([f"seed {rd['metadata']['seed']}" for rd in all_run_data],
                            rotation=45, fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, loc="lower right")
    else:
        ax.text(0.5, 0.5, "No exact states available",
                transform=ax.transAxes, ha="center", va="center", fontsize=11)
        ax.set_title("(b) Coverage", fontweight="bold")

    title = f"Ensemble Sampling Quality ({n_runs} runs)"
    if ensemble_label:
        title += f" — {ensemble_label}"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fpath = os.path.join(output_path, "ensemble_panel2_sampling_quality.png")
    fig.savefig(fpath)
    plt.close(fig)
    return fpath


def plot_ensemble_summary_card(
    all_run_data: list[dict],
    ensemble_summary: Optional[dict],
    output_path: str,
    ensemble_label: str = "",
) -> str:
    """Panel 4: Ensemble summary card with mean +/- std for all key metrics."""
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.axis("off")

    metadata = all_run_data[0]["metadata"]
    n_runs = len(all_run_data)

    # Compute final metrics from run data directly
    final_kls = [rd["kl_history"][-1] for rd in all_run_data if len(rd["kl_history"]) > 0]
    final_hammings = [rd["hamming_history"][-1] for rd in all_run_data if len(rd["hamming_history"]) > 0]
    final_ess = [rd["ess_history"][-1] for rd in all_run_data if len(rd["ess_history"]) > 0]

    # Coverage
    exact_states = all_run_data[0].get("exact_states")
    coverages = []
    if exact_states is not None and len(exact_states) > 0:
        n_exact = len(exact_states)
        state_set_ref = set(tuple(s.astype(np.int8)) for s in exact_states)
        for rd in all_run_data:
            found = set(tuple(s.astype(np.int8)) for s in rd["final_samples"])
            n_found = len(found & state_set_ref)
            coverages.append(n_found / n_exact)
    else:
        n_exact = -1

    def _fmt(values):
        if not values:
            return "N/A"
        m, s = np.mean(values), np.std(values)
        return f"{m:.4f} +/- {s:.4f}"

    def _fmt_int(values):
        if not values:
            return "N/A"
        m, s = np.mean(values), np.std(values)
        return f"{m:.1f} +/- {s:.1f}"

    train_times = [rd["metadata"]["train_time_s"] for rd in all_run_data]

    lines = [
        ("Ensemble", ensemble_label or "N/A"),
        ("N runs", str(n_runs)),
        ("Lattice", f"{metadata['lattice_name']} ({metadata['boundary']} BC)"),
        ("Size", f"{metadata['size_label']}  ({metadata['n_vertices']}v, {metadata['n_edges']}e)"),
        ("beta_1", str(metadata["beta_1"])),
        ("Reachable states", str(n_exact) if n_exact > 0 else "not enumerated"),
        ("", ""),
        ("Model params", str(metadata["n_model_params"])),
        ("Layers / equ / inv", f"{metadata['n_layers']} / {metadata['equ_dim']} / {metadata['inv_dim']}"),
        ("Epochs", str(metadata["n_epochs"])),
        ("Batch size", str(metadata["batch_size"])),
        ("LR", str(metadata["lr"])),
        ("", ""),
        ("Final KL", _fmt(final_kls)),
        ("Final Hamming", _fmt(final_hammings)),
        ("Final ESS", _fmt_int(final_ess)),
        ("Coverage", _fmt(coverages) if coverages else "N/A"),
        ("Train time", f"{np.mean(train_times):.1f} +/- {np.std(train_times):.1f}s"),
        ("", ""),
        ("PASS/FAIL", ""),
    ]

    # Add pass/fail evaluation
    if final_kls:
        kl_mean = np.mean(final_kls)
        if kl_mean < 0.05:
            lines.append(("  KL", f"PASS ({kl_mean:.4f} < 0.05)"))
        elif kl_mean < 0.10:
            lines.append(("  KL", f"MARGINAL ({kl_mean:.4f})"))
        else:
            lines.append(("  KL", f"FAIL ({kl_mean:.4f} > 0.10)"))

    if coverages:
        cov_mean = np.mean(coverages)
        if cov_mean > 0.90:
            lines.append(("  Coverage", f"PASS ({cov_mean:.3f} > 0.90)"))
        elif cov_mean > 0.80:
            lines.append(("  Coverage", f"MARGINAL ({cov_mean:.3f})"))
        else:
            lines.append(("  Coverage", f"FAIL ({cov_mean:.3f} < 0.80)"))

    y = 0.97
    for label, value in lines:
        if label == "" and value == "":
            y -= 0.015
            continue
        color = "black"
        if "PASS" in value:
            color = _GREEN
        elif "FAIL" in value:
            color = _RED
        elif "MARGINAL" in value:
            color = "#f39c12"
        ax.text(0.05, y, label, fontsize=9.5, fontweight="bold",
                transform=ax.transAxes, verticalalignment="top",
                fontfamily="monospace")
        ax.text(0.40, y, value, fontsize=9.5, color=color,
                transform=ax.transAxes, verticalalignment="top",
                fontfamily="monospace")
        y -= 0.04

    fig.suptitle("Ensemble Summary", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fpath = os.path.join(output_path, "ensemble_panel4_summary.png")
    fig.savefig(fpath)
    plt.close(fig)
    return fpath


def main():
    parser = argparse.ArgumentParser(
        description="Generate ensemble-averaged diagnostic plots"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--ensemble-dir",
        help="Path to ensemble directory (contains ensemble_summary.json)"
    )
    group.add_argument(
        "--run-dirs", nargs="+",
        help="Explicit list of run directories"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for plots (default: {ensemble-dir}/plots)"
    )
    args = parser.parse_args()

    # Resolve run directories
    if args.ensemble_dir:
        summary_path = os.path.join(args.ensemble_dir, "ensemble_summary.json")
        if not os.path.exists(summary_path):
            logger.error(f"ensemble_summary.json not found in {args.ensemble_dir}")
            return 1
        with open(summary_path) as f:
            ensemble_summary = json.load(f)
        run_dirs = ensemble_summary["run_dirs"]
        ensemble_label = ensemble_summary.get("ensemble_id", "")
        output_dir = args.output_dir or os.path.join(args.ensemble_dir, "plots")
    else:
        run_dirs = args.run_dirs
        ensemble_summary = None
        ensemble_label = ""
        output_dir = args.output_dir or "results/ensemble_plots"

    os.makedirs(output_dir, exist_ok=True)

    # Load all runs
    logger.info(f"Loading {len(run_dirs)} runs...")
    all_run_data = []
    for rd in run_dirs:
        if not os.path.exists(rd):
            logger.warning(f"Run directory not found, skipping: {rd}")
            continue
        try:
            data = load_training_run(rd)
            all_run_data.append(data)
            logger.info(f"  Loaded: {rd} (seed={data['metadata']['seed']})")
        except Exception as e:
            logger.warning(f"  Failed to load {rd}: {e}")

    if len(all_run_data) < 2:
        logger.error(f"Need at least 2 runs for ensemble plots, got {len(all_run_data)}")
        return 1

    logger.info(f"\nGenerating ensemble plots for {len(all_run_data)} runs -> {output_dir}")
    paths = []

    # Panel 1: Training curves
    paths.append(plot_ensemble_training_curves(all_run_data, output_dir, ensemble_label))
    logger.info(f"  Panel 1: {paths[-1]}")

    # Panel 2: Sampling quality
    paths.append(plot_ensemble_sampling_quality(all_run_data, output_dir, ensemble_label))
    logger.info(f"  Panel 2: {paths[-1]}")

    # Panel 4: Summary card
    paths.append(plot_ensemble_summary_card(all_run_data, ensemble_summary, output_dir, ensemble_label))
    logger.info(f"  Panel 4: {paths[-1]}")

    # Panel 5: Gradient diagnostics
    paths.append(plot_ensemble_gradient_diagnostics(all_run_data, output_dir, ensemble_label))
    logger.info(f"  Panel 5: {paths[-1]}")

    logger.info(f"\nDone. {len(paths)} plots saved to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
