#!/usr/bin/env python3
"""Run an ensemble of N training runs for a single lattice configuration.

Each run uses a different random seed. After all runs complete, prints
summary statistics (mean +/- std) for key metrics.

Usage:
    python -m scripts.train_ensemble --lattice square --nx 4 --ny 4 --boundary open \
        --epochs 2000 --n-runs 5

    # Pass through any train_lattice.py arguments:
    python -m scripts.train_ensemble --lattice kagome --nx 2 --ny 2 --boundary periodic \
        --epochs 3000 --batch-size 128 --entropy-bonus 0.05 --n-runs 5
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_summary_from_log(log_lines: list[str]) -> dict:
    """Extract key metrics from train_lattice.py log output."""
    metrics = {}
    for line in log_lines:
        if "Final KL =" in line:
            try:
                metrics["kl"] = float(line.split("Final KL = ")[1].strip())
            except (ValueError, IndexError):
                pass
        elif "Final Hamming =" in line:
            try:
                metrics["hamming"] = float(line.split("Final Hamming = ")[1].strip())
            except (ValueError, IndexError):
                pass
        elif "Final ESS =" in line:
            try:
                metrics["ess"] = float(line.split("Final ESS = ")[1].strip())
            except (ValueError, IndexError):
                pass
        elif "Ice violations =" in line:
            try:
                metrics["violations"] = float(line.split("Ice violations = ")[1].strip())
            except (ValueError, IndexError):
                pass
        elif "Unique states =" in line:
            try:
                metrics["unique_states"] = int(line.split("Unique states = ")[1].strip())
            except (ValueError, IndexError):
                pass
        elif "State coverage:" in line:
            try:
                # "State coverage: 210/355 = 0.5915"
                parts = line.split("= ")
                metrics["coverage"] = float(parts[-1].strip())
            except (ValueError, IndexError):
                pass
        elif "Reachable states =" in line:
            try:
                val = line.split("Reachable states = ")[1].strip()
                if val != "not enumerated":
                    metrics["reachable_states"] = int(val)
            except (ValueError, IndexError):
                pass
        elif "Training time =" in line:
            try:
                metrics["train_time"] = float(line.split("Training time = ")[1].replace("s", "").strip())
            except (ValueError, IndexError):
                pass
        elif "Sample time" in line:
            try:
                metrics["sample_time"] = float(line.split("= ")[1].replace("s", "").strip())
            except (ValueError, IndexError):
                pass
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run ensemble of training runs with different seeds"
    )
    parser.add_argument("--n-runs", type=int, default=5, help="Number of ensemble runs")
    parser.add_argument("--base-seed", type=int, default=42, help="Base seed (incremented per run)")
    parser.add_argument("--ensemble-id", type=str, default=None,
                        help="Ensemble identifier (default: auto-generated)")

    # All other args are passed through to train_lattice.py
    args, passthrough = parser.parse_known_args()

    # Build the base command
    base_cmd = [sys.executable, "-m", "scripts.train_lattice"] + passthrough

    # Extract lattice info from passthrough for naming
    lattice = "unknown"
    nx = "?"
    boundary = "open"
    for i, arg in enumerate(passthrough):
        if arg == "--lattice" and i + 1 < len(passthrough):
            lattice = passthrough[i + 1]
        elif arg == "--nx" and i + 1 < len(passthrough):
            nx = passthrough[i + 1]
        elif arg == "--boundary" and i + 1 < len(passthrough):
            boundary = passthrough[i + 1]

    ensemble_id = args.ensemble_id or f"ensemble-{lattice}-{nx}-{boundary}"

    logger.info(f"=== Ensemble Training: {args.n_runs} runs ===")
    logger.info(f"Lattice: {lattice} {nx}x{nx} ({boundary})")
    logger.info(f"Ensemble ID: {ensemble_id}")
    logger.info(f"Base command: {' '.join(base_cmd)}")

    all_metrics = []
    run_dirs = []
    total_t0 = time.perf_counter()

    for run_idx in range(args.n_runs):
        seed = args.base_seed + run_idx
        run_id = f"{ensemble_id}-run{run_idx}"

        # Build per-run command with unique seed and run-id
        cmd = base_cmd.copy()

        # Remove any existing --seed or --run-id from passthrough
        filtered_cmd = []
        skip_next = False
        for i, arg in enumerate(cmd):
            if skip_next:
                skip_next = False
                continue
            if arg in ("--seed", "--run-id"):
                skip_next = True
                continue
            filtered_cmd.append(arg)
        cmd = filtered_cmd + ["--seed", str(seed), "--run-id", run_id]

        logger.info(f"\n--- Run {run_idx + 1}/{args.n_runs} (seed={seed}) ---")
        t0 = time.perf_counter()

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )

        elapsed = time.perf_counter() - t0

        if result.returncode != 0:
            logger.error(f"Run {run_idx + 1} FAILED (exit code {result.returncode})")
            logger.error(f"stderr: {result.stderr[-500:]}")
            continue

        # Parse metrics from stdout+stderr (logging goes to stderr)
        log_lines = (result.stdout + result.stderr).split("\n")
        metrics = parse_summary_from_log(log_lines)
        metrics["seed"] = seed
        metrics["run_id"] = run_id
        metrics["elapsed"] = elapsed

        all_metrics.append(metrics)
        run_dirs.append(f"results/neural_training/{run_id}")

        logger.info(
            f"Run {run_idx + 1} complete in {elapsed:.1f}s: "
            f"KL={metrics.get('kl', 'N/A')}, "
            f"coverage={metrics.get('coverage', 'N/A')}, "
            f"ESS={metrics.get('ess', 'N/A')}"
        )

    total_time = time.perf_counter() - total_t0

    if not all_metrics:
        logger.error("All runs failed!")
        return 1

    # Compute ensemble statistics
    logger.info(f"\n{'=' * 60}")
    logger.info(f"=== Ensemble Summary: {len(all_metrics)}/{args.n_runs} runs completed ===")
    logger.info(f"{'=' * 60}")

    summary = {}
    for key in ["kl", "hamming", "ess", "violations", "unique_states", "coverage", "train_time"]:
        values = [m[key] for m in all_metrics if key in m]
        if values:
            arr = np.array(values)
            summary[key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "values": [float(v) for v in arr],
            }
            logger.info(
                f"  {key:20s}: {np.mean(arr):.4f} +/- {np.std(arr):.4f} "
                f"(range: {np.min(arr):.4f} - {np.max(arr):.4f})"
            )

    logger.info(f"\n  Total ensemble time: {total_time:.1f}s")
    logger.info(f"  Run directories: {run_dirs}")

    # Save ensemble summary
    ensemble_dir = os.path.join("results", "neural_training", ensemble_id)
    os.makedirs(ensemble_dir, exist_ok=True)

    ensemble_summary = {
        "ensemble_id": ensemble_id,
        "n_runs": args.n_runs,
        "n_completed": len(all_metrics),
        "lattice": lattice,
        "nx": nx,
        "boundary": boundary,
        "base_seed": args.base_seed,
        "total_time_s": total_time,
        "summary": summary,
        "per_run": all_metrics,
        "run_dirs": run_dirs,
    }

    summary_path = os.path.join(ensemble_dir, "ensemble_summary.json")
    with open(summary_path, "w") as f:
        json.dump(ensemble_summary, f, indent=2)
    logger.info(f"\n  Ensemble summary saved to: {summary_path}")

    # Print a formatted results table for easy copy-paste into training-experiments.md
    logger.info("\n--- Markdown Table Row ---")
    kl_str = f"{summary.get('kl', {}).get('mean', float('nan')):.3f} +/- {summary.get('kl', {}).get('std', float('nan')):.3f}"
    cov_str = f"{summary.get('coverage', {}).get('mean', float('nan')):.2f} +/- {summary.get('coverage', {}).get('std', float('nan')):.2f}"
    ess_str = f"{summary.get('ess', {}).get('mean', float('nan')):.0f} +/- {summary.get('ess', {}).get('std', float('nan')):.0f}"
    time_str = f"{summary.get('train_time', {}).get('mean', float('nan')):.0f}s"
    logger.info(f"| {lattice} {nx}x{nx} {boundary} | {kl_str} | {cov_str} | {ess_str} | {time_str} |")

    return 0


if __name__ == "__main__":
    sys.exit(main())
