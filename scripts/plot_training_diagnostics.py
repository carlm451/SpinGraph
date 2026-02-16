#!/usr/bin/env python3
"""Generate diagnostic plots from a saved training run.

Usage:
    python -m scripts.plot_training_diagnostics --run-dir results/neural_training/{run_id}
    python -m scripts.plot_training_diagnostics --run-dir ... --n-samples 12
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

from src.neural.checkpointing import load_training_run
from src.neural.training_plots import (
    compute_vertex_charges,
    generate_all_panels,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate diagnostic plots from a saved training run"
    )
    parser.add_argument(
        "--run-dir", required=True,
        help="Path to the training run directory (e.g. results/neural_training/{run_id})"
    )
    parser.add_argument(
        "--n-samples", type=int, default=6,
        help="Number of individual sample PNGs to generate (default: 6)"
    )
    args = parser.parse_args()

    logger.info(f"Loading run data from: {args.run_dir}")
    run_data = load_training_run(args.run_dir)

    metadata = run_data["metadata"]
    logger.info(
        f"Run: {metadata['run_id']} | "
        f"{metadata['lattice_name']} {metadata['size_label']} ({metadata['boundary']})"
    )

    # Ice-rule validation across all final samples
    logger.info("\n=== Ice Rule Validation ===")
    samples = run_data["final_samples"]
    edge_list = run_data["edge_list"]
    coordination = run_data["coordination"]
    n_total = len(samples)
    n_violating_samples = 0
    total_monopoles = 0

    for i in range(n_total):
        vci = compute_vertex_charges(samples[i], edge_list, coordination)
        if vci.n_violations > 0:
            n_violating_samples += 1
            total_monopoles += vci.n_violations

    logger.info(
        f"Samples with monopoles: {n_violating_samples}/{n_total} "
        f"({100 * n_violating_samples / n_total:.1f}%)"
    )
    if n_violating_samples > 0:
        logger.info(f"Total monopole vertices across all samples: {total_monopoles}")
        avg_monopoles = total_monopoles / n_violating_samples
        logger.info(f"Avg monopoles per violating sample: {avg_monopoles:.1f}")
    else:
        logger.info("All samples are perfect ice states (Q_v minimal at every vertex)")

    # Charge statistics across all samples
    all_charges = []
    for i in range(min(n_total, 500)):  # cap to avoid slow loop on huge sample sets
        vci = compute_vertex_charges(samples[i], edge_list, coordination)
        all_charges.append(np.abs(vci.charge))
    all_charges = np.array(all_charges)  # (n_samples, n_vertices)
    logger.info(
        f"Charge magnitude stats: "
        f"mean |Q_v| = {all_charges.mean():.4f}, "
        f"max |Q_v| = {all_charges.max():.0f}"
    )

    # Generate plots
    output_dir = os.path.join(args.run_dir, "plots")
    logger.info(f"\nGenerating plots to: {output_dir}")

    paths = generate_all_panels(run_data, output_dir, n_individual=args.n_samples)

    # Separate panel paths from sample paths
    panel_paths = [p for p in paths if "sample_" not in os.path.basename(p)]
    sample_paths = [p for p in paths if "sample_" in os.path.basename(p)]

    logger.info(f"\nGenerated {len(panel_paths)} diagnostic panels:")
    for p in panel_paths:
        logger.info(f"  {p}")

    logger.info(f"\nGenerated {len(sample_paths)} individual sample plots:")
    for p in sample_paths:
        logger.info(f"  {p}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
