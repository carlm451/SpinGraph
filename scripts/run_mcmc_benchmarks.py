#!/usr/bin/env python3
"""Run MCMC benchmark suite across lattices and sizes.

Usage:
    python -m scripts.run_mcmc_benchmarks [--lattices square kagome] [--sizes XS S M]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.sampling.benchmark import run_full_benchmark_suite

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="MCMC benchmark suite")
    parser.add_argument(
        "--lattices", nargs="+",
        default=["square", "kagome", "shakti", "tetris", "santa_fe"],
    )
    parser.add_argument("--sizes", nargs="+", default=["XS", "S", "M"])
    parser.add_argument("--boundary", default="periodic")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--flips", nargs="+", type=int, default=[10, 20, 50, 100])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/mcmc_benchmarks")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running MCMC benchmarks: {args.lattices} x {args.sizes}")

    results = run_full_benchmark_suite(
        lattices=args.lattices,
        sizes=args.sizes,
        boundary=args.boundary,
        n_samples=args.n_samples,
        flip_values=args.flips,
        seed=args.seed,
    )

    # Save results as JSON
    output = {}
    for key, result_list in results.items():
        output[key] = []
        for r in result_list:
            output[key].append({
                "lattice_name": r.lattice_name,
                "size_label": r.size_label,
                "boundary": r.boundary,
                "n_samples": r.n_samples,
                "n_flips_between": r.n_flips_between,
                "wall_time_seconds": r.wall_time_seconds,
                "time_per_sample": r.time_per_sample,
                "mean_energy": r.mean_energy,
                "energy_std": r.energy_std,
                "ice_rule_violation_rate": r.ice_rule_violation_rate,
                "mean_hamming_distance": r.mean_hamming_distance,
                "hamming_std": r.hamming_std,
                "unique_fraction": r.unique_fraction,
                "n_vertices": r.n_vertices,
                "n_edges": r.n_edges,
                "autocorrelation_time": r.autocorrelation_time,
            })

    output_file = output_dir / "benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved to {output_file}")

    # Print summary table
    logger.info("\n=== MCMC Benchmark Summary ===")
    logger.info(f"{'Config':<25} {'Flips':>6} {'Time(s)':>8} {'t/sample':>10} "
                f"{'Hamming':>8} {'Unique':>8} {'Tau':>8}")
    logger.info("-" * 90)
    for key, result_list in sorted(results.items()):
        for r in result_list:
            tau_str = f"{r.autocorrelation_time:.2f}" if r.autocorrelation_time == r.autocorrelation_time else "N/A"
            logger.info(
                f"{key:<25} {r.n_flips_between:>6} {r.wall_time_seconds:>8.2f} "
                f"{r.time_per_sample:>10.6f} {r.mean_hamming_distance:>8.4f} "
                f"{r.unique_fraction:>8.4f} {tau_str:>8}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
