#!/usr/bin/env python
"""CLI entry point for computing the spectral catalog.

Usage:
    python scripts/run_catalog.py --lattices square kagome --sizes XS S --strategies all none
    python scripts/run_catalog.py --all
    python scripts/run_catalog.py --all --sizes XS S M
    python scripts/run_catalog.py --all --sizes XS S M L --boundary open
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lattices.registry import list_lattices
from src.spectral.catalog import run_full_catalog, SIZE_CONFIGS
from src.io.serialize import save_result, update_catalog_index


def main():
    parser = argparse.ArgumentParser(description="Compute spectral catalog for ASI lattices")
    parser.add_argument("--lattices", nargs="+", default=None,
                        help="Lattice types to compute (default: all)")
    parser.add_argument("--sizes", nargs="+", default=None,
                        help=f"Size labels (choices: {list(SIZE_CONFIGS.keys())})")
    parser.add_argument("--strategies", nargs="+", default=None,
                        help="Face strategies: 'all', 'none' (default: both)")
    parser.add_argument("--boundary", default="periodic",
                        choices=["periodic", "open"],
                        help="Boundary condition (default: periodic)")
    parser.add_argument("--all", action="store_true",
                        help="Run all lattices (shorthand for --lattices with all types)")
    parser.add_argument("--output", default="results/catalog",
                        help="Output directory (default: results/catalog)")
    args = parser.parse_args()

    lattices = args.lattices
    if args.all:
        lattices = sorted(list_lattices())
    if lattices is None:
        lattices = sorted(list_lattices())

    sizes = args.sizes
    if sizes is None:
        sizes = ["XS", "S", "M"]

    strategies = args.strategies
    if strategies is None:
        strategies = ["all", "none"]

    print(f"Lattices: {lattices}")
    print(f"Sizes: {sizes}")
    print(f"Strategies: {strategies}")
    print(f"Boundary: {args.boundary}")
    print(f"Output: {args.output}")
    print()

    # Save results incrementally (so completed results survive if process is killed)
    os.makedirs(args.output, exist_ok=True)

    results = run_full_catalog(
        lattices=lattices,
        sizes=sizes,
        strategies=strategies,
        boundary=args.boundary,
        save_dir=args.output,
    )

    update_catalog_index(args.output)

    print(f"\nSaved {len(results)} results to {args.output}/")
    print(f"Catalog index: {args.output}/catalog_index.json")


if __name__ == "__main__":
    main()
