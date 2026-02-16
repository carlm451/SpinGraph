#!/usr/bin/env python
"""CLI entry point for generating publication-quality matplotlib figures.

Usage:
    python scripts/export_figures.py
    python scripts/export_figures.py --boundary both
    python scripts/export_figures.py --catalog results/catalog --output results/figures
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lattices.registry import list_lattices
from src.viz.matplotlib_figures import (
    figure_lattice_gallery,
    figure_eigenvalue_histograms,
    figure_spectral_overlay,
    figure_beta1_scaling,
    figure_harmonic_modes,
    figure_l1_decomposition,
    figure_ice_manifold_scaling,
    figure_ice_manifold_fraction,
    figure_ice_states,
)


def _generate_for_boundary(args, boundary):
    """Generate all figures for a single boundary condition."""
    bc_label = "periodic" if boundary == "periodic" else "open"
    print(f"\n=== Generating figures for {bc_label} boundary conditions ===\n")

    # Figure 1: Lattice zoo gallery
    print(f"  Fig 1: Lattice gallery ({bc_label})...", end=" ", flush=True)
    f = figure_lattice_gallery(args.output, size=4, boundary=boundary)
    print(f"saved: {f}")

    # Figure 2: Eigenvalue histograms (both strategies)
    for strat in ["all", "none"]:
        print(f"  Fig 2: Eigenvalue histograms (faces={strat}, {bc_label})...", end=" ", flush=True)
        try:
            f = figure_eigenvalue_histograms(args.catalog, args.output,
                                             size_label="S", strategy=strat,
                                             boundary=boundary)
            print(f"saved: {f}")
        except Exception as e:
            print(f"skipped: {e}")

    # Figure 3: Spectral overlay
    for strat in ["all", "none"]:
        print(f"  Fig 3: Spectral overlay (faces={strat}, {bc_label})...", end=" ", flush=True)
        try:
            f = figure_spectral_overlay(args.catalog, args.output,
                                        size_label="S", strategy=strat,
                                        boundary=boundary)
            print(f"saved: {f}")
        except Exception as e:
            print(f"skipped: {e}")

    # Figure 4: beta_1 scaling
    print(f"  Fig 4: beta_1 scaling ({bc_label})...", end=" ", flush=True)
    try:
        f = figure_beta1_scaling(args.catalog, args.output, sizes=args.sizes,
                                 boundary=boundary)
        print(f"saved: {f}")
    except Exception as e:
        print(f"skipped: {e}")

    # Figure 5: Harmonic modes (all 5 lattices)
    all_lattices = list_lattices()
    for name in all_lattices:
        for strat in ["all", "none"]:
            print(f"  Fig 5: Harmonic modes ({name}, {strat}, {bc_label})...", end=" ", flush=True)
            try:
                f = figure_harmonic_modes(args.catalog, args.output,
                                          lattice_name=name,
                                          size_label="XS", strategy=strat,
                                          boundary=boundary)
                print(f"saved: {f}")
            except Exception as e:
                print(f"skipped: {e}")

    # Figure 6: L1 decomposition
    for strat in ["all", "none"]:
        print(f"  Fig 6: L1 decomposition (faces={strat}, {bc_label})...", end=" ", flush=True)
        try:
            f = figure_l1_decomposition(args.catalog, args.output,
                                        size_label="S", strategy=strat,
                                        boundary=boundary)
            print(f"saved: {f}")
        except Exception as e:
            print(f"skipped: {e}")

    # Figure 9: Ice state samples (all lattices at XS size)
    for name in all_lattices:
        print(f"  Fig 9: Ice states ({name}, {bc_label})...", end=" ", flush=True)
        try:
            f = figure_ice_states(args.output, lattice_name=name, size=4,
                                  boundary=boundary, n_samples=6)
            print(f"saved: {f}")
        except Exception as e:
            print(f"skipped: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate publication figures")
    parser.add_argument("--catalog", default="results/catalog",
                        help="Catalog directory with precomputed results")
    parser.add_argument("--output", default="results/figures",
                        help="Output directory for figures")
    parser.add_argument("--sizes", nargs="+", default=["XS", "S", "M", "L"],
                        help="Sizes available in catalog for scaling plot")
    parser.add_argument("--boundary", default="both",
                        choices=["periodic", "open", "both"],
                        help="Boundary conditions to generate figures for")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Generating figures...")

    if args.boundary == "both":
        boundaries = ["periodic", "open"]
    else:
        boundaries = [args.boundary]

    for boundary in boundaries:
        _generate_for_boundary(args, boundary)

    # Figures 7-8: Ice manifold (overlay both BCs on one plot)
    print("\n=== Generating ice manifold figures (both BCs overlaid) ===\n")

    print("  Fig 7: Ice manifold scaling...", end=" ", flush=True)
    try:
        f = figure_ice_manifold_scaling(args.catalog, args.output, sizes=args.sizes)
        print(f"saved: {f}")
    except Exception as e:
        print(f"skipped: {e}")

    print("  Fig 8: Ice manifold fraction...", end=" ", flush=True)
    try:
        f = figure_ice_manifold_fraction(args.catalog, args.output, sizes=args.sizes)
        print(f"saved: {f}")
    except Exception as e:
        print(f"skipped: {e}")

    print(f"\nDone! All figures saved to {args.output}")


if __name__ == "__main__":
    main()
