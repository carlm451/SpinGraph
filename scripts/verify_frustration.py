#!/usr/bin/env python3
"""Verify vertex-frustration properties of tetris and santa_fe lattices.

A loop is vertex-frustrated if it contains an odd number of z=2 (long island)
vertices. The tetris should be MAXIMALLY frustrated (all loops), while santa_fe
should have a MIX of frustrated and non-frustrated loops.
"""
import sys
sys.path.insert(0, "/Users/carlmerrigan/DeckerCode/SpinIceTDL")

import numpy as np
from src.lattices.tetris import TetrisGenerator
from src.lattices.santa_fe import SantaFeGenerator


def check_frustration(name, generator, nx=4, ny=4):
    """Build lattice and check frustration of each face."""
    result = generator.build(nx, ny, boundary="periodic")
    coord = result.coordination

    print(f"\n{'='*60}")
    print(f"  {name} lattice ({nx}x{ny} periodic)")
    print(f"{'='*60}")
    print(f"  Vertices: {result.n_vertices}")
    print(f"  Edges:    {result.n_edges}")
    print(f"  Faces:    {result.n_faces}")
    print(f"  Coordination dist: {result.coordination_distribution}")

    # Per-cell counts
    n_cells = nx * ny
    print(f"  Per cell: V={result.n_vertices/n_cells:.1f}, "
          f"E={result.n_edges/n_cells:.1f}, "
          f"F={result.n_faces/n_cells:.1f}")

    # Euler characteristic check
    chi = result.n_vertices - result.n_edges + result.n_faces
    print(f"  Euler char: V-E+F = {chi} (should be 0 for torus)")

    # Check each face for frustration
    n_frustrated = 0
    n_unfrustrated = 0
    face_details = []

    for i, face in enumerate(result.face_list):
        z2_count = sum(1 for v in face if coord[v] == 2)
        is_frustrated = (z2_count % 2 == 1)
        face_details.append({
            'face_idx': i,
            'size': len(face),
            'z2_count': z2_count,
            'frustrated': is_frustrated,
            'coord_seq': [int(coord[v]) for v in face],
        })
        if is_frustrated:
            n_frustrated += 1
        else:
            n_unfrustrated += 1

    print(f"\n  Frustration analysis:")
    print(f"    Frustrated faces:     {n_frustrated} / {len(result.face_list)}")
    print(f"    Non-frustrated faces: {n_unfrustrated} / {len(result.face_list)}")
    frac = n_frustrated / len(result.face_list) if result.face_list else 0
    print(f"    Fraction frustrated:  {frac:.3f}")

    # Show first few face details
    print(f"\n  First 6 face details:")
    for d in face_details[:6]:
        status = "FRUSTRATED" if d['frustrated'] else "not frustrated"
        print(f"    Face {d['face_idx']}: size={d['size']}, "
              f"z2_count={d['z2_count']}, coords={d['coord_seq']}, "
              f"{status}")

    # Summary of face sizes
    sizes = [d['size'] for d in face_details]
    unique_sizes = sorted(set(sizes))
    print(f"\n  Face size distribution:")
    for s in unique_sizes:
        count = sizes.count(s)
        frust_of_size = sum(1 for d in face_details
                           if d['size'] == s and d['frustrated'])
        print(f"    Size {s}: {count} faces ({frust_of_size} frustrated)")

    return face_details


if __name__ == "__main__":
    tetris_details = check_frustration("Tetris", TetrisGenerator())
    santa_fe_details = check_frustration("Santa Fe", SantaFeGenerator())

    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)

    tetris_all_frustrated = all(d['frustrated'] for d in tetris_details)
    print(f"  Tetris ALL frustrated?  {tetris_all_frustrated}  "
          f"(EXPECTED: True for maximal frustration)")

    sf_mixed = (any(d['frustrated'] for d in santa_fe_details) and
                any(not d['frustrated'] for d in santa_fe_details))
    print(f"  Santa Fe MIXED?         {sf_mixed}  "
          f"(EXPECTED: True for mixed frustration)")
