"""Kagome (honeycomb) lattice generator.

In the ASI convention, "kagome spin ice" refers to spins on the edges of a
honeycomb lattice.  The edge midpoints form a kagome pattern, but the vertex
graph -- where ice rules are enforced -- is the honeycomb lattice with z=3
at every vertex.

This module implements the honeycomb vertex graph (z=3), consistent with the
spin ice literature (e.g. Nature Comms 7, 11446 (2016): "A honeycomb ice is
often called the Kagome spin ice, as the spins reside on the edges of a
honeycomb lattice").
"""
import math
from .base import LatticeGenerator, UnitCell


class KagomeGenerator(LatticeGenerator):
    """Honeycomb lattice (kagome ice): 2 vertices per cell, 3 edges per cell.

    Bravais lattice: a1 = (sqrt(3), 0), a2 = (sqrt(3)/2, 3/2)
    Vertices per cell:
      v0 = (0, 0)  (sublattice A)
      v1 = (0, 1)  (sublattice B)

    Each vertex has coordination z=3.

    Periodic NxN:
      n0 = 2N^2, n1 = 3N^2, n2 = N^2 (one hexagonal face per cell)
      chi = 0 (torus), beta_1(all) = 2, beta_1(none) = N^2 + 1
    """

    name = "kagome"

    def _define_unit_cell(self) -> UnitCell:
        s3 = math.sqrt(3)
        return UnitCell(
            a1=(s3, 0.0),
            a2=(s3 / 2, 1.5),
            vertices=[
                (0.0, 0.0),    # v0 (sublattice A)
                (0.0, 1.0),    # v1 (sublattice B)
            ],
            edges=[
                # "up" bond: v0 -> v1 in same cell
                (0, 1, 0, 0),
                # "upper-right" bond: v1 -> v0 in cell (i, j+1)
                (1, 0, 0, 1),
                # "upper-left" bond: v1 -> v0 in cell (i-1, j+1)
                (1, 0, -1, 1),
            ],
            faces=[
                # Hexagonal face (CCW):
                # v0(0,0), v1(0,0), v0(0,1), v1(1,0), v0(1,0), v1(1,-1)
                [(0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 1, 0), (0, 1, 0), (1, 1, -1)],
            ],
            expected_coordination={0: 3, 1: 3},
        )
