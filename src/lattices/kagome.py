"""Kagome lattice generator.

The kagome lattice is the trihexagonal tiling: corner-sharing triangles
with hexagonal voids. Every vertex has coordination z=4.

In the ASI context, CLAUDE.md says "all z=3" referring to the honeycomb
vertex coordination (the kagome is the medial lattice of the honeycomb).
As an actual graph, kagome vertices have degree 4.
"""
import math
from .base import LatticeGenerator, UnitCell


class KagomeGenerator(LatticeGenerator):
    """Kagome lattice: 3 vertices per cell, 6 edges per cell.

    Bravais lattice: a1 = (2, 0), a2 = (1, sqrt(3))
    Vertices per cell:
      v0 = (0, 0), v1 = (1, 0), v2 = (0.5, sqrt(3)/2)

    Each vertex has coordination z=4.

    Periodic N*N:
      n0 = 3N^2, n1 = 6N^2
      n2 = 3N^2 (all filled: N^2 up-tri + N^2 down-tri + N^2 hex)
      chi = 0 (torus), beta_1 = 2
    """

    name = "kagome"

    def _define_unit_cell(self) -> UnitCell:
        s3 = math.sqrt(3)
        return UnitCell(
            a1=(2.0, 0.0),
            a2=(1.0, s3),
            vertices=[
                (0.0, 0.0),       # v0
                (1.0, 0.0),       # v1
                (0.5, s3 / 2),    # v2
            ],
            edges=[
                # Intra-cell edges (upward triangle)
                (0, 1, 0, 0),   # v0 -> v1 same cell
                (1, 2, 0, 0),   # v1 -> v2 same cell
                (0, 2, 0, 0),   # v0 -> v2 same cell
                # Inter-cell edges
                (1, 0, 1, 0),   # v1 -> v0 in cell (i+1, j)
                (2, 0, 0, 1),   # v2 -> v0 in cell (i, j+1)
                (2, 1, -1, 1),  # v2 -> v1 in cell (i-1, j+1)
            ],
            faces=[
                # Upward triangle (intra-cell): v0, v1, v2 in cell (i,j)
                [(0, 0, 0), (1, 0, 0), (2, 0, 0)],
                # Downward triangle: v1(i,j), v0(i+1,j), v2(i+1,j-1)
                [(1, 0, 0), (0, 1, 0), (2, 1, -1)],
                # Hexagon (CCW): v1(i,j), v0(i+1,j), v2(i+1,j),
                #   v1(i,j+1), v0(i,j+1), v2(i,j)
                [(1, 0, 0), (0, 1, 0), (2, 1, 0), (1, 0, 1), (0, 0, 1), (2, 0, 0)],
            ],
            expected_coordination={0: 4, 1: 4, 2: 4},
        )
