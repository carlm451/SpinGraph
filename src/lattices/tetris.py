"""Tetris ice lattice generator (mixed z=2,3,4).

The tetris lattice is a vertex-frustrated ASI topology introduced by
Morrison, Nelson & Nisoli (2013). It is maximally frustrated: every
minimal loop contains an odd number of long (fused) islands, and thus
cannot satisfy the ice rule at all vertices simultaneously.

The lattice decomposes into alternating horizontal bands:
  - BACKBONES: chains of z=4 crossroads connected through z=2 bridge
    vertices (long/fused horizontal islands). Completely ordered in
    ground state.
  - STAIRCASES: chains of z=3 T-junctions connected by short horizontal
    islands. Host the vertex frustration and remain disordered in
    ground state ("sliding phase").

The bridge positions on consecutive backbone rows are STAGGERED (shifted
by one lattice spacing), ensuring every minimal loop passes through
exactly one z=2 vertex — the signature of maximal vertex frustration.

References:
  Morrison, Nelson & Nisoli, New J. Phys. 15, 045009 (2013)
  Gilbert et al., Nature Physics 12, 162 (2016)

Key properties:
  - Mixed coordination: z=2 (bridges), z=3 (T-junctions), z=4 (crossroads)
  - Maximal vertex frustration (ALL loops frustrated)
  - Anisotropic: ordered in one direction, disordered in other ("sliding phase")
  - Lower rotational symmetry than shakti
"""
from .base import LatticeGenerator, UnitCell


class TetrisGenerator(LatticeGenerator):
    """Tetris lattice generator.

    Unit cell: a1=(2,0), a2=(0,4) containing 8 vertices.

    The cell spans two backbone rows (y=0,2) and two staircase rows (y=1,3).
    Bridge positions are STAGGERED between the two backbone rows:
      - Row 0 backbone: bridge at x=1 (between z4 at x=0 and z4 in next cell)
      - Row 2 backbone: bridge at x=0 (shifted, between z4 in prev cell and x=1)

    Vertex layout (one unit cell, y increases upward):

        v6(0,3) ---- v7(1,3)        staircase row 3 (z=3)
           |            |
        v4(0,2) ---- v5(1,2)        backbone row 2 (z=2, z=4) bridge at LEFT
           |            |
        v2(0,1) ---- v3(1,1)        staircase row 1 (z=3)
           |            |
        v0(0,0) ---- v1(1,0)        backbone row 0 (z=4, z=2) bridge at RIGHT

    Coordination per vertex:
      v0: z=4  (backbone row 0, crossroad at x=0)
      v1: z=2  (backbone row 0, bridge/fused at x=1)
      v2: z=3  (staircase row 1, T-junction at x=0)
      v3: z=3  (staircase row 1, T-junction at x=1)
      v4: z=2  (backbone row 2, bridge/fused at x=0)
      v5: z=4  (backbone row 2, crossroad at x=1)
      v6: z=3  (staircase row 3, T-junction at x=0)
      v7: z=3  (staircase row 3, T-junction at x=1)

    Per cell: 2x z=4, 4x z=3, 2x z=2
    Total degree: 2*4 + 4*3 + 2*2 = 24
    Edges per cell: 12
    Faces per cell: 4 (all hexagons, each with exactly 1 z=2 vertex)

    Periodic NxM: n0 = 8*N*M, n1 = 12*N*M, n2 = 4*N*M
    """

    name = "tetris"

    def _define_unit_cell(self) -> UnitCell:
        return UnitCell(
            a1=(2.0, 0.0),
            a2=(0.0, 4.0),
            vertices=[
                (0.0, 0.0),    # v0: z=4 crossroad (backbone row 0)
                (1.0, 0.0),    # v1: z=2 bridge (backbone row 0)
                (0.0, 1.0),    # v2: z=3 T-junction (staircase row 1)
                (1.0, 1.0),    # v3: z=3 T-junction (staircase row 1)
                (0.0, 2.0),    # v4: z=2 bridge (backbone row 2, SHIFTED)
                (1.0, 2.0),    # v5: z=4 crossroad (backbone row 2, SHIFTED)
                (0.0, 3.0),    # v6: z=3 T-junction (staircase row 3)
                (1.0, 3.0),    # v7: z=3 T-junction (staircase row 3)
            ],
            edges=[
                # Backbone row 0 (horizontal): z4-z2-z4 chain
                (0, 1, 0, 0),     # e0: v0--v1 (intra-cell, right)
                (1, 0, 1, 0),     # e1: v1--v0(i+1,j) (cross-cell, right)

                # Staircase row 1 (horizontal): z3--z3 chain
                (2, 3, 0, 0),     # e2: v2--v3 (intra-cell, right)
                (3, 2, 1, 0),     # e3: v3--v2(i+1,j) (cross-cell, right)

                # Backbone row 2 (horizontal): z2-z4-z2 chain (STAGGERED)
                (4, 5, 0, 0),     # e4: v4--v5 (intra-cell, right)
                (5, 4, 1, 0),     # e5: v5--v4(i+1,j) (cross-cell, right)

                # Staircase row 3 (horizontal): z3--z3 chain
                (6, 7, 0, 0),     # e6: v6--v7 (intra-cell, right)
                (7, 6, 1, 0),     # e7: v7--v6(i+1,j) (cross-cell, right)

                # Vertical edges (connecting backbones to staircases)
                # z=4 vertices have vertical edges; z=2 vertices do NOT
                (0, 2, 0, 0),     # e8:  v0--v2  (backbone0 z4 up to staircase1)
                (3, 5, 0, 0),     # e9:  v3--v5  (staircase1 up to backbone2 z4)
                (5, 7, 0, 0),     # e10: v5--v7  (backbone2 z4 up to staircase3)
                (6, 0, 0, 1),     # e11: v6--v0(i,j+1) (staircase3 up to backbone0 z4, cross-cell)
            ],
            faces=[
                # Face 1: backbone0-staircase1 hexagon (contains v1, z=2)
                # v0(0,0)->v1(0,0)->v0(+1,0)->v2(+1,0)->v3(0,0)->v2(0,0)
                [(0, 0, 0), (1, 0, 0), (0, 1, 0), (2, 1, 0), (3, 0, 0), (2, 0, 0)],

                # Face 2: staircase1-backbone2 hexagon (contains v4, z=2)
                # v2(0,0)->v3(0,0)->v5(0,0)->v4(0,0)->v5(-1,0)->v3(-1,0)
                [(2, 0, 0), (3, 0, 0), (5, 0, 0), (4, 0, 0), (5, -1, 0), (3, -1, 0)],

                # Face 3: backbone2-staircase3 hexagon (contains v4, z=2)
                # v5(0,0)->v7(0,0)->v6(0,0)->v7(-1,0)->v5(-1,0)->v4(0,0)
                [(5, 0, 0), (7, 0, 0), (6, 0, 0), (7, -1, 0), (5, -1, 0), (4, 0, 0)],

                # Face 4: staircase3-backbone0(j+1) hexagon (contains v1, z=2)
                # v7(0,0)->v6(0,0)->v0(0,+1)->v1(0,+1)->v0(+1,+1)->v6(+1,0)
                [(7, 0, 0), (6, 0, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (6, 1, 0)],
            ],
            expected_coordination={
                0: 4, 1: 2, 2: 3, 3: 3,
                4: 2, 5: 4, 6: 3, 7: 3,
            },
        )
