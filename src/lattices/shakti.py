"""Shakti ice lattice generator (mixed z=2,3,4).

The shakti lattice is a vertex-frustrated ASI topology introduced by
Morrison, Nelson & Nisoli (2013).  It features maximally frustrated
vertex interactions with mixed coordination numbers z=2, z=3, z=4.

The structure consists of a square grid of z=4 corner vertices.  Each
square plaquette contains two parallel bars (dominos) connected by a
bridge.  Bars alternate between horizontal pairs (H) and vertical pairs
(V) in a checkerboard pattern:

    +---H---+---V---+
    |       |       |
    V       H       V
    |       |       |
    +---H---+---V---+

Each bar is split into two half-edges by a z=3 midpoint vertex.  The
two z=3 midpoints in a plaquette are connected through a z=2 bridge
vertex.

Reference image: Lao et al., Nature Physics 14, 723 (2018), panels a/c/d.

Key properties:
  - Mixed coordination: z=2 (bridges), z=3 (bar midpoints), z=4 (corners)
  - Maximal vertex frustration
  - Extensive degeneracy with topological order
  - Per cell: 4x z=4, 8x z=3, 4x z=2 = 16 vertices
  - Per cell: 24 edges, 8 hexagonal faces
"""
from .base import LatticeGenerator, UnitCell


class ShaktiGenerator(LatticeGenerator):
    """Shakti lattice generator.

    Unit cell: a1=(2,0), a2=(0,2) containing 16 vertices and 24 edges.

    The 2x2 super-cell contains four plaquettes in a checkerboard:

        Plaquette A (H-pair, bottom-left):  bars horizontal at y~0.25, y~0.75
        Plaquette B (V-pair, bottom-right): bars vertical at x~1.25, x~1.75
        Plaquette C (V-pair, top-left):     bars vertical at x~0.25, x~0.75
        Plaquette D (H-pair, top-right):    bars horizontal at y~1.25, y~1.75

    Corner vertices (z=4): v0(0,0), v1(1,0), v2(0,1), v3(1,1)
    Bar midpoints (z=3):   v4,v5 (A), v7,v8 (B), v10,v11 (C), v13,v14 (D)
    Bridges (z=2):         v6 (A), v9 (B), v12 (C), v15 (D)

    Periodic N*N: n0 = 16*N^2, n1 = 24*N^2, n2 = 8*N^2
    Euler: 16 - 24 + 8 = 0 (torus)
    """

    name = "shakti"

    def _define_unit_cell(self) -> UnitCell:
        return UnitCell(
            a1=(2.0, 0.0),
            a2=(0.0, 2.0),
            vertices=[
                # Corner vertices (z=4)
                (0.0, 0.0),     # v0
                (1.0, 0.0),     # v1
                (0.0, 1.0),     # v2
                (1.0, 1.0),     # v3
                # Plaquette A (H-pair): bottom bar midpoint, top bar midpoint, bridge
                (0.5, 0.25),    # v4:  z=3
                (0.5, 0.75),    # v5:  z=3
                (0.5, 0.5),     # v6:  z=2
                # Plaquette B (V-pair): left bar midpoint, right bar midpoint, bridge
                (1.25, 0.5),    # v7:  z=3
                (1.75, 0.5),    # v8:  z=3
                (1.5, 0.5),     # v9:  z=2
                # Plaquette C (V-pair): left bar midpoint, right bar midpoint, bridge
                (0.25, 1.5),    # v10: z=3
                (0.75, 1.5),    # v11: z=3
                (0.5, 1.5),     # v12: z=2
                # Plaquette D (H-pair): bottom bar midpoint, top bar midpoint, bridge
                (1.5, 1.25),    # v13: z=3
                (1.5, 1.75),    # v14: z=3
                (1.5, 1.5),     # v15: z=2
            ],
            edges=[
                # --- Plaquette A (H-pair, bottom-left quadrant) ---
                # Bottom bar: v0 -> v4 -> v1
                (0, 4, 0, 0),
                (4, 1, 0, 0),
                # Top bar: v2 -> v5 -> v3
                (2, 5, 0, 0),
                (5, 3, 0, 0),
                # Bridge: v4 -> v6 -> v5
                (4, 6, 0, 0),
                (6, 5, 0, 0),

                # --- Plaquette B (V-pair, bottom-right quadrant) ---
                # Left bar: v1 -> v7 -> v3
                (1, 7, 0, 0),
                (7, 3, 0, 0),
                # Right bar: v8 -> v0(+1,0), v8 -> v2(+1,0)
                (8, 0, 1, 0),
                (8, 2, 1, 0),
                # Bridge: v7 -> v9 -> v8
                (7, 9, 0, 0),
                (9, 8, 0, 0),

                # --- Plaquette C (V-pair, top-left quadrant) ---
                # Left bar: v2 -> v10 -> v0(0,+1)
                (2, 10, 0, 0),
                (10, 0, 0, 1),
                # Right bar: v3 -> v11 -> v1(0,+1)
                (3, 11, 0, 0),
                (11, 1, 0, 1),
                # Bridge: v10 -> v12 -> v11
                (10, 12, 0, 0),
                (12, 11, 0, 0),

                # --- Plaquette D (H-pair, top-right quadrant) ---
                # Bottom bar: v3 -> v13 -> v2(+1,0)
                (3, 13, 0, 0),
                (13, 2, 1, 0),
                # Top bar: v14 -> v1(0,+1), v14 -> v0(+1,+1)
                (14, 1, 0, 1),
                (14, 0, 1, 1),
                # Bridge: v13 -> v15 -> v14
                (13, 15, 0, 0),
                (15, 14, 0, 0),
            ],
            faces=[
                # 8 hexagonal faces (each visits 2 z=4, 3 z=3, 1 z=2 vertex)

                # Faces around v0:
                # F1: v0-v4-v6-v5-v2-v8(-1,0)  (left of plaquette A)
                [(0, 0, 0), (4, 0, 0), (6, 0, 0), (5, 0, 0),
                 (2, 0, 0), (8, -1, 0)],
                # F2: v0-v8(-1,0)-v9(-1,0)-v7(-1,0)-v1(-1,0)-v14(-1,-1)
                [(0, 0, 0), (8, -1, 0), (9, -1, 0), (7, -1, 0),
                 (1, -1, 0), (14, -1, -1)],
                # F3: v0-v14(-1,-1)-v15(-1,-1)-v13(-1,-1)-v2(0,-1)-v10(0,-1)
                [(0, 0, 0), (14, -1, -1), (15, -1, -1), (13, -1, -1),
                 (2, 0, -1), (10, 0, -1)],
                # F4: v0-v10(0,-1)-v12(0,-1)-v11(0,-1)-v1-v4
                [(0, 0, 0), (10, 0, -1), (12, 0, -1), (11, 0, -1),
                 (1, 0, 0), (4, 0, 0)],

                # Faces around v3:
                # F5: v3-v13-v15-v14-v1(0,+1)-v11
                [(3, 0, 0), (13, 0, 0), (15, 0, 0), (14, 0, 0),
                 (1, 0, 1), (11, 0, 0)],
                # F6: v3-v11-v12-v10-v2-v5
                [(3, 0, 0), (11, 0, 0), (12, 0, 0), (10, 0, 0),
                 (2, 0, 0), (5, 0, 0)],
                # F7: v3-v5-v6-v4-v1-v7  (right of plaquette A)
                [(3, 0, 0), (5, 0, 0), (6, 0, 0), (4, 0, 0),
                 (1, 0, 0), (7, 0, 0)],
                # F8: v3-v7-v9-v8-v2(+1,0)-v13
                [(3, 0, 0), (7, 0, 0), (9, 0, 0), (8, 0, 0),
                 (2, 1, 0), (13, 0, 0)],
            ],
            expected_coordination={
                0: 4, 1: 4, 2: 4, 3: 4,       # corners
                4: 3, 5: 3, 7: 3, 8: 3,        # bar midpoints (A, B)
                10: 3, 11: 3, 13: 3, 14: 3,    # bar midpoints (C, D)
                6: 2, 9: 2, 12: 2, 15: 2,      # bridges
            },
        )
