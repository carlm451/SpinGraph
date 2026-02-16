"""Santa Fe ice lattice generator (mixed z=2,3,4).

The Santa Fe lattice is an ASI topology from Morrison, Nelson & Nisoli
(2013). It features both geometric AND vertex frustration, with
polymer-like strings of unhappy vertices in its low-energy states.

The Santa Fe is obtained by removing 1 island and bridging 2 others per
2a x 2a unit cell of the underlying square lattice. The bridges are placed
on one horizontal AND one vertical edge (mixed direction), which creates
both types of frustration simultaneously.

References:
  Morrison, Nelson & Nisoli, New J. Phys. 15, 045009 (2013)
  Zhang et al., Nature Communications 12, 6514 (2021)

Key properties:
  - Mixed coordination: z=2 (bridges), z=3 (T-junctions), z=4 (crossroads)
  - Both geometric and vertex frustration
  - Polymer-like string defects in ground state

Note on frustration ratio:
  Morrison describes 2/8 = 25% of "rectangular plaquettes" per composite
  square as frustrated. This cell produces 2/3 = 67% frustrated minimal
  faces because the 3 faces per cell are minimal graph faces (hexagon,
  pentagon, heptagon), not the "8 elementary plaquettes" Morrison counts
  (which require a larger unit cell to enumerate). The GRAPH topology
  (vertices, edges, coordination) correctly captures the Santa Fe structure;
  only the face enumeration differs from Morrison's composite-square
  accounting. This does not affect L0, L1_down, or the "no-faces" beta1.
"""
from .base import LatticeGenerator, UnitCell


class SantaFeGenerator(LatticeGenerator):
    """Santa Fe lattice generator.

    Unit cell: a1=(2,0), a2=(0,2) containing 6 vertices.

    Vertex layout:

        v2(0,1) --------- v3(1,1)
        |                  |
        v5(0,0.5)          |
        |                  |
        v0(0,0) --- v4(0.5,0) --- v1(1,0)

    Bridge v4 subdivides the BOTTOM HORIZONTAL edge (v0--v1).
    Bridge v5 subdivides the LEFT VERTICAL edge (v0--v2).
    The RIGHT VERTICAL (v1--v3) and TOP HORIZONTAL (v2--v3) are direct.

    This mixed-direction bridge placement distinguishes Santa Fe from
    tetris (which has bridges only on horizontal edges).

    Cross-cell connections (all distance 1.0 in real space):
      v1 -- v0 in cell (i+1, j):  i-direction connectivity
      v0 -- v2 in cell (i, j-1):  j-direction connectivity
      v1 -- v3 in cell (i, j-1):  j-direction connectivity

    Coordination per vertex:
      v0: z=4 (v4, v5, v1(i-1,j), v2(i,j+1))
      v1: z=4 (v4, v3, v0(i+1,j), v3(i,j+1))
      v2: z=3 (v5, v3, v0(i,j-1))
      v3: z=3 (v1, v2, v1(i,j-1))
      v4: z=2 (v0, v1)
      v5: z=2 (v0, v2)

    Per cell: 2x z=4, 2x z=3, 2x z=2
    Total degree per cell: 2*4 + 2*3 + 2*2 = 18
    Edges per cell: 9

    Periodic N*N: n0 = 6*N^2, n1 = 9*N^2
    """

    name = "santa_fe"

    def _define_unit_cell(self) -> UnitCell:
        return UnitCell(
            a1=(2.0, 0.0),
            a2=(0.0, 2.0),
            vertices=[
                (0.0, 0.0),    # v0: z=4 crossroad
                (1.0, 0.0),    # v1: z=4 crossroad
                (0.0, 1.0),    # v2: z=3 T-junction
                (1.0, 1.0),    # v3: z=3 T-junction
                (0.5, 0.0),    # v4: z=2 bridge (v0--v1 horizontal)
                (0.0, 0.5),    # v5: z=2 bridge (v0--v2 vertical)
            ],
            edges=[
                # Bridge edges (4 edges)
                (0, 4, 0, 0),    # v0 -- v4 (bottom bridge left half)
                (4, 1, 0, 0),    # v4 -- v1 (bottom bridge right half)
                (0, 5, 0, 0),    # v0 -- v5 (left bridge bottom half)
                (5, 2, 0, 0),    # v5 -- v2 (left bridge top half)

                # Direct edges (2 edges, no bridges)
                (1, 3, 0, 0),    # v1 -- v3 (right vertical, direct)
                (2, 3, 0, 0),    # v2 -- v3 (top horizontal, direct)

                # Cross-cell edges (3 edges, all distance 1.0)
                (1, 0, 1, 0),    # v1(ci,cj) -- v0(ci+1,cj)  [i-direction]
                (0, 2, 0, -1),   # v0(ci,cj) -- v2(ci,cj-1)  [j-direction]
                (1, 3, 0, -1),   # v1(ci,cj) -- v3(ci,cj-1)  [j-direction]
            ],
            faces=[
                # Face 1: Intra-cell hexagon
                # v0-v5-v2-v3-v1-v4, all in same cell
                [(0, 0, 0), (5, 0, 0), (2, 0, 0), (3, 0, 0),
                 (1, 0, 0), (4, 0, 0)],
                # Face 2: Pentagon (j-crossing)
                # v2(0,0)-v0(0,+1)-v4(0,+1)-v1(0,+1)-v3(0,0)
                [(2, 0, 0), (0, 0, 1), (4, 0, 1), (1, 0, 1),
                 (3, 0, 0)],
                # Face 3: Heptagon (i+j-crossing)
                # v1(0,0)-v3(0,0)-v1(0,+1)-v0(+1,+1)-v2(+1,0)-v5(+1,0)-v0(+1,0)
                [(1, 0, 0), (3, 0, 0), (1, 0, 1), (0, 1, 1),
                 (2, 1, 0), (5, 1, 0), (0, 1, 0)],
            ],
            expected_coordination={
                0: 4, 1: 4, 2: 3, 3: 3,
                4: 2, 5: 2,
            },
        )
