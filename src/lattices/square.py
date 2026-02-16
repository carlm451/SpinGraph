"""Square ice lattice generator (all z=4)."""
from .base import LatticeGenerator, UnitCell


class SquareGenerator(LatticeGenerator):
    """Square lattice: 1 vertex per cell, 2 edges per cell, 1 face per cell.

    Unit cell: a1=(1,0), a2=(0,1)
    Vertex: v0 at (0,0)
    Edges: horizontal v0->v0(+1,0), vertical v0->v0(0,+1)
    Face: square (v0(0,0), v0(1,0), v0(1,1), v0(0,1))
    All vertices have coordination z=4.

    Periodic N×N: n0=N², n1=2N², n2=N² (all filled)
    """

    name = "square"

    def _define_unit_cell(self) -> UnitCell:
        return UnitCell(
            a1=(1.0, 0.0),
            a2=(0.0, 1.0),
            vertices=[(0.0, 0.0)],
            edges=[
                (0, 0, 1, 0),  # horizontal: v0 -> v0 in cell (i+1, j)
                (0, 0, 0, 1),  # vertical: v0 -> v0 in cell (i, j+1)
            ],
            faces=[
                # Square face: v0(0,0) -> v0(1,0) -> v0(1,1) -> v0(0,1)
                [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)],
            ],
            expected_coordination={0: 4},
        )
