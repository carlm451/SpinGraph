"""Tests for periodic edge classification and stub geometry.

4x4 periodic square lattice on a torus:
  n0 = 16, n1 = 32, n2 = 16
  Periodic edges: 4 horizontal wraps + 4 vertical wraps = 8
  Interior edges: 32 - 8 = 24
"""
import numpy as np
import pytest

from src.lattices.square import SquareGenerator
from src.viz.periodic_edges import classify_edges, classify_faces, PeriodicEdge


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def square_4x4_periodic():
    gen = SquareGenerator()
    return gen.build(4, 4, boundary="periodic")


@pytest.fixture
def square_4x4_open():
    gen = SquareGenerator()
    return gen.build(4, 4, boundary="open")


# ---------------------------------------------------------------------------
# Edge classification
# ---------------------------------------------------------------------------

class TestClassifyEdgesSquarePeriodic:
    """Test edge classification on 4x4 periodic square lattice."""

    def test_total_edge_count(self, square_4x4_periodic):
        lat = square_4x4_periodic
        interior, periodic, is_periodic = classify_edges(
            lat.positions, lat.edge_list,
            lat.unit_cell.a1, lat.unit_cell.a2,
            lat.nx_size, lat.ny_size, lat.boundary,
        )
        assert len(interior) + len(periodic) == len(lat.edge_list)

    def test_periodic_edge_count(self, square_4x4_periodic):
        """4x4 square: 4 horizontal wraps + 4 vertical wraps = 8 periodic edges."""
        lat = square_4x4_periodic
        _, periodic, is_periodic = classify_edges(
            lat.positions, lat.edge_list,
            lat.unit_cell.a1, lat.unit_cell.a2,
            lat.nx_size, lat.ny_size, lat.boundary,
        )
        assert len(periodic) == 8
        assert np.sum(is_periodic) == 8

    def test_interior_edge_count(self, square_4x4_periodic):
        """4x4 square: 32 total - 8 periodic = 24 interior."""
        lat = square_4x4_periodic
        interior, _, _ = classify_edges(
            lat.positions, lat.edge_list,
            lat.unit_cell.a1, lat.unit_cell.a2,
            lat.nx_size, lat.ny_size, lat.boundary,
        )
        assert len(interior) == 24

    def test_mask_matches_lists(self, square_4x4_periodic):
        lat = square_4x4_periodic
        interior, periodic, is_periodic = classify_edges(
            lat.positions, lat.edge_list,
            lat.unit_cell.a1, lat.unit_cell.a2,
            lat.nx_size, lat.ny_size, lat.boundary,
        )
        # All interior edge indices should not be periodic
        for _, _, idx in interior:
            assert not is_periodic[idx]
        # All periodic edge indices should be periodic
        for pe in periodic:
            assert is_periodic[pe.edge_index]

    def test_edge_indices_cover_all(self, square_4x4_periodic):
        """Every edge index 0..n_edges-1 appears exactly once."""
        lat = square_4x4_periodic
        interior, periodic, _ = classify_edges(
            lat.positions, lat.edge_list,
            lat.unit_cell.a1, lat.unit_cell.a2,
            lat.nx_size, lat.ny_size, lat.boundary,
        )
        all_indices = sorted([idx for _, _, idx in interior] +
                             [pe.edge_index for pe in periodic])
        assert all_indices == list(range(len(lat.edge_list)))


class TestClassifyEdgesSquareOpen:
    """Test that open boundary produces no periodic edges."""

    def test_no_periodic_edges(self, square_4x4_open):
        lat = square_4x4_open
        interior, periodic, is_periodic = classify_edges(
            lat.positions, lat.edge_list,
            lat.unit_cell.a1, lat.unit_cell.a2,
            lat.nx_size, lat.ny_size, lat.boundary,
        )
        assert len(periodic) == 0
        assert np.sum(is_periodic) == 0
        assert len(interior) == len(lat.edge_list)


# ---------------------------------------------------------------------------
# Stub geometry
# ---------------------------------------------------------------------------

class TestStubGeometry:
    """Verify stub positions for periodic wrapping edges on the square lattice."""

    def test_stub_u_starts_at_vertex(self, square_4x4_periodic):
        """Each stub_u starts at the position of vertex u."""
        lat = square_4x4_periodic
        _, periodic, _ = classify_edges(
            lat.positions, lat.edge_list,
            lat.unit_cell.a1, lat.unit_cell.a2,
            lat.nx_size, lat.ny_size, lat.boundary,
        )
        for pe in periodic:
            start, _ = pe.stub_u
            np.testing.assert_allclose(start, lat.positions[pe.u], atol=1e-12)

    def test_stub_v_starts_at_vertex(self, square_4x4_periodic):
        """Each stub_v starts at the position of vertex v."""
        lat = square_4x4_periodic
        _, periodic, _ = classify_edges(
            lat.positions, lat.edge_list,
            lat.unit_cell.a1, lat.unit_cell.a2,
            lat.nx_size, lat.ny_size, lat.boundary,
        )
        for pe in periodic:
            start, _ = pe.stub_v
            np.testing.assert_allclose(start, lat.positions[pe.v], atol=1e-12)

    def test_stub_length_half_unit(self, square_4x4_periodic):
        """For square lattice, stubs should be half a lattice spacing long (0.5)."""
        lat = square_4x4_periodic
        _, periodic, _ = classify_edges(
            lat.positions, lat.edge_list,
            lat.unit_cell.a1, lat.unit_cell.a2,
            lat.nx_size, lat.ny_size, lat.boundary,
        )
        for pe in periodic:
            su_start, su_end = pe.stub_u
            sv_start, sv_end = pe.stub_v
            len_u = np.linalg.norm(su_end - su_start)
            len_v = np.linalg.norm(sv_end - sv_start)
            np.testing.assert_allclose(len_u, 0.5, atol=1e-12)
            np.testing.assert_allclose(len_v, 0.5, atol=1e-12)

    def test_stubs_point_opposite_directions(self, square_4x4_periodic):
        """The two stubs of a periodic edge should point in opposite directions."""
        lat = square_4x4_periodic
        _, periodic, _ = classify_edges(
            lat.positions, lat.edge_list,
            lat.unit_cell.a1, lat.unit_cell.a2,
            lat.nx_size, lat.ny_size, lat.boundary,
        )
        for pe in periodic:
            su_start, su_end = pe.stub_u
            sv_start, sv_end = pe.stub_v
            dir_u = su_end - su_start
            dir_v = sv_end - sv_start
            # Should be anti-parallel
            dot = np.dot(dir_u, dir_v)
            np.testing.assert_allclose(dot, -np.dot(dir_u, dir_u), atol=1e-12)


# ---------------------------------------------------------------------------
# Face classification
# ---------------------------------------------------------------------------

class TestClassifyFaces:
    """Test face classification based on periodic edge membership."""

    def test_face_split_counts(self, square_4x4_periodic):
        """Interior + periodic faces == total faces."""
        lat = square_4x4_periodic
        _, _, is_periodic = classify_edges(
            lat.positions, lat.edge_list,
            lat.unit_cell.a1, lat.unit_cell.a2,
            lat.nx_size, lat.ny_size, lat.boundary,
        )
        interior_faces, periodic_faces = classify_faces(
            lat.face_list, lat.edge_list, is_periodic,
        )
        assert len(interior_faces) + len(periodic_faces) == len(lat.face_list)

    def test_periodic_face_count(self, square_4x4_periodic):
        """4x4 square: faces touching boundary wraps.

        The square lattice has faces defined as
        (i,j)->(i+1,j)->(i+1,j+1)->(i,j+1).  A face is periodic if any
        of its edges wraps.  The faces at i=3 (right wrap) and j=3 (top wrap)
        are periodic.  That's 4 + 4 - 1 = 7 faces touching at least one
        boundary wrap (the corner at (3,3) has both horizontal and vertical
        wrap edges).
        """
        lat = square_4x4_periodic
        _, _, is_periodic = classify_edges(
            lat.positions, lat.edge_list,
            lat.unit_cell.a1, lat.unit_cell.a2,
            lat.nx_size, lat.ny_size, lat.boundary,
        )
        _, periodic_faces = classify_faces(
            lat.face_list, lat.edge_list, is_periodic,
        )
        # 4x4 square: rightmost column (4 faces) + topmost row (4 faces)
        # minus corner overlap (1 face) = 7
        assert len(periodic_faces) == 7

    def test_interior_face_count(self, square_4x4_periodic):
        """4x4 square: 16 total - 7 periodic = 9 interior faces."""
        lat = square_4x4_periodic
        _, _, is_periodic = classify_edges(
            lat.positions, lat.edge_list,
            lat.unit_cell.a1, lat.unit_cell.a2,
            lat.nx_size, lat.ny_size, lat.boundary,
        )
        interior_faces, _ = classify_faces(
            lat.face_list, lat.edge_list, is_periodic,
        )
        assert len(interior_faces) == 9

    def test_open_no_periodic_faces(self, square_4x4_open):
        """Open BCs: all faces are interior."""
        lat = square_4x4_open
        _, _, is_periodic = classify_edges(
            lat.positions, lat.edge_list,
            lat.unit_cell.a1, lat.unit_cell.a2,
            lat.nx_size, lat.ny_size, lat.boundary,
        )
        interior_faces, periodic_faces = classify_faces(
            lat.face_list, lat.edge_list, is_periodic,
        )
        assert len(periodic_faces) == 0
        assert len(interior_faces) == len(lat.face_list)
