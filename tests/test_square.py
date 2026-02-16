"""Tests for square lattice pipeline against known analytical values.

4×4 periodic square lattice (torus):
  n0 = 16, n1 = 32, n2 = 16 (all filled), all z=4
  β₀ = 1, β₁ = 2, β₂ = 1, Euler χ = 0

With no faces (n2=0):
  β₁ = n1 - (n0 - β₀) = 32 - 15 = 17
"""
import numpy as np
import pytest

from src.lattices.square import SquareGenerator
from src.topology.incidence import build_B1, build_B2, verify_chain_complex
from src.topology.laplacians import build_all_laplacians
from src.spectral.eigensolve import eigendecompose, count_zero_eigenvalues, extract_harmonic_basis
from src.spectral.betti import compute_betti_numbers, validate_harmonic_vectors


@pytest.fixture
def square_4x4():
    gen = SquareGenerator()
    return gen.build(4, 4, boundary="periodic")


@pytest.fixture
def square_4x4_open():
    gen = SquareGenerator()
    return gen.build(4, 4, boundary="open")


class TestSquareCounts:
    """Verify basic lattice counts."""

    def test_vertex_count(self, square_4x4):
        assert square_4x4.n_vertices == 16

    def test_edge_count(self, square_4x4):
        assert square_4x4.n_edges == 32

    def test_face_count(self, square_4x4):
        assert square_4x4.n_faces == 16

    def test_all_coordination_4(self, square_4x4):
        assert np.all(square_4x4.coordination == 4)

    def test_euler_counts(self, square_4x4):
        # χ = n0 - n1 + n2 = 16 - 32 + 16 = 0 (torus)
        chi = square_4x4.n_vertices - square_4x4.n_edges + square_4x4.n_faces
        assert chi == 0


class TestSquareIncidence:
    """Test incidence matrix properties."""

    def test_B1_shape(self, square_4x4):
        B1 = build_B1(square_4x4.n_vertices, square_4x4.edge_list)
        assert B1.shape == (16, 32)

    def test_B1_column_sums(self, square_4x4):
        """Each B1 column sums to 0 (one +1 and one -1)."""
        B1 = build_B1(square_4x4.n_vertices, square_4x4.edge_list)
        col_sums = np.array(B1.sum(axis=0)).flatten()
        np.testing.assert_allclose(col_sums, 0, atol=1e-14)

    def test_B1_entries_per_column(self, square_4x4):
        """Each B1 column has exactly 2 nonzeros."""
        B1 = build_B1(square_4x4.n_vertices, square_4x4.edge_list)
        nnz_per_col = np.diff(B1.tocsc().indptr)
        assert np.all(nnz_per_col == 2)

    def test_B2_shape(self, square_4x4):
        B1 = build_B1(square_4x4.n_vertices, square_4x4.edge_list)
        B2 = build_B2(square_4x4.n_edges, square_4x4.face_list, square_4x4.edge_list)
        assert B2.shape == (32, 16)

    def test_chain_complex(self, square_4x4):
        """B1 @ B2 = 0 (boundary of boundary is zero)."""
        B1 = build_B1(square_4x4.n_vertices, square_4x4.edge_list)
        B2 = build_B2(square_4x4.n_edges, square_4x4.face_list, square_4x4.edge_list)
        assert verify_chain_complex(B1, B2)

    def test_B2_column_sums(self, square_4x4):
        """Each B2 column sums to 0 (faces are closed cycles)."""
        B2 = build_B2(square_4x4.n_edges, square_4x4.face_list, square_4x4.edge_list)
        col_sums = np.array(B2.sum(axis=0)).flatten()
        np.testing.assert_allclose(col_sums, 0, atol=1e-14)


class TestSquareLaplacians:
    """Test Laplacian properties."""

    def test_L0_graph_laplacian_identity(self, square_4x4):
        """L0 = D - A for the graph Laplacian."""
        import networkx as nx
        B1 = build_B1(square_4x4.n_vertices, square_4x4.edge_list)
        laps = build_all_laplacians(
            B1,
            build_B2(square_4x4.n_edges, square_4x4.face_list, square_4x4.edge_list),
        )
        L0 = laps["L0"].toarray()

        # Build D - A from networkx
        A = nx.adjacency_matrix(square_4x4.graph, nodelist=range(16)).toarray().astype(float)
        D = np.diag(square_4x4.coordination.astype(float))
        L0_expected = D - A

        np.testing.assert_allclose(L0, L0_expected, atol=1e-12)

    def test_laplacians_symmetric(self, square_4x4):
        B1 = build_B1(square_4x4.n_vertices, square_4x4.edge_list)
        B2 = build_B2(square_4x4.n_edges, square_4x4.face_list, square_4x4.edge_list)
        laps = build_all_laplacians(B1, B2)
        for name, L in laps.items():
            Ld = L.toarray()
            np.testing.assert_allclose(Ld, Ld.T, atol=1e-14, err_msg=f"{name} not symmetric")

    def test_laplacians_psd(self, square_4x4):
        """All Laplacians are positive semi-definite."""
        B1 = build_B1(square_4x4.n_vertices, square_4x4.edge_list)
        B2 = build_B2(square_4x4.n_edges, square_4x4.face_list, square_4x4.edge_list)
        laps = build_all_laplacians(B1, B2)
        for name, L in laps.items():
            evals = np.linalg.eigvalsh(L.toarray())
            assert np.all(evals > -1e-10), f"{name} has negative eigenvalue: {evals.min()}"


class TestSquareBettiAllFaces:
    """Betti numbers for 4×4 periodic square with all faces filled."""

    @pytest.fixture(autouse=True)
    def setup(self, square_4x4):
        self.lat = square_4x4
        B1 = build_B1(square_4x4.n_vertices, square_4x4.edge_list)
        B2 = build_B2(square_4x4.n_edges, square_4x4.face_list, square_4x4.edge_list)
        laps = build_all_laplacians(B1, B2)
        L0_result = eigendecompose(laps["L0"])
        L1_result = eigendecompose(laps["L1"])
        self.betti = compute_betti_numbers(
            B1, B2, L0_result["eigenvalues"], L1_result["eigenvalues"]
        )
        self.L1_result = L1_result
        self.B1 = B1
        self.B2 = B2

    def test_beta_0(self):
        assert self.betti["beta_0"] == 1

    def test_beta_1(self):
        assert self.betti["beta_1"] == 2

    def test_beta_2(self):
        assert self.betti["beta_2"] == 1

    def test_euler_consistent(self):
        assert self.betti["euler_consistent"]

    def test_euler_zero(self):
        # Torus: χ = 0
        assert self.betti["euler_betti"] == 0

    def test_methods_agree(self):
        assert self.betti["method_agreement"]

    def test_harmonic_vectors(self):
        """Harmonic vectors are divergence-free and curl-free."""
        harm = extract_harmonic_basis(
            self.L1_result["eigenvectors"],
            self.L1_result["eigenvalues"],
        )
        assert harm.shape[1] == 2  # β₁ = 2
        results = validate_harmonic_vectors(self.B1, self.B2, harm)
        for r in results:
            assert r["is_harmonic"], f"Mode {r['mode_index']} not harmonic: {r}"


class TestSquareBettiNoFaces:
    """Betti numbers for 4×4 periodic square with NO faces filled."""

    @pytest.fixture(autouse=True)
    def setup(self, square_4x4):
        self.lat = square_4x4
        B1 = build_B1(square_4x4.n_vertices, square_4x4.edge_list)
        # No faces
        B2 = build_B2(square_4x4.n_edges, [], square_4x4.edge_list)
        laps = build_all_laplacians(B1, B2)
        L0_result = eigendecompose(laps["L0"])
        L1_result = eigendecompose(laps["L1"])
        self.betti = compute_betti_numbers(
            B1, B2, L0_result["eigenvalues"], L1_result["eigenvalues"]
        )
        self.L1_result = L1_result
        self.B1 = B1
        self.B2 = B2

    def test_beta_0(self):
        assert self.betti["beta_0"] == 1

    def test_beta_1(self):
        # β₁ = n1 - rank(B1) = 32 - (16 - 1) = 17
        assert self.betti["beta_1"] == 17

    def test_beta_2(self):
        assert self.betti["beta_2"] == 0

    def test_euler_consistent(self):
        assert self.betti["euler_consistent"]

    def test_methods_agree(self):
        assert self.betti["method_agreement"]

    def test_harmonic_count(self):
        harm = extract_harmonic_basis(
            self.L1_result["eigenvectors"],
            self.L1_result["eigenvalues"],
        )
        assert harm.shape[1] == 17


class TestSquareScaling:
    """Verify counts scale correctly with system size."""

    @pytest.mark.parametrize("n", [3, 5, 8])
    def test_periodic_counts(self, n):
        gen = SquareGenerator()
        lat = gen.build(n, n, boundary="periodic")
        assert lat.n_vertices == n * n
        assert lat.n_edges == 2 * n * n
        assert lat.n_faces == n * n

    @pytest.mark.parametrize("n", [3, 5, 8])
    def test_beta1_all_faces_periodic(self, n):
        """β₁ = 2 for any size periodic square with all faces."""
        gen = SquareGenerator()
        lat = gen.build(n, n, boundary="periodic")
        B1 = build_B1(lat.n_vertices, lat.edge_list)
        B2 = build_B2(lat.n_edges, lat.face_list, lat.edge_list)
        laps = build_all_laplacians(B1, B2)
        L1_result = eigendecompose(laps["L1"])
        beta_1 = count_zero_eigenvalues(L1_result["eigenvalues"])
        assert beta_1 == 2
