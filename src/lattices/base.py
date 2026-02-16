"""Base classes for lattice generation.

UnitCell defines a tileable lattice pattern. LatticeGenerator handles
the shared tiling logic for periodic and open boundary conditions.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np


@dataclass
class UnitCell:
    """Definition of a repeating lattice unit cell.

    Attributes:
        a1: First lattice translation vector.
        a2: Second lattice translation vector.
        vertices: Fractional positions of vertices within the cell.
        edges: List of (src_vertex, dst_vertex, di, dj) where di, dj are
            cell offsets for the destination vertex.
        faces: Each face is an ordered list of (vertex_idx, di, dj) tuples
            describing the vertex cycle (counterclockwise).
        expected_coordination: Mapping from intra-cell vertex index to
            expected bulk coordination number (degree).
    """
    a1: Tuple[float, float]
    a2: Tuple[float, float]
    vertices: List[Tuple[float, float]]
    edges: List[Tuple[int, int, int, int]]
    faces: List[List[Tuple[int, int, int]]]
    expected_coordination: Dict[int, int] = field(default_factory=dict)


@dataclass
class LatticeResult:
    """Output of lattice construction."""
    name: str
    nx_size: int
    ny_size: int
    boundary: str  # 'periodic' or 'open'
    positions: np.ndarray  # (n_vertices, 2)
    edge_list: List[Tuple[int, int]]  # canonical orientation: lower -> higher index
    face_list: List[List[int]]  # each face = ordered list of global vertex indices
    coordination: np.ndarray  # per-vertex degree
    graph: nx.Graph
    n_vertices: int
    n_edges: int
    n_faces: int
    unit_cell: UnitCell

    @property
    def coordination_distribution(self) -> Dict[int, int]:
        unique, counts = np.unique(self.coordination, return_counts=True)
        return {int(u): int(c) for u, c in zip(unique, counts)}


class LatticeGenerator(abc.ABC):
    """Base class for lattice generators with shared tiling logic."""

    name: str = "base"

    @abc.abstractmethod
    def _define_unit_cell(self) -> UnitCell:
        """Return the unit cell definition for this lattice type."""
        ...

    def build(
        self,
        nx_size: int,
        ny_size: int,
        boundary: str = "periodic",
    ) -> LatticeResult:
        """Tile the unit cell and construct the full lattice.

        Args:
            nx_size: Number of unit cells in the x direction.
            ny_size: Number of unit cells in the y direction.
            boundary: 'periodic' or 'open'.

        Returns:
            LatticeResult with all lattice data.
        """
        uc = self._define_unit_cell()
        n_vert_per_cell = len(uc.vertices)
        a1 = np.array(uc.a1)
        a2 = np.array(uc.a2)

        # --- Map (vertex_in_cell, cell_i, cell_j) -> global index ---
        def global_idx(v: int, ci: int, cj: int) -> int:
            return v + n_vert_per_cell * (ci * ny_size + cj)

        # --- Build vertex positions ---
        n_total_verts = n_vert_per_cell * nx_size * ny_size
        positions = np.zeros((n_total_verts, 2))
        for ci in range(nx_size):
            for cj in range(ny_size):
                origin = ci * a1 + cj * a2
                for v, (fx, fy) in enumerate(uc.vertices):
                    gidx = global_idx(v, ci, cj)
                    positions[gidx] = origin + np.array([fx, fy])

        # --- Build edges with canonical orientation (lower -> higher index) ---
        edge_set = set()
        edge_list = []
        for ci in range(nx_size):
            for cj in range(ny_size):
                for src_v, dst_v, di, dj in uc.edges:
                    ni = ci + di
                    nj = cj + dj
                    if boundary == "periodic":
                        ni = ni % nx_size
                        nj = nj % ny_size
                    else:  # open
                        if ni < 0 or ni >= nx_size or nj < 0 or nj >= ny_size:
                            continue

                    g_src = global_idx(src_v, ci, cj)
                    g_dst = global_idx(dst_v, ni, nj)

                    if g_src == g_dst:
                        continue  # skip self-loops

                    # Canonical: smaller index first
                    e = (min(g_src, g_dst), max(g_src, g_dst))
                    if e not in edge_set:
                        edge_set.add(e)
                        edge_list.append(e)

        # --- Build faces ---
        face_list = []
        if uc.faces:
            face_set = set()
            for ci in range(nx_size):
                for cj in range(ny_size):
                    for face_def in uc.faces:
                        face_verts = []
                        valid = True
                        for v_idx, fdi, fdj in face_def:
                            ni = ci + fdi
                            nj = cj + fdj
                            if boundary == "periodic":
                                ni = ni % nx_size
                                nj = nj % ny_size
                            else:
                                if ni < 0 or ni >= nx_size or nj < 0 or nj >= ny_size:
                                    valid = False
                                    break
                            face_verts.append(global_idx(v_idx, ni, nj))

                        if not valid:
                            continue

                        # Check all edges of this face exist
                        n_fv = len(face_verts)
                        all_edges_exist = True
                        for k in range(n_fv):
                            u, v = face_verts[k], face_verts[(k + 1) % n_fv]
                            e = (min(u, v), max(u, v))
                            if e not in edge_set:
                                all_edges_exist = False
                                break

                        if not all_edges_exist:
                            continue

                        # Deduplicate faces (canonical form = sorted tuple)
                        face_key = tuple(sorted(face_verts))
                        if len(set(face_verts)) != len(face_verts):
                            continue  # degenerate face
                        if face_key not in face_set:
                            face_set.add(face_key)
                            face_list.append(face_verts)

        # --- Build NetworkX graph ---
        G = nx.Graph()
        for i in range(n_total_verts):
            G.add_node(i, pos=tuple(positions[i]))
        for u, v in edge_list:
            G.add_edge(u, v)

        coordination = np.array([G.degree(i) for i in range(n_total_verts)])

        return LatticeResult(
            name=self.name,
            nx_size=nx_size,
            ny_size=ny_size,
            boundary=boundary,
            positions=positions,
            edge_list=edge_list,
            face_list=face_list,
            coordination=coordination,
            graph=G,
            n_vertices=n_total_verts,
            n_edges=len(edge_list),
            n_faces=len(face_list),
            unit_cell=uc,
        )
