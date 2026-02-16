"""Shared lattice rendering logic used by both matplotlib and Plotly backends."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# Coordination-number color scheme (consistent across all figures)
COORD_COLORS = {
    2: "#3498db",  # blue
    3: "#2ecc71",  # green
    4: "#e74c3c",  # red
    5: "#9b59b6",  # purple
    6: "#f39c12",  # orange
}

DEFAULT_EDGE_COLOR = "#7f8c8d"
DEFAULT_FACE_ALPHA = 0.15
FACE_COLOR = "#95a5a6"


def coord_color(z: int) -> str:
    """Return color for coordination number z."""
    return COORD_COLORS.get(z, "#34495e")


def compute_layout_bounds(
    positions: np.ndarray,
    padding: float = 0.5,
) -> Tuple[float, float, float, float]:
    """Compute axis bounds from vertex positions with padding."""
    xmin, ymin = positions.min(axis=0)
    xmax, ymax = positions.max(axis=0)
    return (xmin - padding, xmax + padding, ymin - padding, ymax + padding)


def edge_segments(
    positions: np.ndarray,
    edge_list: List[Tuple[int, int]],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return list of (xs, ys) pairs for edge line segments."""
    segments = []
    for u, v in edge_list:
        xs = np.array([positions[u, 0], positions[v, 0]])
        ys = np.array([positions[u, 1], positions[v, 1]])
        segments.append((xs, ys))
    return segments


def face_polygons(
    positions: np.ndarray,
    face_list: List[List[int]],
) -> List[np.ndarray]:
    """Return list of polygon vertex arrays (N, 2) for faces."""
    polys = []
    for face in face_list:
        coords = positions[face]
        polys.append(coords)
    return polys


def edge_segments_periodic(
    positions: np.ndarray,
    edge_list: List[Tuple[int, int]],
    a1: Tuple[float, float],
    a2: Tuple[float, float],
    nx_size: int,
    ny_size: int,
    boundary: str = "periodic",
) -> Tuple[
    List[Tuple[np.ndarray, np.ndarray, int]],
    List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]],
]:
    """Return interior segments and periodic stub pairs for matplotlib.

    Parameters
    ----------
    positions, edge_list, a1, a2, nx_size, ny_size, boundary
        Same as :func:`src.viz.periodic_edges.classify_edges`.

    Returns
    -------
    interior_segs : list of (xs, ys, edge_index)
        Interior edge segments (2-element arrays).
    periodic_stubs : list of (xs_u, ys_u, xs_v, ys_v, edge_index)
        Each periodic edge yields two stubs: one from u, one from v.
    """
    from src.viz.periodic_edges import classify_edges

    interior_edges, periodic_edges, _ = classify_edges(
        positions, edge_list, a1, a2, nx_size, ny_size, boundary,
    )

    interior_segs = []
    for u, v, idx in interior_edges:
        xs = np.array([positions[u, 0], positions[v, 0]])
        ys = np.array([positions[u, 1], positions[v, 1]])
        interior_segs.append((xs, ys, idx))

    periodic_stubs = []
    for pe in periodic_edges:
        su_start, su_end = pe.stub_u
        sv_start, sv_end = pe.stub_v
        xs_u = np.array([su_start[0], su_end[0]])
        ys_u = np.array([su_start[1], su_end[1]])
        xs_v = np.array([sv_start[0], sv_end[0]])
        ys_v = np.array([sv_start[1], sv_end[1]])
        periodic_stubs.append((xs_u, ys_u, xs_v, ys_v, pe.edge_index))

    return interior_segs, periodic_stubs


def harmonic_edge_colors(
    edge_values: np.ndarray,
    cmap_name: str = "RdBu",
) -> Tuple[np.ndarray, float]:
    """Normalize harmonic eigenvector values for edge coloring.

    Returns (normalized_values, vmax) where values are in [-vmax, vmax].
    """
    vmax = np.max(np.abs(edge_values))
    if vmax < 1e-15:
        vmax = 1.0
    return edge_values / vmax, vmax
