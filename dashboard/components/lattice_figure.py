"""Plotly figure for visualizing ASI lattice geometry.

Renders vertices colored by coordination number, edges as lines (optionally
colored by a harmonic eigenvector), and faces as semi-transparent shaded
polygons.  Periodic (wrap-around) edges are drawn as dashed stubs extending
from each endpoint toward the lattice boundary.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Coordination-number color palette
# ---------------------------------------------------------------------------

COORDINATION_COLORS: Dict[int, str] = {
    2: "#3498db",   # blue
    3: "#2ecc71",   # green
    4: "#e74c3c",   # red
    5: "#9b59b6",   # purple
    6: "#f39c12",   # orange
}

_DEFAULT_VERTEX_COLOR = "#95a5a6"  # gray fallback for unexpected z values
_EDGE_COLOR = "#7f8c8d"
_FACE_COLOR = "#95a5a6"
_FACE_OPACITY = 0.15
_PERIODIC_EDGE_COLOR = "#b0b0b0"


def _color_for_z(z: int) -> str:
    """Return the hex color for a coordination number."""
    return COORDINATION_COLORS.get(z, _DEFAULT_VERTEX_COLOR)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_lattice_figure(
    positions: np.ndarray,
    edges: List[Tuple[int, int]],
    coordination: np.ndarray,
    faces: Optional[List[List[int]]] = None,
    harmonic_vector: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    a1: Optional[Tuple[float, float]] = None,
    a2: Optional[Tuple[float, float]] = None,
    nx_size: Optional[int] = None,
    ny_size: Optional[int] = None,
    boundary: Optional[str] = None,
) -> go.Figure:
    """Build an interactive Plotly figure of a lattice.

    Parameters
    ----------
    positions : np.ndarray, shape (N, 2)
        Vertex positions.
    edges : list of (int, int)
        Edge list as pairs of vertex indices.
    coordination : np.ndarray, shape (N,)
        Per-vertex coordination number (degree).
    faces : list of list[int], optional
        Each face is an ordered list of vertex indices forming a polygon.
        Rendered as semi-transparent shaded regions.
    harmonic_vector : np.ndarray, shape (n_edges,), optional
        If provided, edges are colored by this vector's values using a
        diverging RdBu colorscale instead of the default gray.
    title : str, optional
        Figure title.
    a1, a2 : tuple of float, optional
        Unit-cell lattice vectors (needed for periodic edge visualization).
    nx_size, ny_size : int, optional
        Unit-cell repetition counts (needed for periodic edge visualization).
    boundary : str, optional
        ``'periodic'`` or ``'open'``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()

    # --- Classify edges if periodic info is available ---
    interior_edges = None
    periodic_edges = None
    is_periodic = None
    use_periodic = (
        boundary == "periodic"
        and a1 is not None
        and a2 is not None
        and nx_size is not None
        and ny_size is not None
    )

    if use_periodic:
        from src.viz.periodic_edges import classify_edges, classify_faces

        interior_edges, periodic_edges, is_periodic = classify_edges(
            positions, edges, a1, a2, nx_size, ny_size, boundary,
        )

    # --- Faces (draw first so they sit behind everything) ---
    if faces:
        if use_periodic and is_periodic is not None:
            interior_faces, _ = classify_faces(faces, edges, is_periodic)
            _add_faces(fig, positions, interior_faces)
        else:
            _add_faces(fig, positions, faces)

    # --- Edges ---
    if harmonic_vector is not None:
        _add_edges_colored(
            fig, positions, edges, harmonic_vector,
            interior_edges=interior_edges,
            periodic_edges=periodic_edges,
        )
    else:
        _add_edges_plain(
            fig, positions, edges,
            interior_edges=interior_edges,
            periodic_edges=periodic_edges,
        )

    # --- Vertices grouped by coordination number ---
    _add_vertices(fig, positions, coordination)

    # --- Layout ---
    fig.update_layout(
        title=dict(text=title or "", x=0.5, xanchor="center"),
        xaxis=dict(
            scaleanchor="y",
            scaleratio=1,
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            visible=False,
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            visible=False,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=True,
        legend=dict(
            title="Coordination z",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#cccccc",
            borderwidth=1,
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="closest",
    )

    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _add_faces(
    fig: go.Figure,
    positions: np.ndarray,
    faces: List[List[int]],
) -> None:
    """Add semi-transparent face polygons."""
    for face in faces:
        # Close the polygon by repeating the first vertex
        xs = [positions[v, 0] for v in face] + [positions[face[0], 0]]
        ys = [positions[v, 1] for v in face] + [positions[face[0], 1]]

        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            fill="toself",
            fillcolor=f"rgba(149, 165, 166, {_FACE_OPACITY})",
            line=dict(width=0),
            mode="lines",
            hoverinfo="skip",
            showlegend=False,
        ))


def _add_edges_plain(
    fig: go.Figure,
    positions: np.ndarray,
    edges: List[Tuple[int, int]],
    interior_edges=None,
    periodic_edges=None,
) -> None:
    """Add edges as gray lines.  Periodic edges drawn as dashed stubs."""
    if interior_edges is not None and periodic_edges is not None:
        # Interior edges: solid lines in a single trace
        xs: list = []
        ys: list = []
        for u, v, _ in interior_edges:
            xs.extend([positions[u, 0], positions[v, 0], None])
            ys.extend([positions[u, 1], positions[v, 1], None])

        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode="lines",
                line=dict(color=_EDGE_COLOR, width=1.5),
                hoverinfo="skip",
                showlegend=False,
            ))

        # Periodic edges: dashed stubs in a single trace
        pxs: list = []
        pys: list = []
        for pe in periodic_edges:
            # Stub from u
            su_start, su_end = pe.stub_u
            pxs.extend([su_start[0], su_end[0], None])
            pys.extend([su_start[1], su_end[1], None])
            # Stub from v
            sv_start, sv_end = pe.stub_v
            pxs.extend([sv_start[0], sv_end[0], None])
            pys.extend([sv_start[1], sv_end[1], None])

        if pxs:
            fig.add_trace(go.Scatter(
                x=pxs, y=pys,
                mode="lines",
                line=dict(color=_PERIODIC_EDGE_COLOR, width=1.5, dash="dash"),
                hoverinfo="skip",
                showlegend=False,
            ))
    else:
        # Fallback: draw all edges as solid (no periodic info)
        xs = []
        ys = []
        for u, v in edges:
            xs.extend([positions[u, 0], positions[v, 0], None])
            ys.extend([positions[u, 1], positions[v, 1], None])

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            line=dict(color=_EDGE_COLOR, width=1.5),
            hoverinfo="skip",
            showlegend=False,
        ))


def _add_edges_colored(
    fig: go.Figure,
    positions: np.ndarray,
    edges: List[Tuple[int, int]],
    harmonic_vector: np.ndarray,
    interior_edges=None,
    periodic_edges=None,
) -> None:
    """Add edges colored by harmonic_vector values using RdBu colorscale.

    Interior edges are drawn as solid colored segments.  Periodic edges are
    drawn as two dashed colored stubs per edge.
    """
    vmax = np.max(np.abs(harmonic_vector)) if len(harmonic_vector) > 0 else 1.0
    if vmax == 0:
        vmax = 1.0

    if interior_edges is not None and periodic_edges is not None:
        # Build periodic edge index set for fast lookup
        periodic_idx_set = {pe.edge_index for pe in periodic_edges}

        # Background: faint gray for interior edges only
        xs_bg: list = []
        ys_bg: list = []
        for u, v, idx in interior_edges:
            xs_bg.extend([positions[u, 0], positions[v, 0], None])
            ys_bg.extend([positions[u, 1], positions[v, 1], None])

        if xs_bg:
            fig.add_trace(go.Scatter(
                x=xs_bg, y=ys_bg,
                mode="lines",
                line=dict(color="rgba(200,200,200,0.4)", width=3),
                hoverinfo="skip",
                showlegend=False,
            ))

        # Interior colored segments
        for u, v, idx in interior_edges:
            val = harmonic_vector[idx] if idx < len(harmonic_vector) else 0.0
            normed = val / vmax
            r, g, b = _rdbu_color(normed)
            color = f"rgb({r},{g},{b})"

            fig.add_trace(go.Scatter(
                x=[positions[u, 0], positions[v, 0]],
                y=[positions[u, 1], positions[v, 1]],
                mode="lines",
                line=dict(color=color, width=3),
                hoverinfo="text",
                hovertext=f"edge {idx}: {val:.4f}",
                showlegend=False,
            ))

        # Periodic colored dashed stubs
        for pe in periodic_edges:
            val = harmonic_vector[pe.edge_index] if pe.edge_index < len(harmonic_vector) else 0.0
            normed = val / vmax
            r, g, b = _rdbu_color(normed)
            color = f"rgb({r},{g},{b})"

            # Stub from u
            su_start, su_end = pe.stub_u
            fig.add_trace(go.Scatter(
                x=[su_start[0], su_end[0]],
                y=[su_start[1], su_end[1]],
                mode="lines",
                line=dict(color=color, width=3, dash="dash"),
                hoverinfo="text",
                hovertext=f"edge {pe.edge_index} (periodic): {val:.4f}",
                showlegend=False,
            ))
            # Stub from v
            sv_start, sv_end = pe.stub_v
            fig.add_trace(go.Scatter(
                x=[sv_start[0], sv_end[0]],
                y=[sv_start[1], sv_end[1]],
                mode="lines",
                line=dict(color=color, width=3, dash="dash"),
                hoverinfo="text",
                hovertext=f"edge {pe.edge_index} (periodic): {val:.4f}",
                showlegend=False,
            ))

    else:
        # Fallback: no periodic info, draw everything as solid
        xs_bg: list = []
        ys_bg: list = []
        for u, v in edges:
            xs_bg.extend([positions[u, 0], positions[v, 0], None])
            ys_bg.extend([positions[u, 1], positions[v, 1], None])

        fig.add_trace(go.Scatter(
            x=xs_bg, y=ys_bg,
            mode="lines",
            line=dict(color="rgba(200,200,200,0.4)", width=3),
            hoverinfo="skip",
            showlegend=False,
        ))

        for idx, (u, v) in enumerate(edges):
            val = harmonic_vector[idx] if idx < len(harmonic_vector) else 0.0
            normed = val / vmax
            r, g, b = _rdbu_color(normed)
            color = f"rgb({r},{g},{b})"

            fig.add_trace(go.Scatter(
                x=[positions[u, 0], positions[v, 0]],
                y=[positions[u, 1], positions[v, 1]],
                mode="lines",
                line=dict(color=color, width=3),
                hoverinfo="text",
                hovertext=f"edge {idx}: {val:.4f}",
                showlegend=False,
            ))

    # Colorbar via invisible scatter
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(
            size=0.001,
            color=[0],
            colorscale="RdBu",
            cmin=-vmax,
            cmax=vmax,
            colorbar=dict(
                title="Harmonic<br>amplitude",
                thickness=15,
                len=0.6,
            ),
            showscale=True,
        ),
        hoverinfo="skip",
        showlegend=False,
    ))


def _rdbu_color(t: float) -> Tuple[int, int, int]:
    """Map a value in [-1, 1] to an RGB tuple on a Red-Blue diverging scale.

    -1 -> blue (67, 147, 195)
     0 -> white (247, 247, 247)
    +1 -> red  (214, 96, 77)
    """
    t_clamp = max(-1.0, min(1.0, t))
    if t_clamp < 0:
        frac = -t_clamp  # 0..1, how blue
        r = int(247 - frac * (247 - 67))
        g = int(247 - frac * (247 - 147))
        b = int(247 - frac * (247 - 195))
    else:
        frac = t_clamp  # 0..1, how red
        r = int(247 - frac * (247 - 214))
        g = int(247 - frac * (247 - 96))
        b = int(247 - frac * (247 - 77))
    return r, g, b


def _add_vertices(
    fig: go.Figure,
    positions: np.ndarray,
    coordination: np.ndarray,
) -> None:
    """Add vertex markers, one trace per coordination number for legend grouping."""
    unique_z = sorted(set(int(z) for z in coordination))

    for z in unique_z:
        mask = coordination == z
        idxs = np.where(mask)[0]
        color = _color_for_z(z)

        fig.add_trace(go.Scatter(
            x=positions[idxs, 0],
            y=positions[idxs, 1],
            mode="markers",
            marker=dict(
                size=7,
                color=color,
                line=dict(width=0.5, color="white"),
            ),
            name=f"z = {z}",
            legendgroup=f"z{z}",
            hoverinfo="text",
            hovertext=[f"v{i} (z={z})" for i in idxs],
        ))
